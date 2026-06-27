//! Keyword search over the `documents` table.
//!
//! Candidate documents are found with LanceDB's full-text (BM25) index over the
//! concatenated `search_text` column, then scored and ranked in Rust by the
//! number of distinct query terms they contain, reporting per-field provenance.
//! Matching is metadata-only; artifact OCR text is searched by [`crate::hybrid`].

use crate::index::Index;
use crate::model::{Document, SearchHit};

/// Parameters accepted by [`Index::search`].
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Free-text query; whitespace-separated terms are matched independently.
    pub query: String,
    /// Restrict to documents that have an artifact of this media type.
    pub media_type: Option<String>,
    /// Restrict to documents whose custodian contains this substring.
    pub custodian: Option<String>,
    /// Maximum number of hits to return.
    pub limit: u32,
    /// Number of leading hits to skip (pagination).
    pub offset: u32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            query: String::new(),
            media_type: None,
            custodian: None,
            limit: 10,
            offset: 0,
        }
    }
}

impl Index {
    /// Keyword-search documents, returning ranked hits with provenance.
    pub async fn search(&self, params: &SearchParams) -> anyhow::Result<Vec<SearchHit>> {
        let terms = normalize_terms(&params.query);
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // Over-fetch FTS candidates so post-filtering and pagination still have
        // enough rows to satisfy `offset + limit`.
        let want = params.offset as usize + params.limit as usize;
        let fetch = want.saturating_mul(4).max(want).max(50);
        let docs = self.documents_fts(&params.query, fetch).await?;

        let mut hits: Vec<SearchHit> = docs
            .into_iter()
            .filter(|d| passes_filters(d, params))
            .filter_map(|d| {
                let (score, matched) = score_document(&d, &terms);
                (score > 0).then(|| to_hit(d, score, matched))
            })
            .collect();

        // Rank by term count, then by how many artifacts the document has.
        hits.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then(b.artifact_count.cmp(&a.artifact_count))
        });

        let start = (params.offset as usize).min(hits.len());
        let end = (start + params.limit as usize).min(hits.len());
        Ok(hits[start..end].to_vec())
    }
}

/// Apply the optional media-type and custodian filters to a candidate document.
fn passes_filters(doc: &Document, params: &SearchParams) -> bool {
    if let Some(mt) = &params.media_type
        && !doc.artifact_types.iter().any(|t| t == mt)
    {
        return false;
    }
    if let Some(c) = &params.custodian {
        let needle = c.to_lowercase();
        if !doc
            .custodian
            .iter()
            .any(|v| v.to_lowercase().contains(&needle))
        {
            return false;
        }
    }
    true
}

/// Count the distinct query terms present in the document's searchable fields
/// and report which named fields contributed.
fn score_document(doc: &Document, terms: &[String]) -> (u32, Vec<String>) {
    let fields: [(&str, String); 8] = [
        ("title", opt(&doc.title)),
        ("bn", opt(&doc.bn)),
        ("topic", opt(&doc.topic)),
        ("description", opt(&doc.description)),
        ("keywords", opt(&doc.keywords)),
        ("authors", doc.authors.join(" ")),
        ("custodian", doc.custodian.join(" ")),
        ("recipients", doc.recipients.join(" ")),
    ];
    let lowered: Vec<(&str, String)> = fields
        .iter()
        .map(|(name, val)| (*name, val.to_lowercase()))
        .collect();

    let haystack = lowered
        .iter()
        .map(|(_, v)| v.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let score = terms.iter().filter(|t| haystack.contains(t.as_str())).count() as u32;

    let matched = lowered
        .into_iter()
        .filter(|(_, val)| terms.iter().any(|t| val.contains(t.as_str())))
        .map(|(name, _)| name.to_string())
        .collect();

    (score, matched)
}

/// Build a [`SearchHit`] from a matched document.
fn to_hit(doc: Document, score: u32, matched_fields: Vec<String>) -> SearchHit {
    SearchHit {
        id: doc.id,
        bn: doc.bn,
        title: doc.title,
        date_sent: doc.date_sent,
        artifact_types: doc.artifact_types,
        artifact_count: doc.artifact_count,
        score,
        matched_fields,
    }
}

fn opt(s: &Option<String>) -> String {
    s.clone().unwrap_or_default()
}

/// Lowercase, split on whitespace, and deduplicate query terms.
fn normalize_terms(query: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| !t.is_empty() && seen.insert(t.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_and_dedups_terms() {
        assert_eq!(normalize_terms("Foo foo BAR"), vec!["foo", "bar"]);
        assert!(normalize_terms("   ").is_empty());
    }

    #[test]
    fn scores_by_distinct_term_presence() {
        let doc = Document {
            title: Some("Quarterly Report".into()),
            authors: vec!["Jane Doe".into()],
            ..Document::default()
        };
        let (score, matched) = score_document(&doc, &["report".into(), "jane".into()]);
        assert_eq!(score, 2);
        assert!(matched.contains(&"title".to_string()));
        assert!(matched.contains(&"authors".to_string()));
    }

    #[test]
    fn media_type_filter_uses_artifact_types() {
        let doc = Document {
            artifact_types: vec!["application/pdf".into()],
            ..Document::default()
        };
        let yes = SearchParams {
            media_type: Some("application/pdf".into()),
            ..SearchParams::default()
        };
        let no = SearchParams {
            media_type: Some("text/plain".into()),
            ..SearchParams::default()
        };
        assert!(passes_filters(&doc, &yes));
        assert!(!passes_filters(&doc, &no));
    }
}
