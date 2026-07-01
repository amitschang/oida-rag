//! Keyword search over the `documents` table.
//!
//! Candidate documents are found with LanceDB's full-text (BM25) index over the
//! concatenated `search_text` column, then scored and ranked in Rust by the
//! number of distinct query terms they contain, reporting per-field provenance.
//! Matching is metadata-only; artifact OCR text is searched by [`crate::hybrid`].
//!
//! The engine is generic over the provider's result type `D: SearchableRow`:
//! `D::columns()` sets the (narrow) projection, and `D::searchable_fields()`
//! supplies the field/text pairs to score and report — the only place a corpus
//! enumerates its document fields by name.

use crate::index::Index;
use crate::model::SearchHit;
use crate::row::SearchableRow;

/// Parameters accepted by [`Index::search`].
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Free-text query; whitespace-separated terms are matched independently.
    pub query: String,
    /// Restrict to documents that have an artifact of this media type.
    pub media_type: Option<String>,
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
            limit: 10,
            offset: 0,
        }
    }
}

impl Index {
    /// Keyword-search documents, returning ranked hits with provenance.
    pub async fn search<D: SearchableRow>(
        &self,
        params: &SearchParams,
    ) -> anyhow::Result<Vec<SearchHit<D>>> {
        let terms = normalize_terms(&params.query);
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // Over-fetch FTS candidates so post-filtering and pagination still have
        // enough rows to satisfy `offset + limit`.
        let want = params.offset as usize + params.limit as usize;
        let fetch = want.saturating_mul(4).max(want).max(50);
        let docs: Vec<D> = self.documents_fts_rows::<D>(&params.query, fetch).await?;

        let mut hits: Vec<SearchHit<D>> = docs
            .into_iter()
            .filter(|d| passes_filters(d, params))
            .filter_map(|d| {
                let (score, matched) = score_document(&d, &terms);
                (score > 0).then(|| SearchHit {
                    document: d,
                    score,
                    matched_fields: matched,
                })
            })
            .collect();

        // Rank by distinct-term count; ties keep the candidates' BM25 order.
        hits.sort_by(|a, b| b.score.cmp(&a.score));

        let start = (params.offset as usize).min(hits.len());
        let end = (start + params.limit as usize).min(hits.len());
        Ok(hits.drain(start..end).collect())
    }
}

/// Apply the optional media-type filter to a candidate document, testing it
/// against the document's artifact media types (the framework-fixed manifest).
fn passes_filters<D: SearchableRow>(doc: &D, params: &SearchParams) -> bool {
    if let Some(mt) = &params.media_type
        && !doc.artifact_types().iter().any(|t| t == mt)
    {
        return false;
    }
    true
}

/// Count the distinct query terms present in the document's searchable fields
/// and report which named fields contributed.
fn score_document<D: SearchableRow>(doc: &D, terms: &[String]) -> (u32, Vec<String>) {
    let lowered: Vec<(&'static str, String)> = doc
        .searchable_fields()
        .into_iter()
        .map(|(name, val)| (name, val.to_lowercase()))
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
    use crate::row::DocumentRow;
    use anyhow::Result;
    use arrow::array::RecordBatch;
    use schemars::JsonSchema;
    use serde::Serialize;

    /// A minimal in-crate [`SearchableRow`] so the engine's scoring/filtering can
    /// be tested without a concrete corpus's document type.
    #[derive(Default, Serialize, JsonSchema)]
    struct TestRow {
        id: String,
        title: String,
        authors: Vec<String>,
        artifact_types: Vec<String>,
    }

    impl DocumentRow for TestRow {
        fn columns() -> &'static [&'static str] {
            &["id", "title", "authors", "artifact_types"]
        }
        fn from_row(_batch: &RecordBatch, _i: usize) -> Result<Self> {
            unimplemented!("not exercised by these unit tests")
        }
        fn id(&self) -> &str {
            &self.id
        }
    }

    impl SearchableRow for TestRow {
        fn searchable_fields(&self) -> Vec<(&'static str, String)> {
            vec![
                ("title", self.title.clone()),
                ("authors", self.authors.join(" ")),
            ]
        }
        fn artifact_types(&self) -> &[String] {
            &self.artifact_types
        }
    }

    #[test]
    fn normalizes_and_dedups_terms() {
        assert_eq!(normalize_terms("Foo foo BAR"), vec!["foo", "bar"]);
        assert!(normalize_terms("   ").is_empty());
    }

    #[test]
    fn scores_by_distinct_term_presence() {
        let doc = TestRow {
            title: "Quarterly Report".into(),
            authors: vec!["Jane Doe".into()],
            ..TestRow::default()
        };
        let (score, matched) = score_document(&doc, &["report".into(), "jane".into()]);
        assert_eq!(score, 2);
        assert!(matched.contains(&"title".to_string()));
        assert!(matched.contains(&"authors".to_string()));
    }

    #[test]
    fn media_type_filter_uses_artifact_types() {
        let doc = TestRow {
            artifact_types: vec!["application/pdf".into()],
            ..TestRow::default()
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
