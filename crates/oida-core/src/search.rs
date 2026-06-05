//! Keyword search over the cached `documents` table.
//!
//! v1 uses case-insensitive substring matching (`LIKE`) over the key metadata
//! fields, ranking documents by how many distinct query terms they contain.
//! Matching is metadata-only; artifact OCR text is not indexed in v1.

use duckdb::ToSql;

use crate::index::{DOC_COLS, Index, row_to_document};
use crate::model::{Document, SearchHit};

/// Concatenation of the searchable metadata fields, lowercased for matching.
const HAYSTACK: &str = "lower(concat_ws(' ', \
     coalesce(title, ''), coalesce(bn, ''), coalesce(topic, ''), \
     coalesce(description, ''), coalesce(keywords, ''), \
     coalesce(array_to_string(authors, ' '), ''), \
     coalesce(array_to_string(custodian, ' '), ''), \
     coalesce(array_to_string(recipients, ' '), '')))";

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
    pub fn search(&self, params: &SearchParams) -> anyhow::Result<Vec<SearchHit>> {
        let terms = normalize_terms(&params.query);
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // One scoring term per query term: 1 if the haystack contains it.
        let score_expr = terms
            .iter()
            .map(|_| format!("(CASE WHEN {HAYSTACK} LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END)"))
            .collect::<Vec<_>>()
            .join(" + ");

        // Owned bind values, kept alive for the duration of the query.
        let like_params: Vec<String> = terms
            .iter()
            .map(|t| format!("%{}%", escape_like(t)))
            .collect();

        let mut filters = String::new();
        let mut filter_vals: Vec<String> = Vec::new();
        if let Some(mt) = &params.media_type {
            filters.push_str(" AND id IN (SELECT id FROM artifacts WHERE media_type = ?)");
            filter_vals.push(mt.clone());
        }
        if let Some(c) = &params.custodian {
            filters.push_str(" AND array_to_string(custodian, ' ') ILIKE ?");
            filter_vals.push(format!("%{}%", escape_like(c)));
        }

        let sql = format!(
            "WITH scored AS (
                 SELECT {DOC_COLS}, ({score_expr}) AS score
                 FROM documents
                 WHERE true{filters}
             )
             SELECT * FROM scored
             WHERE score > 0
             ORDER BY score DESC, artifact_count DESC
             LIMIT ? OFFSET ?"
        );

        // Bind order: scoring terms, then filters, then limit/offset.
        let limit = params.limit as i64;
        let offset = params.offset as i64;
        let mut binds: Vec<&dyn ToSql> = Vec::new();
        for p in &like_params {
            binds.push(p);
        }
        for v in &filter_vals {
            binds.push(v);
        }
        binds.push(&limit);
        binds.push(&offset);

        let conn = self.conn_lock();
        let mut stmt = conn.prepare(&sql)?;
        // `scored` projects DOC_COLS (0..=20) followed by score (21).
        let rows = stmt.query_map(binds.as_slice(), |row| {
            let doc = row_to_document(row)?;
            let score: i64 = row.get(21)?;
            Ok((doc, score as u32))
        })?;

        let mut hits = Vec::new();
        for r in rows {
            let (doc, score) = r?;
            hits.push(to_hit(doc, score, &terms));
        }
        Ok(hits)
    }
}

/// Build a [`SearchHit`] from a matched document, computing field provenance.
fn to_hit(doc: Document, score: u32, terms: &[String]) -> SearchHit {
    let matched_fields = matched_fields(&doc, terms);
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

/// Determine which named fields contained at least one query term.
fn matched_fields(doc: &Document, terms: &[String]) -> Vec<String> {
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
    fields
        .into_iter()
        .filter(|(_, val)| {
            let low = val.to_lowercase();
            terms.iter().any(|t| low.contains(t.as_str()))
        })
        .map(|(name, _)| name.to_string())
        .collect()
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

/// Escape `LIKE` metacharacters so terms match literally (with `ESCAPE '\'`).
fn escape_like(term: &str) -> String {
    term.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}
