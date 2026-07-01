//! Corpus-agnostic result and schema types.
//!
//! These types are fixed by the framework: the artifact manifest
//! ([`Artifact`]/[`RawArtifact`]) and the schema-agnostic query results
//! ([`SqlQueryResult`], [`TableSchema`]). They know nothing about any
//! particular document schema.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A single stored artifact belonging to a document.
///
/// Each artifact is one file (e.g. the `.ocr` text, the `.pdf`, or a
/// `_thumb.png`); all of a document's artifacts are grouped under its `id`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Artifact {
    /// Document id this artifact belongs to.
    pub document_id: String,
    /// File name on disk (also the lookup key under `artifact_root`).
    pub name: String,
    /// MIME type, e.g. `text/plain`, `application/pdf`, `image/png`.
    pub media_type: Option<String>,
    /// Size in bytes, if known.
    pub size: Option<u64>,
    /// MD5 checksum, if known.
    pub md5: Option<String>,
}

/// The stored bytes of a single non-text artifact, returned by a `(id, name)`
/// point lookup against the `raw_artifacts` table.
///
/// That pair uniquely identifies one artifact file, so it is the only
/// retrieval mode raw storage supports — the table exists to return the
/// original bytes, never to be scanned in bulk.
#[derive(Debug, Clone)]
pub struct RawArtifact {
    /// Document id this artifact belongs to.
    pub document_id: String,
    /// File name on disk (the second half of the lookup key).
    pub name: String,
    /// MIME type, e.g. `application/pdf`, `image/png`.
    pub media_type: Option<String>,
    /// MD5 checksum, if known.
    pub md5: Option<String>,
    /// Size in bytes, if known. Reflects the source's reported size, which the
    /// stored `data` length should match.
    pub size: Option<u64>,
    /// The original file bytes.
    pub data: Vec<u8>,
}

/// A metadata-search result: the matched document plus why it matched.
///
/// Generic over the provider's result type `D` (a lean summary for search, the
/// full document for a point lookup), reshaped to carry the whole document
/// rather than a hand-maintained subset of inlined fields.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SearchHit<D> {
    /// The matched document, decoded at the provider's chosen projection.
    pub document: D,
    /// Number of distinct query terms that matched anywhere in the document.
    pub score: u32,
    /// Field names that contained at least one query term.
    pub matched_fields: Vec<String>,
}

/// A hybrid (keyword + semantic) search hit over artifact text.
///
/// Results are folded to one entry per document: `score` is the document's
/// fused (RRF) relevance and `snippet` is the best-matching passage of its
/// OCR/plain text. Generic over the hydrated document type `D`.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct HybridHit<D> {
    /// The matched document, hydrated at the provider's chosen projection.
    pub document: D,
    /// Fused relevance score (higher is better) from Reciprocal Rank Fusion.
    pub score: f32,
    /// The artifact file the best-matching passage came from.
    pub artifact_name: Option<String>,
    /// Best-matching passage of the document's text, for context.
    pub snippet: Option<String>,
}

/// The result of running a read-only SQL query against the cache.
///
/// On success, `columns` names the projected columns and `rows` holds the
/// result as a JSON array of objects keyed by column name (lists/structs become
/// JSON arrays/objects). Rows is expected to be a valid json string, hence the
/// RawValue. On failure (invalid or rejected SQL, or a DataFusion execution
/// error), `error` carries a human-readable message and `rows` is an empty
/// array — letting the model read the error and correct its query.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SqlQueryResult {
    /// Column names in projection order.
    pub columns: Vec<String>,
    /// Result rows as a JSON array of objects keyed by column name.
    #[schemars(with = "Vec<serde_json::Value>")]
    pub rows: Box<serde_json::value::RawValue>,
    /// Number of rows returned.
    pub row_count: usize,
    /// True if more rows existed than the requested row cap.
    pub truncated: bool,
    /// Error message when the query was rejected or failed; `None` on success.
    pub error: Option<String>,
}

impl Default for SqlQueryResult {
    fn default() -> Self {
        Self {
            columns: Vec::new(),
            rows: empty_json_rows(),
            row_count: 0,
            truncated: false,
            error: None,
        }
    }
}

impl SqlQueryResult {
    /// Build a failed result carrying only an error message.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            error: Some(message.into()),
            ..Self::default()
        }
    }
}

/// An empty JSON array (`[]`) for use as the default/error-case `rows` value.
fn empty_json_rows() -> Box<serde_json::value::RawValue> {
    serde_json::value::RawValue::from_string("[]".to_string()).expect("`[]` is valid JSON")
}

/// One column of a table, as reported by `DESCRIBE`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ColumnInfo {
    /// Column name.
    pub name: String,
    /// Arrow type, e.g. `Utf8`, `Int64`, `List(Utf8)`.
    pub type_: String,
}

/// The schema of one cache table.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TableSchema {
    /// Table name (`documents` or `artifacts`).
    pub table: String,
    /// Columns in definition order.
    pub columns: Vec<ColumnInfo>,
}
