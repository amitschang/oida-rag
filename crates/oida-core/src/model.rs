//! Domain types describing OIDA documents, artifacts, and their relationships.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A single stored artifact belonging to a [`Document`].
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

/// Document-level metadata: one row per source document.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct Document {
    /// Stable document identifier (groups all of a document's artifacts).
    pub id: String,
    /// Bates number — the identifier used by relationship references.
    pub bn: Option<String>,
    /// Human-readable title (`ti`, falling back to `filename`).
    pub title: Option<String>,
    pub industry: Option<String>,
    pub collection: Option<String>,
    pub genre: Option<String>,
    /// Date sent, as stored (free-form string such as `2008 March 05`).
    pub date_sent: Option<String>,
    pub date_received: Option<String>,
    pub topic: Option<String>,
    pub description: Option<String>,
    pub keywords: Option<String>,
    /// Conversation / thread identifier, if part of an email thread.
    pub conversation: Option<String>,
    pub custodian: Vec<String>,
    /// Authors / senders.
    pub authors: Vec<String>,
    /// Recipients.
    pub recipients: Vec<String>,
    /// CC recipients.
    pub cc: Vec<String>,
    /// Bates numbers of attachments.
    pub attachments: Vec<String>,
    /// Bates numbers of related documents.
    pub related: Vec<String>,
    /// Bates numbers mentioned within this document.
    pub mentions: Vec<String>,
    /// Distinct artifact media types available for this document.
    pub artifact_types: Vec<String>,
    /// Number of artifacts attached to this document.
    pub artifact_count: u64,
}

/// A search result: a document summary plus provenance about why it matched.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchHit {
    pub id: String,
    pub bn: Option<String>,
    pub title: Option<String>,
    pub date_sent: Option<String>,
    pub artifact_types: Vec<String>,
    pub artifact_count: u64,
    /// Number of query terms that matched anywhere in the document.
    pub score: u32,
    /// Field names that contained at least one query term.
    pub matched_fields: Vec<String>,
}

/// A hybrid (keyword + semantic) search hit over artifact text.
///
/// Results are folded to one entry per document: `score` is the document's
/// fused (RRF) relevance and `snippet` is the best-matching passage of its
/// OCR/plain text. Document metadata is hydrated from the cache.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HybridHit {
    pub id: String,
    pub bn: Option<String>,
    pub title: Option<String>,
    pub date_sent: Option<String>,
    pub artifact_types: Vec<String>,
    pub artifact_count: u64,
    /// Fused relevance score (higher is better) from Reciprocal Rank Fusion.
    pub score: f32,
    /// The artifact file the best-matching passage came from.
    pub artifact_name: Option<String>,
    /// Best-matching passage of the document's text, for context.
    pub snippet: Option<String>,
}

/// The kind of relationship connecting two documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    Attachment,
    Related,
    Mention,
    Conversation,
}

impl RelationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RelationKind::Attachment => "attachment",
            RelationKind::Related => "related",
            RelationKind::Mention => "mention",
            RelationKind::Conversation => "conversation",
        }
    }
}

/// An edge in the document relationship graph.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RelatedEdge {
    /// Document id the edge originates from.
    pub from_id: String,
    /// Relationship type.
    pub kind: RelationKind,
    /// The Bates/conversation reference that produced this edge.
    pub reference: String,
    /// Resolved neighbor document, if one exists in the index.
    pub neighbor: Option<Document>,
    /// BFS depth at which this edge was discovered (1 = direct).
    pub depth: u32,
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
    serde_json::value::RawValue::from_string("[]".to_string())
        .expect("`[]` is valid JSON")
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
