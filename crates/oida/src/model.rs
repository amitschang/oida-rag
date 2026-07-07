//! OIDA-specific document, search-result, and relationship types.
//!
//! These describe the OIDA corpus schema: the typed [`Document`], its lean
//! [`DocumentSummary`], and the relationship graph
//! ([`RelationKind`]/[`RelatedEdge`]). A different corpus would supply its own.

use anyhow::Result;
use arrow::array::RecordBatch;
use corpus_index::index::{row_list, row_opt_str, row_u64};
use corpus_index::row::{DocumentRow, SearchableRow};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Columns of the `documents` table mapped onto [`Document`], in definition
/// order. `search_text` (an FTS-only concatenation) is intentionally excluded.
const DOC_COLS: &[&str] = &[
    "id",
    "bn",
    "title",
    "industry",
    "collection",
    "genre",
    "date_sent",
    "date_received",
    "topic",
    "description",
    "keywords",
    "conversation",
    "custodian",
    "authors",
    "recipients",
    "cc",
    "attachments",
    "related",
    "mentions",
    "artifact_types",
    "artifact_count",
];

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

/// Lean document projection used for search and hybrid results.
///
/// Carries the fields shown in a result (id, bn, title, date, artifact summary)
/// plus the free-text fields the metadata search scores on (topic, description,
/// keywords, authors, custodian, recipients) — but not the graph/admin columns
/// of the full [`Document`], so the search scan stays narrow.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct DocumentSummary {
    pub id: String,
    pub bn: Option<String>,
    pub title: Option<String>,
    pub date_sent: Option<String>,
    pub topic: Option<String>,
    pub description: Option<String>,
    pub keywords: Option<String>,
    pub authors: Vec<String>,
    pub custodian: Vec<String>,
    pub recipients: Vec<String>,
    pub artifact_types: Vec<String>,
    pub artifact_count: u64,
}

/// Columns of [`DocumentSummary`], in decode order.
const SUMMARY_COLS: &[&str] = &[
    "id",
    "bn",
    "title",
    "date_sent",
    "topic",
    "description",
    "keywords",
    "authors",
    "custodian",
    "recipients",
    "artifact_types",
    "artifact_count",
];

impl DocumentRow for DocumentSummary {
    fn columns() -> &'static [&'static str] {
        SUMMARY_COLS
    }

    fn from_row(batch: &RecordBatch, i: usize) -> Result<Self> {
        Ok(Self {
            id: row_opt_str(batch, "id", i)?.unwrap_or_default(),
            bn: row_opt_str(batch, "bn", i)?,
            title: row_opt_str(batch, "title", i)?,
            date_sent: row_opt_str(batch, "date_sent", i)?,
            topic: row_opt_str(batch, "topic", i)?,
            description: row_opt_str(batch, "description", i)?,
            keywords: row_opt_str(batch, "keywords", i)?,
            authors: row_list(batch, "authors", i)?,
            custodian: row_list(batch, "custodian", i)?,
            recipients: row_list(batch, "recipients", i)?,
            artifact_types: row_list(batch, "artifact_types", i)?,
            artifact_count: row_u64(batch, "artifact_count", i)?,
        })
    }

    fn id(&self) -> &str {
        &self.id
    }
}

impl SearchableRow for DocumentSummary {
    fn searchable_fields(&self) -> Vec<(&'static str, String)> {
        fn opt(s: &Option<String>) -> String {
            s.clone().unwrap_or_default()
        }
        vec![
            ("title", opt(&self.title)),
            ("bn", opt(&self.bn)),
            ("topic", opt(&self.topic)),
            ("description", opt(&self.description)),
            ("keywords", opt(&self.keywords)),
            ("authors", self.authors.join(" ")),
            ("custodian", self.custodian.join(" ")),
            ("recipients", self.recipients.join(" ")),
        ]
    }

    fn artifact_types(&self) -> &[String] {
        &self.artifact_types
    }
}

impl DocumentRow for Document {
    fn columns() -> &'static [&'static str] {
        DOC_COLS
    }

    fn from_row(batch: &RecordBatch, i: usize) -> Result<Self> {
        Ok(Self {
            id: row_opt_str(batch, "id", i)?.unwrap_or_default(),
            bn: row_opt_str(batch, "bn", i)?,
            title: row_opt_str(batch, "title", i)?,
            industry: row_opt_str(batch, "industry", i)?,
            collection: row_opt_str(batch, "collection", i)?,
            genre: row_opt_str(batch, "genre", i)?,
            date_sent: row_opt_str(batch, "date_sent", i)?,
            date_received: row_opt_str(batch, "date_received", i)?,
            topic: row_opt_str(batch, "topic", i)?,
            description: row_opt_str(batch, "description", i)?,
            keywords: row_opt_str(batch, "keywords", i)?,
            conversation: row_opt_str(batch, "conversation", i)?,
            custodian: row_list(batch, "custodian", i)?,
            authors: row_list(batch, "authors", i)?,
            recipients: row_list(batch, "recipients", i)?,
            cc: row_list(batch, "cc", i)?,
            attachments: row_list(batch, "attachments", i)?,
            related: row_list(batch, "related", i)?,
            mentions: row_list(batch, "mentions", i)?,
            artifact_types: row_list(batch, "artifact_types", i)?,
            artifact_count: row_u64(batch, "artifact_count", i)?,
        })
    }

    fn id(&self) -> &str {
        &self.id
    }
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
pub struct GraphEdge {
    /// Document id the edge originates from.
    pub from_id: String,
    /// Relationship type.
    pub kind: RelationKind,
    /// The Bates/conversation reference that produced this edge.
    pub reference: String,
    /// Id of the resolved neighbor in `RelatedGraph::nodes`, if it exists in the index.
    pub neighbor_id: Option<String>,
    /// BFS depth at which this edge was discovered (1 = direct).
    pub depth: u32,
}

/// A document relationship graph returned by [`CorpusQueries::related`].
///
/// Nodes are deduplicated: a document referenced by multiple edges appears once
/// in `nodes` and is pointed to by `neighbor_id` on each edge.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RelatedGraph {
    /// All resolved documents in the graph, keyed by document id.
    pub nodes: std::collections::HashMap<String, Document>,
    /// Typed edges discovered during BFS traversal.
    pub edges: Vec<GraphEdge>,
}
