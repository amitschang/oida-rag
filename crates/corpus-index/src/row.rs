//! The read-side document contract: how a corpus row decodes into a typed
//! result.
//!
//! This is the dual of the write-side contract (the source provider maps
//! records → Arrow batches): here the provider maps an Arrow row back into its
//! own type. The framework's generic readers ([`crate::index::Index::get`],
//! [`get_many`](crate::index::Index::get_many), [`search`](crate::index::Index::search))
//! are written against these traits, so the engine never names a concrete
//! document type.

use anyhow::Result;
use arrow::array::RecordBatch;
use schemars::JsonSchema;
use serde::Serialize;

/// A type the framework can read out of the `documents` table.
///
/// The provider declares the projection it needs ([`columns`](DocumentRow::columns))
/// and how to decode one row ([`from_row`](DocumentRow::from_row)); the engine
/// honors the projection so a lean result type scans fewer columns.
pub trait DocumentRow: Serialize + JsonSchema + Send + 'static {
    /// Columns to project when reading this type (a narrow scan).
    fn columns() -> &'static [&'static str];
    /// Decode row `i` of `batch` (which was selected with [`columns`](DocumentRow::columns)).
    fn from_row(batch: &RecordBatch, i: usize) -> Result<Self>
    where
        Self: Sized;
    /// The document id of a decoded row (used to preserve/restore ordering).
    fn id(&self) -> &str;
}

/// A [`DocumentRow`] usable as a metadata-search result.
///
/// `searchable_fields` powers the term-count ranking and `matched_fields`
/// provenance: it is the one place that enumerates a corpus's document fields
/// by name, and it lives with the provider's type rather than in the engine.
pub trait SearchableRow: DocumentRow {
    /// `(field name, text)` pairs the metadata search scores and reports on.
    fn searchable_fields(&self) -> Vec<(&'static str, String)>;
    /// The distinct artifact media types of the document — the framework-fixed
    /// manifest projection the `media_type` search filter tests against.
    fn artifact_types(&self) -> &[String];
}
