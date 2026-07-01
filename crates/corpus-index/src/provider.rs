//! The ingest boundary: a corpus source, decoupled from LanceDB.
//!
//! A [`SourceProvider`] streams a corpus as pages of already-built Arrow
//! batches ([`SourcePage`]); the framework's [`build_metadata`](crate::ingest::build_metadata)
//! driver owns everything downstream (table writes, index creation, watermark).
//! Yielding *batches per page* — not per-row callbacks — keeps the provider's
//! per-document mapping hot loop monomorphic inside the impl, with zero dynamic
//! dispatch crossing the boundary.

use anyhow::Result;
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use futures::Stream;

/// What the framework must know about a provider's `documents` table.
///
/// The provider owns its whole document schema *in code*; the framework fixes
/// only that a `documents` table exists, has an `id`, and declares which column
/// feeds full-text search and which scalar columns to index.
pub struct DocumentsContract {
    /// The `documents` Arrow schema. Must contain a Utf8 `id` column.
    pub schema: SchemaRef,
    /// Column the FTS index is built over (e.g. `search_text`).
    pub fts_column: &'static str,
    /// `documents` columns to build scalar (BTree) indexes on.
    pub scalar_index_cols: &'static [&'static str],
}

/// One page of a corpus scan: document and artifact rows, already mapped to the
/// framework-consumable Arrow batches.
///
/// The same page type drives both the full build (which ignores `redacted`) and
/// the incremental apply (which uses it plus the `digest` column inside
/// `documents`).
pub struct SourcePage {
    /// `documents` rows for this page, matching [`DocumentsContract::schema`]
    /// (including the `digest` column).
    pub documents: RecordBatch,
    /// `artifacts` rows for this page: `(id, name, media_type, size, md5)`.
    pub artifacts: RecordBatch,
    /// Per-document *policy* withdrawal flag, aligned to `documents` rows.
    /// All-false for sources with no withdrawal concept; the framework derives
    /// *structural* "no artifacts" redaction itself. Ignored by the full build.
    pub redacted: Vec<bool>,
    /// Greatest watermark value seen on this page (e.g. max modified-date), or
    /// `None` when the source has no watermark concept.
    pub watermark: Option<String>,
    /// Total documents the source reports for the query (same on every page).
    pub num_found: u64,
}

/// A corpus source the framework can ingest and incrementally update.
///
/// A single `scan` drives both the full build and the incremental apply — the
/// full build ignores [`SourcePage::redacted`], the incremental path uses it.
pub trait SourceProvider {
    /// The provider's `documents` contract (schema + index declarations).
    fn documents_contract(&self) -> &DocumentsContract;
    /// Stream the corpus at or after `since` (an inclusive watermark lower
    /// bound), a page of Arrow batches at a time.
    fn scan(&self, since: Option<&str>) -> impl Stream<Item = Result<SourcePage>> + Send;
}
