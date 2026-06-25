//! Parquet → LanceDB ingest.
//!
//! Loads the source parquet into the embedded LanceDB store using DataFusion —
//! the same query engine [`crate::sql`] serves ad-hoc SQL with, and the same one
//! LanceDB itself is built on, so there is a single ingest path and no external
//! database dependency.
//!
//! The source parquet holds one row per document, with the document's artifacts
//! inline as a `list<struct<md5, mediaType, name, size>>` column. Two tables are
//! produced from a single streaming pass over it:
//! - `documents`: one row per document — the metadata columns, an FTS-only
//!   `search_text` column concatenating the searchable fields, and the artifact
//!   list summarized into `artifact_types` (distinct media types) and
//!   `artifact_count`.
//! - `artifacts`: one row per artifact — the inline struct list exploded so
//!   artifact fields (`name`, `media_type`, …) can be indexed and queried.
//!
//! Because each row is already one document, there is no deduplication and no
//! aggregation: every output row is final the instant it is read, so the pass
//! streams batch-by-batch with bounded memory. The per-document `artifact_types`
//! summary and the artifact explode both require reaching into the struct list,
//! which SQL cannot express over a `list<struct>`, so each batch's artifact
//! column is walked once in Rust to produce both.
//!
//! DataFusion is pinned to the same version `lance` uses, so the arrow types it
//! produces match exactly the arrow LanceDB consumes.

//! Solr → LanceDB metadata ingest.
//!
//! Streams the archive Solr corpus into the embedded LanceDB store, mapping each
//! source document into two tables via [`crate::solr_map`]:
//! - `documents`: one row per document — the metadata columns, an FTS-only
//!   `search_text` column concatenating the searchable fields, the artifact list
//!   summarized into `artifact_types` (distinct media types) and
//!   `artifact_count`, plus the incremental-update columns `ddmudate` and
//!   `digest`.
//! - `artifacts`: one row per artifact — `(id, name, media_type, size, md5)`.
//!
//! The pass streams page-by-page with bounded memory (see [`TableWriter`]) and,
//! on completion, records the high-water `ddmudate` so a later incremental
//! update can resume from it. The sibling [`crate::hybrid`] module builds the
//! text/vector index over the same database from the `artifacts` table.

use anyhow::{Context, Result, bail};
use arrow::array::RecordBatch;
use indicatif::{ProgressBar, ProgressStyle};
use lancedb::index::Index as LanceIndex;
use lancedb::index::scalar::BTreeIndexBuilder;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::{Connection, Table};
use std::io::IsTerminal;

use crate::config::Config;
use crate::index::{ARTIFACTS_TABLE, DOCUMENTS_TABLE, write_watermark};
use crate::solr::CURSOR_START;
use crate::{solr_map, update};

/// Names of the text-search and metadata tables created by a full-text build,
/// dropped alongside metadata on a forced metadata re-ingest.
const FULLTEXT_TABLES: &[&str] = &["chunks", "_meta"];

/// Row counts produced by a metadata ingest.
#[derive(Debug, Clone, Copy)]
pub struct MetadataStats {
    /// Documents written to the `documents` table.
    pub documents: u64,
    /// Artifacts written to the `artifacts` table.
    pub artifacts: u64,
}

/// Open (or create) the LanceDB database at the configured path.
pub(crate) async fn connect(config: &Config) -> Result<Connection> {
    let path = config
        .lance_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("lance_path is not valid UTF-8"))?;
    lancedb::connect(path)
        .execute()
        .await
        .with_context(|| format!("connecting to LanceDB at {path}"))
}

/// Ingest document and artifact metadata from the Solr source into LanceDB.
///
/// This is the canonical metadata ingest now that the parquet route is retired:
/// it streams the Solr corpus by `cursorMark` and maps each document into the
/// `documents`/`artifacts` tables via [`crate::solr_map`], producing the same
/// schema the serving path and full-text build expect (plus the new
/// `ddmudate`/`digest` columns). It rebuilds those tables from scratch — and
/// drops the derived `chunks`/`_meta` tables, which a `--full-text` pass
/// regenerates.
///
/// `since` (an inclusive `ddmudate` lower bound) is intended for testing on a
/// small window; omit it for a complete rebuild.
///
/// Refuses to overwrite an existing metadata index unless `force` is set, and
/// only drops the current tables *after* Solr has returned its first page, so a
/// misconfiguration (e.g. a missing `solr_url`) or a connectivity failure never
/// destroys a usable index.
pub async fn ingest_from_solr(
    config: &Config,
    since: Option<&str>,
    force: bool,
) -> Result<MetadataStats> {
    let db = connect(config).await?;
    let existing = db.table_names().execute().await.context("listing tables")?;

    // Never replace an existing metadata index implicitly; require --force.
    if !force && existing.iter().any(|n| n == DOCUMENTS_TABLE) {
        bail!(
            "a metadata index already exists at {}; pass --force to rebuild it, \
             or use `update --apply` for an incremental update",
            config.lance_path.display()
        );
    }

    // Build the Solr client and fetch the first page *before* dropping anything,
    // so a misconfiguration or a connectivity failure leaves the current index
    // intact.
    let client = update::solr_client(config)?;
    let mut page = client
        .scan_page(since, CURSOR_START, solr_map::SOURCE_FIELDS)
        .await
        .context("fetching first Solr page")?;
    if page.docs.is_empty() {
        bail!("solr ingest produced no documents");
    }

    // Solr responded with data: it is now safe to replace the tables.
    for name in [DOCUMENTS_TABLE, ARTIFACTS_TABLE]
        .iter()
        .chain(FULLTEXT_TABLES.iter())
    {
        if existing.iter().any(|n| n == *name) {
            db.drop_table(name, &[])
                .await
                .with_context(|| format!("dropping table {name}"))?;
        }
    }

    let mut docs = TableWriter::new(&db, DOCUMENTS_TABLE, config.ingest_buffer_bytes);
    let mut arts = TableWriter::new(&db, ARTIFACTS_TABLE, config.ingest_buffer_bytes);

    tracing::info!("ingesting documents and artifacts from solr (streaming)...");
    // On a TTY render a live progress bar; off a TTY indicatif hides it, so the
    // periodic `tracing` lines below keep pod logs informative.
    let tty = std::io::stderr().is_terminal();
    let bar = ProgressBar::new(page.num_found);
    bar.set_style(
        ProgressStyle::with_template(
            "{prefix:<9} {bar:28.yellow/blue} {pos:>9}/{len:>9} docs │ {per_sec}",
        )
        .expect("valid template")
        .progress_chars("=>-"),
    );
    bar.set_prefix("documents");
    let mut cursor = CURSOR_START.to_string();
    let mut scanned: u64 = 0;
    let mut watermark: Option<String> = None;
    loop {
        if page.docs.is_empty() {
            break;
        }
        scanned += page.docs.len() as u64;
        for doc in &page.docs {
            if let Some(m) = solr_map::doc_modified(doc, &config.solr_modified_field)
                && watermark.as_deref().is_none_or(|cur| m.as_str() > cur)
            {
                watermark = Some(m);
            }
        }
        docs.push(solr_map::documents_batch(
            &page.docs,
            &config.solr_modified_field,
        )?)
        .await?;
        arts.push(solr_map::artifacts_batch(&page.docs)?).await?;
        bar.set_length(page.num_found);
        bar.set_position(scanned);
        if !tty {
            tracing::info!("scanned {scanned}/{} documents", page.num_found);
        }

        if page.next_cursor.is_empty() || page.next_cursor == cursor {
            break;
        }
        cursor = page.next_cursor.clone();
        page = client
            .scan_page(since, &cursor, solr_map::SOURCE_FIELDS)
            .await?;
    }
    bar.finish();

    let documents = docs.finish().await?;
    let artifacts = arts.finish().await?;
    if documents == 0 {
        bail!("solr ingest produced no documents");
    }

    tracing::info!("creating indexes...");
    let docs_table = db
        .open_table(DOCUMENTS_TABLE)
        .execute()
        .await
        .context("opening documents table for indexing")?;
    create_indexes(&docs_table, &db).await?;

    // Record the high-water `ddmudate` so the next incremental update resumes
    // from it. Written last, after the tables and indexes are durable.
    if let Some(w) = &watermark {
        write_watermark(&db, w).await?;
    }

    tracing::info!("solr ingest complete: {documents} documents, {artifacts} artifacts");
    Ok(MetadataStats {
        documents,
        artifacts,
    })
}

/// Buffers output batches and flushes them to a Lance table once their in-memory
/// size reaches `flush_bytes`, creating the table from the first flush. A
/// byte-based threshold keeps fragment sizes uniform across tables whose rows
/// differ in width (a document row is far wider than an artifact row).
struct TableWriter<'a> {
    db: &'a Connection,
    name: &'static str,
    flush_bytes: usize,
    table: Option<Table>,
    buf: Vec<RecordBatch>,
    buf_bytes: usize,
    total: u64,
}

impl<'a> TableWriter<'a> {
    fn new(db: &'a Connection, name: &'static str, flush_bytes: usize) -> Self {
        Self {
            db,
            name,
            flush_bytes,
            table: None,
            buf: Vec::new(),
            buf_bytes: 0,
            total: 0,
        }
    }

    /// Buffer a batch, flushing once the buffer reaches `flush_bytes`.
    async fn push(&mut self, batch: RecordBatch) -> Result<()> {
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(());
        }
        self.total += rows as u64;
        self.buf_bytes += batch.get_array_memory_size();
        self.buf.push(batch);
        if self.buf_bytes >= self.flush_bytes {
            self.flush().await?;
        }
        Ok(())
    }

    /// Write any buffered batches as a single Lance fragment.
    async fn flush(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let batches = std::mem::take(&mut self.buf);
        self.buf_bytes = 0;
        match &self.table {
            Some(t) => {
                t.add(batches)
                    .execute()
                    .await
                    .with_context(|| format!("appending to {}", self.name))?;
            }
            None => {
                let t = self
                    .db
                    .create_table(self.name, batches)
                    .execute()
                    .await
                    .with_context(|| format!("creating table {}", self.name))?;
                self.table = Some(t);
            }
        }
        Ok(())
    }

    /// Flush the remainder and return the total rows written.
    async fn finish(mut self) -> Result<u64> {
        self.flush().await?;
        Ok(self.total)
    }
}

/// Create the FTS and scalar indexes the serving queries rely on.
async fn create_indexes(documents: &Table, db: &Connection) -> Result<()> {
    documents
        .create_index(&["search_text"], LanceIndex::FTS(FtsIndexBuilder::default()))
        .execute()
        .await
        .context("creating documents FTS index")?;
    for col in ["id", "bn", "conversation"] {
        documents
            .create_index(&[col], LanceIndex::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await
            .with_context(|| format!("creating documents.{col} index"))?;
    }
    let artifacts = db
        .open_table(ARTIFACTS_TABLE)
        .execute()
        .await
        .context("opening artifacts table for indexing")?;
    for col in ["id", "name", "media_type"] {
        artifacts
            .create_index(&[col], LanceIndex::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await
            .with_context(|| format!("creating artifacts.{col} index"))?;
    }
    Ok(())
}

