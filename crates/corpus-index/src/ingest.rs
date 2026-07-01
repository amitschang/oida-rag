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
use futures::TryStreamExt;
use lancedb::index::Index as LanceIndex;
use lancedb::index::scalar::BTreeIndexBuilder;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::{Connection, Table};
use std::io::IsTerminal;

use crate::config::CoreConfig;
use crate::index::{ARTIFACTS_TABLE, DOCUMENTS_TABLE, has_table, track_max, write_watermark};
use crate::progress::documents_bar;
use crate::provider::{DocumentsContract, SourceProvider};

/// Names of the text-search and metadata tables created by a full-text build,
/// dropped alongside metadata on a forced metadata re-ingest.
const FULLTEXT_TABLES: &[&str] = &[crate::index::CHUNKS_TABLE, crate::hybrid::META_TABLE];

/// Row counts produced by a metadata ingest.
#[derive(Debug, Clone, Copy)]
pub struct MetadataStats {
    /// Documents written to the `documents` table.
    pub documents: u64,
    /// Artifacts written to the `artifacts` table.
    pub artifacts: u64,
}

/// Open (or create) the LanceDB database at the configured path.
pub(crate) async fn connect(config: &CoreConfig) -> Result<Connection> {
    let path = config
        .lance_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("lance_path is not valid UTF-8"))?;
    lancedb::connect(path)
        .execute()
        .await
        .with_context(|| format!("connecting to LanceDB at {path}"))
}

/// Build the `documents`/`artifacts` metadata tables from a [`SourceProvider`],
/// from scratch.
///
/// Streams the provider page by page ([`SourcePage`](crate::provider::SourcePage))
/// into the two tables with bounded memory, then builds the FTS + scalar indexes
/// the provider's [`DocumentsContract`] declares and records the high-water
/// watermark so a later incremental update can resume from it. Also drops the
/// derived `chunks`/`_meta` tables, which a `--full-text` pass regenerates.
///
/// `since` (an inclusive watermark lower bound) is intended for testing on a
/// small window; omit it for a complete rebuild.
///
/// Refuses to overwrite an existing metadata index unless `force` is set, and
/// only drops the current tables *after* the provider has returned its first
/// page, so a misconfiguration or a connectivity failure never destroys a
/// usable index.
pub async fn build_metadata<P: SourceProvider>(
    provider: &P,
    config: &CoreConfig,
    since: Option<&str>,
    force: bool,
) -> Result<MetadataStats> {
    let db = connect(config).await?;

    // Never replace an existing metadata index implicitly; require --force.
    if !force && has_table(&db, DOCUMENTS_TABLE).await? {
        bail!(
            "a metadata index already exists at {}; pass --force to rebuild it, \
             or run `ingest` (without --force) for an incremental update",
            config.lance_path.display()
        );
    }

    // Pull the first page *before* dropping anything, so a misconfiguration or a
    // connectivity failure leaves the current index intact.
    let mut stream = std::pin::pin!(provider.scan(since));
    let Some(mut page) = stream.try_next().await.context("fetching first source page")? else {
        bail!("source produced no documents");
    };

    // The source responded with data: it is now safe to replace the tables.
    for name in [DOCUMENTS_TABLE, ARTIFACTS_TABLE]
        .iter()
        .chain(FULLTEXT_TABLES.iter())
    {
        match db.drop_table(name, &[]).await {
            Ok(()) | Err(lancedb::Error::TableNotFound { .. }) => {}
            Err(e) => return Err(e).with_context(|| format!("dropping table {name}")),
        }
    }

    let mut docs = TableWriter::new(&db, DOCUMENTS_TABLE, config.ingest_buffer_bytes);
    let mut arts = TableWriter::new(&db, ARTIFACTS_TABLE, config.ingest_buffer_bytes);

    tracing::info!("ingesting documents and artifacts from source (streaming)...");
    // On a TTY render a live progress bar; off a TTY indicatif hides it, so the
    // periodic `tracing` lines below keep pod logs informative.
    let tty = std::io::stderr().is_terminal();
    let bar = documents_bar(page.num_found);
    let mut scanned: u64 = 0;
    let mut watermark: Option<String> = None;
    loop {
        scanned += page.documents.num_rows() as u64;
        if let Some(w) = page.watermark.take() {
            track_max(&mut watermark, w);
        }
        docs.push(page.documents).await?;
        arts.push(page.artifacts).await?;
        bar.set_length(page.num_found);
        bar.set_position(scanned);
        if !tty {
            tracing::info!("scanned {scanned}/{} documents", page.num_found);
        }

        match stream.try_next().await? {
            Some(next) => page = next,
            None => break,
        }
    }
    bar.finish();

    let documents = docs.finish().await?;
    let artifacts = arts.finish().await?;
    if documents == 0 {
        bail!("source produced no documents");
    }

    tracing::info!("creating indexes...");
    let docs_table = db
        .open_table(DOCUMENTS_TABLE)
        .execute()
        .await
        .context("opening documents table for indexing")?;
    create_indexes(&docs_table, &db, provider.documents_contract()).await?;

    // Record the high-water watermark so the next incremental update resumes
    // from it. Written last, after the tables and indexes are durable.
    if let Some(w) = &watermark {
        write_watermark(&db, w).await?;
    }

    tracing::info!("metadata ingest complete: {documents} documents, {artifacts} artifacts");
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

/// Create the FTS and scalar indexes the serving queries rely on. The
/// `documents` FTS column and scalar-index columns come from the provider's
/// [`DocumentsContract`]; the `artifacts` indexes are framework-fixed.
async fn create_indexes(
    documents: &Table,
    db: &Connection,
    contract: &DocumentsContract,
) -> Result<()> {
    documents
        .create_index(
            &[contract.fts_column],
            LanceIndex::FTS(FtsIndexBuilder::default()),
        )
        .execute()
        .await
        .context("creating documents FTS index")?;
    for col in contract.scalar_index_cols {
        documents
            .create_index(&[*col], LanceIndex::BTree(BTreeIndexBuilder::default()))
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

