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

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use arrow::array::{
    Array, AsArray, Int64Array, Int64Builder, ListArray, ListBuilder, RecordBatch, StringArray,
    StringBuilder, StructArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use futures::TryStreamExt;
use lancedb::index::Index as LanceIndex;
use lancedb::index::scalar::BTreeIndexBuilder;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::{Connection, Table};

use crate::config::Config;
use crate::index::{ARTIFACTS_TABLE, DOCUMENTS_TABLE};
use crate::schema;

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

/// Ingest document and artifact metadata from the parquet into LanceDB.
///
/// Pass `force` to replace existing `documents`/`artifacts` tables (this also
/// drops any full-text `chunks`/`_meta` tables, since they are derived from the
/// same source and would otherwise be left stale).
pub async fn ingest_metadata(config: &Config, force: bool) -> Result<MetadataStats> {
    if !config.parquet_path.exists() {
        bail!("parquet {} not found", config.parquet_path.display());
    }

    let db = connect(config).await?;
    let existing = db.table_names().execute().await.context("listing tables")?;
    let have_meta = existing.iter().any(|n| n == DOCUMENTS_TABLE);
    if have_meta && !force {
        bail!(
            "index already exists at {}; pass force to re-ingest",
            config.lance_path.display()
        );
    }
    if force {
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
    }

    let ctx = session_context()?;
    let parquet = config
        .parquet_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("parquet_path is not valid UTF-8"))?;
    ctx.register_parquet("src", parquet, ParquetReadOptions::default())
        .await
        .with_context(|| format!("registering parquet {parquet}"))?;
    schema::validate_registered(&ctx).await?;

    tracing::info!("ingesting documents and artifacts (streaming)...");
    let stats = stream_ingest(&db, &ctx, config.ingest_buffer_bytes).await?;

    tracing::info!("creating indexes...");
    let docs_table = db
        .open_table(DOCUMENTS_TABLE)
        .execute()
        .await
        .context("opening documents table for indexing")?;
    create_indexes(&docs_table, &db).await?;

    tracing::info!(
        "ingest complete: {} documents, {} artifacts",
        stats.documents,
        stats.artifacts
    );
    Ok(stats)
}

/// Build the DataFusion context used to read the parquet.
fn session_context() -> Result<SessionContext> {
    // DataFusion 53 reads parquet string/binary columns as Arrow `Utf8View`/
    // `BinaryView` by default, but Lance's schema layer rejects view types
    // (LanceError(Schema): Unsupported data type: Utf8View). Force plain
    // `Utf8`/`Binary` so the ingest batches land in a Lance-compatible schema.
    let mut cfg = SessionConfig::new();
    cfg.options_mut().execution.parquet.schema_force_view_types = false;
    Ok(SessionContext::new_with_config(cfg))
}

/// Stream the source parquet, transforming each batch into `documents` and
/// `artifacts` rows and appending them to their respective Lance tables.
async fn stream_ingest(
    db: &Connection,
    ctx: &SessionContext,
    flush_bytes: usize,
) -> Result<MetadataStats> {
    let df = ctx.sql(STREAM_SQL).await.context("planning ingest query")?;
    let mut stream = df.execute_stream().await.context("executing ingest query")?;

    let mut docs = TableWriter::new(db, DOCUMENTS_TABLE, flush_bytes);
    let mut arts = TableWriter::new(db, ARTIFACTS_TABLE, flush_bytes);

    while let Some(batch) = stream
        .try_next()
        .await
        .context("reading ingest result batch")?
    {
        if batch.num_rows() == 0 {
            continue;
        }
        let artifact = batch
            .column(COL_ARTIFACT)
            .as_list_opt::<i32>()
            .ok_or_else(|| anyhow!("`artifact` column is not a List<Struct>"))?;
        docs.push(build_documents_batch(&batch, artifact)?).await?;
        arts.push(build_artifacts_batch(&batch, artifact)?).await?;
    }

    let documents = docs.finish().await?;
    let artifacts = arts.finish().await?;
    if documents == 0 {
        bail!("ingest produced no documents");
    }
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

/// Column indices in [`STREAM_SQL`]'s output. Columns `0..=COL_LAST_META` are
/// document metadata carried through unchanged; the rest are derived.
const COL_LAST_META: usize = 18; // id (0) .. mentions (18)
const COL_SEARCH_TEXT: usize = 19;
const COL_ARTIFACT: usize = 20;

/// Build a `documents` batch: the metadata columns carried through, the FTS
/// `search_text` column, and the artifact list summarized into `artifact_types`
/// (distinct media types) and `artifact_count`.
fn build_documents_batch(input: &RecordBatch, artifact: &ListArray) -> Result<RecordBatch> {
    let n = input.num_rows();
    let mut types = ListBuilder::new(StringBuilder::new());
    let mut count = Int64Builder::with_capacity(n);

    for row in 0..n {
        if artifact.is_null(row) {
            count.append_value(0);
            types.append(true); // empty (non-null) media-type list
            continue;
        }
        let elem = artifact.value(row);
        let structs = elem
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| anyhow!("`artifact` element is not a Struct"))?;
        count.append_value(structs.len() as i64);
        let media = struct_str(structs, "mediaType")?;
        let mut seen: HashSet<&str> = HashSet::new();
        for j in 0..structs.len() {
            if media.is_null(j) {
                continue;
            }
            let v = media.value(j);
            if seen.insert(v) {
                types.values().append_value(v);
            }
        }
        types.append(true);
    }

    // Schema: carried metadata fields, then the three derived columns. (Column
    // order is immaterial downstream — readers select by name — but kept stable
    // so every batch shares one schema.)
    let in_schema = input.schema();
    let mut fields: Vec<Arc<Field>> = (0..=COL_LAST_META)
        .map(|i| in_schema.field(i).clone().into())
        .collect();
    fields.push(Arc::new(Field::new(
        "artifact_types",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        true,
    )));
    fields.push(Arc::new(Field::new("artifact_count", DataType::Int64, false)));
    fields.push(in_schema.field(COL_SEARCH_TEXT).clone().into());

    let mut columns = input.columns()[..=COL_LAST_META].to_vec();
    columns.push(Arc::new(types.finish()));
    columns.push(Arc::new(count.finish()));
    columns.push(input.column(COL_SEARCH_TEXT).clone());

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .context("assembling documents batch")
}

/// Build an `artifacts` batch by exploding each document's inline artifact list
/// into one row per artifact: `(id, name, media_type, size, md5)`.
fn build_artifacts_batch(input: &RecordBatch, artifact: &ListArray) -> Result<RecordBatch> {
    let n = input.num_rows();
    let ids = input
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("`id` column is not Utf8"))?;

    let mut out_id = StringBuilder::new();
    let mut out_name = StringBuilder::new();
    let mut out_media = StringBuilder::new();
    let mut out_size = Int64Builder::new();
    let mut out_md5 = StringBuilder::new();

    for row in 0..n {
        if artifact.is_null(row) {
            continue;
        }
        let elem = artifact.value(row);
        let structs = elem
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| anyhow!("`artifact` element is not a Struct"))?;
        let name = struct_str(structs, "name")?;
        let media = struct_str(structs, "mediaType")?;
        let md5 = struct_str(structs, "md5")?;
        let size = struct_i64(structs, "size")?;
        let id = (!ids.is_null(row)).then(|| ids.value(row));
        for j in 0..structs.len() {
            append_opt(&mut out_id, id);
            append_opt(&mut out_name, (!name.is_null(j)).then(|| name.value(j)));
            append_opt(&mut out_media, (!media.is_null(j)).then(|| media.value(j)));
            append_opt(&mut out_md5, (!md5.is_null(j)).then(|| md5.value(j)));
            if size.is_null(j) {
                out_size.append_null();
            } else {
                out_size.append_value(size.value(j));
            }
        }
    }

    let fields = vec![
        Field::new("id", DataType::Utf8, true),
        Field::new("name", DataType::Utf8, true),
        Field::new("media_type", DataType::Utf8, true),
        Field::new("size", DataType::Int64, true),
        Field::new("md5", DataType::Utf8, true),
    ];
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        vec![
            Arc::new(out_id.finish()),
            Arc::new(out_name.finish()),
            Arc::new(out_media.finish()),
            Arc::new(out_size.finish()),
            Arc::new(out_md5.finish()),
        ],
    )
    .context("assembling artifacts batch")
}

/// Borrow a named `Utf8` field from an artifact struct array.
fn struct_str<'a>(structs: &'a StructArray, field: &str) -> Result<&'a StringArray> {
    structs
        .column_by_name(field)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| anyhow!("artifact struct missing `{field}: Utf8`"))
}

/// Borrow a named `Int64` field from an artifact struct array.
fn struct_i64<'a>(structs: &'a StructArray, field: &str) -> Result<&'a Int64Array> {
    structs
        .column_by_name(field)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| anyhow!("artifact struct missing `{field}: Int64`"))
}

/// Append an optional string, writing null when absent.
fn append_opt(builder: &mut StringBuilder, value: Option<&str>) {
    match value {
        Some(v) => builder.append_value(v),
        None => builder.append_null(),
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

/// Stream the source parquet, projecting/renaming the document metadata columns,
/// deriving the FTS `search_text`, and carrying the inline `artifact` struct
/// list through for the Rust pass to summarize and explode. Column order here
/// must match the `COL_*` indices above.
const STREAM_SQL: &str = "\
    SELECT \
      id, bn, \
      coalesce(ti, filename) AS title, \
      industry, collection, genre, \
      datesent AS date_sent, datereceived AS date_received, \
      topic, \"desc\" AS description, kw AS keywords, conversation, \
      custodian, au AS authors, rc AS recipients, cc, \
      attachment AS attachments, related, men AS mentions, \
      concat_ws(' ', \
        coalesce(coalesce(ti, filename), ''), coalesce(bn, ''), coalesce(topic, ''), \
        coalesce(\"desc\", ''), coalesce(kw, ''), \
        array_to_string(au, ' '), array_to_string(custodian, ' '), \
        array_to_string(rc, ' ')) AS search_text, \
      artifact \
    FROM src";
