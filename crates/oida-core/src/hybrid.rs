//! LanceDB-backed hybrid (keyword + semantic) search over artifact text.
//!
//! The metadata index ([`crate::Index`]) answers questions about a document's
//! fields. This module answers questions about what a document *says*: it reads
//! the plain-text (OCR) artifacts, splits them into overlapping chunks, embeds
//! each chunk with an Ollama model, and stores both the text and its vector in
//! a single LanceDB table. Queries then run a full-text search and a vector
//! search in parallel and fuse the two rankings with Reciprocal Rank Fusion
//! (RRF), collapsing chunk hits back to their parent document.
//!
//! # Embedding-model consistency
//!
//! A vector index is only meaningful when queries are embedded with the *same*
//! model that produced the stored vectors. To guarantee that, the build writes
//! a `_meta` row recording the embed model name, its vector dimension, and the
//! model's content digest. At query time we read that row back and embed the
//! query with the *stored* model name (never the live config), assert the
//! dimension matches, and verify the model's current digest still equals the
//! stored one — refusing to serve stale results if the model changed under us.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use futures::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::index::Index as LanceIndex;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::rerankers::rrf::RRFReranker;
use lancedb::table::{CompactionOptions, OptimizeAction};
use lancedb::{Connection, Table};

use crate::artifacts::{ArtifactTextStatus, read_artifact_text};
use crate::config::Config;
use crate::embed::Embedder;
use crate::index::Index;
use crate::ingest::connect;

/// Name of the table holding text chunks and their embeddings.
const CHUNKS_TABLE: &str = "chunks";
/// Name of the single-row table holding index metadata.
const META_TABLE: &str = "_meta";
/// Number of chunks embedded per Ollama request during a build.
const EMBED_BATCH: usize = 64;
/// Below this row count a vector (ANN) index is skipped; flat search is exact
/// and fast enough, and IVF/PQ training needs a reasonable number of rows.
const MIN_VECTOR_INDEX_ROWS: usize = 256;

/// A single fused chunk hit, before hydration into a full document.
#[derive(Debug, Clone)]
pub struct HybridChunkHit {
    /// Document id the matching chunk belongs to.
    pub doc_id: String,
    /// RRF-fused relevance score (higher is better).
    pub score: f32,
    /// Name of the text artifact the chunk came from.
    pub artifact_name: String,
    /// A short single-line excerpt of the matching chunk.
    pub snippet: String,
}

/// Stored index metadata plus live counts.
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Distinct documents represented in the index.
    pub documents: u64,
    /// Total text chunks stored.
    pub chunks: u64,
    /// Embedding vector dimension.
    pub dim: usize,
    /// Ollama model used to produce the embeddings.
    pub embed_model: String,
    /// Content digest of the embed model at build time.
    pub model_digest: String,
    /// Unix seconds when the index was built.
    pub built_at: i64,
    /// Chunk size (bytes) used when splitting text.
    pub chunk_bytes: usize,
    /// Chunk overlap (bytes) used when splitting text.
    pub chunk_overlap: usize,
}

/// Index metadata read back from the `_meta` table.
#[derive(Debug, Clone)]
struct Meta {
    embed_model: String,
    dim: usize,
    model_digest: String,
    built_at: i64,
    chunk_bytes: usize,
    chunk_overlap: usize,
    documents: u64,
    chunks: u64,
}

/// A handle to the hybrid text index for serving queries.
pub struct HybridIndex {
    table: Table,
    embedder: Embedder,
    meta: Meta,
}

impl HybridIndex {
    /// Open an existing hybrid index for querying.
    ///
    /// The embedder is bound to the model recorded in the index metadata, not
    /// to [`Config::embed_model`], so a query can never use a model that
    /// disagrees with the stored vectors.
    pub async fn open(config: &Config) -> Result<Self> {
        let db = connect(config).await?;
        let names = db.table_names().execute().await.context("listing tables")?;
        if !names.iter().any(|n| n == CHUNKS_TABLE) {
            bail!(
                "hybrid index not found at {}; build it first (oida-cli ingest --full-text)",
                config.lance_path.display()
            );
        }
        let meta = read_meta(&db).await?;
        let table = db
            .open_table(CHUNKS_TABLE)
            .execute()
            .await
            .context("opening chunks table")?;
        let embedder = Embedder::new(&config.ollama_host, meta.embed_model.clone())?;
        Ok(Self {
            table,
            embedder,
            meta,
        })
    }

    /// Return index metadata and live row counts.
    pub async fn stats(&self) -> Result<IndexStats> {
        let chunks = self.table.count_rows(None).await.context("counting chunks")? as u64;
        Ok(IndexStats {
            documents: self.meta.documents,
            chunks,
            dim: self.meta.dim,
            embed_model: self.meta.embed_model.clone(),
            model_digest: self.meta.model_digest.clone(),
            built_at: self.meta.built_at,
            chunk_bytes: self.meta.chunk_bytes,
            chunk_overlap: self.meta.chunk_overlap,
        })
    }

    /// Run a hybrid keyword + semantic search, returning up to `limit`
    /// documents ranked by fused relevance.
    ///
    /// Guards against a silently changed embedding model: the query is embedded
    /// with the stored model name, its dimension is checked against the stored
    /// dimension, and the model's live digest is compared with the stored one.
    pub async fn query(&self, text: &str, limit: usize) -> Result<Vec<HybridChunkHit>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let limit = limit.max(1);

        // Refuse to serve if the model changed out from under the index.
        let live_digest = self
            .embedder
            .digest()
            .await
            .context("verifying embed model digest")?;
        if live_digest != self.meta.model_digest {
            bail!(
                "embed model '{}' has changed (digest {} != index digest {}); \
                 rebuild the index (oida-cli ingest --full-text --force)",
                self.meta.embed_model,
                live_digest,
                self.meta.model_digest
            );
        }

        let qvec = self.embedder.embed_one(text).await?;
        if qvec.len() != self.meta.dim {
            bail!(
                "query embedding has dimension {} but index dimension is {}",
                qvec.len(),
                self.meta.dim
            );
        }

        // Over-fetch chunks so that, after collapsing to one hit per document,
        // we still have enough distinct documents to satisfy `limit`.
        let candidates = limit.saturating_mul(5).max(limit);
        let batches: Vec<RecordBatch> = self
            .table
            .query()
            .nearest_to(qvec)
            .context("building vector query")?
            .full_text_search(FullTextSearchQuery::new(text.to_string()))
            .rerank(Arc::new(RRFReranker::default()))
            .select(Select::columns(&["doc_id", "artifact_name", "text"]))
            .limit(candidates)
            .execute()
            .await
            .context("executing hybrid search")?
            .try_collect()
            .await
            .context("collecting hybrid results")?;

        collapse_to_documents(&batches, limit)
    }
}

/// Build (or rebuild) the hybrid text index from the readable text artifacts.
///
/// Reads each `text/plain`/`.ocr` artifact, splits it into overlapping chunks,
/// embeds the chunks with `embedder`, and writes them to LanceDB along with a
/// `_meta` row pinning the embed model and its digest. Pass `force` to replace
/// an existing index.
pub async fn build(
    config: &Config,
    index: &Index,
    embedder: &Embedder,
    force: bool,
) -> Result<IndexStats> {
    if config.artifact_root.is_none() {
        bail!("artifact_root is not configured; the hybrid index needs the text files on disk");
    }

    let db = connect(config).await?;
    let existing = db.table_names().execute().await.context("listing tables")?;
    let have_index = existing.iter().any(|n| n == CHUNKS_TABLE);
    if have_index && !force {
        bail!(
            "hybrid index already exists at {}; pass force to rebuild",
            config.lance_path.display()
        );
    }
    if force {
        for name in [CHUNKS_TABLE, META_TABLE] {
            if existing.iter().any(|n| n == name) {
                db.drop_table(name, &[]).await.with_context(|| format!("dropping table {name}"))?;
            }
        }
    }

    // Resolve the model digest up front; this also fails fast if the model is
    // not installed before we do any expensive work.
    let model_digest = embedder.digest().await.context("reading embed model digest")?;

    let artifacts = index.text_artifacts().await.context("listing text artifacts")?;
    let total_artifacts = artifacts.len();
    tracing::info!("indexing text from {total_artifacts} artifacts");

    let mut table: Option<Table> = None;
    let mut dim: Option<usize> = None;
    // `pending` accumulates chunk rows for the next (small) Ollama embed
    // request; `buffer` accumulates the resulting embedded batches and is only
    // flushed to LanceDB once it crosses `write_buffer_bytes`, so each
    // `Table::add` writes one large fragment instead of one per embed batch.
    let mut pending: Vec<ChunkRow> = Vec::new();
    let mut buffer: Vec<RecordBatch> = Vec::new();
    let mut buffer_bytes: usize = 0;
    let mut chunk_count: u64 = 0;
    let mut doc_count: u64 = 0;
    let mut last_doc: Option<String> = None;
    // Throttle progress logging to once per this many artifacts scanned.
    const PROGRESS_EVERY: usize = 500;

    for (scanned, (doc_id, name, media_type)) in artifacts.iter().enumerate() {
        if last_doc.as_deref() != Some(doc_id.as_str()) {
            doc_count += 1;
            last_doc = Some(doc_id.clone());
        }
        if scanned > 0 && scanned % PROGRESS_EVERY == 0 {
            tracing::info!(
                "embedding progress: {scanned}/{total_artifacts} artifacts, \
                 {chunk_count} chunks embedded so far"
            );
        }
        // u64::MAX / 2 reads the whole file without risking an offset overflow.
        let loaded = read_artifact_text(config, name, media_type.as_deref(), 0, u64::MAX / 2);
        if loaded.status != ArtifactTextStatus::TextLoaded {
            continue;
        }
        let Some(body) = loaded.text else { continue };
        for (idx, chunk) in chunk_text(&body, config.chunk_bytes, config.chunk_overlap)
            .into_iter()
            .enumerate()
        {
            pending.push(ChunkRow {
                doc_id: doc_id.clone(),
                chunk_idx: idx as i32,
                artifact_name: name.clone(),
                text: chunk,
            });
        }
        if pending.len() >= EMBED_BATCH {
            chunk_count += pending.len() as u64;
            let batch = embed_rows(&mut dim, embedder, &pending).await?;
            pending.clear();
            buffer_bytes += batch.get_array_memory_size();
            buffer.push(batch);
            if buffer_bytes >= config.write_buffer_bytes {
                write_buffer(&db, &mut table, &mut buffer).await?;
                buffer_bytes = 0;
            }
        }
    }
    if !pending.is_empty() {
        chunk_count += pending.len() as u64;
        let batch = embed_rows(&mut dim, embedder, &pending).await?;
        pending.clear();
        buffer.push(batch);
    }
    if !buffer.is_empty() {
        write_buffer(&db, &mut table, &mut buffer).await?;
    }

    let table = table.ok_or_else(|| {
        anyhow::anyhow!("no readable text artifacts found; nothing to index")
    })?;
    let dim = dim.expect("dim is set whenever a batch was flushed");

    // Compact before indexing so the FTS/vector indexes build over a clean
    // fragment layout. With large write buffers there is usually little to do,
    // but this keeps the result insensitive to how the row counts fell.
    if config.compact_on_build {
        tracing::info!("compacting chunks table before indexing");
        let stats = table
            .optimize(OptimizeAction::Compact {
                options: CompactionOptions::default(),
                remap_options: None,
            })
            .await
            .context("compacting chunks table")?;
        if let Some(m) = stats.compaction {
            tracing::info!(
                "compaction: {} fragments removed, {} fragments added",
                m.fragments_removed,
                m.fragments_added
            );
        }
    }

    tracing::info!("creating full-text index on {chunk_count} chunks");
    table
        .create_index(&["text"], LanceIndex::FTS(FtsIndexBuilder::default()))
        .execute()
        .await
        .context("creating FTS index")?;

    if (chunk_count as usize) >= MIN_VECTOR_INDEX_ROWS {
        tracing::info!("creating vector index");
        if let Err(e) = table
            .create_index(&["vector"], LanceIndex::Auto)
            .execute()
            .await
        {
            tracing::warn!("vector index creation failed ({e}); queries will use flat search");
        }
    } else {
        tracing::info!("skipping vector index ({chunk_count} chunks < {MIN_VECTOR_INDEX_ROWS})");
    }

    let built_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let meta = Meta {
        embed_model: embedder.model().to_string(),
        dim,
        model_digest,
        built_at,
        chunk_bytes: config.chunk_bytes,
        chunk_overlap: config.chunk_overlap,
        documents: doc_count,
        chunks: chunk_count,
    };
    write_meta(&db, &meta).await?;

    Ok(IndexStats {
        documents: meta.documents,
        chunks: meta.chunks,
        dim: meta.dim,
        embed_model: meta.embed_model,
        model_digest: meta.model_digest,
        built_at: meta.built_at,
        chunk_bytes: meta.chunk_bytes,
        chunk_overlap: meta.chunk_overlap,
    })
}

/// A pending chunk awaiting embedding and write.
struct ChunkRow {
    doc_id: String,
    chunk_idx: i32,
    artifact_name: String,
    text: String,
}

/// Embed `rows` and append them to the chunks table, creating it on first use.
/// Embed a batch of chunk rows and assemble the Arrow [`RecordBatch`], tracking
/// the embedding dimension. This does not touch LanceDB — the caller buffers
/// the returned batches and flushes them with [`write_buffer`].
async fn embed_rows(
    dim: &mut Option<usize>,
    embedder: &Embedder,
    rows: &[ChunkRow],
) -> Result<RecordBatch> {
    let texts: Vec<String> = rows.iter().map(|r| r.text.clone()).collect();
    let vectors = embedder.embed(&texts).await?;
    let d = vectors[0].len();
    match dim {
        Some(existing) if *existing != d => {
            bail!("embedding dimension changed from {existing} to {d} mid-build")
        }
        None => *dim = Some(d),
        _ => {}
    }
    build_record_batch(rows, &vectors, d as i32)
}

/// Flush the buffered batches to LanceDB in a single `Table::add` (creating the
/// table on first call), then clear the buffer. One call writes one fragment.
async fn write_buffer(
    db: &Connection,
    table: &mut Option<Table>,
    buffer: &mut Vec<RecordBatch>,
) -> Result<()> {
    if buffer.is_empty() {
        return Ok(());
    }
    let batches = std::mem::take(buffer);
    match table {
        Some(t) => {
            t.add(batches).execute().await.context("appending chunk batches")?;
        }
        None => {
            let t = db
                .create_table(CHUNKS_TABLE, batches)
                .execute()
                .await
                .context("creating chunks table")?;
            *table = Some(t);
        }
    }
    Ok(())
}

/// Assemble an Arrow [`RecordBatch`] for a batch of chunk rows.
fn build_record_batch(rows: &[ChunkRow], vectors: &[Vec<f32>], dim: i32) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Utf8, false),
        Field::new("chunk_idx", DataType::Int32, false),
        Field::new("artifact_name", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            false,
        ),
    ]));

    let doc_ids = StringArray::from_iter_values(rows.iter().map(|r| r.doc_id.as_str()));
    let chunk_idxs = Int32Array::from_iter_values(rows.iter().map(|r| r.chunk_idx));
    let names = StringArray::from_iter_values(rows.iter().map(|r| r.artifact_name.as_str()));
    let texts = StringArray::from_iter_values(rows.iter().map(|r| r.text.as_str()));
    let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vectors
            .iter()
            .map(|v| Some(v.iter().map(|f| Some(*f)).collect::<Vec<_>>())),
        dim,
    );

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(doc_ids),
            Arc::new(chunk_idxs),
            Arc::new(names),
            Arc::new(texts),
            Arc::new(vector_array),
        ],
    )
    .context("assembling chunk record batch")
}

/// Collapse ranked chunk rows to one hit per document, preserving the fused
/// relevance order, and truncate to `limit` documents.
fn collapse_to_documents(batches: &[RecordBatch], limit: usize) -> Result<Vec<HybridChunkHit>> {
    let mut seen: HashMap<String, ()> = HashMap::new();
    let mut out: Vec<HybridChunkHit> = Vec::new();

    for batch in batches {
        let doc_ids = str_col(batch, "doc_id")?;
        let names = str_col(batch, "artifact_name")?;
        let texts = str_col(batch, "text")?;
        let scores = batch
            .column_by_name("_relevance_score")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        for row in 0..batch.num_rows() {
            let doc_id = doc_ids.value(row).to_string();
            if seen.contains_key(&doc_id) {
                continue;
            }
            seen.insert(doc_id.clone(), ());
            let score = scores.map(|s| s.value(row)).unwrap_or(0.0);
            out.push(HybridChunkHit {
                doc_id,
                score,
                artifact_name: names.value(row).to_string(),
                snippet: snippet(texts.value(row)),
            });
            if out.len() >= limit {
                return Ok(out);
            }
        }
    }
    Ok(out)
}

/// Downcast a named column to a [`StringArray`].
fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("result missing column {name}"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("column {name} is not a string column"))
}

/// Produce a short single-line excerpt from chunk text.
fn snippet(text: &str) -> String {
    const MAX: usize = 280;
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.len() <= MAX {
        return collapsed;
    }
    let mut end = MAX;
    while end > 0 && !collapsed.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &collapsed[..end])
}

/// Split text into overlapping byte-bounded chunks, respecting char boundaries.
fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let size = size.max(1);
    let n = text.len();
    if n <= size {
        return vec![text.to_string()];
    }
    let step = size.saturating_sub(overlap).max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while start < n {
        while start < n && !text.is_char_boundary(start) {
            start += 1;
        }
        let mut end = (start + size).min(n);
        while end < n && !text.is_char_boundary(end) {
            end += 1;
        }
        if start >= end {
            break;
        }
        out.push(text[start..end].to_string());
        if end >= n {
            break;
        }
        start += step;
    }
    out
}

/// Write the single-row `_meta` table describing the index.
async fn write_meta(db: &Connection, meta: &Meta) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("embed_model", DataType::Utf8, false),
        Field::new("dim", DataType::Int32, false),
        Field::new("model_digest", DataType::Utf8, false),
        Field::new("built_at", DataType::Int64, false),
        Field::new("chunk_bytes", DataType::Int32, false),
        Field::new("chunk_overlap", DataType::Int32, false),
        Field::new("documents", DataType::Int64, false),
        Field::new("chunks", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec![meta.embed_model.clone()])),
            Arc::new(Int32Array::from(vec![meta.dim as i32])),
            Arc::new(StringArray::from(vec![meta.model_digest.clone()])),
            Arc::new(Int64Array::from(vec![meta.built_at])),
            Arc::new(Int32Array::from(vec![meta.chunk_bytes as i32])),
            Arc::new(Int32Array::from(vec![meta.chunk_overlap as i32])),
            Arc::new(Int64Array::from(vec![meta.documents as i64])),
            Arc::new(Int64Array::from(vec![meta.chunks as i64])),
        ],
    )
    .context("assembling meta record batch")?;
    db.create_table(META_TABLE, vec![batch])
        .execute()
        .await
        .context("creating meta table")?;
    Ok(())
}

/// Read the single `_meta` row back.
async fn read_meta(db: &Connection) -> Result<Meta> {
    let names = db.table_names().execute().await.context("listing tables")?;
    if !names.iter().any(|n| n == META_TABLE) {
        bail!("hybrid index is missing its {META_TABLE} table; rebuild it");
    }
    let table = db.open_table(META_TABLE).execute().await.context("opening meta table")?;
    let batches: Vec<RecordBatch> = table
        .query()
        .limit(1)
        .execute()
        .await
        .context("reading meta")?
        .try_collect()
        .await
        .context("collecting meta")?;
    let batch = batches
        .into_iter()
        .find(|b| b.num_rows() > 0)
        .ok_or_else(|| anyhow::anyhow!("hybrid index {META_TABLE} table is empty; rebuild it"))?;

    let embed_model = str_col(&batch, "embed_model")?.value(0).to_string();
    let model_digest = str_col(&batch, "model_digest")?.value(0).to_string();
    let dim = i32_col(&batch, "dim")?.value(0) as usize;
    let chunk_bytes = i32_col(&batch, "chunk_bytes")?.value(0) as usize;
    let chunk_overlap = i32_col(&batch, "chunk_overlap")?.value(0) as usize;
    let built_at = i64_col(&batch, "built_at")?.value(0);
    let documents = i64_col(&batch, "documents")?.value(0).max(0) as u64;
    let chunks = i64_col(&batch, "chunks")?.value(0).max(0) as u64;

    Ok(Meta {
        embed_model,
        dim,
        model_digest,
        built_at,
        chunk_bytes,
        chunk_overlap,
        documents,
        chunks,
    })
}

/// Downcast a named column to an [`Int32Array`].
fn i32_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int32Array> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("meta missing column {name}"))?
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| anyhow::anyhow!("meta column {name} is not int32"))
}

/// Downcast a named column to an [`Int64Array`].
fn i64_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("meta missing column {name}"))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow::anyhow!("meta column {name} is not int64"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_short_text_into_one() {
        assert_eq!(chunk_text("hello world", 2048, 256), vec!["hello world"]);
    }

    #[test]
    fn chunks_long_text_with_overlap() {
        let text = "abcdefghij".repeat(10); // 100 bytes
        let chunks = chunk_text(&text, 40, 10);
        assert!(chunks.len() > 1);
        // Each chunk is at most `size` bytes.
        assert!(chunks.iter().all(|c| c.len() <= 40));
        // Consecutive chunks overlap, so the concatenated length exceeds the input.
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert!(total > text.len());
    }

    #[test]
    fn empty_text_yields_no_chunks() {
        assert!(chunk_text("   ", 2048, 256).is_empty());
    }

    #[test]
    fn snippet_collapses_and_truncates() {
        let s = snippet("a\n\n   b\tc");
        assert_eq!(s, "a b c");
        let long = "x ".repeat(400);
        let s = snippet(&long);
        assert!(s.ends_with('…'));
    }
}
