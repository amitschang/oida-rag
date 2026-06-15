//! LanceDB-backed hybrid (keyword + semantic) search over artifact text.
//!
//! The metadata index ([`crate::Index`]) answers questions about a document's
//! fields. This module answers questions about what a document *says*: it reads
//! the plain-text (OCR) artifacts, splits them into overlapping chunks, embeds
//! each chunk with an OpenAI-compatible model (Ollama or vLLM), and stores both
//! the text and its vector in a single LanceDB table. Queries then run a
//! full-text search and a vector
//! search in parallel and fuse the two rankings with Reciprocal Rank Fusion
//! (RRF), collapsing chunk hits back to their parent document.
//!
//! # Embedding-model consistency
//!
//! A vector index is only meaningful when queries are embedded with the *same*
//! model that produced the stored vectors. To guarantee that, the build writes
//! a `_meta` row recording the embed model name and its vector dimension. At
//! query time we embed with the *stored* model name (never the live config),
//! assert the dimension matches, and — unless verification is disabled — refuse
//! to serve when the configured model name disagrees with the stored one. There
//! is no portable content digest across embed servers, so the model name is the
//! pin: encode any weights change (a commit hash, a quantization tag) into it.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use futures::{StreamExt, TryStreamExt};
use tokio::sync::{mpsc, oneshot};
use lance_index::scalar::FullTextSearchQuery;
use lancedb::index::Index as LanceIndex;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::rerankers::rrf::RRFReranker;
use lancedb::table::{CompactionOptions, OptimizeAction};
use lancedb::{Connection, Table};

use crate::artifacts::{ArtifactTextStatus, artifact_path, read_artifact_text};
use crate::config::Config;
use crate::embed::Embedder;
use crate::index::Index;
use crate::ingest::connect;

/// Name of the table holding text chunks and their embeddings.
const CHUNKS_TABLE: &str = "chunks";
/// Name of the single-row table holding index metadata.
const META_TABLE: &str = "_meta";
/// Below this row count a vector (ANN) index is skipped; flat search is exact
/// and fast enough, and IVF/PQ training needs a reasonable number of rows.
const MIN_VECTOR_INDEX_ROWS: usize = 256;

/// Live build counters, updated lock-free by the pipeline stages and sampled by
/// the progress ticker. Decoupling counting (hot path) from rendering (timer)
/// keeps display cadence independent of work cadence.
#[derive(Default)]
struct Progress {
    /// Artifacts the reader has walked (whether read, skipped, or missing).
    scanned: AtomicU64,
    /// Chunks embedded so far (includes any seeded from a resume).
    chunks: AtomicU64,
    /// Bytes of chunk text embedded so far, for a throughput/token estimate.
    text_bytes: AtomicU64,
    /// Referenced artifact files that were not on disk.
    missing: AtomicU64,
    /// Live pipeline gauges — instantaneous depths, not cumulative totals — to
    /// locate the bottleneck. The stage where work piles up is downstream of the
    /// stall; the stage that runs below its limit is being starved by it.
    ///
    /// Read tasks currently executing on the blocking pool. Pinned near
    /// `read_concurrency` when reads keep up; collapsing toward 1 despite a high
    /// limit is the signature of head-of-line blocking in the order-preserving
    /// reader (one slow artifact stalls every read queued behind it).
    reads_inflight: AtomicU64,
    /// Read+chunked jobs waiting in the channel for a free embed slot. Near the
    /// channel cap ⇒ the embedder is the bottleneck; near 0 ⇒ the reader can't
    /// keep the embedder fed.
    jobs_queued: AtomicU64,
    /// Embed requests in flight to the backend. Pinned near `embed_concurrency`
    /// ⇒ the GPUs are saturated; below it ⇒ the embedder is starved upstream.
    embeds_inflight: AtomicU64,
    /// Embedded batches waiting in the channel for the writer. Near the channel
    /// cap ⇒ LanceDB write backpressure is the bottleneck.
    out_queued: AtomicU64,
}

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
    /// Model used to produce the embeddings.
    pub embed_model: String,
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
    /// The embed model the caller configured; compared against `meta.embed_model`
    /// at query time when `verify_model` is set.
    configured_model: String,
    /// Whether to refuse a query when `configured_model` disagrees with the
    /// model recorded in the index.
    verify_model: bool,
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
        let embedder = Embedder::new(
            &config.embed_host,
            meta.embed_model.clone(),
            config.embed_api_key.clone(),
        )?;
        Ok(Self {
            table,
            embedder,
            meta,
            configured_model: config.embed_model.clone(),
            verify_model: config.embed_verify_model,
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
            built_at: self.meta.built_at,
            chunk_bytes: self.meta.chunk_bytes,
            chunk_overlap: self.meta.chunk_overlap,
        })
    }

    /// Run a hybrid keyword + semantic search, returning up to `limit`
    /// documents ranked by fused relevance.
    ///
    /// Guards against querying with a model that disagrees with the stored
    /// vectors: unless verification is disabled, the configured model name must
    /// match the one recorded in the index (pinning is by name alone), and the
    /// query embedding's dimension is checked against the stored dimension.
    pub async fn query(&self, text: &str, limit: usize) -> Result<Vec<HybridChunkHit>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let limit = limit.max(1);

        // Refuse to serve if the configured model name differs from the one the
        // index was built with. Names are the pin, so encode any weights change
        // into the name; bypass with `embed_verify_model = false` when intended.
        if self.verify_model && self.configured_model != self.meta.embed_model {
            bail!(
                "configured embed model '{}' does not match index model '{}'; \
                 rebuild the index (oida-cli ingest --full-text --force) or set \
                 embed_verify_model = false to override",
                self.configured_model,
                self.meta.embed_model
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
/// `_meta` row pinning the embed model name. Pass `force` to replace
/// an existing index.
///
/// Pass `resume` to continue an interrupted build: the existing `chunks` table
/// is kept, every `doc_id` already present is skipped, and embedding picks up
/// from the first un-indexed document. This is safe because durable writes are
/// aligned to document boundaries (see the scan loop), so a `doc_id` in the
/// table is always complete — never a partial document. If no `chunks` table
/// exists yet (e.g. a crash before the first checkpoint), `resume` simply
/// builds the index from scratch; either way it never re-ingests the
/// `documents`/`artifacts` metadata. `resume` and `force` are mutually
/// exclusive.
pub async fn build(
    config: &Config,
    index: &Index,
    embedder: &Embedder,
    force: bool,
    resume: bool,
) -> Result<IndexStats> {
    if config.artifact_root.is_none() {
        bail!("artifact_root is not configured; the hybrid index needs the text files on disk");
    }

    if force && resume {
        bail!("force and resume are mutually exclusive");
    }

    let db = connect(config).await?;
    let existing = db.table_names().execute().await.context("listing tables")?;
    let have_index = existing.iter().any(|n| n == CHUNKS_TABLE);
    if have_index && !force && !resume {
        bail!(
            "hybrid index already exists at {}; pass force to rebuild or resume to continue",
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

    let artifacts = index.text_artifacts().await.context("listing text artifacts")?;
    let total_artifacts = artifacts.len();
    tracing::info!("indexing text from {total_artifacts} artifacts");

    // Resume seeds: a prior (interrupted) build's already-indexed documents are
    // skipped, and its chunk/doc/dim counts carry into the totals.
    let mut resumed_table: Option<Table> = None;
    let mut seed_dim: Option<usize> = None;
    let mut prior_chunks: u64 = 0;
    // Documents already present from a prior (interrupted) build; skipped here.
    let mut done: HashSet<String> = HashSet::new();

    if resume {
        // The interrupted build never wrote `_meta`; drop any stale one so the
        // finalize step can recreate it cleanly.
        if existing.iter().any(|n| n == META_TABLE) {
            db.drop_table(META_TABLE, &[]).await.context("dropping stale _meta table")?;
        }
        if have_index {
            let t = db
                .open_table(CHUNKS_TABLE)
                .execute()
                .await
                .context("opening chunks table to resume")?;
            // Seed the dimension from the stored vectors so the embed stage will
            // catch a model swap (a differently-sized embedding) on the first
            // new batch.
            seed_dim = Some(vector_dim(&t).await?);
            done = existing_doc_ids(&t).await?;
            prior_chunks = t.count_rows(None).await.context("counting existing chunks")? as u64;
            resumed_table = Some(t);
            tracing::info!(
                "resuming: {} documents ({prior_chunks} chunks) already indexed; skipping them",
                done.len()
            );
        } else {
            // Nothing embedded yet (e.g. a crash before the first checkpoint):
            // resume just means build the full-text index from scratch while
            // leaving the already-ingested documents/artifacts tables alone.
            tracing::info!("resume requested with no existing chunks; building the full-text index from scratch");
        }
    }
    let prior_docs = done.len() as u64;

    // ---- Pipelined build: read+chunk → embed (concurrent) → write ----
    //
    // The three stages run concurrently, connected by bounded channels, so disk
    // reads and LanceDB writes overlap GPU embedding instead of serializing with
    // it. The reader emits chunks in document order; `buffered` preserves that
    // order through the concurrent embed stage; the writer therefore sees chunks
    // in document order and keeps durable writes aligned to document boundaries
    // — the invariant `resume` depends on (a `doc_id` in the table is complete).
    let concurrency = config.embed_concurrency.max(1);
    let read_concurrency = config.read_concurrency.max(1);
    let embed_batch = config.embed_batch.max(1);
    let (jobs_tx, jobs_rx) = mpsc::channel::<Vec<ChunkRow>>(concurrency * 2);
    let (out_tx, out_rx) = mpsc::channel::<RecordBatch>(concurrency * 2);

    // Shared live counters and the timer-driven ticker that renders them. The
    // chunk counter is seeded with the resumed total so the displayed figure is
    // cumulative.
    let progress = Arc::new(Progress::default());
    progress.chunks.store(prior_chunks, Ordering::Relaxed);
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let ticker = {
        let progress = progress.clone();
        tokio::spawn(run_ticker(progress, total_artifacts as u64, stop_rx))
    };

    // Reader: read + chunk up to `read_concurrency` files at once (each on the
    // blocking pool), emitting chunks in document order. Concurrent reads keep a
    // fast embed backend fed when per-file storage latency would otherwise starve
    // a single serial reader.
    let reader = {
        let cfg = config.clone();
        let progress = progress.clone();
        tokio::spawn(async move {
            run_reader(&cfg, artifacts, done, embed_batch, read_concurrency, &progress, jobs_tx)
                .await
        })
    };

    // Embedder: keep `concurrency` embed requests in flight, results in order.
    let embed_stage = {
        let embedder = embedder.clone();
        let progress = progress.clone();
        tokio::spawn(async move {
            run_embedder(embedder, jobs_rx, out_tx, seed_dim, prior_chunks, concurrency, &progress)
                .await
        })
    };

    // Writer: drain embedded batches and flush complete documents to LanceDB.
    // Runs inline so `build` ends up owning the resulting table handle. A writer
    // failure drops `out_rx`, which unblocks the upstream stages (their sends
    // fail and they wind down), so we can surface it before joining them.
    let writer_db = db.clone();
    let table = run_writer(&writer_db, resumed_table, out_rx, config.write_buffer_bytes, &progress)
        .await
        .context("writing chunk fragments")?;

    let ReaderStats { docs: new_docs, missing } =
        reader.await.context("reader task panicked")?;
    let (dim, chunk_count) = embed_stage.await.context("embedder task panicked")??;
    // Stop the ticker and let it print its final summary line.
    let _ = stop_tx.send(());
    let _ = ticker.await;
    let doc_count = prior_docs + new_docs;

    if missing > 0 {
        let pct = 100.0 * missing as f64 / total_artifacts.max(1) as f64;
        tracing::warn!(
            "{missing}/{total_artifacts} referenced artifact files ({pct:.1}%) were missing \
             and skipped (e.g. redacted documents); the index is built from the rest"
        );
    }

    let table = table.ok_or_else(|| {
        anyhow::anyhow!("no readable text artifacts found; nothing to index")
    })?;
    let dim = dim.expect("dim is set whenever a batch was written");

    // Write `_meta` now, before the (optional, slow) compaction and index builds.
    // Every value it records is already final and durable: the chunks are on
    // disk, `dim`/`chunk_count`/`doc_count` are known, and compaction/indexing
    // only change physical layout, never these numbers. Writing it here is what
    // makes the index openable — `HybridIndex::open` requires `_meta` — so an
    // interruption during compaction or the vector-index build leaves a fully
    // queryable index (vector search falls back to a flat scan) rather than one
    // the server refuses to load. `_meta` is a single tiny row, so deferring it
    // this far costs no memory.
    let built_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let meta = Meta {
        embed_model: embedder.model().to_string(),
        dim,
        built_at,
        chunk_bytes: config.chunk_bytes,
        chunk_overlap: config.chunk_overlap,
        documents: doc_count,
        chunks: chunk_count,
    };
    write_meta(&db, &meta).await?;

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

    Ok(IndexStats {
        documents: meta.documents,
        chunks: meta.chunks,
        dim: meta.dim,
        embed_model: meta.embed_model,
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

/// Tallies returned by the reader stage.
struct ReaderStats {
    /// New (non-skipped) documents whose artifacts were scanned.
    docs: u64,
    /// Referenced artifact files that were not on disk (skipped, not fatal).
    missing: u64,
}

/// Outcome of reading + chunking one artifact, produced on the blocking pool and
/// consumed in document order by the reader's sequencing loop.
enum ReadOutcome {
    /// Chunk rows for one artifact (empty if the text was blank).
    Chunks(Vec<ChunkRow>),
    /// The referenced file was not on disk (skipped, not fatal).
    Missing { doc_id: String, name: String },
    /// Not a text artifact we read, or no text extracted; nothing to do.
    Empty,
}

/// Read and chunk one artifact. Pure blocking file I/O plus CPU chunking, so it
/// is safe — and the point — to run many of these at once on the blocking pool.
fn read_one(config: &Config, doc_id: &str, name: &str, media_type: Option<&str>) -> ReadOutcome {
    // u64::MAX / 2 reads the whole file without risking an offset overflow.
    let loaded = read_artifact_text(config, name, media_type, 0, u64::MAX / 2);
    match loaded.status {
        ArtifactTextStatus::TextLoaded => {}
        // A referenced file may legitimately be absent (e.g. redacted documents
        // are pulled from disk), so a miss is not fatal; the caller counts it.
        ArtifactTextStatus::ArtifactFileMissing => {
            return ReadOutcome::Missing { doc_id: doc_id.to_string(), name: name.to_string() };
        }
        // Not a text type we read in v1 (the listing should already exclude
        // these, but skip defensively rather than fail the whole build).
        ArtifactTextStatus::UnsupportedArtifactType
        | ArtifactTextStatus::ArtifactRootNotConfigured => return ReadOutcome::Empty,
    }
    let Some(body) = loaded.text else { return ReadOutcome::Empty };
    let rows = chunk_text(&body, config.chunk_bytes, config.chunk_overlap)
        .into_iter()
        .enumerate()
        .map(|(idx, chunk)| ChunkRow {
            doc_id: doc_id.to_string(),
            chunk_idx: idx as i32,
            artifact_name: name.to_string(),
            text: chunk,
        })
        .collect();
    ReadOutcome::Chunks(rows)
}

/// Reader stage: read and chunk the (document-ordered) artifacts with up to
/// `read_concurrency` files in flight on the blocking pool, sending chunk rows
/// downstream in `embed_batch`-sized jobs. `buffered` preserves document order,
/// so the writer still sees whole documents in order — the invariant `resume`
/// depends on. Returns once the listing is exhausted or the embed stage has gone
/// away.
async fn run_reader(
    config: &Config,
    artifacts: Vec<(String, String, Option<String>)>,
    done: HashSet<String>,
    embed_batch: usize,
    read_concurrency: usize,
    progress: &Progress,
    jobs_tx: mpsc::Sender<Vec<ChunkRow>>,
) -> ReaderStats {
    // Sequential pre-pass: drop documents a prior build already indexed and count
    // distinct (non-skipped) documents. This is cheap (no I/O), so it does not
    // hold up the concurrent reads that follow.
    let mut to_read: Vec<(String, String, Option<String>)> = Vec::with_capacity(artifacts.len());
    let mut docs = 0u64;
    let mut last_doc: Option<&str> = None;
    for (doc_id, name, media_type) in artifacts.iter() {
        if done.contains(doc_id) {
            progress.scanned.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if last_doc != Some(doc_id.as_str()) {
            docs += 1;
            last_doc = Some(doc_id);
        }
        to_read.push((doc_id.clone(), name.clone(), media_type.clone()));
    }

    // Share the config into the per-file blocking reads without recloning it.
    let cfg = Arc::new(config.clone());
    let reads = futures::stream::iter(to_read)
        .map(|(doc_id, name, media_type)| {
            let cfg = cfg.clone();
            async move {
                progress.reads_inflight.fetch_add(1, Ordering::Relaxed);
                let outcome = tokio::task::spawn_blocking(move || {
                    read_one(&cfg, &doc_id, &name, media_type.as_deref())
                })
                .await;
                progress.reads_inflight.fetch_sub(1, Ordering::Relaxed);
                progress.scanned.fetch_add(1, Ordering::Relaxed);
                outcome
            }
        })
        .buffered(read_concurrency);
    futures::pin_mut!(reads);

    let mut pending: Vec<ChunkRow> = Vec::new();
    let mut missing = 0u64;
    while let Some(joined) = reads.next().await {
        let outcome = match joined {
            Ok(o) => o,
            // A read task panicked — log and skip that artifact rather than abort
            // the whole build over one file.
            Err(e) => {
                tracing::error!("artifact read task panicked: {e}");
                continue;
            }
        };
        match outcome {
            ReadOutcome::Chunks(rows) => pending.extend(rows),
            // Warn on the first miss so a wrong artifact_root is obvious; the
            // end-of-build summary reports the total.
            ReadOutcome::Missing { doc_id, name } => {
                if missing == 0 {
                    let root = config.artifact_root.as_deref().unwrap_or(Path::new(""));
                    tracing::warn!(
                        "artifact '{name}' (doc {doc_id}) not found at {}; \
                         skipping it and any further missing files (e.g. redacted)",
                        artifact_path(root, &name).display(),
                    );
                }
                missing += 1;
                progress.missing.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            ReadOutcome::Empty => continue,
        }
        // Hand off in `embed_batch`-sized jobs. Drain whole batches in a loop
        // rather than shipping the leftover `pending` in one shot: a single large
        // artifact can chunk into thousands of rows, and sending them as one
        // embed request crashes the model runner. Jobs may span documents — the
        // writer realigns to document boundaries — so small documents still pack
        // into full embed requests for backend efficiency.
        while pending.len() >= embed_batch {
            let job: Vec<ChunkRow> = pending.drain(..embed_batch).collect();
            progress.jobs_queued.fetch_add(1, Ordering::Relaxed);
            if jobs_tx.send(job).await.is_err() {
                // Embed stage went away (it errored); stop reading.
                return ReaderStats { docs, missing };
            }
        }
    }
    if !pending.is_empty() {
        progress.jobs_queued.fetch_add(1, Ordering::Relaxed);
        let _ = jobs_tx.send(pending).await;
    }
    ReaderStats { docs, missing }
}

/// Embedder stage: embed jobs with up to `concurrency` Ollama requests in flight
/// at once (via `buffered`, which preserves input order) and forward the
/// resulting batches to the writer in document order. Returns the embedding
/// dimension (seeded from a resume, else discovered) and the cumulative chunk
/// count (prior chunks plus those embedded here).
async fn run_embedder(
    embedder: Embedder,
    jobs_rx: mpsc::Receiver<Vec<ChunkRow>>,
    out_tx: mpsc::Sender<RecordBatch>,
    seed_dim: Option<usize>,
    prior_chunks: u64,
    concurrency: usize,
    progress: &Progress,
) -> Result<(Option<usize>, u64)> {
    // Adapt the channel into a Stream so `buffered` can drive N embeds at once.
    let jobs = futures::stream::unfold(jobs_rx, |mut rx| async move {
        rx.recv().await.map(|rows| (rows, rx))
    });
    let embedded = jobs
        .map(|rows| {
            let embedder = embedder.clone();
            async move {
                // A job leaves the read→embed backlog as it enters the embed stage.
                progress.jobs_queued.fetch_sub(1, Ordering::Relaxed);
                let n = rows.len();
                let bytes: u64 = rows.iter().map(|r| r.text.len() as u64).sum();
                progress.embeds_inflight.fetch_add(1, Ordering::Relaxed);
                let result = embed_job(&embedder, &rows).await;
                progress.embeds_inflight.fetch_sub(1, Ordering::Relaxed);
                let (batch, d) = result?;
                Ok::<(RecordBatch, usize, usize, u64), anyhow::Error>((batch, d, n, bytes))
            }
        })
        .buffered(concurrency);
    // `unfold` yields a `!Unpin` stream; pin it so `.next()` is callable.
    futures::pin_mut!(embedded);

    let mut dim = seed_dim;
    let mut chunk_count = prior_chunks;
    while let Some(item) = embedded.next().await {
        let (batch, d, n, bytes) = item?;
        match dim {
            Some(existing) if existing != d => {
                bail!("embedding dimension changed from {existing} to {d} mid-build")
            }
            None => dim = Some(d),
            _ => {}
        }
        chunk_count += n as u64;
        // Publish for the ticker (rendering happens on its own timer).
        progress.chunks.fetch_add(n as u64, Ordering::Relaxed);
        progress.text_bytes.fetch_add(bytes, Ordering::Relaxed);
        progress.out_queued.fetch_add(1, Ordering::Relaxed);
        if out_tx.send(batch).await.is_err() {
            // Writer stopped (it errored); its error is the real failure and the
            // caller surfaces it, so just stop pulling.
            break;
        }
    }
    Ok((dim, chunk_count))
}

/// Embed one job of chunk rows into a [`RecordBatch`], returning the embedding
/// dimension alongside it. Touches no shared state, so it is safe to run many
/// of these concurrently.
async fn embed_job(embedder: &Embedder, rows: &[ChunkRow]) -> Result<(RecordBatch, usize)> {
    debug_assert!(!rows.is_empty(), "reader never sends an empty job");
    let texts: Vec<String> = rows.iter().map(|r| r.text.clone()).collect();
    let vectors = embedder.embed(&texts).await?;
    let d = vectors[0].len();
    let batch = build_record_batch(rows, &vectors, d as i32)?;
    Ok((batch, d))
}

/// How far back the sliding-window ("recent") rate looks.
const RATE_WINDOW: Duration = Duration::from_secs(60);

/// Progress ticker: every interval, sample the shared counters and render a
/// throughput line carrying two rates — a sliding-window "recent" rate (last
/// [`RATE_WINDOW`]) and a cumulative average. A per-tick instantaneous rate is
/// avoided because batches complete in bursts, so most ticks would read zero.
///
/// Throughput is measured from the first *embedded* chunk, so a resume's
/// skip/scan phase (which produces no chunks) never enters the rate. On a TTY
/// it rewrites one line in place; off a TTY (pod logs, pipes) it emits a
/// periodic newline log. Stops when `stop` fires, printing a final summary.
async fn run_ticker(progress: Arc<Progress>, total_artifacts: u64, mut stop: oneshot::Receiver<()>) {
    let tty = std::io::stderr().is_terminal();
    let mut interval =
        tokio::time::interval(Duration::from_millis(if tty { 500 } else { 5000 }));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let initial_chunks = progress.chunks.load(Ordering::Relaxed);
    // Set on the first tick that observes new chunks — the throughput clock.
    let mut run_start: Option<Instant> = None;
    // Recent (timestamp, chunks, bytes) samples within the window, for the
    // sliding-window rate.
    let mut samples: VecDeque<(Instant, u64, u64)> = VecDeque::new();

    loop {
        let stopping = tokio::select! {
            _ = interval.tick() => false,
            _ = &mut stop => true,
        };
        let now = Instant::now();
        let chunks = progress.chunks.load(Ordering::Relaxed);
        let bytes = progress.text_bytes.load(Ordering::Relaxed);
        let scanned = progress.scanned.load(Ordering::Relaxed);
        let missing = progress.missing.load(Ordering::Relaxed);
        let reads_inflight = progress.reads_inflight.load(Ordering::Relaxed);
        let jobs_queued = progress.jobs_queued.load(Ordering::Relaxed);
        let embeds_inflight = progress.embeds_inflight.load(Ordering::Relaxed);
        let out_queued = progress.out_queued.load(Ordering::Relaxed);

        if run_start.is_none() && chunks > initial_chunks {
            run_start = Some(now);
        }

        // Sliding-window rate: compare now against the oldest sample still inside
        // the window. This smooths over the bursts that make a per-tick delta read
        // zero most of the time.
        samples.push_back((now, chunks, bytes));
        while matches!(samples.front(), Some(&(t, _, _)) if now.duration_since(t) > RATE_WINDOW) {
            samples.pop_front();
        }
        let (recent_cps, recent_bps) = match samples.front() {
            Some(&(t0, c0, b0)) if now > t0 => {
                let dt = now.duration_since(t0).as_secs_f64();
                ((chunks - c0) as f64 / dt, (bytes - b0) as f64 / dt)
            }
            _ => (0.0, 0.0),
        };
        // Cumulative average, measured from the first embedded chunk.
        let avg_cps = run_start.map_or(0.0, |rs| {
            (chunks - initial_chunks) as f64 / now.duration_since(rs).as_secs_f64().max(1e-6)
        });
        // ~4 bytes per token is a rough English heuristic; labelled an estimate.
        let recent_tps = recent_bps / 4.0;

        let line = format!(
            "{chunks} chunks | 1m: {recent_cps:.0} ch/s, {:.1} MB/s, ~{} tok/s | \
             avg: {avg_cps:.0} ch/s | pipe: rd {reads_inflight} \u{2192} q {jobs_queued} \u{2192} \
             emb {embeds_inflight} \u{2192} q {out_queued} | {scanned}/{total_artifacts} scanned | \
             {missing} missing",
            recent_bps / 1.0e6,
            human_count(recent_tps as u64),
        );
        if tty {
            // \r returns to column 0; \x1b[K clears the rest of the line.
            eprint!("\r\x1b[K{line}");
            let _ = std::io::stderr().flush();
        } else {
            tracing::info!("{line}");
        }

        if stopping {
            if tty {
                eprintln!();
            }
            let embedded = chunks - initial_chunks;
            let secs = run_start.map_or(0.0, |rs| now.duration_since(rs).as_secs_f64());
            tracing::info!(
                "embedding throughput: {embedded} chunks in {secs:.0}s ({:.0} chunks/s avg)",
                embedded as f64 / secs.max(1e-6),
            );
            break;
        }
    }
}

/// Format a count compactly with a K/M suffix (for the high-magnitude tok/s).
fn human_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1.0e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1.0e3)
    } else {
        n.to_string()
    }
}

/// Writer stage: accumulate embedded batches (already in document order) and
/// flush them to LanceDB. A flush writes every *complete* document and carries
/// the still-open trailing document forward, so a durable fragment never splits
/// a document — the invariant that makes `resume` correct. Returns the table
/// handle (created on first write, or the resumed one).
async fn run_writer(
    db: &Connection,
    mut table: Option<Table>,
    mut out_rx: mpsc::Receiver<RecordBatch>,
    threshold: usize,
    progress: &Progress,
) -> Result<Option<Table>> {
    let mut buffer: Vec<RecordBatch> = Vec::new();
    let mut buffer_bytes: usize = 0;
    while let Some(batch) = out_rx.recv().await {
        progress.out_queued.fetch_sub(1, Ordering::Relaxed);
        buffer_bytes += batch.get_array_memory_size();
        buffer.push(batch);
        if buffer_bytes >= threshold {
            let open = last_doc_id(&buffer)?;
            let (mut flush, carry) = split_open_doc(std::mem::take(&mut buffer), &open)?;
            write_buffer(db, &mut table, &mut flush).await?;
            buffer_bytes = carry.iter().map(|b| b.get_array_memory_size()).sum();
            buffer = carry;
        }
    }
    // Stream drained: the final document is complete, so persist the remainder.
    if !buffer.is_empty() {
        write_buffer(db, &mut table, &mut buffer).await?;
    }
    Ok(table)
}

/// The `doc_id` of the last row in the buffer — the document still open (more of
/// its chunks may yet arrive), which a flush must carry rather than write.
fn last_doc_id(buffer: &[RecordBatch]) -> Result<String> {
    let last = buffer.last().expect("writer only splits a non-empty buffer");
    let ids = str_col(last, "doc_id")?;
    Ok(ids.value(last.num_rows() - 1).to_string())
}

/// Split document-ordered chunk batches into `(flush, carry)`: every batch up to
/// the start of the trailing `open` document goes to `flush`; the open
/// document's contiguous trailing run is carried forward. Because document ids
/// are contiguous in the stream, the open document is always a suffix, so the
/// carried run is exactly the rows a later flush (or the final drain) will
/// complete.
fn split_open_doc(
    buffer: Vec<RecordBatch>,
    open: &str,
) -> Result<(Vec<RecordBatch>, Vec<RecordBatch>)> {
    // Global row index one past the last row that is *not* the open document; if
    // the whole buffer is the open document, nothing can be flushed yet.
    let mut last_non_open: Option<usize> = None;
    let mut base = 0usize;
    for b in &buffer {
        let ids = str_col(b, "doc_id")?;
        for r in 0..b.num_rows() {
            if ids.value(r) != open {
                last_non_open = Some(base + r);
            }
        }
        base += b.num_rows();
    }
    let carry_start = last_non_open.map_or(0, |i| i + 1);

    let mut flush = Vec::new();
    let mut carry = Vec::new();
    let mut base = 0usize;
    for b in buffer {
        let len = b.num_rows();
        let (start, end) = (base, base + len);
        if end <= carry_start {
            flush.push(b);
        } else if start >= carry_start {
            carry.push(b);
        } else {
            // The boundary falls inside this batch (slices are zero-copy).
            let cut = carry_start - start;
            flush.push(b.slice(0, cut));
            carry.push(b.slice(cut, len - cut));
        }
        base = end;
    }
    Ok((flush, carry))
}

/// Read the embedding dimension from an existing chunks table's schema, used to
/// seed `dim` when resuming so a mid-build model swap is still caught.
async fn vector_dim(table: &Table) -> Result<usize> {
    let schema = table.schema().await.context("reading chunks schema")?;
    let field = schema
        .field_with_name("vector")
        .context("chunks table has no vector column")?;
    match field.data_type() {
        DataType::FixedSizeList(_, n) => Ok(*n as usize),
        other => bail!("chunks vector column has unexpected type {other:?}"),
    }
}

/// Collect the set of `doc_id`s already present in the chunks table so a resumed
/// build can skip them. Reads only the `doc_id` column.
async fn existing_doc_ids(table: &Table) -> Result<HashSet<String>> {
    let batches: Vec<RecordBatch> = table
        .query()
        .select(Select::columns(&["doc_id"]))
        .execute()
        .await
        .context("scanning existing doc_ids")?
        .try_collect()
        .await
        .context("collecting existing doc_ids")?;
    let mut out = HashSet::new();
    for batch in &batches {
        let ids = str_col(batch, "doc_id")?;
        for row in 0..batch.num_rows() {
            out.insert(ids.value(row).to_string());
        }
    }
    Ok(out)
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
    let dim = i32_col(&batch, "dim")?.value(0) as usize;
    let chunk_bytes = i32_col(&batch, "chunk_bytes")?.value(0) as usize;
    let chunk_overlap = i32_col(&batch, "chunk_overlap")?.value(0) as usize;
    let built_at = i64_col(&batch, "built_at")?.value(0);
    let documents = i64_col(&batch, "documents")?.value(0).max(0) as u64;
    let chunks = i64_col(&batch, "chunks")?.value(0).max(0) as u64;

    Ok(Meta {
        embed_model,
        dim,
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

    /// Build a chunk batch from a list of `doc_id`s (one row each, dummy vector).
    fn batch_of(doc_ids: &[&str]) -> RecordBatch {
        let rows: Vec<ChunkRow> = doc_ids
            .iter()
            .enumerate()
            .map(|(i, id)| ChunkRow {
                doc_id: (*id).to_string(),
                chunk_idx: i as i32,
                artifact_name: format!("{id}.ocr"),
                text: format!("text-{i}"),
            })
            .collect();
        let vectors: Vec<Vec<f32>> = rows.iter().map(|_| vec![0.0, 1.0]).collect();
        build_record_batch(&rows, &vectors, 2).unwrap()
    }

    /// Flatten the `doc_id` column of a list of batches back into a `Vec`.
    fn doc_ids(batches: &[RecordBatch]) -> Vec<String> {
        let mut out = Vec::new();
        for b in batches {
            let ids = str_col(b, "doc_id").unwrap();
            for r in 0..b.num_rows() {
                out.push(ids.value(r).to_string());
            }
        }
        out
    }

    #[test]
    fn split_carries_open_doc_across_batches() {
        // Open doc "C" spans the batch boundary; everything before it flushes.
        let buffer = vec![batch_of(&["A", "A", "B"]), batch_of(&["B", "C", "C"])];
        let open = last_doc_id(&buffer).unwrap();
        assert_eq!(open, "C");
        let (flush, carry) = split_open_doc(buffer, &open).unwrap();
        assert_eq!(doc_ids(&flush), ["A", "A", "B", "B"]);
        assert_eq!(doc_ids(&carry), ["C", "C"]);
    }

    #[test]
    fn split_carries_everything_when_buffer_is_one_open_doc() {
        // A single document larger than the buffer cannot be checkpointed yet:
        // nothing is flushed (which would split it) and all rows carry forward.
        let buffer = vec![batch_of(&["C", "C"]), batch_of(&["C"])];
        let open = last_doc_id(&buffer).unwrap();
        let (flush, carry) = split_open_doc(buffer, &open).unwrap();
        assert!(flush.is_empty());
        assert_eq!(doc_ids(&carry), ["C", "C", "C"]);
    }

    #[test]
    fn split_flushes_all_when_boundary_aligns() {
        // The open doc occupies exactly the final batch: earlier batches flush
        // whole, the last carries whole, no slicing.
        let buffer = vec![batch_of(&["A", "A"]), batch_of(&["B", "B"])];
        let open = last_doc_id(&buffer).unwrap();
        let (flush, carry) = split_open_doc(buffer, &open).unwrap();
        assert_eq!(doc_ids(&flush), ["A", "A"]);
        assert_eq!(doc_ids(&carry), ["B", "B"]);
    }
}
