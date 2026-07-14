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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use futures::{StreamExt, TryStreamExt};
use tokio::sync::{Semaphore, mpsc, oneshot};
use lance_index::scalar::FullTextSearchQuery;
use lancedb::arrow::SendableRecordBatchStream;
use lancedb::index::Index as LanceIndex;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::query::{ExecutableQuery, HasQuery, QueryBase, Select};
use lancedb::rerankers::rrf::RRFReranker;
use lancedb::table::{CompactionOptions, OptimizeAction};
use lancedb::{Connection, Table};

use crate::artifacts::is_text;
use crate::config::{CoreConfig, DEFAULT_EMBED_LOOKAHEAD_FACTOR};
use crate::embed::Embedder;
use crate::index::{
    CHUNKS_TABLE, Index, contains_table, has_table, i32_col, i64_col, str_col, text_refs_from_batch,
};
use crate::ingest::connect;
use crate::progress::{FullTextProgress, run_ticker};
use crate::source::ArtifactSource;

/// Name of the single-row table holding index metadata.
pub(crate) const META_TABLE: &str = "_meta";
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
    /// to [`CoreConfig::embed_model`], so a query can never use a model that
    /// disagrees with the stored vectors.
    pub async fn open(config: &CoreConfig) -> Result<Self> {
        let db = connect(config).await?;
        if !has_table(&db, CHUNKS_TABLE).await? {
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
        let mut query = self
            .table
            .query()
            .nearest_to(qvec)
            .context("building vector query")?;
        // Adopt the future Lance behavior: don't auto-project `_score` into the
        // output. We rely on `_relevance_score` from the reranker instead.
        // TODO: remove once Lance makes this the default (currently lancedb 0.30).
        query.mut_query().disable_scoring_autoprojection = true;
        let batches: Vec<RecordBatch> = query
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
    config: &CoreConfig,
    index: &Index,
    embedder: &Embedder,
    force: bool,
    resume: bool,
) -> Result<IndexStats> {
    // Standalone build owns its own progress and ticker. The concurrent
    // orchestrator instead drives `build_with_progress` with a shared ticker so
    // raw storage and the text build render on one status line.
    let progress = Arc::new(FullTextProgress::default());
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let ticker = {
        let progress = progress.clone();
        tokio::spawn(run_ticker(Some(progress), None, stop_rx))
    };
    let result = build_with_progress(config, index, embedder, force, resume, progress).await;
    let _ = stop_tx.send(());
    let _ = ticker.await;
    result
}

/// Build the full-text index, reporting into the supplied shared `progress`.
///
/// Identical to [`build`] but with the progress counters and ticker owned by
/// the caller, so the concurrent raw + full-text orchestrator can render both
/// passes on a single status line.
pub(crate) async fn build_with_progress(
    config: &CoreConfig,
    index: &Index,
    embedder: &Embedder,
    force: bool,
    resume: bool,
    progress: Arc<FullTextProgress>,
) -> Result<IndexStats> {
    if force && resume {
        bail!("force and resume are mutually exclusive");
    }

    // Resolve the artifact source (local directory or S3 bucket) once; the
    // reader pulls every text artifact through it.
    let source = Arc::new(ArtifactSource::from_config(config)?.ok_or_else(|| {
        anyhow::anyhow!(
            "no artifact source configured (set artifact_root or s3_bucket); \
             the hybrid index needs the text files"
        )
    })?);

    let db = connect(config).await?;
    let existing = db.table_names().execute().await.context("listing tables")?;
    let have_index = contains_table(&existing, CHUNKS_TABLE);
    if have_index && !force && !resume {
        bail!(
            "hybrid index already exists at {}; pass force to rebuild or resume to continue",
            config.lance_path.display()
        );
    }
    if force {
        for name in [CHUNKS_TABLE, META_TABLE] {
            if contains_table(&existing, name) {
                db.drop_table(name, &[]).await.with_context(|| format!("dropping table {name}"))?;
            }
        }
    }

    let total_artifacts = index.text_count().await.context("counting text artifacts")? as usize;
    progress.text_total.store(total_artifacts as u64, Ordering::Relaxed);
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
        if contains_table(&existing, META_TABLE) {
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
    // Ordered look-ahead window for the embed stage. 0 means auto; never below
    // `concurrency`, or the window would throttle the request slots it feeds.
    let lookahead = if config.embed_lookahead == 0 {
        concurrency.saturating_mul(DEFAULT_EMBED_LOOKAHEAD_FACTOR)
    } else {
        config.embed_lookahead
    }
    .max(concurrency);
    let (jobs_tx, jobs_rx) = mpsc::channel::<Vec<ChunkRow>>(concurrency * 2);
    let (out_tx, out_rx) = mpsc::channel::<RecordBatch>(concurrency * 2);

    // Shared live counters seeded with the resumed chunk total so the displayed
    // figure is cumulative. The ticker that renders `progress` is owned by the
    // caller ([`build`] or the concurrent orchestrator).
    progress.chunks.store(prior_chunks, Ordering::Relaxed);

    // Reader: read + chunk up to `read_concurrency` files at once (each on the
    // blocking pool), emitting chunks in document order. Concurrent reads keep a
    // fast embed backend fed when per-file storage latency would otherwise starve
    // a single serial reader.
    let artifacts = index
        .text_artifacts_stream()
        .await
        .context("streaming text artifacts")?;
    let reader = {
        let cfg = config.clone();
        let progress = progress.clone();
        let source = source.clone();
        tokio::spawn(async move {
            run_reader(&cfg, source, artifacts, done, &progress, jobs_tx).await
        })
    };

    // Embedder: keep `concurrency` embed requests in flight, results in order.
    let embed_stage = {
        let embedder = embedder.clone();
        let progress = progress.clone();
        tokio::spawn(async move {
            run_embedder(
                embedder, jobs_rx, out_tx, seed_dim, prior_chunks, concurrency, lookahead,
                &progress,
            )
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

    // The embed pipeline has drained; everything below (the `_meta` write, the
    // optional compaction, and the FTS/vector index builds) reports no
    // incremental progress. Signal the ticker so it retires the full-text bar
    // and prints a phase message instead of redrawing frozen counters.
    progress.post_embed.store(true, Ordering::Relaxed);

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
        let started = Instant::now();
        let stats = table
            .optimize(OptimizeAction::Compact {
                options: CompactionOptions::default(),
                remap_options: None,
            })
            .await
            .context("compacting chunks table")?;
        let (removed, added) = stats
            .compaction
            .map_or((0, 0), |m| (m.fragments_removed, m.fragments_added));
        tracing::info!(
            "compaction done in {:.1}s ({removed} fragments removed, {added} fragments added)",
            started.elapsed().as_secs_f64()
        );
    }

    tracing::info!("creating full-text index on {chunk_count} chunks");
    let started = Instant::now();
    table
        .create_index(&["text"], LanceIndex::FTS(FtsIndexBuilder::default()))
        .execute()
        .await
        .context("creating FTS index")?;
    tracing::info!(
        "full-text index done in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    if (chunk_count as usize) >= MIN_VECTOR_INDEX_ROWS {
        tracing::info!("creating vector index");
        let started = Instant::now();
        match table.create_index(&["vector"], LanceIndex::Auto).execute().await {
            Ok(()) => tracing::info!(
                "vector index done in {:.1}s",
                started.elapsed().as_secs_f64()
            ),
            Err(e) => {
                tracing::warn!("vector index creation failed ({e}); queries will use flat search")
            }
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
    Missing,
    /// Not a text artifact we read, or no text extracted; nothing to do.
    Empty,
}

/// Read and chunk one artifact. Fetches the text through the [`ArtifactSource`]
/// (local or S3) and splits it into overlapping chunks. A missing file is not
/// fatal (the caller counts it); a non-text artifact or empty body yields
/// nothing.
async fn read_one(
    source: &ArtifactSource,
    config: &CoreConfig,
    doc_id: &str,
    name: &str,
    media_type: Option<&str>,
) -> ReadOutcome {
    // The listing should already exclude non-text artifacts, but skip defensively
    // rather than fetch bytes we cannot use.
    if !is_text(name, media_type) {
        return ReadOutcome::Empty;
    }
    let bytes = match source.get(doc_id, name).await {
        Ok(Some(bytes)) => bytes,
        // A referenced file may legitimately be absent (e.g. redacted documents),
        // so a miss is not fatal; the caller counts it.
        Ok(None) => {
            return ReadOutcome::Missing;
        }
        Err(e) => {
            tracing::warn!("error reading artifact '{name}' (doc {doc_id}): {e:#}; skipping");
            return ReadOutcome::Empty;
        }
    };
    let body = String::from_utf8_lossy(&bytes);
    if body.is_empty() {
        return ReadOutcome::Empty;
    }
    let rows = chunk_text(body.as_ref(), config.chunk_bytes, config.chunk_overlap)
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
///
/// `artifacts` is a stream ordered by `(id, name)` (see
/// [`Index::text_artifacts_stream`]), drained one batch at a time so the listing
/// is never fully resident — at full-corpus scale it would be gigabytes. A batch
/// is read to completion before the next begins, which keeps document order
/// intact across batch boundaries: a document is a contiguous run in the ordered
/// stream, so its trailing artifacts in one batch are immediately followed by
/// the rest at the head of the next, with no other document interleaved.
async fn run_reader(
    config: &CoreConfig,
    source: Arc<ArtifactSource>,
    mut artifacts: SendableRecordBatchStream,
    done: HashSet<String>,
    progress: &FullTextProgress,
    jobs_tx: mpsc::Sender<Vec<ChunkRow>>,
) -> ReaderStats {
    let read_concurrency = config.read_concurrency.max(1);
    let embed_batch = config.embed_batch.max(1);
    // Share the config into the per-file reads without recloning it.
    let cfg = Arc::new(config.clone());

    let mut pending: Vec<ChunkRow> = Vec::new();
    let mut missing = 0u64;
    let mut docs = 0u64;
    // Last document fed into the read pipeline, tracked across batches so the
    // distinct-document count is right even when a document's artifacts straddle
    // a batch boundary (the stream is ordered, so a document is a contiguous
    // run).
    let mut last_doc: Option<String> = None;

    loop {
        let batch = match artifacts.try_next().await {
            Ok(Some(batch)) => batch,
            Ok(None) => break,
            Err(e) => {
                tracing::error!("error reading text-artifact listing: {e:#}");
                break;
            }
        };
        let refs = match text_refs_from_batch(&batch) {
            Ok(refs) => refs,
            Err(e) => {
                tracing::error!("error decoding text-artifact batch: {e:#}");
                break;
            }
        };

        // Per-batch pre-pass: drop documents a prior build already indexed and
        // count distinct (non-skipped) documents. This is cheap (no I/O), so it
        // does not hold up the concurrent reads that follow.
        let mut to_read: Vec<(String, String, Option<String>)> = Vec::with_capacity(refs.len());
        for (doc_id, name, media_type) in refs {
            if done.contains(&doc_id) {
                progress.scanned.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if last_doc.as_deref() != Some(doc_id.as_str()) {
                docs += 1;
                last_doc = Some(doc_id.clone());
            }
            to_read.push((doc_id, name, media_type));
        }

        let reads = futures::stream::iter(to_read)
            .map(|(doc_id, name, media_type)| {
                let cfg = cfg.clone();
                let source = source.clone();
                async move {
                    progress.reads_inflight.fetch_add(1, Ordering::Relaxed);
                    let outcome =
                        read_one(&source, &cfg, &doc_id, &name, media_type.as_deref()).await;
                    progress.reads_inflight.fetch_sub(1, Ordering::Relaxed);
                    progress.scanned.fetch_add(1, Ordering::Relaxed);
                    outcome
                }
            })
            .buffered(read_concurrency);
        futures::pin_mut!(reads);

        while let Some(outcome) = reads.next().await {
            match outcome {
                ReadOutcome::Chunks(rows) => pending.extend(rows),
                // Warn on the first miss so a wrong source is obvious; the
                // end-of-build summary reports the total.
                ReadOutcome::Missing => {
                    missing += 1;
                    progress.missing.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                ReadOutcome::Empty => continue,
            }
            // Hand off in `embed_batch`-sized jobs. Drain whole batches in a loop
            // rather than shipping the leftover `pending` in one shot: a single
            // large artifact can chunk into thousands of rows, and sending them as
            // one embed request crashes the model runner. Jobs may span documents
            // — the writer realigns to document boundaries — so small documents
            // still pack into full embed requests for backend efficiency.
            while pending.len() >= embed_batch {
                let job: Vec<ChunkRow> = pending.drain(..embed_batch).collect();
                progress.jobs_queued.fetch_add(1, Ordering::Relaxed);
                if jobs_tx.send(job).await.is_err() {
                    // Embed stage went away (it errored); stop reading.
                    return ReaderStats { docs, missing };
                }
            }
        }
    }
    if !pending.is_empty() {
        progress.jobs_queued.fetch_add(1, Ordering::Relaxed);
        let _ = jobs_tx.send(pending).await;
    }
    ReaderStats { docs, missing }
}

/// Embedder stage: embed jobs while keeping up to `concurrency` requests in
/// flight, and forward the resulting batches to the writer in document order.
///
/// Ordering and concurrency are decoupled. `buffered(lookahead)` preserves input
/// (document) order with a window of `lookahead` jobs, while a semaphore caps the
/// actual in-flight embed requests at `concurrency`. This keeps the ordering
/// guarantee the writer/resume invariant depends on, but stops a single slow
/// request from starving the backend: a slow head only stalls *output* ordering,
/// while the other windowed jobs keep all `concurrency` request slots busy. (With
/// `lookahead == concurrency` the two collapse and a slow head idles the backend
/// until it completes — the pathology this split avoids.) Returns the embedding
/// dimension (seeded from a resume, else discovered) and the cumulative chunk
/// count (prior chunks plus those embedded here).
#[allow(clippy::too_many_arguments)]
async fn run_embedder(
    embedder: Embedder,
    jobs_rx: mpsc::Receiver<Vec<ChunkRow>>,
    out_tx: mpsc::Sender<RecordBatch>,
    seed_dim: Option<usize>,
    prior_chunks: u64,
    concurrency: usize,
    lookahead: usize,
    progress: &FullTextProgress,
) -> Result<(Option<usize>, u64)> {
    // Caps concurrent embed requests independently of the ordered window above.
    let in_flight = Arc::new(Semaphore::new(concurrency));
    // Adapt the channel into a Stream so `buffered` can drive the window.
    let jobs = futures::stream::unfold(jobs_rx, |mut rx| async move {
        rx.recv().await.map(|rows| (rows, rx))
    });
    let embedded = jobs
        .map(|rows| {
            let embedder = embedder.clone();
            let in_flight = in_flight.clone();
            async move {
                // A job leaves the read→embed backlog as it enters the window.
                progress.jobs_queued.fetch_sub(1, Ordering::Relaxed);
                let n = rows.len();
                let bytes: u64 = rows.iter().map(|r| r.text.len() as u64).sum();
                // Hold a permit only across the request itself, so the gauge and
                // the backend see exactly `concurrency` in flight regardless of
                // how far the ordered window reads ahead.
                let permit = in_flight.acquire().await.expect("embed semaphore is never closed");
                progress.embeds_inflight.fetch_add(1, Ordering::Relaxed);
                let result = embed_job(&embedder, &rows).await;
                progress.embeds_inflight.fetch_sub(1, Ordering::Relaxed);
                drop(permit);
                let (batch, d) = result?;
                Ok::<(RecordBatch, usize, usize, u64), anyhow::Error>((batch, d, n, bytes))
            }
        })
        .buffered(lookahead);
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
    progress: &FullTextProgress,
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
    if !has_table(db, META_TABLE).await? {
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
