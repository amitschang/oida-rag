//! Corpus-agnostic engine configuration.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Default OpenAI-compatible endpoint used for embeddings. Defaults to a local
/// Ollama; point it at a vLLM sidecar (e.g. `http://localhost:8000`) for higher
/// throughput.
pub const DEFAULT_EMBED_HOST: &str = "http://localhost:11434";
/// Default path to the LanceDB database holding the document index.
pub const DEFAULT_LANCE: &str = "oida-lance";
/// Default model used to embed document text for semantic search.
pub const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";
/// Default chunk size, in bytes, used when splitting artifact text.
pub const DEFAULT_CHUNK_BYTES: usize = 2048;
/// Default overlap, in bytes, between adjacent text chunks.
pub const DEFAULT_CHUNK_OVERLAP: usize = 256;
/// Default in-memory write-buffer target, in bytes, used when building the
/// hybrid index. Embedded chunk batches accumulate until they reach this size,
/// then flush to LanceDB in a single `Table::add`. This decouples the (small)
/// embed batch from the (large) Lance fragment, keeping fragment churn low. Each
/// flush is also a durable checkpoint that a re-run of an incremental
/// `ingest --full-text` can restart from, so this is kept modest (128 MiB) to
/// bound how much embedding work a crash can lose while still yielding healthy
/// Lance fragments.
pub const DEFAULT_WRITE_BUFFER_BYTES: usize = 128 << 20;
/// Whether to compact the chunks table after a hybrid-index build by default.
pub const DEFAULT_COMPACT_ON_BUILD: bool = true;
/// Default target size, in bytes, of each `raw_artifacts` LanceDB fragment.
/// Raw storage fetches whole files (PDFs, images, …) of widely varying size, so
/// flushing a fixed *count* of blobs yields fragments of erratic size. Instead,
/// fetched blobs accumulate until their combined size reaches this target, then
/// flush as one fragment — keeping file sizes consistent regardless of the
/// per-artifact size distribution. Larger means fewer, bigger files (less
/// metadata) at the cost of higher peak memory and a coarser resume checkpoint.
pub const DEFAULT_RAW_FILE_BYTES: usize = 256 << 20;
/// Default in-memory buffer target, in bytes, before the metadata ingest flushes
/// a LanceDB fragment. Larger values yield fewer, bigger fragments (better read
/// performance, less metadata) at the cost of higher peak memory during ingest.
pub const DEFAULT_INGEST_BUFFER_BYTES: usize = 512 * 1024 * 1024;
/// Default number of embed requests kept in flight concurrently while building
/// the hybrid index. The build pipelines reading/chunking, embedding, and
/// writing; this is how many embed calls overlap to keep the GPU fed across
/// request round-trips. To benefit, the server must allow at least this many
/// parallel requests (`OLLAMA_NUM_PARALLEL`).
pub const DEFAULT_EMBED_CONCURRENCY: usize = 4;
/// Default number of artifact files read and chunked concurrently while building
/// the hybrid index. The reader hands chunks to the embed stage; with a fast
/// embed backend (e.g. vLLM) a single serial reader becomes the bottleneck, so
/// several reads overlap to keep storage (which itself handles concurrency, like
/// Ceph) busy and the embed connections saturated. Defaults higher than embed
/// concurrency because per-file read latency, not bandwidth, is the limiter.
pub const DEFAULT_READ_CONCURRENCY: usize = 16;
/// Default number of text chunks sent per embed request. Larger batches amortize
/// per-request overhead (HTTP, tokenization setup, JSON float encoding) across
/// more chunks, which matters because a small embed model is usually
/// overhead-bound rather than GPU-bound. Too large can overflow the model
/// runner's context, so it is bounded and tunable.
pub const DEFAULT_EMBED_BATCH: usize = 64;
/// Default ordered look-ahead window for the embed stage, as a multiple of
/// `embed_concurrency`, used when `embed_lookahead` is left at 0 (auto). The
/// embed stage emits results in document order (the writer/resume invariant),
/// but decouples that ordering window from the count of concurrent requests: a
/// slow request only stalls *output*, while this many jobs stay available to
/// keep `embed_concurrency` requests in flight. Larger absorbs more per-request
/// latency variance (better GPU utilization) at the cost of more buffered jobs
/// in memory; 8× covers heavy-tailed latency without much memory.
pub const DEFAULT_EMBED_LOOKAHEAD_FACTOR: usize = 8;
/// Whether, by default, a query verifies that the configured embed model name
/// matches the one recorded in the index. Pinning is by name alone (no portable
/// content digest across servers), so set the name to encode a weights identity
/// — a commit hash, a quantization tag, a `num_ctx` variant — and keep this on.
pub const DEFAULT_EMBED_VERIFY_MODEL: bool = true;

/// Corpus-agnostic engine configuration: everything the framework drivers
/// (index open, hybrid build, artifact source, raw store) need, with no
/// knowledge of any particular source or app.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CoreConfig {
    /// Directory containing artifact files on disk, keyed by `artifact_name`.
    ///
    /// Optional: when unset, artifact-text tools degrade gracefully.
    pub artifact_root: Option<PathBuf>,
    /// Base URL of the OpenAI-compatible server used for embeddings. Kept
    /// separate from the chat host so embeddings can be served by a faster
    /// sidecar (e.g. vLLM) while chat stays on Ollama.
    ///
    /// May be a comma-separated list of replica addresses serving the same model;
    /// requests are then balanced across them by least connections (client-side,
    /// so no external load balancer is required).
    pub embed_host: String,
    /// Optional bearer token sent with embed requests (vLLM `--api-key`).
    pub embed_api_key: Option<String>,
    /// Path to the LanceDB database holding the hybrid keyword+vector index
    /// over artifact text.
    pub lance_path: PathBuf,
    /// Model used to embed document text and queries for semantic search. This
    /// is only the *default* used when building the index; the model actually
    /// used for a query is read back from the index metadata so search can never
    /// use a model that disagrees with the stored vectors.
    pub embed_model: String,
    /// Target size, in bytes, of each text chunk that is embedded and indexed.
    pub chunk_bytes: usize,
    /// Number of bytes adjacent chunks overlap, to avoid splitting matches
    /// across a boundary.
    pub chunk_overlap: usize,
    /// Target size, in bytes, of the in-memory write buffer used when building
    /// the hybrid index. Larger values mean fewer, bigger LanceDB fragments
    /// (less churn) at the cost of higher peak memory during the build.
    pub write_buffer_bytes: usize,
    /// Compact the chunks table once the build finishes, merging any remaining
    /// small fragments and pruning old versions before the indexes are built.
    pub compact_on_build: bool,
    /// In-memory buffer target, in bytes, before the metadata ingest flushes a
    /// LanceDB fragment. Larger = fewer, bigger fragments (better reads) but
    /// higher peak memory during ingest.
    pub ingest_buffer_bytes: usize,
    /// Number of embed requests kept concurrently in flight while building the
    /// hybrid index, overlapping round-trips to keep the GPU saturated. Requires
    /// a matching `OLLAMA_NUM_PARALLEL` on the server to take effect.
    pub embed_concurrency: usize,
    /// Number of artifact files read and chunked concurrently while building the
    /// hybrid index, overlapping per-file storage latency so the reader can keep
    /// a fast embed backend fed.
    pub read_concurrency: usize,
    /// Number of text chunks per embed request during a hybrid build. Larger
    /// amortizes per-request overhead; keep below what the model runner's context
    /// can hold for one request.
    pub embed_batch: usize,
    /// Ordered look-ahead window for the embed stage, in jobs. The stage still
    /// emits in document order, but this many embed jobs may be buffered ahead so
    /// a slow request stalls only the *output*, not the `embed_concurrency`
    /// requests kept in flight — keeping the GPU fed despite per-request latency
    /// variance. 0 means auto (`DEFAULT_EMBED_LOOKAHEAD_FACTOR` × concurrency);
    /// values below `embed_concurrency` are raised to it. Costs buffered jobs in
    /// memory, bounded by this window.
    pub embed_lookahead: usize,
    /// Verify, before serving a query, that `embed_model` matches the model
    /// name recorded in the index. Pinning is by name only; turn this off to
    /// bypass the check (e.g. when intentionally serving with a renamed model).
    pub embed_verify_model: bool,
    /// Whether ingest stores non-text/plain ("raw") artifacts in a
    /// `raw_artifacts` table, fetched from the artifact source. Off by default;
    /// the text/plain chunk index is built regardless.
    pub store_raw_artifacts: bool,
    /// Target size, in bytes, of each `raw_artifacts` LanceDB fragment. Fetched
    /// blobs accumulate until their combined size reaches this target, then
    /// flush as one fragment, so file sizes stay consistent despite widely
    /// varying per-artifact sizes. Larger = fewer, bigger files (less metadata)
    /// at the cost of higher peak memory and a coarser resume checkpoint.
    pub raw_file_bytes: usize,
    /// S3 bucket holding the artifact files. When set, the ingest/full-text
    /// reader fetches artifacts from S3 instead of `artifact_root`, using the
    /// same fan-out key layout under `s3_prefix`. Credentials are read from the
    /// standard AWS environment (`AWS_ACCESS_KEY_ID`, …).
    pub s3_bucket: Option<String>,
    /// AWS region for `s3_bucket` (e.g. `us-east-1`). Optional for S3-compatible
    /// endpoints that ignore it.
    pub s3_region: Option<String>,
    /// Custom S3 endpoint URL, for S3-compatible stores (MinIO, Ceph RGW, R2).
    /// HTTP endpoints are allowed when this is set.
    pub s3_endpoint: Option<String>,
    /// Key prefix prepended to the fan-out artifact path within `s3_bucket`.
    pub s3_prefix: Option<String>,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            artifact_root: None,
            embed_host: DEFAULT_EMBED_HOST.to_string(),
            embed_api_key: None,
            lance_path: PathBuf::from(DEFAULT_LANCE),
            embed_model: DEFAULT_EMBED_MODEL.to_string(),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
            write_buffer_bytes: DEFAULT_WRITE_BUFFER_BYTES,
            compact_on_build: DEFAULT_COMPACT_ON_BUILD,
            ingest_buffer_bytes: DEFAULT_INGEST_BUFFER_BYTES,
            embed_concurrency: DEFAULT_EMBED_CONCURRENCY,
            read_concurrency: DEFAULT_READ_CONCURRENCY,
            embed_batch: DEFAULT_EMBED_BATCH,
            embed_lookahead: 0,
            embed_verify_model: DEFAULT_EMBED_VERIFY_MODEL,
            store_raw_artifacts: false,
            raw_file_bytes: DEFAULT_RAW_FILE_BYTES,
            s3_bucket: None,
            s3_region: None,
            s3_endpoint: None,
            s3_prefix: None,
        }
    }
}
