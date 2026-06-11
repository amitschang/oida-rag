//! Configuration shared by the MCP server and the CLI client.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Default Ollama model used to drive tool calling.
pub const DEFAULT_MODEL: &str = "qwen2.5-coder:latest";
/// Default Ollama HTTP endpoint.
pub const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";
/// Default path to the OIDA parquet index (one row per document, artifacts
/// inline as a `list<struct>` column).
pub const DEFAULT_PARQUET: &str = "oida-index.parquet";
/// Default path to the LanceDB database holding the document index.
pub const DEFAULT_LANCE: &str = "oida-lance";
/// Default Ollama model used to embed document text for semantic search.
pub const DEFAULT_EMBED_MODEL: &str = "nomic-embed-text";
/// Default chunk size, in bytes, used when splitting artifact text.
pub const DEFAULT_CHUNK_BYTES: usize = 2048;
/// Default overlap, in bytes, between adjacent text chunks.
pub const DEFAULT_CHUNK_OVERLAP: usize = 256;
/// Default in-memory write-buffer target, in bytes, used when building the
/// hybrid index. Embedded chunk batches accumulate until they reach this size,
/// then flush to LanceDB in a single `Table::add`. This decouples the (small)
/// Ollama embed batch from the (large) Lance fragment, keeping fragment churn
/// low. Each flush is also a durable checkpoint that `--resume` can restart
/// from, so this is kept modest (128 MiB) to bound how much embedding work a
/// crash can lose while still yielding healthy Lance fragments.
pub const DEFAULT_WRITE_BUFFER_BYTES: usize = 128 << 20;
/// Whether to compact the chunks table after a hybrid-index build by default.
pub const DEFAULT_COMPACT_ON_BUILD: bool = true;
/// Default in-memory buffer target, in bytes, before the metadata ingest flushes
/// a LanceDB fragment. Larger values yield fewer, bigger fragments (better read
/// performance, less metadata) at the cost of higher peak memory during ingest.
pub const DEFAULT_INGEST_BUFFER_BYTES: usize = 512 * 1024 * 1024;

/// Runtime configuration.
///
/// Values are resolved from (in increasing priority): built-in defaults, an
/// optional TOML config file, then explicit overrides supplied by the caller
/// (env vars / CLI flags are applied by the binaries themselves).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Path to the source parquet index (the source of truth).
    pub parquet_path: PathBuf,
    /// Directory containing artifact files on disk, keyed by `artifact_name`.
    ///
    /// Optional: when unset, artifact-text tools degrade gracefully.
    pub artifact_root: Option<PathBuf>,
    /// Base URL of the Ollama server.
    pub ollama_host: String,
    /// Ollama model name used by the CLI agent.
    pub ollama_model: String,
    /// Path to the LanceDB database holding the hybrid keyword+vector index
    /// over artifact text.
    pub lance_path: PathBuf,
    /// Ollama model used to embed document text and queries for semantic
    /// search. This is only the *default* used when building the index; the
    /// model actually used for a query is read back from the index metadata
    /// so search can never use a model that disagrees with the stored vectors.
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            parquet_path: PathBuf::from(DEFAULT_PARQUET),
            artifact_root: None,
            ollama_host: DEFAULT_OLLAMA_HOST.to_string(),
            ollama_model: DEFAULT_MODEL.to_string(),
            lance_path: PathBuf::from(DEFAULT_LANCE),
            embed_model: DEFAULT_EMBED_MODEL.to_string(),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
            write_buffer_bytes: DEFAULT_WRITE_BUFFER_BYTES,
            compact_on_build: DEFAULT_COMPACT_ON_BUILD,
            ingest_buffer_bytes: DEFAULT_INGEST_BUFFER_BYTES,
        }
    }
}

impl Config {
    /// Load configuration from a TOML file. Missing files yield defaults.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)?;
        let cfg: Config = toml::from_str(&text)?;
        Ok(cfg)
    }
}
