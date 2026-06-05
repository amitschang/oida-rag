//! Configuration shared by the MCP server and the CLI client.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Default Ollama model used to drive tool calling.
pub const DEFAULT_MODEL: &str = "qwen2.5-coder:latest";
/// Default Ollama HTTP endpoint.
pub const DEFAULT_OLLAMA_HOST: &str = "http://localhost:11434";
/// Default path to the OIDA parquet index.
pub const DEFAULT_PARQUET: &str = "oida-index-by-artifact.parquet";
/// Default path to the persistent DuckDB cache built from the parquet.
pub const DEFAULT_CACHE: &str = "oida.duckdb";

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
    /// Path to the persistent DuckDB cache derived from the parquet.
    pub cache_path: PathBuf,
    /// Directory containing artifact files on disk, keyed by `artifact_name`.
    ///
    /// Optional: when unset, artifact-text tools degrade gracefully.
    pub artifact_root: Option<PathBuf>,
    /// Base URL of the Ollama server.
    pub ollama_host: String,
    /// Ollama model name used by the CLI agent.
    pub ollama_model: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            parquet_path: PathBuf::from(DEFAULT_PARQUET),
            cache_path: PathBuf::from(DEFAULT_CACHE),
            artifact_root: None,
            ollama_host: DEFAULT_OLLAMA_HOST.to_string(),
            ollama_model: DEFAULT_MODEL.to_string(),
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
