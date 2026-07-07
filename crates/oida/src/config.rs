//! OIDA runtime configuration: the domain (Solr) and app (chat) slices, plus
//! the flat aggregate that layers them over the framework's [`CoreConfig`].

use std::path::Path;

use corpus_index::CoreConfig;
use serde::{Deserialize, Serialize};

/// Default model used to drive tool calling.
pub const DEFAULT_MODEL: &str = "qwen2.5-coder:latest";
/// Default OpenAI-compatible chat endpoint (Ollama's `/v1` layer by default;
/// point it at a vLLM sidecar, e.g. `http://localhost:8000`, to serve chat
/// there).
pub const DEFAULT_CHAT_HOST: &str = "http://localhost:11434";
/// Default Solr `q` selecting the tracked corpus for incremental updates.
pub const DEFAULT_SOLR_QUERY: &str = "industrycode:OPIOIDS";
/// Default rows per Solr page (cursorMark paging) during an update.
pub const DEFAULT_SOLR_PAGE_ROWS: usize = 1000;
/// Default Solr field holding the archiver metadata-modified date, used as the
/// incremental watermark (day-resolution; filtered via `fq`, never sorted).
pub const DEFAULT_SOLR_MODIFIED_FIELD: &str = "ddmudate";

/// The default chat system prompt. OIDA-specific text lives here (config), not
/// hard-coded in the generic agent loop, so a different corpus can override it.
pub const DEFAULT_SYSTEM_PROMPT: &str = "
You are an assistant for exploring the OIDA (Opioids Industry Document Archive)
archive of documents (OCR'd PDFs, emails, images) and the network of connections
between them. Use the available tools to ground every answer in the index:
search_documents to find documents, get_document for metadata and artifact
lists, get_artifact_text to read OCR text, and get_related to follow
attachments, mentions, and email threads. Prefer calling tools over guessing.
Cite document ids and Bates numbers. If a tool returns no results, say so
plainly.
";
/// The default assistant label used in the interactive REPL's banners/prompt.
pub const DEFAULT_ASSISTANT_LABEL: &str = "OIDA";

/// Solr source-provider configuration (the OIDA domain slice).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SolrConfig {
    /// Base URL of the archive Solr core used by `ingest`, e.g.
    /// `https://metadata.idl.ucsf.edu/solr/ltdl3`. `None` disables ingest/update;
    /// unused by the serving path.
    pub solr_url: Option<String>,
    /// Solr `q` selecting the corpus to track for updates.
    pub solr_query: String,
    /// Rows per Solr page (cursorMark paging) during an update.
    pub solr_page_rows: usize,
    /// Solr field carrying the archiver metadata-modified date, used as the
    /// inclusive lower-bound watermark for incremental updates.
    pub solr_modified_field: String,
}

impl Default for SolrConfig {
    fn default() -> Self {
        Self {
            solr_url: None,
            solr_query: DEFAULT_SOLR_QUERY.to_string(),
            solr_page_rows: DEFAULT_SOLR_PAGE_ROWS,
            solr_modified_field: DEFAULT_SOLR_MODIFIED_FIELD.to_string(),
        }
    }
}

/// Chat-agent configuration (the CLI app slice; unused by the server).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatConfig {
    /// Base URL of the OpenAI-compatible chat server (the chat agent). Defaults
    /// to a local Ollama; point it at a vLLM sidecar for higher throughput.
    pub chat_host: String,
    /// Optional bearer token for the chat server, sent as `Authorization:
    /// Bearer`. Needed only for a locked-down vLLM (`--api-key`); Ollama ignores
    /// it. Prefer setting it via the `OIDA_CHAT_API_KEY` env var over the file.
    pub chat_api_key: Option<String>,
    /// Chat model name used by the CLI agent.
    pub chat_model: String,
    /// System prompt establishing the assistant's role and tool workflow. The
    /// generic agent loop takes this verbatim — the OIDA-specific wording is a
    /// config value, not code.
    pub system_prompt: String,
    /// Short label the interactive REPL uses in its banner and prompt (e.g.
    /// `OIDA` → `OIDA assistant…` / `oida> `).
    pub assistant_label: String,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            chat_host: DEFAULT_CHAT_HOST.to_string(),
            chat_api_key: None,
            chat_model: DEFAULT_MODEL.to_string(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            assistant_label: DEFAULT_ASSISTANT_LABEL.to_string(),
        }
    }
}

/// Runtime configuration for the OIDA binaries.
///
/// The flat on-disk TOML (and environment/CLI overrides) deserialize into three
/// disjoint slices via `serde(flatten)`: the framework's [`CoreConfig`], the
/// Solr provider's [`SolrConfig`], and the chat app's [`ChatConfig`]. Framework
/// drivers take `&CoreConfig` and never see the other two.
///
/// Values are resolved from (in increasing priority): built-in defaults, an
/// optional TOML config file, then explicit overrides supplied by the caller
/// (env vars / CLI flags are applied by the binaries themselves).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct OidaConfig {
    #[serde(flatten)]
    pub core: CoreConfig,
    #[serde(flatten)]
    pub solr: SolrConfig,
    #[serde(flatten)]
    pub chat: ChatConfig,
}

impl OidaConfig {
    /// Load configuration from a TOML file. Missing files yield defaults.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(path)?;
        let cfg: OidaConfig = toml::from_str(&text)?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_toml_layers_into_slices() {
        // The on-disk format stays flat; `serde(flatten)` must route each key to
        // its slice and leave unmentioned keys at their defaults.
        let toml = r#"
            lance_path = "/data/idx"
            embed_host = "http://vllm:8000"
            solr_url = "https://example/solr/ltdl3"
            solr_page_rows = 500
            chat_model = "qwen2.5-coder:latest"
        "#;
        let cfg: OidaConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.core.lance_path, std::path::PathBuf::from("/data/idx"));
        assert_eq!(cfg.core.embed_host, "http://vllm:8000");
        assert_eq!(cfg.solr.solr_url.as_deref(), Some("https://example/solr/ltdl3"));
        assert_eq!(cfg.solr.solr_page_rows, 500);
        assert_eq!(cfg.chat.chat_model, "qwen2.5-coder:latest");
        // Untouched keys keep their slice defaults.
        assert_eq!(cfg.core.chunk_bytes, corpus_index::config::DEFAULT_CHUNK_BYTES);
        assert_eq!(cfg.solr.solr_modified_field, DEFAULT_SOLR_MODIFIED_FIELD);
    }
}
