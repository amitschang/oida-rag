//! Config-file schema and defaults for the `mcp-chat` binary.
//!
//! The file is one input in the precedence chain `defaults < file < env < flag`;
//! the binary ([`crate::config`] consumers live in `src/bin/mcp-chat.rs`) loads
//! it, then overlays environment and command-line values on top.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

use crate::McpServer;

/// Default OpenAI-compatible chat endpoint (Ollama's `/v1` layer).
pub const DEFAULT_CHAT_HOST: &str = "http://localhost:11434";
/// Default chat model.
pub const DEFAULT_MODEL: &str = "qwen2.5-coder:latest";
/// Default REPL label.
pub const DEFAULT_LABEL: &str = "mcp-chat";
/// A generic default system prompt. Callers with a domain should override it.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a helpful assistant with access to tools provided over MCP. Use the \
available tools to ground every answer; prefer calling a tool over guessing. If \
a tool returns no results, say so plainly.";

/// The on-disk config file. Every field is optional so a partial file layers
/// cleanly over the defaults; flags and env override it in turn.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    /// Base URL of the OpenAI-compatible chat server.
    pub chat_host: Option<String>,
    /// Bearer token for the chat server (prefer the `MCP_CHAT_API_KEY` env var).
    pub chat_api_key: Option<String>,
    /// Chat model name.
    pub model: Option<String>,
    /// Inline system prompt. Mutually exclusive with `system_prompt_file`.
    pub system_prompt: Option<String>,
    /// Path to a file whose contents become the system prompt.
    pub system_prompt_file: Option<String>,
    /// REPL banner/prompt label.
    pub label: Option<String>,
    /// MCP servers to connect to (`[[server]]` tables).
    #[serde(default, rename = "server")]
    pub servers: Vec<ServerEntry>,
}

/// One `[[server]]` entry. Exactly one of `command` (stdio) or `url` (HTTP) must
/// be set.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerEntry {
    /// Namespace override used to disambiguate this server's tools when several
    /// servers are connected. Defaults to a slug derived from the endpoint.
    pub name: Option<String>,
    /// Child-process command for a stdio server.
    pub command: Option<String>,
    /// Arguments for the stdio command.
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables for the stdio command.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// URL for an HTTP (streamable-HTTP) server.
    pub url: Option<String>,
    /// `Authorization` header value for an HTTP server (e.g. `"Bearer …"`).
    pub auth_header: Option<String>,
    /// Extra request headers for an HTTP server.
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

impl ServerEntry {
    /// Convert this entry into a connectable [`McpServer`], validating that
    /// exactly one transport is specified.
    pub fn to_mcp_server(&self) -> anyhow::Result<McpServer> {
        match (&self.command, &self.url) {
            (Some(command), None) => Ok(McpServer::Stdio {
                bin: command.into(),
                args: self.args.clone(),
                env: self.env.clone().into_iter().collect(),
                name: self.name.clone(),
            }),
            (None, Some(url)) => Ok(McpServer::Http {
                url: url.clone(),
                auth_header: self.auth_header.clone(),
                headers: self.headers.clone().into_iter().collect(),
                name: self.name.clone(),
            }),
            (Some(_), Some(_)) => {
                anyhow::bail!("server entry sets both `command` and `url`; use exactly one")
            }
            (None, None) => {
                anyhow::bail!("server entry sets neither `command` nor `url`; use exactly one")
            }
        }
    }
}

impl FileConfig {
    /// Load and parse a TOML config file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing config file {}", path.display()))
    }
}
