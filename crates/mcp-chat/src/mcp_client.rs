//! MCP client: connects to an MCP server over one of several transports (a
//! child process over stdio, or a streamable-HTTP endpoint) and exposes helpers
//! to list and call its tools.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{Context, anyhow};
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, Tool};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use serde_json::{Map, Value};

/// How to reach one MCP server.
pub enum McpServer {
    /// Spawn a child process that speaks MCP over stdio. The child inherits this
    /// process's environment (plus `env`) and stderr (so its logs are visible).
    Stdio {
        /// Path to the server binary.
        bin: PathBuf,
        /// Extra command-line arguments.
        args: Vec<String>,
        /// Extra environment variables layered onto the inherited environment.
        env: Vec<(String, String)>,
    },
    /// Connect to a streamable-HTTP MCP endpoint.
    Http {
        /// Base URL of the MCP endpoint, e.g. `http://localhost:8000/mcp`.
        url: String,
        /// Value for the `Authorization` header (e.g. `"Bearer …"`), if any.
        auth_header: Option<String>,
        /// Extra request headers, as `(name, value)` pairs.
        headers: Vec<(String, String)>,
    },
}

impl McpServer {
    /// A short human-readable description of the endpoint for banners/logs.
    pub fn describe(&self) -> String {
        match self {
            McpServer::Stdio { bin, .. } => format!("child process {}", bin.display()),
            McpServer::Http { url, .. } => format!("HTTP endpoint {url}"),
        }
    }

    /// A short default namespace slug for this server, used to disambiguate tool
    /// names when more than one server is connected (§ tool namespacing). For a
    /// child process it's the binary's file stem; for HTTP, the URL host.
    pub fn default_namespace(&self) -> String {
        match self {
            McpServer::Stdio { bin, .. } => bin
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("server")
                .to_string(),
            McpServer::Http { url, .. } => reqwest::Url::parse(url)
                .ok()
                .and_then(|u| u.host_str().map(str::to_string))
                .unwrap_or_else(|| "server".to_string()),
        }
    }
}

/// A connected MCP client, transport-agnostic once the handshake completes.
pub struct McpClient {
    service: RunningService<RoleClient, ()>,
}

impl McpClient {
    /// Connect over the transport described by `server` and complete the MCP
    /// handshake.
    pub async fn connect(server: &McpServer) -> anyhow::Result<Self> {
        let service = match server {
            McpServer::Stdio { bin, args, env } => {
                let mut cmd = tokio::process::Command::new(bin);
                cmd.args(args);
                for (key, value) in env {
                    cmd.env(key, value);
                }
                cmd.stderr(Stdio::inherit());
                let transport = TokioChildProcess::new(cmd)
                    .with_context(|| format!("spawning MCP server {}", bin.display()))?;
                ()
                    .serve(transport)
                    .await
                    .context("MCP handshake with server failed")?
            }
            McpServer::Http {
                url,
                auth_header,
                headers,
            } => {
                let config = build_http_config(url, auth_header.as_deref(), headers)?;
                // Let rmcp build its own HTTP client (its reqwest version differs
                // from ours); we only supply the config.
                let transport = StreamableHttpClientTransport::from_config(config);
                ()
                    .serve(transport)
                    .await
                    .with_context(|| format!("MCP handshake with {url} failed"))?
            }
        };
        Ok(Self { service })
    }

    /// List the tools advertised by the server.
    pub async fn list_tools(&self) -> anyhow::Result<Vec<Tool>> {
        self.service
            .list_all_tools()
            .await
            .context("listing MCP tools")
    }

    /// Invoke a tool and return a string representation of its result,
    /// preferring structured JSON content.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Map<String, Value>,
    ) -> anyhow::Result<String> {
        let mut params = CallToolRequestParams::new(name.to_string());
        params.arguments = Some(arguments);
        let result = self
            .service
            .call_tool(params)
            .await
            .with_context(|| format!("calling tool {name}"))?;

        if result.is_error == Some(true) {
            let text = render_content(&result);
            return Err(anyhow!("tool {name} reported an error: {text}"));
        }

        if let Some(structured) = result.structured_content {
            return Ok(serde_json::to_string(&structured)?);
        }
        Ok(render_content(&result))
    }

    /// Shut down the server connection.
    pub async fn shutdown(self) {
        let _ = self.service.cancel().await;
    }
}

/// Build the streamable-HTTP transport config, applying an optional
/// `Authorization` header and any extra custom headers. The header types come
/// from the `http` crate, which is version-unified across the tree (unlike
/// reqwest, whose rmcp copy differs from ours — hence `from_config`).
fn build_http_config(
    url: &str,
    auth_header: Option<&str>,
    headers: &[(String, String)],
) -> anyhow::Result<rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig> {
    use http::{HeaderName, HeaderValue};
    use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;

    reqwest::Url::parse(url).with_context(|| format!("invalid MCP URL {url}"))?;

    let mut config = StreamableHttpClientTransportConfig::with_uri(url.to_string());
    if let Some(auth) = auth_header {
        config = config.auth_header(auth.to_string());
    }
    if !headers.is_empty() {
        let mut map = HashMap::new();
        for (name, value) in headers {
            let header_name = HeaderName::from_bytes(name.as_bytes())
                .with_context(|| format!("invalid header name {name}"))?;
            let header_value = HeaderValue::from_str(value)
                .with_context(|| format!("invalid value for header {name}"))?;
            map.insert(header_name, header_value);
        }
        config = config.custom_headers(map);
    }
    Ok(config)
}

/// Concatenate the textual parts of a tool result.
fn render_content(result: &rmcp::model::CallToolResult) -> String {
    let mut out = String::new();
    for c in &result.content {
        if let Some(text) = c.as_text() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&text.text);
        }
    }
    out
}
