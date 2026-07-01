//! MCP client: spawns the OIDA MCP server as a child process over stdio and
//! exposes helpers to list and call its tools.

use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{Context, anyhow};
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, Tool};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use serde_json::{Map, Value};

/// A connected MCP client backed by a child-process server.
pub struct McpClient {
    service: RunningService<RoleClient, ()>,
}

impl McpClient {
    /// Spawn the server binary and complete the MCP handshake.
    ///
    /// The server inherits this process's environment (so `OIDA_*` config is
    /// passed through) and stderr (so its logs are visible to the user).
    pub async fn connect(server_bin: PathBuf) -> anyhow::Result<Self> {
        let mut cmd = tokio::process::Command::new(&server_bin);
        cmd.stderr(Stdio::inherit());
        let transport = TokioChildProcess::new(cmd)
            .with_context(|| format!("spawning MCP server {}", server_bin.display()))?;
        let service = ()
            .serve(transport)
            .await
            .context("MCP handshake with server failed")?;
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
    pub async fn call_tool(&self, name: &str, arguments: Map<String, Value>) -> anyhow::Result<String> {
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
