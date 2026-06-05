//! OIDA assistant CLI.
//!
//! Connects to the OIDA MCP server (spawned as a child process) and drives a
//! local Ollama model that calls the server's tools to answer questions about
//! the document archive.

use std::path::PathBuf;

use anyhow::{Context, anyhow};
use clap::Parser;
use oida_core::Config;

mod agent;
mod mcp_client;
mod ollama;
mod repl;

use agent::Agent;
use mcp_client::McpClient;

/// CLI arguments. Flags override config-file and environment values.
#[derive(Debug, Parser)]
#[command(name = "oida-cli", about = "Chat with the OIDA document archive via a local LLM")]
struct Args {
    /// Path to a TOML config file.
    #[arg(long, env = "OIDA_CONFIG", default_value = "oida.toml")]
    config: PathBuf,

    /// Ollama model to use (overrides config).
    #[arg(long, env = "OIDA_MODEL")]
    model: Option<String>,

    /// Ollama host URL (overrides config).
    #[arg(long, env = "OIDA_OLLAMA_HOST")]
    ollama_host: Option<String>,

    /// Directory containing artifact files on disk (overrides config).
    #[arg(long, env = "OIDA_ARTIFACT_ROOT")]
    artifact_root: Option<PathBuf>,

    /// Path to the oida-mcp-server binary (defaults to a sibling of this exe).
    #[arg(long, env = "OIDA_SERVER_BIN")]
    server_bin: Option<PathBuf>,

    /// Run a single query non-interactively and exit.
    #[arg(long)]
    once: Option<String>,
}

/// Resolve the server binary path, defaulting to a sibling of the current exe.
fn resolve_server_bin(explicit: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p);
    }
    let exe = std::env::current_exe().context("locating current executable")?;
    let dir = exe
        .parent()
        .ok_or_else(|| anyhow!("current exe has no parent directory"))?;
    let candidate = dir.join("oida-mcp-server");
    if candidate.exists() {
        Ok(candidate)
    } else {
        // Fall back to PATH lookup.
        Ok(PathBuf::from("oida-mcp-server"))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let args = Args::parse();

    let mut config = Config::load(&args.config)?;
    if let Some(m) = &args.model {
        config.ollama_model = m.clone();
    }
    if let Some(h) = &args.ollama_host {
        config.ollama_host = h.clone();
    }
    if let Some(r) = &args.artifact_root {
        config.artifact_root = Some(r.clone());
    }

    let server_bin = resolve_server_bin(args.server_bin.clone())?;
    eprintln!("Starting OIDA MCP server: {} (first run builds the cache)…", server_bin.display());

    let mcp = McpClient::connect(server_bin).await?;
    let tools = mcp.list_tools().await?;
    repl::print_tools(&tools);

    let agent = Agent::new(&config.ollama_host, config.ollama_model.clone(), mcp, &tools)?;
    eprintln!("Using model: {}\n", config.ollama_model);

    let result = if let Some(query) = args.once.clone() {
        repl::run_once(&agent, query).await
    } else {
        repl::run_repl(&agent).await
    };

    agent.shutdown().await;
    result
}
