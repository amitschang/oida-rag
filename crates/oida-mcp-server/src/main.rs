//! OIDA MCP server.
//!
//! Exposes the OIDA index as MCP tools over stdio. All logging goes to stderr
//! so stdout stays a clean JSON-RPC channel.
//!
//! Usage:
//!   oida-mcp-server                # serve over stdio (builds cache if absent)
//!   oida-mcp-server build-cache    # build the DuckDB cache and exit
//!   oida-mcp-server build-cache --force

use std::sync::Arc;

use anyhow::Context;
use oida_core::{Config, Index};
use rmcp::ServiceExt;
use rmcp::transport::stdio;

mod tools;
use tools::OidaServer;

fn load_config() -> anyhow::Result<Config> {
    let path = std::env::var("OIDA_CONFIG").unwrap_or_else(|_| "oida.toml".to_string());
    let mut config = Config::load(&path)?;
    // Environment overrides (handy without a config file).
    if let Ok(v) = std::env::var("OIDA_PARQUET") {
        config.parquet_path = v.into();
    }
    if let Ok(v) = std::env::var("OIDA_CACHE") {
        config.cache_path = v.into();
    }
    if let Ok(v) = std::env::var("OIDA_ARTIFACT_ROOT") {
        config.artifact_root = Some(v.into());
    }
    Ok(config)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = load_config()?;
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.first().map(String::as_str) == Some("build-cache") {
        let force = args.iter().any(|a| a == "--force");
        Index::build_cache(&config, force)?;
        return Ok(());
    }

    // Ensure the cache exists before serving.
    if !config.cache_path.exists() {
        tracing::info!("cache not found; building it (one-time)...");
        Index::build_cache(&config, false)?;
    }

    let index = Arc::new(Index::open(&config).context("opening index")?);
    tracing::info!("OIDA MCP server ready; serving over stdio");

    let service = OidaServer::new(index, Arc::new(config))
        .serve(stdio())
        .await
        .context("starting MCP server")?;
    service.waiting().await?;
    Ok(())
}
