//! OIDA MCP server.
//!
//! Exposes the OIDA index as MCP tools over stdio. All logging goes to stderr
//! so stdout stays a clean JSON-RPC channel.
//!
//! The index must already be ingested (`oida-cli ingest`); this server never
//! builds it. If no index is found the server still starts, but the tools
//! return a helpful error telling the caller to run an ingest.

use std::sync::Arc;

use anyhow::Context;
use oida_core::{ArtifactSource, Config, Index};
use rmcp::ServiceExt;
use rmcp::transport::stdio;

mod tools;
use tools::OidaServer;

fn load_config() -> anyhow::Result<Config> {
    let path = std::env::var("OIDA_CONFIG").unwrap_or_else(|_| "oida.toml".to_string());
    let mut config = Config::load(&path)?;
    // Environment overrides (handy without a config file).
    if let Ok(v) = std::env::var("OIDA_ARTIFACT_ROOT") {
        config.artifact_root = Some(v.into());
    }
    if let Ok(v) = std::env::var("OIDA_LANCE") {
        config.lance_path = v.into();
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

    let index = Arc::new(
        Index::open(&config)
            .await
            .context("opening index (run `oida-cli ingest` if it has not been built)")?,
    );

    // Open the hybrid text index if it has been built; degrade gracefully so
    // the metadata tools still work when it is absent.
    let hybrid = match oida_core::hybrid::HybridIndex::open(&config).await {
        Ok(h) => {
            tracing::info!("hybrid text index loaded");
            Some(h)
        }
        Err(e) => {
            tracing::info!("hybrid text index unavailable ({e}); hybrid_search disabled");
            None
        }
    };

    // Build the artifact byte source (local dir or S3) once; `get_artifact_text`
    // returns a status when none is configured.
    let source = match ArtifactSource::from_config(&config) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("artifact source unavailable ({e}); get_artifact_text disabled");
            None
        }
    };

    tracing::info!("OIDA MCP server ready; serving over stdio");

    let service = OidaServer::new(index, hybrid, source)
        .serve(stdio())
        .await
        .context("starting MCP server")?;
    service.waiting().await?;
    Ok(())
}
