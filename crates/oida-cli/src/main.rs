//! OIDA assistant CLI.
//!
//! Connects to the OIDA MCP server (spawned as a child process) and drives a
//! local Ollama model that calls the server's tools to answer questions about
//! the document archive.

use std::path::PathBuf;

use anyhow::{Context, anyhow};
use clap::{Parser, Subcommand};
use oida_core::{Config, Embedder, Index, hybrid};

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

    /// Path to the source parquet index (overrides config).
    #[arg(long, env = "OIDA_PARQUET")]
    parquet_path: Option<PathBuf>,

    /// Path to the LanceDB database directory (overrides config).
    #[arg(long, env = "OIDA_LANCE")]
    lance_path: Option<PathBuf>,

    /// Ollama model to use (overrides config).
    #[arg(long, env = "OIDA_MODEL")]
    model: Option<String>,

    /// Ollama host URL (overrides config).
    #[arg(long, env = "OIDA_OLLAMA_HOST")]
    ollama_host: Option<String>,

    /// Directory containing artifact files on disk (overrides config).
    #[arg(long, env = "OIDA_ARTIFACT_ROOT")]
    artifact_root: Option<PathBuf>,

    /// Embedding model for the full-text index (overrides config).
    #[arg(long, env = "OIDA_EMBED_MODEL")]
    embed_model: Option<String>,

    /// Target size, in bytes, of each embedded text chunk (overrides config).
    #[arg(long, env = "OIDA_CHUNK_BYTES")]
    chunk_bytes: Option<usize>,

    /// Overlap, in bytes, between adjacent text chunks (overrides config).
    #[arg(long, env = "OIDA_CHUNK_OVERLAP")]
    chunk_overlap: Option<usize>,

    /// Write-buffer target, in bytes, for the hybrid index build (overrides config).
    #[arg(long, env = "OIDA_WRITE_BUFFER_BYTES")]
    write_buffer_bytes: Option<usize>,

    /// Compact the chunks table after a hybrid build (overrides config).
    #[arg(long, env = "OIDA_COMPACT_ON_BUILD")]
    compact_on_build: Option<bool>,

    /// Buffer target, in bytes, before the metadata ingest flushes (overrides config).
    #[arg(long, env = "OIDA_INGEST_BUFFER_BYTES")]
    ingest_buffer_bytes: Option<usize>,

    /// Concurrent embed requests in flight during a full-text build (overrides config).
    #[arg(long, env = "OIDA_EMBED_CONCURRENCY")]
    embed_concurrency: Option<usize>,

    /// Context window (tokens) sent as `num_ctx` on embed requests; 0 omits it
    /// and uses the model/server default (overrides config).
    #[arg(long, env = "OIDA_EMBED_NUM_CTX")]
    embed_num_ctx: Option<usize>,

    /// Path to the oida-mcp-server binary (defaults to a sibling of this exe).
    #[arg(long, env = "OIDA_SERVER_BIN")]
    server_bin: Option<PathBuf>,

    /// Run a single query non-interactively and exit.
    #[arg(long)]
    once: Option<String>,

    /// Subcommand to run. When omitted, starts an interactive chat session.
    #[command(subcommand)]
    command: Option<Command>,
}

/// Top-level subcommands. Chat is the default when none is given.
#[derive(Debug, Subcommand)]
enum Command {
    /// Ingest the parquet into LanceDB (required before chatting).
    ///
    /// Loads document/artifact metadata always; pass `--full-text` to also
    /// build the hybrid keyword + semantic text index over the OCR artifacts.
    Ingest {
        /// Also build the hybrid full-text/semantic index.
        #[arg(long)]
        full_text: bool,
        /// Replace an existing index instead of failing.
        #[arg(long)]
        force: bool,
        /// Continue an interrupted full-text build, skipping documents that are
        /// already indexed. Implies `--full-text`; cannot combine with `--force`.
        #[arg(long, conflicts_with = "force")]
        resume: bool,
    },
    /// Show statistics about the ingested index.
    Stats,
}


/// Overlay CLI-flag / env-var values onto the config loaded from file. Each
/// field is only touched when the caller supplied it, so the precedence is
/// defaults < config file < env var < CLI flag (clap resolves flag-over-env).
fn apply_overrides(config: &mut Config, args: &Args) {
    if let Some(p) = &args.parquet_path {
        config.parquet_path = p.clone();
    }
    if let Some(p) = &args.lance_path {
        config.lance_path = p.clone();
    }
    if let Some(m) = &args.model {
        config.ollama_model = m.clone();
    }
    if let Some(h) = &args.ollama_host {
        config.ollama_host = h.clone();
    }
    if let Some(r) = &args.artifact_root {
        config.artifact_root = Some(r.clone());
    }
    if let Some(m) = &args.embed_model {
        config.embed_model = m.clone();
    }
    if let Some(v) = args.chunk_bytes {
        config.chunk_bytes = v;
    }
    if let Some(v) = args.chunk_overlap {
        config.chunk_overlap = v;
    }
    if let Some(v) = args.write_buffer_bytes {
        config.write_buffer_bytes = v;
    }
    if let Some(v) = args.compact_on_build {
        config.compact_on_build = v;
    }
    if let Some(v) = args.ingest_buffer_bytes {
        config.ingest_buffer_bytes = v;
    }
    if let Some(v) = args.embed_concurrency {
        config.embed_concurrency = v;
    }
    if let Some(v) = args.embed_num_ctx {
        // 0 is the escape hatch for "omit num_ctx and defer to the model default".
        config.embed_num_ctx = (v > 0).then_some(v);
    }
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
                .unwrap_or_else(|_| "warn,oida_core=info".into()),
        )
        .init();

    let args = Args::parse();

    let mut config = Config::load(&args.config)?;
    apply_overrides(&mut config, &args);

    // Dispatch management subcommands before touching the MCP server.
    match &args.command {
        Some(Command::Ingest { full_text, force, resume }) => {
            return run_ingest(&config, *full_text, *force, *resume).await;
        }
        Some(Command::Stats) => return run_stats(&config).await,
        None => {}
    }

    // Chatting requires an ingested index; never build it implicitly.
    if !Index::is_ingested(&config).await {
        eprintln!(
            "No index found at {}.\nRun `oida-cli ingest` (add --full-text for semantic search) \
             before chatting.",
            config.lance_path.display()
        );
        std::process::exit(1);
    }

    let server_bin = resolve_server_bin(args.server_bin.clone())?;
    eprintln!("Starting OIDA MCP server: {}…", server_bin.display());

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

/// Run the `ingest` subcommand: load metadata, then optionally the full-text
/// index, reporting progress to stderr.
async fn run_ingest(
    config: &Config,
    full_text: bool,
    force: bool,
    resume: bool,
) -> anyhow::Result<()> {
    // Resuming continues an interrupted full-text build only; the metadata
    // tables it relies on are already present, so skip re-ingesting them (which
    // would otherwise refuse to run or wipe the partial chunks table).
    if resume {
        let index = Index::open(config)
            .await
            .context("opening index to resume (run a metadata ingest first)")?;
        let model = &config.embed_model;
        let embedder =
            Embedder::new(&config.ollama_host, model.to_string(), config.embed_num_ctx)?;
        eprintln!("Resuming hybrid text index build with embed model '{model}'…");
        let hstats = hybrid::build(config, &index, &embedder, false, true).await?;
        eprintln!(
            "Indexed {} chunks across {} documents (dim {}).",
            hstats.chunks, hstats.documents, hstats.dim
        );
        return Ok(());
    }

    eprintln!("Ingesting metadata from {}…", config.parquet_path.display());
    let stats = Index::ingest_metadata(config, force)
        .await
        .context("ingesting metadata")?;
    eprintln!(
        "Ingested {} documents and {} artifacts.",
        stats.documents, stats.artifacts
    );

    if full_text {
        let index = Index::open(config).await.context("opening index")?;
        let model = &config.embed_model;
        let embedder =
            Embedder::new(&config.ollama_host, model.to_string(), config.embed_num_ctx)?;
        eprintln!("Building hybrid text index with embed model '{model}'…");
        let hstats = hybrid::build(config, &index, &embedder, force, false).await?;
        eprintln!(
            "Indexed {} chunks across {} documents (dim {}).",
            hstats.chunks, hstats.documents, hstats.dim
        );
    }
    Ok(())
}

/// Run the `stats` subcommand: report index row counts and hybrid metadata.
async fn run_stats(config: &Config) -> anyhow::Result<()> {
    let index = Index::open(config).await.context("opening index")?;
    let (documents, artifacts) = index.counts().await?;
    println!("OIDA index ({})", config.lance_path.display());
    println!("  documents:      {documents}");
    println!("  artifacts:      {artifacts}");

    match hybrid::HybridIndex::open(config).await {
        Ok(h) => {
            let s = h.stats().await?;
            println!("  full-text:      built");
            println!("    chunks:       {}", s.chunks);
            println!("    embed model:  {}", s.embed_model);
            println!("    vector dim:   {}", s.dim);
            println!("    model digest: {}", s.model_digest);
            println!("    chunk bytes:  {}", s.chunk_bytes);
            println!("    chunk overlap:{}", s.chunk_overlap);
            println!("    built at:     {} (unix)", s.built_at);
        }
        Err(_) => println!("  full-text:      not built (run `oida-cli ingest --full-text`)"),
    }
    Ok(())
}

