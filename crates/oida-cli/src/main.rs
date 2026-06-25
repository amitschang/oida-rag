//! OIDA assistant CLI.
//!
//! Connects to the OIDA MCP server (spawned as a child process) and drives a
//! local Ollama model that calls the server's tools to answer questions about
//! the document archive.

use std::path::PathBuf;

use anyhow::{Context, anyhow};
use clap::{Parser, Subcommand};
use oida_core::{Config, Embedder, Index, hybrid, ingest, raw, update};

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

    /// Path to the LanceDB database directory (overrides config).
    #[arg(long, env = "OIDA_LANCE")]
    lance_path: Option<PathBuf>,

    /// Ollama model to use (overrides config).
    #[arg(long, env = "OIDA_MODEL")]
    model: Option<String>,

    /// Ollama host URL for the chat agent (overrides config).
    #[arg(long, env = "OIDA_OLLAMA_HOST")]
    ollama_host: Option<String>,

    /// OpenAI-compatible host URL for embeddings, e.g. a vLLM sidecar at
    /// `http://localhost:8000` (overrides config). Accepts a comma-separated list
    /// of replica addresses to balance across by least connections, e.g.
    /// `http://vllm-a:8000,http://vllm-b:8000`.
    #[arg(long, env = "OIDA_EMBED_HOST")]
    embed_host: Option<String>,

    /// Bearer token sent with embed requests, for a server started with an API
    /// key (overrides config).
    #[arg(long, env = "OIDA_EMBED_API_KEY")]
    embed_api_key: Option<String>,

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

    /// Concurrent artifact-file reads during a full-text build (overrides config).
    #[arg(long, env = "OIDA_READ_CONCURRENCY")]
    read_concurrency: Option<usize>,

    /// Text chunks per embed request during a full-text build (overrides config).
    #[arg(long, env = "OIDA_EMBED_BATCH")]
    embed_batch: Option<usize>,

    /// Ordered look-ahead window (in jobs) for the embed stage during a full-text
    /// build; 0 = auto (8× concurrency). Decouples output ordering from request
    /// concurrency so a slow request can't starve the backend (overrides config).
    #[arg(long, env = "OIDA_EMBED_LOOKAHEAD")]
    embed_lookahead: Option<usize>,

    /// Verify the configured embed model name matches the index's at query time
    /// (overrides config). Pass `--embed-verify-model false` to bypass.
    #[arg(long, env = "OIDA_EMBED_VERIFY_MODEL")]
    embed_verify_model: Option<bool>,

    /// Path to the oida-mcp-server binary (defaults to a sibling of this exe).
    #[arg(long, env = "OIDA_SERVER_BIN")]
    server_bin: Option<PathBuf>,

    /// Solr core base URL for `update` (overrides config), e.g.
    /// https://metadata.idl.ucsf.edu/solr/ltdl3.
    #[arg(long, env = "OIDA_SOLR_URL", global = true)]
    solr_url: Option<String>,

    /// Solr `q` selecting the corpus for `update` (overrides config).
    #[arg(long, env = "OIDA_SOLR_QUERY", global = true)]
    solr_query: Option<String>,

    /// S3 bucket holding the artifact files; when set, ingest/full-text reads
    /// fetch artifacts from S3 instead of `artifact_root` (overrides config).
    #[arg(long, env = "OIDA_S3_BUCKET", global = true)]
    s3_bucket: Option<String>,

    /// AWS region for the S3 bucket (overrides config).
    #[arg(long, env = "OIDA_S3_REGION", global = true)]
    s3_region: Option<String>,

    /// Custom S3 endpoint URL for S3-compatible stores, e.g. MinIO/Ceph/R2
    /// (overrides config). Credentials come from the standard AWS environment.
    #[arg(long, env = "OIDA_S3_ENDPOINT", global = true)]
    s3_endpoint: Option<String>,

    /// Key prefix prepended to the fan-out artifact path within the bucket
    /// (overrides config).
    #[arg(long, env = "OIDA_S3_PREFIX", global = true)]
    s3_prefix: Option<String>,

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
        /// Store the raw (non-text) artifact bytes in a `raw_artifacts` blob
        /// table, fetched from the configured artifact source (local or S3).
        #[arg(long)]
        store_raw: bool,
        /// Continue an interrupted build, skipping work that is already done.
        /// Select what to resume with `--full-text` and/or `--store-raw` (at
        /// least one required); cannot combine with `--force`.
        #[arg(long, conflicts_with = "force")]
        resume: bool,
    },
    /// Show statistics about the ingested index.
    Stats,
    /// Show the delta an update would apply from the Solr source (read-only).
    ///
    /// Pages the archive Solr core from a modified-date watermark and classifies
    /// each document against the live index without writing anything.
    Update {
        /// Perform the read-only dry run (the default when no mode flag is set).
        #[arg(long)]
        dry_run: bool,
        /// Apply the update by writing to the index. Without --rebuild this is
        /// an incremental in-place apply (upsert new/changed, delete redacted,
        /// invalidate stale chunks); with --rebuild it is a full Solr re-ingest.
        #[arg(long)]
        apply: bool,
        /// With --apply, rebuild the `documents`/`artifacts` tables from scratch
        /// by re-ingesting the whole Solr corpus (drops derived chunks/_meta).
        #[arg(long)]
        rebuild: bool,
        /// Inclusive lower-bound modified-date (ISO-8601, e.g.
        /// 2026-01-01T00:00:00Z). Omit for a full scan from the beginning.
        #[arg(long)]
        since: Option<String>,
        /// Fetch and print one full Solr document (all fields), then exit — to
        /// learn the source schema. Ignores --dry-run.
        #[arg(long)]
        sample_doc: bool,
    },
}


/// Overlay CLI-flag / env-var values onto the config loaded from file. Each
/// field is only touched when the caller supplied it, so the precedence is
/// defaults < config file < env var < CLI flag (clap resolves flag-over-env).
fn apply_overrides(config: &mut Config, args: &Args) {
    if let Some(p) = &args.lance_path {
        config.lance_path = p.clone();
    }
    if let Some(m) = &args.model {
        config.ollama_model = m.clone();
    }
    if let Some(h) = &args.ollama_host {
        config.ollama_host = h.clone();
    }
    if let Some(h) = &args.embed_host {
        config.embed_host = h.clone();
    }
    if let Some(k) = &args.embed_api_key {
        config.embed_api_key = Some(k.clone());
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
    if let Some(v) = args.read_concurrency {
        config.read_concurrency = v;
    }
    if let Some(v) = args.embed_batch {
        config.embed_batch = v;
    }
    if let Some(v) = args.embed_lookahead {
        config.embed_lookahead = v;
    }
    if let Some(v) = args.embed_verify_model {
        config.embed_verify_model = v;
    }
    if let Some(u) = &args.solr_url {
        config.solr_url = Some(u.clone());
    }
    if let Some(q) = &args.solr_query {
        config.solr_query = q.clone();
    }
    if let Some(b) = &args.s3_bucket {
        config.s3_bucket = Some(b.clone());
    }
    if let Some(r) = &args.s3_region {
        config.s3_region = Some(r.clone());
    }
    if let Some(e) = &args.s3_endpoint {
        config.s3_endpoint = Some(e.clone());
    }
    if let Some(p) = &args.s3_prefix {
        config.s3_prefix = Some(p.clone());
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
        Some(Command::Ingest { full_text, force, store_raw, resume }) => {
            if *store_raw {
                config.store_raw_artifacts = true;
            }
            return run_ingest(&config, *full_text, *force, *resume).await;
        }
        Some(Command::Stats) => return run_stats(&config).await,
        Some(Command::Update { dry_run, apply, rebuild, since, sample_doc }) => {
            return run_update(&config, *dry_run, *apply, *rebuild, since.clone(), *sample_doc).await;
        }
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
    let do_raw = config.store_raw_artifacts;
    let do_text = full_text;

    // Resuming continues an interrupted build only; the metadata tables it
    // relies on are already present, so skip re-ingesting them (which would
    // otherwise refuse to run or wipe the partial tables). `--store-raw` and
    // `--full-text` select *which* derived data to resume; at least one is
    // required. Re-run either after `update --apply` to pick up new/changed
    // documents.
    if resume {
        if !do_raw && !do_text {
            anyhow::bail!(
                "--resume needs a selector: pass --full-text and/or --store-raw \
                 to choose what to resume"
            );
        }
        let index = Index::open(config)
            .await
            .context("opening index to resume (run a metadata ingest first)")?;
        return run_derived_builds(config, &index, do_raw, do_text, false, true).await;
    }

    eprintln!(
        "Ingesting metadata from Solr {} (q={:?})…",
        config.solr_url.as_deref().unwrap_or("<unset>"),
        config.solr_query
    );
    let stats = ingest::ingest_from_solr(config, None, force)
        .await
        .context("ingesting metadata")?;
    eprintln!(
        "Ingested {} documents and {} artifacts.",
        stats.documents, stats.artifacts
    );

    if do_raw || do_text {
        let index = Index::open(config).await.context("opening index")?;
        run_derived_builds(config, &index, do_raw, do_text, force, false).await?;
    }
    Ok(())
}

/// Build the requested derived stores. When both raw storage and the full-text
/// index are requested they run concurrently on a single shared status line
/// (raw is network/disk bound, full-text is GPU bound, so overlapping them
/// keeps both busy); otherwise each runs on its own.
async fn run_derived_builds(
    config: &Config,
    index: &Index,
    do_raw: bool,
    do_text: bool,
    force: bool,
    resume: bool,
) -> anyhow::Result<()> {
    match (do_raw, do_text) {
        (true, true) => {
            let embedder = make_embedder(config)?;
            eprintln!(
                "Building raw store + full-text index concurrently (embed model '{}')…",
                config.embed_model
            );
            let (rstats, hstats) =
                oida_core::build_raw_and_text(config, index, &embedder, force, resume).await?;
            eprintln!(
                "Stored {} raw artifacts ({} already present, {} missing).",
                rstats.stored, rstats.skipped, rstats.missing
            );
            eprintln!(
                "Indexed {} chunks across {} documents (dim {}).",
                hstats.chunks, hstats.documents, hstats.dim
            );
        }
        (true, false) => {
            eprintln!("Storing raw (non-text) artifacts…");
            let rstats = raw::build(config, index, resume).await?;
            eprintln!(
                "Stored {} raw artifacts ({} already present, {} missing).",
                rstats.stored, rstats.skipped, rstats.missing
            );
        }
        (false, true) => {
            let embedder = make_embedder(config)?;
            eprintln!(
                "Building hybrid text index with embed model '{}'…",
                config.embed_model
            );
            let hstats = hybrid::build(config, index, &embedder, force, resume).await?;
            eprintln!(
                "Indexed {} chunks across {} documents (dim {}).",
                hstats.chunks, hstats.documents, hstats.dim
            );
        }
        (false, false) => {}
    }
    Ok(())
}

/// Construct the embedding client from the configured host, model, and key.
fn make_embedder(config: &Config) -> anyhow::Result<Embedder> {
    Embedder::new(
        &config.embed_host,
        config.embed_model.clone(),
        config.embed_api_key.clone(),
    )
}

/// Run the `stats` subcommand: report index row counts and hybrid metadata.
async fn run_stats(config: &Config) -> anyhow::Result<()> {
    let index = Index::open(config).await.context("opening index")?;
    let (documents, artifacts) = index.counts().await?;
    println!("OIDA index ({})", config.lance_path.display());
    println!("  documents:      {documents}");
    println!("  artifacts:      {artifacts}");

    let sizes = index.store_sizes().await.context("summarising artifact sizes")?;
    println!("  full-text artifacts:");
    println!(
        "    referenced:   {} ({})",
        sizes.text_logical_count,
        human_bytes(sizes.text_logical_bytes)
    );
    println!(
        "    in archive:   {} ({})",
        sizes.text_real_count,
        human_bytes(sizes.text_real_bytes)
    );
    println!("  raw artifacts:");
    println!(
        "    referenced:   {} ({})",
        sizes.raw_logical_count,
        human_bytes(sizes.raw_logical_bytes)
    );
    match (sizes.raw_real_count, sizes.raw_real_bytes) {
        (Some(count), Some(bytes)) => {
            println!("    in archive:   {} ({})", count, human_bytes(bytes));
        }
        _ => println!("    in archive:   not stored (run `oida-cli ingest --store-raw`)"),
    }

    match hybrid::HybridIndex::open(config).await {
        Ok(h) => {
            let s = h.stats().await?;
            println!("  full-text:      built");
            println!("    chunks:       {}", s.chunks);
            println!("    embed model:  {}", s.embed_model);
            println!("    vector dim:   {}", s.dim);
            println!("    chunk bytes:  {}", s.chunk_bytes);
            println!("    chunk overlap:{}", s.chunk_overlap);
            println!("    built at:     {} (unix)", s.built_at);
        }
        Err(_) => println!("  full-text:      not built (run `oida-cli ingest --full-text`)"),
    }
    Ok(())
}

/// Format a byte count as a human-readable size using binary (1024) units.
fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Run the `update` subcommand. With `--sample-doc` it dumps one Solr document
/// for schema inspection; with `--apply --rebuild` it re-ingests the whole Solr
/// corpus into the `documents`/`artifacts` tables; otherwise it runs the
/// read-only dry-run differ, reporting the delta without writing.
async fn run_update(
    config: &Config,
    dry_run: bool,
    apply: bool,
    rebuild: bool,
    since: Option<String>,
    sample_doc: bool,
) -> anyhow::Result<()> {
    if sample_doc {
        let doc = update::sample_doc(config, since.as_deref())
            .await
            .context("fetching sample Solr document")?;
        match doc {
            Some(d) => println!("{}", serde_json::to_string_pretty(&d)?),
            None => println!("No documents matched."),
        }
        return Ok(());
    }

    if apply {
        if rebuild {
            eprintln!(
                "Rebuilding documents/artifacts from Solr {} (q={:?}){}…",
                config.solr_url.as_deref().unwrap_or("<unset>"),
                config.solr_query,
                since
                    .as_deref()
                    .map(|s| format!(" since {s}"))
                    .unwrap_or_default()
            );
            let stats = ingest::ingest_from_solr(config, since.as_deref(), true)
                .await
                .context("re-ingesting from Solr")?;
            println!("Solr re-ingest complete ({})", config.lance_path.display());
            println!("  documents:  {}", stats.documents);
            println!("  artifacts:  {}", stats.artifacts);
            if config.store_raw_artifacts {
                let index = Index::open(config).await.context("opening index")?;
                eprintln!("Storing raw (non-text) artifacts…");
                let rstats = raw::build(config, &index, false).await?;
                println!("  raw:        {}", rstats.stored);
            }
            eprintln!(
                "\nNote: derived chunks/_meta were dropped; run `ingest --full-text` to rebuild \
                 the text-search index."
            );
            return Ok(());
        }

        let index = Index::open(config)
            .await
            .context("opening index (run a metadata ingest first)")?;
        eprintln!(
            "Applying incremental update from Solr {} (q={:?}){}…",
            config.solr_url.as_deref().unwrap_or("<unset>"),
            config.solr_query,
            since
                .as_deref()
                .map(|s| format!(" since {s}"))
                .unwrap_or_else(|| " since stored watermark".to_string())
        );
        let stats = update::apply(config, &index, since.as_deref())
            .await
            .context("applying incremental update")?;
        println!("Incremental update applied ({})", config.lance_path.display());
        match &stats.since {
            Some(s) => println!("  since (modified ≥):  {s}"),
            None => println!("  since (modified ≥):  <full scan>"),
        }
        println!("  solr numFound:       {}", stats.num_found);
        println!("  pages fetched:       {}", stats.pages);
        println!("  documents scanned:   {}", stats.scanned);
        println!("  new (inserted):      {}", stats.new);
        println!("  changed (upserted):  {}", stats.changed);
        println!("  unchanged (skipped): {}", stats.unchanged);
        println!("  redacted (deleted):  {}", stats.redacted);
        println!("  documents upserted:  {}", stats.upserted);
        println!("  chunks invalidated:  {}", stats.chunks_invalidated);
        println!("  raw invalidated:     {}", stats.raw_invalidated);
        match &stats.watermark {
            Some(w) => println!("  watermark written:   {w}"),
            None => println!("  watermark written:   <unchanged>"),
        }
        if stats.changed > 0 || stats.redacted > 0 {
            eprintln!(
                "\nNote: stale chunks (and raw artifacts) for changed/redacted documents were \
                 removed; run `ingest --resume` (add `--store-raw` to also refresh raw bytes) to \
                 re-process the affected documents."
            );
        }
        return Ok(());
    }

    if !dry_run {
        anyhow::bail!(
            "pass --dry-run for the read-only differ, or --apply --rebuild for a full Solr re-ingest"
        );
    }

    let index = Index::open(config)
        .await
        .context("opening index (run a metadata ingest first)")?;
    eprintln!(
        "Querying Solr {} (q={:?}){}…",
        config.solr_url.as_deref().unwrap_or("<unset>"),
        config.solr_query,
        since
            .as_deref()
            .map(|s| format!(" since {s}"))
            .unwrap_or_default()
    );

    let plan = update::dry_run(config, &index, since.as_deref()).await?;

    println!("Update dry run ({})", config.lance_path.display());
    match &plan.since {
        Some(s) => println!("  since (modified ≥):  {s}"),
        None => println!("  since (modified ≥):  <full scan>"),
    }
    println!("  solr numFound:       {}", plan.num_found);
    println!("  pages fetched:       {}", plan.pages);
    println!("  documents scanned:   {}", plan.scanned);
    println!("  new (insert):        {}", plan.new);
    println!("  changed (re-embed):  {}", plan.changed);
    println!("  unchanged (skip):    {}", plan.unchanged);
    println!("  redacted (delete):   {}", plan.redacted);
    println!("  text artifacts to fetch: {}", plan.refetch_text_artifacts);
    match &plan.max_modified {
        Some(m) => println!("  next watermark:      {m}"),
        None => println!("  next watermark:      <none>"),
    }
    eprintln!(
        "\nNote: classification compares the Solr artifact name/md5 set against the \
         indexed artifacts (content changes / redactions). No writes performed. Use \
         --apply to apply the delta in place, or --apply --rebuild for a full Solr re-ingest."
    );
    Ok(())
}


