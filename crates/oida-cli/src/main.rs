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
#[command(
    name = "oida-cli",
    about = "Query and maintain the OIDA document archive",
    arg_required_else_help = true
)]
struct Args {
    #[command(flatten)]
    global: GlobalArgs,

    /// Subcommand to run.
    #[command(subcommand)]
    command: Command,
}

/// Options shared by every subcommand (resolved before dispatch).
#[derive(Debug, clap::Args)]
struct GlobalArgs {
    /// Path to a TOML config file.
    #[arg(long, env = "OIDA_CONFIG", default_value = "oida.toml", global = true)]
    config: PathBuf,

    /// Path to the LanceDB database directory (overrides config).
    #[arg(long, env = "OIDA_LANCE", global = true)]
    lance_path: Option<PathBuf>,

    /// OpenAI-compatible host URL for embeddings, e.g. a vLLM sidecar at
    /// `http://localhost:8000` (overrides config). Accepts a comma-separated list
    /// of replica addresses to balance across by least connections, e.g.
    /// `http://vllm-a:8000,http://vllm-b:8000`. Used when building the full-text
    /// index and when embedding queries for semantic search.
    #[arg(long, env = "OIDA_EMBED_HOST", global = true)]
    embed_host: Option<String>,

    /// Bearer token sent with embed requests, for a server started with an API
    /// key (overrides config).
    #[arg(long, env = "OIDA_EMBED_API_KEY", global = true)]
    embed_api_key: Option<String>,

    /// Embedding model for the full-text index (overrides config).
    #[arg(long, env = "OIDA_EMBED_MODEL", global = true)]
    embed_model: Option<String>,

    /// Verify the configured embed model name matches the index's at query time
    /// (overrides config). Pass `--embed-verify-model false` to bypass.
    #[arg(long, env = "OIDA_EMBED_VERIFY_MODEL", global = true)]
    embed_verify_model: Option<bool>,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
// `IngestArgs` carries many tuning fields, making it the largest variant. The
// command enum is parsed once at startup, so the size asymmetry is irrelevant
// and boxing would fight clap's `Subcommand` derive.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Chat with the archive via a local LLM (interactive, or one-shot with
    /// `--once`).
    ///
    /// Requires an ingested index; build one with `oida-cli ingest --force`
    /// first.
    Chat(ChatArgs),
    /// Ingest or update document/artifact metadata from Solr, and optionally
    /// build the derived full-text and raw-artifact stores.
    ///
    /// With no mode flag (or `--update`) this performs an incremental in-place
    /// update from the stored watermark; `--force` rebuilds the metadata tables
    /// from a full Solr scan; `--dry-run` previews the incremental delta without
    /// writing. Add `--full-text` and/or `--store-raw` to (re)build those stores
    /// after the metadata sync.
    Ingest(IngestArgs),
    /// Show statistics about the ingested index.
    Stats,
}

/// Arguments for the `chat` subcommand.
#[derive(Debug, clap::Args)]
struct ChatArgs {
    /// Run a single query non-interactively and exit (otherwise interactive).
    #[arg(long)]
    once: Option<String>,

    /// Ollama model to use for the chat agent (overrides config).
    #[arg(long, env = "OIDA_MODEL")]
    model: Option<String>,

    /// Ollama host URL for the chat agent (overrides config).
    #[arg(long, env = "OIDA_OLLAMA_HOST")]
    ollama_host: Option<String>,

    /// Path to the oida-mcp-server binary (defaults to a sibling of this exe).
    #[arg(long, env = "OIDA_SERVER_BIN")]
    server_bin: Option<PathBuf>,
}

/// Arguments for the `ingest` subcommand.
#[derive(Debug, clap::Args)]
struct IngestArgs {
    /// Drop and rebuild the `documents`/`artifacts` tables from a full Solr
    /// scan (drops derived chunks/_meta). Mutually exclusive with
    /// `--update`/`--dry-run`.
    #[arg(long, conflicts_with_all = ["update", "dry_run"])]
    force: bool,

    /// Incrementally sync metadata from Solr (upsert new/changed, delete
    /// redacted, invalidate stale chunks) from the stored watermark. This is the
    /// default when neither `--force` nor `--dry-run` is given.
    #[arg(long)]
    update: bool,

    /// Preview the incremental delta without writing anything (read-only).
    /// Mutually exclusive with `--force`.
    #[arg(long, conflicts_with = "force")]
    dry_run: bool,

    /// Also (re)build the hybrid keyword + semantic full-text index over the OCR
    /// artifacts. With `--force` it is rebuilt from scratch; otherwise only
    /// new/changed documents are (re-)embedded.
    #[arg(long)]
    full_text: bool,

    /// Also store raw (non-text) artifact bytes in the `raw_artifacts` blob
    /// table, fetched from the configured artifact source. With `--force` it is
    /// rebuilt; otherwise only new/changed documents are fetched.
    #[arg(long)]
    store_raw: bool,

    /// Inclusive lower-bound modified-date (ISO-8601, e.g.
    /// 2026-01-01T00:00:00Z). With `--force` omit for a full scan; otherwise
    /// omit to use the stored watermark.
    #[arg(long)]
    since: Option<String>,

    /// Fetch and print one full Solr document (all fields), then exit — to
    /// inspect the source schema. Ignores all other flags.
    #[arg(long)]
    sample_doc: bool,

    #[command(flatten)]
    solr: SolrArgs,

    #[command(flatten)]
    source: ArtifactSourceArgs,

    #[command(flatten)]
    tuning: TuningArgs,
}

/// Solr source options (used by `ingest`).
#[derive(Debug, clap::Args)]
#[command(next_help_heading = "Solr source")]
struct SolrArgs {
    /// Solr core base URL (overrides config), e.g.
    /// https://metadata.idl.ucsf.edu/solr/ltdl3.
    #[arg(long, env = "OIDA_SOLR_URL")]
    solr_url: Option<String>,

    /// Solr `q` selecting the corpus (overrides config).
    #[arg(long, env = "OIDA_SOLR_QUERY")]
    solr_query: Option<String>,
}

/// Artifact-byte source options (used by `ingest` for `--full-text`/`--store-raw`).
#[derive(Debug, clap::Args)]
#[command(next_help_heading = "Artifact source (local or S3)")]
struct ArtifactSourceArgs {
    /// Directory containing artifact files on disk (overrides config).
    #[arg(long, env = "OIDA_ARTIFACT_ROOT")]
    artifact_root: Option<PathBuf>,

    /// S3 bucket holding the artifact files; when set, reads fetch artifacts
    /// from S3 instead of `artifact_root` (overrides config).
    #[arg(long, env = "OIDA_S3_BUCKET")]
    s3_bucket: Option<String>,

    /// AWS region for the S3 bucket (overrides config).
    #[arg(long, env = "OIDA_S3_REGION")]
    s3_region: Option<String>,

    /// Custom S3 endpoint URL for S3-compatible stores, e.g. MinIO/Ceph/R2
    /// (overrides config). Credentials come from the standard AWS environment.
    #[arg(long, env = "OIDA_S3_ENDPOINT")]
    s3_endpoint: Option<String>,

    /// Key prefix prepended to the fan-out artifact path within the bucket
    /// (overrides config).
    #[arg(long, env = "OIDA_S3_PREFIX")]
    s3_prefix: Option<String>,
}

/// Build-tuning options for the metadata ingest and full-text build (used by
/// `ingest`).
#[derive(Debug, clap::Args)]
#[command(next_help_heading = "Build tuning")]
struct TuningArgs {
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
}


/// Overlay the global flags / env vars (shared by every subcommand) onto the
/// config loaded from file. Each field is only touched when the caller supplied
/// it, so the precedence is defaults < config file < env var < CLI flag (clap
/// resolves flag-over-env).
fn apply_global_overrides(config: &mut Config, g: &GlobalArgs) {
    if let Some(p) = &g.lance_path {
        config.lance_path = p.clone();
    }
    if let Some(h) = &g.embed_host {
        config.embed_host = h.clone();
    }
    if let Some(k) = &g.embed_api_key {
        config.embed_api_key = Some(k.clone());
    }
    if let Some(m) = &g.embed_model {
        config.embed_model = m.clone();
    }
    if let Some(v) = g.embed_verify_model {
        config.embed_verify_model = v;
    }
}

/// Overlay the `chat`-specific flags onto the config.
fn apply_chat_overrides(config: &mut Config, c: &ChatArgs) {
    if let Some(m) = &c.model {
        config.ollama_model = m.clone();
    }
    if let Some(h) = &c.ollama_host {
        config.ollama_host = h.clone();
    }
}

/// Overlay the `ingest`-specific flags (Solr source, artifact source, build
/// tuning, raw-store toggle) onto the config.
fn apply_ingest_overrides(config: &mut Config, a: &IngestArgs) {
    if let Some(u) = &a.solr.solr_url {
        config.solr_url = Some(u.clone());
    }
    if let Some(q) = &a.solr.solr_query {
        config.solr_query = q.clone();
    }
    if let Some(r) = &a.source.artifact_root {
        config.artifact_root = Some(r.clone());
    }
    if let Some(b) = &a.source.s3_bucket {
        config.s3_bucket = Some(b.clone());
    }
    if let Some(r) = &a.source.s3_region {
        config.s3_region = Some(r.clone());
    }
    if let Some(e) = &a.source.s3_endpoint {
        config.s3_endpoint = Some(e.clone());
    }
    if let Some(p) = &a.source.s3_prefix {
        config.s3_prefix = Some(p.clone());
    }
    if let Some(v) = a.tuning.chunk_bytes {
        config.chunk_bytes = v;
    }
    if let Some(v) = a.tuning.chunk_overlap {
        config.chunk_overlap = v;
    }
    if let Some(v) = a.tuning.write_buffer_bytes {
        config.write_buffer_bytes = v;
    }
    if let Some(v) = a.tuning.compact_on_build {
        config.compact_on_build = v;
    }
    if let Some(v) = a.tuning.ingest_buffer_bytes {
        config.ingest_buffer_bytes = v;
    }
    if let Some(v) = a.tuning.embed_concurrency {
        config.embed_concurrency = v;
    }
    if let Some(v) = a.tuning.read_concurrency {
        config.read_concurrency = v;
    }
    if let Some(v) = a.tuning.embed_batch {
        config.embed_batch = v;
    }
    if let Some(v) = a.tuning.embed_lookahead {
        config.embed_lookahead = v;
    }
    if a.store_raw {
        config.store_raw_artifacts = true;
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

    let mut config = Config::load(&args.global.config)?;
    apply_global_overrides(&mut config, &args.global);

    match args.command {
        Command::Ingest(a) => {
            apply_ingest_overrides(&mut config, &a);
            run_ingest(&config, &a).await
        }
        Command::Stats => run_stats(&config).await,
        Command::Chat(c) => {
            apply_chat_overrides(&mut config, &c);
            run_chat(&config, &c).await
        }
    }
}

/// Run the `chat` subcommand: connect to the MCP server and drive the agent,
/// either interactively or for a single `--once` query.
async fn run_chat(config: &Config, args: &ChatArgs) -> anyhow::Result<()> {
    // Chatting requires an ingested index; never build it implicitly.
    if !Index::is_ingested(config).await {
        eprintln!(
            "No index found at {}.\nRun `oida-cli ingest --force` (add --full-text for semantic \
             search) before chatting.",
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

/// Run the `ingest` subcommand. Dispatches between the read-only sample/dry-run
/// inspectors, a forced full rebuild, and the default incremental update,
/// reporting progress to stderr.
async fn run_ingest(config: &Config, a: &IngestArgs) -> anyhow::Result<()> {
    if a.sample_doc {
        return run_sample_doc(config, a.since.as_deref()).await;
    }
    if a.dry_run {
        return run_dry_run(config, a.since.as_deref()).await;
    }
    if a.force {
        return run_force_ingest(config, a).await;
    }
    // A plain (incremental) ingest against an empty location has nothing to
    // update in place, so bootstrap it with a full build automatically — the
    // first `ingest` should "just work" without requiring `--force`.
    if !Index::is_ingested(config).await {
        eprintln!("No index found at {}; building it fresh…", config.lance_path.display());
        return run_force_ingest(config, a).await;
    }
    run_incremental(config, a).await
}

/// `ingest --force`: drop and rebuild the metadata tables from a full Solr
/// scan, then (re)build the requested derived stores from scratch.
async fn run_force_ingest(config: &Config, a: &IngestArgs) -> anyhow::Result<()> {
    eprintln!(
        "Rebuilding documents/artifacts from Solr {} (q={:?}){}…",
        config.solr_url.as_deref().unwrap_or("<unset>"),
        config.solr_query,
        a.since
            .as_deref()
            .map(|s| format!(" since {s}"))
            .unwrap_or_default()
    );
    let stats = ingest::ingest_from_solr(config, a.since.as_deref(), true)
        .await
        .context("ingesting metadata")?;
    eprintln!(
        "Ingested {} documents and {} artifacts.",
        stats.documents, stats.artifacts
    );

    if a.store_raw || a.full_text {
        let index = Index::open(config).await.context("opening index")?;
        run_derived_builds(config, &index, a.store_raw, a.full_text, true, false).await?;
    }
    Ok(())
}

/// `ingest` (default) / `ingest --update`: incrementally sync metadata from
/// Solr in place, then resume the requested derived stores so new/changed
/// documents are (re-)processed in the same command.
async fn run_incremental(config: &Config, a: &IngestArgs) -> anyhow::Result<()> {
    let index = Index::open(config)
        .await
        .context("opening index (run `oida-cli ingest --force` first to build it)")?;
    eprintln!(
        "Applying incremental update from Solr {} (q={:?}){}…",
        config.solr_url.as_deref().unwrap_or("<unset>"),
        config.solr_query,
        a.since
            .as_deref()
            .map(|s| format!(" since {s}"))
            .unwrap_or_else(|| " since stored watermark".to_string())
    );
    let stats = update::apply(config, &index, a.since.as_deref())
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

    if a.store_raw || a.full_text {
        run_derived_builds(config, &index, a.store_raw, a.full_text, false, true).await?;
    } else if stats.changed > 0 || stats.redacted > 0 {
        eprintln!(
            "\nNote: stale chunks (and raw artifacts) for changed/redacted documents were \
             removed; re-run with --full-text (add --store-raw to also refresh raw bytes) to \
             re-process the affected documents."
        );
    }
    Ok(())
}

/// `ingest --sample-doc`: fetch and print one full Solr document for schema
/// inspection, then exit without writing.
async fn run_sample_doc(config: &Config, since: Option<&str>) -> anyhow::Result<()> {
    let doc = update::sample_doc(config, since)
        .await
        .context("fetching sample Solr document")?;
    match doc {
        Some(d) => println!("{}", serde_json::to_string_pretty(&d)?),
        None => println!("No documents matched."),
    }
    Ok(())
}

/// `ingest --dry-run`: classify the Solr delta against the live index and print
/// it without writing anything.
async fn run_dry_run(config: &Config, since: Option<&str>) -> anyhow::Result<()> {
    let index = Index::open(config)
        .await
        .context("opening index (run `oida-cli ingest --force` first to build it)")?;
    eprintln!(
        "Querying Solr {} (q={:?}){}…",
        config.solr_url.as_deref().unwrap_or("<unset>"),
        config.solr_query,
        since.map(|s| format!(" since {s}")).unwrap_or_default()
    );

    let plan = update::dry_run(config, &index, since).await?;

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
         indexed artifacts (content changes / redactions). No writes performed. Run \
         `oida-cli ingest` (or `--update`) to apply the delta in place, or \
         `oida-cli ingest --force` for a full Solr re-ingest."
    );
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

