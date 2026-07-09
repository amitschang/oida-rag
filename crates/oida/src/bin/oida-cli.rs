//! OIDA assistant CLI.
//!
//! Connects to the OIDA MCP server (spawned as a child process) and drives an
//! OpenAI-compatible chat model (a local Ollama or a vLLM sidecar) that calls
//! the server's tools to answer questions about the document archive.

use std::path::PathBuf;

use anyhow::{Context, anyhow};
use clap::{Parser, Subcommand};
use corpus_index::cli;
use mcp_chat::{ChatOptions, McpServer};
use oida::{ChatConfig, CoreConfig, Index, OidaConfig, SolrConfig, apply, ingest, update};

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

    /// Chat model to use for the agent (overrides config).
    #[arg(long, env = "OIDA_CHAT_MODEL")]
    model: Option<String>,

    /// OpenAI-compatible chat host URL for the agent, e.g. a local Ollama
    /// (`http://localhost:11434`) or a vLLM sidecar (`http://localhost:8000`).
    #[arg(long, env = "OIDA_CHAT_HOST")]
    chat_host: Option<String>,

    /// Bearer token for the chat host (only for a locked-down vLLM).
    #[arg(long, env = "OIDA_CHAT_API_KEY")]
    chat_api_key: Option<String>,

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


// Each arg group overlays its *disjoint* config slice. Every field is only
// touched when the caller supplied it, so the precedence is
// defaults < config file < env var < CLI flag (clap resolves flag-over-env).
// These are inherent methods, not a trait: the binary knows every concrete type,
// so there is nothing to abstract over (the CLI analogue of merging routers).

impl GlobalArgs {
    /// Overlay the framework-global flags onto the core config slice.
    fn overlay(&self, c: &mut CoreConfig) {
        if let Some(p) = &self.lance_path {
            c.lance_path = p.clone();
        }
        if let Some(h) = &self.embed_host {
            c.embed_host = h.clone();
        }
        if let Some(k) = &self.embed_api_key {
            c.embed_api_key = Some(k.clone());
        }
        if let Some(m) = &self.embed_model {
            c.embed_model = m.clone();
        }
        if let Some(v) = self.embed_verify_model {
            c.embed_verify_model = v;
        }
    }
}

impl ChatArgs {
    /// Overlay the `chat` flags onto the chat config slice.
    fn overlay(&self, c: &mut ChatConfig) {
        if let Some(m) = &self.model {
            c.chat_model = m.clone();
        }
        if let Some(h) = &self.chat_host {
            c.chat_host = h.clone();
        }
        if let Some(k) = &self.chat_api_key {
            c.chat_api_key = Some(k.clone());
        }
    }
}

impl SolrArgs {
    /// Overlay the Solr source flags onto the Solr config slice.
    fn overlay(&self, c: &mut SolrConfig) {
        if let Some(u) = &self.solr_url {
            c.solr_url = Some(u.clone());
        }
        if let Some(q) = &self.solr_query {
            c.solr_query = q.clone();
        }
    }
}

impl ArtifactSourceArgs {
    /// Overlay the artifact-source flags onto the core config slice.
    fn overlay(&self, c: &mut CoreConfig) {
        if let Some(r) = &self.artifact_root {
            c.artifact_root = Some(r.clone());
        }
        if let Some(b) = &self.s3_bucket {
            c.s3_bucket = Some(b.clone());
        }
        if let Some(r) = &self.s3_region {
            c.s3_region = Some(r.clone());
        }
        if let Some(e) = &self.s3_endpoint {
            c.s3_endpoint = Some(e.clone());
        }
        if let Some(p) = &self.s3_prefix {
            c.s3_prefix = Some(p.clone());
        }
    }
}

impl TuningArgs {
    /// Overlay the build-tuning flags onto the core config slice.
    fn overlay(&self, c: &mut CoreConfig) {
        if let Some(v) = self.chunk_bytes {
            c.chunk_bytes = v;
        }
        if let Some(v) = self.chunk_overlap {
            c.chunk_overlap = v;
        }
        if let Some(v) = self.write_buffer_bytes {
            c.write_buffer_bytes = v;
        }
        if let Some(v) = self.compact_on_build {
            c.compact_on_build = v;
        }
        if let Some(v) = self.ingest_buffer_bytes {
            c.ingest_buffer_bytes = v;
        }
        if let Some(v) = self.embed_concurrency {
            c.embed_concurrency = v;
        }
        if let Some(v) = self.read_concurrency {
            c.read_concurrency = v;
        }
        if let Some(v) = self.embed_batch {
            c.embed_batch = v;
        }
        if let Some(v) = self.embed_lookahead {
            c.embed_lookahead = v;
        }
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
                .unwrap_or_else(|_| "warn,oida=info".into()),
        )
        .init();

    let args = Args::parse();

    let mut config = OidaConfig::load(&args.global.config)?;
    args.global.overlay(&mut config.core);

    match args.command {
        Command::Ingest(a) => {
            a.solr.overlay(&mut config.solr);
            a.source.overlay(&mut config.core);
            a.tuning.overlay(&mut config.core);
            if a.store_raw {
                config.core.store_raw_artifacts = true;
            }
            run_ingest(&config, &a).await
        }
        Command::Stats => cli::print_stats(&config.core).await,
        Command::Chat(c) => {
            c.overlay(&mut config.chat);
            run_chat(&config, &c).await
        }
    }
}

/// Run the `chat` subcommand: connect to the MCP server and drive the agent,
/// either interactively or for a single `--once` query. The agent loop, MCP
/// client, and REPL are the generic `mcp-chat` crate; the CLI supplies only the
/// branding, endpoints, and server binary.
async fn run_chat(config: &OidaConfig, args: &ChatArgs) -> anyhow::Result<()> {
    // Chatting requires an ingested index; never build it implicitly.
    if !Index::is_ingested(&config.core).await {
        eprintln!(
            "No index found at {}.\nRun `oida-cli ingest --force` (add --full-text for semantic \
             search) before chatting.",
            config.core.lance_path.display()
        );
        std::process::exit(1);
    }

    mcp_chat::run(ChatOptions {
        servers: vec![McpServer::Stdio {
            bin: resolve_server_bin(args.server_bin.clone())?,
            args: Vec::new(),
            env: Vec::new(),
        }],
        chat_host: config.chat.chat_host.clone(),
        chat_api_key: config.chat.chat_api_key.clone(),
        model: config.chat.chat_model.clone(),
        system_prompt: config.chat.system_prompt.clone(),
        label: config.chat.assistant_label.clone(),
        once: args.once.clone(),
    })
    .await
}

/// Run the `ingest` subcommand. Dispatches between the read-only sample/dry-run
/// inspectors, a forced full rebuild, and the default incremental update,
/// reporting progress to stderr.
async fn run_ingest(config: &OidaConfig, a: &IngestArgs) -> anyhow::Result<()> {
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
    if !Index::is_ingested(&config.core).await {
        eprintln!("No index found at {}; building it fresh…", config.core.lance_path.display());
        return run_force_ingest(config, a).await;
    }
    run_incremental(config, a).await
}

/// `ingest --force`: drop and rebuild the metadata tables from a full Solr
/// scan, then (re)build the requested derived stores from scratch.
async fn run_force_ingest(config: &OidaConfig, a: &IngestArgs) -> anyhow::Result<()> {
    eprintln!(
        "Rebuilding documents/artifacts from Solr {} (q={:?}){}…",
        config.solr.solr_url.as_deref().unwrap_or("<unset>"),
        config.solr.solr_query,
        a.since
            .as_deref()
            .map(|s| format!(" since {s}"))
            .unwrap_or_default()
    );
    let provider = oida::SolrProvider::from_config(&config.solr)
        .context("building Solr source provider")?;
    let stats = ingest::build_metadata(&provider, &config.core, a.since.as_deref(), true)
        .await
        .context("ingesting metadata")?;
    eprintln!(
        "Ingested {} documents and {} artifacts.",
        stats.documents, stats.artifacts
    );

    if a.store_raw || a.full_text {
        let index = Index::open(&config.core).await.context("opening index")?;
        cli::build_derived(&config.core, &index, a.store_raw, a.full_text, true, false).await?;
    }
    Ok(())
}

/// `ingest` (default) / `ingest --update`: incrementally sync metadata from
/// Solr in place, then resume the requested derived stores so new/changed
/// documents are (re-)processed in the same command.
async fn run_incremental(config: &OidaConfig, a: &IngestArgs) -> anyhow::Result<()> {
    let index = Index::open(&config.core)
        .await
        .context("opening index (run `oida-cli ingest --force` first to build it)")?;
    eprintln!(
        "Applying incremental update from Solr {} (q={:?}){}…",
        config.solr.solr_url.as_deref().unwrap_or("<unset>"),
        config.solr.solr_query,
        a.since
            .as_deref()
            .map(|s| format!(" since {s}"))
            .unwrap_or_else(|| " since stored watermark".to_string())
    );
    let provider = oida::SolrProvider::from_config(&config.solr)
        .context("building Solr source provider")?;
    let stats = apply::apply(&provider, &config.core, &index, a.since.as_deref())
        .await
        .context("applying incremental update")?;
    println!("Incremental update applied ({})", config.core.lance_path.display());
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
        cli::build_derived(&config.core, &index, a.store_raw, a.full_text, false, true).await?;
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
async fn run_sample_doc(config: &OidaConfig, since: Option<&str>) -> anyhow::Result<()> {
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
async fn run_dry_run(config: &OidaConfig, since: Option<&str>) -> anyhow::Result<()> {
    let index = Index::open(&config.core)
        .await
        .context("opening index (run `oida-cli ingest --force` first to build it)")?;
    eprintln!(
        "Querying Solr {} (q={:?}){}…",
        config.solr.solr_url.as_deref().unwrap_or("<unset>"),
        config.solr.solr_query,
        since.map(|s| format!(" since {s}")).unwrap_or_default()
    );

    let plan = update::dry_run(config, &index, since).await?;

    println!("Update dry run ({})", config.core.lance_path.display());
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


