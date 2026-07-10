//! `mcp-chat`: a config-driven chat + MCP REPL.
//!
//! Drives an OpenAI-compatible model that calls tools from one or more MCP
//! servers (child process over stdio, or an HTTP endpoint). Everything domain-
//! specific — the system prompt, the servers, the endpoints — comes from an
//! optional TOML config file and command-line flags, following the precedence
//! `defaults < file < env < flag`.

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use mcp_chat::config::{
    DEFAULT_CHAT_HOST, DEFAULT_LABEL, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, FileConfig, ServerEntry,
};
use mcp_chat::{ChatOptions, McpServer};

/// Chat with an OpenAI-compatible model backed by MCP tools.
#[derive(Debug, Parser)]
#[command(name = "mcp-chat", version, about)]
struct Args {
    /// Path to a TOML config file. Flags and env vars override its values.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Base URL of the OpenAI-compatible chat server.
    #[arg(long, env = "MCP_CHAT_HOST")]
    chat_host: Option<String>,

    /// Bearer token for the chat server (only for a locked-down endpoint).
    #[arg(long, env = "MCP_CHAT_API_KEY")]
    api_key: Option<String>,

    /// Chat model name.
    #[arg(long, env = "MCP_CHAT_MODEL")]
    model: Option<String>,

    /// System prompt. A value beginning with `@` is read from that file path.
    #[arg(long)]
    system_prompt: Option<String>,

    /// REPL banner/prompt label.
    #[arg(long)]
    label: Option<String>,

    /// Connect to a stdio MCP server by spawning this binary. At most one stdio
    /// server is allowed; it may be combined with any number of `--mcp-http`
    /// servers. Any `--mcp-*` flag overrides the config file's servers.
    #[arg(long, value_name = "BIN")]
    mcp_stdio: Option<PathBuf>,

    /// Namespace override for the `--mcp-stdio` server, used to disambiguate
    /// tool names. Omit to keep its default namespace.
    #[arg(long, value_name = "NS", requires = "mcp_stdio")]
    mcp_stdio_ns: Option<String>,

    /// Connect to an HTTP MCP server at this URL. Repeatable to connect to
    /// several HTTP servers. Any `--mcp-*` flag overrides the config file's
    /// servers.
    #[arg(long, value_name = "URL")]
    mcp_http: Vec<String>,

    /// `Authorization` header value (e.g. `"Bearer …"`) for the correspondingly
    /// positioned `--mcp-http` server. Repeatable; the Nth `--mcp-auth` pairs
    /// with the Nth `--mcp-http`. Use an empty string to skip a position.
    #[arg(long, value_name = "HEADER", requires = "mcp_http")]
    mcp_auth: Vec<String>,

    /// Namespace override for the correspondingly positioned `--mcp-http`
    /// server, used to disambiguate tool names. Repeatable; the Nth `--mcp-ns`
    /// pairs with the Nth `--mcp-http`. Use an empty string to keep a server's
    /// default namespace. (For the stdio server, use `--mcp-stdio-ns`.)
    #[arg(long, value_name = "NS", requires = "mcp_http")]
    mcp_ns: Vec<String>,

    /// Run this single query non-interactively and exit.
    #[arg(long)]
    once: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    let file = match &args.config {
        Some(path) => FileConfig::load(path)?,
        None => FileConfig::default(),
    };

    let opts = resolve(args, file)?;
    mcp_chat::run(opts).await
}

/// Fold the file config, environment (already folded into `args` by clap), and
/// flags into the final [`ChatOptions`], applying `defaults < file < flag`.
fn resolve(args: Args, file: FileConfig) -> anyhow::Result<ChatOptions> {
    let servers = resolve_servers(&args, &file)?;
    let system_prompt = resolve_system_prompt(&args, &file)?;

    Ok(ChatOptions {
        servers,
        chat_host: args
            .chat_host
            .or(file.chat_host)
            .unwrap_or_else(|| DEFAULT_CHAT_HOST.to_string()),
        chat_api_key: args.api_key.or(file.chat_api_key),
        model: args
            .model
            .or(file.model)
            .unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        system_prompt,
        label: args
            .label
            .or(file.label)
            .unwrap_or_else(|| DEFAULT_LABEL.to_string()),
        once: args.once,
    })
}

/// The `--mcp-*` flags define the servers and override the file: at most one
/// `--mcp-stdio` plus any number of `--mcp-http` endpoints. When no such flag is
/// given, the file's `[[server]]` entries are used.
///
/// `--mcp-auth` and `--mcp-ns` both pair positionally with `--mcp-http`; the
/// stdio server takes its namespace from `--mcp-stdio-ns`.
fn resolve_servers(args: &Args, file: &FileConfig) -> anyhow::Result<Vec<McpServer>> {
    check_pairing("--mcp-auth", args.mcp_auth.len(), args.mcp_http.len())?;
    check_pairing("--mcp-ns", args.mcp_ns.len(), args.mcp_http.len())?;

    let mut servers = Vec::new();
    if let Some(bin) = &args.mcp_stdio {
        servers.push(McpServer::Stdio {
            bin: bin.clone(),
            args: Vec::new(),
            env: Vec::new(),
            name: non_empty(args.mcp_stdio_ns.clone()),
        });
    }
    for (i, url) in args.mcp_http.iter().enumerate() {
        servers.push(McpServer::Http {
            url: url.clone(),
            auth_header: non_empty(args.mcp_auth.get(i).cloned()),
            headers: Vec::new(),
            name: non_empty(args.mcp_ns.get(i).cloned()),
        });
    }
    if !servers.is_empty() {
        return Ok(servers);
    }
    // No server configured is allowed: the session runs as a plain chat with no
    // tools. `--mcp-stdio`/`--mcp-http`/`[[server]]` add tools when wanted.
    file.servers.iter().map(ServerEntry::to_mcp_server).collect()
}

/// Reject more positional `flag` values than there are `--mcp-http` servers to
/// pair them with.
fn check_pairing(flag: &str, given: usize, http_count: usize) -> anyhow::Result<()> {
    if given > http_count {
        anyhow::bail!(
            "got {given} {flag} value(s) but only {http_count} --mcp-http server(s); each {flag} pairs with one --mcp-http"
        );
    }
    Ok(())
}

/// Treat an absent value or an empty string alike as "not set".
fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.is_empty())
}

/// Resolve the system prompt: a `--system-prompt` flag (with `@file` sugar) wins,
/// then the file's inline `system_prompt` or `system_prompt_file`, else the
/// built-in default.
fn resolve_system_prompt(args: &Args, file: &FileConfig) -> anyhow::Result<String> {
    if let Some(prompt) = &args.system_prompt {
        return load_prompt_arg(prompt);
    }
    if let Some(prompt) = &file.system_prompt {
        return Ok(prompt.clone());
    }
    if let Some(path) = &file.system_prompt_file {
        return std::fs::read_to_string(path)
            .with_context(|| format!("reading system_prompt_file {path}"));
    }
    Ok(DEFAULT_SYSTEM_PROMPT.to_string())
}

/// A `--system-prompt` value: read from a file when it begins with `@`, else use
/// the literal string.
fn load_prompt_arg(value: &str) -> anyhow::Result<String> {
    match value.strip_prefix('@') {
        Some(path) => {
            std::fs::read_to_string(path).with_context(|| format!("reading system prompt file {path}"))
        }
        None => Ok(value.to_string()),
    }
}
