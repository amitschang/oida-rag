//! Generic Ollama + MCP-tools chat agent (`feature = "chat"`).
//!
//! Engine-independent: it spawns an MCP server subprocess, advertises whatever
//! tools that server exposes, and drives an Ollama model that calls them. A
//! corpus's chat CLI supplies only branding and endpoints — the system prompt,
//! a label, the server binary, and the Ollama host/model — via [`ChatOptions`]
//! and calls [`run`].

use std::path::PathBuf;

pub mod agent;
pub mod mcp_client;
pub mod ollama;
pub mod repl;

pub use agent::Agent;
pub use mcp_client::McpClient;

/// Everything a one-call chat session needs. Nothing here is corpus-specific;
/// the domain enters only through `system_prompt` and `label`.
pub struct ChatOptions {
    /// Path to the MCP server binary to spawn.
    pub server_bin: PathBuf,
    /// Base URL of the Ollama server driving the conversation.
    pub ollama_host: String,
    /// Ollama model name.
    pub model: String,
    /// System prompt establishing the assistant's role and tool workflow.
    pub system_prompt: String,
    /// Short label for the REPL banner/prompt (e.g. `OIDA` → `oida> `).
    pub label: String,
    /// When set, run this single query non-interactively and exit; otherwise
    /// start the interactive REPL.
    pub once: Option<String>,
}

/// Connect to the MCP server, advertise its tools, and drive the agent —
/// interactively or for a single `--once` query.
pub async fn run(opts: ChatOptions) -> anyhow::Result<()> {
    eprintln!(
        "Starting {} MCP server: {}…",
        opts.label,
        opts.server_bin.display()
    );
    let mcp = McpClient::connect(opts.server_bin).await?;
    let tools = mcp.list_tools().await?;
    repl::print_tools(&tools, &opts.label);

    let agent = Agent::new(&opts.ollama_host, opts.model.clone(), opts.system_prompt, mcp, &tools)?;
    eprintln!("Using model: {}\n", opts.model);

    let result = if let Some(query) = opts.once {
        repl::run_once(&agent, query).await
    } else {
        repl::run_repl(&agent, &opts.label).await
    };

    agent.shutdown().await;
    result
}
