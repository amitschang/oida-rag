//! A generic OpenAI-compatible chat agent driven over MCP tools.
//!
//! Domain-independent: it connects to one or more MCP servers (a child process
//! over stdio, or a streamable-HTTP endpoint), advertises whatever tools those
//! servers expose, and drives an OpenAI-compatible model (Ollama's `/v1` layer,
//! vLLM, or any compatible server) that calls them. A caller supplies only
//! branding and endpoints — the system prompt, a label, the MCP server(s), and
//! the chat host/model — via [`ChatOptions`] and calls [`run`].
//!
//! The crate is both a library (embed [`run`]/[`Agent`] in an app) and a binary
//! (`mcp-chat`, behind the default `bin` feature) that reads all of this from
//! command-line flags and an optional config file.

pub mod agent;
pub mod mcp_client;
pub mod openai;
pub mod repl;

#[cfg(feature = "bin")]
pub mod config;

pub use agent::{Agent, ServerTools};
pub use mcp_client::{McpClient, McpServer};

/// Everything a chat session needs. Nothing here is domain-specific; the domain
/// enters only through `system_prompt`, `label`, and the MCP server(s).
pub struct ChatOptions {
    /// MCP servers to connect to. With one server, tools are advertised under
    /// their bare names; with several, each is namespaced (see [`agent`]).
    pub servers: Vec<McpServer>,
    /// Base URL of the OpenAI-compatible chat server driving the conversation.
    pub chat_host: String,
    /// Optional bearer token for the chat server (needed only for a locked-down
    /// endpoint launched with an API key).
    pub chat_api_key: Option<String>,
    /// Chat model name.
    pub model: String,
    /// System prompt establishing the assistant's role and tool workflow.
    pub system_prompt: String,
    /// Short label for the REPL banner/prompt (e.g. `OIDA` → `oida> `).
    pub label: String,
    /// When set, run this single query non-interactively and exit; otherwise
    /// start the interactive REPL.
    pub once: Option<String>,
}

/// Connect to the configured MCP server(s), advertise their tools, and drive the
/// agent — interactively or for a single `--once` query.
pub async fn run(opts: ChatOptions) -> anyhow::Result<()> {
    // No servers is a valid configuration: a plain chat with no tools. Everything
    // below operates the same, the model just never has a tool to call.
    if opts.servers.is_empty() {
        eprintln!("No MCP servers configured; running as a plain chat (no tools).");
    }

    let mut server_tools = Vec::with_capacity(opts.servers.len());
    for server in &opts.servers {
        eprintln!("Connecting to MCP server: {}…", server.describe());
        let client = McpClient::connect(server).await?;
        let tools = client.list_tools().await?;
        repl::print_tools(&tools, &opts.label);
        server_tools.push(ServerTools {
            namespace: server.namespace(),
            client,
            tools,
        });
    }

    let agent = Agent::new(
        &opts.chat_host,
        opts.chat_api_key,
        opts.model.clone(),
        opts.system_prompt,
        server_tools,
    )?;
    eprintln!("Using model: {}\n", opts.model);

    let result = if let Some(query) = opts.once {
        repl::run_once(&agent, query).await
    } else {
        repl::run_repl(&agent, &opts.label).await
    };

    agent.shutdown().await;
    result
}
