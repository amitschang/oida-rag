//! The agent loop: drives an OpenAI-compatible model that calls MCP tools until
//! it produces a final answer.
//!
//! Tools may come from more than one MCP server. When a single server is
//! connected, tools are advertised to the model under their bare names (so a
//! system prompt can reference them directly). When two or more servers are
//! connected, each tool is advertised as `{namespace}__{tool}` to keep names
//! from colliding across servers; the [`Agent`] routes an advertised name back
//! to the owning client and its bare name at call time.

use std::collections::HashMap;
use std::collections::HashSet;

use rmcp::model::Tool;
use serde_json::{Map, Value};

use super::mcp_client::McpClient;
use super::openai::{ChatMessage, OpenAiChat, ToolFunctionInfo, ToolInfo, ToolType};

/// Maximum tool-calling rounds per user turn (loop guard).
const MAX_ITERATIONS: usize = 8;
/// Tool results longer than this are truncated before being sent back.
const MAX_TOOL_RESULT_CHARS: usize = 6000;
/// OpenAI caps function names at 64 chars; a namespaced name over this is likely
/// to be rejected by the server, so we warn rather than silently truncate (which
/// would break routing).
const MAX_TOOL_NAME_LEN: usize = 64;

/// One connected MCP server and the tools it advertises.
pub struct ServerTools {
    /// Namespace slug used to disambiguate this server's tool names when more
    /// than one server is connected.
    pub namespace: String,
    /// The connected client.
    pub client: McpClient,
    /// The tools this server advertises.
    pub tools: Vec<Tool>,
}

/// Where an advertised tool name routes: which client backs it and the bare name
/// to call on that client (the advertised name may be namespaced; the server
/// only knows its own bare name).
struct ToolEntry {
    client_idx: usize,
    real_name: String,
}

/// One tool call the model asked for: the server-assigned id (absent when the
/// call was recovered from message text rather than the native `tool_calls`
/// field), the tool name, and its parsed arguments.
struct ToolCallReq {
    id: Option<String>,
    name: String,
    args: Value,
}

/// Orchestrates conversation turns against an OpenAI-compatible chat server with
/// MCP-backed tools drawn from one or more servers.
///
/// The loop is domain-agnostic; the domain wording enters only through the
/// caller-supplied `system_prompt` and the tools the connected servers expose.
pub struct Agent {
    chat: OpenAiChat,
    model: String,
    system_prompt: String,
    /// Tool definitions advertised to the model, under their advertised (bare or
    /// namespaced) names.
    tools: Vec<ToolInfo>,
    /// Advertised name → owning client + bare name.
    route: HashMap<String, ToolEntry>,
    /// The advertised names, used to recognise tool calls emitted as message text.
    tool_names: HashSet<String>,
    /// Connected clients, indexed by [`ToolEntry::client_idx`].
    clients: Vec<McpClient>,
}

impl Agent {
    /// Build an agent from a configured chat endpoint and one or more connected
    /// MCP servers. The `system_prompt` establishes the assistant's role and is
    /// supplied by the app (config), not baked into the loop. `api_key`, when
    /// set, is sent as a bearer token (needed only for a locked-down chat host).
    ///
    /// With a single server, tools keep their bare names; with several, each is
    /// advertised as `{namespace}__{tool}`.
    pub fn new(
        chat_host: &str,
        api_key: Option<String>,
        model: String,
        system_prompt: String,
        servers: Vec<ServerTools>,
    ) -> anyhow::Result<Self> {
        let chat = OpenAiChat::try_new(chat_host, api_key)?;
        let namespaced = servers.len() > 1;

        let mut tools = Vec::new();
        let mut route: HashMap<String, ToolEntry> = HashMap::new();
        let mut clients = Vec::with_capacity(servers.len());

        for server in servers {
            let client_idx = clients.len();
            for tool in &server.tools {
                let real_name = tool.name.to_string();
                let advertised = if namespaced {
                    format!("{}__{real_name}", server.namespace)
                } else {
                    real_name.clone()
                };
                if advertised.len() > MAX_TOOL_NAME_LEN {
                    tracing::warn!(
                        tool = %advertised,
                        "advertised tool name exceeds {MAX_TOOL_NAME_LEN} chars; the model may reject it — use a shorter server namespace"
                    );
                }
                if route.contains_key(&advertised) {
                    tracing::warn!(
                        tool = %advertised,
                        "duplicate advertised tool name across servers; the later one wins"
                    );
                }
                tools.push(to_tool_info(tool, &advertised));
                route.insert(advertised, ToolEntry { client_idx, real_name });
            }
            clients.push(server.client);
        }

        let tool_names = route.keys().cloned().collect();
        Ok(Self {
            chat,
            model,
            system_prompt,
            tools,
            route,
            tool_names,
            clients,
        })
    }

    /// A fresh conversation history seeded with the system prompt.
    pub fn new_history(&self) -> Vec<ChatMessage> {
        vec![ChatMessage::system(self.system_prompt.clone())]
    }

    /// Run one user turn to completion, returning the model's final answer.
    pub async fn ask(
        &self,
        history: &mut Vec<ChatMessage>,
        user_input: String,
    ) -> anyhow::Result<String> {
        history.push(ChatMessage::user(user_input));

        // Guard against the model looping on identical tool calls.
        let mut seen_calls: HashSet<String> = HashSet::new();

        for _ in 0..MAX_ITERATIONS {
            let message = self.chat.chat(&self.model, history, &self.tools).await?;
            history.push(message.clone());

            let calls = collect_tool_calls(&message, &self.tool_names);
            if calls.is_empty() {
                return Ok(message.text().to_string());
            }

            for call in &calls {
                let result = self
                    .dispatch(&call.name, as_object(call.args.clone()), &mut seen_calls)
                    .await;
                eprintln!("  [tool] {} -> {} chars", call.name, result.len());
                // Native calls carry an id, so the result goes back as a `tool`
                // message correlated to it. Calls recovered from message text
                // have no id (the assistant message had no native `tool_calls`
                // for a `tool` reply to reference), so feed the result back as a
                // plain user message the model can read.
                let reply = match &call.id {
                    Some(id) => ChatMessage::tool(result, id.clone()),
                    None => ChatMessage::user(format!("Result of {}:\n{result}", call.name)),
                };
                history.push(reply);
            }
        }

        Ok("(Stopped after reaching the maximum number of tool calls. \
            Here is what I gathered so far — ask me to continue if needed.)"
            .to_string())
    }

    /// Validate, deduplicate, dispatch a single tool call, and bound its output.
    async fn dispatch(
        &self,
        name: &str,
        args: Map<String, Value>,
        seen_calls: &mut HashSet<String>,
    ) -> String {
        let Some(entry) = self.route.get(name) else {
            let mut names: Vec<_> = self.route.keys().cloned().collect();
            names.sort();
            return format!(
                "ERROR: unknown tool '{name}'. Available tools: {}",
                names.join(", ")
            );
        };

        let signature = format!("{name}:{}", Value::Object(args.clone()));
        if !seen_calls.insert(signature) {
            return format!(
                "NOTE: tool '{name}' was already called with these exact arguments this turn. \
                 Use the previous result instead of repeating the call."
            );
        }

        let client = &self.clients[entry.client_idx];
        // The payload actually sent to the server: the bare tool name and its
        // JSON arguments. Enable with `RUST_LOG=mcp_chat=debug`.
        let payload = Value::Object(args.clone());
        tracing::debug!(
            tool = %name,
            real_name = %entry.real_name,
            %payload,
            "calling MCP tool"
        );
        match client.call_tool(&entry.real_name, args).await {
            Ok(text) => {
                tracing::debug!(tool = %name, result_chars = text.len(), "tool returned");
                truncate(text)
            }
            // Surface the error to the model rather than aborting the turn.
            Err(e) => {
                tracing::debug!(tool = %name, error = %e, "tool call failed");
                format!("ERROR calling '{name}': {e}")
            }
        }
    }

    /// Release all MCP server connections.
    pub async fn shutdown(self) {
        for client in self.clients {
            client.shutdown().await;
        }
    }
}

/// Coerce tool-call arguments into a JSON object.
fn as_object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        Value::Null => Map::new(),
        other => {
            let mut m = Map::new();
            m.insert("value".to_string(), other);
            m
        }
    }
}

/// Gather tool calls from a model message.
///
/// Prefers the native `tool_calls` field, parsing each call's JSON-encoded
/// `arguments` string into a value. Some models (e.g. qwen2.5-coder) instead
/// emit the call as JSON text in `content` when the server has no tool-call
/// parser configured; we parse that as a fallback so tool use works regardless
/// of the model's template (this is model behaviour, not server-specific).
fn collect_tool_calls(message: &ChatMessage, known: &HashSet<String>) -> Vec<ToolCallReq> {
    let native: Vec<ToolCallReq> = message
        .tool_calls
        .iter()
        .map(|c| ToolCallReq {
            id: c.id.clone(),
            name: c.function.name.clone(),
            // Arguments are a JSON-encoded string; parse it, tolerating an empty
            // or malformed string by falling back to null (→ empty arg object).
            args: serde_json::from_str(&c.function.arguments).unwrap_or(Value::Null),
        })
        .collect();
    if !native.is_empty() {
        return native;
    }
    parse_tool_calls_from_text(message.text(), known)
}

/// Best-effort extraction of `{ "name", "arguments" }` tool calls from text.
/// These carry no id (there is no native call to correlate against).
fn parse_tool_calls_from_text(content: &str, known: &HashSet<String>) -> Vec<ToolCallReq> {
    let Some(value) = extract_json(content) else {
        return Vec::new();
    };
    let mut calls = Vec::new();
    collect_from_value(&value, known, &mut calls);
    calls
}

/// Recursively pull tool-call objects out of a JSON value.
fn collect_from_value(value: &Value, known: &HashSet<String>, out: &mut Vec<ToolCallReq>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_from_value(item, known, out);
            }
        }
        Value::Object(map) => {
            // Unwrap common wrappers: {"function": {...}} and {"tool_calls": [...]}.
            if let Some(func) = map.get("function") {
                collect_from_value(func, known, out);
                return;
            }
            if let Some(tcs) = map.get("tool_calls") {
                collect_from_value(tcs, known, out);
                return;
            }
            if let Some(Value::String(name)) = map.get("name")
                && known.contains(name)
            {
                let args = map
                    .get("arguments")
                    .or_else(|| map.get("parameters"))
                    .cloned()
                    .unwrap_or(Value::Object(Map::new()));
                out.push(ToolCallReq {
                    id: None,
                    name: name.clone(),
                    args,
                });
            }
        }
        _ => {}
    }
}

/// Extract a JSON value from text that may be fenced or surrounded by prose.
fn extract_json(content: &str) -> Option<Value> {
    let trimmed = content.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }
    // Strip ```json ... ``` / ``` ... ``` fences.
    let unfenced = trimmed
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if let Ok(v) = serde_json::from_str::<Value>(unfenced) {
        return Some(v);
    }
    // Fall back to the first balanced `{...}` or `[...]` span.
    let start = unfenced.find(['{', '['])?;
    let end = unfenced.rfind(['}', ']'])?;
    if end > start {
        serde_json::from_str::<Value>(&unfenced[start..=end]).ok()
    } else {
        None
    }
}

/// Truncate an oversized tool result, noting that it was cut.
fn truncate(mut text: String) -> String {
    if text.chars().count() > MAX_TOOL_RESULT_CHARS {
        let cut: String = text.chars().take(MAX_TOOL_RESULT_CHARS).collect();
        text = format!("{cut}\n…[truncated; refine your query or use paging to see more]");
    }
    text
}

/// Convert an MCP tool definition into an OpenAI tool definition, advertising it
/// under `advertised_name` (bare or namespaced).
fn to_tool_info(tool: &Tool, advertised_name: &str) -> ToolInfo {
    ToolInfo {
        tool_type: ToolType::Function,
        function: ToolFunctionInfo {
            name: advertised_name.to_string(),
            description: tool
                .description
                .as_ref()
                .map(|d| d.to_string())
                .unwrap_or_default(),
            parameters: Value::Object((*tool.input_schema).clone()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known() -> HashSet<String> {
        ["search_documents", "get_document"]
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn parses_plain_json_tool_call() {
        let calls = parse_tool_calls_from_text(
            r#"{"name": "search_documents", "arguments": {"query": "opioid"}}"#,
            &known(),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search_documents");
        assert_eq!(calls[0].args["query"], "opioid");
        assert!(calls[0].id.is_none(), "text-parsed calls carry no id");
    }

    #[test]
    fn parses_fenced_and_prose_wrapped_call() {
        let content = "Sure, let me look that up.\n```json\n{\"name\": \"get_document\", \
            \"arguments\": {\"id\": \"thdb0402\"}}\n```";
        let calls = parse_tool_calls_from_text(content, &known());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_document");
    }

    #[test]
    fn accepts_parameters_alias_and_function_wrapper() {
        let calls = parse_tool_calls_from_text(
            r#"{"function": {"name": "search_documents", "parameters": {"query": "x"}}}"#,
            &known(),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].args["query"], "x");
    }

    #[test]
    fn parses_array_of_calls() {
        let calls = parse_tool_calls_from_text(
            r#"[{"name":"search_documents","arguments":{"query":"a"}},
                {"name":"get_document","arguments":{"id":"b"}}]"#,
            &known(),
        );
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn ignores_unknown_tools_and_plain_text() {
        assert!(parse_tool_calls_from_text("just a normal answer", &known()).is_empty());
        assert!(
            parse_tool_calls_from_text(r#"{"name":"not_a_tool","arguments":{}}"#, &known())
                .is_empty()
        );
    }
}
