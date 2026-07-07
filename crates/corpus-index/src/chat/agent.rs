//! The agent loop: drives an OpenAI-compatible model that calls MCP tools until
//! it produces a final answer.

use std::collections::HashSet;

use anyhow::Context;
use rmcp::model::Tool;
use serde_json::{Map, Value};

use super::mcp_client::McpClient;
use super::openai::{ChatMessage, OpenAiChat, ToolFunctionInfo, ToolInfo, ToolType};

/// Maximum tool-calling rounds per user turn (loop guard).
const MAX_ITERATIONS: usize = 8;
/// Tool results longer than this are truncated before being sent back.
const MAX_TOOL_RESULT_CHARS: usize = 6000;

/// One tool call the model asked for: the server-assigned id (absent when the
/// call was recovered from message text rather than the native `tool_calls`
/// field), the tool name, and its parsed arguments.
struct ToolCallReq {
    id: Option<String>,
    name: String,
    args: Value,
}

/// Orchestrates conversation turns against an OpenAI-compatible chat server with
/// MCP-backed tools.
///
/// The loop is corpus-agnostic; the domain wording enters only through the
/// caller-supplied `system_prompt`.
pub struct Agent {
    chat: OpenAiChat,
    model: String,
    system_prompt: String,
    tools: Vec<ToolInfo>,
    tool_names: HashSet<String>,
    mcp: McpClient,
}

impl Agent {
    /// Build an agent from a configured chat endpoint and MCP tool list. The
    /// `system_prompt` establishes the assistant's role and is supplied by the
    /// app (config), not baked into the loop. `api_key`, when set, is sent as a
    /// bearer token (needed only for a locked-down vLLM).
    pub fn new(
        chat_host: &str,
        api_key: Option<String>,
        model: String,
        system_prompt: String,
        mcp: McpClient,
        mcp_tools: &[Tool],
    ) -> anyhow::Result<Self> {
        let chat = OpenAiChat::try_new(chat_host, api_key)
            .with_context(|| format!("invalid chat host {chat_host}"))?;
        let tools: Vec<ToolInfo> = mcp_tools.iter().map(to_tool_info).collect();
        let tool_names = tools.iter().map(|t| t.function.name.clone()).collect();
        Ok(Self {
            chat,
            model,
            system_prompt,
            tools,
            tool_names,
            mcp,
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
            let message = self
                .chat
                .chat(&self.model, history, &self.tools)
                .await
                .context("chat request failed")?;
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
        if !self.tool_names.contains(name) {
            return format!("ERROR: unknown tool '{name}'. Available tools: {}", {
                let mut names: Vec<_> = self.tool_names.iter().cloned().collect();
                names.sort();
                names.join(", ")
            });
        }

        let signature = format!("{name}:{}", Value::Object(args.clone()));
        if !seen_calls.insert(signature) {
            return format!(
                "NOTE: tool '{name}' was already called with these exact arguments this turn. \
                 Use the previous result instead of repeating the call."
            );
        }

        match self.mcp.call_tool(name, args).await {
            Ok(text) => truncate(text),
            // Surface the error to the model rather than aborting the turn.
            Err(e) => format!("ERROR calling '{name}': {e}"),
        }
    }

    /// Release the MCP server connection.
    pub async fn shutdown(self) {
        self.mcp.shutdown().await;
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

/// Convert an MCP tool definition into an OpenAI tool definition.
fn to_tool_info(tool: &Tool) -> ToolInfo {
    ToolInfo {
        tool_type: ToolType::Function,
        function: ToolFunctionInfo {
            name: tool.name.to_string(),
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
