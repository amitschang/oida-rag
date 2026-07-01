//! The agent loop: drives an Ollama model that calls MCP tools until it
//! produces a final answer.

use std::collections::HashSet;

use anyhow::Context;
use rmcp::model::Tool;
use serde_json::{Map, Value};

use super::mcp_client::McpClient;
use super::ollama::{ChatMessage, Ollama, ToolFunctionInfo, ToolInfo, ToolType};

/// Maximum tool-calling rounds per user turn (loop guard).
const MAX_ITERATIONS: usize = 8;
/// Tool results longer than this are truncated before being sent back.
const MAX_TOOL_RESULT_CHARS: usize = 6000;

/// Orchestrates conversation turns against Ollama with MCP-backed tools.
///
/// The loop is corpus-agnostic; the domain wording enters only through the
/// caller-supplied `system_prompt`.
pub struct Agent {
    ollama: Ollama,
    model: String,
    system_prompt: String,
    tools: Vec<ToolInfo>,
    tool_names: HashSet<String>,
    mcp: McpClient,
}

impl Agent {
    /// Build an agent from a configured Ollama endpoint and MCP tool list. The
    /// `system_prompt` establishes the assistant's role and is supplied by the
    /// app (config), not baked into the loop.
    pub fn new(
        ollama_host: &str,
        model: String,
        system_prompt: String,
        mcp: McpClient,
        mcp_tools: &[Tool],
    ) -> anyhow::Result<Self> {
        let ollama = Ollama::try_new(ollama_host)
            .with_context(|| format!("invalid ollama host {ollama_host}"))?;
        let tools: Vec<ToolInfo> = mcp_tools.iter().map(to_tool_info).collect();
        let tool_names = tools.iter().map(|t| t.function.name.clone()).collect();
        Ok(Self {
            ollama,
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
                .ollama
                .chat(&self.model, history, &self.tools)
                .await
                .context("ollama chat request failed")?;
            history.push(message.clone());

            let calls = collect_tool_calls(&message, &self.tool_names);
            if calls.is_empty() {
                return Ok(message.content);
            }

            for (name, args) in &calls {
                let result = self
                    .dispatch(name, as_object(args.clone()), &mut seen_calls)
                    .await;
                eprintln!("  [tool] {name} -> {} chars", result.len());
                history.push(ChatMessage::tool(result));
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
/// Prefers Ollama's native `tool_calls`. Some models (e.g. qwen2.5-coder)
/// instead emit the call as JSON text in `content`; we parse that as a
/// fallback so tool use works regardless of the model's template.
fn collect_tool_calls(message: &ChatMessage, known: &HashSet<String>) -> Vec<(String, Value)> {
    let native: Vec<(String, Value)> = message
        .tool_calls
        .iter()
        .map(|c| (c.function.name.clone(), c.function.arguments.clone()))
        .collect();
    if !native.is_empty() {
        return native;
    }
    parse_tool_calls_from_text(&message.content, known)
}

/// Best-effort extraction of `{ "name", "arguments" }` tool calls from text.
fn parse_tool_calls_from_text(content: &str, known: &HashSet<String>) -> Vec<(String, Value)> {
    let Some(value) = extract_json(content) else {
        return Vec::new();
    };
    let mut calls = Vec::new();
    collect_from_value(&value, known, &mut calls);
    calls
}

/// Recursively pull tool-call objects out of a JSON value.
fn collect_from_value(value: &Value, known: &HashSet<String>, out: &mut Vec<(String, Value)>) {
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
                out.push((name.clone(), args));
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

/// Convert an MCP tool definition into an Ollama tool definition.
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
        assert_eq!(calls[0].0, "search_documents");
        assert_eq!(calls[0].1["query"], "opioid");
    }

    #[test]
    fn parses_fenced_and_prose_wrapped_call() {
        let content = "Sure, let me look that up.\n```json\n{\"name\": \"get_document\", \
            \"arguments\": {\"id\": \"thdb0402\"}}\n```";
        let calls = parse_tool_calls_from_text(content, &known());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "get_document");
    }

    #[test]
    fn accepts_parameters_alias_and_function_wrapper() {
        let calls = parse_tool_calls_from_text(
            r#"{"function": {"name": "search_documents", "parameters": {"query": "x"}}}"#,
            &known(),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1["query"], "x");
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
