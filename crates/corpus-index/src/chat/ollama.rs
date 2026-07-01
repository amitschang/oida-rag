//! Minimal Ollama client built directly on `reqwest`.
//!
//! This intentionally covers only the slice of the Ollama HTTP API the agent
//! loop needs: a single non-streaming `POST /api/chat` call with tool
//! definitions. The request and response types map 1:1 onto the documented
//! wire format, so the whole LLM round-trip is visible here rather than hidden
//! behind a third-party wrapper.
//!
//! Wire format (request):
//! ```json
//! {
//!   "model": "qwen2.5-coder",
//!   "messages": [{"role": "user", "content": "..."}],
//!   "tools": [{"type": "function", "function": {"name": "...", "parameters": {}}}],
//!   "stream": false
//! }
//! ```
//! Wire format (response, the parts we use):
//! ```json
//! { "message": { "role": "assistant", "content": "...",
//!                "tool_calls": [{"function": {"name": "...", "arguments": {}}}] } }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A reusable Ollama HTTP client bound to a base URL (e.g. `http://localhost:11434`).
#[derive(Clone, Debug)]
pub struct Ollama {
    http: reqwest::Client,
    base: String,
}

impl Ollama {
    /// Build a client from a base URL, validating that it parses.
    pub fn try_new(base: &str) -> Result<Self> {
        reqwest::Url::parse(base).with_context(|| format!("invalid ollama host {base}"))?;
        Ok(Self {
            http: reqwest::Client::new(),
            base: base.trim_end_matches('/').to_string(),
        })
    }

    /// Send one non-streaming chat request and return the assistant message.
    pub async fn chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: &[ToolInfo],
    ) -> Result<ChatMessage> {
        let url = format!("{}/api/chat", self.base);
        let body = ChatRequest {
            model,
            messages,
            tools,
            stream: false,
        };
        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("sending chat request to {url}"))?
            .error_for_status()
            .context("ollama returned an error status")?;
        let parsed: ChatResponse = response
            .json()
            .await
            .context("decoding ollama chat response")?;
        Ok(parsed.message)
    }
}

/// Request body for `POST /api/chat`.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    tools: &'a [ToolInfo],
    stream: bool,
}

/// The subset of the chat response we consume.
#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: ChatMessage,
}

/// A single chat message. Roles: `system`, `user`, `assistant`, `tool`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    /// A `system` message (instructions/role priming).
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// A `user` message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// A `tool` message carrying a tool-call result back to the model.
    pub fn tool(content: impl Into<String>) -> Self {
        Self::new("tool", content)
    }

    fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_string(),
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }
}

/// A tool call emitted by the model in an assistant message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: FunctionCall,
}

/// The function name and arguments of a tool call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

/// A tool definition advertised to the model.
#[derive(Clone, Debug, Serialize)]
pub struct ToolInfo {
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub function: ToolFunctionInfo,
}

/// The kind of tool. Ollama currently only supports function tools.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

/// The callable schema for a function tool.
#[derive(Clone, Debug, Serialize)]
pub struct ToolFunctionInfo {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's parameters.
    pub parameters: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serializes_tool_with_function_type_tag() {
        let tool = ToolInfo {
            tool_type: ToolType::Function,
            function: ToolFunctionInfo {
                name: "search_documents".to_string(),
                description: "find documents".to_string(),
                parameters: json!({"type": "object", "properties": {}}),
            },
        };
        let value = serde_json::to_value(&tool).unwrap();
        assert_eq!(value["type"], "function");
        assert_eq!(value["function"]["name"], "search_documents");
    }

    #[test]
    fn omits_empty_tool_calls_when_serializing() {
        let value = serde_json::to_value(ChatMessage::user("hi")).unwrap();
        assert_eq!(value["role"], "user");
        assert!(value.get("tool_calls").is_none());
    }

    #[test]
    fn parses_response_with_tool_calls() {
        let body = json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_document", "arguments": {"id": "thdb0402"}}}
                ]
            }
        });
        let parsed: ChatResponse = serde_json::from_value(body).unwrap();
        assert_eq!(parsed.message.tool_calls.len(), 1);
        assert_eq!(parsed.message.tool_calls[0].function.name, "get_document");
        assert_eq!(
            parsed.message.tool_calls[0].function.arguments["id"],
            "thdb0402"
        );
    }
}
