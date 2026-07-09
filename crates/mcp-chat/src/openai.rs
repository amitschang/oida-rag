//! Minimal OpenAI-compatible chat client built directly on `reqwest`.
//!
//! This intentionally covers only the slice of the OpenAI Chat Completions API
//! the agent loop needs: a single non-streaming `POST /v1/chat/completions` call
//! with tool definitions. Both Ollama (via its `/v1` compatibility layer) and
//! vLLM speak this protocol, so a single client serves either backend — the
//! endpoint URL (and, for a locked-down vLLM, an API key) is the only thing that
//! changes. The request and response types map 1:1 onto the documented wire
//! format, so the whole LLM round-trip is visible here rather than hidden behind
//! a third-party wrapper.
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
//! { "choices": [ { "message": {
//!     "role": "assistant", "content": "...",
//!     "tool_calls": [{"id": "call_0", "type": "function",
//!                     "function": {"name": "...", "arguments": "{...}"}}]
//! } } ] }
//! ```
//! Note the tool-call `arguments` are a JSON-*encoded string* here, unlike
//! Ollama's native `/api/chat`, which nests them as an object; the agent loop
//! parses the string back into a value.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// A reusable OpenAI-compatible chat client bound to a base URL (e.g.
/// `http://localhost:11434` for Ollama or `http://localhost:8000` for vLLM).
#[derive(Clone, Debug)]
pub struct OpenAiChat {
    http: reqwest::Client,
    base: String,
    /// Optional bearer token sent as `Authorization: Bearer <key>`. vLLM ignores
    /// it unless launched with `--api-key`; Ollama ignores it entirely.
    api_key: Option<String>,
}

impl OpenAiChat {
    /// Build a client from a base URL, validating that it parses. `api_key`,
    /// when set, is sent as a bearer token.
    pub fn try_new(base: &str, api_key: Option<String>) -> Result<Self> {
        reqwest::Url::parse(base).with_context(|| format!("invalid chat host {base}"))?;
        Ok(Self {
            http: reqwest::Client::new(),
            base: base.trim_end_matches('/').to_string(),
            api_key,
        })
    }

    /// Send one non-streaming chat request and return the assistant message.
    pub async fn chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: &[ToolInfo],
    ) -> Result<ChatMessage> {
        let url = format!("{}/v1/chat/completions", self.base);
        let body = ChatRequest {
            model,
            messages,
            tools,
            stream: false,
        };
        let mut request = self.http.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .with_context(|| format!("sending chat request to {url}"))?;
        let response = check_status(response, "/v1/chat/completions").await?;
        let parsed: ChatResponse = response
            .json()
            .await
            .context("decoding chat completions response")?;
        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .context("chat completions response had no choices")
    }
}

/// Return the response unchanged on a 2xx status, otherwise fail with an error
/// that folds in the server's response body.
///
/// Both Ollama and OpenAI-style servers report failures as a JSON payload
/// alongside the HTTP status; `reqwest::Response::error_for_status` discards that
/// body, so an error would otherwise surface as a bare status code with no hint
/// as to what the server rejected.
async fn check_status(response: reqwest::Response, endpoint: &str) -> Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    let detail = ErrorBody::message_from(&body).unwrap_or_else(|| {
        let trimmed = body.trim();
        if trimmed.is_empty() {
            "no response body".to_string()
        } else {
            trimmed.to_string()
        }
    });
    bail!("chat server returned {status} for {endpoint}: {detail}");
}

/// The error payload returned alongside a failure status. Tolerates the two
/// shapes these servers use: OpenAI/vLLM's nested `{"error": {"message": ...}}`
/// and Ollama's flat `{"error": "..."}` string.
#[derive(Debug, Deserialize)]
struct ErrorBody {
    #[serde(default)]
    error: Option<ErrorField>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ErrorField {
    Message(String),
    Object { message: String },
}

impl ErrorBody {
    /// Extract a human-readable message from a raw error body, if it parses.
    fn message_from(body: &str) -> Option<String> {
        let parsed: ErrorBody = serde_json::from_str(body).ok()?;
        let message = match parsed.error? {
            ErrorField::Message(m) => m,
            ErrorField::Object { message } => message,
        };
        (!message.is_empty()).then_some(message)
    }
}

/// Request body for `POST /v1/chat/completions`.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    /// Omitted entirely when empty: some servers reject a `"tools": []`.
    #[serde(skip_serializing_if = "slice_is_empty")]
    tools: &'a [ToolInfo],
    stream: bool,
}

/// `skip_serializing_if` predicate for the borrowed `tools` slice.
fn slice_is_empty(tools: &&[ToolInfo]) -> bool {
    tools.is_empty()
}

/// The subset of the chat response we consume: the first choice's message.
#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
}

/// One completion choice.
#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessage,
}

/// A single chat message. Roles: `system`, `user`, `assistant`, `tool`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Optional because assistant messages that only call tools carry a `null`
    /// content, and omitting it on the way back keeps the request valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Correlates a `tool` result back to the assistant's tool call. Required by
    /// the OpenAI spec on `role: "tool"` messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
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

    /// A `tool` message carrying a tool-call result back to the model, tagged
    /// with the id of the call it answers.
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_string(),
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// The message text, or empty when the model returned only tool calls.
    pub fn text(&self) -> &str {
        self.content.as_deref().unwrap_or_default()
    }
}

/// A tool call emitted by the model in an assistant message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    /// Server-assigned id echoed back on the corresponding `tool` result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default = "function_type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

/// Default `type` for a tool call when a server omits it.
fn function_type() -> String {
    "function".to_string()
}

/// The function name and arguments of a tool call. Per the OpenAI spec the
/// arguments are a JSON-encoded *string* (not a nested object); the agent loop
/// parses it back into a value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

/// A tool definition advertised to the model.
#[derive(Clone, Debug, Serialize)]
pub struct ToolInfo {
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub function: ToolFunctionInfo,
}

/// The kind of tool. Only function tools are supported.
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
    pub parameters: serde_json::Value,
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
    fn omits_empty_tools_from_request() {
        let body = ChatRequest {
            model: "m",
            messages: &[],
            tools: &[],
            stream: false,
        };
        let value = serde_json::to_value(&body).unwrap();
        assert!(value.get("tools").is_none(), "empty tools omitted");
    }

    #[test]
    fn omits_empty_content_and_tool_calls_when_serializing() {
        let value = serde_json::to_value(ChatMessage::user("hi")).unwrap();
        assert_eq!(value["role"], "user");
        assert_eq!(value["content"], "hi");
        assert!(value.get("tool_calls").is_none());
        assert!(value.get("tool_call_id").is_none());
    }

    #[test]
    fn tool_message_carries_call_id() {
        let value = serde_json::to_value(ChatMessage::tool("result", "call_0")).unwrap();
        assert_eq!(value["role"], "tool");
        assert_eq!(value["content"], "result");
        assert_eq!(value["tool_call_id"], "call_0");
    }

    #[test]
    fn parses_response_with_tool_calls() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {"id": "call_0", "type": "function",
                         "function": {"name": "get_document", "arguments": "{\"id\":\"thdb0402\"}"}}
                    ]
                }
            }]
        });
        let parsed: ChatResponse = serde_json::from_value(body).unwrap();
        let message = parsed.choices.into_iter().next().unwrap().message;
        assert_eq!(message.text(), "");
        assert_eq!(message.tool_calls.len(), 1);
        let call = &message.tool_calls[0];
        assert_eq!(call.id.as_deref(), Some("call_0"));
        assert_eq!(call.function.name, "get_document");
        // Arguments arrive as a JSON-encoded string, not a nested object.
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        assert_eq!(args["id"], "thdb0402");
    }

    #[test]
    fn round_trips_assistant_tool_call_message() {
        // The loop pushes the raw assistant message back into history; its
        // tool_calls must re-serialize with id, type, and string arguments so
        // the following tool result correlates.
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_7", "type": "function",
                         "function": {"name": "x", "arguments": "{}"}}
                    ]
                }
            }]
        });
        let parsed: ChatResponse = serde_json::from_value(body).unwrap();
        let message = parsed.choices.into_iter().next().unwrap().message;
        let value = serde_json::to_value(&message).unwrap();
        assert_eq!(value["tool_calls"][0]["id"], "call_7");
        assert_eq!(value["tool_calls"][0]["type"], "function");
        assert_eq!(value["tool_calls"][0]["function"]["arguments"], "{}");
    }

    #[test]
    fn error_message_from_nested_openai_shape() {
        let body = r#"{"error": {"message": "context length exceeded", "type": "x"}}"#;
        assert_eq!(
            ErrorBody::message_from(body).as_deref(),
            Some("context length exceeded")
        );
    }

    #[test]
    fn error_message_from_flat_ollama_shape() {
        let body = r#"{"error": "model not found"}"#;
        assert_eq!(ErrorBody::message_from(body).as_deref(), Some("model not found"));
    }
}
