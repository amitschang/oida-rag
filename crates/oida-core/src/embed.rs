//! Minimal Ollama embedding client.
//!
//! Covers only the slice of the Ollama HTTP API the hybrid text index needs:
//! `POST /api/embed` to turn text into vectors, and `GET /api/tags` to read a
//! model's content digest. The digest lets the index pin the exact model that
//! produced its vectors so a later query can detect a silently changed model
//! (see [`crate::hybrid`]).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// A reusable Ollama embedding client bound to a base URL.
#[derive(Clone, Debug)]
pub struct Embedder {
    http: reqwest::Client,
    base: String,
    model: String,
}

impl Embedder {
    /// Build a client for `model` against the Ollama server at `base`.
    pub fn new(base: &str, model: impl Into<String>) -> Result<Self> {
        reqwest::Url::parse(base).with_context(|| format!("invalid ollama host {base}"))?;
        Ok(Self {
            http: reqwest::Client::new(),
            base: base.trim_end_matches('/').to_string(),
            model: model.into(),
        })
    }

    /// The model name this embedder targets.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Embed a batch of texts, returning one vector per input in order.
    ///
    /// Returns an error if the server yields a different number of embeddings
    /// than inputs, or if any embedding is empty.
    pub async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/api/embed", self.base);
        let body = EmbedRequest {
            model: &self.model,
            input: inputs,
        };
        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("sending embed request to {url}"))?;
        let response = check_status(response, "/api/embed").await?;
        let parsed: EmbedResponse = response
            .json()
            .await
            .context("decoding ollama embed response")?;
        if parsed.embeddings.len() != inputs.len() {
            bail!(
                "ollama returned {} embeddings for {} inputs",
                parsed.embeddings.len(),
                inputs.len()
            );
        }
        if parsed.embeddings.iter().any(|e| e.is_empty()) {
            bail!("ollama returned an empty embedding for model {}", self.model);
        }
        Ok(parsed.embeddings)
    }

    /// Embed a single text and return its vector.
    pub async fn embed_one(&self, input: &str) -> Result<Vec<f32>> {
        let mut out = self.embed(std::slice::from_ref(&input.to_string())).await?;
        out.pop()
            .ok_or_else(|| anyhow::anyhow!("ollama returned no embedding"))
    }

    /// Fetch the model's content digest from `GET /api/tags`.
    ///
    /// Each locally installed model carries a sha256 `digest` that changes
    /// whenever the underlying weights change, even if the tag name stays the
    /// same — letting the index detect a model that was retagged out from
    /// under it. We match the configured model name against the `name`/`model`
    /// fields, tolerating an implicit `:latest` tag.
    pub async fn digest(&self) -> Result<String> {
        let url = format!("{}/api/tags", self.base);
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("sending tags request to {url}"))?;
        let response = check_status(response, "/api/tags").await?;
        let parsed: TagsResponse = response
            .json()
            .await
            .context("decoding ollama tags response")?;

        let wanted = &self.model;
        let with_latest = format!("{wanted}:latest");
        let found = parsed.models.into_iter().find(|m| {
            [&m.name, &m.model]
                .into_iter()
                .flatten()
                .any(|n| n == wanted || n == &with_latest)
        });
        match found {
            Some(m) => m
                .digest
                .filter(|d| !d.is_empty())
                .ok_or_else(|| anyhow::anyhow!("model {wanted} has no digest in /api/tags")),
            None => bail!("model {wanted} is not installed (not listed by /api/tags)"),
        }
    }
}

/// Return the response unchanged on a 2xx status, otherwise fail with an error
/// that includes Ollama's response body.
///
/// Ollama reports failures as a JSON `{"error": "..."}` payload alongside the
/// HTTP status; `reqwest::Response::error_for_status` discards that body, so a
/// 400 would otherwise surface as a bare status code with no hint as to which
/// input it rejected or why (e.g. an input that exceeds the model's context
/// length). We read the body and fold the `error` string into the message.
async fn check_status(response: reqwest::Response, endpoint: &str) -> Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().await.unwrap_or_default();
    let detail = serde_json::from_str::<ErrorBody>(&body)
        .ok()
        .and_then(|e| e.error)
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| {
            let trimmed = body.trim();
            if trimmed.is_empty() {
                "no response body".to_string()
            } else {
                trimmed.to_string()
            }
        });
    bail!("ollama returned {status} for {endpoint}: {detail}");
}

/// The `{"error": "..."}` payload Ollama returns alongside an error status.
#[derive(Debug, Deserialize)]
struct ErrorBody {
    #[serde(default)]
    error: Option<String>,
}

/// Request body for `POST /api/embed`.
#[derive(Debug, Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

/// The subset of the embed response we consume.
#[derive(Debug, Deserialize)]
struct EmbedResponse {
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
}

/// The subset of the `GET /api/tags` response we consume.
#[derive(Debug, Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagEntry>,
}

#[derive(Debug, Deserialize)]
struct TagEntry {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    digest: Option<String>,
}
