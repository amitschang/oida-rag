//! Minimal OpenAI-compatible embedding client.
//!
//! Covers only the slice of the OpenAI embeddings API the hybrid text index
//! needs: `POST /v1/embeddings` to turn text into vectors. Both Ollama (via its
//! `/v1` compatibility layer) and vLLM speak this protocol, so a single client
//! serves either backend — the endpoint URL is the only thing that changes.
//!
//! The index pins the embedding model by *name* alone (see [`crate::hybrid`]):
//! there is no portable content digest across these servers, so the convention
//! is to encode a weights identity (a commit hash, a quantization tag, a
//! `num_ctx` variant) into the model name itself and let the name be the pin.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// One embedding backend: a base URL and its live in-flight request count, used
/// to balance load across replicas by fewest connections.
#[derive(Debug)]
struct Backend {
    base: String,
    /// Embed requests currently in flight to this backend (the least-connections
    /// signal). Shared across [`Embedder`] clones via the `Arc` on `backends`.
    inflight: AtomicUsize,
}

/// A reusable embedding client over one or more interchangeable backends.
///
/// With multiple backends (replicas of the same model served behind separate
/// addresses), each request is routed to the backend with the fewest in-flight
/// requests — client-side least-connections balancing, so no external load
/// balancer is needed and a slow replica naturally receives less work. A single
/// backend (the common case: Ollama, or the query path) skips the bookkeeping.
#[derive(Clone, Debug)]
pub struct Embedder {
    http: reqwest::Client,
    /// Shared so cloned `Embedder`s (the build clones one per job) balance over
    /// the *same* live in-flight counters rather than diverging per clone.
    backends: Arc<Vec<Backend>>,
    model: String,
    /// Optional bearer token sent as `Authorization: Bearer <key>`. vLLM ignores
    /// it unless launched with `--api-key`; Ollama ignores it entirely.
    api_key: Option<String>,
}

/// A reservation on the least-loaded backend: increments its in-flight count on
/// acquisition and decrements on drop, so the count tracks a request for its full
/// lifetime (including early returns on error).
struct BackendLease<'a> {
    backend: &'a Backend,
}

impl<'a> BackendLease<'a> {
    fn base(&self) -> &str {
        &self.backend.base
    }
}

impl Drop for BackendLease<'_> {
    fn drop(&mut self) {
        self.backend.inflight.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Embedder {
    /// Build a client for `model` against the OpenAI-compatible server(s) at
    /// `base` (e.g. `http://localhost:11434` for Ollama or `http://localhost:8000`
    /// for vLLM). `api_key`, when set, is sent as a bearer token.
    ///
    /// `base` may be a comma-separated list of addresses serving the same model
    /// (replicas); requests are then balanced across them by least connections.
    pub fn new(base: &str, model: impl Into<String>, api_key: Option<String>) -> Result<Self> {
        let backends: Vec<Backend> = base
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                reqwest::Url::parse(s).with_context(|| format!("invalid embed host {s}"))?;
                Ok(Backend {
                    base: s.trim_end_matches('/').to_string(),
                    inflight: AtomicUsize::new(0),
                })
            })
            .collect::<Result<_>>()?;
        if backends.is_empty() {
            bail!("no embed host provided in {base:?}");
        }
        Ok(Self {
            http: reqwest::Client::new(),
            backends: Arc::new(backends),
            model: model.into(),
            api_key,
        })
    }

    /// Reserve the backend with the fewest in-flight requests (least-connections),
    /// counting this request against it until the returned lease is dropped. A
    /// single backend short-circuits the scan.
    fn lease(&self) -> BackendLease<'_> {
        let backend = self
            .backends
            .iter()
            .min_by_key(|b| b.inflight.load(Ordering::Relaxed))
            .expect("Embedder always has at least one backend");
        backend.inflight.fetch_add(1, Ordering::Relaxed);
        BackendLease { backend }
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
        // Held for the whole request so the in-flight count — and thus the
        // least-connections choice for concurrent calls — stays accurate.
        let lease = self.lease();
        let url = format!("{}/v1/embeddings", lease.base());
        let body = EmbedRequest {
            model: &self.model,
            input: inputs,
        };
        let mut request = self.http.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .with_context(|| format!("sending embed request to {url}"))?;
        if !response.status().is_success() {
            dump_failing_request(&body);
        }
        let response = check_status(response, "/v1/embeddings").await?;
        let parsed: EmbedResponse = response
            .json()
            .await
            .context("decoding embed response")?;
        if parsed.data.len() != inputs.len() {
            bail!(
                "embed server returned {} embeddings for {} inputs",
                parsed.data.len(),
                inputs.len()
            );
        }
        // The OpenAI spec returns each embedding tagged with its input `index`;
        // sort by it so we never depend on the server preserving request order.
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);
        if data.iter().any(|d| d.embedding.is_empty()) {
            bail!("embed server returned an empty embedding for model {}", self.model);
        }
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }

    /// Embed a single text and return its vector.
    pub async fn embed_one(&self, input: &str) -> Result<Vec<f32>> {
        let mut out = self.embed(std::slice::from_ref(&input.to_string())).await?;
        out.pop()
            .ok_or_else(|| anyhow::anyhow!("embed server returned no embedding"))
    }
}

/// Return the response unchanged on a 2xx status, otherwise fail with an error
/// that includes the server's response body.
///
/// Both Ollama and OpenAI-style servers report failures as a JSON payload
/// alongside the HTTP status; `reqwest::Response::error_for_status` discards
/// that body, so a 400 would otherwise surface as a bare status code with no
/// hint as to which input it rejected or why (e.g. an input that exceeds the
/// model's context length). We read the body and fold the error message into
/// the result.
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
    bail!("embed server returned {status} for {endpoint}: {detail}");
}

/// On an embed failure, write the exact request body to a file so the offending
/// input can be isolated and replayed (e.g. `curl -d @oida-failed-embed.json`).
/// A model runner can crash on a specific chunk — typically one that tokenizes
/// past the model's context window — and the error alone doesn't say which of
/// the batch's inputs did it. Best-effort: a dump failure must not mask the
/// original embed error, so I/O errors here are only logged.
fn dump_failing_request(body: &EmbedRequest<'_>) {
    const PATH: &str = "oida-failed-embed.json";
    match serde_json::to_vec_pretty(body) {
        Ok(json) => match std::fs::write(PATH, &json) {
            Ok(()) => tracing::error!(
                "embed request failed; dumped {} inputs to {PATH} for isolation",
                body.input.len()
            ),
            Err(e) => tracing::error!("embed request failed; could not write {PATH}: {e}"),
        },
        Err(e) => tracing::error!("embed request failed; could not serialize request: {e}"),
    }
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

/// Request body for `POST /v1/embeddings`.
#[derive(Debug, Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

/// The subset of the embed response we consume.
#[derive(Debug, Deserialize)]
struct EmbedResponse {
    #[serde(default)]
    data: Vec<EmbedDatum>,
}

/// A single `{index, embedding}` entry from the response's `data` array.
#[derive(Debug, Deserialize)]
struct EmbedDatum {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_single_and_multi_host() {
        let one = Embedder::new("http://h:8000/", "m", None).unwrap();
        assert_eq!(one.backends.len(), 1);
        // Trailing slash trimmed, so the request path joins cleanly.
        assert_eq!(one.backends[0].base, "http://h:8000");

        let many = Embedder::new("http://a:8000, http://b:8000 ,", "m", None).unwrap();
        assert_eq!(many.backends.len(), 2, "blank entry from trailing comma ignored");
        assert_eq!(many.backends[1].base, "http://b:8000");

        assert!(Embedder::new("not a url", "m", None).is_err());
        assert!(Embedder::new(" , ", "m", None).is_err(), "no usable host");
    }

    #[test]
    fn lease_picks_least_loaded_and_releases_on_drop() {
        let e = Embedder::new("http://a:8000,http://b:8000,http://c:8000", "m", None).unwrap();

        // First lease takes any backend (all at 0); pin the rest above it so the
        // next picks count deterministically.
        let l1 = e.lease();
        assert_eq!(e.backends.iter().map(|b| b.inflight.load(Ordering::Relaxed)).sum::<usize>(), 1);

        // Two more leases must land on the two still at zero — never doubling up
        // on l1's backend while idle ones remain.
        let l2 = e.lease();
        let l3 = e.lease();
        for b in e.backends.iter() {
            assert_eq!(b.inflight.load(Ordering::Relaxed), 1, "load spread one-each");
        }

        drop(l2);
        // The freed backend is now the least loaded, so the next lease reuses it.
        let l4 = e.lease();
        let max = e.backends.iter().map(|b| b.inflight.load(Ordering::Relaxed)).max().unwrap();
        assert_eq!(max, 1, "reused the idle backend rather than stacking");
        drop((l1, l3, l4));

        for b in e.backends.iter() {
            assert_eq!(b.inflight.load(Ordering::Relaxed), 0, "all released on drop");
        }
    }

    #[test]
    fn request_serializes_model_and_input() {
        let input = ["hello".to_string()];
        let body = EmbedRequest { model: "nomic-embed-text", input: &input };
        let value = serde_json::to_value(&body).unwrap();
        assert_eq!(value["model"], "nomic-embed-text");
        assert_eq!(value["input"][0], "hello");
        assert!(value.get("options").is_none());
    }

    #[test]
    fn parses_openai_data_array() {
        let body = json!({
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]}
            ]
        });
        let parsed: EmbedResponse = serde_json::from_value(body).unwrap();
        assert_eq!(parsed.data.len(), 2);
        assert_eq!(parsed.data[1].embedding, vec![0.3, 0.4]);
    }

    #[test]
    fn error_message_from_nested_openai_shape() {
        let body = r#"{"error": {"message": "context length exceeded", "type": "invalid_request"}}"#;
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
