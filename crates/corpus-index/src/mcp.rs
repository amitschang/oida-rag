//! Optional MCP tool layer (`feature = "mcp"`).
//!
//! Corpus-agnostic MCP tools built over the [`CorpusBackend`] trait, so any
//! corpus's server gets the reusable tools (search, artifact text, SQL, schema,
//! hybrid) for free — a `CorpusBackend` impl plus a one-line router compose. A
//! server with no bespoke tools needs nothing else; one with domain tools merges
//! its own [`ToolRouter`] on top with `+`.
//!
//! The tools are assembled with rmcp's low-level [`ToolRoute::new_dyn`] rather
//! than the `#[tool]` macros, because the macros only expand over a concrete
//! server type — these builders are generic over the server `S` and (for search
//! and hybrid) the corpus's result type `D`.

use std::collections::HashMap;
use std::future::Future;

use rmcp::ErrorData as McpError;
use rmcp::handler::server::common::schema_for_type;
use rmcp::handler::server::router::tool::{ToolRoute, ToolRouter};
use rmcp::handler::server::tool::{IntoCallToolResult, ToolCallContext};
use rmcp::handler::server::wrapper::Json;
use rmcp::model::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::artifacts::read_artifact_text;
use crate::hybrid::HybridIndex;
use crate::index::Index;
use crate::model::{HybridHit, SearchHit, TableSchema};
use crate::row::{DocumentRow, SearchableRow};
use crate::search::SearchParams;
use crate::source::ArtifactReader;

/// Default and maximum number of search hits returned in one call.
const DEFAULT_SEARCH_LIMIT: u32 = 10;
const MAX_SEARCH_LIMIT: u32 = 50;
/// Default and maximum bytes of artifact text returned in one call.
const DEFAULT_TEXT_BYTES: u64 = 8 * 1024;
const MAX_TEXT_BYTES: u64 = 64 * 1024;
/// Default and maximum rows returned by a `run_sql` query.
const DEFAULT_SQL_ROWS: u32 = 200;
const MAX_SQL_ROWS: u32 = 2000;

/// A server the generic MCP tools can query: the framework [`Index`], the
/// optional [`HybridIndex`], and the artifact [`ArtifactReader`].
///
/// The one seam a corpus's MCP server implements to inherit the generic tools.
pub trait CorpusBackend: Clone + Send + Sync + 'static {
    fn index(&self) -> &Index;
    fn hybrid(&self) -> Option<&HybridIndex>;
    fn artifacts(&self) -> &ArtifactReader;
}

/// Convert an `anyhow` error into an MCP internal error.
fn internal(e: anyhow::Error) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

/// Build one [`ToolRoute`] from a name, description, and an async handler taking
/// the (cloned) server and the deserialized request.
///
/// Uses [`ToolRoute::new_dyn`] so the boxed-future return coerces cleanly; the
/// request schema is derived from `Req`'s [`JsonSchema`], and the response is
/// serialized through rmcp's [`Json`] wrapper — the same shape the `#[tool]`
/// macro produces.
fn make_route<S, Req, Resp, H, Fut>(
    name: &'static str,
    description: &'static str,
    handler: H,
) -> ToolRoute<S>
where
    S: CorpusBackend,
    Req: for<'de> Deserialize<'de> + JsonSchema + 'static,
    Resp: Serialize + JsonSchema + 'static,
    H: Fn(S, Req) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<Resp, McpError>> + Send + 'static,
{
    let tool = Tool::new(name, description, schema_for_type::<Req>());
    ToolRoute::new_dyn(tool, move |mut ctx: ToolCallContext<S>| {
        let handler = handler.clone();
        let server = ctx.service.clone();
        let args = ctx.arguments.take().unwrap_or_default();
        Box::pin(async move {
            let req: Req = serde_json::from_value(serde_json::Value::Object(args))
                .map_err(|e| McpError::invalid_params(format!("invalid arguments: {e}"), None))?;
            let resp = handler(server, req).await?;
            Json(resp).into_call_tool_result()
        })
    })
}

// ---- request / response DTOs -------------------------------------------------

/// Empty request for a no-argument tool (e.g. `describe_schema`).
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct NoArgs {}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchDocumentsRequest {
    /// Free-text query. Whitespace-separated terms are matched independently;
    /// documents containing more terms rank higher.
    pub query: String,
    /// Restrict to documents that have an artifact of this MIME type
    /// (e.g. `application/pdf`, `text/plain`).
    #[serde(default)]
    pub media_type: Option<String>,
    /// Max hits to return (default 10, max 50).
    #[serde(default)]
    pub limit: Option<u32>,
    /// Number of leading hits to skip, for pagination (default 0).
    #[serde(default)]
    pub offset: Option<u32>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchResponse<D> {
    pub count: usize,
    pub hits: Vec<SearchHit<D>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetArtifactTextRequest {
    /// Id of the document that owns the artifact.
    pub id: String,
    /// Artifact file name (e.g. `thdb0402.ocr`).
    pub name: String,
    /// MIME type of the artifact, if known (helps decide readability).
    #[serde(default)]
    pub media_type: Option<String>,
    /// Byte offset to start reading from (default 0).
    #[serde(default)]
    pub offset: Option<u64>,
    /// Max bytes of text to return (default 8192, max 65536).
    #[serde(default)]
    pub max_bytes: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunSqlRequest {
    /// A single read-only SQL statement (SELECT/WITH/DESCRIBE/EXPLAIN/SHOW).
    /// Writes, DDL and multiple statements are rejected.
    pub sql: String,
    /// Max rows to return (default 200, max 2000). Excess rows set `truncated`.
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SchemaResponse {
    pub tables: Vec<TableSchema>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct HybridSearchRequest {
    /// Natural-language or keyword query matched against the *contents* of
    /// documents (OCR text), combining semantic similarity and keyword search.
    pub query: String,
    /// Max documents to return (default 10, max 50).
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct HybridSearchResponse<D> {
    pub count: usize,
    pub hits: Vec<HybridHit<D>>,
}

// ---- generic tool descriptions ----------------------------------------------

const GET_ARTIFACT_TEXT_DESC: &str =
    "Read the text of an artifact (intended for .ocr / text/plain files) by document id and \
     file name. Returns a status: text_loaded, artifact_file_missing, unsupported_artifact_type, \
     or artifact_root_not_configured. Supports paging via offset/max_bytes.";

const RUN_SQL_DESC: &str =
    "Run a single read-only SQL query (DataFusion SQL dialect) against the index for ad-hoc \
     counting, grouping, and filtering. The index has a `documents` table (one row per document) \
     and an `artifacts` table (one row per artifact, joinable on `documents.id = artifacts.id`); \
     list-typed columns can be expanded with UNNEST. Call describe_schema first for the live \
     column names and Arrow types. Only SELECT/WITH/EXPLAIN/SHOW are allowed; writes, DDL and \
     multiple statements are rejected. Returns columns and JSON-object rows keyed by column name \
     (lists become arrays); on a bad query the `error` field explains why so you can fix and retry.";

const DESCRIBE_SCHEMA_DESC: &str =
    "List the columns and Arrow types of the `documents` and `artifacts` tables (and `chunks` \
     when the full-text index is built). Use this to write correct run_sql queries.";

// ---- route builders ----------------------------------------------------------

/// The fully corpus-agnostic tools: artifact-text retrieval, read-only SQL, and
/// schema introspection. None mention any corpus concept.
pub fn generic_router<S: CorpusBackend>() -> ToolRouter<S> {
    ToolRouter::new()
        .with_route(make_route(
            "get_artifact_text",
            GET_ARTIFACT_TEXT_DESC,
            |server: S, req: GetArtifactTextRequest| async move {
                let max_bytes = req
                    .max_bytes
                    .unwrap_or(DEFAULT_TEXT_BYTES)
                    .clamp(1, MAX_TEXT_BYTES);
                Ok(read_artifact_text(
                    Some(server.artifacts()),
                    &req.id,
                    &req.name,
                    req.media_type.as_deref(),
                    req.offset.unwrap_or(0),
                    max_bytes,
                )
                .await)
            },
        ))
        .with_route(make_route(
            "run_sql",
            RUN_SQL_DESC,
            |server: S, req: RunSqlRequest| async move {
                let limit = req
                    .limit
                    .unwrap_or(DEFAULT_SQL_ROWS)
                    .clamp(1, MAX_SQL_ROWS) as usize;
                Ok(server.index().run_sql(&req.sql, limit).await)
            },
        ))
        .with_route(make_route(
            "describe_schema",
            DESCRIBE_SCHEMA_DESC,
            |server: S, _req: NoArgs| async move {
                let tables = server.index().describe_schema().await.map_err(internal)?;
                Ok(SchemaResponse { tables })
            },
        ))
}

/// The metadata keyword-search tool, instantiated at the corpus's search result
/// type `D`. `description` is corpus-supplied so it can name the fields the
/// corpus actually searches (e.g. Bates numbers, custodians).
pub fn search_route<S: CorpusBackend, D: SearchableRow>(
    description: &'static str,
) -> ToolRouter<S> {
    ToolRouter::new().with_route(make_route(
        "search_documents",
        description,
        |server: S, req: SearchDocumentsRequest| async move {
            let limit = req
                .limit
                .unwrap_or(DEFAULT_SEARCH_LIMIT)
                .clamp(1, MAX_SEARCH_LIMIT);
            let params = SearchParams {
                query: req.query,
                media_type: req.media_type,
                limit,
                offset: req.offset.unwrap_or(0),
            };
            let hits = server.index().search::<D>(&params).await.map_err(internal)?;
            Ok(SearchResponse {
                count: hits.len(),
                hits,
            })
        },
    ))
}

/// The hybrid (keyword + semantic) content-search tool, hydrating results as the
/// corpus's document type `D`. `description` is corpus-supplied.
pub fn hybrid_route<S: CorpusBackend, D: DocumentRow + Default>(
    description: &'static str,
) -> ToolRouter<S> {
    ToolRouter::new().with_route(make_route(
        "hybrid_search",
        description,
        |server: S, req: HybridSearchRequest| async move {
            let Some(hybrid) = server.hybrid() else {
                return Err(McpError::internal_error(
                    "hybrid text index is not built; run the full-text ingest",
                    None,
                ));
            };
            let limit = req
                .limit
                .unwrap_or(DEFAULT_SEARCH_LIMIT)
                .clamp(1, MAX_SEARCH_LIMIT) as usize;
            let chunk_hits = hybrid.query(&req.query, limit).await.map_err(internal)?;

            // Hydrate document metadata in one query, preserving rank order.
            let ids: Vec<String> = chunk_hits.iter().map(|h| h.doc_id.clone()).collect();
            let docs = server.index().get_many::<D>(&ids).await.map_err(internal)?;
            let mut by_id: HashMap<String, D> =
                docs.into_iter().map(|d| (d.id().to_string(), d)).collect();

            let hits: Vec<HybridHit<D>> = chunk_hits
                .iter()
                .map(|h| HybridHit {
                    document: by_id.remove(&h.doc_id).unwrap_or_default(),
                    score: h.score,
                    artifact_name: Some(h.artifact_name.clone()),
                    snippet: Some(h.snippet.clone()),
                })
                .collect();

            Ok(HybridSearchResponse {
                count: hits.len(),
                hits,
            })
        },
    ))
}
