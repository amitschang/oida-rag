//! MCP tool surface exposing the OIDA index to a model.
//!
//! Each tool is a thin, well-typed wrapper over [`oida_core`]. Inputs and
//! outputs are JSON-schema'd so the client can advertise them to the LLM and
//! receive structured results.

use std::sync::Arc;

use oida_core::artifacts::{ArtifactText, read_artifact_text};
use oida_core::model::{Artifact, Document, RelatedEdge, SearchHit};
use oida_core::{Config, Index, SearchParams, SqlQueryResult, TableSchema};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{ErrorData as McpError, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Default and maximum number of search hits returned in one call.
const DEFAULT_SEARCH_LIMIT: u32 = 10;
const MAX_SEARCH_LIMIT: u32 = 50;
/// Default and maximum bytes of artifact text returned in one call.
const DEFAULT_TEXT_BYTES: u64 = 8 * 1024;
const MAX_TEXT_BYTES: u64 = 64 * 1024;
/// Maximum relationship-traversal depth a caller may request.
const MAX_DEPTH: u32 = 3;
/// Default and maximum rows returned by a `run_sql` query.
const DEFAULT_SQL_ROWS: u32 = 200;
const MAX_SQL_ROWS: u32 = 2000;

/// The MCP server state: a shared index plus configuration.
#[derive(Clone)]
pub struct OidaServer {
    index: Arc<Index>,
    config: Arc<Config>,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl OidaServer {
    pub fn new(index: Arc<Index>, config: Arc<Config>) -> Self {
        Self {
            index,
            config,
            tool_router: Self::tool_router(),
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchDocumentsRequest {
    /// Free-text query. Whitespace-separated terms are matched independently;
    /// documents containing more terms rank higher.
    pub query: String,
    /// Restrict to documents that have an artifact of this MIME type
    /// (e.g. `application/pdf`, `text/plain`).
    #[serde(default)]
    pub media_type: Option<String>,
    /// Restrict to documents whose custodian contains this substring.
    #[serde(default)]
    pub custodian: Option<String>,
    /// Max hits to return (default 10, max 50).
    #[serde(default)]
    pub limit: Option<u32>,
    /// Number of leading hits to skip, for pagination (default 0).
    #[serde(default)]
    pub offset: Option<u32>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchResponse {
    pub count: usize,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetDocumentRequest {
    /// Document id. Provide this or `bn`.
    #[serde(default)]
    pub id: Option<String>,
    /// Bates number. Used when `id` is not provided.
    #[serde(default)]
    pub bn: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct DocumentResponse {
    pub found: bool,
    pub document: Option<Document>,
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetArtifactTextRequest {
    /// Artifact file name (e.g. `thdb0402.ocr`), as listed by `get_document`.
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
pub struct GetRelatedRequest {
    /// Starting document: either an `id` or a Bates number (`bn`).
    pub start: String,
    /// Traversal depth (1 = direct neighbors; default 1, max 3).
    #[serde(default)]
    pub depth: Option<u32>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct RelatedResponse {
    pub count: usize,
    pub edges: Vec<RelatedEdge>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunSqlRequest {
    /// A single read-only SQL statement (SELECT/WITH/DESCRIBE/EXPLAIN/SHOW/
    /// SUMMARIZE). Writes, DDL, COPY, ATTACH, INSTALL/LOAD and multiple
    /// statements are rejected.
    pub sql: String,
    /// Max rows to return (default 200, max 2000). Excess rows set `truncated`.
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SchemaResponse {
    pub tables: Vec<TableSchema>,
}

/// Convert an `anyhow` error into an MCP internal error.
fn internal(e: anyhow::Error) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

#[tool_router]
impl OidaServer {
    /// Keyword-search the OIDA index for documents.
    #[tool(
        description = "Search OIDA documents by keyword over metadata (title, Bates number, \
        authors, custodian, topic, description). Returns ranked document summaries with \
        provenance. Matching is metadata-only; use get_artifact_text for OCR content."
    )]
    async fn search_documents(
        &self,
        Parameters(req): Parameters<SearchDocumentsRequest>,
    ) -> Result<Json<SearchResponse>, McpError> {
        let limit = req
            .limit
            .unwrap_or(DEFAULT_SEARCH_LIMIT)
            .clamp(1, MAX_SEARCH_LIMIT);
        let params = SearchParams {
            query: req.query,
            media_type: req.media_type,
            custodian: req.custodian,
            limit,
            offset: req.offset.unwrap_or(0),
        };
        let hits = self.index.search(&params).map_err(internal)?;
        Ok(Json(SearchResponse {
            count: hits.len(),
            hits,
        }))
    }

    /// Fetch a single document's metadata and its artifacts.
    #[tool(
        description = "Get full metadata for one OIDA document (by id or Bates number) \
        along with the list of its artifacts (OCR text, PDF, images, etc.)."
    )]
    async fn get_document(
        &self,
        Parameters(req): Parameters<GetDocumentRequest>,
    ) -> Result<Json<DocumentResponse>, McpError> {
        let doc = match (&req.id, &req.bn) {
            (Some(id), _) => self.index.get_document_by_id(id).map_err(internal)?,
            (None, Some(bn)) => self.index.get_document_by_bn(bn).map_err(internal)?,
            (None, None) => {
                return Err(McpError::invalid_params(
                    "provide either `id` or `bn`",
                    None,
                ));
            }
        };
        let artifacts = match &doc {
            Some(d) => self.index.get_artifacts(&d.id).map_err(internal)?,
            None => Vec::new(),
        };
        Ok(Json(DocumentResponse {
            found: doc.is_some(),
            document: doc,
            artifacts,
        }))
    }

    /// Read text from an artifact file on disk.
    #[tool(
        description = "Read the text of an artifact (intended for .ocr / text/plain files) \
        by file name. Returns a status: text_loaded, artifact_file_missing, \
        unsupported_artifact_type, or artifact_root_not_configured. Supports paging via \
        offset/max_bytes."
    )]
    async fn get_artifact_text(
        &self,
        Parameters(req): Parameters<GetArtifactTextRequest>,
    ) -> Result<Json<ArtifactText>, McpError> {
        let max_bytes = req
            .max_bytes
            .unwrap_or(DEFAULT_TEXT_BYTES)
            .clamp(1, MAX_TEXT_BYTES);
        let result = read_artifact_text(
            &self.config,
            &req.name,
            req.media_type.as_deref(),
            req.offset.unwrap_or(0),
            max_bytes,
        );
        Ok(Json(result))
    }

    /// Traverse the document relationship graph.
    #[tool(
        description = "Find documents connected to a starting document (by id or Bates \
        number) through attachments, related references, mentions, or shared email \
        conversation. Returns typed edges with resolved neighbor documents."
    )]
    async fn get_related(
        &self,
        Parameters(req): Parameters<GetRelatedRequest>,
    ) -> Result<Json<RelatedResponse>, McpError> {
        let depth = req.depth.unwrap_or(1).clamp(1, MAX_DEPTH);
        let edges: Vec<RelatedEdge> = self.index.related(&req.start, depth).map_err(internal)?;
        Ok(Json(RelatedResponse {
            count: edges.len(),
            edges,
        }))
    }

    /// Run an arbitrary read-only SQL query against the cache.
    #[tool(
        description = "Run a single read-only SQL query against the cache for ad-hoc \
        counting, grouping, and filtering. Two tables: `documents` (one row per document) \
        and `artifacts` (one row per artifact, joinable on `documents.id = artifacts.id`). \
        Key `documents` columns: id, bn, title, industry, collection, genre, date_sent, \
        date_received, topic, description, keywords, conversation (all VARCHAR), \
        artifact_count (BIGINT), and list columns custodian, authors, recipients, cc, \
        attachments, related, mentions, artifact_types (all VARCHAR[] -- use UNNEST to \
        group by their elements). `artifacts` columns: id, name, media_type, md5 (VARCHAR), \
        size (BIGINT). Only SELECT/WITH/DESCRIBE/EXPLAIN/SHOW/SUMMARIZE are allowed; writes, \
        DDL, file/network access and multiple statements are rejected. Returns columns and \
        JSON-valued rows (lists become arrays); on a bad query the `error` field explains \
        why so you can fix and retry. Call describe_schema for the full column list."
    )]
    async fn run_sql(
        &self,
        Parameters(req): Parameters<RunSqlRequest>,
    ) -> Result<Json<SqlQueryResult>, McpError> {
        let limit = req
            .limit
            .unwrap_or(DEFAULT_SQL_ROWS)
            .clamp(1, MAX_SQL_ROWS) as usize;
        Ok(Json(self.index.run_sql(&req.sql, limit)))
    }

    /// Report the columns and types of the cache tables.
    #[tool(
        description = "List the columns and DuckDB types of the `documents` and `artifacts` \
        tables. Use this to write correct run_sql queries."
    )]
    async fn describe_schema(&self) -> Result<Json<SchemaResponse>, McpError> {
        let tables = self.index.describe_schema().map_err(internal)?;
        Ok(Json(SchemaResponse { tables }))
    }
}

#[tool_handler]
impl ServerHandler for OidaServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "OIDA document assistant. Tools query a local index of document metadata and \
             relationships (emails, attachments, mentions) plus on-disk artifacts (OCR text, \
             PDFs, images). Typical flow: search_documents to find candidates, get_document for \
             details and artifact lists, get_artifact_text for OCR content, get_related to \
             explore the document network. For counts, grouping, or filters the fixed tools \
             don't cover, use run_sql (read-only SQL over the documents/artifacts tables); call \
             describe_schema first to learn the columns.",
        )
    }
}
