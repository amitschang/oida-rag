//! MCP tool surface exposing the OIDA index to a model.
//!
//! The corpus-agnostic tools (search, artifact text, SQL, schema, hybrid) come
//! from the framework's [`corpus_index::mcp`] builders, instantiated at OIDA's
//! result types; this crate adds only the OIDA-specific tools (document lookup
//! with Bates resolution, relationship graph) and wires them together.

use std::sync::Arc;

use corpus_index::mcp::{CorpusBackend, generic_router, hybrid_route, search_route};
use oida::{Artifact, ArtifactReader, CorpusQueries, Document, DocumentSummary, HybridIndex, Index, RelatedEdge};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{ErrorData as McpError, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Maximum relationship-traversal depth a caller may request.
const MAX_DEPTH: u32 = 3;

/// Keyword-search tool description — names OIDA's searchable fields so the model
/// knows what metadata it can match on.
const SEARCH_DESC: &str = "Search OIDA documents by keyword over metadata (title, Bates number, \
    authors, custodian, topic, description). Returns ranked document summaries with provenance. \
    Matching is metadata-only; use get_artifact_text for OCR content.";

/// Hybrid content-search tool description.
const HYBRID_DESC: &str = "Search the *contents* of documents (OCR text) using a hybrid of \
    semantic (vector) and keyword (full-text) matching fused with Reciprocal Rank Fusion. Unlike \
    search_documents (which matches metadata only), this finds documents by what they actually \
    say. Returns ranked documents with a matching text snippet and the source artifact. Requires \
    the hybrid index to be built (oida-cli ingest --full-text).";

/// The MCP server state: a shared index, the hybrid index, and the artifact
/// resolver.
#[derive(Clone)]
pub struct OidaServer {
    index: Arc<Index>,
    /// The hybrid text index, present only when it has been built. Tools that
    /// need it return a helpful error when it is absent.
    hybrid: Arc<Option<HybridIndex>>,
    /// The serving-time artifact resolver (materialized LanceDB tiers, then the
    /// original source). `get_artifact_text` returns a status when it reports
    /// no configured tier.
    reader: Arc<ArtifactReader>,
}

impl OidaServer {
    pub fn new(index: Arc<Index>, hybrid: Option<HybridIndex>, reader: ArtifactReader) -> Self {
        Self {
            index,
            hybrid: Arc::new(hybrid),
            reader: Arc::new(reader),
        }
    }

    /// The full advertised tool set: the framework's corpus-agnostic tools —
    /// instantiated at OIDA's `DocumentSummary` — merged with OIDA's own tools.
    /// `ToolRouter`'s `+` is the composition primitive.
    fn router() -> ToolRouter<Self> {
        generic_router::<Self>()
            + search_route::<Self, DocumentSummary>(SEARCH_DESC)
            + hybrid_route::<Self, DocumentSummary>(HYBRID_DESC)
            + Self::domain_router()
    }
}

/// The one seam that lets `OidaServer` inherit the framework's generic tools.
impl CorpusBackend for OidaServer {
    fn index(&self) -> &Index {
        &self.index
    }
    fn hybrid(&self) -> Option<&HybridIndex> {
        self.hybrid.as_ref().as_ref()
    }
    fn artifacts(&self) -> &ArtifactReader {
        &self.reader
    }
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

/// Convert an `anyhow` error into an MCP internal error.
fn internal(e: anyhow::Error) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

// OIDA-specific tools: their interface names a domain concept (Bates numbers,
// the relationship graph) that has no corpus-independent meaning. A different
// corpus supplies its own domain router (or none) and merges it with the
// framework's generic routers.
#[tool_router(router = domain_router)]
impl OidaServer {
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
            (Some(id), _) => self.index.get_document_by_id(id).await.map_err(internal)?,
            (None, Some(bn)) => self.index.get_document_by_bn(bn).await.map_err(internal)?,
            (None, None) => {
                return Err(McpError::invalid_params(
                    "provide either `id` or `bn`",
                    None,
                ));
            }
        };
        let artifacts = match &doc {
            Some(d) => self.index.get_artifacts(&d.id).await.map_err(internal)?,
            None => Vec::new(),
        };
        Ok(Json(DocumentResponse {
            found: doc.is_some(),
            document: doc,
            artifacts,
        }))
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
        let edges: Vec<RelatedEdge> = self.index.related(&req.start, depth).await.map_err(internal)?;
        Ok(Json(RelatedResponse {
            count: edges.len(),
            edges,
        }))
    }
}

#[tool_handler(router = Self::router())]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advertises_the_full_tool_set() {
        // The framework generic tools + OIDA domain tools must merge to exactly
        // the seven advertised tools.
        let all = OidaServer::router().list_all();
        let names: Vec<&str> = all.iter().map(|t| t.name.as_ref()).collect();
        for expected in [
            "search_documents",
            "get_artifact_text",
            "run_sql",
            "describe_schema",
            "hybrid_search",
            "get_document",
            "get_related",
        ] {
            assert!(names.contains(&expected), "missing tool {expected}");
        }
        assert_eq!(all.len(), 7, "expected 7 tools, got {}", all.len());

        // The OIDA-only tools live in the domain router.
        let domain = OidaServer::domain_router();
        assert!(domain.has_route("get_document"));
        assert!(domain.has_route("get_related"));
    }
}
