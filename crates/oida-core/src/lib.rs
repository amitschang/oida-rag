//! Core domain logic for the OIDA assistant: configuration, the DuckDB-backed
//! index (search + relationship graph), and artifact access. This crate is
//! transport-agnostic — it knows nothing about MCP or Ollama.

pub mod artifacts;
pub mod config;
pub mod graph;
pub mod index;
pub mod model;
pub mod schema;
pub mod search;

pub use config::Config;
pub use index::Index;
pub use model::{Artifact, Document, RelatedEdge, RelationKind, SearchHit};
pub use search::SearchParams;
