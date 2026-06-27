//! Core domain logic for the OIDA assistant: configuration, the LanceDB-backed
//! index (search + relationship graph + hybrid text search), and artifact
//! access. This crate is transport-agnostic — it knows nothing about MCP or
//! Ollama.

pub mod artifacts;
pub mod config;
pub mod derived;
pub mod embed;
pub mod graph;
pub mod hybrid;
pub mod index;
pub mod ingest;
pub mod model;
pub mod progress;
pub mod raw;
pub mod search;
pub mod solr;
pub mod solr_map;
pub mod source;
pub mod sql;
pub mod update;

pub use config::Config;
pub use derived::build_raw_and_text;
pub use embed::Embedder;
pub use hybrid::{HybridIndex, IndexStats};
pub use index::Index;
pub use ingest::MetadataStats;
pub use model::{Artifact, Document, HybridHit, RawArtifact, RelatedEdge, RelationKind, SearchHit};
pub use model::{ColumnInfo, SqlQueryResult, TableSchema};
pub use search::SearchParams;
pub use source::ArtifactSource;
pub use update::{ApplyStats, UpdatePlan};
