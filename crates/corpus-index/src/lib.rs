//! Corpus-agnostic engine for metadata → LanceDB hybrid search.
//!
//! This crate is the reusable framework: an embedded LanceDB store with scalar,
//! full-text (BM25), and vector indexes; a hybrid (RRF) search engine; a raw
//! artifact store and layered retrieval resolver; read-only SQL; and the
//! ingest/incremental-apply drivers. It knows nothing about any particular
//! corpus — a domain crate supplies a [`SourceProvider`], its document row
//! types ([`DocumentRow`]/[`SearchableRow`]), and an artifact key closure.

pub mod apply;
pub mod artifacts;
#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "cli")]
pub mod cli;
pub mod config;
pub mod derived;
pub mod embed;
pub mod hybrid;
pub mod index;
pub mod ingest;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod model;
pub mod progress;
pub mod provider;
pub mod raw;
pub mod row;
pub mod search;
pub mod source;
pub mod sql;

pub use apply::ApplyStats;
pub use config::CoreConfig;
pub use derived::build_raw_and_text;
pub use embed::Embedder;
pub use hybrid::{HybridIndex, IndexStats};
pub use index::Index;
pub use ingest::{MetadataStats, build_metadata};
pub use model::{Artifact, ColumnInfo, HybridHit, RawArtifact, SearchHit, SqlQueryResult, TableSchema};
pub use provider::{DocumentsContract, SourcePage, SourceProvider};
pub use row::{DocumentRow, SearchableRow};
pub use search::SearchParams;
pub use source::{
    ArtifactReader, ArtifactSource, ArtifactStore, ObjectArtifactStore, build_object_store,
    fanout_key,
};
