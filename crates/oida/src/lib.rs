//! OIDA domain layer over the [`corpus_index`] framework.
//!
//! Owns everything OIDA-specific: the Solr [`SourceProvider`], the document
//! schema and its row decode, the relationship-graph queries, and the config
//! slices. The corpus-agnostic engine (hybrid search, artifact store, SQL,
//! ingest/apply drivers) is inherited from [`corpus_index`] and re-exported
//! here, so the binaries compose against a single `oida::` surface.

pub mod config;
pub mod model;
pub mod queries;
pub mod solr;
pub mod solr_map;
pub mod solr_provider;
pub mod update;

// Domain public API.
pub use config::{ChatConfig, OidaConfig, SolrConfig};
pub use model::{Document, DocumentSummary, GraphEdge, RelatedGraph, RelationKind};
pub use queries::CorpusQueries;
pub use solr_provider::SolrProvider;

// Re-export the framework modules and types the binaries compose with, so they
// depend only on `oida`.
pub use corpus_index::{apply, artifacts, hybrid, ingest, raw};
pub use corpus_index::{
    Artifact, ArtifactReader, ArtifactSource, ArtifactStore, ColumnInfo, CoreConfig,
    DocumentRow, DocumentsContract, Embedder, HybridHit, HybridIndex, Index, MetadataStats,
    ObjectArtifactStore, RawArtifact, SearchHit, SearchParams, SearchableRow, SourcePage,
    SourceProvider, SqlQueryResult, TableSchema, build_metadata, build_object_store,
    build_raw_and_text, fanout_key,
};
pub use corpus_index::apply::ApplyStats;
pub use update::UpdatePlan;
