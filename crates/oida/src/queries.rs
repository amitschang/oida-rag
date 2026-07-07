//! OIDA domain queries and relationship-graph traversal.
//!
//! These name corpus concepts — Bates numbers (`bn`), conversation threads, and
//! the reference graph (`attachments`/`related`/`mentions`) — that have no
//! corpus-independent meaning, so they live in the domain layer as an *extension
//! trait* on the framework [`Index`] rather than as methods on it. Each is built
//! from the framework's generic query primitives ([`Index::documents_where`],
//! [`Index::get`]).

use std::collections::{HashMap, HashSet};

use anyhow::Result;

use corpus_index::Index;
use corpus_index::index::sql_str;

use crate::model::{Document, GraphEdge, RelatedGraph, RelationKind};

/// Document lookups and graph traversal specific to the OIDA corpus.
#[allow(async_fn_in_trait)]
pub trait CorpusQueries {
    /// Fetch a single document by its `id`.
    async fn get_document_by_id(&self, id: &str) -> Result<Option<Document>>;
    /// Fetch a single document by its Bates number `bn`.
    async fn get_document_by_bn(&self, bn: &str) -> Result<Option<Document>>;
    /// Look up documents by a set of Bates numbers.
    async fn get_documents_by_bns(&self, bns: &[String]) -> Result<Vec<Document>>;
    /// Look up documents sharing any of the given conversation threads.
    async fn get_documents_by_conversations(
        &self,
        conversations: &[String],
    ) -> Result<Vec<Document>>;
    /// Resolve the documents connected to `start` (an `id` or Bates `bn`) by a
    /// breadth-first walk over the reference graph, up to `depth` hops.
    async fn related(&self, start: &str, depth: u32) -> Result<RelatedGraph>;
}

impl CorpusQueries for Index {
    async fn get_document_by_id(&self, id: &str) -> Result<Option<Document>> {
        Ok(self
            .documents_where::<Document>(&format!("id = {}", sql_str(id)), Some(1))
            .await?
            .into_iter()
            .next())
    }

    async fn get_document_by_bn(&self, bn: &str) -> Result<Option<Document>> {
        Ok(self
            .documents_where::<Document>(&format!("bn = {}", sql_str(bn)), Some(1))
            .await?
            .into_iter()
            .next())
    }

    async fn get_documents_by_bns(&self, bns: &[String]) -> Result<Vec<Document>> {
        if bns.is_empty() {
            return Ok(Vec::new());
        }
        let list = bns.iter().map(|b| sql_str(b)).collect::<Vec<_>>().join(", ");
        self.documents_where::<Document>(&format!("bn IN ({list})"), None)
            .await
    }

    async fn get_documents_by_conversations(
        &self,
        conversations: &[String],
    ) -> Result<Vec<Document>> {
        if conversations.is_empty() {
            return Ok(Vec::new());
        }
        let list = conversations
            .iter()
            .map(|c| sql_str(c))
            .collect::<Vec<_>>()
            .join(", ");
        self.documents_where::<Document>(&format!("conversation IN ({list})"), None)
            .await
    }

    async fn related(&self, start: &str, depth: u32) -> Result<RelatedGraph> {
        let max_depth = depth.max(1);

        let Some(root) = resolve(self, start).await? else {
            return Ok(RelatedGraph { nodes: HashMap::new(), edges: Vec::new() });
        };

        let mut edges: Vec<GraphEdge> = Vec::new();
        let mut nodes: HashMap<String, Document> = HashMap::new();
        let mut visited: HashSet<String> = HashSet::new();

        visited.insert(root.id.clone());
        nodes.insert(root.id.clone(), root.clone());
        let mut frontier: Vec<Document> = vec![root];

        for next_depth in 1..=max_depth {
            // Collect all BN refs from the whole frontier: (from_id, kind, bn).
            let mut bn_refs: Vec<(String, RelationKind, String)> = Vec::new();
            // Collect conversation IDs: conv → [from_id, ...]
            let mut conv_from: Vec<(String, String)> = Vec::new();

            for doc in &frontier {
                let bn_groups: [(RelationKind, &Vec<String>); 3] = [
                    (RelationKind::Attachment, &doc.attachments),
                    (RelationKind::Related, &doc.related),
                    (RelationKind::Mention, &doc.mentions),
                ];
                for (kind, refs) in bn_groups {
                    for bn in refs {
                        bn_refs.push((doc.id.clone(), kind, bn.clone()));
                    }
                }
                if let Some(conv) = &doc.conversation {
                    conv_from.push((conv.clone(), doc.id.clone()));
                }
            }

            let mut next_frontier: Vec<Document> = Vec::new();

            // One query for all BN references across the frontier.
            // Insert resolved docs into the node map; build a bn→id index for edge assembly.
            let unique_bns: Vec<String> = {
                let mut seen = HashSet::new();
                bn_refs
                    .iter()
                    .filter_map(|(_, _, bn)| seen.insert(bn.clone()).then(|| bn.clone()))
                    .collect()
            };
            let mut bn_to_id: HashMap<String, String> = HashMap::new();
            for doc in self.get_documents_by_bns(&unique_bns).await? {
                if let Some(bn) = doc.bn.clone() {
                    bn_to_id.insert(bn, doc.id.clone());
                }
                nodes.entry(doc.id.clone()).or_insert(doc);
            }

            for (from_id, kind, bn) in &bn_refs {
                let neighbor_id = bn_to_id.get(bn.as_str()).cloned();
                push_edge(
                    &mut edges,
                    &mut visited,
                    &mut next_frontier,
                    &nodes,
                    from_id,
                    *kind,
                    bn.clone(),
                    neighbor_id,
                    next_depth,
                );
            }

            // One query for all conversation siblings across the frontier.
            // Same pattern: insert into nodes, build conv→[id] for edge assembly.
            if !conv_from.is_empty() {
                let unique_convs: Vec<String> = {
                    let mut seen = HashSet::new();
                    conv_from
                        .iter()
                        .filter_map(|(c, _)| seen.insert(c.clone()).then(|| c.clone()))
                        .collect()
                };
                let mut by_conv: HashMap<String, Vec<String>> = HashMap::new();
                for sib in self.get_documents_by_conversations(&unique_convs).await? {
                    if let Some(c) = sib.conversation.clone() {
                        by_conv.entry(c).or_default().push(sib.id.clone());
                    }
                    nodes.entry(sib.id.clone()).or_insert(sib);
                }
                for (conv, from_id) in &conv_from {
                    for sib_id in by_conv.get(conv.as_str()).into_iter().flatten() {
                        push_edge(
                            &mut edges,
                            &mut visited,
                            &mut next_frontier,
                            &nodes,
                            from_id,
                            RelationKind::Conversation,
                            conv.clone(),
                            Some(sib_id.clone()),
                            next_depth,
                        );
                    }
                }
            }

            frontier = next_frontier;
        }

        Ok(RelatedGraph { nodes, edges })
    }
}

/// Resolve a starting key that may be either a document `id` or a `bn`.
async fn resolve(index: &Index, key: &str) -> Result<Option<Document>> {
    if let Some(doc) = index.get_document_by_id(key).await? {
        return Ok(Some(doc));
    }
    index.get_document_by_bn(key).await
}

/// Record an edge and add an unvisited neighbor to the next frontier.
#[allow(clippy::too_many_arguments)]
fn push_edge(
    edges: &mut Vec<GraphEdge>,
    visited: &mut HashSet<String>,
    next_frontier: &mut Vec<Document>,
    nodes: &HashMap<String, Document>,
    from_id: &str,
    kind: RelationKind,
    reference: String,
    neighbor_id: Option<String>,
    depth: u32,
) {
    if let Some(id) = &neighbor_id
        && visited.insert(id.clone())
    {
        if let Some(doc) = nodes.get(id) {
            next_frontier.push(doc.clone());
        }
    }
    edges.push(GraphEdge {
        from_id: from_id.to_string(),
        kind,
        reference,
        neighbor_id,
        depth,
    });
}
