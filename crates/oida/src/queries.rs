//! OIDA domain queries and relationship-graph traversal.
//!
//! These name corpus concepts — Bates numbers (`bn`), conversation threads, and
//! the reference graph (`attachments`/`related`/`mentions`) — that have no
//! corpus-independent meaning, so they live in the domain layer as an *extension
//! trait* on the framework [`Index`] rather than as methods on it. Each is built
//! from the framework's generic query primitives ([`Index::documents_where`],
//! [`Index::get`]).

use std::collections::{HashSet, VecDeque};

use anyhow::Result;

use corpus_index::Index;
use corpus_index::index::sql_str;

use crate::model::{Document, RelatedEdge, RelationKind};

/// Document lookups and graph traversal specific to the OIDA corpus.
#[allow(async_fn_in_trait)]
pub trait CorpusQueries {
    /// Fetch a single document by its `id`.
    async fn get_document_by_id(&self, id: &str) -> Result<Option<Document>>;
    /// Fetch a single document by its Bates number `bn`.
    async fn get_document_by_bn(&self, bn: &str) -> Result<Option<Document>>;
    /// Look up documents by a set of Bates numbers.
    async fn get_documents_by_bns(&self, bns: &[String]) -> Result<Vec<Document>>;
    /// Look up documents sharing a conversation thread, excluding one id.
    async fn get_documents_by_conversation(
        &self,
        conversation: &str,
        exclude_id: &str,
    ) -> Result<Vec<Document>>;
    /// Resolve the documents connected to `start` (an `id` or Bates `bn`) by a
    /// breadth-first walk over the reference graph, up to `depth` hops.
    async fn related(&self, start: &str, depth: u32) -> Result<Vec<RelatedEdge>>;
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

    async fn get_documents_by_conversation(
        &self,
        conversation: &str,
        exclude_id: &str,
    ) -> Result<Vec<Document>> {
        self.documents_where::<Document>(
            &format!(
                "conversation = {} AND id != {}",
                sql_str(conversation),
                sql_str(exclude_id)
            ),
            None,
        )
        .await
    }

    async fn related(&self, start: &str, depth: u32) -> Result<Vec<RelatedEdge>> {
        let max_depth = depth.max(1);

        let Some(root) = resolve(self, start).await? else {
            return Ok(Vec::new());
        };

        let mut edges = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(root.id.clone());

        let mut queue: VecDeque<(Document, u32)> = VecDeque::new();
        queue.push_back((root, 0));

        while let Some((doc, d)) = queue.pop_front() {
            if d >= max_depth {
                continue;
            }
            let next_depth = d + 1;

            // Bates-number references grouped by relation kind.
            let bn_groups: [(RelationKind, &Vec<String>); 3] = [
                (RelationKind::Attachment, &doc.attachments),
                (RelationKind::Related, &doc.related),
                (RelationKind::Mention, &doc.mentions),
            ];
            for (kind, refs) in bn_groups {
                let resolved = self.get_documents_by_bns(refs).await?;
                for reference in refs {
                    let neighbor = resolved
                        .iter()
                        .find(|n| n.bn.as_deref() == Some(reference.as_str()))
                        .cloned();
                    push_edge(
                        &mut edges,
                        &mut visited,
                        &mut queue,
                        &doc.id,
                        kind,
                        reference.clone(),
                        neighbor,
                        next_depth,
                    );
                }
            }

            // Conversation/thread siblings.
            if let Some(conv) = doc.conversation.clone() {
                let siblings = self.get_documents_by_conversation(&conv, &doc.id).await?;
                for sib in siblings {
                    push_edge(
                        &mut edges,
                        &mut visited,
                        &mut queue,
                        &doc.id,
                        RelationKind::Conversation,
                        conv.clone(),
                        Some(sib),
                        next_depth,
                    );
                }
            }
        }

        Ok(edges)
    }
}

/// Resolve a starting key that may be either a document `id` or a `bn`.
async fn resolve(index: &Index, key: &str) -> Result<Option<Document>> {
    if let Some(doc) = index.get_document_by_id(key).await? {
        return Ok(Some(doc));
    }
    index.get_document_by_bn(key).await
}

/// Record an edge and enqueue an unvisited neighbor for further traversal.
#[allow(clippy::too_many_arguments)]
fn push_edge(
    edges: &mut Vec<RelatedEdge>,
    visited: &mut HashSet<String>,
    queue: &mut VecDeque<(Document, u32)>,
    from_id: &str,
    kind: RelationKind,
    reference: String,
    neighbor: Option<Document>,
    depth: u32,
) {
    if let Some(n) = &neighbor
        && visited.insert(n.id.clone())
    {
        queue.push_back((n.clone(), depth));
    }
    edges.push(RelatedEdge {
        from_id: from_id.to_string(),
        kind,
        reference,
        neighbor,
        depth,
    });
}
