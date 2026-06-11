//! Relationship-graph traversal over the document network.
//!
//! Documents reference one another by Bates number (`bn`) through the
//! `attachments`, `related`, and `mentions` fields, and share an email thread
//! through `conversation`. [`Index::related`] resolves those references into
//! concrete neighbor documents using a breadth-first walk with a cycle guard.
//! Each hop is an indexed lookup against the LanceDB `documents` table.

use std::collections::{HashSet, VecDeque};

use crate::index::Index;
use crate::model::{Document, RelatedEdge, RelationKind};

impl Index {
    /// Resolve the documents connected to `start` (an `id` or Bates `bn`).
    ///
    /// `depth` controls how many hops to traverse (1 = direct neighbors).
    pub async fn related(&self, start: &str, depth: u32) -> anyhow::Result<Vec<RelatedEdge>> {
        let max_depth = depth.max(1);

        let Some(root) = self.resolve(start).await? else {
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

    /// Resolve a starting key that may be either a document `id` or a `bn`.
    async fn resolve(&self, key: &str) -> anyhow::Result<Option<Document>> {
        if let Some(doc) = self.get_document_by_id(key).await? {
            return Ok(Some(doc));
        }
        self.get_document_by_bn(key).await
    }
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
