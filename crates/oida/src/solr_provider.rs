//! The OIDA Solr [`SourceProvider`] implementation.
//!
//! Wraps the cursor-paged [`SolrClient`] and the [`crate::solr_map`] mapping
//! behind the framework's ingest boundary: each Solr page becomes a
//! [`SourcePage`] of `documents`/`artifacts` Arrow batches, its per-document
//! policy-withdrawal flags, and its watermark.

use anyhow::Result;
use futures::Stream;

use corpus_index::index::track_max;
use corpus_index::{DocumentsContract, SourcePage, SourceProvider};

use crate::config::SolrConfig;
use crate::solr::{CURSOR_START, SolrClient};
use crate::solr_map;

/// The OIDA Solr source provider.
pub struct SolrProvider {
    client: SolrClient,
    modified_field: String,
    contract: DocumentsContract,
}

impl SolrProvider {
    /// Build the provider from config, erroring clearly when `solr_url` is unset.
    pub fn from_config(config: &SolrConfig) -> Result<Self> {
        let client = crate::update::solr_client(config)?;
        let modified_field = config.solr_modified_field.clone();
        // Derive the exact `documents` schema from the mapper by building an
        // empty batch — one source of truth for the schema.
        let schema = solr_map::documents_batch(&[], &modified_field)?.schema();
        let contract = DocumentsContract {
            schema,
            fts_column: "search_text",
            scalar_index_cols: &["id", "bn", "conversation"],
        };
        Ok(Self {
            client,
            modified_field,
            contract,
        })
    }
}

impl SourceProvider for SolrProvider {
    fn documents_contract(&self) -> &DocumentsContract {
        &self.contract
    }

    fn scan(&self, since: Option<&str>) -> impl Stream<Item = Result<SourcePage>> + Send {
        // Clone the paging state into a 'static stream so the driver can pull it
        // in a loop without borrowing the provider.
        let client = self.client.clone();
        let modified_field = self.modified_field.clone();
        let since = since.map(str::to_string);
        futures::stream::try_unfold(Some(CURSOR_START.to_string()), move |state| {
            let client = client.clone();
            let modified_field = modified_field.clone();
            let since = since.clone();
            async move {
                let Some(cursor) = state else {
                    return Ok(None);
                };
                let page = client
                    .scan_page(since.as_deref(), &cursor, solr_map::SOURCE_FIELDS)
                    .await?;
                if page.docs.is_empty() {
                    return Ok(None);
                }
                let documents = solr_map::documents_batch(&page.docs, &modified_field)?;
                let artifacts = solr_map::artifacts_batch(&page.docs)?;
                let redacted = page.docs.iter().map(solr_map::is_redacted_policy).collect();
                let mut watermark = None;
                for doc in &page.docs {
                    if let Some(m) = solr_map::doc_modified(doc, &modified_field) {
                        track_max(&mut watermark, m);
                    }
                }
                // Solr signals exhaustion by returning the cursor it was given.
                let next = if page.next_cursor.is_empty() || page.next_cursor == cursor {
                    None
                } else {
                    Some(page.next_cursor)
                };
                let source_page = SourcePage {
                    documents,
                    artifacts,
                    redacted,
                    watermark,
                    num_found: page.num_found,
                };
                Ok(Some((source_page, next)))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_declares_id_and_fts_column() {
        let cfg = SolrConfig {
            solr_url: Some("http://localhost:8983/solr/ltdl3".into()),
            ..Default::default()
        };
        let provider = SolrProvider::from_config(&cfg).unwrap();
        let contract = provider.documents_contract();
        assert!(
            contract.schema.field_with_name("id").is_ok(),
            "contract schema must contain `id`"
        );
        assert_eq!(contract.fts_column, "search_text");
        assert!(contract.scalar_index_cols.contains(&"id"));
    }
}
