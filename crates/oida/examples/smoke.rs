//! Manual smoke test for the core index against the real Solr corpus.
//! Run with: `cargo run -p oida --example smoke -- "search terms"`

use oida::{CorpusQueries, Index, OidaConfig, SearchParams};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter("info")
        .init();

    let config = OidaConfig::default();

    if !Index::is_ingested(&config.core).await {
        eprintln!("== ingesting metadata from Solr (one-time) ==");
        let provider = oida::SolrProvider::from_config(&config.solr)?;
        oida::build_metadata(&provider, &config.core, None, false).await?;
    }

    let index = Index::open(&config.core).await?;

    let query = std::env::args().nth(1).unwrap_or_else(|| "report".into());
    eprintln!("== search: {query:?} ==");
    let hits = index
        .search::<oida::DocumentSummary>(&SearchParams {
            query,
            limit: 5,
            ..Default::default()
        })
        .await?;
    for h in &hits {
        let d = &h.document;
        println!(
            "[{}] id={} bn={:?} fields={:?} types={:?}\n      title={:?}",
            h.score, d.id, d.bn, h.matched_fields, d.artifact_types, d.title
        );
    }

    if let Some(first) = hits.first() {
        let target =
            std::env::var("OIDA_RELATED_ID").unwrap_or_else(|_| first.document.id.clone());
        eprintln!("== get_document: {target} ==");
        if let Some(doc) = index.get_document_by_id(&target).await? {
            println!(
                "doc id={} bn={:?} authors={:?} attachments={:?} conversation={:?}",
                doc.id, doc.bn, doc.authors, doc.attachments, doc.conversation
            );
            for a in index.get_artifacts(&doc.id).await? {
                println!(
                    "  artifact {} [{:?}] {} bytes",
                    a.name,
                    a.media_type,
                    a.size.unwrap_or(0)
                );
            }
        }

        eprintln!("== related (depth 1): {target} ==");
        let edges = index.related(&target, 1).await?;
        for e in edges.iter().take(10) {
            println!(
                "  {} -> {} ({}) neighbor={:?}",
                e.from_id,
                e.reference,
                e.kind.as_str(),
                e.neighbor.as_ref().map(|n| n.id.clone())
            );
        }
        println!("  ({} edges total)", edges.len());
    }

    Ok(())
}
