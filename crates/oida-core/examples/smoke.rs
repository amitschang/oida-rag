//! Manual smoke test for the core index against the real parquet.
//! Run with: `cargo run -p oida-core --example smoke -- "search terms"`

use oida_core::{Config, Index, SearchParams};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter("info")
        .init();

    let config = Config::default();

    if !config.cache_path.exists() {
        eprintln!("== building cache (one-time) ==");
        Index::build_cache(&config, false)?;
    }

    let index = Index::open(&config)?;

    let query = std::env::args().nth(1).unwrap_or_else(|| "report".into());
    eprintln!("== search: {query:?} ==");
    let hits = index.search(&SearchParams {
        query,
        limit: 5,
        ..Default::default()
    })?;
    for h in &hits {
        println!(
            "[{}] id={} bn={:?} fields={:?} types={:?}\n      title={:?}",
            h.score, h.id, h.bn, h.matched_fields, h.artifact_types, h.title
        );
    }

    if let Some(first) = hits.first() {
        let target = std::env::var("OIDA_RELATED_ID").unwrap_or_else(|_| first.id.clone());
        eprintln!("== get_document: {target} ==");
        if let Some(doc) = index.get_document_by_id(&target)? {
            println!(
                "doc id={} bn={:?} authors={:?} attachments={:?} conversation={:?}",
                doc.id, doc.bn, doc.authors, doc.attachments, doc.conversation
            );
            for a in index.get_artifacts(&doc.id)? {
                println!(
                    "  artifact {} [{:?}] {} bytes",
                    a.name,
                    a.media_type,
                    a.size.unwrap_or(0)
                );
            }
        }

        eprintln!("== related (depth 1): {target} ==");
        let edges = index.related(&target, 1)?;
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
