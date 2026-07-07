//! Find documents with non-empty relationship fields.
//! Run with: `cargo run -p oida --example find-related`
//!
//! Optional env vars:
//!   FIELD=related|mentions|attachments|conversation  (default: related)
//!   LIMIT=N                                          (default: 10)

use oida::{CorpusQueries, Index, OidaConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = OidaConfig::default();
    let index = Index::open(&config.core).await?;

    // Sample a few docs unconditionally to see raw field contents.
    eprintln!("== sample docs (up to 5) ==");
    let sample = index.documents_where::<oida::Document>("id IS NOT NULL", Some(5)).await?;
    for doc in &sample {
        println!(
            "id={} bn={:?} attachments={:?} related={:?} mentions={:?} conversation={:?}",
            doc.id, doc.bn, doc.attachments, doc.related, doc.mentions, doc.conversation,
        );
    }

    // Check which relationship fields have any populated entries at all.
    eprintln!("\n== field population check ==");
    for (label, filter) in [
        ("attachments", "cardinality(attachments) > 0"),
        ("related",     "cardinality(related) > 0"),
        ("mentions",    "cardinality(mentions) > 0"),
        ("conversation","conversation IS NOT NULL"),
    ] {
        let found = index.documents_where::<oida::Document>(filter, Some(1)).await?;
        println!("  {label}: {}", if found.is_empty() { "none" } else { "has entries" });
    }

    let field = std::env::var("FIELD").unwrap_or_else(|_| "related".into());
    let limit: usize = std::env::var("LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let filter = match field.as_str() {
        "conversation" => format!("{field} IS NOT NULL"),
        _ => format!("cardinality({field}) > 0"),
    };

    eprintln!("\n== documents where {filter} (limit {limit}) ==");

    let docs = index
        .documents_where::<oida::Document>(&filter, Some(limit))
        .await?;

    if docs.is_empty() {
        println!("none found");
        return Ok(());
    }

    for doc in &docs {
        println!(
            "id={} bn={:?} attachments={:?} related={:?} mentions={:?} conversation={:?}",
            doc.id, doc.bn, doc.attachments, doc.related, doc.mentions, doc.conversation,
        );
    }

    // Follow through with the graph for the first hit.
    let first = &docs[0];
    eprintln!("\n== related graph (depth 1) for {} ==", first.id);
    let graph = index.related(&first.id, 1).await?;
    for e in &graph.edges {
        println!(
            "  {} -> {} ({}) neighbor={:?}",
            e.from_id, e.reference, e.kind.as_str(), e.neighbor_id,
        );
    }
    println!("({} edges, {} nodes)", graph.edges.len(), graph.nodes.len());

    Ok(())
}
