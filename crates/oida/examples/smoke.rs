//! Manual smoke test for the core index against the real Solr corpus.
//! Run with: `cargo run -p oida --example smoke -- "search terms"`
//!
//! Exercises three read paths end to end:
//!   1. metadata search + graph traversal (always on);
//!   2. hybrid keyword+semantic search over document *contents*
//!      (needs `oida-cli ingest --full-text`);
//!   3. raw artifact extraction — text via the tiered [`ArtifactReader`] and
//!      original bytes via the `raw_artifacts` point lookup (needs an artifact
//!      source configured, and `--store-raw` for the raw tier / binaries).

use std::sync::Arc;

use oida::{
    ArtifactReader, ArtifactSource, ArtifactStore, CorpusQueries, HybridIndex, Index, OidaConfig,
    SearchParams,
};

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
            query: query.clone(),
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
                "doc id={} bn={:?} authors={:?} conversation={:?}\n  attachments={:?}\n  related={:?}\n  mentions={:?}",
                doc.id, doc.bn, doc.authors, doc.conversation,
                doc.attachments, doc.related, doc.mentions,
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
        let graph = index.related(&target, 1).await?;
        for e in graph.edges.iter().take(10) {
            println!(
                "  {} -> {} ({}) neighbor={:?}",
                e.from_id,
                e.reference,
                e.kind.as_str(),
                e.neighbor_id,
            );
        }
        println!("  ({} edges, {} nodes)", graph.edges.len(), graph.nodes.len());
    }

    // == hybrid search over document contents ==
    // Degrades gracefully when the full-text index has not been built, exactly
    // as the MCP server does.
    match HybridIndex::open(&config.core).await {
        Ok(hybrid) => {
            let stats = hybrid.stats().await?;
            eprintln!(
                "== hybrid search: {query:?} (index: {} docs, {} chunks, model={}) ==",
                stats.documents, stats.chunks, stats.embed_model,
            );
            for hit in hybrid.query(&query, 5).await? {
                println!(
                    "[{:.4}] {} <{}>\n      {}",
                    hit.score, hit.doc_id, hit.artifact_name, hit.snippet,
                );
            }
        }
        Err(e) => {
            eprintln!(
                "== hybrid search unavailable ({e}); build it with \
                 `oida-cli ingest --full-text` =="
            );
        }
    }

    // == raw artifact extraction ==
    // Build the same tiered resolver the server uses: the `raw_artifacts`
    // LanceDB blob table first, then the original artifact source (local dir or
    // S3). Then pull one text artifact's text and one artifact's raw bytes for
    // the first search hit's document.
    let raw_table = index.open_raw_table().await.unwrap_or(None);
    let fallback = match ArtifactSource::from_config(&config.core) {
        Ok(s) => s.map(|s| Arc::new(s) as Arc<dyn ArtifactStore>),
        Err(e) => {
            eprintln!("artifact source unavailable ({e}); extraction degraded");
            None
        }
    };
    let reader = ArtifactReader::new(raw_table, fallback);

    if let Some(first) = hits.first() {
        let doc_id = &first.document.id;
        let artifacts = index.get_artifacts(doc_id).await?;
        eprintln!(
            "== artifacts for {doc_id} (reader configured: {}) ==",
            reader.is_configured()
        );

        // Text extraction through the tiered reader (text/plain or `.ocr`).
        if let Some(text_art) = artifacts
            .iter()
            .find(|a| a.media_type.as_deref() == Some("text/plain") || a.name.ends_with(".ocr"))
        {
            let out = oida::artifacts::read_artifact_text(
                Some(&reader),
                doc_id,
                &text_art.name,
                text_art.media_type.as_deref(),
                0,
                400,
            )
            .await;
            let preview = out.text.as_deref().unwrap_or("").replace('\n', " ");
            println!(
                "  text {} [{:?}] status={:?} tier={:?} {} of {:?} bytes\n    {:?}",
                out.name,
                out.media_type,
                out.status,
                out.source,
                out.returned_bytes,
                out.total_bytes,
                preview,
            );
        } else {
            println!("  (no text/plain or .ocr artifact on {doc_id})");
        }

        // Raw bytes via the `(id, name)` point lookup against `raw_artifacts`.
        if let Some(art) = artifacts.first() {
            match index.get_raw_artifact(doc_id, &art.name).await? {
                Some(raw) => println!(
                    "  raw {} [{:?}] md5={:?} {} bytes stored",
                    raw.name,
                    raw.media_type,
                    raw.md5,
                    raw.data.len(),
                ),
                None => println!(
                    "  raw bytes for {} not stored (run `oida-cli ingest --store-raw`)",
                    art.name
                ),
            }
        }
    }

    Ok(())
}
