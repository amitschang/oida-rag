//! Optional CLI helpers (`feature = "cli"`) for building a corpus command line.
//!
//! Corpus-agnostic operations over the engine — construct the embedder, build
//! the derived stores, and report index statistics — so a corpus binary is
//! mostly argument parsing and provider wiring. Output is deliberately neutral
//! (no binary or corpus name baked in).

use anyhow::{Context, Result};

use crate::config::CoreConfig;
use crate::embed::Embedder;
use crate::hybrid::{self, HybridIndex};
use crate::index::Index;
use crate::raw;

/// Construct the embedding client from the configured host, model, and key.
pub fn make_embedder(config: &CoreConfig) -> Result<Embedder> {
    Embedder::new(
        &config.embed_host,
        config.embed_model.clone(),
        config.embed_api_key.clone(),
    )
}

/// Build the requested derived stores. When both raw storage and the full-text
/// index are requested they run concurrently on a single shared status line
/// (raw is network/disk bound, full-text is GPU bound, so overlapping them
/// keeps both busy); otherwise each runs on its own.
pub async fn build_derived(
    config: &CoreConfig,
    index: &Index,
    do_raw: bool,
    do_text: bool,
    force: bool,
    resume: bool,
) -> Result<()> {
    match (do_raw, do_text) {
        (true, true) => {
            let embedder = make_embedder(config)?;
            eprintln!(
                "Building raw store + full-text index concurrently (embed model '{}')…",
                config.embed_model
            );
            let (rstats, hstats) =
                crate::build_raw_and_text(config, index, &embedder, force, resume).await?;
            eprintln!(
                "Stored {} raw artifacts ({} already present, {} missing).",
                rstats.stored, rstats.skipped, rstats.missing
            );
            eprintln!(
                "Indexed {} chunks across {} documents (dim {}).",
                hstats.chunks, hstats.documents, hstats.dim
            );
        }
        (true, false) => {
            eprintln!("Storing raw (non-text) artifacts…");
            let rstats = raw::build(config, index, resume).await?;
            eprintln!(
                "Stored {} raw artifacts ({} already present, {} missing).",
                rstats.stored, rstats.skipped, rstats.missing
            );
        }
        (false, true) => {
            let embedder = make_embedder(config)?;
            eprintln!(
                "Building hybrid text index with embed model '{}'…",
                config.embed_model
            );
            let hstats = hybrid::build(config, index, &embedder, force, resume).await?;
            eprintln!(
                "Indexed {} chunks across {} documents (dim {}).",
                hstats.chunks, hstats.documents, hstats.dim
            );
        }
        (false, false) => {}
    }
    Ok(())
}

/// Report index row counts, artifact byte sizes, and hybrid metadata.
pub async fn print_stats(config: &CoreConfig) -> Result<()> {
    let index = Index::open(config).await.context("opening index")?;
    let (documents, artifacts) = index.counts().await?;
    println!("Index ({})", config.lance_path.display());
    println!("  documents:      {documents}");
    println!("  artifacts:      {artifacts}");

    let sizes = index.store_sizes().await.context("summarising artifact sizes")?;
    println!("  full-text artifacts:");
    println!(
        "    referenced:   {} ({})",
        sizes.text_logical_count,
        human_bytes(sizes.text_logical_bytes)
    );
    println!(
        "    in archive:   {} ({})",
        sizes.text_real_count,
        human_bytes(sizes.text_real_bytes)
    );
    println!("  raw artifacts:");
    println!(
        "    referenced:   {} ({})",
        sizes.raw_logical_count,
        human_bytes(sizes.raw_logical_bytes)
    );
    match (sizes.raw_real_count, sizes.raw_real_bytes) {
        (Some(count), Some(bytes)) => {
            println!("    in archive:   {} ({})", count, human_bytes(bytes));
        }
        _ => println!("    in archive:   not stored (run the raw-artifact ingest)"),
    }

    match HybridIndex::open(config).await {
        Ok(h) => {
            let s = h.stats().await?;
            println!("  full-text:      built");
            println!("    chunks:       {}", s.chunks);
            println!("    embed model:  {}", s.embed_model);
            println!("    vector dim:   {}", s.dim);
            println!("    chunk bytes:  {}", s.chunk_bytes);
            println!("    chunk overlap:{}", s.chunk_overlap);
            println!("    built at:     {} (unix)", s.built_at);
        }
        Err(_) => println!("  full-text:      not built (run the full-text ingest)"),
    }
    Ok(())
}

/// Format a byte count as a human-readable size using binary (1024) units.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
