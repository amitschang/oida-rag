//! Concurrent build of the two derived stores: the raw-artifact table and the
//! full-text index.
//!
//! The two passes bottleneck on disjoint resources — raw storage on
//! network/disk download (every non-text blob), the full-text build on GPU
//! embedding (its text fetches are tiny by comparison). Running them
//! sequentially leaves one of those resources idle for the duration of the
//! other pass; overlapping them keeps both busy, so wall-clock collapses toward
//! the slower of the two rather than their sum. They write to different tables
//! (`raw_artifacts` vs `chunks`/`_meta`), so there is no data dependency forcing
//! an order.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::oneshot;

use crate::config::Config;
use crate::embed::Embedder;
use crate::hybrid::{self, IndexStats, Progress};
use crate::index::Index;
use crate::raw::{self, RawProgress, RawStats};

/// Build the raw-artifact store and the full-text index concurrently, sharing a
/// single live status line.
///
/// `force`/`resume` carry the same meaning as for the individual builds
/// ([`raw::build`] and [`hybrid::build`]) and are forwarded to both passes. If
/// either pass fails the other is cancelled (both are resumable, so a re-run
/// picks up where it stopped).
pub async fn build_raw_and_text(
    config: &Config,
    index: &Index,
    embedder: &Embedder,
    force: bool,
    resume: bool,
) -> Result<(RawStats, IndexStats)> {
    let progress = Arc::new(Progress::default());
    let raw_progress = Arc::new(RawProgress::default());

    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let ticker = tokio::spawn(hybrid::run_ticker(
        progress.clone(),
        Some(raw_progress.clone()),
        stop_rx,
    ));

    let raw_fut = raw::build_with_progress(config, index, resume, Some(raw_progress.as_ref()));
    let text_fut =
        hybrid::build_with_progress(config, index, embedder, force, resume, progress.clone());

    let result = tokio::try_join!(raw_fut, text_fut);

    // Stop the shared ticker and let it print its final line regardless of
    // whether the builds succeeded.
    let _ = stop_tx.send(());
    let _ = ticker.await;

    result
}
