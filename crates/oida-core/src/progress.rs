//! Live build-progress reporting shared by the ingest and derived-store builds.
//!
//! The metadata ingest, raw-artifact storage, and full-text index builds all
//! report progress through this module so their status lines look and behave
//! the same. Counting (the hot path in each pipeline stage) is decoupled from
//! rendering (a timer-driven ticker) through lock-free atomic counters, so the
//! display cadence never couples to the work cadence.
//!
//! On a TTY each active pass renders its own [`indicatif`] bar inside a shared
//! [`MultiProgress`]; off a TTY (pod logs, pipes) `indicatif` hides the bars, so
//! the ticker emits a periodic single-line `tracing` log instead. Any
//! combination of the raw and full-text passes can be driven at once, so the
//! standalone raw build, the standalone full-text build, and the concurrent
//! raw + full-text build all share one ticker.

use std::collections::VecDeque;
use std::io::IsTerminal;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::sync::oneshot;

/// Sliding window over which the ticker's "recent" rates are averaged. A
/// per-tick instantaneous rate is avoided because batches complete in bursts, so
/// most ticks would otherwise read zero.
const RATE_WINDOW: Duration = Duration::from_secs(60);

/// Live full-text build counters, updated lock-free by the pipeline stages and
/// sampled by [`run_ticker`]. Decoupling counting (hot path) from rendering
/// (timer) keeps the display cadence independent of the work cadence.
#[derive(Default)]
pub(crate) struct FullTextProgress {
    /// Artifacts the reader has walked (whether read, skipped, or missing).
    pub(crate) scanned: AtomicU64,
    /// Total text artifacts to index, the progress denominator (0 until the
    /// reader has been handed the artifact list).
    pub(crate) text_total: AtomicU64,
    /// Chunks embedded so far (includes any seeded from a resume).
    pub(crate) chunks: AtomicU64,
    /// Bytes of chunk text embedded so far, for a throughput/token estimate.
    pub(crate) text_bytes: AtomicU64,
    /// Referenced artifact files that were not on disk.
    pub(crate) missing: AtomicU64,
    /// Live pipeline gauges — instantaneous depths, not cumulative totals — to
    /// locate the bottleneck. The stage where work piles up is downstream of the
    /// stall; the stage that runs below its limit is being starved by it.
    ///
    /// Read tasks currently executing on the blocking pool. Pinned near
    /// `read_concurrency` when reads keep up; collapsing toward 1 despite a high
    /// limit is the signature of head-of-line blocking in the order-preserving
    /// reader (one slow artifact stalls every read queued behind it).
    pub(crate) reads_inflight: AtomicU64,
    /// Read+chunked jobs waiting in the channel for a free embed slot. Near the
    /// channel cap ⇒ the embedder is the bottleneck; near 0 ⇒ the reader can't
    /// keep the embedder fed.
    pub(crate) jobs_queued: AtomicU64,
    /// Embed requests in flight to the backend. Pinned near `embed_concurrency`
    /// ⇒ the GPUs are saturated; below it ⇒ the embedder is starved upstream.
    pub(crate) embeds_inflight: AtomicU64,
    /// Embedded batches waiting in the channel for the writer. Near the channel
    /// cap ⇒ LanceDB write backpressure is the bottleneck.
    pub(crate) out_queued: AtomicU64,
}

/// Live counters for the raw-storage pass, updated lock-free as artifacts are
/// fetched and sampled by [`run_ticker`]. Mirrors the figures in
/// [`crate::raw::RawStats`] plus the in-flight download count and bytes
/// transferred.
#[derive(Default)]
pub(crate) struct RawProgress {
    /// Candidate artifacts to fetch this run (after skipping already-stored).
    pub(crate) total: AtomicU64,
    /// Artifacts already present and skipped (resume only).
    pub(crate) skipped: AtomicU64,
    /// Artifacts fetched and written so far.
    pub(crate) stored: AtomicU64,
    /// Referenced artifacts whose bytes were absent from the source.
    pub(crate) missing: AtomicU64,
    /// Bytes downloaded so far.
    pub(crate) bytes: AtomicU64,
    /// Fetches currently in flight.
    pub(crate) inflight: AtomicU64,
}

/// Construct the metadata-ingest "documents" bar with the shared style. Updated
/// inline by the ingest scan loop (a simple monotonic counter), so it does not
/// go through the [`run_ticker`] sampling path the way the raw and full-text
/// passes do.
pub(crate) fn documents_bar(total: u64) -> ProgressBar {
    let bar = ProgressBar::new(total);
    bar.set_style(
        ProgressStyle::with_template(
            "{prefix:<9} {bar:28.yellow/blue} {pos:>9}/{len:>9} docs │ {per_sec}",
        )
        .expect("valid template")
        .progress_chars("=>-"),
    );
    bar.set_prefix("documents");
    bar
}

/// Periodically render live build progress until `stop` fires.
///
/// `text` and/or `raw` are supplied for whichever passes are running, so a
/// standalone raw build, a standalone full-text build, and the concurrent
/// raw + full-text build all share this one ticker. The recent rates are
/// sliding-window averages over [`RATE_WINDOW`]; the full-text throughput is
/// measured from the first *embedded* chunk, so a resume's skip/scan phase
/// (which produces no chunks) never enters the rate.
///
/// On a TTY it drives the live bars in place; off a TTY (pod logs, pipes) it
/// emits a periodic single-line `tracing` log. Stops when `stop` fires, printing
/// a final summary.
pub(crate) async fn run_ticker(
    text: Option<Arc<FullTextProgress>>,
    raw: Option<Arc<RawProgress>>,
    mut stop: oneshot::Receiver<()>,
) {
    let tty = std::io::stderr().is_terminal();
    let mut interval =
        tokio::time::interval(Duration::from_millis(if tty { 500 } else { 5000 }));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // On a TTY render each active pass on its own bar inside a shared
    // MultiProgress; off a TTY indicatif hides the bars, so the single-line
    // `tracing` log below keeps pod logs informative.
    let bars = tty.then(|| TickerBars::new(text.is_some(), raw.is_some()));

    let initial_chunks = text.as_ref().map_or(0, |p| p.chunks.load(Ordering::Relaxed));
    // Set on the first tick that observes new chunks — the throughput clock.
    let mut run_start: Option<Instant> = None;
    // Recent (timestamp, chunks, text bytes, raw bytes) samples within the
    // window, for the sliding-window rates.
    let mut samples: VecDeque<(Instant, u64, u64, u64)> = VecDeque::new();

    loop {
        let stopping = tokio::select! {
            _ = interval.tick() => false,
            _ = &mut stop => true,
        };
        let now = Instant::now();

        // Sample each pass's counters once per tick; shared by both render paths.
        let text_vals = text.as_ref().map(|p| TextVals {
            chunks: p.chunks.load(Ordering::Relaxed),
            bytes: p.text_bytes.load(Ordering::Relaxed),
            scanned: p.scanned.load(Ordering::Relaxed),
            missing: p.missing.load(Ordering::Relaxed),
            total_artifacts: p.text_total.load(Ordering::Relaxed),
            reads_inflight: p.reads_inflight.load(Ordering::Relaxed),
            jobs_queued: p.jobs_queued.load(Ordering::Relaxed),
            embeds_inflight: p.embeds_inflight.load(Ordering::Relaxed),
            out_queued: p.out_queued.load(Ordering::Relaxed),
        });
        let raw_vals = raw.as_ref().map(|raw| RawVals {
            stored: raw.stored.load(Ordering::Relaxed),
            total: raw.total.load(Ordering::Relaxed),
            inflight: raw.inflight.load(Ordering::Relaxed),
            bytes: raw.bytes.load(Ordering::Relaxed),
            missing: raw.missing.load(Ordering::Relaxed),
        });

        let chunks = text_vals.as_ref().map_or(0, |v| v.chunks);
        let bytes = text_vals.as_ref().map_or(0, |v| v.bytes);
        let raw_bytes = raw_vals.as_ref().map_or(0, |v| v.bytes);

        if run_start.is_none() && chunks > initial_chunks {
            run_start = Some(now);
        }

        // Sliding-window rates: compare now against the oldest sample still
        // inside the window. This smooths over the bursts that make a per-tick
        // delta read zero most of the time.
        samples.push_back((now, chunks, bytes, raw_bytes));
        while matches!(samples.front(), Some(&(t, ..)) if now.duration_since(t) > RATE_WINDOW) {
            samples.pop_front();
        }
        let (recent_cps, recent_bps, recent_raw_bps) = match samples.front() {
            Some(&(t0, c0, b0, rb0)) if now > t0 => {
                let dt = now.duration_since(t0).as_secs_f64();
                (
                    (chunks - c0) as f64 / dt,
                    (bytes - b0) as f64 / dt,
                    (raw_bytes - rb0) as f64 / dt,
                )
            }
            _ => (0.0, 0.0, 0.0),
        };
        // Cumulative average, measured from the first embedded chunk.
        let avg_cps = run_start.map_or(0.0, |rs| {
            (chunks - initial_chunks) as f64 / now.duration_since(rs).as_secs_f64().max(1e-6)
        });
        // ~4 bytes per token is a rough English heuristic; labelled an estimate.
        let recent_tps = recent_bps / 4.0;

        match &bars {
            // TTY: drive the live progress bars.
            Some(bars) => {
                if let (Some(raw_bar), Some(v)) = (&bars.raw, &raw_vals) {
                    raw_bar.set_length(v.total);
                    raw_bar.set_position(v.stored);
                    raw_bar.set_message(format!(
                        "{} dl │ {:.1} MB/s │ {} act │ {} msng",
                        human_bytes(v.bytes),
                        recent_raw_bps / 1.0e6,
                        v.inflight,
                        v.missing,
                    ));
                }
                if let (Some(text_bar), Some(v)) = (&bars.text, &text_vals) {
                    text_bar.set_length(v.total_artifacts);
                    text_bar.set_position(v.scanned);
                    text_bar.set_message(format!(
                        "{} chk │ 1m {recent_cps:.0} ch/s, {:.1} MB/s, ~{} tk/s │ \
                         av {avg_cps:.0} chk/s │ r {}\u{2192}q {}\
                         \u{2192}e {}\u{2192}q {} │ {} msng",
                        v.chunks,
                        recent_bps / 1.0e6,
                        human_count(recent_tps as u64),
                        v.reads_inflight,
                        v.jobs_queued,
                        v.embeds_inflight,
                        v.out_queued,
                        v.missing,
                    ));
                }
            }
            // Non-TTY: emit a periodic single-line log built from whichever
            // passes are active.
            None => {
                let mut segments: Vec<String> = Vec::new();
                if let Some(v) = &raw_vals {
                    segments.push(format!(
                        "raw: {}/{} files, {} dl, {:.1} MB/s, {} active, {} missing",
                        v.stored,
                        v.total,
                        human_bytes(v.bytes),
                        recent_raw_bps / 1.0e6,
                        v.inflight,
                        v.missing,
                    ));
                }
                if let Some(v) = &text_vals {
                    segments.push(format!(
                        "{} chunks | 1m: {recent_cps:.0} ch/s, {:.1} MB/s, ~{} tok/s | \
                         avg: {avg_cps:.0} ch/s | pipe: rd {} \u{2192} q {} \
                         \u{2192} emb {} \u{2192} q {} | {}/{} scanned | {} missing",
                        v.chunks,
                        recent_bps / 1.0e6,
                        human_count(recent_tps as u64),
                        v.reads_inflight,
                        v.jobs_queued,
                        v.embeds_inflight,
                        v.out_queued,
                        v.scanned,
                        v.total_artifacts,
                        v.missing,
                    ));
                }
                if !segments.is_empty() {
                    tracing::info!("{}", segments.join(" || "));
                }
            }
        }

        if stopping {
            if let Some(bars) = &bars {
                if let Some(raw_bar) = &bars.raw {
                    raw_bar.finish();
                }
                if let Some(text_bar) = &bars.text {
                    text_bar.finish();
                }
            }
            // The throughput summary is meaningful only for the full-text pass.
            if text.is_some() {
                let embedded = chunks - initial_chunks;
                let secs = run_start.map_or(0.0, |rs| now.duration_since(rs).as_secs_f64());
                tracing::info!(
                    "embedding throughput: {embedded} chunks in {secs:.0}s ({:.0} chunks/s avg)",
                    embedded as f64 / secs.max(1e-6),
                );
            }
            break;
        }
    }
}

/// A snapshot of the full-text counters, sampled once per tick and shared by the
/// bar and log render paths.
struct TextVals {
    chunks: u64,
    bytes: u64,
    scanned: u64,
    missing: u64,
    total_artifacts: u64,
    reads_inflight: u64,
    jobs_queued: u64,
    embeds_inflight: u64,
    out_queued: u64,
}

/// A snapshot of the raw-download counters, sampled once per tick and shared by
/// the bar and log render paths.
struct RawVals {
    stored: u64,
    total: u64,
    inflight: u64,
    bytes: u64,
    missing: u64,
}

/// The live progress bars driven by [`run_ticker`] on a TTY: a raw-download bar
/// and/or a full-text bar, for whichever passes are running. Held in a
/// [`MultiProgress`] so they redraw together without clobbering each other.
struct TickerBars {
    _multi: MultiProgress,
    raw: Option<ProgressBar>,
    text: Option<ProgressBar>,
}

impl TickerBars {
    /// Build the bar set, including the raw-download bar above the full-text bar
    /// for whichever passes are active.
    fn new(with_text: bool, with_raw: bool) -> Self {
        let multi = MultiProgress::new();
        let raw = with_raw.then(|| {
            let bar = multi.add(ProgressBar::new(0));
            bar.set_style(
                ProgressStyle::with_template(
                    "{prefix:<9} {bar:28.green/blue} {pos:>9}/{len:>9} files\n    {msg}",
                )
                .expect("valid template")
                .progress_chars("=>-"),
            );
            bar.set_prefix("raw");
            bar
        });
        let text = with_text.then(|| {
            let bar = multi.add(ProgressBar::new(0));
            bar.set_style(
                ProgressStyle::with_template(
                    "{prefix:<9} {bar:28.cyan/blue} {pos:>9}/{len:>9} arts\n    {msg}",
                )
                .expect("valid template")
                .progress_chars("=>-"),
            );
            bar.set_prefix("full-text");
            bar
        });
        Self {
            _multi: multi,
            raw,
            text,
        }
    }
}

/// Format a count compactly with a K/M suffix (for the high-magnitude tok/s).
fn human_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1.0e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1.0e3)
    } else {
        n.to_string()
    }
}

/// Format a byte count with a binary (1024) unit suffix, for the raw-download
/// segment of the shared progress line.
fn human_bytes(n: u64) -> String {
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
