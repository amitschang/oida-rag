//! Storing raw (non-text) artifact bytes in a LanceDB table.
//!
//! The hybrid index already reads the plain-text (OCR/`text/plain`) artifacts
//! to power keyword + semantic search. Raw storage covers the *complement*:
//! every artifact that is not plain text (PDFs, images, spreadsheets, …) is
//! fetched from the configured [`ArtifactSource`] and written, bytes and all,
//! into a `raw_artifacts` table keyed by document id and artifact name, so a
//! later point lookup can return the original file.
//!
//! Like [`crate::hybrid::build`], the build is a separate pass driven by the
//! already-ingested `artifacts` table — not by the Solr stream — so it composes
//! with the incremental workflow:
//!
//! * `ingest --force --store-raw` rebuilds the table from scratch.
//! * `ingest --store-raw` (incremental) fills in only the artifacts not already
//!   stored, picking up new and changed documents after the metadata update in
//!   the same command. (The incremental update deletes the raw rows of changed
//!   and redacted documents up front, so this re-fetches their current bytes.)

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result};
use arrow::array::{Int64Builder, LargeBinaryBuilder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use futures::stream::{self, StreamExt, TryStreamExt};
use lancedb::index::Index as LanceIndex;
use lancedb::index::scalar::BTreeIndexBuilder;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{Connection, Table};
use tokio::sync::{mpsc, oneshot};

use crate::config::Config;
use crate::index::{Index, RAW_ARTIFACTS_TABLE, RawArtifactRef, has_table, raw_refs_from_batch, str_col};
use crate::ingest::connect;
use crate::progress::{RawProgress, run_ticker};
use crate::source::ArtifactSource;

/// Number of artifacts fetched concurrently per round and buffered before they
/// are appended to the pending flush set. Bounds the fetch window (peak memory
/// is roughly this many blobs plus the pending flush buffer); it does *not*
/// determine fragment size — that is governed by `config.raw_file_bytes`.
pub(crate) const FETCH_CHUNK: usize = 128;

/// Buffered `FETCH_CHUNK`-sized ref groups between the producer (index scan) and
/// the fetcher. Small: it only paces the cheap ref stream, and a couple of
/// groups in flight is enough to keep the fetcher fed without letting the scan
/// race far ahead.
const REFS_CHANNEL_CAP: usize = 2;

/// Buffered assembled fragments between the fetcher and the writer. Each holds
/// up to `config.raw_file_bytes` of artifact bytes, so this is kept small to
/// bound peak memory while still letting fetching overlap a LanceDB flush.
const WRITE_CHANNEL_CAP: usize = 2;

/// Outcome counts from a raw-storage build.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawStats {
    /// Raw artifacts fetched and written in this run.
    pub stored: u64,
    /// Candidates skipped because they were already present (resume only).
    pub skipped: u64,
    /// Referenced artifacts whose bytes were absent from the source.
    pub missing: u64,
}

/// Build (or extend) the `raw_artifacts` table.
///
/// Fresh (`resume == false`) drops any existing table and stores every non-text
/// artifact. Resume (`resume == true`) keeps the table and stores only the
/// artifacts whose `(id, name)` is not already present, so it can be re-run
/// incrementally after an in-place metadata update.
///
/// Owns its own progress and ticker so a standalone raw build still renders a
/// live status line. The concurrent orchestrator instead drives
/// [`build_with_progress`] with a shared ticker so raw storage and the full-text
/// build render on one status line.
pub async fn build(config: &Config, index: &Index, resume: bool) -> Result<RawStats> {
    let progress = Arc::new(RawProgress::default());
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let ticker = {
        let progress = progress.clone();
        tokio::spawn(run_ticker(None, Some(progress), stop_rx))
    };
    let result = build_with_progress(config, index, resume, Some(progress.as_ref())).await;
    let _ = stop_tx.send(());
    let _ = ticker.await;
    result
}

/// Like [`build`], but reports live progress into `progress` when present. Used
/// by the concurrent orchestrator so the shared status line can show download
/// throughput; the standalone [`build`] passes `None`.
pub(crate) async fn build_with_progress(
    config: &Config,
    index: &Index,
    resume: bool,
    progress: Option<&RawProgress>,
) -> Result<RawStats> {
    let source = ArtifactSource::from_config(config)?.ok_or_else(|| {
        anyhow::anyhow!(
            "no artifact source configured (set artifact_root or s3_bucket); \
             raw storage needs the artifact files"
        )
    })?;

    let db = connect(config).await?;
    let have_table = has_table(&db, RAW_ARTIFACTS_TABLE).await?;

    if !resume && have_table {
        db.drop_table(RAW_ARTIFACTS_TABLE, &[])
            .await
            .context("dropping raw_artifacts table for a fresh build")?;
    }

    // On resume, the keys already stored — hashed to 128 bits so the resident
    // set stays a few hundred MB rather than gigabytes of (id, name) strings
    // across a >24M-artifact corpus. A 128-bit hash makes a false "already
    // stored" (which would silently skip an artifact) astronomically unlikely.
    // One sequential scan builds it: raw_artifacts carries no scalar index, and
    // on a resume (an interrupted prior run) none was ever built, so a per-chunk
    // indexed probe would flat-scan the whole table every time instead.
    let already: HashSet<u128> = if resume && have_table {
        existing_keys(&db).await?
    } else {
        HashSet::new()
    };

    // Progress denominator: all non-text candidates minus those already stored,
    // counted rather than materialized so the candidate list never lands in RAM.
    if let Some(p) = progress {
        let total = index.nontext_count().await?;
        p.total
            .store(total.saturating_sub(already.len() as u64), Ordering::Relaxed);
    }

    // Target fragment size. Accumulating fetched blobs to this byte target before
    // flushing decouples file size from the (variable) per-artifact size, so
    // fragments stay an even size instead of swinging with whatever blobs a
    // fixed-count batch happened to hold. Each flush is also the resume
    // checkpoint, so the target bounds how much fetched-but-unwritten work an
    // interruption can lose.
    let target = config.raw_file_bytes.max(1);

    // Run the build as a three-stage pipeline so a slow fetch or a LanceDB flush
    // never stalls the index scan: each stage hands off through a bounded channel
    // and the three run concurrently.
    //
    //   produce_refs  scan the index, skip already-stored keys, emit FETCH_CHUNK
    //                 ref groups
    //         │  refs channel
    //   fetch_stage   download each group's bytes, assemble byte-target fragments
    //         │  write channel
    //   write_stage   append fragments to LanceDB, in order, on the lone writer
    //
    // The channels are small (REFS_CHANNEL_CAP, WRITE_CHANNEL_CAP) so a fast
    // producer applies backpressure rather than racing arbitrarily far ahead.
    let (refs_tx, refs_rx) = mpsc::channel::<Vec<RawArtifactRef>>(REFS_CHANNEL_CAP);
    let (write_tx, write_rx) = mpsc::channel::<RecordBatch>(WRITE_CHANNEL_CAP);

    let (skipped, (stored, missing), table) = tokio::try_join!(
        produce_refs(index, &already, progress, refs_tx),
        fetch_stage(
            &source,
            config.read_concurrency,
            target,
            progress,
            refs_rx,
            write_tx,
        ),
        write_stage(&db, resume, have_table, write_rx),
    )?;

    // Build the `(id, name)` scalar indexes the point-lookup retrieval
    // ([`crate::index::Index::get_raw_artifact`]) rides. Recreated each run so a
    // resume's appended rows are covered, not just the initial set. Skipped only
    // when nothing was ever written (no table handle).
    if let Some(t) = &table {
        ensure_indexes(t).await?;
    }

    Ok(RawStats {
        stored,
        skipped,
        missing,
    })
}

/// Stage 1: stream the non-text candidates from the index, skip the keys already
/// stored (resume), and emit them in `FETCH_CHUNK`-sized groups onto `tx`.
///
/// Streamed batch-by-batch instead of materializing the whole list: over >24M
/// artifacts it would be gigabytes. Returns the number of candidates skipped.
async fn produce_refs(
    index: &Index,
    already: &HashSet<u128>,
    progress: Option<&RawProgress>,
    tx: mpsc::Sender<Vec<RawArtifactRef>>,
) -> Result<u64> {
    let mut skipped: u64 = 0;
    let mut fetch_buf: Vec<RawArtifactRef> = Vec::with_capacity(FETCH_CHUNK);

    let mut stream = index
        .nontext_artifacts_stream()
        .await
        .context("streaming non-text artifacts")?;
    while let Some(batch) = stream
        .try_next()
        .await
        .context("reading non-text artifact batch")?
    {
        for r in raw_refs_from_batch(&batch)? {
            if already.contains(&key128(&r.id, &r.name)) {
                skipped += 1;
                if let Some(p) = progress {
                    p.skipped.store(skipped, Ordering::Relaxed);
                }
                continue;
            }
            fetch_buf.push(r);
            if fetch_buf.len() >= FETCH_CHUNK {
                let chunk = std::mem::replace(&mut fetch_buf, Vec::with_capacity(FETCH_CHUNK));
                // A send error means a downstream stage failed and dropped the
                // receiver; that stage's error surfaces through try_join, so just
                // stop producing.
                if tx.send(chunk).await.is_err() {
                    return Ok(skipped);
                }
            }
        }
    }
    // Emit any trailing candidates that didn't fill a FETCH_CHUNK.
    if !fetch_buf.is_empty() {
        let _ = tx.send(fetch_buf).await;
    }
    Ok(skipped)
}

/// Stage 2: fetch each ref group's bytes and assemble them into byte-target
/// fragments, forwarding completed fragments to the writer on `tx`.
///
/// Returns `(stored, missing)`: artifacts fetched and written, and referenced
/// artifacts whose bytes were absent from the source.
async fn fetch_stage(
    source: &ArtifactSource,
    read_concurrency: usize,
    target: usize,
    progress: Option<&RawProgress>,
    mut rx: mpsc::Receiver<Vec<RawArtifactRef>>,
    tx: mpsc::Sender<RecordBatch>,
) -> Result<(u64, u64)> {
    let mut stored: u64 = 0;
    let mut missing: u64 = 0;
    let mut pending: Vec<(RawArtifactRef, Vec<u8>)> = Vec::new();
    let mut pending_bytes: usize = 0;

    while let Some(chunk) = rx.recv().await {
        let fetched = fetch_chunk(source, &chunk, read_concurrency, progress).await?;
        missing += (chunk.len() - fetched.len()) as u64;
        if let Some(p) = progress {
            p.missing.store(missing, Ordering::Relaxed);
        }
        if !fetched.is_empty() {
            stored += fetched.len() as u64;
            if let Some(p) = progress {
                p.stored.store(stored, Ordering::Relaxed);
            }
            for row in fetched {
                pending_bytes += row.1.len();
                pending.push(row);
            }
        }
        if pending_bytes >= target {
            let batch = raw_artifacts_batch(&pending)?;
            // Writer gone (it failed); its error surfaces through try_join.
            if tx.send(batch).await.is_err() {
                return Ok((stored, missing));
            }
            pending.clear();
            pending_bytes = 0;
        }
    }
    // Flush whatever remains below the target as the final (smaller) fragment.
    if !pending.is_empty() {
        let batch = raw_artifacts_batch(&pending)?;
        let _ = tx.send(batch).await;
    }
    Ok((stored, missing))
}

/// Stage 3: the lone writer. Appends each fragment to LanceDB in arrival order,
/// creating the table on the first write (or opening it to resume). Returns the
/// table handle so the caller can build the `(id, name)` indexes, or `None` when
/// nothing was ever written.
async fn write_stage(
    db: &Connection,
    resume: bool,
    have_table: bool,
    mut rx: mpsc::Receiver<RecordBatch>,
) -> Result<Option<Table>> {
    let mut table: Option<Table> = if resume && have_table {
        Some(
            db.open_table(RAW_ARTIFACTS_TABLE)
                .execute()
                .await
                .context("opening raw_artifacts table to resume")?,
        )
    } else {
        None
    };
    while let Some(batch) = rx.recv().await {
        write_batch(db, &mut table, batch).await?;
    }
    Ok(table)
}

/// Create (or replace) the `(id, name)` BTree scalar indexes that make the
/// `(id, name)` point lookup a true index probe instead of a full scan. Run
/// after a build or resume so the indexes cover every current row.
async fn ensure_indexes(table: &Table) -> Result<()> {
    for col in ["id", "name"] {
        table
            .create_index(&[col], LanceIndex::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await
            .with_context(|| format!("creating raw_artifacts.{col} index"))?;
    }
    Ok(())
}

/// Read the `(id, name)` keys already present in `raw_artifacts`, hashed to 128
/// bits ([`key128`]) so the resident set stays compact. Streamed batch-by-batch
/// so the raw key strings are never all held at once — only the hash set grows.
async fn existing_keys(db: &Connection) -> Result<HashSet<u128>> {
    let table = db
        .open_table(RAW_ARTIFACTS_TABLE)
        .execute()
        .await
        .context("opening raw_artifacts table to read keys")?;
    let mut stream = table
        .query()
        .select(Select::columns(&["id", "name"]))
        .execute()
        .await
        .context("scanning existing raw keys")?;
    let mut out = HashSet::new();
    while let Some(batch) = stream
        .try_next()
        .await
        .context("collecting existing raw keys")?
    {
        let ids = str_col(&batch, "id")?;
        let names = str_col(&batch, "name")?;
        for row in 0..batch.num_rows() {
            out.insert(key128(ids.value(row), names.value(row)));
        }
    }
    Ok(out)
}

/// A 128-bit hash of an artifact's `(id, name)` key. Holds the already-stored
/// resume set compactly — ~16 bytes per key instead of two heap strings. Two
/// differently-seeded SipHash passes give the 128 bits, making a collision
/// (which would wrongly skip an artifact) negligible across tens of millions of
/// keys.
fn key128(id: &str, name: &str) -> u128 {
    use std::hash::{Hash, Hasher};
    let hash_seeded = |seed: u8| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut h);
        id.hash(&mut h);
        0u8.hash(&mut h);
        name.hash(&mut h);
        h.finish()
    };
    ((hash_seeded(0) as u128) << 64) | hash_seeded(1) as u128
}

/// Append one assembled fragment to LanceDB, creating the `raw_artifacts` table
/// on the first write and reusing the handle thereafter.
async fn write_batch(db: &Connection, table: &mut Option<Table>, batch: RecordBatch) -> Result<()> {
    match table {
        Some(t) => {
            t.add(vec![batch]).execute().await.context("appending raw_artifacts batch")?;
        }
        None => {
            let t = db
                .create_table(RAW_ARTIFACTS_TABLE, vec![batch])
                .execute()
                .await
                .context("creating raw_artifacts table")?;
            *table = Some(t);
        }
    }
    Ok(())
}

/// Fetch a slice of artifacts' bytes concurrently, dropping any whose file is
/// absent from the source (returns `None`). Updates `progress` (if present)
/// with the in-flight count and bytes transferred as each fetch completes.
async fn fetch_chunk(
    source: &ArtifactSource,
    refs: &[RawArtifactRef],
    concurrency: usize,
    progress: Option<&RawProgress>,
) -> Result<Vec<(RawArtifactRef, Vec<u8>)>> {
    stream::iter(refs.iter().map(|r| {
        let r = r.clone();
        async move {
            if let Some(p) = progress {
                p.inflight.fetch_add(1, Ordering::Relaxed);
            }
            let result = source.get(&r.id, &r.name).await;
            if let Some(p) = progress {
                p.inflight.fetch_sub(1, Ordering::Relaxed);
                if let Ok(Some(bytes)) = &result {
                    p.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                }
            }
            Ok::<_, anyhow::Error>(result?.map(|b| (r, b)))
        }
    }))
    .buffer_unordered(concurrency.max(1))
    .try_filter_map(|opt| async move { Ok(opt) })
    .try_collect()
    .await
}

/// Build the `raw_artifacts` record batch for a set of fetched artifacts.
fn raw_artifacts_batch(rows: &[(RawArtifactRef, Vec<u8>)]) -> Result<RecordBatch> {
    let mut id = StringBuilder::new();
    let mut name = StringBuilder::new();
    let mut media_type = StringBuilder::new();
    let mut md5 = StringBuilder::new();
    let mut size = Int64Builder::new();
    let mut data = LargeBinaryBuilder::new();

    for (r, bytes) in rows {
        id.append_value(&r.id);
        name.append_value(&r.name);
        match &r.media_type {
            Some(v) => media_type.append_value(v),
            None => media_type.append_null(),
        }
        match &r.md5 {
            Some(v) => md5.append_value(v),
            None => md5.append_null(),
        }
        match r.size {
            Some(v) => size.append_value(v),
            None => size.append_null(),
        }
        data.append_value(bytes);
    }

    // `data` holds the whole file; retrieval is by id/name point lookup, so a
    // plain binary column suffices (no scan ever materializes it in bulk).
    let fields = vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("media_type", DataType::Utf8, true),
        Field::new("md5", DataType::Utf8, true),
        Field::new("size", DataType::Int64, true),
        Field::new("data", DataType::LargeBinary, false),
    ];
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        vec![
            Arc::new(id.finish()),
            Arc::new(name.finish()),
            Arc::new(media_type.finish()),
            Arc::new(md5.finish()),
            Arc::new(size.finish()),
            Arc::new(data.finish()),
        ],
    )
    .context("assembling raw_artifacts batch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, LargeBinaryArray};

    fn pdf_ref() -> RawArtifactRef {
        RawArtifactRef {
            id: "abc123".into(),
            name: "abc123.pdf".into(),
            media_type: Some("application/pdf".into()),
            md5: Some("bbb".into()),
            size: Some(20),
        }
    }

    #[test]
    fn raw_batch_carries_bytes() {
        let rows = vec![(pdf_ref(), b"%PDF-1.7".to_vec())];
        let batch = raw_artifacts_batch(&rows).unwrap();
        assert_eq!(batch.num_rows(), 1);
        let idx = batch.schema().index_of("data").unwrap();
        let data = batch.column(idx).as_any().downcast_ref::<LargeBinaryArray>().unwrap();
        assert_eq!(data.value(0), b"%PDF-1.7");
    }

    /// LanceDB must accept the `data` column and round-trip the bytes through a
    /// normal point-lookup query.
    #[tokio::test]
    async fn raw_table_round_trips_through_lancedb() {
        let dir = std::env::temp_dir().join(format!("oida-raw-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db = lancedb::connect(dir.to_str().unwrap()).execute().await.unwrap();

        let rows = vec![(pdf_ref(), b"%PDF-1.7".to_vec())];
        let batch = raw_artifacts_batch(&rows).unwrap();
        let table = db
            .create_table(RAW_ARTIFACTS_TABLE, vec![batch])
            .execute()
            .await
            .unwrap();
        assert_eq!(table.count_rows(None).await.unwrap(), 1);

        let got: Vec<RecordBatch> = table
            .query()
            .only_if("id = 'abc123'")
            .execute()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let batch = &got[0];
        let idx = batch.schema().index_of("data").unwrap();
        let data = batch.column(idx).as_any().downcast_ref::<LargeBinaryArray>().unwrap();
        assert_eq!(data.value(0), b"%PDF-1.7");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// After `ensure_indexes`, the `(id, name)` key uniquely selects one
    /// artifact's bytes — the retrieval the index exists to serve.
    #[tokio::test]
    async fn id_name_lookup_returns_one_artifact() {
        let dir = std::env::temp_dir().join(format!("oida-raw-idx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db = lancedb::connect(dir.to_str().unwrap()).execute().await.unwrap();

        // Two artifacts under the same id, distinguished only by name.
        let mut other = pdf_ref();
        other.name = "abc123_thumb.png".into();
        let rows = vec![
            (pdf_ref(), b"%PDF-1.7".to_vec()),
            (other, b"\x89PNG".to_vec()),
        ];
        let batch = raw_artifacts_batch(&rows).unwrap();
        let table = db
            .create_table(RAW_ARTIFACTS_TABLE, vec![batch])
            .execute()
            .await
            .unwrap();
        ensure_indexes(&table).await.unwrap();

        let got: Vec<RecordBatch> = table
            .query()
            .only_if("id = 'abc123' AND name = 'abc123.pdf'")
            .execute()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let total: usize = got.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);
        let batch = got.iter().find(|b| b.num_rows() > 0).unwrap();
        let idx = batch.schema().index_of("data").unwrap();
        let data = batch.column(idx).as_any().downcast_ref::<LargeBinaryArray>().unwrap();
        assert_eq!(data.value(0), b"%PDF-1.7");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
