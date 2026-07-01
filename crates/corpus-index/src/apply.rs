//! Incremental metadata apply — the generic, digest-based update driver.
//!
//! The provider streams the same [`SourcePage`](crate::provider::SourcePage)s
//! the full build consumes; this driver owns all the LanceDB machinery:
//! change-detection, upserts, deletes, chunk/raw invalidation, and the
//! watermark. The provider contributes only the per-document policy-withdrawal
//! flag on each page.
//!
//! Change detection is **digest-based**. `documents.digest` (a provider-computed
//! fingerprint over metadata *and* the artifact md5 set) is compared as a plain
//! string — unequal ⇒ upsert. The artifact `(name,md5)` set is the secondary
//! gate: only a real *content* change invalidates chunks/raw so re-embedding
//! fires only when needed, while a metadata-only edit upserts the row alone.

use std::collections::{BTreeSet, HashMap, HashSet};

use anyhow::{Result, anyhow};
use arrow::array::{Array, BooleanArray, RecordBatch};
use arrow::compute::filter_record_batch;
use futures::TryStreamExt;

use crate::config::CoreConfig;
use crate::index::{Index, str_col, track_max};
use crate::provider::SourceProvider;

/// Summary of an incremental apply — what was actually written to the index.
#[derive(Debug, Clone, Default)]
pub struct ApplyStats {
    /// The effective inclusive watermark lower bound the scan used.
    pub since: Option<String>,
    /// Total documents the source reports for the query.
    pub num_found: u64,
    /// Documents scanned and classified.
    pub scanned: u64,
    /// Documents upserted into `documents`/`artifacts` (new + changed).
    pub upserted: u64,
    /// Documents not previously indexed (inserted).
    pub new: u64,
    /// Indexed documents whose fingerprint changed (re-upserted).
    pub changed: u64,
    /// Indexed documents with an unchanged fingerprint (skipped).
    pub unchanged: u64,
    /// Documents withdrawn at the source (deleted from the index).
    pub redacted: u64,
    /// Chunk rows removed so stale embedded text is never served.
    pub chunks_invalidated: u64,
    /// Raw-artifact rows removed so stale bytes are never returned; a later
    /// incremental `ingest --store-raw` re-fetches them.
    pub raw_invalidated: u64,
    /// Source pages fetched.
    pub pages: u64,
    /// Greatest watermark seen — the value persisted on success.
    pub watermark: Option<String>,
}

/// Apply an incremental metadata update to the live index in place.
///
/// The effective lower bound is `since` if given, else the index's persisted
/// watermark, else a full scan. Each source page is classified against the live
/// index by digest; new/changed docs are upserted (by slicing the provider's
/// batches), withdrawn docs are deleted, and the chunks/raw of every doc whose
/// artifact set changed are invalidated. Upserts run per page to bound memory;
/// deletions and the watermark are applied after the full scan and the watermark
/// is written last, so a crash mid-apply re-scans rather than silently skips
/// un-applied work.
pub async fn apply<P: SourceProvider>(
    provider: &P,
    config: &CoreConfig,
    index: &Index,
    since: Option<&str>,
) -> Result<ApplyStats> {
    let _ = config; // reserved for future tuning knobs; scan drives everything.
    let effective_since = match since {
        Some(s) => Some(s.to_string()),
        None => index.read_watermark().await?,
    };
    let mut stats = ApplyStats {
        since: effective_since.clone(),
        ..Default::default()
    };
    let mut redacted_ids: Vec<String> = Vec::new();
    let mut invalidate_ids: Vec<String> = Vec::new();
    let mut watermark: Option<String> = None;

    let mut stream = std::pin::pin!(provider.scan(effective_since.as_deref()));
    while let Some(page) = stream.try_next().await? {
        stats.num_found = page.num_found;
        let n = page.documents.num_rows();
        if n == 0 {
            continue;
        }
        if let Some(w) = &page.watermark {
            track_max(&mut watermark, w.clone());
        }

        // Extract the row-aligned id/digest columns and the incoming artifact
        // `(name,md5)` sets, then read the stored digests/artifact sets for the
        // ids on this page in one batched query each.
        let id_opts = column_opt_strings(&page.documents, "id")?;
        let digest_opts = column_opt_strings(&page.documents, "digest")?;
        let present_ids: Vec<String> = id_opts.iter().flatten().cloned().collect();
        let stored_digest = index.document_digests(&present_ids).await?;
        let stored_arts = index.artifact_digests(&present_ids).await?;
        let incoming_arts = artifact_sets(&page.artifacts)?;

        let mut upsert_mask = vec![false; n];
        let mut upsert_ids: Vec<String> = Vec::new();
        for i in 0..n {
            stats.scanned += 1;
            let Some(id) = &id_opts[i] else { continue };

            // Structural redaction ("no artifacts") is framework-derived; the
            // provider supplies only the policy-withdrawal flag.
            let structurally_empty = incoming_arts.get(id).is_none_or(BTreeSet::is_empty);
            let withdrawn = page.redacted.get(i).copied().unwrap_or(false) || structurally_empty;

            if withdrawn {
                stats.redacted += 1;
                redacted_ids.push(id.clone());
                invalidate_ids.push(id.clone());
            } else if !stored_digest.contains_key(id) {
                stats.new += 1;
                upsert_mask[i] = true;
                upsert_ids.push(id.clone());
            } else if stored_digest.get(id).map(String::as_str) == digest_opts[i].as_deref() {
                stats.unchanged += 1;
            } else {
                stats.changed += 1;
                upsert_mask[i] = true;
                upsert_ids.push(id.clone());
                // Re-embed only on real content change: compare the artifact
                // `(name,md5)` set (digest also covers metadata-only edits).
                let incoming = incoming_arts.get(id).cloned().unwrap_or_default();
                let stored = stored_arts.get(id).cloned().unwrap_or_default();
                if incoming != stored {
                    invalidate_ids.push(id.clone());
                }
            }
        }

        if !upsert_ids.is_empty() {
            // Slice the provider's batches to the changed rows — reusing the
            // full-build mapping instead of a second mapper.
            let upsert_docs = filter_record_batch(&page.documents, &BooleanArray::from(upsert_mask))
                .map_err(|e| anyhow!("filtering upsert documents: {e}"))?;
            let upsert_set: HashSet<String> = upsert_ids.iter().cloned().collect();
            let art_ids = str_col(&page.artifacts, "id")?;
            let art_mask: BooleanArray = (0..page.artifacts.num_rows())
                .map(|i| !art_ids.is_null(i) && upsert_set.contains(art_ids.value(i)))
                .collect();
            let upsert_arts = filter_record_batch(&page.artifacts, &art_mask)
                .map_err(|e| anyhow!("filtering upsert artifacts: {e}"))?;
            index
                .upsert_documents(upsert_docs, upsert_arts, &upsert_ids)
                .await?;
            stats.upserted += upsert_ids.len() as u64;
        }
        stats.pages += 1;
    }

    // Deletes and the watermark are applied after the full scan, watermark last,
    // so a crash mid-apply re-scans rather than silently skips un-applied work.
    index.delete_documents(&redacted_ids).await?;
    stats.chunks_invalidated = index.delete_chunks_for(&invalidate_ids).await?;
    stats.raw_invalidated = index.delete_raw_for(&invalidate_ids).await?;

    if let Some(w) = &watermark {
        index.write_watermark(w).await?;
    }
    stats.watermark = watermark;
    Ok(stats)
}

/// Read a `documents`/`artifacts` string column into row-aligned optionals.
fn column_opt_strings(batch: &RecordBatch, name: &str) -> Result<Vec<Option<String>>> {
    let col = str_col(batch, name)?;
    Ok((0..col.len())
        .map(|i| (!col.is_null(i)).then(|| col.value(i).to_string()))
        .collect())
}

/// Group an `artifacts` batch into each document's `name\0md5` set — the content
/// fingerprint [`Index::artifact_digests`] stores, so the two compare directly.
fn artifact_sets(batch: &RecordBatch) -> Result<HashMap<String, BTreeSet<String>>> {
    let id = str_col(batch, "id")?;
    let name = str_col(batch, "name")?;
    let md5 = str_col(batch, "md5")?;
    let mut map: HashMap<String, BTreeSet<String>> = HashMap::new();
    for i in 0..batch.num_rows() {
        if id.is_null(i) {
            continue;
        }
        let n = if name.is_null(i) { "" } else { name.value(i) };
        let m = if md5.is_null(i) { "" } else { md5.value(i) };
        map.entry(id.value(i).to_string())
            .or_default()
            .insert(format!("{n}\u{0}{m}"));
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    /// Build a small `artifacts`-shaped batch for the pure decode helpers.
    fn artifacts_batch(rows: &[(&str, &str, &str)]) -> RecordBatch {
        let ids: Vec<&str> = rows.iter().map(|(i, _, _)| *i).collect();
        let names: Vec<&str> = rows.iter().map(|(_, n, _)| *n).collect();
        let md5s: Vec<&str> = rows.iter().map(|(_, _, m)| *m).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("md5", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(names)),
                Arc::new(StringArray::from(md5s)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn artifact_sets_group_by_id_with_name_md5_encoding() {
        let batch = artifacts_batch(&[
            ("abc", "abc.ocr", "aaa"),
            ("abc", "abc.pdf", "bbb"),
            ("xyz", "xyz.ocr", "ccc"),
        ]);
        let sets = artifact_sets(&batch).unwrap();
        assert_eq!(sets["abc"].len(), 2);
        // Same `name\0md5` encoding `Index::artifact_digests` stores.
        assert!(sets["abc"].contains("abc.ocr\u{0}aaa"));
        assert!(sets["abc"].contains("abc.pdf\u{0}bbb"));
        assert_eq!(sets["xyz"].len(), 1);
    }

    #[test]
    fn column_opt_strings_preserves_row_alignment() {
        let batch = artifacts_batch(&[("a", "a.ocr", "1"), ("b", "b.ocr", "2")]);
        let ids = column_opt_strings(&batch, "id").unwrap();
        assert_eq!(ids, vec![Some("a".to_string()), Some("b".to_string())]);
    }
}
