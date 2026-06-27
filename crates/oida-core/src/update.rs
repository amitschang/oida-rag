//! Read-only "what would an update change?" differ over the Solr source.
//!
//! This is the validation slice of the incremental-update path: it pages the
//! archive Solr core from a modified-date watermark (`since`) and classifies
//! each document against the live LanceDB index *without writing anything*.
//!
//! Classification compares the artifact set Solr reports (`name` + `md5`,
//! parsed from the stringified `artifact` field) against the artifacts already
//! indexed for that `id`:
//! - **new** — the `id` is not in the index yet.
//! - **changed** — the `id` exists but its artifact `name`/`md5` set differs, so
//!   its content changed and it must be re-fetched and re-embedded.
//! - **unchanged** — same artifact set; a boundary-day re-scan to skip.
//! - **redacted** — the document is withdrawn (`deaccessioned`, unpublished, or
//!   left with no artifacts) and would be deleted from the index.
//!
//! The md5 set is the authoritative content signal because the `artifact` field
//! maps exactly onto the `artifacts` table; document *metadata* columns are
//! produced by a separate scraper and are intentionally out of scope here. The
//! report also tallies the `text/plain` artifacts a real update would fetch to
//! re-embed, and the watermark (max `ddmudate`) it would persist.

use std::collections::{BTreeSet, HashMap};

use anyhow::{Result, anyhow};
use serde_json::Value;

use crate::config::Config;
use crate::index::Index;
use crate::solr::{CURSOR_START, SolrClient};
use crate::solr_map::{ArtifactMeta, doc_id, doc_modified, parse_artifacts};

/// How an update would treat a single source document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Not yet in the index — would be inserted.
    New,
    /// In the index but its artifact set changed — would be re-embedded.
    Changed,
    /// In the index with the same artifact set — a boundary re-scan, skipped.
    Unchanged,
    /// Withdrawn at the source — would be deleted from the index.
    Redacted,
}

/// Summary of the delta a (dry-run) update would apply.
#[derive(Debug, Clone, Default)]
pub struct UpdatePlan {
    /// The inclusive modified-date lower bound the scan used (`None` = full scan).
    pub since: Option<String>,
    /// Total documents Solr reports for the query (`numFound`).
    pub num_found: u64,
    /// Documents actually scanned and classified.
    pub scanned: u64,
    /// Documents not yet indexed (would be inserted).
    pub new: u64,
    /// Indexed documents whose artifact set changed (would be re-embedded).
    pub changed: u64,
    /// Indexed documents with an unchanged artifact set (skipped).
    pub unchanged: u64,
    /// Documents withdrawn at the source (would be deleted).
    pub redacted: u64,
    /// `text/plain` artifacts across new+changed docs — the re-embed fetch list.
    pub refetch_text_artifacts: u64,
    /// Solr pages fetched.
    pub pages: u64,
    /// Greatest modified-date seen — the watermark a real update would persist.
    pub max_modified: Option<String>,
}

/// Summary of an incremental apply — what was actually written to the index.
#[derive(Debug, Clone, Default)]
pub struct ApplyStats {
    /// The effective inclusive modified-date lower bound the scan used.
    pub since: Option<String>,
    /// Total documents Solr reports for the query (`numFound`).
    pub num_found: u64,
    /// Documents scanned and classified.
    pub scanned: u64,
    /// Documents upserted into `documents`/`artifacts` (new + changed).
    pub upserted: u64,
    /// Documents not previously indexed (inserted).
    pub new: u64,
    /// Indexed documents whose artifact set changed (re-upserted + re-embedded).
    pub changed: u64,
    /// Indexed documents with an unchanged artifact set (skipped).
    pub unchanged: u64,
    /// Documents withdrawn at the source (deleted from the index).
    pub redacted: u64,
    /// Chunk rows removed so stale embedded text is never served.
    pub chunks_invalidated: u64,
    /// Raw-artifact rows removed so stale bytes are never returned; a later
    /// incremental `ingest --store-raw` re-fetches them.
    pub raw_invalidated: u64,
    /// Solr pages fetched.
    pub pages: u64,
    /// Greatest modified-date seen — the watermark persisted on success.
    pub watermark: Option<String>,
}

/// Build a Solr client from config, erroring clearly when `solr_url` is unset.
pub(crate) fn solr_client(config: &Config) -> Result<SolrClient> {
    let base = config.solr_url.as_deref().ok_or_else(|| {
        anyhow!(
            "solr_url is not configured; set it to the archive Solr core, e.g. \
             https://metadata.idl.ucsf.edu/solr/ltdl3"
        )
    })?;
    SolrClient::new(
        base,
        config.solr_query.clone(),
        config.solr_modified_field.clone(),
        config.solr_page_rows,
    )
}

/// Classify the Solr documents at or after `since` against the live index,
/// without writing anything.
///
/// The lean dry-run page (`fl=id,ddmudate`) is not enough to classify content,
/// so this scan requests the artifact and redaction fields too.
pub async fn dry_run(config: &Config, index: &Index, since: Option<&str>) -> Result<UpdatePlan> {
    let client = solr_client(config)?;
    let mut plan = UpdatePlan {
        since: since.map(str::to_string),
        ..Default::default()
    };
    let mut cursor = CURSOR_START.to_string();
    loop {
        let page = client.classify_page(since, &cursor).await?;
        plan.num_found = page.num_found;
        if page.docs.is_empty() {
            break;
        }

        // One batched index lookup per page keeps memory bounded over a large
        // delta while amortizing the query over `solr_page_rows` ids.
        let ids: Vec<String> = page.docs.iter().filter_map(doc_id).collect();
        let indexed = index.artifact_digests(&ids).await?;

        for doc in &page.docs {
            plan.scanned += 1;
            if let Some(m) = doc_modified(doc, &config.solr_modified_field)
                && plan
                    .max_modified
                    .as_deref()
                    .is_none_or(|cur| m.as_str() > cur)
            {
                plan.max_modified = Some(m);
            }

            let Some(id) = doc_id(doc) else { continue };
            let artifacts = parse_artifacts(doc);
            match classify(doc, &id, &artifacts, &indexed) {
                Class::New => {
                    plan.new += 1;
                    plan.refetch_text_artifacts += count_text(&artifacts);
                }
                Class::Changed => {
                    plan.changed += 1;
                    plan.refetch_text_artifacts += count_text(&artifacts);
                }
                Class::Unchanged => plan.unchanged += 1,
                Class::Redacted => plan.redacted += 1,
            }
        }
        plan.pages += 1;

        // Solr signals exhaustion by returning the cursor it was given.
        if page.next_cursor.is_empty() || page.next_cursor == cursor {
            break;
        }
        cursor = page.next_cursor;
    }
    Ok(plan)
}

/// Fetch one full Solr document (all fields) for schema inspection.
pub async fn sample_doc(config: &Config, since: Option<&str>) -> Result<Option<Value>> {
    solr_client(config)?.sample_doc(since).await
}

/// Apply an incremental metadata update to the live index in place.
///
/// The effective lower bound is `since` if given, else the index's persisted
/// watermark, else a full scan. Each Solr page is classified against the live
/// index (full source fields, so document/artifact rows can be rebuilt):
/// - **new**/**changed** docs are upserted into `documents`/`artifacts`;
/// - **redacted** docs are deleted;
/// - the chunks of every **changed** or **redacted** doc are invalidated so
///   stale embedded text is never served — a later incremental
///   `ingest --full-text` re-embeds exactly the affected documents.
///
/// Upserts run per page to bound memory; deletions and the watermark are
/// applied after the full scan and the watermark is written last, so a crash
/// mid-apply re-scans rather than silently skips un-applied work.
pub async fn apply(config: &Config, index: &Index, since: Option<&str>) -> Result<ApplyStats> {
    let effective_since = match since {
        Some(s) => Some(s.to_string()),
        None => index.read_watermark().await?,
    };
    let client = solr_client(config)?;
    let mut stats = ApplyStats {
        since: effective_since.clone(),
        ..Default::default()
    };
    let mut cursor = CURSOR_START.to_string();
    let mut redacted_ids: Vec<String> = Vec::new();
    let mut invalidate_ids: Vec<String> = Vec::new();
    let mut watermark: Option<String> = None;
    loop {
        let page = client
            .scan_page(
                effective_since.as_deref(),
                &cursor,
                crate::solr_map::SOURCE_FIELDS,
            )
            .await?;
        stats.num_found = page.num_found;
        if page.docs.is_empty() {
            break;
        }

        let ids: Vec<String> = page.docs.iter().filter_map(doc_id).collect();
        let indexed = index.artifact_digests(&ids).await?;
        let mut upsert: Vec<Value> = Vec::new();
        for doc in &page.docs {
            stats.scanned += 1;
            if let Some(m) = doc_modified(doc, &config.solr_modified_field)
                && watermark.as_deref().is_none_or(|cur| m.as_str() > cur)
            {
                watermark = Some(m);
            }

            let Some(id) = doc_id(doc) else { continue };
            let artifacts = parse_artifacts(doc);
            match classify(doc, &id, &artifacts, &indexed) {
                Class::New => {
                    stats.new += 1;
                    upsert.push(doc.clone());
                }
                Class::Changed => {
                    stats.changed += 1;
                    upsert.push(doc.clone());
                    invalidate_ids.push(id);
                }
                Class::Unchanged => stats.unchanged += 1,
                Class::Redacted => {
                    stats.redacted += 1;
                    redacted_ids.push(id.clone());
                    invalidate_ids.push(id);
                }
            }
        }

        if !upsert.is_empty() {
            index
                .upsert_documents(&upsert, &config.solr_modified_field)
                .await?;
            stats.upserted += upsert.len() as u64;
        }
        stats.pages += 1;

        if page.next_cursor.is_empty() || page.next_cursor == cursor {
            break;
        }
        cursor = page.next_cursor;
    }

    index.delete_documents(&redacted_ids).await?;
    stats.chunks_invalidated = index.delete_chunks_for(&invalidate_ids).await?;
    stats.raw_invalidated = index.delete_raw_for(&invalidate_ids).await?;

    if let Some(w) = &watermark {
        index.write_watermark(w).await?;
    }
    stats.watermark = watermark;
    Ok(stats)
}

/// Decide how a source document compares to what is indexed.
fn classify(
    doc: &Value,
    id: &str,
    artifacts: &[ArtifactMeta],
    indexed: &HashMap<String, BTreeSet<String>>,
) -> Class {
    if is_redacted(doc, artifacts) {
        return Class::Redacted;
    }
    match indexed.get(id) {
        None => Class::New,
        Some(prev) if *prev == fingerprint(artifacts) => Class::Unchanged,
        Some(_) => Class::Changed,
    }
}

/// A document is withdrawn if it is deaccessioned, not published, or has been
/// stripped of all artifacts.
fn is_redacted(doc: &Value, artifacts: &[ArtifactMeta]) -> bool {
    let flagged = |field: &str, redacted_when: bool| {
        matches!(doc.get(field), Some(Value::Bool(b)) if *b == redacted_when)
    };
    artifacts.is_empty() || flagged("deaccessioned", true) || flagged("published", false)
}

/// The content fingerprint of an artifact set: the order-independent set of
/// `name\0md5` strings, matching [`Index::artifact_digests`].
fn fingerprint(artifacts: &[ArtifactMeta]) -> BTreeSet<String> {
    artifacts
        .iter()
        .map(|a| {
            format!(
                "{}\u{0}{}",
                a.name.as_deref().unwrap_or(""),
                a.md5.as_deref().unwrap_or("")
            )
        })
        .collect()
}

/// Count the `text/plain` artifacts — the OCR text an update would fetch and
/// re-embed for a new or changed document.
fn count_text(artifacts: &[ArtifactMeta]) -> u64 {
    artifacts
        .iter()
        .filter(|a| a.media_type.as_deref() == Some("text/plain"))
        .count() as u64
}
