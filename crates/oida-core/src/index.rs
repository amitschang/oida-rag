//! LanceDB-backed access to the OIDA index.
//!
//! The Solr ingest maps the archive corpus (~24M artifacts) into a
//! document-level `documents` table plus a thin `artifacts` table, both stored
//! in a single embedded LanceDB database and indexed (scalar + FTS). All
//! metadata queries run against that store. Text/vector search lives in the
//! sibling [`crate::hybrid`] module against the same database.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use arrow::array::{
    Array, Int32Array, Int64Array, LargeBinaryArray, ListArray, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::arrow::SendableRecordBatchStream;
use lancedb::query::{ColumnOrdering, ExecutableQuery, QueryBase, QueryExecutionOptions, Select};
use lancedb::{Connection, Table};
use serde_json::Value;

use crate::config::Config;
use crate::ingest;
use crate::model::{Artifact, Document, RawArtifact};
use crate::solr_map;

/// Name of the document-level metadata table.
pub(crate) const DOCUMENTS_TABLE: &str = "documents";
/// Name of the per-artifact table.
pub(crate) const ARTIFACTS_TABLE: &str = "artifacts";
/// Name of the hybrid text-index chunks table (defined in [`crate::hybrid`]).
pub(crate) const CHUNKS_TABLE: &str = "chunks";
/// Name of the table holding raw (non-text) artifact bytes.
pub(crate) const RAW_ARTIFACTS_TABLE: &str = "raw_artifacts";
/// Single-row table persisting the incremental-update watermark.
///
/// Excluded from the metadata-rebuild drop set so it survives a full Solr
/// re-ingest (which overwrites it on completion anyway).
pub(crate) const INGEST_STATE_TABLE: &str = "_ingest_state";
/// Column of [`INGEST_STATE_TABLE`] holding the high-water modified-date.
const WATERMARK_COL: &str = "watermark";
/// Maximum ids per `id IN (...)` predicate, to bound generated SQL length.
const IN_LIST_CHUNK: usize = 500;
/// Predicate selecting the plain-text artifacts the hybrid index reads. The
/// source's `media_type` is authoritative for text (every OCR/text file is
/// `text/plain`), so this is a single equality the `media_type` scalar index
/// can satisfy — no `name LIKE '%.ocr'` scan.
const TEXT_FILTER: &str = "media_type = 'text/plain'";
/// Predicate selecting the non-text artifacts raw storage keeps. SQL `<>`
/// yields NULL (not true) for a NULL `media_type`, so this also excludes the
/// no-`media_type`/no-`name` rows that are not real artifacts — exactly the
/// complement of [`TEXT_FILTER`] over the real artifacts.
const NONTEXT_FILTER: &str = "media_type <> 'text/plain'";

/// Output batch size for the streamed artifact scans ([`Index::text_artifacts_stream`]
/// and [`Index::nontext_artifacts_stream`]). LanceDB caps batches at 1024 rows
/// by default; their consumers drain one batch's reads before pulling the next,
/// so a small batch makes read concurrency wind down to zero at every boundary.
/// The listing rows are tiny (a few ids and a media type), so a much wider batch
/// costs only a few MB while making those boundaries rare — keeping the
/// read→embed and download pipelines saturated when reads are the bottleneck.
/// It does not affect the `ORDER BY` sort buffer, which materializes regardless.
const SCAN_BATCH_ROWS: u32 = 16_384;

/// Columns of the `documents` table mapped onto [`Document`], in definition
/// order. `search_text` (an FTS-only concatenation) is intentionally excluded.
pub(crate) const DOC_COLS: &[&str] = &[
    "id",
    "bn",
    "title",
    "industry",
    "collection",
    "genre",
    "date_sent",
    "date_received",
    "topic",
    "description",
    "keywords",
    "conversation",
    "custodian",
    "authors",
    "recipients",
    "cc",
    "attachments",
    "related",
    "mentions",
    "artifact_types",
    "artifact_count",
];

/// Handle to the LanceDB-backed OIDA index.
///
/// LanceDB connection and table handles are cheap, `Clone`, and `Send + Sync`
/// (internally `Arc`-based), so no external locking is required for the
/// read-only serving path.
pub struct Index {
    pub(crate) db: Connection,
    pub(crate) documents: Table,
    pub(crate) artifacts: Table,
}

impl Index {
    /// Open the index, returning an error if metadata has not been ingested.
    pub async fn open(config: &Config) -> Result<Self> {
        if !config.lance_path.exists() {
            bail!(
                "no index found at {}; ingest it first (oida-cli ingest)",
                config.lance_path.display()
            );
        }
        let db = ingest::connect(config).await?;
        let documents = db
            .open_table(DOCUMENTS_TABLE)
            .execute()
            .await
            .with_context(|| {
                format!(
                    "opening documents table at {}; ingest it first (oida-cli ingest)",
                    config.lance_path.display()
                )
            })?;
        let artifacts = db
            .open_table(ARTIFACTS_TABLE)
            .execute()
            .await
            .with_context(|| {
                format!(
                    "opening artifacts table at {}; ingest it first (oida-cli ingest)",
                    config.lance_path.display()
                )
            })?;
        Ok(Self {
            db,
            documents,
            artifacts,
        })
    }

    /// True when the document metadata has been ingested at `config.lance_path`.
    ///
    /// Cheap existence probe used by the CLI/server to decide whether to serve
    /// or to instruct the user to run an ingest. Never mutates the store.
    pub async fn is_ingested(config: &Config) -> bool {
        if !config.lance_path.exists() {
            return false;
        }
        let Ok(db) = ingest::connect(config).await else {
            return false;
        };
        has_table(&db, DOCUMENTS_TABLE).await.unwrap_or(false)
    }

    /// Read the persisted incremental-update watermark (the high-water
    /// modified-date seen by a prior ingest/update), or `None` if no state has
    /// been recorded yet.
    pub async fn read_watermark(&self) -> Result<Option<String>> {
        read_watermark(&self.db).await
    }

    /// Persist `watermark` as the sole row of [`INGEST_STATE_TABLE`].
    pub async fn write_watermark(&self, watermark: &str) -> Result<()> {
        write_watermark(&self.db, watermark).await
    }

    /// Upsert `docs` (raw Solr docs) into the `documents` and `artifacts`
    /// tables.
    ///
    /// Documents are merged on `id` (insert-or-replace). For each document the
    /// previous artifact rows are deleted and the current set re-appended, so a
    /// changed content fingerprint is reflected exactly. Callers should batch
    /// per Solr page to bound memory.
    pub(crate) async fn upsert_documents(&self, docs: &[Value], modified_field: &str) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let batch = solr_map::documents_batch(docs, modified_field)?;
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut builder = self.documents.merge_insert(&["id"]);
        builder
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        builder
            .execute(Box::new(reader))
            .await
            .context("merge-inserting documents")?;

        let ids: Vec<String> = docs.iter().filter_map(solr_map::doc_id).collect();
        delete_in(&self.artifacts, "id", &ids).await?;
        let arts: RecordBatch = solr_map::artifacts_batch(docs)?;
        if arts.num_rows() > 0 {
            self.artifacts
                .add(vec![arts])
                .execute()
                .await
                .context("appending artifacts")?;
        }
        Ok(())
    }

    /// Delete the given document ids from both `documents` and `artifacts`
    /// (used for redacted/deaccessioned documents). Returns the number of
    /// document rows removed.
    pub(crate) async fn delete_documents(&self, ids: &[String]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }
        let removed = delete_in(&self.documents, "id", ids).await?;
        delete_in(&self.artifacts, "id", ids).await?;
        Ok(removed)
    }

    /// Delete every chunk whose `doc_id` is in `doc_ids` from the hybrid text
    /// index, so stale embedded text is never served. A later incremental
    /// `ingest --full-text` re-embeds the affected documents. Returns the number
    /// of chunk rows removed (0 if the chunks table does not exist yet).
    pub(crate) async fn delete_chunks_for(&self, doc_ids: &[String]) -> Result<u64> {
        if doc_ids.is_empty() {
            return Ok(0);
        }
        if !has_table(&self.db, CHUNKS_TABLE).await? {
            return Ok(0);
        }
        let chunks = self
            .db
            .open_table(CHUNKS_TABLE)
            .execute()
            .await
            .context("opening chunks table")?;
        delete_in(&chunks, "doc_id", doc_ids).await
    }

    /// Return the live `(documents, artifacts)` row counts.
    pub async fn counts(&self) -> Result<(u64, u64)> {
        let documents = self
            .documents
            .count_rows(None)
            .await
            .context("counting documents")? as u64;
        let artifacts = self
            .artifacts
            .count_rows(None)
            .await
            .context("counting artifacts")? as u64;
        Ok((documents, artifacts))
    }

    /// Summarise artifact byte sizes split into the full-text and raw
    /// (non-text) sets, contrasting *logical* (every referenced artifact) with
    /// *real* (bytes actually present in the archive).
    ///
    /// Logical totals sum the metadata `size` of every artifact in the
    /// `artifacts` table — what the corpus would weigh if all files were
    /// downloaded. Real totals count only artifacts whose bytes are stored: a
    /// text artifact is "real" once it has been chunked into the `chunks`
    /// table, a non-text artifact once it lives in `raw_artifacts`. Both
    /// reuse the same metadata `size`, so the two are directly comparable.
    pub async fn store_sizes(&self) -> Result<StoreSizes> {
        use std::collections::HashSet;

        let names = self.db.table_names().execute().await.context("listing tables")?;

        // Which text artifacts have been chunked into the full-text index.
        let mut ingested: HashSet<(String, String)> = HashSet::new();
        if contains_table(&names, CHUNKS_TABLE) {
            let chunks = self
                .db
                .open_table(CHUNKS_TABLE)
                .execute()
                .await
                .context("opening chunks table")?;
            let batches: Vec<RecordBatch> = chunks
                .query()
                .select(Select::columns(&["doc_id", "artifact_name"]))
                .execute()
                .await
                .context("scanning chunks for ingested artifacts")?
                .try_collect()
                .await
                .context("collecting ingested artifacts")?;
            for batch in &batches {
                let ids = str_col(batch, "doc_id")?;
                let nms = str_col(batch, "artifact_name")?;
                for row in 0..batch.num_rows() {
                    ingested.insert((ids.value(row).to_string(), nms.value(row).to_string()));
                }
            }
        }

        let mut sizes = StoreSizes::default();

        // Walk every referenced artifact once, classifying text vs non-text,
        // summing logical bytes, and marking text artifacts present in `chunks`.
        let batches: Vec<RecordBatch> = self
            .artifacts
            .query()
            .select(Select::columns(&["id", "name", "media_type", "size"]))
            .execute()
            .await
            .context("scanning artifacts for sizes")?
            .try_collect()
            .await
            .context("collecting artifact sizes")?;
        for batch in &batches {
            let ids = str_col(batch, "id")?;
            let nms = str_col(batch, "name")?;
            let media = str_col(batch, "media_type")?;
            let size = i64_col(batch, "size")?;
            for row in 0..batch.num_rows() {
                let bytes = if size.is_null(row) {
                    0
                } else {
                    size.value(row).max(0) as u64
                };
                let name = nms.value(row);
                let media_type = if media.is_null(row) {
                    None
                } else {
                    Some(media.value(row))
                };
                if crate::artifacts::is_text(name, media_type) {
                    sizes.text_logical_count += 1;
                    sizes.text_logical_bytes += bytes;
                    if ingested.contains(&(ids.value(row).to_string(), name.to_string())) {
                        sizes.text_real_count += 1;
                        sizes.text_real_bytes += bytes;
                    }
                } else {
                    sizes.raw_logical_count += 1;
                    sizes.raw_logical_bytes += bytes;
                }
            }
        }

        // Real non-text bytes: sum the `size` column of stored raw artifacts.
        if contains_table(&names, RAW_ARTIFACTS_TABLE) {
            let raws = self
                .db
                .open_table(RAW_ARTIFACTS_TABLE)
                .execute()
                .await
                .context("opening raw_artifacts table")?;
            let batches: Vec<RecordBatch> = raws
                .query()
                .select(Select::columns(&["size"]))
                .execute()
                .await
                .context("scanning raw_artifacts sizes")?
                .try_collect()
                .await
                .context("collecting raw_artifacts sizes")?;
            let mut count = 0u64;
            let mut total = 0u64;
            for batch in &batches {
                let size = i64_col(batch, "size")?;
                for row in 0..batch.num_rows() {
                    count += 1;
                    if !size.is_null(row) {
                        total += size.value(row).max(0) as u64;
                    }
                }
            }
            sizes.raw_real_count = Some(count);
            sizes.raw_real_bytes = Some(total);
        }

        Ok(sizes)
    }

    /// Return the subset of `ids` that already exist in the `documents` table.
    ///
    /// Used by the update differ to classify Solr documents as new vs.
    /// already-indexed in batches (one `id IN (...)` query per Solr page),
    /// selecting only the `id` column so the lookup stays cheap.
    pub async fn existing_ids(&self, ids: &[String]) -> Result<std::collections::HashSet<String>> {
        use std::collections::HashSet;
        if ids.is_empty() {
            return Ok(HashSet::new());
        }
        let list = ids.iter().map(|i| sql_str(i)).collect::<Vec<_>>().join(", ");
        let batches: Vec<RecordBatch> = self
            .documents
            .query()
            .only_if(format!("id IN ({list})"))
            .select(Select::columns(&["id"]))
            .execute()
            .await
            .context("executing existing-ids query")?
            .try_collect()
            .await
            .context("collecting existing ids")?;
        let mut out = HashSet::with_capacity(ids.len());
        for batch in &batches {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow::anyhow!("documents.id column is not Utf8"))?;
            for i in 0..col.len() {
                if !col.is_null(i) {
                    out.insert(col.value(i).to_string());
                }
            }
        }
        Ok(out)
    }

    /// Map each of `ids` to the set of `name\0md5` strings for its indexed
    /// artifacts (the document's content fingerprint).
    ///
    /// The update differ compares this against the artifact set Solr reports to
    /// decide whether a document's *content* changed (any md5 differs → must
    /// re-embed) versus a boundary-day re-scan of an unchanged document. Ids with
    /// no indexed artifacts are simply absent from the map.
    pub async fn artifact_digests(
        &self,
        ids: &[String],
    ) -> Result<std::collections::HashMap<String, std::collections::BTreeSet<String>>> {
        use std::collections::{BTreeSet, HashMap};
        let mut map: HashMap<String, BTreeSet<String>> = HashMap::new();
        if ids.is_empty() {
            return Ok(map);
        }
        let list = ids.iter().map(|i| sql_str(i)).collect::<Vec<_>>().join(", ");
        let batches: Vec<RecordBatch> = self
            .artifacts
            .query()
            .only_if(format!("id IN ({list})"))
            .select(Select::columns(&["id", "name", "md5"]))
            .execute()
            .await
            .context("executing artifact-digests query")?
            .try_collect()
            .await
            .context("collecting artifact digests")?;
        for batch in &batches {
            let id = str_col(batch, "id")?;
            let name = str_col(batch, "name")?;
            let md5 = str_col(batch, "md5")?;
            for i in 0..batch.num_rows() {
                if id.is_null(i) {
                    continue;
                }
                let name = if name.is_null(i) { "" } else { name.value(i) };
                let md5 = if md5.is_null(i) { "" } else { md5.value(i) };
                map.entry(id.value(i).to_string())
                    .or_default()
                    .insert(format!("{name}\u{0}{md5}"));
            }
        }
        Ok(map)
    }

    /// Run a `documents` query restricted by `filter` (a LanceDB SQL predicate),
    /// optionally capped at `limit` rows, mapping each row to a [`Document`].
    pub(crate) async fn documents_where(
        &self,
        filter: &str,
        limit: Option<usize>,
    ) -> Result<Vec<Document>> {
        let mut q = self
            .documents
            .query()
            .only_if(filter)
            .select(Select::columns(DOC_COLS));
        if let Some(l) = limit {
            q = q.limit(l);
        }
        let batches: Vec<RecordBatch> = q
            .execute()
            .await
            .context("executing documents query")?
            .try_collect()
            .await
            .context("collecting documents")?;
        collect_documents(&batches)
    }

    /// Full-text search the `documents` table's `search_text` column, returning
    /// up to `limit` matching documents ranked by the FTS index.
    pub(crate) async fn documents_fts(&self, query: &str, limit: usize) -> Result<Vec<Document>> {
        let batches: Vec<RecordBatch> = self
            .documents
            .query()
            .full_text_search(FullTextSearchQuery::new(query.to_string()))
            .select(Select::columns(DOC_COLS))
            .limit(limit)
            .execute()
            .await
            .context("executing documents FTS query")?
            .try_collect()
            .await
            .context("collecting documents")?;
        collect_documents(&batches)
    }

    /// Fetch a single document by its `id`.
    pub async fn get_document_by_id(&self, id: &str) -> Result<Option<Document>> {
        Ok(self
            .documents_where(&format!("id = {}", sql_str(id)), Some(1))
            .await?
            .into_iter()
            .next())
    }

    /// Fetch a single document by its Bates number `bn`.
    pub async fn get_document_by_bn(&self, bn: &str) -> Result<Option<Document>> {
        Ok(self
            .documents_where(&format!("bn = {}", sql_str(bn)), Some(1))
            .await?
            .into_iter()
            .next())
    }

    /// Fetch documents for a list of `ids`, preserving the order of `ids`.
    ///
    /// Ids with no matching document are skipped. Used to hydrate ranked
    /// hybrid-search results back into full document metadata while keeping the
    /// relevance ordering intact.
    pub async fn get_documents_by_ids(&self, ids: &[String]) -> Result<Vec<Document>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let list = ids.iter().map(|i| sql_str(i)).collect::<Vec<_>>().join(", ");
        let docs = self
            .documents_where(&format!("id IN ({list})"), None)
            .await?;
        // `IN` does not preserve order; reorder to match `ids`.
        let mut by_id: std::collections::HashMap<&str, Document> =
            docs.iter().map(|d| (d.id.as_str(), d.clone())).collect();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(doc) = by_id.remove(id.as_str()) {
                out.push(doc);
            }
        }
        Ok(out)
    }

    /// Look up documents by a set of Bates numbers.
    pub async fn get_documents_by_bns(&self, bns: &[String]) -> Result<Vec<Document>> {
        if bns.is_empty() {
            return Ok(Vec::new());
        }
        let list = bns.iter().map(|b| sql_str(b)).collect::<Vec<_>>().join(", ");
        self.documents_where(&format!("bn IN ({list})"), None).await
    }

    /// Look up documents sharing a conversation thread, excluding one id.
    pub async fn get_documents_by_conversation(
        &self,
        conversation: &str,
        exclude_id: &str,
    ) -> Result<Vec<Document>> {
        self.documents_where(
            &format!(
                "conversation = {} AND id != {}",
                sql_str(conversation),
                sql_str(exclude_id)
            ),
            None,
        )
        .await
    }

    /// List the artifacts belonging to a document.
    pub async fn get_artifacts(&self, id: &str) -> Result<Vec<Artifact>> {
        let batches: Vec<RecordBatch> = self
            .artifacts
            .query()
            .only_if(format!("id = {}", sql_str(id)))
            .select(Select::columns(&["id", "name", "media_type", "size", "md5"]))
            .execute()
            .await
            .context("executing artifacts query")?
            .try_collect()
            .await
            .context("collecting artifacts")?;
        let mut out = collect_artifacts(&batches)?;
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// Fetch the stored bytes of a single non-text artifact by its `(id, name)`
    /// key — the only retrieval mode raw storage supports, since that pair
    /// uniquely identifies one artifact file.
    ///
    /// Returns `None` if the `raw_artifacts` table has not been built or holds
    /// no such row. The lookup rides the `(id, name)` scalar indexes
    /// [`crate::raw::build`] creates, so it is a point read, never a scan.
    pub async fn get_raw_artifact(&self, id: &str, name: &str) -> Result<Option<RawArtifact>> {
        if !has_table(&self.db, RAW_ARTIFACTS_TABLE).await? {
            return Ok(None);
        }
        let table = self
            .db
            .open_table(RAW_ARTIFACTS_TABLE)
            .execute()
            .await
            .context("opening raw_artifacts table")?;
        let batches: Vec<RecordBatch> = table
            .query()
            .only_if(format!("id = {} AND name = {}", sql_str(id), sql_str(name)))
            .select(Select::columns(&["id", "name", "media_type", "md5", "size", "data"]))
            .limit(1)
            .execute()
            .await
            .context("executing raw-artifact lookup")?
            .try_collect()
            .await
            .context("collecting raw artifact")?;
        raw_artifact_from_batches(&batches)
    }

    /// Stream every artifact whose contents are plain text ([`TEXT_FILTER`]) as
    /// record batches of `(id, name, media_type)`, the columns
    /// [`text_refs_from_batch`] decodes — the artifacts the hybrid text index
    /// reads and embeds.
    ///
    /// Ordered by `(id, name)` so all artifacts of a document arrive
    /// contiguously, the doc-boundary grouping the full-text build's resume
    /// invariant depends on (a `doc_id` in the chunks table is always complete).
    /// The order is enforced by the engine rather than assumed from physical
    /// layout: a plain scan's row order is not a LanceDB guarantee, and the
    /// update path re-appends a changed document's artifacts at the table's end,
    /// so insertion order stops being document order after any update. Returned
    /// as a stream so the reader pulls batches lazily instead of materializing
    /// the whole listing, which is multiple gigabytes at full-corpus scale.
    pub(crate) async fn text_artifacts_stream(&self) -> Result<SendableRecordBatchStream> {
        self.artifacts
            .query()
            .only_if(TEXT_FILTER)
            .select(Select::columns(&["id", "name", "media_type"]))
            .order_by(Some(vec![
                ColumnOrdering::asc_nulls_first("id".to_string()),
                ColumnOrdering::asc_nulls_first("name".to_string()),
            ]))
            .execute_with_options(scan_options())
            .await
            .context("executing text-artifacts query")
    }

    /// Count the plain-text artifacts ([`TEXT_FILTER`]) without materializing
    /// them — the progress denominator for the streamed full-text pass.
    pub(crate) async fn text_count(&self) -> Result<u64> {
        Ok(self
            .artifacts
            .count_rows(Some(TEXT_FILTER.to_string()))
            .await
            .context("counting text artifacts")? as u64)
    }

    /// Stream every artifact whose contents are *not* plain text (the
    /// complement of [`Index::text_artifacts_stream`]) as record batches of
    /// `(id, name, media_type, md5, size)`, the columns [`raw_refs_from_batch`]
    /// decodes into [`RawArtifactRef`]s.
    ///
    /// Returned as a stream, not a materialized `Vec`, so raw storage can pull
    /// candidates batch-by-batch: over >24M artifacts the full list would be
    /// gigabytes resident. Raw storage checkpoints each `(id, name)`
    /// independently, so — unlike the full-text build — it needs no document
    /// grouping and the scan can stream unordered.
    pub(crate) async fn nontext_artifacts_stream(&self) -> Result<SendableRecordBatchStream> {
        self.artifacts
            .query()
            .only_if(NONTEXT_FILTER)
            .select(Select::columns(&["id", "name", "media_type", "md5", "size"]))
            .execute_with_options(scan_options())
            .await
            .context("executing non-text-artifacts query")
    }

    /// Count the non-text artifacts ([`NONTEXT_FILTER`]) without materializing
    /// them — the progress denominator for a streamed raw-storage pass.
    pub(crate) async fn nontext_count(&self) -> Result<u64> {
        Ok(self
            .artifacts
            .count_rows(Some(NONTEXT_FILTER.to_string()))
            .await
            .context("counting non-text artifacts")? as u64)
    }

    /// Delete every raw-artifact row whose `id` (document id) is in `doc_ids`
    /// from the `raw_artifacts` table, so stale bytes for changed/redacted
    /// documents are never returned. A later incremental `ingest --store-raw`
    /// re-fetches the affected artifacts. Returns the number of rows removed
    /// (0 if the table does not exist yet).
    pub(crate) async fn delete_raw_for(&self, doc_ids: &[String]) -> Result<u64> {
        if doc_ids.is_empty() {
            return Ok(0);
        }
        if !has_table(&self.db, RAW_ARTIFACTS_TABLE).await? {
            return Ok(0);
        }
        let raws = self
            .db
            .open_table(RAW_ARTIFACTS_TABLE)
            .execute()
            .await
            .context("opening raw_artifacts table")?;
        delete_in(&raws, "id", doc_ids).await
    }
}

/// A non-text artifact selected for raw storage, before its bytes are fetched.
#[derive(Debug, Clone)]
pub(crate) struct RawArtifactRef {
    pub id: String,
    pub name: String,
    pub media_type: Option<String>,
    pub md5: Option<String>,
    pub size: Option<i64>,
}

/// Decode one batch from [`Index::nontext_artifacts_stream`] (columns
/// `id, name, media_type, md5, size`) into [`RawArtifactRef`]s. The batch is
/// bounded by the scan's read size, so the returned `Vec` stays small.
pub(crate) fn raw_refs_from_batch(batch: &RecordBatch) -> Result<Vec<RawArtifactRef>> {
    let ids = str_col(batch, "id")?;
    let names = str_col(batch, "name")?;
    let media = str_col(batch, "media_type")?;
    let md5 = str_col(batch, "md5")?;
    let size = i64_col(batch, "size")?;
    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        out.push(RawArtifactRef {
            id: ids.value(row).to_string(),
            name: names.value(row).to_string(),
            media_type: opt_at(media, row),
            md5: opt_at(md5, row),
            size: if size.is_null(row) {
                None
            } else {
                Some(size.value(row))
            },
        });
    }
    Ok(out)
}

/// Decode one batch from [`Index::text_artifacts_stream`] (columns
/// `id, name, media_type`) into owned `(id, name, media_type)` tuples. The
/// batch is bounded by the scan's read size, so the returned `Vec` stays small.
pub(crate) fn text_refs_from_batch(
    batch: &RecordBatch,
) -> Result<Vec<(String, String, Option<String>)>> {
    let ids = str_col(batch, "id")?;
    let names = str_col(batch, "name")?;
    let media = str_col(batch, "media_type")?;
    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        out.push((
            ids.value(row).to_string(),
            names.value(row).to_string(),
            opt_at(media, row),
        ));
    }
    Ok(out)
}

/// Artifact byte totals for the `stats` view, split into the full-text and raw
/// (non-text) sets, with logical (referenced) vs real (stored) figures.
///
/// See [`Index::store_sizes`]. `raw_real_*` are `None` until the
/// `raw_artifacts` table has been built.
#[derive(Debug, Clone, Copy, Default)]
pub struct StoreSizes {
    pub text_logical_count: u64,
    pub text_logical_bytes: u64,
    pub text_real_count: u64,
    pub text_real_bytes: u64,
    pub raw_logical_count: u64,
    pub raw_logical_bytes: u64,
    pub raw_real_count: Option<u64>,
    pub raw_real_bytes: Option<u64>,
}

/// Quote a string as a SQL literal for a LanceDB filter, escaping `'`.
pub(crate) fn sql_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// True when `db` contains a table named `name`. Centralises the
/// list-then-membership-test guard the existence-checked paths share.
pub(crate) async fn has_table(db: &Connection, name: &str) -> Result<bool> {
    let names = db.table_names().execute().await.context("listing tables")?;
    Ok(contains_table(&names, name))
}

/// True when a table named `name` is in an already-listed set of table names.
/// Used by the paths that list once and test membership several times, where
/// re-listing per check (as [`has_table`] does) would be wasteful.
pub(crate) fn contains_table(names: &[String], name: &str) -> bool {
    names.iter().any(|n| n == name)
}

/// Update `cur` to `candidate` when it is the greater string, tracking a
/// running maximum (used for the high-water modified-date watermark).
pub(crate) fn track_max(cur: &mut Option<String>, candidate: String) {
    if cur.as_deref().is_none_or(|c| candidate.as_str() > c) {
        *cur = Some(candidate);
    }
}

/// Delete every row of `table` whose `column` value is in `ids`, issued in
/// bounded `id IN (...)` batches. Returns the total rows removed.
pub(crate) async fn delete_in(table: &Table, column: &str, ids: &[String]) -> Result<u64> {
    let mut total = 0u64;
    for group in ids.chunks(IN_LIST_CHUNK) {
        let list = group.iter().map(|i| sql_str(i)).collect::<Vec<_>>().join(", ");
        let predicate = format!("{column} IN ({list})");
        let res = table
            .delete(predicate.as_str())
            .await
            .with_context(|| format!("deleting from {column} IN (...)"))?;
        total += res.num_deleted_rows;
    }
    Ok(total)
}

/// Read the incremental-update watermark from [`INGEST_STATE_TABLE`], or `None`
/// if the table is absent or empty.
pub(crate) async fn read_watermark(db: &Connection) -> Result<Option<String>> {
    if !has_table(db, INGEST_STATE_TABLE).await? {
        return Ok(None);
    }
    let table = db
        .open_table(INGEST_STATE_TABLE)
        .execute()
        .await
        .context("opening ingest-state table")?;
    let batches: Vec<RecordBatch> = table
        .query()
        .select(Select::columns(&[WATERMARK_COL]))
        .execute()
        .await
        .context("querying ingest-state table")?
        .try_collect()
        .await
        .context("collecting ingest-state rows")?;
    let mut max: Option<String> = None;
    for batch in &batches {
        let col = str_col(batch, WATERMARK_COL)?;
        for i in 0..col.len() {
            if !col.is_null(i) {
                track_max(&mut max, col.value(i).to_string());
            }
        }
    }
    Ok(max)
}

/// Persist `watermark` as the single row of [`INGEST_STATE_TABLE`], replacing
/// any prior value. Written last by an ingest/update so a crash mid-apply never
/// advances the watermark past un-applied work.
pub(crate) async fn write_watermark(db: &Connection, watermark: &str) -> Result<()> {
    if has_table(db, INGEST_STATE_TABLE).await? {
        db.drop_table(INGEST_STATE_TABLE, &[])
            .await
            .context("dropping stale ingest-state table")?;
    }
    let schema = Arc::new(Schema::new(vec![Field::new(
        WATERMARK_COL,
        DataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec![watermark]))])
        .context("building ingest-state row")?;
    db.create_table(INGEST_STATE_TABLE, vec![batch])
        .execute()
        .await
        .context("creating ingest-state table")?;
    Ok(())
}

/// Map every row across `batches` (selected with [`DOC_COLS`]) into [`Document`]s.
fn collect_documents(batches: &[RecordBatch]) -> Result<Vec<Document>> {
    let mut out = Vec::new();
    for batch in batches {
        let id = str_col(batch, "id")?;
        let bn = str_col(batch, "bn")?;
        let title = str_col(batch, "title")?;
        let industry = str_col(batch, "industry")?;
        let collection = str_col(batch, "collection")?;
        let genre = str_col(batch, "genre")?;
        let date_sent = str_col(batch, "date_sent")?;
        let date_received = str_col(batch, "date_received")?;
        let topic = str_col(batch, "topic")?;
        let description = str_col(batch, "description")?;
        let keywords = str_col(batch, "keywords")?;
        let conversation = str_col(batch, "conversation")?;
        let count = i64_col(batch, "artifact_count")?;
        for row in 0..batch.num_rows() {
            out.push(Document {
                id: id.value(row).to_string(),
                bn: opt_at(bn, row),
                title: opt_at(title, row),
                industry: opt_at(industry, row),
                collection: opt_at(collection, row),
                genre: opt_at(genre, row),
                date_sent: opt_at(date_sent, row),
                date_received: opt_at(date_received, row),
                topic: opt_at(topic, row),
                description: opt_at(description, row),
                keywords: opt_at(keywords, row),
                conversation: opt_at(conversation, row),
                custodian: list_at(batch, "custodian", row)?,
                authors: list_at(batch, "authors", row)?,
                recipients: list_at(batch, "recipients", row)?,
                cc: list_at(batch, "cc", row)?,
                attachments: list_at(batch, "attachments", row)?,
                related: list_at(batch, "related", row)?,
                mentions: list_at(batch, "mentions", row)?,
                artifact_types: list_at(batch, "artifact_types", row)?,
                artifact_count: count.value(row).max(0) as u64,
            });
        }
    }
    Ok(out)
}

/// Map every row across `batches` into [`Artifact`]s (unsorted).
fn collect_artifacts(batches: &[RecordBatch]) -> Result<Vec<Artifact>> {
    let mut out = Vec::new();
    for batch in batches {
        let ids = str_col(batch, "id")?;
        let names = str_col(batch, "name")?;
        let media = str_col(batch, "media_type")?;
        let size = i64_col(batch, "size")?;
        let md5 = str_col(batch, "md5")?;
        for row in 0..batch.num_rows() {
            out.push(Artifact {
                document_id: ids.value(row).to_string(),
                name: names.value(row).to_string(),
                media_type: opt_at(media, row),
                size: if size.is_null(row) {
                    None
                } else {
                    Some(size.value(row).max(0) as u64)
                },
                md5: opt_at(md5, row),
            });
        }
    }
    Ok(out)
}

/// Execution options for the streamed scans: widen the output batch to
/// [`SCAN_BATCH_ROWS`] so the per-batch read drain in their consumers hits
/// boundaries rarely.
fn scan_options() -> QueryExecutionOptions {
    let mut opts = QueryExecutionOptions::default();
    opts.max_batch_length = SCAN_BATCH_ROWS;
    opts
}

/// Downcast a named column to a [`StringArray`].
pub(crate) fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("result missing column {name}"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("column {name} is not a string column"))
}

/// Downcast a named column to a [`LargeBinaryArray`].
fn bin_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a LargeBinaryArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("result missing column {name}"))?
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| anyhow::anyhow!("column {name} is not a large-binary column"))
}

/// Decode the first row of a `raw_artifacts` point-lookup result (columns
/// `id, name, media_type, md5, size, data`) into a [`RawArtifact`], or `None`
/// when the lookup matched no row.
fn raw_artifact_from_batches(batches: &[RecordBatch]) -> Result<Option<RawArtifact>> {
    for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let ids = str_col(batch, "id")?;
        let names = str_col(batch, "name")?;
        let media = str_col(batch, "media_type")?;
        let md5 = str_col(batch, "md5")?;
        let size = i64_col(batch, "size")?;
        let data = bin_col(batch, "data")?;
        return Ok(Some(RawArtifact {
            document_id: ids.value(0).to_string(),
            name: names.value(0).to_string(),
            media_type: opt_at(media, 0),
            md5: opt_at(md5, 0),
            size: if size.is_null(0) {
                None
            } else {
                Some(size.value(0).max(0) as u64)
            },
            data: data.value(0).to_vec(),
        }));
    }
    Ok(None)
}

/// Downcast a named column to an [`Int64Array`].
pub(crate) fn i64_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("result missing column {name}"))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow::anyhow!("column {name} is not an int64 column"))
}

/// Downcast a named column to an [`Int32Array`].
pub(crate) fn i32_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int32Array> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("result missing column {name}"))?
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| anyhow::anyhow!("column {name} is not an int32 column"))
}

/// Read a nullable string cell, mapping SQL null and the empty string to `None`.
fn opt_at(col: &StringArray, row: usize) -> Option<String> {
    if col.is_null(row) {
        return None;
    }
    let v = col.value(row);
    if v.is_empty() { None } else { Some(v.to_string()) }
}

/// Read a `List<Utf8>` cell into a `Vec<String>`, dropping null/empty elements.
fn list_at(batch: &RecordBatch, name: &str, row: usize) -> Result<Vec<String>> {
    let list = batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("result missing column {name}"))?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow::anyhow!("column {name} is not a list column"))?;
    if list.is_null(row) {
        return Ok(Vec::new());
    }
    let values = list.value(row);
    let strs = values
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("column {name} is not a list of strings"))?;
    let mut out = Vec::with_capacity(strs.len());
    for i in 0..strs.len() {
        if !strs.is_null(i) {
            let v = strs.value(i);
            if !v.is_empty() {
                out.push(v.to_string());
            }
        }
    }
    Ok(out)
}
