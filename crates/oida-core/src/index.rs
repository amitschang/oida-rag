//! LanceDB-backed access to the OIDA index.
//!
//! The 2.7 GB parquet has ~24M artifact-level rows. Ingest deduplicates them
//! into a document-level `documents` table plus a thin `artifacts` table, both
//! stored in a single embedded LanceDB database and indexed (scalar + FTS).
//! All metadata queries run against that store. Text/vector search lives in the
//! sibling [`crate::hybrid`] module against the same database.

use anyhow::{Context, Result, bail};
use arrow::array::{Array, Int64Array, ListArray, RecordBatch, StringArray};
use futures::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{Connection, Table};

use crate::config::Config;
use crate::ingest;
use crate::model::{Artifact, Document};

/// Name of the document-level metadata table.
pub(crate) const DOCUMENTS_TABLE: &str = "documents";
/// Name of the per-artifact table.
pub(crate) const ARTIFACTS_TABLE: &str = "artifacts";

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
        let names = db.table_names().execute().await.context("listing tables")?;
        let have = |n: &str| names.iter().any(|t| t == n);
        if !have(DOCUMENTS_TABLE) || !have(ARTIFACTS_TABLE) {
            bail!(
                "index at {} is missing the documents/artifacts tables; ingest it \
                 (oida-cli ingest)",
                config.lance_path.display()
            );
        }
        let documents = db
            .open_table(DOCUMENTS_TABLE)
            .execute()
            .await
            .context("opening documents table")?;
        let artifacts = db
            .open_table(ARTIFACTS_TABLE)
            .execute()
            .await
            .context("opening artifacts table")?;
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
        match db.table_names().execute().await {
            Ok(names) => names.iter().any(|n| n == DOCUMENTS_TABLE),
            Err(_) => false,
        }
    }

    /// Ingest document/artifact metadata from the parquet into LanceDB.
    pub async fn ingest_metadata(config: &Config, force: bool) -> Result<ingest::MetadataStats> {
        ingest::ingest_metadata(config, force).await
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

    /// List every artifact whose contents are plain text (OCR output or
    /// `text/plain`), as `(document_id, name, media_type)` tuples.
    ///
    /// These are the artifacts the hybrid text index can read and embed.
    pub async fn text_artifacts(&self) -> Result<Vec<(String, String, Option<String>)>> {
        let batches: Vec<RecordBatch> = self
            .artifacts
            .query()
            .only_if("media_type = 'text/plain' OR lower(name) LIKE '%.ocr'")
            .select(Select::columns(&["id", "name", "media_type"]))
            .execute()
            .await
            .context("executing text-artifacts query")?
            .try_collect()
            .await
            .context("collecting text artifacts")?;

        let mut out = Vec::new();
        for batch in &batches {
            let ids = str_col(batch, "id")?;
            let names = str_col(batch, "name")?;
            let media = str_col(batch, "media_type")?;
            for row in 0..batch.num_rows() {
                out.push((
                    ids.value(row).to_string(),
                    names.value(row).to_string(),
                    opt_at(media, row),
                ));
            }
        }
        out.sort();
        Ok(out)
    }
}

/// Quote a string as a SQL literal for a LanceDB filter, escaping `'`.
pub(crate) fn sql_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
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

/// Downcast a named column to a [`StringArray`].
pub(crate) fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("result missing column {name}"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("column {name} is not a string column"))
}

/// Downcast a named column to an [`Int64Array`].
fn i64_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("result missing column {name}"))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow::anyhow!("column {name} is not an int64 column"))
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
