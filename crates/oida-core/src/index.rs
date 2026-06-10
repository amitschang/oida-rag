//! DuckDB-backed access to the OIDA index.
//!
//! The 2.7 GB parquet has ~24M artifact-level rows. Scanning it with `ILIKE`
//! on every query would not be interactive, so we build a persistent DuckDB
//! cache once: a deduplicated document-level `documents` table plus a thin
//! `artifacts` table, both indexed. All queries run against that cache.

use std::sync::{Mutex, MutexGuard};

use anyhow::{Context, bail};
use duckdb::{Connection, Row, ToSql};

use crate::config::Config;
use crate::model::{Artifact, Document};
use crate::schema;

/// Field separator used to flatten DuckDB list columns into a single string.
/// ASCII unit separator (0x1F) does not occur in the textual metadata.
const SEP: char = '\u{1f}';

/// Column projection that maps the `documents` table onto [`Document`].
/// List columns are flattened with [`SEP`] and split back in [`row_to_document`].
pub(crate) const DOC_COLS: &str = "id, bn, title, industry, collection, genre, \
     date_sent, date_received, topic, description, keywords, conversation, \
     array_to_string(custodian, chr(31)) AS custodian, \
     array_to_string(authors, chr(31)) AS authors, \
     array_to_string(recipients, chr(31)) AS recipients, \
     array_to_string(cc, chr(31)) AS cc, \
     array_to_string(attachments, chr(31)) AS attachments, \
     array_to_string(related, chr(31)) AS related, \
     array_to_string(mentions, chr(31)) AS mentions, \
     array_to_string(artifact_types, chr(31)) AS artifact_types, \
     artifact_count";

/// Handle to the cached OIDA index.
pub struct Index {
    /// DuckDB connection. Wrapped in a mutex because queries are synchronous
    /// and short; the MCP server may invoke tools from multiple threads.
    conn: Mutex<Connection>,
}

impl Index {
    /// Open the cache, returning an error if it has not been built yet.
    pub fn open(config: &Config) -> anyhow::Result<Self> {
        if !config.cache_path.exists() {
            bail!(
                "cache {} not found; build it first (build_cache)",
                config.cache_path.display()
            );
        }
        // Open the cache hardened: read-only access, no external file/network
        // access (blocks COPY TO, read_csv/read_parquet, httpfs, etc.), and a
        // locked configuration so a query cannot re-enable those at runtime.
        // This is the safety boundary that makes the arbitrary-SQL tool safe.
        let db_config = duckdb::Config::default()
            .access_mode(duckdb::AccessMode::ReadOnly)?
            .enable_external_access(false)?
            .with("lock_configuration", "true")?;
        let conn = Connection::open_with_flags(&config.cache_path, db_config)
            .with_context(|| format!("opening cache {}", config.cache_path.display()))?;
        let has_tables: i64 = conn.query_row(
            "SELECT count(*) FROM information_schema.tables \
             WHERE table_name IN ('documents', 'artifacts')",
            [],
            |r| r.get(0),
        )?;
        if has_tables != 2 {
            bail!(
                "cache {} is missing the documents/artifacts tables; rebuild it",
                config.cache_path.display()
            );
        }
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Build (or rebuild) the persistent cache from the parquet.
    ///
    /// This is a one-time, relatively expensive operation: it deduplicates the
    /// artifact rows to one row per document and indexes the result.
    pub fn build_cache(config: &Config, force: bool) -> anyhow::Result<()> {
        if config.cache_path.exists() {
            if force {
                std::fs::remove_file(&config.cache_path).ok();
            } else {
                bail!(
                    "cache {} already exists; pass force to rebuild",
                    config.cache_path.display()
                );
            }
        }
        if !config.parquet_path.exists() {
            bail!("parquet {} not found", config.parquet_path.display());
        }

        let conn = Connection::open(&config.cache_path)
            .with_context(|| format!("creating cache {}", config.cache_path.display()))?;
        schema::validate_parquet(&conn, &config.parquet_path)?;

        // Tune for a large one-time build: allow spilling to disk and don't
        // pay to preserve row order.
        conn.execute_batch(
            "PRAGMA threads=4;
             PRAGMA memory_limit='20GB';
             PRAGMA temp_directory='oida-cache.tmp';
             PRAGMA preserve_insertion_order=false;",
        )
        .ok();

        let parquet = config.parquet_path.to_string_lossy().replace('\'', "''");

        // Build the thin artifact table first; the document summary reuses it.
        tracing::info!("building artifact table...");
        conn.execute_batch(&format!(
            "CREATE TABLE artifacts AS
             SELECT
               id,
               artifact_name                 AS name,
               artifact_mediaType            AS media_type,
               CAST(artifact_size AS BIGINT) AS size,
               artifact_md5                  AS md5
             FROM read_parquet('{parquet}');"
        ))
        .context("building artifacts table")?;

        // Deduplicate to one document per id. Metadata is identical across an
        // id's artifact rows, so we take a representative row via a window
        // (spill-friendly) rather than a memory-heavy list aggregation, and
        // join a cheap per-id artifact summary.
        tracing::info!("building document table (deduplicating artifact rows)...");
        conn.execute_batch(&format!(
            "CREATE TABLE documents AS
             WITH meta AS (
               SELECT
                 id,
                 bn,
                 coalesce(ti, filename) AS title,
                 industry,
                 collection,
                 genre,
                 datesent      AS date_sent,
                 datereceived  AS date_received,
                 topic,
                 \"desc\"      AS description,
                 kw            AS keywords,
                 conversation,
                 custodian,
                 au            AS authors,
                 rc            AS recipients,
                 cc,
                 attachment    AS attachments,
                 related,
                 men           AS mentions
               FROM read_parquet('{parquet}')
               QUALIFY row_number() OVER (PARTITION BY id ORDER BY artifact_name) = 1
             ),
             summ AS (
               SELECT id,
                      list(DISTINCT media_type) AS artifact_types,
                      count(*)                  AS artifact_count
               FROM artifacts
               GROUP BY id
             )
             SELECT meta.*, summ.artifact_types, summ.artifact_count
             FROM meta JOIN summ USING (id);"
        ))
        .context("building documents table")?;

        tracing::info!("creating indexes...");
        // DuckDB's parallel ART index build can abort on very large tables;
        // single-threaded index creation is the reliable path here.
        conn.execute_batch("PRAGMA threads=1;").ok();
        conn.execute_batch(
            "CREATE UNIQUE INDEX idx_doc_id ON documents(id);
             CREATE INDEX idx_doc_bn        ON documents(bn);
             CREATE INDEX idx_doc_conv      ON documents(conversation);
             CREATE INDEX idx_art_id        ON artifacts(id);
             CREATE INDEX idx_art_name      ON artifacts(name);
             CREATE INDEX idx_art_media     ON artifacts(media_type);",
        )
        .context("creating indexes")?;

        let (docs, arts): (i64, i64) = conn.query_row(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM artifacts)",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        tracing::info!("cache built: {docs} documents, {arts} artifacts");
        Ok(())
    }

    /// Lock the underlying connection for sibling modules (search, graph).
    pub(crate) fn conn_lock(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock().expect("index mutex poisoned")
    }

    /// Run a `documents` query whose `tail` is appended after the projection
    /// (e.g. `"WHERE id = ?"`), mapping each row to a [`Document`].
    pub(crate) fn documents_query(
        &self,
        tail: &str,
        params: &[&dyn ToSql],
    ) -> anyhow::Result<Vec<Document>> {
        let sql = format!("SELECT {DOC_COLS} FROM documents {tail}");
        let conn = self.conn.lock().expect("index mutex poisoned");
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params, row_to_document)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Fetch a single document by its `id`.
    pub fn get_document_by_id(&self, id: &str) -> anyhow::Result<Option<Document>> {
        Ok(self
            .documents_query("WHERE id = ? LIMIT 1", &[&id])?
            .into_iter()
            .next())
    }

    /// Fetch a single document by its Bates number `bn`.
    pub fn get_document_by_bn(&self, bn: &str) -> anyhow::Result<Option<Document>> {
        Ok(self
            .documents_query("WHERE bn = ? LIMIT 1", &[&bn])?
            .into_iter()
            .next())
    }

    /// List the artifacts belonging to a document.
    pub fn get_artifacts(&self, id: &str) -> anyhow::Result<Vec<Artifact>> {
        let conn = self.conn.lock().expect("index mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, name, media_type, size, md5 FROM artifacts WHERE id = ? ORDER BY name",
        )?;
        let rows = stmt.query_map([id], |row| {
            Ok(Artifact {
                document_id: row.get(0)?,
                name: row.get(1)?,
                media_type: row.get(2)?,
                size: row.get::<_, Option<i64>>(3)?.map(|s| s.max(0) as u64),
                md5: row.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }
}

/// Split a flattened list column back into its elements.
fn split_list(s: Option<String>) -> Vec<String> {
    match s {
        Some(s) if !s.is_empty() => s
            .split(SEP)
            .filter(|p| !p.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

/// Map a row selected with [`DOC_COLS`] into a [`Document`].
pub(crate) fn row_to_document(row: &Row<'_>) -> duckdb::Result<Document> {
    Ok(Document {
        id: row.get(0)?,
        bn: row.get(1)?,
        title: row.get(2)?,
        industry: row.get(3)?,
        collection: row.get(4)?,
        genre: row.get(5)?,
        date_sent: row.get(6)?,
        date_received: row.get(7)?,
        topic: row.get(8)?,
        description: row.get(9)?,
        keywords: row.get(10)?,
        conversation: row.get(11)?,
        custodian: split_list(row.get(12)?),
        authors: split_list(row.get(13)?),
        recipients: split_list(row.get(14)?),
        cc: split_list(row.get(15)?),
        attachments: split_list(row.get(16)?),
        related: split_list(row.get(17)?),
        mentions: split_list(row.get(18)?),
        artifact_types: split_list(row.get(19)?),
        artifact_count: row.get::<_, i64>(20)?.max(0) as u64,
    })
}
