//! Arbitrary read-only SQL over the LanceDB-backed index.
//!
//! [`Index::run_sql`] lets a caller run ad-hoc `SELECT`-style queries against
//! the `documents`, `artifacts`, and (when present) `chunks` tables using
//! DataFusion. The tables are exposed to a fresh, per-call [`SessionContext`]
//! via LanceDB's [`BaseTableAdapter`]; the context registers nothing else, so
//! there are no UDFs, catalogs, or filesystem sources reachable from the SQL.
//! Safety rests on two layers:
//!
//! 1. The session context only knows about the read-only Lance tables. There
//!    is no write path, no external file/network function, and no extension
//!    loading reachable from a query.
//! 2. [`validate_sql`] parses the input with DataFusion's SQL parser and
//!    restricts queries to a single, read-only statement so accidental or
//!    malformed input fails fast with a clear message.

use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::RecordBatch;
use arrow::json::WriterBuilder;
use arrow::json::writer::JsonArray;
use futures::TryStreamExt;
use datafusion::catalog::TableProvider;
use datafusion::prelude::SessionContext;
use datafusion::sql::parser::{DFParser, Statement as DfStatement};
use datafusion::sql::sqlparser::ast::{SetExpr, Statement as SqlStatement};
use lancedb::Table;
use lancedb::table::datafusion::BaseTableAdapter;
use serde_json::value::RawValue;

use crate::index::{ARTIFACTS_TABLE, CHUNKS_TABLE, DOCUMENTS_TABLE, Index};
use crate::model::{ColumnInfo, SqlQueryResult, TableSchema};

/// Validate that `sql` is a single, read-only statement.
///
/// Returns `Ok(())` when the query may run, or `Err(message)` describing why it
/// was rejected. The query is parsed with DataFusion's own SQL parser so the
/// check inspects the statement's AST rather than guessing from leading
/// keywords; this is layered on top of a context that only exposes the
/// read-only Lance tables, and is intentionally conservative.
pub fn validate_sql(sql: &str) -> Result<(), String> {
    if sql.trim().is_empty() {
        return Err("empty query".to_string());
    }

    let statements =
        DFParser::parse_sql(sql).map_err(|e| format!("could not parse query: {e}"))?;

    let statement = match statements.len() {
        0 => return Err("empty query".to_string()),
        1 => &statements[0],
        n => return Err(format!("only a single statement is allowed, found {n}")),
    };

    if is_read_only(statement) {
        Ok(())
    } else {
        Err("statement is not allowed; only read-only queries \
             (SELECT, WITH, DESCRIBE, EXPLAIN) are permitted"
            .to_string())
    }
}

/// True if `statement` cannot mutate data, schema, or session state.
///
/// `EXPLAIN` is unwrapped and its inner statement is checked, because
/// `EXPLAIN ANALYZE` actually executes the wrapped statement.
fn is_read_only(statement: &DfStatement) -> bool {
    match statement {
        DfStatement::Explain(explain) => is_read_only(&explain.statement),
        // A query body can itself be a write (e.g. `WITH cte AS (..) INSERT
        // ..` parses as a query whose body is an INSERT), so inspect it.
        DfStatement::Statement(inner) => match inner.as_ref() {
            SqlStatement::Query(query) => set_expr_is_read_only(&query.body),
            _ => false,
        },
        _ => false,
    }
}

/// True if a query body contains no write/DML node.
///
/// `SELECT`, `VALUES`, and `TABLE` read; nested queries and set operations are
/// checked recursively; `INSERT`/`UPDATE`/`DELETE`/`MERGE` bodies are writes.
fn set_expr_is_read_only(body: &SetExpr) -> bool {
    match body {
        SetExpr::Select(_) | SetExpr::Values(_) | SetExpr::Table(_) => true,
        SetExpr::Query(query) => set_expr_is_read_only(&query.body),
        SetExpr::SetOperation { left, right, .. } => {
            set_expr_is_read_only(left) && set_expr_is_read_only(right)
        }
        _ => false,
    }
}

impl Index {
    /// Run a read-only SQL query, returning at most `max_rows` rows.
    ///
    /// The query is validated by [`validate_sql`] first. Any rejection or
    /// DataFusion execution error is captured in [`SqlQueryResult::error`]
    /// rather than returned as `Err`, so the model can read the message and
    /// retry.
    pub async fn run_sql(&self, sql: &str, max_rows: usize) -> SqlQueryResult {
        if let Err(msg) = validate_sql(sql) {
            return SqlQueryResult::error(msg);
        }
        match self.run_sql_inner(sql, max_rows).await {
            Ok(result) => result,
            Err(e) => SqlQueryResult::error(e.to_string()),
        }
    }

    /// Execute the (already validated) query, mapping rows to JSON.
    async fn run_sql_inner(&self, sql: &str, max_rows: usize) -> Result<SqlQueryResult> {
        let ctx = self.session_context().await?;
        let df = ctx.sql(sql).await.context("planning query")?;
        let mut stream = df.execute_stream().await.context("executing query")?;

        let columns: Vec<String> = stream
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();

        let mut batches: Vec<RecordBatch> = Vec::new();
        let mut row_count = 0usize;
        let mut truncated = false;
        while let Some(batch) = stream.try_next().await.context("reading results")? {
            if batch.num_rows() == 0 {
                continue;
            }
            if row_count >= max_rows {
                // The cap is already met but the stream still has rows.
                truncated = true;
                break;
            }
            // Trim the batch so the result never exceeds `max_rows`; dropping
            // any rows means the result is truncated.
            let remaining = max_rows - row_count;
            let batch = if batch.num_rows() > remaining {
                truncated = true;
                batch.slice(0, remaining)
            } else {
                batch
            };
            row_count += batch.num_rows();
            batches.push(batch);
        }

        Ok(SqlQueryResult {
            columns,
            rows: batches_to_json(&batches)?,
            row_count,
            truncated,
            error: None,
        })
    }

    /// Describe the queryable tables (`documents`, `artifacts`, and `chunks`
    /// when present) for the caller.
    pub async fn describe_schema(&self) -> Result<Vec<TableSchema>> {
        let mut out = Vec::new();
        let mut tables: Vec<(&str, &Table)> = vec![
            (DOCUMENTS_TABLE, &self.documents),
            (ARTIFACTS_TABLE, &self.artifacts),
        ];
        let chunks = self.open_chunks().await?;
        if let Some(t) = &chunks {
            tables.push((CHUNKS_TABLE, t));
        }

        for (name, table) in tables {
            let provider = lance_provider(table).await?;
            let columns = provider
                .schema()
                .fields()
                .iter()
                .map(|f| ColumnInfo {
                    name: f.name().clone(),
                    type_: f.data_type().to_string(),
                })
                .collect();
            out.push(TableSchema {
                table: name.to_string(),
                columns,
            });
        }
        Ok(out)
    }

    /// Build a fresh DataFusion context with the Lance tables registered.
    async fn session_context(&self) -> Result<SessionContext> {
        let ctx = SessionContext::new();
        ctx.register_table(DOCUMENTS_TABLE, lance_provider(&self.documents).await?)
            .context("registering documents")?;
        ctx.register_table(ARTIFACTS_TABLE, lance_provider(&self.artifacts).await?)
            .context("registering artifacts")?;
        if let Some(chunks) = self.open_chunks().await? {
            ctx.register_table(CHUNKS_TABLE, lance_provider(&chunks).await?)
                .context("registering chunks")?;
        }
        Ok(ctx)
    }

    /// Open the optional `chunks` table if the full-text index has been built.
    async fn open_chunks(&self) -> Result<Option<Table>> {
        if crate::index::has_table(&self.db, CHUNKS_TABLE).await? {
            let table = self
                .db
                .open_table(CHUNKS_TABLE)
                .execute()
                .await
                .context("opening chunks table")?;
            Ok(Some(table))
        } else {
            Ok(None)
        }
    }
}

/// Build a DataFusion table provider backed by a LanceDB table.
///
/// Uses LanceDB's [`BaseTableAdapter`] so scans go through LanceDB's query
/// planner: `WHERE` filters are pushed down (and can hit scalar indices) and
/// full-text predicates can use the FTS index.
async fn lance_provider(table: &Table) -> Result<Arc<dyn TableProvider>> {
    let adapter = BaseTableAdapter::try_new(table.base_table().clone())
        .await
        .context("building table provider")?;
    Ok(Arc::new(adapter))
}

/// Serialize the result batches into a single JSON array of row objects using
/// Arrow's JSON writer.
///
/// Each row becomes an object keyed by column name (lists/structs become JSON
/// arrays/objects). `explicit_nulls` keeps null cells in the output so every
/// row carries the full set of columns. The bytes Arrow produces are wrapped in
/// a [`RawValue`] so they pass through into the response verbatim, without being
/// parsed into intermediate values and re-serialized.
fn batches_to_json(batches: &[RecordBatch]) -> Result<Box<RawValue>> {
    let mut writer = WriterBuilder::new()
        .with_explicit_nulls(true)
        .build::<_, JsonArray>(Vec::new());
    for batch in batches {
        writer.write(batch).context("serializing results to JSON")?;
    }
    writer.finish().context("finishing JSON output")?;
    let bytes = writer.into_inner();
    let json = String::from_utf8(bytes).context("results were not valid UTF-8")?;
    RawValue::from_string(json).context("wrapping JSON results")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_read_only_statements() {
        assert!(validate_sql("SELECT * FROM documents").is_ok());
        assert!(validate_sql("  select count(*) from artifacts  ").is_ok());
        assert!(validate_sql("WITH t AS (SELECT 1) SELECT * FROM t").is_ok());
        assert!(validate_sql("EXPLAIN SELECT 1").is_ok());
        assert!(validate_sql("VALUES (1), (2)").is_ok());
        assert!(validate_sql("SELECT 1 UNION ALL SELECT 2").is_ok());
        assert!(validate_sql("SELECT 1;").is_ok());
        assert!(validate_sql("(SELECT 1)").is_ok());
    }

    #[test]
    fn rejects_writes_and_ddl() {
        assert!(validate_sql("INSERT INTO documents VALUES (1)").is_err());
        assert!(validate_sql("UPDATE documents SET id = '1'").is_err());
        assert!(validate_sql("DELETE FROM documents").is_err());
        assert!(validate_sql("DROP TABLE documents").is_err());
        assert!(validate_sql("CREATE TABLE t (x INT)").is_err());
        assert!(validate_sql("COPY documents TO '/tmp/x.csv'").is_err());
    }

    #[test]
    fn rejects_writes_hidden_in_a_query_body() {
        // A leading `WITH` does not make a statement read-only: the query body
        // can be an INSERT. The parser-based check looks past the keyword.
        assert!(
            validate_sql("WITH t AS (SELECT 1) INSERT INTO documents SELECT * FROM t").is_err()
        );
        // EXPLAIN ANALYZE executes its inner statement, so a write there is
        // still rejected.
        assert!(validate_sql("EXPLAIN ANALYZE COPY documents TO '/tmp/x.csv'").is_err());
    }

    #[test]
    fn rejects_multiple_statements() {
        assert!(validate_sql("SELECT 1; SELECT 2").is_err());
        assert!(validate_sql("SELECT 1; DROP TABLE documents").is_err());
        assert!(validate_sql("SELECT 1;DELETE FROM documents").is_err());
    }

    #[test]
    fn semicolon_inside_string_is_allowed() {
        assert!(validate_sql("SELECT 'a; b' AS c").is_ok());
        assert!(validate_sql("SELECT * FROM documents WHERE title = 'x;y'").is_ok());
    }

    #[test]
    fn rejects_empty() {
        assert!(validate_sql("").is_err());
        assert!(validate_sql("   ").is_err());
    }

    #[test]
    fn rejects_unparseable_input() {
        assert!(validate_sql("SELECT * FROM (").is_err());
        assert!(validate_sql("not sql at all").is_err());
    }
}
