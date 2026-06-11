//! Arbitrary read-only SQL over the LanceDB-backed index.
//!
//! [`Index::run_sql`] lets a caller run ad-hoc `SELECT`-style queries against
//! the `documents`, `artifacts`, and (when present) `chunks` tables using
//! DataFusion. The tables are exposed to a fresh, per-call [`SessionContext`]
//! via Lance's [`LanceTableProvider`]; the context registers nothing else, so
//! there are no UDFs, catalogs, or filesystem sources reachable from the SQL.
//! Safety rests on two layers:
//!
//! 1. The session context only knows about the read-only Lance tables. There
//!    is no write path, no external file/network function, and no extension
//!    loading reachable from a query.
//! 2. [`validate_sql`] restricts queries to a single, read-only statement so
//!    accidental or malformed input fails fast with a clear message.

use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, LargeListArray, LargeStringArray, ListArray, StringArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use futures::TryStreamExt;
use lance::datafusion::LanceTableProvider;
use datafusion::catalog::TableProvider;
use datafusion::prelude::SessionContext;
use lancedb::Table;

use crate::index::{ARTIFACTS_TABLE, DOCUMENTS_TABLE, Index};
use crate::model::{ColumnInfo, SqlQueryResult, TableSchema};

/// Name of the optional full-text chunks table (created by `--full-text`).
const CHUNKS_TABLE: &str = "chunks";

/// Statement keywords permitted as the first token of a query. All are
/// read-only; anything else (INSERT, UPDATE, DELETE, CREATE, DROP, COPY,
/// INSERT INTO, ...) is rejected.
const ALLOWED_LEADING_KEYWORDS: &[&str] =
    &["select", "with", "describe", "explain", "show", "values"];

/// Validate that `sql` is a single, read-only statement.
///
/// Returns `Ok(())` when the query may run, or `Err(message)` describing why it
/// was rejected. This is a syntactic guard layered on top of a context that
/// only exposes the read-only Lance tables; it is intentionally conservative.
pub fn validate_sql(sql: &str) -> Result<(), String> {
    let trimmed = sql.trim();
    if trimmed.is_empty() {
        return Err("empty query".to_string());
    }

    // Reject multiple statements. Allow a single optional trailing semicolon,
    // but any other top-level `;` (outside string/identifier quotes) indicates
    // more than one statement.
    if has_multiple_statements(trimmed) {
        return Err("only a single statement is allowed".to_string());
    }

    // Check the leading keyword against the read-only allowlist.
    let first = trimmed
        .split(|c: char| c.is_whitespace() || c == '(')
        .find(|t| !t.is_empty())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !ALLOWED_LEADING_KEYWORDS.contains(&first.as_str()) {
        return Err(format!(
            "statement starting with `{first}` is not allowed; only read-only queries \
             (SELECT, WITH, DESCRIBE, EXPLAIN, SHOW, VALUES) are permitted"
        ));
    }

    Ok(())
}

/// True if `sql` contains more than one statement, ignoring a single optional
/// trailing semicolon and any `;` inside single quotes, double quotes, or
/// backtick-quoted identifiers.
fn has_multiple_statements(sql: &str) -> bool {
    let mut quote: Option<char> = None;
    let mut chars = sql.char_indices().peekable();
    while let Some((idx, c)) = chars.next() {
        match quote {
            Some(q) => {
                if c == q {
                    // Doubled quote is an escaped quote, not a terminator.
                    if chars.peek().map(|&(_, n)| n) == Some(q) {
                        chars.next();
                    } else {
                        quote = None;
                    }
                }
            }
            None => match c {
                '\'' | '"' | '`' => quote = Some(c),
                ';' => {
                    // A semicolon is fine only if nothing but whitespace
                    // follows it (a single trailing terminator).
                    if sql[idx + 1..].trim().is_empty() {
                        return false;
                    }
                    return true;
                }
                _ => {}
            },
        }
    }
    false
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
        let ncols = columns.len();

        let mut out: Vec<Vec<serde_json::Value>> = Vec::new();
        let mut truncated = false;
        'outer: while let Some(batch) = stream.try_next().await.context("reading results")? {
            for row in 0..batch.num_rows() {
                if out.len() >= max_rows {
                    truncated = true;
                    break 'outer;
                }
                let mut record = Vec::with_capacity(ncols);
                for col in 0..ncols {
                    record.push(array_to_json(batch.column(col).as_ref(), row));
                }
                out.push(record);
            }
        }

        Ok(SqlQueryResult {
            columns,
            row_count: out.len(),
            truncated,
            rows: out,
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
        let names = self
            .db
            .table_names()
            .execute()
            .await
            .context("listing tables")?;
        if !names.iter().any(|n| n == CHUNKS_TABLE) {
            return Ok(None);
        }
        let table = self
            .db
            .open_table(CHUNKS_TABLE)
            .execute()
            .await
            .context("opening chunks table")?;
        Ok(Some(table))
    }
}

/// Build a DataFusion table provider backed by a LanceDB table's dataset.
async fn lance_provider(table: &Table) -> Result<Arc<dyn TableProvider>> {
    let dataset = table
        .dataset()
        .context("table has no backing dataset")?
        .get()
        .await
        .context("loading dataset")?;
    Ok(Arc::new(LanceTableProvider::new(dataset, false, false)))
}

/// Convert one cell of an Arrow array into a [`serde_json::Value`].
///
/// Scalars map to their JSON equivalents; list/array types become JSON arrays.
/// Types without a natural JSON form fall back to a string rendering so a
/// result is always serializable.
fn array_to_json(array: &dyn Array, row: usize) -> serde_json::Value {
    use serde_json::Value as J;
    if array.is_null(row) {
        return J::Null;
    }
    match array.data_type() {
        DataType::Boolean => J::Bool(downcast::<BooleanArray>(array).value(row)),
        DataType::Int8 => J::from(downcast::<Int8Array>(array).value(row)),
        DataType::Int16 => J::from(downcast::<Int16Array>(array).value(row)),
        DataType::Int32 => J::from(downcast::<Int32Array>(array).value(row)),
        DataType::Int64 => J::from(downcast::<Int64Array>(array).value(row)),
        DataType::UInt8 => J::from(downcast::<UInt8Array>(array).value(row)),
        DataType::UInt16 => J::from(downcast::<UInt16Array>(array).value(row)),
        DataType::UInt32 => J::from(downcast::<UInt32Array>(array).value(row)),
        DataType::UInt64 => J::from(downcast::<UInt64Array>(array).value(row)),
        DataType::Float32 => json_from_f64(downcast::<Float32Array>(array).value(row) as f64),
        DataType::Float64 => json_from_f64(downcast::<Float64Array>(array).value(row)),
        DataType::Utf8 => J::String(downcast::<StringArray>(array).value(row).to_string()),
        DataType::LargeUtf8 => J::String(downcast::<LargeStringArray>(array).value(row).to_string()),
        DataType::List(_) => list_to_json(downcast::<ListArray>(array).value(row).as_ref()),
        DataType::LargeList(_) => {
            list_to_json(downcast::<LargeListArray>(array).value(row).as_ref())
        }
        DataType::FixedSizeList(_, _) => {
            list_to_json(downcast::<FixedSizeListArray>(array).value(row).as_ref())
        }
        // No lossless JSON form: render textually so output stays serializable.
        _ => format_fallback(array, row),
    }
}

/// Convert an entire (already-extracted) list element array into a JSON array.
fn list_to_json(values: &dyn Array) -> serde_json::Value {
    serde_json::Value::Array((0..values.len()).map(|i| array_to_json(values, i)).collect())
}

/// Render a cell as a string via Arrow's display formatter (fallback path).
fn format_fallback(array: &dyn Array, row: usize) -> serde_json::Value {
    match ArrayFormatter::try_new(array, &FormatOptions::default()) {
        Ok(formatter) => serde_json::Value::String(formatter.value(row).to_string()),
        Err(_) => serde_json::Value::Null,
    }
}

/// Downcast an `&dyn Array` to a concrete array type, panicking on mismatch.
///
/// The caller has already matched on `array.data_type()`, so a mismatch is a
/// programmer error rather than a runtime condition.
fn downcast<T: 'static>(array: &dyn Array) -> &T {
    array
        .as_any()
        .downcast_ref::<T>()
        .expect("array type matched its DataType")
}

/// Build a JSON number from an `f64`, falling back to `null` for non-finite
/// values (JSON cannot represent NaN/Infinity).
fn json_from_f64(n: f64) -> serde_json::Value {
    serde_json::Number::from_f64(n)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_read_only_statements() {
        assert!(validate_sql("SELECT * FROM documents").is_ok());
        assert!(validate_sql("  select count(*) from artifacts  ").is_ok());
        assert!(validate_sql("WITH t AS (SELECT 1) SELECT * FROM t").is_ok());
        assert!(validate_sql("DESCRIBE documents").is_ok());
        assert!(validate_sql("EXPLAIN SELECT 1").is_ok());
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
}
