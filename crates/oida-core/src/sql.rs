//! Arbitrary read-only SQL over the cache.
//!
//! [`Index::run_sql`] lets a caller run ad-hoc `SELECT`-style queries against
//! the `documents` and `artifacts` tables. Safety rests on two layers:
//!
//! 1. The serving connection is opened read-only with external access disabled
//!    and configuration locked (see [`crate::index::Index::open`]). That blocks
//!    writes, file/network access (COPY TO, read_csv/read_parquet, httpfs), and
//!    extension loading regardless of the SQL submitted.
//! 2. [`validate_sql`] additionally restricts queries to a single, read-only
//!    statement so accidental or malformed input fails fast with a clear
//!    message instead of partially executing.

use duckdb::types::Value;

use crate::index::Index;
use crate::model::{ColumnInfo, SqlQueryResult, TableSchema};

/// Statement keywords permitted as the first token of a query. All are
/// read-only; anything else (INSERT, UPDATE, DELETE, COPY, CREATE, DROP,
/// ATTACH, INSTALL, LOAD, PRAGMA, SET, CALL, ...) is rejected.
const ALLOWED_LEADING_KEYWORDS: &[&str] = &[
    "select", "with", "describe", "explain", "show", "summarize", "table", "values", "from",
];

/// Validate that `sql` is a single, read-only statement.
///
/// Returns `Ok(())` when the query may run, or `Err(message)` describing why it
/// was rejected. This is a syntactic guard layered on top of the read-only
/// connection; it is intentionally conservative.
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
             (SELECT, WITH, DESCRIBE, EXPLAIN, SHOW, SUMMARIZE) are permitted"
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
    /// DuckDB execution error is captured in [`SqlQueryResult::error`] rather
    /// than returned as `Err`, so the model can read the message and retry.
    pub fn run_sql(&self, sql: &str, max_rows: usize) -> SqlQueryResult {
        if let Err(msg) = validate_sql(sql) {
            return SqlQueryResult::error(msg);
        }
        match self.run_sql_inner(sql, max_rows) {
            Ok(result) => result,
            Err(e) => SqlQueryResult::error(e.to_string()),
        }
    }

    /// Execute the (already validated) query, mapping rows to JSON.
    fn run_sql_inner(&self, sql: &str, max_rows: usize) -> duckdb::Result<SqlQueryResult> {
        let conn = self.conn_lock();
        let mut stmt = conn.prepare(sql)?;
        // `query` runs the statement, which populates the result schema.
        let mut rows = stmt.query([])?;
        let columns = rows
            .as_ref()
            .map(|s| s.column_names())
            .unwrap_or_default();
        let ncols = columns.len();

        let mut out: Vec<Vec<serde_json::Value>> = Vec::new();
        let mut truncated = false;
        while let Some(row) = rows.next()? {
            if out.len() >= max_rows {
                truncated = true;
                break;
            }
            let mut record = Vec::with_capacity(ncols);
            for i in 0..ncols {
                record.push(value_to_json(row.get::<_, Value>(i)?));
            }
            out.push(record);
        }

        Ok(SqlQueryResult {
            columns,
            row_count: out.len(),
            truncated,
            rows: out,
            error: None,
        })
    }

    /// Describe the cache tables (`documents`, `artifacts`) for the caller.
    pub fn describe_schema(&self) -> anyhow::Result<Vec<TableSchema>> {
        let conn = self.conn_lock();
        let mut out = Vec::new();
        for table in ["documents", "artifacts"] {
            let mut stmt =
                conn.prepare(&format!("SELECT column_name, column_type FROM (DESCRIBE {table})"))?;
            let columns = stmt
                .query_map([], |row| {
                    Ok(ColumnInfo {
                        name: row.get(0)?,
                        type_: row.get(1)?,
                    })
                })?
                .collect::<duckdb::Result<Vec<_>>>()?;
            out.push(TableSchema {
                table: table.to_string(),
                columns,
            });
        }
        Ok(out)
    }
}

/// Convert an owned DuckDB [`Value`] into a [`serde_json::Value`].
///
/// Numeric and textual scalars map to their JSON equivalents; lists/arrays
/// become JSON arrays and structs/maps become JSON objects. Types without a
/// natural JSON form (blobs, intervals, decimals, timestamps) fall back to a
/// string rendering so a result is always serializable.
fn value_to_json(value: Value) -> serde_json::Value {
    use serde_json::Value as J;
    match value {
        Value::Null => J::Null,
        Value::Boolean(b) => J::Bool(b),
        Value::TinyInt(n) => J::from(n),
        Value::SmallInt(n) => J::from(n),
        Value::Int(n) => J::from(n),
        Value::BigInt(n) => J::from(n),
        Value::UTinyInt(n) => J::from(n),
        Value::USmallInt(n) => J::from(n),
        Value::UInt(n) => J::from(n),
        Value::UBigInt(n) => J::from(n),
        Value::HugeInt(n) => {
            // serde_json numbers do not cover i128; keep precision as a string.
            i64::try_from(n).map(J::from).unwrap_or_else(|_| J::String(n.to_string()))
        }
        Value::Float(n) => json_from_f64(n as f64),
        Value::Double(n) => json_from_f64(n),
        Value::Text(s) | Value::Enum(s) => J::String(s),
        Value::List(items) | Value::Array(items) => {
            J::Array(items.into_iter().map(value_to_json).collect())
        }
        Value::Struct(fields) => J::Object(
            fields
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v.clone())))
                .collect(),
        ),
        Value::Map(entries) => J::Array(
            entries
                .iter()
                .map(|(k, v)| {
                    J::Array(vec![value_to_json(k.clone()), value_to_json(v.clone())])
                })
                .collect(),
        ),
        Value::Union(inner) => value_to_json(*inner),
        // No lossless JSON form: render textually so output stays serializable.
        other => J::String(format!("{other:?}")),
    }
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
        assert!(validate_sql("INSTALL httpfs").is_err());
        assert!(validate_sql("LOAD httpfs").is_err());
        assert!(validate_sql("ATTACH 'other.db'").is_err());
        assert!(validate_sql("PRAGMA threads=4").is_err());
        assert!(validate_sql("SET enable_external_access=true").is_err());
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
