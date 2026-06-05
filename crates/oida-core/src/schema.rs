//! Parquet schema validation.
//!
//! The cache builder relies on a known set of columns existing in the source
//! parquet. We validate them up front so failures are clear rather than
//! surfacing as cryptic SQL errors deep inside the build.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, bail};
use duckdb::Connection;

/// Columns the cache builder reads from the parquet.
pub const REQUIRED_COLUMNS: &[&str] = &[
    "id",
    "bn",
    "ti",
    "filename",
    "industry",
    "collection",
    "genre",
    "datesent",
    "datereceived",
    "topic",
    "desc",
    "kw",
    "conversation",
    "custodian",
    "au",
    "rc",
    "cc",
    "attachment",
    "related",
    "men",
    "artifact_name",
    "artifact_mediaType",
    "artifact_size",
    "artifact_md5",
];

/// Confirm the parquet exposes every column the cache builder needs.
pub fn validate_parquet(conn: &Connection, parquet_path: &Path) -> anyhow::Result<()> {
    let path = parquet_path.to_string_lossy();
    let mut stmt = conn
        .prepare("SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet(?))")
        .context("preparing DESCRIBE for parquet")?;
    let names: HashSet<String> = stmt
        .query_map([path.as_ref()], |row| row.get::<_, String>(0))
        .context("describing parquet schema")?
        .collect::<Result<_, _>>()?;

    let missing: Vec<&str> = REQUIRED_COLUMNS
        .iter()
        .copied()
        .filter(|c| !names.contains(*c))
        .collect();

    if !missing.is_empty() {
        bail!(
            "parquet {} is missing required columns: {}",
            parquet_path.display(),
            missing.join(", ")
        );
    }
    Ok(())
}
