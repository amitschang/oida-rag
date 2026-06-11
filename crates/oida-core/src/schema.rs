//! Parquet schema validation.
//!
//! The ingest relies on a known set of columns existing in the source parquet.
//! We validate them up front (against the DataFusion-registered `src` table) so
//! failures are clear rather than surfacing as cryptic SQL errors deep inside
//! the ingest query.

use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use datafusion::prelude::SessionContext;

/// Columns the ingest reads from the parquet. The source has one row per
/// document; `artifact` is a `list<struct<md5, mediaType, name, size>>` holding
/// the document's artifacts inline.
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
    "artifact",
];

/// Confirm the registered `src` table exposes every column the ingest needs.
pub(crate) async fn validate_registered(ctx: &SessionContext) -> Result<()> {
    let df = ctx
        .sql("SELECT * FROM src LIMIT 0")
        .await
        .context("describing parquet schema")?;
    let names: HashSet<String> = df
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();

    let missing: Vec<&str> = REQUIRED_COLUMNS
        .iter()
        .copied()
        .filter(|c| !names.contains(*c))
        .collect();

    if !missing.is_empty() {
        bail!(
            "parquet is missing required columns: {}",
            missing.join(", ")
        );
    }
    Ok(())
}
