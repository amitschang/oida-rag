//! Mapping from raw Solr documents to LanceDB `documents`/`artifacts` batches.
//!
//! Solr is the single source of truth (the parquet route is retired), so this
//! module owns the schema. Each Solr document becomes one `documents` row and
//! its parsed `artifact` list becomes several `artifacts` rows. The only
//! transform the old parquet scraper applied was flattening single-valued array
//! fields (dates, `collection`, …) to scalars; we reproduce that robustly by
//! accepting either a scalar or a one-element array for every scalar column,
//! without needing to know which fields are arrays.
//!
//! The produced schemas match the existing tables (so the serving path and the
//! full-text build are unchanged) plus two new `documents` columns:
//! - `ddmudate`: the document's modified-date watermark (max of the field).
//! - `digest`: a stable fingerprint of the metadata + artifact md5 set, for
//!   future metadata-change detection.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{
    Int64Builder, ListBuilder, RecordBatch, StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema};
use serde::Deserialize;
use serde_json::Value;

/// Scalar (flattened) `documents` columns, paired with the Solr field they come
/// from. `title` is special-cased (coalesce of `ti`/`filename`).
const SCALAR_COLS: &[(&str, &str)] = &[
    ("id", "id"),
    ("bn", "bn"),
    // ("title", _) handled separately.
    ("industry", "industry"),
    ("collection", "collection"),
    ("genre", "genre"),
    ("date_sent", "datesent"),
    ("date_received", "datereceived"),
    ("topic", "topic"),
    ("description", "desc"),
    ("keywords", "kw"),
    ("conversation", "conversation"),
];

/// List-valued `documents` columns, paired with their Solr field.
const LIST_COLS: &[(&str, &str)] = &[
    ("custodian", "custodian"),
    ("authors", "au"),
    ("recipients", "rc"),
    ("cc", "cc"),
    ("attachments", "attachment"),
    ("related", "related"),
    ("mentions", "men"),
];

/// Solr fields the mapper reads (excluding `id` and the modified field, which
/// the scan always requests). Used to keep the Solr ingest's `fl` lean — in
/// particular it omits the heavy `ot` full-text column the mapper never reads.
/// `deaccessioned`/`published` are the redaction flags the incremental apply
/// classifier needs; the full ingest fetches them harmlessly and ignores them.
pub(crate) const SOURCE_FIELDS: &[&str] = &[
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
    "deaccessioned",
    "published",
];

/// One artifact parsed from the Solr `artifact` field.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ArtifactMeta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default, rename = "mediaType")]
    pub media_type: Option<String>,
    #[serde(default)]
    pub md5: Option<String>,
    #[serde(default)]
    pub size: Option<i64>,
}

/// Build the `documents` record batch for a slice of Solr documents.
///
/// Column order mirrors the original ingest (scalars, then list columns, then
/// the derived `artifact_types`/`artifact_count`/`search_text`) followed by the
/// new `ddmudate` and `digest` columns. `modified_field` names the Solr
/// watermark field (e.g. `ddmudate`).
pub(crate) fn documents_batch(docs: &[Value], modified_field: &str) -> Result<RecordBatch> {
    // Scalar builders, in SCALAR_COLS order with `title` inserted after `bn`.
    let mut id = StringBuilder::new();
    let mut bn = StringBuilder::new();
    let mut title = StringBuilder::new();
    let mut scalars: Vec<StringBuilder> = (0..SCALAR_COLS.len() - 2)
        .map(|_| StringBuilder::new())
        .collect();
    let mut lists: Vec<ListBuilder<StringBuilder>> =
        (0..LIST_COLS.len()).map(|_| ListBuilder::new(StringBuilder::new())).collect();
    let mut artifact_types = ListBuilder::new(StringBuilder::new());
    let mut artifact_count = Int64Builder::new();
    let mut search_text = StringBuilder::new();
    let mut ddmudate = StringBuilder::new();
    let mut digest = StringBuilder::new();

    for doc in docs {
        // id + bn + title (coalesce ti/filename).
        append_opt(&mut id, scalar_field(doc, "id"));
        append_opt(&mut bn, scalar_field(doc, "bn"));
        let title_val = scalar_field(doc, "ti").or_else(|| scalar_field(doc, "filename"));
        append_opt(&mut title, title_val.clone());

        // Remaining scalar columns (skip the first two: id, bn).
        for (builder, (_, solr)) in scalars.iter_mut().zip(SCALAR_COLS.iter().skip(2)) {
            append_opt(builder, scalar_field(doc, solr));
        }

        // List columns.
        let mut list_values: Vec<Vec<String>> = Vec::with_capacity(LIST_COLS.len());
        for (builder, (_, solr)) in lists.iter_mut().zip(LIST_COLS.iter()) {
            let vals = list_field(doc, solr);
            for v in &vals {
                builder.values().append_value(v);
            }
            builder.append(true);
            list_values.push(vals);
        }

        // Artifacts → distinct media types + count.
        let artifacts = parse_artifacts(doc);
        artifact_count.append_value(artifacts.len() as i64);
        let mut seen: HashSet<&str> = HashSet::new();
        for a in &artifacts {
            if let Some(m) = a.media_type.as_deref()
                && seen.insert(m)
            {
                artifact_types.values().append_value(m);
            }
        }
        artifact_types.append(true);

        // Derived search text (same fields the FTS column has always covered).
        search_text.append_value(build_search_text(doc, title_val.as_deref()));

        // Watermark + digest.
        append_opt(&mut ddmudate, doc_modified(doc, modified_field));
        digest.append_value(compute_digest(doc, &artifacts));
    }

    let mut fields: Vec<Arc<Field>> = Vec::new();
    let mut columns: Vec<arrow::array::ArrayRef> = Vec::new();

    fields.push(Arc::new(Field::new("id", DataType::Utf8, true)));
    columns.push(Arc::new(id.finish()));
    fields.push(Arc::new(Field::new("bn", DataType::Utf8, true)));
    columns.push(Arc::new(bn.finish()));
    fields.push(Arc::new(Field::new("title", DataType::Utf8, true)));
    columns.push(Arc::new(title.finish()));
    for (builder, (name, _)) in scalars.iter_mut().zip(SCALAR_COLS.iter().skip(2)) {
        fields.push(Arc::new(Field::new(*name, DataType::Utf8, true)));
        columns.push(Arc::new(builder.finish()));
    }
    for (builder, (name, _)) in lists.iter_mut().zip(LIST_COLS.iter()) {
        fields.push(Arc::new(Field::new(*name, list_type(), true)));
        columns.push(Arc::new(builder.finish()));
    }
    fields.push(Arc::new(Field::new("artifact_types", list_type(), true)));
    columns.push(Arc::new(artifact_types.finish()));
    fields.push(Arc::new(Field::new("artifact_count", DataType::Int64, false)));
    columns.push(Arc::new(artifact_count.finish()));
    fields.push(Arc::new(Field::new("search_text", DataType::Utf8, false)));
    columns.push(Arc::new(search_text.finish()));
    fields.push(Arc::new(Field::new("ddmudate", DataType::Utf8, true)));
    columns.push(Arc::new(ddmudate.finish()));
    fields.push(Arc::new(Field::new("digest", DataType::Utf8, false)));
    columns.push(Arc::new(digest.finish()));

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .context("assembling documents batch from solr")
}

/// Build the `artifacts` record batch: one row per artifact across `docs`.
pub(crate) fn artifacts_batch(docs: &[Value]) -> Result<RecordBatch> {
    let mut out_id = StringBuilder::new();
    let mut out_name = StringBuilder::new();
    let mut out_media = StringBuilder::new();
    let mut out_size = Int64Builder::new();
    let mut out_md5 = StringBuilder::new();

    for doc in docs {
        let id = scalar_field(doc, "id");
        for a in parse_artifacts(doc) {
            append_opt(&mut out_id, id.clone());
            append_opt(&mut out_name, a.name);
            append_opt(&mut out_media, a.media_type);
            match a.size {
                Some(s) => out_size.append_value(s),
                None => out_size.append_null(),
            }
            append_opt(&mut out_md5, a.md5);
        }
    }

    let fields = vec![
        Field::new("id", DataType::Utf8, true),
        Field::new("name", DataType::Utf8, true),
        Field::new("media_type", DataType::Utf8, true),
        Field::new("size", DataType::Int64, true),
        Field::new("md5", DataType::Utf8, true),
    ];
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        vec![
            Arc::new(out_id.finish()),
            Arc::new(out_name.finish()),
            Arc::new(out_media.finish()),
            Arc::new(out_size.finish()),
            Arc::new(out_md5.finish()),
        ],
    )
    .context("assembling artifacts batch from solr")
}

/// The Arrow type of our list columns (`List<Utf8>` with child field `item`).
fn list_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)))
}

/// Concatenate the searchable fields into the FTS `search_text` value.
fn build_search_text(doc: &Value, title: Option<&str>) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut push = |s: Option<String>| {
        if let Some(s) = s
            && !s.is_empty()
        {
            parts.push(s);
        }
    };
    push(title.map(str::to_string));
    push(scalar_field(doc, "bn"));
    push(scalar_field(doc, "topic"));
    push(scalar_field(doc, "desc"));
    push(scalar_field(doc, "kw"));
    push(Some(list_field(doc, "au").join(" ")));
    push(Some(list_field(doc, "custodian").join(" ")));
    push(Some(list_field(doc, "rc").join(" ")));
    parts.join(" ")
}

/// A stable per-document fingerprint over the metadata and artifact md5 set.
///
/// Used as `documents.digest` so a future incremental run can detect
/// metadata-only edits (content changes are already caught by the artifact md5
/// set). FNV-1a/64 keeps it deterministic across Rust versions with no extra
/// dependency.
fn compute_digest(doc: &Value, artifacts: &[ArtifactMeta]) -> String {
    const SEP: char = '\u{1f}';
    let mut canon = String::new();
    canon.push_str(scalar_field(doc, "id").as_deref().unwrap_or(""));
    let title = scalar_field(doc, "ti").or_else(|| scalar_field(doc, "filename"));
    canon.push(SEP);
    canon.push_str(title.as_deref().unwrap_or(""));
    for (_, solr) in SCALAR_COLS.iter().skip(2) {
        canon.push(SEP);
        canon.push_str(scalar_field(doc, solr).as_deref().unwrap_or(""));
    }
    for (_, solr) in LIST_COLS {
        canon.push(SEP);
        canon.push_str(&list_field(doc, solr).join("\u{1e}"));
    }
    let mut arts: Vec<String> = artifacts
        .iter()
        .map(|a| {
            format!(
                "{}\u{0}{}",
                a.name.as_deref().unwrap_or(""),
                a.md5.as_deref().unwrap_or("")
            )
        })
        .collect();
    arts.sort();
    canon.push(SEP);
    canon.push_str(&arts.join("\u{1e}"));
    format!("{:016x}", fnv1a64(&canon))
}

/// FNV-1a 64-bit hash.
fn fnv1a64(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Read a scalar string column, flattening a single-element array to its value.
pub(crate) fn scalar_field(doc: &Value, field: &str) -> Option<String> {
    match doc.get(field) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(a)) => a.first().and_then(Value::as_str).map(str::to_string),
        Some(Value::Number(n)) => Some(n.to_string()),
        Some(Value::Bool(b)) => Some(b.to_string()),
        _ => None,
    }
}

/// Read a list-valued string column, wrapping a lone scalar into a one-element
/// list. Non-string array elements are skipped.
pub(crate) fn list_field(doc: &Value, field: &str) -> Vec<String> {
    match doc.get(field) {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

/// Extract the `id` of a Solr document.
pub(crate) fn doc_id(doc: &Value) -> Option<String> {
    doc.get("id").and_then(Value::as_str).map(str::to_string)
}

/// Extract the (possibly multi-valued) modified-date field, taking the greatest
/// value — a document can carry several and the watermark tracks the latest.
pub(crate) fn doc_modified(doc: &Value, field: &str) -> Option<String> {
    match doc.get(field) {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).max().map(str::to_string),
        _ => None,
    }
}

/// Parse the Solr `artifact` field into artifact metadata.
///
/// Solr returns it as a list containing a single *stringified* JSON array of
/// `{name, size, mediaType, md5}` objects, so the inner string is parsed again.
/// For robustness this also accepts an already-decoded array or object element.
pub(crate) fn parse_artifacts(doc: &Value) -> Vec<ArtifactMeta> {
    let mut out = Vec::new();
    match doc.get("artifact") {
        Some(Value::Array(items)) => {
            for item in items {
                push_artifacts(item, &mut out);
            }
        }
        Some(item @ Value::String(_)) => push_artifacts(item, &mut out),
        _ => {}
    }
    out
}

/// Decode one element of the `artifact` field into zero or more artifacts.
fn push_artifacts(item: &Value, out: &mut Vec<ArtifactMeta>) {
    match item {
        Value::String(s) => {
            if let Ok(metas) = serde_json::from_str::<Vec<ArtifactMeta>>(s) {
                out.extend(metas);
            }
        }
        Value::Object(_) => {
            if let Ok(meta) = serde_json::from_value::<ArtifactMeta>(item.clone()) {
                out.push(meta);
            }
        }
        _ => {}
    }
}

/// Append an optional string, writing null when absent.
fn append_opt(builder: &mut StringBuilder, value: Option<String>) {
    match value {
        Some(v) => builder.append_value(v),
        None => builder.append_null(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, ListArray, StringArray};
    use serde_json::json;

    /// A Solr doc mixing scalars, single-element arrays (flatten), multi-valued
    /// lists, and the doubly-encoded `artifact` field.
    fn sample() -> Value {
        json!({
            "id": "abc123",
            "bn": "BATES-1",
            "ti": "Hello World",
            "filename": "fallback.txt",
            "industry": "Opioids",
            "collection": ["Some Collection"],          // single-element array → flatten
            "genre": "Email",
            "datesent": ["2012 August 07"],             // flatten
            "conversation": "thread-9",
            "custodian": ["Smith, John", "Doe, Jane"],   // real multi-valued
            "au": ["sender@example.com"],
            "rc": ["a@example.com", "b@example.com"],
            "ddmudate": ["2026-02-26T00:00:00Z", "2026-05-01T00:00:00Z"],
            "deaccessioned": false,
            "published": true,
            "artifact": [
                "[{\"name\":\"abc123.ocr\",\"size\":10,\"mediaType\":\"text/plain\",\"md5\":\"aaa\"}, \
                  {\"name\":\"abc123.pdf\",\"size\":20,\"mediaType\":\"application/pdf\",\"md5\":\"bbb\"}]"
            ]
        })
    }

    fn str_at(batch: &RecordBatch, col: &str, row: usize) -> Option<String> {
        let idx = batch.schema().index_of(col).unwrap();
        let arr = batch.column(idx).as_any().downcast_ref::<StringArray>().unwrap();
        (!arr.is_null(row)).then(|| arr.value(row).to_string())
    }

    #[test]
    fn documents_batch_maps_and_flattens() {
        let docs = vec![sample()];
        let batch = documents_batch(&docs, "ddmudate").unwrap();
        assert_eq!(batch.num_rows(), 1);

        assert_eq!(str_at(&batch, "id", 0).as_deref(), Some("abc123"));
        // title coalesces ti over filename.
        assert_eq!(str_at(&batch, "title", 0).as_deref(), Some("Hello World"));
        // single-element arrays flattened to scalars.
        assert_eq!(str_at(&batch, "collection", 0).as_deref(), Some("Some Collection"));
        assert_eq!(str_at(&batch, "date_sent", 0).as_deref(), Some("2012 August 07"));
        // ddmudate is the max of the multi-valued field.
        assert_eq!(str_at(&batch, "ddmudate", 0).as_deref(), Some("2026-05-01T00:00:00Z"));
        // digest is a stable 16-hex-char fingerprint.
        let digest = str_at(&batch, "digest", 0).unwrap();
        assert_eq!(digest.len(), 16);

        // artifact_count = 2.
        let idx = batch.schema().index_of("artifact_count").unwrap();
        let count = batch.column(idx).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(count.value(0), 2);

        // multi-valued list preserved.
        let idx = batch.schema().index_of("custodian").unwrap();
        let custodian = batch.column(idx).as_any().downcast_ref::<ListArray>().unwrap();
        let vals = custodian.value(0);
        assert_eq!(vals.as_any().downcast_ref::<StringArray>().unwrap().len(), 2);

        // search_text contains title and a recipient.
        let st = str_at(&batch, "search_text", 0).unwrap();
        assert!(st.contains("Hello World"));
        assert!(st.contains("b@example.com"));
    }

    #[test]
    fn artifacts_batch_explodes_rows() {
        let docs = vec![sample()];
        let batch = artifacts_batch(&docs).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(str_at(&batch, "id", 0).as_deref(), Some("abc123"));
        assert_eq!(str_at(&batch, "name", 0).as_deref(), Some("abc123.ocr"));
        assert_eq!(str_at(&batch, "media_type", 0).as_deref(), Some("text/plain"));
        assert_eq!(str_at(&batch, "md5", 1).as_deref(), Some("bbb"));
    }

    #[test]
    fn digest_changes_with_artifact_md5() {
        let a = sample();
        let mut b = sample();
        b["artifact"] = json!([
            "[{\"name\":\"abc123.ocr\",\"size\":10,\"mediaType\":\"text/plain\",\"md5\":\"CHANGED\"}]"
        ]);
        let da = compute_digest(&a, &parse_artifacts(&a));
        let db = compute_digest(&b, &parse_artifacts(&b));
        assert_ne!(da, db);
    }
}
