//! Minimal Solr client for the archive source index.
//!
//! The OIDA archive is a Solr core (scraped at UCSF). For incremental
//! updates we page the core by `cursorMark` ordered by the unique `id` field —
//! the archiver metadata-modified date (`ddmudate`) has no docValues, so it can
//! only be *filtered* (`fq`), never sorted — and use that date purely as an
//! inclusive lower-bound watermark. This client covers just that slice:
//! cursor-paged `select` queries returning raw JSON documents.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

/// The opaque starting cursor Solr expects for the first page.
pub const CURSOR_START: &str = "*";

/// A cursor-paged Solr `select` client bound to one core and corpus query.
#[derive(Clone, Debug)]
pub struct SolrClient {
    http: reqwest::Client,
    select_url: String,
    query: String,
    modified_field: String,
    rows: usize,
}

/// One page of results plus the cursor needed to fetch the next page.
pub struct Page {
    /// Raw Solr documents (each a JSON object).
    pub docs: Vec<Value>,
    /// Cursor to pass as `cursorMark` for the following page. Equal to the
    /// cursor that produced this page once the result set is exhausted.
    pub next_cursor: String,
    /// Total documents matching the query (the same for every page).
    pub num_found: u64,
}

#[derive(Deserialize)]
struct RawResponse {
    response: RawBody,
    #[serde(rename = "nextCursorMark")]
    next_cursor_mark: Option<String>,
}

#[derive(Deserialize)]
struct RawBody {
    #[serde(rename = "numFound")]
    num_found: u64,
    docs: Vec<Value>,
}

impl SolrClient {
    /// Build a client for the core at `base` (e.g.
    /// `https://metadata.idl.ucsf.edu/solr/ltdl3`), tracking documents matched by
    /// `query` and watermarked on `modified_field`, `rows` per page.
    pub fn new(
        base: &str,
        query: impl Into<String>,
        modified_field: impl Into<String>,
        rows: usize,
    ) -> Result<Self> {
        let base = base.trim_end_matches('/');
        reqwest::Url::parse(base).with_context(|| format!("invalid solr url {base}"))?;
        Ok(Self {
            http: reqwest::Client::new(),
            select_url: format!("{base}/select"),
            query: query.into(),
            modified_field: modified_field.into(),
            rows: rows.max(1),
        })
    }

    /// The `fq` range clause for an inclusive lower-bound watermark, if any.
    fn since_filter(&self, since: Option<&str>) -> Option<String> {
        since.map(|s| format!("{}:[{} TO *]", self.modified_field, s))
    }

    /// Fetch one page of `id` + modified-date for documents at or after `since`,
    /// starting from `cursor` ([`CURSOR_START`] for the first page).
    pub async fn page(&self, since: Option<&str>, cursor: &str) -> Result<Page> {
        let rows = self.rows.to_string();
        let fl = format!("id,{}", self.modified_field);
        let mut params: Vec<(&str, &str)> = vec![
            ("q", self.query.as_str()),
            ("sort", "id asc"),
            ("wt", "json"),
            ("rows", rows.as_str()),
            ("cursorMark", cursor),
            ("fl", fl.as_str()),
        ];
        let fq = self.since_filter(since);
        if let Some(f) = &fq {
            params.push(("fq", f.as_str()));
        }
        self.fetch(&params).await
    }

    /// Like [`page`](Self::page) but also returns the fields needed to classify
    /// each document's content and redaction state: the `artifact` list and the
    /// `deaccessioned`/`published` flags.
    pub async fn classify_page(&self, since: Option<&str>, cursor: &str) -> Result<Page> {
        let rows = self.rows.to_string();
        let fl = format!("id,{},artifact,deaccessioned,published", self.modified_field);
        let mut params: Vec<(&str, &str)> = vec![
            ("q", self.query.as_str()),
            ("sort", "id asc"),
            ("wt", "json"),
            ("rows", rows.as_str()),
            ("cursorMark", cursor),
            ("fl", fl.as_str()),
        ];
        let fq = self.since_filter(since);
        if let Some(f) = &fq {
            params.push(("fq", f.as_str()));
        }
        self.fetch(&params).await
    }

    /// Page the full set of `fields` (plus `id` and the modified field) needed to
    /// build `documents`/`artifacts` rows. Used by the Solr ingest; `fields`
    /// should exclude `ot` and other heavy columns the mapper does not read.
    pub async fn scan_page(
        &self,
        since: Option<&str>,
        cursor: &str,
        fields: &[&str],
    ) -> Result<Page> {
        let rows = self.rows.to_string();
        let mut fl_parts: Vec<&str> = vec!["id", self.modified_field.as_str()];
        fl_parts.extend_from_slice(fields);
        let fl = fl_parts.join(",");
        let mut params: Vec<(&str, &str)> = vec![
            ("q", self.query.as_str()),
            ("sort", "id asc"),
            ("wt", "json"),
            ("rows", rows.as_str()),
            ("cursorMark", cursor),
            ("fl", fl.as_str()),
        ];
        let fq = self.since_filter(since);
        if let Some(f) = &fq {
            params.push(("fq", f.as_str()));
        }
        self.fetch(&params).await
    }

    /// Fetch a single full document (all fields) for schema inspection.
    pub async fn sample_doc(&self, since: Option<&str>) -> Result<Option<Value>> {
        let mut params: Vec<(&str, &str)> = vec![
            ("q", self.query.as_str()),
            ("sort", "id asc"),
            ("wt", "json"),
            ("rows", "1"),
            ("cursorMark", CURSOR_START),
            ("fl", "*"),
        ];
        let fq = self.since_filter(since);
        if let Some(f) = &fq {
            params.push(("fq", f.as_str()));
        }
        Ok(self.fetch(&params).await?.docs.into_iter().next())
    }

    async fn fetch(&self, params: &[(&str, &str)]) -> Result<Page> {
        let resp = self
            .http
            .get(&self.select_url)
            .query(params)
            .send()
            .await
            .with_context(|| format!("querying solr at {}", self.select_url))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("solr returned {status}: {body}");
        }
        let parsed: RawResponse = resp.json().await.context("parsing solr response")?;
        Ok(Page {
            docs: parsed.response.docs,
            next_cursor: parsed.next_cursor_mark.unwrap_or_default(),
            num_found: parsed.response.num_found,
        })
    }
}
