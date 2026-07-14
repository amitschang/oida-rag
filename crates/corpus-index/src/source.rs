//! Artifact byte retrieval — the corpus-agnostic store abstraction plus the
//! OIDA-keyed source and the serving-time resolver.
//!
//! Three layers, smallest first:
//!
//! - [`ArtifactStore`] — the framework's pluggable retrieval interface: given a
//!   `(doc_id, name)`, return the bytes (or `None` if absent). The default impl,
//!   [`ObjectArtifactStore`], maps the pair to an [`object_store`] key via a
//!   caller-supplied closure and works identically over a local directory or an
//!   S3-compatible bucket ([`build_object_store`]). [`fanout_key`] is the one
//!   convenience the framework ships: the `<prefix>/i/d/x/x/<id>/<name>` fan-out.
//! - [`ArtifactSource`] — OIDA's concrete fallback store: an
//!   [`ObjectArtifactStore`] wired with [`fanout_key`] at depth 4 (the archive's
//!   `m/s/k/f/<id>/<name>` layout). Used by the ingest readers and as the
//!   serving fallback.
//! - [`ArtifactReader`] — the serving-time resolver: prefer materialized LanceDB
//!   tiers (the `raw_artifacts` blob table), fall back to an [`ArtifactStore`]
//!   when those are not built.
//!
//! S3 credentials are taken from the standard AWS environment; only the bucket,
//! region, endpoint, and prefix come from [`CoreConfig`]. When no AWS access key
//! is present, requests are sent unsigned so public buckets still work.

use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{LargeBinaryArray, RecordBatch};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::Table;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use object_store::{Error as ObjectStoreError, ObjectStore, ObjectStoreExt};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::config::CoreConfig;
use crate::index::sql_str;

/// Which tier of an [`ArtifactReader`] served a set of bytes, surfaced so the
/// artifact-read tools can report provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactTier {
    /// The materialized `raw_artifacts` LanceDB blob table.
    Raw,
    /// The original artifact source (local directory or S3 bucket).
    ArtifactStore,
}

/// The framework's pluggable artifact-retrieval interface.
///
/// Given a document id and an artifact file name, return the bytes, or `None`
/// when the artifact does not exist (a referenced file may legitimately be
/// absent, e.g. a redacted document). This is the *only* provider-specific tier
/// of [`ArtifactReader`]; everything above it is shared.
#[async_trait]
pub trait ArtifactStore: Send + Sync {
    async fn get(&self, doc_id: &str, name: &str) -> Result<Option<Vec<u8>>>;
}

/// Build the backing [`ObjectStore`] from config: S3 when `s3_bucket` is set,
/// otherwise the local `artifact_root`. Returns `None` when neither is
/// configured, so the caller can decide whether that is fatal.
///
/// The key *prefix* is not applied here — it belongs to the keying closure (see
/// [`fanout_key`]) — because a local store is rooted at `artifact_root` (no
/// prefix) while an S3 store is rooted at the bucket (prefix = `s3_prefix`).
pub fn build_object_store(config: &CoreConfig) -> Result<Option<Arc<dyn ObjectStore>>> {
    if let Some(bucket) = &config.s3_bucket {
        let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);
        if let Some(region) = &config.s3_region {
            builder = builder.with_region(region);
        }
        if let Some(endpoint) = &config.s3_endpoint {
            builder = builder.with_endpoint(endpoint).with_allow_http(true);
        }
        // Public buckets (e.g. the open archive dataset) serve objects
        // anonymously. When no AWS access key is present in the environment
        // there are no credentials to sign with, so skip request signing
        // rather than failing — matching `aws s3 --no-sign-request`.
        let have_access_key = std::env::var_os("AWS_ACCESS_KEY_ID").is_some_and(|v| !v.is_empty());
        if !have_access_key {
            tracing::info!("no AWS_ACCESS_KEY_ID set; using anonymous (unsigned) S3 requests");
            builder = builder.with_skip_signature(true);
        }
        let store = builder.build().context("building S3 object store")?;
        return Ok(Some(Arc::new(store)));
    }
    if let Some(root) = &config.artifact_root {
        let store = LocalFileSystem::new_with_prefix(root)
            .with_context(|| format!("opening local artifact root {}", root.display()))?;
        return Ok(Some(Arc::new(store)));
    }
    Ok(None)
}

/// A fan-out keying closure: maps `(id, name)` to
/// `<prefix>/i/d/x/x…/<id>/<name>`, where the first `depth` characters of `id`
/// each become a directory. This mirrors the archive's on-disk layout (depth 4
/// gives `m/s/k/f/<id>/<name>`) and keeps all of a document's files — original,
/// OCR text, derived thumbnails — under the one `<id>` directory.
pub fn fanout_key(
    prefix: String,
    depth: usize,
) -> impl Fn(&str, &str) -> ObjectPath + Send + Sync + Clone {
    let prefix = prefix.trim_matches('/').to_string();
    move |id: &str, name: &str| {
        let mut raw = String::new();
        if !prefix.is_empty() {
            raw.push_str(&prefix);
            raw.push('/');
        }
        for c in id.chars().take(depth) {
            raw.push(c);
            raw.push('/');
        }
        raw.push_str(id);
        raw.push('/');
        raw.push_str(name);
        ObjectPath::from(raw)
    }
}

/// The default [`ArtifactStore`]: an [`ObjectStore`] addressed by a key closure.
///
/// `K` is monomorphized (key derivation is a direct call, no vtable) and erased
/// the moment the store is held as `Arc<dyn ArtifactStore>`, so the genericity
/// costs nothing and infects no long-lived type.
pub struct ObjectArtifactStore<K> {
    store: Arc<dyn ObjectStore>,
    key: K,
}

impl<K> ObjectArtifactStore<K> {
    pub fn new(store: Arc<dyn ObjectStore>, key: K) -> Self {
        Self { store, key }
    }
}

#[async_trait]
impl<K> ArtifactStore for ObjectArtifactStore<K>
where
    K: Fn(&str, &str) -> ObjectPath + Send + Sync,
{
    async fn get(&self, doc_id: &str, name: &str) -> Result<Option<Vec<u8>>> {
        object_get(&self.store, (self.key)(doc_id, name)).await
    }
}

/// Fetch one object's bytes, mapping a not-found error to `None`.
async fn object_get(store: &Arc<dyn ObjectStore>, path: ObjectPath) -> Result<Option<Vec<u8>>> {
    match store.get(&path).await {
        Ok(result) => {
            let bytes = result.bytes().await.context("reading artifact bytes")?;
            Ok(Some(bytes.to_vec()))
        }
        Err(ObjectStoreError::NotFound { .. }) => Ok(None),
        Err(e) => Err(e).with_context(|| format!("fetching artifact at {path}")),
    }
}

/// OIDA's artifact source: a local directory or an S3 bucket, both addressed
/// with the archive's depth-4 fan-out (`<prefix>/m/s/k/f/<id>/<name>`).
///
/// Used by the ingest readers (full-text/raw) to fetch artifact bytes, and as
/// the fallback tier of an [`ArtifactReader`] at serving time.
pub struct ArtifactSource {
    store: Arc<dyn ObjectStore>,
    /// S3 key prefix; empty for a local root (already rooted at `artifact_root`).
    prefix: String,
}

impl ArtifactSource {
    /// Build the source from config: S3 when `s3_bucket` is set, otherwise the
    /// local `artifact_root`. Returns `None` when neither is configured.
    pub fn from_config(config: &CoreConfig) -> Result<Option<Self>> {
        let Some(store) = build_object_store(config)? else {
            return Ok(None);
        };
        // The S3 prefix keys objects under the bucket; a local store is already
        // rooted at `artifact_root`, so it carries no prefix.
        let prefix = if config.s3_bucket.is_some() {
            config.s3_prefix.clone().unwrap_or_default()
        } else {
            String::new()
        };
        Ok(Some(Self {
            store,
            prefix: prefix.trim_matches('/').to_string(),
        }))
    }

    /// The depth-4 fan-out keyer this source uses.
    fn keyer(&self) -> impl Fn(&str, &str) -> ObjectPath + Send + Sync + Clone {
        fanout_key(self.prefix.clone(), 4)
    }

    /// A display string for an artifact's resolved key, for log messages.
    pub fn key_display(&self, id: &str, name: &str) -> String {
        self.keyer()(id, name).to_string()
    }

    /// Fetch an artifact's bytes. Returns `None` when the object does not exist.
    pub async fn get(&self, id: &str, name: &str) -> Result<Option<Vec<u8>>> {
        object_get(&self.store, self.keyer()(id, name)).await
    }
}

#[async_trait]
impl ArtifactStore for ArtifactSource {
    async fn get(&self, doc_id: &str, name: &str) -> Result<Option<Vec<u8>>> {
        ArtifactSource::get(self, doc_id, name).await
    }
}

/// Serving-time artifact resolver: a layered store, materialized tiers first.
///
/// Lookup order: (1) the `raw_artifacts` LanceDB blob table when it has been
/// built (a `(id, name)` point read); (2) the [`ArtifactStore`] fallback (the
/// original source). Text artifacts are never materialized into `raw_artifacts`,
/// so they always resolve through the fallback — preserving today's behavior
/// while letting stored raw bytes be served from LanceDB.
pub struct ArtifactReader {
    raw: Option<Table>,
    fallback: Option<Arc<dyn ArtifactStore>>,
}

impl ArtifactReader {
    pub fn new(raw: Option<Table>, fallback: Option<Arc<dyn ArtifactStore>>) -> Self {
        Self { raw, fallback }
    }

    /// True when at least one tier can serve bytes; `false` means no artifact
    /// access is configured at all.
    pub fn is_configured(&self) -> bool {
        self.raw.is_some() || self.fallback.is_some()
    }

    /// Resolve an artifact's bytes through the tiers, tagged with the tier that
    /// served them, or `None` if absent.
    ///
    /// `media_type` is an advisory hint; correctness does not depend on it (a
    /// text artifact simply misses the `raw_artifacts` tier and falls through
    /// when raw storage was built binary-only).
    pub async fn bytes(
        &self,
        id: &str,
        name: &str,
        _media_type: Option<&str>,
    ) -> Result<Option<(Vec<u8>, ArtifactTier)>> {
        if let Some(raw) = &self.raw
            && let Some(blob) = raw_blob(raw, id, name).await?
        {
            return Ok(Some((blob, ArtifactTier::Raw)));
        }
        if let Some(fallback) = &self.fallback {
            return Ok(fallback
                .get(id, name)
                .await?
                .map(|b| (b, ArtifactTier::ArtifactStore)));
        }
        Ok(None)
    }
}

/// Point-lookup the `data` column of `raw_artifacts` by `(id, name)`.
async fn raw_blob(table: &Table, id: &str, name: &str) -> Result<Option<Vec<u8>>> {
    let batches: Vec<RecordBatch> = table
        .query()
        .only_if(format!("id = {} AND name = {}", sql_str(id), sql_str(name)))
        .select(Select::columns(&["data"]))
        .limit(1)
        .execute()
        .await
        .context("executing raw-artifact byte lookup")?
        .try_collect()
        .await
        .context("collecting raw-artifact bytes")?;
    for batch in &batches {
        if batch.num_rows() == 0 {
            continue;
        }
        let data = batch
            .column_by_name("data")
            .ok_or_else(|| anyhow::anyhow!("raw_artifacts result missing data column"))?
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| anyhow::anyhow!("raw_artifacts.data is not large-binary"))?;
        return Ok(Some(data.value(0).to_vec()));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(root: std::path::PathBuf) -> CoreConfig {
        CoreConfig {
            artifact_root: Some(root),
            ..CoreConfig::default()
        }
    }

    #[test]
    fn fanout_key_applies_depth_under_prefix() {
        let key = fanout_key("artifacts".to_string(), 4);
        assert_eq!(
            key("mskf0352", "mskf0352.ocr").to_string(),
            "artifacts/m/s/k/f/mskf0352/mskf0352.ocr"
        );
        // Derived files (e.g. thumbnails) live under the document id directory.
        assert_eq!(
            key("mskf0352", "mskf0352_thumb.png").to_string(),
            "artifacts/m/s/k/f/mskf0352/mskf0352_thumb.png"
        );
    }

    #[test]
    fn source_keys_s3_under_prefix() {
        let mut config = cfg(std::path::PathBuf::from("/tmp"));
        config.artifact_root = None;
        config.s3_bucket = Some("bucket".into());
        config.s3_prefix = Some("artifacts".into());
        let source = ArtifactSource::from_config(&config).unwrap().unwrap();
        assert_eq!(
            source.key_display("mskf0352", "mskf0352.ocr"),
            "artifacts/m/s/k/f/mskf0352/mskf0352.ocr"
        );
    }

    /// A text artifact materialized in `raw_artifacts` is served byte-exact by
    /// `get_artifact_text`, tagged as coming from the raw tier — the behavior
    /// `store_text_in_raw` enables (no chunk reconstruction, no source round-trip).
    #[tokio::test]
    async fn text_served_from_raw_table_tagged_raw() {
        use arrow::array::{LargeBinaryArray, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};

        use crate::artifacts::{ArtifactTextStatus, read_artifact_text};
        use crate::index::RAW_ARTIFACTS_TABLE;

        let dir = std::env::temp_dir().join(format!("oida-rawtext-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db = lancedb::connect(dir.to_str().unwrap()).execute().await.unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("data", DataType::LargeBinary, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["doc1"])),
                Arc::new(StringArray::from(vec!["doc1.ocr"])),
                Arc::new(LargeBinaryArray::from(vec![Some(b"hello world".as_ref())])),
            ],
        )
        .unwrap();
        let table = db.create_table(RAW_ARTIFACTS_TABLE, vec![batch]).execute().await.unwrap();

        // Raw-only reader (no fallback): bytes come from LanceDB, tagged Raw.
        let reader = ArtifactReader::new(Some(table), None);
        let (bytes, tier) = reader
            .bytes("doc1", "doc1.ocr", Some("text/plain"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(bytes, b"hello world");
        assert_eq!(tier, ArtifactTier::Raw);

        // get_artifact_text serves that text byte-exact and reports the raw tier.
        let r =
            read_artifact_text(Some(&reader), "doc1", "doc1.ocr", Some("text/plain"), 0, 100).await;
        assert_eq!(r.status, ArtifactTextStatus::TextLoaded);
        assert_eq!(r.text.as_deref(), Some("hello world"));
        assert_eq!(r.source, Some(ArtifactTier::Raw));

        // A miss in the raw tier with no fallback is a graceful absence.
        assert!(reader.bytes("nope", "nope.ocr", None).await.unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// With no raw tier, bytes resolve through the fallback store and are tagged
    /// `ArtifactStore`.
    #[tokio::test]
    async fn fallback_bytes_tagged_artifact_store() {
        let dir = std::env::temp_dir().join(format!("oida-fb-{}", std::process::id()));
        let path = dir.join("m").join("s").join("k").join("f").join("mskf0352").join("mskf0352.ocr");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"body").unwrap();

        let source = ArtifactSource::from_config(&cfg(dir.clone())).unwrap().unwrap();
        let reader = ArtifactReader::new(None, Some(Arc::new(source) as Arc<dyn ArtifactStore>));
        let (bytes, tier) = reader
            .bytes("mskf0352", "mskf0352.ocr", None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(bytes, b"body");
        assert_eq!(tier, ArtifactTier::ArtifactStore);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn local_get_reads_fanned_out_file() {
        let dir = std::env::temp_dir().join(format!("oida-src-{}", std::process::id()));
        let path = dir
            .join("m")
            .join("s")
            .join("k")
            .join("f")
            .join("mskf0352")
            .join("mskf0352.ocr");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"hello").unwrap();

        let source = ArtifactSource::from_config(&cfg(dir.clone())).unwrap().unwrap();
        let bytes = source.get("mskf0352", "mskf0352.ocr").await.unwrap();
        assert_eq!(bytes.as_deref(), Some(b"hello".as_ref()));
        assert!(source.get("missing", "missing.ocr").await.unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
