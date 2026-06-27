//! Reading artifact bytes from a local directory or an S3 bucket.
//!
//! Both ingest paths that touch artifact files — the raw-artifact store and the
//! full-text/OCR reader — fetch bytes through this single abstraction so they
//! work identically against a local `artifact_root` or a remote S3 bucket. The
//! key layout is the same fan-out the on-disk store uses
//! (`<prefix>/m/s/k/f/<id>/<name>`), so the same corpus can live in either
//! backend without re-keying.
//!
//! Backed by [`object_store`] (already in the dependency tree via `lance`),
//! which gives a uniform async `get` over `LocalFileSystem` and `AmazonS3`. S3
//! credentials are taken from the standard AWS environment; only the bucket,
//! region, endpoint, and prefix come from [`Config`]. When no AWS access key is
//! present, requests are sent unsigned so public buckets still work.

use std::sync::Arc;

use anyhow::{Context, Result};
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectPath;
use object_store::{Error as ObjectStoreError, ObjectStore, ObjectStoreExt};

use crate::config::Config;

/// A source of artifact bytes: a local directory or an S3 bucket, both sharing
/// the fan-out key layout `<prefix>/m/s/k/f/<id>/<name>`.
pub struct ArtifactSource {
    store: Arc<dyn ObjectStore>,
    /// Key prefix prepended to every artifact path (empty for a bare local root).
    prefix: String,
}

impl ArtifactSource {
    /// Build the source from config: S3 when `s3_bucket` is set, otherwise the
    /// local `artifact_root`. Returns `None` when neither is configured, so the
    /// caller can decide whether that is fatal.
    pub fn from_config(config: &Config) -> Result<Option<Self>> {
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
            let have_access_key =
                std::env::var_os("AWS_ACCESS_KEY_ID").is_some_and(|v| !v.is_empty());
            if !have_access_key {
                tracing::info!(
                    "no AWS_ACCESS_KEY_ID set; using anonymous (unsigned) S3 requests"
                );
                builder = builder.with_skip_signature(true);
            }
            let store = builder.build().context("building S3 object store")?;
            let prefix = config.s3_prefix.clone().unwrap_or_default();
            return Ok(Some(Self {
                store: Arc::new(store),
                prefix: prefix.trim_matches('/').to_string(),
            }));
        }
        if let Some(root) = &config.artifact_root {
            let store = LocalFileSystem::new_with_prefix(root)
                .with_context(|| format!("opening local artifact root {}", root.display()))?;
            return Ok(Some(Self {
                store: Arc::new(store),
                prefix: String::new(),
            }));
        }
        Ok(None)
    }

    /// Object key for an artifact, applying the fan-out layout under the
    /// prefix: the first four characters of the document `id` each become a
    /// directory, then the full `id`, then the file `name` (mirrors the
    /// archive's on-disk layout `i/d/x/x/idxx…/<name>`).
    ///
    /// All of a document's files — the original, its OCR text, and derived
    /// files like `<id>_thumb.png` — share the one `<id>` directory.
    fn key(&self, id: &str, name: &str) -> ObjectPath {
        let mut raw = String::new();
        if !self.prefix.is_empty() {
            raw.push_str(&self.prefix);
            raw.push('/');
        }
        for c in id.chars().take(4) {
            raw.push(c);
            raw.push('/');
        }
        raw.push_str(id);
        raw.push('/');
        raw.push_str(name);
        ObjectPath::from(raw)
    }

    /// A display string for an artifact's resolved key, for log messages.
    pub fn key_display(&self, id: &str, name: &str) -> String {
        self.key(id, name).to_string()
    }

    /// Fetch an artifact's bytes. Returns `None` when the object does not exist
    /// (a referenced file may legitimately be absent, e.g. a redacted document).
    pub async fn get(&self, id: &str, name: &str) -> Result<Option<Vec<u8>>> {
        let path = self.key(id, name);
        match self.store.get(&path).await {
            Ok(result) => {
                let bytes = result.bytes().await.context("reading artifact bytes")?;
                Ok(Some(bytes.to_vec()))
            }
            Err(ObjectStoreError::NotFound { .. }) => Ok(None),
            Err(e) => Err(e).with_context(|| format!("fetching artifact {id}/{name}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(root: std::path::PathBuf) -> Config {
        Config { artifact_root: Some(root), ..Config::default() }
    }

    #[test]
    fn key_applies_fanout_under_prefix() {
        let mut config = cfg(std::path::PathBuf::from("/tmp"));
        config.artifact_root = None;
        config.s3_bucket = Some("bucket".into());
        config.s3_prefix = Some("artifacts".into());
        let source = ArtifactSource::from_config(&config).unwrap().unwrap();
        assert_eq!(
            source.key_display("mskf0352", "mskf0352.ocr"),
            "artifacts/m/s/k/f/mskf0352/mskf0352.ocr"
        );
        // Derived files (e.g. thumbnails) live under the document id directory.
        assert_eq!(
            source.key_display("mskf0352", "mskf0352_thumb.png"),
            "artifacts/m/s/k/f/mskf0352/mskf0352_thumb.png"
        );
    }

    #[tokio::test]
    async fn local_get_reads_fanned_out_file() {
        let dir = std::env::temp_dir().join(format!("oida-src-{}", std::process::id()));
        let path = dir.join("m").join("s").join("k").join("f").join("mskf0352").join("mskf0352.ocr");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"hello").unwrap();

        let source = ArtifactSource::from_config(&cfg(dir.clone())).unwrap().unwrap();
        let bytes = source.get("mskf0352", "mskf0352.ocr").await.unwrap();
        assert_eq!(bytes.as_deref(), Some(b"hello".as_ref()));
        assert!(source.get("missing", "missing.ocr").await.unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
