//! Reading artifact text from disk.
//!
//! Artifacts are stored under `artifact_root`, keyed by `artifact_name`. v1
//! only treats `.ocr` / `text/plain` artifacts as directly readable text;
//! everything else (and any missing files) returns a structured status so the
//! model can reason about the outcome instead of seeing an opaque error.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::config::Config;

/// Outcome of an artifact-text request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactTextStatus {
    /// No `artifact_root` is configured, so files cannot be located.
    ArtifactRootNotConfigured,
    /// The file does not exist under `artifact_root`.
    ArtifactFileMissing,
    /// The artifact is not a text type readable in v1.
    UnsupportedArtifactType,
    /// Text was loaded successfully.
    TextLoaded,
}

/// Result of reading (a slice of) an artifact's text.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ArtifactText {
    pub name: String,
    pub media_type: Option<String>,
    pub status: ArtifactTextStatus,
    /// Loaded text, present only when `status == TextLoaded`.
    pub text: Option<String>,
    /// Byte offset the returned text started at.
    pub offset: u64,
    /// Number of bytes returned.
    pub returned_bytes: u64,
    /// Total size of the file in bytes, when known.
    pub total_bytes: Option<u64>,
    /// Whether more text remains beyond what was returned.
    pub truncated: bool,
}

impl ArtifactText {
    fn status_only(name: &str, media_type: Option<String>, status: ArtifactTextStatus) -> Self {
        Self {
            name: name.to_string(),
            media_type,
            status,
            text: None,
            offset: 0,
            returned_bytes: 0,
            total_bytes: None,
            truncated: false,
        }
    }
}

/// Whether an artifact should be treated as readable text in v1.
fn is_text(name: &str, media_type: Option<&str>) -> bool {
    media_type == Some("text/plain") || name.to_lowercase().ends_with(".ocr")
}

/// Read up to `max_bytes` of an artifact's text starting at `offset`.
pub fn read_artifact_text(
    config: &Config,
    name: &str,
    media_type: Option<&str>,
    offset: u64,
    max_bytes: u64,
) -> ArtifactText {
    let Some(root) = &config.artifact_root else {
        return ArtifactText::status_only(
            name,
            media_type.map(str::to_string),
            ArtifactTextStatus::ArtifactRootNotConfigured,
        );
    };

    if !is_text(name, media_type) {
        return ArtifactText::status_only(
            name,
            media_type.map(str::to_string),
            ArtifactTextStatus::UnsupportedArtifactType,
        );
    }

    // paths are under a hierarchy of the first four characters. Construct the
    // path from the filename
    let prefix_chars = name.chars().take(4).collect::<Vec<char>>();
    let mut artifact_path = PathBuf::new();
    for c in prefix_chars {
        artifact_path.push(c.to_string());
    }
    artifact_path.push(name);

    let path = root.join(artifact_path);
    if !path_is_file(&path) {
        return ArtifactText::status_only(
            name,
            media_type.map(str::to_string),
            ArtifactTextStatus::ArtifactFileMissing,
        );
    }

    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => {
            return ArtifactText::status_only(
                name,
                media_type.map(str::to_string),
                ArtifactTextStatus::ArtifactFileMissing,
            );
        }
    };

    let total = bytes.len() as u64;
    let start = offset.min(total) as usize;
    let end = (start as u64 + max_bytes).min(total) as usize;
    let slice = &bytes[start..end];
    let text = String::from_utf8_lossy(slice).into_owned();

    ArtifactText {
        name: name.to_string(),
        media_type: media_type.map(str::to_string),
        status: ArtifactTextStatus::TextLoaded,
        offset: start as u64,
        returned_bytes: (end - start) as u64,
        total_bytes: Some(total),
        truncated: (end as u64) < total,
        text: Some(text),
    }
}

fn path_is_file(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(root: Option<std::path::PathBuf>) -> Config {
        Config {
            artifact_root: root,
            ..Config::default()
        }
    }

    #[test]
    fn reports_missing_root() {
        let r = read_artifact_text(&cfg(None), "x.ocr", Some("text/plain"), 0, 100);
        assert_eq!(r.status, ArtifactTextStatus::ArtifactRootNotConfigured);
    }

    #[test]
    fn rejects_non_text_types() {
        let dir = std::env::temp_dir();
        let r = read_artifact_text(&cfg(Some(dir)), "x.pdf", Some("application/pdf"), 0, 100);
        assert_eq!(r.status, ArtifactTextStatus::UnsupportedArtifactType);
    }

    #[test]
    fn reports_missing_file() {
        let dir = std::env::temp_dir();
        let r = read_artifact_text(&cfg(Some(dir)), "does-not-exist.ocr", None, 0, 100);
        assert_eq!(r.status, ArtifactTextStatus::ArtifactFileMissing);
    }

    #[test]
    fn loads_and_pages_text() {
        let dir = std::env::temp_dir().join(format!("oida-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a").join("b").join("c").join("d").join("abcd_doc.ocr");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"hello world").unwrap();

        let r = read_artifact_text(&cfg(Some(dir.clone())), "abcd_doc.ocr", Some("text/plain"), 0, 5);
        assert_eq!(r.status, ArtifactTextStatus::TextLoaded);
        assert_eq!(r.text.as_deref(), Some("hello"));
        assert_eq!(r.total_bytes, Some(11));
        assert!(r.truncated);

        let r2 = read_artifact_text(&cfg(Some(dir)), "abcd_doc.ocr", Some("text/plain"), 6, 100);
        assert_eq!(r2.text.as_deref(), Some("world"));
        assert!(!r2.truncated);
        let _ = std::fs::remove_file(path);
    }
}
