//! HuggingFace LFS protocol primitives — batch endpoint, signed-URL
//! streaming PUT, verify, commit. Internal-only siblings of
//! [`super::upload::upload_file_to_hf`]; see that file for the
//! orchestration shape.
//!
//! Module layout (round-6 split, 2026-05-10):
//! - `batch`        — `lfs_batch_upload` + JSON parsing
//! - `stream`       — `stream_put_with_progress` (PUT to signed URL)
//! - `finalize`     — `lfs_verify` + `commit_lfs_file`
//! - `mod` (here)   — `CountingReader`, action types, `upload_lfs` orchestrator
//! - `test_support` — shared test fixtures (cfg(test))

mod batch;
mod finalize;
mod stream;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod test_support;

use std::collections::HashMap;
use std::path::Path;

use crate::error::VindexError;

use super::PublishCallbacks;
use batch::lfs_batch_upload;
use finalize::{commit_lfs_file, lfs_verify};
use stream::stream_put_with_progress;

/// Counting `Read` adapter — increments a shared atomic on every read so
/// a poll thread can report upload progress without per-chunk syscalls.
pub(super) struct CountingReader<R: std::io::Read> {
    pub(super) inner: R,
    pub(super) counter: std::sync::Arc<portable_atomic::AtomicU64>,
}

impl<R: std::io::Read> std::io::Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.counter
            .fetch_add(n as u64, portable_atomic::Ordering::Relaxed);
        Ok(n)
    }
}

#[derive(Debug)]
pub(super) struct LfsAction {
    pub(super) href: String,
    pub(super) header: HashMap<String, String>,
}

#[derive(Debug)]
pub(super) struct LfsBatchResponse {
    pub(super) upload: Option<LfsAction>,
    pub(super) verify: Option<LfsAction>,
}

/// LFS-mode upload: batch → PUT to signed URL → verify → commit pointer.
#[allow(clippy::too_many_arguments)]
pub(super) fn upload_lfs(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    size: u64,
    sha256: &str,
    callbacks: &mut dyn PublishCallbacks,
    repo_type: &str,
) -> Result<(), VindexError> {
    let batch = lfs_batch_upload(repo_id, token, sha256, size, repo_type)?;

    if let Some(ref upload) = batch.upload {
        stream_put_with_progress(
            &upload.href,
            &upload.header,
            local_path,
            size,
            remote_filename,
            callbacks,
        )?;
    } else {
        // Tick the bar to 100% so the UX matches the upload path.
        callbacks.on_file_progress(remote_filename, size, size);
    }

    if let Some(ref verify) = batch.verify {
        lfs_verify(&verify.href, &verify.header, token, sha256, size)?;
    }

    commit_lfs_file(repo_id, token, remote_filename, sha256, size, repo_type)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::test_support::{write_temp_bytes, CapturingCallbacks, EnvBaseGuard};
    use super::*;
    use serial_test::serial;
    use std::io::Read;

    // ─── CountingReader ────────────────────────────────────────────

    #[test]
    fn counting_reader_counts_bytes_read() {
        use std::sync::atomic::Ordering;
        let bytes = b"hello world".to_vec();
        let counter = std::sync::Arc::new(portable_atomic::AtomicU64::new(0));
        let mut reader = CountingReader {
            inner: bytes.as_slice(),
            counter: counter.clone(),
        };
        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
        assert_eq!(counter.load(Ordering::Relaxed), 5);

        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b" world");
        assert_eq!(counter.load(Ordering::Relaxed), 11);
    }

    #[test]
    fn counting_reader_counter_starts_at_zero() {
        use std::sync::atomic::Ordering;
        let counter = std::sync::Arc::new(portable_atomic::AtomicU64::new(0));
        let mut reader = CountingReader {
            inner: std::io::empty(),
            counter: counter.clone(),
        };
        let mut buf = [0u8; 16];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    // ─── upload_lfs orchestrator (batch → PUT → verify → commit) ───

    #[test]
    #[serial]
    fn upload_lfs_full_path_with_upload_action() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"payload");

        let put_url = format!("{}/lfs/up", server.url());
        let verify_url = format!("{}/lfs/v", server.url());

        let batch_mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "objects": [{
                        "actions": {
                            "upload": {"href": put_url, "header": {}},
                            "verify": {"href": verify_url, "header": {}}
                        }
                    }]
                })
                .to_string(),
            )
            .create();
        let put_mock = server.mock("PUT", "/lfs/up").with_status(200).create();
        let verify_mock = server.mock("POST", "/lfs/v").with_status(200).create();
        let commit_mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        upload_lfs("org/repo", "t", &path, "p.bin", 7, "sha", &mut cb, "model").unwrap();
        batch_mock.assert();
        put_mock.assert();
        verify_mock.assert();
        commit_mock.assert();
    }

    #[test]
    #[serial]
    fn upload_lfs_skips_put_when_object_already_present() {
        // Batch returns no `actions.upload` ⇒ HF says the LFS object is
        // already stored. upload_lfs must skip the PUT and proceed
        // straight to verify (if present) + commit.
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"payload");

        let verify_url = format!("{}/lfs/v", server.url());

        let batch_mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "objects": [{
                        "actions": {
                            "verify": {"href": verify_url, "header": {}}
                        }
                    }]
                })
                .to_string(),
            )
            .create();
        let verify_mock = server.mock("POST", "/lfs/v").with_status(200).create();
        let commit_mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        upload_lfs("org/repo", "t", &path, "p.bin", 7, "sha", &mut cb, "model").unwrap();
        batch_mock.assert();
        verify_mock.assert();
        commit_mock.assert();
        assert!(cb
            .progress_calls
            .iter()
            .any(|(_, sent, total)| sent == total && *sent == 7));
    }

    #[test]
    #[serial]
    fn upload_lfs_no_verify_action_skips_verify_and_commits() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"payload");

        let put_url = format!("{}/lfs/up", server.url());

        let batch_mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "objects": [{
                        "actions": {
                            "upload": {"href": put_url, "header": {}}
                        }
                    }]
                })
                .to_string(),
            )
            .create();
        let put_mock = server.mock("PUT", "/lfs/up").with_status(200).create();
        let commit_mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        upload_lfs("org/repo", "t", &path, "p.bin", 7, "sha", &mut cb, "model").unwrap();
        batch_mock.assert();
        put_mock.assert();
        commit_mock.assert();
    }
}
