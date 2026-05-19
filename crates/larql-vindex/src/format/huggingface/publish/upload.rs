//! Per-file upload pipeline. Decides between LFS and inline-base64
//! commits based on HuggingFace's preupload response, then dispatches
//! into [`super::lfs`] or runs the regular commit inline.

use std::path::Path;

use crate::error::VindexError;

use super::lfs::upload_lfs;
use super::protocol::{hf_base, repo_type_plural, CONTENT_TYPE_NDJSON, HF_PREUPLOAD_SAMPLE_BYTES};
use super::PublishCallbacks;

/// Upload a single file to a HuggingFace dataset repo via the real HF
/// protocol:
///
///   1. **Preupload** — `POST /api/datasets/{repo}/preupload/main` with a
///      base64 sample of the first 512 bytes. HF decides `lfs` vs `regular`
///      based on size + `.gitattributes`.
///   2. **LFS batch** (LFS path only) — `POST {repo}.git/info/lfs/objects/batch`
///      returns a signed upload URL or tells us the file is already there.
///   3. **Streaming PUT** to the signed URL, ticking `on_file_progress` as
///      bytes flow. `CountingReader` + worker thread keeps the main thread
///      free to poll.
///   4. **Verify** — `POST {verify.href}` with `{oid, size}`.
///   5. **Commit** — `POST /api/datasets/{repo}/commit/main` as NDJSON with
///      a `lfsFile` (LFS) or `file` (regular, base64-inline) operation.
///
/// The old single-PUT "upload endpoint" this replaced was fictional — HF
/// never exposed `PUT /api/datasets/{repo}/upload/main/{file}`. Requests
/// to it 404 after the first few megabytes of body, which was the bug
/// that triggered this rewrite.
pub(super) fn upload_file_to_hf(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    callbacks: &mut dyn PublishCallbacks,
    repo_type: &str,
) -> Result<(), VindexError> {
    let size = std::fs::metadata(local_path)?.len();
    let sha256 = crate::format::checksums::sha256_file(local_path)?;

    let decision = preupload_decide(repo_id, token, remote_filename, local_path, size, repo_type)?;

    if decision.should_ignore {
        // HF's preupload told us the server would ignore this path
        // (matches `.gitignore` / similar). Skip silently.
        return Ok(());
    }

    match decision.mode.as_str() {
        "lfs" => upload_lfs(
            repo_id,
            token,
            local_path,
            remote_filename,
            size,
            &sha256,
            callbacks,
            repo_type,
        ),
        "regular" => upload_regular(
            repo_id,
            token,
            local_path,
            remote_filename,
            size,
            callbacks,
            repo_type,
        ),
        other => Err(VindexError::Parse(format!(
            "HF preupload returned unknown mode `{other}` for {remote_filename}"
        ))),
    }
}

#[derive(Debug)]
struct PreuploadDecision {
    mode: String,
    should_ignore: bool,
}

/// Call `POST /api/datasets/{repo}/preupload/main` for a single file and
/// return whether HF wants it uploaded via LFS or inlined in a regular
/// commit. HF requires a base64 sample of the first ~512 bytes so it
/// can sniff the file's format (text vs binary, etc.).
fn preupload_decide(
    repo_id: &str,
    token: &str,
    remote_filename: &str,
    local_path: &Path,
    size: u64,
    repo_type: &str,
) -> Result<PreuploadDecision, VindexError> {
    use base64::Engine;
    use std::io::Read;

    // Read up to the configured byte count for the format-sniff sample. HF accepts a
    // smaller sample for small files without complaint.
    let mut sample_buf = vec![0u8; HF_PREUPLOAD_SAMPLE_BYTES.min(size as usize)];
    if !sample_buf.is_empty() {
        let mut file = std::fs::File::open(local_path)?;
        file.read_exact(&mut sample_buf)?;
    }
    let sample_b64 = base64::prelude::BASE64_STANDARD.encode(&sample_buf);

    let plural = repo_type_plural(repo_type);
    let url = format!("{}/api/{plural}/{repo_id}/preupload/main", hf_base());
    let body = serde_json::json!({
        "files": [{
            "path":   remote_filename,
            "sample": sample_b64,
            "size":   size,
        }],
    });
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("preupload failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "preupload ({status}) for {remote_filename}: {body}"
        )));
    }
    let json: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("preupload JSON: {e}")))?;
    parse_preupload_response(&json)
}

/// Parse the JSON body of an HF preupload response into our routing
/// decision. Pulled out as a pure helper so the JSON contract can be
/// unit-tested without an HTTP server.
fn parse_preupload_response(json: &serde_json::Value) -> Result<PreuploadDecision, VindexError> {
    let files = json
        .get("files")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("preupload response missing `files`".into()))?;
    let entry = files
        .first()
        .ok_or_else(|| VindexError::Parse("preupload response files[] empty".into()))?;
    let mode = entry
        .get("uploadMode")
        .and_then(|v| v.as_str())
        .unwrap_or("lfs")
        .to_string();
    let should_ignore = entry
        .get("shouldIgnore")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok(PreuploadDecision {
        mode,
        should_ignore,
    })
}

/// Small-file path: commit directly with the content inlined as base64
/// in the NDJSON commit body. HF's preupload flags tiny text files for
/// this path.
fn upload_regular(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    size: u64,
    callbacks: &mut dyn PublishCallbacks,
    repo_type: &str,
) -> Result<(), VindexError> {
    use base64::Engine;
    let data = std::fs::read(local_path)?;
    // Fire start+end of the progress bar even though we don't stream —
    // keeps the UX consistent across file sizes.
    callbacks.on_file_progress(remote_filename, 0, size);
    let encoded = base64::prelude::BASE64_STANDARD.encode(&data);

    let plural = repo_type_plural(repo_type);
    let url = format!("{}/api/{plural}/{repo_id}/commit/main", hf_base());
    let mut ndjson = String::new();
    ndjson.push_str(
        &serde_json::to_string(&serde_json::json!({
            "key": "header",
            "value": {
                "summary": format!("Upload {remote_filename}"),
            },
        }))
        .unwrap(),
    );
    ndjson.push('\n');
    ndjson.push_str(
        &serde_json::to_string(&serde_json::json!({
            "key": "file",
            "value": {
                "path":     remote_filename,
                "encoding": "base64",
                "content":  encoded,
            },
        }))
        .unwrap(),
    );
    ndjson.push('\n');

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", CONTENT_TYPE_NDJSON)
        .body(ndjson)
        .send()
        .map_err(|e| VindexError::Parse(format!("commit (regular) failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "commit (regular) {remote_filename} ({status}): {body}"
        )));
    }
    callbacks.on_file_progress(remote_filename, size, size);
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn parse_preupload_lfs_mode() {
        let json = serde_json::json!({
            "files": [{"uploadMode": "lfs"}]
        });
        let d = parse_preupload_response(&json).unwrap();
        assert_eq!(d.mode, "lfs");
        assert!(!d.should_ignore);
    }

    #[test]
    fn parse_preupload_regular_mode() {
        let json = serde_json::json!({
            "files": [{"uploadMode": "regular"}]
        });
        let d = parse_preupload_response(&json).unwrap();
        assert_eq!(d.mode, "regular");
    }

    #[test]
    fn parse_preupload_should_ignore_true() {
        // HF flags files matching `.gitignore` patterns; caller skips them.
        let json = serde_json::json!({
            "files": [{"uploadMode": "regular", "shouldIgnore": true}]
        });
        let d = parse_preupload_response(&json).unwrap();
        assert!(d.should_ignore);
    }

    #[test]
    fn parse_preupload_missing_uploadmode_defaults_to_lfs() {
        // Defensive default: if HF doesn't tell us, route to LFS (the
        // safer choice — small files commit-fail with a clearer error
        // than corruption from misrouting binary as base64-text).
        let json = serde_json::json!({"files": [{}]});
        let d = parse_preupload_response(&json).unwrap();
        assert_eq!(d.mode, "lfs");
        assert!(!d.should_ignore);
    }

    #[test]
    fn parse_preupload_missing_files_array_errors() {
        let json = serde_json::json!({});
        let err = parse_preupload_response(&json).expect_err("missing files[] errors");
        assert!(err.to_string().contains("files"));
    }

    #[test]
    fn parse_preupload_empty_files_array_errors() {
        let json = serde_json::json!({"files": []});
        let err = parse_preupload_response(&json).expect_err("empty files[] errors");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn parse_preupload_unknown_mode_passes_through() {
        // Unknown modes flow through to the upload_file_to_hf dispatch
        // which surfaces them with a targeted error. Don't reject here.
        let json = serde_json::json!({
            "files": [{"uploadMode": "future-mode"}]
        });
        let d = parse_preupload_response(&json).unwrap();
        assert_eq!(d.mode, "future-mode");
    }

    // ─── HTTP-mocked integration tests ─────────────────────────────

    use super::super::PublishCallbacks;
    use crate::format::huggingface::publish::protocol::TEST_BASE_ENV;
    use serial_test::serial;
    use std::io::Write as _;

    struct EnvBaseGuard {
        prev: Option<String>,
    }
    impl EnvBaseGuard {
        fn new(value: &str) -> Self {
            let prev = std::env::var(TEST_BASE_ENV).ok();
            std::env::set_var(TEST_BASE_ENV, value);
            Self { prev }
        }
    }
    impl Drop for EnvBaseGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var(TEST_BASE_ENV, v),
                None => std::env::remove_var(TEST_BASE_ENV),
            }
        }
    }

    #[derive(Default)]
    struct CapturingCallbacks {
        progress_calls: Vec<(String, u64, u64)>,
    }
    impl PublishCallbacks for CapturingCallbacks {
        fn on_file_progress(&mut self, filename: &str, bytes_sent: u64, total_bytes: u64) {
            self.progress_calls
                .push((filename.to_string(), bytes_sent, total_bytes));
        }
    }

    fn write_temp_bytes(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        (dir, path)
    }

    #[test]
    #[serial]
    fn preupload_decide_returns_lfs_mode_for_large_file() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(&vec![0u8; 1024]);

        let mock = server
            .mock("POST", "/api/models/org/repo/preupload/main")
            .match_header("authorization", "Bearer t")
            .with_status(200)
            .with_body(r#"{"files":[{"uploadMode":"lfs"}]}"#)
            .create();

        let d = preupload_decide("org/repo", "t", "weights.bin", &path, 1024, "model").unwrap();
        mock.assert();
        assert_eq!(d.mode, "lfs");
        assert!(!d.should_ignore);
    }

    #[test]
    #[serial]
    fn preupload_decide_dataset_uses_datasets_path() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"small");

        let mock = server
            .mock("POST", "/api/datasets/org/repo/preupload/main")
            .with_status(200)
            .with_body(r#"{"files":[{"uploadMode":"regular"}]}"#)
            .create();

        let d = preupload_decide("org/repo", "t", "f.txt", &path, 5, "dataset").unwrap();
        mock.assert();
        assert_eq!(d.mode, "regular");
    }

    #[test]
    #[serial]
    fn preupload_decide_propagates_should_ignore() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"x");

        let mock = server
            .mock("POST", "/api/models/org/repo/preupload/main")
            .with_status(200)
            .with_body(r#"{"files":[{"uploadMode":"regular","shouldIgnore":true}]}"#)
            .create();

        let d = preupload_decide("org/repo", "t", "ignored", &path, 1, "model").unwrap();
        mock.assert();
        assert!(d.should_ignore);
    }

    #[test]
    #[serial]
    fn preupload_decide_http_error_propagates() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"x");

        let mock = server
            .mock("POST", "/api/models/org/repo/preupload/main")
            .with_status(500)
            .with_body("server boom")
            .create();

        let err =
            preupload_decide("org/repo", "t", "f", &path, 1, "model").expect_err("500 must error");
        mock.assert();
        let msg = err.to_string();
        assert!(msg.contains("500"), "{msg}");
    }

    #[test]
    #[serial]
    fn upload_regular_inlines_file_as_base64_commit() {
        // Small text file: HF expects an NDJSON commit body with the
        // file base64-encoded inline.
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"hello");

        let mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .match_header("authorization", "Bearer t")
            .match_header("content-type", "application/x-ndjson")
            .match_body(mockito::Matcher::Regex(r#""path":\s*"index\.json""#.into()))
            .match_body(mockito::Matcher::Regex(r#""encoding":\s*"base64""#.into()))
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        upload_regular("org/repo", "t", &path, "index.json", 5, &mut cb, "model").unwrap();
        mock.assert();
        // Bar should hit 100% by the end of the upload.
        assert!(
            cb.progress_calls
                .iter()
                .any(|(name, sent, total)| name == "index.json" && *sent == 5 && *total == 5),
            "missing 100% progress tick: {:?}",
            cb.progress_calls
        );
    }

    #[test]
    #[serial]
    fn upload_regular_http_error_propagates_with_filename_in_message() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"x");

        let mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .with_status(403)
            .with_body("denied")
            .create();

        let mut cb = CapturingCallbacks::default();
        let err = upload_regular("org/repo", "t", &path, "rejected.bin", 1, &mut cb, "model")
            .expect_err("403 must error");
        mock.assert();
        let msg = err.to_string();
        assert!(msg.contains("rejected.bin"), "filename in error: {msg}");
        assert!(msg.contains("403"), "status in error: {msg}");
    }

    #[test]
    #[serial]
    fn upload_file_to_hf_dispatches_regular_path() {
        // Full orchestrator: preupload says "regular", upload_regular
        // commits inline. One mockito server handles both endpoints.
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"tiny");

        let preupload_mock = server
            .mock("POST", "/api/models/org/repo/preupload/main")
            .with_status(200)
            .with_body(r#"{"files":[{"uploadMode":"regular"}]}"#)
            .create();
        let commit_mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        upload_file_to_hf("org/repo", "t", &path, "small.txt", &mut cb, "model").unwrap();
        preupload_mock.assert();
        commit_mock.assert();
    }

    #[test]
    #[serial]
    fn upload_file_to_hf_skips_when_should_ignore() {
        // HF flags the file as ignored — upload short-circuits with
        // Ok(()) and never hits the commit endpoint.
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"x");

        let preupload_mock = server
            .mock("POST", "/api/models/org/repo/preupload/main")
            .with_status(200)
            .with_body(r#"{"files":[{"uploadMode":"regular","shouldIgnore":true}]}"#)
            .create();
        // No commit mock — if we hit it, mockito returns 501 and the
        // call would fail.

        let mut cb = CapturingCallbacks::default();
        upload_file_to_hf("org/repo", "t", &path, "ignore-me", &mut cb, "model").unwrap();
        preupload_mock.assert();
    }

    #[test]
    #[serial]
    fn upload_file_to_hf_unknown_mode_errors_with_filename() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"x");

        let _preupload_mock = server
            .mock("POST", "/api/models/org/repo/preupload/main")
            .with_status(200)
            .with_body(r#"{"files":[{"uploadMode":"future-mode"}]}"#)
            .create();

        let mut cb = CapturingCallbacks::default();
        let err = upload_file_to_hf("org/repo", "t", &path, "weird.bin", &mut cb, "model")
            .expect_err("unknown mode must error");
        let msg = err.to_string();
        assert!(msg.contains("future-mode"), "{msg}");
        assert!(msg.contains("weird.bin"), "{msg}");
    }
}
