//! Post-PUT finalisation: verify the LFS object reached storage, then
//! commit the pointer file via the HF NDJSON commit endpoint.

use std::collections::HashMap;

use crate::error::VindexError;

use super::super::protocol::{
    hf_base, repo_type_plural, CONTENT_TYPE_LFS_JSON, CONTENT_TYPE_NDJSON, HASH_ALGO_SHA256,
};

/// POST `{oid, size}` to the verify URL the LFS batch returned. HF uses
/// this to confirm the object made it to storage intact before the
/// commit references it.
pub(super) fn lfs_verify(
    href: &str,
    extra_headers: &HashMap<String, String>,
    token: &str,
    sha256: &str,
    size: u64,
) -> Result<(), VindexError> {
    let body = serde_json::json!({"oid": sha256, "size": size});
    let client = reqwest::blocking::Client::new();
    let mut req = client
        .post(href)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", CONTENT_TYPE_LFS_JSON)
        .header("Content-Type", CONTENT_TYPE_LFS_JSON);
    for (k, v) in extra_headers {
        req = req.header(k.as_str(), v.as_str());
    }
    let resp = req
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("LFS verify failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!("LFS verify ({status}): {body}")));
    }
    Ok(())
}

/// Commit a single LFS pointer into the repo via NDJSON. HF's commit
/// API is one request per change set; we commit per file for simplicity
/// (batching every file into one commit is a future optimisation).
pub(super) fn commit_lfs_file(
    repo_id: &str,
    token: &str,
    remote_filename: &str,
    sha256: &str,
    size: u64,
    repo_type: &str,
) -> Result<(), VindexError> {
    let plural = repo_type_plural(repo_type);
    let url = format!("{}/api/{plural}/{repo_id}/commit/main", hf_base());
    let mut ndjson = String::new();
    ndjson.push_str(
        &serde_json::to_string(&serde_json::json!({
            "key": "header",
            "value": {"summary": format!("Upload {remote_filename}")},
        }))
        .unwrap(),
    );
    ndjson.push('\n');
    ndjson.push_str(
        &serde_json::to_string(&serde_json::json!({
            "key": "lfsFile",
            "value": {
                "path": remote_filename,
                "algo": HASH_ALGO_SHA256,
                "oid":  sha256,
                "size": size,
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
        .map_err(|e| VindexError::Parse(format!("commit (LFS) failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "commit (LFS) {remote_filename} ({status}): {body}"
        )));
    }
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::test_support::EnvBaseGuard;
    use super::*;
    use serial_test::serial;

    // ── lfs_verify ──────────────────────────────────────────────────

    #[test]
    #[serial]
    fn lfs_verify_posts_oid_and_size() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/lfs/verify")
            .match_header("authorization", "Bearer t")
            .match_header("accept", "application/vnd.git-lfs+json")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"oid": "sha", "size": 42}),
            ))
            .with_status(200)
            .create();

        let href = format!("{}/lfs/verify", server.url());
        let extra: HashMap<String, String> = HashMap::new();
        lfs_verify(&href, &extra, "t", "sha", 42).unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn lfs_verify_http_error_propagates() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server.mock("POST", "/lfs/verify").with_status(500).create();

        let href = format!("{}/lfs/verify", server.url());
        let extra: HashMap<String, String> = HashMap::new();
        let err = lfs_verify(&href, &extra, "t", "sha", 42).expect_err("500 errors");
        mock.assert();
        assert!(err.to_string().contains("500"));
    }

    // ── commit_lfs_file ─────────────────────────────────────────────

    #[test]
    #[serial]
    fn commit_lfs_file_posts_ndjson_pointer() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .match_header("authorization", "Bearer t")
            .match_header("content-type", "application/x-ndjson")
            .match_body(mockito::Matcher::Regex(r#""key":\s*"lfsFile""#.into()))
            .match_body(mockito::Matcher::Regex(r#""algo":\s*"sha256""#.into()))
            .match_body(mockito::Matcher::Regex(r#""path":\s*"file\.bin""#.into()))
            .with_status(200)
            .create();

        commit_lfs_file("org/repo", "t", "file.bin", "deadbeef", 100, "model").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn commit_lfs_file_dataset_uses_datasets_path() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/api/datasets/org/repo/commit/main")
            .with_status(200)
            .create();

        commit_lfs_file("org/repo", "t", "file.bin", "deadbeef", 100, "dataset").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn commit_lfs_file_http_error_includes_filename() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .with_status(409)
            .with_body("conflict")
            .create();

        let err =
            commit_lfs_file("org/repo", "t", "conf.bin", "x", 1, "model").expect_err("409 errors");
        mock.assert();
        let msg = err.to_string();
        assert!(msg.contains("conf.bin"), "{msg}");
        assert!(msg.contains("409"), "{msg}");
    }
}
