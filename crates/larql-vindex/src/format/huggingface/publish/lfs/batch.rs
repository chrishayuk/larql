//! LFS batch endpoint — request signed upload + verify URLs for one
//! object, parse the response into actions our caller dispatches on.

use std::collections::HashMap;

use crate::error::VindexError;

use super::super::hf_repo_url;
use super::super::protocol::{
    CONTENT_TYPE_LFS_JSON, HASH_ALGO_SHA256, LFS_OP_UPLOAD, LFS_OP_VERIFY, LFS_TRANSFER_BASIC,
};
use super::{LfsAction, LfsBatchResponse};

/// POST to the LFS batch endpoint asking for an upload URL for one
/// object. Returns the upload + verify actions (either or both may be
/// absent — an absent `upload` means the object is already stored).
pub(super) fn lfs_batch_upload(
    repo_id: &str,
    token: &str,
    sha256: &str,
    size: u64,
    repo_type: &str,
) -> Result<LfsBatchResponse, VindexError> {
    let url = format!(
        "{}.git/info/lfs/objects/batch",
        hf_repo_url(repo_type, repo_id)
    );
    let body = serde_json::json!({
        "operation":  LFS_OP_UPLOAD,
        "transfers":  [LFS_TRANSFER_BASIC],
        "hash_algo":  HASH_ALGO_SHA256,
        "objects":    [{"oid": sha256, "size": size}],
    });
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Accept", CONTENT_TYPE_LFS_JSON)
        .header("Content-Type", CONTENT_TYPE_LFS_JSON)
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("LFS batch failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!("LFS batch ({status}): {body}")));
    }
    let json: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("LFS batch JSON: {e}")))?;
    parse_lfs_batch_response(&json)
}

/// Parse the JSON body of an LFS batch response into the upload/verify
/// actions our caller dispatches on. Pulled out as a pure helper so the
/// JSON contract can be unit-tested without an HTTP server.
pub(super) fn parse_lfs_batch_response(
    json: &serde_json::Value,
) -> Result<LfsBatchResponse, VindexError> {
    let objects = json
        .get("objects")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("LFS batch response missing `objects`".into()))?;
    let obj = objects
        .first()
        .ok_or_else(|| VindexError::Parse("LFS batch objects[] empty".into()))?;

    if let Some(err) = obj.get("error") {
        return Err(VindexError::Parse(format!("LFS batch object error: {err}")));
    }

    let actions = obj.get("actions");
    let parse_action = |key: &str| -> Option<LfsAction> {
        let a = actions?.get(key)?;
        let href = a.get("href").and_then(|v| v.as_str())?.to_string();
        let header: HashMap<String, String> = a
            .get("header")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        Some(LfsAction { href, header })
    };
    Ok(LfsBatchResponse {
        upload: parse_action(LFS_OP_UPLOAD),
        verify: parse_action(LFS_OP_VERIFY),
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::test_support::EnvBaseGuard;
    use super::*;
    use serial_test::serial;

    // ─── parse_lfs_batch_response ──────────────────────────────────

    #[test]
    fn parse_lfs_batch_with_upload_and_verify_actions() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {
                        "href": "https://lfs.example/upload",
                        "header": {"Authorization": "Bearer x", "X-Amz": "sig"}
                    },
                    "verify": {
                        "href": "https://lfs.example/verify",
                        "header": {}
                    }
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        let upload = parsed.upload.expect("upload action present");
        assert_eq!(upload.href, "https://lfs.example/upload");
        assert_eq!(
            upload.header.get("Authorization").map(|s| s.as_str()),
            Some("Bearer x")
        );
        assert_eq!(upload.header.get("X-Amz").map(|s| s.as_str()), Some("sig"));
        let verify = parsed.verify.expect("verify action present");
        assert_eq!(verify.href, "https://lfs.example/verify");
        assert!(verify.header.is_empty());
    }

    #[test]
    fn parse_lfs_batch_with_no_upload_means_object_already_present() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "verify": {"href": "https://lfs.example/verify", "header": {}}
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        assert!(parsed.upload.is_none(), "upload action absent");
        assert!(parsed.verify.is_some(), "verify action present");
    }

    #[test]
    fn parse_lfs_batch_with_no_actions_returns_both_none() {
        let json = serde_json::json!({"objects": [{}]});
        let parsed = parse_lfs_batch_response(&json).unwrap();
        assert!(parsed.upload.is_none());
        assert!(parsed.verify.is_none());
    }

    #[test]
    fn parse_lfs_batch_missing_objects_array_errors() {
        let json = serde_json::json!({});
        let err = parse_lfs_batch_response(&json).expect_err("objects[] missing must error");
        assert!(err.to_string().contains("missing `objects`"));
    }

    #[test]
    fn parse_lfs_batch_empty_objects_array_errors() {
        let json = serde_json::json!({"objects": []});
        let err = parse_lfs_batch_response(&json).expect_err("empty objects[] must error");
        assert!(err.to_string().contains("objects[] empty"));
    }

    #[test]
    fn parse_lfs_batch_per_object_error_surfaces() {
        let json = serde_json::json!({
            "objects": [{
                "error": {"code": 422, "message": "object too large"}
            }]
        });
        let err = parse_lfs_batch_response(&json).expect_err("inline object error");
        let msg = err.to_string();
        assert!(msg.contains("LFS batch object error"), "{msg}");
        assert!(msg.contains("too large"), "{msg}");
    }

    #[test]
    fn parse_lfs_batch_action_without_href_is_skipped() {
        let json = serde_json::json!({
            "objects": [{
                "actions": {
                    "upload": {"header": {}}
                }
            }]
        });
        let parsed = parse_lfs_batch_response(&json).unwrap();
        assert!(parsed.upload.is_none());
    }

    // ─── lfs_batch_upload (HTTP-mocked) ─────────────────────────────

    #[test]
    #[serial]
    fn lfs_batch_upload_returns_actions_from_server() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .match_header("authorization", "Bearer t")
            .match_header("accept", "application/vnd.git-lfs+json")
            .match_header("content-type", "application/vnd.git-lfs+json")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "operation": "upload",
                "transfers": ["basic"],
                "hash_algo": "sha256",
            })))
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "objects": [{
                        "actions": {
                            "upload": {"href": "https://lfs.example/up", "header": {"X-Sig": "abc"}},
                            "verify": {"href": "https://lfs.example/v", "header": {}}
                        }
                    }]
                })
                .to_string(),
            )
            .create();

        let resp = lfs_batch_upload("org/repo", "t", "deadbeef", 1024, "model").unwrap();
        mock.assert();
        let upload = resp.upload.expect("upload action present");
        assert_eq!(upload.href, "https://lfs.example/up");
        assert_eq!(upload.header.get("X-Sig").map(|s| s.as_str()), Some("abc"));
        assert!(resp.verify.is_some());
    }

    #[test]
    #[serial]
    fn lfs_batch_upload_dataset_repo_path() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/datasets/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(r#"{"objects":[{"actions":{}}]}"#)
            .create();

        let _ = lfs_batch_upload("org/repo", "t", "x", 1, "dataset").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn lfs_batch_upload_http_error_propagates() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(500)
            .with_body("boom")
            .create();

        let err = lfs_batch_upload("org/repo", "t", "x", 1, "model").expect_err("500 must error");
        mock.assert();
        assert!(err.to_string().contains("500"));
    }

    #[test]
    #[serial]
    fn lfs_batch_upload_per_object_error_surfaces() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(r#"{"objects":[{"error":{"code":422,"message":"too big"}}]}"#)
            .create();

        let err = lfs_batch_upload("org/repo", "t", "x", 1, "model").expect_err("inline error");
        mock.assert();
        assert!(err.to_string().contains("LFS batch object error"));
    }
}
