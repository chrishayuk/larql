//! HuggingFace API helpers for repo-level state — remote LFS index lookup
//! and repo creation. Both are blocking HTTP calls used by the publish
//! orchestrator before any per-file upload runs.

use std::collections::HashMap;

use crate::error::VindexError;

use super::protocol::{hf_base, repo_type_plural, HTTP_STATUS_CONFLICT};

/// List remote files and return `filename → lfs.oid` for every LFS-tracked
/// file at the repo root. Files without an `lfs.oid` (git-tracked small
/// text) are omitted; callers skip only what's in the map.
pub(super) fn fetch_remote_lfs_oids(
    repo_id: &str,
    token: &str,
    repo_type: &str,
) -> Result<HashMap<String, String>, VindexError> {
    let plural = repo_type_plural(repo_type);
    let base = hf_base();
    let url = format!("{base}/api/{plural}/{repo_id}/tree/main?recursive=true");
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF tree fetch failed: {e}")))?;

    if !resp.status().is_success() {
        // 404 on a fresh repo → no remote files, can't skip anything.
        return Ok(HashMap::new());
    }

    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF tree JSON: {e}")))?;
    Ok(parse_lfs_oid_index(&body))
}

/// Walk the HF tree-listing JSON and return `filename → lfs.oid` for
/// every LFS-tracked file. Files without an `lfs.oid` (small text /
/// directories) are omitted. Pulled out as a pure helper so the JSON
/// contract can be unit-tested without an HTTP server.
fn parse_lfs_oid_index(body: &serde_json::Value) -> HashMap<String, String> {
    let arr = match body.as_array() {
        Some(a) => a,
        None => return HashMap::new(),
    };

    let mut out = HashMap::new();
    for entry in arr {
        if entry.get("type").and_then(|v| v.as_str()) != Some("file") {
            continue;
        }
        let path = match entry.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => continue,
        };
        if let Some(lfs_oid) = entry
            .get("lfs")
            .and_then(|v| v.get("oid"))
            .and_then(|v| v.as_str())
        {
            out.insert(path.to_string(), lfs_oid.to_string());
        }
    }
    out
}

pub(super) fn create_hf_repo(
    repo_id: &str,
    token: &str,
    repo_type: &str,
    private: bool,
) -> Result<(), VindexError> {
    let client = reqwest::blocking::Client::new();
    let url = format!("{}/api/repos/create", hf_base());
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({
            "name": repo_id.split('/').next_back().unwrap_or(repo_id),
            "type": repo_type,
            "private": private,
        }))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF API error: {e}")))?;

    // 409 Conflict = already exists, that's fine
    if resp.status().is_success() || resp.status().as_u16() == HTTP_STATUS_CONFLICT {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "HF repo create failed ({status}): {body}"
        )))
    }
}

/// Flip an already-created repo's visibility. Used at RELEASE
/// (docs/vindex-factory.md §7/§8.3): a build publishes PRIVATE, verifies
/// the published bytes, and only then flips PUBLIC — nothing goes live
/// unverified. `PUT /api/{repo_type}s/{repo_id}/settings` with
/// `{"private": bool}` (HF Hub API, confirmed against the OpenAPI spec
/// at huggingface.co/.well-known/openapi.md — the modern `visibility`
/// enum field also works, but `private` is the simpler two-state case
/// this needs).
pub(super) fn update_repo_visibility(
    repo_id: &str,
    token: &str,
    repo_type: &str,
    private: bool,
) -> Result<(), VindexError> {
    let plural = repo_type_plural(repo_type);
    let url = format!("{}/api/{plural}/{repo_id}/settings", hf_base());
    let client = reqwest::blocking::Client::new();
    let resp = client
        .put(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({ "private": private }))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF API error: {e}")))?;

    if resp.status().is_success() {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "HF repo visibility update failed ({status}): {body}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_oid_index_extracts_lfs_files_only() {
        // Mixed entries: LFS file, non-LFS file (small text), directory.
        // Only the LFS entry should appear in the map.
        let body = serde_json::json!([
            {
                "type": "file",
                "path": "weights.bin",
                "lfs": {"oid": "abc123", "size": 1024}
            },
            {
                "type": "file",
                "path": "index.json"
                // no lfs key — git-tracked small text
            },
            {
                "type": "directory",
                "path": "layers",
                "lfs": {"oid": "should-not-appear"}
            }
        ]);
        let map = parse_lfs_oid_index(&body);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("weights.bin").map(|s| s.as_str()), Some("abc123"));
    }

    #[test]
    fn parse_oid_index_handles_subdir_paths() {
        // HF returns paths like "layers/layer_00.weights" — they should
        // round-trip through the map verbatim (the publish code uses
        // them as filename keys).
        let body = serde_json::json!([
            {
                "type": "file",
                "path": "layers/layer_00.weights",
                "lfs": {"oid": "deadbeef"}
            }
        ]);
        let map = parse_lfs_oid_index(&body);
        assert_eq!(
            map.get("layers/layer_00.weights").map(|s| s.as_str()),
            Some("deadbeef"),
        );
    }

    #[test]
    fn parse_oid_index_non_array_body_yields_empty_map() {
        // Fresh repo / unauth → HF can return a non-array body. Non-fatal:
        // caller falls back to "upload everything".
        let body = serde_json::json!({"error": "not found"});
        assert!(parse_lfs_oid_index(&body).is_empty());
    }

    #[test]
    fn parse_oid_index_empty_array_yields_empty_map() {
        let body = serde_json::json!([]);
        assert!(parse_lfs_oid_index(&body).is_empty());
    }

    #[test]
    fn parse_oid_index_missing_path_skips_entry() {
        // Defensive: malformed entries don't poison the whole walk.
        let body = serde_json::json!([
            {"type": "file", "lfs": {"oid": "x"}},
            {
                "type": "file",
                "path": "good.bin",
                "lfs": {"oid": "y"}
            }
        ]);
        let map = parse_lfs_oid_index(&body);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("good.bin").map(|s| s.as_str()), Some("y"));
    }

    // ─── HTTP-mocked integration tests ─────────────────────────────
    //
    // These set `LARQL_HF_TEST_BASE` to a per-test mockito URL and
    // serialize via `#[serial]` because env vars are process-global.

    use crate::format::huggingface::publish::protocol::TEST_BASE_ENV;
    use serial_test::serial;

    /// RAII-style env-var override: sets the var, restores on drop.
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

    #[test]
    #[serial]
    fn fetch_remote_lfs_oids_parses_tree_response() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
            .match_header("authorization", "Bearer t")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!([
                    {"type": "file", "path": "weights.bin",
                     "lfs": {"oid": "abc"}},
                    {"type": "file", "path": "index.json"}
                ])
                .to_string(),
            )
            .create();

        let map = fetch_remote_lfs_oids("org/repo", "t", "model").unwrap();
        mock.assert();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("weights.bin").map(|s| s.as_str()), Some("abc"));
    }

    #[test]
    #[serial]
    fn fetch_remote_lfs_oids_dataset_uses_datasets_path_segment() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("GET", "/api/datasets/org/repo/tree/main?recursive=true")
            .with_status(200)
            .with_body("[]")
            .create();

        let map = fetch_remote_lfs_oids("org/repo", "t", "dataset").unwrap();
        mock.assert();
        assert!(map.is_empty());
    }

    #[test]
    #[serial]
    fn fetch_remote_lfs_oids_404_returns_empty_map() {
        // Fresh repo: tree endpoint 404s before the first commit.
        // Caller falls back to "upload everything", so this MUST NOT
        // surface as an error.
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
            .with_status(404)
            .create();

        let map = fetch_remote_lfs_oids("org/repo", "t", "model").unwrap();
        mock.assert();
        assert!(map.is_empty());
    }

    #[test]
    #[serial]
    fn fetch_remote_lfs_oids_non_array_body_yields_empty_map() {
        // 200 OK with a JSON object (not array) body — defensive path
        // already covered by the pure parser test, but exercise the
        // full HTTP path here too.
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
            .with_status(200)
            .with_body(r#"{"error": "weird"}"#)
            .create();

        let map = fetch_remote_lfs_oids("org/repo", "t", "model").unwrap();
        mock.assert();
        assert!(map.is_empty());
    }

    #[test]
    #[serial]
    fn create_hf_repo_success() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/api/repos/create")
            .match_header("authorization", "Bearer t")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"name": "repo", "type": "model"}),
            ))
            .with_status(200)
            .with_body("{}")
            .create();

        create_hf_repo("org/repo", "t", "model", false).unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn create_hf_repo_private_true_sets_the_field() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/api/repos/create")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"private": true}),
            ))
            .with_status(200)
            .with_body("{}")
            .create();

        create_hf_repo("org/repo", "t", "model", true).unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn create_hf_repo_409_conflict_is_ok() {
        // 409 Conflict means "already exists" — that's fine, the publish
        // path proceeds to commit. Must NOT surface as an error.
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/api/repos/create")
            .with_status(409)
            .with_body("conflict")
            .create();

        create_hf_repo("org/repo", "t", "model", false).unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn create_hf_repo_other_error_propagates() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/api/repos/create")
            .with_status(500)
            .with_body("boom")
            .create();

        let err = create_hf_repo("org/repo", "t", "model", false).expect_err("500 must error");
        mock.assert();
        let msg = err.to_string();
        assert!(msg.contains("500"), "{msg}");
    }

    #[test]
    #[serial]
    fn create_hf_repo_uses_last_path_segment_as_name() {
        // HF's repos/create body uses just the repo name (the part after
        // the slash), not the full owner/repo. A repo_id without a slash
        // should pass through verbatim.
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("POST", "/api/repos/create")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"name": "loose-repo"}),
            ))
            .with_status(200)
            .create();

        create_hf_repo("loose-repo", "t", "model", false).unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn update_repo_visibility_success() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("PUT", "/api/models/org/repo/settings")
            .match_header("authorization", "Bearer t")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"private": false}),
            ))
            .with_status(200)
            .with_body("{}")
            .create();

        update_repo_visibility("org/repo", "t", "model", false).unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn update_repo_visibility_uses_dataset_plural() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("PUT", "/api/datasets/org/repo/settings")
            .with_status(200)
            .with_body("{}")
            .create();

        update_repo_visibility("org/repo", "t", "dataset", true).unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn update_repo_visibility_error_propagates() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());

        let mock = server
            .mock("PUT", "/api/models/org/repo/settings")
            .with_status(403)
            .with_body("forbidden")
            .create();

        let err =
            update_repo_visibility("org/repo", "t", "model", false).expect_err("403 must error");
        mock.assert();
        let msg = err.to_string();
        assert!(msg.contains("403"), "{msg}");
    }
}
