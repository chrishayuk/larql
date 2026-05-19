//! HuggingFace collection / repo discovery — listing + existence
//! probes used by the CLI to wire vindexes into HF collections.
//!
//! Carved out of the monolithic `huggingface.rs` in the 2026-04-25
//! reorg. See `super::mod.rs` for the module map.
//!
//! Module layout (round-6 split, 2026-05-10):
//! - `collection`  — find/create/add-item/fetch-items HF endpoints.
//! - `repo`        — `repo_exists` / `dataset_repo_exists` HEAD probes.
//! - `mod` (here)  — `CollectionItem` type + `ensure_collection` orchestrator.
//! - `test_support` (cfg(test)) — shared test fixtures.

mod collection;
mod repo;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod test_support;

use crate::error::VindexError;

use super::publish::get_hf_token;
use super::publish::protocol::hf_base;

pub use collection::{add_collection_item, fetch_collection_items};
pub use repo::{dataset_repo_exists, repo_exists};

use collection::{create_collection, find_collection_slug};

/// One repo in a collection.
#[derive(Clone, Debug)]
pub struct CollectionItem {
    /// Repo id (`owner/name`). Full form including namespace.
    pub repo_id: String,
    /// `"model"` (vindex repos, default) or `"dataset"`.
    pub repo_type: String,
    /// Optional short note rendered on the collection card.
    pub note: Option<String>,
}

/// Ensure a collection titled `title` exists in `namespace`, then add
/// every item to it. Idempotent: re-runs reuse the slug (matched by
/// case-insensitive title) and treat HTTP 409 on add-item as success.
/// Returns the collection URL on success.
pub fn ensure_collection(
    namespace: &str,
    title: &str,
    description: Option<&str>,
    items: &[CollectionItem],
) -> Result<String, VindexError> {
    let token = get_hf_token()?;
    let slug = match find_collection_slug(namespace, title, &token)? {
        Some(existing) => existing,
        None => create_collection(namespace, title, description, &token)?,
    };
    for item in items {
        add_collection_item(&slug, item, &token)?;
    }
    Ok(format!("{}/collections/{slug}", hf_base()))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::is_hf_path;
    use super::test_support::TestEnvGuard;
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_is_hf_path() {
        assert!(is_hf_path("hf://chrishayuk/gemma-3-4b-it-vindex"));
        assert!(is_hf_path("hf://user/repo@v1.0"));
        assert!(!is_hf_path("./local.vindex"));
        assert!(!is_hf_path("/absolute/path"));
    }

    #[test]
    fn test_parse_hf_path() {
        let path = "hf://chrishayuk/gemma-3-4b-it-vindex@v2.0";
        let stripped = path.strip_prefix("hf://").unwrap();
        let (repo, rev) = stripped.split_once('@').unwrap();
        assert_eq!(repo, "chrishayuk/gemma-3-4b-it-vindex");
        assert_eq!(rev, "v2.0");
    }

    // ── ensure_collection orchestrator ──────────────────────────────

    #[test]
    #[serial]
    fn ensure_collection_creates_when_missing() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(200)
            .with_body("[]")
            .create();
        let create_mock = server
            .mock("POST", "/api/collections")
            .with_status(200)
            .with_body(r#"{"slug": "org/new-title-abc"}"#)
            .create();
        let add_mock = server
            .mock("POST", "/api/collections/org/new-title-abc/items")
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        let url = ensure_collection("org", "New Title", None, &[item]).unwrap();
        list_mock.assert();
        create_mock.assert();
        add_mock.assert();
        assert!(url.ends_with("/collections/org/new-title-abc"));
    }

    #[test]
    #[serial]
    fn ensure_collection_reuses_existing_slug_case_insensitive() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(200)
            .with_body(
                serde_json::json!([
                    {"title": "MY EXISTING TITLE", "slug": "org/existing-abc"}
                ])
                .to_string(),
            )
            .create();
        let add_mock = server
            .mock("POST", "/api/collections/org/existing-abc/items")
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        let url = ensure_collection("org", "my existing title", None, &[item]).unwrap();
        list_mock.assert();
        add_mock.assert();
        assert!(url.ends_with("/collections/org/existing-abc"));
    }

    #[test]
    #[serial]
    fn ensure_collection_uses_409_slug_when_create_conflicts() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(200)
            .with_body("[]")
            .create();
        let create_mock = server
            .mock("POST", "/api/collections")
            .with_status(409)
            .with_body(r#"{"slug": "org/raced-abc"}"#)
            .create();
        let add_mock = server
            .mock("POST", "/api/collections/org/raced-abc/items")
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        ensure_collection("org", "Raced Title", None, &[item]).unwrap();
        list_mock.assert();
        create_mock.assert();
        add_mock.assert();
    }

    #[test]
    #[serial]
    fn ensure_collection_create_response_missing_slug_errors() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(200)
            .with_body("[]")
            .create();
        let create_mock = server
            .mock("POST", "/api/collections")
            .with_status(200)
            .with_body(r#"{"oops": "no slug"}"#)
            .create();

        let err = ensure_collection("org", "Bad", None, &[]).expect_err("missing slug errors");
        list_mock.assert();
        create_mock.assert();
        assert!(err.to_string().contains("slug"));
    }

    #[test]
    #[serial]
    fn find_collection_slug_handles_404_as_no_match() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(404)
            .create();
        let create_mock = server
            .mock("POST", "/api/collections")
            .with_status(200)
            .with_body(r#"{"slug": "org/created-abc"}"#)
            .create();
        let add_mock = server
            .mock("POST", "/api/collections/org/created-abc/items")
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        ensure_collection("org", "Title", None, &[item]).unwrap();
        list_mock.assert();
        create_mock.assert();
        add_mock.assert();
    }

    #[test]
    #[serial]
    fn find_collection_slug_other_list_error_propagates() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(500)
            .with_body("server error")
            .create();

        let err = ensure_collection("org", "Title", None, &[]).expect_err("500 errors");
        list_mock.assert();
        assert!(err.to_string().contains("500"));
    }

    #[test]
    #[serial]
    fn create_collection_includes_description_when_provided() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let list_mock = server
            .mock("GET", "/api/users/org/collections?limit=100")
            .with_status(200)
            .with_body("[]")
            .create();
        let create_mock = server
            .mock("POST", "/api/collections")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "description": "my collection"
            })))
            .with_status(200)
            .with_body(r#"{"slug": "org/with-desc"}"#)
            .create();
        let add_mock = server
            .mock("POST", "/api/collections/org/with-desc/items")
            .with_status(200)
            .create();

        ensure_collection("org", "Titled", Some("my collection"), &[]).unwrap();
        list_mock.assert();
        create_mock.assert();
        let _ = add_mock;
    }
}
