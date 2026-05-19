//! HF collection list / create / item-add / fetch operations. The
//! orchestrator (`ensure_collection`) lives in `mod.rs` and composes
//! these primitives.

use crate::error::VindexError;

use super::super::publish::get_hf_token;
use super::super::publish::protocol::hf_base;
use super::CollectionItem;

pub(super) fn find_collection_slug(
    namespace: &str,
    title: &str,
    token: &str,
) -> Result<Option<String>, VindexError> {
    let client = reqwest::blocking::Client::new();
    let url = format!("{}/api/users/{namespace}/collections?limit=100", hf_base());
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collections list failed: {e}")))?;
    if !resp.status().is_success() {
        if resp.status().as_u16() == 404 {
            return Ok(None);
        }
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "HF collections list ({status}): {body}"
        )));
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF collections JSON: {e}")))?;
    let arr = match body.as_array() {
        Some(a) => a,
        None => return Ok(None),
    };
    let target = title.to_ascii_lowercase();
    for entry in arr {
        let entry_title = entry.get("title").and_then(|v| v.as_str()).unwrap_or("");
        if entry_title.to_ascii_lowercase() == target {
            if let Some(slug) = entry.get("slug").and_then(|v| v.as_str()) {
                return Ok(Some(slug.to_string()));
            }
        }
    }
    Ok(None)
}

pub(super) fn create_collection(
    namespace: &str,
    title: &str,
    description: Option<&str>,
    token: &str,
) -> Result<String, VindexError> {
    let client = reqwest::blocking::Client::new();
    let mut body = serde_json::json!({
        "title": title,
        "namespace": namespace,
        "private": false,
    });
    if let Some(desc) = description {
        body["description"] = serde_json::Value::String(desc.to_string());
    }
    let url = format!("{}/api/collections", hf_base());
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection create failed: {e}")))?;

    let status = resp.status();
    let body_text = resp.text().unwrap_or_default();

    if status.is_success() {
        let json: serde_json::Value = serde_json::from_str(&body_text)
            .map_err(|e| VindexError::Parse(format!("HF collection JSON: {e}")))?;
        let slug = json
            .get("slug")
            .and_then(|v| v.as_str())
            .ok_or_else(|| VindexError::Parse("HF collection response missing slug".into()))?;
        return Ok(slug.to_string());
    }

    // 409 Conflict — collection already exists. HF returns the existing
    // slug in the error body. We hit this when `find_collection_slug`
    // failed to find it (e.g. auth scope / list pagination issues) but
    // the collection does exist. Short-circuiting here is the robust
    // path regardless of why find missed it.
    if status.as_u16() == 409 {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body_text) {
            if let Some(slug) = json.get("slug").and_then(|v| v.as_str()) {
                return Ok(slug.to_string());
            }
        }
    }

    Err(VindexError::Parse(format!(
        "HF collection create ({status}): {body_text}"
    )))
}

pub fn add_collection_item(
    slug: &str,
    item: &CollectionItem,
    token: &str,
) -> Result<(), VindexError> {
    let client = reqwest::blocking::Client::new();
    // HF's collection API uses `/items` (plural) for POST-to-append.
    // The singular form is only valid as `PATCH/DELETE
    // /api/collections/{slug}/item/{item_id}` for editing an existing
    // entry. Got caught by this on the first real publish — the add
    // failed with 404 after the four repos had already uploaded fine.
    let url = format!("{}/api/collections/{slug}/items", hf_base());
    let mut body = serde_json::json!({
        "item": {
            "type": item.repo_type,
            "id": item.repo_id,
        },
    });
    if let Some(note) = &item.note {
        body["note"] = serde_json::Value::String(note.clone());
    }
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection add-item failed: {e}")))?;
    if resp.status().is_success() || resp.status().as_u16() == 409 {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "HF collection add-item ({status}): {body}"
        )))
    }
}

/// Fetch a collection by slug (or full collection URL) and return its
/// items as `(type, id)` pairs — typically `("dataset", "owner/name")`.
pub fn fetch_collection_items(slug_or_url: &str) -> Result<Vec<(String, String)>, VindexError> {
    let slug = slug_or_url
        .trim_start_matches("https://huggingface.co/collections/")
        .trim_start_matches("http://huggingface.co/collections/")
        .trim_start_matches("hf://collections/")
        .trim_start_matches('/');
    let token = get_hf_token().ok();
    let url = format!("{}/api/collections/{slug}", hf_base());
    let client = reqwest::blocking::Client::new();
    let mut req = client.get(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection fetch failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "HF collection fetch ({status}): {body}"
        )));
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF collection JSON: {e}")))?;
    let items = body
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("collection response missing items".into()))?;
    let mut out = Vec::new();
    for item in items {
        let kind = match item.get("type").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let id = match item.get("id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        out.push((kind, id));
    }
    Ok(out)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::test_support::TestEnvGuard;
    use super::*;
    use serial_test::serial;

    // ── fetch_collection_items ──

    #[test]
    #[serial]
    fn fetch_collection_items_parses_type_id_pairs() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("GET", "/api/collections/org/title-abc")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "items": [
                        {"type": "dataset", "id": "owner/repo-a"},
                        {"type": "model", "id": "owner/repo-b"},
                    ]
                })
                .to_string(),
            )
            .create();

        let items = fetch_collection_items("org/title-abc").unwrap();
        mock.assert();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], ("dataset".into(), "owner/repo-a".into()));
        assert_eq!(items[1], ("model".into(), "owner/repo-b".into()));
    }

    #[test]
    #[serial]
    fn fetch_collection_items_skips_malformed_entries() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("GET", "/api/collections/org/title-abc")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "items": [
                        {"type": "dataset"},
                        {"id": "owner/no-type"},
                        {"type": "model", "id": "owner/good"},
                    ]
                })
                .to_string(),
            )
            .create();

        let items = fetch_collection_items("org/title-abc").unwrap();
        mock.assert();
        assert_eq!(items, vec![("model".into(), "owner/good".into())]);
    }

    #[test]
    #[serial]
    fn fetch_collection_items_strips_url_prefix() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("GET", "/api/collections/org/title-abc")
            .with_status(200)
            .with_body(r#"{"items":[]}"#)
            .create();

        let items =
            fetch_collection_items("https://huggingface.co/collections/org/title-abc").unwrap();
        mock.assert();
        assert!(items.is_empty());
    }

    #[test]
    #[serial]
    fn fetch_collection_items_strips_hf_uri_scheme() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("GET", "/api/collections/org/title-abc")
            .with_status(200)
            .with_body(r#"{"items":[]}"#)
            .create();

        let items = fetch_collection_items("hf://collections/org/title-abc").unwrap();
        mock.assert();
        assert!(items.is_empty());
    }

    #[test]
    #[serial]
    fn fetch_collection_items_http_error_propagates() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("GET", "/api/collections/org/title-abc")
            .with_status(404)
            .with_body("not found")
            .create();

        let err = fetch_collection_items("org/title-abc").expect_err("404 errors");
        mock.assert();
        assert!(err.to_string().contains("404"));
    }

    // ── add_collection_item ──

    #[test]
    #[serial]
    fn add_collection_item_sends_typed_repo_payload() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("POST", "/api/collections/org/title-abc/items")
            .match_header("authorization", "Bearer t")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "item": {"type": "dataset", "id": "org/data-repo"}
            })))
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data-repo".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        add_collection_item("org/title-abc", &item, "t").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn add_collection_item_with_note_sends_note_field() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("POST", "/api/collections/org/title-abc/items")
            .match_body(mockito::Matcher::Regex(
                r#""note":\s*"why this repo""#.into(),
            ))
            .with_status(200)
            .create();

        let item = CollectionItem {
            repo_id: "org/data-repo".into(),
            repo_type: "dataset".into(),
            note: Some("why this repo".into()),
        };
        add_collection_item("org/title-abc", &item, "t").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn add_collection_item_409_is_ok() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("POST", "/api/collections/org/title-abc/items")
            .with_status(409)
            .with_body("already in collection")
            .create();

        let item = CollectionItem {
            repo_id: "org/data-repo".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        add_collection_item("org/title-abc", &item, "t").unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn add_collection_item_other_error_propagates() {
        let mut server = mockito::Server::new();
        let _g = TestEnvGuard::new(&server.url());
        let mock = server
            .mock("POST", "/api/collections/org/title-abc/items")
            .with_status(403)
            .with_body("denied")
            .create();

        let item = CollectionItem {
            repo_id: "org/data-repo".into(),
            repo_type: "dataset".into(),
            note: None,
        };
        let err = add_collection_item("org/title-abc", &item, "t").expect_err("403 errors");
        mock.assert();
        assert!(err.to_string().contains("403"));
    }
}
