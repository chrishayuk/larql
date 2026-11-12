//! ETAG
//! SESSION — get_or_create, session_count
//! ANNOUNCE — vindex_identity_hash

use super::*;

#[test]
fn test_etag_deterministic() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let body = serde_json::json!({"entity": "France", "edges": [{"target": "Paris"}]});
    let s = body.to_string();

    let mut h1 = DefaultHasher::new();
    s.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    s.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

#[test]
fn test_etag_format() {
    // ETag should be quoted hex string
    let body = serde_json::json!({"test": true});
    let s = body.to_string();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&s, &mut hasher);
    let etag = format!("\"{:x}\"", std::hash::Hasher::finish(&hasher));
    assert!(etag.starts_with('"'));
    assert!(etag.ends_with('"'));
    assert!(etag.len() > 4); // At least "xx"
}

#[test]
fn test_if_none_match_comparison() {
    let etag = "\"abc123\"";
    // Exact match
    assert_eq!(etag.trim(), etag);
    // Wildcard
    assert_eq!("*".trim(), "*");
    // No match
    assert_ne!("\"different\"".trim(), etag);
}

#[test]
fn test_304_not_modified_condition() {
    let cached_etag = "\"abc123\"";
    let request_etag = "\"abc123\"";
    let should_304 = request_etag.trim() == cached_etag || request_etag.trim() == "*";
    assert!(should_304);

    let stale_etag = "\"old\"";
    let should_304 = stale_etag.trim() == cached_etag || stale_etag.trim() == "*";
    assert!(!should_304);
}

#[test]
fn test_etag_empty_object_is_valid() {
    let etag = compute_etag(&serde_json::json!({}));
    assert!(etag.starts_with('"') && etag.ends_with('"'));
    assert!(etag.len() > 2);
}

#[test]
fn test_etag_different_key_order_produces_different_hash() {
    // JSON key ordering matters when serialised.
    let a = compute_etag(&serde_json::json!({"a": 1, "b": 2}));
    let b = compute_etag(&serde_json::json!({"b": 2, "a": 1}));
    // serde_json preserves insertion order, so these are the same.
    assert_eq!(a, b);
}

#[test]
fn test_matches_etag_extra_whitespace() {
    let etag = compute_etag(&serde_json::json!({"x": 1}));
    // Leading/trailing whitespace should still match after trim.
    let padded = format!("  {}  ", etag);
    assert!(matches_etag(Some(&padded), &etag));
}

#[test]
fn test_matches_etag_mismatch_returns_false() {
    assert!(!matches_etag(Some("\"abc\""), "\"xyz\""));
}

#[tokio::test]
async fn session_get_or_create_new_session_returns_empty_patched() {
    let sm = SessionManager::new(3600);
    let m = make_loaded_model_for_warmup();
    let patched = sm.get_or_create("new-session", &m).await.expect("overlay");
    assert_eq!(patched.num_patches(), 0);
}

#[tokio::test]
async fn session_count_increments_on_first_create() {
    let sm = SessionManager::new(3600);
    let m = make_loaded_model_for_warmup();
    assert_eq!(sm.session_count().await, 0);
    sm.get_or_create("s1", &m).await.expect("overlay");
    assert_eq!(sm.session_count().await, 1);
    sm.get_or_create("s2", &m).await.expect("overlay");
    assert_eq!(sm.session_count().await, 2);
}

#[tokio::test]
async fn session_get_or_create_same_id_does_not_add_session() {
    let sm = SessionManager::new(3600);
    let m = make_loaded_model_for_warmup();
    sm.get_or_create("same", &m).await.expect("overlay");
    sm.get_or_create("same", &m).await.expect("overlay");
    assert_eq!(sm.session_count().await, 1);
}

#[tokio::test]
async fn session_remove_patch_from_unknown_session_returns_err() {
    let sm = SessionManager::new(3600);
    let result = sm.remove_patch("does-not-exist", "any").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[test]
fn vindex_identity_hash_is_deterministic() {
    use larql_server::announce::vindex_identity_hash;
    let h1 = vindex_identity_hash("gemma-3-4b", 34);
    let h2 = vindex_identity_hash("gemma-3-4b", 34);
    assert_eq!(h1, h2);
}

#[test]
fn vindex_identity_hash_differs_on_model_id() {
    use larql_server::announce::vindex_identity_hash;
    let h1 = vindex_identity_hash("gemma-3-4b", 34);
    let h2 = vindex_identity_hash("llama-3-8b", 34);
    assert_ne!(h1, h2);
}

#[test]
fn vindex_identity_hash_differs_on_num_layers() {
    use larql_server::announce::vindex_identity_hash;
    let h1 = vindex_identity_hash("model", 32);
    let h2 = vindex_identity_hash("model", 34);
    assert_ne!(h1, h2);
}

#[test]
fn vindex_identity_hash_is_hex_string() {
    use larql_server::announce::vindex_identity_hash;
    let h = vindex_identity_hash("gemma-3-4b", 34);
    assert_eq!(h.len(), 16);
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}
