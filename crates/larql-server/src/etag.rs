//! ETag generation for CDN edge caching.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compute a short ETag from a JSON response body.
pub fn compute_etag(body: &serde_json::Value) -> String {
    let mut hasher = DefaultHasher::new();
    body.to_string().hash(&mut hasher);
    format!("\"{:x}\"", hasher.finish())
}

/// Check if the request's If-None-Match header matches the ETag.
pub fn matches_etag(if_none_match: Option<&str>, etag: &str) -> bool {
    match if_none_match {
        Some(val) => val.trim() == etag || val.trim() == "*",
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(test, target_arch = "wasm32"))]
    use wasm_bindgen_test::wasm_bindgen_test;
    #[cfg(all(test, target_arch = "wasm32"))]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn etag_is_quoted() {
        let etag = compute_etag(&serde_json::json!({"hello": "world"}));
        assert!(etag.starts_with('"'));
        assert!(etag.ends_with('"'));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn same_body_same_etag() {
        let body = serde_json::json!({"entity": "France", "edges": []});
        let e1 = compute_etag(&body);
        let e2 = compute_etag(&body);
        assert_eq!(e1, e2);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn different_body_different_etag() {
        let e1 = compute_etag(&serde_json::json!({"entity": "France"}));
        let e2 = compute_etag(&serde_json::json!({"entity": "Germany"}));
        assert_ne!(e1, e2);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn matches_exact() {
        let etag = compute_etag(&serde_json::json!({"x": 1}));
        assert!(matches_etag(Some(&etag), &etag));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn matches_wildcard() {
        let etag = compute_etag(&serde_json::json!({"x": 1}));
        assert!(matches_etag(Some("*"), &etag));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn no_match_on_none() {
        let etag = compute_etag(&serde_json::json!({"x": 1}));
        assert!(!matches_etag(None, &etag));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn no_match_on_different() {
        let etag = compute_etag(&serde_json::json!({"x": 1}));
        assert!(!matches_etag(Some("\"wrong\""), &etag));
    }
}
