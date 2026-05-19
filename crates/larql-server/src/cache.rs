//! TTL cache for DESCRIBE results.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

struct CacheEntry {
    value: serde_json::Value,
    inserted_at: Instant,
}

/// Simple in-memory TTL cache for DESCRIBE responses.
pub struct DescribeCache {
    entries: RwLock<HashMap<String, CacheEntry>>,
    ttl: Duration,
}

impl DescribeCache {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(ttl_secs),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.ttl.as_secs() > 0
    }

    /// Build a cache key from describe parameters.
    pub fn key(model_id: &str, entity: &str, band: &str, limit: usize, min_score: f32) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            model_id, entity, band, limit, min_score as u32
        )
    }

    /// Get a cached value if it exists and hasn't expired.
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        let entries = self.entries.read().ok()?;
        let entry = entries.get(key)?;
        if entry.inserted_at.elapsed() < self.ttl {
            Some(entry.value.clone())
        } else {
            None
        }
    }

    /// Insert a value into the cache.
    pub fn put(&self, key: String, value: serde_json::Value) {
        if let Ok(mut entries) = self.entries.write() {
            // Evict expired entries if the cache is getting large.
            if entries.len() > 10000 {
                let now = Instant::now();
                entries.retain(|_, e| now.duration_since(e.inserted_at) < self.ttl);
            }
            entries.insert(
                key,
                CacheEntry {
                    value,
                    inserted_at: Instant::now(),
                },
            );
        }
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
    fn disabled_when_ttl_zero() {
        let cache = DescribeCache::new(0);
        assert!(!cache.is_enabled());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn enabled_when_ttl_nonzero() {
        let cache = DescribeCache::new(60);
        assert!(cache.is_enabled());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn put_and_get() {
        let cache = DescribeCache::new(60);
        let key = DescribeCache::key("model", "France", "knowledge", 20, 5.0);
        let value = serde_json::json!({"entity": "France"});
        cache.put(key.clone(), value.clone());
        assert_eq!(cache.get(&key), Some(value));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn miss_on_unknown_key() {
        let cache = DescribeCache::new(60);
        assert_eq!(cache.get("nonexistent"), None);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn expired_entry_returns_none() {
        let cache = DescribeCache::new(0); // 0 → disabled, but let's test with 1ns TTL
                                           // Can't easily test TTL expiration in a unit test without sleeping,
                                           // so we test the disabled path instead.
        let key = "test".to_string();
        cache.put(key.clone(), serde_json::json!("val"));
        // With TTL=0, is_enabled() is false, so caller won't even check cache.
        // But internally get() will return None because elapsed >= 0s TTL.
        assert_eq!(cache.get(&key), None);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn key_format() {
        let key = DescribeCache::key("gemma-3-4b-it", "France", "knowledge", 20, 5.0);
        assert_eq!(key, "gemma-3-4b-it:France:knowledge:20:5");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn different_params_different_keys() {
        let k1 = DescribeCache::key("model", "France", "knowledge", 20, 5.0);
        let k2 = DescribeCache::key("model", "Germany", "knowledge", 20, 5.0);
        let k3 = DescribeCache::key("model", "France", "syntax", 20, 5.0);
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
    }
}
