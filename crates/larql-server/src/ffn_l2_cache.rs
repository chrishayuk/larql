//! L2 server-side FFN output cache for WalkFfn.
//!
//! Shared across all clients for the lifetime of the server process.
//! Key: hash of sorted gate-KNN feature IDs per layer (same scheme as L1).
//! Value: FFN output vector (hidden_size floats) wrapped in Arc to avoid clones
//!        when multiple concurrent requests read the same entry.
//! Eviction: simple capacity cap per layer — entries are dropped when the
//!           per-layer map is full (FIFO drop via HashMap entry churn).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

pub const L2_DEFAULT_MAX_ENTRIES: usize = 4096;

pub struct FfnL2Cache {
    layers: Vec<RwLock<HashMap<u64, Arc<Vec<f32>>>>>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl FfnL2Cache {
    pub fn new(num_layers: usize) -> Self {
        Self::with_max_entries(num_layers, L2_DEFAULT_MAX_ENTRIES)
    }

    pub fn with_max_entries(num_layers: usize, max_entries: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Stable u64 key from sorted feature IDs — matches L1 key scheme.
    pub fn key(feature_ids: &[usize]) -> u64 {
        let mut ids = feature_ids.to_vec();
        ids.sort_unstable();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        ids.hash(&mut hasher);
        hasher.finish()
    }

    pub fn get(&self, layer: usize, key: u64) -> Option<Arc<Vec<f32>>> {
        let map = self.layers.get(layer)?.read().ok()?;
        match map.get(&key) {
            Some(v) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(v.clone())
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    pub fn insert(&self, layer: usize, key: u64, value: Vec<f32>) {
        if let Some(lock) = self.layers.get(layer) {
            if let Ok(mut map) = lock.write() {
                if map.len() < self.max_entries {
                    map.insert(key, Arc::new(value));
                }
            }
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn hit_rate(&self) -> f64 {
        let h = self.hits();
        let m = self.misses();
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        }
    }

    /// Snapshot for /v1/stats or logging.
    #[allow(dead_code)]
    pub fn stats(&self) -> serde_json::Value {
        let h = self.hits();
        let m = self.misses();
        let total = h + m;
        let hit_rate = if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        };
        serde_json::json!({
            "hits": h,
            "misses": m,
            "total": total,
            "hit_rate": (hit_rate * 1000.0).round() / 1000.0,
            "layers": self.layers.len(),
            "max_entries_per_layer": self.max_entries,
        })
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
    fn key_matches_l1_scheme() {
        // L1 and L2 use identical key derivation — cross-tier consistency.
        fn l1_key(ids: &[usize]) -> u64 {
            use std::hash::{Hash, Hasher};
            let mut sorted = ids.to_vec();
            sorted.sort_unstable();
            let mut h = std::collections::hash_map::DefaultHasher::new();
            sorted.hash(&mut h);
            h.finish()
        }
        let ids = vec![5usize, 2, 9, 1];
        assert_eq!(FfnL2Cache::key(&ids), l1_key(&ids));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn key_is_order_independent() {
        let k1 = FfnL2Cache::key(&[3, 1, 4, 1, 5]);
        let k2 = FfnL2Cache::key(&[5, 4, 3, 1, 1]);
        assert_eq!(k1, k2);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn miss_then_hit() {
        let cache = FfnL2Cache::new(4);
        let key = FfnL2Cache::key(&[10, 20]);
        assert!(cache.get(0, key).is_none());
        assert_eq!(cache.misses(), 1);

        cache.insert(0, key, vec![1.0, 2.0, 3.0]);
        let result = cache.get(0, key);
        assert!(result.is_some());
        assert_eq!(*result.unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.hits(), 1);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn hit_rate_computation() {
        let cache = FfnL2Cache::new(2);
        let k = FfnL2Cache::key(&[1]);
        cache.insert(0, k, vec![0.0]); // insert does not affect counters
        cache.get(0, k); // hit  → hits=1
        cache.get(0, k); // hit  → hits=2
        let miss_k = FfnL2Cache::key(&[999]);
        cache.get(0, miss_k); // miss → misses=1

        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
        // 2 hits / 3 total = 0.666...
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn capacity_cap() {
        let cache = FfnL2Cache::with_max_entries(1, 2);
        let k0 = FfnL2Cache::key(&[0]);
        let k1 = FfnL2Cache::key(&[1]);
        let k2 = FfnL2Cache::key(&[2]);

        cache.insert(0, k0, vec![0.0]);
        cache.insert(0, k1, vec![1.0]);
        // Full — k2 dropped silently
        cache.insert(0, k2, vec![2.0]);

        assert!(cache.get(0, k0).is_some());
        assert!(cache.get(0, k1).is_some());
        assert!(cache.get(0, k2).is_none());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn layers_are_independent() {
        let cache = FfnL2Cache::new(4);
        let key = FfnL2Cache::key(&[7]);
        cache.insert(0, key, vec![0.0]);
        cache.insert(2, key, vec![2.0]);

        assert_eq!(*cache.get(0, key).unwrap(), vec![0.0]);
        assert_eq!(*cache.get(2, key).unwrap(), vec![2.0]);
        assert!(cache.get(1, key).is_none());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn out_of_range_layer_is_safe() {
        let cache = FfnL2Cache::new(2);
        let key = FfnL2Cache::key(&[1]);
        assert!(cache.get(99, key).is_none());
        cache.insert(99, key, vec![1.0]); // must not panic
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn arc_values_are_shared_not_cloned() {
        let cache = FfnL2Cache::new(2);
        let key = FfnL2Cache::key(&[42]);
        cache.insert(0, key, vec![2.5]);
        let a = cache.get(0, key).unwrap();
        let b = cache.get(0, key).unwrap();
        // Both Arcs point at the same allocation
        assert!(std::sync::Arc::ptr_eq(&a, &b));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn concurrent_reads_do_not_panic() {
        use std::sync::Arc as StdArc;
        let cache = StdArc::new(FfnL2Cache::new(4));
        let key = FfnL2Cache::key(&[1, 2, 3]);
        cache.insert(0, key, vec![1.0, 2.0]);

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let c = StdArc::clone(&cache);
                std::thread::spawn(move || {
                    assert!(c.get(0, key).is_some());
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn stats_json_has_expected_fields() {
        let cache = FfnL2Cache::new(3);
        let stats = cache.stats();
        assert!(stats["hits"].is_number());
        assert!(stats["misses"].is_number());
        assert!(stats["total"].is_number());
        assert!(stats["hit_rate"].is_number());
        assert_eq!(stats["layers"], 3);
        assert_eq!(stats["max_entries_per_layer"], L2_DEFAULT_MAX_ENTRIES);
    }
}
