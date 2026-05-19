//! L1 in-process FFN output cache for WalkFfn.
//!
//! Key: hash of sorted gate-KNN feature IDs per layer.
//! Value: FFN output vector (hidden_size floats).
//! Scope: single WalkFfn instance — one inference session or one HTTP request.
//! Eviction: bounded by max_entries per layer (FIFO, no LRU).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub const L1_DEFAULT_MAX_ENTRIES: usize = 4096;

pub struct FfnL1Cache {
    layers: Vec<RefCell<HashMap<u64, Vec<f32>>>>,
    max_entries: usize,
    hits: Cell<u64>,
    misses: Cell<u64>,
}

impl FfnL1Cache {
    pub fn new(num_layers: usize) -> Self {
        Self::with_max_entries(num_layers, L1_DEFAULT_MAX_ENTRIES)
    }

    pub fn with_max_entries(num_layers: usize, max_entries: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| RefCell::new(HashMap::new()))
                .collect(),
            max_entries,
            hits: Cell::new(0),
            misses: Cell::new(0),
        }
    }

    /// Stable u64 cache key from feature IDs — sorted before hashing so
    /// gate-score order doesn't affect the key.
    ///
    /// Used by `walk_ffn_sparse` (bounded top-k path).
    pub fn key(feature_ids: &[usize]) -> u64 {
        let mut ids = feature_ids.to_vec();
        ids.sort_unstable();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        ids.hash(&mut hasher);
        hasher.finish()
    }

    /// Cache key from a raw residual vector, for dense paths where no sparse
    /// feature set is available (interleaved / full-mmap walks).
    ///
    /// Quantises each float to i16 (scale ×256) before hashing so that
    /// paraphrase-collapsed residuals at cos≥0.999 — which differ by less
    /// than 1 ulp at i16 precision — map to the same key.  The quantisation
    /// step is fast (~1µs for hidden=2560) and makes the key robust to the
    /// floating-point noise that would otherwise prevent cache hits across
    /// identical tokens at different context lengths.
    pub fn residual_key(residual: &[f32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &v in residual {
            let q = (v * 256.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            q.hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn get(&self, layer: usize, key: u64) -> Option<Vec<f32>> {
        let map = self.layers.get(layer)?.borrow();
        if let Some(v) = map.get(&key) {
            self.hits.set(self.hits.get() + 1);
            Some(v.clone())
        } else {
            self.misses.set(self.misses.get() + 1);
            None
        }
    }

    pub fn insert(&self, layer: usize, key: u64, value: Vec<f32>) {
        if let Some(cell) = self.layers.get(layer) {
            let mut map = cell.borrow_mut();
            if map.len() < self.max_entries {
                map.insert(key, value);
            }
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits.get()
    }
    pub fn misses(&self) -> u64 {
        self.misses.get()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits.get() + self.misses.get();
        if total == 0 {
            0.0
        } else {
            self.hits.get() as f64 / total as f64
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn key_is_order_independent() {
        // Same features in different order → same key (gate-score order doesn't matter)
        let k1 = FfnL1Cache::key(&[5, 2, 8, 1]);
        let k2 = FfnL1Cache::key(&[1, 2, 5, 8]);
        let k3 = FfnL1Cache::key(&[8, 1, 2, 5]);
        assert_eq!(k1, k2);
        assert_eq!(k2, k3);
    }

    #[test]
    fn key_differs_for_different_feature_sets() {
        let ka = FfnL1Cache::key(&[1, 2, 3]);
        let kb = FfnL1Cache::key(&[1, 2, 4]);
        let kc = FfnL1Cache::key(&[1, 2, 3, 4]);
        assert_ne!(ka, kb);
        assert_ne!(ka, kc);
        assert_ne!(kb, kc);
    }

    #[test]
    fn key_stable_across_calls() {
        let ids = vec![10usize, 3, 7, 42];
        assert_eq!(FfnL1Cache::key(&ids), FfnL1Cache::key(&ids));
    }

    #[test]
    fn empty_feature_set_has_stable_key() {
        assert_eq!(FfnL1Cache::key(&[]), FfnL1Cache::key(&[]));
    }

    #[test]
    fn miss_then_hit() {
        let cache = FfnL1Cache::new(4);
        let key = FfnL1Cache::key(&[1, 2, 3]);
        assert_eq!(cache.get(0, key), None);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);

        cache.insert(0, key, vec![1.0, 2.0, 3.0]);
        let result = cache.get(0, key);
        assert_eq!(result, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn hit_rate_zero_when_empty() {
        let cache = FfnL1Cache::new(4);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn hit_rate_100_percent() {
        let cache = FfnL1Cache::new(2);
        let key = FfnL1Cache::key(&[7]);
        cache.insert(0, key, vec![0.5]);
        cache.get(0, key);
        cache.get(0, key);
        assert_eq!(cache.hit_rate(), 1.0);
    }

    #[test]
    fn hit_rate_50_percent() {
        let cache = FfnL1Cache::new(2);
        let hit_key = FfnL1Cache::key(&[1]);
        let miss_key = FfnL1Cache::key(&[99]);
        cache.insert(0, hit_key, vec![1.0]);
        cache.get(0, hit_key); // hit
        cache.get(0, miss_key); // miss
        assert!((cache.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn capacity_cap_prevents_insert() {
        let cache = FfnL1Cache::with_max_entries(2, 2);
        let k0 = FfnL1Cache::key(&[0]);
        let k1 = FfnL1Cache::key(&[1]);
        let k2 = FfnL1Cache::key(&[2]);

        cache.insert(0, k0, vec![0.0]);
        cache.insert(0, k1, vec![1.0]);
        // At capacity — k2 must be silently dropped
        cache.insert(0, k2, vec![2.0]);

        assert!(cache.get(0, k0).is_some());
        assert!(cache.get(0, k1).is_some());
        assert_eq!(cache.get(0, k2), None);
    }

    #[test]
    fn layers_are_independent() {
        let cache = FfnL1Cache::new(4);
        let key = FfnL1Cache::key(&[5]);
        cache.insert(0, key, vec![10.0]);
        cache.insert(1, key, vec![20.0]);

        assert_eq!(cache.get(0, key), Some(vec![10.0]));
        assert_eq!(cache.get(1, key), Some(vec![20.0]));
        // Layer 2 was never written
        assert_eq!(cache.get(2, key), None);
    }

    #[test]
    fn out_of_range_layer_is_safe() {
        let cache = FfnL1Cache::new(2);
        let key = FfnL1Cache::key(&[1]);
        // Layer 99 is out of range — should return None, not panic
        assert_eq!(cache.get(99, key), None);
        // Insert to out-of-range layer — should be a no-op
        cache.insert(99, key, vec![1.0]);
    }

    // ── residual_key tests ────────────────────────────────────────────────

    #[test]
    fn residual_key_is_deterministic() {
        let r: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        assert_eq!(FfnL1Cache::residual_key(&r), FfnL1Cache::residual_key(&r));
    }

    #[test]
    fn residual_key_differs_for_different_residuals() {
        let r1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let r2: Vec<f32> = vec![1.0, 2.0, 4.0];
        assert_ne!(FfnL1Cache::residual_key(&r1), FfnL1Cache::residual_key(&r2));
    }

    #[test]
    fn residual_key_matches_for_near_identical_residuals() {
        // Residuals that differ by << 1/256 in each dimension → same i16 bucket
        let base: Vec<f32> = (0..32).map(|i| i as f32 * 0.001).collect();
        let noise: Vec<f32> = base.iter().map(|&v| v + 1e-5).collect();
        assert_eq!(
            FfnL1Cache::residual_key(&base),
            FfnL1Cache::residual_key(&noise)
        );
    }

    #[test]
    fn residual_key_empty_vec() {
        assert_eq!(FfnL1Cache::residual_key(&[]), FfnL1Cache::residual_key(&[]));
    }

    #[test]
    fn different_values_same_key_overwrites() {
        let cache = FfnL1Cache::new(2);
        let key = FfnL1Cache::key(&[3, 7]);
        cache.insert(0, key, vec![1.0, 2.0]);
        cache.insert(0, key, vec![9.0, 8.0]); // overwrite
                                              // Should have the second value (HashMap semantics)
        assert_eq!(cache.get(0, key), Some(vec![9.0, 8.0]));
    }
}
