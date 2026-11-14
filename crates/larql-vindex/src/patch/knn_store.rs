//! Architecture B: per-layer retrieval-override KNN store.
//!
//! Each INSERT stores an L2-normalized residual key alongside target metadata.
//! At inference time, cosine similarity against the stored keys determines
//! whether to override the model's prediction. No FFN slot allocation, no
//! orthogonality constraint, no Hopfield bound — unlimited scale.
//!
//! Port of Python `RetrievalVindex` from ~/chris-source/chris-experiments/compilation/15_v11_model/vindex_build_wordnet_b.py.
//!
//! ## Unified retrieval (2026-07-31, review item 21)
//!
//! This type no longer owns a parallel scoring implementation. Keys
//! are held as rows in the same [`GateOverlay`] structure that serves
//! `PatchedVindex::gate_knn`'s gate overrides — one dot-product kernel,
//! one per-layer snapshot cache, one set of width guards for both
//! retrieval paths. What remains KnnStore-specific is the *policy*:
//! entry metadata (`entity`/`relation`/`target`), L2-normalization on
//! insert, and ranking by raw cosine descending (`gate_knn` ranks by
//! `|score|` merged with base hits — different statistic, same kernel).

use std::collections::HashMap;

use ndarray::Array1;
use serde::{Deserialize, Serialize};

use super::gate_overlay::GateOverlay;

/// A single entry in the retrieval-override KNN store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnnEntry {
    /// L2-normalized residual key (hidden_size floats).
    pub key: Vec<f32>,
    /// Target token ID to emit when this entry matches.
    pub target_id: u32,
    /// Decoded target token string.
    pub target_token: String,
    /// Source entity name (for DESCRIBE lookups).
    pub entity: String,
    /// Relation label.
    pub relation: String,
    /// Confidence score (user-specified or default 1.0).
    pub confidence: f32,
}

/// Per-layer retrieval-override store. Entries are independent —
/// no orthogonality constraint, no FFN-slot budget.
///
/// Storage invariant: `keys` holds one row per entry at
/// `(layer, entry_index)`, index-aligned with `entries[layer]`. Every
/// mutation that renumbers a layer's entries rebuilds that layer's
/// rows.
#[derive(Debug, Default, Clone)]
pub struct KnnStore {
    /// layer -> Vec<KnnEntry>
    entries: HashMap<usize, Vec<KnnEntry>>,
    /// The unified retrieval structure (shared kernel with
    /// `PatchedVindex::gate_knn`): L2-normalized keys as rows keyed
    /// `(layer, entry_index)`.
    keys: GateOverlay,
}

impl KnnStore {
    /// Add an entry. The key is L2-normalized before storage.
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        layer: usize,
        key: Vec<f32>,
        target_id: u32,
        target_token: String,
        entity: String,
        relation: String,
        confidence: f32,
    ) {
        let normalized = l2_normalize(&key);
        let layer_entries = self.entries.entry(layer).or_default();
        self.keys
            .insert(layer, layer_entries.len(), normalized.clone());
        layer_entries.push(KnnEntry {
            key: normalized,
            target_id,
            target_token,
            entity,
            relation,
            confidence,
        });
    }

    /// Remove all entries matching a predicate; layers whose entry
    /// lists shrink get their key rows rebuilt (indices renumber).
    fn remove_matching(&mut self, keep: impl Fn(&KnnEntry) -> bool) {
        for (&layer, entries) in &mut self.entries {
            let before = entries.len();
            entries.retain(&keep);
            if entries.len() != before {
                self.keys.remove_layer(layer);
                for (idx, entry) in entries.iter().enumerate() {
                    self.keys.insert(layer, idx, entry.key.clone());
                }
            }
        }
        self.entries.retain(|_, v| !v.is_empty());
    }

    /// Remove all entries matching an entity name.
    pub fn remove_by_entity(&mut self, entity: &str) {
        let entity_lower = entity.to_lowercase();
        self.remove_matching(|e| e.entity.to_lowercase() != entity_lower);
    }

    /// Remove entries matching entity + relation.
    pub fn remove_by_entity_relation(&mut self, entity: &str, relation: &str) {
        let entity_lower = entity.to_lowercase();
        let relation_lower = relation.to_lowercase();
        self.remove_matching(|e| {
            e.entity.to_lowercase() != entity_lower || e.relation.to_lowercase() != relation_lower
        });
    }

    /// Top-1 KNN query at a layer. Returns (&entry, cosine_score).
    pub fn query_top1(&self, layer: usize, residual: &[f32]) -> Option<(&KnnEntry, f32)> {
        let results = self.query_knn(layer, residual, 1);
        results.into_iter().next()
    }

    /// Top-K KNN query at a layer. Returns Vec<(&entry, cosine_score)> descending.
    ///
    /// Returns borrowed references to stored entries; callers clone only the
    /// fields they need. Cloning an entire `KnnEntry` duplicates the
    /// `hidden_size`-wide `key` vector, which is the hot-path waste this
    /// signature avoids.
    pub fn query_knn(&self, layer: usize, residual: &[f32], k: usize) -> Vec<(&KnnEntry, f32)> {
        let entries = match self.entries.get(&layer) {
            Some(e) if !e.is_empty() => e,
            _ => return Vec::new(),
        };

        // L2-normalize query; keys are stored normalized, so the
        // shared kernel's dot product IS the cosine.
        let q_arr = Array1::from_vec(l2_normalize(residual));
        let mut indexed = self.keys.score_layer(layer, &q_arr);

        // Rank by raw cosine descending (NOT `|score|` — an
        // anti-aligned key must not fire the override), NaN-tolerant,
        // ties broken by insertion order for determinism.
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        indexed.truncate(k.min(entries.len()));

        indexed
            .into_iter()
            .map(|(idx, score)| (&entries[idx], score))
            .collect()
    }

    /// All entries for a given entity (for DESCRIBE). Returns (layer, &KnnEntry).
    pub fn entries_for_entity(&self, entity: &str) -> Vec<(usize, &KnnEntry)> {
        let entity_lower = entity.to_lowercase();
        let mut results = Vec::new();
        for (&layer, entries) in &self.entries {
            for entry in entries {
                if entry.entity.to_lowercase() == entity_lower {
                    results.push((layer, entry));
                }
            }
        }
        results.sort_by_key(|(l, _)| *l);
        results
    }

    /// Which layers have entries.
    pub fn layers(&self) -> Vec<usize> {
        let mut layers: Vec<usize> = self.entries.keys().copied().collect();
        layers.sort();
        layers
    }

    /// Total entry count across all layers.
    pub fn len(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Direct access to entries.
    pub fn entries(&self) -> &HashMap<usize, Vec<KnnEntry>> {
        &self.entries
    }

    /// Whether `self` and `other` hold the same entries, layer by layer and in order. The key rows are derived
    /// from the entries, so they are not compared.
    pub fn same_entries(&self, other: &KnnStore) -> bool {
        self.entries == other.entries
    }

    /// Construct from a fully-populated entries map. Used by
    /// `super::knn_store_io::load`. Keys are assumed already
    /// normalized (they were normalized on the original `add`).
    pub(super) fn from_entries(entries: HashMap<usize, Vec<KnnEntry>>) -> Self {
        let mut keys = GateOverlay::default();
        for (&layer, layer_entries) in &entries {
            for (idx, entry) in layer_entries.iter().enumerate() {
                keys.insert(layer, idx, entry.key.clone());
            }
        }
        Self { entries, keys }
    }
}

/// L2-normalize a vector. Returns zero vector if norm is zero.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x / norm).collect()
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 + seed).sin()).collect()
    }

    /// Regression (unification 2026-07-31): removal renumbers a
    /// layer's entries, so the overlay key rows must be rebuilt
    /// index-aligned — a stale row set would return the WRONG entry
    /// for a matching key, not just a wrong count. Pin query
    /// correctness after removal, not just `len()`.
    #[test]
    fn query_returns_correct_entry_after_removal_renumbers() {
        let mut store = KnnStore::default();
        let keys: Vec<Vec<f32>> = (0..3).map(|i| make_key(32, i as f32)).collect();
        for (i, key) in keys.iter().enumerate() {
            store.add(
                26,
                key.clone(),
                40 + i as u32,
                format!("tok{i}"),
                format!("ent{i}"),
                "rel".into(),
                1.0,
            );
        }
        // Warm the shared kernel's snapshot, then remove the FIRST
        // entry so every survivor's index shifts down by one.
        let _ = store.query_top1(26, &keys[2]);
        store.remove_by_entity("ent0");
        assert_eq!(store.len(), 2);
        for i in [1usize, 2] {
            let (entry, score) = store.query_top1(26, &keys[i]).expect("survivor must match");
            assert_eq!(entry.entity, format!("ent{i}"), "index misalignment");
            assert!(score > 0.99, "exact key must score ~1.0, got {score}");
        }
    }

    /// Clone must carry the key rows over (the snapshot cache resets
    /// lazily) — a clone that lost the retrieval rows would return
    /// empty results while `len()` still reports entries.
    #[test]
    fn clone_preserves_query_results() {
        let mut store = KnnStore::default();
        let key = make_key(16, 1.0);
        store.add(
            26,
            key.clone(),
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            1.0,
        );
        let _ = store.query_top1(26, &key); // warm the original's cache
        let cloned = store.clone();
        let (entry, score) = cloned.query_top1(26, &key).expect("clone must retrieve");
        assert_eq!(entry.target_token, "Paris");
        assert!(score > 0.99);
    }

    #[test]
    fn test_add_and_len() {
        let mut store = KnnStore::default();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.add(
            26,
            make_key(8, 1.0),
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            1.0,
        );
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        store.add(
            26,
            make_key(8, 2.0),
            43,
            "Berlin".into(),
            "Germany".into(),
            "capital".into(),
            1.0,
        );
        assert_eq!(store.len(), 2);

        store.add(
            10,
            make_key(8, 3.0),
            44,
            "French".into(),
            "France".into(),
            "language".into(),
            1.0,
        );
        assert_eq!(store.len(), 3);
        assert_eq!(store.layers(), vec![10, 26]);
    }

    #[test]
    fn test_query_top1_exact_match() {
        let mut store = KnnStore::default();
        let key = make_key(64, 1.0);
        store.add(
            26,
            key.clone(),
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            1.0,
        );

        // Query with same key should return cosine ~1.0
        let result = store.query_top1(26, &key);
        assert!(result.is_some());
        let (entry, score) = result.unwrap();
        assert_eq!(entry.target_id, 42);
        assert_eq!(entry.target_token, "Paris");
        assert!(score > 0.99, "expected ~1.0, got {score}");
    }

    #[test]
    fn test_query_top1_no_match() {
        let store = KnnStore::default();
        let result = store.query_top1(26, &make_key(64, 1.0));
        assert!(result.is_none());
    }

    #[test]
    fn test_query_knn_ordering() {
        let mut store = KnnStore::default();
        let key1 = make_key(64, 1.0);
        let key2 = make_key(64, 2.0);
        let key3 = make_key(64, 3.0);
        store.add(
            26,
            key1.clone(),
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            1.0,
        );
        store.add(
            26,
            key2.clone(),
            43,
            "Berlin".into(),
            "Germany".into(),
            "capital".into(),
            1.0,
        );
        store.add(
            26,
            key3.clone(),
            44,
            "Rome".into(),
            "Italy".into(),
            "capital".into(),
            1.0,
        );

        // Query with key1 — should return Paris first (exact match)
        let results = store.query_knn(26, &key1, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0.target_token, "Paris");
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_remove_by_entity() {
        let mut store = KnnStore::default();
        store.add(
            26,
            make_key(8, 1.0),
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            1.0,
        );
        store.add(
            10,
            make_key(8, 2.0),
            43,
            "French".into(),
            "France".into(),
            "language".into(),
            1.0,
        );
        store.add(
            26,
            make_key(8, 3.0),
            44,
            "Berlin".into(),
            "Germany".into(),
            "capital".into(),
            1.0,
        );
        assert_eq!(store.len(), 3);

        store.remove_by_entity("France");
        assert_eq!(store.len(), 1);
        assert_eq!(store.entries_for_entity("France").len(), 0);
        assert_eq!(store.entries_for_entity("Germany").len(), 1);
    }

    #[test]
    fn test_remove_by_entity_case_insensitive() {
        let mut store = KnnStore::default();
        store.add(
            26,
            make_key(8, 1.0),
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            1.0,
        );
        store.remove_by_entity("france");
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_entries_for_entity() {
        let mut store = KnnStore::default();
        store.add(
            10,
            make_key(8, 1.0),
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            1.0,
        );
        store.add(
            26,
            make_key(8, 2.0),
            43,
            "French".into(),
            "France".into(),
            "language".into(),
            1.0,
        );
        store.add(
            26,
            make_key(8, 3.0),
            44,
            "Berlin".into(),
            "Germany".into(),
            "capital".into(),
            1.0,
        );

        let france = store.entries_for_entity("France");
        assert_eq!(france.len(), 2);
        assert_eq!(france[0].0, 10); // sorted by layer
        assert_eq!(france[1].0, 26);
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero() {
        let v = vec![0.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut store = KnnStore::default();
        store.add(
            26,
            make_key(16, 1.0),
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            0.95,
        );
        store.add(
            26,
            make_key(16, 2.0),
            43,
            "Berlin".into(),
            "Germany".into(),
            "capital".into(),
            0.87,
        );
        store.add(
            10,
            make_key(16, 3.0),
            44,
            "French".into(),
            "France".into(),
            "language".into(),
            1.0,
        );

        let dir = std::env::temp_dir().join("larql_knn_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_knn_store.bin");

        store.save(&path).expect("save failed");
        let loaded = KnnStore::load(&path).expect("load failed");

        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.layers(), vec![10, 26]);

        // Verify entity lookup works after load
        let france = loaded.entries_for_entity("France");
        assert_eq!(france.len(), 2);

        // Verify KNN still works (f16 round-trip loses some precision)
        let key = make_key(16, 1.0);
        let result = loaded.query_top1(26, &key);
        assert!(result.is_some());
        let (entry, score) = result.unwrap();
        assert_eq!(entry.target_token, "Paris");
        assert!(
            score > 0.95,
            "expected high cosine after f16 round-trip, got {score}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_query_different_layer_empty() {
        let mut store = KnnStore::default();
        store.add(
            26,
            make_key(8, 1.0),
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            1.0,
        );

        // Query at layer 10 which has no entries
        let result = store.query_top1(10, &make_key(8, 1.0));
        assert!(result.is_none());
    }

    #[test]
    fn test_orthogonal_keys_low_score() {
        let mut store = KnnStore::default();
        // Two orthogonal keys
        let mut key1 = vec![0.0; 64];
        key1[0] = 1.0;
        let mut key2 = vec![0.0; 64];
        key2[1] = 1.0;

        store.add(
            26,
            key1.clone(),
            42,
            "Paris".into(),
            "France".into(),
            "capital".into(),
            1.0,
        );
        store.add(
            26,
            key2.clone(),
            43,
            "Berlin".into(),
            "Germany".into(),
            "capital".into(),
            1.0,
        );

        // Query with key1 — should return Paris with score=1.0, Berlin with score=0.0
        let results = store.query_knn(26, &key1, 2);
        assert_eq!(results[0].0.target_token, "Paris");
        assert!(results[0].1 > 0.99);
        assert_eq!(results[1].0.target_token, "Berlin");
        assert!(results[1].1.abs() < 0.01);
    }
}
