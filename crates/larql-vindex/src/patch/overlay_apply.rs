//! Patch application — `apply_patch`, `remove_patch`,
//! `rebuild_overrides` for `PatchedVindex`.
//!
//! Walks `VindexPatch::operations` and resolves each one into the
//! overlay's override maps (or the L0 KNN store for arch-B ops).
//! Pulled out of `overlay.rs` so the file holding `PatchedVindex`'s
//! query/mutation API stays focused.

use crate::index::types::DEFAULT_C_SCORE;
use crate::index::FeatureMeta;

use super::format::{decode_gate_vector, PatchOp, VindexPatch};
use super::overlay::PatchedVindex;

impl PatchedVindex {
    /// Apply a patch. Operations are resolved into the override maps.
    pub fn apply_patch(&mut self, patch: VindexPatch) {
        // Collect the layers whose `overrides_gate` we touch so we
        // can invalidate exactly those cache entries at the end. A
        // single patch usually touches one or a handful of layers;
        // the HashSet bounds invalidation cost to O(touched_layers).
        let mut touched_layers: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for op in &patch.operations {
            match op {
                PatchOp::InsertKnn {
                    layer,
                    entity,
                    relation,
                    target,
                    target_id,
                    confidence,
                    key_vector_b64,
                } => {
                    if let Ok(key_vec) = decode_gate_vector(key_vector_b64) {
                        self.knn_store.add(
                            *layer,
                            key_vec,
                            *target_id,
                            target.clone(),
                            entity.clone(),
                            relation.clone(),
                            confidence.unwrap_or(1.0),
                        );
                    }
                    continue;
                }
                PatchOp::DeleteKnn { entity } => {
                    self.knn_store.remove_by_entity(entity);
                    continue;
                }
                _ => {}
            }
            let key = op.key().unwrap(); // safe: only Arch A ops reach here
            match op {
                PatchOp::Insert {
                    target,
                    confidence,
                    gate_vector_b64,
                    up_vector_b64,
                    down_vector_b64,
                    down_meta,
                    ..
                } => {
                    let meta = if let Some(dm) = down_meta {
                        FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        }
                    } else {
                        FeatureMeta {
                            top_token: target.clone(),
                            top_token_id: 0,
                            c_score: confidence.unwrap_or(DEFAULT_C_SCORE),
                            top_k: vec![],
                        }
                    };
                    self.overrides_meta.insert(key, Some(meta));
                    self.deleted.remove(&key);
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key, vec);
                            touched_layers.insert(key.0);
                        }
                    }
                    if let Some(b64) = up_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.base.set_up_vector(key.0, key.1, vec);
                        }
                    }
                    if let Some(b64) = down_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.base.set_down_vector(key.0, key.1, vec);
                        }
                    }
                }
                PatchOp::Update {
                    gate_vector_b64,
                    up_vector_b64,
                    down_vector_b64,
                    down_meta,
                    ..
                } => {
                    if let Some(dm) = down_meta {
                        let meta = FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        };
                        self.overrides_meta.insert(key, Some(meta));
                    }
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key, vec);
                            touched_layers.insert(key.0);
                        }
                    }
                    if let Some(b64) = up_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.base.set_up_vector(key.0, key.1, vec);
                        }
                    }
                    if let Some(b64) = down_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.base.set_down_vector(key.0, key.1, vec);
                        }
                    }
                }
                PatchOp::Delete { .. } => {
                    self.overrides_meta.insert(key, None);
                    self.deleted.insert(key);
                    // Always invalidate on Delete — even if the gate
                    // entry was absent, the per-layer feature set
                    // shrunk and any cached matrix is stale.
                    let was_present = self.overrides_gate.remove(&key).is_some();
                    if was_present {
                        touched_layers.insert(key.0);
                    }
                }
                PatchOp::InsertKnn { .. } | PatchOp::DeleteKnn { .. } => {
                    unreachable!("KNN ops handled above");
                }
            }
        }
        // Per-layer invalidation: only drop cache entries for layers
        // this patch actually mutated. Patches that touch one layer
        // leave every other layer's cache hot, which matters for
        // high-frequency single-layer patch streams (e.g. Exp 52
        // compile loops, INSERT bursts at a single L26 cache).
        for layer in touched_layers {
            self.invalidate_gate_cache_layer(layer);
        }
        self.patches.push(patch);
    }

    /// Remove the last applied patch and rebuild overrides.
    pub fn remove_patch(&mut self, index: usize) {
        if index < self.patches.len() {
            self.patches.remove(index);
            self.rebuild_overrides();
        }
    }

    /// Rebuild override maps from scratch (after removing a patch).
    fn rebuild_overrides(&mut self) {
        self.overrides_meta.clear();
        self.overrides_gate.clear();
        self.deleted.clear();
        self.knn_store = super::knn_store::KnnStore::default();
        self.invalidate_gate_cache();
        let patches: Vec<VindexPatch> = self.patches.drain(..).collect();
        for patch in patches {
            self.apply_patch(patch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::VectorIndex;
    use crate::patch::format::{encode_gate_vector, PatchDownMeta, PatchOp, VindexPatch};

    fn empty_pv() -> PatchedVindex {
        PatchedVindex::new(VectorIndex::new(vec![], vec![], 0, 0))
    }

    fn make_patch(ops: Vec<PatchOp>) -> VindexPatch {
        VindexPatch {
            version: 1,
            base_model: "test".into(),
            base_checksum: None,
            created_at: "2026-01-01T00:00:00Z".into(),
            description: None,
            author: None,
            tags: vec![],
            operations: ops,
        }
    }

    #[test]
    fn apply_insert_populates_overrides_meta() {
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::Insert {
            layer: 2,
            feature: 5,
            relation: None,
            entity: "France".into(),
            target: "Paris".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(patch);
        assert!(pv.overrides_meta.contains_key(&(2, 5)));
        let meta = pv.overrides_meta[&(2, 5)].as_ref().unwrap();
        assert_eq!(meta.top_token, "Paris");
    }

    #[test]
    fn apply_insert_with_down_meta_uses_down_meta_token() {
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::Insert {
            layer: 1,
            feature: 10,
            relation: None,
            entity: "Germany".into(),
            target: "Berlin".into(),
            confidence: Some(0.8),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(PatchDownMeta {
                top_token: "Berlin".into(),
                top_token_id: 42,
                c_score: 0.75,
            }),
        }]);
        pv.apply_patch(patch);
        let meta = pv.overrides_meta[&(1, 10)].as_ref().unwrap();
        assert_eq!(meta.top_token, "Berlin");
        assert_eq!(meta.top_token_id, 42);
        assert!((meta.c_score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn apply_insert_with_gate_vector_populates_overrides_gate() {
        let mut pv = empty_pv();
        let gv = vec![1.0f32, 0.0, -1.0];
        let b64 = encode_gate_vector(&gv);
        let patch = make_patch(vec![PatchOp::Insert {
            layer: 3,
            feature: 7,
            relation: None,
            entity: "Spain".into(),
            target: "Madrid".into(),
            confidence: None,
            gate_vector_b64: Some(b64),
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(patch);
        assert!(pv.overrides_gate.contains_key(&(3, 7)));
        let stored = &pv.overrides_gate[&(3, 7)];
        assert_eq!(stored.len(), 3);
        assert_eq!(stored[0].to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn apply_insert_with_up_and_down_vectors_populates_base_overrides() {
        // Compose-mode INSERT writes gate + up + down overrides; the .vlp
        // must round-trip all three. Without up_vector_b64 /
        // down_vector_b64 in the patch, re-applying the file (e.g. on
        // `larql apply` after a save) would lose up + down and
        // `COMPILE INTO VINDEX` would bake nothing.
        let mut pv = empty_pv();
        let gate = vec![1.0f32, 2.0, 3.0];
        let up = vec![0.1f32, 0.2, 0.3];
        let down = vec![-0.5f32, 0.0, 0.5];
        let patch = make_patch(vec![PatchOp::Insert {
            layer: 4,
            feature: 9,
            relation: Some("capital".into()),
            entity: "France".into(),
            target: "Paris".into(),
            confidence: Some(0.9),
            gate_vector_b64: Some(encode_gate_vector(&gate)),
            up_vector_b64: Some(encode_gate_vector(&up)),
            down_vector_b64: Some(encode_gate_vector(&down)),
            down_meta: None,
        }]);
        pv.apply_patch(patch);
        assert_eq!(pv.overrides_gate_at(4, 9), Some(gate.as_slice()));
        assert_eq!(pv.up_override_at(4, 9), Some(up.as_slice()));
        assert_eq!(pv.down_override_at(4, 9), Some(down.as_slice()));
    }

    #[test]
    fn apply_delete_tombstones_feature() {
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::Delete {
            layer: 0,
            feature: 3,
            reason: None,
        }]);
        pv.apply_patch(patch);
        assert!(pv.deleted.contains(&(0, 3)));
        assert!(pv.overrides_meta[&(0, 3)].is_none());
    }

    #[test]
    fn insert_then_delete_removes_gate_override() {
        let mut pv = empty_pv();
        let gv = vec![1.0f32, 2.0];
        let b64 = encode_gate_vector(&gv);
        let insert_patch = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 1,
            relation: None,
            entity: "A".into(),
            target: "B".into(),
            confidence: None,
            gate_vector_b64: Some(b64),
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(insert_patch);
        assert!(pv.overrides_gate.contains_key(&(0, 1)));

        let delete_patch = make_patch(vec![PatchOp::Delete {
            layer: 0,
            feature: 1,
            reason: None,
        }]);
        pv.apply_patch(delete_patch);
        assert!(!pv.overrides_gate.contains_key(&(0, 1)));
        assert!(pv.deleted.contains(&(0, 1)));
    }

    #[test]
    fn apply_update_sets_meta_only() {
        let mut pv = empty_pv();
        let patch = make_patch(vec![PatchOp::Update {
            layer: 0,
            feature: 2,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(PatchDownMeta {
                top_token: "updated".into(),
                top_token_id: 99,
                c_score: 0.5,
            }),
        }]);
        pv.apply_patch(patch);
        let meta = pv.overrides_meta[&(0, 2)].as_ref().unwrap();
        assert_eq!(meta.top_token, "updated");
        // No gate override set
        assert!(!pv.overrides_gate.contains_key(&(0, 2)));
    }

    #[test]
    fn apply_patches_accumulate_in_order() {
        let mut pv = empty_pv();
        let p1 = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 0,
            relation: None,
            entity: "X".into(),
            target: "Y".into(),
            confidence: Some(0.5),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        let p2 = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 1,
            relation: None,
            entity: "A".into(),
            target: "B".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(p1);
        pv.apply_patch(p2);
        assert_eq!(pv.patches.len(), 2);
        assert!(pv.overrides_meta.contains_key(&(0, 0)));
        assert!(pv.overrides_meta.contains_key(&(0, 1)));
    }

    #[test]
    fn remove_patch_rebuilds_overrides() {
        let mut pv = empty_pv();
        let p1 = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 5,
            relation: None,
            entity: "X".into(),
            target: "first".into(),
            confidence: None,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        let p2 = make_patch(vec![PatchOp::Insert {
            layer: 0,
            feature: 6,
            relation: None,
            entity: "Y".into(),
            target: "second".into(),
            confidence: None,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]);
        pv.apply_patch(p1);
        pv.apply_patch(p2);
        assert_eq!(pv.patches.len(), 2);

        pv.remove_patch(0);
        assert_eq!(pv.patches.len(), 1);
        // Feature 5 (from patch 0) should be gone
        assert!(!pv.overrides_meta.contains_key(&(0, 5)));
        // Feature 6 (from patch 1) should still be present
        assert!(pv.overrides_meta.contains_key(&(0, 6)));
    }

    #[test]
    fn remove_patch_out_of_bounds_is_noop() {
        let mut pv = empty_pv();
        pv.remove_patch(999); // should not panic
        assert!(pv.patches.is_empty());
    }

    #[test]
    fn apply_insert_knn_adds_to_knn_store() {
        let mut pv = empty_pv();
        let kv = encode_gate_vector(&[1.0f32, 0.0, 0.0]);
        let patch = make_patch(vec![PatchOp::InsertKnn {
            layer: 0,
            entity: "France".into(),
            relation: "capital".into(),
            target: "Paris".into(),
            target_id: 1234,
            confidence: Some(1.0),
            key_vector_b64: kv,
        }]);
        pv.apply_patch(patch);
        assert_eq!(pv.knn_store.len(), 1);
    }

    #[test]
    fn apply_delete_knn_removes_from_knn_store() {
        let mut pv = empty_pv();
        let kv = encode_gate_vector(&[1.0f32, 0.0, 0.0]);
        let insert = make_patch(vec![PatchOp::InsertKnn {
            layer: 0,
            entity: "France".into(),
            relation: "capital".into(),
            target: "Paris".into(),
            target_id: 1,
            confidence: None,
            key_vector_b64: kv,
        }]);
        let delete = make_patch(vec![PatchOp::DeleteKnn {
            entity: "France".into(),
        }]);
        pv.apply_patch(insert);
        assert_eq!(pv.knn_store.len(), 1);
        pv.apply_patch(delete);
        assert_eq!(pv.knn_store.len(), 0);
    }
}
