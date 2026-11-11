//! Patch application — `apply_patch`, `remove_patch`,
//! `rebuild_overrides` for `PatchedVindex`.
//!
//! Walks `VindexPatch::operations` and resolves each one into the
//! overlay's override maps (or the L0 KNN store for arch-B ops).
//! Pulled out of `overlay.rs` so the file holding `PatchedVindex`'s
//! query/mutation API stays focused.

use crate::error::VindexError;
use crate::index::types::DEFAULT_C_SCORE;
use crate::index::FeatureMeta;

use super::format::{decode_gate_vector, PatchOp, VindexPatch};
use super::overlay::PatchedVindex;
use super::validate::PatchGeometry;

/// Decode-check every embedded vector in `patch` before any overlay
/// state is mutated, so a corrupt vector rejects the patch wholesale.
/// Without this, `apply_patch` used to drop the vector while still
/// applying the op's metadata — a silently half-applied patch.
fn validate_patch_vectors(patch: &VindexPatch) -> Result<(), VindexError> {
    fn check(op_idx: usize, field: &str, b64: &str) -> Result<(), VindexError> {
        decode_gate_vector(b64)
            .map(|_| ())
            .map_err(|e| VindexError::Parse(format!("patch op {op_idx}: corrupt {field}: {e}")))
    }
    for (i, op) in patch.operations.iter().enumerate() {
        match op {
            PatchOp::Insert {
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            }
            | PatchOp::Update {
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } => {
                for (field, b64) in [
                    ("gate_vector_b64", gate_vector_b64),
                    ("up_vector_b64", up_vector_b64),
                    ("down_vector_b64", down_vector_b64),
                ] {
                    if let Some(b64) = b64 {
                        check(i, field, b64)?;
                    }
                }
            }
            PatchOp::InsertKnn { key_vector_b64, .. } => {
                check(i, "key_vector_b64", key_vector_b64)?;
            }
            PatchOp::Delete { .. } | PatchOp::DeleteKnn { .. } => {}
            PatchOp::WeightDelta(_) => {
                return Err(VindexError::Parse(format!(
                    "patch op {i}: `weight_delta` edits the stored weights of a VINDEX3 container; \
                     a VINDEX2 overlay cannot hold it"
                )));
            }
        }
    }
    Ok(())
}

impl PatchedVindex {
    /// Apply a patch, all of it or none of it.
    ///
    /// The whole patch is checked before any operation is applied: every
    /// embedded vector decodes and is `hidden_size` wide, and every layer
    /// and feature slot it names exists in the base. On `Err` no overlay
    /// state has been touched and the patch is not recorded.
    ///
    /// This is the only way to apply a patch. There is deliberately no
    /// infallible form: one existed, discarded this result, and let callers
    /// report a dropped patch as applied.
    ///
    /// The base's model name is not known here. A caller that knows it
    /// checks [`VindexPatch::check_base_model`] first.
    pub fn try_apply_patch(&mut self, patch: VindexPatch) -> Result<(), VindexError> {
        validate_patch_vectors(&patch)?;
        let base = &self.base;
        patch.check_geometry(&PatchGeometry {
            hidden_size: Some(base.hidden_size),
            num_layers: base.num_layers,
            features_at: &|layer| base.num_features(layer),
        })?;
        self.apply_patch_unchecked(patch);
        Ok(())
    }

    /// Application body. Every `*_b64` field has already been decode-
    /// checked by `validate_patch_vectors`, so the inner `if let Ok`
    /// arms cannot silently skip a vector. Gate-snapshot invalidation
    /// is handled per-layer inside `GateOverlay`'s mutators.
    fn apply_patch_unchecked(&mut self, patch: VindexPatch) {
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
                PatchOp::WeightDelta(_) => {
                    unreachable!("rejected whole by validate_patch_vectors")
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
                            self.overrides_gate.insert(key.0, key.1, vec);
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
                    // Resurrect: Update on a tombstoned slot implies
                    // the feature exists again (mirrors Insert; the
                    // `deleted` contract on `PatchedVindex`, review
                    // 2026-07-30 M6). If the earlier Delete pinned a
                    // `None` meta override and this Update carries no
                    // replacement meta, drop the pin so `feature_meta`
                    // falls through to the base — keeping
                    // `feature_meta()` and `gate_knn()` in agreement.
                    if self.deleted.remove(&key)
                        && matches!(self.overrides_meta.get(&key), Some(None))
                    {
                        self.overrides_meta.remove(&key);
                    }
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key.0, key.1, vec);
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
                    // `GateOverlay::remove` invalidates the layer's
                    // snapshot even when the gate entry was absent —
                    // the per-layer feature set shrunk either way.
                    self.overrides_gate.remove(key.0, key.1);
                }
                PatchOp::InsertKnn { .. } | PatchOp::DeleteKnn { .. } | PatchOp::WeightDelta(_) => {
                    unreachable!("KNN and weight ops handled above");
                }
            }
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
        // Clears rows AND every layer's cached snapshot.
        self.overrides_gate.clear();
        self.deleted.clear();
        self.knn_store = super::knn_store::KnnStore::default();
        let patches: Vec<VindexPatch> = std::mem::take(&mut self.patches);
        // Each was checked against this same base when it was admitted, so
        // re-resolving the stack cannot reject one.
        for patch in patches {
            self.apply_patch_unchecked(patch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::VectorIndex;
    use crate::patch::format::{encode_gate_vector, PatchDownMeta, PatchOp, VindexPatch};

    /// Layers in the test base: past the highest layer any test names.
    const LAYERS: usize = 5;
    /// Feature slots per layer: past the highest slot any test names.
    const FEATURES: usize = 11;

    /// A base with real dimensions. A patch is checked against them, so a
    /// test that applies one needs slots to apply it to and a width for its
    /// vectors.
    fn sized_pv(hidden: usize) -> PatchedVindex {
        let gates = vec![Some(ndarray::Array2::<f32>::zeros((FEATURES, hidden))); LAYERS];
        PatchedVindex::new(VectorIndex::new(gates, vec![None; LAYERS], LAYERS, hidden))
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
        let mut pv = sized_pv(3);
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
        pv.try_apply_patch(patch).expect("patch applies");
        assert!(pv.overrides_meta.contains_key(&(2, 5)));
        let meta = pv.overrides_meta[&(2, 5)].as_ref().unwrap();
        assert_eq!(meta.top_token, "Paris");
    }

    #[test]
    fn apply_insert_with_down_meta_uses_down_meta_token() {
        let mut pv = sized_pv(3);
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
        pv.try_apply_patch(patch).expect("patch applies");
        let meta = pv.overrides_meta[&(1, 10)].as_ref().unwrap();
        assert_eq!(meta.top_token, "Berlin");
        assert_eq!(meta.top_token_id, 42);
        assert!((meta.c_score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn apply_insert_with_gate_vector_populates_overrides_gate() {
        let mut pv = sized_pv(3);
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
        pv.try_apply_patch(patch).expect("patch applies");
        assert!(pv.overrides_gate.contains(3, 7));
        let stored = pv.overrides_gate.get(3, 7).unwrap();
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
        let mut pv = sized_pv(3);
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
        pv.try_apply_patch(patch).expect("patch applies");
        assert_eq!(pv.overrides_gate_at(4, 9), Some(gate.as_slice()));
        assert_eq!(pv.up_override_at(4, 9), Some(up.as_slice()));
        assert_eq!(pv.down_override_at(4, 9), Some(down.as_slice()));
    }

    #[test]
    fn apply_delete_tombstones_feature() {
        let mut pv = sized_pv(3);
        let patch = make_patch(vec![PatchOp::Delete {
            layer: 0,
            feature: 3,
            reason: None,
        }]);
        pv.try_apply_patch(patch).expect("patch applies");
        assert!(pv.deleted.contains(&(0, 3)));
        assert!(pv.overrides_meta[&(0, 3)].is_none());
    }

    #[test]
    fn insert_then_delete_removes_gate_override() {
        let mut pv = sized_pv(2);
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
        pv.try_apply_patch(insert_patch).expect("patch applies");
        assert!(pv.overrides_gate.contains(0, 1));

        let delete_patch = make_patch(vec![PatchOp::Delete {
            layer: 0,
            feature: 1,
            reason: None,
        }]);
        pv.try_apply_patch(delete_patch).expect("patch applies");
        assert!(!pv.overrides_gate.contains(0, 1));
        assert!(pv.deleted.contains(&(0, 1)));
    }

    #[test]
    fn apply_update_sets_meta_only() {
        let mut pv = sized_pv(3);
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
        pv.try_apply_patch(patch).expect("patch applies");
        let meta = pv.overrides_meta[&(0, 2)].as_ref().unwrap();
        assert_eq!(meta.top_token, "updated");
        // No gate override set
        assert!(!pv.overrides_gate.contains(0, 2));
    }

    #[test]
    fn apply_patches_accumulate_in_order() {
        let mut pv = sized_pv(3);
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
        pv.try_apply_patch(p1).expect("patch applies");
        pv.try_apply_patch(p2).expect("patch applies");
        assert_eq!(pv.patches.len(), 2);
        assert!(pv.overrides_meta.contains_key(&(0, 0)));
        assert!(pv.overrides_meta.contains_key(&(0, 1)));
    }

    #[test]
    fn remove_patch_rebuilds_overrides() {
        let mut pv = sized_pv(3);
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
        pv.try_apply_patch(p1).expect("patch applies");
        pv.try_apply_patch(p2).expect("patch applies");
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
        let mut pv = sized_pv(3);
        pv.remove_patch(999); // should not panic
        assert!(pv.patches.is_empty());
    }

    #[test]
    fn try_apply_patch_rejects_corrupt_vector_without_half_applying() {
        // Regression: a corrupt gate vector used to be silently dropped
        // while the op's metadata still landed — a half-applied patch.
        // The whole patch must be rejected and no state mutated,
        // including ops *before* the corrupt one.
        let mut pv = sized_pv(2);
        let patch = make_patch(vec![
            PatchOp::Insert {
                layer: 0,
                feature: 0,
                relation: None,
                entity: "A".into(),
                target: "B".into(),
                confidence: Some(0.9),
                gate_vector_b64: Some(encode_gate_vector(&[1.0, 2.0])),
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
            PatchOp::Insert {
                layer: 1,
                feature: 3,
                relation: None,
                entity: "C".into(),
                target: "D".into(),
                confidence: None,
                gate_vector_b64: Some("!!!!".into()), // invalid base64
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
        ]);
        let err = pv.try_apply_patch(patch).expect_err("corrupt vector");
        assert!(err.to_string().contains("gate_vector_b64"), "{err}");
        assert!(pv.overrides_meta.is_empty(), "no meta may be applied");
        assert!(pv.overrides_gate.is_empty(), "no gate may be applied");
        assert_eq!(pv.num_patches(), 0, "corrupt patch must not be recorded");
    }

    #[test]
    fn try_apply_patch_rejects_truncated_up_vector_on_update() {
        // Truncated base64 (non-multiple-of-4 length) in an Update op's
        // up vector: the op's down_meta must NOT be applied.
        let full = encode_gate_vector(&[1.0f32, -1.0]);
        let truncated = full[..full.len() - 1].to_string();
        let mut pv = sized_pv(2);
        let patch = make_patch(vec![PatchOp::Update {
            layer: 2,
            feature: 4,
            gate_vector_b64: None,
            up_vector_b64: Some(truncated),
            down_vector_b64: None,
            down_meta: Some(PatchDownMeta {
                top_token: "x".into(),
                top_token_id: 1,
                c_score: 0.5,
            }),
        }]);
        assert!(pv.try_apply_patch(patch).is_err());
        assert!(pv.overrides_meta.is_empty());
        assert!(pv.up_override_at(2, 4).is_none());
    }

    #[test]
    fn try_apply_patch_rejects_corrupt_knn_key() {
        let mut pv = sized_pv(3);
        let patch = make_patch(vec![PatchOp::InsertKnn {
            layer: 0,
            entity: "e".into(),
            relation: "r".into(),
            target: "t".into(),
            target_id: 1,
            confidence: None,
            key_vector_b64: "???".into(),
        }]);
        assert!(pv.try_apply_patch(patch).is_err());
        assert_eq!(pv.knn_store.len(), 0);
    }

    #[test]
    fn update_after_delete_clears_tombstone() {
        // Regression (review 2026-07-30 M6): PatchOp::Update on a
        // tombstoned slot must resurrect it, mirroring Insert. Before
        // the fix the tombstone survived, so feature_meta() (via the
        // Some(meta) override) and gate_knn() (tombstone filter)
        // disagreed about the same slot.
        let mut pv = sized_pv(2);
        pv.try_apply_patch(make_patch(vec![PatchOp::Delete {
            layer: 0,
            feature: 1,
            reason: None,
        }]))
        .expect("patch applies");
        assert!(pv.deleted.contains(&(0, 1)));

        pv.try_apply_patch(make_patch(vec![PatchOp::Update {
            layer: 0,
            feature: 1,
            gate_vector_b64: Some(encode_gate_vector(&[1.0f32, 0.0])),
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(PatchDownMeta {
                top_token: "revived".into(),
                top_token_id: 7,
                c_score: 0.6,
            }),
        }]))
        .expect("patch applies");

        assert!(
            !pv.deleted.contains(&(0, 1)),
            "Update must clear the tombstone"
        );
        // Both query paths agree the feature exists again.
        assert_eq!(pv.feature_meta(0, 1).unwrap().top_token, "revived");
        let q = ndarray::Array1::from_vec(vec![1.0_f32, 0.0]);
        let hits = pv.gate_knn(0, &q, 1);
        assert_eq!(hits.len(), 1, "got {hits:?}");
        assert_eq!(hits[0].0, 1, "resurrected feature must be visible to KNN");
    }

    #[test]
    fn update_without_meta_after_delete_drops_pinned_none_meta() {
        // Delete pins `overrides_meta[key] = None`; an Update carrying
        // no down_meta must not leave that pin behind after clearing
        // the tombstone, or feature_meta() would keep reporting None
        // for a slot gate_knn() now returns.
        let mut pv = sized_pv(2);
        pv.try_apply_patch(make_patch(vec![PatchOp::Delete {
            layer: 0,
            feature: 2,
            reason: None,
        }]))
        .expect("patch applies");
        pv.try_apply_patch(make_patch(vec![PatchOp::Update {
            layer: 0,
            feature: 2,
            gate_vector_b64: Some(encode_gate_vector(&[0.0f32, 1.0])),
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }]))
        .expect("patch applies");
        assert!(!pv.deleted.contains(&(0, 2)));
        assert!(
            !matches!(pv.overrides_meta.get(&(0, 2)), Some(None)),
            "pinned-None meta override must be dropped on resurrection"
        );
    }

    #[test]
    fn apply_insert_knn_adds_to_knn_store() {
        let mut pv = sized_pv(3);
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
        pv.try_apply_patch(patch).expect("patch applies");
        assert_eq!(pv.knn_store.len(), 1);
    }

    #[test]
    fn apply_delete_knn_removes_from_knn_store() {
        let mut pv = sized_pv(3);
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
        pv.try_apply_patch(insert).expect("patch applies");
        assert_eq!(pv.knn_store.len(), 1);
        pv.try_apply_patch(delete).expect("patch applies");
        assert_eq!(pv.knn_store.len(), 0);
    }
}
