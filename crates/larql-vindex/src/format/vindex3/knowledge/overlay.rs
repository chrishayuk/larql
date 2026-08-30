//! The V3 mutation overlay — logical edits over a read-only container.
//!
//! V3-LQL-3B: a VINDEX3 container is immutable on disk ("the bytes
//! executed are the bytes stored" is a construction-level property of
//! the format). Mutation therefore lives in an overlay addressed by
//! **semantic identity** — entity-keyed KNN entries today, feature-slot
//! overrides when the compose rung lands — never by byte offsets, so
//! an edit survives repacking or an alternative physical layout.
//!
//! The overlay speaks the same logical patch language as VINDEX2
//! ([`VindexPatch`] / [`PatchOp`]): one `.vlp` file applies to either
//! format, and the V2 semantics are the contract — all-or-nothing
//! apply, removal rebuilds by replaying the remaining patch list
//! (mirroring `PatchedVindex::rebuild_overrides`, session state
//! included in the reset).
//!
//! Feature-slot state (V3-LQL-3B rung 2) carries the V2 tombstone
//! contract verbatim (`PatchedVindex`, review 2026-07-30 M6): DELETE
//! pins a `None` meta override *and* tombstones the slot; a later
//! UPDATE resurrects it, and every read path — `feature_meta`, the
//! gate scan, WALK — must agree about the slot's existence.
//!
//! Since the compose rung, vector-bearing operations (compose
//! installs) apply too: gate/up rows and down columns land in the
//! overlay and reach execution through the operand-source seam
//! (`KnowledgeOverlay::operand_overrides` → `OperandSource`), so a compose patch
//! alters what the program computes without touching the container.

use std::collections::{HashMap, HashSet};

use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::operands::{OperandEdit, OperandOverrides};
use crate::format::vindex3::opplan::{ComponentOpPlan, LayerFfn};
use crate::index::types::{FeatureMeta, DEFAULT_C_SCORE};
use crate::patch::format::{decode_gate_vector, PatchOp, VindexPatch};
use crate::patch::knn_store::KnnStore;

use super::KnowledgeView;

/// Logical mutation state over one bound VINDEX3 container.
#[derive(Default)]
pub struct KnowledgeOverlay {
    /// Entity-keyed retrieval entries (Architecture B). Shared store
    /// type with V2 — same logical semantics, same `knn_store.bin`
    /// persistence format.
    pub knn_store: KnnStore,
    /// Patches applied to this session, in application order.
    pub patches: Vec<VindexPatch>,
    /// Feature-meta overrides: `Some(meta)` replaces the base view's
    /// annotation, a pinned `None` (from DELETE) hides it.
    overrides_meta: HashMap<(usize, usize), Option<FeatureMeta>>,
    /// Tombstoned feature slots — excluded from every read path until
    /// an UPDATE resurrects them.
    deleted: HashSet<(usize, usize)>,
    /// Compose-install vector state (V3-LQL-3B compose): per-slot gate
    /// and up **rows** and down **columns**, in f32. Browse merges the
    /// gate rows into its scan; execution observes all three through
    /// the operand-source seam ([`Self::operand_overrides`]).
    overrides_gate: HashMap<(usize, usize), Vec<f32>>,
    overrides_up: HashMap<(usize, usize), Vec<f32>>,
    overrides_down: HashMap<(usize, usize), Vec<f32>>,
}

impl KnowledgeOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tombstone a feature slot (V2's `PatchedVindex::delete_feature`
    /// contract: pin a `None` meta AND record the tombstone, so the
    /// meta path and the gate-scan path agree the slot is gone).
    pub fn delete_feature(&mut self, layer: usize, feature: usize) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, None);
        self.deleted.insert(key);
        // V2's contract: deleting the slot drops its gate override
        // (the per-layer feature set shrinks either way).
        self.overrides_gate.remove(&key);
    }

    /// Override a feature's metadata; a prior tombstone on the slot is
    /// cleared (resurrection — updating a feature implies it exists).
    pub fn update_feature_meta(&mut self, layer: usize, feature: usize, meta: FeatureMeta) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
        self.deleted.remove(&key);
    }

    /// Install a compose slot: gate row + annotation, resurrecting a
    /// tombstoned slot (V2's `PatchedVindex::insert_feature` contract).
    pub fn insert_feature(
        &mut self,
        layer: usize,
        feature: usize,
        gate: Vec<f32>,
        meta: FeatureMeta,
    ) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
        self.deleted.remove(&key);
        if !gate.is_empty() {
            self.overrides_gate.insert(key, gate);
        }
    }

    /// Replace a composed slot's gate row without touching its
    /// annotation — the refine pass's write-back (V2's
    /// `set_gate_override`).
    pub fn set_gate_vector(&mut self, layer: usize, feature: usize, gate: Vec<f32>) {
        self.overrides_gate.insert((layer, feature), gate);
    }

    pub fn set_up_vector(&mut self, layer: usize, feature: usize, up: Vec<f32>) {
        self.overrides_up.insert((layer, feature), up);
    }

    pub fn set_down_vector(&mut self, layer: usize, feature: usize, down: Vec<f32>) {
        self.overrides_down.insert((layer, feature), down);
    }

    /// One slot's overridden gate row, when present.
    pub fn gate_override_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.overrides_gate
            .get(&(layer, feature))
            .map(Vec::as_slice)
    }

    /// One slot's overridden up row, when present.
    pub fn up_override_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.overrides_up.get(&(layer, feature)).map(Vec::as_slice)
    }

    /// One slot's overridden down column, when present.
    pub fn down_override_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.overrides_down
            .get(&(layer, feature))
            .map(Vec::as_slice)
    }

    /// The overridden gate rows at one layer — browse merges these
    /// into its scan the way V2's `GateOverlay` merge does.
    pub fn gate_overrides_at(&self, layer: usize) -> Vec<(usize, &[f32])> {
        self.overrides_gate
            .iter()
            .filter(|((l, _), _)| *l == layer)
            .map(|((_, f), v)| (*f, v.as_slice()))
            .collect()
    }

    /// Slot-state override count — V2's `num_overrides` semantics
    /// (meta and vector overrides; NOT the KNN store, which is L0
    /// knowledge a container legitimately carries as `knn_store.bin`).
    pub fn num_overrides(&self) -> usize {
        let mut keys: std::collections::BTreeSet<(usize, usize)> =
            self.overrides_meta.keys().copied().collect();
        keys.extend(self.overrides_gate.keys());
        keys.extend(self.overrides_up.keys());
        keys.extend(self.overrides_down.keys());
        keys.len()
    }

    /// Overlay state a clean-container bake cannot represent: V3
    /// annotations are DERIVED from weights (`embed · feature_down`),
    /// so a tombstone or a meta-only relabel (an UPDATE that changed
    /// no vectors) has no physical form in a compiled container —
    /// they live as overlay/patch state only. COMPILE refuses while
    /// any exist rather than silently dropping them.
    pub fn bake_blockers(&self) -> Vec<String> {
        let mut blockers: Vec<String> = self
            .deleted
            .iter()
            .map(|(l, f)| format!("tombstone at ({l},{f})"))
            .collect();
        for (&(l, f), meta) in &self.overrides_meta {
            let has_vectors = self.overrides_gate.contains_key(&(l, f))
                || self.overrides_up.contains_key(&(l, f))
                || self.overrides_down.contains_key(&(l, f));
            if meta.is_some() && !has_vectors && !self.deleted.contains(&(l, f)) {
                blockers.push(format!("meta-only override at ({l},{f})"));
            }
        }
        blockers.sort();
        blockers
    }

    /// Whether any compose vector state exists — execution only takes
    /// the overlaid path while this is true.
    pub fn has_vector_state(&self) -> bool {
        !self.overrides_gate.is_empty()
            || !self.overrides_up.is_empty()
            || !self.overrides_down.is_empty()
    }

    /// V2's free-slot rule (`PatchedVindex::find_free_feature`): first
    /// preference a slot with no base metadata and no overlay claim;
    /// else the weakest-`c_score` unclaimed slot. A pinned-`None` meta
    /// (tombstone) leaves the slot free.
    pub fn find_free_feature(&self, view: &KnowledgeView, layer: usize) -> Option<usize> {
        let n = view.num_features(layer);
        if n == 0 {
            return None;
        }
        let taken_by_overlay = |i: usize| -> bool {
            self.overrides_gate.contains_key(&(layer, i))
                || matches!(self.overrides_meta.get(&(layer, i)), Some(Some(_)))
        };
        for i in 0..n {
            if view.feature_meta(layer, i).is_none() && !taken_by_overlay(i) {
                return Some(i);
            }
        }
        let mut weakest: Option<(usize, f32)> = None;
        for i in 0..n {
            if taken_by_overlay(i) {
                continue;
            }
            let Some(meta) = view.feature_meta(layer, i) else {
                continue;
            };
            if weakest.is_none_or(|(_, score)| meta.c_score < score) {
                weakest = Some((i, meta.c_score));
            }
        }
        weakest.map(|(i, _)| i)
    }

    /// Derive the executor-facing operand edits from the compose state:
    /// gate/up rows and down columns mapped onto the plan's own FFN
    /// operand identities. Fails closed on a layer whose FFN the
    /// overlay cannot yet address (routed/MoE — a later role rung).
    pub fn operand_overrides(
        &self,
        plan: &ComponentOpPlan,
    ) -> Result<OperandOverrides, VindexError> {
        let mut overrides = OperandOverrides::new();
        // Deterministic derivation order (HashMap iteration is not).
        let keys: std::collections::BTreeSet<(usize, usize)> = self
            .overrides_gate
            .keys()
            .chain(self.overrides_up.keys())
            .chain(self.overrides_down.keys())
            .copied()
            .collect();

        for (layer, feature) in keys {
            let key = (layer, feature);
            let (gate, up, down) = (
                self.overrides_gate.get(&key),
                self.overrides_up.get(&key),
                self.overrides_down.get(&key),
            );
            let layer_plan = plan.layers.get(layer).ok_or_else(|| {
                VindexError::Parse(format!("overlay addresses layer {layer} beyond the plan"))
            })?;
            let Some(LayerFfn::Dense(ffn)) = &layer_plan.ffn else {
                return Err(VindexError::Parse(format!(
                    "layer {layer} is routed — compose installs on MoE layers are a later                      role rung"
                )));
            };
            if let Some(gate) = gate {
                let target = ffn.gate.as_ref().unwrap_or(&ffn.up);
                overrides.push(
                    target,
                    OperandEdit::Row {
                        index: feature,
                        values: gate.clone(),
                    },
                );
            }
            if let Some(up) = up {
                overrides.push(
                    &ffn.up,
                    OperandEdit::Row {
                        index: feature,
                        values: up.clone(),
                    },
                );
            }
            if let Some(down) = down {
                overrides.push(
                    &ffn.down,
                    OperandEdit::Column {
                        index: feature,
                        values: down.clone(),
                    },
                );
            }
        }
        Ok(overrides)
    }

    /// Resolve a slot's metadata over the base view's answer — V2's
    /// exact read rule: an override wins (its pinned `None` hides the
    /// slot), a bare tombstone hides it, otherwise the base answers.
    pub fn resolve_feature_meta(
        &self,
        layer: usize,
        feature: usize,
        base: Option<FeatureMeta>,
    ) -> Option<FeatureMeta> {
        let key = (layer, feature);
        if let Some(override_meta) = self.overrides_meta.get(&key) {
            return override_meta.clone();
        }
        if self.deleted.contains(&key) {
            return None;
        }
        base
    }

    /// Whether the slot is currently tombstoned.
    pub fn is_tombstoned(&self, layer: usize, feature: usize) -> bool {
        self.deleted.contains(&(layer, feature))
    }

    /// How many slots are tombstoned at `layer` — the exact oversample
    /// a gate scan needs to stay full after filtering.
    pub fn tombstones_at(&self, layer: usize) -> usize {
        self.deleted.iter().filter(|&&(l, _)| l == layer).count()
    }

    /// Whether any feature-slot state exists (meta overrides or
    /// tombstones) — lets read paths skip the merge entirely.
    pub fn has_feature_state(&self) -> bool {
        !self.overrides_meta.is_empty() || !self.deleted.is_empty()
    }

    /// Apply the overlay onto one layer's full annotation vector
    /// (`feature_metas`-shaped reads).
    pub fn apply_meta_overrides(&self, layer: usize, metas: &mut [Option<FeatureMeta>]) {
        for (feature, slot) in metas.iter_mut().enumerate() {
            let key = (layer, feature);
            if let Some(override_meta) = self.overrides_meta.get(&key) {
                *slot = override_meta.clone();
            } else if self.deleted.contains(&key) {
                *slot = None;
            }
        }
    }

    /// Apply a patch, all-or-nothing: on `Err` no overlay state has
    /// been touched and the patch is not recorded. Errors when an
    /// embedded vector fails to decode, or when the patch contains
    /// feature-slot operations the V3 overlay cannot yet represent.
    pub fn try_apply_patch(&mut self, patch: VindexPatch) -> Result<(), VindexError> {
        validate_v3_patch(&patch)?;
        self.apply_unchecked(&patch);
        self.patches.push(patch);
        Ok(())
    }

    /// Remove a previously applied patch and rebuild the overlay by
    /// replaying the remaining patch list — V2's removal contract:
    /// session-added state outside a patch is reset too.
    pub fn remove_patch(&mut self, index: usize) {
        if index >= self.patches.len() {
            return;
        }
        self.patches.remove(index);
        self.knn_store = KnnStore::default();
        self.overrides_meta.clear();
        self.deleted.clear();
        self.overrides_gate.clear();
        self.overrides_up.clear();
        self.overrides_down.clear();
        let patches = std::mem::take(&mut self.patches);
        for patch in patches {
            self.apply_unchecked(&patch);
            self.patches.push(patch);
        }
    }

    fn apply_unchecked(&mut self, patch: &VindexPatch) {
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
                }
                PatchOp::DeleteKnn { entity } => {
                    self.knn_store.remove_by_entity(entity);
                }
                PatchOp::Delete { layer, feature, .. } => {
                    self.delete_feature(*layer, *feature);
                }
                // V2's Update resolution (overlay_apply) verbatim:
                // carried vectors land in the overlay, a carried meta
                // becomes the override, and the resurrect rule drops a
                // pinned `None` only when this Update carries no
                // replacement meta.
                PatchOp::Update {
                    layer,
                    feature,
                    down_meta,
                    gate_vector_b64,
                    up_vector_b64,
                    down_vector_b64,
                } => {
                    let key = (*layer, *feature);
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key, vec);
                        }
                    }
                    if let Some(b64) = up_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_up.insert(key, vec);
                        }
                    }
                    if let Some(b64) = down_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_down.insert(key, vec);
                        }
                    }
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
                    if self.deleted.remove(&key)
                        && matches!(self.overrides_meta.get(&key), Some(None))
                    {
                        self.overrides_meta.remove(&key);
                    }
                }
                // V2's Insert resolution (`overlay_apply`): meta from
                // `down_meta` when carried, else synthesised from the
                // op's target/confidence; vectors land in the overlay.
                // (V2 splits up/down onto the base index so COMPILE can
                // hard-link gate_vectors.bin — a physical-bake concern
                // V3 does not have; all three live in the overlay
                // here, resolved through the operand-source seam.)
                PatchOp::Insert {
                    layer,
                    feature,
                    target,
                    confidence,
                    gate_vector_b64,
                    up_vector_b64,
                    down_vector_b64,
                    down_meta,
                    ..
                } => {
                    let key = (*layer, *feature);
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
                        }
                    }
                    if let Some(b64) = up_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_up.insert(key, vec);
                        }
                    }
                    if let Some(b64) = down_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_down.insert(key, vec);
                        }
                    }
                }
            }
        }
    }
}

/// Check every operation is representable on the V3 overlay and every
/// embedded vector decodes, before any state changes.
fn validate_v3_patch(patch: &VindexPatch) -> Result<(), VindexError> {
    for (i, op) in patch.operations.iter().enumerate() {
        match op {
            PatchOp::InsertKnn { key_vector_b64, .. } => {
                decode_gate_vector(key_vector_b64).map_err(|e| {
                    VindexError::Parse(format!("patch op {i}: corrupt key_vector_b64: {e}"))
                })?;
            }
            PatchOp::DeleteKnn { .. } | PatchOp::Delete { .. } => {}
            PatchOp::Update {
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            }
            | PatchOp::Insert {
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
                        decode_gate_vector(b64).map_err(|e| {
                            VindexError::Parse(format!("patch op {i}: corrupt {field}: {e}"))
                        })?;
                    }
                }
            }
        }
    }
    Ok(())
}
