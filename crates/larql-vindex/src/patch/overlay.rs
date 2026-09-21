//! PatchedVindex — runtime overlay on an immutable base index.
//!
//! Holds the resolved override maps (`overrides_meta`, `overrides_gate`,
//! `deleted`) plus the L0 `KnnStore`. Knows how to apply a `VindexPatch`
//! (from `super::format`) to its overlay state, query the result via
//! `gate_knn` / `walk` / `feature_meta`, and bake everything back into
//! a clean `VectorIndex` via `bake_down`.
//!
//! The on-the-wire patch format (`VindexPatch`, `PatchOp`,
//! `PatchDownMeta`, base64 helpers) lives in `super::format`.

use std::collections::HashMap;

use ndarray::Array1;

use crate::index::storage::vindex_storage::VindexStorage;
use crate::index::{FeatureMeta, VectorIndex, WalkHit, WalkTrace};

use super::format::VindexPatch;
use super::gate_overlay::GateOverlay;

/// Oversampling factor for the base KNN query when the overlay has
/// gate overrides and/or tombstoned deletions at the queried layer.
/// Overrides can re-score base hits and tombstones remove them, so
/// the merge/filter/sort step needs headroom beyond `top_k` to stay
/// correct. Kept small — the common case is a handful of overlay
/// entries, and this factor prices every patched-layer query.
const BASE_KNN_OVERSAMPLE_FACTOR: usize = 2;

/// First escalation of [`BASE_KNN_OVERSAMPLE_FACTOR`] when tombstoned
/// deletions hollow out the oversampled window (review 2026-07-30
/// M11): if more than `top_k` of the top `2·top_k` base hits are
/// deleted, re-query at 4× before falling back to an all-features
/// query. Keeps the common few-deletions case on the cheap 2× path
/// while guaranteeing `top_k` survivors whenever the base has them.
const BASE_KNN_OVERSAMPLE_RETRY_FACTOR: usize = 4;

// ═══════════════════════════════════════════════════════════════
// PatchedVindex — overlay on immutable base
// ═══════════════════════════════════════════════════════════════

/// A vindex with patches applied as an overlay.
/// The base **files on disk** are never modified.
///
/// ## Layering: gate overrides vs down vector overrides
///
/// `PatchedVindex` deliberately stores its overrides in **two different
/// places** depending on what they are:
///
/// - **Gate vectors** (`insert_feature`, `update_feature_meta`) live in
///   `self.overrides_gate` and `self.overrides_meta` — true overlays
///   that don't touch the base. `gate_knn` consults these on top of the
///   base scores.
///
/// - **Down vectors** (`set_down_vector`) are forwarded to
///   `self.base.set_down_vector`, which mutates the base's
///   `down_overrides` HashMap in place. The base files on disk remain
///   unchanged, but the in-memory base picks up the override directly.
///   `walk_ffn`'s `down_override(layer, feat)` lookup then finds the
///   override on the base.
///
/// This asymmetry is **intentional** and load-bearing for
/// `COMPILE INTO VINDEX`. The dense FFN inference path
/// (`walk_ffn_full_mmap`) reads gate scores from `gate_vectors.bin` via
/// `gate_scores_batch`. If the inserted (norm-matched) gate vector were
/// baked into that file, the dense activation at the inserted slot
/// would become moderate-to-large; combined with the override down
/// vector (multi-layer constellation install at α=0.25 per layer) the
/// residual stream blows up. Keeping the source's weak free-slot gate
/// at the inserted index leaves the dense activation small, so
/// `small_activation × poseidon_vector` per layer accumulates into the
/// validated constellation effect.
///
/// `COMPILE INTO VINDEX` therefore:
///   - Hard-links `gate_vectors.bin` from source (unchanged), and
///   - Bakes the down vectors into `down_weights.bin` via column-rewrite
///     at the inserted slots.
///
/// This is why `down_overrides()` reaches through to the base while
/// `overrides_gate_at()` reads the patch overlay — the two types of
/// override live in different places by design. Don't "fix" this by
/// moving down vectors into a separate overlay map, or you'll have to
/// re-solve the activation-blowup problem.
pub struct PatchedVindex {
    /// Immutable base index. Note: `set_down_vector` mutates
    /// `base.metadata.down_overrides` in place — see the layering doc above.
    pub base: VectorIndex,
    /// Applied patches (in order).
    pub patches: Vec<VindexPatch>,
    /// Resolved meta overrides: (layer, feature) → effective metadata.
    /// Later patches override earlier ones for the same feature.
    pub(crate) overrides_meta: HashMap<(usize, usize), Option<FeatureMeta>>,
    /// Resolved gate vector overrides: (layer, feature) → gate vector.
    /// Lives in the overlay (not on `base`) so that the source
    /// `gate_vectors.bin` stays clean — see layering doc above.
    /// Stored in the shared [`GateOverlay`] retrieval structure (the
    /// same kernel serves the arch-B `KnnStore` since the 2026-07-31
    /// unification), which owns the per-layer flattened snapshot cache
    /// and invalidates it automatically on every mutation.
    pub(crate) overrides_gate: GateOverlay,
    /// Tombstones for deleted features.
    ///
    /// Contract (review 2026-07-30 M6): `Delete` adds the key;
    /// `Insert` **and** `Update` on the same slot remove it —
    /// mutating a feature implies it exists (resurrection). Every
    /// query path (`feature_meta`, `gate_knn`, `bake_down`) must
    /// agree with this set; a path that consults the tombstones
    /// differently from the others is a bug.
    pub(crate) deleted: std::collections::HashSet<(usize, usize)>,
    /// Architecture B: per-layer retrieval-override KNN store.
    pub knn_store: super::knn_store::KnnStore,
}

impl PatchedVindex {
    /// Create a patched vindex from a base index.
    pub fn new(base: VectorIndex) -> Self {
        Self {
            base,
            patches: Vec::new(),
            overrides_meta: HashMap::new(),
            overrides_gate: GateOverlay::default(),
            deleted: std::collections::HashSet::new(),
            knn_store: super::knn_store::KnnStore::default(),
        }
    }

    /// Insert a feature directly into the overlay (auto-patch mode).
    ///
    /// An empty `gate_vec` means "no gate override" (metadata-only
    /// insert, e.g. a Vindexfile INSERT without embeddings to
    /// synthesise a gate): the slot is claimed via `overrides_meta`
    /// but nothing lands in `overrides_gate` — a zero-width row would
    /// poison the flattened per-layer gate cache.
    pub fn insert_feature(
        &mut self,
        layer: usize,
        feature: usize,
        gate_vec: Vec<f32>,
        meta: FeatureMeta,
    ) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
        if gate_vec.is_empty() {
            self.overrides_gate.remove(layer, feature);
        } else {
            self.overrides_gate.insert(layer, feature, gate_vec);
        }
        self.deleted.remove(&key);
    }

    /// Delete a feature via the overlay.
    pub fn delete_feature(&mut self, layer: usize, feature: usize) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, None);
        self.deleted.insert(key);
        self.overrides_gate.remove(layer, feature);
    }

    /// Update feature metadata via the overlay.
    ///
    /// Mirrors [`Self::insert_feature`]'s tombstone contract:
    /// updating a feature implies it exists, so a prior
    /// [`Self::delete_feature`] tombstone on the slot is cleared
    /// (resurrection). Without this, `feature_meta()` reported the
    /// feature while `gate_knn()` permanently filtered it out — two
    /// query paths disagreeing about the same overlay state
    /// (review 2026-07-30 M6).
    pub fn update_feature_meta(&mut self, layer: usize, feature: usize, meta: FeatureMeta) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
        self.deleted.remove(&key);
    }

    /// Check if a (layer, feature) has been overridden.
    pub fn is_overridden(&self, layer: usize, feature: usize) -> bool {
        self.overrides_meta.contains_key(&(layer, feature))
    }

    /// Access the underlying base index (readonly).
    pub fn base(&self) -> &VectorIndex {
        &self.base
    }

    /// Access the underlying base index (mutable, for down vector overrides).
    pub fn base_mut(&mut self) -> &mut VectorIndex {
        &mut self.base
    }

    /// Set a down vector override for a feature.
    pub fn set_down_vector(&mut self, layer: usize, feature: usize, vector: Vec<f32>) {
        self.base.set_down_vector(layer, feature, vector);
    }

    /// Set an up vector override for a feature. Mirrors
    /// `set_down_vector`; both forward to the base index. INSERT calls
    /// this so the slot's activation `silu(gate · x) * (up · x)`
    /// reflects the constellation install.
    pub fn set_up_vector(&mut self, layer: usize, feature: usize, vector: Vec<f32>) {
        self.base.set_up_vector(layer, feature, vector);
    }

    /// All in-memory up vector overrides on the underlying base vindex.
    /// Parallel to `down_overrides()`. Used by `COMPILE INTO VINDEX` to
    /// bake them into a fresh copy of `up_features.bin`.
    pub fn up_overrides(&self) -> &std::collections::HashMap<(usize, usize), Vec<f32>> {
        self.base.up_overrides()
    }

    /// Up vector override for `(layer, feature)`. Forwards to the base
    /// vindex (up vectors live on `VectorIndex.metadata.up_overrides`, not on the
    /// patch overlay — same layering as `down_override_at`).
    pub fn up_override_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.base.up_override_at(layer, feature)
    }

    /// All in-memory down vector overrides on the underlying base vindex.
    /// Used by `COMPILE INTO VINDEX` to bake them into a fresh copy of
    /// `down_weights.bin`.
    ///
    /// For a single (layer, feature) lookup, use `down_override_at`.
    pub fn down_overrides(&self) -> &std::collections::HashMap<(usize, usize), Vec<f32>> {
        self.base.down_overrides()
    }

    /// Down vector override for `(layer, feature)`, if any. Forwards to
    /// the base vindex (down vectors live on `VectorIndex.metadata.down_overrides`,
    /// not on the patch overlay — see the layering doc on `PatchedVindex`).
    pub fn down_override_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.base.down_override_at(layer, feature)
    }

    /// Override gate vector for `(layer, feature)`, if present in the
    /// patch overlay. Used by `COMPILE INTO VINDEX` to read each
    /// inserted gate vector for sidecar serialisation.
    pub fn overrides_gate_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.overrides_gate.get(layer, feature)
    }

    /// Read-only iterator over every gate override slot in the overlay.
    /// Used by `COMPILE INTO VINDEX WITH REFINE` to enumerate the
    /// constellation before refining.
    pub fn overrides_gate_iter(&self) -> impl Iterator<Item = (usize, usize, &[f32])> + '_ {
        self.overrides_gate.iter()
    }

    /// Replace the gate override for `(layer, feature)` with a new
    /// vector. Used by `COMPILE INTO VINDEX WITH REFINE` to write the
    /// refined gate back into the overlay before the bake step. Has no
    /// effect if the slot does not already have a gate override (we
    /// only refine slots that were already touched by a patch).
    pub fn set_gate_override(&mut self, layer: usize, feature: usize, vector: Vec<f32>) {
        self.overrides_gate.replace_existing(layer, feature, vector);
    }

    /// Find a free feature slot at this layer that is NOT already
    /// claimed by the patch overlay. The base index only knows about
    /// its own gate matrix and `down_meta`, so its
    /// `find_free_feature` keeps returning the same "weakest" slot
    /// across calls — which is catastrophic for multi-fact INSERT:
    /// every new INSERT picks the same slot and overwrites the
    /// previous install (validated by the `refine_demo` "last fact
    /// always wins" diagnostic). This wrapper asks the base for
    /// candidate slots and skips any that the overlay has already
    /// taken, scanning linearly until it finds one that's free both
    /// in the base AND in the overlay.
    pub fn find_free_feature(&self, layer: usize) -> Option<usize> {
        let n = self.base.num_features(layer);
        if n == 0 {
            return None;
        }

        // A slot is overlay-claimed if it has a gate override OR a
        // live meta override (metadata-only INSERTs carry no gate
        // vector; a `None` meta is a tombstone, so the slot is free).
        let taken_by_overlay = |i: usize| -> bool {
            self.overrides_gate.contains(layer, i)
                || matches!(self.overrides_meta.get(&(layer, i)), Some(Some(_)))
        };

        // First preference: a slot with no base metadata AND no
        // overlay entry. This matches the base's "no metadata = free"
        // semantics but also respects the overlay.
        for i in 0..n {
            let taken_by_base = self.base.feature_meta(layer, i).is_some();
            if !taken_by_base && !taken_by_overlay(i) {
                return Some(i);
            }
        }

        // Second preference: a slot with base metadata (some c_score)
        // that the overlay has NOT claimed, picking the weakest c_score.
        // This mirrors the base's fallback path but filters out
        // overlay-claimed slots.
        let mut weakest_idx: Option<usize> = None;
        let mut weakest_score = f32::MAX;
        for i in 0..n {
            if taken_by_overlay(i) {
                continue;
            }
            if let Some(meta) = self.base.feature_meta(layer, i) {
                if meta.c_score < weakest_score {
                    weakest_score = meta.c_score;
                    weakest_idx = Some(i);
                }
            }
        }
        weakest_idx
    }

    /// Look up feature metadata, checking overrides first.
    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        let key = (layer, feature);
        if let Some(override_meta) = self.overrides_meta.get(&key) {
            return override_meta.clone();
        }
        if self.deleted.contains(&key) {
            return None;
        }
        self.base.feature_meta(layer, feature)
    }

    /// Base-index KNN at `layer` that keeps escalating the oversample
    /// window until `top_k` non-tombstoned hits survive or the base is
    /// exhausted (review 2026-07-30 M11).
    ///
    /// The cheap path oversamples by [`BASE_KNN_OVERSAMPLE_FACTOR`].
    /// If the base returned a full (unfiltered) result set — meaning
    /// more features exist — yet fewer than `top_k` hits survive the
    /// tombstone filter, the query escalates to
    /// [`BASE_KNN_OVERSAMPLE_RETRY_FACTOR`] and finally to a single
    /// all-features query, so callers never silently under-fill while
    /// live features remain. Returned hits are already
    /// tombstone-filtered and stay sorted by `|score|` descending
    /// (`retain` preserves the base's ordering).
    fn base_knn_surviving_deletions(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let num_features = self.base.num_features(layer);
        for factor in [BASE_KNN_OVERSAMPLE_FACTOR, BASE_KNN_OVERSAMPLE_RETRY_FACTOR] {
            let requested = top_k.saturating_mul(factor);
            let mut hits = self.base.gate_knn(layer, residual, requested);
            // A short (non-full) return means the base has nothing
            // more to offer — retrying with a bigger window can't help.
            let base_exhausted =
                requested >= num_features || hits.len() < requested.min(num_features);
            hits.retain(|(f, _)| !self.deleted.contains(&(layer, *f)));
            if hits.len() >= top_k || base_exhausted {
                return hits;
            }
        }
        // Final rung: score every feature once. Bounded — runs at most
        // once per query, and only when both fixed windows were
        // hollowed out by deletions.
        let mut hits = self.base.gate_knn(layer, residual, num_features);
        hits.retain(|(f, _)| !self.deleted.contains(&(layer, *f)));
        hits
    }

    /// Gate KNN with patched vectors.
    /// For features with overridden gate vectors, uses the patch vector.
    /// For deleted features, excludes them from results.
    pub fn gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // Cheap pre-check: are there any patches at this layer?
        // Mirrors `gate_knn_batch`'s short-circuit. When neither map
        // touches `layer`, the base index's sorted top-k is already
        // the answer — skip the 2× oversample, the override merge,
        // and the re-sort.
        let has_overrides = self.overrides_gate.has_layer(layer);
        let has_deletions = self.deleted.iter().any(|&(l, _)| l == layer);
        if !has_overrides && !has_deletions {
            return self.base.gate_knn(layer, residual, top_k);
        }

        // #1: When the base layer has zero features (e.g. an INSERT-
        // only Exp 53 shard cache, or a layer-range-restricted vindex
        // where this layer is unowned), skip the entire
        // `base.gate_knn` dispatch chain — atomic HNSW flag load →
        // `gate_knn_mmap_fast` → `resolve_gate` → returning empty.
        // That's ~3–5 µs of pure overhead per call.
        //
        // When the base does carry features, oversample by
        // `BASE_KNN_OVERSAMPLE_FACTOR` so the sort step has headroom
        // to keep top_k correct after override merge. When tombstones
        // exist at this layer, a fixed window can be hollowed out by
        // deletions — go through the escalating helper instead.
        // `saturating_mul` guards `usize::MAX` callers.
        let mut hits = if self.base.num_features(layer) > 0 {
            if has_deletions {
                self.base_knn_surviving_deletions(layer, residual, top_k)
            } else {
                self.base.gate_knn(
                    layer,
                    residual,
                    top_k.saturating_mul(BASE_KNN_OVERSAMPLE_FACTOR),
                )
            }
        } else {
            Vec::new()
        };

        if has_overrides {
            // Score every override row at this layer through the
            // shared `GateOverlay` kernel (flattened per-layer
            // snapshot with the zero-width/mixed-width slow-path
            // fallback — the same code path the arch-B `KnnStore`
            // queries through), then merge into the base hits.
            let override_scores = self.overrides_gate.score_layer(layer, residual);
            if hits.is_empty() {
                // Fast path: empty base → the override scores ARE
                // the candidate set.
                hits = override_scores;
            } else {
                // Build a feat → hit_idx lookup once so the merge
                // is O(overrides + hits) instead of O(overrides ×
                // hits). Previous `hits.iter_mut().find(...)`
                // per override was quadratic at n ≈ 256.
                let mut idx: HashMap<usize, usize> =
                    hits.iter().enumerate().map(|(i, (f, _))| (*f, i)).collect();
                for (feat, score) in override_scores {
                    match idx.get(&feat) {
                        Some(&hit_idx) => hits[hit_idx].1 = score,
                        None => {
                            idx.insert(feat, hits.len());
                            hits.push((feat, score));
                        }
                    }
                }
            }
        }

        if has_deletions {
            hits.retain(|(f, _)| !self.deleted.contains(&(layer, *f)));
        }

        // NaN-tolerant `|score|` comparator — a poisoned score
        // collapses to Equal rather than panicking the whole query.
        let cmp_abs_desc = |a: &(usize, f32), b: &(usize, f32)| -> std::cmp::Ordering {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        };

        if has_overrides {
            // Overrides pushed entries in arbitrary order; need a
            // re-rank pass before truncating.
            if top_k == 1 {
                // #4: argmax in one pass instead of O(n log n) sort.
                // The 256-entry sort was ~2 µs at large n; max_by is
                // a tight linear scan over the abs-score.
                let winner = hits.iter().copied().max_by(|a, b| {
                    a.1.abs()
                        .partial_cmp(&b.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                hits.clear();
                if let Some(w) = winner {
                    hits.push(w);
                }
            } else {
                hits.sort_unstable_by(cmp_abs_desc);
                hits.truncate(top_k);
            }
        } else {
            // #5: deletion-only path — `base.gate_knn` returned hits
            // sorted by `|score|` descending, `retain` preserves
            // order. Just truncate.
            hits.truncate(top_k);
        }
        hits
    }

    /// Walk with patch overrides.
    /// Walk the given layers, collecting top-K gate hits per layer.
    ///
    /// Parallel across layers: each layer's `gate_knn` is an independent
    /// gemv over that layer's gate matrix, so there is no ordering or data
    /// dependency between them. `describe()` scans 12--30 layers, which is
    /// enough work per item to cover rayon's dispatch overhead.
    ///
    /// What this does and does not buy, measured rather than assumed:
    /// it cuts **single-query latency** by spreading one query's layers
    /// across cores. It does **not** raise saturated throughput, because
    /// the total bytes read per query are unchanged and the scan is
    /// memory-bound, not compute-bound (15% of 32 vCPUs under 8 concurrent
    /// requests -- threads stalled on memory, not busy). Under concurrent
    /// load the queries already fill the cores; this helps the single-query
    /// case and the lightly-loaded case.
    ///
    /// `map` on a parallel iterator preserves input order, so the returned
    /// trace is still layer-ordered.
    pub fn walk(&self, residual: &Array1<f32>, layers: &[usize], top_k: usize) -> WalkTrace {
        use rayon::prelude::*;
        let trace_layers: Vec<(usize, Vec<WalkHit>)> = layers
            .par_iter()
            .map(|&layer| {
                let hits = self.gate_knn(layer, residual, top_k);
                let walk_hits: Vec<WalkHit> = hits
                    .into_iter()
                    .filter_map(|(feature, gate_score)| {
                        let meta = self.feature_meta(layer, feature)?.clone();
                        Some(WalkHit::from_gate(layer, feature, gate_score, meta))
                    })
                    .collect();
                (layer, walk_hits)
            })
            .collect();
        WalkTrace {
            layers: trace_layers,
        }
    }

    /// Flatten all patches into the base, producing a new clean VectorIndex (heap mode).
    pub fn bake_down(&self) -> VectorIndex {
        let mut new_gate = Vec::new();
        let mut new_meta = Vec::new();

        for layer in 0..self.base.num_layers {
            // Get base gate vectors (from heap or mmap)
            let base_gate = if let Some(g) = self.base.gate_vectors_at(layer) {
                Some(g.clone())
            } else if let Some(view) = self.base.storage.gate_layer_view(layer) {
                // Mmap mode — decode this layer's slice to an Array2
                if view.slice.num_features == 0 {
                    None
                } else {
                    let bpf = crate::config::dtype::bytes_per_float(view.dtype);
                    let byte_offset = view.slice.float_offset * bpf;
                    let byte_count = view.slice.num_features * self.base.hidden_size * bpf;
                    let byte_end = byte_offset + byte_count;
                    let mmap: &[u8] = view.bytes.as_ref();
                    if byte_end > mmap.len() {
                        None
                    } else {
                        let floats = crate::config::dtype::decode_floats(
                            &mmap[byte_offset..byte_end],
                            view.dtype,
                        );
                        ndarray::Array2::from_shape_vec(
                            (view.slice.num_features, self.base.hidden_size),
                            floats,
                        )
                        .ok()
                    }
                }
            } else {
                None
            };

            let gate = base_gate.map(|mut g| {
                // Apply gate vector overrides
                for (l, f, vec) in self.overrides_gate.iter() {
                    if l != layer {
                        continue;
                    }
                    if f < g.shape()[0] && vec.len() == g.shape()[1] {
                        for (j, val) in vec.iter().enumerate() {
                            g[[f, j]] = *val;
                        }
                    }
                }
                g
            });
            new_gate.push(gate);

            // Build metadata from heap or mmap
            let num_features = self.base.num_features(layer);
            let mut new_metas: Vec<Option<FeatureMeta>> =
                if let Some(heap) = self.base.down_meta_at(layer) {
                    heap.to_vec()
                } else if num_features > 0 {
                    // Mmap: read each feature on demand
                    (0..num_features)
                        .map(|f| self.base.feature_meta(layer, f))
                        .collect()
                } else {
                    Vec::new()
                };

            // Apply meta overrides
            for (&(l, f), override_meta) in &self.overrides_meta {
                if l != layer {
                    continue;
                }
                while new_metas.len() <= f {
                    new_metas.push(None);
                }
                new_metas[f] = override_meta.clone();
            }
            // Apply deletes
            for &(l, f) in &self.deleted {
                if l == layer && f < new_metas.len() {
                    new_metas[f] = None;
                }
            }

            new_meta.push(if new_metas.is_empty() {
                None
            } else {
                Some(new_metas)
            });
        }

        VectorIndex::new(
            new_gate,
            new_meta,
            self.base.num_layers,
            self.base.hidden_size,
        )
    }

    /// Number of active patches.
    pub fn num_patches(&self) -> usize {
        self.patches.len()
    }

    /// Total override count.
    pub fn num_overrides(&self) -> usize {
        self.overrides_meta.len()
    }

    // ── Forwarding methods to base (for compatibility) ──

    /// Layers that have gate vectors loaded (delegates to base).
    pub fn loaded_layers(&self) -> Vec<usize> {
        self.base.loaded_layers()
    }

    /// Number of features at a layer (delegates to base).
    pub fn num_features(&self, layer: usize) -> usize {
        self.base.num_features(layer)
    }

    /// Access down metadata for a layer (base only — does not include overrides).
    /// For override-aware lookups, use `feature_meta()`.
    pub fn down_meta_at(&self, layer: usize) -> Option<&[Option<FeatureMeta>]> {
        self.base.down_meta_at(layer)
    }

    /// Access gate vectors matrix for a layer (base only).
    pub fn gate_vectors_at(&self, layer: usize) -> Option<&ndarray::Array2<f32>> {
        self.base.gate_vectors_at(layer)
    }

    /// Number of layers (delegates to base).
    pub fn num_layers(&self) -> usize {
        self.base.num_layers
    }

    /// Hidden size (delegates to base).
    pub fn hidden_size(&self) -> usize {
        self.base.hidden_size
    }
}

#[cfg(test)]
mod gate_override_tests;
