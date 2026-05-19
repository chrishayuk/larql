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
use std::sync::{Arc, RwLock};

use ndarray::Array1;

use crate::index::storage::vindex_storage::VindexStorage;
use crate::index::{FeatureMeta, VectorIndex, WalkHit, WalkTrace};

use super::format::VindexPatch;

/// Per-layer contiguous gate-override snapshot built lazily by
/// `gate_knn`. Keeps the override matvec cache-friendly — same memory
/// layout as `ShardCache` — instead of pointer-chasing through a
/// `HashMap<(layer, feat), Vec<f32>>`.
///
/// Built on first query at a layer, invalidated on any mutation to
/// `overrides_gate` via [`PatchedVindex::invalidate_gate_cache`].
/// Memory cost: one f32-per-element per cached layer, which doubles
/// the override storage; pays off after ≥1 cache hit per build.
struct LayerGateCache {
    /// Feature IDs in row-order of `gate_matrix`. Indices are stable
    /// for the lifetime of one cache entry — invalidating + rebuilding
    /// the cache renumbers them.
    feature_ids: Vec<usize>,
    /// `n × d` row-major; n = `feature_ids.len()`. Direct slice access
    /// gives the gate vector for the i-th feature ID.
    gate_matrix: Vec<f32>,
    /// Hidden dim — needed for slicing the matrix. Cached so callers
    /// don't have to pull it from `base.hidden_size` each call.
    d: usize,
}

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
    pub(crate) overrides_gate: HashMap<(usize, usize), Vec<f32>>,
    /// Tombstones for deleted features.
    pub(crate) deleted: std::collections::HashSet<(usize, usize)>,
    /// Architecture B: per-layer retrieval-override KNN store.
    pub knn_store: super::knn_store::KnnStore,
    /// Lazy per-layer cache of `overrides_gate` flattened into a
    /// contiguous matrix. Built on first `gate_knn` query at each
    /// layer; per-layer invalidation by every mutator that touches
    /// `overrides_gate` so a single-layer patch stream leaves
    /// unrelated layers' caches hot. `Arc` lets cached entries be
    /// read without holding the outer `RwLock` for the whole matvec.
    gate_cache: RwLock<HashMap<usize, Arc<LayerGateCache>>>,
}

impl PatchedVindex {
    /// Create a patched vindex from a base index.
    pub fn new(base: VectorIndex) -> Self {
        Self {
            base,
            patches: Vec::new(),
            overrides_meta: HashMap::new(),
            overrides_gate: HashMap::new(),
            deleted: std::collections::HashSet::new(),
            knn_store: super::knn_store::KnnStore::default(),
            gate_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Drop every layer's cached `overrides_gate` matrix. Used by
    /// `rebuild_overrides`, which clears the entire override state at
    /// once and re-applies the patch list from scratch. Per-mutation
    /// sites prefer [`invalidate_gate_cache_layer`] so a busy
    /// patch-stream doesn't keep evicting unrelated layers.
    ///
    /// `&mut self` is explicit about who's invalidating; the `RwLock`
    /// is internal and never poisoned in practice.
    pub(crate) fn invalidate_gate_cache(&mut self) {
        if let Ok(mut g) = self.gate_cache.write() {
            g.clear();
        }
    }

    /// Drop the cached `overrides_gate` matrix for one layer. Called
    /// by `insert_feature` / `delete_feature` / `set_gate_override`
    /// (which all know their target layer) and by `apply_patch` (once
    /// per touched layer, deduplicated). Leaves caches for other
    /// layers intact — important for high-frequency patch streams
    /// that touch a single layer repeatedly.
    pub(crate) fn invalidate_gate_cache_layer(&mut self, layer: usize) {
        if let Ok(mut g) = self.gate_cache.write() {
            g.remove(&layer);
        }
    }

    /// Build (or reuse) the per-layer contiguous gate-override
    /// snapshot for `layer`. The first call per layer pays the
    /// flatten cost; every subsequent `gate_knn` query at that layer
    /// reuses the `Arc` without holding the outer `RwLock`.
    ///
    /// Returns `None` when there are no gate overrides for `layer`,
    /// avoiding allocating an empty cache entry. Callers fall back
    /// to the per-entry `overrides_gate` iteration in that case
    /// (which the fast-path short-circuit usually avoids anyway).
    fn layer_gate_cache(&self, layer: usize) -> Option<Arc<LayerGateCache>> {
        // Fast path: read-lock check.
        if let Ok(g) = self.gate_cache.read() {
            if let Some(cache) = g.get(&layer) {
                return Some(Arc::clone(cache));
            }
        }
        // Slow path: flatten under a write lock.
        // Counts overrides at this layer first so we can pre-size
        // the matrix exactly — saves a reallocation per build.
        let mut feature_ids: Vec<usize> = Vec::new();
        let mut total_floats = 0usize;
        let mut d = 0usize;
        for (&(l, f), gate_vec) in &self.overrides_gate {
            if l != layer {
                continue;
            }
            if d == 0 {
                d = gate_vec.len();
            } else if gate_vec.len() != d {
                // Mixed widths inside a single layer — caller messed
                // up the API. Skip the cache and let the slow path
                // handle it via the regular iterator.
                return None;
            }
            feature_ids.push(f);
            total_floats += gate_vec.len();
        }
        if feature_ids.is_empty() {
            return None;
        }
        let mut gate_matrix = Vec::with_capacity(total_floats);
        for &feat in &feature_ids {
            // Indexed by `feature_ids` order so the matrix rows align
            // with the IDs. `overrides_gate[(layer, feat)]` is the
            // canonical store and is guaranteed to exist by the loop
            // above.
            let gate_vec = &self.overrides_gate[&(layer, feat)];
            gate_matrix.extend_from_slice(gate_vec);
        }
        let cache = Arc::new(LayerGateCache {
            feature_ids,
            gate_matrix,
            d,
        });
        if let Ok(mut g) = self.gate_cache.write() {
            g.insert(layer, Arc::clone(&cache));
        }
        Some(cache)
    }

    /// Insert a feature directly into the overlay (auto-patch mode).
    pub fn insert_feature(
        &mut self,
        layer: usize,
        feature: usize,
        gate_vec: Vec<f32>,
        meta: FeatureMeta,
    ) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
        self.overrides_gate.insert(key, gate_vec);
        self.deleted.remove(&key);
        self.invalidate_gate_cache_layer(layer);
    }

    /// Delete a feature via the overlay.
    pub fn delete_feature(&mut self, layer: usize, feature: usize) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, None);
        self.deleted.insert(key);
        self.overrides_gate.remove(&key);
        self.invalidate_gate_cache_layer(layer);
    }

    /// Update feature metadata via the overlay.
    pub fn update_feature_meta(&mut self, layer: usize, feature: usize, meta: FeatureMeta) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
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
        self.overrides_gate
            .get(&(layer, feature))
            .map(|v| v.as_slice())
    }

    /// Read-only iterator over every gate override slot in the overlay.
    /// Used by `COMPILE INTO VINDEX WITH REFINE` to enumerate the
    /// constellation before refining.
    pub fn overrides_gate_iter(&self) -> impl Iterator<Item = (usize, usize, &[f32])> + '_ {
        self.overrides_gate
            .iter()
            .map(|(&(l, f), v)| (l, f, v.as_slice()))
    }

    /// Replace the gate override for `(layer, feature)` with a new
    /// vector. Used by `COMPILE INTO VINDEX WITH REFINE` to write the
    /// refined gate back into the overlay before the bake step. Has no
    /// effect if the slot does not already have a gate override (we
    /// only refine slots that were already touched by a patch).
    pub fn set_gate_override(&mut self, layer: usize, feature: usize, vector: Vec<f32>) {
        let key = (layer, feature);
        if let Some(slot) = self.overrides_gate.get_mut(&key) {
            *slot = vector;
            self.invalidate_gate_cache_layer(layer);
        }
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

        // First preference: a slot with no base metadata AND no
        // overlay entry. This matches the base's "no metadata = free"
        // semantics but also respects the overlay.
        for i in 0..n {
            let taken_by_base = self.base.feature_meta(layer, i).is_some();
            let taken_by_overlay = self.overrides_gate.contains_key(&(layer, i));
            if !taken_by_base && !taken_by_overlay {
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
            if self.overrides_gate.contains_key(&(layer, i)) {
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
        let has_overrides = self.overrides_gate.keys().any(|&(l, _)| l == layer);
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
        // When the base does carry features, oversample by 2× so the
        // sort step has headroom to keep top_k correct after override
        // merge. `saturating_mul` guards `usize::MAX` callers.
        let mut hits = if self.base.num_features(layer) > 0 {
            self.base.gate_knn(layer, residual, top_k.saturating_mul(2))
        } else {
            Vec::new()
        };

        if has_overrides {
            // #2 + #3: pull the per-layer contiguous gate matrix
            // (cached after the first query at this layer). One
            // n × d row-major slab + a parallel `feature_ids` Vec
            // give us a cache-friendly matvec — same memory layout
            // as `ShardCache::cosine_similarities`, no
            // pointer-chasing through the `overrides_gate` HashMap.
            // Falls back to per-entry iteration only when mixed gate
            // widths poison the cache (`layer_gate_cache` returns
            // `None`).
            if let Some(cache) = self.layer_gate_cache(layer) {
                // Fast path: empty base → push every override at once.
                if hits.is_empty() {
                    hits.reserve(cache.feature_ids.len());
                    for (i, &feat) in cache.feature_ids.iter().enumerate() {
                        let row = &cache.gate_matrix[i * cache.d..(i + 1) * cache.d];
                        let score: f32 = row.iter().zip(residual.iter()).map(|(a, b)| a * b).sum();
                        hits.push((feat, score));
                    }
                } else {
                    // Build a feat → hit_idx lookup once so the merge
                    // is O(overrides + hits) instead of O(overrides ×
                    // hits). Previous `hits.iter_mut().find(...)`
                    // per override was quadratic at n ≈ 256.
                    let mut idx: HashMap<usize, usize> =
                        hits.iter().enumerate().map(|(i, (f, _))| (*f, i)).collect();
                    for (i, &feat) in cache.feature_ids.iter().enumerate() {
                        let row = &cache.gate_matrix[i * cache.d..(i + 1) * cache.d];
                        let score: f32 = row.iter().zip(residual.iter()).map(|(a, b)| a * b).sum();
                        match idx.get(&feat) {
                            Some(&hit_idx) => hits[hit_idx].1 = score,
                            None => {
                                idx.insert(feat, hits.len());
                                hits.push((feat, score));
                            }
                        }
                    }
                }
            } else {
                // Slow path: mixed-width overrides (caller bug or
                // unusual configuration). Iterate the canonical
                // `overrides_gate` HashMap directly.
                let mut idx: HashMap<usize, usize> =
                    hits.iter().enumerate().map(|(i, (f, _))| (*f, i)).collect();
                for (&(l, f), gate_vec) in &self.overrides_gate {
                    if l != layer {
                        continue;
                    }
                    let score: f32 = gate_vec
                        .iter()
                        .zip(residual.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    match idx.get(&f) {
                        Some(&hit_idx) => hits[hit_idx].1 = score,
                        None => {
                            idx.insert(f, hits.len());
                            hits.push((f, score));
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
    pub fn walk(&self, residual: &Array1<f32>, layers: &[usize], top_k: usize) -> WalkTrace {
        let mut trace_layers = Vec::with_capacity(layers.len());
        for &layer in layers {
            let hits = self.gate_knn(layer, residual, top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.feature_meta(layer, feature)?.clone();
                    Some(WalkHit {
                        layer,
                        feature,
                        gate_score,
                        meta,
                    })
                })
                .collect();
            trace_layers.push((layer, walk_hits));
        }
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
                for (&(l, f), vec) in &self.overrides_gate {
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
mod gate_override_tests {
    //! Direct unit tests for the gate-override accessors and mutator
    //! used by `COMPILE INTO VINDEX WITH REFINE`. The integration tests
    //! in `larql-lql` exercise these via the executor; these tests
    //! cover them at the API surface so a regression in the layering
    //! contract gets caught here without needing the full executor.
    use super::*;
    use crate::index::core::VectorIndex;
    use larql_models::TopKEntry;
    use ndarray::Array2;

    fn make_meta(token: &str) -> FeatureMeta {
        FeatureMeta {
            top_token: token.into(),
            top_token_id: 0,
            c_score: 0.9,
            top_k: vec![TopKEntry {
                token: token.into(),
                token_id: 0,
                logit: 0.9,
            }],
        }
    }

    /// A 2-layer × 3-feature × 4-hidden empty base index for these
    /// tests. Gate vectors and metas are zero — overrides land on top.
    fn make_empty_base() -> PatchedVindex {
        let gate0 = Array2::<f32>::zeros((3, 4));
        let gate1 = Array2::<f32>::zeros((3, 4));
        let down_meta = vec![Some(vec![None, None, None]), Some(vec![None, None, None])];
        let index = VectorIndex::new(vec![Some(gate0), Some(gate1)], down_meta, 2, 4);
        PatchedVindex::new(index)
    }

    #[test]
    fn set_gate_override_replaces_existing_slot() {
        let mut p = make_empty_base();
        p.insert_feature(0, 1, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
        p.set_gate_override(0, 1, vec![0.0, 1.0, 0.0, 0.0]);
        let read = p.overrides_gate_at(0, 1).unwrap();
        assert_eq!(read, &[0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn set_gate_override_is_no_op_when_slot_absent() {
        // The contract is "only refine slots that were already touched
        // by a patch" — set_gate_override should NOT create a new entry
        // out of nothing. Verifying this stops a future caller from
        // accidentally inserting half-state (gate without meta).
        let mut p = make_empty_base();
        p.set_gate_override(0, 1, vec![1.0, 1.0, 1.0, 1.0]);
        assert!(p.overrides_gate_at(0, 1).is_none());
    }

    #[test]
    fn overrides_gate_iter_yields_every_inserted_slot() {
        let mut p = make_empty_base();
        p.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
        p.insert_feature(0, 2, vec![0.0, 1.0, 0.0, 0.0], make_meta("b"));
        p.insert_feature(1, 1, vec![0.0, 0.0, 1.0, 0.0], make_meta("c"));
        let mut entries: Vec<(usize, usize)> =
            p.overrides_gate_iter().map(|(l, f, _)| (l, f)).collect();
        entries.sort();
        assert_eq!(entries, vec![(0, 0), (0, 2), (1, 1)]);
    }

    #[test]
    fn overrides_gate_iter_returns_actual_vectors() {
        let mut p = make_empty_base();
        let g = vec![0.5_f32, -0.5, 0.25, -0.25];
        p.insert_feature(0, 0, g.clone(), make_meta("x"));
        let mut found = false;
        for (l, f, vec) in p.overrides_gate_iter() {
            if (l, f) == (0, 0) {
                assert_eq!(vec, g.as_slice());
                found = true;
            }
        }
        assert!(found, "iter should yield the inserted slot");
    }

    #[test]
    fn set_up_vector_round_trip() {
        // Up overrides parallel down overrides — set, read back, verify.
        // Used by INSERT to write the slot's up component when installing
        // a constellation fact (mutation.rs install_compiled_slot port).
        let mut p = make_empty_base();
        let up = vec![0.3_f32, -0.4, 0.5, -0.6];
        p.set_up_vector(0, 1, up.clone());
        assert_eq!(p.up_override_at(0, 1), Some(up.as_slice()));
        // Different slot is unaffected.
        assert!(p.up_override_at(0, 2).is_none());
    }

    #[test]
    fn up_and_down_overrides_are_independent() {
        // INSERT writes both per layer; verifying they don't overwrite
        // each other's storage (separate HashMaps on the base index).
        let mut p = make_empty_base();
        let up = vec![1.0_f32, 0.0, 0.0, 0.0];
        let down = vec![0.0_f32, 1.0, 0.0, 0.0];
        p.set_up_vector(0, 0, up.clone());
        p.set_down_vector(0, 0, down.clone());
        assert_eq!(p.up_override_at(0, 0), Some(up.as_slice()));
        assert_eq!(p.down_override_at(0, 0), Some(down.as_slice()));
    }

    #[test]
    fn up_overrides_iterator_yields_every_slot() {
        let mut p = make_empty_base();
        p.set_up_vector(0, 0, vec![1.0_f32, 0.0, 0.0, 0.0]);
        p.set_up_vector(0, 2, vec![0.0_f32, 1.0, 0.0, 0.0]);
        p.set_up_vector(1, 1, vec![0.0_f32, 0.0, 1.0, 0.0]);
        let mut keys: Vec<(usize, usize)> = p.up_overrides().keys().copied().collect();
        keys.sort();
        assert_eq!(keys, vec![(0, 0), (0, 2), (1, 1)]);
    }

    #[test]
    fn iter_then_set_round_trip_preserves_other_slots() {
        // Simulate what run_refine_pass does: snapshot via iter,
        // mutate one slot via set_gate_override, verify the other
        // slot's gate is unchanged.
        let mut p = make_empty_base();
        let original_a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let original_b = vec![0.0_f32, 1.0, 0.0, 0.0];
        p.insert_feature(0, 0, original_a.clone(), make_meta("a"));
        p.insert_feature(0, 1, original_b.clone(), make_meta("b"));

        // Snapshot.
        let snapshot: Vec<(usize, usize, Vec<f32>)> = p
            .overrides_gate_iter()
            .map(|(l, f, v)| (l, f, v.to_vec()))
            .collect();
        assert_eq!(snapshot.len(), 2);

        // Mutate slot a only.
        p.set_gate_override(0, 0, vec![0.5, 0.5, 0.0, 0.0]);

        assert_eq!(p.overrides_gate_at(0, 0).unwrap(), &[0.5, 0.5, 0.0, 0.0]);
        assert_eq!(p.overrides_gate_at(0, 1).unwrap(), original_b.as_slice());
    }

    // ── Coverage for the 2026-05-16 cache + accessor paths ─────────────

    /// Second `gate_knn` query at the same layer should hit the
    /// `layer_gate_cache` fast path (read-lock branch).
    #[test]
    fn gate_knn_second_query_uses_cache_fast_path() {
        let mut p = make_empty_base();
        p.insert_feature(0, 1, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
        let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
        // First call: builds the cache under a write lock.
        let _ = p.gate_knn(0, &q, 1);
        // Second call: should reach the `g.get(&layer)` branch and
        // skip the rebuild. Result equivalence is the load-bearing
        // assertion; the perf benefit is measured in
        // `larql-server/benches/shard_query.rs`.
        let second = p.gate_knn(0, &q, 1);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].0, 1);
    }

    /// After mutating layer 1, layer 0's cached entry must survive
    /// (per-layer invalidation, 2026-05-16).
    #[test]
    fn cross_layer_mutation_preserves_other_layer_cache() {
        let mut p = make_empty_base();
        p.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("l0"));
        let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
        // Warm layer 0's cache.
        let _ = p.gate_knn(0, &q, 1);
        // Mutate layer 1 — should NOT invalidate layer 0's cache.
        p.insert_feature(1, 0, vec![0.0, 1.0, 0.0, 0.0], make_meta("l1"));
        // Re-query layer 0 — still hits the cached path with the
        // same result.
        let hits = p.gate_knn(0, &q, 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 0);
    }

    /// `update_feature_meta` overwrites only meta, leaves gate alone.
    #[test]
    fn update_feature_meta_replaces_meta_only() {
        let mut p = make_empty_base();
        p.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
        p.update_feature_meta(0, 0, make_meta("b"));
        assert_eq!(p.feature_meta(0, 0).unwrap().top_token, "b");
        // Gate vector untouched.
        assert_eq!(p.overrides_gate_at(0, 0), Some(&[1.0, 0.0, 0.0, 0.0][..]));
    }

    /// `is_overridden` reports `true` for inserted slots, `false`
    /// otherwise. Trivial accessor — pin behavior so a regression in
    /// the storage map shape gets caught.
    #[test]
    fn is_overridden_tracks_inserted_slots() {
        let mut p = make_empty_base();
        assert!(!p.is_overridden(0, 0));
        p.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
        assert!(p.is_overridden(0, 0));
        assert!(!p.is_overridden(0, 1));
        assert!(!p.is_overridden(1, 0));
    }

    /// `base()` / `base_mut()` round-trip the underlying VectorIndex.
    #[test]
    fn base_and_base_mut_expose_the_inner_index() {
        let mut p = make_empty_base();
        assert_eq!(p.base().num_layers, 2);
        assert_eq!(p.base().hidden_size, 4);
        // `base_mut` is used by callers that need to set down/up
        // vectors directly — verify it round-trips.
        let _: &mut VectorIndex = p.base_mut();
    }

    /// `find_free_feature` picks the first overlay-and-base-free
    /// slot.
    #[test]
    fn find_free_feature_picks_first_overlay_free_slot() {
        // Empty base + empty overlay → slot 0 is free.
        let p = make_empty_base();
        assert_eq!(p.find_free_feature(0), Some(0));

        // Overlay claims slot 0 → next free is slot 1.
        let mut p = make_empty_base();
        p.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
        assert_eq!(p.find_free_feature(0), Some(1));

        // Overlay claims 0 and 1, base claims 2 (via metadata) →
        // first preference fails (no slot is *both* base-free AND
        // overlay-free); fallback returns the weakest base-claimed
        // slot that the overlay hasn't taken, but there are no
        // overlay-free base-claimed slots here, so the result is
        // `None`.
        let mut p2 = make_empty_base();
        // Inject base metadata at slot 2 so `feature_meta` returns
        // Some — simulates a populated base slot.
        p2.base_mut().metadata.down_meta[0] = Some(vec![None, None, Some(make_meta("base"))]);
        p2.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
        p2.insert_feature(0, 1, vec![0.0, 1.0, 0.0, 0.0], make_meta("b"));
        // Slot 2 has base metadata but no overlay claim → returned
        // by the fallback (weakest-c_score) loop.
        assert_eq!(p2.find_free_feature(0), Some(2));
    }

    /// `find_free_feature` returns `None` on a layer with zero features.
    #[test]
    fn find_free_feature_returns_none_when_layer_empty() {
        // Use an index where layer 0 has zero features to hit the
        // `n == 0` early return.
        let index = VectorIndex::empty(2, 4);
        let p = PatchedVindex::new(index);
        assert!(p.find_free_feature(0).is_none());
    }
}
