//! HNSW lifecycle — enable/disable, lazy + eager build, per-layer +
//! per-(layer, expert) caches, plus the HNSW-backed knn variants
//! consumed by `dispatch.rs`.
//!
//! Lock pattern across all build helpers: brief check under the cache
//! mutex, build the HNSW outside the lock, install only if no other
//! thread raced ahead. A duplicated build is cheaper than a corrupted
//! cache.

use ndarray::{Array1, ArrayView2};

use crate::index::core::VectorIndex;

impl VectorIndex {
    /// Enable HNSW search. Indexes are built lazily on first query per layer.
    ///
    /// `ef_search`: beam width for search (50-200). Higher = better recall, slower.
    pub fn enable_hnsw(&self, ef_search: usize) {
        self.gate
            .hnsw_enabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.gate
            .hnsw_ef_search
            .store(ef_search, std::sync::atomic::Ordering::Relaxed);
    }

    /// Disable HNSW, revert to brute-force matmul.
    pub fn disable_hnsw(&self) {
        self.gate
            .hnsw_enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether HNSW is currently enabled.
    pub fn is_hnsw_enabled(&self) -> bool {
        self.gate
            .hnsw_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get the gate vector matrix for a layer as owned contiguous f32.
    /// Used by HNSW build which needs owned data.
    fn gate_matrix_f32(&self, layer: usize) -> Option<(Vec<f32>, usize)> {
        let gate = self.resolve_gate(layer)?;
        Some((gate.data, gate.num_features))
    }

    /// Build a fresh HNSW for `layer` *without* holding the cache lock.
    /// Returns `None` when the layer has no gate data (caller decides
    /// what to do). Two callers race-safely concurrent on different
    /// layers since this never touches `hnsw_cache`.
    fn build_hnsw_layer(&self, layer: usize) -> Option<super::super::hnsw::HnswLayer> {
        let (data, num_features) = self.gate_matrix_f32(layer)?;
        let view = ArrayView2::from_shape((num_features, self.hidden_size), &data).unwrap();
        Some(super::super::hnsw::HnswLayer::build(&view, 8, 32))
    }

    /// Build an HNSW for a single `(layer, expert_id)` unit — i.e. the gate
    /// vectors for one expert's intermediate slice.  Index covers vectors
    /// `feat_start..feat_end` in the layer's global feature space; entries
    /// returned from the HNSW search are still in the local (0-based) range
    /// and the caller offsets them back to global indices.
    ///
    /// Returns `None` when the layer has no gate data or the slice is empty.
    fn build_hnsw_unit_at(
        &self,
        layer: usize,
        feat_start: usize,
        feat_end: usize,
    ) -> Option<super::super::hnsw::HnswLayer> {
        let (data, num_features) = self.gate_matrix_f32(layer)?;
        let end = feat_end.min(num_features);
        if feat_start >= end {
            return None;
        }
        let view = ArrayView2::from_shape((num_features, self.hidden_size), &data).ok()?;
        let slice = view.slice(ndarray::s![feat_start..end, ..]);
        // Smaller `m` and `ef_construction` for the per-expert case — at
        // ~704 vectors the layer-level (8, 32) is overkill; (6, 16) builds
        // ~3× faster with comparable recall on this size class.
        Some(super::super::hnsw::HnswLayer::build(&slice, 6, 16))
    }

    /// Get-or-build the per-(layer, expert) HNSW unit, race-safely.
    ///
    /// Lock pattern mirrors `get_or_build_hnsw`: brief check under the
    /// mutex, build outside the lock, install only if no other thread
    /// raced ahead.
    fn get_or_build_hnsw_unit(&self, layer: usize, feat_start: usize, feat_end: usize) -> bool {
        let key = (layer, feat_start);
        {
            let cache = self.gate.hnsw_unit_cache.lock().unwrap();
            if cache.contains_key(&key) {
                return true;
            }
        }
        let Some(hnsw) = self.build_hnsw_unit_at(layer, feat_start, feat_end) else {
            return false;
        };
        let mut cache = self.gate.hnsw_unit_cache.lock().unwrap();
        cache.entry(key).or_insert(hnsw);
        true
    }

    /// Eager-build per-(layer, expert) HNSW units in parallel.  Equivalent of
    /// [`Self::warmup_hnsw_all_layers`] for the fine-grained shard layout —
    /// caller passes `(layer, feat_start, feat_end)` triples for every unit
    /// the shard owns.  Returns the number of units actually built (skipping
    /// already-cached entries and empty slices).
    pub fn warmup_hnsw_units(&self, units: &[(usize, usize, usize)]) -> usize {
        use rayon::prelude::*;
        // Snapshot which units still need building under the lock.
        let to_build: Vec<(usize, usize, usize)> = {
            let cache = self.gate.hnsw_unit_cache.lock().unwrap();
            units
                .iter()
                .filter(|(l, fs, _)| !cache.contains_key(&(*l, *fs)))
                .copied()
                .collect()
        };
        if to_build.is_empty() {
            return 0;
        }
        let built: Vec<((usize, usize), super::super::hnsw::HnswLayer)> = to_build
            .par_iter()
            .filter_map(|&(l, fs, fe)| self.build_hnsw_unit_at(l, fs, fe).map(|h| ((l, fs), h)))
            .collect();
        let n = built.len();
        let mut cache = self.gate.hnsw_unit_cache.lock().unwrap();
        for (key, hnsw) in built {
            cache.entry(key).or_insert(hnsw);
        }
        n
    }

    /// Atomically install `hnsw` at `layer` if no other thread already
    /// did. A concurrent racer's index is dropped — the loss is one
    /// duplicated build, not a corrupted cache.
    fn install_hnsw_layer(&self, layer: usize, hnsw: super::super::hnsw::HnswLayer) {
        let mut cache = self.gate.hnsw_cache.lock().unwrap();
        if cache.len() <= layer {
            cache.resize_with(layer + 1, || None);
        }
        if cache[layer].is_none() {
            cache[layer] = Some(hnsw);
        }
    }

    /// Get or build the HNSW index for a layer (lazy). Holds the cache
    /// lock only briefly at check + install — the ~76 ms build itself
    /// runs lock-free, so concurrent KNN queries on other layers don't
    /// block on this layer's build.
    fn get_or_build_hnsw(&self, layer: usize) -> bool {
        {
            let cache = self.gate.hnsw_cache.lock().unwrap();
            if cache.get(layer).and_then(|s| s.as_ref()).is_some() {
                return true;
            }
        }
        let Some(hnsw) = self.build_hnsw_layer(layer) else {
            return false;
        };
        self.install_hnsw_layer(layer, hnsw);
        true
    }

    /// Eager-build HNSW for every layer, in parallel. One-shot startup
    /// helper for grid servers and interp pipelines that will query all
    /// layers — single call replaces N × ~76 ms lazy builds with one
    /// parallel batch (≈ 76 ms ÷ N_threads on the slowest layer's bound).
    /// Already-built layers are skipped.
    ///
    /// Holds the cache lock only at the snapshot + install boundaries;
    /// the per-layer build runs lock-free across rayon's pool. Memory
    /// note — each parallel build clones its layer's gate data
    /// (`gate_matrix_f32`), so peak transient RSS is ≈
    /// `min(num_layers, num_threads) × layer_gate_bytes`. Shrink with
    /// `rayon::ThreadPoolBuilder::num_threads(...).build_scoped(...)`
    /// if you need to bound it.
    pub fn warmup_hnsw_all_layers(&self) {
        use rayon::prelude::*;
        let num_layers = self.num_layers;
        let to_build: Vec<usize> = {
            let cache = self.gate.hnsw_cache.lock().unwrap();
            (0..num_layers)
                .filter(|&l| cache.get(l).and_then(|s| s.as_ref()).is_none())
                .collect()
        };
        if to_build.is_empty() {
            return;
        }
        let built: Vec<(usize, super::super::hnsw::HnswLayer)> = to_build
            .par_iter()
            .filter_map(|&l| self.build_hnsw_layer(l).map(|h| (l, h)))
            .collect();
        for (layer, hnsw) in built {
            self.install_hnsw_layer(layer, hnsw);
        }
    }

    /// Gate KNN via HNSW: graph search instead of brute-force matmul.
    ///
    /// Re-rank uses a zero-copy view onto the gate data when the layer
    /// is f32-mmap'd; only the f16-mmap and heap paths fall back to
    /// `gate_matrix_f32` (which clones). Dense 4B with f32 mmap pays
    /// only the search cost; the 100 MB-per-query clone is gone.
    ///
    /// **Ranking semantics.** The brute-force `gate_knn` path returns
    /// the top-K features by |dot| (absolute magnitude — matches the
    /// gate-activation strength regardless of sign). HNSW's internal
    /// rank is by signed dot, which would systematically drop
    /// large-negative features. We oversample HNSW (4× top_k) and then
    /// re-rank by abs at the seam to match the brute path's semantics.
    pub(super) fn gate_knn_hnsw(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        if !self.get_or_build_hnsw(layer) {
            return None;
        }

        let ef = self
            .gate
            .hnsw_ef_search
            .load(std::sync::atomic::Ordering::Relaxed);
        // Oversample so the abs-rank seam below has signed candidates
        // from both tails to choose from.
        let hnsw_k = top_k.saturating_mul(4).max(top_k);
        let cache = self.gate.hnsw_cache.lock().unwrap();
        let hnsw = cache[layer].as_ref()?;

        let mut candidates = if let (true, Some(mmap)) = (
            self.gate.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32,
            self.gate.gate_mmap_bytes.as_ref(),
        ) {
            // Zero-copy view onto f32-mmap.
            let slice = self.gate.gate_mmap_slices.get(layer)?;
            if slice.num_features == 0 {
                return None;
            }
            let byte_offset = slice.float_offset * 4;
            let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
            if byte_end > mmap.len() {
                return None;
            }
            let data = unsafe {
                let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, slice.num_features * self.hidden_size)
            };
            let view =
                ArrayView2::from_shape((slice.num_features, self.hidden_size), data).unwrap();
            hnsw.search(&view, residual, hnsw_k, ef)
        } else {
            // Fallback (f16 mmap or heap): owned clone.
            let (data, num_features) = self.gate_matrix_f32(layer)?;
            let view = ArrayView2::from_shape((num_features, self.hidden_size), &data).unwrap();
            hnsw.search(&view, residual, hnsw_k, ef)
        };

        // Re-rank by |dot| to match brute-force semantics.
        candidates.sort_unstable_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(top_k);
        Some(candidates)
    }

    /// Per-(layer, expert) HNSW search.  Returns `None` when the unit index
    /// can't be built (empty slice, no gate data) or when gate matrix decode
    /// fails — caller falls back to the brute paths in `gate_knn_expert`.
    ///
    /// Same `|dot|` ranking semantics as `gate_knn_hnsw` (oversample 4×, then
    /// re-rank by absolute value).  Indices in the returned vector are in
    /// **global** feature space — `feat_start` is added back so the caller
    /// can use them interchangeably with the brute path's output.
    pub(super) fn gate_knn_expert_hnsw(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        feat_start: usize,
        feat_end: usize,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        if !self.get_or_build_hnsw_unit(layer, feat_start, feat_end) {
            return None;
        }
        let ef = self
            .gate
            .hnsw_ef_search
            .load(std::sync::atomic::Ordering::Relaxed);
        let hnsw_k = top_k.saturating_mul(4).max(top_k);

        // Need a view onto the expert's slice for re-ranking.  Cheapest path
        // is the f32-mmap zero-copy slice; otherwise fall back to a
        // gate_matrix_f32 clone and slice into it.
        let (data, num_features) = self.gate_matrix_f32(layer)?;
        let view = ArrayView2::from_shape((num_features, self.hidden_size), &data).ok()?;
        let end = feat_end.min(num_features);
        if feat_start >= end {
            return None;
        }
        let slice = view.slice(ndarray::s![feat_start..end, ..]);

        let cache = self.gate.hnsw_unit_cache.lock().unwrap();
        let hnsw = cache.get(&(layer, feat_start))?;
        let mut candidates = hnsw.search(&slice, residual, hnsw_k, ef);
        drop(cache);

        // Re-rank by |dot| to match brute-force semantics.
        candidates.sort_unstable_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(top_k);
        // HNSW returned indices in slice-local space (0..end-feat_start).
        // Offset to global feature indices.
        for hit in &mut candidates {
            hit.0 += feat_start;
        }
        Some(candidates)
    }
}
