//! Top-level KNN entry points + the batched matmul gate_walk.
//!
//! Every public KNN call lands here. The methods pick between BLAS,
//! HNSW (`hnsw_lifecycle.rs`), GPU full-batch (`scores_batch.rs`), and
//! Q4 backend matvec, then funnel through `Self::top_k_from_scores`
//! (`mod.rs`) for the K-with-largest-|val| extraction.

use ndarray::{Array1, Array2, ArrayView2};

use super::top_k_by_abs;
use crate::index::core::VectorIndex;
use crate::index::storage::gate_store::{gate_matmul, gemv};
use crate::index::storage::vindex_storage::VindexStorage;
use crate::index::types::*;

impl VectorIndex {
    /// Gate KNN: find the top-K features at a layer whose gate vectors have
    /// the highest dot product with the input residual. Uses BLAS matmul.
    ///
    /// In mmap mode, slices directly from the mmap'd file — zero heap allocation.
    /// Returns (feature_index, dot_product) sorted by absolute magnitude descending.
    pub fn gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // HNSW path
        if self
            .gate
            .hnsw_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            if let Some(results) = self.gate_knn_hnsw(layer, residual, top_k) {
                return results;
            }
        }

        // Fast path: f32 mmap zero-copy (no allocation, no clone)
        if let Some(scores) = self.gate_knn_mmap_fast(layer, residual) {
            return Self::top_k_from_scores(&scores, top_k);
        }

        // Fallback: resolve_gate (copies data for heap/f16 paths)
        let gate = match self.resolve_gate(layer) {
            Some(g) => g,
            None => return vec![],
        };
        let view = gate.view(self.hidden_size);
        let scores = gemv(&view, residual);
        Self::top_k_from_scores(&scores, top_k)
    }

    /// Batched gate walk: scores all features via a single BLAS `gemv`, then
    /// extracts the top-K. Despite the name, this is batched matrix-vector —
    /// see [`Self::gate_walk_pure`] for a true per-feature implementation.
    pub fn gate_walk(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        let num_features = self.num_features(layer);
        if num_features == 0 {
            return None;
        }

        // Get gate data as contiguous f32 (from mmap or warmed cache)
        let gate_data: &[f32];
        let _owned: Vec<f32>;

        // Try zero-copy f32 mmap first
        let mmap_slice = if self.storage.gate_dtype() == crate::config::dtype::StorageDtype::F32 {
            self.storage.gate_layer_view(layer).and_then(|view| {
                if view.slice.num_features == 0 {
                    return None;
                }
                let byte_offset = view.slice.float_offset * 4;
                let byte_end = byte_offset + view.slice.num_features * self.hidden_size * 4;
                let mmap: &[u8] = view.bytes.as_ref();
                if byte_end > mmap.len() {
                    return None;
                }
                Some(unsafe {
                    std::slice::from_raw_parts(
                        mmap[byte_offset..byte_end].as_ptr() as *const f32,
                        view.slice.num_features * self.hidden_size,
                    )
                })
            })
        } else {
            None
        };

        if let Some(data) = mmap_slice {
            gate_data = data;
        } else {
            // Fallback: resolve gate (may clone)
            let gate = self.resolve_gate(layer)?;
            _owned = gate.data;
            gate_data = &_owned;
        }

        let hidden = self.hidden_size;

        // Single BLAS gemv: gate[N, hidden] × residual[hidden] → scores[N].
        let gate_view = ArrayView2::from_shape((num_features, hidden), gate_data).unwrap();
        let scores = gemv(&gate_view, residual);
        Some(Self::top_k_from_scores(&scores, top_k))
    }

    /// Gate KNN within a specific feature range (for MoE expert-scoped queries).
    /// Only computes dot products for features [feat_start..feat_end].
    /// Returns (global_feature_index, score) pairs.
    pub fn gate_knn_expert(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        feat_start: usize,
        feat_end: usize,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // HNSW-on-unit fast path: when the master toggle is on, search the
        // per-(layer, expert) HNSW (lazily built on first hit).  At ~704
        // vectors per Gemma-4-26B-A4B expert this is sub-µs vs ~50µs brute.
        // Falls through to the brute paths below if the index can't be
        // built (empty slice, no gate data) or if the toggle is off.
        if self
            .gate
            .hnsw_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            if let Some(hits) =
                self.gate_knn_expert_hnsw(layer, residual, feat_start, feat_end, top_k)
            {
                return hits;
            }
        }

        // If promoted to heap, use heap path
        if let Some(Some(ref matrix)) = self.gate.gate_vectors.get(layer) {
            let end = feat_end.min(matrix.shape()[0]);
            if feat_start >= end {
                return vec![];
            }
            let slice = matrix.slice(ndarray::s![feat_start..end, ..]);
            let scores = gemv(&slice, residual);
            let mut hits = Self::top_k_from_scores(&scores, top_k);
            for hit in &mut hits {
                hit.0 += feat_start;
            }
            return hits;
        }

        if let Some(view) = self.storage.gate_layer_view(layer) {
            if view.slice.num_features == 0 || feat_start >= view.slice.num_features {
                return vec![];
            }
            let end = feat_end.min(view.slice.num_features);
            let bpf = crate::config::dtype::bytes_per_float(view.dtype);

            // Compute byte range for just this expert's features
            let layer_byte_start = view.slice.float_offset * bpf;
            let expert_byte_start = layer_byte_start + feat_start * self.hidden_size * bpf;
            let expert_byte_end = layer_byte_start + end * self.hidden_size * bpf;
            let n_features = end - feat_start;
            let mmap: &[u8] = view.bytes.as_ref();

            if expert_byte_end > mmap.len() {
                return vec![];
            }

            match view.dtype {
                crate::config::dtype::StorageDtype::F32 => {
                    let data = unsafe {
                        let ptr = mmap[expert_byte_start..expert_byte_end].as_ptr() as *const f32;
                        std::slice::from_raw_parts(ptr, n_features * self.hidden_size)
                    };
                    let v = ndarray::ArrayView2::from_shape((n_features, self.hidden_size), data)
                        .unwrap();
                    let scores = gemv(&v, residual);
                    let mut hits = Self::top_k_from_scores(&scores, top_k);
                    // Offset indices to global feature space
                    for hit in &mut hits {
                        hit.0 += feat_start;
                    }
                    return hits;
                }
                crate::config::dtype::StorageDtype::F16 => {
                    let raw = &mmap[expert_byte_start..expert_byte_end];
                    let floats = larql_models::quant::half::decode_f16(raw);
                    let v =
                        ndarray::ArrayView2::from_shape((n_features, self.hidden_size), &floats)
                            .unwrap();
                    let scores = gemv(&v, residual);
                    let mut hits = Self::top_k_from_scores(&scores, top_k);
                    for hit in &mut hits {
                        hit.0 += feat_start;
                    }
                    return hits;
                }
            }
        }
        // Fallback: full KNN filtered (slower)
        self.gate_knn(layer, residual, top_k * 10)
            .into_iter()
            .filter(|(f, _)| *f >= feat_start && *f < feat_end)
            .take(top_k)
            .collect()
    }

    /// Full walk: gate KNN at each layer, annotated with down token metadata.
    pub fn walk(&self, residual: &Array1<f32>, layers: &[usize], top_k: usize) -> WalkTrace {
        let mut trace_layers = Vec::with_capacity(layers.len());

        for &layer in layers {
            let hits = self.gate_knn(layer, residual, top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.feature_meta(layer, feature)?;
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

    /// Batched gate KNN: compute scores for ALL sequence positions in one BLAS gemm.
    ///
    /// Input: x is [seq_len, hidden]. Computes gate_vectors @ x^T = [features, seq_len].
    /// Returns the union of per-position top-K feature indices (sorted).
    /// One gemm replaces seq_len separate gemv calls.
    ///
    /// Per-position top-K extraction runs in parallel via rayon when
    /// `seq_len >= PARALLEL_TOPK_THRESHOLD` (16 — below that the rayon
    /// scheduling overhead matches or exceeds the per-position savings;
    /// at seq_len 64 the parallel branch saves ~7 % and at seq_len 256
    /// it saves ~24 % on Gemma-shape gates).
    pub fn gate_knn_batch(&self, layer: usize, x: &Array2<f32>, top_k: usize) -> Vec<usize> {
        let seq_len = x.shape()[0];
        if seq_len == 0 {
            return vec![];
        }

        // Fast path: zero-copy f32 mmap/warmed
        let scores_2d = if let Some(s) = self.gate_scores_2d_fast(layer, x) {
            s
        } else if let Some(gate) = self.resolve_gate(layer) {
            gate_matmul(&gate.view(self.hidden_size), &x.view())
        } else {
            return vec![];
        };

        // scores_2d is [num_features, seq_len].
        // For each position, take top-K features; union the indices.
        let num_features = scores_2d.shape()[0];
        let k = top_k.min(num_features);

        const PARALLEL_TOPK_THRESHOLD: usize = 16;
        let position_hits: Vec<Vec<usize>> = if seq_len >= PARALLEL_TOPK_THRESHOLD {
            use rayon::prelude::*;
            (0..seq_len)
                .into_par_iter()
                .map(|s| {
                    top_k_by_abs(scores_2d.column(s).iter().copied(), k)
                        .into_iter()
                        .map(|(idx, _)| idx)
                        .collect()
                })
                .collect()
        } else {
            (0..seq_len)
                .map(|s| {
                    top_k_by_abs(scores_2d.column(s).iter().copied(), k)
                        .into_iter()
                        .map(|(idx, _)| idx)
                        .collect()
                })
                .collect()
        };

        let mut feature_set = std::collections::BTreeSet::new();
        for hits in position_hits {
            feature_set.extend(hits);
        }
        feature_set.into_iter().collect()
    }

    /// Adaptive gate KNN — automatically picks the fastest path per layer.
    ///
    /// Dispatch order:
    /// 1. Pinned Q4 → backend.q4_matvec (pre-loaded, no page faults)
    /// 2. Mmap Q4 → backend.q4_matvec (paged on demand)
    /// 3. f32 mmap/heap → BLAS brute-force (fallback)
    ///
    /// The residency manager tracks which layers are pinned.
    /// More memory budget → more pinned layers → faster walk.
    pub fn gate_knn_adaptive(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        residency: &mut crate::index::storage::residency::ResidencyManager,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(usize, f32)> {
        residency.record_access(layer);

        // 1. Pinned Q4 (fastest — data already in RAM)
        if let Some(q4_data) = residency.pinned_q4(layer) {
            if backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
                let x = residual.as_slice().unwrap();
                let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x);
                let num_features = self.num_features(layer);
                if let Some(scores_vec) =
                    backend.q4_matvec(q4_data, &q8_x, &q8_scales, num_features, self.hidden_size)
                {
                    return Self::top_k_from_scores(&Array1::from_vec(scores_vec), top_k);
                }
            }
        }

        // 2. Mmap Q4 (Q4 file loaded but not pinned — OS pages on demand)
        if let Some(hits) = self.gate_knn_q4(layer, residual, top_k, backend) {
            return hits;
        }

        // 3. f32 brute-force (fallback)
        self.gate_knn(layer, residual, top_k)
    }

    /// Gate KNN via Q4 matvec — scored by a ComputeBackend.
    ///
    /// The vindex provides the raw Q4 data. The backend scores it.
    /// Works with any backend: CPU C kernel, Metal GPU, CUDA, WASM.
    ///
    /// Returns None if Q4 gate data isn't loaded or backend doesn't support Q4.
    pub fn gate_knn_q4(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(usize, f32)>> {
        if !backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
            return None;
        }
        let q4_data = self.gate_q4_data(layer)?;
        let slice = self.storage.gate_q4_layer_slice(layer)?;
        if slice.num_features == 0 {
            return None;
        }

        let (q8_x, q8_scales) =
            larql_compute::cpu::q4::quantize_to_q8(residual.as_slice().unwrap());
        let scores_vec = backend.q4_matvec(
            q4_data,
            &q8_x,
            &q8_scales,
            slice.num_features,
            self.hidden_size,
        )?;

        let scores = Array1::from_vec(scores_vec);
        Some(Self::top_k_from_scores(&scores, top_k))
    }
}

#[cfg(test)]
mod tests {
    //! Inline coverage of the dispatch entry points. Exercises the
    //! heap-only and empty-index branches end-to-end so the fast-path
    //! short-circuits and fallback paths each get hit at least once.
    //! The mmap-mode hot paths and HNSW lifecycle are covered by the
    //! integration tests in `tests/compute_storage_regressions.rs`
    //! and the `gate_cache_lru_tests` in `gate_store.rs`.

    use super::*;
    use crate::index::FeatureMeta;
    use ndarray::array;

    /// Simple heap-mode index: 3 features at layer 0, hidden=4.
    /// Each row is a known direction so we can predict KNN hits.
    fn heap_idx() -> VectorIndex {
        let gate = ndarray::Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, // f0: dot with [1,2,3,4] = 1
                0.0, 2.0, 0.0, 0.0, // f1: dot = 4
                0.0, 0.0, 3.0, 0.0, // f2: dot = 9 (winner)
            ],
        )
        .unwrap();
        VectorIndex::new(vec![Some(gate)], vec![None], 1, 4)
    }

    // ── gate_knn ─────────────────────────────────────────────────

    #[test]
    fn gate_knn_heap_returns_top_k_by_dot_product() {
        let v = heap_idx();
        let q = array![1.0, 2.0, 3.0, 4.0];
        let hits = v.gate_knn(0, &q, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 2, "feature 2 has highest dot product");
    }

    #[test]
    fn gate_knn_empty_index_returns_empty() {
        let v = VectorIndex::empty(1, 4);
        let q = array![1.0, 1.0, 1.0, 1.0];
        assert!(v.gate_knn(0, &q, 5).is_empty());
    }

    // ── gate_walk ────────────────────────────────────────────────

    #[test]
    fn gate_walk_heap_returns_top_k() {
        let v = heap_idx();
        let q = array![1.0, 2.0, 3.0, 4.0];
        let hits = v.gate_walk(0, &q, 2).expect("heap path");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 2);
    }

    #[test]
    fn gate_walk_returns_none_when_num_features_zero() {
        let v = VectorIndex::empty(1, 4);
        assert!(v.gate_walk(0, &array![1.0, 2.0, 3.0, 4.0], 1).is_none());
    }

    // ── gate_knn_expert ──────────────────────────────────────────

    #[test]
    fn gate_knn_expert_heap_returns_global_indices() {
        let v = heap_idx();
        let q = array![1.0, 2.0, 3.0, 4.0];
        // Restrict to features [1, 3): top-1 should be feature 2.
        let hits = v.gate_knn_expert(0, &q, 1, 3, 2);
        assert!(!hits.is_empty());
        // Indices are global — the heap-mode path offsets correctly.
        assert!(hits.iter().all(|(f, _)| *f >= 1 && *f < 3));
    }

    #[test]
    fn gate_knn_expert_invalid_range_returns_empty() {
        let v = heap_idx();
        let q = array![1.0, 1.0, 1.0, 1.0];
        // feat_start >= matrix.shape()[0] → empty.
        assert!(v.gate_knn_expert(0, &q, 99, 100, 5).is_empty());
    }

    // ── walk ─────────────────────────────────────────────────────

    #[test]
    fn walk_enriches_hits_with_feature_metadata() {
        // Build a heap-mode index and seed metadata at one feature.
        let mut v = heap_idx();
        v.metadata.down_meta[0] = Some(vec![
            None,
            None,
            Some(FeatureMeta {
                top_token: "test".into(),
                top_token_id: 7,
                c_score: 0.9,
                top_k: vec![],
            }),
        ]);
        let q = array![1.0, 2.0, 3.0, 4.0];
        let trace = v.walk(&q, &[0], 3);
        assert_eq!(trace.layers.len(), 1);
        let (layer, hits) = &trace.layers[0];
        assert_eq!(*layer, 0);
        assert_eq!(hits.len(), 1, "only feature 2 has metadata");
        assert_eq!(hits[0].feature, 2);
        assert_eq!(hits[0].meta.top_token_id, 7);
    }

    // ── gate_knn_batch ───────────────────────────────────────────

    #[test]
    fn gate_knn_batch_empty_seq_returns_empty() {
        let v = heap_idx();
        let x = ndarray::Array2::<f32>::zeros((0, 4));
        assert!(v.gate_knn_batch(0, &x, 1).is_empty());
    }

    #[test]
    fn gate_knn_batch_single_position_hits_top_features() {
        let v = heap_idx();
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let hits = v.gate_knn_batch(0, &x, 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0], 2, "feature 2 has highest dot product");
    }

    /// `gate_knn_batch` switches to the rayon-parallel branch at
    /// `seq_len >= PARALLEL_TOPK_THRESHOLD` (16). Use seq_len = 20
    /// to exercise that path.
    #[test]
    fn gate_knn_batch_parallel_branch_unions_per_position_hits() {
        let v = heap_idx();
        let mut x = ndarray::Array2::<f32>::zeros((20, 4));
        // Each position points at a different feature so the union
        // across positions covers all 3.
        for s in 0..20 {
            let f = s % 3;
            x[[s, f]] = 1.0;
        }
        let hits = v.gate_knn_batch(0, &x, 1);
        assert_eq!(hits.len(), 3, "union should be all 3 features");
        assert!(hits.iter().all(|&i| i < 3));
    }

    #[test]
    fn gate_knn_batch_resolve_gate_fallback_returns_empty_on_unowned_layer() {
        let v = VectorIndex::empty(1, 4);
        // Layer 0 has no gate vectors → fast-path fails, resolve_gate
        // also fails → returns empty.
        let x = array![[1.0, 1.0, 1.0, 1.0]];
        assert!(v.gate_knn_batch(0, &x, 1).is_empty());
    }

    // ── gate_knn_q4 ──────────────────────────────────────────────

    #[test]
    fn gate_knn_q4_returns_none_when_backend_lacks_q4() {
        // CpuBackend supports Q4 — to exercise the early-return we
        // need a backend without Q4 support. The fake-backend
        // pattern in `compute_storage_regressions.rs` does this; for
        // this inline test we exercise the no-Q4-data path instead:
        // an empty index has no Q4 file, so `gate_q4_data` returns
        // None and the function returns None.
        let v = heap_idx();
        let q = array![1.0, 2.0, 3.0, 4.0];
        let cpu = larql_compute::CpuBackend;
        assert!(v.gate_knn_q4(0, &q, 5, &cpu).is_none());
    }

    // ── gate_knn_adaptive ────────────────────────────────────────

    #[test]
    fn gate_knn_adaptive_heap_falls_through_to_brute_force() {
        let v = heap_idx();
        let q = array![1.0, 2.0, 3.0, 4.0];
        let mut residency =
            crate::index::storage::residency::ResidencyManager::new(0, 1, 4, vec![3]);
        let cpu = larql_compute::CpuBackend;
        let hits = v.gate_knn_adaptive(0, &q, 2, &mut residency, &cpu);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 2);
    }
}
