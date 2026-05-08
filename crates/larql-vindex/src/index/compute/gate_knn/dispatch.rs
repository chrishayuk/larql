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
use crate::index::storage::metadata_store::GgufGateLayerManifest;
use crate::index::types::*;

fn read_gguf_manifest_row(
    file: &mut std::fs::File,
    entry: &GgufGateLayerManifest,
    feature: usize,
) -> Option<Vec<f32>> {
    let feature_count = entry.feature_count();
    if feature >= feature_count || entry.cols == 0 || entry.rows == 0 || entry.experts == 0 {
        return None;
    }
    let element_offset = feature.checked_mul(entry.cols)?;
    let byte_offset = gguf_type_byte_offset(entry.tensor_type, element_offset)?;
    let byte_len = gguf_type_byte_len(entry.tensor_type, entry.cols)?;
    let absolute = entry
        .data_offset
        .checked_add(entry.tensor_offset)?
        .checked_add(byte_offset as u64)?;
    use std::io::{Read, Seek};
    file.seek(std::io::SeekFrom::Start(absolute)).ok()?;
    let mut raw = vec![0u8; byte_len];
    file.read_exact(&mut raw).ok()?;
    larql_models::quant::ggml::dequantize(&raw, entry.tensor_type, entry.cols).ok()
}

fn gguf_type_byte_offset(tensor_type: u32, element_offset: usize) -> Option<usize> {
    use larql_models::quant::ggml;
    match tensor_type {
        ggml::TYPE_F32 => element_offset.checked_mul(4),
        ggml::TYPE_F16 | ggml::TYPE_BF16 => element_offset.checked_mul(2),
        ggml::TYPE_Q4_K if element_offset.is_multiple_of(ggml::K_QUANT_BLOCK_ELEMS) => {
            Some(element_offset / ggml::K_QUANT_BLOCK_ELEMS * ggml::Q4_K_BLOCK_BYTES)
        }
        ggml::TYPE_Q6_K if element_offset.is_multiple_of(ggml::K_QUANT_BLOCK_ELEMS) => {
            Some(element_offset / ggml::K_QUANT_BLOCK_ELEMS * ggml::Q6_K_BLOCK_BYTES)
        }
        _ => None,
    }
}

fn gguf_type_byte_len(tensor_type: u32, n_elements: usize) -> Option<usize> {
    larql_models::quant::ggml::tensor_data_size(tensor_type, n_elements).ok()
}

impl VectorIndex {
    fn gate_knn_gguf_manifest(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        let manifest = self.metadata.gguf_gate_manifest.as_ref()?;
        let entry = manifest.layer(layer)?;
        let feature_count = entry.feature_count();
        if entry.cols != self.hidden_size || feature_count == 0 {
            return None;
        }
        let scan_limit = std::env::var("LARQL_GGUF_MANIFEST_GATE_SCAN_FEATURES")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(64)
            .min(feature_count);
        let mut scores = Vec::with_capacity(scan_limit);
        let mut file = std::fs::File::open(&entry.source_file).ok()?;
        for feature in 0..scan_limit {
            let row = read_gguf_manifest_row(&mut file, entry, feature)?;
            let score = row
                .iter()
                .zip(residual.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
            scores.push((feature, score));
        }
        scores.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(top_k);
        Some(scores)
    }

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

        // Fallback: manifest-backed GGUF/Kimi scan. This deliberately scans a
        // bounded feature prefix per layer so browse queries can execute without
        // materializing multi-hundred-GB dense gate tables.
        if let Some(results) = self.gate_knn_gguf_manifest(layer, residual, top_k) {
            return results;
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
        let mmap_slice = if self.gate.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            self.gate.gate_mmap_bytes.as_ref().and_then(|mmap| {
                let slice = self.gate.gate_mmap_slices.get(layer)?;
                if slice.num_features == 0 {
                    return None;
                }
                let byte_offset = slice.float_offset * 4;
                let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
                if byte_end > mmap.len() {
                    return None;
                }
                Some(unsafe {
                    std::slice::from_raw_parts(
                        mmap[byte_offset..byte_end].as_ptr() as *const f32,
                        slice.num_features * self.hidden_size,
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

        if let Some(ref mmap) = self.gate.gate_mmap_bytes {
            if let Some(slice) = self.gate.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 || feat_start >= slice.num_features {
                    return vec![];
                }
                let end = feat_end.min(slice.num_features);
                let bpf = crate::config::dtype::bytes_per_float(self.gate.gate_mmap_dtype);

                // Compute byte range for just this expert's features
                let layer_byte_start = slice.float_offset * bpf;
                let expert_byte_start = layer_byte_start + feat_start * self.hidden_size * bpf;
                let expert_byte_end = layer_byte_start + end * self.hidden_size * bpf;
                let n_features = end - feat_start;

                if expert_byte_end > mmap.len() {
                    return vec![];
                }

                match self.gate.gate_mmap_dtype {
                    crate::config::dtype::StorageDtype::F32 => {
                        let data = unsafe {
                            let ptr =
                                mmap[expert_byte_start..expert_byte_end].as_ptr() as *const f32;
                            std::slice::from_raw_parts(ptr, n_features * self.hidden_size)
                        };
                        let view =
                            ndarray::ArrayView2::from_shape((n_features, self.hidden_size), data)
                                .unwrap();
                        let scores = gemv(&view, residual);
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
                        let view = ndarray::ArrayView2::from_shape(
                            (n_features, self.hidden_size),
                            &floats,
                        )
                        .unwrap();
                        let scores = gemv(&view, residual);
                        let mut hits = Self::top_k_from_scores(&scores, top_k);
                        for hit in &mut hits {
                            hit.0 += feat_start;
                        }
                        return hits;
                    }
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
            if backend.has_q4() {
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
        if !backend.has_q4() {
            return None;
        }
        let q4_data = self.gate_q4_data(layer)?;
        let slice = self.gate.gate_q4_slices.get(layer)?;
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
