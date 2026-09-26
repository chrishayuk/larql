//! Full-batch score computation feeding the dispatch entry points.
//!
//! `gate_scores_batch` is the public API used by inference for the
//! seq_len-wide gate matmul; `gate_scores_batch_backend` adds the GPU
//! gemv fast path for single-row decode. The two private helpers
//! (`gate_scores_2d_gpu`, `gate_scores_2d_fast`) own the zero-copy
//! mmap/warmed slicing logic and the f16 lazy-decode cache.

use ndarray::{Array2, ArrayView2};

use crate::index::core::VectorIndex;
use crate::index::storage::gate_store::{gate_gemv_gpu, gate_matmul};
use crate::index::storage::vindex_storage::VindexStorage;

impl VectorIndex {
    /// Compute gate scores for all features × all positions in one BLAS gemm.
    /// Returns [seq_len, intermediate] matrix = x @ gate_vectors^T.
    /// These scores are the gate projections — the same as x @ W_gate.T.
    pub fn gate_scores_batch(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        self.gate_scores_batch_backend(layer, x, None)
    }

    /// Backend-aware gate scores. When `backend` is present and `x` is
    /// a single row (seq_len == 1), route through `f32_gemv` — the
    /// same row-per-simdgroup path that closed lm_head. On Gemma 4 31B
    /// decode (hidden = 5376, ~18 K features, 60 layers) the CPU-BLAS
    /// path clocks ~4.3 ms/layer × 60 = 258 ms/token = 60 % of decode.
    /// Metal f32_gemv was measured at ~1 ms/layer on the lm_head of
    /// similar shape, so the upside is ~200 ms/token.
    pub fn gate_scores_batch_backend(
        &self,
        layer: usize,
        x: &Array2<f32>,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Array2<f32>> {
        if x.shape()[0] == 0 {
            return None;
        }

        // Metal gemv fast path (decode / single-row prefill).
        if let Some(be) = backend {
            if x.shape()[0] == 1 {
                if let Some(scores_2d) = self.gate_scores_2d_gpu(layer, x, be) {
                    return Some(scores_2d.t().to_owned());
                }
            }
        }

        // BLAS paths — warmed f32 / mmap f32 / lazy-decoded f16.
        let scores_2d = if let Some(s) = self.gate_scores_2d_fast(layer, x) {
            s
        } else {
            let gate = self.resolve_gate(layer)?;
            gate_matmul(&gate.view(self.hidden_size), &x.view())
        };
        Some(scores_2d.t().to_owned())
    }

    /// Zero-copy GPU gate scores for f32 mmap/warmed, single-row `x`.
    /// Matches `gate_scores_2d_fast` shape contract: returns [N, 1].
    fn gate_scores_2d_gpu(
        &self,
        layer: usize,
        x: &Array2<f32>,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        // Warmed cache (f32 heap).
        {
            let warmed = self.gate.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self
                    .storage
                    .gate_layer_slice(layer)
                    .map(|s| s.num_features)
                    .unwrap_or(0);
                if nf > 0 {
                    let view =
                        ArrayView2::from_shape((nf, self.hidden_size), data.as_slice()).unwrap();
                    if let Some(scores) = gate_gemv_gpu(&view, &x.view(), backend) {
                        return Some(scores);
                    }
                }
            }
        }
        // f32 mmap (zero-copy, the production path for f32 gate vectors).
        if self.storage.gate_dtype() == crate::config::dtype::StorageDtype::F32 {
            if let Some(view) = self.storage.gate_layer_view(layer) {
                if view.slice.num_features == 0 {
                    return None;
                }
                let byte_offset = view.slice.float_offset * 4;
                let byte_end = byte_offset + view.slice.num_features * self.hidden_size * 4;
                let mmap: &[u8] = view.bytes.as_ref();
                if byte_end > mmap.len() {
                    return None;
                }
                let data = unsafe {
                    let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                    std::slice::from_raw_parts(ptr, view.slice.num_features * self.hidden_size)
                };
                let arr = ArrayView2::from_shape((view.slice.num_features, self.hidden_size), data)
                    .unwrap();
                if let Some(scores) = gate_gemv_gpu(&arr, &x.view(), backend) {
                    return Some(scores);
                }
            }
        }
        // f16 mmap: zero-copy pass of raw f16 bytes to Metal's f16_gemv
        // shader, skipping the f16→f32 decode cache entirely. On 31B with
        // an ~18 K × 5376 gate matrix (387 MB f32, 194 MB f16) halving
        // the memory bandwidth is the difference between hitting the
        // CPU-BLAS ceiling and going faster on Metal.
        if self.storage.gate_dtype() == crate::config::dtype::StorageDtype::F16 && x.shape()[0] == 1
        {
            let view = self.storage.gate_layer_view(layer)?;
            if view.slice.num_features == 0 {
                return None;
            }
            let mmap: &[u8] = view.bytes.as_ref();
            let byte_offset = view.slice.float_offset * 2;
            let byte_end = byte_offset + view.slice.num_features * self.hidden_size * 2;
            if byte_end <= mmap.len() {
                let raw = &mmap[byte_offset..byte_end];
                let x_row = x.row(0);
                if let Some(x_slice) = x_row.as_slice() {
                    if let Some(scores) = backend.f16_gemv_force(
                        raw,
                        x_slice,
                        view.slice.num_features,
                        self.hidden_size,
                    ) {
                        return Array2::from_shape_vec((view.slice.num_features, 1), scores).ok();
                    }
                }
            }
        }
        None
    }

    /// Zero-copy batch gate scores for f32 mmap/warmed — returns [features, seq].
    pub(super) fn gate_scores_2d_fast(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        // Warmed cache
        {
            let warmed = self.gate.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self
                    .storage
                    .gate_layer_slice(layer)
                    .map(|s| s.num_features)
                    .unwrap_or(0);
                if nf > 0 {
                    let view =
                        ArrayView2::from_shape((nf, self.hidden_size), data.as_slice()).unwrap();
                    return Some(gate_matmul(&view, &x.view()));
                }
            }
        }
        // f32 mmap
        if self.storage.gate_dtype() == crate::config::dtype::StorageDtype::F32 {
            if let Some(view) = self.storage.gate_layer_view(layer) {
                if view.slice.num_features == 0 {
                    return None;
                }
                let byte_offset = view.slice.float_offset * 4;
                let byte_end = byte_offset + view.slice.num_features * self.hidden_size * 4;
                let mmap: &[u8] = view.bytes.as_ref();
                if byte_end > mmap.len() {
                    return None;
                }
                let data = unsafe {
                    let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                    std::slice::from_raw_parts(ptr, view.slice.num_features * self.hidden_size)
                };
                let arr = ArrayView2::from_shape((view.slice.num_features, self.hidden_size), data)
                    .unwrap();
                return Some(gate_matmul(&arr, &x.view()));
            }
        }
        // f16 mmap — lazy decode into cache, then borrow (no per-call clone).
        // Holding the Mutex for the matmul is fine: forward passes are serial
        // per-layer, and this replaces a 462MB clone with a direct view.
        if self.storage.gate_dtype() == crate::config::dtype::StorageDtype::F16 {
            let view = self.storage.gate_layer_view(layer)?;
            if view.slice.num_features == 0 {
                return None;
            }
            let mmap: &[u8] = view.bytes.as_ref();
            // Take an `Arc` handle and release the lock before the matmul,
            // for the same reason as `gate_knn_mmap_fast`: holding it across
            // the multiply serialises concurrent callers on unrelated layers.
            let data = {
                let mut cache = self.gate.f16_decode_cache.lock().unwrap();
                if cache.len() <= layer {
                    cache.resize(layer + 1, None);
                }
                let miss = cache[layer].is_none();
                if miss {
                    let byte_offset = view.slice.float_offset * 2;
                    let byte_end = byte_offset + view.slice.num_features * self.hidden_size * 2;
                    if byte_end > mmap.len() {
                        return None;
                    }
                    let raw = &mmap[byte_offset..byte_end];
                    cache[layer] = Some(std::sync::Arc::new(
                        larql_models::quant::half::decode_f16(raw),
                    ));
                }
                self.touch_gate_cache_lru(layer, miss, &mut cache);
                std::sync::Arc::clone(cache[layer].as_ref().unwrap())
            };
            let arr = ArrayView2::from_shape(
                (view.slice.num_features, self.hidden_size),
                data.as_slice(),
            )
            .unwrap();
            return Some(gate_matmul(&arr, &x.view()));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    //! Inline coverage of `gate_scores_batch`. Heap and f16-mmap
    //! happy paths plus the empty-input + missing-data fall-throughs.
    //! The f32-mmap fast path is exercised by the integration test
    //! in `tests/compute_storage_regressions.rs`; here we focus on
    //! the branches that aren't reached from there.

    use crate::index::core::VectorIndex;
    use crate::index::types::GateLayerSlice;
    use ndarray::array;

    fn heap_idx() -> VectorIndex {
        let gate = ndarray::Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        )
        .unwrap();
        VectorIndex::new(vec![Some(gate)], vec![None], 1, 4)
    }

    /// `gate_scores_batch` rejects an empty seq.
    #[test]
    fn empty_seq_returns_none() {
        let v = heap_idx();
        let x = ndarray::Array2::<f32>::zeros((0, 4));
        assert!(v.gate_scores_batch(0, &x).is_none());
    }

    /// Past-end layer falls through (no gate data) and returns None.
    #[test]
    fn out_of_range_layer_returns_none() {
        let v = heap_idx();
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        assert!(v.gate_scores_batch(99, &x).is_none());
    }

    /// Heap-mode happy path — falls through the fast paths and lands
    /// on `resolve_gate` + `gate_matmul`.
    #[test]
    fn heap_path_returns_seq_x_features_scores() {
        let v = heap_idx();
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let scores = v.gate_scores_batch(0, &x).expect("heap path");
        // [seq_len=1, num_features=3]
        assert_eq!(scores.shape(), &[1, 3]);
        // f0 dot = 1, f1 dot = 4, f2 dot = 9 (3*3).
        assert_eq!(scores[[0, 0]], 1.0);
        assert_eq!(scores[[0, 1]], 4.0);
        assert_eq!(scores[[0, 2]], 9.0);
    }

    /// f16 mmap fast path — populates storage with f16 bytes + slice
    /// meta, then exercises `gate_scores_2d_fast`'s f16 lazy-decode
    /// branch (which populates the f16 decode cache as a side effect).
    #[test]
    fn f16_mmap_fast_path_populates_decode_cache() {
        // 3 features × 4 hidden, encoded as f16.
        let gate_floats: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, //
            0.0, 0.0, 3.0, 0.0, //
        ];
        let f16_bytes = larql_models::quant::half::encode_f16(&gate_floats);
        let mut anon = memmap2::MmapOptions::new()
            .len(f16_bytes.len())
            .map_anon()
            .expect("anon mmap");
        anon.copy_from_slice(&f16_bytes);
        let mmap = anon.make_read_only().expect("freeze");

        let v = VectorIndex::new_mmap(
            mmap,
            vec![GateLayerSlice {
                float_offset: 0,
                num_features: 3,
            }],
            crate::config::dtype::StorageDtype::F16,
            None,
            1,
            4,
        );

        // Cache should be empty before the call.
        assert!(v.gate.f16_decode_cache.lock().unwrap()[0].is_none());

        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let scores = v.gate_scores_batch(0, &x).expect("f16 fast path");
        assert_eq!(scores.shape(), &[1, 3]);
        // Within f16 quant noise: f0≈1, f1≈4, f2≈9.
        assert!((scores[[0, 0]] - 1.0).abs() < 0.01);
        assert!((scores[[0, 1]] - 4.0).abs() < 0.01);
        assert!((scores[[0, 2]] - 9.0).abs() < 0.05);

        // Cache should be populated as a side effect.
        assert!(v.gate.f16_decode_cache.lock().unwrap()[0].is_some());
    }

    /// Empty layer slice (`num_features == 0`) on the f16 path
    /// short-circuits without panicking.
    #[test]
    fn f16_path_returns_none_when_layer_unowned() {
        // Windows' `MapViewOfFile` rejects zero-length mappings
        // (ERROR_INVALID_PARAMETER, code 87). One byte is enough —
        // the test only cares that `num_features == 0` short-circuits
        // before the mmap is read.
        let mmap = memmap2::MmapOptions::new()
            .len(1)
            .map_anon()
            .unwrap()
            .make_read_only()
            .unwrap();
        let v = VectorIndex::new_mmap(
            mmap,
            vec![GateLayerSlice {
                float_offset: 0,
                num_features: 0,
            }],
            crate::config::dtype::StorageDtype::F16,
            None,
            1,
            4,
        );
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        assert!(v.gate_scores_batch(0, &x).is_none());
    }

    /// `gate_scores_batch_backend` with a CPU backend on a heap index
    /// — exercises the backend path's fall-through to BLAS when the
    /// backend doesn't have `f32_gemv_force` for the heap shape.
    #[test]
    fn backend_path_heap_falls_back_to_blas() {
        let v = heap_idx();
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let cpu = larql_compute::CpuBackend;
        let scores = v
            .gate_scores_batch_backend(0, &x, Some(&cpu))
            .expect("backend + heap");
        assert_eq!(scores.shape(), &[1, 3]);
    }

    /// `gate_scores_batch_backend` short-circuits on empty seq.
    #[test]
    fn backend_path_empty_seq_returns_none() {
        let v = heap_idx();
        let x = ndarray::Array2::<f32>::zeros((0, 4));
        let cpu = larql_compute::CpuBackend;
        assert!(v.gate_scores_batch_backend(0, &x, Some(&cpu)).is_none());
    }

    /// Multi-position seq exercises the gemm path (vs single-row gemv).
    #[test]
    fn heap_path_multi_position_returns_correct_shape() {
        let v = heap_idx();
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
        ];
        let scores = v.gate_scores_batch(0, &x).expect("multi-pos");
        assert_eq!(scores.shape(), &[3, 3]);
    }

    /// Warmed-cache fast path: pre-populate `warmed_gates` and
    /// install matching gate slice metadata so `gate_scores_2d_fast`
    /// hits the warmed branch.
    #[test]
    fn warmed_cache_path_returns_scores() {
        // Build a dummy mmap so the storage's gate_layer_slice is
        // populated; the warmed cache will short-circuit before any
        // mmap data is read.
        let mmap = memmap2::MmapOptions::new()
            .len(24) // 3 features × 4 hidden × 2 bytes (f16)
            .map_anon()
            .unwrap()
            .make_read_only()
            .unwrap();
        let v = VectorIndex::new_mmap(
            mmap,
            vec![GateLayerSlice {
                float_offset: 0,
                num_features: 3,
            }],
            crate::config::dtype::StorageDtype::F16,
            None,
            1,
            4,
        );
        // Populate warmed cache directly.
        v.gate.warmed_gates.write().unwrap()[0] = Some(vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, 0.0, //
            0.0, 0.0, 3.0, 0.0, //
        ]);

        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let scores = v.gate_scores_batch(0, &x).expect("warmed path");
        assert_eq!(scores.shape(), &[1, 3]);
        assert_eq!(scores[[0, 2]], 9.0);
    }
}
