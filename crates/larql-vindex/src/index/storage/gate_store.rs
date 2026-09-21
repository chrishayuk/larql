//! Gate matrix storage — resolve / mmap-fast-path / decode cache LRU.
//!
//! The compute side (`crate::index::compute::gate_knn`) consumes
//! gate vectors but never reaches into the mmap or LRU machinery
//! directly — it goes through this module's accessors.
//!
//! What lives here:
//!
//! - `GateData`             — owned f32 contiguous gate matrix.
//! - `gemv`, `gate_matmul`,
//!   `gate_gemv_gpu`        — small BLAS / GPU wrappers used by KNN.
//! - `set_gate_cache_max_layers` (pub) and the LRU bookkeeping that
//!   pairs with it (`touch_gate_cache_lru`).
//! - `resolve_gate`         — warm → heap → mmap-f32 → mmap-f16
//!                            unified accessor.
//! - `gate_knn_mmap_fast`   — zero-copy f32 mmap path used as the
//!                            `gate_knn` happy path.

use std::sync::{Mutex, RwLock};

use larql_compute::{ComputeBackend, MatMul};
use ndarray::{Array1, Array2, ArrayView2};

use crate::index::core::VectorIndex;
use crate::index::storage::vindex_storage::VindexStorage;

// ── GateStore — composes all gate-matrix-and-cache state ────────────────

/// Gate matrix storage + decode caches + HNSW index.
///
/// Carved out of the monolithic `VectorIndex` god struct in the
/// 2026-04-25 reorg. Field names match the legacy flat ones so call
/// sites can be migrated mechanically; a future PR can drop the
/// redundant `gate_` prefixes.
pub struct GateStore {
    /// Per-layer gate vectors (heap mode).
    pub gate_vectors: Vec<Option<Array2<f32>>>,
    /// Lazy decode cache for f16 gate vectors.
    ///
    /// `Arc` per layer so a reader can take a cheap handle and release the
    /// mutex *before* scoring. Holding the lock across `gemv` would
    /// serialise `PatchedVindex::walk`'s rayon-parallel layers against each
    /// other, which is the whole point of parallelising them; cloning the
    /// data instead would reintroduce the ~71 MB/layer copy this cache
    /// exists to avoid. An `Arc` clone is a refcount bump.
    pub f16_decode_cache: Mutex<Vec<Option<std::sync::Arc<Vec<f32>>>>>,
    /// LRU queue for `f16_decode_cache`. Back is oldest, front is newest.
    pub gate_cache_lru: Mutex<std::collections::VecDeque<usize>>,
    /// Cap on live entries in `f16_decode_cache`. 0 = unlimited.
    pub gate_cache_max_layers: std::sync::atomic::AtomicUsize,
    /// Warm-up cache (RwLock — lock-free reads).
    pub warmed_gates: RwLock<Vec<Option<Vec<f32>>>>,
    /// HNSW per-layer index, lazily built on first query when enabled.
    pub hnsw_cache: Mutex<Vec<Option<super::super::hnsw::HnswLayer>>>,
    /// Fine-grained HNSW indexed by `(layer, expert_id)` over each expert's
    /// gate-vector slice (704 vectors per expert on Gemma 4 26B-A4B vs 90k
    /// per-layer in `hnsw_cache`).  Lazily populated by `gate_knn_expert`
    /// when HNSW is enabled.  Used for the per-unit shard architecture
    /// where each shard hosts only its own (layer, expert) units and a
    /// query's KNN search space is bounded by one expert's slice.
    pub hnsw_unit_cache:
        Mutex<std::collections::HashMap<(usize, usize), super::super::hnsw::HnswLayer>>,
    /// HNSW master toggle.
    pub hnsw_enabled: std::sync::atomic::AtomicBool,
    /// HNSW beam width.
    pub hnsw_ef_search: std::sync::atomic::AtomicUsize,
}

impl GateStore {
    /// Inert default — every Option is None, every cache is empty.
    pub fn empty(num_layers: usize) -> Self {
        Self {
            gate_vectors: vec![None; num_layers],
            f16_decode_cache: Mutex::new(vec![None; num_layers]),
            gate_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            gate_cache_max_layers: std::sync::atomic::AtomicUsize::new(0),
            warmed_gates: RwLock::new(vec![None; num_layers]),
            hnsw_cache: Mutex::new((0..num_layers).map(|_| None).collect()),
            hnsw_unit_cache: Mutex::new(std::collections::HashMap::new()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(false),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(200),
        }
    }
}

impl Clone for GateStore {
    /// Heap data + atomics carry over; mutex-guarded caches reset to
    /// fresh state per the existing VectorIndex Clone contract
    /// (caches are working memory, not durable state). Mmap fields
    /// moved to `MmapStorage::mmap_handles` in step 6 of the
    /// `VindexStorage` migration.
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering;
        let nl = self.gate_vectors.len();
        Self {
            gate_vectors: self.gate_vectors.clone(),
            f16_decode_cache: Mutex::new(vec![None; nl]),
            gate_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            gate_cache_max_layers: std::sync::atomic::AtomicUsize::new(
                self.gate_cache_max_layers.load(Ordering::Relaxed),
            ),
            warmed_gates: RwLock::new(vec![None; nl]),
            hnsw_cache: Mutex::new((0..nl).map(|_| None).collect()),
            hnsw_unit_cache: Mutex::new(std::collections::HashMap::new()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(
                self.hnsw_enabled.load(Ordering::Relaxed),
            ),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(
                self.hnsw_ef_search.load(Ordering::Relaxed),
            ),
        }
    }
}

// ── BLAS / GPU helpers ──────────────────────────────────────────────────

/// Matrix-vector multiply: view[N, hidden] × vec[hidden] → scores[N].
/// All compute goes through larql-compute.
pub(crate) fn gemv(view: &ArrayView2<f32>, vec: &Array1<f32>) -> Array1<f32> {
    // `gate[N, hidden] . vec[hidden] -> [N]`, expressed as a real
    // matrix-vector product rather than a 1-row matmul against a
    // transpose.
    //
    // The previous form was `matmul_transb(vec.into_shape((1, hidden)),
    // gate)`, i.e. `a.dot(&b.t())` with `a` shaped [1, hidden]. ndarray
    // only dispatches to BLAS when the operand layouts qualify, and that
    // shape does not reach `sgemv`: it fell to ndarray's own path.
    // Measured on one layer of a real BitNet 2B browse vindex
    // (6912 features x 2560 dims, f32 cached):
    //
    //   a.dot(&b.t())     15.5 ms/layer    4.6 GB/s
    //   gate.dot(&vec)     3.3 ms/layer   21.4 GB/s   <- this
    //   manual row dot     9.8 ms/layer    7.2 GB/s
    //
    // 4.7x, from removing the transpose. `describe()` scans 12-30 layers
    // per call, so this is the dominant term in its latency.
    //
    // `Array2::dot(&Array1)` is ndarray's gemv entry point and hits
    // `cblas_sgemv` for f32 with a standard-layout operand, which the
    // gate view is (contiguous rows straight out of the mmap or the f16
    // decode cache).
    view.dot(vec)
}

/// Gate scores batch: gate[N, hidden] × x[seq, hidden]^T → [N, seq].
pub(crate) fn gate_matmul(gate: &ArrayView2<f32>, x: &ArrayView2<f32>) -> Array2<f32> {
    let cpu = larql_compute::CpuBackend;
    cpu.matmul_transb(*gate, *x)
}

/// GPU-accelerated gate matmul for the single-position decode case.
///
/// When `x` is a single row (seq_len == 1) and the caller passes a
/// Metal backend, route the gate gemv through `f32_gemv_force` — the
/// dedicated row-per-simdgroup kernel that closed lm_head on Gemma 3 4B.
/// Returns `None` if `seq_len > 1` or if the backend has no f32_gemv;
/// caller falls back to `gate_matmul` (CPU BLAS).
///
/// Shape note: the [N, 1] column vector is laid out flat as [N];
/// caller wraps it back into `Array2` shape.
pub(crate) fn gate_gemv_gpu(
    gate: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    backend: &dyn ComputeBackend,
) -> Option<Array2<f32>> {
    if x.shape()[0] != 1 {
        return None;
    }
    let x_row = x.row(0);
    let x_slice = x_row.as_slice()?;
    // Force GPU dispatch regardless of the backend's flop_threshold —
    // per-layer gate gemvs are ~50–200 M FLOPs, below the default
    // 500 M threshold that protects tiny one-off gemvs. At 34/60
    // layers × every decode token the aggregated saving is real even
    // if each call alone would be dispatch-bound.
    let scores = backend.f32_gemv_force(*gate, x_slice)?;
    Array2::from_shape_vec((gate.shape()[0], 1), scores).ok()
}

// ── Owned-data wrapper ──────────────────────────────────────────────────

/// Resolved gate matrix data — owned f32 with feature count.
pub(crate) struct GateData {
    pub(crate) data: Vec<f32>,
    pub(crate) num_features: usize,
}

impl GateData {
    pub(crate) fn view(&self, hidden_size: usize) -> ArrayView2<'_, f32> {
        ArrayView2::from_shape((self.num_features, hidden_size), &self.data).unwrap()
    }
}

// ── Storage-side methods on VectorIndex ────────────────────────────────

impl VectorIndex {
    /// Cap the number of decoded f16 gate layers held in
    /// `f16_decode_cache`. Call with 0 for unlimited (default);
    /// non-zero enables LRU eviction on the next insert that would
    /// exceed the cap.
    ///
    /// Typical use: `larql serve --max-gate-cache-layers N` to bound
    /// a long-running server's RSS. A 31B f16 gate table decodes to
    /// ~433 MB per layer, so `--max-gate-cache-layers 4` caps decoded
    /// gates at ~1.7 GB (at the cost of repeated decode on evicted
    /// layers).
    pub fn set_gate_cache_max_layers(&self, max_layers: usize) {
        self.gate
            .gate_cache_max_layers
            .store(max_layers, std::sync::atomic::Ordering::Relaxed);
        // Shrink eagerly if the new cap is below the current cache size.
        if max_layers > 0 {
            let mut cache = self.gate.f16_decode_cache.lock().unwrap();
            let mut lru = self.gate.gate_cache_lru.lock().unwrap();
            while lru.len() > max_layers {
                if let Some(evict) = lru.pop_back() {
                    if evict < cache.len() {
                        cache[evict] = None;
                    }
                }
            }
        }
    }

    /// Record a cache hit/miss on `layer`, evicting LRU entries if the
    /// cap is reached. Must be called with `cache` already locked by
    /// the caller; `just_inserted` is true when the caller *just*
    /// decoded and wrote `cache[layer]`.
    pub(crate) fn touch_gate_cache_lru(
        &self,
        layer: usize,
        just_inserted: bool,
        cache: &mut [Option<std::sync::Arc<Vec<f32>>>],
    ) {
        let max = self
            .gate
            .gate_cache_max_layers
            .load(std::sync::atomic::Ordering::Relaxed);
        if max == 0 {
            return;
        }
        let mut lru = self.gate.gate_cache_lru.lock().unwrap();
        // Move `layer` to the front (newest). If it's not in the queue
        // yet, push it; otherwise rotate.
        if let Some(pos) = lru.iter().position(|&l| l == layer) {
            lru.remove(pos);
        }
        lru.push_front(layer);
        if just_inserted {
            while lru.len() > max {
                if let Some(evict) = lru.pop_back() {
                    if evict < cache.len() && evict != layer {
                        cache[evict] = None;
                    }
                }
            }
        }
    }

    /// Resolve the gate matrix for a layer as contiguous f32.
    /// Handles all storage paths: warmed → heap → mmap f32 → mmap f16.
    /// Returns owned data (zero-copy from mmap via `to_vec` on the
    /// hot path).
    pub(crate) fn resolve_gate(&self, layer: usize) -> Option<GateData> {
        // 1. Warmed cache
        {
            let warmed = self.gate.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self
                    .storage
                    .gate_layer_slice(layer)
                    .map(|s| s.num_features)
                    .unwrap_or(0);
                if nf > 0 {
                    return Some(GateData {
                        data: data.clone(),
                        num_features: nf,
                    });
                }
            }
        }

        // 2. Heap
        if let Some(Some(ref matrix)) = self.gate.gate_vectors.get(layer) {
            return Some(GateData {
                data: matrix.as_slice().unwrap().to_vec(),
                num_features: matrix.shape()[0],
            });
        }

        // 3. Mmap
        if let Some(view) = self.storage.gate_layer_view(layer) {
            if view.slice.num_features == 0 {
                return None;
            }
            let bpf = crate::config::dtype::bytes_per_float(view.dtype);
            let byte_offset = view.slice.float_offset * bpf;
            let byte_count = view.slice.num_features * self.hidden_size * bpf;
            let byte_end = byte_offset + byte_count;
            let mmap: &[u8] = view.bytes.as_ref();
            if byte_end > mmap.len() {
                return None;
            }

            let data = match view.dtype {
                crate::config::dtype::StorageDtype::F32 => {
                    let float_count = view.slice.num_features * self.hidden_size;
                    unsafe {
                        let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                        std::slice::from_raw_parts(ptr, float_count).to_vec()
                    }
                }
                crate::config::dtype::StorageDtype::F16 => {
                    // `GateData` owns its buffer, so this arm still copies.
                    // It is the slow path: `gate_knn_mmap_fast` handles f16
                    // without copying and is what the scan actually takes.
                    // Reached only by callers that need an owned matrix.
                    let mut cache = self.gate.f16_decode_cache.lock().unwrap();
                    if cache.len() <= layer {
                        cache.resize(layer + 1, None);
                    }
                    let miss = cache[layer].is_none();
                    if miss {
                        let raw = &mmap[byte_offset..byte_end];
                        cache[layer] = Some(std::sync::Arc::new(
                            larql_models::quant::half::decode_f16(raw),
                        ));
                    }
                    self.touch_gate_cache_lru(layer, miss, &mut cache);
                    cache[layer].as_ref().unwrap().as_ref().clone()
                }
            };
            return Some(GateData {
                data,
                num_features: view.slice.num_features,
            });
        }

        None
    }

    /// Zero-copy gate KNN scoring for the mmap path — no allocation of a
    /// gate copy. Returns `None` if the layer cannot be scored here;
    /// caller falls back to `resolve_gate`.
    ///
    /// Handles f32 (direct reinterpret of the mmap) *and* f16 (score out of
    /// the decode cache in place). The f16 arm is the load-bearing one:
    /// without it every f16 layer fell through to `resolve_gate`, which
    /// ends in `cache[layer].as_ref().unwrap().clone()` — a full f32 copy
    /// of the layer's gate matrix on **every query**. At 6912 features ×
    /// 2560 dims that is ~71 MB cloned per layer, so a 20-layer
    /// `describe()` spent ~1.4 GB on allocation and memcpy before scoring
    /// a single feature, and another ~1.4 GB reading it back in `gemv`.
    /// Measured effect on a real 2B browse container: `describe()` 288 ms
    /// → see `bench_graph.csv`. The decode itself was already cached; it
    /// was purely the copy.
    pub(crate) fn gate_knn_mmap_fast(
        &self,
        layer: usize,
        residual: &Array1<f32>,
    ) -> Option<Array1<f32>> {
        // Warmed cache (RwLock read — lock-free when no writers).
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
                    return Some(gemv(&view, residual));
                }
            }
        }

        // f32 mmap zero-copy.
        if self.storage.gate_dtype() == crate::config::dtype::StorageDtype::F32 {
            if let Some(view) = self.storage.gate_layer_view(layer) {
                if view.slice.num_features == 0 {
                    return None;
                }
                let bpf = 4;
                let byte_offset = view.slice.float_offset * bpf;
                let byte_end = byte_offset + view.slice.num_features * self.hidden_size * bpf;
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
                return Some(gemv(&arr, residual));
            }
        }

        // f16 mmap: score out of the decode cache without copying it.
        //
        // Decoding on a miss is unavoidable (it is what the cache is for),
        // but on a hit the previous path cloned the whole layer purely to
        // hand an owned `Vec` back to the caller. `gemv` only needs a view,
        // so take one over the cached buffer while the lock is held.
        //
        // The lock is released before `gemv`: the cache holds an `Arc` per
        // layer, so a reader takes a refcount bump and scores outside the
        // critical section. Holding it across `gemv` would serialise
        // `PatchedVindex::walk`'s rayon-parallel layers, and cloning the
        // buffer to release early would reintroduce the ~71 MB/layer copy
        // this path exists to remove.
        if self.storage.gate_dtype() == crate::config::dtype::StorageDtype::F16 {
            let view = self.storage.gate_layer_view(layer)?;
            if view.slice.num_features == 0 {
                return None;
            }
            let bpf = 2;
            let byte_offset = view.slice.float_offset * bpf;
            let byte_end = byte_offset + view.slice.num_features * self.hidden_size * bpf;
            let mmap: &[u8] = view.bytes.as_ref();
            if byte_end > mmap.len() {
                return None;
            }

            let data = {
                let mut cache = self.gate.f16_decode_cache.lock().unwrap();
                if cache.len() <= layer {
                    cache.resize(layer + 1, None);
                }
                let miss = cache[layer].is_none();
                if miss {
                    cache[layer] = Some(std::sync::Arc::new(
                        larql_models::quant::half::decode_f16(&mmap[byte_offset..byte_end]),
                    ));
                }
                self.touch_gate_cache_lru(layer, miss, &mut cache);
                std::sync::Arc::clone(cache[layer].as_ref()?)
            };
            let arr = ArrayView2::from_shape(
                (view.slice.num_features, self.hidden_size),
                data.as_slice(),
            )
            .ok()?;
            return Some(gemv(&arr, residual));
        }

        None
    }
}

// ══════════════════════════════════════════════════════════════
// Gate cache LRU tests
//
// Cover `set_gate_cache_max_layers` and `touch_gate_cache_lru` on an
// f16 mmap-backed VectorIndex. Each `gate_knn` call at a new layer
// lazily decodes the layer's gate matrix into `f16_decode_cache`;
// callers should cap the number of resident decoded layers via
// `set_gate_cache_max_layers` to bound RSS on long-running servers.
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod gate_cache_lru_tests {
    use crate::config::dtype::StorageDtype;
    use crate::index::core::VectorIndex;
    use crate::index::types::GateLayerSlice;
    use ndarray::Array1;

    /// Build a minimal f16 mmap-backed VectorIndex suitable for
    /// exercising the f16 decode cache. `num_layers` layers, each
    /// with `num_features` features over `hidden` dims. The gate
    /// matrix at each layer is a scaled identity (row i, col
    /// `i % hidden` = 1.0) so a query that's 1.0 in dim 0 always
    /// hits feature 0.
    fn f16_mmap_index(num_layers: usize, num_features: usize, hidden: usize) -> VectorIndex {
        let per_layer_floats = num_features * hidden;
        let per_layer_bytes = per_layer_floats * 2; // f16
        let total_bytes = per_layer_bytes * num_layers;

        let mut anon = memmap2::MmapMut::map_anon(total_bytes).unwrap();

        let mut slices = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let mut data = vec![0.0f32; per_layer_floats];
            for i in 0..num_features {
                data[i * hidden + (i % hidden)] = 1.0;
            }
            let bytes = larql_models::quant::half::encode_f16(&data);
            let off = l * per_layer_bytes;
            anon[off..off + per_layer_bytes].copy_from_slice(&bytes);
            slices.push(GateLayerSlice {
                float_offset: (l * per_layer_bytes) / 2,
                num_features,
            });
        }

        let mmap = anon.make_read_only().unwrap();
        VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, num_layers, hidden)
    }

    /// Touch layer `l` to force a gate cache decode (or a hit if
    /// already cached).
    fn touch(idx: &VectorIndex, layer: usize) {
        let q = Array1::from_vec(vec![1.0f32; idx.hidden_size]);
        let _ = idx.gate_knn(layer, &q, 1);
    }

    fn resident_layers(idx: &VectorIndex) -> usize {
        idx.gate
            .f16_decode_cache
            .lock()
            .unwrap()
            .iter()
            .filter(|slot| slot.is_some())
            .count()
    }

    fn lru_snapshot(idx: &VectorIndex) -> Vec<usize> {
        idx.gate
            .gate_cache_lru
            .lock()
            .unwrap()
            .iter()
            .copied()
            .collect()
    }

    #[test]
    fn unlimited_cache_grows_without_eviction() {
        let idx = f16_mmap_index(4, 2, 4);
        for l in 0..4 {
            touch(&idx, l);
        }
        assert_eq!(resident_layers(&idx), 4, "all 4 layers must stay resident");
        assert_eq!(
            lru_snapshot(&idx).len(),
            0,
            "LRU queue should stay empty when the cap is unlimited"
        );
    }

    #[test]
    fn cap_two_evicts_lru_on_third_access() {
        let idx = f16_mmap_index(4, 2, 4);
        idx.set_gate_cache_max_layers(2);

        touch(&idx, 0);
        touch(&idx, 1);
        assert_eq!(resident_layers(&idx), 2);

        touch(&idx, 2);
        assert_eq!(resident_layers(&idx), 2, "cap of 2 holds");

        let cache = idx.gate.f16_decode_cache.lock().unwrap();
        assert!(cache[0].is_none(), "layer 0 should have been evicted");
        assert!(cache[1].is_some(), "layer 1 still cached");
        assert!(cache[2].is_some(), "layer 2 newly cached");
    }

    #[test]
    fn cache_hit_promotes_layer_to_newest() {
        let idx = f16_mmap_index(4, 2, 4);
        idx.set_gate_cache_max_layers(2);

        touch(&idx, 0);
        touch(&idx, 1);
        assert_eq!(lru_snapshot(&idx), vec![1, 0]);

        touch(&idx, 0);
        assert_eq!(lru_snapshot(&idx), vec![0, 1]);

        touch(&idx, 2);
        let cache = idx.gate.f16_decode_cache.lock().unwrap();
        assert!(cache[0].is_some(), "layer 0 was promoted on hit, must stay");
        assert!(cache[1].is_none(), "layer 1 was oldest, must be evicted");
        assert!(cache[2].is_some(), "layer 2 newly cached");
    }

    #[test]
    fn shrinking_cap_evicts_down_to_new_bound() {
        let idx = f16_mmap_index(4, 2, 4);
        idx.set_gate_cache_max_layers(4);
        for l in 0..4 {
            touch(&idx, l);
        }
        assert_eq!(resident_layers(&idx), 4);
        assert_eq!(lru_snapshot(&idx).len(), 4);

        idx.set_gate_cache_max_layers(1);
        assert_eq!(resident_layers(&idx), 1);
        assert_eq!(lru_snapshot(&idx).len(), 1);

        let cache = idx.gate.f16_decode_cache.lock().unwrap();
        assert!(cache[3].is_some(), "newest layer should be the survivor");
        for l in 0..3 {
            assert!(cache[l].is_none(), "layer {l} should have been evicted");
        }
    }

    #[test]
    fn set_cap_zero_is_noop_on_existing_entries() {
        let idx = f16_mmap_index(3, 2, 4);
        idx.set_gate_cache_max_layers(2);
        touch(&idx, 0);
        touch(&idx, 1);
        assert_eq!(resident_layers(&idx), 2);

        idx.set_gate_cache_max_layers(0);
        assert_eq!(resident_layers(&idx), 2);
    }
    #[test]
    fn f16_fast_path_scores_without_cloning_the_layer() {
        // The f16 arm of `gate_knn_mmap_fast` is what the gate scan takes on
        // an f16 container. Before it existed, f16 layers fell through to
        // `resolve_gate`, which clones the whole decoded layer per query.
        //
        // Asserts the arm is reached and correct: the fixture's gate matrix
        // is a scaled identity, so a query that is 1.0 in every dim scores
        // every feature at 1.0, and feature 0 is among the top hits.
        let idx = f16_mmap_index(2, 4, 4);
        let q = Array1::from_vec(vec![1.0f32; 4]);

        let scores = idx
            .gate_knn_mmap_fast(0, &q)
            .expect("f16 mmap layers must be scored by the fast path, not resolve_gate");
        assert_eq!(scores.len(), 4, "one score per feature");
        for (i, s) in scores.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-3,
                "feature {i} scored {s}, expected ~1.0 from the identity fixture"
            );
        }

        // Scoring populated the decode cache (the buffer the Arc points at),
        // so a second call is a cache hit and must agree exactly.
        assert_eq!(resident_layers(&idx), 1, "scoring must populate the cache");
        let again = idx.gate_knn_mmap_fast(0, &q).expect("cache hit");
        assert_eq!(scores, again, "a cache hit must not change the scores");
    }

    #[test]
    fn f16_fast_path_agrees_with_the_resolve_gate_slow_path() {
        // Two routes to the same numbers: the fast path scores in place out
        // of the Arc'd cache, `resolve_gate` hands back an owned copy that
        // the caller multiplies itself. They must not disagree -- that would
        // mean the optimisation changed answers, which is the failure mode
        // worth a test rather than the speed.
        let idx = f16_mmap_index(1, 6, 4);
        let q = Array1::from_vec(vec![0.5f32, 0.25, 0.125, 1.0]);

        let fast = idx.gate_knn_mmap_fast(0, &q).expect("fast path");
        let gate = idx.resolve_gate(0).expect("slow path");
        let view = gate.view(idx.hidden_size);
        let slow = super::gemv(&view, &q);

        assert_eq!(fast.len(), slow.len());
        for (i, (f, s)) in fast.iter().zip(slow.iter()).enumerate() {
            assert!(
                (f - s).abs() < 1e-6,
                "feature {i}: fast={f} slow={s} -- the paths disagree"
            );
        }
    }

    #[test]
    fn f16_cache_hands_out_arc_handles_not_copies() {
        // The cache holds `Arc<Vec<f32>>` specifically so a reader can take a
        // handle and release the mutex before scoring; holding it across the
        // multiply would serialise `PatchedVindex::walk`'s parallel layers.
        // Two handles to the same layer must therefore be the same
        // allocation, not two copies of it.
        let idx = f16_mmap_index(1, 4, 4);
        touch(&idx, 0);

        let (a, b) = {
            let cache = idx.gate.f16_decode_cache.lock().unwrap();
            let entry = cache[0].as_ref().expect("layer 0 cached");
            (std::sync::Arc::clone(entry), std::sync::Arc::clone(entry))
        };
        assert!(
            std::sync::Arc::ptr_eq(&a, &b),
            "cache handles must alias one buffer, not clone it"
        );
    }
}
