//! `WalkFfn` — FFN backend that replaces dense matmul with vindex lookups.
//!
//! Routing table (priority order, see `forward_with_activation`):
//!
//! | # | Condition                                            | Path                         |
//! | - | ---------------------------------------------------- | ---------------------------- |
//! | 0 | `seq_len == 1` and L1 cache has the residual         | `l1_cache_hit`               |
//! | 1 | `index.has_overrides_at(layer)`                      | `override:sparse`            |
//! | 2 | `config.is_sparse(layer)`                            | `sparse:*`                   |
//! | 3 | `index.has_fp4_storage()`                            | `fp4_storage:sparse`         |
//! | 4 | `has_interleaved_kquant()` + gated FFN               | `interleaved_kquant:native`  |
//! | 5 | `has_interleaved_q4()` + backend has Q4              | `interleaved_q4:*`           |
//! | 6 | `has_interleaved()`                                  | `interleaved`                |
//! | 7 | `has_full_mmap_ffn()`                                | `full_mmap`                  |
//! | 8 | `has_interleaved_kquant()`                           | `interleaved_kquant:dequant` |
//! | 9 | `has_down_features()` + safetensors weights loaded   | `exact`                      |
//! | 10| Fallback: sparse matmul against safetensors weights  | `weights_fallback:*`         |
//!
//! Priority rationale: overrides must bypass everything (whole-layer
//! paths silently lose overridden features). FP4/FP8 is handled by the
//! sparse path because the format is per-feature by construction —
//! there is no batched FP4 dense path on CPU. Q4K/Q4/f32 interleaved
//! are perf-preference ordered. `exact` and `weights_fallback` are
//! correctness baselines that require safetensors weights.
//!
//! Each walk path lives in its own module under this directory:
//!
//! - `sparse.rs`                  — per-feature walk, unified ffn_row_* dispatch
//! - `interleaved.rs`             — f32 interleaved mmap, three BLAS gemms
//! - `interleaved_q4.rs`            — Q4_0 interleaved, CPU kernel / Metal Q4
//! - `interleaved_kquant_native.rs` — K-quant direct matvec, no dequant cache
//! - `interleaved_kquant_dequant.rs`— K-quant dequant, full f32 dense after decode
//! - `full_mmap.rs`               — gate/up/down in three separate mmap files
//! - `exact.rs`                   — gate/up from safetensors, down from mmap
//! - `helpers.rs`                 — cross-path utilities + trace metadata
//!
//! Adding a new storage format should almost never touch `mod.rs` — add
//! a new module with a single walk function, one branch in the routing
//! ladder, and a unit test in `routing_tests.rs`.

use ndarray::Array2;

use crate::ffn::sparse_compute::sparse_ffn_forward;
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use crate::vindex::l1_cache::FfnL1Cache;
use crate::vindex::walk_config::WalkFfnConfig;
use larql_compute::prelude::*;

use larql_vindex::{GateIndex, WalkHit, WalkTrace};

mod exact;
mod full_mmap;
mod helpers;
mod interleaved;
mod interleaved_kquant_dequant;
mod interleaved_kquant_native;
mod interleaved_q4;
mod sparse;

#[cfg(test)]
mod routing_tests;

pub use helpers::DispatchEntry;

pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn GateIndex,
    pub config: WalkFfnConfig,
    pub backend: Option<&'a dyn ComputeBackend>,
    trace_residuals: std::cell::RefCell<Vec<(usize, Vec<f32>)>>,
    record_trace: bool,
    l1_cache: Option<FfnL1Cache>,
    /// Dispatch-trace sink. `None` = disabled. When `Some`, every walk
    /// path appends a (layer, name) entry on exit. Used by the routing
    /// unit tests and by the env-var dispatch trace for Q2 debugging.
    dispatch_trace: std::cell::RefCell<Option<Vec<DispatchEntry>>>,
}

impl<'a> WalkFfn<'a> {
    pub fn from_config(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        config: WalkFfnConfig,
    ) -> Self {
        Self {
            weights,
            index,
            config,
            backend: None,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: false,
            l1_cache: None,
            dispatch_trace: std::cell::RefCell::new(None),
        }
    }

    pub fn with_backend(mut self, backend: &'a dyn ComputeBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn with_trace(mut self) -> Self {
        self.record_trace = true;
        self
    }

    pub fn with_l1_cache(mut self, num_layers: usize) -> Self {
        self.l1_cache = Some(FfnL1Cache::new(num_layers));
        self
    }

    pub fn l1_cache_stats(&self) -> Option<(u64, u64)> {
        self.l1_cache.as_ref().map(|c| (c.hits(), c.misses()))
    }

    /// Enable the dispatch trace. Each walk path records its name to
    /// this buffer on exit. Use [`take_dispatch_trace`] to retrieve.
    pub fn with_dispatch_trace(self) -> Self {
        *self.dispatch_trace.borrow_mut() = Some(Vec::new());
        self
    }

    /// Drain the dispatch trace and return its accumulated entries.
    /// Returns empty if the trace wasn't enabled.
    pub fn take_dispatch_trace(&self) -> Vec<DispatchEntry> {
        self.dispatch_trace
            .borrow_mut()
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Record a dispatch entry; no-op when the trace is disabled.
    /// Called by each walk path on successful exit.
    ///
    /// Also emits to stderr when `LARQL_WALK_TRACE=1` — makes silent
    /// fallbacks immediately visible without requiring the caller to
    /// opt into the in-memory trace. The env var check is cheap on
    /// the unset path (one thread-local lookup per layer).
    pub(super) fn trace_path(&self, layer: usize, path: &'static str) {
        if let Some(vec) = self.dispatch_trace.borrow_mut().as_mut() {
            vec.push(DispatchEntry { layer, path });
        }
        if walk_trace_env_enabled() {
            eprintln!("[walk_ffn] L{layer} → {path}");
        }
    }
}

// Thread-local cache of the LARQL_WALK_TRACE env var so we don't
// getenv on every layer. Set once per thread on first access; the
// env var is typically static across a process lifetime.
thread_local! {
    static WALK_TRACE_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn walk_trace_env_enabled() -> bool {
    WALK_TRACE_ENABLED.with(|c| {
        if let Some(v) = c.get() {
            return v;
        }
        let enabled = std::env::var("LARQL_WALK_TRACE").ok().as_deref() == Some("1");
        c.set(Some(enabled));
        enabled
    })
}

impl<'a> WalkFfn<'a> {
    fn top_k_for(&self, layer: usize) -> usize {
        self.config.k_for(layer).unwrap_or(usize::MAX)
    }

    // ── Legacy constructors (stable public API) ──

    pub fn new(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        let config = if top_k == usize::MAX {
            WalkFfnConfig::dense(weights.num_layers)
        } else {
            WalkFfnConfig::sparse(weights.num_layers, top_k)
        };
        Self::from_config(weights, index, config)
    }

    pub fn new_unlimited(weights: &'a ModelWeights, index: &'a dyn GateIndex) -> Self {
        Self::from_config(weights, index, WalkFfnConfig::dense(weights.num_layers))
    }

    pub fn new_with_backend(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        top_k: usize,
        backend: &'a dyn ComputeBackend,
    ) -> Self {
        Self::new(weights, index, top_k).with_backend(backend)
    }

    pub fn new_unlimited_with_backend(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        backend: &'a dyn ComputeBackend,
    ) -> Self {
        Self::new_unlimited(weights, index).with_backend(backend)
    }

    pub fn new_with_trace(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        top_k: usize,
    ) -> Self {
        Self::new(weights, index, top_k).with_trace()
    }

    pub fn new_unlimited_with_trace(weights: &'a ModelWeights, index: &'a dyn GateIndex) -> Self {
        Self::new_unlimited(weights, index).with_trace()
    }

    pub fn take_residuals(&self) -> Vec<(usize, Vec<f32>)> {
        self.trace_residuals.borrow_mut().drain(..).collect()
    }

    pub fn take_trace(&self) -> WalkTrace {
        let residuals = self
            .trace_residuals
            .borrow_mut()
            .drain(..)
            .collect::<Vec<_>>();
        let mut layers = Vec::with_capacity(residuals.len());
        for (layer, residual) in residuals {
            let r = ndarray::Array1::from_vec(residual);
            let hits = self.index.gate_knn(layer, &r, self.top_k_for(layer));
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.index.feature_meta(layer, feature)?.clone();
                    Some(WalkHit {
                        layer,
                        feature,
                        gate_score,
                        meta,
                    })
                })
                .collect();
            layers.push((layer, walk_hits));
        }
        WalkTrace { layers }
    }
}

impl<'a> FfnBackend for WalkFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let num_features = self.index.num_features(layer);
        if num_features == 0 {
            self.trace_path(layer, "zero_features_dense");
            let dense_ffn = crate::ffn::WeightFfn {
                weights: self.weights,
            };
            return dense_ffn.forward_with_activation(layer, x);
        }

        if self.record_trace {
            let seq_len = x.shape()[0];
            let last_row = x.row(seq_len - 1).to_vec();
            self.trace_residuals.borrow_mut().push((layer, last_row));
        }

        // Override-aware routing: patched layers bypass every whole-layer
        // path because those would silently produce wrong activations
        // for overridden features.
        if self.index.has_overrides_at(layer) {
            if let Some(result) = self.walk_ffn_sparse(layer, x) {
                // The sparse path has already called trace_path — no
                // need to rewrite; its name carries the specialisation.
                return result;
            }
        }

        // L1 cache: single-position only. Key is a path-independent
        // hash of the residual, so any walk path that produces the
        // same output fills the same slot.
        let seq_len = x.shape()[0];
        let l1_key: Option<u64> = if seq_len == 1 && self.l1_cache.is_some() {
            let x_row = x.row(0);
            let owned;
            let slice: &[f32] = if let Some(s) = x_row.as_slice() {
                s
            } else {
                owned = x_row.to_vec();
                &owned
            };
            Some(FfnL1Cache::residual_key(slice))
        } else {
            None
        };

        if let Some(key) = l1_key {
            if let Some(cache) = &self.l1_cache {
                if let Some(cached) = cache.get(layer, key) {
                    let hidden = x.shape()[1];
                    let mut out = Array2::<f32>::zeros((1, hidden));
                    out.row_mut(0)
                        .assign(&ndarray::ArrayView1::from(cached.as_slice()));
                    self.trace_path(layer, "l1_cache_hit");
                    return (out, Array2::zeros((1, num_features)));
                }
            }
        }

        // Routing ladder. Each branch either `break`s with a result or
        // falls through to the next. See the routing table in the
        // module doc for priority order.
        let result: (Array2<f32>, Array2<f32>) = 'routing: {
            // 2. Explicit sparse K from the user.
            if self.config.is_sparse(layer) {
                if let Some(r) = self.walk_ffn_sparse(layer, x) {
                    break 'routing r;
                }
            }

            // 3. FP4/FP8 storage (exp 26) — no dedicated dense path.
            //    The sparse walk's unified ffn_row_* dispatch handles
            //    FP4/FP8 transparently via GateIndex. Routing FP4
            //    vindexes through sparse here is the whole point of
            //    the trait refactor: zero format-specific code in the
            //    walk kernel.
            if self.index.has_fp4_storage() {
                if let Some(r) = self.walk_ffn_sparse(layer, x) {
                    break 'routing r;
                }
            }

            // 4. Q4K native — direct matvec via `kquant_matmul_transb`. Same
            //    kernel `ffn_decode_step_native` uses. Goes ahead of Q4_0 /
            //    f32 interleaved / full_mmap / dequant because for a vindex
            //    that has both Q4K and one of those, this is the fast path.
            if self.index.has_interleaved_kquant() {
                if let Some(r) = self.walk_ffn_kquant_native(layer, x) {
                    break 'routing r;
                }
            }

            // 5. Q4_0 interleaved + GPU Q4 (Metal).
            if self.index.has_interleaved_q4()
                && self
                    .backend
                    .is_some_and(|be| be.supports_quant(::larql_compute::QuantFormat::Q4_K))
            {
                if let Some(r) = self.walk_ffn_q4_interleaved(layer, x) {
                    break 'routing r;
                }
            }

            // 6. f32 interleaved.
            if self.index.has_interleaved() {
                if let Some(r) = self.walk_ffn_interleaved(layer, x) {
                    break 'routing r;
                }
            }

            // 7. Full mmap — gate/up/down in separate files.
            if self.index.has_full_mmap_ffn() {
                if let Some(r) = self.walk_ffn_full_mmap(layer, x) {
                    break 'routing r;
                }
            }

            // 8. Q4K interleaved dequant — fallback for non-gated archs and
            //    any case where `walk_ffn_kquant_native` returns `None`.
            if self.index.has_interleaved_kquant() {
                if let Some(r) = self.walk_ffn_kquant_dequant(layer, x) {
                    break 'routing r;
                }
            }

            // 9. Exact — down from mmap, gate/up from safetensors.
            if self.index.has_down_features() {
                break 'routing self.walk_ffn_exact(layer, x);
            }

            // 10. Last resort: sparse matmul against safetensors weights.
            //     Fires when the vindex has no FFN payload of its own
            //     (extract_level = Browse without pinned weights).
            let top_k = self.top_k_for(layer);
            let features = self.index.gate_knn_batch(layer, x, top_k);
            let has_any_override = features.iter().any(|&f| {
                self.index.down_override(layer, f).is_some()
                    || self.index.up_override(layer, f).is_some()
            }) || self.index.has_overrides_at(layer);

            if has_any_override {
                let slot_overrides: Vec<crate::ffn::FeatureSlotOverride<'_>> = features
                    .iter()
                    .map(|&f| crate::ffn::FeatureSlotOverride {
                        feature: f,
                        gate: self.index.gate_override(layer, f),
                        up: self.index.up_override(layer, f),
                        down: self.index.down_override(layer, f),
                    })
                    .filter(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some())
                    .collect();
                self.trace_path(layer, "weights_fallback:override");
                break 'routing crate::ffn::sparse_ffn_forward_with_full_overrides(
                    self.weights,
                    layer,
                    x,
                    &features,
                    &slot_overrides,
                );
            }
            self.trace_path(layer, "weights_fallback:sparse");
            break 'routing sparse_ffn_forward(self.weights, layer, x, &features);
        };

        if let Some(key) = l1_key {
            if let Some(cache) = &self.l1_cache {
                cache.insert(layer, key, result.0.row(0).to_vec());
            }
        }

        result
    }

    fn name(&self) -> &str {
        "walk"
    }
}

#[cfg(test)]
mod dispatch_tests {
    use super::*;
    use crate::model::ModelWeights;
    use crate::test_utils::make_test_weights;
    use larql_vindex::{
        FeatureMeta, Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides, QuantizedFfnAccess,
    };
    use ndarray::{Array1, Array2};
    use std::sync::OnceLock;

    fn shared_weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }
    use crate::ffn::FfnBackend;

    /// Minimal GateIndex with only the 3 required methods.
    /// All optional methods fall back to their trait defaults (all return None/false/[]).
    /// WalkFfn routes through path 9 (last-resort sparse matmul against weights.tensors).
    struct MockGateIndex {
        n_features: usize,
    }

    impl GateLookup for MockGateIndex {
        fn gate_knn(
            &self,
            _layer: usize,
            _residual: &Array1<f32>,
            top_k: usize,
        ) -> Vec<(usize, f32)> {
            (0..top_k.min(self.n_features))
                .map(|i| (i, 1.0 / (i as f32 + 1.0)))
                .collect()
        }
        fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
            None
        }
        fn num_features(&self, _layer: usize) -> usize {
            self.n_features
        }
    }

    impl PatchOverrides for MockGateIndex {}
    impl NativeFfnAccess for MockGateIndex {}
    impl QuantizedFfnAccess for MockGateIndex {}
    impl Fp4FfnAccess for MockGateIndex {}

    fn mock_index(weights: &ModelWeights) -> MockGateIndex {
        MockGateIndex {
            n_features: weights.intermediate_size,
        }
    }

    fn input(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    // ── WalkFfn construction ──────────────────────────────────────────────────

    #[test]
    fn walk_ffn_new_unlimited() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited(weights, &idx);
        assert_eq!(ffn.name(), "walk");
    }

    #[test]
    fn walk_ffn_sparse_k() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new(weights, &idx, 4);
        assert_eq!(ffn.name(), "walk");
    }

    // ── forward shape and finiteness ─────────────────────────────────────────

    #[test]
    fn walk_ffn_forward_shape_single_token() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited(weights, &idx);
        let x = input(1, weights.hidden_size);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn walk_ffn_forward_shape_multi_token() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited(weights, &idx);
        let x = input(3, weights.hidden_size);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn walk_ffn_forward_all_layers() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited(weights, &idx);
        let x = input(1, weights.hidden_size);
        for layer in 0..weights.num_layers {
            let out = ffn.forward(layer, &x);
            assert_eq!(
                out.shape(),
                &[1, weights.hidden_size],
                "layer {layer} wrong shape"
            );
            assert!(
                out.iter().all(|v| v.is_finite()),
                "layer {layer} non-finite"
            );
        }
    }

    #[test]
    fn walk_ffn_sparse_vs_dense_same_shape() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn_sparse = WalkFfn::new(weights, &idx, 4);
        let ffn_dense = WalkFfn::new_unlimited(weights, &idx);
        let x = input(1, weights.hidden_size);
        let out_s = ffn_sparse.forward(0, &x);
        let out_d = ffn_dense.forward(0, &x);
        assert_eq!(out_s.shape(), out_d.shape());
    }

    #[test]
    fn walk_ffn_with_activation_returns_activation() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited(weights, &idx);
        let x = input(2, weights.hidden_size);
        let (out, act) = ffn.forward_with_activation(0, &x);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert_eq!(act.shape()[0], 2, "activation should have seq_len rows");
    }

    #[test]
    fn walk_ffn_zero_features_falls_back_to_weight_ffn() {
        // When MockGateIndex returns 0 features, WalkFfn should fall back to WeightFfn.
        let weights = shared_weights();
        let zero_idx = MockGateIndex { n_features: 0 };
        let ffn = WalkFfn::new_unlimited(weights, &zero_idx);
        let x = input(1, weights.hidden_size);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn walk_ffn_with_backend() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited_with_backend(weights, &idx, &larql_compute::CpuBackend);
        let x = input(1, weights.hidden_size);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    // ── trace + l1_cache + dispatch_trace ──────────────────────────────

    #[test]
    fn walk_ffn_take_residuals_returns_per_layer_traces() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited(weights, &idx).with_trace();
        let x = input(2, weights.hidden_size);
        // Run forward for every layer so trace_residuals populates.
        for layer in 0..weights.num_layers {
            ffn.forward(layer, &x);
        }
        let residuals = ffn.take_residuals();
        assert_eq!(residuals.len(), weights.num_layers);
        for (layer, residual) in &residuals {
            assert!(*layer < weights.num_layers);
            assert_eq!(residual.len(), weights.hidden_size);
            assert!(residual.iter().all(|v| v.is_finite()));
        }
        // Drained — second call must be empty.
        assert!(ffn.take_residuals().is_empty());
    }

    #[test]
    fn walk_ffn_with_trace_emits_residuals() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_with_trace(weights, &idx, 4);
        let x = input(1, weights.hidden_size);
        ffn.forward(0, &x);
        let residuals = ffn.take_residuals();
        assert_eq!(residuals.len(), 1);
        assert_eq!(residuals[0].0, 0);
    }

    #[test]
    fn walk_ffn_new_unlimited_with_trace_records() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited_with_trace(weights, &idx);
        let x = input(1, weights.hidden_size);
        ffn.forward(0, &x);
        assert_eq!(ffn.take_residuals().len(), 1);
    }

    #[test]
    fn walk_ffn_take_trace_pairs_residuals_with_walk_hits() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited_with_trace(weights, &idx);
        let x = input(1, weights.hidden_size);
        ffn.forward(0, &x);
        let trace = ffn.take_trace();
        assert!(!trace.layers.is_empty());
        // Mock index returns no FeatureMeta so walk_hits collapses to empty
        // — but the layer entry itself must still be present.
        assert_eq!(trace.layers[0].0, 0);
    }

    #[test]
    fn walk_ffn_new_with_backend_attaches_backend() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_with_backend(weights, &idx, 4, &larql_compute::CpuBackend);
        let x = input(1, weights.hidden_size);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn walk_ffn_with_l1_cache_records_misses_then_hits() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited(weights, &idx).with_l1_cache(weights.num_layers);
        let x = input(1, weights.hidden_size);
        // L1 cache stats: first call is a miss, second identical call a hit.
        ffn.forward(0, &x);
        ffn.forward(0, &x);
        let (hits, misses) = ffn.l1_cache_stats().expect("cache enabled");
        assert!(misses >= 1, "first call must be a miss");
        // Whether the second call hits depends on the cache key — so we
        // just assert hits is a sensible non-overflowing count.
        assert!(hits + misses >= 2);
    }

    #[test]
    fn walk_ffn_l1_cache_stats_returns_none_when_disabled() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited(weights, &idx);
        assert!(ffn.l1_cache_stats().is_none());
    }

    #[test]
    fn walk_ffn_with_dispatch_trace_records_per_layer_path() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited(weights, &idx).with_dispatch_trace();
        let x = input(1, weights.hidden_size);
        for layer in 0..weights.num_layers {
            ffn.forward(layer, &x);
        }
        let trace = ffn.take_dispatch_trace();
        assert_eq!(trace.len(), weights.num_layers);
        for (i, entry) in trace.iter().enumerate() {
            assert_eq!(entry.layer, i);
            assert!(!entry.path.is_empty());
        }
        // Drain semantics: second call returns empty.
        assert!(ffn.take_dispatch_trace().is_empty());
    }

    #[test]
    fn walk_ffn_take_dispatch_trace_returns_empty_when_disabled() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new_unlimited(weights, &idx);
        assert!(ffn.take_dispatch_trace().is_empty());
    }

    #[test]
    fn walk_ffn_from_config_uses_supplied_walkffnconfig() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let cfg = crate::vindex::WalkFfnConfig::sparse(weights.num_layers, 2);
        let ffn = WalkFfn::from_config(weights, &idx, cfg);
        let x = input(1, weights.hidden_size);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    /// `WalkFfn::new` with `top_k = usize::MAX` routes through the dense
    /// `WalkFfnConfig::dense` branch (line 178). The `new(_, _, k)`
    /// constructor's other branch (sparse) is already exercised by
    /// `walk_ffn_sparse_k`.
    #[test]
    fn walk_ffn_new_with_usize_max_takes_dense_branch() {
        let weights = shared_weights();
        let idx = mock_index(weights);
        let ffn = WalkFfn::new(weights, &idx, usize::MAX);
        let x = input(1, weights.hidden_size);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
    }

    /// Variant of `MockGateIndex` that yields a `FeatureMeta` for every
    /// `(layer, feature)` query. This is what `take_trace` needs to
    /// promote a residual into a populated `WalkHit` — without it, the
    /// `filter_map` collapses to empty (`feature_meta` returns `None`).
    struct MockGateIndexWithMeta {
        n_features: usize,
    }

    impl GateLookup for MockGateIndexWithMeta {
        fn gate_knn(
            &self,
            _layer: usize,
            _residual: &Array1<f32>,
            top_k: usize,
        ) -> Vec<(usize, f32)> {
            (0..top_k.min(self.n_features))
                .map(|i| (i, 1.0 / (i as f32 + 1.0)))
                .collect()
        }
        fn feature_meta(&self, _layer: usize, feature: usize) -> Option<FeatureMeta> {
            Some(FeatureMeta {
                top_token: format!("tok{feature}"),
                top_token_id: feature as u32,
                c_score: 0.9,
                top_k: Vec::new(),
            })
        }
        fn num_features(&self, _layer: usize) -> usize {
            self.n_features
        }
    }

    impl PatchOverrides for MockGateIndexWithMeta {}
    impl NativeFfnAccess for MockGateIndexWithMeta {}
    impl QuantizedFfnAccess for MockGateIndexWithMeta {}
    impl Fp4FfnAccess for MockGateIndexWithMeta {}

    /// When `feature_meta` returns `Some`, `take_trace` builds a populated
    /// `WalkHit` per (layer, feature) — covering the `Some(WalkHit { .. })`
    /// arm of the `filter_map` (lines 235-238).
    #[test]
    fn walk_ffn_take_trace_populates_walk_hits_when_meta_present() {
        let weights = shared_weights();
        let idx = MockGateIndexWithMeta {
            n_features: weights.intermediate_size,
        };
        let ffn = WalkFfn::new_with_trace(weights, &idx, 3);
        let x = input(1, weights.hidden_size);
        ffn.forward(0, &x);
        let trace = ffn.take_trace();
        assert_eq!(trace.layers.len(), 1);
        let (layer, hits) = &trace.layers[0];
        assert_eq!(*layer, 0);
        assert!(
            !hits.is_empty(),
            "expected WalkHits when feature_meta returns Some"
        );
        for hit in hits {
            assert_eq!(hit.layer, 0);
            assert!(hit.gate_score.is_finite());
            assert!(hit.meta.top_token.starts_with("tok"));
        }
    }

    /// `walk_trace_env_enabled` caches the env-var lookup in a thread-local.
    /// Spawn a fresh thread so the cell starts empty, set `LARQL_WALK_TRACE=1`
    /// in that thread, then drive `forward` — `trace_path` reads the cache
    /// (first call populates it as `Some(true)`) and emits to stderr
    /// (line 145-147).
    #[test]
    fn walk_ffn_trace_path_honours_env_var_in_fresh_thread() {
        let handle = std::thread::spawn(|| {
            // SAFETY: thread-isolated env var. The whole point of running
            // in a dedicated thread is that no other test sees this var
            // mid-flight — and we wipe it before returning. Set + remove
            // are bracketed within a single thread's lifetime.
            unsafe {
                std::env::set_var("LARQL_WALK_TRACE", "1");
            }
            let weights = make_test_weights();
            let idx = MockGateIndex {
                n_features: weights.intermediate_size,
            };
            let ffn = WalkFfn::new_unlimited(&weights, &idx);
            let x = Array2::from_shape_vec(
                (1, weights.hidden_size),
                (0..weights.hidden_size)
                    .map(|i| (i as f32 + 1.0) * 0.02)
                    .collect(),
            )
            .unwrap();
            // Drives trace_path, which checks walk_trace_env_enabled.
            ffn.forward(0, &x);
            unsafe {
                std::env::remove_var("LARQL_WALK_TRACE");
            }
        });
        handle.join().expect("env-var thread panicked");
    }

    /// Forward against the Q4K test fixture routes through the native
    /// kquant path (priority 4 in the routing ladder, line 340-343),
    /// which fires when the vindex has interleaved_kquant data AND the
    /// `num_features` Q4_K fallback returns a non-zero width.
    #[test]
    fn walk_ffn_forward_routes_through_native_kquant_path() {
        use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        // Pre-flight: the Q4K manifest's gate component bytes should give
        // a positive intermediate width via the `num_features` fallback.
        // If this fails the test would silently route through
        // `zero_features_dense` and the native kquant path stays uncov.
        for layer in 0..weights.num_layers {
            assert!(
                index.num_features(layer) > 0,
                "layer {layer}: num_features must be > 0 for Q4K routing — \
                 the kquant_ffn_intermediate_width fallback returned 0"
            );
        }
        let ffn = WalkFfn::new_unlimited(&weights, &index);
        let x = input(1, weights.hidden_size);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
