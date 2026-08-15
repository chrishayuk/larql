//! `WalkFfn` — FFN backend that replaces dense matmul with vindex lookups.
//!
//! Routing table (priority order, see `forward_routed`):
//!
//! | # | Condition                                            | Path                         |
//! | - | ---------------------------------------------------- | ---------------------------- |
//! | 0 | `seq_len == 1`, L1 cache hit, `Observe::Skip` only   | `l1_cache_hit`               |
//! | 1a| `index.has_overrides_at(layer)` + delta preconditions| `override:base_delta`        |
//! | 1b| `index.has_overrides_at(layer)`                      | `override:sparse`            |
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
//! paths silently lose overridden features). Overridden layers first
//! try the exact base+delta path (`base_delta.rs` — dense base plus
//! O(|patch|) per-slot corrections, 2026-07-30 review item 16), which
//! declines to the walk when its exactness preconditions don't hold.
//! FP4/FP8 is handled by the
//! sparse path because the format is per-feature by construction —
//! there is no batched FP4 dense path on CPU. Q4K/Q4/f32 interleaved
//! are perf-preference ordered. `exact` and `weights_fallback` are
//! correctness baselines that require safetensors weights.
//!
//! Each walk path lives in its own module under this directory:
//!
//! - `base_delta.rs`              — exact base+delta execution for patched layers
//! - `sparse.rs`                  — per-feature walk, unified ffn_row_* dispatch
//!   (`sparse_gemv.rs` full-K gemv, `sparse_route.rs` hit selection,
//!   `sparse_parallel.rs` parallel Q4K down, `sparse_gather.rs` gather kernel)
//! - `interleaved.rs`             — f32 interleaved mmap, three BLAS gemms
//! - `interleaved_q4.rs`            — Q4_0 interleaved, CPU kernel / Metal Q4
//! - `interleaved_kquant_native.rs` — K-quant direct matvec, no dequant cache
//! - `interleaved_kquant_dequant.rs`— K-quant dequant, full f32 dense after decode
//! - `full_mmap.rs`               — gate/up/down in three separate mmap files
//! - `exact.rs`                   — gate/up from safetensors, down from mmap
//! - `observe.rs`                 — Observe mode (forward / forward_observed split)
//! - `trace.rs`                   — runtime trace records + take_trace rebuild
//! - `helpers.rs`                 — cross-path utilities + trace metadata
//! - `builders.rs`                — constructors and builder methods
//! - `timings.rs`                 — phase-timing sink + env-var trace plumbing
//!
//! - `plan.rs` / `planner.rs`      — explicit execution plan + the planning half
//!   of the ladder (2026-07-30 review, item 18)
//!
//! Adding a new storage format should almost never touch `mod.rs` — add
//! a new module with a single walk function, one rung in the planner
//! (`planner.rs`), one arm in `execute_plan`, and a unit test in
//! `routing_tests.rs` / `plan_tests.rs`.
//!
//! Planning (2026-07-30 review, item 18, `plan.rs` + `planner.rs`):
//! path selection is an explicit decision value. `plan_layer` walks
//! the CONDITIONS above and returns an [`FfnPlan`] — the rung that
//! will run plus a structured [`PlanReason`] (why this rung, why not
//! each rung above it, which thresholds fired); `forward_ladder` then
//! just executes the plan. A path that declines mid-execution
//! (returns `None`) triggers a re-plan with that rung excluded, and
//! the eventual plan's reason records the decline — inspect it via
//! [`WalkFfn::plan_for`] (pure, no side effects) or the runtime
//! trace's `plan_reason` field.
//!
//! Observation (2026-07-30 review, item 15): `forward` runs every path
//! in `Observe::Skip` mode, which by construction never allocates or
//! fills an activation buffer; `forward_observed` runs `Observe::Record`
//! and returns an honest [`FfnActivations`] — sparse paths emit exactly
//! the `(feature, activation)` pairs they computed, dense paths hand
//! over their intrinsic activation matrix, and the L1 cache read is
//! bypassed (recompute) rather than fabricating zeros.
//!
//! Runtime trace (2026-07-30 review, item 17, `trace.rs`): enabling
//! `with_trace` upgrades every call to `Observe::Record` and folds the
//! executed path's observation into per-(position, layer) records at
//! the ladder exit — `take_trace` reports the features the walk
//! EXECUTED (selector/pool/cell-router included), not a post-hoc
//! `gate_knn` re-run. Tracing therefore shares Record-mode semantics:
//! the L1 cache read is bypassed while the trace is on.

use ndarray::Array2;

use crate::ffn::sparse_compute::{sparse_ffn_forward, sparse_ffn_forward_observed};
use crate::ffn::{FfnActivations, FfnBackend};
use crate::model::ModelWeights;
use crate::vindex::l1_cache::FfnL1Cache;
use crate::vindex::walk_config::WalkFfnConfig;
use larql_compute::prelude::*;
use observe::Observe;

use larql_vindex::GateIndex;

mod base_delta;
mod builders;
mod compact_dense;
mod exact;
mod full_mmap;
mod helpers;
mod interleaved;
mod interleaved_kquant_dequant;
mod interleaved_kquant_native;
mod interleaved_q4;
mod observe;
mod plan;
mod planner;
mod selector;
mod shortlist;
mod sparse;
mod sparse_gather;
mod sparse_gemv;
mod sparse_parallel;
mod sparse_route;
mod thresholds;
mod timings;
mod trace;

#[cfg(test)]
mod base_delta_tests;
#[cfg(test)]
mod dispatch_tests;
#[cfg(test)]
mod plan_tests;
#[cfg(test)]
mod routing_tests;
#[cfg(test)]
mod shortlist_tests;
#[cfg(test)]
mod trace_tests;

pub use compact_dense::CompactDenseLayer;
pub use helpers::DispatchEntry;
pub use plan::{FfnPlan, PlanReason, PlanRung, SkippedRung, ThresholdCheck};
pub use timings::PhaseTimingsHandle;
pub use trace::{LayerTraceRecord, TracedFeature};

use plan::PlanExclusions;

pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn GateIndex,
    pub config: WalkFfnConfig,
    pub backend: Option<&'a dyn ComputeBackend>,
    trace_residuals: std::cell::RefCell<Vec<(usize, Vec<f32>)>>,
    record_trace: bool,
    /// Runtime trace sink (2026-07-30 review, item 17). Populated by
    /// `fold_runtime_trace` at every routing-ladder exit when
    /// `record_trace` is set — the records describe the EXECUTED path,
    /// not a post-hoc `gate_knn` re-run. Drained by `take_trace` /
    /// `take_runtime_trace` (`trace.rs`).
    runtime_trace: std::cell::RefCell<Vec<trace::LayerTraceRecord>>,
    /// Name of the most recent `trace_path` call — the executed ladder
    /// path for the current layer call by the time the trace folds.
    last_path: std::cell::Cell<&'static str>,
    l1_cache: Option<FfnL1Cache>,
    /// Dispatch-trace sink. `None` = disabled. When `Some`, every walk
    /// path appends a (layer, name) entry on exit. Used by the routing
    /// unit tests and by the env-var dispatch trace for Q2 debugging.
    dispatch_trace: std::cell::RefCell<Option<Vec<DispatchEntry>>>,
    /// Phase-timing sink for `sparse:parallel_q4k_down`. `None` =
    /// disabled. When `Some`, the branch records cache_fetch / scan /
    /// reduce timings via atomic adds.
    pub(super) phase_timings: Option<std::sync::Arc<PhaseTimingsHandle>>,
    /// Lazy cache of per-feature `‖down_row‖` per layer. Built on first
    /// use when the selector is `GateXDownNorm` or `GateXUpDownNorm`.
    pub(super) down_norms_cache: std::cell::RefCell<Vec<Option<std::sync::Arc<Vec<f32>>>>>,
    /// Lazy cache of per-feature `‖up_row‖` per layer. Built on first
    /// use when the selector is `GateXUpDownNorm`.
    pub(super) up_norms_cache: std::cell::RefCell<Vec<Option<std::sync::Arc<Vec<f32>>>>>,
    /// Count of joint-selector calls that silently degraded to the
    /// production GateOnly chain (missing batched scores or norms).
    /// A non-zero count on an A/B sweep means the labelled selector
    /// was not the selector that ran (2026-07-30 review, M10). Also
    /// surfaced per-call as a `selector:fallback` dispatch-trace entry.
    pub(super) selector_fallbacks: std::cell::Cell<u64>,
    /// Count of configured two-stage shortlist selections that
    /// declined to single-stage (M < K, `Random` selector, or stage-2
    /// inputs unavailable — 2026-07-30 review, item 19). Non-zero
    /// means at least one layer call did NOT get the O(M·d)
    /// shortlist cost shape the config claims. Also surfaced per-call
    /// as a `shortlist:declined` dispatch-trace entry.
    pub(super) shortlist_declines: std::cell::Cell<u64>,
}

impl<'a> WalkFfn<'a> {
    fn top_k_for(&self, layer: usize) -> usize {
        self.config
            .k_for(layer)
            .unwrap_or_else(|| self.index.num_features(layer))
    }

    /// Number of joint-selector calls that degraded to the production
    /// GateOnly chain so far. Check this after an A/B sweep: non-zero
    /// means the sweep's selector label lies for at least some calls.
    pub fn selector_fallback_count(&self) -> u64 {
        self.selector_fallbacks.get()
    }

    /// Number of configured two-stage shortlist selections that
    /// declined to single-stage so far (item 19). Check after a run
    /// with `shortlist_m` set: non-zero means some calls paid the
    /// single-stage full-projection cost, not the shortlist's O(M·d).
    pub fn shortlist_decline_count(&self) -> u64 {
        self.shortlist_declines.get()
    }

    pub fn l1_cache_stats(&self) -> Option<(u64, u64)> {
        self.l1_cache.as_ref().map(|c| (c.hits(), c.misses()))
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
        self.last_path.set(path);
        if let Some(vec) = self.dispatch_trace.borrow_mut().as_mut() {
            vec.push(DispatchEntry { layer, path });
        }
        if timings::walk_trace_env_enabled() {
            eprintln!("[walk_ffn] L{layer} → {path}");
        }
    }

    pub fn take_residuals(&self) -> Vec<(usize, Vec<f32>)> {
        self.trace_residuals.borrow_mut().drain(..).collect()
    }

    /// Non-draining snapshot of the residuals captured so far. Used by the
    /// early-exit path to inspect the stored-layer residuals at the resolved
    /// layer *mid-forward* (after layer L*, before the tail runs) without
    /// consuming the trace — the full forward, if it continues on a miss,
    /// keeps appending.
    pub fn peek_residuals(&self) -> Vec<(usize, Vec<f32>)> {
        self.trace_residuals.borrow().clone()
    }

    /// Ladder steps 4–9 — the whole-layer, override-blind paths, in
    /// priority order. Returns `None` when every step declines (caller
    /// falls through to `weights_fallback`).
    ///
    /// Factored out of `forward_with_activation` so the base+delta
    /// path (`base_delta.rs`) can produce its dense base through the
    /// EXACT same code the unpatched ladder runs — base numerics
    /// identical to the unpatched model by construction (2026-07-30
    /// review, item 16 exactness condition 4). On a patched index
    /// these paths read only base bytes, so their output is the
    /// pre-patch dense result — which is precisely what base+delta
    /// needs to correct, and precisely why the ordinary override arm
    /// must never land here directly.
    fn forward_unpatched_whole_layer(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        for rung in planner::WHOLE_LAYER_RUNGS {
            if !self.whole_layer_rung_available(rung) {
                continue;
            }
            if let Some(r) = self.execute_whole_layer_rung(rung, layer, x) {
                return Some(r);
            }
        }
        None
    }

    /// Run ONE whole-layer rung (4–9). `None` = the path declined —
    /// the caller falls through (`forward_unpatched_whole_layer`) or
    /// re-plans (`forward_ladder`). Rung priority and availability
    /// conditions live in `planner.rs` (`WHOLE_LAYER_RUNGS` /
    /// `whole_layer_rung_available`); rationale for the ordering is in
    /// the module doc's routing table.
    fn execute_whole_layer_rung(
        &self,
        rung: PlanRung,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        debug_assert!(
            planner::WHOLE_LAYER_RUNGS.contains(&rung),
            "not a whole-layer rung: {rung:?}"
        );
        match rung {
            PlanRung::KquantNative => self.walk_ffn_kquant_native(layer, x),
            PlanRung::InterleavedQ4 => self.walk_ffn_q4_interleaved(layer, x),
            PlanRung::InterleavedF32 => self.walk_ffn_interleaved(layer, x),
            PlanRung::FullMmap => self.walk_ffn_full_mmap(layer, x),
            PlanRung::KquantDequant => self.walk_ffn_kquant_dequant(layer, x),
            PlanRung::Exact => Some(self.walk_ffn_exact(layer, x)),
            // Guarded by the debug_assert above; nothing to execute.
            _ => None,
        }
    }

    /// Ladder step 10 — sparse matmul against safetensors weights, and
    /// the only path besides the sparse walk that honours feature
    /// overrides. Doubles as the hard fallback for overridden layers
    /// whose sparse walk fails: every whole-layer ladder path is
    /// override-blind, so an overridden layer must land here, never
    /// fall through (2026-07-30 review, finding M7).
    fn weights_fallback(
        &self,
        layer: usize,
        x: &Array2<f32>,
        observe: Observe,
    ) -> (Array2<f32>, Option<FfnActivations>) {
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
            return if observe.recording() {
                let (out, obs) = crate::ffn::sparse_ffn_forward_with_full_overrides_observed(
                    self.weights,
                    layer,
                    x,
                    &features,
                    &slot_overrides,
                );
                (out, Some(obs))
            } else {
                (
                    crate::ffn::sparse_ffn_forward_with_full_overrides(
                        self.weights,
                        layer,
                        x,
                        &features,
                        &slot_overrides,
                    ),
                    None,
                )
            };
        }
        self.trace_path(layer, "weights_fallback:sparse");
        if observe.recording() {
            let (out, obs) = sparse_ffn_forward_observed(self.weights, layer, x, &features);
            (out, Some(obs))
        } else {
            (sparse_ffn_forward(self.weights, layer, x, &features), None)
        }
    }
}

impl<'a> WalkFfn<'a> {
    /// Route one layer, then — when the runtime trace is enabled —
    /// fold the executed path's observation into the trace sink
    /// (2026-07-30 review, item 17). Tracing upgrades `Skip` to
    /// `Record` so the sparse kernels emit their per-feature records;
    /// that also bypasses the L1 cache read (a hit observes nothing),
    /// exactly like an ordinary observed call.
    fn forward_routed(
        &self,
        layer: usize,
        x: &Array2<f32>,
        observe: Observe,
    ) -> (Array2<f32>, Option<FfnActivations>) {
        let observe = if self.record_trace {
            Observe::Record
        } else {
            observe
        };
        let (result, plan) = self.forward_ladder(layer, x, observe);
        if self.record_trace {
            self.fold_runtime_trace(layer, &result.0, result.1.as_ref(), &plan);
        }
        result
    }

    /// The routing ladder, parameterised on observation mode. Both trait
    /// entry points run THIS body, so `forward` and `forward_observed`
    /// can never route differently — the only mode-dependent behaviour
    /// is whether activations are recorded and whether the L1 cache
    /// read is consulted.
    ///
    /// Selection is the planner's (`plan_layer`, item 18): this body
    /// only performs the L1 probe (the one side-effecting input the
    /// planner must not own — hit/miss stats), executes the planned
    /// rung, and re-plans with the rung excluded when a fallible path
    /// declines. Returns the executed plan alongside the result so the
    /// runtime trace can record the reason.
    ///
    /// Invariant: returns `Some` observation iff `observe` is `Record`
    /// — except the `Skip`-only L1 hit, which returns `None` trivially.
    fn forward_ladder(
        &self,
        layer: usize,
        x: &Array2<f32>,
        observe: Observe,
    ) -> ((Array2<f32>, Option<FfnActivations>), FfnPlan) {
        let num_features = self.index.num_features(layer);
        let has_overrides = self.index.has_overrides_at(layer);
        let seq_len = x.shape()[0];

        if num_features > 0 && self.record_trace {
            let last_row = x.row(seq_len - 1).to_vec();
            self.trace_residuals.borrow_mut().push((layer, last_row));
        }

        // L1 cache probe — single-position, non-overridden layers only
        // (patched layers must bypass the cache: its entries are
        // override-blind and can never be invalidated by the override
        // machinery). Key is a path-independent hash of the residual,
        // so any walk path that produces the same output fills the
        // same slot. The read is Skip-mode only: the cache stores
        // outputs, so a hit has nothing honest to say about
        // activations — an observed call recomputes and refreshes the
        // slot (2026-07-30 review, item 15). Exactly one `get` per
        // eligible call, preserving hit/miss accounting.
        let l1_key =
            (num_features > 0 && !has_overrides && seq_len == 1 && self.l1_cache.is_some())
                .then(|| planner::residual_l1_key(x));
        let l1_row: Option<Vec<f32>> = if observe.recording() {
            None
        } else {
            l1_key.and_then(|key| self.l1_cache.as_ref().and_then(|c| c.get(layer, key)))
        };

        let mut exclude = PlanExclusions::default();
        loop {
            let plan = self.plan_layer(layer, x, observe, l1_row.is_some(), exclude);
            let rung = plan.rung();
            match self.execute_plan(rung, layer, x, observe, l1_row.as_deref()) {
                Some(result) => {
                    // Fill the L1 slot from the fallthrough-ladder
                    // rungs only — a hit re-inserting itself is a
                    // pointless write, and override/zero-feature plans
                    // never have a key.
                    if rung != PlanRung::L1CacheHit {
                        if let (Some(key), Some(cache)) = (l1_key, self.l1_cache.as_ref()) {
                            cache.insert(layer, key, result.0.row(0).to_vec());
                        }
                    }
                    return (result, plan);
                }
                // Runtime decline: honest re-plan from the next rung.
                // The next plan's `skipped` list carries this rung
                // with a "declined at execution" reason.
                None => exclude = exclude.with(rung),
            }
        }
    }

    /// Execute one planned rung. `None` = the path declined at
    /// runtime; `forward_ladder` re-plans. Selection logic lives in
    /// the planner — this match is intentionally condition-free.
    fn execute_plan(
        &self,
        rung: PlanRung,
        layer: usize,
        x: &Array2<f32>,
        observe: Observe,
        l1_row: Option<&[f32]>,
    ) -> Option<(Array2<f32>, Option<FfnActivations>)> {
        match rung {
            PlanRung::ZeroFeaturesDense => {
                self.trace_path(layer, "zero_features_dense");
                let dense_ffn = crate::ffn::WeightFfn {
                    weights: self.weights,
                };
                Some(if observe.recording() {
                    let (out, obs) = dense_ffn.forward_observed(layer, x);
                    (out, Some(obs))
                } else {
                    (dense_ffn.forward(layer, x), None)
                })
            }
            PlanRung::L1CacheHit => {
                let row = l1_row.expect("L1CacheHit is only planned when the probe returned a row");
                let hidden = x.shape()[1];
                let mut out = Array2::<f32>::zeros((1, hidden));
                out.row_mut(0).assign(&ndarray::ArrayView1::from(row));
                self.trace_path(layer, "l1_cache_hit");
                Some((out, None))
            }
            // 1a. Exact base+delta (2026-07-30 review, item 16): dense
            //     base through the unpatched whole-layer ladder, plus
            //     O(|patch|) per-slot corrections. Declines when its
            //     exactness preconditions don't hold mid-execution.
            PlanRung::OverrideBaseDelta => self.walk_ffn_base_delta(layer, x, observe),
            // 1b / 2 / 3 all run the sparse walk (FP4/FP8 is handled
            // by the unified ffn_row_* dispatch — that transparency is
            // the point of the trait refactor). The sparse path calls
            // trace_path itself; its name carries the specialisation.
            PlanRung::OverrideSparse | PlanRung::Sparse | PlanRung::Fp4Sparse => {
                self.walk_ffn_sparse(layer, x, observe)
            }
            // 4–9. Whole-layer paths (shared with base+delta's base).
            //      Their activation matrix is an intrinsic intermediate
            //      (the down projection consumes it), so `Skip` merely
            //      drops it — no extra allocation either way.
            PlanRung::KquantNative
            | PlanRung::InterleavedQ4
            | PlanRung::InterleavedF32
            | PlanRung::FullMmap
            | PlanRung::KquantDequant
            | PlanRung::Exact => self
                .execute_whole_layer_rung(rung, layer, x)
                .map(|(out, act)| (out, observe.dense(act))),
            // 10. Last resort — and the only override-aware path
            //     besides the sparse walk, so overridden layers whose
            //     walk declines are PLANNED here directly (never fall
            //     through override-blind paths).
            PlanRung::WeightsFallback => Some(self.weights_fallback(layer, x, observe)),
        }
    }
}

impl<'a> FfnBackend for WalkFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        // `Observe::Skip` — by construction no walk path touches an
        // activation buffer on this route (unless the runtime trace is
        // enabled, which deliberately upgrades to `Record` so the trace
        // can report the executed features — see `forward_routed`).
        self.forward_routed(layer, x, Observe::Skip).0
    }

    fn forward_observed(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, FfnActivations) {
        let (out, obs) = self.forward_routed(layer, x, Observe::Record);
        (
            out,
            obs.expect("forward_routed(Record) always yields an observation"),
        )
    }

    fn name(&self) -> &str {
        "walk"
    }
}
