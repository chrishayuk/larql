//! walk_path_audit — per-path equivalence harness for WalkFfn dispatch paths.
//!
//! For each path the live vindex makes available, force dispatch via a
//! `MaskedGateIndex` wrapper and compare every FFN layer's output against
//! `WeightFfn` (dense matmul reference). Aggregates per-path stats across a
//! small fixed prompt corpus (anchor + factual + code). Emits markdown +
//! JSON artifacts and exits non-zero on bound violations.
//!
//! Assertion metrics are **cos** and **relative L2** (`L2 / ‖primary‖`),
//! both magnitude-invariant. Absolute L2 and max-element drift are kept in
//! the per-layer table for diagnosis (e.g. surfacing residual-magnitude
//! outliers like the L11/code/1 ` fibonacci` spike on Gemma 3 4B), but are
//! not what the gate fires on.
//!
//! Opening bounds (overridable per-path via the `bound_for` table). Each
//! cosine floor is set one decimal less precise than the measured worst on
//! the canonical baseline — tight enough to catch a real regression, loose
//! enough to survive an Accelerate point release reordering FMAs:
//!
//!   - exact paths (interleaved, full_mmap, exact):            cos ≥ 0.99999, rel_L2 ≤ 1e-2
//!   - quantized (interleaved_kquant:dequant, interleaved_q4):    cos ≥ 0.99,    rel_L2 ≤ 5e-3
//!   - fp4 (fp4_storage:sparse):                               cos ≥ 0.98,    rel_L2 ≤ 1e-2
//!
//! `rel_L2` opens loose; tighten to `measured_worst × 4` per path in a
//! follow-up PR after first-baseline measurements land.
//!
//! Plus, for every path: top-1 token match on each prompt + Paris probability
//! within 5e-3 of dense.
//!
//! `weights_fallback` is **not** in this audit — it's the no-vindex-data
//! corner case and at any finite K it's measuring approximation quality
//! rather than path equivalence. That belongs in a separate
//! `walk_approximation_quality` example that sweeps K.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example walk_path_audit -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex /path/to/gemma3-4b.vindex \
//!     [--out-md walk_path_audit.md] [--out-json walk_path_audit.json]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::{Array1, Array2};

use larql_inference::{
    predict, predict_with_ffn,
    vindex::{WalkFfn, WalkFfnConfig},
    FfnBackend, InferenceModel, WeightFfn,
};
use larql_vindex::{
    FeatureMeta, FfnRowAccess, Fp4FfnAccess, GateIndex, GateLookup, NativeFfnAccess,
    PatchOverrides, QuantizedFfnAccess, SilentLoadCallbacks, StorageBucket, VectorIndex,
};

// ── Corpus ─────────────────────────────────────────────────────────────

/// Three prompts: a Paris-style anchor that matches the April measurement
/// and the bench corpus, a mid-length factual to vary the residual content,
/// and a code fragment to push the walk into FFN features the natural-
/// language prompts don't touch. Aggregating max L2 across all of them
/// gives worst-case drift; averaging would hide exactly what we're
/// trying to catch.
const PROMPTS: &[(&str, &str)] = &[
    ("paris", "The capital of France is"),
    (
        "apollo",
        "The Apollo 11 mission landed on the Moon on July 20, 1969. The commander was",
    ),
    ("code", "def fibonacci(n):"),
];

const PARIS_KEY: &str = "paris";

// ── Bounds ─────────────────────────────────────────────────────────────

/// Per-path assertion floor. Both metrics are magnitude-invariant.
///
/// The baseline rule for `min_cos`: take the measured worst across the
/// canonical-vindex run and back off one decimal place. On Gemma 3 4B f16
/// (1326 obs / 3 prompts × 34 layers / 13 avg pos): worst measured cos =
/// 0.999996 → floor 0.99999. Tight enough to catch a real regression,
/// loose enough that an Accelerate point release shuffling an FMA doesn't
/// red CI.
///
/// `rel_l2` opens generous on first commit because we don't have a per-path
/// measurement yet; tighten to `measured_worst × 4` in the follow-up PR.
#[derive(Clone, Copy, Debug)]
struct PathBound {
    /// Bucket label — surfaced in the markdown summary header.
    kind: &'static str,
    /// Per-observation cos floor. Min cos across all (layer, prompt, pos)
    /// observations must be ≥ this.
    min_cos: f32,
    /// Per-observation rel_L2 ceiling, where rel_L2 = L2 / max(‖primary‖, EPS).
    /// Magnitude-invariant; doesn't blow up on outlier-magnitude residuals.
    rel_l2: f32,
    /// End-to-end gate on the Paris-anchor prompt: |walk_prob − dense_prob|
    /// at the top-1 token must be ≤ this. The Paris prompt is the
    /// fixed sampler-stability check across all paths; per-bucket budgets
    /// reflect that quantized/FP4 paths can drift further on softmax
    /// while still preserving model behavior (top-1 + ranking).
    paris_prob_budget: f64,
}

const BOUND_EXACT: PathBound = PathBound {
    kind: "exact",
    min_cos: 0.99999,
    // rel_L2 floor 1e-2 is intentionally loose pending measure-then-tighten
    // across Q4K/FP4 paths; canonical f16 measurement on Gemma 3 4B is
    // 1.881e-3 (worst at L32/paris/0), target post-matrix tightening ~7.5e-3
    // (= measured × 4). Don't tighten this in isolation — wait until the
    // Q4K and FP4 baselines land and apply the same rule per bucket.
    rel_l2: 1e-2,
    paris_prob_budget: 5e-3,
};

const BOUND_QUANTIZED: PathBound = PathBound {
    kind: "quantized",
    min_cos: 0.99,
    // Quantized rel_L2 ceiling is loose by design — cos is the meaningful
    // assertion for this bucket. The two metrics aren't independent: for
    // similar-magnitude vectors, rel_L2 ≈ √(2(1-cos)), so cos = 0.99
    // implies rel_L2 ≈ 0.14, and the f16-style 1e-2 ceiling would be
    // mathematically impossible here. Canonical Q4K measurement on Gemma
    // 3 4B is rel_L2 = 1.205e-1 (worst at L10/code/1, interleaved_kquant
    // path); 4× headroom puts the ceiling at ~5e-1. See
    // walk_path_audit_gemma3_4b_q4k_baseline.md for the derivation.
    rel_l2: 5e-1,
    // Matches walk_correctness.rs Q4K-down threshold (0.035) with margin
    // for prompts more sensitive to softmax redistribution than Paris.
    // If walk_correctness later tightens its Q4K-down gate, revisit this
    // budget so the two thresholds stay in sync.
    paris_prob_budget: 5e-2,
};

const BOUND_FP4: PathBound = PathBound {
    kind: "fp4",
    min_cos: 0.98,
    rel_l2: 1e-2,
    // Provisional pending FP4 baseline measurement on
    // gemma3-4b-fp4a.vindex; same reasoning as quantized — FP4 dequant
    // moves softmax further than f16-class noise. Tighten via
    // measure-then-tighten when the FP4 baseline lands.
    paris_prob_budget: 5e-2,
};

/// Map a [`StorageBucket`] to its assertion bound. This is the source of
/// truth for "what's the right floor for this bucket"; paths set their
/// `bound` field by calling this on the bucket they're walking against.
fn bound_for_bucket(bucket: StorageBucket) -> PathBound {
    match bucket {
        StorageBucket::Exact => BOUND_EXACT,
        StorageBucket::Quantized => BOUND_QUANTIZED,
        StorageBucket::Fp4 => BOUND_FP4,
    }
}

/// Fallback only — prefer `PathSpec.bound` (set explicitly per spec in
/// `enumerate_paths`). Kept as a path-name → default-bucket primitive in
/// case a future caller needs to look up a bucket without a `PathSpec`.
/// Loose prefix-matching so paths with sub-labels (`sparse:gemv_full_k`,
/// `interleaved_q4:metal`, …) all land on the right bucket.
#[allow(dead_code)]
fn bound_for(path: &str) -> PathBound {
    if path.starts_with("fp4_storage") {
        BOUND_FP4
    } else if path.starts_with("interleaved_kquant") || path.starts_with("interleaved_q4") {
        BOUND_QUANTIZED
    } else {
        BOUND_EXACT
    }
}

/// Floor for the divisor in `rel_L2 = L2 / max(‖primary‖, EPS)`. Prevents a
/// near-zero residual at e.g. position 0 (BOS) from producing a misleading
/// rel_L2 = nonzero / ~0. Below this magnitude cos is the more robust
/// metric anyway.
const REL_L2_NORM_EPS: f32 = 1e-6;

// ── CLI ────────────────────────────────────────────────────────────────

struct Args {
    model: String,
    vindex: PathBuf,
    out_md: Option<PathBuf>,
    out_json: Option<PathBuf>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut vindex = PathBuf::new();
    let mut out_md: Option<PathBuf> = None;
    let mut out_json: Option<PathBuf> = None;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                i += 1;
                model = argv[i].clone();
            }
            "--vindex" => {
                i += 1;
                vindex = PathBuf::from(&argv[i]);
            }
            "--out-md" => {
                i += 1;
                out_md = Some(PathBuf::from(&argv[i]));
            }
            "--out-json" => {
                i += 1;
                out_json = Some(PathBuf::from(&argv[i]));
            }
            _ => {}
        }
        i += 1;
    }

    if model.is_empty() || !vindex.is_dir() {
        eprintln!(
            "Usage: walk_path_audit --model MODEL --vindex PATH \\\n\
             \t[--out-md walk_path_audit.md] [--out-json walk_path_audit.json]"
        );
        std::process::exit(1);
    }

    Args {
        model,
        vindex,
        out_md,
        out_json,
    }
}

// ── MaskedGateIndex ────────────────────────────────────────────────────

/// Newtype wrapper that selectively reports availability flags as `false`,
/// forcing the WalkFfn dispatcher down a specific path. Data methods are
/// pure delegations; only the `has_*` booleans are masked.
///
/// Soundness: verified against every walk path in
/// `crates/larql-inference/src/vindex/walk_ffn/*.rs`. Each path gates on a
/// `has_*` flag at the dispatcher *and* early-exits on `Option::None` from
/// data methods, so masking is sufficient — we don't need to override data.
/// The unified `ffn_row_*` default impls also re-check `has_*` on `self`,
/// which is us, so the mask cascades through the row-level dispatch too.
#[derive(Default, Clone, Copy, Debug)]
struct PathMask {
    hide_fp4: bool,
    hide_q4: bool,
    hide_interleaved: bool,
    hide_full_mmap: bool,
    hide_q4k: bool,
    hide_down_features: bool,
}

struct MaskedGateIndex<'a> {
    inner: &'a dyn GateIndex,
    mask: PathMask,
}

impl<'a> GateLookup for MaskedGateIndex<'a> {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.inner.gate_knn(layer, residual, top_k)
    }
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.inner.feature_meta(layer, feature)
    }
    fn num_features(&self, layer: usize) -> usize {
        self.inner.num_features(layer)
    }

    fn gate_scores_batch(&self, l: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        self.inner.gate_scores_batch(l, x)
    }
    fn gate_scores_batch_backend(
        &self,
        l: usize,
        x: &Array2<f32>,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Array2<f32>> {
        self.inner.gate_scores_batch_backend(l, x, backend)
    }
    fn gate_knn_q4(
        &self,
        l: usize,
        residual: &Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(usize, f32)>> {
        self.inner.gate_knn_q4(l, residual, top_k, backend)
    }
    fn gate_walk(
        &self,
        l: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        self.inner.gate_walk(l, residual, top_k)
    }
}

impl<'a> PatchOverrides for MaskedGateIndex<'a> {
    fn down_override(&self, l: usize, f: usize) -> Option<&[f32]> {
        self.inner.down_override(l, f)
    }
    fn up_override(&self, l: usize, f: usize) -> Option<&[f32]> {
        self.inner.up_override(l, f)
    }
    fn gate_override(&self, l: usize, f: usize) -> Option<&[f32]> {
        self.inner.gate_override(l, f)
    }
    fn has_overrides_at(&self, layer: usize) -> bool {
        self.inner.has_overrides_at(layer)
    }
}

impl<'a> NativeFfnAccess for MaskedGateIndex<'a> {
    fn has_down_features(&self) -> bool {
        !self.mask.hide_down_features && self.inner.has_down_features()
    }
    fn down_feature_vector(&self, l: usize, f: usize) -> Option<&[f32]> {
        self.inner.down_feature_vector(l, f)
    }
    fn down_layer_matrix(&self, l: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.down_layer_matrix(l)
    }
    fn up_layer_matrix(&self, l: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.up_layer_matrix(l)
    }
    fn has_full_mmap_ffn(&self) -> bool {
        !self.mask.hide_full_mmap && self.inner.has_full_mmap_ffn()
    }
    fn has_interleaved(&self) -> bool {
        !self.mask.hide_interleaved && self.inner.has_interleaved()
    }
    fn interleaved_gate(&self, l: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.interleaved_gate(l)
    }
    fn interleaved_up(&self, l: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.interleaved_up(l)
    }
    fn interleaved_down(&self, l: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.interleaved_down(l)
    }
    fn prefetch_interleaved_layer(&self, l: usize) {
        self.inner.prefetch_interleaved_layer(l)
    }
}

impl<'a> QuantizedFfnAccess for MaskedGateIndex<'a> {
    fn has_interleaved_q4(&self) -> bool {
        !self.mask.hide_q4 && self.inner.has_interleaved_q4()
    }
    fn interleaved_q4_gate(&self, l: usize) -> Option<ndarray::Array2<f32>> {
        self.inner.interleaved_q4_gate(l)
    }
    fn interleaved_q4_up(&self, l: usize) -> Option<ndarray::Array2<f32>> {
        self.inner.interleaved_q4_up(l)
    }
    fn interleaved_q4_down(&self, l: usize) -> Option<ndarray::Array2<f32>> {
        self.inner.interleaved_q4_down(l)
    }
    fn prefetch_interleaved_q4_layer(&self, l: usize) {
        self.inner.prefetch_interleaved_q4_layer(l)
    }
    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        self.inner.interleaved_q4_mmap_ref()
    }
    fn has_interleaved_kquant(&self) -> bool {
        !self.mask.hide_q4k && self.inner.has_interleaved_kquant()
    }
    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        self.inner.interleaved_kquant_mmap_ref()
    }
    fn prefetch_interleaved_kquant_layer(&self, l: usize) {
        self.inner.prefetch_interleaved_kquant_layer(l)
    }
    fn interleaved_kquant_layer_data(&self, l: usize) -> Option<[(&[u8], &str); 3]> {
        self.inner.interleaved_kquant_layer_data(l)
    }
    fn has_down_features_kquant(&self) -> bool {
        self.inner.has_down_features_kquant()
    }
    fn kquant_ffn_layer(&self, l: usize, c: usize) -> Option<std::sync::Arc<Vec<f32>>> {
        self.inner.kquant_ffn_layer(l, c)
    }
    fn kquant_ffn_row_into(&self, l: usize, c: usize, f: usize, out: &mut [f32]) -> bool {
        self.inner.kquant_ffn_row_into(l, c, f, out)
    }
    fn kquant_ffn_row_dot(&self, l: usize, c: usize, f: usize, x: &[f32]) -> Option<f32> {
        self.inner.kquant_ffn_row_dot(l, c, f, x)
    }
    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        l: usize,
        c: usize,
        f: usize,
        a: f32,
        out: &mut [f32],
    ) -> bool {
        self.inner
            .kquant_ffn_row_scaled_add_via_cache(l, c, f, a, out)
    }
    fn kquant_ffn_row_scaled_add(
        &self,
        l: usize,
        c: usize,
        f: usize,
        a: f32,
        out: &mut [f32],
    ) -> bool {
        self.inner.kquant_ffn_row_scaled_add(l, c, f, a, out)
    }
    fn kquant_down_feature_scaled_add(&self, l: usize, f: usize, a: f32, out: &mut [f32]) -> bool {
        self.inner.kquant_down_feature_scaled_add(l, f, a, out)
    }
    fn kquant_matmul_transb(
        &self,
        l: usize,
        c: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        self.inner.kquant_matmul_transb(l, c, x, x_rows, backend)
    }
}

impl<'a> Fp4FfnAccess for MaskedGateIndex<'a> {
    fn has_fp4_storage(&self) -> bool {
        !self.mask.hide_fp4 && self.inner.has_fp4_storage()
    }
    fn fp4_ffn_row_dot(&self, l: usize, c: usize, f: usize, x: &[f32]) -> Option<f32> {
        self.inner.fp4_ffn_row_dot(l, c, f, x)
    }
    fn fp4_ffn_row_scaled_add(
        &self,
        l: usize,
        c: usize,
        f: usize,
        a: f32,
        out: &mut [f32],
    ) -> bool {
        self.inner.fp4_ffn_row_scaled_add(l, c, f, a, out)
    }
    fn fp4_ffn_row_into(&self, l: usize, c: usize, f: usize, out: &mut [f32]) -> bool {
        self.inner.fp4_ffn_row_into(l, c, f, out)
    }
}

// ── Path catalog ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct PathSpec {
    /// Display name; matches the dispatch trace label prefix.
    name: &'static str,
    /// Mask to apply on top of the live vindex flags.
    mask: PathMask,
    /// Sparse-K config (`Some`) or dense ladder (`None`).
    sparse_k: Option<usize>,
    /// Assertion bound for this path. Set explicitly per spec — for paths
    /// whose precision is fixed by the path itself (e.g. `interleaved` is
    /// always f32; `interleaved_kquant` is always Q4K), this is hardcoded to
    /// the right bucket. For `sparse`, which dispatches through the
    /// unified `ffn_row_*` chain and walks whatever data the vindex
    /// carries, the bucket is determined by `index.primary_storage_bucket()`.
    bound: PathBound,
}

/// Probe the live vindex and return the paths that are actually testable.
/// Q4 metal/CPU and fp4 paths only show up when the corresponding flag is
/// set on the underlying index — skip them silently otherwise.
fn enumerate_paths(index: &VectorIndex) -> Vec<PathSpec> {
    let mut out = Vec::new();

    // sparse:* — config-forced walk_ffn_sparse over whatever the unified
    // ffn_row_* dispatch picks. Always available since it doesn't depend
    // on any has_* flag. Bucket is *vindex-dependent*: on an f16 vindex
    // sparse walks f32 features (Exact); on a Q4K vindex sparse walks
    // Q4K via kquant_ffn_row_dot (Quantized). primary_storage_bucket()
    // encapsulates that mapping so future storage formats inherit it.
    out.push(PathSpec {
        name: "sparse",
        mask: PathMask::default(),
        sparse_k: Some(usize::MAX),
        bound: bound_for_bucket(index.primary_storage_bucket()),
    });

    // fp4_storage:sparse — only if the vindex carries FP4 storage.
    if index.has_fp4_storage() {
        out.push(PathSpec {
            name: "fp4_storage",
            mask: PathMask {
                // Don't mask anything: fp4 fires from the dense ladder
                // when has_fp4_storage()=true, which is what we want.
                ..PathMask::default()
            },
            sparse_k: None,
            bound: BOUND_FP4,
        });
    }

    // interleaved_q4 — requires a backend with q4 support; skipped in v1
    // since this example doesn't pass a backend. Documented for clarity:
    if index.has_interleaved_q4() {
        eprintln!(
            "[walk_path_audit] interleaved_q4 path skipped (requires Metal/Q4 backend; not wired in v1)"
        );
    }

    // interleaved (f32) — mask fp4 + q4 above it. Always Exact: this
    // path reads f32 interleaved data directly, regardless of what
    // other storage variants the vindex carries.
    if index.has_interleaved() {
        out.push(PathSpec {
            name: "interleaved",
            mask: PathMask {
                hide_fp4: true,
                hide_q4: true,
                ..PathMask::default()
            },
            sparse_k: None,
            bound: BOUND_EXACT,
        });
    }

    // full_mmap — mask everything above it. Always Exact: walks f32
    // mmap'd gate/up/down.
    if index.has_full_mmap_ffn() {
        out.push(PathSpec {
            name: "full_mmap",
            mask: PathMask {
                hide_fp4: true,
                hide_q4: true,
                hide_interleaved: true,
                ..PathMask::default()
            },
            sparse_k: None,
            bound: BOUND_EXACT,
        });
    }

    // interleaved_kquant:dequant — mask everything above it. Always
    // Quantized: dequants Q4K bytes per layer.
    if index.has_interleaved_kquant() {
        out.push(PathSpec {
            name: "interleaved_kquant",
            mask: PathMask {
                hide_fp4: true,
                hide_q4: true,
                hide_interleaved: true,
                hide_full_mmap: true,
                ..PathMask::default()
            },
            sparse_k: None,
            bound: BOUND_QUANTIZED,
        });
    }

    // exact — mask everything above it. Needs has_down_features=true.
    // Always Exact: gate/up from safetensors (f32), down from features
    // (f32).
    if index.has_down_features() {
        out.push(PathSpec {
            name: "exact",
            mask: PathMask {
                hide_fp4: true,
                hide_q4: true,
                hide_interleaved: true,
                hide_full_mmap: true,
                hide_q4k: true,
                ..PathMask::default()
            },
            sparse_k: None,
            bound: BOUND_EXACT,
        });
    }

    // weights_fallback:* is intentionally not in this audit. It's the
    // no-vindex-data corner case (extract_level = Browse without pinned
    // weights), and at any finite K it's measuring approximation quality
    // ("how good is K=N sparse walk vs dense matmul") rather than path
    // equivalence ("do the walk paths agree with dense matmul"). Those
    // are different questions; mixing them muddies the audit headline.
    // The K-sweep belongs in a separate `walk_approximation_quality`
    // example.

    out
}

// ── Diff plumbing ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
struct PositionDiff {
    l2: f32,
    cos: f32,
    max_abs: f32,
    /// ‖primary‖ at this position. Carried so downstream can compute
    /// `rel_L2 = L2 / max(primary_norm, REL_L2_NORM_EPS)` without
    /// re-walking the array. Diagnostic-only; not directly asserted on.
    primary_norm: f32,
}

/// Per-(layer, position) diff between primary and secondary. Last-position
/// diff is what walk_correctness reports; we capture every position so we
/// can report worst-case across the whole prompt.
fn diff_all_positions(a: &Array2<f32>, b: &Array2<f32>) -> Vec<PositionDiff> {
    let seq_len = a.shape()[0];
    let hidden = a.shape()[1];
    let mut out = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let mut l2_sq = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut dot = 0.0f32;
        let mut a_norm_sq = 0.0f32;
        let mut b_norm_sq = 0.0f32;
        for j in 0..hidden {
            let ai = a[[s, j]];
            let bi = b[[s, j]];
            let d = ai - bi;
            l2_sq += d * d;
            let abs_d = d.abs();
            if abs_d > max_abs {
                max_abs = abs_d;
            }
            dot += ai * bi;
            a_norm_sq += ai * ai;
            b_norm_sq += bi * bi;
        }
        let an = a_norm_sq.sqrt();
        let bn = b_norm_sq.sqrt();
        let cos = if an > 0.0 && bn > 0.0 {
            dot / (an * bn)
        } else {
            0.0
        };
        out.push(PositionDiff {
            l2: l2_sq.sqrt(),
            cos,
            max_abs,
            primary_norm: an,
        });
    }
    out
}

/// DualFfn that records, per layer, the full `[seq_len]` diff vector. The
/// primary drives the residual stream onward (so this measures secondary
/// drift relative to the dense reference at the *same* input residual).
struct DualFfn<'a> {
    primary: &'a dyn FfnBackend,
    secondary: &'a dyn FfnBackend,
    /// Vec<(layer, per-position diffs)> in the order calls arrive.
    diffs: RefCell<Vec<(usize, Vec<PositionDiff>)>>,
}

impl<'a> FfnBackend for DualFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let (p_out, p_act) = self.primary.forward_with_activation(layer, x);
        let (s_out, _) = self.secondary.forward_with_activation(layer, x);
        let positions = diff_all_positions(&p_out, &s_out);
        self.diffs.borrow_mut().push((layer, positions));
        (p_out, p_act)
    }
    fn name(&self) -> &str {
        "dual"
    }
}

// ── Per-path run state ─────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
struct LayerSummary {
    // ── Assertion metrics (magnitude-invariant) ─────────────────────
    /// Worst (min) cos across all observations at this layer.
    min_cos: f32,
    /// Worst (max) rel_L2 = L2 / max(‖primary‖, EPS) across all observations.
    max_rel_l2: f32,
    /// Prompt key at which `max_rel_l2` was observed.
    worst_rel_l2_prompt: String,
    /// Sequence position at which `max_rel_l2` was observed.
    worst_rel_l2_pos: usize,

    // ── Diagnostic metrics (magnitude-dependent; for triage, not assertion) ─
    /// Worst absolute L2 across all observations at this layer.
    max_l2: f32,
    /// Worst max-element drift.
    max_abs: f32,
    /// Prompt key at which `max_l2` was observed (often the residual-magnitude
    /// outlier — see L11/code/1 ` fibonacci` on Gemma 3 4B).
    worst_prompt: String,
    /// Sequence position at which `max_l2` was observed.
    worst_pos: usize,

    // ── Bookkeeping ─────────────────────────────────────────────────
    /// Number of observations folded in (sum of seq_len across prompts).
    n_obs: usize,
    /// Dispatch label observed for this layer (any label seen across runs;
    /// they should all match for a forced path, modulo `exact` fallthrough).
    dispatch_label: String,
    /// Set when the harness detected `exact` traced for this layer but
    /// `down_layer_matrix(layer).is_none()` — silently relayed to
    /// `walk_ffn_full_mmap` despite the trace label.
    fallthrough: bool,
}

#[derive(Debug, Default)]
struct PromptResult {
    /// Top-1 token from the path's prediction.
    walk_top1_token: String,
    walk_top1_prob: f64,
    /// Top-1 from dense (reference, cached across paths).
    dense_top1_token: String,
    dense_top1_prob: f64,
    /// True iff walk_top1_token == dense_top1_token.
    top1_match: bool,
    /// |walk_prob - dense_prob| at top-1 token (only meaningful on Paris).
    prob_delta: f64,
}

#[derive(Debug, Default)]
struct PathRun {
    name: String,
    mask: PathMask,
    sparse_k: Option<usize>,
    /// Assertion floor (cos + rel_L2). `None` only used for the default-
    /// constructed Default impl; populated for every real run.
    bound: Option<PathBound>,
    layers: Vec<LayerSummary>,
    /// path-name → layer count, taken from drained dispatch trace.
    dispatch_counts: BTreeMap<String, usize>,
    /// Layers where exact-fallthrough was detected post-run.
    fallthrough_layers: Vec<usize>,
    /// Per-prompt result (keyed by prompt name).
    per_prompt: BTreeMap<String, PromptResult>,
    /// Verdict and reason.
    pass: bool,
    fail_reasons: Vec<String>,
    // ── Aggregate path-level stats ──────────────────────────────────
    /// Assertion: worst cos across the whole path.
    path_min_cos: f32,
    /// Assertion: worst rel_L2 across the whole path.
    path_max_rel_l2: f32,
    path_worst_rel_l2_layer: usize,
    path_worst_rel_l2_prompt: String,
    path_worst_rel_l2_pos: usize,
    /// Diagnostic: worst absolute L2 across the whole path.
    path_max_l2: f32,
    path_mean_l2: f32,
    path_max_abs: f32,
    path_worst_layer: usize,
    path_worst_prompt: String,
    path_worst_pos: usize,
    n_total_obs: usize,
}

/// Run one prompt through DualFfn + secondary-only, fold per-(layer,
/// position) diffs into `per_layer`, and capture top-1 prediction.
#[allow(clippy::too_many_arguments)]
fn run_prompt_for_path(
    weights: &larql_inference::model::ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_key: &str,
    prompt: &str,
    spec: &PathSpec,
    inner: &dyn GateIndex,
    weight_ffn: &WeightFfn<'_>,
    per_layer: &mut [Option<LayerSummary>],
    dispatch_counts: &mut BTreeMap<String, usize>,
    exact_layers_seen: &mut Vec<usize>,
) -> (String, f64) {
    let masked = MaskedGateIndex {
        inner,
        mask: spec.mask,
    };
    let config = match spec.sparse_k {
        Some(k) => WalkFfnConfig::sparse(weights.num_layers, k),
        None => WalkFfnConfig::dense(weights.num_layers),
    };
    // Fresh WalkFfn per (path, prompt) — gives us a clean L1 cache state
    // per measurement and isolates dispatch trace per prompt.
    let walk = WalkFfn::from_config(weights, &masked, config).with_dispatch_trace();

    let dual = DualFfn {
        primary: weight_ffn,
        secondary: &walk,
        diffs: RefCell::new(Vec::with_capacity(weights.num_layers)),
    };

    // Tokenize and run. Use predict_with_ffn for the dual; we'll re-run
    // walk solo afterwards to get the path's own top-1 prediction.
    let encoding = tokenizer
        .encode(prompt, true)
        .unwrap_or_else(|e| panic!("tokenize prompt {prompt_key}: {e}"));
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let _ = predict_with_ffn(weights, tokenizer, &token_ids, 5, &dual);

    let trace = walk.take_dispatch_trace();
    let trace_by_layer: BTreeMap<usize, &'static str> =
        trace.iter().map(|e| (e.layer, e.path)).collect();
    for entry in &trace {
        *dispatch_counts.entry(entry.path.to_string()).or_insert(0) += 1;
    }

    // Collapse per-(layer, position) diffs into per-layer summaries,
    // tracking which (prompt, position) gave the worst L2 at each layer.
    let diffs = dual.diffs.borrow();
    for (layer, positions) in diffs.iter() {
        let slot = per_layer
            .get_mut(*layer)
            .expect("per_layer indexed by layer < num_layers");
        let mut entry = slot.take().unwrap_or_else(|| LayerSummary {
            min_cos: 1.0,
            ..Default::default()
        });
        for (pos, d) in positions.iter().enumerate() {
            entry.n_obs += 1;
            let rel = d.l2 / d.primary_norm.max(REL_L2_NORM_EPS);
            // Assertion metrics first.
            if rel > entry.max_rel_l2 {
                entry.max_rel_l2 = rel;
                entry.worst_rel_l2_prompt = prompt_key.to_string();
                entry.worst_rel_l2_pos = pos;
            }
            if d.cos < entry.min_cos {
                entry.min_cos = d.cos;
            }
            // Diagnostic metrics.
            if d.l2 > entry.max_l2 {
                entry.max_l2 = d.l2;
                entry.worst_prompt = prompt_key.to_string();
                entry.worst_pos = pos;
            }
            if d.max_abs > entry.max_abs {
                entry.max_abs = d.max_abs;
            }
        }
        if entry.dispatch_label.is_empty() {
            if let Some(lbl) = trace_by_layer.get(layer) {
                entry.dispatch_label = (*lbl).to_string();
                if *lbl == "exact" {
                    exact_layers_seen.push(*layer);
                }
            }
        }
        *slot = Some(entry);
    }

    // Re-run walk solo to capture top-1 prediction. Cheaper than reusing
    // dual's predict result because dual may bias predictions through
    // primary's residual stream — we want the path's own answer.
    let masked2 = MaskedGateIndex {
        inner,
        mask: spec.mask,
    };
    let config2 = match spec.sparse_k {
        Some(k) => WalkFfnConfig::sparse(weights.num_layers, k),
        None => WalkFfnConfig::dense(weights.num_layers),
    };
    let walk2 = WalkFfn::from_config(weights, &masked2, config2);
    let walk_pred = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk2);
    let (top1_tok, top1_prob) = walk_pred.predictions.into_iter().next().unwrap_or_default();
    (top1_tok, top1_prob)
}

// ── Markdown / JSON emit ───────────────────────────────────────────────

fn render_markdown(model: &str, vindex: &Path, runs: &[PathRun]) -> String {
    let mut s = String::new();
    s.push_str("# walk_path_audit\n\n");
    s.push_str(&format!("**Model:** `{}`  \n", model));
    s.push_str(&format!("**Vindex:** `{}`  \n", vindex.display()));
    s.push_str(&format!("**Prompts:** {}\n\n", PROMPTS.len()));
    s.push_str(
        "**Metrics.** Assertion: `min cos`, `max rel L2 = L2 / ‖primary‖` — both \
         magnitude-invariant. Diagnostic: `max abs L2`, `max|Δ|` — vary with residual \
         magnitude, included for triage of outlier observations (e.g. residual-norm \
         spikes at specific (layer, token) pairs).\n\n",
    );

    // Summary table
    s.push_str("## Summary\n\n");
    s.push_str(
        "| path | bound | min cos (assert) | max rel L2 (assert) | top-1 ok | Paris ΔP | max abs L2 (diag) | worst rel-L2 layer | worst rel-L2 prompt | verdict |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|---|\n");
    for r in runs {
        let top1_ok = if r.per_prompt.values().all(|p| p.top1_match) {
            "✓".to_string()
        } else {
            let bad: Vec<_> = r
                .per_prompt
                .iter()
                .filter(|(_, p)| !p.top1_match)
                .map(|(k, _)| k.as_str())
                .collect();
            format!("✗ ({})", bad.join(","))
        };
        let paris_delta = r
            .per_prompt
            .get(PARIS_KEY)
            .map(|p| format!("{:.3e}", p.prob_delta))
            .unwrap_or_else(|| "—".to_string());
        let verdict = if r.pass { "PASS" } else { "FAIL" };
        let bound = r.bound.expect("bound populated for all real runs");
        s.push_str(&format!(
            "| `{}` | {} (cos≥{:.5}, rel_L2≤{:.0e}) | {:.6} | {:.3e} | {} | {} | {:.3e} | {} | {} | **{}** |\n",
            r.name,
            bound.kind,
            bound.min_cos,
            bound.rel_l2,
            r.path_min_cos,
            r.path_max_rel_l2,
            top1_ok,
            paris_delta,
            r.path_max_l2,
            r.path_worst_rel_l2_layer,
            r.path_worst_rel_l2_prompt,
            verdict,
        ));
    }
    s.push('\n');

    // Per-path detail
    for r in runs {
        let bound = r.bound.expect("bound populated for all real runs");
        s.push_str(&format!("## `{}`\n\n", r.name));
        s.push_str(&format!(
            "**Mask:** fp4={} q4={} interleaved={} full_mmap={} q4k={} down_features={}  \n",
            r.mask.hide_fp4,
            r.mask.hide_q4,
            r.mask.hide_interleaved,
            r.mask.hide_full_mmap,
            r.mask.hide_q4k,
            r.mask.hide_down_features,
        ));
        s.push_str(&format!(
            "**Sparse K:** {}  \n",
            r.sparse_k
                .map(|k| if k == usize::MAX {
                    "MAX".to_string()
                } else {
                    k.to_string()
                })
                .unwrap_or_else(|| "—".to_string())
        ));
        s.push_str(&format!(
            "**Bound ({}):** cos ≥ {:.5}, rel_L2 ≤ {:.0e}  \n",
            bound.kind, bound.min_cos, bound.rel_l2,
        ));
        s.push_str(&format!(
            "**Assertion aggregate:** min cos = {:.6}, max rel_L2 = {:.3e} (layer {}, prompt {}, pos {})  \n",
            r.path_min_cos,
            r.path_max_rel_l2,
            r.path_worst_rel_l2_layer,
            r.path_worst_rel_l2_prompt,
            r.path_worst_rel_l2_pos,
        ));
        s.push_str(&format!(
            "**Diagnostic aggregate:** max abs_L2 = {:.3e} (layer {}, prompt {}, pos {}), max|Δ| = {:.3e}, n_obs = {}  \n",
            r.path_max_l2,
            r.path_worst_layer,
            r.path_worst_prompt,
            r.path_worst_pos,
            r.path_max_abs,
            r.n_total_obs,
        ));
        if !r.dispatch_counts.is_empty() {
            s.push_str("**Dispatch counts:** ");
            let parts: Vec<String> = r
                .dispatch_counts
                .iter()
                .map(|(k, v)| format!("`{}`={}", k, v))
                .collect();
            s.push_str(&parts.join(", "));
            s.push_str("  \n");
        }
        if !r.fallthrough_layers.is_empty() {
            s.push_str(&format!(
                "**⚠ exact→full_mmap fallthrough at layers:** {:?}  \n",
                r.fallthrough_layers
            ));
        }
        if !r.fail_reasons.is_empty() {
            s.push_str("**Fail reasons:**\n");
            for reason in &r.fail_reasons {
                s.push_str(&format!("- {}\n", reason));
            }
        }
        s.push('\n');

        // Per-prompt block
        s.push_str("### Per-prompt\n\n");
        s.push_str("| prompt | walk top-1 | dense top-1 | match | walk P | dense P | ΔP |\n");
        s.push_str("|---|---|---|---|---|---|---|\n");
        for (key, p) in &r.per_prompt {
            s.push_str(&format!(
                "| `{}` | `{}` | `{}` | {} | {:.6} | {:.6} | {:.3e} |\n",
                key,
                p.walk_top1_token,
                p.dense_top1_token,
                if p.top1_match { "✓" } else { "✗" },
                p.walk_top1_prob,
                p.dense_top1_prob,
                p.prob_delta,
            ));
        }
        s.push('\n');

        // Per-layer block. Assertion columns first, then diagnostic.
        s.push_str("### Per-layer\n\n");
        s.push_str(
            "| layer | dispatch | min cos (assert) | max rel L2 (assert) | rel L2 worst (prompt/pos) | max abs L2 (diag) | max\\|Δ\\| (diag) | abs L2 worst (prompt/pos) | n |\n",
        );
        s.push_str("|---|---|---|---|---|---|---|---|---|\n");
        for (i, ls) in r.layers.iter().enumerate() {
            s.push_str(&format!(
                "| {} | `{}`{} | {:.6} | {:.3e} | {}/{} | {:.3e} | {:.3e} | {}/{} | {} |\n",
                i,
                ls.dispatch_label,
                if ls.fallthrough { " ⚠" } else { "" },
                ls.min_cos,
                ls.max_rel_l2,
                ls.worst_rel_l2_prompt,
                ls.worst_rel_l2_pos,
                ls.max_l2,
                ls.max_abs,
                ls.worst_prompt,
                ls.worst_pos,
                ls.n_obs,
            ));
        }
        s.push('\n');
    }

    s
}

fn render_json(model: &str, vindex: &Path, runs: &[PathRun]) -> String {
    use serde_json::{json, Value};
    let paths: Vec<Value> = runs
        .iter()
        .map(|r| {
            json!({
                "name": r.name,
                "mask": {
                    "hide_fp4": r.mask.hide_fp4,
                    "hide_q4": r.mask.hide_q4,
                    "hide_interleaved": r.mask.hide_interleaved,
                    "hide_full_mmap": r.mask.hide_full_mmap,
                    "hide_q4k": r.mask.hide_q4k,
                    "hide_down_features": r.mask.hide_down_features,
                },
                "sparse_k": r.sparse_k.map(|k| if k == usize::MAX { -1i64 } else { k as i64 }),
                "bound": r.bound.map(|b| json!({
                    "kind": b.kind,
                    "min_cos": b.min_cos,
                    "rel_l2": b.rel_l2,
                })),
                "aggregate": {
                    "assertion": {
                        "min_cos": r.path_min_cos,
                        "max_rel_l2": r.path_max_rel_l2,
                        "worst_rel_l2_layer": r.path_worst_rel_l2_layer,
                        "worst_rel_l2_prompt": r.path_worst_rel_l2_prompt,
                        "worst_rel_l2_pos": r.path_worst_rel_l2_pos,
                    },
                    "diagnostic": {
                        "max_abs_l2": r.path_max_l2,
                        "mean_abs_l2": r.path_mean_l2,
                        "max_abs": r.path_max_abs,
                        "worst_layer": r.path_worst_layer,
                        "worst_prompt": r.path_worst_prompt,
                        "worst_pos": r.path_worst_pos,
                    },
                    "n_obs": r.n_total_obs,
                },
                "dispatch_counts": r.dispatch_counts,
                "fallthrough_layers": r.fallthrough_layers,
                "per_prompt": r.per_prompt.iter().map(|(k, p)| (k.clone(), json!({
                    "walk_top1_token": p.walk_top1_token,
                    "walk_top1_prob": p.walk_top1_prob,
                    "dense_top1_token": p.dense_top1_token,
                    "dense_top1_prob": p.dense_top1_prob,
                    "top1_match": p.top1_match,
                    "prob_delta": p.prob_delta,
                }))).collect::<serde_json::Map<_, _>>(),
                "per_layer": r.layers.iter().enumerate().map(|(i, ls)| json!({
                    "layer": i,
                    "dispatch": ls.dispatch_label,
                    "fallthrough": ls.fallthrough,
                    "assertion": {
                        "min_cos": ls.min_cos,
                        "max_rel_l2": ls.max_rel_l2,
                        "worst_rel_l2_prompt": ls.worst_rel_l2_prompt,
                        "worst_rel_l2_pos": ls.worst_rel_l2_pos,
                    },
                    "diagnostic": {
                        "max_abs_l2": ls.max_l2,
                        "max_abs": ls.max_abs,
                        "worst_prompt": ls.worst_prompt,
                        "worst_pos": ls.worst_pos,
                    },
                    "n_obs": ls.n_obs,
                })).collect::<Vec<_>>(),
                "verdict": if r.pass { "pass" } else { "fail" },
                "fail_reasons": r.fail_reasons,
            })
        })
        .collect();

    let root = json!({
        "model": model,
        "vindex": vindex.display().to_string(),
        "prompts": PROMPTS.iter().map(|(k, p)| json!({"key": k, "text": p})).collect::<Vec<_>>(),
        "paths": paths,
    });
    serde_json::to_string_pretty(&root).unwrap()
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    eprintln!("=== walk_path_audit ===\n");
    eprintln!("Model:  {}", args.model);
    eprintln!("Vindex: {}\n", args.vindex.display());

    let t0 = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    eprintln!(
        "Model loaded in {:.1}s ({} layers, hidden={})",
        t0.elapsed().as_secs_f64(),
        model.weights().num_layers,
        model.weights().hidden_size
    );

    let t0 = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&args.vindex, &mut cb)?;
    eprintln!(
        "Vindex loaded in {:.1}s ({} vectors)\n",
        t0.elapsed().as_secs_f64(),
        index.total_gate_vectors()
    );

    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    let weight_ffn = WeightFfn { weights };

    // Cache dense baseline predictions per prompt — same for every path,
    // no point re-running.
    let mut dense_by_prompt: BTreeMap<String, (String, f64)> = BTreeMap::new();
    for (key, prompt) in PROMPTS {
        let encoding = tokenizer
            .encode(*prompt, true)
            .map_err(|e| format!("tokenize {key}: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let pred = predict(weights, tokenizer, &token_ids, 5);
        let (tok, prob) = pred.predictions.into_iter().next().unwrap_or_default();
        eprintln!("[dense] {:>7}: top1=`{}` p={:.6}", key, tok, prob);
        dense_by_prompt.insert((*key).to_string(), (tok, prob));
    }
    eprintln!();

    let paths = enumerate_paths(&index);
    eprintln!("Testing {} path(s):\n", paths.len());
    for p in &paths {
        eprintln!("  - {}", p.name);
    }
    eprintln!();

    let mut runs: Vec<PathRun> = Vec::with_capacity(paths.len());
    for spec in &paths {
        let t0 = Instant::now();
        let bound = spec.bound;

        let mut per_layer: Vec<Option<LayerSummary>> = (0..num_layers).map(|_| None).collect();
        let mut dispatch_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut exact_layers_seen: Vec<usize> = Vec::new();
        let mut per_prompt: BTreeMap<String, PromptResult> = BTreeMap::new();

        for (key, prompt) in PROMPTS {
            let (walk_tok, walk_prob) = run_prompt_for_path(
                weights,
                tokenizer,
                key,
                prompt,
                spec,
                &index,
                &weight_ffn,
                &mut per_layer,
                &mut dispatch_counts,
                &mut exact_layers_seen,
            );
            let (dense_tok, dense_prob) = dense_by_prompt.get(*key).cloned().unwrap_or_default();
            let top1_match = walk_tok == dense_tok;
            let prob_delta = (walk_prob - dense_prob).abs();
            per_prompt.insert(
                (*key).to_string(),
                PromptResult {
                    walk_top1_token: walk_tok,
                    walk_top1_prob: walk_prob,
                    dense_top1_token: dense_tok,
                    dense_top1_prob: dense_prob,
                    top1_match,
                    prob_delta,
                },
            );
        }

        // Detect exact→full_mmap fallthrough on every layer the trace
        // labelled `exact`. We can't see this in the dispatch trace
        // itself — exact.rs falls through silently when the per-layer
        // `down_layer_matrix` returns None despite has_down_features=true.
        let mut fallthrough_layers: Vec<usize> = Vec::new();
        if spec.name == "exact" {
            for layer in &exact_layers_seen {
                if index.down_layer_matrix(*layer).is_none() {
                    fallthrough_layers.push(*layer);
                    if let Some(slot) = per_layer.get_mut(*layer) {
                        if let Some(s) = slot.as_mut() {
                            s.fallthrough = true;
                            s.dispatch_label = "exact:fallthrough_to_full_mmap".to_string();
                        }
                    }
                }
            }
            fallthrough_layers.sort();
            fallthrough_layers.dedup();
        }

        // Materialise per-layer summaries; fill empty slots with a default
        // so the table has one row per layer.
        let layers: Vec<LayerSummary> = per_layer
            .into_iter()
            .map(|opt| opt.unwrap_or_default())
            .collect();

        // Aggregate path-level stats. Assertion metrics first, then
        // diagnostic. Both worst-case observations carry their (prompt,
        // pos) coordinates so the failure message points at the exact
        // residual that breached.
        let mut path_max_rel_l2 = 0.0f32;
        let mut path_worst_rel_l2_layer = 0usize;
        let mut path_worst_rel_l2_prompt = String::new();
        let mut path_worst_rel_l2_pos = 0usize;
        let mut path_min_cos = 1.0f32;

        let mut path_max_l2 = 0.0f32;
        let mut path_max_abs = 0.0f32;
        let mut path_worst_layer = 0usize;
        let mut path_worst_prompt = String::new();
        let mut path_worst_pos = 0usize;
        let mut sum_l2 = 0.0f64;
        let mut n_total_obs = 0usize;
        for (i, ls) in layers.iter().enumerate() {
            sum_l2 += (ls.max_l2 as f64) * (ls.n_obs as f64);
            n_total_obs += ls.n_obs;
            // Assertion metrics.
            if ls.max_rel_l2 > path_max_rel_l2 {
                path_max_rel_l2 = ls.max_rel_l2;
                path_worst_rel_l2_layer = i;
                path_worst_rel_l2_prompt = ls.worst_rel_l2_prompt.clone();
                path_worst_rel_l2_pos = ls.worst_rel_l2_pos;
            }
            if ls.min_cos < path_min_cos {
                path_min_cos = ls.min_cos;
            }
            // Diagnostic metrics.
            if ls.max_l2 > path_max_l2 {
                path_max_l2 = ls.max_l2;
                path_worst_layer = i;
                path_worst_prompt = ls.worst_prompt.clone();
                path_worst_pos = ls.worst_pos;
            }
            if ls.max_abs > path_max_abs {
                path_max_abs = ls.max_abs;
            }
        }
        let path_mean_l2 = if n_total_obs > 0 {
            (sum_l2 / n_total_obs as f64) as f32
        } else {
            0.0
        };

        // Verdict: cos ≥ bound.min_cos, rel_L2 ≤ bound.rel_l2, all prompts
        // top-1 match, Paris prob delta ≤ bound.paris_prob_budget. Multiple
        // failures collected together so the first run gives a complete
        // picture instead of failing fast and hiding the rest.
        let mut fail_reasons: Vec<String> = Vec::new();
        if path_min_cos < bound.min_cos {
            fail_reasons.push(format!(
                "min cos {:.6} below floor {:.6}",
                path_min_cos, bound.min_cos,
            ));
        }
        if path_max_rel_l2 > bound.rel_l2 {
            fail_reasons.push(format!(
                "max rel L2 {:.3e} exceeds bound {:.0e} at layer {} (prompt {}, pos {})",
                path_max_rel_l2,
                bound.rel_l2,
                path_worst_rel_l2_layer,
                path_worst_rel_l2_prompt,
                path_worst_rel_l2_pos,
            ));
        }
        for (key, p) in &per_prompt {
            if !p.top1_match {
                fail_reasons.push(format!(
                    "top-1 mismatch on `{}`: walk=`{}` dense=`{}`",
                    key, p.walk_top1_token, p.dense_top1_token,
                ));
            }
        }
        if let Some(p) = per_prompt.get(PARIS_KEY) {
            if p.prob_delta > bound.paris_prob_budget {
                fail_reasons.push(format!(
                    "Paris prob delta {:.3e} exceeds {:.0e}",
                    p.prob_delta, bound.paris_prob_budget
                ));
            }
        }
        let pass = fail_reasons.is_empty();

        eprintln!(
            "[{:>16}] cos={:.6} rel_L2={:.3e} (L{}/{}/{})  abs_L2={:.3e}(diag)  {}  ({:.1}s)",
            spec.name,
            path_min_cos,
            path_max_rel_l2,
            path_worst_rel_l2_layer,
            path_worst_rel_l2_prompt,
            path_worst_rel_l2_pos,
            path_max_l2,
            if pass { "PASS" } else { "FAIL" },
            t0.elapsed().as_secs_f64(),
        );
        if !fallthrough_layers.is_empty() {
            eprintln!(
                "                  ⚠ exact→full_mmap fallthrough at {:?}",
                fallthrough_layers
            );
        }

        runs.push(PathRun {
            name: spec.name.to_string(),
            mask: spec.mask,
            sparse_k: spec.sparse_k,
            bound: Some(bound),
            layers,
            dispatch_counts,
            fallthrough_layers,
            per_prompt,
            pass,
            fail_reasons,
            path_min_cos,
            path_max_rel_l2,
            path_worst_rel_l2_layer,
            path_worst_rel_l2_prompt,
            path_worst_rel_l2_pos,
            path_max_l2,
            path_mean_l2,
            path_max_abs,
            path_worst_layer,
            path_worst_prompt,
            path_worst_pos,
            n_total_obs,
        });
    }

    // Emit artifacts.
    let md = render_markdown(&args.model, &args.vindex, &runs);
    if let Some(path) = &args.out_md {
        std::fs::write(path, &md)?;
        eprintln!("\nMarkdown → {}", path.display());
    } else {
        println!("{}", md);
    }
    let json = render_json(&args.model, &args.vindex, &runs);
    if let Some(path) = &args.out_json {
        std::fs::write(path, &json)?;
        eprintln!("JSON → {}", path.display());
    }

    // Exit code = number of failed paths (so CI can `exit_code != 0` test).
    let failed = runs.iter().filter(|r| !r.pass).count();
    eprintln!(
        "\n=== {} path(s) tested, {} passed, {} failed ===",
        runs.len(),
        runs.len() - failed,
        failed
    );
    if failed > 0 {
        std::process::exit(failed as i32);
    }
    Ok(())
}
