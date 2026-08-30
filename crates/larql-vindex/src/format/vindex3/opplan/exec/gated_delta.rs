//! Reference Gated DeltaNet: the recurrence, written to be read.
//!
//! Deliberately slow and literal. No fused kernel, no vectorisation, no
//! reassociation — this exists so someone can put it beside
//! `torch_recurrent_gated_delta_rule` in `transformers` and compare it line
//! by line. Speed is QW-4's problem.
//!
//! The state is owned here, on purpose. A DeltaNet layer's continuation
//! state is one dense `Dk × Dv` matrix per value head and does not grow
//! with the sequence, so it is not a KV cache and must not be forced
//! through one. Whether the engine should have a single abstraction
//! covering both is a real question, and it is QW-3's — answering it before
//! the arithmetic is proven would risk baking a wrong recurrence into a
//! nice-looking generic interface.

// Explicit index loops on purpose. This module's stated job is to sit
// beside `torch_recurrent_gated_delta_rule` and be checkable line by line,
// and the reference indexes `[..., i]` over named axes. Iterator chains
// would read better in isolation and worse against the thing they must be
// verified against — and verification is the entire point of the file.
#![allow(clippy::needless_range_loop)]

use super::super::gated_delta::StateDtype;
use super::super::GatedDeltaOp;
use super::continuation::{
    RecurrentBuffer, RecurrentBufferGeometry, RecurrentGeometry, RecurrentState,
    StateInitialization,
};
use super::cpu::WeightRows;
use super::timing::{timed, OpClass};

/// L2-normalisation epsilon, from the reference kernel's own default.
const L2NORM_EPS: f32 = 1e-6;

/// This operator's state, expressed in the engine's generic terms.
///
/// A Gated DeltaNet layer keeps one `Dk × Dv` matrix per value head. That
/// is a [`RecurrentGeometry`] like any other — the shape is the operator's
/// business, and the storage is not. There is deliberately no
/// `GatedDeltaState` type: a state named after one operator is exactly what
/// KDA would have had to work around.
///
/// `None` when the checkpoint declared no state precision this build
/// represents. It does NOT fall back to a default, for the same reason
/// [`plan_continuation_geometry`](super::continuation::plan_continuation_geometry)
/// does not: a default that happens to be right for one checkpoint lets
/// every test pass while the architecture is wrong, and the next operator
/// inherits the accident.
pub fn state_geometry(op: &GatedDeltaOp) -> Option<RecurrentGeometry> {
    Some(RecurrentGeometry {
        buffers: vec![
            // Buffer 0 — the delta matrix.
            RecurrentBufferGeometry {
                shape: vec![op.num_value_heads, op.key_head_dim, op.value_head_dim],
                dtype: op.state_dtype?,
                initialization: StateInitialization::Zeros,
            },
            // Buffer 1 — the causal convolution's history. See
            // `plan_continuation_geometry` for why its precision is not
            // taken from `state_dtype`.
            RecurrentBufferGeometry {
                shape: vec![op.qkv_channels(), op.conv_kernel],
                dtype: StateDtype::Float32,
                initialization: StateInitialization::Zeros,
            },
        ],
    })
}

/// Buffer indices this operator assigns. The storage layer holds a list;
/// these names are the operator's private knowledge of what is in it.
pub const DELTA_MATRIX: usize = 0;
pub const CONV_HISTORY: usize = 1;

/// Compile-time reminder that the two dtype spellings are one type.
const _: fn() = || {
    let _: Option<StateDtype> = None::<StateDtype>;
};

/// Flat index of one state cell, and the layout authority the hoisted
/// [`step_inner`] slices by.
///
/// `#[cfg(test)]` because the shipped path no longer indexes cell by
/// cell — that was the cost CPU-2D1 removed. It stays as the written form
/// of the layout, with `the_head_block_is_the_cell_formula` pinning that
/// `h * dk * dv + kk * dv + vv` really is what this computes; the hoist's
/// correctness rests on exactly that identity.
#[cfg(test)]
pub(super) fn cell(op: &GatedDeltaOp, head: usize, k: usize, v: usize) -> usize {
    (head * op.key_head_dim + k) * op.value_head_dim + v
}

/// One position's inputs to the recurrence, per value head, already split
/// and head-expanded by the caller.
///
/// Taking them pre-derived keeps this function the *recurrence* and nothing
/// else: the projections, convolution and head expansion are separate
/// stages with their own comparison planes.
pub struct RecurrenceStep<'a> {
    /// `[value_heads * key_head_dim]`, NOT yet L2-normalised or scaled.
    pub query: &'a [f32],
    /// `[value_heads * key_head_dim]`, NOT yet L2-normalised.
    pub key: &'a [f32],
    /// `[value_heads * value_head_dim]`.
    pub value: &'a [f32],
    /// `[value_heads]` — already `-exp(A_log) * softplus(a + dt_bias)`, so
    /// it is negative and `exp(g)` is a decay in `(0, 1]`.
    pub g: &'a [f32],
    /// `[value_heads]` — already through the sigmoid.
    pub beta: &'a [f32],
}

fn l2_normalise(row: &mut [f32]) {
    let sum_sq: f32 = row.iter().map(|x| x * x).sum();
    let inv = 1.0 / (sum_sq + L2NORM_EPS).sqrt();
    for x in row.iter_mut() {
        *x *= inv;
    }
}

/// Advance the state by one position and return that position's output.
///
/// Returns `[value_heads * value_head_dim]` — the recurrence's own output,
/// before the gated norm and the output projection.
///
/// Transcribed from `torch_recurrent_gated_delta_rule`. The order is the
/// specification, not an implementation detail:
///
/// ```text
/// S  = S * exp(g)                  decay first
/// kv = k · S                       read with the key
/// d  = (v - kv) * beta             the delta rule
/// S  = S + outer(k, d)             rank-1 write
/// o  = q · S                       read with the query, AFTER the write
/// ```
///
/// That last ordering is why a single-position test cannot validate this:
/// the current position reads a state it has just written, so an
/// implementation that reads before writing produces a plausible first
/// output and a wrong second one.
pub fn recurrence_step(
    op: &GatedDeltaOp,
    step: &RecurrenceStep<'_>,
    state: &mut RecurrentBuffer,
) -> Vec<f32> {
    step_inner(op, step, state, Mutation::None)
}

/// Deliberate defects, for the negative controls.
///
/// Test-only, but they perturb the REAL function rather than a copy of it:
/// a control that mutates a duplicate proves only that the duplicate is
/// detectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutation {
    None,
    /// Apply the query scale BEFORE the L2-norm, so normalisation undoes it.
    ScaleBeforeNorm,
    /// Read the state with q BEFORE the rank-1 write, so the current
    /// position cannot see its own contribution.
    ReadBeforeWrite,
    /// Skip the decay entirely.
    NoDecay,
    /// Drop beta from the delta rule.
    NoBeta,
    /// Use g directly instead of exp(g).
    RawGate,
    /// Apply SiLU BEFORE the convolution instead of after.
    SiluBeforeConv,
    /// Centre the convolution window instead of making it causal, so a
    /// position sees its own future.
    CentredConv,
    /// Tile q/k across heads (`h + j*Hk`) instead of `repeat_interleave`
    /// (`h*3 + j`). Same shape, different pairing with the value heads.
    TiledHeadExpansion,
}

/// The delta rule, with the geometry and the state slice lifted OUT of
/// the inner loops.
///
/// Same arithmetic, same order, same rounding — [`step_inner_literal`] is
/// kept beside it and `the_hoisted_recurrence_is_bit_identical` asserts
/// both the output and the STATE match exactly. Nothing here is a
/// numerical decision.
///
/// **What changed is what the compiler can see.** The literal form reads
/// `dk`/`dv` from a `&GatedDeltaOp` and reaches every element through
/// `state.cells_mut()[cell(op, h, kk, vv)]`, so each of the four sweeps
/// is an indexed access through a reborrow with a bound the optimiser
/// cannot prove loop-invariant. Lifting the head's own `dk * dv` block
/// into a slice and walking it as `chunks_exact(dv)` says the same thing
/// in a form that vectorises: the rows are contiguous, the length is
/// known, and there is no aliasing question left to answer.
///
/// A transcription of this loop with compile-time dimensions ran 1.92x
/// faster than one with runtime dimensions and the accessor — same
/// arithmetic, same machine, same process. That gap is what this
/// function exists to close, and it is why the rung is a REFACTOR and
/// hand-written SIMD is a later, separate question.
///
/// `kv`, `delta`, `q` and `k` are allocated once for the layer instead of
/// once per head — 192 allocations per layer at Qwen3.8's 48 heads — for
/// the same reason: it is bookkeeping, not arithmetic.
fn step_inner(
    op: &GatedDeltaOp,
    step: &RecurrenceStep<'_>,
    state: &mut RecurrentBuffer,
    mutation: Mutation,
) -> Vec<f32> {
    let (hv, dk, dv) = (op.num_value_heads, op.key_head_dim, op.value_head_dim);
    let mut out = vec![0.0f32; hv * dv];
    // The reference L2-normalises inside the kernel and applies the query
    // scale AFTERWARDS. Scaling first would rescale the normalisation and
    // is a different function.
    let scale = 1.0 / (dk as f32).sqrt();

    let cells = state.cells_mut();
    let (mut q, mut k) = (vec![0.0f32; dk], vec![0.0f32; dk]);
    let (mut kv, mut delta) = (vec![0.0f32; dv], vec![0.0f32; dv]);

    for h in 0..hv {
        q.copy_from_slice(&step.query[h * dk..(h + 1) * dk]);
        k.copy_from_slice(&step.key[h * dk..(h + 1) * dk]);
        if mutation == Mutation::ScaleBeforeNorm {
            for x in q.iter_mut() {
                *x *= scale;
            }
        }
        l2_normalise(&mut q);
        l2_normalise(&mut k);
        if mutation != Mutation::ScaleBeforeNorm {
            for x in q.iter_mut() {
                *x *= scale;
            }
        }
        let v = &step.value[h * dv..(h + 1) * dv];
        let decay = match mutation {
            Mutation::NoDecay => 1.0,
            Mutation::RawGate => step.g[h],
            _ => step.g[h].exp(),
        };
        let beta = if mutation == Mutation::NoBeta {
            1.0
        } else {
            step.beta[h]
        };

        // `cell(op, h, kk, vv)` is `h * dk * dv + kk * dv + vv`, so this
        // head owns one contiguous block and each `kk` is one contiguous
        // row of `dv`.
        let head = &mut cells[h * dk * dv..(h + 1) * dk * dv];

        for row in head.chunks_exact_mut(dv) {
            for cell in row.iter_mut() {
                *cell *= decay;
            }
        }
        // kv = sum over the KEY axis, weighted by k.
        kv.fill(0.0);
        for (kk, row) in head.chunks_exact(dv).enumerate() {
            let kw = k[kk];
            for (vv, cell) in row.iter().enumerate() {
                kv[vv] += *cell * kw;
            }
        }
        for vv in 0..dv {
            delta[vv] = (v[vv] - kv[vv]) * beta;
        }

        let out_row = &mut out[h * dv..(h + 1) * dv];
        let read = |head: &[f32], out_row: &mut [f32]| {
            for (kk, row) in head.chunks_exact(dv).enumerate() {
                let qw = q[kk];
                for (vv, cell) in row.iter().enumerate() {
                    out_row[vv] += *cell * qw;
                }
            }
        };
        if mutation == Mutation::ReadBeforeWrite {
            read(head, out_row);
        }
        for (kk, row) in head.chunks_exact_mut(dv).enumerate() {
            let kw = k[kk];
            for (vv, cell) in row.iter_mut().enumerate() {
                *cell += kw * delta[vv];
            }
        }
        if mutation != Mutation::ReadBeforeWrite {
            read(head, out_row);
        }
    }
    out
}

/// The literal delta rule, kept as a permanent ORACLE.
///
/// Not dead code and not history: a recurrence can produce a plausible
/// output while corrupting the state every following token reads, so the
/// optimised form needs something to be exactly equal TO. This is the
/// version that reads line by line beside the operator spec, and
/// `the_hoisted_recurrence_is_bit_identical` is what makes keeping it
/// worth the lines.
#[cfg(test)]
pub(super) fn step_inner_literal(
    op: &GatedDeltaOp,
    step: &RecurrenceStep<'_>,
    state: &mut RecurrentBuffer,
    mutation: Mutation,
) -> Vec<f32> {
    let (hv, dk, dv) = (op.num_value_heads, op.key_head_dim, op.value_head_dim);
    let mut out = vec![0.0f32; hv * dv];
    // The reference L2-normalises inside the kernel and applies the query
    // scale AFTERWARDS. Scaling first would rescale the normalisation and
    // is a different function.
    let scale = 1.0 / (dk as f32).sqrt();

    for h in 0..hv {
        let mut q: Vec<f32> = step.query[h * dk..(h + 1) * dk].to_vec();
        let mut k: Vec<f32> = step.key[h * dk..(h + 1) * dk].to_vec();
        if mutation == Mutation::ScaleBeforeNorm {
            for x in q.iter_mut() {
                *x *= scale;
            }
        }
        l2_normalise(&mut q);
        l2_normalise(&mut k);
        if mutation != Mutation::ScaleBeforeNorm {
            for x in q.iter_mut() {
                *x *= scale;
            }
        }
        let v = &step.value[h * dv..(h + 1) * dv];
        let decay = match mutation {
            Mutation::NoDecay => 1.0,
            Mutation::RawGate => step.g[h],
            _ => step.g[h].exp(),
        };
        let beta = if mutation == Mutation::NoBeta {
            1.0
        } else {
            step.beta[h]
        };

        for kk in 0..dk {
            for vv in 0..dv {
                let idx = cell(op, h, kk, vv);
                state.cells_mut()[idx] *= decay;
            }
        }
        // kv = sum over the KEY axis, weighted by k.
        let mut kv = vec![0.0f32; dv];
        for kk in 0..dk {
            let kw = k[kk];
            for vv in 0..dv {
                kv[vv] += state.cells_mut()[cell(op, h, kk, vv)] * kw;
            }
        }
        let delta: Vec<f32> = (0..dv).map(|vv| (v[vv] - kv[vv]) * beta).collect();
        let mut read = |state: &RecurrentBuffer| {
            for kk in 0..dk {
                let qw = q[kk];
                for vv in 0..dv {
                    out[h * dv + vv] += state.cells()[cell(op, h, kk, vv)] * qw;
                }
            }
        };
        if mutation == Mutation::ReadBeforeWrite {
            read(state);
        }
        for kk in 0..dk {
            let kw = k[kk];
            for vv in 0..dv {
                let idx = cell(op, h, kk, vv);
                state.cells_mut()[idx] += kw * delta[vv];
            }
        }
        if mutation != Mutation::ReadBeforeWrite {
            read(state);
        }
    }
    out
}

/// Run the recurrence with a deliberate defect. Negative controls only.
pub fn recurrence_step_mutated(
    op: &GatedDeltaOp,
    step: &RecurrenceStep<'_>,
    state: &mut RecurrentBuffer,
    mutation: Mutation,
) -> Vec<f32> {
    step_inner(op, step, state, mutation)
}

/// The nine operands in the representations they are resident as, in the
/// checkpoint's own layouts.
///
/// Linear weights are `[out, in]` row-major, as PyTorch stores them, so a
/// projection is `y[o] = sum_i x[i] * w[o][i]`. Nothing here re-derives a
/// tensor from its name: the caller resolves the operands through the
/// `GatedDeltaOp` that QW-1 built, which is the single architecture
/// authority.
pub struct GatedDeltaWeights<'a> {
    /// The five DENSE projections, in whatever representation they are
    /// resident as. `WeightRows` and not `&[f32]` because the point of
    /// the rung is that a 100 MB matrix reaches the kernel still compact
    /// — a widening accessor here would put the whole model back in f32
    /// while every test still passed.
    pub in_proj_qkv: WeightRows<'a>,
    pub in_proj_a: WeightRows<'a>,
    pub in_proj_b: WeightRows<'a>,
    pub in_proj_z: WeightRows<'a>,
    pub out_proj: WeightRows<'a>,
    /// The elementwise glue and the depthwise convolution, always f32.
    /// Six MB across the whole model against 11 GB of projection — there
    /// is no traffic here to halve, and narrowing it would only cost a
    /// widen per use.
    pub conv1d: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
    pub norm_eps: f32,
}

/// Every boundary the operator crosses, kept so a disagreement names its
/// own stage instead of being debugged backwards from the layer output.
#[derive(Debug, Default)]
pub struct LayerPlanes {
    /// Post-conv, post-SiLU, split and head-expanded: `[T][Hv*Dk]`.
    pub query: Vec<Vec<f32>>,
    pub key: Vec<Vec<f32>>,
    /// `[T][Hv*Dv]`.
    pub value: Vec<Vec<f32>>,
    /// `[T][Hv]`.
    pub g: Vec<Vec<f32>>,
    pub beta: Vec<Vec<f32>>,
    /// `[T][Hv*Dv]`.
    pub z: Vec<Vec<f32>>,
    pub core: Vec<Vec<f32>>,
    /// `[T][hidden]`.
    pub output: Vec<Vec<f32>>,
}

/// How a Gated DeltaNet layer performs its DENSE projections.
///
/// The seam exists so a backend can accelerate the five matrix products
/// around the recurrence **without touching the recurrence**. Everything
/// QW-2/QW-2E proved stage-by-stage against HF — the convolution, the
/// head expansion, the gates, the delta rule, the gated norm — is below
/// this trait and identical for every implementation.
///
/// `ReferenceBackend` keeps [`ScalarProjections`], which is the literal
/// transcription and shares no arithmetic with `larql-compute`. That
/// independence is what makes it an oracle, so it is never routed
/// through BLAS however much faster BLAS is.
///
/// Named for the five products rather than for one, so it cannot be
/// confused with `exec::cpu::DenseProjector` — the row-range KERNEL
/// trait one layer below. This one asks "compute this whole projection";
/// that one asks "compute these rows", and the executor sits between
/// them deciding how the rows were cut.
pub trait DenseProjections: Sync {
    /// `y = W x`, with `W` row-major `[out_dim, x.len()]` in whatever
    /// representation it is resident as.
    ///
    /// Infallible, and a representation the implementation cannot
    /// consume PANICS rather than returning an error. The pairing of
    /// format to kernel is made once, by `PhysicalProjectionPlan`, and
    /// observed rather than re-derived — so a mismatch here is not a bad
    /// input, it is that invariant broken, and the same posture the
    /// kernels below already take.
    fn project(&self, weight: WeightRows<'_>, x: &[f32], out_dim: usize) -> Vec<f32>;

    /// The same projection against every position, sharing ONE weight
    /// traversal where the backend has a kernel that can.
    ///
    /// Default is the loop it replaces, so the reference realisation and
    /// every other implementation keep working untouched.
    ///
    /// **A claim about SCHEDULE, never about arithmetic.** Each position
    /// keeps its own activation quantisation and its own accumulator, and
    /// nothing is summed across positions — so each returned vector is
    /// bit-identical to what [`Self::project`] would have produced for
    /// that position alone.
    fn project_many(&self, weight: WeightRows<'_>, xs: &[&[f32]], out_dim: usize) -> Vec<Vec<f32>> {
        xs.iter()
            .map(|x| self.project(weight, x, out_dim))
            .collect()
    }

    /// Whether [`Self::project_many`] would actually share the traversal
    /// here. `false` means the loop — correct, and not the thing a
    /// CPU-7C timing is about.
    fn is_weight_stationary(&self, _weight: WeightRows<'_>, _in_dim: usize, _n: usize) -> bool {
        false
    }
}

/// The literal projection: one scalar dot per row.
///
/// Measured at a flat 5.6 GB/s across every Qwen3.8 projection shape —
/// which is why it is the oracle and not the execution strategy.
pub struct ScalarProjections;

impl DenseProjections for ScalarProjections {
    fn project(&self, weight: WeightRows<'_>, x: &[f32], out_dim: usize) -> Vec<f32> {
        let WeightRows::F32(w) = weight else {
            panic!(
                "the reference projection consumes f32 weights only; the reference backend \
                 declares F32 for every operand, so a compact slab arriving here means the \
                 format resolver and the backend that answered it have come apart"
            );
        };
        matvec(w, x, out_dim)
    }
}

fn matvec(w: &[f32], x: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    // The numerically stable form; large x must not overflow exp.
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// The whole operator: hidden states in, layer output out, state advanced.
///
/// Stage order is the specification. Three of these are places a plausible
/// implementation goes wrong without looking wrong:
/// the convolution is causal by left-pad and right-truncate (not centred),
/// SiLU comes AFTER it, and q/k are `repeat_interleave`d 3x — head `e`
/// takes original head `e / 3`, not `e % Hk`.
pub fn layer_forward(
    op: &GatedDeltaOp,
    w: &GatedDeltaWeights<'_>,
    hidden: &[Vec<f32>],
    state: &mut RecurrentState,
    mutation: Mutation,
) -> LayerPlanes {
    layer_forward_with(op, w, hidden, state, mutation, &ScalarProjections)
}

/// [`layer_forward`] with a chosen projection strategy. The public entry
/// point above always passes [`ScalarProjections`], so the reference
/// path is unchanged and the oracle stays independent.
pub fn layer_forward_with(
    op: &GatedDeltaOp,
    w: &GatedDeltaWeights<'_>,
    hidden: &[Vec<f32>],
    state: &mut RecurrentState,
    mutation: Mutation,
    proj: &dyn DenseProjections,
) -> LayerPlanes {
    let (hk, hv) = (op.num_key_heads, op.num_value_heads);
    let (dk, dv) = (op.key_head_dim, op.value_head_dim);
    let (key_dim, value_dim) = (hk * dk, hv * dv);
    let conv_dim = op.qkv_channels();
    let kernel = op.conv_kernel;
    let repeat = hv / hk;
    let mut planes = LayerPlanes::default();
    // Attributes every projection below to the recurrent class, so `g`
    // can be reported as `g_GD` and `g_FFN` separately in one binary.
    let _site = super::cpu::ledger::in_site(super::cpu::ledger::Site::Recurrent);

    // Stage 1: the fused projection, for every position.
    //
    // **CPU-7C.** One weight traversal serves all of them where the
    // backend has a stationary kernel; where it does not, `project_many`
    // is the same loop this used to be. `hidden` is the LAYER INPUT, so
    // every position's operand exists before the recurrence has run — the
    // eligibility that makes this legal is a property of the operand, not
    // of the schedule.
    let rows: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
    let mixed: Vec<Vec<f32>> = proj.project_many(w.in_proj_qkv, &rows, conv_dim);

    // Stage 2: depthwise causal convolution, then SiLU.
    //
    // The window reaches back `kernel-1` positions. Those may lie BEFORE
    // this batch, which is why the convolution keeps a durable history
    // rather than left-padding with zeros: zeros are only correct at the
    // start of a sequence, and a continuation that assumed them would
    // silently truncate every layer's receptive field at the batch
    // boundary. Invisible on a whole-prefix forward — which is exactly
    // how it stayed missing until now — and wrong on the first
    // continuation step.
    let t_len = hidden.len();
    let history_len = kernel - 1;
    let history: Vec<f32> = state.buffer(CONV_HISTORY).cells().to_vec();
    // `history` is `[conv_dim][kernel]`, oldest first; the newest
    // `kernel-1` entries are the positions preceding this batch.
    let past = |c: usize, back: usize| -> f32 {
        // `back` = 1 means the position immediately before the batch.
        let slot = kernel - back;
        history[c * kernel + slot]
    };
    let mut conv: Vec<Vec<f32>> = vec![vec![0.0; conv_dim]; t_len];
    let convolution = timed(OpClass::DeltaConv);
    for c in 0..conv_dim {
        let taps = &w.conv1d[c * kernel..(c + 1) * kernel];
        for t in 0..t_len {
            let mut acc = 0.0f32;
            for (i, tap) in taps.iter().enumerate() {
                // Causal: left-padded by kernel-1, so tap i reads
                // position t - (kernel-1) + i. Centring it would let the
                // position read its own future.
                let offset = if mutation == Mutation::CentredConv {
                    t as isize - (kernel as isize / 2) + i as isize
                } else {
                    t as isize - (kernel as isize - 1) + i as isize
                };
                if offset < 0 {
                    // Before this batch: the durable history answers, and
                    // its zeros ARE correct at sequence start because the
                    // buffer was initialised to zeros there.
                    let back = (-offset) as usize;
                    if back <= history_len {
                        acc += tap * past(c, back);
                    }
                    continue;
                }
                if (offset as usize) < t_len {
                    let x = mixed[offset as usize][c];
                    acc += tap
                        * if mutation == Mutation::SiluBeforeConv {
                            silu(x)
                        } else {
                            x
                        };
                }
            }
            conv[t][c] = if mutation == Mutation::SiluBeforeConv {
                acc
            } else {
                silu(acc)
            };
        }
    }
    drop(convolution);

    // **CPU-7C.** `a`, `b` and `z` read `hidden[t]` — the layer input —
    // and never the recurrent state, so all three are eligible across
    // positions and are lifted out of the sequential loop below. The
    // recurrence still advances strictly in order; what moved is where
    // its OPERANDS are computed, which is a schedule and not a semantics.
    let gate_a = proj.project_many(w.in_proj_a, &rows, hv);
    let gate_b = proj.project_many(w.in_proj_b, &rows, hv);
    let gate_z = proj.project_many(w.in_proj_z, &rows, value_dim);

    // `out_proj` is downstream of the recurrence: its operand for position
    // `t` does not exist until `step_inner` has consumed `t-1`. It is
    // therefore collected here and projected AFTER the loop — legal
    // because nothing feeds `out_proj`'s result back into the recurrence.
    let mut gated: Vec<Vec<f32>> = Vec::with_capacity(t_len);

    for t in 0..t_len {
        // Stage 3/4: split, then expand q/k from Hk heads to Hv.
        let expand = timed(OpClass::DeltaHeadExpand);
        let mut q = vec![0.0f32; hv * dk];
        let mut k = vec![0.0f32; hv * dk];
        for e in 0..hv {
            let src = if mutation == Mutation::TiledHeadExpansion {
                e % hk
            } else {
                e / repeat
            };
            for d in 0..dk {
                q[e * dk + d] = conv[t][src * dk + d];
                k[e * dk + d] = conv[t][key_dim + src * dk + d];
            }
        }
        let value = conv[t][key_dim * 2..key_dim * 2 + value_dim].to_vec();
        drop(expand);

        // Stage 5: the gates. The three projections are timed by the
        // executor; only the elementwise part is this leaf.
        let (a, b, z) = (&gate_a[t], &gate_b[t], &gate_z[t]);
        let gates = timed(OpClass::DeltaGates);
        let beta: Vec<f32> = b.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let g: Vec<f32> = (0..hv)
            .map(|h| -w.a_log[h].exp() * softplus(a[h] + w.dt_bias[h]))
            .collect();
        drop(gates);

        // Stage 6: the proven recurrence.
        let recurrence = timed(OpClass::DeltaRecurrence);
        let core = step_inner(
            op,
            &RecurrenceStep {
                query: &q,
                key: &k,
                value: &value,
                g: &g,
                beta: &beta,
            },
            state.buffer_mut(DELTA_MATRIX),
            mutation,
        );

        drop(recurrence);

        // Stage 7: gated RMSNorm, per value head over Dv. Norm, then the
        // weight, then the SiLU'd gate — that order is the reference's.
        let gated_norm = timed(OpClass::DeltaGatedNorm);
        let mut normed = vec![0.0f32; value_dim];
        for h in 0..hv {
            let row = &core[h * dv..(h + 1) * dv];
            let var: f32 = row.iter().map(|x| x * x).sum::<f32>() / dv as f32;
            let inv = 1.0 / (var + w.norm_eps).sqrt();
            for d in 0..dv {
                normed[h * dv + d] = w.norm[d] * (row[d] * inv) * silu(z[h * dv + d]);
            }
        }

        drop(gated_norm);

        // Stage 8: held for the batched `out_proj` below.
        gated.push(normed);
        planes.query.push(q);
        planes.key.push(k);
        planes.value.push(value);
        planes.g.push(g);
        planes.beta.push(beta);
        planes.z.push(z.clone());
        planes.core.push(core);
    }

    // Stage 8, continued: one traversal of `out_proj` for every position,
    // now that the recurrence has produced all of their operands.
    if let Some(width) = hidden.first().map(Vec::len) {
        let gated_rows: Vec<&[f32]> = gated.iter().map(Vec::as_slice).collect();
        planes.output = proj.project_many(w.out_proj, &gated_rows, width);
    }

    // Roll the convolution history forward: the last `kernel` positions of
    // the PRE-convolution projection, oldest first — the same window HF's
    // `conv_state.copy_(cat([conv_state, x])[..., -state_len:])` keeps.
    //
    // Taken from `mixed`, not from `conv`: the buffer holds the
    // convolution's INPUT, and seeding it with the output would feed the
    // next batch a doubly-convolved signal.
    {
        let history = state.buffer_mut(CONV_HISTORY);
        let cells = history.cells_mut();
        for c in 0..conv_dim {
            for slot in 0..kernel {
                // `slot` counts from oldest to newest across the window
                // ending at the last position of this batch.
                let back = kernel - 1 - slot;
                let value = if back < t_len {
                    mixed[t_len - 1 - back][c]
                } else {
                    // The batch was shorter than the window, so the tail
                    // of the PREVIOUS history still occupies these slots.
                    let older = back - t_len;
                    if older < history_len {
                        past(c, older + 1)
                    } else {
                        0.0
                    }
                };
                cells[c * kernel + slot] = value;
            }
        }
    }
    planes
}
