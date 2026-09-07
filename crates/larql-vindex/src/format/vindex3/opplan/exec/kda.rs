//! Kimi Delta Attention, executed.
//!
//! The recurrent path only, for `T ≤ 64` — the range where the reference
//! itself uses `fused_recurrent_kda` rather than its chunked kernel. The
//! chunk path is a separate rung: it is a different *implementation* of
//! the same recurrence, and proving the recurrence first is what makes a
//! later chunked version checkable against something.
//!
//! **The specification is the reference forward, not the config.** That
//! distinction is load-bearing here and not a slogan:
//! [`KdaOp::gate_lower_bound`](super::super::kda::KdaOp) is declared by
//! both observed checkpoints and applied by neither, and applying it moves
//! this layer's output by a relative 1.75 with every shape still closing.
//! [`Mutation::ApplyGateLowerBound`] exists so that stays a measurement.
//!
//! ## Fault localisation, established before this code was written
//!
//! Ablating the two L2 normalisations against the oracle gives an
//! asymmetry worth keeping in mind while debugging:
//!
//! | ablation | output moves | recurrent state moves |
//! |---|---|---|
//! | omit **q** normalisation | yes | **no — exactly zero** |
//! | omit **k** normalisation | yes | yes |
//!
//! `q` never touches the state; it appears only in the readout. So a
//! disagreement that moves the STATE cannot be in the q path, and one that
//! moves only the OUTPUT is most likely q, the gated norm, or `o_proj`.

use larql_models::config::{KdaGateForm, KdaGeometry};

use super::continuation::{
    RecurrentBufferGeometry, RecurrentGeometry, RecurrentState, StateInitialization,
};
use super::cpu::executor;
use super::cpu::projector::{DenseProjector, WeightRows};
use super::timing::{timed, OpClass};

/// One layer's fifteen operands.
///
/// `q_proj`/`k_proj`/`v_proj`/`o_proj` arrive in whatever representation
/// they are RESIDENT as — the plan binds them through the same
/// `OperandRole` → `LoadedWeight` path as every other matrix, and the
/// backend's own format policy decides. BF16 is the measured fast case
/// (P4c-4: the checkpoint's own representation for KDA's four largest
/// projections, ~26 ms/token of the pre-fusion 44.97 ms KDA bucket,
/// routed through the executor's ROW-parallel path — see
/// [`matvec_bf16`]), and it stays exactly that; anything else goes
/// through the executor's format-aware dispatcher rather than being
/// refused for not being bf16.
///
/// Everything else stays plain `f32`: the convolution, gate and recurrence
/// arithmetic is KDA-specific and small enough that compacting it is its
/// own later decision (banked, not bundled into this one — see this
/// module's own doc history for why bundling representation changes with
/// execution-strategy changes made the earlier expert rung hard to read).
#[derive(Clone, Copy)]
pub struct KdaWeights<'a> {
    pub q_proj: WeightRows<'a>,
    pub k_proj: WeightRows<'a>,
    pub v_proj: WeightRows<'a>,
    pub q_conv1d: &'a [f32],
    pub k_conv1d: &'a [f32],
    pub v_conv1d: &'a [f32],
    pub f_a_proj: &'a [f32],
    pub f_b_proj: &'a [f32],
    /// The output gate's projection in its DECLARED form. The low-rank
    /// pair is glue (f32, narrow); Kimi-K3's full-rank `g_proj` is a
    /// `[Hv·Dv, hidden]` matrix and rides the same row representation the
    /// four wide projections do. Only this projection differs between
    /// forms — its sigmoid and the gated norm below are the same code.
    pub output_gate: KdaOutputGateWeights<'a>,
    pub b_proj: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub o_norm: &'a [f32],
    pub o_proj: WeightRows<'a>,
    pub norm_eps: f32,
    /// The inner rank the two gate factorisations meet at — `f_a_proj`'s
    /// row count and `f_b_proj`'s column count, `KdaOp::gate_rank`.
    ///
    /// Carried explicitly because **no config declares it** and the
    /// executor cannot recover it: it is resolved once, from the bound
    /// operands, at plan-build time. This module used to substitute
    /// `head_dim`, which is right on both observed checkpoints (Kimi
    /// Linear and GLM-5.3-Flash both factor through 128) and right for
    /// no stated reason — a coincidence two checkpoints agreed on, and
    /// the kind a fixture whose widths are all distinct is built to
    /// expose. It is a separate fact, so it is a separate field.
    pub gate_rank: usize,
    /// Which decay gate this checkpoint's family actually computes.
    ///
    /// Carried, never defaulted, because the two observed checkpoints
    /// declare the SAME `gate_lower_bound: -5.0` and do different things
    /// with it: Kimi Linear's reference reads the field nowhere,
    /// GLM-5.3-Flash's applies it. Measured on the real GLM layer 0
    /// against the pinned reference, swapping the forms moves the layer
    /// output by relative 2.50e-2 and the gate's own mean from -0.906 to
    /// -2.528 — a 2.8x error in the per-step decay that compounds with
    /// context and leaves every shape closing.
    pub gate_form: KdaGateForm,
}

/// Buffer indices this operator assigns within its
/// [`RecurrentState`] — the same contract [`super::mamba2`] states: the
/// storage layer holds four indexed buffers and knows nothing about what
/// they mean; this is the one place that mapping is written down.
pub const RECURRENT: usize = 0;
pub const CONV_Q: usize = 1;
pub const CONV_K: usize = 2;
pub const CONV_V: usize = 3;

/// The recurrent and convolution state a KDA layer carries between
/// calls, in the engine's generic terms.
///
/// Nothing here is indexed by position: the recurrent part is one
/// `D × D` matrix per head whatever the sequence length, and the
/// convolution part is the last `kernel - 1` inputs of each stream. That
/// is the whole reason a continuation planner is told state elements and
/// never a span.
///
/// **fp32 is a judgment, not a default.** Kimi Linear declares no state
/// precision anywhere — no `mamba_ssm_dtype`, no equivalent under any
/// spelling (`linear_attn_config` carries `num_heads`, `head_dim` and
/// `short_conv_kernel_size`, and nothing else) — so the schema has no
/// declared value to carry and the planner does not get to pick one. The
/// reference does: `fla`'s `naive_recurrent_kda`, which the checkpoint's
/// own `modeling_kimi.py` calls through `fused_recurrent_kda`, holds the
/// state as `torch.float32` and casts q, k, v, g and beta into it every
/// step — transcribed, sha-pinned, in `scripts/kda_reference.py:130`.
/// The state is therefore held at the precision the reference computes
/// at, exactly as [`super::mamba2::state_geometry`] holds its sibling
/// judgment, and this function is the one place it lives.
///
/// Corroborated rather than merely read: the P3d parity ladder scores
/// this executor against that oracle at 2.1e-7 – 4.7e-7 relative on real
/// weights, a bound a bf16 state could not reach.
pub fn state_geometry(g: KdaGeometry) -> RecurrentGeometry {
    let width = g.value_width();
    // The window is `kernel - 1` inputs, not `kernel`: the current
    // position's own input is not history, it arrives with the call.
    // (Gated DeltaNet's conv buffer is a full `kernel` wide because HF
    // seeds it by left-padding the position INTO the buffer — a
    // different reference, a different shape, and the two must not be
    // made to agree by symmetry.)
    let tail = g.conv_kernel.saturating_sub(1);
    let conv = || RecurrentBufferGeometry {
        shape: vec![width, tail],
        dtype: super::super::gated_delta::StateDtype::Float32,
        initialization: StateInitialization::Zeros,
    };
    RecurrentGeometry {
        buffers: vec![
            RecurrentBufferGeometry {
                shape: vec![g.num_heads, g.head_dim, g.head_dim],
                dtype: super::super::gated_delta::StateDtype::Float32,
                initialization: StateInitialization::Zeros,
            },
            conv(),
            conv(),
            conv(),
        ],
    }
}

/// The zero state a sequence starts from.
pub fn zero_state(g: KdaGeometry) -> RecurrentState {
    RecurrentState::zeros(&state_geometry(g))
}

/// Every boundary the operator crosses, kept so a disagreement names its
/// own stage rather than being debugged backwards from the layer output.
///
/// The names and order match `BOUNDARIES` in `scripts/kda_reference.py`,
/// so a fixture and a report cannot drift on what a stage means.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct KdaPlanes {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub q_conv: Vec<f32>,
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub f_lowrank: Vec<f32>,
    pub g_decay: Vec<f32>,
    pub beta: Vec<f32>,
    pub recurrent_out: Vec<f32>,
    pub o_gate: Vec<f32>,
    pub o_norm: Vec<f32>,
    pub output: Vec<f32>,
}

/// Deliberate defects, for the negative controls.
///
/// These perturb the REAL function rather than a copy of it: a control
/// that mutates a duplicate proves only that the duplicate is detectable.
/// Same posture as Gated DeltaNet's `Mutation`, and the same reason.
/// The output gate's projection weights, one variant per declared form
/// ([`KdaOutputGate`](super::super::kda::KdaOutputGate)).
#[derive(Clone, Copy)]
pub enum KdaOutputGateWeights<'a> {
    /// `g = g_b_proj · (g_a_proj · x)` — Kimi Linear, GLM-5.3-Flash.
    LowRank {
        /// `[rank, hidden]`.
        g_a_proj: &'a [f32],
        /// `[Hv·Dv, rank]`.
        g_b_proj: &'a [f32],
    },
    /// `g = g_proj · x` — Kimi-K3 (`use_full_rank_gate: true`).
    FullRank {
        /// `[Hv·Dv, hidden]`, at whatever representation it is resident as.
        g_proj: WeightRows<'a>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mutation {
    None,
    /// Force the clamped-sigmoid decay, `bound * sigmoid(exp(A_log) * pre)`.
    ///
    /// On a [`KdaGateForm::Softplus`] checkpoint (Kimi Linear) this is the
    /// form the reference does *not* use, and the control pins
    /// `gate_lower_bound` as provenance by measurement.
    ApplyGateLowerBound(f32),
    /// Force the softplus decay, `-exp(A_log) * softplus(pre)`.
    ///
    /// The mirror of [`Self::ApplyGateLowerBound`], and the one that
    /// matters on a [`KdaGateForm::ClampedSigmoid`] checkpoint
    /// (GLM-5.3-Flash): it is exactly what running Kimi's executor
    /// unchanged on GLM would compute. A single-sided control would have
    /// certified the wrong direction — the gate form has to be shown to
    /// matter on BOTH families, or "it is declared" is untested.
    ForceSoftplusGate,
    /// Skip the query L2 normalisation. Must move the output and leave the
    /// state untouched.
    NoQNorm,
    /// Skip the key L2 normalisation. Must move both.
    NoKNorm,
    /// Round the recurrent state to bf16 precision each step, so an
    /// "optimisation" that drops the f32 promotion is caught.
    Bf16Recurrence,
    /// Read the state with q BEFORE the rank-1 write, so the current
    /// position cannot see its own contribution.
    ReadBeforeWrite,
    /// Skip the decay entirely.
    NoDecay,
    /// Drop beta from the delta rule.
    NoBeta,
    /// The output gate skipped: `sigmoid(0) = 0.5` on every channel.
    /// K3-REP-GATE-1's first gate control; caught at `o_norm`.
    GateSkipped,
    /// The gate applied to the recurrent output BEFORE the RMS norm
    /// instead of after it (`FusedRMSNormGated`'s norm-then-gate order
    /// inverted). Caught at `o_norm`.
    GateBeforeNorm,
    /// The raw pre-activation multiplied in, no sigmoid. Caught at
    /// `o_norm`.
    SigmoidOmitted,
    /// The gate applied to `v` before the recurrence and not after it —
    /// a placement defect, since the reference gates the aggregate.
    /// Caught at `o_norm` (and everything downstream of the recurrence).
    GateOnValueBeforeRecurrence,
    /// Write `v` instead of the prediction error `v - kᵀS` — the single
    /// most plausible wrong transcription of a delta rule, and one that
    /// agrees at `T = 1` from a zero state.
    WriteValueNotError,
}

/// `y = W x` with `W` row-major `[out, inp]`.
///
/// Routed to the crate's existing BLAS projector rather than written out
/// here, and the split is deliberate: the **projections are ordinary
/// linear algebra** that infrastructure already does well, while the
/// convolution, gates, recurrence and gated norm are KDA-specific and stay
/// a plain f32 transcription.
///
/// A scalar loop here cost 57 s on the full-width fixture, which is not a
/// performance problem so much as a debugging one — every later
/// integration rung would have paid it. Swapping only this function leaves
/// the recurrence byte-for-byte the same code, so the full-width parity
/// gate re-run is a real check that acceleration changed no semantics.
fn matvec(w: &[f32], x: &[f32], out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out];
    super::cpu::kernels::BlasF32.project_rows(WeightRows::F32(w), x, &mut y);
    y
}

/// `y = W x` for a BF16-compact `W`, through the executor's ROW-PARALLEL
/// path — `executor::shared().project_as(class, ...)`, not a single
/// `project_rows` call, and NOT `project()`: that would nest the
/// executor's own `OpClass::Projection` timer inside the caller-supplied
/// `class` (both fire, both accumulate — `timing.rs`'s "nothing nests"
/// contract violated, caught by its own `nested()` counter the first
/// real-weight run made). `project_as` times under `class` alone.
///
/// `FusedBf16` is `CpuParallelism::ExternalPool` (`cpu/kernels.rs`): it
/// needs EXTERNAL row-splitting across workers to reach its real
/// throughput, measured directly at P4a — called the single-shot way
/// `BlasF32` is called, it LOSES to `BlasF32`'s own Accelerate-internal
/// threading despite moving half the bytes. P4b-1 already applied that
/// lesson with TASK-level fan-out over the eight independent experts;
/// there is only ONE q/k/v/o_proj per layer, so the equivalent fan-out
/// here is ROW-level, which `executor::project` already implements — no
/// new mechanism, the existing one aimed at what it was built for.
///
/// **Sequential by construction, one call at a time** (P4c-4): P4c-2a
/// measured what happens when several already-threaded projections run
/// CONCURRENTLY on this machine — each one's own time roughly TRIPLED
/// from core contention, for a wall-clock win of only ~12%. Every call
/// site of this function runs on the caller's own thread, never nested
/// inside another parallel dispatch, so the executor's own "at most one
/// layer of parallelism owns the machine" rule gives this the whole pool
/// deliberately, rather than sharing it the way that rung found out costs
/// more than it saves.
/// How the four wide BF16 projections are executed.
///
/// **A seam that can BATCH, which is the whole reason it exists.** The
/// crate's `DenseProjector` takes one matrix at a time, which is right
/// for a CPU kernel and cannot express the shape a device wants: q, k
/// and v read the SAME input, so a backend that submits work in batches
/// should send all three together. Rung 5a priced one CPU↔GPU
/// command-buffer boundary at ~0.23 ms, so four separate crossings a
/// layer is ~0.9 ms of pure orchestration before any arithmetic — which
/// is more than the projections themselves are likely to cost.
///
/// So this trait is shaped by the dependency structure, not by the
/// matrix: `qkv` together because they share an input, `o` alone because
/// it consumes the recurrence's output and cannot be known earlier.
///
/// Everything else about KDA stays exactly where it is. The convolution,
/// norms, gates and recurrence remain the proven CPU path, and an
/// implementation of this trait changes only WHERE four matvecs run.
pub trait KdaProjections: Sync {
    /// `q_proj(x)`, `k_proj(x)`, `v_proj(x)` — all `[width, hidden]`
    /// against the same normalised hidden state.
    fn qkv(&self, w: KdaWeights<'_>, x: &[f32], width: usize) -> [Vec<f32>; 3];

    /// `o_proj(x)` — `[hidden, width]` against the gated, normed value
    /// stream, which does not exist until the recurrence has run.
    fn o(&self, w: WeightRows<'_>, x: &[f32], out: usize) -> Vec<f32>;
}

/// The proven CPU path: three row-parallel projections, one at a time.
///
/// Sequential by construction (P4c-4): P4c-2a measured concurrent
/// already-threaded projections at roughly TRIPLE each one's own time
/// from core contention, for a wall-clock win of only ~12%.
pub struct CpuKdaProjections;

impl KdaProjections for CpuKdaProjections {
    fn qkv(&self, w: KdaWeights<'_>, x: &[f32], width: usize) -> [Vec<f32>; 3] {
        [
            project(OpClass::KdaQProj, w.q_proj, x, width),
            project(OpClass::KdaKProj, w.k_proj, x, width),
            project(OpClass::KdaVProj, w.v_proj, x, width),
        ]
    }

    fn o(&self, w: WeightRows<'_>, x: &[f32], out: usize) -> Vec<f32> {
        project(OpClass::KdaOProj, w, x, out)
    }
}

/// One wide projection, at the resident representation.
///
/// BF16 keeps its own call — the row-parallel `FusedBf16` path P4c-4
/// measured and every Kimi number quotes. Every other representation goes
/// to the executor's format-aware dispatcher, the same one every
/// production matrix uses, rather than being refused here: a container
/// that stores these projections q8 is a representation decision, not a
/// different operator, and the recurrence below is unchanged either way.
fn project(class: OpClass, rows: WeightRows<'_>, x: &[f32], out: usize) -> Vec<f32> {
    match rows {
        WeightRows::Bf16(w) => matvec_bf16(class, w, x, out),
        other => {
            let _t = timed(class);
            super::cpu::physical::project_rows(other, x, out)
                .expect("the CPU executor pool is unavailable")
        }
    }
}

/// KDA's four wide projections through **the backend's own** dense
/// projector — the arm the plan-driven path uses.
///
/// [`CpuKdaProjections`] is this operator's measured CPU realisation and
/// picks its own kernels; this one asks the backend, exactly as
/// [`super::gated_delta`] and [`super::mamba2`] do, so a reference run
/// gets the scalar oracle and a production run gets the executor's
/// format-aware dispatch without KDA choosing for either.
///
/// No timers here on purpose: the projector below already times what it
/// runs, and wrapping it would nest two intervals over one operation —
/// the "nothing nests" contract `timing.rs` enforces with its own
/// counter.
pub struct BackendKdaProjections<'a>(pub &'a dyn super::gated_delta::DenseProjections);

impl KdaProjections for BackendKdaProjections<'_> {
    fn qkv(&self, w: KdaWeights<'_>, x: &[f32], width: usize) -> [Vec<f32>; 3] {
        [
            self.0.project(w.q_proj, x, width),
            self.0.project(w.k_proj, x, width),
            self.0.project(w.v_proj, x, width),
        ]
    }

    fn o(&self, w: WeightRows<'_>, x: &[f32], out: usize) -> Vec<f32> {
        self.0.project(w, x, out)
    }
}

fn matvec_bf16(class: OpClass, w: &[u16], x: &[f32], out: usize) -> Vec<f32> {
    executor::shared()
        .expect("the CPU executor pool is unavailable")
        .project_as(
            class,
            &super::cpu::kernels::FusedBf16,
            WeightRows::Bf16(w),
            x,
            out,
        )
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn softplus(v: f32) -> f32 {
    // `ln(1+e^v)`, in the numerically stable form: for large `v` the naive
    // expression overflows to `inf` and then to a `NaN` gate.
    if v > 20.0 {
        v
    } else {
        v.exp().ln_1p()
    }
}

/// Depthwise causal convolution over one stream, then SiLU.
///
/// Causal because the window is the `kernel-1` PREVIOUS inputs plus the
/// current one; a symmetric padding would let position `t` read `t+1`.
/// `window` carries those previous inputs across calls, so a continuation
/// produces exactly what one pass over the concatenation would.
fn short_conv(
    x: &[f32],
    weight: &[f32],
    window: &mut [f32],
    width: usize,
    kernel: usize,
) -> Vec<f32> {
    let tail = kernel - 1;
    let mut out = vec![0.0f32; width];
    for c in 0..width {
        let w = &weight[c * kernel..(c + 1) * kernel];
        let hist = &window[c * tail..(c + 1) * tail];
        // Oldest first: history then the current sample.
        let mut acc = 0.0f32;
        for (i, weight_i) in w.iter().enumerate().take(tail) {
            acc += weight_i * hist[i];
        }
        acc += w[tail] * x[c];
        out[c] = silu(acc);
    }
    // Slide every window one position, dropping the oldest.
    for c in 0..width {
        let hist = &mut window[c * tail..(c + 1) * tail];
        for i in 0..tail.saturating_sub(1) {
            hist[i] = hist[i + 1];
        }
        if tail > 0 {
            hist[tail - 1] = x[c];
        }
    }
    out
}

/// L2-normalise each head's slice in place.
///
/// Applied to q and k inside the reference's kernel
/// (`use_qk_l2norm_in_kernel=True`), which is why it appears nowhere in
/// the checkpoint's modeling file — and why it is the easiest operation in
/// the whole block to omit by accident.
fn l2_normalise_heads(v: &mut [f32], heads: usize, dim: usize) {
    for h in 0..heads {
        let head = &mut v[h * dim..(h + 1) * dim];
        let norm = head.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Matches `F.normalize`'s clamp: a zero head stays zero rather
        // than becoming NaN.
        let inv = 1.0 / norm.max(1e-12);
        for x in head.iter_mut() {
            *x *= inv;
        }
    }
}

fn bf16_round(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xFFFF_0000)
}

/// One position through the block, advancing `state`.
///
/// Returns the layer output for this position and appends every boundary
/// to `planes`.
#[allow(clippy::too_many_arguments)]
pub fn step(
    x: &[f32],
    w: KdaWeights<'_>,
    g: KdaGeometry,
    state: &mut RecurrentState,
    planes: &mut KdaPlanes,
    mutation: Mutation,
) -> Vec<f32> {
    step_with(&CpuKdaProjections, x, w, g, state, planes, mutation)
}

/// [`step`] with the projections executed somewhere the caller chooses.
#[allow(clippy::too_many_arguments)]
pub fn step_with(
    projections: &dyn KdaProjections,
    x: &[f32],
    w: KdaWeights<'_>,
    g: KdaGeometry,
    state: &mut RecurrentState,
    planes: &mut KdaPlanes,
    mutation: Mutation,
) -> Vec<f32> {
    let (heads, dim) = (g.num_heads, g.head_dim);
    let width = g.value_width();

    // q/k/v/o_proj are BF16, executed ONE AT A TIME through the executor's
    // row-parallel path (P4c-4 — see `matvec_bf16`'s own doc comment for
    // why sequential, not concurrent). The gate arithmetic stays plain f32
    // via the single-call `matvec`, unchanged from before P4c-2a — this
    // rung deliberately narrows to the four wide projections only.
    // All three at once. They are mutually independent — each feeds its
    // own convolution window and its own norm — so hoisting them above
    // the convolutions reorders nothing observable, and it is what lets a
    // batching backend see them together.
    let [q_p, k_p, v_p] = projections.qkv(w, x, width);
    let mut q = {
        let _t = timed(OpClass::KdaConv);
        short_conv(
            &q_p,
            w.q_conv1d,
            state.buffer_mut(CONV_Q).cells_mut(),
            width,
            g.conv_kernel,
        )
    };
    planes.q_conv.extend_from_slice(&q);
    {
        let _t = timed(OpClass::KdaQkNorm);
        if mutation != Mutation::NoQNorm {
            l2_normalise_heads(&mut q, heads, dim);
        }
    }
    planes.q_norm.extend_from_slice(&q);

    let mut k = {
        let _t = timed(OpClass::KdaConv);
        short_conv(
            &k_p,
            w.k_conv1d,
            state.buffer_mut(CONV_K).cells_mut(),
            width,
            g.conv_kernel,
        )
    };
    planes.k_conv.extend_from_slice(&k);
    {
        let _t = timed(OpClass::KdaQkNorm);
        if mutation != Mutation::NoKNorm {
            l2_normalise_heads(&mut k, heads, dim);
        }
    }
    planes.k_norm.extend_from_slice(&k);

    let mut v = {
        let _t = timed(OpClass::KdaConv);
        short_conv(
            &v_p,
            w.v_conv1d,
            state.buffer_mut(CONV_V).cells_mut(),
            width,
            g.conv_kernel,
        )
    };
    planes.v_conv.extend_from_slice(&v);

    // The decay gate. Everything from here is f32 by construction: the
    // gate is an exponential of a softplus, and the recurrence multiplies
    // by it every step, so a narrower accumulator compounds.
    let decay = {
        let _t = timed(OpClass::KdaDecayGate);
        let f_low = matvec(w.f_b_proj, &matvec(w.f_a_proj, x, w.gate_rank), width);
        planes.f_lowrank.extend_from_slice(&f_low);
        let mut decay = vec![0.0f32; width];
        for h in 0..heads {
            let a = w.a_log[h].exp();
            for d in 0..dim {
                let i = h * dim + d;
                let pre = f_low[i] + w.dt_bias[i];
                // The DECLARED form, unless a control overrides it. Both
                // overrides exist so the choice can be falsified from
                // either side.
                let form = match mutation {
                    Mutation::ApplyGateLowerBound(bound) => {
                        KdaGateForm::ClampedSigmoid { lower_bound: bound }
                    }
                    Mutation::ForceSoftplusGate => KdaGateForm::Softplus,
                    _ => w.gate_form,
                };
                decay[i] = match form {
                    KdaGateForm::Softplus => -a * softplus(pre),
                    // `lower_bound * sigmoid(exp(A_log) * pre)`, bounding
                    // the decay below at `exp(lower_bound)`.
                    KdaGateForm::ClampedSigmoid { lower_bound } => {
                        lower_bound * (1.0 / (1.0 + (-(a * pre)).exp()))
                    }
                };
            }
        }
        planes.g_decay.extend_from_slice(&decay);
        decay
    };

    // The output gate's PROJECTION, in the declared form. `project` times
    // the full-rank matvec itself under the same class, so only the
    // low-rank composition is timed here.
    let mut gate = match w.output_gate {
        KdaOutputGateWeights::LowRank { g_a_proj, g_b_proj } => {
            let _t = timed(OpClass::KdaOutputGate);
            matvec(g_b_proj, &matvec(g_a_proj, x, w.gate_rank), width)
        }
        KdaOutputGateWeights::FullRank { g_proj } => {
            project(OpClass::KdaOutputGate, g_proj, x, width)
        }
    };
    if mutation == Mutation::GateSkipped {
        gate.iter_mut().for_each(|g| *g = 0.0);
    }
    planes.o_gate.extend_from_slice(&gate);
    if mutation == Mutation::GateOnValueBeforeRecurrence {
        for (vi, gi) in v.iter_mut().zip(&gate) {
            *vi /= 1.0 + (-gi).exp();
        }
    }

    let beta: Vec<f32> = {
        let _t = timed(OpClass::KdaBProj);
        let beta: Vec<f32> = matvec(w.b_proj, x, heads)
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        planes.beta.extend_from_slice(&beta);
        beta
    };

    // The delta rule, per head. `q` is scaled by `D^-1/2` at the readout.
    //
    // FUSED, P4c-3: the reference is four full `D×D` state traversals —
    // decay-write, predict-read, update-write, readout-read — but two of
    // those pairs are FUSABLE without changing a single summed value.
    //
    // (1) decay + predict: `pred[vv] = Σ_kk kh[kk] * (decay[kk]*s_old[kk,vv])`.
    // Decaying row `kk` is entirely local to that row, so decaying it and
    // immediately folding its contribution into every `pred[vv]` — before
    // moving to row `kk+1` — reads EXACTLY the values a separate later
    // predict pass would have read, and accumulates them in the SAME
    // kk-ascending order the original `.sum()` did. Not an approximation:
    // the identical sequence of additions in the identical order is
    // bit-identical under IEEE-754.
    //
    // (2) write + readout: `out[vv] = Σ_kk qh[kk]*scale*s_new[kk,vv]` where
    // `s_new[kk,vv]` is written earlier in the SAME kk iteration — same
    // argument, same conclusion.
    //
    // `Mutation::ReadBeforeWrite` is the one case that CANNOT take fusion
    // (2): its entire point is that the readout must see the PRE-write
    // state, so write and readout stay two passes there, exactly as
    // before — falling back to the unfused form for one control is not a
    // performance regression on the decode path, since production
    // decoding never uses this mutation.
    let out = {
        let _t = timed(OpClass::KdaRecurrence);
        let scale = (dim as f32).powf(-0.5);
        let mut out = vec![0.0f32; width];
        for h in 0..heads {
            let s =
                &mut state.buffer_mut(RECURRENT).cells_mut()[h * dim * dim..(h + 1) * dim * dim];
            let (qh, kh, vh) = (&q[h * dim..], &k[h * dim..], &v[h * dim..]);

            let mut pred = vec![0.0f32; dim];
            for kk in 0..dim {
                if mutation != Mutation::NoDecay {
                    let d = decay[h * dim + kk].exp();
                    for vv in 0..dim {
                        s[kk * dim + vv] *= d;
                    }
                }
                let kv = kh[kk];
                for vv in 0..dim {
                    pred[vv] += kv * s[kk * dim + vv];
                }
            }
            // The prediction error `v - kᵀS`, which is what the delta
            // rule writes against. Writing `v` instead agrees at T=1
            // from a zero state — see `Mutation::WriteValueNotError`.
            let mut err = vec![0.0f32; dim];
            for vv in 0..dim {
                err[vv] = match mutation {
                    Mutation::WriteValueNotError => vh[vv],
                    _ => vh[vv] - pred[vv],
                };
            }
            let b = if mutation == Mutation::NoBeta {
                1.0
            } else {
                beta[h]
            };

            if mutation == Mutation::ReadBeforeWrite {
                for vv in 0..dim {
                    out[h * dim + vv] = (0..dim).map(|kk| qh[kk] * scale * s[kk * dim + vv]).sum();
                }
                for kk in 0..dim {
                    let write = b * kh[kk];
                    for vv in 0..dim {
                        let cell = &mut s[kk * dim + vv];
                        *cell += write * err[vv];
                        if mutation == Mutation::Bf16Recurrence {
                            *cell = bf16_round(*cell);
                        }
                    }
                }
            } else {
                for kk in 0..dim {
                    let write = b * kh[kk];
                    let qv = qh[kk] * scale;
                    for vv in 0..dim {
                        let cell = &mut s[kk * dim + vv];
                        *cell += write * err[vv];
                        if mutation == Mutation::Bf16Recurrence {
                            *cell = bf16_round(*cell);
                        }
                        out[h * dim + vv] += qv * *cell;
                    }
                }
            }
        }
        planes.recurrent_out.extend_from_slice(&out);
        out
    };

    // `gate` was already computed above — it depends only on `x`, never
    // on `out`.

    // Gated RMSNorm: normalise over ONE head's width, scale by the
    // weight, then gate by `sigmoid(gate)`.
    let normed = {
        let _t = timed(OpClass::KdaGatedNorm);
        let mut normed = vec![0.0f32; width];
        // `GateBeforeNorm` gates `out` first and norms the gated vector;
        // every other arm norms `out` and applies the gate factor after.
        let pre: Vec<f32> = if mutation == Mutation::GateBeforeNorm {
            out.iter()
                .zip(&gate)
                .map(|(o, g)| o / (1.0 + (-g).exp()))
                .collect()
        } else {
            out.clone()
        };
        for h in 0..heads {
            let slice = &pre[h * dim..(h + 1) * dim];
            let ms = slice.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let inv = (ms + w.norm_eps).sqrt().recip();
            for (d, (sv, nv)) in slice.iter().zip(w.o_norm).enumerate() {
                let i = h * dim + d;
                let factor = match mutation {
                    Mutation::GateBeforeNorm | Mutation::GateOnValueBeforeRecurrence => 1.0,
                    Mutation::SigmoidOmitted => gate[i],
                    _ => 1.0 / (1.0 + (-gate[i]).exp()),
                };
                normed[i] = sv * inv * nv * factor;
            }
        }
        planes.o_norm.extend_from_slice(&normed);
        normed
    };

    planes.q_proj.extend_from_slice(&q_p);
    planes.k_proj.extend_from_slice(&k_p);
    planes.v_proj.extend_from_slice(&v_p);

    // `o_proj` is `[hidden, Hv·Dv]`, so its output width is the hidden
    // width this position arrived at — read from the ACTIVATION, never
    // from the weight's slice length: a resident slab is page-padded and
    // a compact representation is not even f32-shaped, so `len / width`
    // was only ever right for one residency.
    let y = projections.o(w.o_proj, &normed, x.len());
    planes.output.extend_from_slice(&y);
    y
}

/// A whole sequence through the block, from `state`.
pub fn layer_forward(
    x: &[f32],
    hidden: usize,
    w: KdaWeights<'_>,
    g: KdaGeometry,
    state: &mut RecurrentState,
    mutation: Mutation,
) -> KdaPlanes {
    layer_forward_with(&CpuKdaProjections, x, hidden, w, g, state, mutation)
}

/// [`layer_forward`] with the projections executed somewhere the caller
/// chooses. Everything but those four matvecs is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward_with(
    projections: &dyn KdaProjections,
    x: &[f32],
    hidden: usize,
    w: KdaWeights<'_>,
    g: KdaGeometry,
    state: &mut RecurrentState,
    mutation: Mutation,
) -> KdaPlanes {
    let mut planes = KdaPlanes::default();
    for pos in x.chunks_exact(hidden) {
        step_with(projections, pos, w, g, state, &mut planes, mutation);
    }
    planes
}
