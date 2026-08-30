//! One Kimi Linear MoE block, executed — deliberately boring.
//!
//! Transcribed from `KimiSparseMoeBlock.forward`/`moe_infer` in the
//! checkpoint's own `modeling_kimi.py`:
//!
//! ```text
//! ids, weights = route(x)                          # kimi_router::route
//! for id in ids:
//!     expert_out[id] = w2(silu(w1(x)) * w3(x))       # KimiBlockSparseMLP
//! routed_sum = Σ weights[i] * expert_out[ids[i]]
//! shared_out = w2(silu(w1(x)) * w3(x))               # KimiMLP, the shared expert
//! output = routed_sum + shared_out                  # UNSCALED — see below
//! ```
//!
//! **Execution order is NOT the reference's textual order** (P4d): the
//! pseudocode above computes `shared_out` sequentially after the routed
//! loop because that is how `modeling_kimi.py` reads, but `shared_out`
//! depends only on `x`, exactly like every routed expert — so this
//! module dispatches all nine (up to eight routed plus the shared
//! branch) as one `parallel_map` fan-out and only THEN computes
//! `routed_sum`/`output`. Same values, same reduction order
//! (`expert_outputs[i]` is still `expert_out[ids[i]]`) — only WHERE and
//! WHEN each branch's arithmetic runs changed.
//!
//! **The shared branch is summed, never scaled.** `routed_scaling_factor`
//! (2.446) is already folded into `weights` by [`kimi_router::route`] — it
//! multiplies each ROUTED expert's contribution before the sum, and
//! `y = y + self.shared_experts(identity)` in the reference adds the
//! shared branch afterward, plain. Multiplying the shared output by the
//! branch scale too would be the Gemma-4-hybrid shape (a dense branch
//! summed with a separately-scaled routed one) grafted onto a checkpoint
//! that does not do that — P3d-e's operand closure work already
//! established the two are distinct roles for exactly this reason.
//!
//! **Deliberately boring, except for one representation choice.** This
//! rung is one MoE block in isolation, the combine a plain f32
//! transcription, and the caller supplies each selected expert's weights
//! ONE AT A TIME rather than this module ever holding all 256 — the
//! entire point of routing is that most of them are never read.
//!
//! **P4a: expert weights are BF16 code units, never F32.** The
//! CANONICAL container already stores every tensor this checkpoint has
//! as BF16 — an f32 expert here would be a materialisation this format
//! does not need, not a representation it started in. Measured before
//! this change: MLA and `lm_head` were 20-22× slower than they needed to
//! be from calling the WRONG KERNEL (`exec::kernels::matvec`, a
//! deliberately naive f32 reference) rather than a wrong representation
//! — that fix alone was a 5.0× decode speedup. This one is a
//! representation fix: [`super::cpu::kernels::FusedBf16`] widens each
//! BF16 code to f32 IN REGISTERS and multiplies against the f32
//! activation immediately — CPU-1B's own finding was that widening to a
//! scratch buffer FIRST costs more than the traffic it saves (27.3 GB/s
//! against a fused kernel's 122.0), so there is no F32 expansion at any
//! point between disk and the matvec's own arithmetic.

use super::cpu::executor;
use super::cpu::kernels::FusedBf16;
use super::cpu::projector::{DenseProjector, WeightRows};
use super::kimi_router::{route, Mutation, RouterTrace};
use super::timing::{timed, OpClass};

/// One expert's three projections, gate(`w1`)/up(`w3`)/down(`w2`) in the
/// checkpoint's own naming — never alphabetic order, see
/// [`larql_models::architectures::kimi`]. BF16 code units: each `u16` is
/// the top 16 bits of the f32 value it denotes (see `WeightRows::Bf16`'s
/// own doc comment) — the SAME bits the checkpoint stores on disk, never
/// upconverted to f32 and held that way.
#[derive(Debug, Clone, Copy)]
pub struct ExpertWeights<'a> {
    /// `w1`, `[inter, hidden]`.
    pub gate: &'a [u16],
    /// `w3`, `[inter, hidden]`.
    pub up: &'a [u16],
    /// `w2`, `[hidden, inter]`.
    pub down: &'a [u16],
}

/// Every stage of one block call: the router's own trace, each selected
/// expert's UNWEIGHTED output (so a test can check expert 3 of 8 in
/// isolation), the weighted routed sum, the shared branch, and the final
/// output — the same "every stage inspectable" posture `kimi_router`
/// takes for its own pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeBlockTrace {
    pub router: RouterTrace,
    /// `expert_outputs[i]` is the UNWEIGHTED output of expert
    /// `router.selected_ids[i]` — same order, same length.
    pub expert_outputs: Vec<Vec<f32>>,
    /// `Σ router.weights[i] * expert_outputs[i]`.
    pub routed_sum: Vec<f32>,
    /// The shared expert's output — computed once, never weighted, never
    /// scaled by `routed_scaling_factor`.
    pub shared_output: Vec<f32>,
    /// `routed_sum + shared_output`.
    pub output: Vec<f32>,
}

fn matvec(w: &[u16], x: &[f32], out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out];
    FusedBf16.project_rows(WeightRows::Bf16(w), x, &mut y);
    y
}

/// `pub(crate)` so an alternative execution of this same FFN (the Metal
/// bf16 rung) can compose the identical activation rather than
/// transcribe it: two arms that differ in the matvec AND in the
/// non-linearity would not isolate either.
pub(crate) fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// One expert's forward: `w2(silu(w1(x)) * w3(x))` — `KimiBlockSparseMLP`
/// and `KimiMLP` (the shared expert) share this exact shape; only the
/// weights and `inter` differ.
pub fn expert_ffn(x: &[f32], w: ExpertWeights<'_>, hidden: usize, inter: usize) -> Vec<f32> {
    let gate = matvec(w.gate, x, inter);
    let up = matvec(w.up, x, inter);
    let h: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
    matvec(w.down, &h, hidden)
}

/// One job in the block's fan-out (P4d): a routed expert by id, or the
/// shared branch. `Copy` — cheap enough to pass by value into a
/// `parallel_map` closure rather than borrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoeJob {
    Routed(usize),
    Shared,
}

/// One MoE block on one token. `expert_weights(id)` is called exactly
/// once per selected id — the caller loads only those, never all
/// `experts`. `shared` is `None` for a checkpoint that declares no shared
/// expert (`shared_output` and hence its contribution to `output` is then
/// all zeros, matching `if self.config.num_shared_experts is not None`).
#[allow(clippy::too_many_arguments)]
pub fn moe_block_forward<'a>(
    x: &[f32],
    hidden: usize,
    inter: usize,
    router_weight: &[f32],
    router_bias: &[f32],
    experts: usize,
    top_k: usize,
    renormalize: bool,
    branch_scale: f64,
    expert_weights: impl Fn(usize) -> ExpertWeights<'a> + Sync,
    shared: Option<(ExpertWeights<'a>, usize)>,
) -> MoeBlockTrace {
    let router = {
        let _t = timed(OpClass::MoeRouter);
        route(
            x,
            router_weight,
            router_bias,
            experts,
            top_k,
            renormalize,
            branch_scale,
            Mutation::None,
        )
    };

    // P4b-1: eight selected experts are eight independent jobs. P4d adds
    // the shared branch as a NINTH — it depends only on `x`, same as
    // every routed expert, so it is independent of them too. One
    // `parallel_map` fan-out over all nine, not a nested `join` between
    // "the routed fan-out" and "the shared branch": nesting a second
    // parallel dispatch INSIDE (or alongside, sharing the pool with) an
    // already-parallel `parallel_map` is exactly the P4c-2a mistake —
    // either the outer dispatch collapses to serial when it detects it
    // is already inside a rayon worker, or, worse, two independent
    // dispatches fight over the same cores. Folding shared into the SAME
    // single-level fan-out sidesteps the question entirely: there is
    // only ever one owner of the pool for this whole block.
    //
    // Order-preserving, same argument as P4b-1: `jobs[i]` maps to
    // `results[i]` regardless of which worker computed it or when, so
    // popping the shared job's result (always last, since it is always
    // pushed last) and using the rest in `selected_ids` order changes
    // WHERE the arithmetic runs, never what it sums.
    let mut jobs: Vec<MoeJob> = router
        .selected_ids
        .iter()
        .map(|&id| MoeJob::Routed(id))
        .collect();
    if shared.is_some() {
        jobs.push(MoeJob::Shared);
    }
    let mut results: Vec<Vec<f32>> = {
        let _t = timed(OpClass::MoeFanout);
        let pool = executor::shared().expect("the CPU executor pool is unavailable");
        pool.parallel_map(&jobs, |job| match *job {
            MoeJob::Routed(id) => {
                let _t = timed(OpClass::MoeRoutedExpert);
                expert_ffn(x, expert_weights(id), hidden, inter)
            }
            MoeJob::Shared => {
                let _t = timed(OpClass::MoeSharedExpert);
                let (w, shared_inter) =
                    shared.expect("a Shared job is only pushed when shared.is_some()");
                expert_ffn(x, w, hidden, shared_inter)
            }
        })
    };
    let shared_output = if shared.is_some() {
        results
            .pop()
            .expect("a Shared job was pushed, so its result exists")
    } else {
        vec![0.0f32; hidden]
    };
    let expert_outputs = results;

    let mut routed_sum = vec![0.0f32; hidden];
    {
        let _t = timed(OpClass::Residual);
        for (out, &w) in expert_outputs.iter().zip(&router.weights) {
            for (s, &v) in routed_sum.iter_mut().zip(out) {
                *s += v * w;
            }
        }
    }

    let output: Vec<f32> = {
        let _t = timed(OpClass::Residual);
        routed_sum
            .iter()
            .zip(&shared_output)
            .map(|(&r, &s)| r + s)
            .collect()
    };

    MoeBlockTrace {
        router,
        expert_outputs,
        routed_sum,
        shared_output,
        output,
    }
}
