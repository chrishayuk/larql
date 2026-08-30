//! Kimi Linear's MoE router, executed — sigmoid scores, a bias-corrected
//! SELECTION distinct from an unbiased WEIGHTING, renormalised, then
//! branch-scaled. Transcribed from `KimiMoEGate.forward` in the
//! checkpoint's own `modeling_kimi.py`:
//!
//! ```text
//! logits            = x @ W.T                                   # [experts]
//! scores            = sigmoid(logits)
//! selection_scores  = scores + e_score_correction_bias
//! ids               = topk(selection_scores, top_k)              # BIASED
//! weights           = scores[ids]                                # UNBIASED
//! weights          /= sum(weights) + 1e-20     # iff top_k > 1 and moe_renormalize
//! weights          *= routed_scaling_factor
//! ```
//!
//! **The selection/weighting split is the whole point of this file.**
//! `e_score_correction_bias` shifts which experts win; it must NEVER shift
//! how much they count once won — the two reads deliberately name
//! different arrays (`selection_scores` vs `scores`) so a future edit
//! cannot collapse them into one gather by accident. [`Mutation::
//! GatherBiasedWeights`] is the control that proves it: same ids, moved
//! weights.
//!
//! **Deliberately not modelled here**: expert groups
//! (`num_expert_group`/`topk_group`). Admission blocks any declaration
//! beyond the identity case (one group, `topk_group` 1 — see
//! `plan/tests/moe_spellings.rs`'s `more_than_one_expert_group_blocks`),
//! at which point `KimiMoEGate`'s group-topk masking is provably a no-op:
//! one group holds every expert, so nothing is masked out before the real
//! top-k. A general grouped-router path belongs with the checkpoint that
//! actually declares one, not guessed in ahead of it.
//!
//! **Boring on purpose.** This rung is router execution in isolation,
//! before touching a real expert — the projection is the crate's trusted
//! BLAS path (ordinary linear algebra, same posture `kda.rs` states for
//! its own projections); everything past it is a plain f32 transcription.

use super::cpu::kernels::BlasF32;
use super::cpu::projector::{DenseProjector, WeightRows};

/// Every stage of one router call, so a test can assert on ANY of them
/// independently rather than only the final weights — logits, sigmoid
/// scores, the bias-corrected selection scores, the selected ids, the
/// gathered (unbiased) weights, the renormalised weights, and the
/// branch-scaled weights actually returned.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterTrace {
    /// `x @ W.T`, `[experts]`.
    pub logits: Vec<f32>,
    /// `sigmoid(logits)`, `[experts]` — what selects AND, unbiased, what
    /// weighs.
    pub scores: Vec<f32>,
    /// `scores + e_score_correction_bias`, `[experts]` — SELECTS ONLY.
    pub selection_scores: Vec<f32>,
    /// Indices of the `top_k` largest `selection_scores`, ties broken by
    /// ascending index for a deterministic result (the reference's
    /// `sorted=False` topk does not promise an order this crate can
    /// reproduce bit-for-bit, and none is needed: the experts are summed).
    pub selected_ids: Vec<usize>,
    /// `scores[selected_ids]` — gathered from the UNBIASED array, `[top_k]`.
    pub gathered_weights: Vec<f32>,
    /// After `/= sum + 1e-20` (iff `top_k > 1` and renormalise is
    /// declared) — before the branch scale.
    pub normalized_weights: Vec<f32>,
    /// `normalized_weights * routed_scaling_factor` — what a routed FFN
    /// actually multiplies each selected expert's output by.
    pub weights: Vec<f32>,
}

/// One deliberate deviation from the correct computation, for a control to
/// perturb the REAL function rather than run a hand-rolled second one —
/// the same posture `exec/kda.rs::Mutation` uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutation {
    None,
    /// Skip adding `e_score_correction_bias` before selection — select on
    /// the unbiased scores too. On a fixture built so the bias flips the
    /// top-k boundary, this must change `selected_ids`.
    OmitBiasCorrection,
    /// Gather the FINAL weights from `selection_scores` (biased) instead
    /// of `scores` (unbiased) — the most plausible wrong transcription,
    /// and the one this file's whole design exists to make impossible.
    /// Must preserve `selected_ids` and move `weights`.
    GatherBiasedWeights,
    /// Skip the `/= sum + eps` renormalisation step.
    OmitRenormalization,
    /// Skip the `routed_scaling_factor` multiply.
    OmitBranchScale,
    /// Replace `sigmoid(logits)` with `softmax(logits)` over every expert
    /// — proves the function genuinely reads `moe_router_activation_func:
    /// sigmoid` rather than something order-preserving-only. Softmax
    /// scores also drive selection under this mutation, so `selected_ids`
    /// may move too: sigmoid and softmax rank experts differently in
    /// general, not just scale them.
    UseSoftmaxInsteadOfSigmoid,
}

/// `y = W x`, `W` row-major `[experts, hidden]` — routed to the crate's
/// trusted BLAS path, the same choice `kda.rs::matvec` makes for its own
/// projections.
fn matvec(w: &[f32], x: &[f32], experts: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; experts];
    BlasF32.project_rows(WeightRows::F32(w), x, &mut y);
    y
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Softmax over the whole slice — [`Mutation::UseSoftmaxInsteadOfSigmoid`]
/// only; Kimi's declared router never runs this.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|v| v / sum).collect()
}

/// Route one token: `x` is `[hidden]`, `router_weight` is `[experts,
/// hidden]` row-major, `router_bias` is `[experts]` (`e_score_correction_
/// bias`). `top_k` must be `<= experts` and `experts` must be nonzero —
/// both are closure facts a caller with a real `RoutedFfnOp` already has.
#[allow(clippy::too_many_arguments)]
pub fn route(
    x: &[f32],
    router_weight: &[f32],
    router_bias: &[f32],
    experts: usize,
    top_k: usize,
    renormalize: bool,
    branch_scale: f64,
    mutation: Mutation,
) -> RouterTrace {
    assert_eq!(router_bias.len(), experts, "router_bias must be [experts]");
    assert!(
        top_k > 0 && top_k <= experts,
        "top_k must be in 1..=experts"
    );

    let logits = matvec(router_weight, x, experts);
    let scores = if mutation == Mutation::UseSoftmaxInsteadOfSigmoid {
        softmax(&logits)
    } else {
        logits.iter().map(|&v| sigmoid(v)).collect()
    };
    let selection_scores: Vec<f32> = if mutation == Mutation::OmitBiasCorrection {
        scores.clone()
    } else {
        scores
            .iter()
            .zip(router_bias)
            .map(|(&s, &b)| s + b)
            .collect()
    };

    // Top-k by selection score, descending, ties broken by ascending
    // index — deterministic, and correct regardless of tie-break policy
    // because the selected experts are summed, order-invariant.
    let mut ranked: Vec<usize> = (0..experts).collect();
    ranked.sort_by(|&a, &b| {
        selection_scores[b]
            .partial_cmp(&selection_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let selected_ids: Vec<usize> = ranked[..top_k].to_vec();

    let gather_from = if mutation == Mutation::GatherBiasedWeights {
        &selection_scores
    } else {
        &scores
    };
    let gathered_weights: Vec<f32> = selected_ids.iter().map(|&i| gather_from[i]).collect();

    let normalized_weights =
        if renormalize && top_k > 1 && mutation != Mutation::OmitRenormalization {
            let sum: f32 = gathered_weights.iter().sum::<f32>() + 1e-20;
            gathered_weights.iter().map(|&w| w / sum).collect()
        } else {
            gathered_weights.clone()
        };

    let scale = if mutation == Mutation::OmitBranchScale {
        1.0
    } else {
        branch_scale
    };
    let weights: Vec<f32> = normalized_weights
        .iter()
        .map(|&w| (w as f64 * scale) as f32)
        .collect();

    RouterTrace {
        logits,
        scores,
        selection_scores,
        selected_ids,
        gathered_weights,
        normalized_weights,
        weights,
    }
}
