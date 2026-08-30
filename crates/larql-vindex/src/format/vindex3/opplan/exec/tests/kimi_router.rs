//! Kimi Linear's router, in isolation — before touching a real expert.
//!
//! One fixture proves every stage (logits, sigmoid scores, bias-corrected
//! selection, selected ids, gathered unbiased weights, renormalised
//! weights, branch-scaled weights) and one control per way the
//! computation could plausibly be wrong. The most important control is
//! [`Mutation::GatherBiasedWeights`]: it must preserve the selected ids
//! while moving the weights, because that is the one mistake that would
//! silently rescale every routed expert's contribution.
//!
//! Built so the correction bias FLIPS the top-k boundary — expert 2 loses
//! to expert 1 unbiased, and beats it once biased — so the
//! selection/weighting split is exercised on a real decision, not just a
//! magnitude change.

use crate::format::vindex3::opplan::exec::kimi_router::{route, Mutation};

const EXPERTS: usize = 4;
const HIDDEN: usize = 3;
const TOP_K: usize = 2;
const BRANCH_SCALE: f64 = 2.446; // Kimi's real `routed_scaling_factor`.

/// `x = [1, 0, 0]`, so `logits[e] = router_weight[e][0]` exactly — every
/// expected value below is computable by hand from `LOGITS` alone, with
/// no hidden dot-product arithmetic to get wrong.
const LOGITS: [f32; EXPERTS] = [2.0, 1.0, 0.5, -1.0];
/// Bias flips the boundary: unbiased, expert 1 (0.731) beats expert 2
/// (0.622); biased, expert 2 (0.622 + 0.3 = 0.922) beats expert 1 AND
/// expert 0 (0.881).
const BIAS: [f32; EXPERTS] = [0.0, 0.0, 0.3, 0.0];

fn x() -> Vec<f32> {
    vec![1.0, 0.0, 0.0]
}

fn router_weight() -> Vec<f32> {
    let mut w = vec![0.0f32; EXPERTS * HIDDEN];
    for e in 0..EXPERTS {
        w[e * HIDDEN] = LOGITS[e];
    }
    w
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn route_with(
    mutation: Mutation,
) -> crate::format::vindex3::opplan::exec::kimi_router::RouterTrace {
    route(
        &x(),
        &router_weight(),
        &BIAS,
        EXPERTS,
        TOP_K,
        true, // moe_renormalize
        BRANCH_SCALE,
        mutation,
    )
}

/// The router weight IS a projection: `logits = x @ W.T`, checked against
/// hand-placed values, not against the function's own internals.
#[test]
fn logits_are_the_router_projection() {
    let trace = route_with(Mutation::None);
    assert_eq!(trace.logits, LOGITS);
}

#[test]
fn scores_are_sigmoid_of_logits() {
    let trace = route_with(Mutation::None);
    for (score, logit) in trace.scores.iter().zip(LOGITS) {
        assert!(
            (score - sigmoid(logit)).abs() < 1e-6,
            "{score} vs sigmoid({logit}) = {}",
            sigmoid(logit)
        );
    }
}

#[test]
fn selection_scores_are_scores_plus_bias() {
    let trace = route_with(Mutation::None);
    for (e, ((&selection, &score), &bias)) in trace
        .selection_scores
        .iter()
        .zip(&trace.scores)
        .zip(&BIAS)
        .enumerate()
    {
        assert!((selection - (score + bias)).abs() < 1e-6, "expert {e}");
    }
}

/// The near-boundary flip: biased selection picks {0, 2}, NOT the
/// unbiased top-2 {0, 1} — expert 2's bias-corrected score (0.922) beats
/// expert 0's (0.881).
#[test]
fn selected_ids_are_chosen_by_the_biased_selection_scores() {
    let trace = route_with(Mutation::None);
    let mut ids = trace.selected_ids.clone();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 2], "{:?}", trace.selection_scores);
}

#[test]
fn gathered_weights_come_from_the_unbiased_scores() {
    let trace = route_with(Mutation::None);
    for (&id, &gathered) in trace.selected_ids.iter().zip(&trace.gathered_weights) {
        assert!(
            (gathered - trace.scores[id]).abs() < 1e-6,
            "expert {id}: gathered {gathered} vs unbiased score {}",
            trace.scores[id]
        );
        // The sharpest statement of the split: the gathered weight must
        // NOT equal the (biased) selection score used to choose it,
        // whenever that expert's bias is nonzero.
        if BIAS[id] != 0.0 {
            assert!(
                (gathered - trace.selection_scores[id]).abs() > 1e-3,
                "expert {id}: gathered weight equals the BIASED score — the split collapsed"
            );
        }
    }
}

#[test]
fn normalized_weights_sum_to_one() {
    let trace = route_with(Mutation::None);
    let sum: f32 = trace.normalized_weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    for (&norm, &gathered) in trace.normalized_weights.iter().zip(&trace.gathered_weights) {
        let expected = gathered / (trace.gathered_weights.iter().sum::<f32>() + 1e-20);
        assert!((norm - expected).abs() < 1e-6);
    }
}

#[test]
fn final_weights_are_normalized_weights_times_the_branch_scale() {
    let trace = route_with(Mutation::None);
    for (&w, &norm) in trace.weights.iter().zip(&trace.normalized_weights) {
        assert!((w as f64 - norm as f64 * BRANCH_SCALE).abs() < 1e-5);
    }
}

/// The load-bearing control: gathering from the BIASED array preserves
/// the selected ids exactly — selection is unaffected, because selection
/// already happened — but moves every weight, because expert 2's biased
/// score (0.922) is far from its real, unbiased one (0.622).
#[test]
fn gathering_biased_scores_preserves_ids_but_moves_every_weight() {
    let correct = route_with(Mutation::None);
    let wrong = route_with(Mutation::GatherBiasedWeights);

    let mut correct_ids = correct.selected_ids.clone();
    let mut wrong_ids = wrong.selected_ids.clone();
    correct_ids.sort_unstable();
    wrong_ids.sort_unstable();
    assert_eq!(
        correct_ids, wrong_ids,
        "selection must be unaffected by how weights are gathered"
    );

    for (&id, (&right, &bad)) in correct
        .selected_ids
        .iter()
        .zip(correct.weights.iter().zip(&wrong.weights))
    {
        assert!(
            (right - bad).abs() > 1e-3,
            "expert {id}: weight did not move ({right} vs {bad}) — the split is not load-bearing"
        );
    }
}

/// Omitting the correction bias changes WHICH experts win — the near-
/// boundary fixture exists precisely to make this observable: unbiased,
/// expert 1 beats expert 2; biased, the reverse.
#[test]
fn omitting_the_correction_bias_changes_the_selected_ids() {
    let biased = route_with(Mutation::None);
    let unbiased = route_with(Mutation::OmitBiasCorrection);

    let mut biased_ids = biased.selected_ids.clone();
    let mut unbiased_ids = unbiased.selected_ids.clone();
    biased_ids.sort_unstable();
    unbiased_ids.sort_unstable();

    assert_eq!(biased_ids, vec![0, 2]);
    assert_eq!(unbiased_ids, vec![0, 1], "{:?}", unbiased.selection_scores);
    assert_ne!(biased_ids, unbiased_ids);
}

#[test]
fn omitting_renormalization_leaves_weights_unnormalized() {
    let normal = route_with(Mutation::None);
    let unnormalized = route_with(Mutation::OmitRenormalization);

    // The gathered (pre-normalisation) weights are identical — only the
    // step downstream of them changed.
    assert_eq!(normal.gathered_weights, unnormalized.gathered_weights);
    assert_eq!(
        unnormalized.normalized_weights, unnormalized.gathered_weights,
        "omitting renormalisation must leave the gathered weights untouched"
    );
    assert_ne!(normal.weights, unnormalized.weights);
}

#[test]
fn omitting_the_branch_scale_leaves_the_normalized_weights_unscaled() {
    let normal = route_with(Mutation::None);
    let unscaled = route_with(Mutation::OmitBranchScale);

    assert_eq!(unscaled.weights, unscaled.normalized_weights);
    assert_ne!(normal.weights, unscaled.weights);
}

/// Softmax over every expert ranks and scales differently from sigmoid —
/// proves the function genuinely reads `moe_router_activation_func:
/// sigmoid`, not something that happens to look right because it is
/// monotonic in the same logits.
#[test]
fn softmax_instead_of_sigmoid_produces_different_scores() {
    let sigmoid_trace = route_with(Mutation::None);
    let softmax_trace = route_with(Mutation::UseSoftmaxInsteadOfSigmoid);

    assert_ne!(sigmoid_trace.scores, softmax_trace.scores);
    // Softmax scores still sum close to 1 over all experts; sigmoid scores
    // do not (each is independent) — a second, independent witness that
    // the two are genuinely different distributions, not a rescale of one.
    let softmax_sum: f32 = softmax_trace.scores.iter().sum();
    let sigmoid_sum: f32 = sigmoid_trace.scores.iter().sum();
    assert!((softmax_sum - 1.0).abs() < 1e-5, "{softmax_sum}");
    assert!((sigmoid_sum - 1.0).abs() > 0.1, "{sigmoid_sum}");
}

/// The reference guards renormalisation with `top_k > 1` — at `top_k ==
/// 1` the single selected weight passes straight through even when
/// `moe_renormalize` is declared true.
#[test]
fn top_k_one_skips_renormalization_even_when_declared() {
    let trace = route(
        &x(),
        &router_weight(),
        &BIAS,
        EXPERTS,
        1,
        true,
        BRANCH_SCALE,
        Mutation::None,
    );
    assert_eq!(trace.normalized_weights, trace.gathered_weights);
    // Still selects by the biased score: expert 2 (0.922) beats expert 0
    // (0.881) alone at top_k = 1.
    assert_eq!(trace.selected_ids, vec![2]);
}
