//! The sigmoid router, witnessed against literals computed by hand — not
//! against the other backend. The Kimi miniature declares the same rule
//! on both CPU backends, so their agreement proves nothing about the rule;
//! these numbers do.
//!
//! Four experts, top-2, logits `[2, 1, 0, -1]`:
//!   sigmoid   = [0.880797, 0.731059, 0.5, 0.268941]
//!   bias      = [-1.5, -0.5, 0.0, 1.4]
//!   corrected = [-0.619203, 0.231059, 0.5, 1.668941]  → selects 3 then 2
//!   raw weights of the selected = [0.268941, 0.5], sum 0.768941
//!   renormalised × branch scale 2 = [0.699511, 1.300489]
//! Without the bias the same rule would have selected experts 0 and 1.

use super::super::backend::{ExpertSlices, RoutedFfnCall};
use super::super::production::select_experts;
use super::super::reference::select_experts_reference;
use larql_models::config::{ExpertRoutingPolicy, GateUpLayout, MoeRouterKind};
use larql_models::{Activation, ExpertGatePolicy};

const LOGITS: [f32; 4] = [2.0, 1.0, 0.0, -1.0];
const BIAS: [f32; 4] = [-1.5, -0.5, 0.0, 1.4];
const TOP_K: usize = 2;
/// Hand-computed to six places; the backends must land within one unit
/// of the last place given.
const TOLERANCE: f32 = 1e-6;

/// A call whose router IS the logits: hidden 1 and `x = [1]`, so
/// `router · x` reproduces the logits in every backend.
fn call<'a>(
    x: &'a [f32],
    router: &'a [f32],
    bias: Option<&'a [f32]>,
    kind: MoeRouterKind,
    policy: ExpertRoutingPolicy,
    branch_scale: f32,
) -> RoutedFfnCall<'a> {
    RoutedFfnCall {
        x,
        hidden: 1,
        intermediate: 1,
        experts: LOGITS.len(),
        top_k: TOP_K,
        router_kind: kind,
        routing_policy: policy,
        branch_scale,
        activation: Activation::Silu,
        gate_policy: ExpertGatePolicy::Gated,
        router,
        router_bias: bias,
        weights: ExpertSlices::Fused {
            gate_up: &[],
            down: &[],
            layout: GateUpLayout::Interleaved,
        },
        gate_up_bias: None,
        down_bias: None,
        router_input: None,
        router_scale: None,
        router_per_expert_scale: None,
        router_norm_eps: None,
    }
}

/// What a router returns: the selected experts, each with its weight.
type Selection = Vec<(usize, f32)>;

fn both(call: &RoutedFfnCall<'_>) -> (Selection, Selection) {
    let mut logits = LOGITS.to_vec();
    let production = select_experts(call, &mut logits).unwrap();
    let reference = select_experts_reference(call).unwrap();
    (production, reference)
}

fn assert_weights(got: &[(usize, f32)], want: &[(usize, f32)], arm: &str) {
    assert_eq!(got.len(), want.len(), "{arm}: {got:?}");
    for ((e, w), (want_e, want_w)) in got.iter().zip(want) {
        assert_eq!(e, want_e, "{arm}: selected {got:?}, wanted {want:?}");
        assert!(
            (w - want_w).abs() <= TOLERANCE,
            "{arm}: expert {e} weighed {w}, wanted {want_w}"
        );
    }
}

#[test]
fn the_correction_bias_moves_the_selection_and_never_the_weights() {
    let x = [1.0f32];
    let c = call(
        &x,
        &LOGITS,
        Some(&BIAS),
        MoeRouterKind::Sigmoid,
        ExpertRoutingPolicy::NormalisedOverSelected,
        2.0,
    );
    let want = [(3, 0.699511), (2, 1.300489)];
    let (production, reference) = both(&c);
    assert_weights(&production, &want, "production");
    assert_weights(&reference, &want, "reference");
}

#[test]
fn without_the_bias_the_top_scores_are_selected_and_renormalised() {
    let x = [1.0f32];
    let c = call(
        &x,
        &LOGITS,
        None,
        MoeRouterKind::Sigmoid,
        ExpertRoutingPolicy::NormalisedOverSelected,
        1.0,
    );
    // 0.880797 + 0.731059 = 1.611856
    let want = [(0, 0.546449), (1, 0.453551)];
    let (production, reference) = both(&c);
    assert_weights(&production, &want, "production");
    assert_weights(&reference, &want, "reference");
}

#[test]
fn an_unrenormalised_sigmoid_router_keeps_the_raw_scores() {
    let x = [1.0f32];
    let c = call(
        &x,
        &LOGITS,
        None,
        MoeRouterKind::Sigmoid,
        ExpertRoutingPolicy::SoftmaxThenSelect,
        1.0,
    );
    let want = [(0, 0.880797), (1, 0.731059)];
    let (production, reference) = both(&c);
    assert_weights(&production, &want, "production");
    assert_weights(&reference, &want, "reference");
}

/// The softmax rule is untouched by the sigmoid arm, and the branch
/// scale reaches it too: `exp(2)/(exp(2)+exp(1)) × 3`.
#[test]
fn a_softmax_router_keeps_its_rule_under_the_branch_scale() {
    let x = [1.0f32];
    let c = call(
        &x,
        &LOGITS,
        None,
        MoeRouterKind::TopKSoftmax,
        ExpertRoutingPolicy::NormalisedOverSelected,
        3.0,
    );
    let want = [(0, 2.193176), (1, 0.806824)];
    let (production, reference) = both(&c);
    assert_weights(&production, &want, "production");
    assert_weights(&reference, &want, "reference");
}

/// A top-k of one has nothing to renormalise against: the raw score, scaled.
#[test]
fn a_single_selected_expert_keeps_its_raw_score() {
    let x = [1.0f32];
    let mut c = call(
        &x,
        &LOGITS,
        Some(&BIAS),
        MoeRouterKind::Sigmoid,
        ExpertRoutingPolicy::NormalisedOverSelected,
        2.446,
    );
    c.top_k = 1;
    // sigmoid(-1) × 2.446 = 0.26894142 × 2.446
    let want = [(3, 0.657831)];
    let (production, reference) = both(&c);
    assert_weights(&production, &want, "production");
    assert_weights(&reference, &want, "reference");
}
