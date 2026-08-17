//! THE 5b-1 gate: the emitted generic program means what the container
//! says — per-layer policy from the table, placement explicit, no layer
//! arithmetic anywhere.

use larql_models::config::{Activation, PositionPolicy};

use super::encoded_fixture;
use crate::format::vindex3::graph::policy::AttentionSpan;
use crate::format::vindex3::opplan::plan_component_ops;

/// Target: 8 four-norm layers, hybrid interleave. Layer 0 must read
/// sliding+RoPE and layer 3 full+NoPE **solely from the persisted
/// table** — the fixture's `% 4` pattern exists only in the config that
/// generated it, never in the planner.
#[test]
fn target_plan_reads_span_and_position_from_the_table() {
    let fixture = encoded_fixture();
    let outcome = plan_component_ops(&fixture.inspection, &fixture.root, "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    assert_eq!(plan.layers.len(), 8);

    let layer0 = &plan.layers[0];
    assert_eq!(layer0.attention.span, AttentionSpan::Sliding);
    assert_eq!(layer0.attention.window, Some(16));
    assert!(matches!(
        layer0.attention.position,
        PositionPolicy::Rope { .. }
    ));

    let layer3 = &plan.layers[3];
    assert_eq!(layer3.attention.span, AttentionSpan::Full);
    assert_eq!(layer3.attention.window, None);
    assert_eq!(layer3.attention.position, PositionPolicy::None);

    // Four-norm placement, explicit positions — not a count.
    assert!(layer0.post_attention_norm.is_some());
    assert!(layer0.post_ffn_norm.is_some());
    assert!(layer0
        .pre_ffn_norm
        .weight
        .tensor
        .contains("pre_feedforward_layernorm"));

    // Full accounting, every layer.
    for layer in &plan.layers {
        assert_eq!(layer.operands_accounted, 12, "layer {}", layer.layer);
        assert_eq!(layer.operands_present, 12);
    }

    // Ends of the program: embedding, final norm, head with the resolved
    // output multiplier.
    let embedding = plan.embedding.as_ref().unwrap();
    assert_eq!(embedding.vocab_size, 128);
    assert!(plan.final_norm.is_some());
    let output = plan.output.as_ref().unwrap();
    let multiplier = output.multiplier.expect("declared output_multiplier");
    assert!((multiplier - 0.196).abs() < 1e-12);

    // Both scales reach the kernel arguments, unfolded.
    let query_scale = layer0
        .attention
        .query_scale
        .expect("declared qk_scale_factor");
    assert!((query_scale - 3.87).abs() < 1e-12);
    assert!((layer0.attention.score_scale - (8f64).powf(-0.5)).abs() < 1e-12);
    // The judged gate binds its operand.
    let gate = layer0.attention.output_gate.as_ref().unwrap();
    assert!(gate.projection.tensor.contains("self_attn.gate_proj"));
    assert!(layer0.attention.parameter_free_qk_norm.q);
}

/// Drafter: two-norm placement — `post_attention_layernorm` *is* the
/// pre-FFN norm, the post positions are absent, and the QK-norm pair is
/// bound with per-head shapes.
#[test]
fn drafter_plan_uses_two_norm_placement_and_qk_norms() {
    let fixture = encoded_fixture();
    let outcome = plan_component_ops(&fixture.inspection, &fixture.root, "draft").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    assert_eq!(plan.layers.len(), 2);

    let layer0 = &plan.layers[0];
    assert!(layer0.post_attention_norm.is_none());
    assert!(layer0.post_ffn_norm.is_none());
    assert!(layer0
        .pre_ffn_norm
        .weight
        .tensor
        .contains("post_attention_layernorm"));

    let qk = layer0.attention.qk_norm.as_ref().unwrap();
    assert_eq!(qk.q.shape, vec![8]);
    assert_eq!(qk.k.shape, vec![8]);

    // No head objects → no embedding/output ops; final norm exists.
    assert!(plan.embedding.is_none());
    assert!(plan.output.is_none());
    assert!(plan.final_norm.is_some());
    assert_eq!(layer0.attention.num_kv_heads, 4);
    assert_eq!(layer0.ffn.dense().unwrap().activation, Activation::Silu);

    // 11 operands: 4 attn + 2 qk norms + 2 norms + 3 mlp.
    assert_eq!(layer0.operands_accounted, 11);
}

/// The plan carries no HF names: every operand reference is a logical
/// object id plus a segment-relative tensor.
#[test]
fn plan_operands_reference_logical_objects_only() {
    let fixture = encoded_fixture();
    let plan = plan_component_ops(&fixture.inspection, &fixture.root, "target")
        .unwrap()
        .plan
        .unwrap();
    let json = serde_json::to_string(&plan).unwrap();
    assert!(!json.contains("model.language_model"), "HF prefix leaked");
    assert!(!json.contains("lm_head"), "HF head name leaked");
    for layer in &plan.layers {
        assert_eq!(layer.attention.q.object, "target.decoder_stack");
    }
}
