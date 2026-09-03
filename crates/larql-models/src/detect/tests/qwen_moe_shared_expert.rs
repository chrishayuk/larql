//! Qwen's MoE facts: a gated shared expert sized by its own key, and a
//! stacked expert bank on the 3.5 generation.
//!
//! Three families sit under one `QwenArch` and disagree about all of it.
//! `qwen3_moe` runs no shared branch; `qwen2_moe` and `qwen3_5_moe` each
//! run exactly one, sized by `shared_expert_intermediate_size` and scaled
//! by `sigmoid(shared_expert_gate(x))` before the sum
//! (`Qwen2MoeSparseMoeBlock.forward`, `Qwen3_5MoeSparseMoeBlock.forward`).
//! `qwen3_5_moe` alone stores its experts stacked.

use crate::config::{
    ExpertFormat, GateActivation, GateCombine, GateUpLayout, SharedExpertGateSource,
};
use crate::detect::detect_from_json;

/// Qwen1.5-MoE-A2.7B's own numbers. The declared shared width and the
/// DeepSeek/Kimi derivation (`moe_intermediate_size * shared experts`)
/// differ FOURFOLD here, which is the whole point of the fixture: on
/// Qwen3.5-35B they coincide at 512 and would pin nothing.
const QWEN15_MOE_INTERMEDIATE: u64 = 1408;
const QWEN15_SHARED_INTERMEDIATE: u64 = 5632;

fn qwen2_moe_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen2_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "intermediate_size": 5632,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_experts": 60,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": QWEN15_MOE_INTERMEDIATE,
        "shared_expert_intermediate_size": QWEN15_SHARED_INTERMEDIATE,
    })
}

#[test]
fn the_shared_branch_is_sized_by_the_declared_key_not_by_the_routed_width() {
    let arch = detect_from_json(&qwen2_moe_config());
    assert_eq!(arch.num_shared_experts(), 1, "one branch, from the block");
    assert_eq!(
        arch.shared_expert_intermediate_size(),
        Some(QWEN15_SHARED_INTERMEDIATE as usize)
    );
    // The derivation this replaced, stated so the test fails if someone
    // reinstates it rather than merely renaming it.
    assert_ne!(
        arch.shared_expert_intermediate_size(),
        Some(arch.moe_intermediate_size() * arch.num_shared_experts())
    );
}

#[test]
fn the_shared_branch_carries_its_sigmoid_output_gate_and_operand() {
    let arch = detect_from_json(&qwen2_moe_config());
    let gate = arch
        .shared_expert_branch_gate()
        .expect("Qwen MoE gates its shared branch");
    assert_eq!(gate.source, SharedExpertGateSource::HiddenStateToScalar);
    assert_eq!(gate.activation, GateActivation::Sigmoid);
    assert_eq!(gate.combine, GateCombine::ElementwiseMultiply);
    // One name apart in the checkpoint, and never the same operand: the
    // branch gate is `[1, hidden]`, the SwiGLU gate is `[5632, hidden]`.
    assert_eq!(
        arch.shared_expert_branch_gate_key(0).as_deref(),
        Some("layers.0.mlp.shared_expert_gate.weight")
    );
    assert_eq!(
        arch.shared_expert_gate_key(0).as_deref(),
        Some("layers.0.mlp.shared_expert.gate_proj.weight")
    );
    assert_eq!(
        arch.shared_expert_up_key(0).as_deref(),
        Some("layers.0.mlp.shared_expert.up_proj.weight")
    );
    assert_eq!(
        arch.shared_expert_down_key(0).as_deref(),
        Some("layers.0.mlp.shared_expert.down_proj.weight")
    );
}

/// The falsifier: a Qwen MoE that declares no shared width runs no shared
/// branch, and must not acquire one from the family name.
#[test]
fn a_qwen_moe_without_the_declared_width_runs_no_shared_branch() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
    }));
    assert_eq!(arch.num_shared_experts(), 0);
    assert_eq!(arch.shared_expert_intermediate_size(), None);
    assert!(arch.shared_expert_branch_gate().is_none());
    assert!(arch.shared_expert_branch_gate_key(0).is_none());
    // And it keeps the per-expert bank its checkpoints ship.
    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert);
    assert!(arch.gate_up_layout().is_none());
}

/// `qwen3_5_moe` stacks its experts; its siblings do not. The nested
/// `_text` spelling is the one every released checkpoint declares.
#[test]
fn only_the_qwen35_moe_generation_stacks_its_expert_bank() {
    let stacked = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
    }));
    assert_eq!(stacked.expert_format(), ExpertFormat::PackedBF16);
    assert_eq!(
        stacked.gate_up_layout(),
        Some(GateUpLayout::ContiguousHalves)
    );
    // The dense generation shares the `qwen3_5` prefix and must NOT be
    // swept in by it.
    let dense = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_5_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "intermediate_size": 17408,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
    }));
    assert_eq!(dense.expert_format(), ExpertFormat::PerExpert);
    assert!(dense.gate_up_layout().is_none());
}

/// Nemotron-H's spelling reaches the same field.
#[test]
fn the_prefixed_spelling_of_the_shared_width_is_the_same_fact() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "nemotron_h",
        "hidden_size": 4480,
        "num_hidden_layers": 62,
        "intermediate_size": 1856,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "n_routed_experts": 128,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 1856,
        "n_shared_experts": 1,
        "moe_shared_expert_intermediate_size": 3712,
    }));
    assert_eq!(arch.num_shared_experts(), 1);
    assert_eq!(arch.shared_expert_intermediate_size(), Some(3712));
}
