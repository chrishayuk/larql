//! Operand closure for Multi-Latent Attention (P3d-f): Kimi Linear's
//! full-attention layers — `q_proj`, the shared compressed KV projection,
//! its RMSNorm, the KV decompression, and `o_proj` — carved to typed
//! roles and bound with zero defects, at the CHECKPOINT's own asymmetric
//! geometry (q_head_dim ≠ v_head_dim).
//!
//! No MoE this file: `num_experts` is left undeclared, so the FFN is
//! plain dense — this rung is MLA operand closure, and pulling MoE
//! machinery into the fixture would test a different rung's mechanism
//! under this one's name. See `kimi_moe_closure.rs` for that one.

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::graph::OperandRole;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{
    plan_component_ops, ClosureDefect, LayerAttention, OpPlanOutcome,
};
use crate::format::vindex3::plan::tests_support::custom_artifact;

const HIDDEN: usize = 32;
const NUM_HEADS: usize = 4;
const KV_LORA_RANK: usize = 16;
const QK_NOPE_HEAD_DIM: usize = 8;
const QK_ROPE_HEAD_DIM: usize = 4;
const V_HEAD_DIM: usize = 8;
const INTER: usize = 64;
const LAYERS: usize = 2;
const VOCAB: usize = 64;

fn q_head_dim() -> usize {
    QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
}

fn kimi_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_HEADS,
        "head_dim": 999, // deliberately NOT MLA's real width — proves
                         // `expected_shape` reads `mla`, never this field,
                         // for an MLA layer's Q/K/V/O contracts.
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "mla_use_nope": true,
        // Every layer full-attention (MLA) — no KDA in this fixture, no
        // `num_experts` declared, so the FFN stays dense.
        "linear_attn_config": {
            "kda_layers": [],
            "full_attn_layers": (1..=LAYERS).collect::<Vec<_>>()
        },
    })
}

fn kimi_layer_tensors(layer: usize) -> Vec<(String, Vec<usize>)> {
    let prefix = format!("model.layers.{layer}.");
    vec![
        (
            format!("{prefix}self_attn.q_proj.weight"),
            vec![NUM_HEADS * q_head_dim(), HIDDEN],
        ),
        (
            format!("{prefix}self_attn.kv_a_proj_with_mqa.weight"),
            vec![KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN],
        ),
        (
            format!("{prefix}self_attn.kv_a_layernorm.weight"),
            vec![KV_LORA_RANK],
        ),
        (
            format!("{prefix}self_attn.kv_b_proj.weight"),
            vec![NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK],
        ),
        (
            format!("{prefix}self_attn.o_proj.weight"),
            vec![HIDDEN, NUM_HEADS * V_HEAD_DIM],
        ),
        (format!("{prefix}input_layernorm.weight"), vec![HIDDEN]),
        (
            format!("{prefix}post_attention_layernorm.weight"),
            vec![HIDDEN],
        ),
        (format!("{prefix}mlp.gate_proj.weight"), vec![INTER, HIDDEN]),
        (format!("{prefix}mlp.up_proj.weight"), vec![INTER, HIDDEN]),
        (format!("{prefix}mlp.down_proj.weight"), vec![HIDDEN, INTER]),
    ]
}

fn kimi_tensors() -> Vec<(String, Vec<usize>)> {
    let mut tensors = vec![
        ("model.embed_tokens.weight".to_string(), vec![VOCAB, HIDDEN]),
        ("model.norm.weight".to_string(), vec![HIDDEN]),
        ("lm_head.weight".to_string(), vec![VOCAB, HIDDEN]),
    ];
    for layer in 0..LAYERS {
        tensors.extend(kimi_layer_tensors(layer));
    }
    tensors
}

fn plan_variant(mutate: impl FnOnce(&mut Vec<(String, Vec<usize>)>)) -> OpPlanOutcome {
    let dir = tempfile::tempdir().unwrap();
    let mut tensors = kimi_tensors();
    mutate(&mut tensors);
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(dir.path(), &kimi_config(), &borrowed);
    let named = vec![("kimi-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), false).unwrap();
    plan_component_ops(&inspection, out.path(), "target").unwrap()
}

/// The whole point of this rung: every MLA layer's five operands — at
/// the checkpoint's own asymmetric geometry, not the config's unrelated
/// `head_dim` — close with zero defects.
#[test]
fn an_mla_kimi_shaped_estate_closes() {
    let outcome = plan_variant(|_| {});
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    for layer in &plan.layers {
        let LayerAttention::Mla(op) = &layer.attention else {
            panic!("layer {}: planned non-MLA", layer.layer);
        };
        assert_eq!(op.num_heads, NUM_HEADS);
        assert_eq!(op.kv_lora_rank, KV_LORA_RANK);
        assert_eq!(op.qk_nope_head_dim, QK_NOPE_HEAD_DIM);
        assert_eq!(op.qk_rope_head_dim, QK_ROPE_HEAD_DIM);
        assert_eq!(op.v_head_dim, V_HEAD_DIM);
        assert_eq!(op.q_head_dim(), QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM);
        assert_eq!(op.compressed_kv_width(), KV_LORA_RANK + QK_ROPE_HEAD_DIM);
        for operand in [
            &op.q_proj,
            &op.kv_a_proj,
            &op.kv_b_proj,
            &op.kv_a_norm,
            &op.out_proj,
        ] {
            assert_eq!(operand.object, "target.decoder_stack");
        }
    }
}

/// The two geometry mismatches the real container showed BEFORE this
/// rung: `q_proj`/`o_proj` checked against the config's plain `head_dim`
/// (999, deliberately wrong in this fixture) instead of MLA's own
/// asymmetric widths. Both must be GONE once the surface carries `mla`.
#[test]
fn q_proj_and_o_proj_are_checked_against_mla_geometry_not_the_config_head_dim() {
    let outcome = plan_variant(|_| {});
    assert!(
        !outcome
            .defects
            .iter()
            .any(|d| matches!(d, ClosureDefect::GeometryMismatch { tensor, .. } if tensor.contains("q_proj") || tensor.contains("o_proj"))),
        "{:?}",
        outcome.defects
    );
}

/// A missing `kv_a_layernorm` — the operand easiest to overlook, since it
/// has no softmax analogue at all — is named precisely, not absorbed
/// into a generic "unclassified operand" report.
#[test]
fn a_missing_kv_a_norm_is_named_precisely() {
    let outcome = plan_variant(|tensors| {
        tensors.retain(|(name, _)| name != "model.layers.0.self_attn.kv_a_layernorm.weight");
    });
    assert!(
        outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::MissingOperand {
                layer: 0,
                role: OperandRole::MlaKvANorm
            }
        )),
        "{:?}",
        outcome.defects
    );
}

/// An MLA layer needs no `self_attn.k_proj`/`v_proj` — demanding them
/// would report two missing operands per layer for tensors the
/// checkpoint never ships (K/V arrive only through the compressed path).
/// A stray `k_proj` tensor is a defect, not a silently-accepted extra.
#[test]
fn no_plain_k_or_v_projection_is_required_and_a_stray_one_is_a_defect() {
    let outcome = plan_variant(|tensors| {
        tensors.push((
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            vec![NUM_HEADS * QK_NOPE_HEAD_DIM, HIDDEN],
        ));
    });
    assert!(
        !outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::MissingOperand {
                role: OperandRole::AttnK,
                ..
            }
        )),
        "{:?}",
        outcome.defects
    );
    assert!(
        outcome.defects.iter().any(|d| matches!(
            d,
            ClosureDefect::OperandImpliesAbsentOp { tensor, .. } if tensor.contains("k_proj")
        )),
        "{:?}",
        outcome.defects
    );
}
