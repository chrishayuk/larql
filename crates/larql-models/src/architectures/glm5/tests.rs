//! Tests for [`super::GlmMoeDsaArch`]: `from_config`/`family`, tensor-key
//! formatting against confirmed real HF weight names, MLA/MoE/DSA config
//! passthrough, and the `detect_from_json` round-trip.

use crate::config::{ExpertFormat, ModelArchitecture};

/// The real published `config.json` shape for `zai-org/GLM-5.2` (fetched
/// from the HF hub 2026-08-14) — not a scaled-down stand-in.
fn real_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 192,
        "kv_lora_rank": 512,
        "q_lora_rank": 2048,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "first_k_dense_replace": 3,
        "index_topk": 2048,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "vocab_size": 154880,
        "max_position_embeddings": 1048576,
        "norm_topk_prob": true,
        "tie_word_embeddings": false,
        "rms_norm_eps": 1e-5,
    })
}

fn arch() -> Box<dyn ModelArchitecture> {
    crate::detect_from_json(&real_config())
}

// ── from_config / family / detection round-trip ────────────────────────────

#[test]
fn detects_family_and_config_dimensions() {
    let a = arch();
    assert_eq!(a.family(), "glm_moe_dsa");
    assert_eq!(a.config().hidden_size, 6144);
    assert_eq!(a.config().num_layers, 78);
    assert_eq!(a.config().vocab_size, Some(154_880));
    assert_eq!(a.config().max_position_embeddings, Some(1_048_576));
}

#[test]
fn detection_round_trip_selects_glm_moe_dsa_not_generic() {
    // Minimal config — must still route to GlmMoeDsaArch, not fall back to
    // GenericArch, purely off `model_type == "glm_moe_dsa"`.
    let a = crate::detect_from_json(&serde_json::json!({
        "model_type": "glm_moe_dsa",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
    }));
    assert_eq!(a.family(), "glm_moe_dsa");
    assert_ne!(a.family(), "generic");
}

// ── Tensor keys: embed / final norm / prefix stripping ──────────────────────

#[test]
fn embed_and_final_norm_keys_use_the_confirmed_hf_convention() {
    let a = arch();
    // Confirmed real weight-map keys: `model.embed_tokens.weight` and
    // (by convention alongside it) `model.norm.weight`, after the base
    // trait's default `model.` prefix strip.
    assert_eq!(a.embed_key(), "embed_tokens.weight");
    assert_eq!(a.final_norm_key(), "norm.weight");
    assert!(a.key_prefixes_to_strip().contains(&"model."));
    assert!(a.has_lm_head());
}

#[test]
fn dense_ffn_keys_match_layers_before_first_k_dense_replace() {
    let a = arch();
    // Confirmed real keys for layer 0 (dense, pre-MoE — layers 0-2 per
    // `first_k_dense_replace: 3`): `model.layers.0.mlp.{gate,up,down}_proj.weight`.
    assert_eq!(a.ffn_gate_key(0), "layers.0.mlp.gate_proj.weight");
    assert_eq!(a.ffn_up_key(0), "layers.0.mlp.up_proj.weight");
    assert_eq!(a.ffn_down_key(0), "layers.0.mlp.down_proj.weight");
}

// ── MLA ───────────────────────────────────────────────────────────────────

#[test]
fn mla_keys_match_confirmed_real_weight_names() {
    let a = arch();
    assert_eq!(
        a.mla_kv_a_key(0),
        Some("layers.0.self_attn.kv_a_proj_with_mqa.weight".to_string())
    );
    assert_eq!(
        a.mla_kv_b_key(0),
        Some("layers.0.self_attn.kv_b_proj.weight".to_string())
    );
    assert_eq!(
        a.mla_q_a_key(0),
        Some("layers.0.self_attn.q_a_proj.weight".to_string())
    );
    assert_eq!(
        a.mla_q_b_key(7),
        Some("layers.7.self_attn.q_b_proj.weight".to_string())
    );
    assert_eq!(a.attn_o_key(7), "layers.7.self_attn.o_proj.weight");
}

#[test]
fn mla_field_passthrough_from_real_config() {
    let a = arch();
    assert!(a.uses_mla());
    assert_eq!(a.kv_lora_rank(), 512);
    assert_eq!(a.q_lora_rank(), 2048);
    assert_eq!(a.mla_qk_nope_head_dim(), Some(192));
    assert_eq!(a.mla_qk_rope_head_dim(), Some(64));
    assert_eq!(a.mla_v_head_dim(), Some(256));
}

// ── MoE ───────────────────────────────────────────────────────────────────

#[test]
fn moe_field_passthrough_from_real_config() {
    let a = arch();
    assert!(a.is_moe());
    assert_eq!(a.num_experts(), 256);
    assert_eq!(a.num_experts_per_token(), 8);
    assert_eq!(a.num_shared_experts(), 1);
    assert_eq!(a.expert_format(), ExpertFormat::PerExpert);
}

#[test]
fn moe_keys_match_confirmed_real_weight_names() {
    let a = arch();
    // Layer 10 is a MoE layer (>= first_k_dense_replace == 3), matching the
    // representative MoE-block layer confirmed in the real weight map.
    assert_eq!(
        a.moe_router_key(10),
        Some("layers.10.mlp.gate.weight".to_string())
    );
    assert_eq!(
        a.moe_router_bias_key(10),
        Some("layers.10.mlp.gate.e_score_correction_bias".to_string())
    );
    assert_eq!(
        a.expert_ffn_gate_key(10, 3),
        Some("layers.10.mlp.experts.3.gate_proj.weight".to_string())
    );
    assert_eq!(
        a.expert_ffn_up_key(10, 3),
        Some("layers.10.mlp.experts.3.up_proj.weight".to_string())
    );
    assert_eq!(
        a.expert_ffn_down_key(10, 3),
        Some("layers.10.mlp.experts.3.down_proj.weight".to_string())
    );
    assert_eq!(
        a.shared_expert_gate_key(10),
        Some("layers.10.mlp.shared_experts.gate_proj.weight".to_string())
    );
    assert_eq!(
        a.shared_expert_up_key(10),
        Some("layers.10.mlp.shared_experts.up_proj.weight".to_string())
    );
    assert_eq!(
        a.shared_expert_down_key(10),
        Some("layers.10.mlp.shared_experts.down_proj.weight".to_string())
    );
}

#[test]
fn distinct_experts_get_distinct_keys() {
    let a = arch();
    assert_ne!(a.expert_ffn_gate_key(10, 0), a.expert_ffn_gate_key(10, 1));
}

// ── DSA sparse-attention indexer (config/key facts only) ────────────────────

#[test]
fn dsa_indexer_field_passthrough_from_real_config() {
    let a = arch();
    assert_eq!(a.dsa_index_topk(), Some(2048));
    assert_eq!(a.dsa_index_n_heads(), Some(32));
    assert_eq!(a.dsa_index_head_dim(), Some(128));
}

#[test]
fn dsa_indexer_keys_match_confirmed_real_weight_names() {
    let a = arch();
    assert_eq!(
        a.dsa_indexer_wq_b_key(0),
        Some("layers.0.self_attn.indexer.wq_b.weight".to_string())
    );
    assert_eq!(
        a.dsa_indexer_wk_key(0),
        Some("layers.0.self_attn.indexer.wk.weight".to_string())
    );
    assert_eq!(
        a.dsa_indexer_k_norm_key(0),
        Some("layers.0.self_attn.indexer.k_norm.weight".to_string())
    );
    assert_eq!(
        a.dsa_indexer_weights_proj_key(0),
        Some("layers.0.self_attn.indexer.weights_proj.weight".to_string())
    );
}

#[test]
fn dsa_numeric_fields_absent_when_config_omits_them() {
    // A partial config must not invent numeric indexer facts it wasn't
    // given `index_topk` / `index_n_heads` / `index_head_dim` have no
    // architecture-class default — absence must stay absence.
    let a = crate::detect_from_json(&serde_json::json!({
        "model_type": "glm_moe_dsa",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
    }));
    assert_eq!(a.dsa_index_topk(), None);
    assert_eq!(a.dsa_index_n_heads(), None);
    assert_eq!(a.dsa_index_head_dim(), None);
}

#[test]
fn dsa_indexer_keys_are_a_structural_fact_independent_of_numeric_config() {
    // The indexer tensor keys are a fact about the `glm_moe_dsa` family
    // (every layer carries an indexer block), not conditioned on whether
    // `index_topk` etc. happen to be present — same posture as
    // `mla_kv_a_key` unconditionally naming a tensor once `uses_mla()`
    // no longer gates it. Every DSA checkpoint has these keys regardless.
    let a = crate::detect_from_json(&serde_json::json!({
        "model_type": "glm_moe_dsa",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
    }));
    assert_eq!(
        a.dsa_indexer_wq_b_key(0),
        Some("layers.0.self_attn.indexer.wq_b.weight".to_string())
    );
    assert_eq!(
        a.dsa_indexer_wk_key(0),
        Some("layers.0.self_attn.indexer.wk.weight".to_string())
    );
    assert_eq!(
        a.dsa_indexer_k_norm_key(0),
        Some("layers.0.self_attn.indexer.k_norm.weight".to_string())
    );
    assert_eq!(
        a.dsa_indexer_weights_proj_key(0),
        Some("layers.0.self_attn.indexer.weights_proj.weight".to_string())
    );
}

// ── Defaults when MLA / MoE fields are missing ───────────────────────────────

#[test]
fn defaults_fire_when_moe_and_mla_fields_are_missing() {
    let a = crate::detect_from_json(&serde_json::json!({
        "model_type": "glm_moe_dsa",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
    }));
    assert_eq!(a.family(), "glm_moe_dsa");

    // No expert count -> is_moe() is false; the size defaults still answer.
    assert!(!a.is_moe());
    assert_eq!(a.num_experts(), 256);
    assert_eq!(a.num_experts_per_token(), 8);
    assert_eq!(a.num_shared_experts(), 1);

    // No kv_lora_rank/q_lora_rank -> uses_mla() is false; defaults still answer.
    assert!(!a.uses_mla());
    assert_eq!(a.kv_lora_rank(), 512);
    assert_eq!(a.q_lora_rank(), 2048);
    assert_eq!(a.mla_qk_nope_head_dim(), None);
    assert_eq!(a.mla_qk_rope_head_dim(), None);
    assert_eq!(a.mla_v_head_dim(), None);
}
