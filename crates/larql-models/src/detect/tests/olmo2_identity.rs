//! OLMo-2 / OLMo-3 resolve to their own identity, and that identity
//! declares only what the references establish.
//!
//! The point of these is not that `olmo2` is "supported". It is that
//! three execution-sensitive facts are STATED rather than inherited, and
//! each of the three is silent when wrong — a Llama-shaped default
//! produces a running model with a different forward pass, which is the
//! failure `architecture_identity` exists as a gate to catch.

use crate::config::QkNormScope;
use crate::detect::detect_from_json;

/// The 0425-1B geometry, from its own `config.json`.
fn olmo2_config(model_type: &str) -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Olmo2ForCausalLM"],
        "model_type": model_type,
        "attention_bias": false,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-06,
        "rope_theta": 500000,
        "tie_word_embeddings": false,
        "vocab_size": 100352
    })
}

#[test]
fn olmo2_and_olmo3_resolve_to_their_own_identity_not_the_generic_fallback() {
    for model_type in ["olmo2", "olmo3"] {
        let arch = detect_from_json(&olmo2_config(model_type));
        assert_eq!(
            arch.family(),
            model_type,
            "the label must be the config's own, so a report says WHICH generation"
        );
    }
    // The falsifier: OLMo v1 and OLMoE are different architectures and
    // must not be swallowed by this entry.
    assert_eq!(detect_from_json(&olmo2_config("olmoe")).family(), "olmoe");
    assert_eq!(detect_from_json(&olmo2_config("olmo")).family(), "generic");
}

/// `Olmo2Attention` normalises the WHOLE projection before the head
/// reshape — the operator OLMoE already declares, and not Qwen3's
/// per-head norm over `head_dim` elements.
#[test]
fn the_qk_norm_is_over_the_whole_projection_like_olmoe_and_not_per_head() {
    let arch = detect_from_json(&olmo2_config("olmo2"));
    assert_eq!(arch.qk_norm_scope(), QkNormScope::FullProjection);
    assert_eq!(
        arch.attn_q_norm_key(0).as_deref(),
        Some("layers.0.self_attn.q_norm.weight")
    );
    assert_eq!(
        arch.attn_k_norm_key(0).as_deref(),
        Some("layers.0.self_attn.k_norm.weight")
    );
    // The contrast that gives the assertion meaning: a Qwen3 config
    // reaching the same accessor answers the other scope.
    let qwen = detect_from_json(&serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 2048, "num_hidden_layers": 16,
        "intermediate_size": 8192, "num_attention_heads": 16,
        "num_key_value_heads": 16, "vocab_size": 100352
    }));
    assert_eq!(qwen.qk_norm_scope(), QkNormScope::PerHead);
}

/// `Olmo2Config.rms_norm_eps` defaults to 1e-5. The 1B ships 1e-06
/// explicitly, so the DEFAULT is what a checkpoint omitting the field
/// would run at — and it is declared from the class, never observed
/// from a row that happens to state it.
#[test]
fn the_norm_epsilon_default_is_the_class_default_not_llamas() {
    let mut config = olmo2_config("olmo2");
    config.as_object_mut().unwrap().remove("rms_norm_eps");
    let arch = detect_from_json(&config);
    assert!(
        (arch.default_norm_eps() - 1e-5).abs() < 1e-12,
        "got {}",
        arch.default_norm_eps()
    );
    // And the declared value still wins where one is declared.
    let declared = detect_from_json(&olmo2_config("olmo2"));
    assert!(
        (declared.norm_eps() - 1e-6).abs() < 1e-12,
        "{}",
        declared.norm_eps()
    );
}

/// `Olmo2RMSNorm` initialises its weight to ONES and applies
/// `weight * x` — not the `(1 + weight)` convention Qwen3.5 stores.
#[test]
fn the_norm_weight_is_applied_directly_not_as_an_offset_from_one() {
    let arch = detect_from_json(&olmo2_config("olmo2"));
    assert_eq!(arch.norm_weight_offset(), 0.0);
    assert_eq!(arch.qk_norm_weight_offset(), 0.0);
}

/// EXAONE-4 shares OLMo-2's post-norm stack and differs in one operator.
/// Registered separately for exactly that reason, and the assertion pair
/// is what stops a later refactor collapsing them.
#[test]
fn exaone4_is_its_own_identity_with_per_head_qk_norm() {
    let exaone = detect_from_json(&serde_json::json!({
        "model_type": "exaone4",
        "hidden_size": 2048, "num_hidden_layers": 30,
        "intermediate_size": 4096, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 64,
        "rms_norm_eps": 1e-05, "vocab_size": 102400
    }));
    assert_eq!(exaone.family(), "exaone4");
    // `Exaone4RMSNorm(head_dim)` applied after the head reshape.
    assert_eq!(exaone.qk_norm_scope(), QkNormScope::PerHead);
    // The nested spelling resolves through the same prefix.
    let nested = detect_from_json(&serde_json::json!({
        "model_type": "exaone4_5_text",
        "hidden_size": 2048, "num_hidden_layers": 30,
        "intermediate_size": 4096, "num_attention_heads": 32,
        "num_key_value_heads": 8, "vocab_size": 102400
    }));
    assert_eq!(nested.family(), "exaone4_5_text");
    assert_eq!(nested.qk_norm_scope(), QkNormScope::PerHead);

    // The contrast that makes the entry worth having: the same
    // post-norm stack in OLMo-2 normalises the WHOLE projection.
    assert_eq!(
        detect_from_json(&olmo2_config("olmo2")).qk_norm_scope(),
        QkNormScope::FullProjection
    );
    // Both take the 1e-5 class default, and neither takes Llama's 1e-6.
    assert!((exaone.default_norm_eps() - 1e-5).abs() < 1e-12);
}

/// LFM2's dialect resolves to a real identity — never to `GenericArch`,
/// which is what the registry's own honoured-by-dispatch gate insists on:
/// a registered NAME that dispatches to the generic path would report a
/// resolved identity while being served generic defaults.
#[test]
fn lfm2_resolves_to_its_own_identity_with_its_own_tensor_spellings() {
    for model_type in ["lfm2", "lfm2_moe"] {
        let arch = detect_from_json(&serde_json::json!({
            "model_type": model_type,
            "hidden_size": 1024, "num_hidden_layers": 16,
            "intermediate_size": 4608, "num_attention_heads": 16,
            "num_key_value_heads": 8, "vocab_size": 65536,
            "norm_eps": 1e-05, "full_attn_idxs": [2, 5, 8, 10, 12, 14]
        }));
        assert_eq!(arch.family(), model_type);
        // Its epsilon is spelled `norm_eps`, and it is read.
        assert!(
            (arch.norm_eps() - 1e-5).abs() < 1e-12,
            "{}",
            arch.norm_eps()
        );
        // Per-head QK norm — `Lfm2RMSNorm(head_dim)` after the reshape.
        assert_eq!(arch.qk_norm_scope(), QkNormScope::PerHead);
        // The four spellings that differ from Llama's.
        assert_eq!(
            arch.attn_q_norm_key(0).as_deref(),
            Some("layers.0.self_attn.q_layernorm.weight")
        );
        assert_eq!(arch.attn_o_key(0), "layers.0.self_attn.out_proj.weight");
        assert_eq!(arch.ffn_gate_key(0), "layers.0.feed_forward.w1.weight");
        assert_eq!(arch.ffn_up_key(0), "layers.0.feed_forward.w3.weight");
        assert_eq!(arch.ffn_down_key(0), "layers.0.feed_forward.w2.weight");
        assert_eq!(arch.final_norm_key(), "model.embedding_norm.weight");
    }
}
