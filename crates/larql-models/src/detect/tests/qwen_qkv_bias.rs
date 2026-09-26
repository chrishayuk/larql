//! Qwen2's attention bias shape: Q/K/V biased, output unbiased.
//!
//! `Qwen2Attention` and `Qwen2MoeAttention` build `q_proj`/`k_proj`/
//! `v_proj` with `bias=True` and `o_proj` with `bias=False`. Checkpoints
//! written by transformers 5 declare it as `qkv_bias`; earlier ones say
//! nothing and the family default answers. Qwen3 dropped the biases for
//! QK-norm, so it gets no default, and a declared `attention_bias` (the
//! all-four form) suppresses it rather than being overridden.

use crate::detect::detect_from_json;

fn config(model_type: &str, extra: serde_json::Value) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": model_type,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    });
    for (key, value) in extra.as_object().unwrap() {
        config[key] = value.clone();
    }
    config
}

#[test]
fn a_declared_qkv_bias_is_read() {
    for value in [true, false] {
        let arch = detect_from_json(&config(
            "qwen2_moe",
            serde_json::json!({ "qkv_bias": value }),
        ));
        assert_eq!(arch.qkv_bias(), Some(value));
        assert_eq!(arch.config().qkv_bias, Some(value));
    }
}

#[test]
fn a_silent_qwen2_config_takes_the_family_default() {
    for model_type in ["qwen2", "qwen2_moe"] {
        let arch = detect_from_json(&config(model_type, serde_json::json!({})));
        assert_eq!(arch.qkv_bias(), Some(true), "{model_type}");
        assert_eq!(
            arch.config().qkv_bias,
            None,
            "the default is not a declaration"
        );
    }
}

#[test]
fn qwen3_and_a_declared_attention_bias_take_no_default() {
    let qwen3 = detect_from_json(&config("qwen3", serde_json::json!({})));
    assert_eq!(qwen3.qkv_bias(), None);
    let declared = detect_from_json(&config(
        "qwen2",
        serde_json::json!({ "attention_bias": false }),
    ));
    assert_eq!(declared.qkv_bias(), None);
    assert_eq!(declared.attention_bias(), Some(false));
}

#[test]
fn a_non_qwen_family_reads_the_declaration_verbatim() {
    let llama = detect_from_json(&config("llama", serde_json::json!({ "qkv_bias": true })));
    assert_eq!(llama.qkv_bias(), Some(true));
    assert_eq!(
        detect_from_json(&config("llama", serde_json::json!({}))).qkv_bias(),
        None
    );
}
