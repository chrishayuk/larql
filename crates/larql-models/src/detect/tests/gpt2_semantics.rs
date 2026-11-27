//! GPT-2's execution semantics as the architecture resolves them: learned
//! absolute positions and no rotation, the declared activation, the
//! score-scaling switch, and the storage layout the inventory normalises.

use crate::config::{Activation, InputPositionEncoding, PositionPolicy};
use crate::detect::*;

fn gpt2(extra: serde_json::Value) -> Box<dyn crate::config::ModelArchitecture> {
    let mut config = serde_json::json!({
        "model_type": "gpt2",
        "n_embd": 64,
        "n_layer": 2,
        "n_head": 4,
        "n_positions": 128,
        "vocab_size": 100,
        "layer_norm_epsilon": 1e-5,
    });
    for (k, v) in extra.as_object().unwrap() {
        config[k] = v.clone();
    }
    detect_from_json(&config)
}

/// GPT-2's config states no rope key; `rope_base` holds the parser's
/// DEFAULT. A family with a learned position table must not rotate off it.
#[test]
fn gpt2_rotates_nothing_and_adds_a_learned_table_at_the_input() {
    let arch = gpt2(serde_json::json!({}));
    assert!(!arch.config().declares_rotary);
    for layer in 0..2 {
        assert_eq!(arch.position_policy_for_layer(layer), PositionPolicy::None);
    }
    assert_eq!(
        arch.input_position_encoding(),
        Some(InputPositionEncoding::LearnedAbsolute { max_positions: 128 })
    );
    // A declared base is still read: the guard is the declaration.
    let declared = gpt2(serde_json::json!({ "rope_theta": 10000.0 }));
    assert!(declared.position_policy_for_layer(0).is_rotary());
}

#[test]
fn a_family_without_a_position_table_declares_no_input_encoding() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 128,
    }));
    assert_eq!(arch.input_position_encoding(), None);
    assert!(arch.position_policy_for_layer(0).is_rotary());
}

#[test]
fn gpt2_reads_activation_function_and_defaults_to_gelu_new_when_silent() {
    assert_eq!(
        gpt2(serde_json::json!({})).activation(),
        Activation::GeluTanh
    );
    assert_eq!(
        gpt2(serde_json::json!({ "activation_function": "gelu_new" })).activation(),
        Activation::GeluTanh
    );
    // The declaration is read, not overridden by the family.
    assert_eq!(
        gpt2(serde_json::json!({ "activation_function": "relu" })).activation(),
        Activation::Relu
    );
}

#[test]
fn scale_attn_weights_false_makes_the_score_multiplier_one() {
    let scaled = gpt2(serde_json::json!({ "scale_attn_weights": true }));
    assert_eq!(scaled.attention_scale(), 1.0 / 16f64.sqrt());
    let silent = gpt2(serde_json::json!({}));
    assert_eq!(silent.attention_scale(), scaled.attention_scale());
    let unscaled = gpt2(serde_json::json!({ "scale_attn_weights": false }));
    assert_eq!(unscaled.attention_scale(), 1.0);
    assert_eq!(unscaled.attention_scale_for_layer(1), 1.0);
}

#[test]
fn gpt2_declares_its_biases_and_its_conv1d_storage_layout() {
    let arch = gpt2(serde_json::json!({}));
    assert_eq!(arch.norm_bias(), Some(true));
    assert_eq!(arch.ffn_bias(), Some(true));
    assert_eq!(arch.attention_bias(), Some(true));
    let layout = arch.source_layout();
    assert!(layout.input_major.contains(&"attn.c_attn"));
    assert_eq!(layout.fused[0].parts.len(), 3);
    assert!(layout.buffers.iter().any(|b| b.suffix == "attn.bias"));
}
