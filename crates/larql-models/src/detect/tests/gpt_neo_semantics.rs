//! GPT-Neo as the architecture resolves it: its own topology spellings,
//! the `attention_types` interleave and `window_size`, unscaled scores, an
//! output-only attention bias, and the GPT-2 primitives it shares.

use crate::config::{Activation, InputPositionEncoding, PositionPolicy};
use crate::detect::*;

fn neo(extra: serde_json::Value) -> Box<dyn crate::config::ModelArchitecture> {
    let mut config = serde_json::json!({
        "model_type": "gpt_neo",
        "hidden_size": 64,
        "num_layers": 4,
        "num_heads": 16,
        "intermediate_size": null,
        "max_position_embeddings": 2048,
        "attention_types": [[["global", "local"], 2]],
        "window_size": 256,
        "layer_norm_epsilon": 1e-5,
        "vocab_size": 100,
    });
    for (k, v) in extra.as_object().unwrap() {
        config[k] = v.clone();
    }
    detect_from_json(&config)
}

#[test]
fn gpt_neo_reads_its_own_topology_spellings() {
    let arch = neo(serde_json::json!({}));
    assert_eq!(arch.family(), "gpt_neo");
    assert_eq!(arch.config().num_layers, 4);
    assert_eq!(arch.config().num_q_heads, 16);
    assert_eq!(arch.config().head_dim, 4);
    // `intermediate_size: null` is `4 * hidden_size`.
    assert_eq!(arch.config().intermediate_size, 256);
}

/// `num_heads` is GPT-Neo's spelling only: on a family that does not
/// name it, it stays the Mamba2 mixer's word and is never read as
/// attention heads.
#[test]
fn num_heads_is_not_read_as_attention_heads_outside_gpt_neo() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_heads": 16,
        "intermediate_size": 128,
    }));
    assert_eq!(arch.config().num_q_heads, 0);
}

#[test]
fn attention_types_expand_into_global_and_local_layers() {
    let arch = neo(serde_json::json!({}));
    let sliding: Vec<bool> = (0..4).map(|l| arch.is_sliding_window_layer(l)).collect();
    assert_eq!(sliding, [false, true, false, true]);
    assert_eq!(arch.sliding_window_size(), Some(256));
    // `attention_layers` alone answers when the run-length form is absent.
    let mut config = serde_json::json!({
        "model_type": "gpt_neo", "hidden_size": 64, "num_layers": 2, "num_heads": 16,
        "attention_layers": ["local", "global"], "window_size": 8,
    });
    let arch = detect_from_json(&config);
    assert!(arch.is_sliding_window_layer(0) && !arch.is_sliding_window_layer(1));
    // An unrecognised word resolves to neither kind.
    config["attention_layers"] = serde_json::json!(["global", "dilated"]);
    let arch = detect_from_json(&config);
    assert_eq!(arch.config().layer_types.as_ref().unwrap()[1], "dilated");
}

#[test]
fn gpt_neo_does_not_scale_scores_and_biases_only_its_output_projection() {
    let arch = neo(serde_json::json!({}));
    assert_eq!(arch.attention_scale(), 1.0);
    assert_eq!(arch.attention_output_bias(), Some(true));
    assert_eq!(arch.attention_bias(), None);
    assert_eq!(arch.qkv_bias(), None);
    assert!(arch.source_layout().input_major.is_empty());
}

#[test]
fn gpt_neo_shares_the_gpt2_primitives() {
    let arch = neo(serde_json::json!({}));
    assert_eq!(arch.position_policy_for_layer(0), PositionPolicy::None);
    assert_eq!(
        arch.input_position_encoding(),
        Some(InputPositionEncoding::LearnedAbsolute {
            max_positions: 2048
        })
    );
    assert_eq!(arch.activation(), Activation::GeluTanh);
    assert_eq!(arch.norm_bias(), Some(true));
    assert_eq!(arch.ffn_bias(), Some(true));
}
