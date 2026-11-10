//! Flat `gemma3_text` plan fixtures (Gemma 3 270M / 1B).
//!
//! The miniature mirrors the real flat configs: no `text_config` wrapper,
//! `Gemma3ForCausalLM`, `model.*` tensor spelling, and the RoPE pair in
//! its legacy flat spelling (`rope_theta` for the full layers,
//! `rope_local_base_freq` for the sliding ones). Two interleave dialects,
//! as shipped: the 270M declares an explicit `layer_types` array beside
//! the BC `_sliding_window_pattern` and `use_bidirectional_attention:
//! false`; the 1B declares only `sliding_window_pattern` and
//! `cache_implementation: "hybrid"`.

use larql_models::config::{
    is_full_at_period, LAYER_TYPE_FULL_ATTENTION, LAYER_TYPE_SLIDING_ATTENTION,
};
use larql_models::inventory::ArchitectureInventory;
use std::path::Path;

#[allow(unused_imports)]
use super::*;

/// Twelve layers at the class-default period: two full layers.
pub const FLAT_GEMMA3_LAYERS: usize = 12;
/// HF's `sliding_window_pattern` default, which both real checkpoints
/// declare.
pub const FLAT_GEMMA3_PERIOD: usize = 6;
pub const FLAT_GEMMA3_HIDDEN: usize = 32;
pub const FLAT_GEMMA3_INTER: usize = 64;
pub const FLAT_GEMMA3_Q_HEADS: usize = 4;
pub const FLAT_GEMMA3_KV_HEADS: usize = 1;
pub const FLAT_GEMMA3_HEAD_DIM: usize = 16;
pub const FLAT_GEMMA3_VOCAB: usize = 300;
pub const FLAT_GEMMA3_WINDOW: usize = 8;
pub const FLAT_GEMMA3_GLOBAL_THETA: f64 = 1_000_000.0;
pub const FLAT_GEMMA3_LOCAL_THETA: f64 = 10_000.0;

/// Which interleave dialect the fixture declares.
#[derive(Debug, Clone, Copy)]
pub enum FlatGemma3Dialect {
    /// 270M: `layer_types` + `_sliding_window_pattern` +
    /// `use_bidirectional_attention`.
    LayerTypes,
    /// 1B: `sliding_window_pattern` + `cache_implementation`.
    Period,
}

/// The `layer_types` array the stride `period` derives, as HF writes it.
pub fn flat_gemma3_layer_types(period: usize) -> Vec<&'static str> {
    (0..FLAT_GEMMA3_LAYERS)
        .map(|layer| {
            if is_full_at_period(layer, period) {
                LAYER_TYPE_FULL_ATTENTION
            } else {
                LAYER_TYPE_SLIDING_ATTENTION
            }
        })
        .collect()
}

/// The flat config of `dialect`, with `mutate` applied before writing.
pub fn flat_gemma3_target_with(
    dir: &Path,
    dialect: FlatGemma3Dialect,
    mutate: impl FnOnce(&mut serde_json::Value),
) -> ArchitectureInventory {
    let mut config = serde_json::json!({
        "architectures": ["Gemma3ForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_logit_softcapping": null,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "final_logit_softcapping": null,
        "head_dim": FLAT_GEMMA3_HEAD_DIM,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": FLAT_GEMMA3_HIDDEN,
        "initializer_range": 0.02,
        "intermediate_size": FLAT_GEMMA3_INTER,
        "max_position_embeddings": 32768,
        "model_type": "gemma3_text",
        "num_attention_heads": FLAT_GEMMA3_Q_HEADS,
        "num_hidden_layers": FLAT_GEMMA3_LAYERS,
        "num_key_value_heads": FLAT_GEMMA3_KV_HEADS,
        "pad_token_id": 0,
        "query_pre_attn_scalar": FLAT_GEMMA3_HEAD_DIM,
        "rms_norm_eps": 1e-06,
        "rope_local_base_freq": FLAT_GEMMA3_LOCAL_THETA,
        "rope_scaling": null,
        "rope_theta": FLAT_GEMMA3_GLOBAL_THETA,
        "sliding_window": FLAT_GEMMA3_WINDOW,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "vocab_size": FLAT_GEMMA3_VOCAB
    });
    match dialect {
        FlatGemma3Dialect::LayerTypes => {
            config["layer_types"] = serde_json::json!(flat_gemma3_layer_types(FLAT_GEMMA3_PERIOD));
            config["_sliding_window_pattern"] = serde_json::json!(FLAT_GEMMA3_PERIOD);
            config["use_bidirectional_attention"] = serde_json::json!(false);
        }
        FlatGemma3Dialect::Period => {
            config["sliding_window_pattern"] = serde_json::json!(FLAT_GEMMA3_PERIOD);
            config["cache_implementation"] = serde_json::json!("hybrid");
        }
    }
    mutate(&mut config);

    let h = FLAT_GEMMA3_HIDDEN;
    let q_rows = FLAT_GEMMA3_Q_HEADS * FLAT_GEMMA3_HEAD_DIM;
    let kv_rows = FLAT_GEMMA3_KV_HEADS * FLAT_GEMMA3_HEAD_DIM;
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![FLAT_GEMMA3_VOCAB, h],
        ),
        ("model.norm.weight".into(), vec![h]),
    ];
    for layer in 0..FLAT_GEMMA3_LAYERS {
        let stack = format!("model.layers.{layer}");
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![q_rows, h]),
            ("self_attn.k_proj.weight", vec![kv_rows, h]),
            ("self_attn.v_proj.weight", vec![kv_rows, h]),
            ("self_attn.o_proj.weight", vec![h, q_rows]),
            ("self_attn.q_norm.weight", vec![FLAT_GEMMA3_HEAD_DIM]),
            ("self_attn.k_norm.weight", vec![FLAT_GEMMA3_HEAD_DIM]),
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("pre_feedforward_layernorm.weight", vec![h]),
            ("post_feedforward_layernorm.weight", vec![h]),
            ("mlp.gate_proj.weight", vec![FLAT_GEMMA3_INTER, h]),
            ("mlp.up_proj.weight", vec![FLAT_GEMMA3_INTER, h]),
            ("mlp.down_proj.weight", vec![h, FLAT_GEMMA3_INTER]),
        ] {
            tensors.push((format!("{stack}.{name}"), shape));
        }
    }
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, shape) in &tensors {
        push_tensor(&mut header, &mut offset, name, shape);
    }
    inventory_from(dir, &config, &serde_json::Value::Object(header))
}
