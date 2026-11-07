//! The hybrid (linear + full attention) f32 model fixture.

use std::path::Path;

#[allow(unused_imports)]
use super::*;

/// A four-layer **hybrid** stack on an `LLLF` cadence: three Gated
/// DeltaNet layers and one softmax layer.
///
/// The fixture the QW-3.6b traversal gates run on. Small, but not too
/// small to discriminate:
///
/// * **three** recurrent layers before the softmax one, so a dispatch
///   that ran the first operator for every layer is caught by position,
///   not just by count;
/// * the softmax layer is **last**, so a traversal that refuses lazily
///   has already emitted three layers before it gets there;
/// * `conv_kernel` 4 against sequences longer than 4, so the convolution
///   history spans a batch boundary.
pub fn hybrid_lllf_f32_model(dir: &Path) {
    hybrid_lllf_f32_model_with_layers(dir, HYBRID_CADENCE);
}

/// Every `HYBRID_CADENCE`-th layer of the hybrid fixture attends.
pub const HYBRID_CADENCE: usize = 4;

/// The hybrid fixture at `layers` depth, still on the `LLLF` cadence.
///
/// Depth is the one thing a single `LLLF` block cannot vary, and a
/// target that reads the attention *interval* off the stack needs at
/// least two attending layers to see one: one block states a position,
/// two state a cadence.
pub fn hybrid_lllf_f32_model_with_layers(dir: &Path, layers: usize) {
    // Linear-attention geometry: 2*Hk*Dk + Hv*Dv channels.
    const KEY_HEADS: usize = 2;
    const VALUE_HEADS: usize = 4;
    const LIN_DIM: usize = 8;
    const CONV_KERNEL: usize = 4;
    let qkv = 2 * KEY_HEADS * LIN_DIM + VALUE_HEADS * LIN_DIM;
    let value_width = VALUE_HEADS * LIN_DIM;
    let attends = |layer: usize| layer % HYBRID_CADENCE == HYBRID_CADENCE - 1;
    let layer_types: Vec<&str> = (0..layers)
        .map(|l| {
            if attends(l) {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();

    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "torch_dtype": "float32",
            "model_type": "qwen3_5",
            "hidden_size": DENSE_HIDDEN,
            "num_hidden_layers": layers,
            "intermediate_size": DENSE_INTERMEDIATE,
            "num_attention_heads": DENSE_Q_HEADS,
            "num_key_value_heads": DENSE_KV_HEADS,
            "head_dim": DENSE_HEAD_DIM,
            "vocab_size": DENSE_VOCAB,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "layer_types": layer_types,
            "full_attention_interval": HYBRID_CADENCE,
            "linear_num_key_heads": KEY_HEADS,
            "linear_num_value_heads": VALUE_HEADS,
            "linear_key_head_dim": LIN_DIM,
            "linear_value_head_dim": LIN_DIM,
            "linear_conv_kernel_dim": CONV_KERNEL,
            "mamba_ssm_dtype": "float32"
        })
        .to_string(),
    )
    .unwrap();

    let q_rows = DENSE_Q_HEADS * DENSE_HEAD_DIM;
    let kv_rows = DENSE_KV_HEADS * DENSE_HEAD_DIM;
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[DENSE_VOCAB, DENSE_HIDDEN],
        &lcg_values(DENSE_VOCAB * DENSE_HIDDEN, 1),
    );
    shard.push(
        "model.norm.weight",
        &[DENSE_HIDDEN],
        &norm_values(DENSE_HIDDEN, 2),
    );
    shard.push(
        "lm_head.weight",
        &[DENSE_VOCAB, DENSE_HIDDEN],
        &lcg_values(DENSE_VOCAB * DENSE_HIDDEN, 3),
    );
    for layer in 0..layers {
        let seed = 100 + layer as u64 * 20;
        let prefix = format!("model.layers.{layer}");
        if attends(layer) {
            // The softmax layer.
            for (name, rows, s) in [
                ("q_proj", q_rows, seed),
                ("k_proj", kv_rows, seed + 1),
                ("v_proj", kv_rows, seed + 2),
            ] {
                shard.push(
                    &format!("{prefix}.self_attn.{name}.weight"),
                    &[rows, DENSE_HIDDEN],
                    &lcg_values(rows * DENSE_HIDDEN, s),
                );
            }
            shard.push(
                &format!("{prefix}.self_attn.o_proj.weight"),
                &[DENSE_HIDDEN, q_rows],
                &lcg_values(DENSE_HIDDEN * q_rows, seed + 3),
            );
        } else {
            // A recurrence: nine operands, no q/k/v/o.
            shard.push(
                &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                &[qkv, DENSE_HIDDEN],
                &lcg_values(qkv * DENSE_HIDDEN, seed),
            );
            for (name, s) in [("in_proj_a", seed + 1), ("in_proj_b", seed + 2)] {
                shard.push(
                    &format!("{prefix}.linear_attn.{name}.weight"),
                    &[VALUE_HEADS, DENSE_HIDDEN],
                    &lcg_values(VALUE_HEADS * DENSE_HIDDEN, s),
                );
            }
            shard.push(
                &format!("{prefix}.linear_attn.in_proj_z.weight"),
                &[value_width, DENSE_HIDDEN],
                &lcg_values(value_width * DENSE_HIDDEN, seed + 3),
            );
            shard.push(
                &format!("{prefix}.linear_attn.conv1d.weight"),
                &[qkv, 1, CONV_KERNEL],
                &lcg_values(qkv * CONV_KERNEL, seed + 4),
            );
            shard.push(
                &format!("{prefix}.linear_attn.A_log"),
                &[VALUE_HEADS],
                &lcg_values(VALUE_HEADS, seed + 5),
            );
            shard.push(
                &format!("{prefix}.linear_attn.dt_bias"),
                &[VALUE_HEADS],
                &lcg_values(VALUE_HEADS, seed + 6),
            );
            shard.push(
                &format!("{prefix}.linear_attn.norm.weight"),
                &[LIN_DIM],
                &norm_values(LIN_DIM, seed + 7),
            );
            shard.push(
                &format!("{prefix}.linear_attn.out_proj.weight"),
                &[DENSE_HIDDEN, value_width],
                &lcg_values(DENSE_HIDDEN * value_width, seed + 8),
            );
        }
        for (name, s) in [
            ("input_layernorm", seed + 10),
            ("post_attention_layernorm", seed + 11),
        ] {
            shard.push(
                &format!("{prefix}.{name}.weight"),
                &[DENSE_HIDDEN],
                &norm_values(DENSE_HIDDEN, s),
            );
        }
        for (name, rows, cols, s) in [
            ("gate_proj", DENSE_INTERMEDIATE, DENSE_HIDDEN, seed + 12),
            ("up_proj", DENSE_INTERMEDIATE, DENSE_HIDDEN, seed + 13),
            ("down_proj", DENSE_HIDDEN, DENSE_INTERMEDIATE, seed + 14),
        ] {
            shard.push(
                &format!("{prefix}.mlp.{name}.weight"),
                &[rows, cols],
                &lcg_values(rows * cols, s),
            );
        }
    }
    shard.write(dir);
}
