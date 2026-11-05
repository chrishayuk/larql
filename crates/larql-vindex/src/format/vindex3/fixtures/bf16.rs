//! A small **BF16** container with a compiled Q8_0 pack, for tests that edit
//! stored weights.
//!
//! Llama-shaped, with the dense fixture's names and geometry (hidden 64,
//! intermediate 256, so every projection is Q8_0-eligible), stored as
//! bfloat16 the way a real checkpoint is, then given its Q8_0 pack by the
//! production compiler. The decoder tensors live in the one object
//! `target.decoder_stack`, as in the Qwen3-0.6B container.

use std::path::{Path, PathBuf};

use super::{
    encode_fixture_container, lcg_values, norm_values, ShardBuilder, DENSE_HEAD_DIM, DENSE_HIDDEN,
    DENSE_INTERMEDIATE, DENSE_KV_HEADS, DENSE_LAYERS, DENSE_Q_HEADS, DENSE_VOCAB,
};
use crate::format::vindex3::represent::policy::{Protections, RolePolicy};
use crate::format::vindex3::represent::{compile_representation, RepresentSpec};

/// The Q8_0 compiler's encoding label.
const Q8_0: &str = "Q8_0";

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn push(shard: &mut ShardBuilder, name: &str, shape: &[usize], values: &[f32]) {
    shard.push_bytes(name, "BF16", shape, &bf16_bytes(values));
}

fn write_bf16_checkpoint(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16", "model_type": "llama",
            "tie_word_embeddings": false, "hidden_size": DENSE_HIDDEN, "num_hidden_layers": DENSE_LAYERS,
            "intermediate_size": DENSE_INTERMEDIATE, "num_attention_heads": DENSE_Q_HEADS,
            "num_key_value_heads": DENSE_KV_HEADS, "head_dim": DENSE_HEAD_DIM, "vocab_size": DENSE_VOCAB,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0
        })
        .to_string(),
    )
    .unwrap();
    let (q, kv) = (
        DENSE_Q_HEADS * DENSE_HEAD_DIM,
        DENSE_KV_HEADS * DENSE_HEAD_DIM,
    );
    let mut s = ShardBuilder::new();
    push(
        &mut s,
        "model.embed_tokens.weight",
        &[DENSE_VOCAB, DENSE_HIDDEN],
        &lcg_values(DENSE_VOCAB * DENSE_HIDDEN, 1),
    );
    push(
        &mut s,
        "model.norm.weight",
        &[DENSE_HIDDEN],
        &norm_values(DENSE_HIDDEN, 2),
    );
    push(
        &mut s,
        "lm_head.weight",
        &[DENSE_VOCAB, DENSE_HIDDEN],
        &lcg_values(DENSE_VOCAB * DENSE_HIDDEN, 3),
    );
    for layer in 0..DENSE_LAYERS {
        let seed = 100 + layer as u64 * 10;
        let p = format!("model.layers.{layer}");
        push(
            &mut s,
            &format!("{p}.self_attn.q_proj.weight"),
            &[q, DENSE_HIDDEN],
            &lcg_values(q * DENSE_HIDDEN, seed),
        );
        push(
            &mut s,
            &format!("{p}.self_attn.k_proj.weight"),
            &[kv, DENSE_HIDDEN],
            &lcg_values(kv * DENSE_HIDDEN, seed + 1),
        );
        push(
            &mut s,
            &format!("{p}.self_attn.v_proj.weight"),
            &[kv, DENSE_HIDDEN],
            &lcg_values(kv * DENSE_HIDDEN, seed + 2),
        );
        push(
            &mut s,
            &format!("{p}.self_attn.o_proj.weight"),
            &[DENSE_HIDDEN, q],
            &lcg_values(DENSE_HIDDEN * q, seed + 3),
        );
        push(
            &mut s,
            &format!("{p}.input_layernorm.weight"),
            &[DENSE_HIDDEN],
            &norm_values(DENSE_HIDDEN, seed + 4),
        );
        push(
            &mut s,
            &format!("{p}.post_attention_layernorm.weight"),
            &[DENSE_HIDDEN],
            &norm_values(DENSE_HIDDEN, seed + 5),
        );
        push(
            &mut s,
            &format!("{p}.mlp.gate_proj.weight"),
            &[DENSE_INTERMEDIATE, DENSE_HIDDEN],
            &lcg_values(DENSE_INTERMEDIATE * DENSE_HIDDEN, seed + 6),
        );
        push(
            &mut s,
            &format!("{p}.mlp.up_proj.weight"),
            &[DENSE_INTERMEDIATE, DENSE_HIDDEN],
            &lcg_values(DENSE_INTERMEDIATE * DENSE_HIDDEN, seed + 7),
        );
        push(
            &mut s,
            &format!("{p}.mlp.down_proj.weight"),
            &[DENSE_HIDDEN, DENSE_INTERMEDIATE],
            &lcg_values(DENSE_HIDDEN * DENSE_INTERMEDIATE, seed + 8),
        );
    }
    s.write(dir);
}

/// Build the container under `work` (which must exist) and return its path:
/// BF16 canonical weights plus a Q8_0 pack compiled by the production compiler.
pub fn bf16_container_with_q8_pack(work: &Path) -> PathBuf {
    let checkpoint = work.join("checkpoint");
    std::fs::create_dir(&checkpoint).unwrap();
    let canonical = work.join("canonical.vindex3");
    encode_fixture_container(write_bf16_checkpoint, &checkpoint, &canonical, "wsp");
    let container = work.join("source.vindex3");
    let spec = RepresentSpec {
        encoding: Q8_0.to_string(),
        objects: Vec::new(),
        roles: RolePolicy::default(),
        deployment: false,
        protect: Protections::default(),
    };
    compile_representation(&canonical, &container, &spec).unwrap();
    container
}
