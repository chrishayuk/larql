//! A Qwen2-MoE miniature: softmax attention with Q/K/V biases and an
//! unbiased output projection (`qkv_bias`), a per-expert routed bank under
//! the `mlp.experts.{id}.gate_proj/up_proj/down_proj` spelling, and a
//! shared expert scaled by its own sigmoid gate.
//!
//! Tensor names and shapes follow `Qwen2MoeForCausalLM`; the values are
//! the deterministic LCG every other miniature uses. The variants exist so
//! a test can take away exactly one fact — a declaration, a bias's values,
//! the gate's values, or add an operand the declaration forbids — and see
//! that single change surface.

use std::path::Path;

use super::fixtures::{lcg_values, norm_values, ShardBuilder};

pub const QWEN_HIDDEN: usize = 16;
pub const QWEN_LAYERS: usize = 2;
pub const QWEN_Q_HEADS: usize = 4;
pub const QWEN_KV_HEADS: usize = 2;
pub const QWEN_HEAD_DIM: usize = QWEN_HIDDEN / QWEN_Q_HEADS;
pub const QWEN_EXPERTS: usize = 4;
pub const QWEN_TOP_K: usize = 2;
pub const QWEN_EXPERT_WIDTH: usize = 8;
pub const QWEN_SHARED_WIDTH: usize = 12;
pub const QWEN_VOCAB: usize = 32;
pub const QWEN_TOKENS: [u32; 6] = [3, 17, 28, 0, 11, 5];
/// The seeded values are small enough that `sigmoid(w · x)` sits near
/// 0.5 — where a zeroed gate also sits — so the gate row is scaled until
/// its sigmoid moves well away from the midpoint and taking it away shows.
pub const QWEN_GATE_SCALE: f32 = 16.0;

/// How the checkpoint declares its attention biases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasDeclaration {
    /// `qkv_bias: true`, as transformers 5 writes a Qwen2-MoE config.
    QkvBias,
    /// No bias key at all, as every pre-transformers-5 Qwen2 config is:
    /// the family default answers.
    Silent,
    /// `qkv_bias: true` beside `attention_bias: true` — a contradiction.
    Both,
}

/// One miniature's deliberate variations from the plain model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenMoeForm {
    pub declaration: BiasDeclaration,
    /// Ship the Q/K/V bias tensors with their seeded values; `false`
    /// ships them as zeros, so the biases are present but inert.
    pub bias_values: bool,
    /// Also ship an `o_proj.bias`, which `qkv_bias` does not declare.
    pub output_bias: bool,
    /// Ship the shared-expert gate as zeros (`sigmoid(0) = 0.5` on every
    /// token) instead of its seeded values.
    pub zero_branch_gate: bool,
}

impl Default for QwenMoeForm {
    fn default() -> Self {
        Self {
            declaration: BiasDeclaration::QkvBias,
            bias_values: true,
            output_bias: false,
            zero_branch_gate: false,
        }
    }
}

pub fn miniature_qwen2_moe(dir: &Path) {
    miniature_qwen2_moe_with(dir, QwenMoeForm::default());
}

pub fn miniature_qwen2_moe_with(dir: &Path, form: QwenMoeForm) {
    let mut config = serde_json::json!({
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "torch_dtype": "float32",
        "hidden_size": QWEN_HIDDEN,
        "num_hidden_layers": QWEN_LAYERS,
        "intermediate_size": QWEN_SHARED_WIDTH * 2,
        "moe_intermediate_size": QWEN_EXPERT_WIDTH,
        "shared_expert_intermediate_size": QWEN_SHARED_WIDTH,
        "num_attention_heads": QWEN_Q_HEADS,
        "num_key_value_heads": QWEN_KV_HEADS,
        "num_experts": QWEN_EXPERTS,
        "num_experts_per_tok": QWEN_TOP_K,
        "norm_topk_prob": false,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "vocab_size": QWEN_VOCAB,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": false
    });
    match form.declaration {
        BiasDeclaration::QkvBias => config["qkv_bias"] = serde_json::json!(true),
        BiasDeclaration::Silent => {}
        BiasDeclaration::Both => {
            config["qkv_bias"] = serde_json::json!(true);
            config["attention_bias"] = serde_json::json!(true);
        }
    }
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let q_rows = QWEN_Q_HEADS * QWEN_HEAD_DIM;
    let kv_rows = QWEN_KV_HEADS * QWEN_HEAD_DIM;
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[QWEN_VOCAB, QWEN_HIDDEN],
        &lcg_values(QWEN_VOCAB * QWEN_HIDDEN, 1),
    );
    shard.push(
        "model.norm.weight",
        &[QWEN_HIDDEN],
        &norm_values(QWEN_HIDDEN, 2),
    );
    shard.push(
        "lm_head.weight",
        &[QWEN_VOCAB, QWEN_HIDDEN],
        &lcg_values(QWEN_VOCAB * QWEN_HIDDEN, 3),
    );
    let bias = |n: usize, seed: u64| {
        if form.bias_values {
            lcg_values(n, seed)
        } else {
            vec![0.0; n]
        }
    };
    for layer in 0..QWEN_LAYERS {
        let seed = 100 + layer as u64 * 100;
        let prefix = format!("model.layers.{layer}");
        let attn = format!("{prefix}.self_attn");
        for (name, rows, cols, s) in [
            ("q_proj", q_rows, QWEN_HIDDEN, seed),
            ("k_proj", kv_rows, QWEN_HIDDEN, seed + 1),
            ("v_proj", kv_rows, QWEN_HIDDEN, seed + 2),
            ("o_proj", QWEN_HIDDEN, q_rows, seed + 3),
        ] {
            shard.push(
                &format!("{attn}.{name}.weight"),
                &[rows, cols],
                &lcg_values(rows * cols, s),
            );
        }
        for (name, rows, s) in [
            ("q_proj", q_rows, seed + 4),
            ("k_proj", kv_rows, seed + 5),
            ("v_proj", kv_rows, seed + 6),
        ] {
            shard.push(&format!("{attn}.{name}.bias"), &[rows], &bias(rows, s));
        }
        if form.output_bias {
            shard.push(
                &format!("{attn}.o_proj.bias"),
                &[QWEN_HIDDEN],
                &lcg_values(QWEN_HIDDEN, seed + 7),
            );
        }
        shard.push(
            &format!("{prefix}.input_layernorm.weight"),
            &[QWEN_HIDDEN],
            &norm_values(QWEN_HIDDEN, seed + 8),
        );
        shard.push(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[QWEN_HIDDEN],
            &norm_values(QWEN_HIDDEN, seed + 9),
        );
        let mlp = format!("{prefix}.mlp");
        shard.push(
            &format!("{mlp}.gate.weight"),
            &[QWEN_EXPERTS, QWEN_HIDDEN],
            &lcg_values(QWEN_EXPERTS * QWEN_HIDDEN, seed + 10),
        );
        for expert in 0..QWEN_EXPERTS {
            let s = seed + 20 + expert as u64 * 3;
            let e = format!("{mlp}.experts.{expert}");
            push_swiglu(&mut shard, &e, QWEN_EXPERT_WIDTH, s);
        }
        push_swiglu(
            &mut shard,
            &format!("{mlp}.shared_expert"),
            QWEN_SHARED_WIDTH,
            seed + 60,
        );
        shard.push(
            &format!("{mlp}.shared_expert_gate.weight"),
            &[1, QWEN_HIDDEN],
            &if form.zero_branch_gate {
                vec![0.0; QWEN_HIDDEN]
            } else {
                lcg_values(QWEN_HIDDEN, seed + 70)
                    .into_iter()
                    .map(|v| v * QWEN_GATE_SCALE)
                    .collect::<Vec<_>>()
            },
        );
    }
    shard.write(dir);
}

/// One SwiGLU block's three projections under `{prefix}.{gate,up,down}_proj`.
fn push_swiglu(shard: &mut ShardBuilder, prefix: &str, width: usize, seed: u64) {
    shard.push(
        &format!("{prefix}.gate_proj.weight"),
        &[width, QWEN_HIDDEN],
        &lcg_values(width * QWEN_HIDDEN, seed),
    );
    shard.push(
        &format!("{prefix}.up_proj.weight"),
        &[width, QWEN_HIDDEN],
        &lcg_values(width * QWEN_HIDDEN, seed + 1),
    );
    shard.push(
        &format!("{prefix}.down_proj.weight"),
        &[QWEN_HIDDEN, width],
        &lcg_values(QWEN_HIDDEN * width, seed + 2),
    );
}
