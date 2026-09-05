//! A miniature Kimi-Linear-shaped checkpoint: KDA recurrences
//! interleaved with Multi-Latent Attention, at the `KKKM` cadence the
//! real 48B carries.
//!
//! Its own file because it is the *second* hybrid vocabulary, and the
//! two must not be read off one another. [`super::fixtures`]'s
//! `hybrid_lllf_f32_model` is Qwen3.8's shape — Gated DeltaNet against
//! softmax — where the recurrence fuses q|k|v into one projection and
//! the full layer is ordinary attention. This one is structurally
//! different on both sides: KDA splits q/k/v into three projections
//! through three convolutions with low-rank gate pairs, and its
//! full-attention layers are MLA, whose `q_proj`/`o_proj` are
//! byte-identical *spellings* to the softmax set at a different width.
//!
//! That collision is the point. A reader that names a mixer by which
//! tensors it can find, rather than by the operator the graph
//! declares, resolves this checkpoint to something it is not — which
//! is exactly what happened: every one of Kimi Linear's 27 layers once
//! reported as Gated DeltaNet. The fixture exists so that answer
//! cannot come back green.

use std::path::Path;

use super::fixtures::{lcg_values, norm_values, ShardBuilder};

const HIDDEN: usize = 32;
const INTER: usize = 64;
const VOCAB: usize = 64;
const LAYERS: usize = 4;

/// KDA geometry. `Hv · Dv` is the width every projection and conv
/// closes at.
const KDA_HEADS: usize = 4;
const KDA_HEAD_DIM: usize = 8;
const KDA_CONV_KERNEL: usize = 4;
/// Inner rank of the f and g gate factorisations — the low-rank pairs
/// that have no Gated DeltaNet counterpart.
const KDA_GATE_RANK: usize = 4;

/// MLA geometry, deliberately asymmetric: the query/key head width and
/// the value head width differ, so nothing can pass by assuming one
/// `head_dim` governs the layer.
const MLA_HEADS: usize = 4;
const MLA_KV_LORA_RANK: usize = 16;
const MLA_QK_NOPE_HEAD_DIM: usize = 8;
const MLA_QK_ROPE_HEAD_DIM: usize = 4;
const MLA_V_HEAD_DIM: usize = 8;

/// Layers that run KDA. Every other layer is MLA — `KKKM`, one-based
/// in the config exactly as the checkpoint spells it.
fn kda_layers() -> Vec<usize> {
    (0..LAYERS).filter(|l| l % 4 != 3).map(|l| l + 1).collect()
}

fn full_attn_layers() -> Vec<usize> {
    (0..LAYERS).filter(|l| l % 4 == 3).map(|l| l + 1).collect()
}

fn kda_width() -> usize {
    KDA_HEADS * KDA_HEAD_DIM
}

fn mla_q_head_dim() -> usize {
    MLA_QK_NOPE_HEAD_DIM + MLA_QK_ROPE_HEAD_DIM
}

/// The `KKKM` hybrid: three KDA layers, then one MLA layer.
pub fn hybrid_kda_mla_f32_model(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["KimiLinearForCausalLM"],
            "model_type": "kimi_linear",
            "torch_dtype": "float32",
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": MLA_HEADS,
            "num_key_value_heads": MLA_HEADS,
            // Deliberately not either operator's real width: an
            // operator-aware reader never consults it for these layers.
            "head_dim": 999,
            "vocab_size": VOCAB,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "kv_lora_rank": MLA_KV_LORA_RANK,
            "qk_nope_head_dim": MLA_QK_NOPE_HEAD_DIM,
            "qk_rope_head_dim": MLA_QK_ROPE_HEAD_DIM,
            "v_head_dim": MLA_V_HEAD_DIM,
            "mla_use_nope": true,
            "linear_attn_config": {
                "kda_layers": kda_layers(),
                "full_attn_layers": full_attn_layers(),
                "num_heads": KDA_HEADS,
                "head_dim": KDA_HEAD_DIM,
                "short_conv_kernel_size": KDA_CONV_KERNEL,
            },
        })
        .to_string(),
    )
    .unwrap();

    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 1),
    );
    shard.push("model.norm.weight", &[HIDDEN], &norm_values(HIDDEN, 2));
    shard.push(
        "lm_head.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 3),
    );
    for layer in 0..LAYERS {
        let seed = 200 + layer as u64 * 32;
        let prefix = format!("model.layers.{layer}");
        if layer % 4 == 3 {
            push_mla_layer(&mut shard, &prefix, seed);
        } else {
            push_kda_layer(&mut shard, &prefix, seed);
        }
        for (name, s) in [
            ("input_layernorm", seed + 20),
            ("post_attention_layernorm", seed + 21),
        ] {
            shard.push(
                &format!("{prefix}.{name}.weight"),
                &[HIDDEN],
                &norm_values(HIDDEN, s),
            );
        }
        for (name, rows, cols, s) in [
            ("gate_proj", INTER, HIDDEN, seed + 22),
            ("up_proj", INTER, HIDDEN, seed + 23),
            ("down_proj", HIDDEN, INTER, seed + 24),
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

/// KDA's fifteen operands. Three projections, three convs, two
/// low-rank gate pairs — none of which a Gated DeltaNet layer has.
fn push_kda_layer(shard: &mut ShardBuilder, prefix: &str, seed: u64) {
    let w = kda_width();
    for (name, s) in [("q_proj", seed), ("k_proj", seed + 1), ("v_proj", seed + 2)] {
        shard.push(
            &format!("{prefix}.self_attn.{name}.weight"),
            &[w, HIDDEN],
            &lcg_values(w * HIDDEN, s),
        );
    }
    for (name, s) in [
        ("q_conv1d", seed + 3),
        ("k_conv1d", seed + 4),
        ("v_conv1d", seed + 5),
    ] {
        shard.push(
            &format!("{prefix}.self_attn.{name}.weight"),
            &[w, 1, KDA_CONV_KERNEL],
            &lcg_values(w * KDA_CONV_KERNEL, s),
        );
    }
    for (name, s) in [("f_a_proj", seed + 6), ("g_a_proj", seed + 7)] {
        shard.push(
            &format!("{prefix}.self_attn.{name}.weight"),
            &[KDA_GATE_RANK, HIDDEN],
            &lcg_values(KDA_GATE_RANK * HIDDEN, s),
        );
    }
    for (name, s) in [("f_b_proj", seed + 8), ("g_b_proj", seed + 9)] {
        shard.push(
            &format!("{prefix}.self_attn.{name}.weight"),
            &[w, KDA_GATE_RANK],
            &lcg_values(w * KDA_GATE_RANK, s),
        );
    }
    shard.push(
        &format!("{prefix}.self_attn.b_proj.weight"),
        &[KDA_HEADS, HIDDEN],
        &lcg_values(KDA_HEADS * HIDDEN, seed + 10),
    );
    shard.push(
        &format!("{prefix}.self_attn.A_log"),
        &[KDA_HEADS],
        &lcg_values(KDA_HEADS, seed + 11),
    );
    shard.push(
        &format!("{prefix}.self_attn.dt_bias"),
        &[w],
        &lcg_values(w, seed + 12),
    );
    shard.push(
        &format!("{prefix}.self_attn.o_norm.weight"),
        &[KDA_HEAD_DIM],
        &norm_values(KDA_HEAD_DIM, seed + 13),
    );
    shard.push(
        &format!("{prefix}.self_attn.o_proj.weight"),
        &[HIDDEN, w],
        &lcg_values(HIDDEN * w, seed + 14),
    );
}

/// MLA's five. `q_proj` and `o_proj` are the same spellings a softmax
/// layer uses, at a width only the operator explains.
fn push_mla_layer(shard: &mut ShardBuilder, prefix: &str, seed: u64) {
    let q_rows = MLA_HEADS * mla_q_head_dim();
    shard.push(
        &format!("{prefix}.self_attn.q_proj.weight"),
        &[q_rows, HIDDEN],
        &lcg_values(q_rows * HIDDEN, seed),
    );
    shard.push(
        &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
        &[MLA_KV_LORA_RANK + MLA_QK_ROPE_HEAD_DIM, HIDDEN],
        &lcg_values((MLA_KV_LORA_RANK + MLA_QK_ROPE_HEAD_DIM) * HIDDEN, seed + 1),
    );
    shard.push(
        &format!("{prefix}.self_attn.kv_a_layernorm.weight"),
        &[MLA_KV_LORA_RANK],
        &norm_values(MLA_KV_LORA_RANK, seed + 2),
    );
    let kv_b_rows = MLA_HEADS * (MLA_QK_NOPE_HEAD_DIM + MLA_V_HEAD_DIM);
    shard.push(
        &format!("{prefix}.self_attn.kv_b_proj.weight"),
        &[kv_b_rows, MLA_KV_LORA_RANK],
        &lcg_values(kv_b_rows * MLA_KV_LORA_RANK, seed + 3),
    );
    shard.push(
        &format!("{prefix}.self_attn.o_proj.weight"),
        &[HIDDEN, MLA_HEADS * MLA_V_HEAD_DIM],
        &lcg_values(HIDDEN * MLA_HEADS * MLA_V_HEAD_DIM, seed + 4),
    );
}

// ── A per-expert MoE miniature with BYTES, for the prepared plan ─────

/// Routed-MoE geometry of the bytes-backed miniature below. Small enough
/// to execute in a test, awkward enough to catch a transposed matrix:
/// the expert width is not the hidden width, and `top_k` is not one.
pub const MOE_EXPERTS: usize = 4;
pub const MOE_TOP_K: usize = 2;
pub const MOE_INTER: usize = 24;
pub const MOE_SHARED_EXPERTS: usize = 1;
/// `first_k_dense_replace`: layer 0 runs a dense MLP, every other layer
/// routes — Kimi Linear's own arrangement.
pub const MOE_DENSE_PREFIX: usize = 1;
pub const MOE_LAYERS: usize = 3;
/// Plain softmax attention on every layer, so the plan executes through
/// the prepared executor today; KDA/MLA execution through the prepared
/// plan is its own binding (K3-RESIDENCY-VERTICAL-1, V3).
const MOE_Q_HEADS: usize = 4;
const MOE_KV_HEADS: usize = 2;
const MOE_HEAD_DIM: usize = 8;

/// The width of layer 0's dense MLP — the routed width, so one seed
/// table serves both; the bytes differ by seed, never by shape.
const MOE_DENSE_INTER: usize = MOE_INTER;

/// A Kimi-Linear-shaped checkpoint whose FFN is a PER-EXPERT bank with a
/// shared expert, written with real f32 values: the executable subject
/// for a mapped bank bound through the prepared plan. The routed layers'
/// experts are addressed by index, so a wrong expert's bytes are a wrong
/// answer, not a tolerance.
///
/// `expert_seed(e)` names each expert's values, so a test can write a
/// twin container in which one expert's bytes are another's and prove
/// the executor read the expert it selected.
pub fn kimi_per_expert_moe_f32_model(dir: &Path) {
    kimi_per_expert_moe_f32_model_with(dir, |e| e as u64)
}

/// [`kimi_per_expert_moe_f32_model`] with the expert-to-seed map chosen
/// by the caller — the twin-container control.
pub fn kimi_per_expert_moe_f32_model_with(dir: &Path, expert_seed: impl Fn(usize) -> u64) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["KimiLinearForCausalLM"],
            "model_type": "kimi_linear",
            "torch_dtype": "float32",
            "hidden_size": HIDDEN,
            "intermediate_size": MOE_DENSE_INTER,
            "num_hidden_layers": MOE_LAYERS,
            "num_attention_heads": MOE_Q_HEADS,
            "num_key_value_heads": MOE_KV_HEADS,
            "head_dim": MOE_HEAD_DIM,
            "vocab_size": VOCAB,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "kda_layers": [],
                "full_attn_layers": (1..=MOE_LAYERS).collect::<Vec<_>>()
            },
            "first_k_dense_replace": MOE_DENSE_PREFIX,
            "num_experts": MOE_EXPERTS,
            "num_experts_per_token": MOE_TOP_K,
            "num_shared_experts": MOE_SHARED_EXPERTS,
            "moe_intermediate_size": MOE_INTER,
            "moe_router_activation_func": "sigmoid",
            "moe_renormalize": true,
            "routed_scaling_factor": 2.446,
        })
        .to_string(),
    )
    .unwrap();

    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 1),
    );
    shard.push("model.norm.weight", &[HIDDEN], &norm_values(HIDDEN, 2));
    shard.push(
        "lm_head.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 3),
    );
    let q = MOE_Q_HEADS * MOE_HEAD_DIM;
    let kv = MOE_KV_HEADS * MOE_HEAD_DIM;
    for layer in 0..MOE_LAYERS {
        let seed = 400 + layer as u64 * 64;
        let prefix = format!("model.layers.{layer}");
        for (name, rows, cols, s) in [
            ("q_proj", q, HIDDEN, seed),
            ("k_proj", kv, HIDDEN, seed + 1),
            ("v_proj", kv, HIDDEN, seed + 2),
            ("o_proj", HIDDEN, q, seed + 3),
        ] {
            shard.push(
                &format!("{prefix}.self_attn.{name}.weight"),
                &[rows, cols],
                &lcg_values(rows * cols, s),
            );
        }
        for (name, s) in [
            ("input_layernorm", seed + 20),
            ("post_attention_layernorm", seed + 21),
        ] {
            shard.push(
                &format!("{prefix}.{name}.weight"),
                &[HIDDEN],
                &norm_values(HIDDEN, s),
            );
        }
        if layer < MOE_DENSE_PREFIX {
            for (name, rows, cols, s) in [
                ("gate_proj", MOE_DENSE_INTER, HIDDEN, seed + 22),
                ("up_proj", MOE_DENSE_INTER, HIDDEN, seed + 23),
                ("down_proj", HIDDEN, MOE_DENSE_INTER, seed + 24),
            ] {
                shard.push(
                    &format!("{prefix}.mlp.{name}.weight"),
                    &[rows, cols],
                    &lcg_values(rows * cols, s),
                );
            }
            continue;
        }
        let moe = format!("{prefix}.block_sparse_moe");
        shard.push(
            &format!("{moe}.gate.weight"),
            &[MOE_EXPERTS, HIDDEN],
            &lcg_values(MOE_EXPERTS * HIDDEN, seed + 30),
        );
        shard.push(
            &format!("{moe}.gate.e_score_correction_bias"),
            &[MOE_EXPERTS],
            &lcg_values(MOE_EXPERTS, seed + 31),
        );
        let shared_inter = MOE_INTER * MOE_SHARED_EXPERTS;
        for (name, rows, cols, s) in [
            ("gate_proj", shared_inter, HIDDEN, seed + 32),
            ("up_proj", shared_inter, HIDDEN, seed + 33),
            ("down_proj", HIDDEN, shared_inter, seed + 34),
        ] {
            shard.push(
                &format!("{moe}.shared_experts.{name}.weight"),
                &[rows, cols],
                &lcg_values(rows * cols, s),
            );
        }
        for expert in 0..MOE_EXPERTS {
            let es = seed + 40 + 3 * expert_seed(expert);
            for (name, rows, cols, s) in [
                ("w1", MOE_INTER, HIDDEN, es),
                ("w3", MOE_INTER, HIDDEN, es + 1),
                ("w2", HIDDEN, MOE_INTER, es + 2),
            ] {
                shard.push(
                    &format!("{moe}.experts.{expert}.{name}.weight"),
                    &[rows, cols],
                    &lcg_values(rows * cols, s),
                );
            }
        }
    }
    shard.write(dir);
}
