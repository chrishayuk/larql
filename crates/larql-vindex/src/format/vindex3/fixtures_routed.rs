//! Shared GPT-OSS miniature: packed MXFP4, biases, routing and sliding attention.
use super::fixtures::{lcg_values, norm_values, ShardBuilder};
use super::opplan::exec::weights::{quantize_mxfp4, LoadedWeight};
use std::path::Path;

// ── Routed miniature geometry: MXFP4 needs k ≡ 0 (mod 32) on both projections ──
pub const HIDDEN: usize = 32;
pub const INTER: usize = 32;
pub const EXPERTS: usize = 4;
pub const TOP_K: usize = 2;
pub const Q_HEADS: usize = 2;
pub const KV_HEADS: usize = 1;
pub const HEAD_DIM: usize = 16;
pub const VOCAB: usize = 29;
pub const LAYERS: usize = 2;
pub const WINDOW: usize = 4;
/// GPT-OSS's published clamp.
pub const SWIGLU_LIMIT: f32 = 7.0;
/// Gate and up share one fused operand.
pub const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;
/// MXFP4 group geometry, as the packer lays it out.
pub const MXFP4_GROUP_ELEMS: usize = larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;
pub const MXFP4_GROUP_BYTES: usize = larql_models::quant::mxfp4::MXFP4_GROUP_BYTES;
/// Bank-relative suffixes of the classified MXFP4 streams and of the
/// unclassified BF16 copies (which land in the bank object because they
/// share its prefix, but block closure — so they are only ever reached by
/// an op edited to point at them).
pub const GATE_UP_BLOCKS: &str = "mlp.experts.gate_up_proj_blocks";
pub const DOWN_BLOCKS: &str = "mlp.experts.down_proj_blocks";
pub const GATE_UP_BF16: &str = "mlp.experts.gate_up_proj_bf16";
pub const DOWN_BF16: &str = "mlp.experts.down_proj_bf16";
pub const BLOCKS_SUFFIX: &str = "_blocks";
pub const SCALES_SUFFIX: &str = "_scales";
pub const BF16_SUFFIX: &str = "_bf16";
pub const DTYPE_U8: &str = "U8";
pub const DTYPE_BF16: &str = "BF16";
/// Seed of the normalised FFN input the routed tests apply.
pub const INPUT_SEED: u64 = 9_001;

/// Loop-vs-BLAS reassociation on 32-wide operands, and the bf16 → f16 /
/// e2m1 → f16 conversions are exact, so f16 realisations sit here too.
pub const NOISE_CEILING: f32 = 1e-5;
/// A hand-built call's epsilon.
pub const EPS: f64 = 1e-5;

/// Round-to-nearest-even bf16 of `v`, back as the f32 it denotes.
pub fn bf16_round(v: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(v)) << 16)
}

/// The bf16 bit pattern nearest `v` (ties to even).
pub fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let rounding = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

/// The GPT-OSS miniature with bf16-representable source values. With
/// `bf16_copies`, each layer's expert bank additionally carries an
/// unquantised BF16 copy of both projections under unclassified names.
pub fn miniature_gpt_oss(dir: &Path, bf16_copies: bool) {
    let config = serde_json::json!({
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "torch_dtype": "float32",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": INTER,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "sliding_window": WINDOW,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_theta": 10000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4096,
            "truncate": false
        },
        "rms_norm_eps": 1e-5,
        "attention_bias": true,
        "swiglu_limit": SWIGLU_LIMIT,
        "tie_word_embeddings": false,
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head"
            ]
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let q_rows = Q_HEADS * HEAD_DIM;
    let kv_rows = KV_HEADS * HEAD_DIM;
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
        let seed = 700 + layer as u64 * 40;
        let prefix = format!("model.layers.{layer}");
        let scaled = |n: usize, s: u64, g: f32| -> Vec<f32> {
            lcg_values(n, s).into_iter().map(|v| v * g).collect()
        };
        for (suffix, shape, values) in [
            (
                "self_attn.q_proj.weight",
                vec![q_rows, HIDDEN],
                scaled(q_rows * HIDDEN, seed, 1.0),
            ),
            (
                "self_attn.k_proj.weight",
                vec![kv_rows, HIDDEN],
                scaled(kv_rows * HIDDEN, seed + 1, 1.0),
            ),
            (
                "self_attn.v_proj.weight",
                vec![kv_rows, HIDDEN],
                scaled(kv_rows * HIDDEN, seed + 2, 1.0),
            ),
            (
                "self_attn.o_proj.weight",
                vec![HIDDEN, q_rows],
                scaled(HIDDEN * q_rows, seed + 3, 1.0),
            ),
            (
                "self_attn.q_proj.bias",
                vec![q_rows],
                scaled(q_rows, seed + 4, 2.0),
            ),
            (
                "self_attn.k_proj.bias",
                vec![kv_rows],
                scaled(kv_rows, seed + 5, 2.0),
            ),
            (
                "self_attn.v_proj.bias",
                vec![kv_rows],
                scaled(kv_rows, seed + 6, 2.0),
            ),
            (
                "self_attn.o_proj.bias",
                vec![HIDDEN],
                scaled(HIDDEN, seed + 7, 2.0),
            ),
            (
                "self_attn.sinks",
                vec![Q_HEADS],
                scaled(Q_HEADS, seed + 8, 4.0),
            ),
            (
                "input_layernorm.weight",
                vec![HIDDEN],
                norm_values(HIDDEN, seed + 9),
            ),
            (
                "post_attention_layernorm.weight",
                vec![HIDDEN],
                norm_values(HIDDEN, seed + 10),
            ),
            (
                "mlp.router.weight",
                vec![EXPERTS, HIDDEN],
                scaled(EXPERTS * HIDDEN, seed + 11, 4.0),
            ),
            (
                "mlp.router.bias",
                vec![EXPERTS],
                scaled(EXPERTS, seed + 12, 1.0),
            ),
            (
                "mlp.experts.gate_up_proj_bias",
                vec![EXPERTS, FUSED_BRANCHES * INTER],
                scaled(EXPERTS * FUSED_BRANCHES * INTER, seed + 13, 1.0),
            ),
            (
                "mlp.experts.down_proj_bias",
                vec![EXPERTS, HIDDEN],
                scaled(EXPERTS * HIDDEN, seed + 14, 1.0),
            ),
        ] {
            shard.push(&format!("{prefix}.{suffix}"), &shape, &values);
        }
        // Packed MXFP4 experts from bf16-representable sources, so a BF16
        // copy of the same bank requantises to these exact bytes.
        for (suffix, bf16_suffix, rows, k, s) in [
            (
                GATE_UP_BLOCKS,
                GATE_UP_BF16,
                FUSED_BRANCHES * INTER,
                HIDDEN,
                seed + 15,
            ),
            (DOWN_BLOCKS, DOWN_BF16, HIDDEN, INTER, seed + 16),
        ] {
            let groups = k / MXFP4_GROUP_ELEMS;
            let mut blocks = Vec::new();
            let mut scales = Vec::new();
            let mut bf16 = Vec::new();
            for e in 0..EXPERTS {
                let values: Vec<f32> = scaled(rows * k, s + e as u64 * 100, 2.0)
                    .into_iter()
                    .map(bf16_round)
                    .collect();
                let LoadedWeight::Mxfp4 { packed, scales: sc } =
                    quantize_mxfp4(&values, rows, k, suffix).unwrap()
                else {
                    unreachable!()
                };
                blocks.extend_from_slice(&packed.as_slice()[..rows * groups * MXFP4_GROUP_BYTES]);
                scales.extend_from_slice(&sc.as_slice()[..rows * groups]);
                bf16.extend(values.iter().flat_map(|v| bf16_bits(*v).to_le_bytes()));
            }
            shard.push_bytes(
                &format!("{prefix}.{suffix}"),
                DTYPE_U8,
                &[EXPERTS, rows, groups, MXFP4_GROUP_BYTES],
                &blocks,
            );
            shard.push_bytes(
                &format!("{prefix}.{}", suffix.replace(BLOCKS_SUFFIX, SCALES_SUFFIX)),
                DTYPE_U8,
                &[EXPERTS, rows, groups],
                &scales,
            );
            if bf16_copies {
                shard.push_bytes(
                    &format!("{prefix}.{bf16_suffix}"),
                    DTYPE_BF16,
                    &[EXPERTS, rows, k],
                    &bf16,
                );
            }
        }
    }
    shard.write(dir);
}

/// Closed whole-model fixture without the optional unclassified BF16 copies.
pub fn miniature_routed(dir: &Path) {
    miniature_gpt_oss(dir, false);
}
