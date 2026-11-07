//! FP8, gated-Q and shrunk-projection model fixtures.

use super::super::encode::encode_system;
use larql_models::inventory::build_inventory;
use std::path::Path;

#[allow(unused_imports)]
use super::*;

/// Write a checkpoint fixture and encode it into a VINDEX3 container in
/// one call — the shape every "open a real container" test needs.
///
/// `write_checkpoint` is one of the writers above (or a caller's own);
/// the encoded system holds that single model under `name`.
/// The dense fixture with **one projection stored as fine-grained FP8** —
/// E4M3 codes plus a `weight_scale_inv` grid, the way GLM-5.3-Flash and
/// the DeepSeek-V3 lineage ship 95.8 % of their bytes.
///
/// Only `layers.0.mlp.gate_proj` is converted. A fixture where everything
/// were FP8 could not show that the pair is bound TOGETHER and separately
/// from its neighbours — the failure that matters is a scale grid
/// reaching the wrong matrix, and a uniform fixture hides it.
///
/// The grid is deliberately **non-square with unequal tiles** (a
/// `[2, 4]` grid over `[out, in]`), because a square one cannot tell a
/// row-major scale index from a transposed one.
pub fn dense_fp8_model(dir: &Path) {
    use larql_models::quant::fp8::f32_to_e4m3;

    dense_f32_model(dir);
    // Re-open the config to add the scheme's declaration. The tensors
    // are appended to a second shard so the base fixture stays exactly
    // what every other test sees.
    let cfg_path = dir.join("config.json");
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    // STORAGE only. `activation_scheme` is deliberately absent: it names
    // an FP8 *compute* path this build does not implement, and declaring
    // it makes the plan block — which is the intended behaviour and is
    // asserted in `fp8_carriage`. A checkpoint that stores FP8 and says
    // nothing about activation quantisation is admissible; one that asks
    // for an FP8 GEMM is not.
    cfg["quantization_config"] = serde_json::json!({
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [FP8_TILE_ROWS, FP8_TILE_COLS],
        "modules_to_not_convert": [],
    });
    std::fs::write(&cfg_path, cfg.to_string()).unwrap();

    // Drop the f32 original and write the FP8 pair in its place.
    let target = "model.layers.0.mlp.gate_proj.weight";
    let (rows, cols) = (DENSE_INTERMEDIATE, DENSE_HIDDEN);
    let scale_rows = rows / FP8_TILE_ROWS;
    let scale_cols = cols / FP8_TILE_COLS;
    assert!(
        scale_rows > 1 && scale_cols > 1 && scale_rows != scale_cols,
        "the fixture's grid must be non-square and larger than 1x1, or it \
         cannot distinguish a transposed scale index"
    );

    let values = lcg_values(rows * cols, 4242);
    let scales: Vec<f32> = (0..scale_rows * scale_cols)
        .map(|i| 0.0625 * (1 + i % 5) as f32)
        .collect();
    // Quantise so the pair MEANS something: each value is divided by its
    // tile's scale before encoding, so dequantising recovers it to E4M3
    // resolution rather than to noise.
    let mut codes = vec![0u8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let s = scales[(r / FP8_TILE_ROWS) * scale_cols + c / FP8_TILE_COLS];
            codes[r * cols + c] = f32_to_e4m3(values[r * cols + c] / s);
        }
    }

    let mut shard = ShardBuilder::new();
    shard.push_bytes(target, "F8_E4M3", &[rows, cols], &codes);
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    shard.push_bytes(
        "model.layers.0.mlp.gate_proj.weight_scale_inv",
        "F32",
        &[scale_rows, scale_cols],
        &scale_bytes,
    );
    shard.write_as(dir, "model-fp8.safetensors");
    strip_tensor(dir, "model.safetensors", target);
}

/// The tile this fixture's FP8 pair uses. Unequal on purpose.
pub const FP8_TILE_ROWS: usize = DENSE_INTERMEDIATE / 2;
pub const FP8_TILE_COLS: usize = DENSE_HIDDEN / 4;

/// Remove one tensor from a written shard, rewriting the payload.
///
/// The FP8 pair REPLACES the f32 original; leaving both would let a
/// loader bind either and the test would not say which.
pub(super) fn strip_tensor(dir: &Path, shard: &str, name: &str) {
    let path = dir.join(shard);
    let raw = std::fs::read(&path).unwrap();
    let hlen = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + hlen]).unwrap();
    let body = &raw[8 + hlen..];

    let mut out = ShardBuilder::new();
    for (k, v) in header.as_object().unwrap() {
        if k == name || k == "__metadata__" {
            continue;
        }
        let off = v["data_offsets"].as_array().unwrap();
        let (a, b) = (
            off[0].as_u64().unwrap() as usize,
            off[1].as_u64().unwrap() as usize,
        );
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        out.push_bytes(k, v["dtype"].as_str().unwrap(), &shape, &body[a..b]);
    }
    std::fs::remove_file(&path).unwrap();
    out.write_as(dir, shard);
}

pub fn encode_fixture_container(
    write_checkpoint: impl FnOnce(&Path),
    checkpoint_dir: &Path,
    container_dir: &Path,
    name: &str,
) {
    write_checkpoint(checkpoint_dir);
    let inventory = build_inventory(checkpoint_dir).unwrap();
    encode_system(&[(name.to_string(), inventory)], container_dir).unwrap();
}

/// A dense model whose query projection is **fused with an attention
/// output gate** — `2 · num_heads · head_dim` rows, query and gate
/// interleaved per head.
///
/// Built for one job: proving the fused gate's layout and ordering
/// semantics. It is deliberately not a general small attention model.
/// The properties that matter are:
///
/// * **more than one query head** (8) — a single-head fixture cannot
///   distinguish a per-head interleave from contiguous halves at all,
///   because with one head the two layouts are the same bytes;
/// * `q_proj` rows `2 · 8 · 8 = 128` against an ungated 64, so the shape
///   contract has something to witness;
/// * real `q_norm`/`k_norm` weights, so `GateGetsQNorm` is a mutation
///   that can actually change a number rather than a no-op.
///
/// `model_type: "qwen3_5"` because the fused gate is judged on the Qwen
/// family; nothing else about the fixture is Qwen-specific.
pub fn gated_q_f32_model(dir: &Path) {
    gated_q_f32_model_with_geometry(dir, GATED_Q_DENSE_GEOMETRY);
}

/// The per-head geometry [`gated_q_f32_model_with_geometry`] varies.
/// Everything else (hidden size, head counts, layers) stays the dense
/// fixture's.
#[derive(Debug, Clone, Copy)]
pub struct GatedQGeometry {
    pub head_dim: usize,
    /// Written as the top-level `partial_rotary_factor` when present.
    pub partial_rotary_factor: Option<f64>,
}

/// [`gated_q_f32_model`]'s own geometry: the dense fixture's head dim,
/// full rotary (no `partial_rotary_factor` key).
pub const GATED_Q_DENSE_GEOMETRY: GatedQGeometry = GatedQGeometry {
    head_dim: DENSE_HEAD_DIM,
    partial_rotary_factor: None,
};

/// Qwen3.5-0.8B's full-attention per-head geometry: head dim 256, a
/// quarter of it rotary. The 256-wide per-head stride is what the fused
/// gate split has to get right at the real model's scale.
pub const GATED_Q_QWEN35_GEOMETRY: GatedQGeometry = GatedQGeometry {
    head_dim: QWEN35_FULL_ATTENTION_HEAD_DIM,
    partial_rotary_factor: Some(QWEN35_PARTIAL_ROTARY_FACTOR),
};

/// Qwen3.5-0.8B's `head_dim` for its full (softmax) attention layers.
pub const QWEN35_FULL_ATTENTION_HEAD_DIM: usize = 256;

/// Qwen3.5-0.8B's `partial_rotary_factor`: RoPE covers 64 of 256 dims.
pub const QWEN35_PARTIAL_ROTARY_FACTOR: f64 = 0.25;

/// [`gated_q_f32_model`] at a chosen per-head geometry. The tensors are
/// sized from `geometry.head_dim`, so the q_proj interleave stride, the
/// `q_norm`/`k_norm` widths and `o_proj`'s input all follow it.
pub fn gated_q_f32_model_with_geometry(dir: &Path, geometry: GatedQGeometry) {
    let head_dim = geometry.head_dim;
    let mut config = serde_json::json!({
        "architectures": ["Qwen3_5ForCausalLM"],
        "torch_dtype": "float32",
        "model_type": "qwen3_5",
        "hidden_size": DENSE_HIDDEN,
        "num_hidden_layers": DENSE_LAYERS,
        "intermediate_size": DENSE_INTERMEDIATE,
        "num_attention_heads": DENSE_Q_HEADS,
        "num_key_value_heads": DENSE_KV_HEADS,
        "head_dim": head_dim,
        "vocab_size": DENSE_VOCAB,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attn_output_gate": true
    });
    if let Some(factor) = geometry.partial_rotary_factor {
        config["partial_rotary_factor"] = serde_json::json!(factor);
    }
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    gated_q_shards(dir, head_dim);
}

fn gated_q_shards(dir: &Path, head_dim: usize) {
    let q_rows = DENSE_Q_HEADS * head_dim;
    let kv_rows = DENSE_KV_HEADS * head_dim;
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
    for layer in 0..DENSE_LAYERS {
        let seed = 100 + layer as u64 * 10;
        let prefix = format!("model.layers.{layer}");
        // The fused projection: DOUBLE width.
        shard.push(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[q_rows * 2, DENSE_HIDDEN],
            &lcg_values(q_rows * 2 * DENSE_HIDDEN, seed),
        );
        shard.push(
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[kv_rows, DENSE_HIDDEN],
            &lcg_values(kv_rows * DENSE_HIDDEN, seed + 1),
        );
        shard.push(
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[kv_rows, DENSE_HIDDEN],
            &lcg_values(kv_rows * DENSE_HIDDEN, seed + 2),
        );
        // `o_proj` is sized by the ATTENTION width, not the projection's.
        shard.push(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[DENSE_HIDDEN, q_rows],
            &lcg_values(DENSE_HIDDEN * q_rows, seed + 3),
        );
        shard.push(
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[head_dim],
            &norm_values(head_dim, seed + 9),
        );
        shard.push(
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[head_dim],
            &norm_values(head_dim, seed + 10),
        );
        shard.push(
            &format!("{prefix}.input_layernorm.weight"),
            &[DENSE_HIDDEN],
            &norm_values(DENSE_HIDDEN, seed + 4),
        );
        shard.push(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[DENSE_HIDDEN],
            &norm_values(DENSE_HIDDEN, seed + 5),
        );
        shard.push(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[DENSE_INTERMEDIATE, DENSE_HIDDEN],
            &lcg_values(DENSE_INTERMEDIATE * DENSE_HIDDEN, seed + 6),
        );
        shard.push(
            &format!("{prefix}.mlp.up_proj.weight"),
            &[DENSE_INTERMEDIATE, DENSE_HIDDEN],
            &lcg_values(DENSE_INTERMEDIATE * DENSE_HIDDEN, seed + 7),
        );
        shard.push(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[DENSE_HIDDEN, DENSE_INTERMEDIATE],
            &lcg_values(DENSE_HIDDEN * DENSE_INTERMEDIATE, seed + 8),
        );
    }
    shard.write(dir);
}

/// Rewrite [`gated_q_f32_model`]'s shards with an ORDINARY-width query
/// projection, leaving `attn_output_gate: true` declared.
///
/// The negative control for the gate's shape contract: a checkpoint that
/// claims a fused gate but ships no rows to hold it.
pub fn shrink_q_proj_to_ungated_width(dir: &Path) {
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
    for layer in 0..DENSE_LAYERS {
        let seed = 100 + layer as u64 * 10;
        let prefix = format!("model.layers.{layer}");
        shard.push(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[q_rows, DENSE_HIDDEN],
            &lcg_values(q_rows * DENSE_HIDDEN, seed),
        );
        for (name, rows, s) in [("k_proj", kv_rows, seed + 1), ("v_proj", kv_rows, seed + 2)] {
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
        for (name, s) in [("q_norm", seed + 9), ("k_norm", seed + 10)] {
            shard.push(
                &format!("{prefix}.self_attn.{name}.weight"),
                &[DENSE_HEAD_DIM],
                &norm_values(DENSE_HEAD_DIM, s),
            );
        }
        for (name, s) in [
            ("input_layernorm", seed + 4),
            ("post_attention_layernorm", seed + 5),
        ] {
            shard.push(
                &format!("{prefix}.{name}.weight"),
                &[DENSE_HIDDEN],
                &norm_values(DENSE_HIDDEN, s),
            );
        }
        for (name, rows, cols, s) in [
            ("gate_proj", DENSE_INTERMEDIATE, DENSE_HIDDEN, seed + 6),
            ("up_proj", DENSE_INTERMEDIATE, DENSE_HIDDEN, seed + 7),
            ("down_proj", DENSE_HIDDEN, DENSE_INTERMEDIATE, seed + 8),
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
