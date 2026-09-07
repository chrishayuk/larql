//! Miniature whole-model checkpoints, public so integration tests in
//! sibling crates can build a real VINDEX3 container without
//! duplicating the frozen geometry.
//!
//! Two fixtures, both deterministic (LCG over the flat index — no RNG
//! state, no clock):
//!
//! - [`dense_f32_model`]: a dense Llama-shaped checkpoint. The plain
//!   anatomy — two norms per layer, RoPE everywhere, no gates.
//! - [`miniature_glimmer`]: the judged-semantics miniature (sigmoid
//!   attention gate, four-norm placement, per-layer sliding/full +
//!   RoPE/NoPE split, softcapping) with deliberately awkward
//!   dimensions so no broadcasting accident can hide.
//!
//! The executor's own oracle tests live in
//! `opplan/exec/tests` and consume these same writers; a sibling crate
//! building a container from one of them is therefore executing the
//! exact program those parity gates certify.
//!
//! This is a sibling of [`super::test_support`] (conformance fixture A),
//! which covers a single routed-MoE *layer bank*; the writers here emit
//! *complete model programs* — embedding, attention stack, output head —
//! the shape a [`DecodeSession`](super::opplan::exec::decode::DecodeSession)
//! needs.

use std::io::Write;
use std::path::Path;

use larql_models::inventory::build_inventory;

use super::encode::encode_system;

/// Deterministic small weights: LCG over the flat index, scaled to
/// ±0.05 so activations stay in a well-conditioned range.
/// One RFC 1950 stream over the row-major little-endian bf16 image of
/// `values` — the only `BF16_ZLIB` encoder in the tree, and it lives with
/// the fixtures deliberately: the representation contract is decode-only,
/// and a compile-path encoder is a later rung's claim, not this one's.
pub fn encode_bf16_zlib(values: &[f32]) -> Vec<u8> {
    use std::io::Write;
    let image = larql_models::quant::half::encode_bf16(values);
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
    encoder
        .write_all(&image)
        .expect("a Vec never refuses a write");
    encoder.finish().expect("a Vec never refuses a write")
}

pub fn lcg_values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 33) as f64) / ((1u64 << 31) as f64);
            ((unit - 0.5) * 0.1) as f32
        })
        .collect()
}

/// Norm weights near 1.0 (never all-zero: that would mask norm bugs).
pub fn norm_values(n: usize, seed: u64) -> Vec<f32> {
    lcg_values(n, seed).into_iter().map(|v| 1.0 + v).collect()
}

/// Write one F32 tensor into a safetensors header/payload pair.
pub struct ShardBuilder {
    header: serde_json::Map<String, serde_json::Value>,
    payload: Vec<u8>,
}

impl Default for ShardBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardBuilder {
    pub fn new() -> Self {
        Self {
            header: serde_json::Map::new(),
            payload: Vec::new(),
        }
    }

    pub fn push(&mut self, name: &str, shape: &[usize], values: &[f32]) {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let start = self.payload.len();
        for v in values {
            self.payload.extend_from_slice(&v.to_le_bytes());
        }
        self.header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, self.payload.len()],
            }),
        );
    }

    /// Write one tensor of any dtype from raw little-endian bytes — for
    /// packed MXFP4 (`U8`) expert banks.
    pub fn push_bytes(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: &[u8]) {
        let start = self.payload.len();
        self.payload.extend_from_slice(bytes);
        self.header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, self.payload.len()],
            }),
        );
    }

    pub fn write(self, dir: &Path) {
        self.write_as(dir, "model.safetensors");
    }

    /// Write under a chosen filename — for a fixture that ships more than
    /// one shard.
    pub fn write_as(self, dir: &Path, name: &str) {
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(self.header)).unwrap();
        let mut file = std::fs::File::create(dir.join(name)).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&self.payload).unwrap();
    }
}

// ── Dense Llama-shaped fixture geometry ──
pub const DENSE_HIDDEN: usize = 64;
pub const DENSE_INTERMEDIATE: usize = 256;
pub const DENSE_VOCAB: usize = 128;
pub const DENSE_Q_HEADS: usize = 8;
pub const DENSE_KV_HEADS: usize = 2;
pub const DENSE_HEAD_DIM: usize = 8;
pub const DENSE_LAYERS: usize = 2;

/// A dense Llama-shaped checkpoint with real F32 weights — loadable by
/// the production path and encodable into a VINDEX3 container.
pub fn dense_f32_model(dir: &Path) {
    dense_f32_model_with(dir, HeadStorage::Separate);
}

/// How a checkpoint stores its output head.
///
/// A logical role and a serialised object are different things, and this
/// is the axis on which they come apart. Qwen3-4B is the real instance:
/// it declares `tie_word_embeddings: true` and ships no `lm_head.weight`
/// at all, while Qwen3-0.6B and Qwen3-1.7B declare the same flag and ship
/// the tensor anyway. Same family, same declaration, different
/// realisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadStorage {
    /// A distinct `lm_head.weight` group — its own object in the
    /// container.
    Separate,
    /// `tie_word_embeddings: true` and no head tensor. The logical role
    /// still exists; nothing is serialised for it.
    Tied,
}

/// The dense Llama-shaped checkpoint, with the head stored either way.
pub fn dense_f32_model_with(dir: &Path, head: HeadStorage) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "float32",
            "model_type": "llama",
            "tie_word_embeddings": head == HeadStorage::Tied,
            "hidden_size": DENSE_HIDDEN,
            "num_hidden_layers": DENSE_LAYERS,
            "intermediate_size": DENSE_INTERMEDIATE,
            "num_attention_heads": DENSE_Q_HEADS,
            "num_key_value_heads": DENSE_KV_HEADS,
            "head_dim": DENSE_HEAD_DIM,
            "vocab_size": DENSE_VOCAB,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
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
    if head == HeadStorage::Separate {
        shard.push(
            "lm_head.weight",
            &[DENSE_VOCAB, DENSE_HIDDEN],
            &lcg_values(DENSE_VOCAB * DENSE_HIDDEN, 3),
        );
    }
    for layer in 0..DENSE_LAYERS {
        let seed = 100 + layer as u64 * 10;
        let prefix = format!("model.layers.{layer}");
        shard.push(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[q_rows, DENSE_HIDDEN],
            &lcg_values(q_rows * DENSE_HIDDEN, seed),
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
        shard.push(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[DENSE_HIDDEN, q_rows],
            &lcg_values(DENSE_HIDDEN * q_rows, seed + 3),
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

// ── The awkward miniature-Glimmer geometry, shared only as *numbers* ──
pub const G_HIDDEN: usize = 12;
pub const G_Q_HEADS: usize = 3;
pub const G_KV_HEADS: usize = 1;
pub const G_HEAD_DIM: usize = 4;
pub const G_FFN: usize = 20;
pub const G_VOCAB: usize = 29;
pub const G_LAYERS: usize = 2;
pub const G_WINDOW: usize = 3;
pub const G_TOKENS: [u32; 5] = [3, 17, 28, 0, 11];

/// Optional attention operands the miniature can carry (A-9.1): Q/K/V/O
/// projection biases (with `attention_bias: true` declared) and
/// per-query-head sink logits. `perturb` names one of those tensors by
/// suffix; the writer scales it by [`PERTURB_GAIN`], so a test can ask
/// whether that single operand is load-bearing.
#[derive(Default, Clone, Copy)]
pub struct MiniatureExtras {
    pub attention_bias: bool,
    pub sinks: bool,
    pub perturb: Option<&'static str>,
}

/// Multiplier applied to a perturbed extra operand.
pub const PERTURB_GAIN: f32 = 3.0;

/// The four projection-bias operands, by layer-relative suffix.
pub const BIAS_SUFFIXES: [&str; 4] = [
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.bias",
];
pub const SINKS_SUFFIX: &str = "self_attn.sinks";

/// The two-layer miniature Glimmer checkpoint (F32, judged family):
///
/// ```text
/// layer 0: Sliding(3) + RoPE(500000) + gated attention
/// layer 1: Full     + NoPE          + gated attention
/// ```
pub fn miniature_glimmer(dir: &Path) {
    miniature_glimmer_with(dir, MiniatureExtras::default());
}

/// The miniature with the A-9.1 extras.
pub fn miniature_glimmer_with(dir: &Path, extras: MiniatureExtras) {
    let mut config = serde_json::json!({
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "torch_dtype": "float32",
            "model_type": "muse_glimmer_text",
            "hidden_size": G_HIDDEN,
            "num_hidden_layers": G_LAYERS,
            "intermediate_size": G_FFN,
            "num_attention_heads": G_Q_HEADS,
            "num_key_value_heads": G_KV_HEADS,
            "head_dim": G_HEAD_DIM,
            "vocab_size": G_VOCAB,
            "sliding_window": G_WINDOW,
            "rms_norm_eps": 1e-5,
            "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" },
            "layer_types": ["sliding_attention", "full_attention"],
            "layer_rope_theta": [500000.0, 0.0],
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.196,
            "post_norm_eps": 1e-8,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 20.0
    });
    if extras.attention_bias {
        config["attention_bias"] = serde_json::json!(true);
    }
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let q_rows = G_Q_HEADS * G_HEAD_DIM;
    let kv_rows = G_KV_HEADS * G_HEAD_DIM;
    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[G_VOCAB, G_HIDDEN],
        &lcg_values(G_VOCAB * G_HIDDEN, 1),
    );
    shard.push("model.norm.weight", &[G_HIDDEN], &norm_values(G_HIDDEN, 2));
    shard.push(
        "lm_head.weight",
        &[G_VOCAB, G_HIDDEN],
        &lcg_values(G_VOCAB * G_HIDDEN, 3),
    );
    for layer in 0..G_LAYERS {
        let seed = 900 + layer as u64 * 30;
        let prefix = format!("model.layers.{layer}");
        for (suffix, shape, values) in [
            (
                "self_attn.q_proj.weight",
                vec![q_rows, G_HIDDEN],
                lcg_values(q_rows * G_HIDDEN, seed),
            ),
            (
                "self_attn.k_proj.weight",
                vec![kv_rows, G_HIDDEN],
                lcg_values(kv_rows * G_HIDDEN, seed + 1),
            ),
            (
                "self_attn.v_proj.weight",
                vec![kv_rows, G_HIDDEN],
                lcg_values(kv_rows * G_HIDDEN, seed + 2),
            ),
            (
                "self_attn.o_proj.weight",
                vec![G_HIDDEN, q_rows],
                lcg_values(G_HIDDEN * q_rows, seed + 3),
            ),
            (
                "self_attn.gate_proj.weight",
                vec![q_rows, G_HIDDEN],
                lcg_values(q_rows * G_HIDDEN, seed + 4),
            ),
            (
                "input_layernorm.weight",
                vec![G_HIDDEN],
                norm_values(G_HIDDEN, seed + 5),
            ),
            (
                "post_attention_layernorm.weight",
                vec![G_HIDDEN],
                norm_values(G_HIDDEN, seed + 6),
            ),
            (
                "pre_feedforward_layernorm.weight",
                vec![G_HIDDEN],
                norm_values(G_HIDDEN, seed + 7),
            ),
            (
                "post_feedforward_layernorm.weight",
                vec![G_HIDDEN],
                norm_values(G_HIDDEN, seed + 8),
            ),
            (
                "mlp.gate_proj.weight",
                vec![G_FFN, G_HIDDEN],
                lcg_values(G_FFN * G_HIDDEN, seed + 9),
            ),
            (
                "mlp.up_proj.weight",
                vec![G_FFN, G_HIDDEN],
                lcg_values(G_FFN * G_HIDDEN, seed + 10),
            ),
            (
                "mlp.down_proj.weight",
                vec![G_HIDDEN, G_FFN],
                lcg_values(G_HIDDEN * G_FFN, seed + 11),
            ),
        ] {
            shard.push(&format!("{prefix}.{suffix}"), &shape, &values);
        }
        // Extras: values well away from zero so an unapplied bias or sink
        // is a visible absence, not a rounding-level one.
        let mut extra = |suffix: &str, len: usize, seed: u64| {
            let mut values: Vec<f32> = lcg_values(len, seed).into_iter().map(|v| v * 4.0).collect();
            if extras.perturb == Some(suffix) {
                for v in &mut values {
                    *v *= PERTURB_GAIN;
                }
            }
            shard.push(&format!("{prefix}.{suffix}"), &[len], &values);
        };
        if extras.attention_bias {
            extra(BIAS_SUFFIXES[0], q_rows, seed + 12);
            extra(BIAS_SUFFIXES[1], kv_rows, seed + 13);
            extra(BIAS_SUFFIXES[2], kv_rows, seed + 14);
            extra(BIAS_SUFFIXES[3], G_HIDDEN, seed + 15);
        }
        if extras.sinks {
            extra(SINKS_SUFFIX, G_Q_HEADS, seed + 16);
        }
    }
    shard.write(dir);
}

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
fn strip_tensor(dir: &Path, shard: &str, name: &str) {
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
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["Qwen3_5ForCausalLM"],
            "torch_dtype": "float32",
            "model_type": "qwen3_5",
            "hidden_size": DENSE_HIDDEN,
            "num_hidden_layers": DENSE_LAYERS,
            "intermediate_size": DENSE_INTERMEDIATE,
            "num_attention_heads": DENSE_Q_HEADS,
            "num_key_value_heads": DENSE_KV_HEADS,
            "head_dim": DENSE_HEAD_DIM,
            "vocab_size": DENSE_VOCAB,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "attn_output_gate": true
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
            &[DENSE_HEAD_DIM],
            &norm_values(DENSE_HEAD_DIM, seed + 9),
        );
        shard.push(
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[DENSE_HEAD_DIM],
            &norm_values(DENSE_HEAD_DIM, seed + 10),
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
    // Linear-attention geometry: 2*Hk*Dk + Hv*Dv channels.
    const KEY_HEADS: usize = 2;
    const VALUE_HEADS: usize = 4;
    const LIN_DIM: usize = 8;
    const CONV_KERNEL: usize = 4;
    let qkv = 2 * KEY_HEADS * LIN_DIM + VALUE_HEADS * LIN_DIM;
    let value_width = VALUE_HEADS * LIN_DIM;
    let layers = 4;

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
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "full_attention_interval": 4,
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
        if layer % 4 == 3 {
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
