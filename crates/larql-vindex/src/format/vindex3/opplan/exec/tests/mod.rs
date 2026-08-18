//! Stage A gates (V3-G5b-2): the plan executor against the
//! checkpoint-driven production forward — layer by layer.
//!
//! The two sides share **nothing but the fixture's weight values**: the
//! oracle loads the HF checkpoint through `larql-models` and runs
//! `larql-compute`'s production layers (BLAS, hooks); the executor reads
//! the encoded container through the closure-verified operand path and
//! computes with its own naive loops. Agreement is therefore a claim
//! about *semantics* — plan interpretation, operand binding, norm
//! placement, RoPE convention, residual order — not shared arithmetic.

mod controls;
mod coverage_backend_decode;
mod coverage_device;
mod coverage_experts_production;
mod decode;
mod device;
mod gemma4_refusals;
mod golden;
mod kernels;
mod parity;
mod routed;
mod seam;
mod sinks_bias;
mod smoke;
mod streaming;

use std::io::Write;
use std::path::Path;

/// Deterministic small weights: LCG over the flat index, scaled to
/// ±0.05 so activations stay in a well-conditioned range.
pub(super) fn lcg_values(n: usize, seed: u64) -> Vec<f32> {
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
pub(super) fn norm_values(n: usize, seed: u64) -> Vec<f32> {
    lcg_values(n, seed).into_iter().map(|v| 1.0 + v).collect()
}

/// Write one F32 tensor into a safetensors header/payload pair.
pub(super) struct ShardBuilder {
    header: serde_json::Map<String, serde_json::Value>,
    payload: Vec<u8>,
}

impl ShardBuilder {
    pub(super) fn new() -> Self {
        Self {
            header: serde_json::Map::new(),
            payload: Vec::new(),
        }
    }

    pub(super) fn push(&mut self, name: &str, shape: &[usize], values: &[f32]) {
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
    pub(super) fn push_bytes(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: &[u8]) {
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

    pub(super) fn write(self, dir: &Path) {
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(self.header)).unwrap();
        let mut file = std::fs::File::create(dir.join("model.safetensors")).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(&self.payload).unwrap();
    }
}

/// Fixture geometry, shared by both sides of the parity gate.
pub(super) const HIDDEN: usize = 64;
pub(super) const INTERMEDIATE: usize = 256;
pub(super) const VOCAB: usize = 128;
pub(super) const Q_HEADS: usize = 8;
pub(super) const KV_HEADS: usize = 2;
pub(super) const HEAD_DIM: usize = 8;
pub(super) const LAYERS: usize = 2;

/// A dense Llama-shaped checkpoint with real F32 weights — loadable by
/// the production path and encodable into a VINDEX3 container.
pub(super) fn dense_f32_model(dir: &Path) {
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "torch_dtype": "float32",
            "model_type": "llama",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "intermediate_size": INTERMEDIATE,
            "num_attention_heads": Q_HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        })
        .to_string(),
    )
    .unwrap();

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
        let seed = 100 + layer as u64 * 10;
        let prefix = format!("model.layers.{layer}");
        shard.push(
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[q_rows, HIDDEN],
            &lcg_values(q_rows * HIDDEN, seed),
        );
        shard.push(
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[kv_rows, HIDDEN],
            &lcg_values(kv_rows * HIDDEN, seed + 1),
        );
        shard.push(
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[kv_rows, HIDDEN],
            &lcg_values(kv_rows * HIDDEN, seed + 2),
        );
        shard.push(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN, q_rows],
            &lcg_values(HIDDEN * q_rows, seed + 3),
        );
        shard.push(
            &format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 4),
        );
        shard.push(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 5),
        );
        shard.push(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
            &lcg_values(INTERMEDIATE * HIDDEN, seed + 6),
        );
        shard.push(
            &format!("{prefix}.mlp.up_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
            &lcg_values(INTERMEDIATE * HIDDEN, seed + 7),
        );
        shard.push(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN, INTERMEDIATE],
            &lcg_values(HIDDEN * INTERMEDIATE, seed + 8),
        );
    }
    shard.write(dir);
}
