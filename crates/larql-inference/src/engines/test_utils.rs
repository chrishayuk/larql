//! Synthetic test fixtures for engine and layer-graph unit tests.
//!
//! Three helpers:
//! - `make_test_weights()` — fully functional 2-layer ModelWeights (no disk I/O)
//! - `make_test_vindex(weights)` — in-memory VectorIndex with random gate vectors
//! - `make_test_tokenizer(vocab_size)` — WordLevel tokenizer mapping token N to "[N]"
//!
//! Dimensions: vocab=32, hidden=16, intermediate=32, 2 q-heads, 1 kv-head,
//! head_dim=8, 2 layers. Forward pass ≈ 10 ms on CPU.

use larql_models::{detect_from_json, ModelWeights, WeightArray};
use ndarray::Array2;
use std::collections::HashMap;

/// Build a synthetic `ModelWeights` with all tensors populated.
/// Uses `TinyModelArch` key conventions (e.g. `"0.attn.q_proj.weight"`).
pub fn make_test_weights() -> ModelWeights {
    make_test_weights_with(SyntheticDims::tiny())
}

/// Caller-supplied dimensions for [`make_test_weights_with`].
/// Useful when you need the same synthetic-weights factory at a
/// larger shape — e.g. for benchmarking the attention-service
/// runner under realistic dims, without paying for a real
/// model checkpoint.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticDims {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
}

impl SyntheticDims {
    /// Tiny: vocab=32, hidden=16, 2 layers. Default for unit tests.
    /// Forward pass < 1 ms on CPU.
    pub fn tiny() -> Self {
        Self {
            vocab: 32,
            hidden: 16,
            intermediate: 32,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            num_layers: 2,
        }
    }

    /// Gemma-3-4B-shaped: vocab=262144, hidden=2560, 33 layers,
    /// num_q=8, num_kv=4, head_dim=320, intermediate=10240.
    /// Useful for realistic-shape benches; weight build is
    /// noticeable (~1 GB f32) so prefer caching with `OnceLock`.
    pub fn gemma_3_4b_like() -> Self {
        Self {
            vocab: 32,
            hidden: 2560,
            intermediate: 10240,
            num_q_heads: 8,
            num_kv_heads: 4,
            head_dim: 320,
            num_layers: 33,
        }
    }

    /// Smaller Gemma-shaped fixture for benches that don't need the
    /// full 33 layers — same hidden/heads but only 4 layers.
    /// Build cost ≈ 100 MB f32; useful for parametric sweeps.
    pub fn gemma_3_4b_4layer() -> Self {
        Self {
            vocab: 32,
            hidden: 2560,
            intermediate: 10240,
            num_q_heads: 8,
            num_kv_heads: 4,
            head_dim: 320,
            num_layers: 4,
        }
    }
}

/// Build synthetic weights at caller-supplied dimensions. The body
/// is identical to [`make_test_weights`] but parameterised. Used by
/// the `larql-server` benchmark harness to reach realistic Gemma
/// shapes without a real checkpoint.
pub fn make_test_weights_with(dims: SyntheticDims) -> ModelWeights {
    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": dims.hidden,
        "num_hidden_layers": dims.num_layers,
        "intermediate_size": dims.intermediate,
        "head_dim": dims.head_dim,
        "num_attention_heads": dims.num_q_heads,
        "num_key_value_heads": dims.num_kv_heads,
        "vocab_size": dims.vocab,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut rng_state = 0xdeadbeef_u64;

    // LCG giving values in [-scale, +scale]
    let mut rand_mat = |rows: usize, cols: usize, scale: f32| -> WeightArray {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (rng_state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data)
            .unwrap()
            .into_shared()
    };

    let hidden = dims.hidden;
    // Embed + lm_head
    let embed = rand_mat(dims.vocab, hidden, 0.1);
    let lm_head = rand_mat(dims.vocab, hidden, 0.1);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Final norm (ones → valid unweighted RMSNorm fallback)
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; hidden]);

    let q_dim = dims.num_q_heads * dims.head_dim;
    let kv_dim = dims.num_kv_heads * dims.head_dim;
    let inter = dims.intermediate;

    for layer in 0..dims.num_layers {
        // Attention projections
        tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, hidden, 0.1));
        tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, hidden, 0.1));
        tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, hidden, 0.1));
        tensors.insert(arch.attn_o_key(layer), rand_mat(hidden, q_dim, 0.1));
        // FFN — missing tensors cause panic, so always provide them
        tensors.insert(arch.ffn_gate_key(layer), rand_mat(inter, hidden, 0.1));
        tensors.insert(arch.ffn_up_key(layer), rand_mat(inter, hidden, 0.1));
        tensors.insert(arch.ffn_down_key(layer), rand_mat(hidden, inter, 0.1));
        // Layer norms
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; hidden]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; hidden]);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        arch,
        num_layers: dims.num_layers,
        hidden_size: hidden,
        intermediate_size: inter,
        vocab_size: dims.vocab,
        head_dim: dims.head_dim,
        num_q_heads: dims.num_q_heads,
        num_kv_heads: dims.num_kv_heads,
        rope_base: 10_000.0,
    }
}

/// Build an in-memory `VectorIndex` with random gate vectors per layer.
/// The VectorIndex has no Q4K or interleaved data — `predict_honest` falls
/// through to the CPU path, and `WalkFfn` routes through the sparse fallback
/// that uses `weights.tensors`.
pub fn make_test_vindex(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    let n_features = weights.intermediate_size;
    let hidden = weights.hidden_size;

    // Each layer gets an independent LCG seed so gate matrices are distinct.
    let gate_vectors: Vec<Option<Array2<f32>>> = (0..weights.num_layers)
        .map(|l| {
            let mut state = 0xabcdef_u64.wrapping_add(l as u64 * 0x9e3779b97f4a7c15);
            let data: Vec<f32> = (0..n_features * hidden)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state as u32) as f32 / u32::MAX as f32 * 0.1 - 0.05
                })
                .collect();
            Some(Array2::from_shape_vec((n_features, hidden), data).unwrap())
        })
        .collect();

    let down_meta = vec![None; weights.num_layers];
    larql_vindex::VectorIndex::new(gate_vectors, down_meta, weights.num_layers, hidden)
}

/// Build a `tokenizers::Tokenizer` with a vocabulary of `vocab_size` tokens.
/// Token N decodes to `"[N]"`, so token IDs from `make_test_weights()` all
/// decode to valid (if meaningless) strings.
pub fn make_test_tokenizer(vocab_size: usize) -> tokenizers::Tokenizer {
    // WordLevel::builder().vocab() requires an AHashMap.
    // Build a simple BPE-less tokenizer via JSON serialization instead.
    let mut vocab_json = serde_json::Map::new();
    for i in 0..vocab_size as u64 {
        vocab_json.insert(format!("[{i}]"), serde_json::Value::Number(i.into()));
    }
    // Add UNK token at the end
    vocab_json.insert("[UNK]".into(), serde_json::Value::Number(vocab_size.into()));

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab_json,
            "unk_token": "[UNK]"
        }
    });

    let bytes = serde_json::to_vec(&tokenizer_json).expect("JSON serialization failed");
    tokenizers::Tokenizer::from_bytes(&bytes).expect("synthetic tokenizer construction failed")
}

/// All three synthetic fixtures bundled together. Build once per test module
/// via `OnceLock`; each field is cheaply borrowed.
pub struct TestFixtures {
    pub weights: ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub index: larql_vindex::VectorIndex,
}

impl TestFixtures {
    pub fn build() -> Self {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_vindex(&weights);
        Self {
            weights,
            tokenizer,
            index,
        }
    }
}
