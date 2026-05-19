//! Synthetic full-f32 vindex fixture for tests that need a
//! `LoadedModel.weights`-populating model directory on disk.
//!
//! The on-disk format produced here satisfies
//! `larql_vindex::load_model_weights_with_opts` — i.e. the lazy loader
//! that `LoadedModel.get_or_load_weights()` calls when a route handler
//! hits a `full_output=true` code path. The existing test helpers in
//! `tests/common/mod.rs` build a tiny in-memory `VectorIndex` for
//! features-only paths; this one is the missing piece for the
//! full-output paths, which is why `routes/walk_ffn.rs`,
//! `routes/explain.rs`, `routes/infer.rs`, the OpenAI generation
//! routes, and the streaming routes are all excluded from per-file
//! coverage gating in `coverage-policy.json`.
//!
//! Trade-off: ~2 layers × hidden=8 × intermediate=4 is small enough
//! that `build_vindex` + load takes a few hundred ms. The same
//! deterministic seed-driven weights are used as `larql-vindex/tests/
//! test_vindex.rs::make_synthetic_model` — we duplicate them here
//! because test-only items don't cross crate boundaries without an
//! explicit feature gate, and that's a heavier refactor.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use larql_vindex::ndarray::Array2;

/// On-disk synthetic vindex sized for fast load. Drop the returned
/// `Fixture` at end-of-test to remove the directory.
pub struct SyntheticVindex {
    pub dir: PathBuf,
    pub num_layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    _tmp: tempfile::TempDir,
}

/// Build a synthetic dense-FFN model in memory. Matches
/// `larql-vindex/tests/test_vindex.rs::make_synthetic_model` — keep
/// the shapes / weight values aligned so both crates can rely on the
/// same deterministic baseline.
fn make_weights() -> larql_models::ModelWeights {
    let num_layers = 2;
    let hidden = 8;
    let intermediate = 4;
    let vocab_size = 16;

    let mut tensors: HashMap<String, larql_models::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for layer in 0..num_layers {
        // FFN gate (intermediate × hidden). Diagonal-ish so the
        // synthetic forward pass is non-trivial but doesn't NaN.
        let mut gate = Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            gate[[i, i % hidden]] = 1.0 + layer as f32;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.gate_proj.weight"),
            gate.into_shared(),
        );

        // FFN up (intermediate × hidden).
        let mut up = Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            up[[i, (i + 1) % hidden]] = 0.5;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.up_proj.weight"),
            up.into_shared(),
        );

        // FFN down (hidden × intermediate).
        let mut down = Array2::<f32>::zeros((hidden, intermediate));
        for i in 0..intermediate {
            down[[i % hidden, i]] = 0.3;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.down_proj.weight"),
            down.into_shared(),
        );

        // Attention Q/K/V/O (hidden × hidden), identity. Sufficient for
        // any test that walks the attention path with a single head.
        for suffix in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let mut attn = Array2::<f32>::zeros((hidden, hidden));
            for i in 0..hidden {
                attn[[i, i]] = 1.0;
            }
            tensors.insert(
                format!("layers.{layer}.self_attn.{suffix}.weight"),
                attn.into_shared(),
            );
        }

        // Norms — unit gain.
        vectors.insert(
            format!("layers.{layer}.input_layernorm.weight"),
            vec![1.0; hidden],
        );
        vectors.insert(
            format!("layers.{layer}.post_attention_layernorm.weight"),
            vec![1.0; hidden],
        );
    }
    vectors.insert("norm.weight".into(), vec![1.0; hidden]);

    // Embeddings (vocab × hidden). Identity-on-diagonal so embed
    // lookups give a non-zero one-hot per token.
    let mut embed = Array2::<f32>::zeros((vocab_size, hidden));
    for i in 0..vocab_size {
        embed[[i, i % hidden]] = 1.0;
    }
    let embed = embed.into_shared();
    let lm_head = embed.clone();

    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "head_dim": hidden,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "vocab_size": vocab_size,
    }));

    larql_models::ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size,
        head_dim: hidden,
        num_q_heads: 1,
        num_kv_heads: 1,
        rope_base: 10000.0,
        arch,
    }
}

/// Build a complete f32 synthetic vindex on disk in a tempdir. The
/// returned `SyntheticVindex` carries the dir path and dimensions;
/// drop it at end of test to clean up.
pub fn build() -> SyntheticVindex {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let dir = tmp.path().to_path_buf();

    let weights = make_weights();
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    let intermediate = weights.intermediate_size;
    let vocab_size = weights.vocab_size;

    // WordLevel tokenizer with a tiny vocab (matches our 16-row
    // synthetic embed table). Crucial: an empty BPE tokenizer
    // encodes every prompt to 0 tokens, which causes the forward
    // pass to produce empty traces — every per-layer loop in
    // routes/explain.rs / routes/walk_ffn.rs becomes uncovered.
    // With real tokens, the predict / walk_ffn paths execute end-
    // to-end and cover the meat of the route handlers.
    let tok_json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"the":0,"capital":1,"of":2,"France":3,"is":4,"Paris":5,"a":6,"b":7,"c":8,"x":9,"y":10,"z":11},"unk_token":"x"}}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).expect("write tokenizer");
    let tokenizer =
        larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).expect("parse tokenizer");

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizer,
        "test/synthetic",
        &dir,
        // down_top_k: keep small — 5 is enough for the per-feature
        // metadata writer; ExtractLevel::All produces every tensor
        // the f32 loader expects.
        5,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut cb,
    )
    .expect("build synthetic vindex");

    SyntheticVindex {
        dir,
        num_layers,
        hidden,
        intermediate,
        vocab_size,
        _tmp: tmp,
    }
}
