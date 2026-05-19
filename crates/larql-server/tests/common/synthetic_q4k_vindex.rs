//! Q4K-quantised synthetic vindex on disk.
//!
//! Companion to [`super::synthetic_vindex`] (the f32 fixture). The
//! generation paths in `routes/openai/chat.rs`,
//! `routes/openai/completions.rs`, `routes/stream.rs`, and
//! `routes/walk_ffn/q8k.rs` panic against an f32 vindex with
//! `attn Q4K slices missing for layer 0` from
//! `vindex/kquant_forward/cached.rs` — those code paths require Q4K
//! storage (`attn_weights_q4k.bin` + `interleaved_kquant.bin` + the
//! Q4K-mode `index.json`). This module builds exactly that.
//!
//! Two stages:
//!   1. Write a tiny Llama-shaped safetensors model to a tempdir
//!      (`write_synthetic_llama_model`).
//!   2. Stream-extract it through `larql_vindex::build_vindex_streaming`
//!      with `QuantFormat::Q4K` so the output dir is a complete Q4K
//!      vindex.
//!
//! Mirror of `larql-vindex/tests/test_vindex_to_q4k.rs::q4k_end_to_end_from_synthetic_safetensors`,
//! except we keep the safetensors stage in a tempdir and go straight
//! to Q4K (the gold-standard test does a float-then-convert two-stage
//! to also exercise `vindex_to_q4k`; we don't need that detour).

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// On-disk synthetic Q4K vindex sized for fast load. Drop the
/// returned fixture at end of test to clean up.
pub struct SyntheticQ4kVindex {
    pub dir: PathBuf,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    _tmp: tempfile::TempDir,
}

/// Write a synthetic Llama-shaped safetensors model into `dir`.
/// Mirrors `larql-vindex/tests/test_vindex_to_q4k.rs::write_synthetic_llama_model`
/// — keep the shapes / weight values aligned so we can rely on the
/// same Q4K extraction path that gold-standard test does.
fn write_synthetic_llama_model(
    model_dir: &Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    std::fs::create_dir_all(model_dir).unwrap();
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        // Deterministic per-tensor ramp — matches the gold-standard fixture.
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };
    push("model.embed_tokens.weight", vec![vocab, hidden]);
    push("model.norm.weight", vec![hidden]);
    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        push(
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        push(&format!("{lp}.input_layernorm.weight"), vec![hidden]);
        push(
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), serialized).unwrap();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

/// Build a complete Q4K vindex on disk. Returns a fixture handle;
/// drop to clean up.
pub fn build() -> SyntheticQ4kVindex {
    // Tiny dims that pad to exactly one 256-element Q4_K super-block
    // per row (hidden=8, intermediate=4) — same shapes the gold-
    // standard test uses.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let vocab = 16usize;

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let model_dir = tmp.path().join("model");
    let vindex_dir = tmp.path().join("vindex");

    let _tokenizer_unused =
        write_synthetic_llama_model(&model_dir, hidden, intermediate, num_layers, vocab);

    // Override the empty BPE tokenizer the helper wrote with a real
    // WordLevel tokenizer (12 entries) so prompts encode to non-zero
    // ids — without this, every chat/completions/stream test exits
    // early on "prompt tokenises to empty" and the generation loop
    // body stays uncovered.
    let tok_json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"the":0,"capital":1,"of":2,"France":3,"is":4,"Paris":5,"a":6,"b":7,"c":8,"x":9,"y":10,"z":11},"unk_token":"x"}}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/synthetic-q4k",
        &vindex_dir,
        4, // down_top_k
        larql_vindex::ExtractLevel::Inference,
        larql_vindex::StorageDtype::F32,
        larql_vindex::QuantFormat::Q4K,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false, // drop_gate_vectors — keep them so Vector KNN works too
        &mut cb,
    )
    .expect("build Q4K synthetic vindex");

    SyntheticQ4kVindex {
        dir: vindex_dir,
        hidden,
        intermediate,
        num_layers,
        vocab_size: vocab,
        _tmp: tmp,
    }
}
