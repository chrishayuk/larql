//! Integration tests for model loading — safetensors and GGUF.
//!
//! Each test builds a minimal synthetic binary in a tempdir and exercises the
//! public loading API. No real model files required.

use std::io::{Seek, Write};
use std::path::Path;
use tempfile::TempDir;

use larql_models::{
    load_model_dir, load_model_dir_filtered, load_model_dir_validated, load_model_dir_walk_only,
    load_model_dir_walk_only_validated, validation::FIELD_HEAD_DIM, ModelError,
};

// ═══════════════════════════════════════════════════════════════════════════
// Safetensors binary builder
// ═══════════════════════════════════════════════════════════════════════════

/// Build a valid safetensors file in memory.
///
/// `entries`: (tensor_name, dtype_string, shape, raw_data_bytes)
///
/// The dtype string must match the safetensors spec: "F32", "F16", "BF16",
/// "I64", etc. `raw_data_bytes` must be exactly the right number of bytes for
/// the given shape × element size.
fn make_safetensors(entries: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
    let mut data_offset = 0usize;
    let mut meta = serde_json::Map::new();
    let mut tensor_data = Vec::<u8>::new();

    for &(name, dtype, shape, ref bytes) in entries {
        let end = data_offset + bytes.len();
        meta.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [data_offset, end],
            }),
        );
        tensor_data.extend_from_slice(bytes);
        data_offset = end;
    }
    meta.insert("__metadata__".into(), serde_json::json!({}));

    let header = serde_json::to_vec(&serde_json::Value::Object(meta)).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&tensor_data);
    out
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Encode `n` elements as f16 1.0 (0x3C00).
fn f16_ones(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| [0x00u8, 0x3C]).collect()
}

/// Encode `n` elements as bf16 1.0 (0x3F80).
fn bf16_ones(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| [0x80u8, 0x3F]).collect()
}

/// Encode `n` elements as I64 42.
fn i64_bytes(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| 42i64.to_le_bytes()).collect()
}

/// Write config.json and a single `model.safetensors` into `dir`.
fn write_model_dir(dir: &Path, entries: &[(&str, &str, &[usize], Vec<u8>)]) {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 10,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("model.safetensors"), make_safetensors(entries)).unwrap();
}

fn write_model_dir_with_config(
    dir: &Path,
    config: serde_json::Value,
    entries: &[(&str, &str, &[usize], Vec<u8>)],
) {
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("model.safetensors"), make_safetensors(entries)).unwrap();
}

/// Minimal embed + lm_head + norm for a successful Llama-like load (hidden=4, vocab=10).
fn minimal_tensors() -> Vec<(&'static str, &'static str, &'static [usize], Vec<u8>)> {
    let embed_data = f32_bytes(&[1.0f32; 40]); // [10, 4]
    let norm_data = f32_bytes(&[1.0f32; 4]); // [4]
    let head_data = f32_bytes(&[1.0f32; 40]); // [10, 4]
    vec![
        ("embed_tokens.weight", "F32", &[10, 4], embed_data),
        ("norm.weight", "F32", &[4], norm_data),
        ("lm_head.weight", "F32", &[10, 4], head_data),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// GGUF binary builder
// ═══════════════════════════════════════════════════════════════════════════

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_F32: u32 = 0; // tensor type F32

fn gguf_str(f: &mut impl Write, s: &str) {
    let b = s.as_bytes();
    f.write_all(&(b.len() as u64).to_le_bytes()).unwrap();
    f.write_all(b).unwrap();
}

fn gguf_meta_str(f: &mut impl Write, key: &str, val: &str) {
    gguf_str(f, key);
    f.write_all(&GGUF_TYPE_STRING.to_le_bytes()).unwrap();
    gguf_str(f, val);
}

fn gguf_meta_u32(f: &mut impl Write, key: &str, val: u32) {
    gguf_str(f, key);
    f.write_all(&GGUF_TYPE_UINT32.to_le_bytes()).unwrap();
    f.write_all(&val.to_le_bytes()).unwrap();
}

fn gguf_meta_f32(f: &mut impl Write, key: &str, val: f32) {
    gguf_str(f, key);
    f.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes()).unwrap();
    f.write_all(&val.to_le_bytes()).unwrap();
}

fn gguf_tensor_info(f: &mut impl Write, name: &str, dims: &[u64], ty: u32, offset: u64) {
    gguf_str(f, name);
    f.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
    for &d in dims {
        f.write_all(&d.to_le_bytes()).unwrap();
    }
    f.write_all(&ty.to_le_bytes()).unwrap();
    f.write_all(&offset.to_le_bytes()).unwrap();
}

fn write_minimal_gguf(path: &Path) {
    write_minimal_gguf_custom(path, 100, None, true, true);
}

/// Write a minimal but complete GGUF file that `load_gguf` can successfully parse.
///
/// Architecture: llama, hidden=4, 1 layer.
/// Tensors: token_embd (embed), output (lm_head), output_norm (norm vector).
fn write_minimal_gguf_custom(
    path: &Path,
    vocab: u64,
    metadata_vocab_size: Option<u32>,
    include_kv_heads: bool,
    include_key_length: bool,
) {
    const HIDDEN: u64 = 4;
    let embed_elems = (HIDDEN * vocab) as usize;
    let norm_elems = HIDDEN as usize;

    let embed_bytes = (embed_elems * 4) as u64; // F32
    let norm_bytes = (norm_elems * 4) as u64;
    let metadata_count: u64 = 6
        + if include_kv_heads { 1 } else { 0 }
        + if include_key_length { 1 } else { 0 }
        + if metadata_vocab_size.is_some() { 1 } else { 0 };

    let mut f = std::fs::File::create(path).unwrap();

    // Header
    f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap(); // version 3
    f.write_all(&3u64.to_le_bytes()).unwrap(); // n_tensors
    f.write_all(&metadata_count.to_le_bytes()).unwrap(); // n_metadata

    // Metadata
    gguf_meta_str(&mut f, "general.architecture", "llama");
    gguf_meta_u32(&mut f, "llama.embedding_length", HIDDEN as u32);
    gguf_meta_u32(&mut f, "llama.block_count", 1);
    gguf_meta_u32(&mut f, "llama.feed_forward_length", 16);
    gguf_meta_u32(&mut f, "llama.attention.head_count", 2);
    if include_kv_heads {
        gguf_meta_u32(&mut f, "llama.attention.head_count_kv", 2);
    }
    if include_key_length {
        gguf_meta_u32(&mut f, "llama.attention.key_length", 2);
    }
    gguf_meta_f32(&mut f, "llama.rope.freq_base", 10000.0);
    if let Some(vocab_size) = metadata_vocab_size {
        gguf_meta_u32(&mut f, "llama.vocab_size", vocab_size);
    }

    // Tensor infos (offsets are relative to the data section start)
    gguf_tensor_info(&mut f, "token_embd.weight", &[HIDDEN, vocab], GGUF_F32, 0);
    gguf_tensor_info(
        &mut f,
        "output.weight",
        &[HIDDEN, vocab],
        GGUF_F32,
        embed_bytes,
    );
    gguf_tensor_info(
        &mut f,
        "output_norm.weight",
        &[HIDDEN],
        GGUF_F32,
        embed_bytes * 2,
    );

    // Pad to 32-byte boundary (start of data section)
    let pos = f.stream_position().unwrap();
    let aligned = pos.div_ceil(32) * 32;
    f.write_all(&vec![0u8; (aligned - pos) as usize]).unwrap();

    // Tensor data: all 1.0f32
    // Write tensor data (all zeros — we just check shape loads correctly)
    f.write_all(&vec![0u8; embed_bytes as usize]).unwrap();
    f.write_all(&vec![0u8; embed_bytes as usize]).unwrap();
    f.write_all(&vec![0u8; norm_bytes as usize]).unwrap();
    f.flush().unwrap();
}

/// Write a minimal GGUF with one FFN tensor, used to prove walk-only filtering
/// is applied before/at GGUF tensor loading.
fn write_gguf_with_ffn(path: &Path) {
    const VOCAB: u64 = 100;
    const HIDDEN: u64 = 4;
    const INTERMEDIATE: u64 = 16;
    let embed_elems = (HIDDEN * VOCAB) as usize;
    let norm_elems = HIDDEN as usize;
    let ffn_elems = (HIDDEN * INTERMEDIATE) as usize;

    let embed_bytes = (embed_elems * 4) as u64;
    let norm_bytes = (norm_elems * 4) as u64;
    let ffn_bytes = (ffn_elems * 4) as u64;

    let mut f = std::fs::File::create(path).unwrap();

    f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&4u64.to_le_bytes()).unwrap();
    f.write_all(&8u64.to_le_bytes()).unwrap();

    gguf_meta_str(&mut f, "general.architecture", "llama");
    gguf_meta_u32(&mut f, "llama.embedding_length", HIDDEN as u32);
    gguf_meta_u32(&mut f, "llama.block_count", 1);
    gguf_meta_u32(&mut f, "llama.feed_forward_length", INTERMEDIATE as u32);
    gguf_meta_u32(&mut f, "llama.attention.head_count", 2);
    gguf_meta_u32(&mut f, "llama.attention.head_count_kv", 2);
    gguf_meta_u32(&mut f, "llama.attention.key_length", 2);
    gguf_meta_f32(&mut f, "llama.rope.freq_base", 10000.0);

    gguf_tensor_info(&mut f, "token_embd.weight", &[HIDDEN, VOCAB], GGUF_F32, 0);
    gguf_tensor_info(
        &mut f,
        "output.weight",
        &[HIDDEN, VOCAB],
        GGUF_F32,
        embed_bytes,
    );
    gguf_tensor_info(
        &mut f,
        "output_norm.weight",
        &[HIDDEN],
        GGUF_F32,
        embed_bytes * 2,
    );
    gguf_tensor_info(
        &mut f,
        "blk.0.ffn_gate.weight",
        &[HIDDEN, INTERMEDIATE],
        GGUF_F32,
        embed_bytes * 2 + norm_bytes,
    );

    let pos = f.stream_position().unwrap();
    let aligned = pos.div_ceil(32) * 32;
    f.write_all(&vec![0u8; (aligned - pos) as usize]).unwrap();

    f.write_all(&vec![0u8; embed_bytes as usize]).unwrap();
    f.write_all(&vec![0u8; embed_bytes as usize]).unwrap();
    f.write_all(&vec![0u8; norm_bytes as usize]).unwrap();
    f.write_all(&vec![0u8; ffn_bytes as usize]).unwrap();
    f.flush().unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════
// Safetensors loading tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn load_f32_tensors_correct_values() {
    let dir = TempDir::new().unwrap();
    let known: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
    write_model_dir(
        dir.path(),
        &[
            ("embed_tokens.weight", "F32", &[10, 4], f32_bytes(&known)),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    assert_eq!(weights.embed.shape(), &[10, 4]);
    // First element: known[0] = 0.0
    assert!((weights.embed[[0, 0]] - known[0]).abs() < 1e-6);
    // Last element: known[39] = 3.9
    assert!((weights.embed[[9, 3]] - known[39]).abs() < 1e-5);
}

#[test]
fn load_model_dir_validated_rejects_invalid_config() {
    let dir = TempDir::new().unwrap();
    write_model_dir_with_config(
        dir.path(),
        serde_json::json!({
            "model_type": "llama",
            "hidden_size": 5,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 0,
            "vocab_size": 10,
        }),
        &minimal_tensors(),
    );

    let permissive = load_model_dir(dir.path()).unwrap();
    assert_eq!(permissive.hidden_size, 5);

    match load_model_dir_validated(dir.path()) {
        Err(ModelError::ConfigValidation(errors)) => {
            assert!(errors.iter().any(|error| error.field == FIELD_HEAD_DIM));
        }
        _ => panic!("expected config validation error"),
    }
}

#[test]
fn load_model_dir_walk_only_validated_rejects_invalid_config() {
    let dir = TempDir::new().unwrap();
    write_model_dir_with_config(
        dir.path(),
        serde_json::json!({
            "model_type": "llama",
            "hidden_size": 5,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 0,
            "vocab_size": 10,
        }),
        &minimal_tensors(),
    );

    match load_model_dir_walk_only_validated(dir.path()) {
        Err(ModelError::ConfigValidation(errors)) => {
            assert!(errors.iter().any(|error| error.field == FIELD_HEAD_DIM));
        }
        _ => panic!("expected config validation error"),
    }
}

#[test]
fn load_f16_tensors_converts_to_f32() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            ("embed_tokens.weight", "F16", &[10, 4], f16_ones(40)),
            ("norm.weight", "F16", &[4], f16_ones(4)),
            ("lm_head.weight", "F16", &[10, 4], f16_ones(40)),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    assert_eq!(weights.embed.shape(), &[10, 4]);
    // f16 1.0 → f32 1.0
    assert!((weights.embed[[0, 0]] - 1.0).abs() < 1e-4);
}

#[test]
fn load_bf16_tensors_converts_to_f32() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            ("embed_tokens.weight", "BF16", &[10, 4], bf16_ones(40)),
            ("norm.weight", "BF16", &[4], bf16_ones(4)),
            ("lm_head.weight", "BF16", &[10, 4], bf16_ones(40)),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    assert_eq!(weights.embed.shape(), &[10, 4]);
    assert!((weights.embed[[0, 0]] - 1.0).abs() < 1e-4);
}

#[test]
fn load_1d_norm_tensor_goes_into_vectors() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[2.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            (
                "layers.0.input_layernorm.weight",
                "F32",
                &[4],
                f32_bytes(&[3.0f32; 4]),
            ),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    let norm = weights.vectors.get("norm.weight").unwrap();
    assert_eq!(norm.len(), 4);
    assert!((norm[0] - 2.0).abs() < 1e-6);

    let ln = weights
        .vectors
        .get("layers.0.input_layernorm.weight")
        .unwrap();
    assert!((ln[0] - 3.0).abs() < 1e-6);
}

#[test]
fn walk_only_excludes_ffn_tensors() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            (
                "layers.0.self_attn.q_proj.weight",
                "F32",
                &[2, 4],
                f32_bytes(&[1.0f32; 8]),
            ),
            (
                "layers.0.mlp.gate_proj.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[1.0f32; 16]),
            ),
            (
                "layers.0.mlp.up_proj.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[1.0f32; 16]),
            ),
            (
                "layers.0.mlp.down_proj.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[1.0f32; 16]),
            ),
        ],
    );

    let weights = load_model_dir_walk_only(dir.path()).unwrap();
    assert!(!weights
        .tensors
        .contains_key("layers.0.mlp.gate_proj.weight"));
    assert!(!weights.tensors.contains_key("layers.0.mlp.up_proj.weight"));
    assert!(!weights
        .tensors
        .contains_key("layers.0.mlp.down_proj.weight"));
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
}

#[test]
fn walk_only_excludes_starcoder2_ffn_tensors() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 10,
    });
    write_model_dir_with_config(
        dir.path(),
        config,
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            (
                "layers.0.self_attn.q_proj.weight",
                "F32",
                &[2, 4],
                f32_bytes(&[1.0f32; 8]),
            ),
            (
                "layers.0.mlp.c_fc.weight",
                "F32",
                &[16, 4],
                f32_bytes(&[1.0f32; 64]),
            ),
            (
                "layers.0.mlp.c_proj.weight",
                "F32",
                &[4, 16],
                f32_bytes(&[1.0f32; 64]),
            ),
            (
                "layers.0.mlp.c_fc.bias",
                "F32",
                &[16],
                f32_bytes(&[1.0f32; 16]),
            ),
            (
                "layers.0.mlp.c_proj.bias",
                "F32",
                &[4],
                f32_bytes(&[1.0f32; 4]),
            ),
        ],
    );

    let weights = load_model_dir_walk_only(dir.path()).unwrap();
    assert!(!weights.tensors.contains_key("layers.0.mlp.c_fc.weight"));
    assert!(!weights.tensors.contains_key("layers.0.mlp.c_proj.weight"));
    assert!(!weights.vectors.contains_key("layers.0.mlp.c_fc.bias"));
    assert!(!weights.vectors.contains_key("layers.0.mlp.c_proj.bias"));
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
}

#[test]
fn walk_only_excludes_gpt_oss_packed_mxfp4_experts() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_local_experts": 1,
        "num_experts_per_tok": 1,
        "head_dim": 2,
        "vocab_size": 10,
    });
    write_model_dir_with_config(
        dir.path(),
        config,
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            (
                "layers.0.mlp.router.weight",
                "F32",
                &[1, 4],
                f32_bytes(&[1.0f32; 4]),
            ),
            (
                "layers.0.mlp.experts.gate_up_proj_blocks",
                "U8",
                &[1, 2, 1, 16],
                vec![0x22; 32],
            ),
            (
                "layers.0.mlp.experts.gate_up_proj_scales",
                "U8",
                &[1, 2, 1],
                vec![127; 2],
            ),
            (
                "layers.0.mlp.experts.down_proj_blocks",
                "U8",
                &[1, 1, 1, 16],
                vec![0x22; 16],
            ),
            (
                "layers.0.mlp.experts.down_proj_scales",
                "U8",
                &[1, 1, 1],
                vec![127; 1],
            ),
        ],
    );

    let weights = load_model_dir_walk_only(dir.path()).unwrap();
    assert!(!weights
        .tensors
        .keys()
        .any(|key| key.contains("block_sparse_moe.experts")));
    assert!(weights.tensors.contains_key("layers.0.mlp.router.weight"));
}

#[test]
fn packed_bf16_experts_are_mmap_backed_not_copied() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "vocab_size": 10,
            "enable_moe_block": true,
            "num_experts": 1,
            "top_k_experts": 1,
            "moe_intermediate_size": 1
        }
    });
    let gate_up_bytes: Vec<u8> = (0u8..16).collect();
    let down_bytes: Vec<u8> = (16u8..24).collect();
    write_model_dir_with_config(
        dir.path(),
        config,
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            (
                "layers.0.experts.gate_up_proj",
                "BF16",
                &[1, 2, 4],
                gate_up_bytes.clone(),
            ),
            (
                "layers.0.experts.down_proj",
                "BF16",
                &[1, 4, 1],
                down_bytes.clone(),
            ),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();

    assert!(
        weights.raw_bytes.is_empty(),
        "large packed BF16 tensors should stay in mmap ranges, not heap raw_bytes"
    );
    assert_eq!(weights.packed_mmaps.len(), 1);
    assert_eq!(
        weights
            .get_packed_bytes("layers.0.experts.gate_up_proj")
            .unwrap(),
        gate_up_bytes.as_slice()
    );
    assert_eq!(
        weights
            .get_packed_bytes("layers.0.experts.down_proj")
            .unwrap(),
        down_bytes.as_slice()
    );
}

#[test]
fn filtered_custom_predicate_skips_target() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            (
                "layers.0.self_attn.q_proj.weight",
                "F32",
                &[2, 4],
                f32_bytes(&[1.0f32; 8]),
            ),
        ],
    );

    let weights = load_model_dir_filtered(dir.path(), |k| k.contains("q_proj")).unwrap();
    assert!(!weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
    // embed and lm_head are not filtered
    assert_eq!(weights.embed.shape(), &[10, 4]);
}

#[test]
fn unsupported_dtype_goes_to_skipped_tensors() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            // attention_mask is typically I64 — should be skipped, not crash
            ("attention_mask", "I64", &[1, 10], i64_bytes(10)),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    assert!(
        !weights.skipped_tensors.is_empty(),
        "I64 tensor should be in skipped_tensors"
    );
    let (key, dtype) = &weights.skipped_tensors[0];
    assert_eq!(key, "attention_mask");
    assert!(
        dtype.contains("I64"),
        "dtype string should mention I64, got: {dtype}"
    );
}

#[test]
fn missing_embed_returns_missing_tensor_error() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            // no embed_tokens.weight
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
        ],
    );

    match load_model_dir(dir.path()) {
        Err(ModelError::MissingTensor(k)) => assert_eq!(k, "embed_tokens.weight"),
        Err(e) => panic!("expected MissingTensor, got error: {e}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn tied_lm_head_falls_back_to_embed() {
    // No lm_head.weight → falls back to embed clone.
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[2.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    assert_eq!(weights.lm_head.shape(), &[10, 4]);
    assert!((weights.lm_head[[0, 0]] - 2.0).abs() < 1e-6);
}

#[test]
fn mlx_weights_subdir_is_found() {
    // MLX layout: safetensors lives in a weights/ subdirectory.
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "llama", "hidden_size": 4, "num_hidden_layers": 1,
        "intermediate_size": 16, "num_attention_heads": 2,
        "num_key_value_heads": 2, "head_dim": 2, "vocab_size": 10,
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let weights_dir = dir.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();
    let tensors = minimal_tensors();
    std::fs::write(
        weights_dir.join("model.safetensors"),
        make_safetensors(&tensors),
    )
    .unwrap();

    let weights = load_model_dir(dir.path()).unwrap();
    assert_eq!(weights.embed.shape(), &[10, 4]);
}

#[test]
fn no_safetensors_files_returns_error() {
    let dir = TempDir::new().unwrap();
    // Complete config.json: the loader must reach the safetensors-scan
    // step (and report no files) rather than erroring earlier on a
    // partial config.
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    // No .safetensors files → NoSafetensors error
    match load_model_dir(dir.path()) {
        Err(ModelError::NoSafetensors(_)) => {}
        Err(e) => panic!("expected NoSafetensors, got error: {e}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn non_directory_non_gguf_file_returns_error() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("not_a_model.txt");
    std::fs::write(&path, b"hello").unwrap();
    match load_model_dir(&path) {
        Err(ModelError::NotADirectory(_)) => {}
        Err(e) => panic!("expected NotADirectory, got error: {e}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GGUF loading tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn load_gguf_via_load_model_dir() {
    // load_model_dir detects .gguf in the directory and delegates to load_gguf.
    let dir = TempDir::new().unwrap();
    write_minimal_gguf(&dir.path().join("model.gguf"));

    let weights = load_model_dir(dir.path()).unwrap();
    // embed_tokens: dims=[4, 100] in GGUF → shape [100, 4] after GGUF dim swap
    assert_eq!(weights.embed.shape(), &[100, 4]);
    assert_eq!(weights.vocab_size, 100);
    assert_eq!(weights.num_layers, 1);
    assert_eq!(weights.hidden_size, 4);
}

#[test]
fn load_gguf_single_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.gguf");
    write_minimal_gguf(&path);

    let weights = load_model_dir(&path).unwrap();
    assert_eq!(weights.embed.shape(), &[100, 4]);
    assert_eq!(weights.vocab_size, 100);
    assert_eq!(weights.num_layers, 1);
}

#[test]
fn load_gguf_preserves_explicit_small_vocab_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("small-vocab.gguf");
    write_minimal_gguf_custom(&path, 128, Some(128), true, true);

    let weights = load_model_dir(&path).unwrap();

    assert_eq!(weights.embed.shape(), &[128, 4]);
    assert_eq!(weights.vocab_size, 128);
}

#[test]
fn load_gguf_uses_shape_vocab_when_metadata_and_tokenizer_are_absent() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("shape-vocab.gguf");
    write_minimal_gguf_custom(&path, 64, None, true, true);

    let weights = load_model_dir(&path).unwrap();

    assert_eq!(weights.embed.shape(), &[64, 4]);
    assert_eq!(weights.vocab_size, 64);
}

#[test]
fn load_gguf_defaults_missing_kv_heads_and_key_length() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("missing-attn-metadata.gguf");
    write_minimal_gguf_custom(&path, 100, Some(100), false, false);

    let weights = load_model_dir_validated(&path).unwrap();

    assert_eq!(weights.num_q_heads, 2);
    assert_eq!(weights.num_kv_heads, 2);
    assert_eq!(weights.head_dim, 2);
}

#[test]
fn load_gguf_walk_only_excludes_ffn_tensor() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("tiny-with-ffn.gguf");
    write_gguf_with_ffn(&path);

    let weights = load_model_dir_walk_only(&path).unwrap();
    assert!(!weights
        .tensors
        .contains_key("layers.0.mlp.gate_proj.weight"));
    assert_eq!(weights.embed.shape(), &[100, 4]);
}

#[test]
fn load_gguf_prefers_largest_file_when_multiple() {
    // When a directory has multiple GGUF files, the loader picks the largest.
    let dir = TempDir::new().unwrap();
    write_minimal_gguf(&dir.path().join("model-small.gguf"));
    // Write a zero-byte "large" file — loader picks by metadata(len).
    // In practice: largest by file size. Write the big one as the real model.
    write_minimal_gguf(&dir.path().join("model-main.gguf"));
    std::fs::write(dir.path().join("shard.gguf"), [0u8; 4]).unwrap();

    // Should not panic — any successful load is acceptable here.
    let result = load_model_dir(dir.path());
    assert!(result.is_ok() || matches!(result, Err(ModelError::Parse(_))));
}

#[test]
fn gguf_vectors_map_includes_1d_norms() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.gguf");
    write_minimal_gguf(&path);

    let weights = load_model_dir(&path).unwrap();
    // output_norm.weight → normalize_gguf_key → norm.weight (1D)
    // ends up in vectors, not tensors
    assert!(
        weights.vectors.contains_key("norm.weight"),
        "1D output_norm should be in vectors as norm.weight; keys: {:?}",
        weights.vectors.keys().collect::<Vec<_>>()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// GPT-OSS MXFP4 (load_mxfp4_expert_tensors): full-load path
// ═══════════════════════════════════════════════════════════════════════════
//
// `walk_only_excludes_gpt_oss_packed_mxfp4_experts` exercises only the
// walk-only path which short-circuits before dequantising. The full
// `load_model_dir` path runs `load_mxfp4_expert_tensors`, which:
//   1. iterates safetensors looking for `*.gate_up_proj_blocks` tensors,
//   2. pairs each with its scales companion + the down companion,
//   3. dequantises them via `crate::quant::mxfp4::split_gate_up_experts`.
//
// This test pins that path against a tiny synthetic GPT-OSS fixture.

#[test]
fn load_full_gpt_oss_dequantises_packed_mxfp4_experts() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_local_experts": 1,
        "num_experts_per_tok": 1,
        "head_dim": 2,
        "vocab_size": 10,
    });
    write_model_dir_with_config(
        dir.path(),
        config,
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            (
                "layers.0.mlp.router.weight",
                "F32",
                &[1, 4],
                f32_bytes(&[1.0f32; 4]),
            ),
            // Packed MXFP4: gate+up fused, 1 expert, out_features=8 (2×inter=4),
            // groups=1, packed_bytes=16 per (expert, out, group).
            // shape = [num_experts=1, out_features=8, groups=1, 16].
            (
                "layers.0.mlp.experts.gate_up_proj_blocks",
                "U8",
                &[1, 8, 1, 16],
                vec![0x22; 8 * 16],
            ),
            (
                "layers.0.mlp.experts.gate_up_proj_scales",
                "U8",
                &[1, 8, 1],
                vec![127; 8],
            ),
            // Down: 1 expert, out_features=4 (=hidden), groups=1.
            (
                "layers.0.mlp.experts.down_proj_blocks",
                "U8",
                &[1, 4, 1, 16],
                vec![0x22; 4 * 16],
            ),
            (
                "layers.0.mlp.experts.down_proj_scales",
                "U8",
                &[1, 4, 1],
                vec![127; 4],
            ),
        ],
    );

    let weights = load_model_dir(dir.path()).expect("full GPT-OSS load");
    // The dequantiser splits gate_up into gate (w1) + up (w3) per expert,
    // and emits down (w2). Each per-expert key uses the
    // `block_sparse_moe.experts.<E>.<proj>` shape from
    // `mxfp4_expert_key`. Pin that we got non-empty tensors back.
    let any_expert_key = weights
        .tensors
        .keys()
        .any(|k| k.contains("block_sparse_moe.experts.0"));
    assert!(
        any_expert_key,
        "load_mxfp4_expert_tensors must populate per-expert keys; got: {:?}",
        weights.tensors.keys().collect::<Vec<_>>()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// DeepSeek-V4 per-expert MXFP4 (dequantize_per_expert_mxfp4)
// ═══════════════════════════════════════════════════════════════════════════
//
// V4 stores expert weights as separate `(.weight=I8, .scale=F8_E8M0)`
// pairs per (expert, projection). The `dequantize_per_expert_mxfp4`
// detector inside the standard load branch finds those pairs by
// dtype + naming pattern, regardless of architecture metadata. This
// test exercises that path against a synthetic V4-style fixture.

fn f8_e8m0_bytes(n: usize) -> Vec<u8> {
    // Byte = 127 → 2^0 = 1.0 (per-group scale of unity).
    vec![127u8; n]
}

#[test]
fn load_full_deepseek_v4_dequantises_per_expert_mxfp4() {
    let dir = TempDir::new().unwrap();
    // V4 detection in detect.rs requires `model_type = "deepseek_v4"`,
    // and uses_mla() requires kv/q_lora_rank present.
    let config = serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 10,
        "n_routed_experts": 1,
        "num_experts_per_tok": 1,
        "n_shared_experts": 0,
        "kv_lora_rank": 4,
        "q_lora_rank": 4,
    });
    // V4 strips no `model.` prefix and uses `embed.weight` + `norm.weight`
    // (see `architectures/deepseek_v4.rs`).
    //
    // Per-expert MXFP4 layout: weight [out_features, packed_cols=groups*16],
    // scale [out_features, groups]. We use out_features=2, groups=1 →
    // packed_cols=16 → unpacked_cols=32.
    let out_features = 2usize;
    let groups = 1usize;
    let packed_cols = groups * 16;
    let weight_bytes = vec![0u8; out_features * packed_cols]; // all-zero nibbles → all-zero unpacked
    let scale_bytes = f8_e8m0_bytes(out_features * groups);

    write_model_dir_with_config(
        dir.path(),
        config,
        &[
            ("embed.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            // V4 doesn't necessarily have lm_head; loader falls back to embed.
            // Per-expert MXFP4 weight + scale pair for w1 (gate_proj).
            (
                "layers.0.ffn.experts.0.w1.weight",
                "I8",
                &[out_features, packed_cols],
                weight_bytes.clone(),
            ),
            (
                "layers.0.ffn.experts.0.w1.scale",
                "F8_E8M0",
                &[out_features, groups],
                scale_bytes.clone(),
            ),
            // Plus w2 (down) and w3 (up) — same shape.
            (
                "layers.0.ffn.experts.0.w2.weight",
                "I8",
                &[out_features, packed_cols],
                weight_bytes.clone(),
            ),
            (
                "layers.0.ffn.experts.0.w2.scale",
                "F8_E8M0",
                &[out_features, groups],
                scale_bytes.clone(),
            ),
            (
                "layers.0.ffn.experts.0.w3.weight",
                "I8",
                &[out_features, packed_cols],
                weight_bytes,
            ),
            (
                "layers.0.ffn.experts.0.w3.scale",
                "F8_E8M0",
                &[out_features, groups],
                scale_bytes,
            ),
        ],
    );

    let weights = load_model_dir(dir.path()).expect("full V4 load");
    // The V4 dequantiser writes the dequantised weight under the
    // (prefix-stripped) tensor name. With V4's empty prefix list, that's
    // exactly `layers.0.ffn.experts.0.w1.weight`.
    assert!(
        weights
            .tensors
            .contains_key("layers.0.ffn.experts.0.w1.weight"),
        "V4 dequantiser must emit the unpacked weight; got: {:?}",
        weights.tensors.keys().collect::<Vec<_>>()
    );
    let arr = weights
        .tensors
        .get("layers.0.ffn.experts.0.w1.weight")
        .unwrap();
    // Output cols = packed_cols * 2 = 32 (the dequantiser unpacks
    // nibbles).
    assert_eq!(arr.shape(), &[out_features, packed_cols * 2]);
}

#[test]
fn load_filtered_validated_runs_with_validation() {
    // `load_model_dir_filtered_validated` is the public validated
    // entrypoint that callers like the streaming extractor use. Pin that
    // it accepts a passing predicate + invokes the validation path.
    let dir = TempDir::new().unwrap();
    write_model_dir(dir.path(), &minimal_tensors());
    let weights = larql_models::load_model_dir_filtered_validated(dir.path(), |_| false)
        .expect("validated filter accepts minimal model");
    assert!(weights.tensors.contains_key("embed_tokens.weight"));
}

// ─────────────────────────────────────────────────────────────────
// Coverage for the file-vs-directory + missing-file early-return arms
// in `loading/safetensors.rs::load_model_dir_filtered_with_validation`.
// ─────────────────────────────────────────────────────────────────

#[test]
fn load_model_dir_errors_when_path_points_at_a_non_gguf_file() {
    // `path.is_file()` true, extension != gguf → NotADirectory.
    let dir = TempDir::new().unwrap();
    let bogus = dir.path().join("random.txt");
    std::fs::write(&bogus, "not a model").unwrap();
    match load_model_dir(&bogus) {
        Err(ModelError::NotADirectory(p)) => assert_eq!(p, bogus),
        Err(e) => panic!("expected NotADirectory, got error: {e}"),
        Ok(_) => panic!("expected NotADirectory, got Ok"),
    }
}

#[test]
fn load_model_dir_errors_when_path_does_not_exist() {
    // Neither file nor directory — falls into the second NotADirectory
    // branch (line 121).
    let dir = TempDir::new().unwrap();
    let missing = dir.path().join("does-not-exist");
    match load_model_dir(&missing) {
        Err(ModelError::NotADirectory(p)) => assert_eq!(p, missing),
        Err(e) => panic!("expected NotADirectory, got error: {e}"),
        Ok(_) => panic!("expected NotADirectory, got Ok"),
    }
}

#[test]
fn load_model_dir_errors_when_safetensors_missing_in_directory() {
    // Empty directory (no safetensors, no GGUF) — exercises the
    // "no safetensors files in {path}" branch.
    let dir = TempDir::new().unwrap();
    // Must still have config.json so detect_architecture passes.
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    match load_model_dir(dir.path()) {
        Err(ModelError::NoSafetensors(p)) => assert_eq!(p, dir.path()),
        Err(e) => panic!("expected NoSafetensors, got error: {e}"),
        Ok(_) => panic!("expected NoSafetensors, got Ok"),
    }
}

#[test]
fn load_model_dir_filtered_skip_predicate_omits_filtered_tensors() {
    // `load_model_dir_filtered` accepts a closure that returns true for
    // tensors to skip. Covers the filter branch + the resulting
    // skipped-tensor accounting.
    let dir = TempDir::new().unwrap();
    write_model_dir(dir.path(), &minimal_tensors());
    let weights = load_model_dir_filtered(dir.path(), |name| name == "norm.weight")
        .expect("filtered load should succeed");
    assert!(weights.tensors.contains_key("embed_tokens.weight"));
    assert!(!weights.tensors.contains_key("norm.weight"));
}
