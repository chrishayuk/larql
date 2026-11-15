//! A miniature **Gemma 3 text** checkpoint, stored BF16, with the
//! capability snapshot a GGUF export reads.
//!
//! Real Gemma 3 anatomy at toy width: four norms per layer plus Q/K norms,
//! a GELU-tanh gated FFN, a sliding/full interleave at the class-default
//! period with the full layers on linearly-scaled RoPE, a tied head, and a
//! SentencePiece-style BPE tokenizer with byte fallback, control tokens and
//! a chat template. The vocabulary is shorter than the embedding so the
//! padding ids are exercised.

use std::path::{Path, PathBuf};

use super::{lcg_values, ShardBuilder};
use crate::format::vindex3::encode::checkpoint::encode_checkpoint;

pub const GEMMA3_FIXTURE_LAYERS: usize = 6;
pub const GEMMA3_FIXTURE_PERIOD: usize = 6;
pub const GEMMA3_FIXTURE_HIDDEN: usize = 32;
pub const GEMMA3_FIXTURE_INTER: usize = 64;
pub const GEMMA3_FIXTURE_Q_HEADS: usize = 4;
pub const GEMMA3_FIXTURE_KV_HEADS: usize = 1;
pub const GEMMA3_FIXTURE_HEAD_DIM: usize = 16;
pub const GEMMA3_FIXTURE_VOCAB: usize = 24;
pub const GEMMA3_FIXTURE_WINDOW: usize = 8;
pub const GEMMA3_FIXTURE_CONTEXT: usize = 4096;
pub const GEMMA3_FIXTURE_GLOBAL_THETA: f64 = 1_000_000.0;
pub const GEMMA3_FIXTURE_LOCAL_THETA: f64 = 10_000.0;
pub const GEMMA3_FIXTURE_ROPE_FACTOR: f64 = 8.0;
pub const GEMMA3_FIXTURE_EPS: f64 = 1e-6;
pub const GEMMA3_FIXTURE_TOP_K: i64 = 64;
pub const GEMMA3_FIXTURE_TOP_P: f64 = 0.95;
/// The checkpoint directory's name, which the container records as its
/// source artifact and the GGUF names itself after.
pub const GEMMA3_FIXTURE_NAME: &str = "gemma-3-tiny-it";
pub const GEMMA3_FIXTURE_TEMPLATE: &str =
    "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}<end_of_turn>\n{% endfor %}";

/// The tokenizer's base vocabulary, in id order. Ids 4..6 are bytes;
/// the two turn markers are control tokens; `▁▁` is user-defined.
pub const GEMMA3_FIXTURE_PIECES: &[&str] = &[
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<0x00>",
    "<0x01>",
    "<0x0A>",
    "\u{2581}",
    "a",
    "b",
    "\u{2581}a",
    "ab",
    "<start_of_turn>",
    "<end_of_turn>",
    "\u{2581}\u{2581}",
    "c",
    "<unused0>",
];
/// `(id, special)` for every added token.
const ADDED: &[(usize, bool)] = &[
    (0, true),
    (1, true),
    (2, true),
    (3, true),
    (12, true),
    (13, true),
    (14, false),
    (16, false),
];

fn bf16(values: &[f32]) -> Vec<u8> {
    larql_models::quant::half::encode_bf16(values)
}

fn push(shard: &mut ShardBuilder, name: &str, shape: &[usize], seed: u64) {
    let n = shape.iter().product();
    shard.push_bytes(name, "BF16", shape, &bf16(&lcg_values(n, seed)));
}

fn config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3_text",
        "attention_bias": false,
        "attn_logit_softcapping": null,
        "final_logit_softcapping": null,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "head_dim": GEMMA3_FIXTURE_HEAD_DIM,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": GEMMA3_FIXTURE_HIDDEN,
        "intermediate_size": GEMMA3_FIXTURE_INTER,
        "max_position_embeddings": GEMMA3_FIXTURE_CONTEXT,
        "num_attention_heads": GEMMA3_FIXTURE_Q_HEADS,
        "num_hidden_layers": GEMMA3_FIXTURE_LAYERS,
        "num_key_value_heads": GEMMA3_FIXTURE_KV_HEADS,
        "query_pre_attn_scalar": GEMMA3_FIXTURE_HEAD_DIM,
        "rms_norm_eps": GEMMA3_FIXTURE_EPS,
        "rope_local_base_freq": GEMMA3_FIXTURE_LOCAL_THETA,
        "rope_scaling": {"rope_type": "linear", "factor": GEMMA3_FIXTURE_ROPE_FACTOR},
        "rope_theta": GEMMA3_FIXTURE_GLOBAL_THETA,
        "sliding_window": GEMMA3_FIXTURE_WINDOW,
        "sliding_window_pattern": GEMMA3_FIXTURE_PERIOD,
        "torch_dtype": "bfloat16",
        "vocab_size": GEMMA3_FIXTURE_VOCAB,
    })
}

fn tokenizer_json() -> serde_json::Value {
    let vocab: serde_json::Map<String, serde_json::Value> = GEMMA3_FIXTURE_PIECES
        .iter()
        .enumerate()
        .map(|(i, p)| (p.to_string(), serde_json::json!(i)))
        .collect();
    let added: Vec<serde_json::Value> = ADDED
        .iter()
        .map(|(id, special)| {
            serde_json::json!({
                "id": id, "content": GEMMA3_FIXTURE_PIECES[*id], "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": special
            })
        })
        .collect();
    serde_json::json!({
        "version": "1.0",
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": "<bos>", "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}}
        },
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
                  "vocab": vocab, "merges": [["\u{2581}", "a"], ["a", "b"]]}
    })
}

fn tokenizer_config() -> serde_json::Value {
    let decoder: serde_json::Map<String, serde_json::Value> = ADDED
        .iter()
        .map(|(id, special)| {
            (
                id.to_string(),
                serde_json::json!({"content": GEMMA3_FIXTURE_PIECES[*id], "lstrip": false,
                    "normalized": false, "rstrip": false, "single_word": false, "special": special}),
            )
        })
        .collect();
    serde_json::json!({
        "add_bos_token": true,
        "add_eos_token": false,
        "added_tokens_decoder": decoder,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "chat_template": GEMMA3_FIXTURE_TEMPLATE,
        "tokenizer_class": "GemmaTokenizer",
    })
}

fn write_json(dir: &Path, name: &str, value: &serde_json::Value) {
    std::fs::write(dir.join(name), serde_json::to_string_pretty(value).unwrap()).unwrap();
}

/// Write the checkpoint into `dir` (config, BF16 safetensors, tokenizer
/// files, generation config).
pub fn gemma3_text_checkpoint(dir: &Path) {
    write_json(dir, "config.json", &config());
    write_json(dir, "tokenizer.json", &tokenizer_json());
    write_json(dir, "tokenizer_config.json", &tokenizer_config());
    write_json(
        dir,
        "generation_config.json",
        &serde_json::json!({"bos_token_id": 2, "eos_token_id": [1, 13], "pad_token_id": 0,
            "do_sample": true, "top_k": GEMMA3_FIXTURE_TOP_K, "top_p": GEMMA3_FIXTURE_TOP_P}),
    );
    let h = GEMMA3_FIXTURE_HIDDEN;
    let hd = GEMMA3_FIXTURE_HEAD_DIM;
    let q = GEMMA3_FIXTURE_Q_HEADS * hd;
    let kv = GEMMA3_FIXTURE_KV_HEADS * hd;
    let inter = GEMMA3_FIXTURE_INTER;
    let mut s = ShardBuilder::new();
    push(
        &mut s,
        "model.embed_tokens.weight",
        &[GEMMA3_FIXTURE_VOCAB, h],
        1,
    );
    push(&mut s, "model.norm.weight", &[h], 2);
    for layer in 0..GEMMA3_FIXTURE_LAYERS {
        let seed = 100 + 20 * layer as u64;
        let p = format!("model.layers.{layer}");
        let tensors: [(&str, Vec<usize>); 13] = [
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("pre_feedforward_layernorm.weight", vec![h]),
            ("post_feedforward_layernorm.weight", vec![h]),
            ("self_attn.q_proj.weight", vec![q, h]),
            ("self_attn.k_proj.weight", vec![kv, h]),
            ("self_attn.v_proj.weight", vec![kv, h]),
            ("self_attn.o_proj.weight", vec![h, q]),
            ("self_attn.q_norm.weight", vec![hd]),
            ("self_attn.k_norm.weight", vec![hd]),
            ("mlp.gate_proj.weight", vec![inter, h]),
            ("mlp.up_proj.weight", vec![inter, h]),
            ("mlp.down_proj.weight", vec![h, inter]),
        ];
        for (i, (name, shape)) in tensors.iter().enumerate() {
            push(&mut s, &format!("{p}.{name}"), shape, seed + i as u64);
        }
    }
    s.write(dir);
}

/// Encode the checkpoint under `work/` into a container and return the
/// container's path. The checkpoint stays at `work/<GEMMA3_FIXTURE_NAME>`.
pub fn gemma3_text_container(work: &Path) -> PathBuf {
    let checkpoint = work.join(GEMMA3_FIXTURE_NAME);
    std::fs::create_dir_all(&checkpoint).unwrap();
    gemma3_text_checkpoint(&checkpoint);
    let container = work.join("container.vindex3");
    encode_checkpoint(&checkpoint, &container).expect("the Gemma 3 fixture encodes");
    container
}
