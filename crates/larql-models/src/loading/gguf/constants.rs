//! GGUF format constants — magic number, type IDs, key names, replacement table.

pub(super) const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian

// Metadata value types
pub(super) const GGUF_TYPE_UINT8: u32 = 0;
pub(super) const GGUF_TYPE_INT8: u32 = 1;
pub(super) const GGUF_TYPE_UINT16: u32 = 2;
pub(super) const GGUF_TYPE_INT16: u32 = 3;
pub(super) const GGUF_TYPE_UINT32: u32 = 4;
pub(super) const GGUF_TYPE_INT32: u32 = 5;
pub(super) const GGUF_TYPE_FLOAT32: u32 = 6;
pub(super) const GGUF_TYPE_BOOL: u32 = 7;
pub(super) const GGUF_TYPE_STRING: u32 = 8;
pub(super) const GGUF_TYPE_ARRAY: u32 = 9;
pub(super) const GGUF_TYPE_UINT64: u32 = 10;
pub(super) const GGUF_TYPE_INT64: u32 = 11;
pub(super) const GGUF_TYPE_FLOAT64: u32 = 12;

pub(super) const GGUF_GENERAL_ARCHITECTURE: &str = "general.architecture";
pub(super) const GGUF_EMBEDDING_LENGTH: &str = "embedding_length";
pub(super) const GGUF_BLOCK_COUNT: &str = "block_count";
pub(super) const GGUF_FEED_FORWARD_LENGTH: &str = "feed_forward_length";
// MoE-only architectures (DeepSeek-V4 family) omit the global
// `feed_forward_length` and emit only the per-expert size — fall back
// to it so config validation doesn't reject the model with
// `intermediate_size: must be greater than 0`.
pub(super) const GGUF_EXPERT_FEED_FORWARD_LENGTH: &str = "expert_feed_forward_length";
pub(super) const GGUF_ATTENTION_HEAD_COUNT: &str = "attention.head_count";
pub(super) const GGUF_ATTENTION_HEAD_COUNT_KV: &str = "attention.head_count_kv";
pub(super) const GGUF_ATTENTION_KEY_LENGTH: &str = "attention.key_length";
pub(super) const GGUF_ROPE_FREQ_BASE: &str = "rope.freq_base";
// MLA-specific metadata keys emitted by llama.cpp for DeepSeek-V2/V3/Kimi-K2
// family models. `_mla` variants carry the pre-absorption per-head dims;
// non-`_mla` variants carry the (possibly larger) absorbed/effective sizes.
// `rope.dimension_count` is the RoPE-positional portion of each Q/K head
// (qk_rope_head_dim in the HF config).
pub(super) const GGUF_ATTENTION_KEY_LENGTH_MLA: &str = "attention.key_length_mla";
pub(super) const GGUF_ATTENTION_VALUE_LENGTH: &str = "attention.value_length";
pub(super) const GGUF_ATTENTION_VALUE_LENGTH_MLA: &str = "attention.value_length_mla";
pub(super) const GGUF_ATTENTION_Q_LORA_RANK: &str = "attention.q_lora_rank";
pub(super) const GGUF_ATTENTION_KV_LORA_RANK: &str = "attention.kv_lora_rank";
pub(super) const GGUF_ROPE_DIMENSION_COUNT: &str = "rope.dimension_count";
pub(super) const GGUF_VOCAB_SIZE: &str = "vocab_size";

pub(super) const HF_MODEL_TYPE: &str = "model_type";
pub(super) const HF_HIDDEN_SIZE: &str = "hidden_size";
pub(super) const HF_NUM_HIDDEN_LAYERS: &str = "num_hidden_layers";
pub(super) const HF_INTERMEDIATE_SIZE: &str = "intermediate_size";
pub(super) const HF_NUM_ATTENTION_HEADS: &str = "num_attention_heads";
pub(super) const HF_NUM_KEY_VALUE_HEADS: &str = "num_key_value_heads";
pub(super) const HF_HEAD_DIM: &str = "head_dim";
pub(super) const HF_ROPE_THETA: &str = "rope_theta";
pub(super) const HF_VOCAB_SIZE: &str = "vocab_size";

pub(super) const TOKENIZER_JSON: &str = "tokenizer.json";
pub(super) const TOKENIZER_MODEL: &str = "model";
pub(super) const TOKENIZER_VOCAB: &str = "vocab";

pub(super) const GGUF_OUTPUT_WEIGHT: &str = "output.weight";
pub(super) const DEFAULT_GGUF_VOCAB_SIZE: usize = 262_144;
pub(super) const GEMMA4_GGUF_HEAD_DIM: u32 = 256;

pub(super) const GGUF_TO_HF_KEY_REPLACEMENTS: &[(&str, &str)] = &[
    ("blk.", "layers."),
    ("attn_qkv.", "self_attn.qkv_proj."),
    ("attn_q.", "self_attn.q_proj."),
    ("attn_k.", "self_attn.k_proj."),
    ("attn_v.", "self_attn.v_proj."),
    ("attn_output.", "self_attn.o_proj."),
    ("ffn_gate.", "mlp.gate_proj."),
    ("ffn_up.", "mlp.up_proj."),
    ("ffn_down.", "mlp.down_proj."),
    ("attn_norm.", "input_layernorm."),
    ("ffn_norm.", "post_attention_layernorm."),
    ("token_embd.", "embed_tokens."),
    ("position_embd.", "wpe."),
    ("output_norm.", "norm."),
    ("output.", "lm_head."),
];

/// Gemma 2/3/4 layers carry four norms plus QK-norms; the generic table
/// above maps `ffn_norm.` to `post_attention_layernorm.` (correct for the
/// llama two-norm layout, wrong here — gemma's `ffn_norm` is the pre-FFN
/// norm and `post_attention_norm` is the real post-attention one) and has
/// no entries for the rest. Applied BEFORE the generic table so `ffn_norm.`
/// is consumed by the gemma rule first. Gemma 1 keeps the generic path.
pub(super) const GGUF_TO_HF_KEY_REPLACEMENTS_GEMMA: &[(&str, &str)] = &[
    ("attn_q_norm.", "self_attn.q_norm."),
    ("attn_k_norm.", "self_attn.k_norm."),
    ("post_attention_norm.", "post_attention_layernorm."),
    ("ffn_norm.", "pre_feedforward_layernorm."),
    ("post_ffw_norm.", "post_feedforward_layernorm."),
    // Gemma 4 per-layer output scalar; larql's key has no `.weight` suffix.
    ("layer_output_scale.weight", "layer_scalar"),
];

// Tensor type constants moved to format::quant::ggml
