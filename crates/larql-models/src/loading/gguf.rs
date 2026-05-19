//! GGUF format reader — parse GGUF files and load tensors as f32.
//!
//! GGUF is the GGML Universal Format used by llama.cpp.
//! We support reading unquantized (F32, F16, BF16) and quantized (Q4_0, Q4_1, Q8_0) tensors.
//! All tensors are dequantized to f32 for use with ModelWeights.

use std::collections::HashMap;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use ndarray::Array2;

use crate::detect::{detect_from_json_validated, ModelError};
use crate::weights::ModelWeights;

// ═══════════════════════════════════════════════════════════════
// GGUF constants
// ═══════════════════════════════════════════════════════════════

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian

// Metadata value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

const GGUF_GENERAL_ARCHITECTURE: &str = "general.architecture";
const GGUF_EMBEDDING_LENGTH: &str = "embedding_length";
const GGUF_BLOCK_COUNT: &str = "block_count";
const GGUF_FEED_FORWARD_LENGTH: &str = "feed_forward_length";
const GGUF_ATTENTION_HEAD_COUNT: &str = "attention.head_count";
const GGUF_ATTENTION_HEAD_COUNT_KV: &str = "attention.head_count_kv";
const GGUF_ATTENTION_KEY_LENGTH: &str = "attention.key_length";
const GGUF_ROPE_FREQ_BASE: &str = "rope.freq_base";
const GGUF_VOCAB_SIZE: &str = "vocab_size";

const HF_MODEL_TYPE: &str = "model_type";
const HF_HIDDEN_SIZE: &str = "hidden_size";
const HF_NUM_HIDDEN_LAYERS: &str = "num_hidden_layers";
const HF_INTERMEDIATE_SIZE: &str = "intermediate_size";
const HF_NUM_ATTENTION_HEADS: &str = "num_attention_heads";
const HF_NUM_KEY_VALUE_HEADS: &str = "num_key_value_heads";
const HF_HEAD_DIM: &str = "head_dim";
const HF_ROPE_THETA: &str = "rope_theta";
const HF_VOCAB_SIZE: &str = "vocab_size";

const TOKENIZER_JSON: &str = "tokenizer.json";
const TOKENIZER_MODEL: &str = "model";
const TOKENIZER_VOCAB: &str = "vocab";

const GGUF_OUTPUT_WEIGHT: &str = "output.weight";
const DEFAULT_GGUF_VOCAB_SIZE: usize = 262_144;
const GEMMA4_GGUF_HEAD_DIM: u32 = 256;

const GGUF_TO_HF_KEY_REPLACEMENTS: &[(&str, &str)] = &[
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

// Tensor type constants moved to format::quant::ggml

// ═══════════════════════════════════════════════════════════════
// GGUF metadata value
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::F32(v) => Some(*v as f64),
            GgufValue::F64(v) => Some(*v),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// GGUF tensor info
// ═══════════════════════════════════════════════════════════════

pub struct GgufTensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    tensor_type: u32,
    offset: u64,
}

// ═══════════════════════════════════════════════════════════════
// GGUF reader
// ═══════════════════════════════════════════════════════════════

pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensor_infos: Vec<GgufTensorInfo>,
    pub data_offset: u64,
    pub path: std::path::PathBuf,
}

impl GgufFile {
    /// Parse a GGUF file header and tensor info (does not read tensor data yet).
    pub fn open(path: &Path) -> Result<Self, ModelError> {
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        // Magic
        let magic = read_u32(&mut r)?;
        if magic != GGUF_MAGIC {
            return Err(ModelError::Parse(format!(
                "not a GGUF file (magic: 0x{:08X}, expected 0x{:08X})",
                magic, GGUF_MAGIC
            )));
        }

        // Version
        let version = read_u32(&mut r)?;
        if !(2..=3).contains(&version) {
            return Err(ModelError::Parse(format!(
                "unsupported GGUF version: {version}"
            )));
        }

        let n_tensors = read_u64(&mut r)? as usize;
        let n_metadata = read_u64(&mut r)? as usize;

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_metadata {
            let key = read_string(&mut r)?;
            let value = read_value(&mut r)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensor_infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut r)?);
            }
            let tensor_type = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;
            tensor_infos.push(GgufTensorInfo {
                name,
                n_dims,
                dims,
                tensor_type,
                offset,
            });
        }

        // Data starts at next alignment boundary (32 bytes)
        let pos = r.stream_position().map_err(ModelError::Io)?;
        let alignment = 32u64;
        let data_offset = pos.div_ceil(alignment) * alignment;

        Ok(GgufFile {
            metadata,
            tensor_infos,
            data_offset,
            path: path.to_path_buf(),
        })
    }

    /// Load all tensors, dequantizing to f32.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors(
        &self,
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
        ),
        ModelError,
    > {
        self.load_tensors_filtered(&|_| false)
    }

    /// Load tensors, skipping normalized keys before reading/dequantizing tensor data.
    ///
    /// `skip_key` sees keys after GGUF-to-HF normalization but before architecture-specific
    /// prefix stripping. GGUF keys do not carry the HF wrapper prefixes, so this is enough for
    /// the current GGUF path and lets walk-only loading avoid FFN dequantization.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors_filtered(
        &self,
        skip_key: &dyn Fn(&str) -> bool,
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
        ),
        ModelError,
    > {
        let file = std::fs::File::open(&self.path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();

        for info in &self.tensor_infos {
            // Normalize key name (strip GGUF prefixes). Do this before data-size/dequant
            // work so filtered loading avoids touching skipped tensor bytes.
            let key = normalize_gguf_key(&info.name);
            if skip_key(&key) {
                continue;
            }

            let abs_offset = self.data_offset.checked_add(info.offset).ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: data_offset {} + tensor offset {} overflows u64",
                    info.name, self.data_offset, info.offset,
                ))
            })?;
            let n_elements: u64 = info.dims.iter().product();

            let data_size = tensor_data_size(info.tensor_type, n_elements as usize)?;
            let abs_offset_usize = usize::try_from(abs_offset).map_err(|_| {
                ModelError::Parse(format!(
                    "tensor {}: absolute offset {} exceeds usize on this platform",
                    info.name, abs_offset,
                ))
            })?;
            let end = abs_offset_usize.checked_add(data_size).ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: offset {} + size {} overflows usize",
                    info.name, abs_offset_usize, data_size,
                ))
            })?;
            if end > mmap.len() {
                return Err(ModelError::Parse(format!(
                    "tensor {} data out of bounds (offset {} + size {} > file {})",
                    info.name,
                    abs_offset,
                    data_size,
                    mmap.len()
                )));
            }

            let raw = &mmap[abs_offset_usize..end];
            let floats = dequantize(raw, info.tensor_type, n_elements as usize)?;

            match info.n_dims {
                2 => {
                    // GGUF/GGML stores tensor dimensions in reverse order:
                    //   dims[0] = number of columns (innermost/fastest)
                    //   dims[1] = number of rows (outermost)
                    // The raw bytes are contiguous along dims[0], so after swapping
                    // to the conventional [rows, cols] shape, ndarray's standard
                    // row-major layout preserves the matrix values.
                    let ne0 = info.dims[0] as usize; // columns in GGML
                    let ne1 = info.dims[1] as usize; // rows in GGML
                    let arr = Array2::from_shape_vec((ne1, ne0), floats)
                        .map_err(|e| ModelError::Parse(format!("tensor {}: {}", info.name, e)))?;
                    tensors.insert(key, arr.into_shared());
                }
                1 => {
                    vectors.insert(key, floats);
                }
                _ => {} // skip higher-dim tensors
            }
        }

        Ok((tensors, vectors))
    }

    /// Build a config.json-equivalent from GGUF metadata for architecture detection.
    pub fn to_config_json(&self) -> serde_json::Value {
        let get_str = |k: &str| {
            self.metadata
                .get(k)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let _get_u32 = |k: &str| self.metadata.get(k).and_then(|v| v.as_u32()).unwrap_or(0);

        // GGUF uses "general.architecture" and "{arch}.*" keys
        let arch = get_str(GGUF_GENERAL_ARCHITECTURE);
        let prefix = format!("{arch}.");

        let get_arch_u32 = |suffix: &str| {
            let key = format!("{prefix}{suffix}");
            if let Some(v) = self.metadata.get(&key) {
                // Try scalar first, then array max (handles Gemma 4 variable FFN sizes)
                if let Some(val) = v.as_u32() {
                    return val;
                }
                if let GgufValue::Array(arr) = v {
                    return arr.iter().filter_map(|x| x.as_u32()).max().unwrap_or(0);
                }
            }
            0
        };
        let get_arch_u32_opt = |suffix: &str| {
            let key = format!("{prefix}{suffix}");
            self.metadata.get(&key).and_then(|v| v.as_u32())
        };
        let get_arch_f64 = |suffix: &str| {
            self.metadata
                .get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_f64())
        };

        // Map GGUF architecture names to HF model_type
        let model_type = match arch.as_str() {
            "llama" => "llama",
            "gemma" | "gemma2" | "gemma3" | "gemma4" => &arch,
            "qwen" | "qwen2" => "qwen2",
            "mistral" => "mistral",
            "mixtral" => "mixtral",
            "phi" | "phi2" | "phi3" => "phi",
            "gpt2" => "gpt2",
            "deepseek" | "deepseek2" => "deepseek_v2",
            other => other,
        };

        let hidden_size = get_arch_u32(GGUF_EMBEDDING_LENGTH);
        let num_heads = get_arch_u32(GGUF_ATTENTION_HEAD_COUNT);
        let num_kv_heads = get_arch_u32(GGUF_ATTENTION_HEAD_COUNT_KV);
        let head_dim = if arch == "gemma4" && num_heads > 0 {
            // Gemma 4 GGUF metadata reports the global key length; known
            // exports use 256 for the per-head dimension that the runtime
            // architecture needs as its base layer head_dim.
            GEMMA4_GGUF_HEAD_DIM
        } else {
            let key_length = get_arch_u32(GGUF_ATTENTION_KEY_LENGTH);
            if key_length > 0 {
                key_length
            } else {
                hidden_size.checked_div(num_heads).unwrap_or(0)
            }
        };
        let num_kv_heads = if num_kv_heads > 0 {
            num_kv_heads
        } else {
            num_heads
        };

        let mut config = serde_json::json!({
            HF_MODEL_TYPE: model_type,
            HF_HIDDEN_SIZE: hidden_size,
            HF_NUM_HIDDEN_LAYERS: get_arch_u32(GGUF_BLOCK_COUNT),
            HF_INTERMEDIATE_SIZE: get_arch_u32(GGUF_FEED_FORWARD_LENGTH),
            HF_NUM_ATTENTION_HEADS: num_heads,
            HF_NUM_KEY_VALUE_HEADS: num_kv_heads,
            HF_HEAD_DIM: head_dim,
        });

        if let Some(rope_base) = get_arch_f64(GGUF_ROPE_FREQ_BASE) {
            config[HF_ROPE_THETA] = serde_json::json!(rope_base);
        }
        if let Some(vocab_size) = get_arch_u32_opt(GGUF_VOCAB_SIZE).filter(|&v| v > 0) {
            config[HF_VOCAB_SIZE] = serde_json::json!(vocab_size);
        }

        config
    }
}

/// Load a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf(path: &Path) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered(path, &|_| false)
}

/// Load and validate a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf_validated(path: &Path) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation(path, &|_| false, true)
}

/// Load a GGUF file into ModelWeights, skipping normalized keys before dequantization.
pub(crate) fn load_gguf_filtered(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation(path, skip_key, false)
}

/// Load a GGUF file into ModelWeights with optional architecture validation.
pub(crate) fn load_gguf_filtered_with_validation(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
    validate_config: bool,
) -> Result<ModelWeights, ModelError> {
    let gguf = GgufFile::open(path)?;

    // Detect architecture from GGUF metadata
    let config_json = gguf.to_config_json();
    let arch = if validate_config {
        detect_from_json_validated(&config_json)?
    } else {
        crate::detect_from_json(&config_json)
    };
    let prefixes = arch.key_prefixes_to_strip();

    // Load and dequantize all tensors
    let (mut tensors, mut vectors) = gguf.load_tensors_filtered(skip_key)?;

    // Re-normalize keys through the architecture's prefix stripping
    let mut normalized_tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    for (k, v) in tensors.drain() {
        let key = super::safetensors::normalize_key(&k, prefixes);
        normalized_tensors.insert(key, v);
    }

    // Some GGUF converters (notably non-standard GPT-2 builds) ship FFN /
    // attention weights in the transpose of the canonical Linear layout. Fix
    // orientation up-front so all downstream consumers see a single shape.
    orient_ffn_tensors(&mut normalized_tensors, &*arch);
    orient_attention_tensors(&mut normalized_tensors, &*arch);

    // Architectures that pack Q/K/V into one Conv1D matrix (GPT-2) ship a
    // single `qkv_proj` tensor. Split into per-projection q/k/v tensors and
    // matching biases so downstream consumers always see the unfused layout
    // returned by `attn_q_key` / `attn_k_key` / `attn_v_key`.
    split_fused_qkv(&mut normalized_tensors, &mut vectors, &*arch);

    let embed_key = arch.embed_key();
    let embed_raw = normalized_tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();
    let cfg = arch.config();
    let tokenizer_vocab_size = read_tokenizer_vocab_size(path);
    let configured_vocab_size = cfg.vocab_size.filter(|&v| v > 0);
    let expected_vocab_size = configured_vocab_size.or(tokenizer_vocab_size);
    let embed = orient_embedding(embed_raw, cfg.hidden_size, expected_vocab_size);

    let lm_head = normalized_tensors
        .get("lm_head.weight")
        .or_else(|| normalized_tensors.get(GGUF_OUTPUT_WEIGHT))
        .cloned()
        .unwrap_or_else(|| embed.clone());
    let position_embed = arch
        .position_embed_key()
        .and_then(|key| normalized_tensors.get(key).cloned());

    // Prefer explicit metadata, then tokenizer.json, then the loaded embedding
    // shape. The final constant is only for malformed files with an empty
    // embedding; normal GGUFs should resolve from one of the first three.
    let vocab_size = expected_vocab_size
        .or_else(|| (embed.shape()[0] > 0).then_some(embed.shape()[0]))
        .unwrap_or(DEFAULT_GGUF_VOCAB_SIZE);

    Ok(ModelWeights {
        tensors: normalized_tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        position_embed,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

fn read_tokenizer_vocab_size(path: &Path) -> Option<usize> {
    let parent = path.parent()?;
    let tok_path = parent.join(TOKENIZER_JSON);
    let data = std::fs::read_to_string(tok_path).ok()?;
    let json = serde_json::from_str::<serde_json::Value>(&data).ok()?;
    json[TOKENIZER_MODEL][TOKENIZER_VOCAB]
        .as_object()
        .map(|v| v.len())
        .filter(|&v| v > 0)
}

fn orient_embedding(
    embed: crate::WeightArray,
    hidden_size: usize,
    vocab_size: Option<usize>,
) -> crate::WeightArray {
    let shape = embed.shape();
    let rows = shape[0];
    let cols = shape[1];

    if cols == hidden_size || vocab_size.is_some_and(|vocab| rows == vocab) {
        return embed;
    }
    if rows == hidden_size || vocab_size.is_some_and(|vocab| cols == vocab) {
        let mut out = ndarray::Array2::<f32>::zeros((cols, rows));
        out.assign(&embed.t());
        return out.into_shared();
    }

    embed
}

/// Walk per-layer FFN tensors and ensure they're in canonical orientation.
///
/// Canonical (Llama / nn.Linear convention):
/// - gate / up:  shape `(intermediate, hidden)`
/// - down:       shape `(hidden, intermediate)`
///
/// Some GGUF converters (notably non-standard GPT-2 builds where Conv1D
/// weights weren't transposed) store FFN weights in the inverse layout.
/// If a tensor's loaded shape matches the inverse of the canonical
/// orientation — and the two dimensions differ so orientation is
/// unambiguous — transpose it. Otherwise leave it untouched.
///
/// Driven entirely by `ModelArchitecture` keys and `ModelConfig` dimensions
/// — no family-specific branching.
fn orient_ffn_tensors(
    tensors: &mut HashMap<String, crate::WeightArray>,
    arch: &dyn crate::config::ModelArchitecture,
) {
    let cfg = arch.config();
    let hidden = cfg.hidden_size;
    let dense_inter = cfg.intermediate_size;
    if cfg.num_layers == 0 || hidden == 0 {
        return;
    }

    let moe_inter = if arch.is_moe() || arch.is_hybrid_moe() {
        let m = arch.moe_intermediate_size();
        (m > 0).then_some(m)
    } else {
        None
    };
    let n_experts = if moe_inter.is_some() {
        arch.num_experts()
    } else {
        0
    };

    for layer in 0..cfg.num_layers {
        // Dense FFN tensors
        if dense_inter > 0 {
            orient_in_place(tensors, &arch.ffn_gate_key(layer), dense_inter, hidden);
            orient_in_place(tensors, &arch.ffn_up_key(layer), dense_inter, hidden);
            orient_in_place(tensors, &arch.ffn_down_key(layer), hidden, dense_inter);
        }

        // Shared-expert FFN tensors share dense intermediate dim.
        if dense_inter > 0 {
            if let Some(key) = arch.shared_expert_gate_key(layer) {
                orient_in_place(tensors, &key, dense_inter, hidden);
            }
            if let Some(key) = arch.shared_expert_up_key(layer) {
                orient_in_place(tensors, &key, dense_inter, hidden);
            }
            if let Some(key) = arch.shared_expert_down_key(layer) {
                orient_in_place(tensors, &key, hidden, dense_inter);
            }
        }

        // Per-expert MoE FFN tensors use the per-expert intermediate dim.
        if let Some(mf) = moe_inter {
            for expert in 0..n_experts {
                if let Some(key) = arch.expert_ffn_gate_key(layer, expert) {
                    orient_in_place(tensors, &key, mf, hidden);
                }
                if let Some(key) = arch.expert_ffn_up_key(layer, expert) {
                    orient_in_place(tensors, &key, mf, hidden);
                }
                if let Some(key) = arch.expert_ffn_down_key(layer, expert) {
                    orient_in_place(tensors, &key, hidden, mf);
                }
            }
        }
    }
}

/// Transpose `tensors[key]` if it's currently shaped `(expected_cols, expected_rows)`
/// while the canonical shape is `(expected_rows, expected_cols)`. No-op when the
/// tensor is missing, already canonical, the dimensions are equal (ambiguous),
/// or the shape matches neither orientation.
fn orient_in_place(
    tensors: &mut HashMap<String, crate::WeightArray>,
    key: &str,
    expected_rows: usize,
    expected_cols: usize,
) {
    if expected_rows == 0 || expected_cols == 0 || expected_rows == expected_cols {
        return;
    }
    let arr = match tensors.get(key) {
        Some(a) => a,
        None => return,
    };
    let shape = arr.shape();
    if shape.len() != 2 {
        return;
    }
    if shape[0] == expected_rows && shape[1] == expected_cols {
        return;
    }
    if shape[0] == expected_cols && shape[1] == expected_rows {
        let mut out = ndarray::Array2::<f32>::zeros((expected_rows, expected_cols));
        out.assign(&arr.t());
        tensors.insert(key.to_string(), out.into_shared());
    }
}

/// Walk per-layer attention tensors and ensure they're in canonical orientation.
///
/// Canonical (Linear convention):
/// - q_proj:   shape `(num_q_heads * head_dim, hidden_size)`
/// - k_proj:   shape `(num_kv_heads * head_dim, hidden_size)`
/// - v_proj:   shape `(num_kv_heads * head_dim, hidden_size)`
/// - o_proj:   shape `(hidden_size, num_q_heads * head_dim)`
/// - qkv_proj: shape `(q_dim + 2 * kv_dim, hidden_size)` — used by fused-QKV
///   architectures (GPT-2). Split happens in `split_fused_qkv` after this.
///
/// `orient_in_place` is a no-op when the two dimensions are equal, so square
/// tensors (e.g. GPT-2 with `q_dim == kv_dim == hidden`) survive untouched.
/// The fused-QKV tensor is asymmetric (`3*hidden vs hidden`) and orientable.
fn orient_attention_tensors(
    tensors: &mut HashMap<String, crate::WeightArray>,
    arch: &dyn crate::config::ModelArchitecture,
) {
    let cfg = arch.config();
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    if cfg.num_layers == 0 || hidden == 0 || head_dim == 0 {
        return;
    }
    let q_dim = cfg.num_q_heads * head_dim;
    let kv_dim = cfg.num_kv_heads * head_dim;

    for layer in 0..cfg.num_layers {
        if q_dim > 0 {
            orient_in_place(tensors, &arch.attn_q_key(layer), q_dim, hidden);
            orient_in_place(tensors, &arch.attn_o_key(layer), hidden, q_dim);
        }
        if kv_dim > 0 {
            orient_in_place(tensors, &arch.attn_k_key(layer), kv_dim, hidden);
            orient_in_place(tensors, &arch.attn_v_key(layer), kv_dim, hidden);
        }
        if let Some(key) = arch.fused_qkv_key(layer) {
            let total = q_dim + 2 * kv_dim;
            if total > 0 {
                orient_in_place(tensors, &key, total, hidden);
            }
        }
    }
}

/// Materialise per-projection q/k/v tensors (and biases) from a fused QKV
/// matrix, when the architecture declares one via `fused_qkv_key`.
///
/// The fused weight is assumed to be in canonical orientation
/// `(q_dim + 2 * kv_dim, hidden_size)` — `orient_attention_tensors` runs
/// first to enforce that. Rows split into:
/// - `0 .. q_dim`                       → `attn_q_key`
/// - `q_dim .. q_dim + kv_dim`          → `attn_k_key`
/// - `q_dim + kv_dim .. q_dim + 2*kv_dim` → `attn_v_key`
///
/// The fused bias (1D, length `q_dim + 2 * kv_dim`) splits identically into
/// the per-projection bias keys returned by the trait.
///
/// Driven entirely by `ModelArchitecture` keys + `ModelConfig` dimensions —
/// no family-specific branching.
fn split_fused_qkv(
    tensors: &mut HashMap<String, crate::WeightArray>,
    vectors: &mut HashMap<String, Vec<f32>>,
    arch: &dyn crate::config::ModelArchitecture,
) {
    let cfg = arch.config();
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    if cfg.num_layers == 0 || hidden == 0 || head_dim == 0 {
        return;
    }
    let q_dim = cfg.num_q_heads * head_dim;
    let kv_dim = cfg.num_kv_heads * head_dim;
    let total = q_dim + 2 * kv_dim;
    if total == 0 {
        return;
    }

    for layer in 0..cfg.num_layers {
        let Some(weight_key) = arch.fused_qkv_key(layer) else {
            continue;
        };

        if let Some(fused) = tensors.remove(&weight_key) {
            let shape = fused.shape();
            if shape.len() == 2 && shape[0] == total && shape[1] == hidden {
                if q_dim > 0 {
                    let q = fused.slice(ndarray::s![..q_dim, ..]).to_owned();
                    tensors.insert(arch.attn_q_key(layer), q.into_shared());
                }
                if kv_dim > 0 {
                    let k = fused
                        .slice(ndarray::s![q_dim..q_dim + kv_dim, ..])
                        .to_owned();
                    let v = fused
                        .slice(ndarray::s![q_dim + kv_dim..total, ..])
                        .to_owned();
                    tensors.insert(arch.attn_k_key(layer), k.into_shared());
                    tensors.insert(arch.attn_v_key(layer), v.into_shared());
                }
            } else {
                // Shape doesn't match expected fused layout — put it back so
                // the caller can surface the mismatch via missing-tensor errors.
                tensors.insert(weight_key, fused);
            }
        }

        if let Some(bias_key) = arch.fused_qkv_bias_key(layer) {
            if let Some(fused_b) = vectors.remove(&bias_key) {
                if fused_b.len() == total {
                    if let (Some(qb_key), true) = (arch.attn_q_bias_key(layer), q_dim > 0) {
                        vectors.insert(qb_key, fused_b[..q_dim].to_vec());
                    }
                    if kv_dim > 0 {
                        if let Some(kb_key) = arch.attn_k_bias_key(layer) {
                            vectors.insert(kb_key, fused_b[q_dim..q_dim + kv_dim].to_vec());
                        }
                        if let Some(vb_key) = arch.attn_v_bias_key(layer) {
                            vectors.insert(vb_key, fused_b[q_dim + kv_dim..total].to_vec());
                        }
                    }
                } else {
                    vectors.insert(bias_key, fused_b);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// GGUF binary reading helpers
// ═══════════════════════════════════════════════════════════════

fn read_u8(r: &mut impl Read) -> Result<u8, ModelError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(r: &mut impl Read) -> Result<i8, ModelError> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut impl Read) -> Result<u16, ModelError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(r: &mut impl Read) -> Result<i16, ModelError> {
    Ok(read_u16(r)? as i16)
}

fn read_u32(r: &mut impl Read) -> Result<u32, ModelError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> Result<i32, ModelError> {
    Ok(read_u32(r)? as i32)
}

fn read_u64(r: &mut impl Read) -> Result<u64, ModelError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> Result<i64, ModelError> {
    Ok(read_u64(r)? as i64)
}

fn read_f32(r: &mut impl Read) -> Result<f32, ModelError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> Result<f64, ModelError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string(r: &mut impl Read) -> Result<String, ModelError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| ModelError::Parse(e.to_string()))
}

fn read_value(r: &mut impl Read) -> Result<GgufValue, ModelError> {
    let vtype = read_u32(r)?;
    match vtype {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let len = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_array_element(r, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        _ => Err(ModelError::Parse(format!(
            "unknown GGUF metadata type: {vtype}"
        ))),
    }
}

fn read_array_element(r: &mut impl Read, elem_type: u32) -> Result<GgufValue, ModelError> {
    match elem_type {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        _ => Err(ModelError::Parse(format!(
            "unknown GGUF array element type: {elem_type}"
        ))),
    }
}

// ═══════════════════════════════════════════════════════════════
// Dequantization — delegates to format::quant module
// ═══════════════════════════════════════════════════════════════

fn tensor_data_size(tensor_type: u32, n_elements: usize) -> Result<usize, ModelError> {
    crate::quant::ggml::tensor_data_size(tensor_type, n_elements)
}

fn dequantize(data: &[u8], tensor_type: u32, n_elements: usize) -> Result<Vec<f32>, ModelError> {
    crate::quant::ggml::dequantize(data, tensor_type, n_elements)
}

/// Normalize GGUF tensor key names to match HuggingFace conventions.
pub fn normalize_gguf_key(name: &str) -> String {
    // GGUF uses "blk.N.attn_q.weight" format
    // HF uses "model.layers.N.self_attn.q_proj.weight" format
    // We normalize to the HF style since that's what ModelArchitecture expects

    GGUF_TO_HF_KEY_REPLACEMENTS
        .iter()
        .fold(name.to_string(), |acc, (from, to)| acc.replace(from, to))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orient_in_place_transposes_inverse_layout() {
        use ndarray::Array2;

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        // Inverse layout: stored (cols, rows) when canonical is (rows, cols).
        // Canonical for ffn_down is (hidden, intermediate).
        let stored = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_shared();
        tensors.insert("layers.0.mlp.down_proj.weight".to_string(), stored);

        // Canonical (hidden=2, intermediate=3): expect shape (2, 3) after orient.
        orient_in_place(&mut tensors, "layers.0.mlp.down_proj.weight", 2, 3);

        let oriented = tensors.get("layers.0.mlp.down_proj.weight").unwrap();
        assert_eq!(oriented.shape(), &[2, 3]);
        // Transpose maps (i,j) → (j,i): row-major buffer becomes 1,3,5,2,4,6.
        assert_eq!(oriented[[0, 0]], 1.0);
        assert_eq!(oriented[[0, 1]], 3.0);
        assert_eq!(oriented[[0, 2]], 5.0);
        assert_eq!(oriented[[1, 0]], 2.0);
        assert_eq!(oriented[[1, 1]], 4.0);
        assert_eq!(oriented[[1, 2]], 6.0);
    }

    #[test]
    fn test_orient_in_place_leaves_canonical_layout_untouched() {
        use ndarray::Array2;

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let canonical = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_shared();
        let original_ptr = canonical.as_ptr();
        tensors.insert("layers.0.mlp.down_proj.weight".to_string(), canonical);

        orient_in_place(&mut tensors, "layers.0.mlp.down_proj.weight", 2, 3);

        let after = tensors.get("layers.0.mlp.down_proj.weight").unwrap();
        // No clone-and-replace: same backing buffer.
        assert_eq!(after.as_ptr(), original_ptr);
    }

    #[test]
    fn test_orient_in_place_skips_ambiguous_square_dims() {
        use ndarray::Array2;

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let square = Array2::from_shape_vec((4, 4), (0..16).map(|x| x as f32).collect())
            .unwrap()
            .into_shared();
        tensors.insert("layers.0.mlp.up_proj.weight".to_string(), square);

        orient_in_place(&mut tensors, "layers.0.mlp.up_proj.weight", 4, 4);

        let after = tensors.get("layers.0.mlp.up_proj.weight").unwrap();
        // Untouched — orientation can't be inferred when rows == cols.
        assert_eq!(after.shape(), &[4, 4]);
        assert_eq!(after[[0, 0]], 0.0);
        assert_eq!(after[[3, 3]], 15.0);
    }

    /// Build a minimal Gpt2-shaped ModelConfig for orientation/split tests.
    fn synth_gpt2_config(
        num_layers: usize,
        hidden: usize,
        head_dim: usize,
        n_heads: usize,
    ) -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            model_type: "gpt2".into(),
            norm_eps: None,
            num_layers,
            hidden_size: hidden,
            intermediate_size: 4 * hidden,
            head_dim,
            num_q_heads: n_heads,
            num_kv_heads: n_heads,
            vocab_size: Some(8),
            rope_base: 10_000.0,
            rope_local_base: None,
            sliding_window: None,
            num_experts: None,
            num_experts_per_token: None,
            num_shared_experts: None,
            enable_moe_block: false,
            top_k_experts: None,
            moe_intermediate_size: None,
            kv_lora_rank: None,
            q_lora_rank: None,
            rope_scaling: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            embedding_multiplier: None,
            residual_multiplier: None,
            attention_multiplier: None,
            logits_scaling: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            per_layer_embed_dim: None,
            num_kv_shared_layers: None,
        }
    }

    #[test]
    fn test_orient_attention_tensors_fixes_inverse_fused_qkv_layout() {
        use ndarray::Array2;

        // hidden=4, head_dim=2, n_heads=2 → q_dim=kv_dim=4, total=12.
        let cfg = synth_gpt2_config(1, 4, 2, 2);
        let arch = crate::architectures::gpt2::Gpt2Arch::from_config(cfg);

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        // Inverse layout: stored (hidden=4, total=12) instead of (12, 4).
        let inverse = Array2::<f32>::zeros((4, 12)).into_shared();
        tensors.insert("layers.0.self_attn.qkv_proj.weight".into(), inverse);

        orient_attention_tensors(&mut tensors, &arch);

        let oriented = tensors.get("layers.0.self_attn.qkv_proj.weight").unwrap();
        assert_eq!(oriented.shape(), &[12, 4]);
    }

    #[test]
    fn test_split_fused_qkv_materialises_per_projection_tensors_and_biases() {
        use ndarray::Array2;

        // hidden=4, head_dim=2, n_heads=2 → q_dim=kv_dim=4, total=12.
        let cfg = synth_gpt2_config(1, 4, 2, 2);
        let arch = crate::architectures::gpt2::Gpt2Arch::from_config(cfg);

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

        // Fused weight: row r has constant value r so we can verify slices.
        let mut data = Vec::with_capacity(12 * 4);
        for r in 0..12 {
            for _c in 0..4 {
                data.push(r as f32);
            }
        }
        let fused_w = Array2::from_shape_vec((12, 4), data).unwrap().into_shared();
        tensors.insert("layers.0.self_attn.qkv_proj.weight".into(), fused_w);

        // Fused bias: 12 distinct values.
        let fused_b: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        vectors.insert("layers.0.self_attn.qkv_proj.bias".into(), fused_b);

        split_fused_qkv(&mut tensors, &mut vectors, &arch);

        // Fused tensor + bias removed.
        assert!(!tensors.contains_key("layers.0.self_attn.qkv_proj.weight"));
        assert!(!vectors.contains_key("layers.0.self_attn.qkv_proj.bias"));

        let q = tensors.get("layers.0.self_attn.q_proj.weight").unwrap();
        let k = tensors.get("layers.0.self_attn.k_proj.weight").unwrap();
        let v = tensors.get("layers.0.self_attn.v_proj.weight").unwrap();
        assert_eq!(q.shape(), &[4, 4]);
        assert_eq!(k.shape(), &[4, 4]);
        assert_eq!(v.shape(), &[4, 4]);
        // Row r maps to constant r in the fused layout. q rows 0..4, k 4..8, v 8..12.
        assert_eq!(q[[0, 0]], 0.0);
        assert_eq!(q[[3, 3]], 3.0);
        assert_eq!(k[[0, 0]], 4.0);
        assert_eq!(k[[3, 3]], 7.0);
        assert_eq!(v[[0, 0]], 8.0);
        assert_eq!(v[[3, 3]], 11.0);

        let qb = vectors.get("layers.0.self_attn.q_proj.bias").unwrap();
        let kb = vectors.get("layers.0.self_attn.k_proj.bias").unwrap();
        let vb = vectors.get("layers.0.self_attn.v_proj.bias").unwrap();
        assert_eq!(qb.len(), 4);
        assert_eq!(kb.len(), 4);
        assert_eq!(vb.len(), 4);
        assert!((qb[0] - 0.0).abs() < 1e-6);
        assert!((kb[0] - 0.4).abs() < 1e-6);
        assert!((vb[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_split_fused_qkv_no_op_when_arch_has_no_fused_key() {
        use ndarray::Array2;

        // Llama-style arch — no fused QKV.
        let cfg = synth_gpt2_config(1, 4, 2, 2);
        let arch = crate::architectures::llama::LlamaArch::from_config(cfg);

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
        let q = Array2::<f32>::zeros((4, 4)).into_shared();
        tensors.insert("layers.0.self_attn.q_proj.weight".into(), q);

        split_fused_qkv(&mut tensors, &mut vectors, &arch);

        // Untouched.
        assert!(tensors.contains_key("layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn test_orient_ffn_tensors_fixes_gpt2_style_inverse_layout() {
        use crate::config::ModelConfig;
        use ndarray::Array2;

        let cfg = ModelConfig {
            model_type: "gpt2".into(),
            norm_eps: None,
            num_layers: 1,
            hidden_size: 4,
            intermediate_size: 12,
            head_dim: 2,
            num_q_heads: 2,
            num_kv_heads: 2,
            vocab_size: Some(8),
            rope_base: 10_000.0,
            rope_local_base: None,
            sliding_window: None,
            num_experts: None,
            num_experts_per_token: None,
            num_shared_experts: None,
            enable_moe_block: false,
            top_k_experts: None,
            moe_intermediate_size: None,
            kv_lora_rank: None,
            q_lora_rank: None,
            rope_scaling: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            embedding_multiplier: None,
            residual_multiplier: None,
            attention_multiplier: None,
            logits_scaling: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            per_layer_embed_dim: None,
            num_kv_shared_layers: None,
        };
        let arch = crate::architectures::gpt2::Gpt2Arch::from_config(cfg);

        // Inverse layouts: ffn_up stored (hidden, inter) instead of (inter, hidden);
        // ffn_down stored (inter, hidden) instead of (hidden, inter).
        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let up_inverse = Array2::<f32>::zeros((4, 12)).into_shared();
        let down_inverse = Array2::<f32>::zeros((12, 4)).into_shared();
        tensors.insert("layers.0.mlp.up_proj.weight".into(), up_inverse);
        tensors.insert("layers.0.mlp.down_proj.weight".into(), down_inverse);

        orient_ffn_tensors(&mut tensors, &arch);

        let up = tensors.get("layers.0.mlp.up_proj.weight").unwrap();
        let down = tensors.get("layers.0.mlp.down_proj.weight").unwrap();
        assert_eq!(up.shape(), &[12, 4]);
        assert_eq!(down.shape(), &[4, 12]);
    }

    #[test]
    fn test_normalize_gguf_key() {
        assert_eq!(
            normalize_gguf_key("blk.0.attn_q.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.15.ffn_gate.weight"),
            "layers.15.mlp.gate_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("token_embd.weight"),
            "embed_tokens.weight"
        );
        assert_eq!(normalize_gguf_key("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_load_tensors_swaps_gguf_2d_dims_to_rows_cols() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        let mut file = std::fs::File::create(&path).unwrap();

        // Header
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap(); // version
        file.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors
        file.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata

        // Tensor info: ggml dims order is [cols, rows].
        let name = b"blk.0.ffn_down.weight";
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims
        file.write_all(&4u64.to_le_bytes()).unwrap(); // cols
        file.write_all(&2u64.to_le_bytes()).unwrap(); // rows
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap(); // tensor data offset

        // Pad tensor data start to 32-byte boundary.
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();

        // Raw row-major data for a logical [2, 4] matrix.
        for v in 1u32..=8 {
            file.write_all(&(v as f32).to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let (tensors, _) = gguf.load_tensors().unwrap();
        let down = tensors.get("layers.0.mlp.down_proj.weight").unwrap();

        assert_eq!(down.shape(), &[2, 4]);
        assert_eq!(down[[0, 0]], 1.0);
        assert_eq!(down[[0, 1]], 2.0);
        assert_eq!(down[[0, 2]], 3.0);
        assert_eq!(down[[0, 3]], 4.0);
        assert_eq!(down[[1, 0]], 5.0);
        assert_eq!(down[[1, 1]], 6.0);
        assert_eq!(down[[1, 2]], 7.0);
        assert_eq!(down[[1, 3]], 8.0);
    }

    #[test]
    fn test_gemma4_gguf_to_config_json_maps_arch_and_overrides_head_dim() {
        // Synthesize GGUF metadata matching gemma-4-e2b's shape.
        // Exercises: (a) gemma4 name pass-through, (b) head_dim=256 override,
        // (c) array metadata (per-layer variable FFN sizes → take max).
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("gemma4".to_string()),
        );
        metadata.insert("gemma4.embedding_length".to_string(), GgufValue::U32(1536));
        metadata.insert("gemma4.block_count".to_string(), GgufValue::U32(35));
        metadata.insert("gemma4.attention.head_count".to_string(), GgufValue::U32(8));
        metadata.insert(
            "gemma4.attention.head_count_kv".to_string(),
            GgufValue::U32(1),
        );
        // Gemma 4 reports attention.key_length=512 (global head_dim), not the
        // per-head 256 we want. Loader must override to 256 for arch="gemma4".
        metadata.insert(
            "gemma4.attention.key_length".to_string(),
            GgufValue::U32(512),
        );
        metadata.insert("gemma4.vocab_size".to_string(), GgufValue::U32(262144));
        // Per-layer variable FFN — some layers 6144, some 12288. Must take max.
        metadata.insert(
            "gemma4.feed_forward_length".to_string(),
            GgufValue::Array(vec![
                GgufValue::U32(6144),
                GgufValue::U32(12288),
                GgufValue::U32(6144),
            ]),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
        };
        let cfg = gguf.to_config_json();

        assert_eq!(cfg["model_type"], "gemma4");
        assert_eq!(cfg["hidden_size"], 1536);
        assert_eq!(cfg["num_hidden_layers"], 35);
        // head_dim override: 256 despite attention.key_length=512
        assert_eq!(cfg["head_dim"], 256);
        // intermediate_size: max of the per-layer FFN array (12288), not 6144
        assert_eq!(cfg["intermediate_size"], 12288);
        assert_eq!(cfg["num_attention_heads"], 8);
        assert_eq!(cfg["num_key_value_heads"], 1);
        assert_eq!(cfg["vocab_size"], 262144);
    }

    #[test]
    fn test_gguf_to_config_json_omits_absent_rope_base_for_arch_default() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
        metadata.insert("llama.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );
        metadata.insert(
            "llama.attention.key_length".to_string(),
            GgufValue::U32(128),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
        };
        let cfg = gguf.to_config_json();

        assert!(cfg.get(HF_ROPE_THETA).is_none());
        let arch = crate::detect_from_json_validated(&cfg).unwrap();
        assert_eq!(arch.config().rope_base, 10_000.0);
    }

    /// Build a minimal GGUF file with one 2-D F32 tensor, but truncate the
    /// tensor data region so that `offset + size > file len`. Loader must
    /// reject this cleanly, not panic on a slice OOB.
    #[test]
    fn test_load_tensors_rejects_truncated_tensor_data() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.gguf");
        let mut file = std::fs::File::create(&path).unwrap();

        // Header
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap(); // version
        file.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors
        file.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata

        // Tensor info: declares 2×4 F32 (32 bytes of data) at tensor offset 0.
        let name = b"blk.0.ffn_down.weight";
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&4u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();

        // Pad to 32-byte boundary, then write only 16 bytes of tensor data
        // (half of the declared 32). Loader must detect the shortfall.
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        file.write_all(&[0u8; 16]).unwrap();
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        match gguf.load_tensors() {
            Err(ModelError::Parse(msg)) => {
                assert!(
                    msg.contains("out of bounds") || msg.contains("too short"),
                    "unexpected error: {msg}"
                );
            }
            Err(other) => panic!("expected Parse error, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    // Dequant tests are in format::quant::ggml::tests

    // ─────────────────────────────────────────────────────────────────
    // Byte-reading helpers + read_value/read_array_element variant
    // coverage. The existing GGUF-builder tests only emit STRING / U32 /
    // FLOAT32 metadata; the read-side dispatch arms for U8, I8, U16,
    // I16, I32, U64, I64, F64, Bool, ARRAY, and the unknown-type error
    // branch are exercised here directly.
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn read_value_dispatches_every_supported_variant() {
        use std::io::Cursor;
        // U8 (tag 0): tag(u32) + 1 byte payload.
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_UINT8.to_le_bytes());
        buf.push(0xAB);
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::U8(0xAB)
        ));

        // I8 (tag 1).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_INT8.to_le_bytes());
        buf.push(0xFFu8); // -1 as i8
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::I8(-1)
        ));

        // U16 (tag 2).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_UINT16.to_le_bytes());
        buf.extend_from_slice(&12345u16.to_le_bytes());
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::U16(12345)
        ));

        // I16 (tag 3).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_INT16.to_le_bytes());
        buf.extend_from_slice(&(-7i16).to_le_bytes());
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::I16(-7)
        ));

        // I32 (tag 5).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_INT32.to_le_bytes());
        buf.extend_from_slice(&(-65_536i32).to_le_bytes());
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::I32(-65_536)
        ));

        // BOOL (tag 7): tag + 1 byte (0 = false, nonzero = true).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_BOOL.to_le_bytes());
        buf.push(1u8);
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::Bool(true)
        ));
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_BOOL.to_le_bytes());
        buf.push(0u8);
        assert!(matches!(
            read_value(&mut Cursor::new(buf)).unwrap(),
            GgufValue::Bool(false)
        ));

        // U64 (tag 10).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_UINT64.to_le_bytes());
        buf.extend_from_slice(&(u64::MAX - 3).to_le_bytes());
        let v = read_value(&mut Cursor::new(buf)).unwrap();
        match v {
            GgufValue::U64(x) => assert_eq!(x, u64::MAX - 3),
            other => panic!("expected U64, got {other:?}"),
        }

        // I64 (tag 11).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_INT64.to_le_bytes());
        buf.extend_from_slice(&(-9_999_999i64).to_le_bytes());
        let v = read_value(&mut Cursor::new(buf)).unwrap();
        match v {
            GgufValue::I64(x) => assert_eq!(x, -9_999_999),
            other => panic!("expected I64, got {other:?}"),
        }

        // F64 (tag 12).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_FLOAT64.to_le_bytes());
        buf.extend_from_slice(&std::f64::consts::PI.to_le_bytes());
        let v = read_value(&mut Cursor::new(buf)).unwrap();
        match v {
            GgufValue::F64(x) => {
                assert!((x - std::f64::consts::PI).abs() < 1e-12);
            }
            other => panic!("expected F64, got {other:?}"),
        }
    }

    #[test]
    fn read_value_array_recurses_through_read_array_element() {
        use std::io::Cursor;
        // Array of 3 U32 values.
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
        buf.extend_from_slice(&GGUF_TYPE_UINT32.to_le_bytes());
        buf.extend_from_slice(&3u64.to_le_bytes());
        for v in [10u32, 20, 30] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        match read_value(&mut Cursor::new(buf)).unwrap() {
            GgufValue::Array(elems) => {
                assert_eq!(elems.len(), 3);
                assert!(matches!(elems[0], GgufValue::U32(10)));
                assert!(matches!(elems[1], GgufValue::U32(20)));
                assert!(matches!(elems[2], GgufValue::U32(30)));
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn read_array_element_dispatches_every_supported_variant() {
        use std::io::Cursor;

        type VariantCase = (u32, Vec<u8>, fn(GgufValue));
        let cases: &[VariantCase] = &[
            (GGUF_TYPE_UINT8, vec![0x42], |v| {
                assert!(matches!(v, GgufValue::U8(0x42)))
            }),
            (GGUF_TYPE_INT8, vec![0xFE], |v| {
                assert!(matches!(v, GgufValue::I8(-2)))
            }),
            (GGUF_TYPE_UINT16, 500u16.to_le_bytes().to_vec(), |v| {
                assert!(matches!(v, GgufValue::U16(500)))
            }),
            (GGUF_TYPE_INT16, (-9i16).to_le_bytes().to_vec(), |v| {
                assert!(matches!(v, GgufValue::I16(-9)))
            }),
            (GGUF_TYPE_UINT32, 7u32.to_le_bytes().to_vec(), |v| {
                assert!(matches!(v, GgufValue::U32(7)))
            }),
            (GGUF_TYPE_INT32, (-77_777i32).to_le_bytes().to_vec(), |v| {
                assert!(matches!(v, GgufValue::I32(-77_777)))
            }),
            (
                GGUF_TYPE_FLOAT32,
                2.5f32.to_le_bytes().to_vec(),
                |v| match v {
                    GgufValue::F32(x) => assert_eq!(x, 2.5),
                    other => panic!("expected F32, got {other:?}"),
                },
            ),
            (GGUF_TYPE_BOOL, vec![1u8], |v| {
                assert!(matches!(v, GgufValue::Bool(true)))
            }),
            (
                GGUF_TYPE_UINT64,
                12345u64.to_le_bytes().to_vec(),
                |v| match v {
                    GgufValue::U64(x) => assert_eq!(x, 12345),
                    other => panic!("expected U64, got {other:?}"),
                },
            ),
            (
                GGUF_TYPE_INT64,
                (-1234i64).to_le_bytes().to_vec(),
                |v| match v {
                    GgufValue::I64(x) => assert_eq!(x, -1234),
                    other => panic!("expected I64, got {other:?}"),
                },
            ),
            (
                GGUF_TYPE_FLOAT64,
                1.5f64.to_le_bytes().to_vec(),
                |v| match v {
                    GgufValue::F64(x) => assert_eq!(x, 1.5),
                    other => panic!("expected F64, got {other:?}"),
                },
            ),
        ];

        for (tag, bytes, check) in cases {
            let v = read_array_element(&mut Cursor::new(bytes.clone()), *tag).unwrap();
            check(v);
        }
    }

    #[test]
    fn read_array_element_string_variant() {
        use std::io::Cursor;
        let mut buf = Vec::new();
        buf.extend_from_slice(&6u64.to_le_bytes());
        buf.extend_from_slice(b"hello!");
        match read_array_element(&mut Cursor::new(buf), GGUF_TYPE_STRING).unwrap() {
            GgufValue::String(s) => assert_eq!(s, "hello!"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn read_value_unknown_metadata_type_errors() {
        use std::io::Cursor;
        let mut buf = Vec::new();
        buf.extend_from_slice(&999u32.to_le_bytes());
        match read_value(&mut Cursor::new(buf)) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("unknown GGUF metadata type"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn read_array_element_unknown_type_errors() {
        use std::io::Cursor;
        match read_array_element(&mut Cursor::new(Vec::new()), 9999) {
            Err(ModelError::Parse(msg)) => {
                assert!(
                    msg.contains("unknown GGUF array element type"),
                    "got: {msg}"
                );
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn read_string_rejects_non_utf8() {
        use std::io::Cursor;
        let mut buf = Vec::new();
        buf.extend_from_slice(&3u64.to_le_bytes());
        buf.extend_from_slice(&[0xFF, 0xFE, 0xFD]); // invalid UTF-8
        match read_string(&mut Cursor::new(buf)) {
            Err(ModelError::Parse(_)) => {}
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    // GgufValue::as_* — coverage for the tiny accessor impls.

    #[test]
    fn gguf_value_as_u32_handles_three_int_variants() {
        assert_eq!(GgufValue::U32(7).as_u32(), Some(7));
        assert_eq!(GgufValue::I32(-1).as_u32(), Some(u32::MAX));
        assert_eq!(GgufValue::U64(42).as_u32(), Some(42));
        assert_eq!(GgufValue::String("x".into()).as_u32(), None);
    }

    #[test]
    fn gguf_value_as_str_returns_string_payload() {
        assert_eq!(GgufValue::String("hi".into()).as_str(), Some("hi"));
        assert_eq!(GgufValue::U32(1).as_str(), None);
    }

    #[test]
    fn gguf_value_as_f64_widens_f32_and_returns_f64_payload() {
        assert_eq!(GgufValue::F32(1.5).as_f64(), Some(1.5));
        assert_eq!(GgufValue::F64(2.5).as_f64(), Some(2.5));
        assert_eq!(GgufValue::U32(1).as_f64(), None);
    }

    #[test]
    fn read_tokenizer_vocab_size_reads_vocab_object_length() {
        let dir = tempfile::TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        let tokenizer_json = serde_json::json!({
            TOKENIZER_MODEL: {
                TOKENIZER_VOCAB: {
                    "<unk>": 0,
                    "<bos>": 1,
                    "<eos>": 2,
                    "a": 3,
                    "b": 4,
                }
            }
        });
        std::fs::write(dir.path().join(TOKENIZER_JSON), tokenizer_json.to_string()).unwrap();
        assert_eq!(read_tokenizer_vocab_size(&gguf), Some(5));
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_when_tokenizer_json_absent() {
        let dir = tempfile::TempDir::new().unwrap();
        // model.gguf path with no tokenizer.json next to it.
        assert_eq!(
            read_tokenizer_vocab_size(&dir.path().join("model.gguf")),
            None
        );
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_when_vocab_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        // Empty vocab object — filtered out by `.filter(|&v| v > 0)`.
        let tokenizer_json = serde_json::json!({
            TOKENIZER_MODEL: {
                TOKENIZER_VOCAB: {}
            }
        });
        std::fs::write(dir.path().join(TOKENIZER_JSON), tokenizer_json.to_string()).unwrap();
        assert_eq!(read_tokenizer_vocab_size(&gguf), None);
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_on_malformed_json() {
        let dir = tempfile::TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        std::fs::write(dir.path().join(TOKENIZER_JSON), b"not-json").unwrap();
        assert_eq!(read_tokenizer_vocab_size(&gguf), None);
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_when_path_has_no_parent() {
        // PathBuf::new() has no parent — exercises the early-return at L525.
        assert_eq!(read_tokenizer_vocab_size(std::path::Path::new("")), None);
    }
}
