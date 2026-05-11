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

// ── Qwen 3.6 (qwen35 / qwen35moe) Gated DeltaNet metadata keys ─────────────
// All are `<arch>.<key>` per GGUF convention; the loader scopes by prefix.
// `pub(crate)` so the Qwen35Arch handler (Phase B) can use the same source
// of truth as the loader.
#[allow(dead_code)]
pub(crate) const GGUF_FULL_ATTENTION_INTERVAL: &str = "full_attention_interval";
#[allow(dead_code)]
pub(crate) const GGUF_SSM_STATE_SIZE: &str = "ssm.state_size";
#[allow(dead_code)]
pub(crate) const GGUF_SSM_INNER_SIZE: &str = "ssm.inner_size";
#[allow(dead_code)]
pub(crate) const GGUF_SSM_DT_RANK: &str = "ssm.time_step_rank"; // = n_v_heads
#[allow(dead_code)]
pub(crate) const GGUF_SSM_GROUP_COUNT: &str = "ssm.group_count"; // = n_k_heads
#[allow(dead_code)]
pub(crate) const GGUF_SSM_CONV_KERNEL: &str = "ssm.conv_kernel"; // = d_conv
#[allow(dead_code)]
pub(crate) const GGUF_ROPE_DIMENSION_SECTIONS: &str = "rope.dimension_sections";

// Per-layer DeltaNet / Qwen3-Next tensor name suffixes (used by the
// `Qwen35Arch` handler landing in Phase B; defined here so the GGUF→
// vindex key normaliser has one source of truth).
#[allow(dead_code)]
pub(crate) const GGUF_TENSOR_ATTN_QKV: &str = "attn_qkv"; // fused Q+K+V projection
#[allow(dead_code)]
pub(crate) const GGUF_TENSOR_ATTN_GATE: &str = "attn_gate"; // Z gate (DeltaNet) / Q-gate (full-attn fused)
#[allow(dead_code)]
pub(crate) const GGUF_TENSOR_SSM_CONV1D: &str = "ssm_conv1d"; // depthwise Conv1D over QKV
#[allow(dead_code)]
pub(crate) const GGUF_TENSOR_SSM_DT: &str = "ssm_dt"; // bias added to alpha
#[allow(dead_code)]
pub(crate) const GGUF_TENSOR_SSM_A: &str = "ssm_a"; // per-head log-decay
#[allow(dead_code)]
pub(crate) const GGUF_TENSOR_SSM_BETA: &str = "ssm_beta"; // delta-rule learning-rate proj
#[allow(dead_code)]
pub(crate) const GGUF_TENSOR_SSM_ALPHA: &str = "ssm_alpha"; // pre-softplus gate proj
#[allow(dead_code)]
pub(crate) const GGUF_TENSOR_SSM_NORM: &str = "ssm_norm"; // post-mixer RMSNorm
#[allow(dead_code)]
pub(crate) const GGUF_TENSOR_SSM_OUT: &str = "ssm_out"; // output projection

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

const GGUF_TO_HF_KEY_REPLACEMENTS: &[(&str, &str)] = &[
    ("blk.", "layers."),
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

        // Map GGUF architecture names to HF model_type.
        //
        // `qwen35` (Qwen 3.6 dense, 27B) and `qwen35moe` (Qwen 3.6 MoE,
        // 35B-A3B) are **hybrid Gated DeltaNet + full-attention**
        // architectures, NOT pure transformer Qwen3. They preserve the
        // qwen-family prefix so detect.rs's `t.starts_with("qwen")` route
        // still picks them up, but downstream (per the
        // `inference-qwen35-deltanet` openspec change) they SHALL be
        // handled by a `Qwen35Arch` / `Qwen35MoeArch` rather than the
        // pure-transformer `QwenArch`.
        let model_type = match arch.as_str() {
            "llama" => "llama",
            "gemma" | "gemma2" | "gemma3" | "gemma4" => &arch,
            "qwen" | "qwen2" => "qwen2",
            "qwen35" | "qwen35moe" => &arch,
            "mistral" => "mistral",
            "mixtral" => "mixtral",
            "phi" | "phi2" | "phi3" => "phi",
            "gpt2" => "gpt2",
            "deepseek" | "deepseek2" => "deepseek_v2",
            other => other,
        };

        // Gemma 4's attention.key_length reports a different dimension than
        // per-head dim; override with hidden_size / num_heads (standard formula)
        let hidden_size = get_arch_u32(GGUF_EMBEDDING_LENGTH);
        let num_heads = get_arch_u32(GGUF_ATTENTION_HEAD_COUNT);
        let head_dim = if arch == "gemma4" && num_heads > 0 {
            // Gemma 4: Q matrix rows = num_heads × head_dim where head_dim = hidden/num_heads × scale
            // For gemma-4-e2b: 1536 / 8 = 192, but actual is 256. Use 2×(hidden/heads) as heuristic.
            // Better: derive from known value 2048 Q rows / 8 heads = 256
            256
        } else {
            get_arch_u32(GGUF_ATTENTION_KEY_LENGTH)
        };

        let mut config = serde_json::json!({
            HF_MODEL_TYPE: model_type,
            HF_HIDDEN_SIZE: hidden_size,
            HF_NUM_HIDDEN_LAYERS: get_arch_u32(GGUF_BLOCK_COUNT),
            HF_INTERMEDIATE_SIZE: get_arch_u32(GGUF_FEED_FORWARD_LENGTH),
            HF_NUM_ATTENTION_HEADS: num_heads,
            HF_NUM_KEY_VALUE_HEADS: get_arch_u32(GGUF_ATTENTION_HEAD_COUNT_KV),
            HF_HEAD_DIM: head_dim,
        });

        if let Some(rope_base) = get_arch_f64(GGUF_ROPE_FREQ_BASE) {
            config[HF_ROPE_THETA] = serde_json::json!(rope_base);
        }
        if let Some(vocab_size) = get_arch_u32_opt(GGUF_VOCAB_SIZE) {
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
    let (mut tensors, vectors) = gguf.load_tensors_filtered(skip_key)?;

    // Re-normalize keys through the architecture's prefix stripping
    let mut normalized_tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    for (k, v) in tensors.drain() {
        let key = super::safetensors::normalize_key(&k, prefixes);
        normalized_tensors.insert(key, v);
    }

    let embed_key = arch.embed_key();
    let embed_raw = normalized_tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();
    // GGUF stores embeddings as [hidden_size, vocab_size] but we need [vocab_size, hidden_size]
    let embed = if embed_raw.shape()[0] < embed_raw.shape()[1] {
        let mut out = ndarray::Array2::<f32>::zeros((embed_raw.shape()[1], embed_raw.shape()[0]));
        out.assign(&embed_raw.t());
        out.into_shared()
    } else {
        embed_raw
    };

    let lm_head = normalized_tensors
        .get("lm_head.weight")
        .or_else(|| normalized_tensors.get(GGUF_OUTPUT_WEIGHT))
        .cloned()
        .unwrap_or_else(|| embed.clone());

    let cfg = arch.config();
    // Gemma3 GGUF does not store vocab_size in arch metadata.
    // Read it from tokenizer.json sitting next to the GGUF file.
    let vocab_size = cfg.vocab_size.filter(|&v| v > 2560).unwrap_or_else(|| {
        // Try to read vocab size from tokenizer.json
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let tok_path = parent.join(TOKENIZER_JSON);
            if let Ok(data) = std::fs::read_to_string(&tok_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                    if let Some(v) = json[TOKENIZER_MODEL][TOKENIZER_VOCAB].as_object() {
                        return v.len();
                    }
                }
            }
        }
        262144 // Gemma3 default
    });

    Ok(ModelWeights {
        tensors: normalized_tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
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
}
