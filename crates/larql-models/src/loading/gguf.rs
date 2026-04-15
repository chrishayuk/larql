//! GGUF format reader — parse GGUF files and load tensors as f32.
//!
//! GGUF is the GGML Universal Format used by llama.cpp.
//! We support reading unquantized (F32, F16, BF16) and quantized (Q4_0, Q4_1, Q8_0) tensors.
//! All tensors are dequantized to f32 for use with ModelWeights.

use std::collections::HashMap;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use ndarray::Array2;

use crate::weights::ModelWeights;
use crate::detect::ModelError;

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
                "not a GGUF file (magic: 0x{:08X}, expected 0x{:08X})", magic, GGUF_MAGIC
            )));
        }

        // Version
        let version = read_u32(&mut r)?;
        if !(2..=3).contains(&version) {
            return Err(ModelError::Parse(format!("unsupported GGUF version: {version}")));
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
            tensor_infos.push(GgufTensorInfo { name, n_dims, dims, tensor_type, offset });
        }

        // Data starts at next alignment boundary (32 bytes)
        let pos = r.stream_position()
            .map_err(ModelError::Io)?;
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
    pub fn load_tensors(&self) -> Result<(HashMap<String, crate::WeightArray>, HashMap<String, Vec<f32>>), ModelError> {
        let file = std::fs::File::open(&self.path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();

        for info in &self.tensor_infos {
            let abs_offset = self.data_offset + info.offset;
            let n_elements: u64 = info.dims.iter().product();

            let data_size = tensor_data_size(info.tensor_type, n_elements as usize)?;
            if abs_offset as usize + data_size > mmap.len() {
                return Err(ModelError::Parse(format!(
                    "tensor {} data out of bounds (offset {} + size {} > file {})",
                    info.name, abs_offset, data_size, mmap.len()
                )));
            }

            let raw = &mmap[abs_offset as usize..abs_offset as usize + data_size];
            let floats = dequantize(raw, info.tensor_type, n_elements as usize)?;

            // Normalize key name (strip GGUF prefixes)
            let key = normalize_gguf_key(&info.name);

            match info.n_dims {
                2 => {
                    // GGUF stores tensor extents in ggml order: n0 = cols, n1 = rows.
                    // Convert to conventional ndarray row-major [rows, cols].
                    let cols = info.dims[0] as usize;
                    let rows = info.dims[1] as usize;
                    let arr = Array2::from_shape_vec((rows, cols), floats)
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
        let get_str = |k: &str| self.metadata.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let _get_u32 = |k: &str| self.metadata.get(k).and_then(|v| v.as_u32()).unwrap_or(0);

        // GGUF uses "general.architecture" and "{arch}.*" keys
        let arch = get_str("general.architecture");
        let prefix = format!("{arch}.");

        let get_arch_u32 = |suffix: &str| {
            self.metadata.get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_u32())
                .unwrap_or(0)
        };
        let get_arch_f64 = |suffix: &str| {
            self.metadata.get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        };

        // Map GGUF architecture names to HF model_type
        let model_type = match arch.as_str() {
            "llama" => "llama",
            "gemma" | "gemma2" | "gemma3" => &arch,
            "qwen" | "qwen2" => "qwen2",
            "mistral" => "mistral",
            "mixtral" => "mixtral",
            "phi" | "phi2" | "phi3" => "phi",
            "gpt2" => "gpt2",
            "deepseek" | "deepseek2" => "deepseek_v2",
            other => other,
        };

        serde_json::json!({
            "model_type": model_type,
            "hidden_size": get_arch_u32("embedding_length"),
            "num_hidden_layers": get_arch_u32("block_count"),
            "intermediate_size": get_arch_u32("feed_forward_length"),
            "num_attention_heads": get_arch_u32("attention.head_count"),
            "num_key_value_heads": get_arch_u32("attention.head_count_kv"),
            "head_dim": get_arch_u32("attention.key_length"),
            "rope_theta": get_arch_f64("rope.freq_base"),
            "vocab_size": get_arch_u32("vocab_size"),
        })
    }
}

/// Load a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf(path: &Path) -> Result<ModelWeights, ModelError> {
    let gguf = GgufFile::open(path)?;

    // Detect architecture from GGUF metadata
    let config_json = gguf.to_config_json();
    let arch = crate::detect_from_json(&config_json);
    let prefixes = arch.key_prefixes_to_strip();

    // Load and dequantize all tensors
    let (mut tensors, vectors) = gguf.load_tensors()?;

    // Re-normalize keys through the architecture's prefix stripping
    let mut normalized_tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    for (k, v) in tensors.drain() {
        let key = super::safetensors::normalize_key_pub(&k, prefixes);
        normalized_tensors.insert(key, v);
    }

    let embed_key = arch.embed_key();
    let embed_raw = normalized_tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();
    // `load_tensors()` already converts GGUF's ggml-order dims into conventional
    // [rows, cols]. Keep this transpose as a defensive fallback for older files.
    let embed = if embed_raw.shape()[0] < embed_raw.shape()[1] {
        let mut out = ndarray::Array2::<f32>::zeros((embed_raw.shape()[1], embed_raw.shape()[0]));
        out.assign(&embed_raw.t());
        out.into_shared()
    } else {
        embed_raw
    };

    let lm_head = normalized_tensors
        .get("lm_head.weight")
        .or_else(|| normalized_tensors.get("output.weight"))
        .cloned()
        .unwrap_or_else(|| embed.clone());

    let cfg = arch.config();
    // Gemma3 GGUF does not store vocab_size in arch metadata.
    // Read it from tokenizer.json sitting next to the GGUF file.
    let vocab_size = cfg.vocab_size
        .filter(|&v| v > 2560)
        .unwrap_or_else(|| {
            // Try to read vocab size from tokenizer.json
            if let Some(parent) = std::path::Path::new(&path).parent() {
                let tok_path = parent.join("tokenizer.json");
                if let Ok(data) = std::fs::read_to_string(&tok_path) {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                        if let Some(v) = json["model"]["vocab"].as_object() {
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
        _ => Err(ModelError::Parse(format!("unknown GGUF metadata type: {vtype}"))),
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
        _ => Err(ModelError::Parse(format!("unknown GGUF array element type: {elem_type}"))),
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

    

    name
        .replace("blk.", "layers.")
        .replace("attn_q.", "self_attn.q_proj.")
        .replace("attn_k.", "self_attn.k_proj.")
        .replace("attn_v.", "self_attn.v_proj.")
        .replace("attn_output.", "self_attn.o_proj.")
        .replace("ffn_gate.", "mlp.gate_proj.")
        .replace("ffn_up.", "mlp.up_proj.")
        .replace("ffn_down.", "mlp.down_proj.")
        .replace("attn_norm.", "input_layernorm.")
        .replace("ffn_norm.", "post_attention_layernorm.")
        .replace("token_embd.", "embed_tokens.")
        .replace("output_norm.", "norm.")
        .replace("output.", "lm_head.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, Write};

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
        assert_eq!(
            normalize_gguf_key("output.weight"),
            "lm_head.weight"
        );
    }

    #[test]
    fn test_load_tensors_swaps_gguf_2d_dims_to_rows_cols() {
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
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap(); // tensor data offset

        // Pad tensor data start to 32-byte boundary.
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize]).unwrap();

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
        assert_eq!(down[[1, 3]], 8.0);
    }

    // Dequant tests are in format::quant::ggml::tests
}
