//! GGUF data types — GgufValue, GgufTensorInfo, ShardInfo, GgufFile struct definitions.

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// GGUF metadata value
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
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
    pub(super) name: String,
    pub(super) n_dims: u32,
    pub(super) dims: Vec<u64>,
    pub(super) tensor_type: u32,
    pub(super) offset: u64,
    /// Index into [`GgufFile::shards`] selecting which file this tensor lives in.
    /// Zero for single-shard models; assigned by `open` when discovering siblings.
    pub(super) shard_idx: usize,
}

impl GgufTensorInfo {
    /// Raw GGUF tensor name (e.g. `blk.0.attn_q.weight`). The HF-equivalent
    /// key (`layers.0.self_attn.q_proj.weight`) is obtained via
    /// [`normalize_gguf_key`].
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn n_dims(&self) -> u32 {
        self.n_dims
    }
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }
    /// GGML tensor-type id (Q4_0, Q8_0, F16, …). See `quant::ggml` constants.
    pub fn tensor_type(&self) -> u32 {
        self.tensor_type
    }
    /// Tensor data offset *within* its shard's data section. Add
    /// `ShardInfo::data_offset` to get the absolute file offset.
    pub fn offset(&self) -> u64 {
        self.offset
    }
    /// Index into [`GgufFile::shards`] selecting which file owns this tensor.
    pub fn shard_idx(&self) -> usize {
        self.shard_idx
    }
}

/// One file in a (possibly multi-shard) GGUF split.
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// Path to the `.gguf` file for this shard.
    pub path: std::path::PathBuf,
    /// Byte offset at which tensor data starts inside this file.
    pub data_offset: u64,
}

// ═══════════════════════════════════════════════════════════════
// GGUF reader
// ═══════════════════════════════════════════════════════════════

pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensor_infos: Vec<GgufTensorInfo>,
    /// Tensor data offset of the first (or only) shard. Kept for back-compat
    /// with single-file callers — multi-shard callers should index into
    /// [`Self::shards`] using `GgufTensorInfo::shard_idx`.
    pub data_offset: u64,
    /// Path to the first (or only) shard. Same back-compat note as
    /// `data_offset` — for multi-shard models the other shards are in
    /// [`Self::shards`].
    pub path: std::path::PathBuf,
    /// All shards making up this GGUF. Always non-empty; length 1 for
    /// single-file models. For multi-shard models opened from a non-first
    /// shard, `self.path` is the user-supplied path (not necessarily shard 0).
    pub shards: Vec<ShardInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
