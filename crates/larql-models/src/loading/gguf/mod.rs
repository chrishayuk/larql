//! GGUF format reader — parse GGUF files and load tensors as f32.
//!
//! GGUF is the GGML Universal Format used by llama.cpp.
//! We support reading unquantized (F32, F16, BF16) and quantized (Q4_0, Q4_1, Q8_0) tensors.
//! All tensors are dequantized to f32 for use with ModelWeights.

mod constants;
mod loader;
mod orient;
mod parser;
mod reader;
mod types;
mod writer;

pub use loader::{
    load_gguf, load_gguf_keep_quant, load_gguf_validated, normalize_gguf_key, I2S_SCALE_SUFFIX,
};
pub use types::{GgufFile, GgufTensorInfo, GgufValue, ShardInfo};
pub use writer::{write_streaming, GgufTensor, GgufTensorDescriptor, GgufWriter};

// Preserve original pub(crate) API surface — some items are consumed only within
// this module today but were pub(crate) in the monolith and may gain external
// callers (e.g. safetensors.rs already uses load_gguf_filtered_with_validation).
#[allow(unused_imports)]
pub(crate) use loader::{load_gguf_filtered, load_gguf_filtered_with_validation};
#[allow(unused_imports)]
pub(crate) use parser::{discover_shard_siblings, parse_shard_filename};
