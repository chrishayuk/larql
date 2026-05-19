//! Model weights serialization to/from .vindex directories.
//!
//! Split format (v2): separate files per component, no duplication.
//!   attn_weights.bin  — Q, K, V, O per layer
//!   up_weights.bin    — FFN up projections (gate is in gate_vectors.bin)
//!   down_weights.bin  — FFN down projections
//!   norms.bin         — all LayerNorm/RMSNorm vectors
//!   lm_head.bin       — output projection
//!
//! - `write_f32`: build + streaming write paths for f32 / Q4_0
//!                weights (`write_model_weights`, `WeightSource` trait,
//!                `StreamingWeights`).
//! - `write_kquant`: Q4_K / Q6_K streaming writer with manifest-aware
//!                output (`write_model_weights_kquant`).
//! - `load`:      reconstruct `ModelWeights` from a vindex directory
//!                (`load_model_weights`, `find_tokenizer_path`).

mod capabilities;
pub mod load;
pub mod manifest;
mod ple_sidecar;
pub mod write_f32;
pub mod write_kquant;
pub mod write_layers;

pub(crate) use capabilities::ensure_extract_level_supported;

pub use load::{
    find_tokenizer_path, load_model_weights, load_model_weights_kquant,
    load_model_weights_kquant_shard, load_model_weights_with_opts, LoadWeightsOptions,
};
pub use manifest::Q4kManifestEntry;
pub use write_f32::{
    write_model_weights, write_model_weights_with_opts, StreamingWeights, WeightSource,
    WriteWeightsOptions,
};
pub use write_kquant::{
    write_model_weights_kquant, write_model_weights_kquant_with_opts, DownProjFormat,
    KquantWriteOptions, QuantBlockFormat,
};
