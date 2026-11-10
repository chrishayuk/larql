//! File format I/O — vindex loading, saving, checksums, HuggingFace.
//! Model loading (safetensors/GGUF) is in larql-models.

pub mod capability;
pub mod checksums;
pub mod completeness;
pub mod describes;
pub mod down_meta;
pub mod filenames;
pub mod fp4_codec;
pub mod fresh_output;
pub mod generation;
#[cfg(test)]
mod generation_tests;
pub mod huggingface;
pub mod le_floats;
pub mod load;
pub mod lyrw2;
pub mod moe_manifest;
pub mod quant;
pub mod spec;
pub mod vindex3;
pub mod weights;

// Back-compat alias — `format::fp4_storage` was renamed to `fp4_codec`
// in the 2026-04-25 round-2 cleanup (the file does encoding-side
// codec work; the runtime store lives at `index::storage::fp4_store`).
// Drop this alias once external callers are migrated.
pub use describes::model_id_at;
pub use fp4_codec as fp4_storage;
