//! Lowering a dense softmax decoder to a llama.cpp GGUF, byte-faithful to
//! llama.cpp's own converter for the same weights.
//!
//! The qwen35 lowering beside this module exists for one hybrid target;
//! this one is shaped for the many dense ones. Everything family-specific
//! is a [`target::DenseTarget`] value — tensor names, arch string, and the
//! behaviour the runtime builds in — so adding a family is adding data.
//! Gemma 3 is the first registered target.
//!
//! What the reference converter computes from its own heuristics (the
//! naming of `general.*`, the special-token flags, the SentencePiece
//! vocabulary) is ported rather than approximated, because the oracle for
//! this module is byte identity with that converter's output.

pub mod export;
pub mod gemma3;
pub mod general;
pub mod metadata;
pub mod naming;
pub mod programme;
pub mod roles;
pub mod sentencepiece;
pub mod special;
pub mod spm_vocab;
pub mod target;
pub mod tensors;
pub mod tokenizer;

pub use export::{export_dense, ContextSource, DenseExport, DenseExportOptions};
