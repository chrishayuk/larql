//! The dense VINDEX3 → GGUF export, tested against a miniature Gemma 3
//! container: what leaves, read back through LARQL's own GGUF reader, and
//! every refusal on the way.

mod common;
mod export_roundtrip;
mod naming;
mod programme;
mod refusals;
mod sentencepiece;
mod special;
mod vocab;
