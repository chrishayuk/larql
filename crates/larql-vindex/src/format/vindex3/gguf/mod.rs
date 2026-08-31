//! Lowering a VINDEX3 artifact to GGUF, for execution by an independent
//! runtime.
//!
//! The boundary this module defends: **semantic concepts in, target
//! vocabulary out.** A lowering reads `context_length`,
//! `attention.num_q_heads`, `ffn.intermediate_size` from the graph and
//! writes `qwen35.context_length` and friends. The moment a lowering
//! needs to know it is looking at Qwen in order to *find* a fact, that
//! fact is missing from the graph — see the format spec's
//! independent-backend test.

pub mod emit;
pub mod geometry;
pub mod metadata;
pub mod plan;
pub mod preflight;
pub mod vocab;
pub mod walk;
