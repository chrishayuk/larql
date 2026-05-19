//! MarkovResidualEngine — residual-stream KV-cache replacement.
//!
//! The pre-layer residual vector is the complete Markov state of the transformer.
//! K/V are recomputed from stored residuals at decode time (KL = 0.0 vs full-KV
//! baseline on Gemma 3 4B, validated 2026-04-23).

pub mod compute;
pub mod engine;
pub mod store;
pub mod walk;

pub use compute::{
    kv_memory_bytes_for_seq, recompute_kv, rs_decode_step, rs_prefill, RsPrefillResult,
};
pub use engine::MarkovResidualEngine;
pub use store::RsStore;
pub use walk::ensure_attn_tensors_dequantised;
