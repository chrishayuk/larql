//! Full pipeline: ALL Q4 (attention + FFN) in ONE Metal command buffer.
//!
//! Correct inference path with norms and residual connections:
//!   Per layer:
//!     1. rms_norm(h, input_norm) → h_norm
//!     2. Q4 Q/K/V projections from h_norm
//!     3. Fused attention (RoPE + GQA + softcap)
//!     4. Q4 O projection
//!     5. Post-attn norm (if post_norms) + residual_add(h, o_out) → h
//!     6. rms_norm(h, post_attn_norm) → h_ffn
//!     7. Q4 gate/up → GEGLU → Q4 down
//!     8. Post-FFN norm (if post_norms) + residual_add(h, ffn_out) → h
//!     9. Q8 quantize h → next layer
//!
//! ## Layout
//!
//! - `dispatch`: orchestrator (`dispatch_full_pipeline`) + the
//!   public `encode_rms_norm` / `encode_residual_add` helpers used
//!   by `prefill.rs`.
//! - `buffers`: [`LayerBuffers`] — pre-allocates every per-layer
//!   scratch buffer + caches the per-layer Q4 weight handles.
//! - `dump`: per-layer file dumps activated by
//!   `LARQL_METAL_DUMP_LAYERS=<dir>`.
//! - `kv_copy`: post-commit KV cache population.

mod buffers;
mod dispatch;
mod dump;
mod kv_copy;
mod stages;

// Public re-exports — these names are part of the crate-level API
// (`prefill.rs` uses the encode helpers, callers reach for
// `dispatch_full_pipeline` directly).
pub use dispatch::{
    dispatch_full_pipeline, encode_residual_add, encode_rms_norm, PipelineIntervention,
};
