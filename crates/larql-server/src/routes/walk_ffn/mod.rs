//! POST /v1/walk-ffn — decoupled inference protocol.
//!
//! L2 FFN cache: single-position (`seq_len == 1`) requests with `full_output`
//! check the per-model L2 cache before running WalkFfn. Cache key is derived
//! from the gate-KNN feature IDs for that layer (same scheme as L1).
//!
//! Client sends a residual vector, server runs either (a) gate KNN only, or
//! (b) the full FFN compute, and returns the result. This enables distributed
//! inference where the client runs attention locally and the server provides
//! the sparse FFN computation.
//!
//! # Features-only mode (default)
//!
//! Single layer:
//!   POST /v1/walk-ffn {"layer": 26, "residual": [0.12, -0.34, ...]}
//!   → {"layer": 26, "features": [f0, f1, ...], "scores": [s0, s1, ...]}
//!
//! Batched:
//!   POST /v1/walk-ffn {"layers": [0,1,...], "residual": [...]}
//!   → {"results": [{"layer": 0, "features": [...], "scores": [...]}, ...]}
//!
//! # Full-output mode (`"full_output": true`)
//!
//! Returns the FFN output vectors for each requested layer, computed via the
//! same `WalkFfn` path used by local inference (gate KNN → activation → up
//! gather → down projection, architecture-correct).
//!
//! The `residual` field is a row-major flat array of length `seq_len *
//! hidden_size`. `seq_len` defaults to 1 and lets the server process a whole
//! sequence (prefill) in one round trip. Output mirrors the shape.
//!
//! Single layer:
//!   POST /v1/walk-ffn {"layer": 26, "residual": [...], "seq_len": 1,
//!                       "full_output": true}
//!   → {"layer": 26, "output": [...], "seq_len": 1}
//!
//! Batched:
//!   POST /v1/walk-ffn {"layers": [...], "residual": [...], "seq_len": N,
//!                       "full_output": true}
//!   → {"results": [{"layer": N, "output": [...], "seq_len": N}, ...]}
//!
//! Full-output mode triggers lazy loading of model weights. On first call it
//! mmaps the vindex weight files; subsequent calls reuse the loaded state.
//!
//! # Binary wire format (`Content-Type: application/x-larql-ffn`)
//!
//! Requires `full_output = true`. Eliminates JSON float parsing overhead.
//!
//! ## Request — single layer
//! ```text
//! Offset  Size  Field
//! 0       4     layer_index (u32 LE, must not be 0xFFFFFFFF)
//! 4       4     seq_len (u32 LE)
//! 8       4     flags (u32 LE, bit 0 = full_output, must be 1)
//! 12      4     top_k (u32 LE)
//! 16      N×4   residual (f32[] LE)
//! ```
//!
//! ## Request — batch
//! ```text
//! 0       4     BATCH_MARKER = 0xFFFFFFFF
//! 4       4     num_layers (u32 LE)
//! 8       K×4   layer_indices (u32[] LE)
//! 8+K*4   4     seq_len (u32 LE)
//! 12+K*4  4     flags (u32 LE)
//! 16+K*4  4     top_k (u32 LE)
//! 20+K*4  N×4   residual (f32[] LE)
//! ```
//!
//! ## Response — single layer
//! ```text
//! 0       4     layer (u32 LE)
//! 4       4     seq_len (u32 LE)
//! 8       4     latency_ms (f32 LE)
//! 12      N×4   output (f32[] LE)
//! ```
//!
//! ## Response — batch
//! ```text
//! 0       4     BATCH_MARKER = 0xFFFFFFFF
//! 4       4     num_results (u32 LE)
//! 8       4     latency_ms (f32 LE)
//! Per result:
//!   0     4     layer (u32 LE)
//!   4     4     seq_len (u32 LE)
//!   8     4     num_output_floats (u32 LE)
//!   12    M×4   output (f32[] LE)
//! ```
//!
//! ## Module layout (post-split, 2026-05-17)
//!
//! - [`types`] — `WalkFfnRequest`, `FfnEntry`, `FfnOutput`, `RifGuard`,
//!   `BINARY_CT`, `BATCH_MARKER`. Pure data.
//! - [`binary`] — `decode_binary_request`, `encode_binary_output(_f16|_i8)`,
//!   `encode_json_full_output`. Pure functions + in-file unit tests.
//! - [`validate`] — `collect_scan_layers`, `validate_residual`,
//!   `validate_owned`. Pure-function correctness checks.
//! - [`core`] — `run_full_output_core` (the FFN compute + MoE-layer
//!   branch). The MoE branch needs a remote-MoE backend; excluded from
//!   per-file coverage gating until a MoE fixture lands.
//! - [`dispatch`] — `run_walk_ffn` (JSON entry point, parse → validate
//!   → dispatch to full or features-only).
//! - [`handler`] — `handle_walk_ffn` (axum entrypoint, negotiates
//!   binary vs JSON; ADR-0009 Accept-header response encoding).
//! - [`q8k`] — `handle_walk_ffn_q8k` (Q8K-prenormed dense-FFN batch
//!   endpoint). Needs a Q4K-quantised vindex; excluded from per-file
//!   coverage until a Q4K fixture lands.

pub(crate) mod binary;
pub(crate) mod core;
pub(crate) mod dispatch;
pub mod handler;
pub mod q8k;
pub(crate) mod types;
pub(crate) mod validate;

// ── Public re-exports (preserve `crate::routes::walk_ffn::X` paths) ──────────

pub use handler::handle_walk_ffn;
pub use q8k::handle_walk_ffn_q8k;
pub use types::WalkFfnRequest;

// utoipa's `OpenApi` derive emits `__path_$fn` types matching the
// path it sees in the `paths(...)` list. To keep that list referencing
// `crate::routes::walk_ffn::handle_walk_ffn(_q8k)` (instead of forcing
// every call site to know the post-split submodule path), re-export
// the generated path types here.
#[doc(hidden)]
pub use handler::__path_handle_walk_ffn;
#[doc(hidden)]
pub use q8k::__path_handle_walk_ffn_q8k;
