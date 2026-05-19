//! Diagnostic and profiling tools for the Metal compute backend.
//!
//! Three categories of diagnostics, now consolidated here:
//!
//! ## 1. Full shader bench (`shader_bench`)
//! Emits a shader inventory and directly times the production-shaped tiled
//! kernels in isolated and batched command-buffer modes. Use this before
//! promoting shader variants; batched timings match decode geometry.
//!
//! ## 2. Per-kernel bandwidth profiler (`kernel_profile`)
//! Measures each production kernel (q6k_matvec, q4k_ffn_gate_up, QKV, lm_head)
//! in isolation AND batched (34x in one command buffer, matching the real decode
//! pipeline). Reports: ms/call, GB/s effective bandwidth, compute- vs bandwidth-bound.
//!
//! ## 3. Decode-stage profiler (`decode::profile`)
//! Per-stage wall-clock timings during a real decode token (attn vs FFN vs norm).
//! `ProfileTimings` is re-exported here for callers that don't want to import from
//! the private `decode` submodule.
//!
//! ## 4. Decode-layer dump (`decode::diag`)
//! Env-gated: `LARQL_DUMP_LAYERS=<dir>` writes per-layer f32 files for CPU/Metal
//! residual diffs. `LARQL_DECODE_DIAG_LAYER=<n>` dumps all sub-stage buffers at
//! layer n and exits. Used to bisect NaN/divergence to a specific sub-stage.
//!
//! ## Usage
//! ```bash
//! # Per-kernel bandwidth profiler
//! cargo run --release --features metal -p larql-compute --example diag_profile_kernels
//!
//! # Full shader bench + inventory
//! cargo run --release --features metal -p larql-compute --example diag_shader_bench
//!
//! # Decode pipeline stage bisect
//! LARQL_METAL_DUMP_LAYERS=/tmp/dump \
//!   cargo run --release --features metal -p larql-compute --example diag_decode_pipeline
//! ```

pub mod kernel_profile;
pub mod shader_bench;

// Re-export the stage-level profiling types from decode::profile so callers
// don't need to know the internal module layout.
pub use crate::decode::ProfileTimings;
