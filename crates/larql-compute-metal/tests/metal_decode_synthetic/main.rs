//! Synthetic end-to-end decode tests.
//!
//! Builds a small `FullPipelineLayer` with synthetic Q4_K (attn) +
//! Q4_0 (FFN) weights and runs `MetalBackend::decode_token` on it.
//! Adapted from `examples/diag_decode_pipeline.rs`.
//!
//! Why this file exists: per-shader tests (`test_metal_shaders.rs` and
//! friends) hit the kernels but never exercise the production decode
//! orchestration code in `metal/decode/encode_{attn,qkv,ffn,post_ffn}.rs`
//! and `metal/decode/mod.rs::decode_token_with_moe_split_fn`. End-to-end
//! tests in `larql-inference/tests/` do, but those don't show up in
//! per-crate `cargo llvm-cov --package larql-compute` runs. This test
//! file fills that gap — a single decode_token call lifts ~2856 LoC of
//! production decode code from 0% to executed.
//!
//! These are smoke tests, not numerical-parity tests. They verify:
//! - decode_token returns a non-NaN, non-zero output buffer
//! - dimensions are right
//! - The `LARQL_FUSED_PRELAYER_NORM=1` D-RMS-FUSE wiring produces
//!   bit-identical output to the unfused path on a non-Gemma-style
//!   layer (no `has_post_norms`).
//!
//! Numerical-correctness against a CPU reference happens in
//! `larql-inference/tests/test_cpu_metal_parity.rs` against real
//! vindexes; it's at the wrong scope to live here. That gate only loads
//! Q4_K vindexes, though, so it never reaches the Q4_0 FFN these layers
//! use. `prefill_decode_ffn_parity` is the one numerical check here: it
//! holds prefill to decode on that FFN, because a prefill that ran the
//! down projection on the wrong vector passed every smoke test above.

#![cfg(target_os = "macos")]

mod attention_hybrid;
mod attention_reaches_residual;

mod attn_options;
mod backend_kv;
mod common;
mod decode_core;
mod env_diag_paths;
mod ffn_ple_routes;
mod ffn_std_profile;
mod moe;
mod o_proj_and_seqpar;
mod padded_attn;
mod prefill_decode_ffn_parity;
mod qkv_routes;
mod state_dump;
mod trait_decode;
