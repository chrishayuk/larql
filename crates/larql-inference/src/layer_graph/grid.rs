//! Autoregressive generation via a sharded expert grid.
//!
//! Uses the Metal pipeline for attention + dense FFN (same as normal `generate`),
//! but intercepts the MoE expert block per layer via a callback that dispatches
//! to remote shards over HTTP instead of calling `cpu_moe_forward` locally.
//!
//! The hook: `ComputeBackend::decode_token_with_moe(layers, x, ..., moe_fn)`
//! where `moe_fn(layer, h_post_attn) -> Vec<f32>` calls
//! `RemoteMoeBackend::forward_moe`.
//!
//! # Diagnostics
//!
//! Set `SKIP_MOE=1` to zero out the expert block on every decode step.
//! This isolates whether errors come from remote dispatch vs. dense FFN.

mod config;
mod remote_ffn;
mod remote_moe;
mod setup;
mod timing;

pub use remote_ffn::{generate_with_remote_ffn, generate_with_remote_ffn_batch};
pub use remote_moe::{generate_with_remote_moe, generate_with_remote_moe_batch};

// ── Bottleneck diagnostic ────────────────────────────────────────────────────
//
// Activated by `LARQL_MOE_TIMING=1`.  The streaming path swaps
// `forward_moe_stream` for an explicit fire/collect_with_timing pair so we can
// see, for every MoE layer of every decoded token:
//
//   - `total_ms`:        wall-clock time inside the moe_fn closure
//   - `route_fire_ms`:   CPU routing + non-blocking fire
//   - `collect_ms`:      condvar-blocking wait for all shards' h2 frames
//   - per-shard `(wall_ms, server_compute_ms)` so `network_ms` is derivable
//     as `wall_ms − server_compute_ms`
//
// Everything is per-MoE-layer; the GPU side (attention + dense FFN) is timed
// independently by `LARQL_GPU_TIMING=1` in the metal backend.

#[derive(Debug)]
pub struct GridGenerateResult {
    pub tokens: Vec<String>,
    pub decode_ms: Vec<f64>,
    /// Sum of remote FFN round-trip time per decode step (all layers, streaming path only).
    /// Empty for MoE paths and the batch predispatch path.
    pub ffn_rtt_ms: Vec<f64>,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::ffn::moe_remote::{RemoteMoeBackend, RemoteMoeError};
    use crate::layer_graph::generate::eos::EosConfig;
    use crate::test_utils::{make_test_tokenizer, make_test_vindex, make_test_weights};
    use larql_compute::CpuBackend;

    // ── generate_with_remote_moe — error path ────────────────────────────────

    #[test]
    fn errors_when_vindex_has_no_q4k_mmap() {
        let weights = make_test_weights();
        let idx = make_test_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);

        // make_test_vindex has no interleaved Q4K or Q4 mmap.
        // The function should fail at the mmap guard, before any GPU or shard call.
        let remote = RemoteMoeBackend::new_disconnected();
        let result = generate_with_remote_moe(
            &weights,
            &tokenizer,
            vec![0u32],
            1,
            &idx,
            &remote,
            &CpuBackend,
            &EosConfig::builtin(),
        );
        match result {
            Err(RemoteMoeError::BadResponse(msg)) => {
                assert!(
                    msg.contains("no interleaved Q4 FFN mmap"),
                    "unexpected error message: {msg}"
                );
            }
            other => panic!("expected BadResponse, got: {other:?}"),
        }
    }
}
