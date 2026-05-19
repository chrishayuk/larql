//! GPU-side prefill phase for [`super::generate_streaming`].
//!
//! Three branches:
//!
//! 1. **PLE** (Gemma 4 E2B) — token-by-token via `decode_token` so the
//!    Metal backend can apply per-layer-input embeddings inside each
//!    layer block. Only compiled when `--features metal`; without it the
//!    branch is unreachable (no `MetalBackend` to downcast to).
//! 2. **Per-layer Q4_K MoE** — token-by-token via `decode_token_q4k_moe`.
//!    The standard `prefill_kquant` path calls `cpu_moe_forward` which expects
//!    BF16 blobs and would panic on Q4_K expert bytes; token-by-token is
//!    correct and builds the KV cache identically.
//! 3. **Standard** — the batched fused `prefill_kquant_prompt` path.
//!
//! Returns the post-prefill `h_vec` (`seq_len × hidden` floats; only the
//! last position is meaningful for the subsequent first-token sample).

use crate::layer_graph::generate::gpu_setup::prefill_kquant_prompt;
use crate::layer_graph::generate::types::GenerateError;
use crate::model::ModelWeights;
use larql_compute::prelude::*;
use larql_compute::FullPipelineLayer;

/// Run the prefill phase for streaming Q4 generation.
///
/// `metal_ple_backend` is `Some(metal)` only when (a) the model uses
/// per-layer embeddings AND (b) `LARQL_METAL_PLE=1` AND (c) we're on
/// macOS+metal. The PLE-upload closure is invoked once per prompt token.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_for_streaming(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
    layers: &[FullPipelineLayer],
    hidden: usize,
    intermediate: usize,
    token_ids: &[u32],
    x: &[f32],
    qk_norm_val: bool,
    softcap_val: f32,
    metal_ple_backend: Option<&larql_compute_metal::MetalBackend>,
    upload_ple: &dyn Fn(&larql_compute_metal::MetalBackend, u32, &[f32]),
) -> Result<Vec<f32>, GenerateError> {
    let seq_len = token_ids.len();

    // Branch 1: Per-Layer Embeddings (Metal-only).
    if let Some(metal) = metal_ple_backend {
        let mut last_h = vec![0.0f32; hidden];
        for pos in 0..seq_len {
            let x_pos: Vec<f32> = x[pos * hidden..(pos + 1) * hidden].to_vec();
            upload_ple(metal, token_ids[pos], &x_pos);
            last_h = backend
                .decode_token(layers, &x_pos, hidden, intermediate)
                .unwrap_or_else(|| vec![0.0f32; hidden]);
        }
        let mut out = vec![0.0f32; seq_len * hidden];
        out[(seq_len - 1) * hidden..].copy_from_slice(&last_h);
        return Ok(out);
    }

    // Branch 2: per-layer Q4_K MoE format.
    if weights.has_per_layer_ffn() {
        return prefill_kquant_moe(weights, backend, layers, hidden, intermediate, token_ids, x);
    }

    // Branch 3: standard fused prefill.
    prefill_kquant_prompt(
        backend,
        layers,
        x,
        hidden,
        intermediate,
        seq_len,
        qk_norm_val,
        softcap_val,
        "GPU Q4 prefill returned no output",
    )
}

/// Non-metal build: PLE branch is unreachable (no `MetalBackend`).
#[cfg(not(all(feature = "metal", target_os = "macos")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_for_streaming(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
    layers: &[FullPipelineLayer],
    hidden: usize,
    intermediate: usize,
    token_ids: &[u32],
    x: &[f32],
    qk_norm_val: bool,
    softcap_val: f32,
) -> Result<Vec<f32>, GenerateError> {
    let seq_len = token_ids.len();

    if weights.has_per_layer_ffn() {
        return prefill_kquant_moe(weights, backend, layers, hidden, intermediate, token_ids, x);
    }

    prefill_kquant_prompt(
        backend,
        layers,
        x,
        hidden,
        intermediate,
        seq_len,
        qk_norm_val,
        softcap_val,
        "GPU Q4 prefill returned no output",
    )
}

/// Per-layer Q4_K MoE prefill: route on CPU, dispatch experts on GPU
/// via `decode_token_q4k_moe` per token. Returns the last-position hidden
/// padded to a `seq_len × hidden` buffer to match the batched-prefill shape.
#[allow(clippy::too_many_arguments)]
fn prefill_kquant_moe(
    weights: &ModelWeights,
    backend: &dyn ComputeBackend,
    layers: &[FullPipelineLayer],
    hidden: usize,
    intermediate: usize,
    token_ids: &[u32],
    x: &[f32],
) -> Result<Vec<f32>, GenerateError> {
    if !backend.supports(Capability::DecodeQ4KMoe) {
        return Err(GenerateError::unsupported_backend(
            "per-layer Q4K expert generation requires backend Q4K MoE decode support",
        ));
    }
    let seq_len = token_ids.len();
    let norm_eps = weights.arch.norm_eps();
    let mut last_h = vec![0.0f32; hidden];
    for pos in 0..seq_len {
        let x_pos: Vec<f32> = x[pos * hidden..(pos + 1) * hidden].to_vec();
        let get_expert =
            |layer_idx, expert_idx| weights.get_layer_entry_bytes(layer_idx, expert_idx);
        last_h = backend
            .decode_token_q4k_moe(layers, &x_pos, hidden, intermediate, norm_eps, &get_expert)
            .unwrap_or_else(|| vec![0.0f32; hidden]);
    }
    let mut out = vec![0.0f32; seq_len * hidden];
    out[(seq_len - 1) * hidden..].copy_from_slice(&last_h);
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! Unit-test the early-return guards. The full prefill paths (PLE,
    //! Q4_K MoE, standard fused) need both a Q4-supporting backend AND a
    //! Q4_K-loaded vindex — out of reach for synthetic fixtures. The
    //! guard tests below exercise the rejection paths reachable with
    //! `CpuBackend`, which doesn't advertise either Q4 or DecodeQ4KMoe.
    use super::*;
    use crate::test_utils::make_test_weights;

    #[test]
    fn prefill_q4k_moe_rejects_backend_without_decode_q4k_moe_capability() {
        // CpuBackend doesn't support `Capability::DecodeQ4KMoe` → the
        // guard fires and returns Err before touching layers.
        let weights = make_test_weights();
        let layers: Vec<FullPipelineLayer<'_>> = Vec::new();
        let token_ids = vec![0u32, 1];
        let x = vec![0.0f32; 2 * weights.hidden_size];
        let err = prefill_kquant_moe(
            &weights,
            &larql_compute::CpuBackend,
            &layers,
            weights.hidden_size,
            weights.intermediate_size,
            &token_ids,
            &x,
        )
        .expect_err("CpuBackend without DecodeQ4KMoe must be rejected");
        // Use Display so we don't depend on private GenerateError variants.
        let msg = format!("{err}");
        assert!(
            msg.contains("Q4K") || msg.contains("decode") || msg.contains("backend"),
            "expected Q4K/decode/backend wording, got: {msg}"
        );
    }

    // The non-metal `prefill_for_streaming` doesn't take a `metal_ple`
    // argument — its branch set is just (per-layer Q4K MoE, standard).
    // Both paths require backend Q4 support, so CpuBackend short-circuits
    // through the moe_q4k guard above before reaching the standard
    // `prefill_kquant_prompt` path.

    /// `prefill_for_streaming` standard branch (non-MoE weights) — falls
    /// through `has_per_layer_ffn=false` → calls `prefill_kquant_prompt`,
    /// which propagates the backend's `None` as `PrefillFailed`.
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    #[test]
    fn prefill_for_streaming_standard_branch_errors_when_backend_returns_none() {
        let weights = make_test_weights();
        // Standard arch (no per-layer FFN) takes the branch-3 path; CpuBackend's
        // default `prefill_kquant` returns None → wraps in PrefillFailed.
        let layers: Vec<FullPipelineLayer<'_>> = Vec::new();
        let token_ids = vec![0u32, 1];
        let x = vec![0.0f32; 2 * weights.hidden_size];
        let result = prefill_for_streaming(
            &weights,
            &larql_compute::CpuBackend,
            &layers,
            weights.hidden_size,
            weights.intermediate_size,
            &token_ids,
            &x,
            false,
            0.0,
        );
        let err = match result {
            Ok(_) => panic!("CpuBackend default prefill_kquant must yield Err"),
            Err(e) => e,
        };
        assert!(matches!(err, GenerateError::PrefillFailed { .. }));
    }

    /// `prefill_for_streaming` standard branch happy path — `MockGpuBackend`
    /// returns `Some(vec![0; seq_len * hidden])` from `prefill_kquant`, the
    /// wrapper unwraps it as `Ok`. Drives the success body of
    /// `prefill_kquant_prompt`.
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    #[test]
    fn prefill_for_streaming_standard_branch_succeeds_with_mock_gpu_backend() {
        use crate::test_utils::MockGpuBackend;
        let weights = make_test_weights();
        let backend = MockGpuBackend::new();
        let layers: Vec<FullPipelineLayer<'_>> = Vec::new();
        let token_ids = vec![0u32, 1, 2];
        let seq = token_ids.len();
        let x = vec![0.0f32; seq * weights.hidden_size];
        let out = prefill_for_streaming(
            &weights,
            &backend,
            &layers,
            weights.hidden_size,
            weights.intermediate_size,
            &token_ids,
            &x,
            false,
            0.0,
        )
        .expect("MockGpuBackend prefill_kquant returns Some");
        assert_eq!(out.len(), seq * weights.hidden_size);
    }

    /// `prefill_kquant_moe` happy path — `MockGpuBackend` advertises
    /// `DecodeQ4KMoe` and returns `Some(vec![0; hidden])` per token.
    #[test]
    fn prefill_q4k_moe_succeeds_with_mock_gpu_backend() {
        use crate::test_utils::MockGpuBackend;
        let weights = make_test_weights();
        let backend = MockGpuBackend::new();
        let layers: Vec<FullPipelineLayer<'_>> = Vec::new();
        let token_ids = vec![0u32, 1, 2];
        let seq = token_ids.len();
        let x = vec![0.0f32; seq * weights.hidden_size];
        let out = prefill_kquant_moe(
            &weights,
            &backend,
            &layers,
            weights.hidden_size,
            weights.intermediate_size,
            &token_ids,
            &x,
        )
        .expect("MockGpuBackend supports DecodeQ4KMoe");
        // Output is the last-position hidden padded out to seq × hidden.
        assert_eq!(out.len(), seq * weights.hidden_size);
    }
}
