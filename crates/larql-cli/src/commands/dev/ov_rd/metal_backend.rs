/// Metal compute backend initialization for AHORD evaluation commands.
///
/// Shared by `eval-program` and `induce-program`. Returns `None` when Metal
/// is unavailable or not requested; callers fall back to the CPU path.

pub(super) type Backend = Box<dyn larql_compute::ComputeBackend + Send + Sync>;

/// Initialize a Metal backend if `use_metal` is true and the `metal` feature
/// is compiled on macOS. Logs the outcome to stderr.
pub(super) fn init(use_metal: bool) -> Option<Backend> {
    if !use_metal {
        return None;
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match larql_compute_metal::MetalBackend::new() {
            Some(b) => {
                eprintln!("Metal backend: initialized (GPU-accelerated forward passes)");
                Some(Box::new(b))
            }
            None => {
                eprintln!("Metal backend: MetalBackend::new() returned None — falling back to CPU");
                None
            }
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        eprintln!(
            "Metal backend: not compiled in — rebuild with `--features metal` on macOS. \
             Falling back to CPU."
        );
        None
    }
}

/// Run a full baseline forward pass on Metal — no intervention, all heads intact.
///
/// Returns the hidden state for ALL positions `[seq_len × hidden]`.
/// Returns `None` if the backend does not support the path; callers fall back to CPU.
pub(super) fn try_metal_baseline(
    weights: &larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &larql_vindex::VectorIndex,
    backend: &Backend,
) -> Option<ndarray::Array2<f32>> {
    larql_inference::vindex::predict_kquant_metal_hidden(
        weights,
        token_ids,
        index,
        backend.as_ref(),
    )
}

/// Capture the target head's pre-W_O output at `target_layer` using GPU,
/// stopping after that layer. Returns `[seq_len × head_dim]` f32.
///
/// ~34× faster than CPU oracle PQ for target_layer=0 (only runs 1/34 GPU layers).
pub(super) fn try_metal_capture_pre_wo(
    weights: &larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &larql_vindex::VectorIndex,
    target_layer: usize,
    target_head: usize,
    backend: &Backend,
) -> Option<Vec<f32>> {
    larql_inference::vindex::predict_kquant_metal_capture_pre_wo(
        weights,
        token_ids,
        index,
        backend.as_ref(),
        target_layer,
        target_head,
    )
}

/// Run the Metal head-replacement forward pass.
///
/// Returns `None` if the backend is not provided or the GPU path fails —
/// callers then run the CPU path.
pub(super) fn try_metal(
    weights: &larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &larql_vindex::VectorIndex,
    head_layer: usize,
    head_idx: usize,
    replacement_delta: &ndarray::Array2<f32>,
    backend: &Backend,
) -> Option<ndarray::Array2<f32>> {
    larql_inference::vindex::predict_kquant_metal_with_replaced_head_residual_delta(
        weights,
        token_ids,
        index,
        backend.as_ref(),
        head_layer,
        head_idx,
        replacement_delta,
    )
}
