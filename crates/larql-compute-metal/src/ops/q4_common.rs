//! Shared Q4 utilities and pipeline state container.

use metal::ComputePipelineState;

use crate::kernels::KernelHandle;
use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;

/// Pipeline states for Q4 operations — compiled from modular shaders.
///
/// `matvec` is a [`KernelHandle`] because its kernel uses simdgroup
/// row-tiling — the dispatcher must agree with the kernel's hardcoded
/// row map. The handle bundles geometry with the pipeline so they
/// cannot drift apart (see `metal::kernel` module docs).
///
/// `vecmat` and `f32_matvec` use flat `dispatch_threads` and don't
/// have per-TG row geometry; bare [`ComputePipelineState`] is enough.
pub struct Q4Pipelines {
    /// Q4 × Q8 matvec (simdgroup-tiled, currently `q4_matvec_v4`).
    pub matvec: KernelHandle,
    /// Q4 vector-matrix scatter (flat dispatch, currently `q4_vecmat`).
    pub vecmat: ComputePipelineState,
    /// Q4 × f32 matvec for transposed down projection (one thread
    /// per output row, currently `q4_f32_matvec`).
    pub f32_matvec: ComputePipelineState,
}

/// Pre-quantize f32 vector to Q8_0 (int8 + per-block f32 scale).
/// Used to prepare input for Q4×Q8 dot product.
pub fn quantize_to_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n_blocks = x.len() / LEGACY_BLOCK_ELEMS;
    let mut q8 = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; n_blocks];
    for (b, scale_out) in scales.iter_mut().enumerate().take(n_blocks) {
        let off = b * LEGACY_BLOCK_ELEMS;
        let block = &x[off..off + LEGACY_BLOCK_ELEMS];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        *scale_out = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for j in 0..LEGACY_BLOCK_ELEMS {
            q8[off + j] = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (q8, scales)
}
