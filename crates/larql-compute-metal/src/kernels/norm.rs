//! Norm + residual + scale-vector pipeline registry.
//!
//! First of four planned `MetalBackend` registries (`NormKernels`,
//! `QuantKernels`, `AttentionKernels`, `FfnKernels`) — see modularity
//! tracker M3 in `ROADMAP.md`. Groups the pipelines that handle:
//!
//! - RMS-norm + Q8-quantised RMS-norm + residual-add (the small loop
//!   ops shared across every layer)
//! - Cooperative residual+norm fusions (`residual_norm`,
//!   `residual_norm_q8`, `residual_norm_store` — D-RMS-FUSE plumbing)
//! - LayerNorm and parameter-free V-norm (StarCoder2 / Gemma 4)
//! - QK-norm and the qk-norm + RoPE fusion (Gemma 3 / 4 attention)
//! - The `post_attn_residual_norm_store` and
//!   `post_ffn_norm_residual_add` triple/double fusions
//! - `scale_vector` (per-layer scalar multiplier — Gemma 4)
//!
//! Why these belong together: dispatch sites that touch one of these
//! almost always touch another (e.g. `encode_post_ffn` reads
//! `residual_norm_store`, `post_ffn_norm_residual_add`, and
//! `residual_add` from the same `&self`). Bundling them removes 16
//! `pub` fields from the top-level `MetalBackend` struct.
//!
//! ## Construction
//!
//! [`NormKernels::build`] takes the device and shader library and
//! produces every pipeline at once. Returns `None` (instead of
//! propagating individual errors) so the caller can keep using `?`-
//! style early returns from `MetalBackend::with_options`.

use metal::{ComputePipelineState, Device, Library};

use crate::shaders;

/// Pipeline registry for norm, residual, and scale-vector kernels.
///
/// All fields are `pub` so existing dispatch sites can read them
/// directly (`backend.norms.rms_norm_pipeline`). The registry adds an
/// organising layer; it does not narrow the surface yet.
pub struct NormKernels {
    // Plain RMS-norm + the Q8-quantised twin used by the Q4_0 / Q8_0
    // attention path.
    pub rms_norm_pipeline: ComputePipelineState,
    pub rms_norm_q8_pipeline: ComputePipelineState,

    // Cooperative residual + norm fusions. `residual_add` is the
    // unfused fallback; `residual_norm_*` are the various fused
    // variants used across the decode pipeline.
    pub residual_add_pipeline: ComputePipelineState,
    pub residual_norm_pipeline: ComputePipelineState,
    pub residual_norm_q8_pipeline: ComputePipelineState,
    /// D-RMS-FUSE Phase 1: residual_add + next-layer rms_norm in one
    /// dispatch. Opt-in via `LARQL_FUSED_PRELAYER_NORM=1`.
    pub residual_norm_store_pipeline: ComputePipelineState,

    // LayerNorm (StarCoder2 / GPT-2 family).
    pub layer_norm_pipeline: ComputePipelineState,
    pub layer_norm_no_bias_pipeline: ComputePipelineState,

    // Parameter-free RMSNorm on the V projection (Gemma 4).
    pub v_norm_pipeline: ComputePipelineState,
    pub v_norm_batched_pipeline: ComputePipelineState,

    // Per-head QK-norm (Gemma 3 / 4) and the QK-norm + RoPE fusion.
    pub qk_norm_pipeline: ComputePipelineState,
    pub qk_norm_qk_pipeline: ComputePipelineState,
    pub qk_norm_rope_fused_pipeline: ComputePipelineState,

    /// Triple fusion: `post_attn_norm + residual + ffn_norm + h_post_attn
    /// store` for the `has_post_norms` decode path.
    pub post_attn_residual_norm_store_pipeline: ComputePipelineState,
    /// Double fusion: `rms_norm(down_out) + residual_add(h_post_attn,
    /// normed_ffn)` for the `has_post_norms + post_ffn_norm` decode
    /// path. Opt out via `LARQL_FUSED_POST_FFN_NORM=0`.
    pub post_ffn_norm_residual_add_pipeline: ComputePipelineState,

    /// Per-layer scalar multiplier (Gemma 4). Element-wise; lives in
    /// the norm registry because it sits in the same residual-stream
    /// "small ops" cluster.
    pub scale_vector_pipeline: ComputePipelineState,
}

impl NormKernels {
    /// Build every pipeline in the registry. Returns `None` if any
    /// individual pipeline creation fails.
    ///
    /// Panics rather than returning `Option` because a pipeline-build
    /// failure here is an internal MSL bug (typo in `KERNEL_NAME`,
    /// undeclared function, etc.) — those are caught by the
    /// `shaders/*` unit tests, not gracefully handleable at runtime.
    /// The previous `Option<Self>` return type forced `?` operators on
    /// every line and left coverage at ~82 % because the early-return
    /// branch is structurally unreachable in production.  Switching to
    /// `expect` collapses each line to a single covered region.
    pub fn build(device: &Device, library: &Library) -> Self {
        use crate::kernels::compile_required as r;
        Self {
            rms_norm_pipeline: r::<shaders::residual_inject::RmsNormKernel>(device, library),
            rms_norm_q8_pipeline: r::<shaders::fused_ops::RmsNormQ8Kernel>(device, library),

            residual_add_pipeline: r::<shaders::residual_inject::ResidualAddKernel>(
                device, library,
            ),
            residual_norm_pipeline: r::<shaders::fused_ops::ResidualNormKernel>(device, library),
            residual_norm_q8_pipeline: r::<shaders::fused_ops::ResidualNormQ8Kernel>(
                device, library,
            ),
            residual_norm_store_pipeline: r::<shaders::fused_ops::ResidualNormStoreKernel>(
                device, library,
            ),

            layer_norm_pipeline: r::<shaders::layer_norm::Kernel>(device, library),
            layer_norm_no_bias_pipeline: r::<shaders::layer_norm::NoBiasKernel>(device, library),

            v_norm_pipeline: r::<shaders::v_norm::Kernel>(device, library),
            v_norm_batched_pipeline: r::<shaders::v_norm::BatchedKernel>(device, library),

            qk_norm_pipeline: r::<shaders::qk_norm::Kernel>(device, library),
            qk_norm_qk_pipeline: r::<shaders::qk_norm::QkKernel>(device, library),
            qk_norm_rope_fused_pipeline: r::<shaders::qk_norm_rope_fused::Kernel>(device, library),

            post_attn_residual_norm_store_pipeline: r::<
                shaders::post_attn_residual_norm_store::Kernel,
            >(device, library),
            post_ffn_norm_residual_add_pipeline: r::<shaders::post_ffn_norm_residual_add::Kernel>(
                device, library,
            ),

            scale_vector_pipeline: r::<shaders::residual_inject::ScaleVectorKernel>(
                device, library,
            ),
        }
    }
}
