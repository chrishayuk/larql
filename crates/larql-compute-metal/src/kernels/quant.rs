//! Quantised matvec / matmul / quantize-input pipeline registry.
//!
//! Second of four planned `MetalBackend` registries (M3) — see
//! `norm_kernels.rs` for the pattern. Groups the **format-primitive**
//! pipelines: per-format matvec / matmul kernels and the f32 → Q8
//! quantiser. Stage-shaped kernels (QKV projection, FFN gate+up /
//! geglu+down) live in their own registries (M3 next steps).
//!
//! ## What's here vs what isn't
//!
//! - **Here**: `q4k_matvec` (4sg / 8sg / stride32 variants), the
//!   `q4k_matvec_pipeline` alias chosen at startup, `q4k_matmul`,
//!   `q6k_matvec` (4sg / 8sg variants + alias), `q8_matvec`, and the
//!   element-wise f32 → Q8 `q8_quant` kernel.
//! - **Elsewhere**: `q4k_qkv_proj` / `q4kf_qkv_proj` (attention
//!   stage — AttentionKernels in the next M3 step), `q4k_ffn_gate_up`
//!   / `q4k_geglu_*_down` / `q6k_geglu_*_down` (FFN stage — FfnKernels),
//!   and the existing `Q4Pipelines` sub-struct (`backend.q4`) which
//!   already bundles the legacy Q4_0 trio.
//!
//! ## Production-active matvec selection
//!
//! `q4k_matvec_pipeline` and `q6k_matvec_pipeline` are picked at build
//! time from [`BackendOptions`]. The 4sg / 8sg variants stay accessible
//! so per-kernel benches and parity tests can dispatch them explicitly.

use metal::{ComputePipelineState, Device, Library};

use crate::kernels::KernelHandle;
use crate::options::BackendOptions;
use crate::shaders;

/// Pipeline registry for quantised primitive matvec / matmul kernels
/// and the `f32 → Q8` quantiser.
pub struct QuantKernels {
    pub q8_quant_pipeline: ComputePipelineState,
    pub q8_matvec_pipeline: KernelHandle,

    /// Production-active Q4_K matvec — picked from [`BackendOptions`]
    /// at construction (`q4k_matvec_use_4sg` flips between the two).
    pub q4k_matvec_pipeline: KernelHandle,
    pub q4k_matvec_4sg_pipeline: KernelHandle,
    pub q4k_matvec_8sg_pipeline: KernelHandle,
    /// Stride-32 lane access variant of `q4k_matvec`. Bit-identical
    /// reduction tree to `f16_gemv`. Currently opt-in (no production
    /// caller); kept as the close-call lm_head insurance kernel.
    pub q4k_matvec_stride32_pipeline: KernelHandle,
    /// Q4_K gemm — used by the prefill amortisation experiments.
    pub q4k_matmul_pipeline: KernelHandle,

    /// Production-active Q6_K matvec — picked from [`BackendOptions`]
    /// at construction (`q6k_use_8sg` flips between the two).
    pub q6k_matvec_pipeline: KernelHandle,
    pub q6k_matvec_4sg_pipeline: KernelHandle,
    pub q6k_matvec_8sg_pipeline: KernelHandle,
}

impl QuantKernels {
    /// Build every pipeline in the registry.  Picks the production
    /// `q4k_matvec_pipeline` and `q6k_matvec_pipeline` aliases from
    /// `options`.  Panics if any individual pipeline fails to compile
    /// — same rationale as
    /// [`NormKernels::build`](super::norm::NormKernels::build).
    pub fn build(device: &Device, library: &Library, options: &BackendOptions) -> Self {
        use crate::kernels::{compile_required as r, compile_required_handle as h};

        let q8_quant_pipeline = r::<shaders::quantize_q8::Kernel>(device, library);
        let q8_matvec_pipeline = h::<shaders::q8_matvec::Kernel>(device, library);

        let q4k_matvec_4sg_pipeline = h::<shaders::q4k_matvec::Kernel>(device, library);
        let q4k_matvec_8sg_pipeline = h::<shaders::q4k_matvec_8sg::Kernel>(device, library);
        let q4k_matvec_stride32_pipeline =
            h::<shaders::q4k_matvec_stride32::Kernel>(device, library);
        let q4k_matvec_pipeline = if options.q4k_matvec_use_4sg {
            q4k_matvec_4sg_pipeline.clone()
        } else {
            q4k_matvec_8sg_pipeline.clone()
        };
        let q4k_matmul_pipeline = h::<shaders::q4k_matmul::Kernel>(device, library);

        let q6k_matvec_4sg_pipeline = h::<shaders::q6k_matvec::Kernel>(device, library);
        let q6k_matvec_8sg_pipeline = h::<shaders::q6k_matvec_8sg::Kernel>(device, library);
        let q6k_matvec_pipeline = if options.q6k_use_8sg {
            q6k_matvec_8sg_pipeline.clone()
        } else {
            q6k_matvec_4sg_pipeline.clone()
        };

        Self {
            q8_quant_pipeline,
            q8_matvec_pipeline,
            q4k_matvec_pipeline,
            q4k_matvec_4sg_pipeline,
            q4k_matvec_8sg_pipeline,
            q4k_matvec_stride32_pipeline,
            q4k_matmul_pipeline,
            q6k_matvec_pipeline,
            q6k_matvec_4sg_pipeline,
            q6k_matvec_8sg_pipeline,
        }
    }
}
