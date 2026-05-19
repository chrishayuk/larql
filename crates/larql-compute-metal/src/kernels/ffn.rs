//! FFN dispatch + activation pipeline registry.
//!
//! Last of the four planned `MetalBackend` registries (M3) — see
//! `norm_kernels.rs` for the pattern. Groups every pipeline that
//! `encode_ffn.rs` and `stages/ffn.rs` reach into:
//!
//! - **Element-wise activations**: `silu`, `gelu_tanh` (Standard FFN
//!   path) and the gated `geglu_*` twins.
//! - **Q4_K gate+up**: production kernel + the three opt-in variants
//!   (`f16acc`, `8sg`, `coop`) that the auto-memory ship-log keeps
//!   alive as opt-ins (`LARQL_F16_ACC`, `LARQL_GATE_UP_8SG`,
//!   `LARQL_GATE_UP_COOP`).
//! - **Q4_KF gate+up**: llama.cpp-exact pre-baked-scales fast path.
//! - **Fused activation+down**: Q4_K silu/geltanh, Q6_K silu/geltanh,
//!   plus the cached-activation Q6_K geltanh variant
//!   (`LARQL_FUSED_Q6K_DOWN` opt-in, currently no-op pending kernel
//!   parity per `encode_ffn.rs` block doc).
//!
//! Why these belong together: every FFN dispatch site reads more than
//! one of these in the same scope. Bundling removes 14 `pub` fields
//! from the top-level `MetalBackend` struct.

use metal::{ComputePipelineState, Device, Library};

use crate::kernels::KernelHandle;
use crate::shaders;

/// Pipeline registry for FFN dispatch (gate+up, activation, down).
pub struct FfnKernels {
    // Gated FFN activations (`act(gate) * up`).
    pub geglu_pipeline: ComputePipelineState,
    pub geglu_gelu_tanh_pipeline: ComputePipelineState,

    // Standard (non-gated) FFN activations.
    pub silu_pipeline: ComputePipelineState,
    pub gelu_tanh_pipeline: ComputePipelineState,

    // Q4_K gate+up (production + three opt-in variants).
    pub q4k_ffn_gate_up_pipeline: KernelHandle,
    /// `LARQL_F16_ACC=1` opt-in. f16 inner accumulator on the legacy
    /// 4sg gate+up.
    pub q4k_ffn_gate_up_f16acc_pipeline: KernelHandle,
    /// `LARQL_GATE_UP_8SG=0` opts back to 4sg; this is the 8sg variant
    /// that the production alias resolves to today.
    pub q4k_ffn_gate_up_8sg_pipeline: KernelHandle,
    /// `LARQL_GATE_UP_COOP=1` opt-in cooperative-scale-load variant.
    pub q4k_ffn_gate_up_coop_pipeline: KernelHandle,

    // Q4_KF gate+up.
    pub q4kf_ffn_gate_up_pipeline: KernelHandle,

    // Fused activation+down — Q4_K and Q6_K twins.
    pub q4k_geglu_silu_down_pipeline: KernelHandle,
    pub q4k_geglu_gelu_tanh_down_pipeline: KernelHandle,
    pub q6k_geglu_silu_down_pipeline: KernelHandle,
    pub q6k_geglu_gelu_tanh_down_pipeline: KernelHandle,
    /// Cached-activation Q6_K + GELU-tanh — `LARQL_FUSED_Q6K_DOWN=1`
    /// opt-in. Currently no-op until kernel-level parity work lands.
    pub q6k_geglu_gelu_tanh_down_cached_pipeline: KernelHandle,

    /// Per-Layer Embeddings gate-apply (Gemma 4 E2B): fused
    /// `gate = gelu_tanh(gate) * per_layer_input`. Wired by the PLE
    /// dispatch helper between the two PLE matvecs (gate proj → up proj).
    pub ple_gate_apply_pipeline: ComputePipelineState,
}

impl FfnKernels {
    /// Build every pipeline in the registry.  Panics if any individual
    /// pipeline fails to compile — same rationale as
    /// [`NormKernels::build`](super::norm::NormKernels::build).
    pub fn build(device: &Device, library: &Library) -> Self {
        use crate::kernels::{compile_required as r, compile_required_handle as h};
        Self {
            geglu_pipeline: r::<shaders::geglu::SiluKernel>(device, library),
            geglu_gelu_tanh_pipeline: r::<shaders::geglu::GeluTanhKernel>(device, library),

            silu_pipeline: r::<shaders::activation::SiluKernel>(device, library),
            gelu_tanh_pipeline: r::<shaders::activation::GeluTanhKernel>(device, library),

            q4k_ffn_gate_up_pipeline: h::<shaders::q4k_ffn_gate_up::Kernel>(device, library),
            q4k_ffn_gate_up_f16acc_pipeline: h::<shaders::q4k_ffn_gate_up_f16acc::Kernel>(
                device, library,
            ),
            q4k_ffn_gate_up_8sg_pipeline: h::<shaders::q4k_ffn_gate_up_8sg::Kernel>(
                device, library,
            ),
            q4k_ffn_gate_up_coop_pipeline: h::<shaders::q4k_ffn_gate_up_coop::Kernel>(
                device, library,
            ),

            q4kf_ffn_gate_up_pipeline: h::<shaders::q4kf_ffn_gate_up::Kernel>(device, library),

            q4k_geglu_silu_down_pipeline: h::<shaders::q4k_geglu_down::SiluKernel>(device, library),
            q4k_geglu_gelu_tanh_down_pipeline: h::<shaders::q4k_geglu_down::GeluTanhKernel>(
                device, library,
            ),
            q6k_geglu_silu_down_pipeline: h::<shaders::q6k_geglu_down::SiluKernel>(device, library),
            q6k_geglu_gelu_tanh_down_pipeline: h::<shaders::q6k_geglu_down::GeluTanhKernel>(
                device, library,
            ),
            q6k_geglu_gelu_tanh_down_cached_pipeline: h::<
                shaders::q6k_geglu_gelu_tanh_down_cached::Kernel,
            >(device, library),

            ple_gate_apply_pipeline: r::<shaders::per_layer_embed::GateApplyKernel>(
                device, library,
            ),
        }
    }
}
