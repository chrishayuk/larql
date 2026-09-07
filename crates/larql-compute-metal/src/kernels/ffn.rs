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

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, Library};

use crate::kernels::KernelHandle;
use crate::shaders;

/// Pipeline registry for FFN dispatch (gate+up, activation, down).
pub struct FfnKernels {
    // Gated FFN activations (`act(gate) * up`).
    pub geglu_pipeline: ComputePipelineState,
    pub geglu_gelu_tanh_pipeline: ComputePipelineState,
    /// GPT-OSS's clamped GLU with fused gate/up bias adds — the MoE
    /// expert activation for `MoeGateRule::ClampedGlu` layers.
    pub clamped_glu_bias_pipeline: ComputePipelineState,
    /// Kimi-K3's SiTU-GLU combine (K3-ACT-1).
    pub situ_glu_pipeline: ComputePipelineState,
    /// GPU weighted MoE combine (`new_h = h_post_attn + Σ w·(out+bias)`)
    /// for identity-combine policies — lets experts + next-layer
    /// attention share one command buffer.
    pub moe_weighted_combine_pipeline: ComputePipelineState,

    // Standard (non-gated) FFN activations.
    pub silu_pipeline: ComputePipelineState,
    pub gelu_tanh_pipeline: ComputePipelineState,

    // Q4_K gate+up (production + three opt-in variants).
    pub q4k_ffn_gate_up_pipeline: KernelHandle,
    /// `LARQL_F16_ACC=1` opt-in (requires `LARQL_GATE_UP_8SG=0` too —
    /// the 8sg default discards it). f16 inner accumulator on the
    /// legacy 4sg gate+up.
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
    /// "Cached-activation" Q6_K + GELU-tanh — **never dispatched**:
    /// `LARQL_FUSED_Q6K_DOWN=1` binds the non-cached pipeline above,
    /// and this kernel's body is a verbatim copy of it (it caches
    /// nothing, contradicting its module doc; audit F21/F22).
    /// Retire-or-implement is tracked on the larql-compute roadmap.
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
            clamped_glu_bias_pipeline: r::<shaders::geglu::ClampedGluBiasKernel>(device, library),
            situ_glu_pipeline: r::<shaders::geglu::SituGluKernel>(device, library),
            moe_weighted_combine_pipeline: r::<shaders::moe_weighted_combine::Kernel>(
                device, library,
            ),

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

/// Bind SiTU-GLU for one expert slot (K3-ACT-1).
///
/// One function for all three routed dispatch sites — they differ only in
/// which scratch offsets they hand it, and a combine transcribed three
/// times is three chances to transcribe it differently. The caller still
/// owns the dispatch, because the three sites size their grids from
/// different places.
///
/// **Expert biases are refused, not dropped.** `situ_glu` has no bias
/// slots: Kimi-K3's experts are bias-free `nn.Linear`s, and the reference
/// gives no composition of a bias with SiTU to transcribe. Ignoring a
/// staged bias here would be the same silent substitution this rung
/// exists to remove, so it asserts instead — reachable only from a SiTU
/// checkpoint that ships expert biases, which would be new architecture
/// and owes its own rung.
#[allow(clippy::too_many_arguments)]
pub fn bind_situ_glu(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    gate: (&Buffer, u64),
    up: (&Buffer, u64),
    out: (&Buffer, u64),
    inter: u32,
    beta: f32,
    linear_beta: Option<f32>,
    staged_biases: bool,
) {
    assert!(
        !staged_biases,
        "this layer stages per-expert gate/up biases and its combine is SiTU-GLU, which has no \
         bias form in the reference; refusing rather than dropping them"
    );
    // `None` is a different function from an infinite bound, so the flag
    // carries on the GPU exactly what `Option<f32>` carries on the CPU.
    let has_linear: u32 = u32::from(linear_beta.is_some());
    let linear = linear_beta.unwrap_or(1.0);

    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(gate.0), gate.1);
    enc.set_buffer(1, Some(up.0), up.1);
    enc.set_buffer(2, Some(out.0), out.1);
    enc.set_bytes(3, 4, &inter as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &beta as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &linear as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &has_linear as *const u32 as *const std::ffi::c_void);
}

/// The one message every refusal of GLM-5.3-Flash's combine quotes —
/// the admission gates and the three dispatch backstops behind them —
/// so a caller reads the same sentence whichever surface caught it.
pub const CLAMPED_GATED_REFUSAL: &str =
    "no Metal expert-activation kernel for MoeGateRule::ClampedGated (GLM-5.3-Flash's \
     clamped SwiGLU); the ClampedGlu shader computes a different function and must not \
     stand in for it";

/// Whether this registry has an expert-activation kernel for a routed
/// layer's combine rule.
///
/// One rule has none: GLM-5.3-Flash's [`MoeGateRule::ClampedGated`] —
/// clamp the gate and up, then plain SwiGLU. The nearest shader,
/// `clamped_glu_bias`, computes `(u+1)·g·σ(αg)`, which is a DIFFERENT
/// function. Standing it in reads as a relative 31.7 against the
/// reference with every shape closing, which is the signature of a
/// silent substitution rather than of a crash.
///
/// Answered HERE so the three routed dispatches — the descriptor
/// dispatch, the zero-copy fast path, and the GPU route — refuse from
/// one fact rather than three copies of it, and so they refuse at
/// ADMISSION, before a command encoder exists. A refusal raised
/// mid-encode leaves the encoder unended and Metal aborts the process
/// instead of reporting it, so the caller never learns which layer was
/// at fault; that is the reason [`bind_situ_glu`]'s bias assert is
/// paired with an admission check in `gpu_route_supported` rather than
/// left to fire alone.
///
/// [`MoeGateRule::ClampedGated`]: larql_compute::MoeGateRule::ClampedGated
pub fn expert_activation_supported(gate_rule: &larql_compute::MoeGateRule) -> bool {
    !matches!(gate_rule, larql_compute::MoeGateRule::ClampedGated { .. })
}
