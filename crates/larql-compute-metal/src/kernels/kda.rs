//! KDA's non-projection stage pipelines, bundled.
//!
//! One struct per operator family, mirroring `FfnKernels`/`NormKernels`.
//! These are the stages rung 5c moves on device so a KDA layer costs one
//! CPU↔GPU crossing instead of two — see
//! [`crate::trait_impl::kda`] for why that is the point rather than the
//! kernels' own speed.

use metal::{ComputePipelineState, Device, Library};

use crate::shaders;

pub struct KdaKernels {
    /// Depthwise causal convolution + SiLU, sliding the window in place.
    pub short_conv_silu: ComputePipelineState,
    /// Per-head L2 normalisation of q and k.
    pub l2_normalise_heads: ComputePipelineState,
    /// `-exp(a_log) * softplus(f_low + dt_bias)`.
    pub decay_gate: ComputePipelineState,
    /// `sigmoid(b_proj x)`.
    pub beta_sigmoid: ComputePipelineState,
    /// The delta rule against device-resident state.
    pub recurrence: ComputePipelineState,
    /// Gated RMSNorm over one head's width.
    pub gated_rms_norm: ComputePipelineState,
}

impl KdaKernels {
    /// Build every pipeline in the registry. Panics if any individual
    /// pipeline fails to compile — the same rationale as
    /// [`FfnKernels::build`](super::ffn::FfnKernels::build), and with a
    /// sharper edge here: `MetalBackend::new` reports a failed shader
    /// library and an absent device identically as `None`, so a `?` on
    /// this path turns a broken build into "no Metal on this host" and
    /// every gate downstream skips green. That happened once, to an
    /// `log1p` MSL does not have.
    pub fn build(device: &Device, library: &Library) -> Self {
        use crate::kernels::compile_required as r;
        Self {
            short_conv_silu: r::<shaders::kda::ShortConvSiluKernel>(device, library),
            l2_normalise_heads: r::<shaders::kda::L2NormaliseHeadsKernel>(device, library),
            decay_gate: r::<shaders::kda::DecayGateKernel>(device, library),
            beta_sigmoid: r::<shaders::kda::BetaSigmoidKernel>(device, library),
            recurrence: r::<shaders::kda::RecurrenceKernel>(device, library),
            gated_rms_norm: r::<shaders::kda::GatedRmsNormKernel>(device, library),
        }
    }
}

/// The router→expert-binding seam and the device-weighted MoE combine —
/// the two kernels a Kimi decoder layer needs beyond KDA and the experts.
pub struct KimiLayerKernels {
    /// Sigmoid + correction bias + deterministic top-k + renorm + scale,
    /// writing the expert offset table the grouped kernel then reads.
    pub router_select: ComputePipelineState,
    /// `residual + Σ w·expert`, with the weights read from a DEVICE
    /// buffer rather than pushed from the host.
    pub moe_combine: ComputePipelineState,
}

impl KimiLayerKernels {
    /// Panics if either fails to compile — same rationale as
    /// [`KdaKernels::build`].
    pub fn build(device: &Device, library: &Library) -> Self {
        use crate::kernels::compile_required as r;
        Self {
            router_select: r::<shaders::kimi_layer::RouterSelectKernel>(device, library),
            moe_combine: r::<shaders::kimi_layer::MoeCombineKernel>(device, library),
        }
    }
}

/// MLA's two non-matvec stages: the per-position latent norm, and the
/// attention over the resident compressed cache.
pub struct MlaKernels {
    pub kv_a_norm: ComputePipelineState,
    pub attention: ComputePipelineState,
}

impl MlaKernels {
    /// Panics if either fails to compile — same rationale as
    /// [`KdaKernels::build`].
    pub fn build(device: &Device, library: &Library) -> Self {
        use crate::kernels::compile_required as r;
        Self {
            kv_a_norm: r::<shaders::mla::KvANormKernel>(device, library),
            attention: r::<shaders::mla::AttentionKernel>(device, library),
        }
    }
}
