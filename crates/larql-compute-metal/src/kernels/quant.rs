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

    /// Direct MXFP4 matvec — consumes packed nibbles + e8m0 scales with no
    /// f32 materialisation (K1 of the fused-MXFP4 ladder).
    pub mxfp4_matvec_pipeline: KernelHandle,

    /// Direct NVFP4 matvec — the same E2M1 elements read through E4M3
    /// group scales plus one f32 tensor scale (VINDEX3-Q2). Sibling of
    /// `mxfp4_matvec_pipeline`, differing only in the scale geometry.
    pub nvfp4_matvec_pipeline: KernelHandle,
    /// `nvfp4_matvec_v2` — arithmetic E2M1 decode, vector loads; the
    /// falsified A-5 hypothesis, retained as an explicit arm
    /// (`LARQL_NVFP4_KERNEL=v2`; default v1). Same values to fp32 rounding.
    pub nvfp4_matvec_v2_pipeline: KernelHandle,
    /// A-5 sweep arms — v1's inner loop at (groups per lane, rows per
    /// TG) ∈ {(2,4),(4,4),(1,2),(1,8),(2,2),(2,8)}; `examples/nvfp4_gemv_shapes.rs`.
    pub nvfp4_sweep_pipelines: [KernelHandle; 11],
    /// A-5b: segmented x2 — Q+K+V or gate+up in one dispatch.
    pub nvfp4_matvec_x2_seg3_pipeline: KernelHandle,
    /// seg3t — per-threadgroup segment resolution (the production form;
    /// seg3's per-row-pair resolve cost 4.8 µs/dispatch at the QKV shape).
    pub nvfp4_matvec_x2_seg3t_pipeline: KernelHandle,
    /// A-5b rung 2a: x2 with the residual add folded into the write.
    pub nvfp4_matvec_x2r_pipeline: KernelHandle,
    /// A-5b rung 2d: x2 with the pre-norm folded into the prologue.
    pub nvfp4_matvec_x2n_pipeline: KernelHandle,
    /// Rung 2d form B: pre-norm staged in threadgroup memory.
    pub nvfp4_matvec_x2m_pipeline: KernelHandle,

    /// Q6_K grouped-expert matvec: every selected expert in one dispatch, so
    /// the grid carries 16x the threadgroups of a single expert matrix (K3a).
    pub q6k_grouped_experts_pipeline: KernelHandle,
    /// Q4_K sibling — what the engine's MoE down projection needs.
    pub q4k_grouped_experts_pipeline: KernelHandle,
    /// Q8_0 sibling — the precision ladder's middle rung (8.5 bpw),
    /// canonical ggml 34-byte blocks. Same grouped ABI as the other two.
    pub q8_0_grouped_experts_pipeline: KernelHandle,

    /// K2 layout/decode tournament: four MXFP4 grouped-expert arms that read
    /// the same weights through different scale layouts and decode strategies.
    /// Candidates, not production paths — see `shaders::mxfp4_grouped_experts`.
    pub mxfp4g_split_lut16_pipeline: KernelHandle,
    /// Arm A2: arm A's layout and math with a vectorised skeleton
    /// (`uint4` weight loads, `float4` X loads). Requires 16-byte-aligned
    /// payload region bases; see the kernel docs.
    pub mxfp4g_split_lut16_vec_pipeline: KernelHandle,
    pub mxfp4g_inter_lut16_pipeline: KernelHandle,
    pub mxfp4g_inter_pair_pipeline: KernelHandle,
    pub mxfp4g_inter_magsign_pipeline: KernelHandle,
    /// Ceiling probes, not candidates: trivial affine decode, and weights-only
    /// (no X gather). They bracket how much of the gap is decode vs skeleton.
    pub mxfp4g_inter_bits_pipeline: KernelHandle,
    pub mxfp4g_inter_affine_pipeline: KernelHandle,
    pub mxfp4g_inter_nox_pipeline: KernelHandle,

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

    /// Production-active MXFP4 grouped-expert arm — picked from
    /// [`BackendOptions`] at construction (`mxfp4_arm`), the same way the
    /// Q4_K and Q6_K aliases above are. Resolved once here rather than
    /// per dispatch: the arm is a startup choice, and reading it inside
    /// the encode path would put an env lookup on the per-layer hot path.
    pub mxfp4_grouped_pipeline: KernelHandle,
    /// A2x2 — the vec arm with two rows per simdgroup sharing X loads
    /// (A-12 expert pass). Bit-identical per row to the vec arm; only
    /// valid where the vec arm is (16-byte-aligned offsets).
    pub mxfp4_grouped_x2_pipeline: KernelHandle,
    /// A2x2gu — gate+up halves in one dispatch (buffers 11–13 add the
    /// second output and row walk).
    pub mxfp4_grouped_x2_gu_pipeline: KernelHandle,
    /// A2dc — down projection + weighted combine in one dispatch (top-4).
    pub mxfp4_down_combine4_pipeline: KernelHandle,
    /// A2x2p / A2x4 — the 313→346 candidate arms (byte-pair LUT; four
    /// rows per lane).
    pub mxfp4_grouped_x2p_pipeline: KernelHandle,
    pub mxfp4_grouped_x4_pipeline: KernelHandle,
    /// How [`Self::mxfp4_grouped_pipeline`] receives its e8m0 exponents.
    /// Travels with the pipeline because the two layouts have different
    /// binding arities, so a call site cannot bind one without knowing
    /// which it got.
    pub mxfp4_grouped_binding: ExpertScaleBinding,
    /// Which arm [`Self::mxfp4_grouped_pipeline`] is. The encode path
    /// needs the identity, not just the pipeline: the vectorised arm
    /// carries an alignment precondition it must check per descriptor
    /// table, falling back to the scalar split arm when it fails.
    pub mxfp4_grouped_arm: shaders::mxfp4_grouped_experts::Mxfp4Arm,
}

/// How a grouped-expert kernel receives its dequantisation scales.
///
/// The arity difference is real, not cosmetic:
///
/// | binding | buffers |
/// |---|---|
/// | [`Self::InlineScales`] | `W(0) offsets(1) X(2) out(3) N(4) K(5) XSTRIDE(6)` |
/// | [`Self::SplitE8M0`] | `Wp(0) offsets(1) Ws(2) s_offsets(3) X(4) out(5) N(6) K(7) XSTRIDE(8) ROWBASE(9) ROWSTRIDE(10)` |
///
/// Binding an inline-scale call site against a split-scale kernel puts
/// activations where the kernel reads exponents, which decodes silently
/// and wrongly.
///
/// The two extra pairs on the split arm are not symmetry for its own sake.
/// `s_offsets` exists because the exponent stream's placement is the
/// container writer's choice, not `payload_offset / 16`; `ROWBASE`/
/// `ROWSTRIDE` exist because a fused gate/up region can arrange its two
/// halves contiguously *or* interleaved, and an inline-scale call site
/// expresses "which half" as a byte offset, which can only say the former.
/// Both are properties a stored bank has and a bench fixture does not,
/// which is why only the arm that serves stored banks carries them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExpertScaleBinding {
    /// Scales ride inside the weight stream (Q4_K, Q6_K, and the
    /// interleaved MXFP4 arms).
    ///
    /// Implies a contiguous-halves fused region: there is no binding here
    /// that could say otherwise, so a call site holding an interleaved
    /// bank must refuse rather than dispatch.
    InlineScales,
    /// A separate e8m0 exponent stream with its own offset table, plus an
    /// explicit fused-row walk (the exact MXFP4 arm).
    SplitE8M0,
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

        let mxfp4_matvec_pipeline = h::<shaders::mxfp4_matvec::Kernel>(device, library);
        let nvfp4_matvec_pipeline = h::<shaders::nvfp4_matvec::Kernel>(device, library);
        let nvfp4_matvec_v2_pipeline = h::<shaders::nvfp4_matvec::KernelV2>(device, library);
        let nvfp4_sweep_pipelines = [
            h::<shaders::nvfp4_matvec::KernelG2R4>(device, library),
            h::<shaders::nvfp4_matvec::KernelG4R4>(device, library),
            h::<shaders::nvfp4_matvec::KernelG1R2>(device, library),
            h::<shaders::nvfp4_matvec::KernelG1R8>(device, library),
            h::<shaders::nvfp4_matvec::KernelG2R2>(device, library),
            h::<shaders::nvfp4_matvec::KernelG2R8>(device, library),
            h::<shaders::nvfp4_matvec::KernelX2>(device, library),
            h::<shaders::nvfp4_matvec::KernelX4>(device, library),
            h::<shaders::nvfp4_matvec::KernelX1B>(device, library),
            h::<shaders::nvfp4_matvec::KernelX2B>(device, library),
            h::<shaders::nvfp4_matvec::KernelX4B>(device, library),
        ];
        let nvfp4_matvec_x2_seg3_pipeline =
            h::<shaders::nvfp4_matvec::KernelX2Seg3>(device, library);
        let nvfp4_matvec_x2_seg3t_pipeline =
            h::<shaders::nvfp4_matvec::KernelX2Seg3T>(device, library);
        let nvfp4_matvec_x2r_pipeline = h::<shaders::nvfp4_matvec::KernelX2R>(device, library);
        let nvfp4_matvec_x2n_pipeline = h::<shaders::nvfp4_matvec::KernelX2N>(device, library);
        let nvfp4_matvec_x2m_pipeline = h::<shaders::nvfp4_matvec::KernelX2M>(device, library);
        let q6k_grouped_experts_pipeline =
            h::<shaders::q6k_grouped_experts::Kernel>(device, library);
        let q4k_grouped_experts_pipeline =
            h::<shaders::q4k_grouped_experts::Kernel>(device, library);
        let q8_0_grouped_experts_pipeline =
            h::<shaders::q8_0_grouped_experts::Kernel>(device, library);

        let mxfp4g_split_lut16_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelSplitLut16>(device, library);
        let mxfp4g_split_lut16_vec_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelSplitLut16Vec>(device, library);
        let mxfp4g_inter_lut16_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelInterLut16>(device, library);
        let mxfp4g_inter_pair_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelInterPair>(device, library);
        let mxfp4g_inter_magsign_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelInterMagSign>(device, library);
        let mxfp4g_inter_bits_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelInterBits>(device, library);
        let mxfp4g_inter_affine_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelInterAffine>(device, library);
        let mxfp4g_inter_nox_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelInterNoX>(device, library);

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

        use shaders::mxfp4_grouped_experts::Mxfp4Arm;
        let mxfp4_grouped_pipeline = match options.mxfp4_arm {
            Mxfp4Arm::SplitLut16 => mxfp4g_split_lut16_pipeline.clone(),
            Mxfp4Arm::SplitLut16Vec => mxfp4g_split_lut16_vec_pipeline.clone(),
            Mxfp4Arm::InterLut16 => mxfp4g_inter_lut16_pipeline.clone(),
            Mxfp4Arm::InterPair => mxfp4g_inter_pair_pipeline.clone(),
            Mxfp4Arm::InterMagSign => mxfp4g_inter_magsign_pipeline.clone(),
        };
        let mxfp4_grouped_x2_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelSplitLut16VecX2>(device, library);
        let mxfp4_grouped_x2_gu_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelSplitLut16VecX2Gu>(device, library);
        let mxfp4_down_combine4_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelDownCombine4>(device, library);
        let mxfp4_grouped_x2p_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelSplitLut16VecX2P>(device, library);
        let mxfp4_grouped_x4_pipeline =
            h::<shaders::mxfp4_grouped_experts::KernelSplitLut16VecX4>(device, library);
        let mxfp4_grouped_arm = options.mxfp4_arm;
        let mxfp4_grouped_binding = if options.mxfp4_arm.is_split_scale() {
            ExpertScaleBinding::SplitE8M0
        } else {
            ExpertScaleBinding::InlineScales
        };

        Self {
            q8_quant_pipeline,
            q8_matvec_pipeline,
            mxfp4_matvec_pipeline,
            nvfp4_matvec_pipeline,
            nvfp4_matvec_v2_pipeline,
            nvfp4_sweep_pipelines,
            nvfp4_matvec_x2_seg3_pipeline,
            nvfp4_matvec_x2_seg3t_pipeline,
            nvfp4_matvec_x2r_pipeline,
            nvfp4_matvec_x2n_pipeline,
            nvfp4_matvec_x2m_pipeline,
            q6k_grouped_experts_pipeline,
            q4k_grouped_experts_pipeline,
            q8_0_grouped_experts_pipeline,
            mxfp4g_split_lut16_pipeline,
            mxfp4g_split_lut16_vec_pipeline,
            mxfp4g_inter_lut16_pipeline,
            mxfp4g_inter_pair_pipeline,
            mxfp4g_inter_magsign_pipeline,
            mxfp4g_inter_bits_pipeline,
            mxfp4g_inter_affine_pipeline,
            mxfp4g_inter_nox_pipeline,
            q4k_matvec_pipeline,
            q4k_matvec_4sg_pipeline,
            q4k_matvec_8sg_pipeline,
            q4k_matvec_stride32_pipeline,
            q4k_matmul_pipeline,
            q6k_matvec_pipeline,
            q6k_matvec_4sg_pipeline,
            q6k_matvec_8sg_pipeline,
            mxfp4_grouped_pipeline,
            mxfp4_grouped_x2_pipeline,
            mxfp4_grouped_x2_gu_pipeline,
            mxfp4_down_combine4_pipeline,
            mxfp4_grouped_x2p_pipeline,
            mxfp4_grouped_x4_pipeline,
            mxfp4_grouped_binding,
            mxfp4_grouped_arm,
        }
    }

    /// The grouped-expert kernel serving `format`, with its binding shape.
    ///
    /// Replaces a `match format { Q6_K => .., _ => q4k }` that was spelled
    /// inline at four dispatch sites. That wildcard is only safe while
    /// every format reaching it is Q4_K-shaped: MXFP4 is not, and routing
    /// its 16-byte groups through the Q4_K kernel's 144-byte superblock
    /// stride produces garbage silently — the failure class the capability
    /// audit (F4) named as the worst available.
    ///
    /// # Panics
    /// If `format` has no grouped-expert kernel. Loud by design.
    pub fn grouped_experts_for(
        &self,
        format: larql_compute::QuantFormat,
    ) -> (&KernelHandle, ExpertScaleBinding) {
        use larql_compute::QuantFormat as Q;
        match format {
            Q::Q6_K => (
                &self.q6k_grouped_experts_pipeline,
                ExpertScaleBinding::InlineScales,
            ),
            Q::Q4_K | Q::Q4_KF => (
                &self.q4k_grouped_experts_pipeline,
                ExpertScaleBinding::InlineScales,
            ),
            Q::MXFP4 => (&self.mxfp4_grouped_pipeline, self.mxfp4_grouped_binding),
            other => panic!(
                "kernels::quant: {other:?} has no grouped-expert kernel. \
                 Implemented for Q4_K, Q4_KF, Q6_K and MXFP4 — add a kernel \
                 and an arm here rather than falling through to Q4_K, which \
                 would read the wrong block stride."
            ),
        }
    }

    /// The per-expert (non-grouped) matvec kernel serving `format`, with
    /// its binding shape. The ragged-path sibling of
    /// [`Self::grouped_experts_for`]; same wildcard hazard, same loud
    /// failure.
    ///
    /// # Panics
    /// If `format` has no per-expert matvec kernel.
    pub fn expert_matvec_for(
        &self,
        format: larql_compute::QuantFormat,
    ) -> (&KernelHandle, ExpertScaleBinding) {
        use larql_compute::QuantFormat as Q;
        match format {
            Q::Q6_K => (&self.q6k_matvec_pipeline, ExpertScaleBinding::InlineScales),
            Q::Q4_K | Q::Q4_KF => (&self.q4k_matvec_pipeline, ExpertScaleBinding::InlineScales),
            // K1 of the ladder consumes the checkpoint's two streams
            // directly, so it is split regardless of the grouped arm.
            Q::MXFP4 => (&self.mxfp4_matvec_pipeline, ExpertScaleBinding::SplitE8M0),
            other => panic!(
                "kernels::quant: {other:?} has no per-expert matvec kernel. \
                 Implemented for Q4_K, Q4_KF, Q6_K and MXFP4 — add a kernel \
                 and an arm here rather than falling through to Q4_K."
            ),
        }
    }
}

#[cfg(test)]
mod tests;
