//! Attention dispatch + RoPE + QKV-projection pipeline registry.
//!
//! Third of four planned `MetalBackend` registries (M3) — see
//! `norm_kernels.rs` for the pattern. Groups:
//!
//! - **Plain attention**: `causal_attn`, `fused_attn` (RoPE + GQA +
//!   softcap fused).
//! - **KV-cached attention**: `kv_attend` (short-span TG-array path),
//!   `kv_attend_long` (auto-selected past `SHORT_ATTENTION_SPAN`),
//!   `kv_append` (write new K/V row to cache).
//! - **Fused KV+attend / qk-norm+rope+kv+attend**: the
//!   `kv_append_attend_fused` and `attn_fused` triple/quadruple-fusion
//!   kernels (default-on; opt-out via env).
//! - **RoPE variants**: `rope_at_pos`, `rope_at_pos_batched`,
//!   `rope_at_pos_batched_qk`.
//! - **QKV projection kernels**: `q4k_qkv_proj`, `q4kf_qkv_proj`,
//!   `q4k_q6k_qkv_proj` (mixed-Q4K-Q6KV), `q4k_q6k_qkv_proj_normed`
//!   (norm-fused opt-in), `q4k_proj`, `q4kf_proj`, `q8_qkv_proj`.
//!
//! Why these belong together: every `encode_*` site that opens an
//! attention encoder reaches into this group; bundling removes 17
//! `pub` fields from `MetalBackend`.

use metal::{ComputePipelineState, Device, Library};

use crate::kernels::KernelHandle;
use crate::shaders;

/// Pipeline registry for attention, RoPE, KV-cache + QKV projection.
pub struct AttentionKernels {
    pub causal_attn_pipeline: ComputePipelineState,
    pub fused_attn_pipeline: ComputePipelineState,

    pub kv_attend_pipeline: ComputePipelineState,
    pub kv_attend_long_pipeline: ComputePipelineState,
    pub kv_append_pipeline: ComputePipelineState,
    /// Default-on; opt out via `LARQL_FUSED_KV_APPEND_ATTEND=0`.
    pub kv_append_attend_fused_pipeline: ComputePipelineState,
    /// Default-on; opt out via `LARQL_FUSED_ATTN=0`.
    pub attn_fused_pipeline: ComputePipelineState,

    pub rope_at_pos_pipeline: ComputePipelineState,
    pub rope_at_pos_batched_pipeline: ComputePipelineState,
    pub rope_at_pos_batched_qk_pipeline: ComputePipelineState,

    pub q4k_qkv_proj_pipeline: KernelHandle,
    /// Fused mixed-quant QKV: Q4_K Q/K rows + Q6_K V rows in one
    /// dispatch (Gemma 3 4B / Gemma 4 ship V as Q6_K).
    pub q4k_q6k_qkv_proj_pipeline: KernelHandle,
    /// Norm-fused alternative to `q4k_q6k_qkv_proj_pipeline`. Opt-in
    /// via `LARQL_QKV_FUSED=1` (defused as default 2026-05-09 per
    /// ADR-016 — bandwidth cost outweighed dispatch saving).
    pub q4k_q6k_qkv_proj_normed_pipeline: KernelHandle,
    pub q4k_proj_pipeline: KernelHandle,
    pub q4kf_qkv_proj_pipeline: KernelHandle,
    pub q4kf_proj_pipeline: KernelHandle,
    pub q8_qkv_proj_pipeline: KernelHandle,
}

impl AttentionKernels {
    /// Build every pipeline in the registry.  Panics if any individual
    /// pipeline fails to compile — same rationale as
    /// [`NormKernels::build`](super::norm::NormKernels::build).
    pub fn build(device: &Device, library: &Library) -> Self {
        use crate::kernels::{compile_required as r, compile_required_handle as h};
        Self {
            causal_attn_pipeline: r::<shaders::causal_attention::Kernel>(device, library),
            fused_attn_pipeline: r::<shaders::fused_attention::Kernel>(device, library),

            kv_attend_pipeline: r::<shaders::kv_attention::AttendKernel>(device, library),
            kv_attend_long_pipeline: r::<shaders::kv_attention::AttendLongKernel>(device, library),
            kv_append_pipeline: r::<shaders::kv_attention::AppendKernel>(device, library),
            kv_append_attend_fused_pipeline: r::<shaders::kv_append_attend_fused::Kernel>(
                device, library,
            ),
            attn_fused_pipeline: r::<shaders::attn_fused::Kernel>(device, library),

            rope_at_pos_pipeline: r::<shaders::rope::RopeAtPosKernel>(device, library),
            rope_at_pos_batched_pipeline: r::<shaders::rope::RopeAtPosBatchedKernel>(
                device, library,
            ),
            rope_at_pos_batched_qk_pipeline: r::<shaders::rope::RopeAtPosBatchedQkKernel>(
                device, library,
            ),

            q4k_qkv_proj_pipeline: h::<shaders::q4k_qkv_proj::QkvKernel>(device, library),
            q4k_q6k_qkv_proj_pipeline: h::<shaders::q4k_q6k_qkv_proj::Kernel>(device, library),
            q4k_q6k_qkv_proj_normed_pipeline: h::<shaders::q4k_q6k_qkv_proj::NormedKernel>(
                device, library,
            ),
            q4k_proj_pipeline: h::<shaders::q4k_qkv_proj::ProjKernel>(device, library),
            q4kf_qkv_proj_pipeline: h::<shaders::q4kf_qkv_proj::QkvKernel>(device, library),
            q4kf_proj_pipeline: h::<shaders::q4kf_qkv_proj::ProjKernel>(device, library),
            q8_qkv_proj_pipeline: h::<shaders::q8_attn_proj::QkvKernel>(device, library),
        }
    }
}
