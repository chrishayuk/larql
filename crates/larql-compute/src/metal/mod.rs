//! Metal GPU compute backend — Apple Silicon.
//!
//! All operations go through the [`ComputeBackend`] trait. Metal-specific
//! optimisations: simdgroup reductions, uint4 vectorised loads, native half reads,
//! zero-copy mmap buffers, single command buffer pipeline.
//!
//! ## Modules
//!
//! - `shaders/`:  Metal Shading Language — one file per kernel (~30 shader files)
//! - `ops/`:      GPU dispatch — one file per operation (6 dispatchers)
//! - `buffers`:   GPU buffer cache (zero-copy mmap, transient allocation)
//! - `f32_ops`:   f32 tiled matmul dispatch with GPU/CPU routing
//! - `calibrate`: CPU vs GPU auto-calibration
//!
//! ## Performance (M3 Max, Gemma 3 4B, 34 layers)
//!
//! - Full decode: ~0.38ms/layer, ~77 tok/s (Q4_KF path)
//! - vs Ollama: ~1.0–1.25× (at parity)
//! - Q4_K matvec: uint4 loads, 8 rows/TG, multi-row (nr0=2)
//! - KV attention: simd_max/simd_sum reductions, float4 Q·K dot products

pub mod shaders;   // modular: shaders/mod.rs → one file per shader
pub mod buffers;
pub mod f32_ops;
pub mod ops;        // modular: ops/mod.rs → one file per operation
pub mod stages;     // modular: stages/mod.rs → one file per pipeline stage
pub mod calibrate;
mod direct_ops;
mod decode;
mod decode_profile;
mod decode_hybrid;
mod pipeline;
mod prefill;
mod trait_impl;

use std::sync::atomic::{AtomicUsize, Ordering};
use ndarray::{Array2, ArrayView2};
use metal::*;

use crate::backend::{ComputeBackend, MatMulOp};
use buffers::BufferCache;
use f32_ops::F32Ops;
use ops::q4_common::Q4Pipelines;

/// Metal GPU compute backend.
pub struct MetalBackend {
    queue: CommandQueue,
    bufs: BufferCache,
    f32_ops: F32Ops,
    pub q4: Q4Pipelines,
    causal_attn_pipeline: ComputePipelineState,
    pub fused_attn_pipeline: ComputePipelineState,
    pub geglu_pipeline: ComputePipelineState,
    pub geglu_gelu_tanh_pipeline: ComputePipelineState,
    q8_quant_pipeline: ComputePipelineState,
    pub kv_attend_pipeline: ComputePipelineState,
    pub kv_append_pipeline: ComputePipelineState,
    q8_matvec_pipeline: ComputePipelineState,
    pub rms_norm_pipeline: ComputePipelineState,
    pub residual_add_pipeline: ComputePipelineState,
    q8_qkv_proj_pipeline: ComputePipelineState,
    pub q4k_matvec_pipeline: ComputePipelineState,
    pub q4k_ffn_gate_up_pipeline: ComputePipelineState,
    pub q4kf_ffn_gate_up_pipeline: ComputePipelineState,
    pub q4k_geglu_silu_down_pipeline: ComputePipelineState,
    pub q4k_geglu_gelu_tanh_down_pipeline: ComputePipelineState,
    pub q6k_matvec_pipeline: ComputePipelineState,
    #[allow(dead_code)]
    rope_pipeline: ComputePipelineState,
    pub rope_at_pos_pipeline: ComputePipelineState,
    pub rope_at_pos_batched_pipeline: ComputePipelineState,
    pub q4k_qkv_proj_pipeline: ComputePipelineState,
    /// Fused mixed-quant QKV: Q4_K Q/K rows + Q6_K V rows in one dispatch.
    /// Gemma 3 4B / Gemma 4 ship `V` as Q6_K; without this shader decode
    /// falls through to three per-projection dispatches per layer.
    pub q4k_q6k_qkv_proj_pipeline: ComputePipelineState,
    q4k_proj_pipeline: ComputePipelineState,
    pub q4kf_qkv_proj_pipeline: ComputePipelineState,
    pub q4kf_proj_pipeline: ComputePipelineState,
    // Standalone activations (non-gated FFN)
    pub silu_pipeline: ComputePipelineState,
    pub gelu_tanh_pipeline: ComputePipelineState,
    // LayerNorm (StarCoder2)
    pub layer_norm_pipeline: ComputePipelineState,
    pub layer_norm_no_bias_pipeline: ComputePipelineState,
    // V-norm (Gemma 4)
    pub v_norm_pipeline: ComputePipelineState,
    pub v_norm_batched_pipeline: ComputePipelineState,
    pub qk_norm_pipeline: ComputePipelineState,
    // Scale vector (per-layer scalar, Gemma 4)
    pub scale_vector_pipeline: ComputePipelineState,
    /// KV cache for decode mode — initialized on first decode_token call.
    kv_cache: std::sync::Mutex<Option<ops::kv_cache::KVCache>>,
    pub rms_norm_q8_pipeline: ComputePipelineState,
    pub residual_norm_pipeline: ComputePipelineState,
    pub residual_norm_q8_pipeline: ComputePipelineState,
    /// Dedicated row-per-simdgroup f32 gemv for the LM head. Used in
    /// autoregressive decode where `matmul_transb(query, lm_head)` shows
    /// up as the dominant per-token cost.
    pub f32_gemv_pipeline: ComputePipelineState,
    /// Same layout as [`Self::f32_gemv_pipeline`], but with a `half`
    /// weight matrix. Halves bandwidth for tied-embedding models whose
    /// lm_head would otherwise live as a 5.6 GB f32 clone on 31B.
    pub f16_gemv_pipeline: ComputePipelineState,
    flop_threshold: AtomicUsize,
}

impl MetalBackend {
    /// Create a Metal backend. Returns None if no Metal device is available.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let all_src = shaders::all_shaders();
        let library = device
            .new_library_with_source(&all_src, &opts)
            .map_err(|e| eprintln!("[metal] shader compile error: {e}"))
            .ok()?;

        let sgemm_fn = library.get_function("sgemm", None).ok()?;
        let transb_fn = library.get_function("sgemm_transb", None).ok()?;
        // Use v4 (uint32 wide loads) as production Q4 matvec — 2× faster than v1
        let q4_matvec_fn = library.get_function("q4_matvec_v4", None).ok()?;
        let q4_vecmat_fn = library.get_function("q4_vecmat", None).ok()?;

        let f32_ops = F32Ops {
            sgemm_pipeline: device.new_compute_pipeline_state_with_function(&sgemm_fn).ok()?,
            transb_pipeline: device.new_compute_pipeline_state_with_function(&transb_fn).ok()?,
        };

        let q4_f32_matvec_fn = library.get_function("q4_f32_matvec", None).ok()?;
        let geglu_fn = library.get_function("geglu_silu", None).ok()?;
        let q8_quant_fn = library.get_function("quantize_q8", None).ok()?;
        let causal_attn_fn = library.get_function("causal_attention", None).ok()?;
        let causal_attn_pipeline = device.new_compute_pipeline_state_with_function(&causal_attn_fn).ok()?;

        let q4 = Q4Pipelines {
            matvec: device.new_compute_pipeline_state_with_function(&q4_matvec_fn).ok()?,
            vecmat: device.new_compute_pipeline_state_with_function(&q4_vecmat_fn).ok()?,
            f32_matvec: device.new_compute_pipeline_state_with_function(&q4_f32_matvec_fn).ok()?,
        };

        let bufs = BufferCache::new(&device);

        let geglu_pipeline = device.new_compute_pipeline_state_with_function(&geglu_fn).ok()?;
        let geglu_gelu_tanh_fn = library.get_function("geglu_gelu_tanh", None).ok()?;
        let geglu_gelu_tanh_pipeline = device.new_compute_pipeline_state_with_function(&geglu_gelu_tanh_fn).ok()?;
        let q8_quant_pipeline = device.new_compute_pipeline_state_with_function(&q8_quant_fn).ok()?;

        // Q8 matvec for attention projections
        let q8_matvec_fn = library.get_function("q8_matvec", None).ok()?;
        let q8_matvec_pipeline = device.new_compute_pipeline_state_with_function(&q8_matvec_fn).ok()?;

        // Norm and residual ops
        let rms_norm_fn = library.get_function("rms_norm", None).ok()?;
        let residual_add_fn = library.get_function("residual_add", None).ok()?;
        let rms_norm_pipeline = device.new_compute_pipeline_state_with_function(&rms_norm_fn).ok()?;
        let residual_add_pipeline = device.new_compute_pipeline_state_with_function(&residual_add_fn).ok()?;

        // Q4_K and Q6_K matvec (Ollama-compatible quantization)
        let q4k_fn = library.get_function("q4k_matvec", None).ok()?;
        let q4k_ffn_gate_up_fn = library.get_function("q4k_ffn_gate_up", None).ok()?;
        let q6k_fn = library.get_function("q6k_matvec", None).ok()?;
        let q4k_matvec_pipeline = device.new_compute_pipeline_state_with_function(&q4k_fn).ok()?;
        let q4k_ffn_gate_up_pipeline = device.new_compute_pipeline_state_with_function(&q4k_ffn_gate_up_fn).ok()?;
        let q4kf_ffn_gate_up_fn = library.get_function("q4kf_ffn_gate_up", None).ok()?;
        let q4kf_ffn_gate_up_pipeline = device.new_compute_pipeline_state_with_function(&q4kf_ffn_gate_up_fn).ok()?;
        let q4k_geglu_silu_down_fn = library.get_function("q4k_geglu_silu_down", None).ok()?;
        let q4k_geglu_silu_down_pipeline = device.new_compute_pipeline_state_with_function(&q4k_geglu_silu_down_fn).ok()?;
        let q4k_geglu_gelu_tanh_down_fn = library.get_function("q4k_geglu_gelu_tanh_down", None).ok()?;
        let q4k_geglu_gelu_tanh_down_pipeline = device.new_compute_pipeline_state_with_function(&q4k_geglu_gelu_tanh_down_fn).ok()?;
        let q6k_matvec_pipeline = device.new_compute_pipeline_state_with_function(&q6k_fn).ok()?;

        // Fused Q8 QKV projection (all 3 in one dispatch)
        let q8_qkv_fn = library.get_function("q8_qkv_proj", None).ok()?;
        let q8_qkv_proj_pipeline = device.new_compute_pipeline_state_with_function(&q8_qkv_fn).ok()?;

        // Fused ops (norm+quantize, residual+norm, residual+norm+quantize)
        let rms_norm_q8_fn = library.get_function("rms_norm_q8", None).ok()?;
        let residual_norm_fn = library.get_function("residual_norm", None).ok()?;
        let residual_norm_q8_fn = library.get_function("residual_norm_q8", None).ok()?;
        let rms_norm_q8_pipeline = device.new_compute_pipeline_state_with_function(&rms_norm_q8_fn).ok()?;
        let residual_norm_pipeline = device.new_compute_pipeline_state_with_function(&residual_norm_fn).ok()?;
        let residual_norm_q8_pipeline = device.new_compute_pipeline_state_with_function(&residual_norm_q8_fn).ok()?;

        // Dedicated f32 gemv for the LM head.
        let f32_gemv_fn = library.get_function("f32_gemv", None).ok()?;
        let f32_gemv_pipeline = device.new_compute_pipeline_state_with_function(&f32_gemv_fn).ok()?;
        // f16 counterpart — half the memory, same shader topology.
        let f16_gemv_fn = library.get_function("f16_gemv", None).ok()?;
        let f16_gemv_pipeline = device.new_compute_pipeline_state_with_function(&f16_gemv_fn).ok()?;

        // RoPE (standalone, for prefill KV cache population)
        let rope_fn = library.get_function("rope_apply", None).ok()?;
        let rope_pipeline = device.new_compute_pipeline_state_with_function(&rope_fn).ok()?;

        // RoPE at position (for KV-cached decode)
        let rope_at_pos_fn = library.get_function("rope_at_pos", None).ok()?;
        let rope_at_pos_pipeline = device.new_compute_pipeline_state_with_function(&rope_at_pos_fn).ok()?;
        let rope_at_pos_batched_fn = library.get_function("rope_at_pos_batched", None).ok()?;
        let rope_at_pos_batched_pipeline = device.new_compute_pipeline_state_with_function(&rope_at_pos_batched_fn).ok()?;

        // Fused Q4_K QKV projection (one dispatch for Q+K+V)
        let q4k_qkv_fn = library.get_function("q4k_qkv_proj", None).ok()?;
        let q4k_qkv_proj_pipeline = device.new_compute_pipeline_state_with_function(&q4k_qkv_fn).ok()?;
        let q4k_q6k_qkv_fn = library.get_function("q4k_q6k_qkv_proj", None).ok()?;
        let q4k_q6k_qkv_proj_pipeline = device.new_compute_pipeline_state_with_function(&q4k_q6k_qkv_fn).ok()?;
        let q4k_proj_fn = library.get_function("q4k_proj", None).ok()?;
        let q4k_proj_pipeline = device.new_compute_pipeline_state_with_function(&q4k_proj_fn).ok()?;

        // Q4_KF: pre-baked scales (faster inference)
        let q4kf_qkv_fn = library.get_function("q4kf_qkv_proj", None).ok()?;
        let q4kf_qkv_proj_pipeline = device.new_compute_pipeline_state_with_function(&q4kf_qkv_fn).ok()?;
        let q4kf_proj_fn = library.get_function("q4kf_proj", None).ok()?;
        let q4kf_proj_pipeline = device.new_compute_pipeline_state_with_function(&q4kf_proj_fn).ok()?;

        // Fused attention (RoPE + GQA + softcap)
        let fused_attn_fn = library.get_function("fused_attention", None).ok()?;
        let fused_attn_pipeline = device.new_compute_pipeline_state_with_function(&fused_attn_fn).ok()?;

        // Standalone activations (non-gated FFN)
        let silu_fn = library.get_function("silu", None).ok()?;
        let gelu_tanh_fn = library.get_function("gelu_tanh", None).ok()?;
        let silu_pipeline = device.new_compute_pipeline_state_with_function(&silu_fn).ok()?;
        let gelu_tanh_pipeline = device.new_compute_pipeline_state_with_function(&gelu_tanh_fn).ok()?;

        // LayerNorm (StarCoder2, GPT-2)
        let layer_norm_fn = library.get_function("layer_norm", None).ok()?;
        let layer_norm_no_bias_fn = library.get_function("layer_norm_no_bias", None).ok()?;
        let layer_norm_pipeline = device.new_compute_pipeline_state_with_function(&layer_norm_fn).ok()?;
        let layer_norm_no_bias_pipeline = device.new_compute_pipeline_state_with_function(&layer_norm_no_bias_fn).ok()?;

        // V-norm (parameter-free RMSNorm, Gemma 4)
        let v_norm_fn = library.get_function("v_norm", None).ok()?;
        let v_norm_pipeline = device.new_compute_pipeline_state_with_function(&v_norm_fn).ok()?;
        let v_norm_batched_fn = library.get_function("v_norm_batched", None).ok()?;
        let v_norm_batched_pipeline = device.new_compute_pipeline_state_with_function(&v_norm_batched_fn).ok()?;

        // QK-norm (learned-weight per-head RMSNorm, Gemma 3/4)
        let qk_norm_fn = library.get_function("qk_norm", None).ok()?;
        let qk_norm_pipeline = device.new_compute_pipeline_state_with_function(&qk_norm_fn).ok()?;

        // Scale vector (per-layer scalar multiplier, Gemma 4)
        let scale_vector_fn = library.get_function("scale_vector", None).ok()?;
        let scale_vector_pipeline = device.new_compute_pipeline_state_with_function(&scale_vector_fn).ok()?;

        // KV cache attention
        let kv_attend_fn = library.get_function("kv_attention", None).ok()?;
        let kv_append_fn = library.get_function("kv_cache_append", None).ok()?;
        let kv_attend_pipeline = device.new_compute_pipeline_state_with_function(&kv_attend_fn).ok()?;
        let kv_append_pipeline = device.new_compute_pipeline_state_with_function(&kv_append_fn).ok()?;

        Some(Self {
            queue, bufs, f32_ops, q4, causal_attn_pipeline, fused_attn_pipeline,
            geglu_pipeline, geglu_gelu_tanh_pipeline, q8_quant_pipeline,
            kv_attend_pipeline, kv_append_pipeline,
            q8_matvec_pipeline,
            rms_norm_pipeline, residual_add_pipeline,
            q8_qkv_proj_pipeline,
            q4k_matvec_pipeline, q4k_ffn_gate_up_pipeline,
            q4kf_ffn_gate_up_pipeline,
            q4k_geglu_silu_down_pipeline, q4k_geglu_gelu_tanh_down_pipeline,
            q6k_matvec_pipeline,
            rope_pipeline, rope_at_pos_pipeline, rope_at_pos_batched_pipeline,
            q4k_qkv_proj_pipeline, q4k_q6k_qkv_proj_pipeline, q4k_proj_pipeline,
            q4kf_qkv_proj_pipeline, q4kf_proj_pipeline,
            silu_pipeline, gelu_tanh_pipeline,
            layer_norm_pipeline, layer_norm_no_bias_pipeline,
            v_norm_pipeline, v_norm_batched_pipeline,
            qk_norm_pipeline,
            scale_vector_pipeline,
            kv_cache: std::sync::Mutex::new(None),
            rms_norm_q8_pipeline, residual_norm_pipeline, residual_norm_q8_pipeline,
            f32_gemv_pipeline,
            f16_gemv_pipeline,
            flop_threshold: AtomicUsize::new(calibrate::DEFAULT_FLOP_THRESHOLD),
        })
    }

    /// Auto-calibrate CPU vs GPU threshold.
    pub fn calibrate(&self) {
        let threshold = calibrate::calibrate(&self.f32_ops, &self.queue, &self.bufs);
        self.flop_threshold.store(threshold, Ordering::Relaxed);
    }

    pub fn flop_threshold(&self) -> usize { self.flop_threshold.load(Ordering::Relaxed) }
    pub fn set_flop_threshold(&self, t: usize) { self.flop_threshold.store(t.max(calibrate::MIN_FLOP_FLOOR), Ordering::Relaxed); }
    pub fn cache_size(&self) -> usize { self.bufs.len() }
    pub fn bufs(&self) -> &BufferCache { &self.bufs }
    pub fn queue(&self) -> &CommandQueue { &self.queue }

    /// Access the KV cache for hybrid decode (GPU attention + CPU FFN).
    /// Creates the cache on first access.
    pub fn kv_cache_mut(&self, num_layers: usize, num_kv_heads: usize, head_dim: usize) -> std::sync::MutexGuard<'_, Option<ops::kv_cache::KVCache>> {
        let mut guard = self.kv_cache.lock().unwrap();
        if guard.is_none() {
            *guard = Some(self.create_kv_cache(num_layers, 4096, num_kv_heads, head_dim));
        }
        guard
    }
}
