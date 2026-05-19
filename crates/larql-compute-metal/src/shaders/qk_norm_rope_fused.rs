//! Fused **QK-norm + RoPE** for Gemma 3/4 attention.
//!
//! Replaces the consecutive `qk_norm_qk` + `rope_at_pos_batched_qk`
//! dispatches in `metal/decode/mod.rs` with a single kernel: each
//! threadgroup handles one (Q or K) head, does the RMS-norm + per-d
//! scale, then applies RoPE rotation in-place — with a single
//! `threadgroup_barrier` between the two phases (no inter-dispatch
//! round-trip).
//!
//! **Why this kernel exists**: in-pipeline GPU timing
//! (`LARQL_GPU_TIMING=1`) on Gemma 3 4B (2026-05-01) shows
//! `decode_token` runs ~340 dispatches/tok at ~30 µs avg = ~10.5 ms
//! GPU compute, vs llama.cpp/ollama's estimated ~200 dispatches/tok
//! → ~8 ms. **Dispatch count, not per-kernel speed, is the bottleneck**
//! after earlier kernel-utilization optimisations came out null
//! (`F16_ACC`, `GATE_UP_COOP`, and the removed NR2 candidate). This fusion is
//! the smallest concrete dispatch-reduction step: 1 dispatch saved
//! per layer × 34 layers = ~34 dispatches/tok × ~7 µs/dispatch ≈
//! 0.24 ms/tok end-to-end.
//!
//! **Math**: identical to the consecutive-dispatch chain. Per head:
//!   1. `rms² = (1/head_dim) Σ x[d]²` (parallel reduction).
//!   2. `x[d] = x[d] / √(rms² + eps) * (offset + weight[d])`
//!      (eqn matches `qk_norm_qk` — `offset = 1.0` on Gemma 2/3,
//!      `0.0` on Gemma 4).
//!   3. RoPE: for each (d, d + rotary_dim/2) pair,
//!      `(re', im') = (re·cos_θ − im·sin_θ, re·sin_θ + im·cos_θ)`,
//!      `θ = pos · rope_base^(-2d/rotary_dim)`. Identical to
//!      `rope_at_pos_batched_qk`.
//!
//! **Geometry**: `(num_q + num_kv)` threadgroups, one per head.
//! Threads-per-TG = ceil(head_dim, 32) (typically 256 on Gemma 3 4B).
//! Bounded by hardware threadgroup-mem usage (~1 KB tg_partial[]).
//!
//! Same `[[buffer]]` numbering convention as `qk_norm_qk` for buffers
//! 0..7, plus the RoPE-specific buffers 8..10
//! (rope_base, pos, rotary_dim) — caller binds them in one go.

pub const SHADER: &str = r#"
kernel void qk_norm_rope_fused(
    device float*       Q          [[buffer(0)]],   // [num_q * head_dim]   in-place
    device float*       K          [[buffer(1)]],   // [num_kv * head_dim]  in-place
    device const float* q_weight   [[buffer(2)]],   // [head_dim]
    device const float* k_weight   [[buffer(3)]],   // [head_dim]
    constant uint&      head_dim   [[buffer(4)]],
    constant uint&      num_q      [[buffer(5)]],
    constant float&     eps        [[buffer(6)]],
    constant float&     offset     [[buffer(7)]],
    constant float&     rope_base  [[buffer(8)]],
    constant uint&      pos        [[buffer(9)]],
    constant uint&      rotary_dim [[buffer(10)]],
    uint h_idx [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tg_w  [[threads_per_threadgroup]])
{
    bool is_q = (h_idx < num_q);
    uint local_head = is_q ? h_idx : (h_idx - num_q);
    device float*       buf    = is_q ? Q : K;
    device const float* weight = is_q ? q_weight : k_weight;
    uint base = local_head * head_dim;

    // ── Phase 1: compute sum-of-squares for this head ──
    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_w) {
        float v = buf[base + i];
        partial += v * v;
    }

    threadgroup float tg_partial[512];
    tg_partial[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_w / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) tg_partial[tid] += tg_partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(tg_partial[0] / float(head_dim) + eps);
    float inv_rms = 1.0f / rms;

    // ── Phase 2: write normalised values back to buf ──
    // After this loop completes, the buffer holds RMS-normed,
    // weight-scaled values — the same state the original
    // `qk_norm_qk` would have left them in.
    for (uint d = tid; d < head_dim; d += tg_w) {
        buf[base + d] = (buf[base + d] * inv_rms) * (offset + weight[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: in-place RoPE rotation ──
    // Each thread handles one (d, d + hdim) rotary pair. `rotary_dim`
    // may be < `head_dim` for partial-RoPE archs (e.g. some Gemma
    // configs). When `rotary_dim == 0` we treat it as full-head.
    uint rdim = (rotary_dim == 0u) ? head_dim : min(rotary_dim, head_dim);
    uint hdim = rdim / 2u;
    for (uint d = tid; d < hdim; d += tg_w) {
        float freq  = 1.0f / pow(rope_base, float(2u * d) / float(rdim));
        float angle = float(pos) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float re = buf[base + d];
        float im = buf[base + d + hdim];
        buf[base + d]        = re * cos_a - im * sin_a;
        buf[base + d + hdim] = re * sin_a + im * cos_a;
    }
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "qk_norm_rope_fused";
}
