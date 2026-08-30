//! Kimi Delta Attention's non-projection stages, on device.
//!
//! Rung 5c of the Kimi Metal ladder, and the reason it exists is not
//! compute. Rung 5b put KDA's four wide projections on Metal and measured
//! the kernel 2.8x faster than the CPU path — 0.25 ms of GPU-busy against
//! 0.70 ms — and then watched the two mandatory CPU↔GPU crossings cost
//! 0.40 ms and hand back 89% of the win. The crossings are mandatory only
//! because the stages BETWEEN the projections live on the host: the
//! convolution, the q/k norms, the gates, the recurrence and the gated
//! norm. Move those and the layer needs one crossing, not two.
//!
//! So these kernels are deliberately boring. Each is a literal
//! transcription of `exec::kda::step`, in the same order, with the same
//! accumulation sequence wherever a sequence exists. The goal is
//! **boundary deletion**; a recurrence microbenchmark is not the point
//! and would be a bad reason to reorder anything.
//!
//! ## The state stays on device
//!
//! `kda_recurrence` reads and writes the `[H][D][D]` recurrent state and
//! the three convolution windows in place. Reading them back to preserve
//! the host's representation would reintroduce exactly the crossing this
//! rung removes.
//!
//! ## Where bit-exactness is and is not claimed
//!
//! The recurrence is bit-exact by construction in the dimension that
//! matters: **one thread owns one value column `vv` for a whole head**,
//! so its accumulation over `kk` is the same ascending sequence the CPU
//! performs, with no cross-thread reduction to reassociate it. The two
//! head-wide reductions — the q/k L2 norm and the gated RMS norm — sum
//! `D` terms in a threadgroup tree where the CPU sums them in order, so
//! those differ by reassociation alone. Measured, not assumed: the gate
//! reports every plane's delta.

/// One thread per value column, one threadgroup per head. `head_dim`
/// must not exceed this; Kimi's is 128.
pub const RECURRENCE_THREADS_PER_TG: u64 = 128;
/// Head-wide reductions (`l2_normalise_heads`, the gated RMS norm) use
/// one threadgroup per head with this many threads.
pub const HEAD_REDUCE_THREADS_PER_TG: u64 = 128;
/// Flat element-wise kernels.
pub const ELEMENTWISE_THREADS_PER_TG: u64 = 256;

pub const SHADER: &str = r#"
// `ln(1+x)`, accurate for small `x`. MSL has no `log1p`, and the naive
// `log(1+x)` loses every bit of a tiny `x` to the rounding of `1+x` —
// which is exactly the regime the softplus tail sits in. Kahan's
// identity recovers it: `log(u) * (x / (u-1))` where `u = 1+x`, and the
// `u == 1` branch returns `x` itself, whose own error is second order.
static inline float kda_log1p(float x) {
    const float u = 1.0f + x;
    if (u == 1.0f) return x;
    return log(u) * (x / (u - 1.0f));
}

// `ln(1+e^v)` in the stable form `exec::kda::softplus` uses: above 20 the
// naive expression overflows to inf and then to a NaN gate.
static inline float kda_softplus(float v) {
    return v > 20.0f ? v : kda_log1p(exp(v));
}

static inline float kda_silu(float v) {
    return v / (1.0f + exp(-v));
}

// Depthwise causal convolution over one stream, then SiLU, then slide the
// window. One thread owns one channel, so the window update is local and
// needs no barrier.
//
// Causal: the window is the `kernel-1` PREVIOUS inputs plus the current
// one. `window` carries those across calls, which is what makes a
// continuation produce what one pass over the concatenation would.
kernel void kda_short_conv_silu(
    device const float* x      [[buffer(0)]],  // [width]
    device const float* weight [[buffer(1)]],  // [width, kernel]
    device float*       window [[buffer(2)]],  // [width, kernel-1], in/out
    device float*       out    [[buffer(3)]],  // [width]
    constant uint&      width  [[buffer(4)]],
    constant uint&      kernel_size [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= width) return;
    const uint tail = kernel_size - 1u;
    device const float* w = weight + c * kernel_size;
    device float* hist = window + c * tail;

    // Oldest first: history then the current sample — the same order
    // `exec::kda::short_conv` accumulates in.
    float acc = 0.0f;
    for (uint i = 0; i < tail; ++i) {
        acc += w[i] * hist[i];
    }
    acc += w[tail] * x[c];
    out[c] = kda_silu(acc);

    // Slide one position, dropping the oldest.
    for (uint i = 0; i + 1 < tail; ++i) {
        hist[i] = hist[i + 1];
    }
    if (tail > 0) hist[tail - 1] = x[c];
}

// Per-head L2 normalisation, matching `F.normalize`'s clamp: a zero head
// stays zero rather than becoming NaN.
kernel void kda_l2_normalise_heads(
    device const float* v   [[buffer(0)]],  // [heads, dim]
    device float*       out [[buffer(1)]],  // [heads, dim]
    constant uint&      dim [[buffer(2)]],
    uint  head [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]],
    uint  tcount [[threads_per_threadgroup]])
{
    threadgroup float partial[128];
    device const float* head_v = v + head * dim;
    device float* head_o = out + head * dim;

    float acc = 0.0f;
    for (uint d = tid; d < dim; d += tcount) acc += head_v[d] * head_v[d];
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tcount / 2u; s > 0u; s >>= 1u) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / max(sqrt(partial[0]), 1e-12f);
    for (uint d = tid; d < dim; d += tcount) head_o[d] = head_v[d] * inv;
}

// `decay[i] = -exp(a_log[h]) * softplus(f_low[i] + dt_bias[i])`.
kernel void kda_decay_gate(
    device const float* f_low   [[buffer(0)]],  // [width]
    device const float* dt_bias [[buffer(1)]],  // [width]
    device const float* a_log   [[buffer(2)]],  // [heads]
    device float*       decay   [[buffer(3)]],  // [width]
    constant uint&      width   [[buffer(4)]],
    constant uint&      dim     [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= width) return;
    const float a = exp(a_log[i / dim]);
    decay[i] = -a * kda_softplus(f_low[i] + dt_bias[i]);
}

// `beta[h] = sigmoid(b_proj_out[h])`.
kernel void kda_beta_sigmoid(
    device const float* pre   [[buffer(0)]],  // [heads]
    device float*       beta  [[buffer(1)]],  // [heads]
    constant uint&      heads [[buffer(2)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= heads) return;
    beta[h] = 1.0f / (1.0f + exp(-pre[h]));
}

// The delta rule, one head per threadgroup, one VALUE COLUMN per thread.
//
// That mapping is the whole design. Column `vv` of a head's state is
// touched only by thread `vv`, so every accumulation over `kk` runs in
// the same ascending order `exec::kda::step` performs and no
// cross-thread reduction exists to reassociate it. It also coalesces:
// for a fixed `kk` the threads read `s[kk*dim + 0 .. dim-1]`, which is
// contiguous.
//
// Both fusions the CPU path documents are preserved. Decay-and-predict
// share one `kk` pass because decaying row `kk` is local to that row;
// write-and-readout share the next because `s_new[kk][vv]` is written
// earlier in the same iteration that reads it.
kernel void kda_recurrence(
    device float*       state [[buffer(0)]],  // [heads, dim, dim], in/out
    device const float* q     [[buffer(1)]],  // [heads, dim], L2-normalised
    device const float* k     [[buffer(2)]],  // [heads, dim], L2-normalised
    device const float* v     [[buffer(3)]],  // [heads, dim]
    device const float* decay [[buffer(4)]],  // [heads, dim]
    device const float* beta  [[buffer(5)]],  // [heads]
    device float*       out   [[buffer(6)]],  // [heads, dim]
    constant uint&      dim   [[buffer(7)]],
    constant float&     scale [[buffer(8)]],  // dim^-0.5
    uint head [[threadgroup_position_in_grid]],
    uint vv   [[thread_position_in_threadgroup]])
{
    if (vv >= dim) return;
    device float* s = state + (ulong)head * dim * dim;
    device const float* qh = q + head * dim;
    device const float* kh = k + head * dim;
    device const float* dh = decay + head * dim;
    const float b = beta[head];

    // Pass 1 — decay each row and fold it into this column's prediction.
    float pred = 0.0f;
    for (uint kk = 0; kk < dim; ++kk) {
        const float cell = s[kk * dim + vv] * exp(dh[kk]);
        s[kk * dim + vv] = cell;
        pred += kh[kk] * cell;
    }
    // The prediction error the delta rule writes against.
    const float err = v[head * dim + vv] - pred;

    // Pass 2 — write the rank-1 update and read the output out of the
    // cell just written.
    float acc = 0.0f;
    for (uint kk = 0; kk < dim; ++kk) {
        const float cell = s[kk * dim + vv] + b * kh[kk] * err;
        s[kk * dim + vv] = cell;
        acc += qh[kk] * scale * cell;
    }
    out[head * dim + vv] = acc;
}

// Gated RMSNorm over one head's width: normalise, scale by the weight,
// gate by `sigmoid(gate)`.
kernel void kda_gated_rms_norm(
    device const float* x      [[buffer(0)]],  // [heads, dim]
    device const float* weight [[buffer(1)]],  // [dim]
    device const float* gate   [[buffer(2)]],  // [heads, dim]
    device float*       out    [[buffer(3)]],  // [heads, dim]
    constant uint&      dim    [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    uint  head   [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  tcount [[threads_per_threadgroup]])
{
    threadgroup float partial[128];
    device const float* head_x = x + head * dim;

    float acc = 0.0f;
    for (uint d = tid; d < dim; d += tcount) acc += head_x[d] * head_x[d];
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tcount / 2u; s > 0u; s >>= 1u) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(partial[0] / float(dim) + eps);
    for (uint d = tid; d < dim; d += tcount) {
        const uint i = head * dim + d;
        out[i] = head_x[d] * inv * weight[d] / (1.0f + exp(-gate[i]));
    }
}
"#;

macro_rules! kda_kernel {
    ($ident:ident, $name:literal) => {
        pub struct $ident;
        impl crate::kernels::ShaderKernel for $ident {
            const KERNEL_NAME: &'static str = $name;
        }
    };
}

kda_kernel!(ShortConvSiluKernel, "kda_short_conv_silu");
kda_kernel!(L2NormaliseHeadsKernel, "kda_l2_normalise_heads");
kda_kernel!(DecayGateKernel, "kda_decay_gate");
kda_kernel!(BetaSigmoidKernel, "kda_beta_sigmoid");
kda_kernel!(RecurrenceKernel, "kda_recurrence");
kda_kernel!(GatedRmsNormKernel, "kda_gated_rms_norm");

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel this module declares is actually emitted, and the
    /// threadgroup constants match the `threadgroup float partial[128]`
    /// the reductions allocate — a reduction dispatched with more
    /// threads than that array holds would write out of bounds.
    #[test]
    fn every_declared_kernel_is_emitted_and_sized() {
        use crate::kernels::ShaderKernel;
        for name in [
            ShortConvSiluKernel::KERNEL_NAME,
            L2NormaliseHeadsKernel::KERNEL_NAME,
            DecayGateKernel::KERNEL_NAME,
            BetaSigmoidKernel::KERNEL_NAME,
            RecurrenceKernel::KERNEL_NAME,
            GatedRmsNormKernel::KERNEL_NAME,
        ] {
            assert!(
                SHADER.contains(&format!("kernel void {name}(")),
                "{name} declared but not emitted"
            );
        }
        assert_eq!(HEAD_REDUCE_THREADS_PER_TG, 128);
        assert_eq!(RECURRENCE_THREADS_PER_TG, 128);
        assert_eq!(SHADER.matches("threadgroup float partial[128];").count(), 2);
    }

    /// The recurrence must keep both fusions the CPU path documents: one
    /// `kk` pass that decays and predicts, one that writes and reads out.
    /// A third pass would mean the transcription drifted.
    #[test]
    fn the_recurrence_keeps_its_two_passes() {
        let body = SHADER
            .split("kernel void kda_recurrence(")
            .nth(1)
            .expect("recurrence kernel");
        let body = body.split("kernel void ").next().unwrap();
        assert_eq!(
            body.matches("for (uint kk = 0; kk < dim; ++kk)").count(),
            2,
            "the recurrence should walk kk exactly twice"
        );
    }
}
