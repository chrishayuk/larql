//! f16 gemv — f16 weights × f32 query → f32 output, for the LM head.
//!
//! Mirror of [`f32_gemv`](super::f32_gemv) but the weight matrix is `half`
//! on disk. Saves the 5.6 GB f32 clone on Gemma 4 31B (2.8 GB on disk as
//! f16) and halves the memory-bandwidth of the per-token logit gemv.
//!
//! Metal promotes the `half` load to `float` inline — there's no explicit
//! conversion cost beyond the reduced bandwidth. The accumulator stays
//! `float` to preserve argmax stability on the 262 K-wide logit vector.

pub const SHADER: &str = r#"
constant uint F16GEMV_SG_PER_TG = 8;
constant uint F16GEMV_ROWS_PER_TG = F16GEMV_SG_PER_TG;

kernel void f16_gemv(
    device const half*  W   [[buffer(0)]],   // [N, K] row-major, f16
    device const float* X   [[buffer(1)]],   // [K]
    device float*       out [[buffer(2)]],   // [N]
    constant uint&      N   [[buffer(3)]],
    constant uint&      K   [[buffer(4)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * F16GEMV_ROWS_PER_TG + sg_id;
    if (row >= N) return;

    device const half* w_row = W + row * K;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint k = lane;
    for (; k + 3 * 32 < K; k += 4 * 32) {
        a0 = fma(float(w_row[k         ]), X[k         ], a0);
        a1 = fma(float(w_row[k + 32    ]), X[k + 32    ], a1);
        a2 = fma(float(w_row[k + 64    ]), X[k + 64    ], a2);
        a3 = fma(float(w_row[k + 96    ]), X[k + 96    ], a3);
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; k < K; k += 32) acc = fma(float(w_row[k]), X[k], acc);

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "f16_gemv";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
