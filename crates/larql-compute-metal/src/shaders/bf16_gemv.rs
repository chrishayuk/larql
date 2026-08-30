//! bf16 gemv — bf16 weights × f32 query → f32 output.
//!
//! Direct sibling of [`f16_gemv`](super::f16_gemv): same row-per-simdgroup
//! tiling, same 4-way unrolled lane stride, same f32 accumulator. Only the
//! weight decode differs.
//!
//! **bf16 is not f16.** `f16_gemv` binds `device const half*` and lets
//! Metal promote; that hardware promotion implements IEEE-754 binary16
//! (5 exponent bits, 10 mantissa). bf16 is the top half of an f32 (8
//! exponent bits, 7 mantissa) — a different type at the same width, so
//! it needs its own binding (`ushort`, no arithmetic meaning) and its own
//! decode ([`decode_bf16_metal`](super::common)). Routing bf16 codes
//! through the half path would not round differently, it would compute
//! unrelated numbers.
//!
//! **Why it exists.** Kimi Linear's checkpoint stores every tensor as
//! bf16, and the CPU arc's [`FusedBf16`] kernel already proved the
//! representation is worth keeping compact all the way into registers
//! (widening to scratch first measured 27.3 GB/s against a fused
//! kernel's 122.0). This is that same architecture on the GPU: load the
//! code, widen in register, FMA against the f32 activation, discard —
//! half the bytes of an f32 gemv, exactly the same values, because the
//! widen is lossless.
//!
//! [`FusedBf16`]: https://docs.rs/larql-vindex — `exec::cpu::kernels::FusedBf16`

pub const SHADER: &str = r#"
constant uint BF16GEMV_SG_PER_TG = 8;
constant uint BF16GEMV_ROWS_PER_TG = BF16GEMV_SG_PER_TG;

kernel void bf16_gemv(
    device const ushort* W   [[buffer(0)]],   // [N, K] row-major, bf16 codes
    device const float*  X   [[buffer(1)]],   // [K]
    device float*        out [[buffer(2)]],   // [N]
    constant uint&       N   [[buffer(3)]],
    constant uint&       K   [[buffer(4)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * BF16GEMV_ROWS_PER_TG + sg_id;
    if (row >= N) return;

    device const ushort* w_row = W + row * K;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint k = lane;
    for (; k + 3 * 32 < K; k += 4 * 32) {
        a0 = fma(decode_bf16_metal(w_row[k         ]), X[k         ], a0);
        a1 = fma(decode_bf16_metal(w_row[k + 32    ]), X[k + 32    ], a1);
        a2 = fma(decode_bf16_metal(w_row[k + 64    ]), X[k + 64    ], a2);
        a3 = fma(decode_bf16_metal(w_row[k + 96    ]), X[k + 96    ], a3);
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; k < K; k += 32) acc = fma(decode_bf16_metal(w_row[k]), X[k], acc);

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "bf16_gemv";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
