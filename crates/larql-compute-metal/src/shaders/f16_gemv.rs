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

// VERIFY-N: RL rows per simdgroup against NR activation rows, each f16
// weight read once for the block. Lane l walks half4 columns l, l+32, ...
// X is [NR, K], out [NR, N], both row-major. Requires K % 4 == 0. The
// reduction is float4-grouped, so parity with f16_gemv is to fp32
// rounding, not bit.
#define F16_MULTIRHS_KERNEL(NAME, RL, NR)                                        \
kernel void NAME(                                                                \
    device const half*  W   [[buffer(0)]],                                       \
    device const float* X   [[buffer(1)]],                                       \
    device float*       out [[buffer(2)]],                                       \
    constant uint&      N   [[buffer(3)]],                                       \
    constant uint&      K   [[buffer(4)]],                                       \
    uint tg_id   [[threadgroup_position_in_grid]],                               \
    uint lane    [[thread_index_in_simdgroup]],                                  \
    uint sg_id   [[simdgroup_index_in_threadgroup]])                             \
{                                                                                \
    const uint row0 = (tg_id * F16GEMV_SG_PER_TG + sg_id) * (RL);                \
    if (row0 >= N) return;                                                       \
    const uint K4 = K / 4u;                                                      \
    device const half4*  W4 = (device const half4*)W;                            \
    device const float4* X4 = (device const float4*)X;                           \
    float acc[RL][NR];                                                           \
    for (uint r = 0u; r < (RL); ++r)                                             \
        for (uint n = 0u; n < (NR); ++n) { acc[r][n] = 0.0f; }                   \
    for (uint k = lane; k < K4; k += 32u) {                                      \
        float4 w[RL];                                                            \
        for (uint r = 0u; r < (RL); ++r) {                                       \
            const uint row = row0 + r;                                           \
            w[r] = row < N ? float4(W4[(ulong)row * K4 + k]) : float4(0.0f);     \
        }                                                                        \
        for (uint n = 0u; n < (NR); ++n) {                                       \
            const float4 x = X4[(ulong)n * K4 + k];                              \
            for (uint r = 0u; r < (RL); ++r) { acc[r][n] += dot(w[r], x); }      \
        }                                                                        \
    }                                                                            \
    for (uint n = 0u; n < (NR); ++n) {                                           \
        for (uint r = 0u; r < (RL); ++r) {                                       \
            const float t = simd_sum(acc[r][n]);                                 \
            if (lane == 0u && row0 + r < N) { out[(ulong)n * N + row0 + r] = t; } \
        }                                                                        \
    }                                                                            \
}

F16_MULTIRHS_KERNEL(f16_matmul_x2_r2, 2u, 2u)
F16_MULTIRHS_KERNEL(f16_matmul_x2_r4, 2u, 4u)
F16_MULTIRHS_KERNEL(f16_matmul_x2_r8, 2u, 8u)

// VERIFY-N tiled: the 8x8 simdgroup MAC with the activation tile shared.
//
// The multi-RHS kernels above are bound by X-operand traffic: every
// simdgroup re-reads all R activation rows for its 2 weight rows, so X
// bytes per output element are ~2K regardless of R. Here a threadgroup
// covers F16MM_TG_ROWS weight rows; per K chunk it stages the [8, KC]
// activation tile in threadgroup memory ONCE (rows >= R zero-filled), and
// each simdgroup multiplies that tile against F16MM_SG_BLOCKS 8-row weight
// tiles loaded straight from device memory:
//
//   C[pos, row] += X[pos, k..k+8] . W[row, k..k+8]^T      (8x8x8 per MAC)
//
// Results pass through threadgroup memory so only the R valid position
// rows are written — out may be a KV-cache slot range, so writing padded
// rows would clobber positions past the block.
//
// Contract: N % 8 == 0, K % F16MM_KC == 0, 1 <= R <= 8. X is [R, K], out
// [R, N], both row-major. Parity to f16_gemv is to fp32 rounding.
constant uint F16MM_SG = 4;
constant uint F16MM_SG_BLOCKS = 4;
constant uint F16MM_SG_ROWS = F16MM_SG_BLOCKS * 8;
constant uint F16MM_TG_ROWS = F16MM_SG * F16MM_SG_ROWS;
constant uint F16MM_KC = 32;

kernel void f16_matmul_sg(
    device const half*  W   [[buffer(0)]],
    device const float* X   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      N   [[buffer(3)]],
    constant uint&      K   [[buffer(4)]],
    constant uint&      R   [[buffer(5)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float xs[8 * F16MM_KC];
    threadgroup float cs[F16MM_SG * 8 * F16MM_SG_ROWS];
    const uint row0 = tg_id * F16MM_TG_ROWS + sg_id * F16MM_SG_ROWS;
    simdgroup_float8x8 acc[F16MM_SG_BLOCKS];
    for (uint b = 0u; b < F16MM_SG_BLOCKS; ++b) {
        acc[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint k0 = 0u; k0 < K; k0 += F16MM_KC) {
        // Stage X[0..8, k0..k0+KC]: 256 floats over 128 threads.
        for (uint i = tid; i < 8u * F16MM_KC; i += F16MM_SG * 32u) {
            const uint pos = i / F16MM_KC;
            const uint kk = i % F16MM_KC;
            xs[i] = pos < R ? X[(ulong)pos * K + k0 + kk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < F16MM_KC; kk += 8u) {
            simdgroup_float8x8 a;
            simdgroup_load(a, xs + kk, F16MM_KC);
            for (uint b = 0u; b < F16MM_SG_BLOCKS; ++b) {
                const uint r = row0 + b * 8u;
                if (r < N) {
                    simdgroup_half8x8 wt;
                    simdgroup_load(wt, W + (ulong)r * K + k0 + kk, K, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc[b], a, wt, acc[b]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* c = cs + sg_id * 8u * F16MM_SG_ROWS;
    for (uint b = 0u; b < F16MM_SG_BLOCKS; ++b) {
        simdgroup_store(acc[b], c + b * 8u, F16MM_SG_ROWS);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 8u * F16MM_SG_ROWS; i += 32u) {
        const uint pos = i / F16MM_SG_ROWS;
        const uint col = i % F16MM_SG_ROWS;
        const uint row = row0 + col;
        if (pos < R && row < N) {
            out[(ulong)pos * N + row] = c[i];
        }
    }
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

/// VERIFY-N f16 multi-RHS arms as (kernel name, activation rows), in
/// `MetalBackend::f16_matmul_pipelines` order. 8 simdgroups x 2 rows.
pub const MATMUL_ARMS: [(&str, usize); 3] = [
    ("f16_matmul_x2_r2", 2),
    ("f16_matmul_x2_r4", 4),
    ("f16_matmul_x2_r8", 8),
];
macro_rules! f16_matmul_kernel {
    ($ty:ident, $name:literal) => {
        /// VERIFY-N f16 multi-RHS arm; see `MATMUL_ARMS`.
        pub struct $ty;
        impl crate::kernels::TiledKernel for $ty {
            const KERNEL_NAME: &'static str = $name;
            const ROWS_PER_TG: u64 = 16;
            const THREADS_PER_TG: u64 = THREADS_PER_TG;
        }
    };
}
f16_matmul_kernel!(KernelMatmulR2, "f16_matmul_x2_r2");
f16_matmul_kernel!(KernelMatmulR4, "f16_matmul_x2_r4");
f16_matmul_kernel!(KernelMatmulR8, "f16_matmul_x2_r8");

/// VERIFY-N tiled f16 matmul: 4 simdgroups x 32 rows per threadgroup;
/// takes the position count `R` (1..=8) at buffer 5.
pub struct KernelMatmulSg;
impl crate::kernels::TiledKernel for KernelMatmulSg {
    const KERNEL_NAME: &'static str = "f16_matmul_sg";
    const ROWS_PER_TG: u64 = 128;
    const THREADS_PER_TG: u64 = 128;
}
/// Positions one tiled dispatch covers at most.
pub const MATMUL_SG_MAX_ROWS: usize = 8;
/// K granularity the tiled kernel stages the activation tile at.
pub const MATMUL_SG_KC: usize = 32;
