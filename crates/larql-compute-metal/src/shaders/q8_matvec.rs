//! Q8_0 × Q8_0 matrix-vector multiply.
//!
//! scores[N] = Q8_weights[N, K] @ Q8_x[K]
//!
//! Q8_0 format: 34 bytes per block of 32 (f16 scale + 32 signed int8).
//! No nibble unpacking — each weight is one byte. Simpler and faster than Q4.
//! Used for V projection where Q4 corrupts the payload.

pub const SHADER: &str = r#"
constant uint Q8_ROWS_PER_TG = 8;

kernel void q8_matvec(
    device const uchar*  W8     [[buffer(0)]],   // Q8_0 weights [N, K] packed
    device const char*   Q8     [[buffer(1)]],   // Q8_0 input [K] int8
    device const float*  W8s    [[buffer(2)]],   // weight per-block scales [N * blocks]
    device const float*  Q8s    [[buffer(3)]],   // input per-block scales [blocks]
    device float*        out    [[buffer(4)]],
    constant uint&       N      [[buffer(5)]],
    constant uint&       K      [[buffer(6)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint blocks = K / 32;
    uint bytes_per_row = 32 * blocks;  // Q8: 32 bytes per block (no scale in data, separate)
    // Actually Q8_0 format: 34 bytes per block (2B f16 scale + 32B int8)
    // But we use a simpler layout: weights as raw int8 + separate scale array
    // This matches our quantize_q8_0 format.

    // Load Q8 input into threadgroup shared memory
    threadgroup char tg_q8[8192];
    threadgroup float tg_q8s[256];
    for (uint i = tid_in_tg; i < K; i += 256) tg_q8[i] = Q8[i];
    for (uint i = tid_in_tg; i < blocks; i += 256) tg_q8s[i] = Q8s[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_idx = tg_id * Q8_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    device const char* row = (device const char*)(W8 + row_idx * K);
    device const float* w_scales = W8s + row_idx * blocks;

    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += 32) {
        float combined_scale = w_scales[b] * tg_q8s[b];
        device const char* wb = row + b * 32;
        threadgroup const char* q8 = tg_q8 + b * 32;

        // int8 × int8 dot product — byte-by-byte, no address space casts
        int isum = 0;
        for (uint j = 0; j < 32; j++) {
            isum += int(wb[j]) * int(q8[j]);
        }

        acc += float(isum) * combined_scale;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q8_matvec";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
