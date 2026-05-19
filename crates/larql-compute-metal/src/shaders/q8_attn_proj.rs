//! Fused Q8 QKV projection: all 3 attention projections in one dispatch.
//!
//! Uses simdgroup reduction (like q4_matvec_v4) for maximum throughput.
//! 8 rows per threadgroup, 32 lanes per simdgroup, simd_sum for reduction.
//!
//! Grid: ((q_rows + k_rows + v_rows + 7) / 8, 1, 1).
//! Each threadgroup handles 8 rows across Q/K/V.

pub const SHADER: &str = r#"
constant uint QKV_ROWS_PER_TG = 8;

// Fused Q+K+V projection: all 3 in one kernel.
// Rows 0..q_rows → Q, q_rows..q_rows+k_rows → K, rest → V.
kernel void q8_qkv_proj(
    device const uchar*  Wq     [[buffer(0)]],
    device const uchar*  Wk     [[buffer(1)]],
    device const uchar*  Wv     [[buffer(2)]],
    device const char*   X8     [[buffer(3)]],   // Q8 input int8 [K]
    device const float*  Wqs    [[buffer(4)]],   // Q weight scales [q_rows * blocks]
    device const float*  Wks    [[buffer(5)]],
    device const float*  Wvs    [[buffer(6)]],
    device const float*  X8s    [[buffer(7)]],   // input scales [blocks]
    device float*        Q_out  [[buffer(8)]],
    device float*        K_out  [[buffer(9)]],
    device float*        V_out  [[buffer(10)]],
    constant uint&       q_rows [[buffer(11)]],
    constant uint&       k_rows [[buffer(12)]],
    constant uint&       v_rows [[buffer(13)]],
    constant uint&       K      [[buffer(14)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * QKV_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    uint blocks = K / 32;

    // Load Q8 input into threadgroup shared memory (once per TG)
    threadgroup int8_t tg_x8[8192];
    threadgroup float tg_xs[256];
    for (uint i = tid_in_tg; i < K; i += 256)
        tg_x8[i] = ((device const int8_t*)X8)[i];
    for (uint i = tid_in_tg; i < blocks; i += 256)
        tg_xs[i] = X8s[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Determine which projection and local row
    device const uchar* W;
    device const float* Ws;
    device float* out_buf;
    uint local_row;

    if (global_row < q_rows) {
        W = Wq; Ws = Wqs; out_buf = Q_out; local_row = global_row;
    } else if (global_row < q_rows + k_rows) {
        W = Wk; Ws = Wks; out_buf = K_out; local_row = global_row - q_rows;
    } else {
        W = Wv; Ws = Wvs; out_buf = V_out; local_row = global_row - q_rows - k_rows;
    }

    // Q8 dot product: W[local_row, :] · X8[:]
    // Each lane processes a stripe of blocks
    float acc = 0.0f;
    device const char* row_data = (device const char*)(W + local_row * K);
    device const float* row_scales = Ws + local_row * blocks;

    for (uint b = lane; b < blocks; b += 32) {
        float combined_scale = row_scales[b] * tg_xs[b];
        device const char* wb = row_data + b * 32;
        threadgroup const int8_t* xb = tg_x8 + b * 32;

        // int8 × int8 dot product (compiler auto-vectorizes)
        int isum = 0;
        for (uint i = 0; i < 32; i++) {
            isum += int(wb[i]) * int(xb[i]);
        }
        acc += float(isum) * combined_scale;
    }

    // Simdgroup reduction — no threadgroup barrier needed
    acc = simd_sum(acc);

    if (lane == 0) {
        out_buf[local_row] = acc;
    }
}

// Single Q8 projection with simdgroup reduction (for O projection).
kernel void q8_proj_rope(
    device const uchar*  W8     [[buffer(0)]],
    device const char*   X8     [[buffer(1)]],
    device const float*  W8s    [[buffer(2)]],
    device const float*  X8s    [[buffer(3)]],
    device float*        out    [[buffer(4)]],
    constant uint&       num_rows [[buffer(5)]],
    constant uint&       K      [[buffer(6)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * QKV_ROWS_PER_TG + sg_id;
    if (row >= num_rows) return;

    uint blocks = K / 32;

    threadgroup int8_t tg_x8[8192];
    threadgroup float tg_xs[256];
    for (uint i = tid_in_tg; i < K; i += 256)
        tg_x8[i] = ((device const int8_t*)X8)[i];
    for (uint i = tid_in_tg; i < blocks; i += 256)
        tg_xs[i] = X8s[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const char* row_data = (device const char*)(W8 + row * K);
    device const float* row_scales = W8s + row * blocks;

    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += 32) {
        float combined_scale = row_scales[b] * tg_xs[b];
        device const char* wb = row_data + b * 32;
        threadgroup const int8_t* xb = tg_x8 + b * 32;

        int isum = 0;
        for (uint i = 0; i < 32; i++) {
            isum += int(wb[i]) * int(xb[i]);
        }
        acc += float(isum) * combined_scale;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Two kernels — the fused QKV projection (`q8_qkv_proj`) and a
/// per-projection variant with RoPE (`q8_proj_rope`).
pub struct QkvKernel;
impl crate::kernels::TiledKernel for QkvKernel {
    const KERNEL_NAME: &'static str = "q8_qkv_proj";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}

pub struct ProjRopeKernel;
impl crate::kernels::TiledKernel for ProjRopeKernel {
    const KERNEL_NAME: &'static str = "q8_proj_rope";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
