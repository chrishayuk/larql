//! Fused Q4_K QKV — GGUF 144-byte super-block layout.
//!
//! Two kernels:
//!
//! - `q4k_qkv_proj`: fused Q+K+V in one dispatch when all three weights
//!   are Q4_K (uniform-quant models). Rows `0..q_rows` come from `Wq`,
//!   `q_rows..q_rows+k_rows` from `Wk`, rest from `Wv`. Each simdgroup
//!   handles one row; `ROWS_PER_TG = 8`.
//! - `q4k_proj`: single-matrix variant used for the O projection
//!   (`attn_out → h`) and wherever a standalone Q4_K matvec is needed.
//!
//! Both use **manual byte offsets on the 144-byte GGUF layout** — not the
//! 148-byte `block_q4_K` MSL struct, whose extra `mins[4]` makes pointer
//! arithmetic mis-stride across rows on GGUF data. Matches the proven
//! decode in `q4k_matvec` and `q4k_q6k_qkv_proj`.

pub const SHADER: &str = r#"
constant uint Q4K_QKV_ROWS_PER_TG = 8;
constant uint Q4K_QKV_BLOCK_SIZE  = 144;

// Unpack the 12 packed scale+min bytes of a GGUF Q4_K super-block into
// parallel arrays of 8 scales and 8 mins (llama.cpp `get_scale_min_k4`).
// Inlined into each kernel; not shared because MSL has no function
// parameter for writable arrays without `thread` qualifier gymnastics.
#define Q4K_UNPACK_SCALES_MINS(sb_bytes, scales, mins) do {                   \
    for (uint j = 0; j < 4; j++) {                                            \
        scales[j] = uint(sb_bytes[j])   & 0x3Fu;                              \
        mins[j]   = uint(sb_bytes[j+4]) & 0x3Fu;                              \
    }                                                                         \
    for (uint j = 4; j < 8; j++) {                                            \
        scales[j] = (uint(sb_bytes[j+4]) & 0x0Fu) | ((uint(sb_bytes[j-4]) >> 6) << 4); \
        mins[j]   = (uint(sb_bytes[j+4]) >> 4)    | ((uint(sb_bytes[j])   >> 6) << 4); \
    }                                                                         \
} while (0)

kernel void q4k_qkv_proj(
    device const uchar*  Wq   [[buffer(0)]],
    device const uchar*  Wk   [[buffer(1)]],
    device const uchar*  Wv   [[buffer(2)]],
    device const float*  X    [[buffer(3)]],
    device float*        Q_out [[buffer(4)]],
    device float*        K_out [[buffer(5)]],
    device float*        V_out [[buffer(6)]],
    constant uint&       q_rows [[buffer(7)]],
    constant uint&       k_rows [[buffer(8)]],
    constant uint&       v_rows [[buffer(9)]],
    constant uint&       K      [[buffer(10)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint total_rows = q_rows + k_rows + v_rows;
    uint global_row = tg_id * Q4K_QKV_ROWS_PER_TG + sg_id;
    if (global_row >= total_rows) return;

    device const uchar* W;
    device float* out_buf;
    uint local_row;
    if (global_row < q_rows) {
        W = Wq; out_buf = Q_out; local_row = global_row;
    } else if (global_row < q_rows + k_rows) {
        W = Wk; out_buf = K_out; local_row = global_row - q_rows;
    } else {
        W = Wv; out_buf = V_out; local_row = global_row - q_rows - k_rows;
    }

    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_QKV_BLOCK_SIZE;
    device const uchar* row = W + local_row * bytes_per_row;

    float acc = 0.0f;
    for (uint sb = lane; sb < superblocks; sb += 32) {
        device const uchar* block = row + sb * Q4K_QKV_BLOCK_SIZE;

        ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d    = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        device const uchar* sb_bytes = block + 4;
        uint scales[8];
        uint mins[8];
        Q4K_UNPACK_SCALES_MINS(sb_bytes, scales, mins);

        device const uchar* qs = block + 16;
        uint x_base = sb * 256;
        float sb_acc = 0.0f;
        for (uint g = 0; g < 4; g++) {
            uint sub_lo = 2 * g;
            uint sub_hi = 2 * g + 1;
            float sc_lo = d * float(scales[sub_lo]);
            float sc_hi = d * float(scales[sub_hi]);
            float mn_lo = dmin * float(mins[sub_lo]);
            float mn_hi = dmin * float(mins[sub_hi]);
            float dot_lo = 0.0f, sum_lo = 0.0f;
            float dot_hi = 0.0f, sum_hi = 0.0f;
            for (uint l = 0; l < 32; l++) {
                uchar byte = qs[g * 32 + l];
                float nib_lo = float(byte & 0x0Fu);
                float nib_hi = float((byte >> 4) & 0x0Fu);
                float xlo = X[x_base + sub_lo * 32 + l];
                float xhi = X[x_base + sub_hi * 32 + l];
                dot_lo += nib_lo * xlo;
                sum_lo += xlo;
                dot_hi += nib_hi * xhi;
                sum_hi += xhi;
            }
            sb_acc += sc_lo * dot_lo - mn_lo * sum_lo;
            sb_acc += sc_hi * dot_hi - mn_hi * sum_hi;
        }
        acc += sb_acc;
    }
    acc = simd_sum(acc);
    if (lane == 0) out_buf[local_row] = acc;
}

kernel void q4k_proj(
    device const uchar*  W4K [[buffer(0)]],
    device const float*  X   [[buffer(1)]],
    device float*        out [[buffer(2)]],
    constant uint&       N   [[buffer(3)]],
    constant uint&       K   [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q4K_QKV_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_QKV_BLOCK_SIZE;
    device const uchar* row = W4K + row_idx * bytes_per_row;

    float acc = 0.0f;
    for (uint sb = lane; sb < superblocks; sb += 32) {
        device const uchar* block = row + sb * Q4K_QKV_BLOCK_SIZE;

        ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d    = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        device const uchar* sb_bytes = block + 4;
        uint scales[8];
        uint mins[8];
        Q4K_UNPACK_SCALES_MINS(sb_bytes, scales, mins);

        device const uchar* qs = block + 16;
        uint x_base = sb * 256;
        float sb_acc = 0.0f;
        for (uint g = 0; g < 4; g++) {
            uint sub_lo = 2 * g;
            uint sub_hi = 2 * g + 1;
            float sc_lo = d * float(scales[sub_lo]);
            float sc_hi = d * float(scales[sub_hi]);
            float mn_lo = dmin * float(mins[sub_lo]);
            float mn_hi = dmin * float(mins[sub_hi]);
            float dot_lo = 0.0f, sum_lo = 0.0f;
            float dot_hi = 0.0f, sum_hi = 0.0f;
            for (uint l = 0; l < 32; l++) {
                uchar byte = qs[g * 32 + l];
                float nib_lo = float(byte & 0x0Fu);
                float nib_hi = float((byte >> 4) & 0x0Fu);
                float xlo = X[x_base + sub_lo * 32 + l];
                float xhi = X[x_base + sub_hi * 32 + l];
                dot_lo += nib_lo * xlo;
                sum_lo += xlo;
                dot_hi += nib_hi * xhi;
                sum_hi += xhi;
            }
            sb_acc += sc_lo * dot_lo - mn_lo * sum_lo;
            sb_acc += sc_hi * dot_hi - mn_hi * sum_hi;
        }
        acc += sb_acc;
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Two kernels share this file's geometry — fused QKV projection
/// (`q4k_qkv_proj`) and the per-projection variant (`q4k_proj`).
/// Each gets its own marker so the binding site picks the right one
/// by type path.
pub struct QkvKernel;
impl crate::kernels::TiledKernel for QkvKernel {
    const KERNEL_NAME: &'static str = "q4k_qkv_proj";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}

pub struct ProjKernel;
impl crate::kernels::TiledKernel for ProjKernel {
    const KERNEL_NAME: &'static str = "q4k_proj";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
