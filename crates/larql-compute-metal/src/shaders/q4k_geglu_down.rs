//! Fused GEGLU activation + Q4_K down projection.
//!
//! Eliminates the GEGLU dispatch entirely by computing `silu(gate) × up`
//! (or `gelu_tanh(gate) × up` for Gemma/GPT-2/Phi) on-the-fly during the
//! down projection. Each lane computes the activation for its assigned
//! sub-block elements and immediately multiplies by the dequantised
//! weight.
//!
//!   `down_out[row] = Σᵢ W_down[row, i] · act(gate[i]) · up[i]`
//!
//! Saves one dispatch + one full read/write of the inter-sized
//! activation buffer.
//!
//! Uses the **GGUF 144-byte Q4_K layout** (manual byte offsets +
//! `get_scale_min_k4` packing), matching `q4k_matvec`.

pub const SHADER: &str = r#"
constant uint Q4K_GD_ROWS_PER_TG = 8;
constant uint Q4K_GD_BLOCK_SIZE  = 144;

#define Q4K_GD_UNPACK_SCALES_MINS(sb_bytes, scales, mins) do {                \
    for (uint j = 0; j < 4; j++) {                                            \
        scales[j] = uint(sb_bytes[j])   & 0x3Fu;                              \
        mins[j]   = uint(sb_bytes[j+4]) & 0x3Fu;                              \
    }                                                                         \
    for (uint j = 4; j < 8; j++) {                                            \
        scales[j] = (uint(sb_bytes[j+4]) & 0x0Fu) | ((uint(sb_bytes[j-4]) >> 6) << 4); \
        mins[j]   = (uint(sb_bytes[j+4]) >> 4)    | ((uint(sb_bytes[j])   >> 6) << 4); \
    }                                                                         \
} while (0)

// SiLU + down (Llama, Mistral, Qwen).
kernel void q4k_geglu_silu_down(
    device const uchar*  W_down [[buffer(0)]],   // down weights [N, inter] Q4_K GGUF
    device const float*  gate   [[buffer(1)]],   // gate output [inter]
    device const float*  up     [[buffer(2)]],   // up output [inter]
    device float*        out    [[buffer(3)]],   // output [N] (hidden)
    constant uint&       N      [[buffer(4)]],   // hidden (output rows)
    constant uint&       K      [[buffer(5)]],   // inter (input dim)
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * Q4K_GD_ROWS_PER_TG + sg_id;
    if (row >= N) return;

    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_GD_BLOCK_SIZE;
    device const uchar* row_bytes = W_down + row * bytes_per_row;
    float acc = 0.0f;

    for (uint sb = lane; sb < superblocks; sb += 32) {
        device const uchar* block = row_bytes + sb * Q4K_GD_BLOCK_SIZE;

        ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d    = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        device const uchar* sb_bytes = block + 4;
        uint scales[8];
        uint mins[8];
        Q4K_GD_UNPACK_SCALES_MINS(sb_bytes, scales, mins);

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
                uint idx_lo = x_base + sub_lo * 32 + l;
                uint idx_hi = x_base + sub_hi * 32 + l;
                float g_lo = gate[idx_lo];
                float act_lo = (g_lo / (1.0f + exp(-g_lo))) * up[idx_lo];
                float g_hi = gate[idx_hi];
                float act_hi = (g_hi / (1.0f + exp(-g_hi))) * up[idx_hi];
                dot_lo += nib_lo * act_lo;
                sum_lo += act_lo;
                dot_hi += nib_hi * act_hi;
                sum_hi += act_hi;
            }
            sb_acc += sc_lo * dot_lo - mn_lo * sum_lo;
            sb_acc += sc_hi * dot_hi - mn_hi * sum_hi;
        }
        acc += sb_acc;
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}

// GELU-tanh + down (Gemma, GPT-2, Phi).
kernel void q4k_geglu_gelu_tanh_down(
    device const uchar*  W_down [[buffer(0)]],
    device const float*  gate   [[buffer(1)]],
    device const float*  up     [[buffer(2)]],
    device float*        out    [[buffer(3)]],
    constant uint&       N      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * Q4K_GD_ROWS_PER_TG + sg_id;
    if (row >= N) return;

    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_GD_BLOCK_SIZE;
    device const uchar* row_bytes = W_down + row * bytes_per_row;
    float acc = 0.0f;

    float c = 0.7978845608f; // sqrt(2/pi)
    for (uint sb = lane; sb < superblocks; sb += 32) {
        device const uchar* block = row_bytes + sb * Q4K_GD_BLOCK_SIZE;

        ushort d_bits    = ushort(block[0]) | (ushort(block[1]) << 8);
        ushort dmin_bits = ushort(block[2]) | (ushort(block[3]) << 8);
        float d    = decode_f16_metal(d_bits);
        float dmin = decode_f16_metal(dmin_bits);

        device const uchar* sb_bytes = block + 4;
        uint scales[8];
        uint mins[8];
        Q4K_GD_UNPACK_SCALES_MINS(sb_bytes, scales, mins);

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
                uint idx_lo = x_base + sub_lo * 32 + l;
                uint idx_hi = x_base + sub_hi * 32 + l;
                float g_lo = gate[idx_lo];
                float t_lo = tanh(c * (g_lo + 0.044715f * g_lo * g_lo * g_lo));
                float act_lo = (0.5f * g_lo * (1.0f + t_lo)) * up[idx_lo];
                float g_hi = gate[idx_hi];
                float t_hi = tanh(c * (g_hi + 0.044715f * g_hi * g_hi * g_hi));
                float act_hi = (0.5f * g_hi * (1.0f + t_hi)) * up[idx_hi];
                dot_lo += nib_lo * act_lo;
                sum_lo += act_lo;
                dot_hi += nib_hi * act_hi;
                sum_hi += act_hi;
            }
            sb_acc += sc_lo * dot_lo - mn_lo * sum_lo;
            sb_acc += sc_hi * dot_hi - mn_hi * sum_hi;
        }
        acc += sb_acc;
    }
    acc = simd_sum(acc);
    if (lane == 0) out[row] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256; // 8 rows × 32 lanes

/// Two activation variants of fused GEGLU+down — SiLU (Llama, Mistral)
/// and GELU-tanh (Gemma). Same geometry, distinct kernels.
pub struct SiluKernel;
impl crate::kernels::TiledKernel for SiluKernel {
    const KERNEL_NAME: &'static str = "q4k_geglu_silu_down";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}

pub struct GeluTanhKernel;
impl crate::kernels::TiledKernel for GeluTanhKernel {
    const KERNEL_NAME: &'static str = "q4k_geglu_gelu_tanh_down";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
