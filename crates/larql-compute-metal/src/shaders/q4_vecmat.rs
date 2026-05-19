//! Q4_0 vector-matrix multiply (scatter-accumulate).
//!
//! out[K] = activation[N] @ Q4[N, K]
//!
//! One thread per output element. Each thread reads one column
//! across all N rows, accumulating weighted Q4 values.
//! GPU-hostile pattern (scatter), but parallel across K output elements.

pub const SHADER: &str = r#"
kernel void q4_vecmat(
    device const float* activation [[buffer(0)]],
    device const uchar* Q4         [[buffer(1)]],
    device float*       out        [[buffer(2)]],
    constant uint&      N          [[buffer(3)]],
    constant uint&      K          [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= K) return;
    uint blocks_per_row = K / 32;
    uint bytes_per_row = blocks_per_row * 18;
    uint block_idx = tid / 32;
    uint elem_in_block = tid % 32;
    uint nibble_idx = elem_in_block / 2;
    bool is_high = (elem_in_block & 1) != 0;
    float acc = 0.0f;
    for (uint row = 0; row < N; row++) {
        float act = activation[row];
        if (act < 1e-10f && act > -1e-10f) continue;
        device const uchar* block = Q4 + row * bytes_per_row + block_idx * 18;
        ushort scale_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        float q4_scale = decode_f16_metal(scale_bits);
        uchar byte = block[2 + nibble_idx];
        int q_val = is_high ? (int(byte >> 4) - 8) : (int(byte & 0x0F) - 8);
        acc += float(q_val) * q4_scale * act;
    }
    out[tid] = acc;
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4_vecmat";
}
