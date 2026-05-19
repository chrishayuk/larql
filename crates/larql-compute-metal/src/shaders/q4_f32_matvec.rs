//! Q4_0 × f32 matrix-vector multiply.
//!
//! scores[N] = Q4[N, K] @ f32_x[K]
//!
//! Input is f32 (not Q8). Used for down projection where
//! activation is sparse and Q8 quantization loses precision.
//! One thread per output row, scalar inner loop.

pub const SHADER: &str = r#"
kernel void q4_f32_matvec(
    device const uchar* Q4    [[buffer(0)]],
    device const float* x     [[buffer(1)]],
    device float*       out   [[buffer(2)]],
    constant uint&      N     [[buffer(3)]],
    constant uint&      K     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    uint blocks = K / 32;
    uint bytes_per_row = blocks * 18;
    device const uchar* row = Q4 + tid * bytes_per_row;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        device const uchar* block = row + b * 18;
        ushort scale_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        float q4_scale = decode_f16_metal(scale_bits);
        device const uchar* quants = block + 2;
        device const float* xb = x + b * 32;
        float block_sum = 0.0f;
        for (uint j = 0; j < 16; j++) {
            uchar byte = quants[j];
            float lo = float(int(byte & 0x0F) - 8);
            float hi = float(int(byte >> 4) - 8);
            block_sum += lo * xb[j * 2] + hi * xb[j * 2 + 1];
        }
        acc += block_sum * q4_scale;
    }
    out[tid] = acc;
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4_f32_matvec";
}
