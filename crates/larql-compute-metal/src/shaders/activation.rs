//! Standalone activation functions for non-gated FFN (StarCoder2, GPT-2).
//!
//! Unlike GEGLU which multiplies gate*up, these apply activation in-place
//! to a single buffer: out[i] = activation(input[i]).
//!
//! Used when ffn_type == Standard: up → activation → down (no gate).

pub const SHADER: &str = r#"
// SiLU / Swish: out = x / (1 + exp(-x))
kernel void silu(
    device const float* input [[buffer(0)]],
    device float*       out   [[buffer(1)]],
    constant uint&      N     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float x = input[tid];
    out[tid] = x / (1.0f + exp(-x));
}

// GELU with tanh approximation: out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_tanh(
    device const float* input [[buffer(0)]],
    device float*       out   [[buffer(1)]],
    constant uint&      N     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float x = input[tid];
    // Clamp the tanh argument to avoid `exp(2y)` overflow inside Apple
    // Silicon's tanh (see note in `geglu_gelu_tanh`). Mathematically
    // equivalent at f32 precision since tanh saturates by |y|=10.
    float c = 0.7978845608f; // sqrt(2/pi)
    float y = c * (x + 0.044715f * x * x * x);
    y = clamp(y, -15.0f, 15.0f);
    float t = tanh(y);
    out[tid] = 0.5f * x * (1.0f + t);
}
"#;

pub struct SiluKernel;
impl crate::kernels::ShaderKernel for SiluKernel {
    const KERNEL_NAME: &'static str = "silu";
}

pub struct GeluTanhKernel;
impl crate::kernels::ShaderKernel for GeluTanhKernel {
    const KERNEL_NAME: &'static str = "gelu_tanh";
}
