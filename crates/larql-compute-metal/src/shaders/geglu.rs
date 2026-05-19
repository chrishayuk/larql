//! GEGLU activation variants:
//!   geglu_silu:       out = silu(gate) × up       (Llama, Mistral, Qwen)
//!   geglu_gelu_tanh:  out = gelu_tanh(gate) × up  (Gemma, GPT-2, Phi)
//!
//! Element-wise, one thread per element.

pub const SHADER: &str = r#"
kernel void geglu_silu(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      N    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float g = gate[tid];
    out[tid] = (g / (1.0f + exp(-g))) * up[tid];
}

kernel void geglu_gelu_tanh(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      N    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float g = gate[tid];
    // GELU with tanh approximation:
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    //
    // Apple Silicon's `tanh` uses `(exp(2y)-1)/(exp(2y)+1)`, which overflows
    // f32 and returns NaN once |y| ≳ 44 (ln(f32_max) / 2). For gate values
    // around ±10 the argument `y` hits ~50 and poisons the activation with
    // NaNs at isolated indices. Clamping at ±15 is safe: tanh(15) differs
    // from 1.0 by < 1e-13, far below f32 precision.
    float c = 0.7978845608f; // sqrt(2/pi)
    float y = c * (g + 0.044715f * g * g * g);
    y = clamp(y, -15.0f, 15.0f);
    float t = tanh(y);
    out[tid] = (0.5f * g * (1.0f + t)) * up[tid];
}
"#;

pub struct SiluKernel;
impl crate::kernels::ShaderKernel for SiluKernel {
    const KERNEL_NAME: &'static str = "geglu_silu";
}

pub struct GeluTanhKernel;
impl crate::kernels::ShaderKernel for GeluTanhKernel {
    const KERNEL_NAME: &'static str = "geglu_gelu_tanh";
}
