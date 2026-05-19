//! Per-Layer Embeddings (PLE) gate-apply: fused `gelu_tanh(gate) * per_layer_input`.
//!
//! Used by Gemma 4 E2B. The full PLE block per layer is:
//!
//! ```text
//! gate    = h × W_input_gate.T          // [ple_dim]   ← f32_gemv
//! gate    = gelu_tanh(gate) * per_layer_input[layer]  ← THIS KERNEL
//! contrib = gate × W_projection.T       // [hidden]    ← f32_gemv
//! h      += rms_norm(contrib) * w       // [hidden]    ← post_ffn_norm_residual_add (reused)
//! ```
//!
//! Fusing the activation with the elementwise multiply against the
//! precomputed per-layer-input row saves one dispatch per layer (35 layers
//! × 1 dispatch = 35 saved per token on E2B).
//!
//! Math matches `crates/larql-inference/src/forward/ple.rs::apply_per_layer_embedding`:
//! the GELU-tanh approximation uses the same constants (sqrt(2/pi),
//! 0.044715, x³ inner term) as `metal/shaders/activation.rs::gelu_tanh`,
//! and the multiply is the same `gate * per_layer_input` step the CPU
//! path performs after the activation.

pub const SHADER: &str = r#"
// gate[i] = gelu_tanh(gate[i]) * per_layer_input[i]
// In-place is supported: bind gate as both input and output.
kernel void ple_gate_apply(
    device const float* gate_in            [[buffer(0)]],   // [ple_dim]
    device const float* per_layer_input    [[buffer(1)]],   // [ple_dim]
    device float*       gate_out           [[buffer(2)]],   // [ple_dim]
    constant uint&      N                  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float x = gate_in[tid];
    // GELU-tanh approximation, same as `gelu_tanh` shader.
    // Clamp the tanh argument to avoid `exp(2y)` overflow inside Apple
    // Silicon's tanh.  Mathematically equivalent at f32 precision since
    // tanh saturates by |y|=10.
    float c = 0.7978845608f; // sqrt(2/pi)
    float y = c * (x + 0.044715f * x * x * x);
    y = clamp(y, -15.0f, 15.0f);
    float g = 0.5f * x * (1.0f + tanh(y));
    gate_out[tid] = g * per_layer_input[tid];
}
"#;

pub struct GateApplyKernel;
impl crate::kernels::ShaderKernel for GateApplyKernel {
    const KERNEL_NAME: &'static str = "ple_gate_apply";
}
