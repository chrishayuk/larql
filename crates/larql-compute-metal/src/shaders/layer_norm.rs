//! LayerNorm — standard layer normalization (mean-subtraction + variance normalization).
//!
//! out = (x - mean) / sqrt(var + eps) * weight + bias
//!
//! Used by StarCoder2, GPT-2, BERT. Most modern LLMs use RMSNorm instead.
//! For RMSNorm, see residual_inject.rs (rms_norm kernel).

pub const SHADER: &str = r#"
// LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
// Grid: (len, 1, 1). Each thread handles one element, but reads all elements for mean/var.
kernel void layer_norm(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float*       out    [[buffer(3)]],
    constant uint&      len    [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    constant float&     offset [[buffer(6)]],   // norm weight offset (0.0 standard, 1.0 Gemma)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;

    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < len; i++) {
        sum += x[i];
    }
    float mean = sum / float(len);

    // Compute variance
    float var_sum = 0.0f;
    for (uint i = 0; i < len; i++) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float inv_std = 1.0f / sqrt(var_sum / float(len) + eps);

    out[tid] = (x[tid] - mean) * inv_std * (weight[tid] + offset) + bias[tid];
}

// LayerNorm without bias: out = (x - mean) / sqrt(var + eps) * weight
kernel void layer_norm_no_bias(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      len    [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    constant float&     offset [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= len) return;

    float sum = 0.0f;
    for (uint i = 0; i < len; i++) {
        sum += x[i];
    }
    float mean = sum / float(len);

    float var_sum = 0.0f;
    for (uint i = 0; i < len; i++) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float inv_std = 1.0f / sqrt(var_sum / float(len) + eps);

    out[tid] = (x[tid] - mean) * inv_std * (weight[tid] + offset);
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "layer_norm";
}

pub struct NoBiasKernel;
impl crate::kernels::ShaderKernel for NoBiasKernel {
    const KERNEL_NAME: &'static str = "layer_norm_no_bias";
}
