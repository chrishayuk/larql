//! Dense-FFN block contribution census.
//!
//! One threadgroup owns one contiguous intermediate-channel block. Threads
//! divide the hidden outputs, reproduce that block's raw down projection, and
//! reduce its squared L2 mass. This reads the down matrix exactly once across
//! all blocks and returns only one f32 per logical object.

pub const THREADS_PER_TG: u64 = 256;

pub const SHADER: &str = r#"
constant uint FFN_BLOCK_THREADS = 256;
constant uint FFN_BLOCK_SIMDGROUPS = FFN_BLOCK_THREADS / 32;

kernel void f16_ffn_block_contribution(
    device const half*  down           [[buffer(0)]], // [hidden, intermediate]
    device const float* inner          [[buffer(1)]], // [intermediate]
    device float*       masses         [[buffer(2)]], // [blocks]
    constant uint&      hidden         [[buffer(3)]],
    constant uint&      intermediate   [[buffer(4)]],
    constant uint&      block_channels [[buffer(5)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]])
{
    uint start = block * block_channels;
    if (start >= intermediate) return;
    uint end = min(start + block_channels, intermediate);

    float local_mass = 0.0f;
    for (uint output = tid; output < hidden; output += FFN_BLOCK_THREADS) {
        device const half* row = down + output * intermediate;
        float value = 0.0f;
        for (uint channel = start; channel < end; ++channel) {
            value = fma(float(row[channel]), inner[channel], value);
        }
        local_mass = fma(value, value, local_mass);
    }

    local_mass = simd_sum(local_mass);
    threadgroup float subgroup_mass[FFN_BLOCK_SIMDGROUPS];
    if (lane == 0) subgroup_mass[sg] = local_mass;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float value = lane < FFN_BLOCK_SIMDGROUPS ? subgroup_mass[lane] : 0.0f;
        value = simd_sum(value);
        if (lane == 0) masses[block] = value;
    }
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "f16_ffn_block_contribution";
}
