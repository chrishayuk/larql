//! Element-wise CUDA kernels used by the device-resident decode path.
//!
//! `cuda-decode-device-resident` Phase 2 introduced GPU versions of
//! the per-layer host helpers (`rms_norm_vec`, `activate`,
//! `add_in_place`) so `decode_token_device` can keep the running
//! residual `h` on the device across the entire layer loop. Each
//! kernel is correctness-first: matches the CPU reference to within
//! 1e-3 max-element absolute difference; tuning is a follow-up.

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

/// Single-block RMSNorm. One CUDA block (1024 threads) processes the
/// whole `[n]` vector via a strided loop and a parallel reduction.
/// Mirrors the CPU body in `decode::rms_norm_vec`:
///
/// ```text
/// inv_rms = rsqrt(mean(x²) + eps)
/// out[i]  = x[i] * inv_rms * (weight[i] + offset)        if has_weight
///         = x[i] * inv_rms                               otherwise
/// ```
const RMS_NORM_SRC: &str = r#"
extern "C" __global__ void rms_norm_vec_f32(
    const float* x,
    const float* weight,
    int n,
    int has_weight,
    float eps,
    float norm_offset,
    float* out
) {
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];

    float ss = 0.0f;
    for (int i = tid; i < n; i += bdim) {
        float v = x[i];
        ss += v * v;
    }
    smem[tid] = ss;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(smem[0] / (float)n + eps);

    if (has_weight) {
        for (int i = tid; i < n; i += bdim) {
            out[i] = x[i] * inv_rms * (weight[i] + norm_offset);
        }
    } else {
        for (int i = tid; i < n; i += bdim) {
            out[i] = x[i] * inv_rms;
        }
    }
}
"#;

/// Element-wise gate × up activation. `out[i] = act(gate[i]) * up[i]`
/// where `act` is either Silu (`g / (1 + e^-g)`) or GeluTanh
/// (`0.5 g (1 + tanh(0.7978846(g + 0.044715 g³)))`). Matches the CPU
/// `decode::activate` body.
const SILU_GATE_UP_SRC: &str = r#"
extern "C" __global__ void silu_gate_up_f32(
    const float* gate,
    const float* up,
    int n,
    int gelu,
    float* out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = gate[idx];
    float a;
    if (gelu) {
        a = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
    } else {
        a = g / (1.0f + expf(-g));
    }
    out[idx] = a * up[idx];
}
"#;

/// `dst[i] += delta[i]`. Trivial element-wise; bandwidth-bound.
const ADD_IN_PLACE_SRC: &str = r#"
extern "C" __global__ void add_in_place_f32(
    float* dst,
    const float* delta,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += delta[idx];
}
"#;

/// `dst[i] *= scalar`. Used for the per-layer residual scale (Gemma's
/// `layer_scalar`). Trivial element-wise.
const SCALE_INPLACE_SRC: &str = r#"
extern "C" __global__ void scale_inplace_f32(
    float* dst,
    int n,
    float scalar
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] *= scalar;
}
"#;

/// Q8_1 quantize. Memory layout matches llama.cpp's `block_q8_1`
/// (`ggml/src/ggml-cuda/common.cuh`):
///
/// ```text
/// struct __align__(4) block_q8_1 {
///     half2  ds;     // (fp16 scale, fp16 scale * sum_of_block)
///     int8_t qs[32]; // 32 quantised input values
/// }; // 36 bytes per block, alignment 4
/// ```
///
/// One warp per 32-element block; warp-shuffle reductions for both
/// the amax (→ scale) and the per-block sum.
///
/// `cuda-q4k-mmvq-int8` Phase 1.
const QUANTIZE_Q8_1_SRC: &str = r#"
__device__ unsigned short f32_to_f16_bits(float v) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}

extern "C" __global__ void quantize_q8_1_f32(
    const float* __restrict__ x,
    int n_blocks,
    unsigned char* __restrict__ out
) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    int t = threadIdx.x; // 0..31

    float v = x[(size_t)b * 32 + t];

    // amax across the warp
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float r = __shfl_xor_sync(0xffffffff, amax, o);
        amax = fmaxf(amax, r);
    }

    float scale = amax / 127.0f;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    int q = __float2int_rn(v * inv_scale);
    if (q > 127) q = 127;
    if (q < -128) q = -128;

    // sum across the warp
    float sum_x = v;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sum_x += __shfl_xor_sync(0xffffffff, sum_x, o);
    }

    // Write the 36-byte block. ds occupies bytes [0..4] as a packed
    // half2 (lo: scale, hi: scale * sum). qs occupies bytes [4..36].
    unsigned char* block_ptr = out + (size_t)b * 36;
    if (t == 0) {
        unsigned short s_h = f32_to_f16_bits(scale);
        unsigned short m_h = f32_to_f16_bits(scale * sum_x);
        unsigned int packed = (unsigned int)s_h | ((unsigned int)m_h << 16);
        *((unsigned int*)block_ptr) = packed;
    }
    block_ptr[4 + t] = (unsigned char)((signed char)q);
}
"#;

static RMS_NORM_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static SILU_GATE_UP_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static ADD_IN_PLACE_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static SCALE_INPLACE_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static QUANTIZE_Q8_1_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();

fn load_kernel(
    drv: &Driver,
    cell: &'static OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)>,
    src: &str,
    fn_name: &str,
) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = cell.get() {
        return Ok(f);
    }
    let ptx = compile_ptx(src)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile {fn_name}: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load {fn_name} module: {e:?}")))?;
    let func = module
        .load_function(fn_name)
        .map_err(|e| CudaInitError::DriverMissing(format!("load {fn_name} function: {e:?}")))?;
    let _ = cell.set((module, func));
    let (_, f) = cell.get().unwrap();
    Ok(f)
}

/// Device-resident RMSNorm. Allocates a fresh output buffer of length
/// `n`. `weight` may be `None` (matches the CPU fallback when the
/// caller passes an empty norm slice).
pub(crate) fn rms_norm_device(
    backend: &CudaBackend,
    x_dev: &CudaSlice<f32>,
    weight_dev: Option<&CudaSlice<f32>>,
    n: usize,
    eps: f32,
    norm_offset: f32,
) -> Result<CudaSlice<f32>, CudaInitError> {
    if x_dev.len() != n {
        return Err(CudaInitError::DriverMissing(format!(
            "rms_norm_device: x_dev.len={} != n={}",
            x_dev.len(),
            n
        )));
    }
    if let Some(w) = weight_dev {
        if w.len() != n {
            return Err(CudaInitError::DriverMissing(format!(
                "rms_norm_device: weight_dev.len={} != n={}",
                w.len(),
                n
            )));
        }
    }
    let drv = backend.driver();
    let func = load_kernel(drv, &RMS_NORM_FUNC, RMS_NORM_SRC, "rms_norm_vec_f32")?;
    let mut out = drv.device_alloc_uninit(n)?;
    // 1024 threads = max blockDim on every supported arch. Single
    // block — sufficient for hidden=2560 with a strided loop.
    let block_dim: u32 = 1024;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: block_dim * std::mem::size_of::<f32>() as u32,
    };
    let n_i = n as i32;
    let has_weight_i: i32 = if weight_dev.is_some() { 1 } else { 0 };

    // Always pass a pointer to the kernel; if has_weight=0 it isn't
    // dereferenced, but we still need a valid device pointer so the
    // launch builder has something to bind. Reuse the output buffer
    // — kernel won't touch it on the unused path.
    let placeholder = out.clone();
    let weight_arg = weight_dev.unwrap_or(&placeholder);

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(x_dev)
            .arg(weight_arg)
            .arg(&n_i)
            .arg(&has_weight_i)
            .arg(&eps)
            .arg(&norm_offset)
            .arg(&mut out)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch rms_norm: {e:?}")))?;
    }
    Ok(out)
}

/// Device-resident silu/gelu gate × up. `gelu_tanh` selects the
/// activation. Allocates a fresh output buffer of length `n`.
pub(crate) fn silu_gate_up_device(
    backend: &CudaBackend,
    gate_dev: &CudaSlice<f32>,
    up_dev: &CudaSlice<f32>,
    n: usize,
    gelu_tanh: bool,
) -> Result<CudaSlice<f32>, CudaInitError> {
    if gate_dev.len() != n || up_dev.len() != n {
        return Err(CudaInitError::DriverMissing(format!(
            "silu_gate_up_device shape mismatch: gate={} up={} n={n}",
            gate_dev.len(),
            up_dev.len(),
        )));
    }
    let drv = backend.driver();
    let func = load_kernel(
        drv,
        &SILU_GATE_UP_FUNC,
        SILU_GATE_UP_SRC,
        "silu_gate_up_f32",
    )?;
    let mut out = drv.device_alloc_uninit(n)?;
    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block_dim), 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let gelu_i: i32 = if gelu_tanh { 1 } else { 0 };
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(gate_dev)
            .arg(up_dev)
            .arg(&n_i)
            .arg(&gelu_i)
            .arg(&mut out)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch silu_gate_up: {e:?}")))?;
    }
    Ok(out)
}

/// `dst += delta` in place on the device.
pub(crate) fn add_in_place_device(
    backend: &CudaBackend,
    dst: &mut CudaSlice<f32>,
    delta: &CudaSlice<f32>,
) -> Result<(), CudaInitError> {
    if dst.len() != delta.len() {
        return Err(CudaInitError::DriverMissing(format!(
            "add_in_place_device shape mismatch: dst={} delta={}",
            dst.len(),
            delta.len(),
        )));
    }
    let n = dst.len();
    let drv = backend.driver();
    let func = load_kernel(
        drv,
        &ADD_IN_PLACE_FUNC,
        ADD_IN_PLACE_SRC,
        "add_in_place_f32",
    )?;
    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block_dim), 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(dst)
            .arg(delta)
            .arg(&n_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch add_in_place: {e:?}")))?;
    }
    Ok(())
}

/// `dst *= scalar` in place on the device.
pub(crate) fn scale_inplace_device(
    backend: &CudaBackend,
    dst: &mut CudaSlice<f32>,
    scalar: f32,
) -> Result<(), CudaInitError> {
    let n = dst.len();
    let drv = backend.driver();
    let func = load_kernel(
        drv,
        &SCALE_INPLACE_FUNC,
        SCALE_INPLACE_SRC,
        "scale_inplace_f32",
    )?;
    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block_dim), 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(dst)
            .arg(&n_i)
            .arg(&scalar)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch scale_inplace: {e:?}")))?;
    }
    Ok(())
}

/// Device-resident Q8_1 buffer. Layout is the packed `block_q8_1[]`
/// from llama.cpp: 36 bytes per 32-element block, of which the first
/// 4 are a `half2 ds` (fp16 scale, fp16 scale × sum) and the
/// following 32 are signed-int8 quants. `cuda-q4k-mmvq-int8` Phase 1.
pub(crate) struct Q8_1Buf {
    /// Raw packed bytes; length = `n_blocks * 36`.
    pub(crate) bytes: cudarc::driver::CudaSlice<u8>,
    /// Number of 32-element blocks; total elements quantised = `n_blocks * 32`.
    pub(crate) n_blocks: usize,
}

/// Quantise a device-resident f32 vector to Q8_1. `n` MUST be a
/// multiple of 32 (one Q8_1 block per 32 elements).
pub(crate) fn quantize_q8_1_device(
    backend: &CudaBackend,
    x_dev: &CudaSlice<f32>,
    n: usize,
) -> Result<Q8_1Buf, CudaInitError> {
    if x_dev.len() != n {
        return Err(CudaInitError::DriverMissing(format!(
            "quantize_q8_1_device: x_dev.len={} != n={}",
            x_dev.len(),
            n
        )));
    }
    if n % 32 != 0 {
        return Err(CudaInitError::DriverMissing(format!(
            "quantize_q8_1_device: n={n} must be a multiple of 32",
        )));
    }
    let n_blocks = n / 32;
    let bytes_len = n_blocks * 36;

    let drv = backend.driver();
    let func = load_kernel(
        drv,
        &QUANTIZE_Q8_1_FUNC,
        QUANTIZE_Q8_1_SRC,
        "quantize_q8_1_f32",
    )?;
    let mut bytes = drv
        .stream
        .alloc_zeros::<u8>(bytes_len)
        .map_err(|e| CudaInitError::DriverMissing(format!("alloc q8_1 bytes: {e:?}")))?;

    let n_blocks_i = n_blocks as i32;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(x_dev)
            .arg(&n_blocks_i)
            .arg(&mut bytes)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch quantize_q8_1: {e:?}")))?;
    }

    Ok(Q8_1Buf { bytes, n_blocks })
}

#[cfg(test)]
mod tests {
    //! Inline kernel tests gated on `LARQL_CUDA_AVAILABLE=1`. Run with:
    //! `LARQL_CUDA_AVAILABLE=1 cargo test -p larql-compute --features cuda --lib`.

    use super::*;
    use larql_models::quant::half::f16_to_f32;

    fn gpu_or_skip() -> Option<CudaBackend> {
        if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
            eprintln!("skipping CUDA q8_1 quantize test: set LARQL_CUDA_AVAILABLE=1");
            return None;
        }
        CudaBackend::new().ok()
    }

    /// Dequantise the packed `block_q8_1[]` byte stream on host.
    /// Returns `(reconstructed_values, per_block_scale)`.
    fn dequant_q8_1_host(bytes: &[u8], n_blocks: usize) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(bytes.len(), n_blocks * 36);
        let mut out = Vec::with_capacity(n_blocks * 32);
        let mut scales = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let base = b * 36;
            let scale_bits = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
            let scale = f16_to_f32(scale_bits);
            scales.push(scale);
            for i in 0..32 {
                let q = bytes[base + 4 + i] as i8;
                out.push(scale * (q as f32));
            }
        }
        (out, scales)
    }

    /// `cuda-q4k-mmvq-int8` Phase 1: quantising and dequantising a
    /// random vector via the GPU kernel SHALL match the original
    /// within one Q8_1 quantum (per-block-`amax / 127`).
    #[test]
    fn q8_1_quantize_roundtrips_to_within_quant_noise() {
        let Some(backend) = gpu_or_skip() else { return };

        let n = 2560;
        // Synthetic input with a wide dynamic range so multiple blocks
        // see different per-block amax values.
        let mut s: u64 = 0xDEAD_BEEF_DEAD_BEEF;
        let x: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFF_FFFF) as f32 / 8_388_608.0) - 0.5
            })
            .collect();

        let x_dev = backend.htod_f32(&x).expect("htod x");
        let buf = quantize_q8_1_device(&backend, &x_dev, n).expect("quantize_q8_1_device");
        backend.driver().sync().expect("sync");
        let bytes: Vec<u8> = backend
            .driver()
            .stream
            .clone_dtoh(&buf.bytes)
            .expect("dtoh q8_1 bytes");

        assert_eq!(bytes.len(), buf.n_blocks * 36);
        assert_eq!(buf.n_blocks, n / 32);

        let (recon, scales) = dequant_q8_1_host(&bytes, buf.n_blocks);
        assert_eq!(recon.len(), n);

        // Per-block: max |original - reconstructed| ≤ per-block amax / 127
        // (one Q8_1 quantum). Allow a small fp16 fudge factor for the
        // half-precision scale rounding.
        let mut worst_per_quantum_ratio: f32 = 0.0;
        for b in 0..buf.n_blocks {
            let block_amax = (0..32).map(|i| x[b * 32 + i].abs()).fold(0.0_f32, f32::max);
            let scale = scales[b];
            // The kernel's scale = block_amax / 127, but quantised
            // through fp16 — so the quantum is ~ scale itself.
            let quantum = scale.max(block_amax / 127.0).max(1e-9);
            for i in 0..32 {
                let err = (x[b * 32 + i] - recon[b * 32 + i]).abs();
                worst_per_quantum_ratio = worst_per_quantum_ratio.max(err / quantum);
            }
        }
        assert!(
            worst_per_quantum_ratio <= 1.05,
            "Q8_1 round-trip error exceeds 1.05 quanta ({worst_per_quantum_ratio}); kernel \
             likely has a quantization or fp16-scale bug",
        );
    }
}
