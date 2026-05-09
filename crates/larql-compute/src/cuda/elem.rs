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

static RMS_NORM_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static SILU_GATE_UP_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static ADD_IN_PLACE_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static SCALE_INPLACE_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();

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
