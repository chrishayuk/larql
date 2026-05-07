//! CUDA fused decode-time attention.
//!
//! Phase `cuda-fused-attention`: a per-row scaled-softmax kernel
//! compiled via NVRTC + a `decode_attention` helper that chains
//! cuBLAS gemm → softmax → cuBLAS gemm into one host roundtrip.
//! Single-head, single-batch. The caller splits heads / GQA before
//! calling.

use std::sync::OnceLock;

use cudarc::cublas::{
    sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
    Gemm, GemmConfig,
};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

/// CUDA C source for a row-per-block scaled-softmax with optional
/// causal mask + softcap. Written to be deterministic, easy to read,
/// and obviously-correct rather than tuned. seq_len up to ~4096
/// supported via the strided loop.
const SOFTMAX_SRC: &str = r#"
// NVRTC compiles without the standard headers, so we provide
// IEEE-754 inf/-inf bit patterns directly.
#define POS_INF (__int_as_float(0x7f800000))
#define NEG_INF (__int_as_float(0xff800000))

extern "C" __global__ void scaled_softmax(
    float *x,
    int n_rows,
    int n_cols,
    float scale,
    float softcap,    // 0 -> no softcap
    int causal        // nonzero -> apply causal mask
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    float *r = x + (size_t)row * n_cols;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float smem[];

    // ── Pass 1: pre-process + max ─────────────────────────────────
    float my_max = NEG_INF;
    for (int j = tid; j < n_cols; j += bdim) {
        float v = r[j] * scale;
        if (softcap > 0.f) {
            v = softcap * tanhf(v / softcap);
        }
        if (causal && j > row) {
            v = NEG_INF;
        }
        r[j] = v;
        if (v > my_max) my_max = v;
    }
    smem[tid] = my_max;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = smem[tid], b = smem[tid + s];
            smem[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    float row_max = smem[0];

    // ── Pass 2: exp + sum ─────────────────────────────────────────
    float my_sum = 0.f;
    for (int j = tid; j < n_cols; j += bdim) {
        float e = expf(r[j] - row_max);
        r[j] = e;
        my_sum += e;
    }
    smem[tid] = my_sum;
    __syncthreads();
    for (int s = bdim / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float row_sum = smem[0];

    // ── Pass 3: normalise ─────────────────────────────────────────
    float inv = 1.f / row_sum;
    for (int j = tid; j < n_cols; j += bdim) {
        r[j] *= inv;
    }
}
"#;

/// Lazily-loaded softmax module + function. cudarc's `CudaContext` is
/// `Send + Sync`; `OnceLock` gives us thread-safe one-time init.
static SOFTMAX_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();

fn softmax_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = SOFTMAX_FUNC.get() {
        return Ok(f);
    }
    // First call: compile PTX and load the function.
    let ptx = compile_ptx(SOFTMAX_SRC)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile softmax: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load module: {e:?}")))?;
    let func = module
        .load_function("scaled_softmax")
        .map_err(|e| CudaInitError::DriverMissing(format!("load function: {e:?}")))?;
    let _ = SOFTMAX_FUNC.set((module, func));
    let (_, f) = SOFTMAX_FUNC.get().unwrap();
    Ok(f)
}

/// Optional per-call attention knobs. The kernel folds these in.
#[derive(Clone, Copy, Debug)]
pub struct AttentionOpts {
    pub causal: bool,
    pub softcap: f32, // 0.0 → no softcap
}

impl Default for AttentionOpts {
    fn default() -> Self {
        AttentionOpts {
            causal: false,
            softcap: 0.0,
        }
    }
}

/// In-place row-wise softmax on a `[n_rows, n_cols]` row-major device
/// buffer. `scale` is applied before the row max (so `1/sqrt(d)` etc).
pub(crate) fn softmax_inplace(
    drv: &Driver,
    x_dev: &mut cudarc::driver::CudaSlice<f32>,
    n_rows: usize,
    n_cols: usize,
    scale: f32,
    opts: AttentionOpts,
) -> Result<(), CudaInitError> {
    let func = softmax_function(drv)?;
    // 1024 threads = max blockDim on every supported arch.
    let block_dim: u32 = 1024;
    let grid_dim: u32 = n_rows as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
    };
    let n_rows_i = n_rows as i32;
    let n_cols_i = n_cols as i32;
    let causal_i: i32 = if opts.causal { 1 } else { 0 };
    let softcap_f = opts.softcap;
    // SAFETY: The kernel writes at most `n_rows * n_cols` f32 values
    // starting at the slice base; the buffer length matches the shape
    // (caller guarantees).
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(x_dev)
            .arg(&n_rows_i)
            .arg(&n_cols_i)
            .arg(&scale)
            .arg(&softcap_f)
            .arg(&causal_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch softmax: {e:?}")))?;
    }
    Ok(())
}

/// Single-head decode-time attention: `out = softmax((Q @ K^T) * scale, mask) @ V`.
///
/// Inputs are row-major contiguous slices:
///   Q: `[n_q, head_dim]`
///   K: `[n_kv, head_dim]`
///   V: `[n_kv, head_dim]`
/// Output: `[n_q, head_dim]`.
///
/// One synchronous host roundtrip total.
pub fn decode_attention(
    backend: &CudaBackend,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    opts: AttentionOpts,
) -> Result<Vec<f32>, CudaInitError> {
    let drv = backend.driver();
    debug_assert_eq!(q.len(), n_q * head_dim);
    debug_assert_eq!(k.len(), n_kv * head_dim);
    debug_assert_eq!(v.len(), n_kv * head_dim);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // ── 1. attn_logits = Q @ K^T  [n_q × n_kv] row-major ─────────────
    // Reuse the same row-major-via-column-major identity from
    // cuda::matmul: passing K as the first cuBLAS arg with op=T.
    let q_dev = drv.device_buf_from(q)?;
    let k_dev = drv.device_buf_from(k)?;
    let v_dev = drv.device_buf_from(v)?;

    let mut logits_dev = drv.device_alloc(n_q * n_kv)?;
    let cfg_qk = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n_kv as i32,
        n: n_q as i32,
        k: head_dim as i32,
        alpha: 1.0_f32,
        lda: head_dim as i32,
        ldb: head_dim as i32,
        beta: 0.0_f32,
        ldc: n_kv as i32,
    };
    // SAFETY: dimensions / leading-dims match buffer lengths.
    unsafe {
        drv.blas
            .gemm(cfg_qk, &k_dev, &q_dev, &mut logits_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("gemm QK^T: {e:?}")))?;
    }

    // ── 2. softmax(logits, scale, mask) in place ────────────────────
    softmax_inplace(drv, &mut logits_dev, n_q, n_kv, scale, opts)?;

    // ── 3. out = attn @ V  [n_q × head_dim] row-major ───────────────
    // Same row-major identity:
    //   row-major attn (n_q, n_kv)  ≡ col-major (n_kv, n_q)
    //   row-major V    (n_kv, head_dim) ≡ col-major (head_dim, n_kv)
    //   want col-major out (head_dim, n_q) = V^T_cm × attn_cm
    // cuBLAS: transa=N, transb=N, M=head_dim, N=n_q, K=n_kv,
    //         lda=head_dim, ldb=n_kv, ldc=head_dim.
    let mut out_dev = drv.device_alloc(n_q * head_dim)?;
    let cfg_av = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: head_dim as i32,
        n: n_q as i32,
        k: n_kv as i32,
        alpha: 1.0_f32,
        lda: head_dim as i32,
        ldb: n_kv as i32,
        beta: 0.0_f32,
        ldc: head_dim as i32,
    };
    unsafe {
        drv.blas
            .gemm(cfg_av, &v_dev, &logits_dev, &mut out_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("gemm attn@V: {e:?}")))?;
    }

    drv.sync()?;
    drv.to_host(&out_dev)
}
