//! CUDA fused decode-time attention.
//!
//! Phase `cuda-fused-attention`: correctness-first fused helpers for
//! the attention path. The legacy `decode_attention` helper is kept as
//! a cuBLAS + softmax reference path; the fused helpers below cover
//! RMSNorm+QKV projection and decode-time QK norm + RoPE + KV append +
//! softmax + V aggregation in a single custom launch.

use std::sync::OnceLock;

use cudarc::cublas::{
    sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
    Gemm, GemmConfig,
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda-oxide")]
use cudarc::nvrtc::Ptx;

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

const QKV_RMS_PROJ_SRC: &str = r#"
extern "C" __global__ void qkv_rms_proj_f32(
    const float* x,
    const float* norm_w,
    const float* wq,
    const float* wk,
    const float* wv,
    float* q,
    float* k,
    float* v,
    int hidden,
    int q_dim,
    int kv_dim,
    float eps,
    float norm_offset
) {
    int row = blockIdx.x;
    int total_rows = q_dim + kv_dim + kv_dim;
    if (row >= total_rows) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];

    float ss = 0.f;
    for (int d = tid; d < hidden; d += bdim) {
        float xv = x[d];
        ss += xv * xv;
    }
    smem[tid] = ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf(smem[0] / (float)hidden + eps);

    const float* w;
    float* out;
    int local_row;
    if (row < q_dim) {
        w = wq;
        out = q;
        local_row = row;
    } else if (row < q_dim + kv_dim) {
        w = wk;
        out = k;
        local_row = row - q_dim;
    } else {
        w = wv;
        out = v;
        local_row = row - q_dim - kv_dim;
    }

    float acc = 0.f;
    const float* w_row = w + (size_t)local_row * hidden;
    for (int d = tid; d < hidden; d += bdim) {
        float xn = x[d] * inv_rms * (norm_w[d] + norm_offset);
        acc += xn * w_row[d];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[local_row] = smem[0];
}
"#;

const FUSED_DECODE_ATTN_SRC: &str = r#"
#define NEG_INF (__int_as_float(0xff800000))

extern "C" __global__ void fused_decode_attention_f32(
    const float* q,
    const float* k_new,
    const float* v_new,
    float* k_cache,
    float* v_cache,
    const float* q_norm,
    const float* k_norm,
    float* out,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int pos,
    int max_seq,
    int rotary_dim,
    float rope_base,
    float eps,
    float qk_norm_offset,
    float attn_scale,
    float softcap,
    int use_qk_norm
) {
    int qh = blockIdx.x;
    if (qh >= num_q_heads || pos >= max_seq) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];
    float* scores = smem;
    float* scratch = smem + max_seq;
    // Pre-rotated Q vector. cuda-attn-rope-hoist Phase 1: computed
    // once per attn call (depends only on `pos`, not `j`), then read
    // by every iteration of the score loop. Saves ~n_ctx redundant
    // (cosf, sinf, powf) triples per (head, d).
    float* q_rot = smem + max_seq + bdim;

    int group = max(1, num_q_heads / max(1, num_kv_heads));
    int kvh = min(num_kv_heads - 1, qh / group);
    const float* q_head = q + (size_t)qh * head_dim;
    const float* k_head = k_new + (size_t)kvh * head_dim;
    const float* v_head = v_new + (size_t)kvh * head_dim;

    float q_ss = 0.f;
    float k_ss = 0.f;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        float kv = k_head[d];
        q_ss += qv * qv;
        k_ss += kv * kv;
    }
    scratch[tid] = q_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float q_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    scratch[tid] = k_ss;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float k_inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    // ── Pre-rotate Q once per attention call ────────────────────────
    // Q's RoPE rotation depends only on `pos`, not on `j`. The score
    // loop below previously recomputed it per `(j, d)` — with
    // n_ctx ≈ 25 active threads and head_dim = 256 that's ~6 400
    // redundant trig triples per Q head per layer call. Hoist:
    int rdim_pre = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
    int hdim_pre = rdim_pre / 2;
    for (int d = tid; d < head_dim; d += bdim) {
        float qv = q_head[d];
        if (use_qk_norm) qv *= q_inv * (q_norm[d] + qk_norm_offset);
        if (d < rdim_pre) {
            int pair = d % hdim_pre;
            bool imag = d >= hdim_pre;
            float re = q_head[pair];
            float im = q_head[pair + hdim_pre];
            if (use_qk_norm) {
                re *= q_inv * (q_norm[pair]            + qk_norm_offset);
                im *= q_inv * (q_norm[pair + hdim_pre] + qk_norm_offset);
            }
            float freq  = 1.0f / powf(rope_base, (float)(2 * pair) / (float)rdim_pre);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            qv = imag ? (re * s + im * c) : (re * c - im * s);
        }
        q_rot[d] = qv;
    }
    __syncthreads();

    // Append the current K/V once per KV head. Other Q heads sharing
    // the same KV head compute against k_new/v_new directly for pos.
    if ((qh % group) == 0) {
        for (int d = tid; d < head_dim; d += bdim) {
            float kv = k_head[d];
            if (use_qk_norm) kv *= k_inv * (k_norm[d] + qk_norm_offset);
            // Compute the rotated current key directly so the append
            // path does not need an extra per-block scratch vector.
            float k_rot;
            int rdim = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
            int hdim = rdim / 2;
            if (d < rdim) {
                int pair = d % hdim;
                bool imag = d >= hdim;
                float re = k_head[pair];
                float im = k_head[pair + hdim];
                if (use_qk_norm) {
                    re *= k_inv * (k_norm[pair] + qk_norm_offset);
                    im *= k_inv * (k_norm[pair + hdim] + qk_norm_offset);
                }
                float freq = 1.0f / powf(rope_base, (float)(2 * pair) / (float)rdim);
                float angle = (float)pos * freq;
                float c = cosf(angle);
                float s = sinf(angle);
                k_rot = imag ? (re * s + im * c) : (re * c - im * s);
            } else {
                k_rot = kv;
            }
            size_t idx = ((size_t)pos * num_kv_heads + kvh) * head_dim + d;
            k_cache[idx] = k_rot;
            v_cache[idx] = v_head[d];
        }
    }
    __syncthreads();

    int n_ctx = pos + 1;
    for (int j = tid; j < n_ctx; j += bdim) {
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++) {
            // Q is pre-rotated above (cuda-attn-rope-hoist).
            float qv = q_rot[d];

            float kv;
            if (j == pos) {
                int rdim = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
                if (d < rdim) {
                    int hdim = rdim / 2;
                    int pair = d % hdim;
                    bool imag = d >= hdim;
                    float re = k_head[pair];
                    float im = k_head[pair + hdim];
                    if (use_qk_norm) {
                        re *= k_inv * (k_norm[pair] + qk_norm_offset);
                        im *= k_inv * (k_norm[pair + hdim] + qk_norm_offset);
                    }
                    float freq = 1.0f / powf(rope_base, (float)(2 * pair) / (float)rdim);
                    float angle = (float)pos * freq;
                    float c = cosf(angle);
                    float s = sinf(angle);
                    kv = imag ? (re * s + im * c) : (re * c - im * s);
                } else {
                    kv = k_head[d];
                    if (use_qk_norm) kv *= k_inv * (k_norm[d] + qk_norm_offset);
                }
            } else {
                kv = k_cache[((size_t)j * num_kv_heads + kvh) * head_dim + d];
            }
            dot += qv * kv;
        }
        float logit = dot * attn_scale;
        if (softcap > 0.f) logit = softcap * tanhf(logit / softcap);
        scores[j] = logit;
    }
    for (int j = tid + n_ctx; j < max_seq; j += bdim) {
        scores[j] = NEG_INF;
    }
    __syncthreads();

    float my_max = NEG_INF;
    for (int j = tid; j < n_ctx; j += bdim) {
        float s = scores[j];
        if (s > my_max) my_max = s;
    }
    scratch[tid] = my_max;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        __syncthreads();
    }
    float row_max = scratch[0];

    float my_sum = 0.f;
    for (int j = tid; j < n_ctx; j += bdim) {
        float e = expf(scores[j] - row_max);
        scores[j] = e;
        my_sum += e;
    }
    scratch[tid] = my_sum;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.f / scratch[0];

    for (int d = tid; d < head_dim; d += bdim) {
        float acc = 0.f;
        for (int j = 0; j < n_ctx; j++) {
            float prob = scores[j] * inv_sum;
            float vv = (j == pos)
                ? v_head[d]
                : v_cache[((size_t)j * num_kv_heads + kvh) * head_dim + d];
            acc += prob * vv;
        }
        out[(size_t)qh * head_dim + d] = acc;
    }
}
"#;

/// Lazily-loaded softmax module + function. cudarc's `CudaContext` is
/// `Send + Sync`; `OnceLock` gives us thread-safe one-time init.
static SOFTMAX_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static QKV_RMS_PROJ_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();
static FUSED_DECODE_ATTN_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> =
    OnceLock::new();

fn softmax_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = SOFTMAX_FUNC.get() {
        return Ok(f);
    }
    #[cfg(feature = "cuda-oxide")]
    let module = {
        let ptx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("larql_compute.ptx");
        drv.ctx
            .load_module(Ptx::from_file(ptx))
            .map_err(|e| CudaInitError::DriverMissing(format!("load cuda-oxide softmax: {e:?}")))?
    };
    #[cfg(feature = "cuda-oxide")]
    let func = module.load_function("scaled_softmax_oxide").map_err(|e| {
        CudaInitError::DriverMissing(format!("load cuda-oxide softmax function: {e:?}"))
    })?;
    #[cfg(not(feature = "cuda-oxide"))]
    let (module, func) = {
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
        (module, func)
    };
    let _ = SOFTMAX_FUNC.set((module, func));
    let (_, f) = SOFTMAX_FUNC.get().unwrap();
    Ok(f)
}

fn qkv_rms_proj_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = QKV_RMS_PROJ_FUNC.get() {
        return Ok(f);
    }
    let ptx = compile_ptx(QKV_RMS_PROJ_SRC)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile qkv_rms_proj: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load qkv module: {e:?}")))?;
    let func = module
        .load_function("qkv_rms_proj_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load qkv function: {e:?}")))?;
    let _ = QKV_RMS_PROJ_FUNC.set((module, func));
    let (_, f) = QKV_RMS_PROJ_FUNC.get().unwrap();
    Ok(f)
}

fn fused_decode_attention_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = FUSED_DECODE_ATTN_FUNC.get() {
        return Ok(f);
    }
    let ptx = compile_ptx(FUSED_DECODE_ATTN_SRC).map_err(|e| {
        CudaInitError::DriverMissing(format!("nvrtc compile fused_decode_attention: {e:?}"))
    })?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load fused attention module: {e:?}")))?;
    let func = module
        .load_function("fused_decode_attention_f32")
        .map_err(|e| {
            CudaInitError::DriverMissing(format!("load fused attention function: {e:?}"))
        })?;
    let _ = FUSED_DECODE_ATTN_FUNC.set((module, func));
    let (_, f) = FUSED_DECODE_ATTN_FUNC.get().unwrap();
    Ok(f)
}

/// Optional per-call attention knobs. The kernel folds these in.
#[derive(Clone, Copy, Debug)]
pub struct AttentionOpts {
    pub causal: bool,
    pub softcap: f32, // 0.0 → no softcap
}

#[derive(Clone, Copy, Debug)]
pub struct QkvProjDims {
    pub hidden: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
}

#[derive(Clone, Debug)]
pub struct QkvProjOutput {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct FusedDecodeAttentionOpts {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub pos: usize,
    pub max_seq: usize,
    pub rotary_dim: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub qk_norm_offset: f32,
    pub attn_scale: f32,
    pub softcap: f32,
}

#[derive(Clone, Debug)]
pub struct FusedDecodeAttentionOutput {
    pub out: Vec<f32>,
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
}

/// Device-resident output of [`fused_decode_attention_device`].
/// `cuda-decode-device-resident` Phase 1.
///
/// `out` stays on the GPU so the next projection (wo) can run
/// without an extra round trip. `k_cache` / `v_cache` come back to
/// the host because Phase 1 still stores the KV cache as
/// `Vec<f32>`. Phase 3 swaps those for `CudaSlice<f32>` and drops
/// the dtoh.
pub struct FusedDecodeAttentionDeviceOutput {
    pub out: CudaSlice<f32>,
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn qkv_rms_proj(
    backend: &CudaBackend,
    x: &[f32],
    norm_weight: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    dims: QkvProjDims,
    eps: f32,
    norm_offset: f32,
) -> Result<QkvProjOutput, CudaInitError> {
    assert_eq!(x.len(), dims.hidden);
    assert_eq!(norm_weight.len(), dims.hidden);
    assert_eq!(wq.len(), dims.q_dim * dims.hidden);
    assert_eq!(wk.len(), dims.kv_dim * dims.hidden);
    assert_eq!(wv.len(), dims.kv_dim * dims.hidden);

    let drv = backend.driver();
    let func = qkv_rms_proj_function(drv)?;
    let x_dev = drv.device_buf_from(x)?;
    let norm_dev = drv.device_buf_from(norm_weight)?;
    let wq_dev = drv.device_buf_from(wq)?;
    let wk_dev = drv.device_buf_from(wk)?;
    let wv_dev = drv.device_buf_from(wv)?;
    let mut q_dev = drv.device_alloc_uninit(dims.q_dim)?;
    let mut k_dev = drv.device_alloc_uninit(dims.kv_dim)?;
    let mut v_dev = drv.device_alloc_uninit(dims.kv_dim)?;

    let block_dim: u32 = 256;
    let total_rows = dims.q_dim + dims.kv_dim + dims.kv_dim;
    let cfg = LaunchConfig {
        grid_dim: (total_rows as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: block_dim * std::mem::size_of::<f32>() as u32,
    };
    let hidden_i = dims.hidden as i32;
    let q_dim_i = dims.q_dim as i32;
    let kv_dim_i = dims.kv_dim as i32;

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(&x_dev)
            .arg(&norm_dev)
            .arg(&wq_dev)
            .arg(&wk_dev)
            .arg(&wv_dev)
            .arg(&mut q_dev)
            .arg(&mut k_dev)
            .arg(&mut v_dev)
            .arg(&hidden_i)
            .arg(&q_dim_i)
            .arg(&kv_dim_i)
            .arg(&eps)
            .arg(&norm_offset)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch qkv_rms_proj: {e:?}")))?;
    }

    drv.sync()?;
    Ok(QkvProjOutput {
        q: drv.to_host(&q_dev)?,
        k: drv.to_host(&k_dev)?,
        v: drv.to_host(&v_dev)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn fused_decode_attention(
    backend: &CudaBackend,
    q: &[f32],
    k_new: &[f32],
    v_new: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    opts: FusedDecodeAttentionOpts,
) -> Result<FusedDecodeAttentionOutput, CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    assert_eq!(q.len(), q_dim);
    assert_eq!(k_new.len(), kv_dim);
    assert_eq!(v_new.len(), kv_dim);
    assert_eq!(k_cache.len(), cache_len);
    assert_eq!(v_cache.len(), cache_len);
    assert!(opts.pos < opts.max_seq);

    let use_qk_norm = q_norm.is_some() && k_norm.is_some();
    let q_norm_owned;
    let k_norm_owned;
    let q_norm = match q_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            q_norm_owned = vec![0.0_f32; opts.head_dim];
            &q_norm_owned
        }
    };
    let k_norm = match k_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            k_norm_owned = vec![0.0_f32; opts.head_dim];
            &k_norm_owned
        }
    };

    let drv = backend.driver();
    let func = fused_decode_attention_function(drv)?;
    let q_dev = drv.device_buf_from(q)?;
    let k_new_dev = drv.device_buf_from(k_new)?;
    let v_new_dev = drv.device_buf_from(v_new)?;
    let mut k_cache_dev = drv.device_buf_from(k_cache)?;
    let mut v_cache_dev = drv.device_buf_from(v_cache)?;
    let q_norm_dev = drv.device_buf_from(q_norm)?;
    let k_norm_dev = drv.device_buf_from(k_norm)?;
    let mut out_dev = drv.device_alloc_uninit(q_dim)?;

    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: ((opts.max_seq + block_dim as usize + opts.head_dim)
            * std::mem::size_of::<f32>()) as u32,
    };
    let num_q_heads_i = opts.num_q_heads as i32;
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    let pos_i = opts.pos as i32;
    let max_seq_i = opts.max_seq as i32;
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(&q_dev)
            .arg(&k_new_dev)
            .arg(&v_new_dev)
            .arg(&mut k_cache_dev)
            .arg(&mut v_cache_dev)
            .arg(&q_norm_dev)
            .arg(&k_norm_dev)
            .arg(&mut out_dev)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(&pos_i)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&opts.attn_scale)
            .arg(&opts.softcap)
            .arg(&use_qk_norm_i)
            .launch(cfg)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!("launch fused_decode_attention: {e:?}"))
            })?;
    }

    drv.sync()?;
    Ok(FusedDecodeAttentionOutput {
        out: drv.to_host(&out_dev)?,
        k_cache: drv.to_host(&k_cache_dev)?,
        v_cache: drv.to_host(&v_cache_dev)?,
    })
}

/// `cuda-decode-device-resident` Phase 3: full device-resident
/// fused decode attention. Q/K-new/V-new are `CudaSlice<f32>`
/// (Phase 1) **and** the K/V cache is now `&mut CudaSlice<f32>`
/// — the kernel reads prior tokens from it and writes the new row
/// in place at `pos`, with zero PCIe traffic. Returns just the
/// attention output as a device-resident slice.
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_attention_device_kv(
    backend: &CudaBackend,
    q_dev: &CudaSlice<f32>,
    k_new_dev: &CudaSlice<f32>,
    v_new_dev: &CudaSlice<f32>,
    k_cache_dev: &mut CudaSlice<f32>,
    v_cache_dev: &mut CudaSlice<f32>,
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    opts: FusedDecodeAttentionOpts,
) -> Result<CudaSlice<f32>, CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    assert_eq!(q_dev.len(), q_dim);
    assert_eq!(k_new_dev.len(), kv_dim);
    assert_eq!(v_new_dev.len(), kv_dim);
    assert_eq!(k_cache_dev.len(), cache_len);
    assert_eq!(v_cache_dev.len(), cache_len);
    assert!(opts.pos < opts.max_seq);

    let use_qk_norm = q_norm.is_some() && k_norm.is_some();
    let q_norm_owned;
    let k_norm_owned;
    let q_norm = match q_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            q_norm_owned = vec![0.0_f32; opts.head_dim];
            &q_norm_owned
        }
    };
    let k_norm = match k_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            k_norm_owned = vec![0.0_f32; opts.head_dim];
            &k_norm_owned
        }
    };

    let drv = backend.driver();
    let func = fused_decode_attention_function(drv)?;
    let q_norm_dev = drv.device_buf_from(q_norm)?;
    let k_norm_dev = drv.device_buf_from(k_norm)?;
    let mut out_dev = drv.device_alloc_uninit(q_dim)?;

    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: ((opts.max_seq + block_dim as usize + opts.head_dim)
            * std::mem::size_of::<f32>()) as u32,
    };
    let num_q_heads_i = opts.num_q_heads as i32;
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    let pos_i = opts.pos as i32;
    let max_seq_i = opts.max_seq as i32;
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(q_dev)
            .arg(k_new_dev)
            .arg(v_new_dev)
            .arg(k_cache_dev)
            .arg(v_cache_dev)
            .arg(&q_norm_dev)
            .arg(&k_norm_dev)
            .arg(&mut out_dev)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(&pos_i)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&opts.attn_scale)
            .arg(&opts.softcap)
            .arg(&use_qk_norm_i)
            .launch(cfg)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!(
                    "launch fused_decode_attention_device_kv: {e:?}"
                ))
            })?;
    }

    // Phase 3: no sync, no dtoh of K/V cache slabs. The kernel
    // wrote the new row into k_cache_dev/v_cache_dev at `pos`;
    // those buffers are persistent across calls so subsequent
    // tokens read them without any PCIe traffic.
    Ok(out_dev)
}

/// Device-resident variant of [`fused_decode_attention`]. Q / K-new /
/// V-new come in as `CudaSlice<f32>` (already produced by
/// `q4k_matvec_device` etc.) and the attention output stays on the
/// GPU. Phase 1 still pulls the K/V cache back to host because
/// `CudaKvCache` is `Vec<f32>`-backed; that goes away in Phase 3.
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_attention_device(
    backend: &CudaBackend,
    q_dev: &CudaSlice<f32>,
    k_new_dev: &CudaSlice<f32>,
    v_new_dev: &CudaSlice<f32>,
    k_cache: &[f32],
    v_cache: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    opts: FusedDecodeAttentionOpts,
) -> Result<FusedDecodeAttentionDeviceOutput, CudaInitError> {
    let q_dim = opts.num_q_heads * opts.head_dim;
    let kv_dim = opts.num_kv_heads * opts.head_dim;
    let cache_len = opts.max_seq * opts.num_kv_heads * opts.head_dim;
    assert_eq!(q_dev.len(), q_dim);
    assert_eq!(k_new_dev.len(), kv_dim);
    assert_eq!(v_new_dev.len(), kv_dim);
    assert_eq!(k_cache.len(), cache_len);
    assert_eq!(v_cache.len(), cache_len);
    assert!(opts.pos < opts.max_seq);

    let use_qk_norm = q_norm.is_some() && k_norm.is_some();
    let q_norm_owned;
    let k_norm_owned;
    let q_norm = match q_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            q_norm_owned = vec![0.0_f32; opts.head_dim];
            &q_norm_owned
        }
    };
    let k_norm = match k_norm {
        Some(w) => {
            assert_eq!(w.len(), opts.head_dim);
            w
        }
        None => {
            k_norm_owned = vec![0.0_f32; opts.head_dim];
            &k_norm_owned
        }
    };

    let drv = backend.driver();
    let func = fused_decode_attention_function(drv)?;
    let mut k_cache_dev = drv.device_buf_from(k_cache)?;
    let mut v_cache_dev = drv.device_buf_from(v_cache)?;
    let q_norm_dev = drv.device_buf_from(q_norm)?;
    let k_norm_dev = drv.device_buf_from(k_norm)?;
    let mut out_dev = drv.device_alloc_uninit(q_dim)?;

    let block_dim: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: ((opts.max_seq + block_dim as usize + opts.head_dim)
            * std::mem::size_of::<f32>()) as u32,
    };
    let num_q_heads_i = opts.num_q_heads as i32;
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    let pos_i = opts.pos as i32;
    let max_seq_i = opts.max_seq as i32;
    let rotary_dim_i = opts.rotary_dim as i32;
    let use_qk_norm_i = if use_qk_norm { 1_i32 } else { 0_i32 };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(q_dev)
            .arg(k_new_dev)
            .arg(v_new_dev)
            .arg(&mut k_cache_dev)
            .arg(&mut v_cache_dev)
            .arg(&q_norm_dev)
            .arg(&k_norm_dev)
            .arg(&mut out_dev)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(&pos_i)
            .arg(&max_seq_i)
            .arg(&rotary_dim_i)
            .arg(&opts.rope_base)
            .arg(&opts.eps)
            .arg(&opts.qk_norm_offset)
            .arg(&opts.attn_scale)
            .arg(&opts.softcap)
            .arg(&use_qk_norm_i)
            .launch(cfg)
            .map_err(|e| {
                CudaInitError::DriverMissing(format!("launch fused_decode_attention_device: {e:?}"))
            })?;
    }

    drv.sync()?;
    let k_cache_host = drv.to_host(&k_cache_dev)?;
    let v_cache_host = drv.to_host(&v_cache_dev)?;
    Ok(FusedDecodeAttentionDeviceOutput {
        out: out_dev,
        k_cache: k_cache_host,
        v_cache: v_cache_host,
    })
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
    #[cfg(not(feature = "cuda-oxide"))]
    // 1024 threads = max blockDim on every supported arch.
    let block_dim: u32 = 1024;
    #[cfg(feature = "cuda-oxide")]
    // cuda-oxide softmax is a correctness-first row-serial pilot kernel.
    let block_dim: u32 = 1;
    let grid_dim: u32 = n_rows as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        #[cfg(not(feature = "cuda-oxide"))]
        shared_mem_bytes: (block_dim as usize * std::mem::size_of::<f32>()) as u32,
        #[cfg(feature = "cuda-oxide")]
        shared_mem_bytes: 0,
    };
    let n_rows_i = n_rows as i32;
    let n_cols_i = n_cols as i32;
    let causal_i: i32 = if opts.causal { 1 } else { 0 };
    let softcap_f = opts.softcap;
    #[cfg(feature = "cuda-oxide")]
    let len = x_dev.len();
    // SAFETY: The kernel writes at most `n_rows * n_cols` f32 values
    // starting at the slice base; the buffer length matches the shape
    // (caller guarantees).
    unsafe {
        let mut builder = drv.stream.launch_builder(func);
        builder.arg(x_dev);
        #[cfg(feature = "cuda-oxide")]
        builder.arg(&len);
        builder
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

    let mut logits_dev = drv.device_alloc_uninit(n_q * n_kv)?;
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
    let mut out_dev = drv.device_alloc_uninit(n_q * head_dim)?;
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
