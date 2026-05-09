//! Batched masked attention primitive — phase 3 task 3.1 of
//! `cuda-speculative-decoding`.
//!
//! Computes `Y = softmax(scale·Q·Kᵀ ⊙ mask) · V` for `q_tokens ≥ 1`
//! with a per-q bitmask over `[0, ctx_len)` attendable positions.
//! Supports GQA (`num_q_heads` ≥ `num_kv_heads`, group ratio = q/kv).
//!
//! Scope:
//!   - Inputs are **pre-prepared**: caller does RoPE, QK norm, and
//!     KV writes via the existing kernels in `cuda::attn`. This
//!     kernel does not touch the KV cache.
//!   - One block per (q_token, q_head) pair — natural parallelism
//!     across the speculative tree.
//!
//! CPU oracle: scalar attention loop in the test file. Token-level
//! parity within `1e-3` absolute (matches existing fused-attention
//! parity tolerance).

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

const TREE_ATTN_SRC: &str = r#"
#define NEG_INF (__int_as_float(0xff800000))

// One block per (q_token, q_head) pair.
// grid_dim = (num_q_heads, q_tokens, 1)
// block_dim = (THREADS, 1, 1)  — typical THREADS = 128
//
// Layout (all row-major, contiguous):
//   q          [q_tokens, num_q_heads, head_dim]            f32
//   k_full     [ctx_len,  num_kv_heads, head_dim]           f32
//   v_full     [ctx_len,  num_kv_heads, head_dim]           f32
//   mask_bits  [q_tokens, words_per_row]                    u32
//                where words_per_row = (ctx_len + 31) / 32;
//                bit j set ⇔ q-token may attend to ctx position j
//   out        [q_tokens, num_q_heads, head_dim]            f32
//
// Shared memory: ctx_len floats (scores) + bdim floats (reduce).
extern "C" __global__ void tree_decode_attention_f32(
    const float *q,
    const float *k_full,
    const float *v_full,
    const unsigned int *mask_bits,
    float *out,
    int q_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int ctx_len,
    int words_per_row,
    float attn_scale,
    float softcap
) {
    int qh = blockIdx.x;
    int qt = blockIdx.y;
    if (qh >= num_q_heads || qt >= q_tokens) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    extern __shared__ float smem[];
    float *scores  = smem;
    float *scratch = smem + ctx_len;

    int group = num_q_heads / (num_kv_heads > 0 ? num_kv_heads : 1);
    if (group < 1) group = 1;
    int kvh = qh / group;
    if (kvh >= num_kv_heads) kvh = num_kv_heads - 1;

    const float *q_vec = q + ((size_t)qt * num_q_heads + qh) * head_dim;
    const unsigned int *row_mask = mask_bits + (size_t)qt * words_per_row;

    // Compute scores[j] for j in [0, ctx_len). Masked positions get NEG_INF.
    for (int j = tid; j < ctx_len; j += bdim) {
        unsigned int w = row_mask[j >> 5];
        bool allowed = ((w >> (j & 31)) & 1u) != 0;
        if (!allowed) {
            scores[j] = NEG_INF;
            continue;
        }
        const float *k_vec = k_full + ((size_t)j * num_kv_heads + kvh) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_vec[d] * k_vec[d];
        }
        float s = dot * attn_scale;
        if (softcap > 0.f) {
            s = softcap * tanhf(s / softcap);
        }
        scores[j] = s;
    }
    __syncthreads();

    // Row max.
    float local_max = NEG_INF;
    for (int j = tid; j < ctx_len; j += bdim) {
        float v = scores[j];
        if (v > local_max) local_max = v;
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = scratch[tid];
            float b = scratch[tid + stride];
            scratch[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    float row_max = scratch[0];

    // Exp + sum.
    float local_sum = 0.f;
    for (int j = tid; j < ctx_len; j += bdim) {
        float v = scores[j];
        float e = (v == NEG_INF) ? 0.f : __expf(v - row_max);
        scores[j] = e;
        local_sum += e;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.f / scratch[0];

    // Out[d] = sum_j scores[j] * inv_sum * V[j, kvh, d].
    float *out_vec = out + ((size_t)qt * num_q_heads + qh) * head_dim;
    for (int d = tid; d < head_dim; d += bdim) {
        float acc = 0.f;
        for (int j = 0; j < ctx_len; j++) {
            float w = scores[j];
            if (w == 0.f) continue;
            const float *v_vec = v_full + ((size_t)j * num_kv_heads + kvh) * head_dim;
            acc += w * inv_sum * v_vec[d];
        }
        out_vec[d] = acc;
    }
}
"#;

static TREE_ATTN_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();

fn tree_attn_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = TREE_ATTN_FUNC.get() {
        return Ok(f);
    }
    let ptx = compile_ptx(TREE_ATTN_SRC)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc tree_attn: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load tree_attn: {e:?}")))?;
    let func = module
        .load_function("tree_decode_attention_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("get tree_attn fn: {e:?}")))?;
    let _ = TREE_ATTN_FUNC.set((module, func));
    Ok(&TREE_ATTN_FUNC.get().unwrap().1)
}

/// Per-call options. Keep it small — RoPE / norm / KV writes are
/// the caller's responsibility (use the existing fused kernels in
/// `cuda::attn`).
#[derive(Clone, Copy, Debug)]
pub struct TreeAttentionOpts {
    pub q_tokens: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ctx_len: usize,
    pub words_per_row: usize,
    pub attn_scale: f32,
    pub softcap: f32,
}

/// Run batched masked attention.
///
/// `q`: row-major `[q_tokens, num_q_heads, head_dim]`.
/// `k_full` / `v_full`: row-major `[ctx_len, num_kv_heads, head_dim]`.
/// `mask_bits`: row-major `[q_tokens, words_per_row]` u32 bitmask.
///
/// Returns row-major `[q_tokens, num_q_heads, head_dim]`.
pub fn tree_decode_attention(
    backend: &CudaBackend,
    q: &[f32],
    k_full: &[f32],
    v_full: &[f32],
    mask_bits: &[u32],
    opts: TreeAttentionOpts,
) -> Result<Vec<f32>, CudaInitError> {
    let q_len = opts.q_tokens * opts.num_q_heads * opts.head_dim;
    let kv_len = opts.ctx_len * opts.num_kv_heads * opts.head_dim;
    let mask_len = opts.q_tokens * opts.words_per_row;
    if q.len() != q_len {
        return Err(CudaInitError::DriverMissing(format!(
            "tree_decode_attention: q len {} != expected {q_len}",
            q.len()
        )));
    }
    if k_full.len() != kv_len || v_full.len() != kv_len {
        return Err(CudaInitError::DriverMissing(format!(
            "tree_decode_attention: kv len mismatch ({}, {}) vs {kv_len}",
            k_full.len(),
            v_full.len()
        )));
    }
    if mask_bits.len() != mask_len {
        return Err(CudaInitError::DriverMissing(format!(
            "tree_decode_attention: mask len {} != expected {mask_len}",
            mask_bits.len()
        )));
    }
    if opts.words_per_row != opts.ctx_len.div_ceil(32) {
        return Err(CudaInitError::DriverMissing(format!(
            "tree_decode_attention: words_per_row {} != ctx_len.div_ceil(32) {}",
            opts.words_per_row,
            opts.ctx_len.div_ceil(32)
        )));
    }

    let drv = backend.driver();
    let func = tree_attn_function(drv)?;
    let q_dev = drv.device_buf_from(q)?;
    let k_dev = drv.device_buf_from(k_full)?;
    let v_dev = drv.device_buf_from(v_full)?;
    let mask_dev = drv.device_u32_buf_from(mask_bits)?;
    let mut out_dev = drv.device_alloc(q_len)?;

    let q_tokens_i = opts.q_tokens as i32;
    let num_q_heads_i = opts.num_q_heads as i32;
    let num_kv_heads_i = opts.num_kv_heads as i32;
    let head_dim_i = opts.head_dim as i32;
    let ctx_len_i = opts.ctx_len as i32;
    let words_per_row_i = opts.words_per_row as i32;
    let scale = opts.attn_scale;
    let softcap = opts.softcap;

    const THREADS: u32 = 128;
    let cfg = LaunchConfig {
        grid_dim: (opts.num_q_heads as u32, opts.q_tokens as u32, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: ((opts.ctx_len as u32) + THREADS) * std::mem::size_of::<f32>() as u32,
    };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(&q_dev)
            .arg(&k_dev)
            .arg(&v_dev)
            .arg(&mask_dev)
            .arg(&mut out_dev)
            .arg(&q_tokens_i)
            .arg(&num_q_heads_i)
            .arg(&num_kv_heads_i)
            .arg(&head_dim_i)
            .arg(&ctx_len_i)
            .arg(&words_per_row_i)
            .arg(&scale)
            .arg(&softcap)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch tree_attn: {e:?}")))?;
    }
    drv.sync()?;
    drv.to_host(&out_dev)
}
