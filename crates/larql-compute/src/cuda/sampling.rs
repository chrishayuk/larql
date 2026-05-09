//! CUDA `verify_tree` kernel — phase 3 of `cuda-speculative-decoding`.
//!
//! Mirrors the CPU oracle in
//! `larql_inference::speculative::verify::verify_tree`: given target
//! probability rows for a most-likely root-to-leaf path through a
//! draft tree, plus the path's draft IDs and per-step `p_draft`,
//! emits the accepted span via the rejection-sampling rule from
//! Leviathan et al. 2022.
//!
//! Phase 3 acceptance criterion (per `cuda-speculative-decoding`
//! design.md §3.2): GPU output SHALL match the CPU oracle on
//! token-ID equality across 64 fixed RNG seeds.
//!
//! This is the **single-thread correctness-first** version. A
//! parallel-reduction follow-up lands once parity is locked.

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

const VERIFY_TREE_SRC: &str = r#"
// SplitMix64 — bit-identical to larql_inference::speculative::verify::VerifyRng.
__device__ float larql_splitmix64_next_f32(unsigned long long *state) {
    *state += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    unsigned int top24 = (unsigned int)(z >> 40);
    return ((float)top24) / 16777216.0f;
}

extern "C" __global__ void verify_tree_kernel(
    const float *p_target,       // [path_len, vocab] row-major
    const int *path_drafts,      // [path_len]
    const float *path_pdraft,    // [path_len]
    int path_len,
    int vocab,
    unsigned long long seed,
    int *accepted_out,           // [path_len]; -1 sentinel
    int *corrected_out,          // [1]
    int *bonus_out               // [1]
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned long long state = seed;

    // Reset outputs to -1 sentinel.
    for (int k = 0; k < path_len; k++) accepted_out[k] = -1;
    *corrected_out = -1;
    *bonus_out = -1;

    // Walk path.
    for (int k = 0; k < path_len; k++) {
        const float *p_row = p_target + (long long)k * (long long)vocab;
        int draft_id = path_drafts[k];
        float pd = path_pdraft[k];
        // Mirror Rust: p_draft.max(f32::MIN_POSITIVE) ≈ 1.175494e-38.
        if (pd < 1.175494351e-38f) pd = 1.175494351e-38f;
        float pt = p_row[draft_id];
        float ratio = pt / pd;
        float accept_prob = ratio < 1.0f ? ratio : 1.0f;

        float u = larql_splitmix64_next_f32(&state);
        if (u < accept_prob) {
            accepted_out[k] = draft_id;
            continue;
        }

        // Rejected at k: sample corrected from residual.
        // residual[i] = max(0, p_row[i] - (i == draft_id ? min(pd, pt) : 0))
        float subtract = pd < pt ? pd : pt;
        float z_sum = 0.0f;
        for (int i = 0; i < vocab; i++) {
            float v = p_row[i];
            if (i == draft_id) v -= subtract;
            if (v < 0.0f) v = 0.0f;
            z_sum += v;
        }

        int picked = vocab - 1;
        if (z_sum <= 0.0f) {
            // Residual collapsed; fall back to p_target.
            float total = 0.0f;
            for (int i = 0; i < vocab; i++) total += p_row[i];
            if (total <= 0.0f) {
                *corrected_out = 0;
                return;
            }
            float u2 = larql_splitmix64_next_f32(&state) * total;
            float acc = 0.0f;
            for (int i = 0; i < vocab; i++) {
                acc += p_row[i];
                if (u2 < acc) { picked = i; break; }
            }
        } else {
            float u2 = larql_splitmix64_next_f32(&state) * z_sum;
            float acc = 0.0f;
            for (int i = 0; i < vocab; i++) {
                float v = p_row[i];
                if (i == draft_id) v -= subtract;
                if (v < 0.0f) v = 0.0f;
                acc += v;
                if (u2 < acc) { picked = i; break; }
            }
        }
        *corrected_out = picked;
        return;
    }

    // All accepted: bonus from p_target at deepest accepted position.
    const float *p_last = p_target + (long long)(path_len - 1) * (long long)vocab;
    float total = 0.0f;
    for (int i = 0; i < vocab; i++) total += p_last[i];
    if (total <= 0.0f) {
        *bonus_out = 0;
        return;
    }
    float u3 = larql_splitmix64_next_f32(&state) * total;
    float acc = 0.0f;
    int picked = vocab - 1;
    for (int i = 0; i < vocab; i++) {
        acc += p_last[i];
        if (u3 < acc) { picked = i; break; }
    }
    *bonus_out = picked;
}

// Parallel variant. One block, blockDim.x = 1024 threads.
// Reduces sums in shared memory (binary tree), then thread 0 walks
// the inverse-CDF serially. Tradeoff: parallel sum order differs
// from CPU serial sum, so token IDs may differ from the serial
// kernel at boundary `u` values. Use the serial kernel for
// bit-exact parity tests; this kernel for production perf.
extern "C" __global__ void verify_tree_kernel_p(
    const float *p_target,
    const int *path_drafts,
    const float *path_pdraft,
    int path_len,
    int vocab,
    unsigned long long seed,
    int *accepted_out,
    int *corrected_out,
    int *bonus_out
) {
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    extern __shared__ float smem[];
    // smem[0..bdim] used for partial sums.

    __shared__ unsigned long long state_s;
    __shared__ int decision_s;
    __shared__ float pt_s;
    __shared__ float pd_s;
    __shared__ float zsum_s;
    __shared__ int rejected_at_s;

    if (tid == 0) {
        state_s = seed;
        rejected_at_s = -1;
        for (int k = 0; k < path_len; k++) accepted_out[k] = -1;
        *corrected_out = -1;
        *bonus_out = -1;
    }
    __syncthreads();

    int reject_k = -1;
    for (int k = 0; k < path_len; k++) {
        const float *p_row = p_target + (long long)k * (long long)vocab;
        if (tid == 0) {
            int draft_id = path_drafts[k];
            float pd = path_pdraft[k];
            if (pd < 1.175494351e-38f) pd = 1.175494351e-38f;
            float pt = p_row[draft_id];
            pt_s = pt;
            pd_s = pd;
            float ratio = pt / pd;
            float accept_prob = ratio < 1.0f ? ratio : 1.0f;
            float u = larql_splitmix64_next_f32(&state_s);
            decision_s = (u < accept_prob) ? 1 : 0;
            if (decision_s == 1) accepted_out[k] = draft_id;
        }
        __syncthreads();
        if (decision_s == 1) {
            continue;
        }
        // Rejected at k. Compute z_sum = sum_i max(0, p_row[i] - subtract_at_draft).
        int draft_id = path_drafts[k];
        float subtract = pd_s < pt_s ? pd_s : pt_s;
        float local = 0.0f;
        for (int i = tid; i < vocab; i += bdim) {
            float v = p_row[i];
            if (i == draft_id) v -= subtract;
            if (v < 0.0f) v = 0.0f;
            local += v;
        }
        smem[tid] = local;
        __syncthreads();
        for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            zsum_s = smem[0];
            rejected_at_s = k;
        }
        __syncthreads();
        reject_k = k;
        break;
    }

    if (rejected_at_s >= 0) {
        if (tid == 0) {
            const float *p_row = p_target + (long long)rejected_at_s * (long long)vocab;
            int draft_id = path_drafts[rejected_at_s];
            float subtract = pd_s < pt_s ? pd_s : pt_s;
            float zsum = zsum_s;
            int picked = vocab - 1;
            if (zsum <= 0.0f) {
                // Fall back to p_target (using already-computed sum if zsum == 0).
                float total = 0.0f;
                for (int i = 0; i < vocab; i++) total += p_row[i];
                if (total > 0.0f) {
                    float u2 = larql_splitmix64_next_f32(&state_s) * total;
                    float acc = 0.0f;
                    for (int i = 0; i < vocab; i++) {
                        acc += p_row[i];
                        if (u2 < acc) { picked = i; break; }
                    }
                } else {
                    picked = 0;
                }
            } else {
                float u2 = larql_splitmix64_next_f32(&state_s) * zsum;
                float acc = 0.0f;
                for (int i = 0; i < vocab; i++) {
                    float v = p_row[i];
                    if (i == draft_id) v -= subtract;
                    if (v < 0.0f) v = 0.0f;
                    acc += v;
                    if (u2 < acc) { picked = i; break; }
                }
            }
            *corrected_out = picked;
        }
        return;
    }

    // All accepted: bonus from p_target at deepest accepted position.
    const float *p_last = p_target + (long long)(path_len - 1) * (long long)vocab;
    float local_total = 0.0f;
    for (int i = tid; i < vocab; i += bdim) local_total += p_last[i];
    smem[tid] = local_total;
    __syncthreads();
    for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        float total = smem[0];
        if (total <= 0.0f) {
            *bonus_out = 0;
            return;
        }
        float u3 = larql_splitmix64_next_f32(&state_s) * total;
        float acc = 0.0f;
        int picked = vocab - 1;
        for (int i = 0; i < vocab; i++) {
            acc += p_last[i];
            if (u3 < acc) { picked = i; break; }
        }
        *bonus_out = picked;
    }
    (void)reject_k;
}
"#;

static VERIFY_TREE_MODULE: OnceLock<std::sync::Arc<CudaModule>> = OnceLock::new();
static VERIFY_TREE_FUNC: OnceLock<CudaFunction> = OnceLock::new();
static VERIFY_TREE_FUNC_P: OnceLock<CudaFunction> = OnceLock::new();

fn verify_tree_module(drv: &Driver) -> Result<&'static std::sync::Arc<CudaModule>, CudaInitError> {
    if let Some(m) = VERIFY_TREE_MODULE.get() {
        return Ok(m);
    }
    let ptx = compile_ptx(VERIFY_TREE_SRC)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc verify_tree: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load verify_tree: {e:?}")))?;
    let _ = VERIFY_TREE_MODULE.set(module);
    Ok(VERIFY_TREE_MODULE.get().unwrap())
}

fn verify_tree_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some(f) = VERIFY_TREE_FUNC.get() {
        return Ok(f);
    }
    let module = verify_tree_module(drv)?;
    let func = module
        .load_function("verify_tree_kernel")
        .map_err(|e| CudaInitError::DriverMissing(format!("get verify_tree_kernel: {e:?}")))?;
    let _ = VERIFY_TREE_FUNC.set(func);
    Ok(VERIFY_TREE_FUNC.get().unwrap())
}

fn verify_tree_function_p(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some(f) = VERIFY_TREE_FUNC_P.get() {
        return Ok(f);
    }
    let module = verify_tree_module(drv)?;
    let func = module
        .load_function("verify_tree_kernel_p")
        .map_err(|e| CudaInitError::DriverMissing(format!("get verify_tree_kernel_p: {e:?}")))?;
    let _ = VERIFY_TREE_FUNC_P.set(func);
    Ok(VERIFY_TREE_FUNC_P.get().unwrap())
}

/// Decoded `AcceptedSpan` returned by the GPU kernel. Mirrors
/// `larql_inference::speculative::verify::AcceptedSpan` but without
/// the optional Vec — sentinels `-1` mark unset positions.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuAcceptedSpan {
    pub accepted: Vec<i32>,
    pub corrected: i32,
    pub bonus: i32,
}

/// Run the GPU `verify_tree` kernel.
///
/// `p_target_rows[k]` is the target probability vector at the k-th
/// position of the most-likely path. `path_drafts[k]` is the
/// drafted token ID at that position; `path_pdraft[k]` is its
/// `p_draft`. `seed` is the SplitMix64 seed shared with the CPU
/// oracle for parity testing.
pub fn verify_tree_gpu(
    backend: &CudaBackend,
    p_target_rows: &[Vec<f32>],
    path_drafts: &[i32],
    path_pdraft: &[f32],
    vocab: usize,
    seed: u64,
) -> Result<GpuAcceptedSpan, CudaInitError> {
    let drv: &Driver = backend.driver();
    let path_len = p_target_rows.len();
    if path_len == 0 || path_drafts.len() != path_len || path_pdraft.len() != path_len {
        return Err(CudaInitError::DriverMissing(format!(
            "verify_tree_gpu shape mismatch: path_len={path_len} drafts={} pdraft={}",
            path_drafts.len(),
            path_pdraft.len()
        )));
    }
    for (i, row) in p_target_rows.iter().enumerate() {
        if row.len() != vocab {
            return Err(CudaInitError::DriverMissing(format!(
                "verify_tree_gpu row {i}: got vocab={}, expected {vocab}",
                row.len()
            )));
        }
    }

    // Flatten p_target rows into a single contiguous buffer.
    let mut p_target_flat = Vec::with_capacity(path_len * vocab);
    for row in p_target_rows {
        p_target_flat.extend_from_slice(row);
    }

    let func = verify_tree_function(drv)?;
    let p_target_dev = drv.device_buf_from(&p_target_flat)?;
    let path_drafts_dev = drv.device_i32_buf_from(path_drafts)?;
    let path_pdraft_dev = drv.device_buf_from(path_pdraft)?;
    let mut accepted_dev = drv.device_alloc_i32(path_len)?;
    let mut corrected_dev = drv.device_alloc_i32(1)?;
    let mut bonus_dev = drv.device_alloc_i32(1)?;

    let path_len_i = path_len as i32;
    let vocab_i = vocab as i32;
    let seed_u64 = seed;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(&p_target_dev)
            .arg(&path_drafts_dev)
            .arg(&path_pdraft_dev)
            .arg(&path_len_i)
            .arg(&vocab_i)
            .arg(&seed_u64)
            .arg(&mut accepted_dev)
            .arg(&mut corrected_dev)
            .arg(&mut bonus_dev)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch verify_tree: {e:?}")))?;
    }
    drv.sync()?;

    let accepted = drv.to_host_i32(&accepted_dev)?;
    let corrected = drv.to_host_i32(&corrected_dev)?[0];
    let bonus = drv.to_host_i32(&bonus_dev)?[0];

    Ok(GpuAcceptedSpan {
        accepted,
        corrected,
        bonus,
    })
}

/// Run the parallel `verify_tree_kernel_p`. Same inputs/outputs as
/// [`verify_tree_gpu`]; uses block-wide parallel sum reduction so
/// vocab=256k completes in ~80 µs instead of ~1.6 ms (single-thread).
///
/// **Caveat**: parallel reduction order differs from CPU serial sum,
/// so token IDs may diverge from the CPU oracle at boundary `u`
/// values. The rejection-sampling distribution is preserved
/// (statistical equivalence); use [`verify_tree_gpu`] for bit-exact
/// CPU parity tests.
pub fn verify_tree_gpu_parallel(
    backend: &CudaBackend,
    p_target_rows: &[Vec<f32>],
    path_drafts: &[i32],
    path_pdraft: &[f32],
    vocab: usize,
    seed: u64,
) -> Result<GpuAcceptedSpan, CudaInitError> {
    let drv: &Driver = backend.driver();
    let path_len = p_target_rows.len();
    if path_len == 0 || path_drafts.len() != path_len || path_pdraft.len() != path_len {
        return Err(CudaInitError::DriverMissing(format!(
            "verify_tree_gpu_parallel shape mismatch: path_len={path_len}"
        )));
    }
    for (i, row) in p_target_rows.iter().enumerate() {
        if row.len() != vocab {
            return Err(CudaInitError::DriverMissing(format!(
                "verify_tree_gpu_parallel row {i}: got vocab={}, expected {vocab}",
                row.len()
            )));
        }
    }

    let mut p_target_flat = Vec::with_capacity(path_len * vocab);
    for row in p_target_rows {
        p_target_flat.extend_from_slice(row);
    }

    let func = verify_tree_function_p(drv)?;
    let p_target_dev = drv.device_buf_from(&p_target_flat)?;
    let path_drafts_dev = drv.device_i32_buf_from(path_drafts)?;
    let path_pdraft_dev = drv.device_buf_from(path_pdraft)?;
    let mut accepted_dev = drv.device_alloc_i32(path_len)?;
    let mut corrected_dev = drv.device_alloc_i32(1)?;
    let mut bonus_dev = drv.device_alloc_i32(1)?;

    let path_len_i = path_len as i32;
    let vocab_i = vocab as i32;
    let seed_u64 = seed;

    const THREADS: u32 = 1024;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: THREADS * std::mem::size_of::<f32>() as u32,
    };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(&p_target_dev)
            .arg(&path_drafts_dev)
            .arg(&path_pdraft_dev)
            .arg(&path_len_i)
            .arg(&vocab_i)
            .arg(&seed_u64)
            .arg(&mut accepted_dev)
            .arg(&mut corrected_dev)
            .arg(&mut bonus_dev)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch verify_tree_p: {e:?}")))?;
    }
    drv.sync()?;

    let accepted = drv.to_host_i32(&accepted_dev)?;
    let corrected = drv.to_host_i32(&corrected_dev)?[0];
    let bonus = drv.to_host_i32(&bonus_dev)?[0];

    Ok(GpuAcceptedSpan {
        accepted,
        corrected,
        bonus,
    })
}
