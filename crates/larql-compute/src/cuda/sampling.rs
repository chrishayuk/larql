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
"#;

static VERIFY_TREE_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();

fn verify_tree_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = VERIFY_TREE_FUNC.get() {
        return Ok(f);
    }
    let ptx = compile_ptx(VERIFY_TREE_SRC)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc verify_tree: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load verify_tree: {e:?}")))?;
    let func = module
        .load_function("verify_tree_kernel")
        .map_err(|e| CudaInitError::DriverMissing(format!("get verify_tree_kernel: {e:?}")))?;
    let _ = VERIFY_TREE_FUNC.set((module, func));
    Ok(&VERIFY_TREE_FUNC.get().unwrap().1)
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
