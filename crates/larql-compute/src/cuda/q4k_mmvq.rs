//! Q4_K × Q8_1 mmvq matvec — INT8 SIMD path via `__dp4a`.
//!
//! `cuda-decode-device-resident` and `cuda-q4k-device-cache` left the
//! Q4_K matvec on a custom f32 SIMT kernel (`q4k_direct.rs`). This
//! module ports the well-tuned llama.cpp pattern:
//!
//! 1. The input vector is quantised to **Q8_1** (`elem::quantize_q8_1_device`).
//! 2. The matvec kernel reads packed Q4_K weight bytes + Q8_1 input
//!    blocks and reduces with `__dp4a` (single-instruction 4-way INT8
//!    SIMD dot product → INT32 accumulator).
//! 3. The Q4_K super-block scale/min decode is folded into the final
//!    fp32 accumulator: `out = d * Σ(d8 · sc · dot1) - dmin * Σ(d8 · m · sum_q8)`.
//!
//! The kernel body is a close-to-verbatim port of
//! `vec_dot_q4_K_q8_1_impl_vmmq` from upstream
//! `ggml/src/ggml-cuda/vecdotq.cuh` (MIT-licensed, llama.cpp
//! authors).
//!
//! `cuda-q4k-mmvq-int8` Phase 2.

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::backend::CudaBackend;
use super::driver::Driver;
use super::elem::Q8_1Buf;
use super::error::CudaInitError;

const Q4K_BLOCK_ELEMS: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;
/// Threads per row. One full warp per output row; 16 lanes share a
/// super-block, the other 16 lanes work on the next super-block in
/// the same warp iteration (matching upstream's `blocks_per_iter = 2`).
const WARP_SIZE: u32 = 32;
/// CUDA blocks pack `ROWS_PER_BLOCK` warps so we get a `(WARP_SIZE,
/// ROWS_PER_BLOCK, 1)` blockDim. 4 keeps occupancy reasonable on
/// sm_89 without spilling shared.
const ROWS_PER_BLOCK: u32 = 4;

/// Q4_K × Q8_1 mmvq kernel. One CUDA block = `ROWS_PER_BLOCK` warps =
/// `ROWS_PER_BLOCK` output rows.
const Q4K_MMVQ_SRC: &str = r#"
// fp16 → f32 helper. NVRTC-friendly (no cuda_fp16.h required).
__device__ float larql_f16_to_f32(unsigned short h) {
    unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
    unsigned int mant = (unsigned int)h & 0x03ffu;
    int exp = ((int)h >> 10) & 0x1f;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0u) {
            bits = sign;
        } else {
            float v = (float)mant * 5.960464477539063e-8f;
            return (sign != 0u) ? -v : v;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((unsigned int)(exp + 112) << 23) | (mant << 13);
    }
    return __uint_as_float(bits);
}

// __dp4a is not exposed as a built-in by NVRTC without including
// cuda_fp16.h / device intrinsics; use inline PTX. dp4a.s32.s32
// computes `r = c + signed_dot4(a, b)` where a/b are INT32 holding
// 4 packed INT8 values each. Single SASS instruction on sm_61+;
// runs on dedicated INT pipelines at ~4× the rate of fp32 MAC.
__device__ int dp4a_s32(int a, int b, int c) {
    int r;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// Q4_K sub-block scale/min decode. `j` is sub-block index 0..7;
// `packed` is the 12-byte scale/min array at offset 4 of each
// 144-byte super-block. Matches the layout used in q4k_direct.rs.
__device__ unsigned char q4k_scale(const unsigned char* packed, int j) {
    if (j < 4) return packed[j] & 0x3fu;
    return (packed[j + 4] & 0x0fu) | ((packed[j - 4] >> 6) << 4);
}
__device__ unsigned char q4k_min(const unsigned char* packed, int j) {
    if (j < 4) return packed[j + 4] & 0x3fu;
    return (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
}

// Core dot product, lifted from llama.cpp's vec_dot_q4_K_q8_1_impl_vmmq
// (MIT, ggml authors). Computes the contribution to one output row
// from a slice of one Q4_K super-block at quant index `iqs` (in
// {0,2,...,30}) against the matching pair of Q8_1 input blocks.
__device__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const unsigned char* bq4_K,        // 144-byte Q4_K super-block
    const unsigned char* y_q8_1_base,  // start of Q8_1 input blocks
    int kby,                           // Q8_1 block index aligned with this super-block
    int iqs                            // 4-bit quant index within super-block: 0,2,...,30
) {
    // d, dmin: per-super-block fp16 → f32
    float d_f    = larql_f16_to_f32((unsigned short)bq4_K[0] | ((unsigned short)bq4_K[1] << 8));
    float dmin_f = larql_f16_to_f32((unsigned short)bq4_K[2] | ((unsigned short)bq4_K[3] << 8));

    const unsigned char* packed = bq4_K + 4;     // 12 bytes of packed scales/mins
    const unsigned char* qs     = bq4_K + 16;    // 128 bytes of 4-bit quants

    // bq8_offset: which pair of Q8_1 blocks this iqs slice consumes
    // (matches upstream: bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2))
    //  = 2 * ((iqs/2) / 4) ∈ {0,2,4,6}).
    int bq8_offset = 2 * ((iqs / 2) / 4);

    // Load 8 bytes of Q4_K quants from offset (16 * bq8_offset + 4 *
    // ((iqs/2) % 4)) and another 8 bytes 16 bytes ahead.
    const unsigned int* q4 =
        (const unsigned int*)(qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    int v0 = q4[0];   // 4 bytes = 8 nibbles
    int v1 = q4[4];   // 4 bytes from 16 ahead

    // Per-sub-block scales (sc) and mins (m) for sub-blocks
    // {bq8_offset, bq8_offset+1}.
    unsigned char sc0 = q4k_scale(packed, bq8_offset + 0);
    unsigned char sc1 = q4k_scale(packed, bq8_offset + 1);
    unsigned char m0  = q4k_min  (packed, bq8_offset + 0);
    unsigned char m1  = q4k_min  (packed, bq8_offset + 1);

    // Pull two Q8_1 blocks (each 36 bytes: half2 ds + 32 i8 qs).
    int u00, u01, u10, u11;
    float d8_0, d8_1;
    {
        const unsigned char* bq8 = y_q8_1_base + (size_t)(kby + bq8_offset + 0) * 36;
        d8_0 = larql_f16_to_f32(
            (unsigned short)bq8[0] | ((unsigned short)bq8[1] << 8)
        );
        const unsigned int* q8 = (const unsigned int*)(bq8 + 4);
        int chunk = (iqs / 2) % 4;
        u00 = q8[chunk];
        u01 = q8[chunk + 4];
    }
    {
        const unsigned char* bq8 = y_q8_1_base + (size_t)(kby + bq8_offset + 1) * 36;
        d8_1 = larql_f16_to_f32(
            (unsigned short)bq8[0] | ((unsigned short)bq8[1] << 8)
        );
        const unsigned int* q8 = (const unsigned int*)(bq8 + 4);
        int chunk = (iqs / 2) % 4;
        u10 = q8[chunk];
        u11 = q8[chunk + 4];
    }

    // QR4_K = 2 unrolled iterations: low/high nibbles of v[0..1].
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    // i=0: low nibbles of v0/v1, paired with bq8_offset Q8_1 block (u00/u01)
    {
        int v0i  = (v0 >> 0) & 0x0F0F0F0F;
        int v1i  = (v1 >> 0) & 0x0F0F0F0F;
        int dot1 = dp4a_s32(v1i, u01, dp4a_s32(v0i, u00, 0));
        int dot2 = dp4a_s32(0x01010101, u01, dp4a_s32(0x01010101, u00, 0));
        sumf_d += d8_0 * (dot1 * (float)sc0);
        sumf_m += d8_0 * (dot2 * (float)m0);
    }
    // i=1: high nibbles of v0/v1, paired with bq8_offset+1 Q8_1 block (u10/u11)
    {
        int v0i  = (v0 >> 4) & 0x0F0F0F0F;
        int v1i  = (v1 >> 4) & 0x0F0F0F0F;
        int dot1 = dp4a_s32(v1i, u11, dp4a_s32(v0i, u10, 0));
        int dot2 = dp4a_s32(0x01010101, u11, dp4a_s32(0x01010101, u10, 0));
        sumf_d += d8_1 * (dot1 * (float)sc1);
        sumf_m += d8_1 * (dot2 * (float)m1);
    }

    return d_f * sumf_d - dmin_f * sumf_m;
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_f32(
    const unsigned char* __restrict__ vbq,   // packed Q4_K weights
    const unsigned char* __restrict__ vy,    // packed block_q8_1[]
    float*               __restrict__ dst,   // output [rows]
    int rows,
    int n_super_blocks                       // hidden / 256
) {
    int tid_x = threadIdx.x;                 // 0..31 (lane within warp)
    int tid_y = threadIdx.y;                 // 0..ROWS_PER_BLOCK-1 (warp within block)
    int row   = blockIdx.x * blockDim.y + tid_y;
    if (row >= rows) return;

    // 16 lanes share a super-block; the other 16 lanes own the
    // adjacent super-block in the same warp iteration.
    int kbx_lane = tid_x / 16;               // 0 or 1
    int iqs      = 2 * (tid_x % 16);         // 0, 2, ..., 30

    const unsigned char* row_base =
        vbq + (size_t)row * (size_t)n_super_blocks * 144ull;

    float partial = 0.0f;
    // Each warp-iteration covers `blocks_per_iter = 2` super-blocks.
    for (int kbx_base = 0; kbx_base + kbx_lane < n_super_blocks; kbx_base += 2) {
        int kbx = kbx_base + kbx_lane;
        if (kbx >= n_super_blocks) break;

        // Each Q4_K super-block (256 weights) aligns with 8 Q8_1 blocks.
        int kby = kbx * 8;

        partial += vec_dot_q4_K_q8_1_impl_vmmq(
            row_base + (size_t)kbx * 144,
            vy,
            kby,
            iqs
        );
    }

    // Warp-reduce partial sums (32 lanes → 1).
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, o);
    }

    if (tid_x == 0) {
        dst[row] = partial;
    }
}
"#;

static Q4K_MMVQ_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();

fn q4k_mmvq_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = Q4K_MMVQ_FUNC.get() {
        return Ok(f);
    }
    // dp4a was introduced in sm_61 (Pascal). NVRTC's default target
    // (`compute_52`) doesn't include it; compile against compute_61
    // so the PTX is JIT'able on every supported card.
    // `LARQL_CUDA_Q4K_ARCH` overrides the NVRTC PTX target (e.g.
    // `compute_89` for sm_89-specific PTX). Default = compute_61,
    // which is forward-compatible with every supported card; the
    // driver JIT specialises to the actual compute capability at
    // runtime, so a higher arch only matters if the kernel uses
    // PTX intrinsics gated on it.
    let arch: Option<&'static str> = match std::env::var("LARQL_CUDA_Q4K_ARCH").ok().as_deref() {
        Some("compute_61") => Some("compute_61"),
        Some("compute_70") => Some("compute_70"),
        Some("compute_75") => Some("compute_75"),
        Some("compute_80") => Some("compute_80"),
        Some("compute_86") => Some("compute_86"),
        Some("compute_89") => Some("compute_89"),
        _ => Some("compute_61"),
    };
    let opts = CompileOptions {
        arch,
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(Q4K_MMVQ_SRC, opts)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile q4k_mmvq: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load q4k_mmvq module: {e:?}")))?;
    let func = module
        .load_function("mul_mat_vec_q4_K_q8_1_f32")
        .map_err(|e| CudaInitError::DriverMissing(format!("load q4k_mmvq function: {e:?}")))?;
    let _ = Q4K_MMVQ_FUNC.set((module, func));
    let (_, f) = Q4K_MMVQ_FUNC.get().unwrap();
    Ok(f)
}

/// Q4_K × Q8_1 device matvec. Both inputs are device-resident; output
/// is device-resident. The packed Q4_K weight bytes route through the
/// shared `with_q4k_device_buf` cache so first call uploads, later
/// calls reuse.
pub(crate) fn matvec_device(
    backend: &CudaBackend,
    q4k_data: &[u8],
    x_q8_1: &Q8_1Buf,
    rows: usize,
    hidden: usize,
) -> Result<CudaSlice<f32>, CudaInitError> {
    let drv = backend.driver();
    let mut y_dev = drv.device_alloc_uninit(rows)?;
    matvec_device_into(backend, q4k_data, x_q8_1, &mut y_dev, rows, hidden)?;
    Ok(y_dev)
}

/// `cuda-decode-cuda-graph`: variant of `matvec_device_into` that
/// takes a pre-resolved Q4_K weight buffer (already in the device
/// cache) instead of host bytes. Avoids the cache-lock acquisition
/// in the captured graph's hot loop.
pub(crate) fn matvec_device_into_with_dev(
    backend: &CudaBackend,
    q4k_dev: &CudaSlice<u8>,
    x_q8_1: &Q8_1Buf,
    y_dev: &mut CudaSlice<f32>,
    rows: usize,
    hidden: usize,
) -> Result<(), CudaInitError> {
    matvec_device_into_with_dev_tiled(
        backend,
        q4k_dev,
        x_q8_1,
        y_dev,
        rows,
        hidden,
        choose_rows_per_block(rows, hidden),
    )
}

/// `cuda-q4k-mmvq-down-tile`: shape-aware tile chooser. The kernel is
/// blockDim.y-agnostic, so different launches can pick different
/// `ROWS_PER_BLOCK` without recompiling. Empirically:
///
/// - Tall, narrow weights (`rows ≥ hidden`, e.g. gate/up at
///   10240 × 2560) want **larger tiles** — 4 is fine; the per-block
///   working set stays ≤ 6 KB and the L1 cache is comfortable.
/// - Short, wide weights (`rows < hidden`, e.g. proj_down at
///   2560 × 10240) need **smaller tiles** — at `rows_per_block = 4`
///   each block reads ~23 KB of weights, and with 5 blocks resident
///   per SM that's ~115 KB / 128 KB L1 — tight enough to evict and
///   thrash. Halving the tile to 2 cuts per-block working set to
///   ~11.5 KB and doubles block count, restoring L1 headroom.
pub(crate) fn choose_rows_per_block(_rows: usize, _hidden: usize) -> u32 {
    // `LARQL_CUDA_Q4K_RPB=N` overrides the default (1, 2, 4, 8, 16).
    // Used by `q4k_mmvq_rows_per_block_sweep` and as a back-out.
    if let Ok(val) = std::env::var("LARQL_CUDA_Q4K_RPB") {
        if let Ok(n) = val.parse::<u32>() {
            if [1, 2, 4, 8, 16].contains(&n) {
                return n;
            }
        }
    }
    // Empirical: a real-decode sweep on Gemma 3 4B Q4_K (RTX 4090,
    // graph path on, all shapes forced to the same `rows_per_block`)
    // showed:
    //
    //   rpb=1: 8.45 ms/tok  rpb=2: 8.66 ms/tok  rpb=4: 8.28 ms/tok
    //   rpb=8: 8.29 ms/tok  rpb=16: 8.31 ms/tok
    //
    // rpb=4 is best across all projections — including the
    // suspected-asymmetric `proj_down` (rows=2560, hidden=10240) — so
    // the original constant stays. The proj_down vs proj_gate_up
    // wallclock gap (47 µs vs 21 µs in `LARQL_CUDA_DECODE_PROFILE=1`
    // buckets) is mostly a profile-sync artifact; HBM bandwidth on
    // the down shape is bounded by access-pattern coalescing, not
    // by tile granularity.
    ROWS_PER_BLOCK
}

/// `cuda-q4k-mmvq-down-tile`: parameterised `matvec_device_into_with_dev`
/// that takes `rows_per_block` explicitly. Used by the shape-aware
/// dispatcher above and by the in-tree microbench
/// (`bench_rows_per_block_sweep`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn matvec_device_into_with_dev_tiled(
    backend: &CudaBackend,
    q4k_dev: &CudaSlice<u8>,
    x_q8_1: &Q8_1Buf,
    y_dev: &mut CudaSlice<f32>,
    rows: usize,
    hidden: usize,
    rows_per_block: u32,
) -> Result<(), CudaInitError> {
    if rows == 0 || hidden == 0 || !hidden.is_multiple_of(Q4K_BLOCK_ELEMS) {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid q4k_mmvq shape rows={rows} hidden={hidden}",
        )));
    }
    if rows_per_block == 0 {
        return Err(CudaInitError::DriverMissing(
            "q4k_mmvq rows_per_block must be > 0".into(),
        ));
    }
    let n_super_blocks = hidden / Q4K_BLOCK_ELEMS;
    let expected_q4k_bytes = rows
        .checked_mul(n_super_blocks)
        .and_then(|v| v.checked_mul(Q4K_BLOCK_BYTES))
        .ok_or_else(|| CudaInitError::DriverMissing("q4k byte size overflow".to_string()))?;
    if q4k_dev.len() != expected_q4k_bytes {
        return Err(CudaInitError::DriverMissing(format!(
            "q4k_dev length mismatch: got {}, expected {expected_q4k_bytes}",
            q4k_dev.len()
        )));
    }
    if x_q8_1.n_blocks * 32 != hidden {
        return Err(CudaInitError::DriverMissing(format!(
            "q8_1 input block count mismatch: {} blocks for hidden={hidden}",
            x_q8_1.n_blocks,
        )));
    }
    if y_dev.len() != rows {
        return Err(CudaInitError::DriverMissing(format!(
            "q4k_mmvq y length mismatch: y={} != rows={rows}",
            y_dev.len(),
        )));
    }

    let drv = backend.driver();
    let func = q4k_mmvq_function(drv)?;
    let rows_i = rows as i32;
    let n_super_blocks_i = n_super_blocks as i32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(rows_per_block), 1, 1),
        block_dim: (WARP_SIZE, rows_per_block, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(q4k_dev)
            .arg(&x_q8_1.bytes)
            .arg(y_dev)
            .arg(&rows_i)
            .arg(&n_super_blocks_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch q4k_mmvq: {e:?}")))?;
    }
    Ok(())
}

/// `cuda-decode-cuda-graph` companion: write q4k_mmvq output into a
/// caller-provided buffer instead of allocating a fresh one. Used by
/// `decode_token_device` once `DecodeScratch` is available so the
/// captured graph sees stable destination pointers across replays.
pub(crate) fn matvec_device_into(
    backend: &CudaBackend,
    q4k_data: &[u8],
    x_q8_1: &Q8_1Buf,
    y_dev: &mut CudaSlice<f32>,
    rows: usize,
    hidden: usize,
) -> Result<(), CudaInitError> {
    if rows == 0 || hidden == 0 || !hidden.is_multiple_of(Q4K_BLOCK_ELEMS) {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid q4k_mmvq shape rows={rows} hidden={hidden}",
        )));
    }
    let n_super_blocks = hidden / Q4K_BLOCK_ELEMS;
    let expected = rows
        .checked_mul(n_super_blocks)
        .and_then(|v| v.checked_mul(Q4K_BLOCK_BYTES))
        .ok_or_else(|| CudaInitError::DriverMissing("q4k byte size overflow".to_string()))?;
    if q4k_data.len() != expected {
        return Err(CudaInitError::DriverMissing(format!(
            "q4k byte length mismatch: got {}, expected {expected}",
            q4k_data.len()
        )));
    }
    if x_q8_1.n_blocks * 32 != hidden {
        return Err(CudaInitError::DriverMissing(format!(
            "q8_1 input block count mismatch: {} blocks for hidden={hidden}",
            x_q8_1.n_blocks,
        )));
    }
    if x_q8_1.bytes.len() != x_q8_1.n_blocks * 36 {
        return Err(CudaInitError::DriverMissing(format!(
            "q8_1 byte length mismatch: bytes={} expected={}",
            x_q8_1.bytes.len(),
            x_q8_1.n_blocks * 36,
        )));
    }
    if y_dev.len() != rows {
        return Err(CudaInitError::DriverMissing(format!(
            "q4k_mmvq y length mismatch: y={} != rows={rows}",
            y_dev.len(),
        )));
    }

    let drv = backend.driver();
    let func = q4k_mmvq_function(drv)?;
    let rows_i = rows as i32;
    let n_super_blocks_i = n_super_blocks as i32;
    let rows_per_block = choose_rows_per_block(rows, hidden);
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(rows_per_block), 1, 1),
        block_dim: (WARP_SIZE, rows_per_block, 1),
        shared_mem_bytes: 0,
    };

    backend.with_q4k_device_buf(q4k_data, |q4k_dev| {
        unsafe {
            drv.stream
                .launch_builder(func)
                .arg(q4k_dev)
                .arg(&x_q8_1.bytes)
                .arg(y_dev)
                .arg(&rows_i)
                .arg(&n_super_blocks_i)
                .launch(cfg)
                .map_err(|e| CudaInitError::DriverMissing(format!("launch q4k_mmvq: {e:?}")))?;
        }
        Ok(())
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Inline parity test gated on `LARQL_CUDA_AVAILABLE=1`.

    use super::*;
    use crate::cpu::ops::q4_common::quantize_q4_k;
    use crate::cuda::elem::quantize_q8_1_device;
    use crate::cuda::q4k_direct;

    fn gpu_or_skip() -> Option<CudaBackend> {
        if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
            eprintln!("skipping q4k_mmvq parity test: set LARQL_CUDA_AVAILABLE=1");
            return None;
        }
        CudaBackend::new().ok()
    }

    fn synth(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s & 0xFF_FFFF) as f32 / 8_388_608.0) - 0.5
            })
            .collect()
    }

    /// `cuda-q4k-mmvq-int8` Phase 2: the new INT8 mmvq kernel SHALL
    /// match the existing f32 direct kernel run on the Q8_1-DEQUANTIZED
    /// input to ≤ 1e-3 max-element. Comparing mmvq to `q4k_direct(x_f32)`
    /// is bounded by Q8_1 quantization noise (≈ scale * sqrt(hidden) per
    /// element); comparing both paths against the SAME Q8_1-dequantized
    /// reference removes that noise floor and isolates the kernel
    /// arithmetic.
    #[test]
    fn q4k_mmvq_matches_q4k_direct_on_dequantized_input() {
        let Some(backend) = gpu_or_skip() else { return };
        use larql_models::quant::half::f16_to_f32;

        for &(rows, hidden, seed) in &[(64usize, 256usize, 0x1100u64), (4096, 2560, 0x2200)] {
            let w_q4k = quantize_q4_k(&synth(rows * hidden, seed ^ 0xA1));
            let x = synth(hidden, seed ^ 0xB1);

            // Quantize x to Q8_1 on-device, copy back, dequantize on host,
            // then re-upload as f32 for the q4k_direct path.
            let x_dev = backend.htod_f32(&x).expect("htod x");
            let q8_1 = quantize_q8_1_device(&backend, &x_dev, hidden).expect("quantize_q8_1");
            backend.driver().sync().expect("sync");
            let q8_1_bytes = backend
                .driver()
                .stream
                .clone_dtoh(&q8_1.bytes)
                .expect("dtoh q8_1");
            let mut x_dq = Vec::with_capacity(hidden);
            for b in 0..(hidden / 32) {
                let base = b * 36;
                let scale =
                    f16_to_f32(u16::from_le_bytes([q8_1_bytes[base], q8_1_bytes[base + 1]]));
                for i in 0..32 {
                    let q = q8_1_bytes[base + 4 + i] as i8;
                    x_dq.push(scale * (q as f32));
                }
            }
            let x_dq_dev = backend.htod_f32(&x_dq).expect("htod x_dq");
            let direct = q4k_direct::matvec_device(&backend, &w_q4k, &x_dq_dev, rows, hidden)
                .expect("q4k_direct(dequantized)");
            backend.driver().sync().expect("sync");
            let direct_host = backend.driver().to_host(&direct).expect("dtoh direct");

            let mmvq = matvec_device(&backend, &w_q4k, &q8_1, rows, hidden).expect("q4k_mmvq");
            backend.driver().sync().expect("sync");
            let mmvq_host = backend.driver().to_host(&mmvq).expect("dtoh mmvq");

            let max_diff = direct_host
                .iter()
                .zip(&mmvq_host)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_diff <= 1e-3,
                "rows={rows} hidden={hidden}: kernel arithmetic max diff {max_diff} > 1e-3 \
                 (Q8_1-dequantized reference)",
            );
        }
    }

    /// `cuda-q4k-mmvq-down-tile` microbench: sweeps `rows_per_block` ∈
    /// {1, 2, 4, 8, 16} for representative Gemma 3 4B Q4_K projection
    /// shapes and prints per-call wall-clock time. Run manually with:
    ///
    /// ```text
    /// LARQL_CUDA_AVAILABLE=1 cargo test -p larql-compute --features cuda \
    ///   --lib q4k_mmvq_rows_per_block_sweep -- --ignored --nocapture
    /// ```
    ///
    /// Picks the best tile per shape and updates `choose_rows_per_block`
    /// accordingly.
    #[test]
    #[ignore = "manual microbench; warmups + repeated launches"]
    fn q4k_mmvq_rows_per_block_sweep() {
        let Some(backend) = gpu_or_skip() else { return };

        // Gemma 3 4B Q4_K projection shapes (rows, hidden):
        //   qkv      :  8*256 ×  2560  (q),   4*256 ×  2560  (k/v)
        //   wo       :   2560 × 8*256
        //   gate/up  :  10240 ×  2560
        //   down     :   2560 × 10240
        let shapes = [
            ("q     ", 2048, 2560),
            ("kv    ", 1024, 2560),
            ("wo    ", 2560, 2048),
            ("gate  ", 10240, 2560),
            ("up    ", 10240, 2560),
            ("down  ", 2560, 10240),
        ];

        let n_iters: usize = 200;

        for (label, rows, hidden) in shapes {
            let w_q4k = quantize_q4_k(&synth(rows * hidden, 0x9000));
            let q4k_dev = backend
                .arc_q4k_device_buf(&w_q4k)
                .expect("htod q4k");

            let x = synth(hidden, 0x9100);
            let x_dev = backend.htod_f32(&x).expect("htod x");
            let q8 = quantize_q8_1_device(&backend, &x_dev, hidden).expect("q8_1");

            let mut y = backend.alloc_f32(rows).expect("alloc y");

            print!("[mmvq_sweep] {label}  rows={rows:>5} hidden={hidden:>5}  ");

            // Warmup once so caches/JIT/scheduling settle.
            for &rpb in &[1u32, 2, 4, 8, 16] {
                let _ = matvec_device_into_with_dev_tiled(
                    &backend,
                    &q4k_dev,
                    &q8,
                    &mut y,
                    rows,
                    hidden,
                    rpb,
                );
            }
            backend.driver().sync().expect("sync");

            for &rpb in &[1u32, 2, 4, 8, 16] {
                if (rows as u32) % rpb != 0 {
                    print!("rpb{rpb}=-      ");
                    continue;
                }
                // Time `n_iters` launches; report per-call µs.
                let t = std::time::Instant::now();
                for _ in 0..n_iters {
                    matvec_device_into_with_dev_tiled(
                        &backend,
                        &q4k_dev,
                        &q8,
                        &mut y,
                        rows,
                        hidden,
                        rpb,
                    )
                    .expect("launch");
                }
                backend.driver().sync().expect("sync");
                let us = t.elapsed().as_secs_f64() * 1e6 / n_iters as f64;
                print!("rpb{rpb}={us:>5.1}µs  ");
            }
            println!();
        }
    }
}
