//! Batched Q4_K matvec — phase 2 of `cuda-speculative-decoding`.
//!
//! Extends [`super::q4k_direct::matvec`] (single x → single y) to
//! process M ≥ 1 input vectors in a single kernel launch, sharing
//! the weight read across the M dimension. This is what unlocks
//! Tensor Cores for the speculative-decode verification pass:
//! batch=1 wastes 15/16 columns of a 16×16 fragment; batch ≥ 4
//! becomes productive.
//!
//! Phase 2 acceptance criterion (per design.md §2.1): output at
//! `M_TILE = 1` SHALL be bit-identical to the single-token kernel,
//! and at `M_TILE > 1` SHALL match running the single-token kernel
//! `M` times within fp32 ordering tolerance (1e-5 absolute).
//!
//! This is the **shared-mem reduction, per-M serial dispatch**
//! version. A weight-cache + register-tile follow-up lands once
//! the M_TILE template proves the architectural win.

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

const Q4K_BLOCK_ELEMS: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;
pub const M_TILE_MAX: usize = 8;
const THREADS_PER_ROW: u32 = 128;
const ROWS_PER_BLOCK: u32 = 4;

const Q4K_MATVEC_BATCHED_SRC: &str = r#"
__device__ float larql_f16_to_f32_b(unsigned short h) {
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

__device__ unsigned char q4k_scale_b(const unsigned char* packed, int j) {
    if (j < 4) return packed[j] & 0x3fu;
    return (packed[j + 4] & 0x0fu) | ((packed[j - 4] >> 6) << 4);
}

__device__ unsigned char q4k_min_b(const unsigned char* packed, int j) {
    if (j < 4) return packed[j + 4] & 0x3fu;
    return (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
}

// Batched Q4_K matvec.
//
// Layout: input X is row-major [m_tile, hidden]; output Y is row-major
// [m_tile, rows]. Each thread block covers ROWS_PER_BLOCK rows; each
// thread within a row's lane accumulates m_tile partial dot products,
// one per input vector. Per-m reduction in shared memory.
//
// Shared memory size: ROWS_PER_BLOCK * M_TILE_MAX * blockDim.x * sizeof(float).
extern "C" __global__ void q4k_matvec_batched(
    const unsigned char* q4k,
    const float* x,
    float* y,
    int rows,
    int hidden,
    int blocks_per_row,
    int m_tile
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int row_lane = threadIdx.y;
    extern __shared__ float smem[];

    // Per-thread accumulators, one per m. Cap at compile-time M_TILE_MAX.
    float acc[8];
    #pragma unroll
    for (int m = 0; m < 8; m++) acc[m] = 0.0f;

    const unsigned char* row_base = q4k + (unsigned long long)row * blocks_per_row * 144ull;

    for (int sb = 0; sb < blocks_per_row; ++sb) {
        const unsigned char* block = row_base + sb * 144;
        float d = larql_f16_to_f32_b((unsigned short)block[0] | ((unsigned short)block[1] << 8));
        float dmin = larql_f16_to_f32_b((unsigned short)block[2] | ((unsigned short)block[3] << 8));
        const unsigned char* packed = block + 4;
        const unsigned char* quants = block + 16;
        int x_base = sb * 256;

        for (int elem = tid; elem < 256; elem += bdim) {
            int sub = elem >> 5;
            int lane = elem & 31;
            int pair = sub >> 1;
            unsigned char byte = quants[pair * 32 + lane];
            unsigned char nibble = (sub & 1) ? (byte >> 4) : (byte & 0x0f);
            float sc = d * (float)q4k_scale_b(packed, sub);
            float mn = dmin * (float)q4k_min_b(packed, sub);
            float w = sc * (float)nibble - mn;

            // Multiply-accumulate over m_tile activations at this hidden index.
            // Weight `w` is shared across all m — this is the architectural win.
            for (int m = 0; m < m_tile; m++) {
                float xv = x[m * hidden + x_base + elem];
                acc[m] += w * xv;
            }
        }
    }

    // Per-m shared-mem reduction. smem is laid out as
    // [row_lane][m][thread] tightly so each (row_lane, m) reduction
    // is a contiguous bdim-element strip.
    for (int m = 0; m < m_tile; m++) {
        int slot = (row_lane * 8 + m) * bdim + tid;
        smem[slot] = acc[m];
        __syncthreads();
        for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smem[slot] += smem[slot + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            y[m * rows + row] = smem[(row_lane * 8 + m) * bdim];
        }
    }
}
"#;

static Q4K_MATVEC_BATCHED_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> =
    OnceLock::new();

fn q4k_matvec_batched_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = Q4K_MATVEC_BATCHED_FUNC.get() {
        return Ok(f);
    }
    let ptx = compile_ptx(Q4K_MATVEC_BATCHED_SRC)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc q4k_batched: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load q4k_batched: {e:?}")))?;
    let func = module
        .load_function("q4k_matvec_batched")
        .map_err(|e| CudaInitError::DriverMissing(format!("load q4k_batched fn: {e:?}")))?;
    let _ = Q4K_MATVEC_BATCHED_FUNC.set((module, func));
    Ok(&Q4K_MATVEC_BATCHED_FUNC.get().unwrap().1)
}

/// Run batched Q4_K matvec.
///
/// `x_batched`: row-major `[m_tile, hidden]`.
/// Returns row-major `[m_tile, rows]`.
pub fn matvec_batched(
    backend: &CudaBackend,
    q4k_data: &[u8],
    x_batched: &[f32],
    rows: usize,
    hidden: usize,
    m_tile: usize,
) -> Result<Vec<f32>, CudaInitError> {
    if rows == 0 || hidden == 0 || m_tile == 0 || m_tile > M_TILE_MAX {
        return Err(CudaInitError::DriverMissing(format!(
            "matvec_batched bad shape: rows={rows} hidden={hidden} m_tile={m_tile} (max {M_TILE_MAX})"
        )));
    }
    if x_batched.len() != m_tile * hidden {
        return Err(CudaInitError::DriverMissing(format!(
            "matvec_batched x len mismatch: got {}, expected {}",
            x_batched.len(),
            m_tile * hidden
        )));
    }
    if !hidden.is_multiple_of(Q4K_BLOCK_ELEMS) {
        return Err(CudaInitError::DriverMissing(format!(
            "matvec_batched hidden={hidden} not multiple of {Q4K_BLOCK_ELEMS}"
        )));
    }
    let blocks_per_row = hidden / Q4K_BLOCK_ELEMS;
    let expected_q4k = rows * blocks_per_row * Q4K_BLOCK_BYTES;
    if q4k_data.len() != expected_q4k {
        return Err(CudaInitError::DriverMissing(format!(
            "matvec_batched q4k len mismatch: got {}, expected {expected_q4k}",
            q4k_data.len()
        )));
    }

    let drv = backend.driver();
    let func = q4k_matvec_batched_function(drv)?;
    let x_dev = drv.device_buf_from(x_batched)?;
    let mut y_dev = drv.device_alloc(m_tile * rows)?;

    let q4k_dev = drv.device_u8_buf_from(q4k_data)?;

    let rows_i = rows as i32;
    let hidden_i = hidden as i32;
    let blocks_per_row_i = blocks_per_row as i32;
    let m_tile_i = m_tile as i32;

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(ROWS_PER_BLOCK), 1, 1),
        block_dim: (THREADS_PER_ROW, ROWS_PER_BLOCK, 1),
        shared_mem_bytes: ROWS_PER_BLOCK
            * (M_TILE_MAX as u32)
            * THREADS_PER_ROW
            * std::mem::size_of::<f32>() as u32,
    };

    unsafe {
        drv.stream
            .launch_builder(func)
            .arg(&q4k_dev)
            .arg(&x_dev)
            .arg(&mut y_dev)
            .arg(&rows_i)
            .arg(&hidden_i)
            .arg(&blocks_per_row_i)
            .arg(&m_tile_i)
            .launch(cfg)
            .map_err(|e| CudaInitError::DriverMissing(format!("launch q4k_batched: {e:?}")))?;
    }
    drv.sync()?;
    drv.to_host(&y_dev)
}
