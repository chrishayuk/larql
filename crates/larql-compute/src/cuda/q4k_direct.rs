//! Direct CUDA Q4_K matrix-vector multiply.
//!
//! Q4_K uses the GGUF 144-byte super-block layout from
//! `larql_models::quant::ggml::q4_k`: f16 `d`, f16 `dmin`, 12 packed
//! scale/min bytes, then 128 quant bytes for 256 values. This kernel maps one
//! CUDA block to one output row and accumulates directly from packed bytes,
//! avoiding the older host-side Q4_K -> f32 matrix materialization.

use std::sync::OnceLock;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::backend::CudaBackend;
use super::driver::Driver;
use super::error::CudaInitError;

const Q4K_BLOCK_ELEMS: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;
const THREADS_PER_ROW: u32 = 128;
const ROWS_PER_BLOCK: u32 = 4;

const Q4K_MATVEC_SRC: &str = r#"
__device__ float larql_f16_to_f32(unsigned short h) {
    unsigned int sign = ((unsigned int)h & 0x8000u) << 16;
    unsigned int mant = (unsigned int)h & 0x03ffu;
    int exp = ((int)h >> 10) & 0x1f;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0u) {
            bits = sign;
        } else {
            float v = (float)mant * 5.960464477539063e-8f; // 2^-24
            return (sign != 0u) ? -v : v;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((unsigned int)(exp + 112) << 23) | (mant << 13);
    }
    return __uint_as_float(bits);
}

__device__ unsigned char q4k_scale(const unsigned char* packed, int j) {
    if (j < 4) return packed[j] & 0x3fu;
    return (packed[j + 4] & 0x0fu) | ((packed[j - 4] >> 6) << 4);
}

__device__ unsigned char q4k_min(const unsigned char* packed, int j) {
    if (j < 4) return packed[j + 4] & 0x3fu;
    return (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
}

extern "C" __global__ void q4k_matvec_direct(
    const unsigned char* q4k,
    const float* x,
    float* y,
    int rows,
    int hidden,
    int blocks_per_row
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int row_lane = threadIdx.y;
    extern __shared__ float smem[];
    float* row_smem = smem + row_lane * bdim;

    float acc = 0.0f;
    const unsigned char* row_base = q4k + (unsigned long long)row * blocks_per_row * 144ull;

    for (int sb = 0; sb < blocks_per_row; ++sb) {
        const unsigned char* block = row_base + sb * 144;
        float d = larql_f16_to_f32((unsigned short)block[0] | ((unsigned short)block[1] << 8));
        float dmin = larql_f16_to_f32((unsigned short)block[2] | ((unsigned short)block[3] << 8));
        const unsigned char* packed = block + 4;
        const unsigned char* quants = block + 16;
        int x_base = sb * 256;

        for (int elem = tid; elem < 256; elem += bdim) {
            int sub = elem >> 5;
            int lane = elem & 31;
            int pair = sub >> 1;
            unsigned char byte = quants[pair * 32 + lane];
            unsigned char nibble = (sub & 1) ? (byte >> 4) : (byte & 0x0f);
            float sc = d * (float)q4k_scale(packed, sub);
            float mn = dmin * (float)q4k_min(packed, sub);
            float w = sc * (float)nibble - mn;
            acc += w * x[x_base + elem];
        }
    }

    row_smem[tid] = acc;
    __syncthreads();
    for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) row_smem[tid] += row_smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[row] = row_smem[0];
}
"#;

static Q4K_MATVEC_FUNC: OnceLock<(std::sync::Arc<CudaModule>, CudaFunction)> = OnceLock::new();

fn q4k_matvec_function(drv: &Driver) -> Result<&'static CudaFunction, CudaInitError> {
    if let Some((_, f)) = Q4K_MATVEC_FUNC.get() {
        return Ok(f);
    }
    let ptx = compile_ptx(Q4K_MATVEC_SRC)
        .map_err(|e| CudaInitError::DriverMissing(format!("nvrtc compile q4k_matvec: {e:?}")))?;
    let module = drv
        .ctx
        .load_module(ptx)
        .map_err(|e| CudaInitError::DriverMissing(format!("load q4k module: {e:?}")))?;
    let func = module
        .load_function("q4k_matvec_direct")
        .map_err(|e| CudaInitError::DriverMissing(format!("load q4k function: {e:?}")))?;
    let _ = Q4K_MATVEC_FUNC.set((module, func));
    let (_, f) = Q4K_MATVEC_FUNC.get().unwrap();
    Ok(f)
}

/// Device-resident Q4_K matvec. Same kernel as [`matvec`] but takes
/// a device-side input slice and returns a device-side output slice
/// — no implicit `htod` / `dtoh` and no `sync` between launches.
/// `cuda-decode-device-resident` Phase 1.
///
/// The kernel itself is unchanged from the host-input variant; only
/// the host↔device plumbing differs. Callers that want a `Vec<f32>`
/// can keep using [`matvec`], which now wraps this function.
pub(crate) fn matvec_device(
    backend: &CudaBackend,
    q4k_data: &[u8],
    x_dev: &CudaSlice<f32>,
    rows: usize,
    hidden: usize,
) -> Result<CudaSlice<f32>, CudaInitError> {
    if rows == 0 || hidden == 0 || x_dev.len() != hidden || !hidden.is_multiple_of(Q4K_BLOCK_ELEMS)
    {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid q4k matvec shape rows={rows} hidden={hidden} x_len={}",
            x_dev.len()
        )));
    }
    let blocks_per_row = hidden / Q4K_BLOCK_ELEMS;
    let expected = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(Q4K_BLOCK_BYTES))
        .ok_or_else(|| CudaInitError::DriverMissing("q4k byte size overflow".to_string()))?;
    if q4k_data.len() != expected {
        return Err(CudaInitError::DriverMissing(format!(
            "q4k byte length mismatch: got {}, expected {expected}",
            q4k_data.len()
        )));
    }

    let drv = backend.driver();
    let func = q4k_matvec_function(drv)?;
    let mut y_dev = drv.device_alloc_uninit(rows)?;
    let rows_i = rows as i32;
    let hidden_i = hidden as i32;
    let blocks_per_row_i = blocks_per_row as i32;
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(ROWS_PER_BLOCK), 1, 1),
        block_dim: (THREADS_PER_ROW, ROWS_PER_BLOCK, 1),
        shared_mem_bytes: THREADS_PER_ROW * ROWS_PER_BLOCK * std::mem::size_of::<f32>() as u32,
    };

    backend.with_q4k_device_buf(q4k_data, |q4k_dev| {
        unsafe {
            drv.stream
                .launch_builder(func)
                .arg(q4k_dev)
                .arg(x_dev)
                .arg(&mut y_dev)
                .arg(&rows_i)
                .arg(&hidden_i)
                .arg(&blocks_per_row_i)
                .launch(cfg)
                .map_err(|e| CudaInitError::DriverMissing(format!("launch q4k_matvec: {e:?}")))?;
        }
        Ok(())
    })?;

    Ok(y_dev)
}

pub(crate) fn matvec(
    backend: &CudaBackend,
    q4k_data: &[u8],
    x: &[f32],
    rows: usize,
    hidden: usize,
) -> Result<Vec<f32>, CudaInitError> {
    if rows == 0 || hidden == 0 || x.len() != hidden || !hidden.is_multiple_of(Q4K_BLOCK_ELEMS) {
        return Err(CudaInitError::DriverMissing(format!(
            "invalid q4k matvec shape rows={rows} hidden={hidden} x_len={}",
            x.len()
        )));
    }
    let trace_min_rows = std::env::var("LARQL_CUDA_Q4K_TRACE_MIN_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100_000);
    let trace = std::env::var("LARQL_CUDA_Q4K_TRACE").ok().as_deref() == Some("1")
        && rows >= trace_min_rows;
    let t0 = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };

    let drv = backend.driver();
    let x_dev = drv.device_buf_from(x)?;
    let y_dev = matvec_device(backend, q4k_data, &x_dev, rows, hidden)?;
    drv.sync()?;
    let out = drv.to_host(&y_dev)?;
    if let Some(t0) = t0 {
        eprintln!(
            "[cuda-q4k] rows={rows} hidden={hidden} bytes={} cache_slots={} elapsed_ms={:.3}",
            q4k_data.len(),
            backend.q4k_device_cache_len(),
            t0.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(out)
}
