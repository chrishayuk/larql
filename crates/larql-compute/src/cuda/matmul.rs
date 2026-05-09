//! cuBLAS-backed f32 GEMM and GEMV.
//!
//! cuBLAS is column-major; ndarray gives us row-major slices. We use
//! the standard transposed-product identity to get a row-major output
//! without reformatting either input. See `design.md` D1.

use cudarc::cublas::{
    sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
    Gemm, GemmConfig,
};
use cudarc::driver::CudaSlice;

use super::driver::Driver;
use super::error::CudaInitError;

/// Compute row-major `C = A * B` where:
///   A is `m × k` row-major (`a.len() == m * k`)
///   B is `k × n` row-major (`b.len() == k * n`)
///   C is `m × n` row-major (returned as `Vec<f32>` of length `m * n`)
pub(crate) fn matmul(
    drv: &Driver,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, CudaInitError> {
    debug_assert_eq!(a.len(), m * k, "A length mismatch");
    debug_assert_eq!(b.len(), k * n, "B length mismatch");

    // Strategy (see design.md D1):
    //   row-major A (m,k) ≡ col-major A^T (k,m)
    //   row-major B (k,n) ≡ col-major B^T (n,k)
    //   want col-major C^T (n,m), which reads back as row-major C (m,n).
    //   C^T = B^T * A^T  → cuBLAS gemm with transa=N, transb=N,
    //                       cuBLAS-M = n, cuBLAS-N = m, cuBLAS-K = k,
    //                       leading dims: lda=n (B^T col-major), ldb=k (A^T col-major), ldc=n.
    //
    // First operand passed to cuBLAS is our row-major B; second is row-major A.
    let a_dev = drv.device_buf_from(a)?;
    let b_dev = drv.device_buf_from(b)?;
    let mut c_dev = drv.device_alloc(m * n)?;

    let cfg = GemmConfig {
        transa: CUBLAS_OP_N,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0_f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0_f32,
        ldc: n as i32,
    };

    // SAFETY: dimensions and leading-dim values match the buffer lengths
    // computed above; cudarc's safe wrapper still requires `unsafe` for
    // the cuBLAS dispatch itself.
    unsafe {
        drv.blas
            .gemm(cfg, &b_dev, &a_dev, &mut c_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("cublas gemm: {e:?}")))?;
    }

    drv.sync()?;
    drv.to_host(&c_dev)
}

/// Compute row-major `C = A * B^T` where:
///   A is `m × k` row-major
///   B is `n × k` row-major (transposed naturally by the multiply)
///   C is `m × n` row-major
pub(crate) fn matmul_transb(
    drv: &Driver,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>, CudaInitError> {
    debug_assert_eq!(a.len(), m * k, "A length mismatch");
    debug_assert_eq!(b.len(), n * k, "B length mismatch");

    // C = A * B^T  (row-major)
    //   row-major B (n,k) ≡ col-major B^T (k,n)
    //   row-major A (m,k) ≡ col-major A^T (k,m)
    //   we want col-major C^T (n,m) = (A * B^T)^T = B * A^T
    //   In col-major: B (k,n)^T * A^T (k,m) — first operand transposed.
    // cuBLAS: transa=T, transb=N, cuBLAS-M=n, cuBLAS-N=m, cuBLAS-K=k,
    //         lda=k (B is col-major k×n, leading dim is k), ldb=k, ldc=n.

    let a_dev = drv.device_buf_from(a)?;
    let b_dev = drv.device_buf_from(b)?;
    let mut c_dev = drv.device_alloc(m * n)?;

    let cfg = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0_f32,
        lda: k as i32,
        ldb: k as i32,
        beta: 0.0_f32,
        ldc: n as i32,
    };

    // SAFETY: see matmul() above.
    unsafe {
        drv.blas
            .gemm(cfg, &b_dev, &a_dev, &mut c_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("cublas gemm_transb: {e:?}")))?;
    }

    drv.sync()?;
    drv.to_host(&c_dev)
}

/// Compute `y = W * x` where:
///   W is `n × k` row-major (the weight matrix, n outputs × k inputs)
///   x has length `k`
///   y has length `n`
///
/// Uses GEMM with M=1 instead of cublasSgemv to keep the path identical
/// to the matmul one. This is the common LM-head shape; tuning a real
/// gemv kernel is a future change.
pub(crate) fn gemv(
    drv: &Driver,
    w: &[f32],
    x: &[f32],
    n: usize,
    k: usize,
) -> Result<Vec<f32>, CudaInitError> {
    debug_assert_eq!(w.len(), n * k, "W length mismatch");
    debug_assert_eq!(x.len(), k, "x length mismatch");
    matmul_transb(drv, x, w, 1, n, k)
}

pub(crate) fn gemv_device_w(
    drv: &Driver,
    w_dev: &CudaSlice<f32>,
    x: &[f32],
    n: usize,
    k: usize,
) -> Result<Vec<f32>, CudaInitError> {
    debug_assert_eq!(w_dev.len(), n * k, "W length mismatch");
    debug_assert_eq!(x.len(), k, "x length mismatch");

    let x_dev = drv.device_buf_from(x)?;
    let mut y_dev = drv.device_alloc(n)?;
    let cfg = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: 1,
        k: k as i32,
        alpha: 1.0_f32,
        lda: k as i32,
        ldb: k as i32,
        beta: 0.0_f32,
        ldc: n as i32,
    };

    unsafe {
        drv.blas
            .gemm(cfg, w_dev, &x_dev, &mut y_dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("cublas gemv_device_w: {e:?}")))?;
    }
    drv.sync()?;
    drv.to_host(&y_dev)
}

/// Device-resident GEMV: `y = W * x` with both `W` and `x` already on
/// device, output also on device. No `htod`, no `dtoh`, no `sync`.
/// `cuda-decode-device-resident` Phase 1.
pub(crate) fn gemv_device_inout(
    drv: &Driver,
    w_dev: &CudaSlice<f32>,
    x_dev: &CudaSlice<f32>,
    n: usize,
    k: usize,
) -> Result<CudaSlice<f32>, CudaInitError> {
    debug_assert_eq!(w_dev.len(), n * k, "W length mismatch");
    debug_assert_eq!(x_dev.len(), k, "x length mismatch");

    let mut y_dev = drv.device_alloc(n)?;
    let cfg = GemmConfig {
        transa: CUBLAS_OP_T,
        transb: CUBLAS_OP_N,
        m: n as i32,
        n: 1,
        k: k as i32,
        alpha: 1.0_f32,
        lda: k as i32,
        ldb: k as i32,
        beta: 0.0_f32,
        ldc: n as i32,
    };

    unsafe {
        drv.blas.gemm(cfg, w_dev, x_dev, &mut y_dev).map_err(|e| {
            CudaInitError::DriverMissing(format!("cublas gemv_device_inout: {e:?}"))
        })?;
    }
    Ok(y_dev)
}

#[cfg(test)]
mod tests {
    // The real correctness gate is in tests/test_cuda_f32.rs (gated on
    // LARQL_CUDA_AVAILABLE). The inline tests stay shape-only so they
    // run on a CPU host without needing a GPU.

    #[test]
    fn cuda_op_constants_are_distinct() {
        // Sanity: catch a regression where the cuBLAS op enum collapses.
        use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
        assert_ne!(CUBLAS_OP_N as i32, CUBLAS_OP_T as i32);
    }
}
