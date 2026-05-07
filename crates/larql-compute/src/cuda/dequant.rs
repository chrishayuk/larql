//! Host-side dequant shims around `larql_models::quant::ggml::*`.
//!
//! Phase-1 (cuda-q4-matvec): dequant happens on CPU, then the f32
//! buffer is uploaded to the device for cuBLAS gemv. This is the
//! correctness-first path; a future `cuda-q4-matvec-fused`
//! sub-change replaces it with on-device dequant kernels.

use larql_models::quant::ggml;

use super::error::CudaInitError;

pub(crate) fn dequant_q4_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, CudaInitError> {
    ggml::legacy::dequantize_q4_0(data, n_elements)
        .map_err(|e| CudaInitError::DriverMissing(format!("dequant_q4_0: {e}")))
}

pub(crate) fn dequant_q4_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, CudaInitError> {
    ggml::q4_k::dequantize_q4_k(data, n_elements)
        .map_err(|e| CudaInitError::DriverMissing(format!("dequant_q4_k: {e}")))
}

pub(crate) fn dequant_q6_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, CudaInitError> {
    ggml::q6_k::dequantize_q6_k(data, n_elements)
        .map_err(|e| CudaInitError::DriverMissing(format!("dequant_q6_k: {e}")))
}
