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

pub(crate) fn dequant_q4_kf(data: &[u8], n_elements: usize) -> Result<Vec<f32>, CudaInitError> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = crate::pipeline::Q4_KF_BLOCK_BYTES;
    if !n_elements.is_multiple_of(BLOCK_ELEMS) {
        return Err(CudaInitError::DriverMissing(format!(
            "dequant_q4_kf: n_elements {n_elements} not multiple of {BLOCK_ELEMS}"
        )));
    }
    let n_blocks = n_elements / BLOCK_ELEMS;
    let expected = n_blocks * BLOCK_BYTES;
    if data.len() < expected {
        return Err(CudaInitError::DriverMissing(format!(
            "dequant_q4_kf: short buffer: {} < {expected}",
            data.len()
        )));
    }

    let mut out = vec![0.0f32; n_elements];
    for sb in 0..n_blocks {
        let block = &data[sb * BLOCK_BYTES..(sb + 1) * BLOCK_BYTES];
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for j in 0..8 {
            let sc = u16::from_le_bytes([block[j * 2], block[j * 2 + 1]]);
            let mn = u16::from_le_bytes([block[16 + j * 2], block[16 + j * 2 + 1]]);
            scales[j] = crate::cpu::ops::q4_common::f16_to_f32(sc);
            mins[j] = crate::cpu::ops::q4_common::f16_to_f32(mn);
        }

        let quants = &block[32..160];
        let sb_base = sb * BLOCK_ELEMS;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let chunk = &quants[g * 32..(g + 1) * 32];
            let base_lo = sb_base + sb_lo * 32;
            let base_hi = sb_base + sb_hi * 32;
            for l in 0..32 {
                let byte = chunk[l];
                out[base_lo + l] = scales[sb_lo] * (byte & 0x0F) as f32 - mins[sb_lo];
                out[base_hi + l] = scales[sb_hi] * ((byte >> 4) & 0x0F) as f32 - mins[sb_hi];
            }
        }
    }
    Ok(out)
}

pub(crate) fn dequant_q6_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, CudaInitError> {
    ggml::q6_k::dequantize_q6_k(data, n_elements)
        .map_err(|e| CudaInitError::DriverMissing(format!("dequant_q6_k: {e}")))
}
