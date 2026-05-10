//! CUDA quantized matvec dispatch.
//!
//! The current correctness-first implementation dequantizes weights on CPU
//! and runs the resulting f32 GEMV through cuBLAS. Fused direct kernels can
//! replace these bodies without changing the public `QuantMatVec` trait.

use crate::backend::QuantMatVec;
use crate::{QuantFormat, QuantWeight};

use super::backend::CudaBackend;
use super::dequant;
use super::matmul as kernels;
use super::q4k_direct;

impl QuantMatVec for CudaBackend {
    fn q4_matvec(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        const Q8_BLOCK: usize = 32;
        if q8_x.len() != hidden || q8_scales.len() * Q8_BLOCK != hidden {
            return None;
        }
        let mut x = Vec::with_capacity(hidden);
        for (block_i, scale) in q8_scales.iter().enumerate() {
            for j in 0..Q8_BLOCK {
                x.push((q8_x[block_i * Q8_BLOCK + j] as f32) * scale);
            }
        }
        let w = dequant::dequant_q4_0(q4_data, num_rows * hidden).ok()?;
        kernels::gemv(self.driver(), &w, &x, num_rows, hidden).ok()
    }

    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden {
            return None;
        }
        if std::env::var("LARQL_CUDA_Q4K_HOST_DEQUANT").ok().as_deref() != Some("1") {
            if let Ok(out) = q4k_direct::matvec(self, q4k_data, x, num_rows, hidden) {
                return Some(out);
            }
            // Fallback: dequant Q4_K once + cuBLAS GEMV via the
            // session-cached f16/f32 weight buffer. Handles the
            // non-multiple-of-256 hidden case the direct kernel
            // rejects (e.g. Gemma 3 270M's lm_head: hidden=640
            // and vocab=262144). Mirrors the per-token decode
            // fallback in `decode::matvec_device_mmvq` so the
            // same dequantized buffer is reused across calls.
            let x_dev = self.htod_f32(x).ok()?;
            let weight = QuantWeight {
                data: q4k_data,
                scales: None,
                format: QuantFormat::Q4_K,
            };
            let y_dev = self.gemm_proj_seq(weight, &x_dev, 1, num_rows, hidden)?;
            return self.dtoh_f32(&y_dev).ok();
        }
        let w = dequant::dequant_q4_k(q4k_data, num_rows * hidden).ok()?;
        kernels::gemv(self.driver(), &w, x, num_rows, hidden).ok()
    }

    fn q4kf_matvec(
        &self,
        q4kf_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden {
            return None;
        }
        let w = dequant::dequant_q4_kf(q4kf_data, num_rows * hidden).ok()?;
        kernels::gemv(self.driver(), &w, x, num_rows, hidden).ok()
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden {
            return None;
        }
        if std::env::var("LARQL_CUDA_Q6K_HOST_DEQUANT").ok().as_deref() != Some("1") {
            return self
                .with_q6k_f32_device_buf(q6k_data, num_rows * hidden, |w_dev| {
                    kernels::gemv_device_w(self.driver(), w_dev, x, num_rows, hidden)
                })
                .ok();
        }
        let w = dequant::dequant_q6_k(q6k_data, num_rows * hidden).ok()?;
        kernels::gemv(self.driver(), &w, x, num_rows, hidden).ok()
    }

    fn has_q4(&self) -> bool {
        true
    }
}
