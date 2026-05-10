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

    /// Phase 4d batched-lm_head: amortised Q4_K matmul for spec-decode
    /// verify. Routes to `gemm_proj_seq` (cached Q4_K → f16 dequant +
    /// cuBLAS hgemm tensor cores) for `seq_len > 1`. The dequantized
    /// f16 weight is cached via `with_q4k_f16_device_buf`, so repeated
    /// calls in the spec loop pay dequant cost once. For `seq_len == 1`
    /// the per-row `q4k_matvec` direct kernel is faster (no f16 cast
    /// overhead) so we delegate there.
    fn q4k_matmul(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        if seq_len == 0 || x.len() != seq_len * hidden {
            return None;
        }
        if seq_len == 1 {
            return self.q4k_matvec(q4k_data, x, num_rows, hidden);
        }

        // GEMM path via cached f16 weight + cuBLAS hgemm. Falls back
        // to per-row matvec when the kernel constraint isn't met.
        let weight = QuantWeight {
            data: q4k_data,
            scales: None,
            format: QuantFormat::Q4_K,
        };
        let x_dev = self.htod_f32(x).ok()?;
        if let Some(y_dev) = self.gemm_proj_seq(weight, &x_dev, seq_len, num_rows, hidden) {
            // Output layout is row-major [seq_len, num_rows] from
            // matmul_transb_device_inout (and its f16 variant). That
            // matches the expected q4k_matmul output, so dtoh directly.
            return self.dtoh_f32(&y_dev).ok();
        }

        // Fallback: per-row q4k_matvec. Loses batching efficiency but
        // preserves correctness for shapes the GEMM path rejects.
        let mut out = Vec::with_capacity(seq_len * num_rows);
        for m in 0..seq_len {
            let row = &x[m * hidden..(m + 1) * hidden];
            let scores = self.q4k_matvec(q4k_data, row, num_rows, hidden)?;
            if scores.len() != num_rows {
                return None;
            }
            out.extend_from_slice(&scores);
        }
        Some(out)
    }
}
