//! `CudaBackend` — owns the [`Driver`] and dispatches to the per-kernel
//! wrappers in `cuda::matmul`. The kernel surface is filled in across
//! the [`cuda-and-rotorquant-kv`][parent] sub-changes; this module's
//! current state is `cuda-f32-baseline`.
//!
//! [parent]: ../../../../openspec/changes/cuda-and-rotorquant-kv/

use std::sync::Arc;

use ndarray::{Array2, ArrayView2};

use crate::backend::{Capability, ComputeBackend, DecodeBackend, MatMul, QuantMatVec};

use super::dequant;
use super::driver::Driver;
use super::error::CudaInitError;
use super::matmul as kernels;

pub struct CudaBackend {
    drv: Arc<Driver>,
}

impl CudaBackend {
    pub fn new() -> Result<Self, CudaInitError> {
        Self::new_with_index(0)
    }

    pub fn new_with_index(ordinal: usize) -> Result<Self, CudaInitError> {
        let drv = Driver::new_with_index(ordinal)?;
        Ok(CudaBackend { drv })
    }

    /// Module-internal accessor used by `cuda::attn` so the helper
    /// can borrow the driver without exposing it crate-wide.
    pub(crate) fn driver(&self) -> &Driver {
        &self.drv
    }

    /// Internal: contiguous row-major view of an `ArrayView2`. The
    /// fast-path is when the view is already standard layout; we only
    /// allocate on the slow-path (transposed / strided views).
    fn as_contiguous<'a>(&self, m: ArrayView2<'a, f32>) -> Vec<f32> {
        if let Some(slice) = m.as_slice() {
            slice.to_vec()
        } else {
            // Strided view — collect through ndarray's iterator into a
            // fresh Vec. Cheap on the dimensions we care about.
            m.iter().copied().collect()
        }
    }
}

// ── MatMul: real cuBLAS calls ──────────────────────────────────────────

impl MatMul for CudaBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2, "matmul shape mismatch: {a:?} × {b:?}");

        let a_buf = self.as_contiguous(a);
        let b_buf = self.as_contiguous(b);
        let out = kernels::matmul(&self.drv, &a_buf, &b_buf, m, n, k)
            .expect("CudaBackend::matmul: cuBLAS failed");

        Array2::from_shape_vec((m, n), out)
            .expect("CudaBackend::matmul: shape mismatch on result")
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        // C = A * B^T  with A: m×k, B: n×k → C: m×n
        let (m, k) = a.dim();
        let (n, k2) = b.dim();
        assert_eq!(k, k2, "matmul_transb shape mismatch: {a:?} × {b:?}^T");

        let a_buf = self.as_contiguous(a);
        let b_buf = self.as_contiguous(b);
        let out = kernels::matmul_transb(&self.drv, &a_buf, &b_buf, m, n, k)
            .expect("CudaBackend::matmul_transb: cuBLAS failed");

        Array2::from_shape_vec((m, n), out)
            .expect("CudaBackend::matmul_transb: shape mismatch on result")
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = w.dim();
        if x.len() != k {
            return None;
        }
        let w_buf = self.as_contiguous(w);
        match kernels::gemv(&self.drv, &w_buf, x, n, k) {
            Ok(v) => Some(v),
            Err(_) => None,
        }
    }
}

impl QuantMatVec for CudaBackend {
    // ── Q4_0 ──────────────────────────────────────────────────────
    // The trait method takes Q8-quantised input (i8 + scales) so the
    // CPU dispatch can avoid re-quantising. We unconditionally
    // dequantise inputs+weights to f32 and run cuBLAS gemv. Scale
    // factors from the Q8 input become a per-block multiplier we
    // fold back into the result.
    fn q4_matvec(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        // Reconstruct the f32 input from (q8_x, q8_scales). Q8 blocks
        // are 32 i8 values + one f32 scale; layout invariant comes
        // from `cpu::ops::q4_common::quantize_to_q8`.
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
        kernels::gemv(&self.drv, &w, &x, num_rows, hidden).ok()
    }

    // ── Q4_K ──────────────────────────────────────────────────────
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
        let w = dequant::dequant_q4_k(q4k_data, num_rows * hidden).ok()?;
        kernels::gemv(&self.drv, &w, x, num_rows, hidden).ok()
    }

    // ── Q6_K ──────────────────────────────────────────────────────
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
        let w = dequant::dequant_q6_k(q6k_data, num_rows * hidden).ok()?;
        kernels::gemv(&self.drv, &w, x, num_rows, hidden).ok()
    }
}

impl DecodeBackend for CudaBackend {
    // Default `decode_token` returns `None`, letting callers fall back
    // to the per-layer matmul path. Override in
    // `cuda-fused-attention`.
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn device_info(&self) -> String {
        self.drv.device_info()
    }

    fn supports(&self, cap: Capability) -> bool {
        // Capability bits flip on as sub-changes land:
        //   cuda-f32-baseline    → Cuda, F32Gemv
        //   cuda-q4-matvec       → +QuantMatVec, +Q4VecMat
        //   cuda-fused-attention → +FlashAttentionV2 (this change)
        //   rotorquant-*         → +KvCompressionRotorQuant
        matches!(
            cap,
            Capability::Cuda
                | Capability::F32Gemv
                | Capability::QuantMatVec
                | Capability::Q4VecMat
                | Capability::FlashAttentionV2
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Base smoke test that runs anywhere — no GPU required when the
    /// driver is missing, the call returns Err and the test passes.
    #[test]
    fn driver_missing_returns_typed_error() {
        match CudaBackend::new() {
            Ok(b) => {
                assert_eq!(b.name(), "cuda");
            }
            Err(CudaInitError::DriverMissing(_))
            | Err(CudaInitError::NoDevices)
            | Err(CudaInitError::ToolkitMismatch { .. }) => {
                // Expected on a host without a working CUDA driver.
            }
            Err(CudaInitError::NotImplemented(_)) => {
                panic!("backend should no longer report NotImplemented after f32 baseline");
            }
        }
    }

    #[test]
    fn supports_f32_gemv_after_baseline() {
        // We only assert the capability set if init succeeded; on
        // hosts without CUDA the test no-ops.
        if let Ok(b) = CudaBackend::new() {
            assert!(b.supports(Capability::Cuda));
            assert!(
                b.supports(Capability::F32Gemv),
                "cuda-f32-baseline must advertise F32Gemv"
            );
            // Decode-token / fused attention land in cuda-fused-attention.
            assert!(!b.supports(Capability::DecodeToken));
        }
    }

    #[test]
    fn supports_q4_matvec_after_q4_baseline() {
        if let Ok(b) = CudaBackend::new() {
            // Capabilities flipped on by cuda-q4-matvec.
            assert!(b.supports(Capability::QuantMatVec));
            assert!(b.supports(Capability::Q4VecMat));
            // Capabilities still off (their sub-changes haven't landed).
            assert!(!b.supports(Capability::KvCompressionRotorQuant));
            assert!(!b.supports(Capability::DecodeToken));
        }
    }

    #[test]
    fn supports_fa2_after_fused_attention() {
        if let Ok(b) = CudaBackend::new() {
            // Cumulative capability set after cuda-fused-attention.
            assert!(b.supports(Capability::Cuda));
            assert!(b.supports(Capability::F32Gemv));
            assert!(b.supports(Capability::QuantMatVec));
            assert!(b.supports(Capability::Q4VecMat));
            assert!(b.supports(Capability::FlashAttentionV2));
            // Still off — RotorQuant lands later.
            assert!(!b.supports(Capability::KvCompressionRotorQuant));
        }
    }
}
