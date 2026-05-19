//! CPU compute backend — BLAS for f32, C kernel for Q4.
//!
//! On macOS: Accelerate BLAS dispatches through Apple's AMX coprocessor.
//! On Linux: OpenBLAS or similar.
//! Q4: C kernel with ARM vdotq_s32 (0.95ms per 105MB matrix on M3 Max).
//!
//! ## Modules
//!
//! - `ops/f32_matmul`: BLAS sgemm dispatch
//! - `ops/q4_matvec`:  C kernel Q4_0 × Q8 matrix-vector
//! - `ops/q4_vecmat`:  C kernel Q4_0 vector-matrix
//! - `ops/q4_common`:  Q8 quantization, C FFI declarations
//! - `ops/q4k_matvec`: Q4_K matrix-vector (llama.cpp super-block format)
//! - `ops/q6k_matvec`: Q6_K matrix-vector
//! - `ops/q8_matvec`:  Q8 matrix-vector
//! - `ops/geglu`:      Element-wise GEGLU activation
//! - `ops/attention`:  Causal attention (fused QK softmax V)
//! - `ops/vector`:     `dot`, `norm`, `cosine` over slices/views
//! - `ops/linalg`:     Cholesky factor/solve, `ridge_decomposition_solve`

pub mod ops;

// Re-export for backward compatibility (used by benchmarks/examples)
pub mod q4 {
    pub use super::ops::q4_common::{q4_0_matvec_c, q4_0_vecmat_c, quantize_q4_0, quantize_to_q8};
    pub use super::ops::q4_matvec::dispatch as q4_matvec;
    pub use super::ops::q4_vecmat::dispatch as q4_vecmat;
}

use crate::backend::{Capability, ComputeBackend, DecodeBackend, MatMul, QuantMatVec};
use ndarray::{Array2, ArrayView2};

/// CPU backend using BLAS (f32) and C kernel (Q4).
pub struct CpuBackend;

impl MatMul for CpuBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        ops::f32_matmul::matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        ops::f32_matmul::matmul_transb(a, b)
    }
}

impl QuantMatVec for CpuBackend {
    fn q4_matvec(
        &self,
        q4_data: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(ops::q4_matvec::dispatch_q8(
            q4_data, q8_x, q8_scales, num_rows, hidden,
        ))
    }

    fn q4_vecmat(
        &self,
        activation: &[f32],
        q4_data: &[u8],
        intermediate: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(ops::q4_vecmat::dispatch(
            activation,
            q4_data,
            intermediate,
            hidden,
        ))
    }

    fn q4k_matvec(
        &self,
        q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        // `ops::q4k_matvec::dispatch` is a tight reference impl but
        // pays a redundant sumy compute inside the row loop. The
        // production path uses `q4_common::q4k_matvec_into`, which
        // precomputes per-sub-block sum_x once and shares it across
        // rows — measurable savings on Gemma-3-4B-class shapes.
        // Parallelised across rows with rayon for the matvec shapes
        // a decode step pulls (2560–8192 rows).
        let mut out = vec![0.0f32; num_rows];
        ops::q4_common::q4k_matvec_into(&mut out, x, q4k_data, num_rows, hidden);
        Some(out)
    }

    fn q6k_matvec(
        &self,
        q6k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(ops::q6k_matvec::dispatch(q6k_data, x, num_rows, hidden))
    }

    fn q4k_dual_matvec(
        &self,
        q4k_a: &[u8],
        q4k_b: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let mut out_a = vec![0.0f32; num_rows];
        let mut out_b = vec![0.0f32; num_rows];
        ops::q4_common::q4k_dual_matvec_into(
            &mut out_a, &mut out_b, x, q4k_a, q4k_b, num_rows, hidden,
        );
        Some((out_a, out_b))
    }

    fn supports_quant(&self, format: crate::QuantFormat) -> bool {
        use crate::QuantFormat;
        matches!(
            format,
            QuantFormat::Q4_0 | QuantFormat::Q4_K | QuantFormat::Q4_KF | QuantFormat::Q6_K
        )
    }
}

// CPU doesn't run the full decode pipeline through ComputeBackend —
// `larql-inference` drives that path. The default `None` impls are
// the right answer here.
impl DecodeBackend for CpuBackend {}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu (BLAS + C Q4 kernel)"
    }

    fn device_info(&self) -> String {
        #[cfg(target_os = "macos")]
        {
            "macOS Accelerate AMX".to_string()
        }
        #[cfg(not(target_os = "macos"))]
        {
            "CPU BLAS".to_string()
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supports(&self, cap: Capability) -> bool {
        matches!(cap, Capability::QuantMatVec | Capability::Q4VecMat,)
    }
}

#[cfg(test)]
mod cpu_backend_tests {
    use super::*;
    use crate::backend::{Capability, ComputeBackend, MatMul, QuantMatVec};
    use crate::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_q6_k, quantize_to_q8};

    // ── MatMul ──────────────────────────────────────────────────────────

    #[test]
    fn cpu_backend_matmul_produces_correct_shape() {
        let a = Array2::<f32>::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let b = Array2::<f32>::from_shape_vec((3, 4), (0..12).map(|i| i as f32).collect()).unwrap();
        let out = CpuBackend.matmul(a.view(), b.view());
        assert_eq!(out.shape(), &[2, 4]);
        // First row of A · first col of B = 1*0 + 2*4 + 3*8 = 32
        assert_eq!(out[[0, 0]], 32.0);
    }

    #[test]
    fn cpu_backend_matmul_transb_matches_manual() {
        let a = Array2::<f32>::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        // b has shape [4, 3]; matmul_transb computes a · b.T → [2, 4]
        let b = Array2::<f32>::from_shape_vec((4, 3), (0..12).map(|i| i as f32).collect()).unwrap();
        let out = CpuBackend.matmul_transb(a.view(), b.view());
        assert_eq!(out.shape(), &[2, 4]);
        // a[0] · b[0] = 1*0 + 2*1 + 3*2 = 8
        assert_eq!(out[[0, 0]], 8.0);
    }

    // ── QuantMatVec ─────────────────────────────────────────────────────

    #[test]
    fn cpu_backend_q4_matvec_returns_some() {
        let rows = 4usize;
        let cols = 32usize; // Q4_0 block size
        let weights: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01).collect();
        let q4 = quantize_q4_0(&weights);
        let x: Vec<f32> = (0..cols).map(|j| (j as f32) * 0.02).collect();
        let (q8_x, q8_scales) = quantize_to_q8(&x);
        let out = CpuBackend
            .q4_matvec(&q4, &q8_x, &q8_scales, rows, cols)
            .expect("q4_matvec must return Some");
        assert_eq!(out.len(), rows);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cpu_backend_q4_vecmat_returns_some() {
        let intermediate = 32usize;
        let hidden = 32usize;
        let weights: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let q4 = quantize_q4_0(&weights);
        let activation: Vec<f32> = (0..intermediate).map(|j| (j as f32) * 0.02).collect();
        let out = CpuBackend
            .q4_vecmat(&activation, &q4, intermediate, hidden)
            .expect("q4_vecmat must return Some");
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cpu_backend_q4k_matvec_routes_through_fast_kernel() {
        let rows = 8usize;
        let cols = 256usize; // single Q4_K super-block
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.003).sin() * 0.1)
            .collect();
        let q4k = quantize_q4_k(&weights);
        let x: Vec<f32> = (0..cols)
            .map(|j| ((j as f32) * 0.007).cos() * 0.5)
            .collect();
        let out = CpuBackend
            .q4k_matvec(&q4k, &x, rows, cols)
            .expect("q4k_matvec must return Some");
        assert_eq!(out.len(), rows);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(
            out.iter().any(|&v| v.abs() > 1e-6),
            "non-degenerate input should produce non-zero output"
        );
    }

    #[test]
    fn cpu_backend_q6k_matvec_returns_some() {
        let rows = 4usize;
        let cols = 256usize;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.005).cos() * 0.1)
            .collect();
        let q6k = quantize_q6_k(&weights);
        let x: Vec<f32> = (0..cols)
            .map(|j| ((j as f32) * 0.009).sin() * 0.5)
            .collect();
        let out = CpuBackend
            .q6k_matvec(&q6k, &x, rows, cols)
            .expect("q6k_matvec must return Some");
        assert_eq!(out.len(), rows);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cpu_backend_q4k_dual_matvec_returns_both_outputs() {
        let rows = 8usize;
        let cols = 256usize;
        let weights_a: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.003).sin() * 0.1)
            .collect();
        let weights_b: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.004).cos() * 0.1)
            .collect();
        let q4k_a = quantize_q4_k(&weights_a);
        let q4k_b = quantize_q4_k(&weights_b);
        let x: Vec<f32> = (0..cols)
            .map(|j| ((j as f32) * 0.011).sin() * 0.5)
            .collect();
        let (out_a, out_b) = CpuBackend
            .q4k_dual_matvec(&q4k_a, &q4k_b, &x, rows, cols)
            .expect("q4k_dual_matvec must return Some");
        assert_eq!(out_a.len(), rows);
        assert_eq!(out_b.len(), rows);
        // A and B come from different weights — outputs must differ.
        let any_diff = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "different weights must produce different outputs");
    }

    #[test]
    fn cpu_backend_supports_q4_k_family_and_q6_k() {
        use crate::QuantFormat;
        assert!(CpuBackend.supports_quant(QuantFormat::Q4_0));
        assert!(CpuBackend.supports_quant(QuantFormat::Q4_K));
        assert!(CpuBackend.supports_quant(QuantFormat::Q4_KF));
        assert!(CpuBackend.supports_quant(QuantFormat::Q6_K));
        // CPU doesn't have a Q8_0 fast path; advertise honestly.
        assert!(!CpuBackend.supports_quant(QuantFormat::Q8_0));
        // Float formats are not "quant" in this trait's sense.
        assert!(!CpuBackend.supports_quant(QuantFormat::BF16));
        assert!(!CpuBackend.supports_quant(QuantFormat::F16));
        assert!(!CpuBackend.supports_quant(QuantFormat::F32));
    }

    // ── ComputeBackend identity ─────────────────────────────────────────

    #[test]
    fn cpu_backend_name_is_descriptive() {
        let name = CpuBackend.name();
        assert!(name.contains("cpu"), "name should mention 'cpu': {name}");
        assert!(name.contains("Q4"), "name should mention Q4 kernel: {name}");
    }

    #[test]
    fn cpu_backend_device_info_is_non_empty() {
        let info = CpuBackend.device_info();
        assert!(!info.is_empty());
        // On macOS we hit the Accelerate branch; elsewhere the generic BLAS branch.
        #[cfg(target_os = "macos")]
        assert!(
            info.contains("Accelerate") || info.contains("AMX"),
            "macOS device_info should mention Accelerate/AMX: {info}"
        );
    }

    #[test]
    fn cpu_backend_as_any_allows_downcast() {
        let backend: Box<dyn ComputeBackend> = Box::new(CpuBackend);
        let any = backend.as_ref().as_any();
        assert!(any.is::<CpuBackend>(), "as_any should expose CpuBackend");
    }

    #[test]
    fn cpu_backend_supports_quant_matvec_and_q4_vecmat() {
        assert!(CpuBackend.supports(Capability::QuantMatVec));
        assert!(CpuBackend.supports(Capability::Q4VecMat));
    }

    #[test]
    fn cpu_backend_does_not_claim_unsupported_caps() {
        // Capabilities the CPU backend explicitly doesn't implement.
        assert!(!CpuBackend.supports(Capability::PrefillQ4));
        assert!(!CpuBackend.supports(Capability::DecodeToken));
    }
}
