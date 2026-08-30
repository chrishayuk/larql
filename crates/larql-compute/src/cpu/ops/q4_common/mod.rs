//! Shared Q4 utilities for CPU backend.
//!
//! C FFI declarations for the vdotq_s32 kernel (csrc/q4_dot.c)
//! and Q8 quantization helper.

mod dequant;
mod dual;
mod f16;
mod matmul;
mod matvec_f32;
mod quantize;

#[cfg(test)]
mod neon_tests;
#[cfg(test)]
mod tests;

extern "C" {
    /// C kernel: Q4_0 × Q8_0 matrix-vector multiply with ARM vdotq_s32.
    pub fn q4_0_matvec_c(
        q4_data: *const u8,
        q8_x: *const i8,
        q8_scales: *const f32,
        scores: *mut f32,
        num_rows: usize,
        hidden: usize,
    );

    /// C kernel: Q4_0 vector-matrix multiply (scatter-accumulate).
    pub fn q4_0_vecmat_c(
        activation: *const f32,
        q4_data: *const u8,
        out: *mut f32,
        intermediate: usize,
        hidden: usize,
    );
}

pub use dequant::dequantize_q4_k;
pub use dual::q4k_dual_matvec_into;
pub use f16::f16_to_f32;
pub(crate) use f16::f32_to_f16;
pub use matmul::{q4k_matmul_into, q6k_matmul_into};
pub use matvec_f32::q4k_matvec_into;
pub use quantize::{
    q4k_to_q4kf, quantize_q4_0, quantize_q4_k, quantize_q4_kf, quantize_q6_k, quantize_to_q8,
};
// Canonical ggml Q8_0 codec (34-byte blocks). Re-exported rather than
// re-implemented: larql-models owns the checkpoint-side block formats,
// and a second encoder here could drift from the one GGUF ingestion
// already trusts.
pub use larql_models::quant::ggml::{dequantize_q8_0, quantize_q8_0};
