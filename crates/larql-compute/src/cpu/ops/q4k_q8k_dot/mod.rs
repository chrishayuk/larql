//! Q4_K weight × Q8_K activation matrix-vector product.
//!
//! The hot path for CPU MoE on Gemma 4 26B-A4B.  Reads 144-byte Q4_K
//! super-blocks straight from the mmapped vindex (no f32 dequant cache),
//! quantises the activation once per call to Q8_K, and accumulates an
//! integer dot product per sub-block.  Math is mathematically equivalent
//! to `q4_common::q4k_matvec_into` (within Q8 quantisation noise on the
//! activation side), but avoids walking ~5.7 GB of f32 weights per token
//! at Gemma 4 26B-A4B sizes — DRAM pressure drops ~4×.
//!
//! Per llama.cpp `ggml_vec_dot_q4_K_q8_K`:
//!
//! ```text
//! per super-block (256 elements, 8 sub-blocks of 32):
//!   d_w    = f16_to_f32(block.d)        (per super-block weight scale)
//!   dmin_w = f16_to_f32(block.dmin)     (per super-block weight min-scale)
//!   d_y    = q8k.d                      (per super-block activation scale)
//!   for sb in 0..8:
//!     sc[sb] (u8 [0..63]), mn[sb] (u8 [0..63])  unpacked from the 12-byte header
//!     dot_sb = Σ_{i in 0..32} q4_nibble[i] * y_q[i]            (i32)
//!     sum_sb = Σ_{i in 0..32} y_q[i]                            (i16, precomputed)
//!     sum1 += sc[sb] * dot_sb
//!     sum2 += mn[sb] * sum_sb
//!   acc += d_w * d_y * sum1 - dmin_w * d_y * sum2
//! out[r] = acc
//! ```
//!
//! Inner kernel uses NEON `sdot` (ARMv8.2-A SDOT instruction, available on
//! Apple M1+ and most modern aarch64 chips) when detected at runtime;
//! aarch64 CPUs without dotprod use baseline NEON widening multiplies.
//! Other architectures use AVX2 when available or a scalar reference. All paths share the
//! Q8_K activation quantiser and the per-super-block aggregation math —
//! only the inner i8×i8 → i32 dot differs.

mod common;
mod dispatch;
mod gate_up;
mod gemm;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod q4k_asm;
#[cfg(target_arch = "x86_64")]
mod q4k_avx2;
#[cfg(target_arch = "aarch64")]
mod q4k_neon;
mod q4k_scalar;
mod q6k;
mod q8k_activation;
#[cfg(test)]
mod tests;

pub use common::kernel_class_summary;
#[doc(hidden)]
pub use common::unpack_scales_mins;
pub use dispatch::{q4k_q8k_matvec_into, q4k_q8k_matvec_parallel};
pub use gate_up::q4k_q8k_gate_up_into;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use gate_up::{q4k_q8k_gate_up_asm, q4k_q8k_gate_up_neon};
pub use gemm::q4k_q8k_gemm_into;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[doc(hidden)]
pub use q4k_asm::q4k_sb_sum1_asm;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use q4k_asm::{q4k_q8k_matvec_asm, q4k_q8k_matvec_asm_v2, q4k_q8k_matvec_asm_v3};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use q4k_neon::{q4k_q8k_matvec_neon, q4k_q8k_matvec_neon_2row};
pub use q4k_scalar::q4k_q8k_matvec_scalar;
pub use q6k::q6k_q8k_matvec_into;
pub use q6k::q6k_q8k_matvec_scalar;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use q6k::{q6k_q8k_matvec_asm, q6k_q8k_matvec_neon};
pub use q8k_activation::{quantize_x_to_q8k, quantize_x_to_q8k_into, Q8KActivation};
