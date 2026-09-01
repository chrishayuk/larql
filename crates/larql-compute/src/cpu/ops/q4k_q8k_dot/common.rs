use larql_models::quant::ggml::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS};

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::q4k_asm::use_asm_kernel;

/// Q4_K super-block layout: 144 bytes per 256 values.
pub(super) const BLOCK_BYTES: usize = Q4_K_BLOCK_BYTES;
/// Number of f32 / i8 elements per Q4_K (and Q8_K) super-block.
pub(super) const ELEMS_PER_BLOCK: usize = Q4_K_BLOCK_ELEMS;
/// Number of 32-element sub-blocks per super-block.
pub(super) const SUBBLOCKS_PER_BLOCK: usize = 8;
/// Sub-block size (matches Q4_K's per-32 nibble groups).
pub(super) const SUBBLOCK_SIZE: usize = 32;

/// Operand validation for every kernel in this module is
/// [`crate::cpu::ops::KernelShapeError::check`]: the hand-asm kernels take
/// `q8k_x.qs.as_ptr()` as a bare pointer with no length of its own and no
/// per-iteration bound in the `asm!` block, so a caller-side mismatch is a
/// real unguarded OOB read. The kernels used to zero-fill and return on a
/// mismatch; they now refuse by name and leave `out` untouched.
/// One-line summary of which SIMD kernel class each Q4_K/Q6_K×Q8_K matvec
/// kernel selects **on this machine, right now** — not what the doc
/// comments claim.
///
/// The three kernels on the serving path don't all have the same SIMD
/// coverage: [`q4k_q8k_matvec_into`] has a runtime-detected AVX2 branch on
/// x86_64, but [`q4k_q8k_gate_up_into`] (the dense `/v1/walk-ffn-q8k` gate+up
/// kernel) and `q6k_q8k_matvec_into` (every Q6_K projection — attention V,
/// lm_head on typical Q4K vindexes) do not; both silently fall back to the
/// ~12–16× slower scalar reference on x86_64. A DEC run must never record a
/// number produced by an unlogged scalar fallback — call this once at
/// server startup. See docs/audits/dec-readiness-review-2026-07-22.md §1b.
pub fn kernel_class_summary() -> String {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        let variant = if use_asm_kernel() { "neon-asm" } else { "neon" };
        format!("q4k_matvec={variant} q6k_matvec={variant} q4k_gate_up={variant}")
    }
    #[cfg(target_arch = "x86_64")]
    {
        let q4k_matvec = if is_x86_feature_detected!("avx2") {
            "avx2"
        } else {
            "scalar"
        };
        // q6k_q8k_matvec_into and q4k_q8k_gate_up_into have no AVX2 branch
        // yet (2026-07-22 review) — always scalar on x86_64 regardless of
        // AVX2 availability.
        format!("q4k_matvec={q4k_matvec} q6k_matvec=scalar q4k_gate_up=scalar")
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        target_arch = "x86_64"
    )))]
    {
        "q4k_matvec=scalar q6k_matvec=scalar q4k_gate_up=scalar".to_string()
    }
}

/// Unpack the 12 packed scale/min bytes at the start of a Q4_K super-block
/// into 8 6-bit scales + 8 6-bit mins.  Matches llama.cpp's
/// `get_scale_min_k4` (and `q4_common::dequantize_q4_k` / `q4k_matvec.rs`).
#[inline(always)]
#[doc(hidden)] // pub for the C12 decomposition microbench (benches/q4k_q8k_matvec.rs)
pub fn unpack_scales_mins(p: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..4 {
        scales[j] = p[j] & 0x3F;
        mins[j] = p[j + 4] & 0x3F;
        scales[j + 4] = (p[j + 8] & 0x0F) | ((p[j] >> 6) << 4);
        mins[j + 4] = (p[j + 8] >> 4) | ((p[j + 4] >> 6) << 4);
    }
    (scales, mins)
}
