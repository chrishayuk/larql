//! # larql-rotorquant
//!
//! K/V cache compression via block-diagonal rotations followed by
//! scalar quantization. Inspired by
//! <https://github.com/scrya-com/rotorquant>.
//!
//! Two production variants land in this crate's first cut:
//!
//! - **Planar3** — per-coordinate-pair Givens rotation, 3-bit Lloyd-Max
//!   quantize. Block size 2.
//! - **Iso3** — per-quaternion-group rotation, 3-bit. Block size 4.
//!
//! 4-bit variants `Planar4` / `Iso4` use the same machinery with a
//! larger codebook; landed for parity with the upstream paper.
//!
//! ## API surface
//!
//! ```rust,no_run
//! use larql_rotorquant::{KvFormat, KvScratch, quantize_k_with_scratch, dequantize_v_with_inverse_rotation};
//!
//! let k = vec![0.0_f32; 128 * 32]; // [seq=128, head_dim=32]
//! let mut scratch = KvScratch::new(KvFormat::Iso3, 128, 32, 1).unwrap();
//! let qk = quantize_k_with_scratch(KvFormat::Iso3, &k, 128, 32, &mut scratch).unwrap();
//! // ... store qk on device or disk ...
//! ```
//!
//! ## Provenance
//!
//! The block-diagonal rotation idea, the deferred-K rule, and the
//! "V dequant must invert the forward rotation" lesson all come from
//! the upstream `feature/planarquant-kv-cache` branch of llama.cpp.
//! The safe Rust CPU reference path lives in `cpu_ref.rs` and is what
//! the GPU kernels parity-test against. Upstream CUDA kernels are
//! vendored under `cuda/upstream/`; see `UPSTREAM.md` for the exact
//! source commit and sync procedure.
//!
//! ## Layout invariant
//!
//! Quantised tensors are emitted as `QuantizedKv { codes, norms,
//! rotation_indices }` where:
//!
//! - `codes`: packed 3- or 4-bit indices, `n_rows * head_dim` codes
//!   total, packed into `Vec<u8>` row-major.
//! - `norms`: one `f32` per row (the L2 norm before normalisation).
//! - `rotation_indices`: one `u16` per (row, block) pair selecting
//!   the rotation chosen for that block. Required for V dequant.
//!
//! The whole struct is reconstructible to within cosine ≥ 0.99 of
//! the input via `dequantize_v_with_inverse_rotation`.

#[cfg(all(feature = "cuda", feature = "cuda-oxide"))]
compile_error!(
    "features `cuda` and `cuda-oxide` are mutually exclusive; enable only one RotorQuant CUDA backend"
);

mod cpu_ref;
mod error;
mod format;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub mod ffi;
#[cfg(feature = "cuda")]
pub use cuda::{
    copy_f16_to_quantized_device, dequantize_to_f32_device, quantized_device_len_bytes, CudaStream,
    CUDA_BLOCK_ELEMENTS,
};

#[cfg(feature = "cuda-oxide")]
pub mod cuda_oxide;

pub use error::RotorQuantError;
pub use format::{KvFormat, KvScratch, QuantizedKv};

/// Quantise the K cache.
///
/// `k` is row-major `[n_rows, head_dim]`. `head_dim` MUST be a
/// multiple of the format's block size (`2` for Planar*, `4` for Iso*).
///
/// Currently a CPU-side reference implementation. The CUDA path is a
/// future optimisation; this entrypoint produces the same bytes either
/// way.
pub fn quantize_k(
    format: KvFormat,
    k: &[f32],
    n_rows: usize,
    head_dim: usize,
) -> Result<QuantizedKv, RotorQuantError> {
    let mut scratch = KvScratch::new(format, n_rows.max(1), head_dim, 1)?;
    quantize_k_with_scratch(format, k, n_rows, head_dim, &mut scratch)
}

/// Quantise the K cache using caller-owned scratch state.
pub fn quantize_k_with_scratch(
    format: KvFormat,
    k: &[f32],
    n_rows: usize,
    head_dim: usize,
    scratch: &mut KvScratch,
) -> Result<QuantizedKv, RotorQuantError> {
    scratch.validate(format, n_rows, head_dim)?;
    cpu_ref::quantize(format, k, n_rows, head_dim)
}

/// Quantise the V cache. Identical machinery to `quantize_k` — the
/// distinct entry point exists so a future deferred-K path can
/// diverge their behaviour.
pub fn quantize_v(
    format: KvFormat,
    v: &[f32],
    n_rows: usize,
    head_dim: usize,
) -> Result<QuantizedKv, RotorQuantError> {
    let mut scratch = KvScratch::new(format, n_rows.max(1), head_dim, 1)?;
    quantize_v_with_scratch(format, v, n_rows, head_dim, &mut scratch)
}

/// Quantise the V cache using caller-owned scratch state.
pub fn quantize_v_with_scratch(
    format: KvFormat,
    v: &[f32],
    n_rows: usize,
    head_dim: usize,
    scratch: &mut KvScratch,
) -> Result<QuantizedKv, RotorQuantError> {
    scratch.validate(format, n_rows, head_dim)?;
    cpu_ref::quantize(format, v, n_rows, head_dim)
}

/// Dequantise K. Applies the rotation chosen at quantize time.
pub fn dequantize_k(qkv: &QuantizedKv) -> Result<Vec<f32>, RotorQuantError> {
    cpu_ref::dequantize(qkv, /* invert_rotation = */ false)
}

/// Dequantise V with the **inverse** rotation. This MUST be used for
/// V (not K) — applying the forward rotation to V on dequantize is
/// the bug upstream commit 6e5a4aa fixed (PPL went from 15K to 7.05).
///
/// The signature deliberately does not let the caller pass a rotation
/// table; the inverse is reconstructed from `qkv.rotation_indices`.
pub fn dequantize_v_with_inverse_rotation(qkv: &QuantizedKv) -> Result<Vec<f32>, RotorQuantError> {
    cpu_ref::dequantize(qkv, /* invert_rotation = */ true)
}
