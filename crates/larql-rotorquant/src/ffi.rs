//! Low-level Rust FFI bindings for the vendored RotorQuant CUDA kernels.
//!
//! This module is intentionally thin: it exposes the C ABI wrapper
//! symbols built from `cuda/larql_rotorquant_kernels.cu` and keeps
//! CUDA stream handles opaque. The safe public API in `lib.rs` remains
//! the default entry point for callers that do not manage device memory.

#![cfg(feature = "cuda")]

use std::ffi::c_void;

use crate::KvFormat;

/// Opaque CUDA stream handle.
///
/// `cudaStream_t` is a pointer-sized runtime handle. Keeping it behind
/// this newtype prevents accidental mixing with host pointers while
/// still allowing CUDA-owning callers to pass an existing stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct CudaStream(*mut c_void);

impl CudaStream {
    /// Use CUDA's default stream.
    pub const fn null() -> Self {
        Self(std::ptr::null_mut())
    }

    /// Wrap a raw `cudaStream_t`.
    ///
    /// # Safety
    ///
    /// `raw` must be either null or a valid CUDA stream for the
    /// current device context, and it must remain valid for the
    /// duration of any kernel launched with this handle.
    pub const unsafe fn from_raw(raw: *mut c_void) -> Self {
        Self(raw)
    }

    /// Return the raw `cudaStream_t`.
    pub const fn as_raw(self) -> *mut c_void {
        self.0
    }
}

unsafe extern "C" {
    fn larql_rotorquant_cpy_f16_planar3(
        src: *const c_void,
        dst: *mut c_void,
        ne: i64,
        stream: *mut c_void,
    );
    fn larql_rotorquant_cpy_f16_planar4(
        src: *const c_void,
        dst: *mut c_void,
        ne: i64,
        stream: *mut c_void,
    );
    fn larql_rotorquant_cpy_f16_iso3(
        src: *const c_void,
        dst: *mut c_void,
        ne: i64,
        stream: *mut c_void,
    );
    fn larql_rotorquant_cpy_f16_iso4(
        src: *const c_void,
        dst: *mut c_void,
        ne: i64,
        stream: *mut c_void,
    );
    fn larql_rotorquant_dequantize_planar3_f32(
        src: *const c_void,
        dst: *mut c_void,
        ne: i64,
        stream: *mut c_void,
    );
    fn larql_rotorquant_dequantize_planar4_f32(
        src: *const c_void,
        dst: *mut c_void,
        ne: i64,
        stream: *mut c_void,
    );
    fn larql_rotorquant_dequantize_iso3_f32(
        src: *const c_void,
        dst: *mut c_void,
        ne: i64,
        stream: *mut c_void,
    );
    fn larql_rotorquant_dequantize_iso4_f32(
        src: *const c_void,
        dst: *mut c_void,
        ne: i64,
        stream: *mut c_void,
    );
    fn larql_rotorquant_dequantize_cpu_ref_iso3_f32(
        codes: *const c_void,
        norms: *const c_void,
        rotation_indices: *const c_void,
        dst: *mut c_void,
        n_rows: i64,
        head_dim: i64,
        stream: *mut c_void,
    );
}

/// Number of FP16 values consumed by one upstream CUDA quantization block.
pub const BLOCK_ELEMENTS: usize = 128;

/// Bytes emitted by one upstream CUDA quantization block.
pub const fn block_bytes(format: KvFormat) -> usize {
    match format {
        KvFormat::Planar3 | KvFormat::Iso3 => 2 + (BLOCK_ELEMENTS / 4) + (BLOCK_ELEMENTS / 8),
        KvFormat::Planar4 | KvFormat::Iso4 => 68,
    }
}

/// Launch the upstream F16-to-RotorQuant copy kernel for device buffers.
///
/// `src` and `dst` must be device pointers. `ne` is the number of
/// FP16 scalar values in `src`, not a byte count. `dst` must have
/// room for `ne / 128 * block_bytes(format)` bytes.
///
/// # Safety
///
/// The caller must ensure:
///
/// - `src` and `dst` are valid device pointers in the active CUDA
///   context.
/// - `ne` is non-negative and divisible by [`BLOCK_ELEMENTS`].
/// - `dst` points to enough writable device memory for the selected
///   format.
/// - `stream` is valid for the active context.
///
/// CUDA runtime failures abort through the vendored `CUDA_CHECK`
/// compatibility shim, matching upstream llama.cpp behaviour.
pub unsafe fn copy_f16_to_quantized(
    format: KvFormat,
    src: *const c_void,
    dst: *mut c_void,
    ne: i64,
    stream: CudaStream,
) {
    debug_assert!(ne >= 0);
    debug_assert_eq!(ne as usize % BLOCK_ELEMENTS, 0);
    match format {
        KvFormat::Planar3 => larql_rotorquant_cpy_f16_planar3(src, dst, ne, stream.as_raw()),
        KvFormat::Planar4 => larql_rotorquant_cpy_f16_planar4(src, dst, ne, stream.as_raw()),
        KvFormat::Iso3 => larql_rotorquant_cpy_f16_iso3(src, dst, ne, stream.as_raw()),
        KvFormat::Iso4 => larql_rotorquant_cpy_f16_iso4(src, dst, ne, stream.as_raw()),
    }
}

/// Launch the upstream-compatible RotorQuant-to-F32 dequantize kernel.
///
/// `src` points to packed RotorQuant blocks produced by
/// [`copy_f16_to_quantized`]. `dst` points to `ne` writable `f32`
/// values. `ne` is the number of scalar values represented by `src`,
/// not a byte count.
///
/// # Safety
///
/// The caller must ensure:
///
/// - `src` and `dst` are valid device pointers in the active CUDA
///   context.
/// - `ne` is non-negative and divisible by [`BLOCK_ELEMENTS`].
/// - `src` points to `ne / 128 * block_bytes(format)` readable bytes.
/// - `dst` points to `ne * sizeof(float)` writable bytes.
/// - `stream` is valid for the active context.
pub unsafe fn dequantize_to_f32(
    format: KvFormat,
    src: *const c_void,
    dst: *mut c_void,
    ne: i64,
    stream: CudaStream,
) {
    debug_assert!(ne >= 0);
    debug_assert_eq!(ne as usize % BLOCK_ELEMENTS, 0);
    match format {
        KvFormat::Planar3 => larql_rotorquant_dequantize_planar3_f32(src, dst, ne, stream.as_raw()),
        KvFormat::Planar4 => larql_rotorquant_dequantize_planar4_f32(src, dst, ne, stream.as_raw()),
        KvFormat::Iso3 => larql_rotorquant_dequantize_iso3_f32(src, dst, ne, stream.as_raw()),
        KvFormat::Iso4 => larql_rotorquant_dequantize_iso4_f32(src, dst, ne, stream.as_raw()),
    }
}

/// Launch a CUDA C parity kernel for the portable CPU-reference Iso3
/// `QuantizedKv` layout.
///
/// This is separate from the upstream packed-block wrappers above: it
/// consumes the same `codes`, `norms`, and `rotation_indices` buffers as
/// the cuda-oxide pilot kernel, which lets the two mutually exclusive
/// cargo features be compared through the same deterministic fixture.
///
/// # Safety
///
/// The caller must ensure all pointers are valid device pointers in the
/// active CUDA context, `dst` is writable for `n_rows * head_dim` `f32`
/// values, and `stream` is valid until the launched kernel completes.
pub unsafe fn dequantize_cpu_ref_iso3_to_f32(
    codes: *const c_void,
    norms: *const c_void,
    rotation_indices: *const c_void,
    dst: *mut c_void,
    n_rows: i64,
    head_dim: i64,
    stream: CudaStream,
) {
    debug_assert!(n_rows >= 0);
    debug_assert!(head_dim >= 0);
    larql_rotorquant_dequantize_cpu_ref_iso3_f32(
        codes,
        norms,
        rotation_indices,
        dst,
        n_rows,
        head_dim,
        stream.as_raw(),
    );
}
