//! CUDA wrappers for the vendored RotorQuant kernels.
//!
//! The CUDA kernels available in this crate cover the device-to-device
//! FP16 → RotorQuant packed copy/quantize path for Planar3, Planar4,
//! Iso3, and Iso4, plus the matching packed RotorQuant → FP32
//! dequantize path. The safe high-level [`crate::quantize_k`] and
//! [`crate::dequantize_k`] APIs continue to use the CPU reference
//! because they operate on the portable [`crate::QuantizedKv`] host
//! format rather than the upstream packed CUDA block format.

#![cfg(feature = "cuda")]

use std::ffi::c_void;

use crate::{ffi, KvFormat, RotorQuantError};

pub use ffi::CudaStream;

/// Number of FP16 scalar values consumed by each vendored CUDA block.
pub const CUDA_BLOCK_ELEMENTS: usize = ffi::BLOCK_ELEMENTS;

/// Return the packed RotorQuant byte length produced for `elements`
/// FP16 scalars by the vendored CUDA copy/quantize kernels.
pub fn quantized_device_len_bytes(
    format: KvFormat,
    elements: usize,
) -> Result<usize, RotorQuantError> {
    if !elements.is_multiple_of(CUDA_BLOCK_ELEMENTS) {
        return Err(RotorQuantError::InvalidBuffer(format!(
            "CUDA RotorQuant input elements ({elements}) must be divisible by {CUDA_BLOCK_ELEMENTS}"
        )));
    }
    let blocks = elements / CUDA_BLOCK_ELEMENTS;
    blocks.checked_mul(ffi::block_bytes(format)).ok_or_else(|| {
        RotorQuantError::InvalidBuffer("CUDA RotorQuant byte length overflow".into())
    })
}

/// Launch the vendored FP16 → packed RotorQuant device copy/quantize kernel.
///
/// `src_device` and `dst_device` are CUDA device pointers in the active
/// context. The source contains `elements` FP16 scalars. The destination
/// must have at least [`quantized_device_len_bytes`] bytes for the same
/// format and element count.
///
/// Returns the required destination byte length for convenience.
///
/// # Safety
///
/// The caller must ensure both pointers are valid device pointers for the
/// active CUDA context, `dst_device` is writable for the returned byte
/// length, and `stream` is valid until the launched kernel completes.
pub unsafe fn copy_f16_to_quantized_device(
    format: KvFormat,
    src_device: *const c_void,
    dst_device: *mut c_void,
    elements: usize,
    stream: CudaStream,
) -> Result<usize, RotorQuantError> {
    if src_device.is_null() {
        return Err(RotorQuantError::InvalidBuffer(
            "CUDA RotorQuant source pointer is null".into(),
        ));
    }
    if dst_device.is_null() {
        return Err(RotorQuantError::InvalidBuffer(
            "CUDA RotorQuant destination pointer is null".into(),
        ));
    }
    let bytes = quantized_device_len_bytes(format, elements)?;
    ffi::copy_f16_to_quantized(format, src_device, dst_device, elements as i64, stream);
    Ok(bytes)
}

/// Launch the packed RotorQuant → FP32 CUDA dequantize kernel.
///
/// `src_device` points to the packed upstream block format produced by
/// [`copy_f16_to_quantized_device`]. `dst_device` points to `elements`
/// writable `f32` scalars in the active CUDA context.
///
/// Returns `elements` for convenience.
///
/// # Safety
///
/// The caller must ensure both pointers are valid device pointers for the
/// active CUDA context, `src_device` is readable for
/// [`quantized_device_len_bytes`] bytes, `dst_device` is writable for
/// `elements * sizeof(f32)` bytes, and `stream` is valid until the
/// launched kernel completes.
pub unsafe fn dequantize_to_f32_device(
    format: KvFormat,
    src_device: *const c_void,
    dst_device: *mut c_void,
    elements: usize,
    stream: CudaStream,
) -> Result<usize, RotorQuantError> {
    if src_device.is_null() {
        return Err(RotorQuantError::InvalidBuffer(
            "CUDA RotorQuant source pointer is null".into(),
        ));
    }
    if dst_device.is_null() {
        return Err(RotorQuantError::InvalidBuffer(
            "CUDA RotorQuant destination pointer is null".into(),
        ));
    }
    let _bytes = quantized_device_len_bytes(format, elements)?;
    ffi::dequantize_to_f32(format, src_device, dst_device, elements as i64, stream);
    Ok(elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_device_len_matches_active_formats() {
        assert_eq!(
            quantized_device_len_bytes(KvFormat::Planar3, 128).unwrap(),
            50
        );
        assert_eq!(quantized_device_len_bytes(KvFormat::Iso3, 128).unwrap(), 50);
        assert_eq!(
            quantized_device_len_bytes(KvFormat::Planar4, 128).unwrap(),
            68
        );
        assert_eq!(quantized_device_len_bytes(KvFormat::Iso4, 128).unwrap(), 68);
    }

    #[test]
    fn quantized_device_len_rejects_partial_blocks() {
        assert!(quantized_device_len_bytes(KvFormat::Iso3, 127).is_err());
    }

    #[test]
    fn copy_f16_to_quantized_device_rejects_null_pointers() {
        let err = unsafe {
            copy_f16_to_quantized_device(
                KvFormat::Iso3,
                std::ptr::null(),
                std::ptr::null_mut(),
                128,
                CudaStream::null(),
            )
        }
        .unwrap_err();
        assert!(matches!(err, RotorQuantError::InvalidBuffer(_)));
    }

    #[test]
    fn dequantize_to_f32_device_rejects_null_pointers() {
        let err = unsafe {
            dequantize_to_f32_device(
                KvFormat::Iso3,
                std::ptr::null(),
                std::ptr::null_mut(),
                128,
                CudaStream::null(),
            )
        }
        .unwrap_err();
        assert!(matches!(err, RotorQuantError::InvalidBuffer(_)));
    }
}
