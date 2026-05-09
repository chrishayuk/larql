//! CUDA driver context + cuBLAS handle.
//!
//! Lazy-initialised on `CudaBackend::new()` and reused for every
//! subsequent kernel launch. Synchronous semantics throughout — host↔
//! device copies block until done. See `design.md` D2 / D3 for the
//! rationale; tuning is a future sub-change.

use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use super::cache;
use super::error::CudaInitError;

/// Owns the CUDA context, default stream, and cuBLAS handle for a
/// single GPU. `Arc<Driver>` is the canonical share point — the
/// backend stores one and clones it for every dispatch site.
pub struct Driver {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) blas: CudaBlas,
}

impl Driver {
    /// Initialise on device 0. Maps cudarc errors to `CudaInitError`
    /// per the parent design (no panics, typed errors).
    #[allow(dead_code)] // public convenience entry — internal callers use new_with_index.
    pub fn new() -> Result<Arc<Self>, CudaInitError> {
        Self::new_with_index(0)
    }

    /// Same as [`Self::new`] with an explicit device ordinal.
    pub fn new_with_index(ordinal: usize) -> Result<Arc<Self>, CudaInitError> {
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| CudaInitError::DriverMissing(format!("{e:?}")))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| CudaInitError::DriverMissing(format!("cuBLAS init: {e:?}")))?;

        // Best-effort: ensure the kernel-cache directory exists. Cache
        // failures don't block backend init — we'll just compile PTX
        // every run.
        cache::ensure_initialised(&ctx).ok();

        Ok(Arc::new(Driver { ctx, stream, blas }))
    }

    /// Block until the default stream has drained. Called at the end
    /// of every wrapper that returns a host buffer so the buffer is
    /// guaranteed populated.
    pub(crate) fn sync(&self) -> Result<(), CudaInitError> {
        self.stream
            .synchronize()
            .map_err(|e| CudaInitError::DriverMissing(format!("stream sync: {e:?}")))
    }

    /// Allocate a device-side `f32` buffer with `len` zero-initialised
    /// elements.
    pub(crate) fn device_alloc(&self, len: usize) -> Result<CudaSlice<f32>, CudaInitError> {
        self.stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| CudaInitError::DriverMissing(format!("device alloc: {e:?}")))
    }

    /// Allocate a device-side `f32` buffer with uninitialised contents.
    /// Cheaper than [`Self::device_alloc`] (skips the async memset)
    /// for short-lived intermediates that will be fully written before
    /// any read — matvec outputs, rms_norm outputs, etc. The caller
    /// MUST guarantee every element is written before being read; the
    /// CudaKvCache init path still uses [`Self::device_alloc`] because
    /// the kernel reads positions [0, pos) on subsequent decode calls.
    pub(crate) fn device_alloc_uninit(&self, len: usize) -> Result<CudaSlice<f32>, CudaInitError> {
        // SAFETY: callers (matvec_device, rms_norm_device, etc.) write
        // every element of the returned buffer with their kernel
        // before the caller observes it.
        unsafe { self.stream.alloc::<f32>(len) }
            .map_err(|e| CudaInitError::DriverMissing(format!("device alloc uninit: {e:?}")))
    }

    /// Copy a host slice to a fresh device buffer.
    pub(crate) fn device_buf_from(&self, host: &[f32]) -> Result<CudaSlice<f32>, CudaInitError> {
        self.stream
            .clone_htod(host)
            .map_err(|e| CudaInitError::DriverMissing(format!("htod copy: {e:?}")))
    }

    /// Copy a host byte slice to a fresh device buffer.
    pub(crate) fn device_u8_buf_from(&self, host: &[u8]) -> Result<CudaSlice<u8>, CudaInitError> {
        self.stream
            .clone_htod(host)
            .map_err(|e| CudaInitError::DriverMissing(format!("htod u8 copy: {e:?}")))
    }

    /// Copy a device buffer back to host (synchronous).
    pub(crate) fn to_host(&self, dev: &CudaSlice<f32>) -> Result<Vec<f32>, CudaInitError> {
        self.stream
            .clone_dtoh(dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("dtoh copy: {e:?}")))
    }

    /// Best-effort device name string for diagnostics.
    pub(crate) fn device_info(&self) -> String {
        // cudarc 0.19 has no high-level "device name" accessor; we
        // settle for the ordinal + a confirmation that cuBLAS is up.
        // Real device name comes from the upcoming
        // `cuda-q4-matvec` change which queries cuDevicePrimaryCtxRetain
        // for arch info.
        format!("cuda(device={}, cublas=ready)", self.ctx.ordinal())
    }
}
