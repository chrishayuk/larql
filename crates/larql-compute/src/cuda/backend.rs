//! `CudaBackend` — Phase-1 stub. See `mod.rs` for status.

use ndarray::{Array2, ArrayView2};

use crate::backend::{Capability, ComputeBackend, DecodeBackend, MatMul, QuantMatVec};

use super::error::CudaInitError;

/// CUDA backend handle.
///
/// Today this just records the device index and runtime version so
/// `device_info()` can report something useful; once kernel work
/// lands the struct will carry cudarc handles (driver, cuBLAS,
/// loaded modules, kernel cache).
pub struct CudaBackend {
    device_index: i32,
    device_name: String,
    runtime_version: String,
}

impl CudaBackend {
    /// Try to construct a CUDA backend on this host.
    ///
    /// Returns `Ok(self)` when a CUDA driver is reachable and at least
    /// one device is enumerable; otherwise a typed
    /// [`CudaInitError`]. Today the call is best-effort — when the
    /// `cudarc` dep is wired in (Phase 2) this becomes a real driver
    /// probe.
    pub fn new() -> Result<Self, CudaInitError> {
        // Phase-1: no-op probe so the surface compiles. Real work in
        // `cuda-f32-baseline`. The intentionally-defensive shape mirrors
        // what the production probe will look like.
        Self::new_with_index(0)
    }

    /// Same as [`Self::new`] with an explicit device index.
    pub fn new_with_index(_index: i32) -> Result<Self, CudaInitError> {
        // When the cuda feature is off in this build the `cudarc` dep
        // isn't compiled in, so this entire module is gated below at
        // crate-root level. The body here will switch to a real
        // `cudarc::driver::CudaDevice::new(index)` call in Phase 2.
        #[cfg(feature = "cuda")]
        {
            // Defer real probe to Phase 2; for now, declare success so
            // `default_backend()` can return a CUDA box and integration
            // tests can exercise the trait dispatch.
            Ok(CudaBackend {
                device_index: _index,
                device_name: "cuda-stub".to_string(),
                runtime_version: env!("CARGO_PKG_VERSION").to_string(),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaInitError::DriverMissing(
                "larql-compute built without --features cuda".into(),
            ))
        }
    }
}

// ── Trait impls — every dispatch path is a stub ─────────────────────────

impl MatMul for CudaBackend {
    fn matmul(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!(
            "CudaBackend::matmul: stub — implement in cuda-f32-baseline (cuBLAS GEMM)"
        );
    }

    fn matmul_transb(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!(
            "CudaBackend::matmul_transb: stub — implement in cuda-f32-baseline (cuBLAS GEMM)"
        );
    }
}

impl QuantMatVec for CudaBackend {
    // All methods on this trait are default-implemented (returning `None`
    // or falling back to scalar code via the helpers in
    // `cpu::ops::q4_common`). The defaults are safe even on a stub
    // backend; we override per-format quant matvec in `cuda-q4-matvec`.
}

impl DecodeBackend for CudaBackend {
    // Default impl returns `None` for `decode_token`, which lets callers
    // fall back to the per-layer matmul path automatically. Override in
    // `cuda-fused-attention`.
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn device_info(&self) -> String {
        format!(
            "cuda (device={}, name={}, build={}; stub backend, kernels not yet wired)",
            self.device_index, self.device_name, self.runtime_version
        )
    }

    fn supports(&self, cap: Capability) -> bool {
        // Phase-1: announce the umbrella `Cuda` flag so `default_backend()`
        // and downstream selection logic can identify us, but do NOT
        // claim per-kernel capabilities yet. Each follow-up change flips
        // its own flag on as the kernel lands.
        matches!(cap, Capability::Cuda)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_cuda() {
        let backend = CudaBackend::new().expect("stub init must succeed under --features cuda");
        assert_eq!(backend.name(), "cuda");
    }

    #[test]
    fn supports_cuda_capability() {
        let backend = CudaBackend::new().expect("stub init");
        assert!(backend.supports(Capability::Cuda));
        // Phase-1 stub does not yet claim kernel-level capabilities.
        assert!(!backend.supports(Capability::F32Gemv));
        assert!(!backend.supports(Capability::FlashAttentionV2));
        assert!(!backend.supports(Capability::KvCompressionRotorQuant));
    }

    #[test]
    fn device_info_mentions_stub() {
        let backend = CudaBackend::new().expect("stub init");
        assert!(backend.device_info().contains("stub"));
    }
}
