//! `CudaBackend` — owns the [`Driver`] and dispatches to the per-kernel
//! wrappers in `cuda::matmul`. The kernel surface is filled in across
//! the [`cuda-and-rotorquant-kv`][parent] sub-changes; this module's
//! current state is `cuda-f32-baseline`.
//!
//! [parent]: ../../../../openspec/changes/cuda-and-rotorquant-kv/

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use cudarc::driver::CudaSlice;
use ndarray::{Array2, ArrayView2};

use crate::backend::{Capability, ComputeBackend, MatMul};

use super::decode::CudaKvCache;
use super::dequant;
use super::driver::Driver;
use super::error::CudaInitError;
use super::matmul as kernels;

pub struct CudaBackend {
    drv: Arc<Driver>,
    pub(crate) kv_cache: Mutex<Option<CudaKvCache>>,
    q4k_device_cache: Mutex<HashMap<DeviceBytesKey, CudaSlice<u8>>>,
    q6k_f32_device_cache: Mutex<HashMap<DeviceBytesKey, CudaSlice<f32>>>,
    /// Per-pointer cache of small f32 weights (norms etc.) so the
    /// per-layer norm-weight htod's collapse to a one-time upload per
    /// host buffer. Keyed by host pointer + length + content hash so
    /// distinct host buffers with identical bytes still collide
    /// safely.
    f32_norm_device_cache: Mutex<HashMap<DeviceBytesKey, CudaSlice<f32>>>,
}

impl CudaBackend {
    pub fn new() -> Result<Self, CudaInitError> {
        Self::new_with_index(0)
    }

    pub fn new_with_index(ordinal: usize) -> Result<Self, CudaInitError> {
        let drv = Driver::new_with_index(ordinal)?;
        Ok(CudaBackend {
            drv,
            kv_cache: Mutex::new(None),
            q4k_device_cache: Mutex::new(HashMap::new()),
            q6k_f32_device_cache: Mutex::new(HashMap::new()),
            f32_norm_device_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Module-internal accessor used by `cuda::attn` so the helper
    /// can borrow the driver without exposing it crate-wide.
    pub(crate) fn driver(&self) -> &Driver {
        &self.drv
    }

    pub(crate) fn with_q4k_device_buf<R>(
        &self,
        host: &[u8],
        f: impl FnOnce(&CudaSlice<u8>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        let mut cache = self
            .q4k_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q4k device cache poisoned".into()))?;
        if !cache.contains_key(&key) {
            let dev = self.drv.device_u8_buf_from(host)?;
            cache.insert(key, dev);
        }
        let dev = cache
            .get(&key)
            .ok_or_else(|| CudaInitError::DriverMissing("q4k cache insert failed".into()))?;
        f(dev)
    }

    pub(crate) fn with_q6k_f32_device_buf<R>(
        &self,
        host: &[u8],
        n_elements: usize,
        f: impl FnOnce(&CudaSlice<f32>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        let mut cache = self
            .q6k_f32_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q6k device cache poisoned".into()))?;
        if !cache.contains_key(&key) {
            let w = dequant::dequant_q6_k(host, n_elements)
                .map_err(|e| CudaInitError::DriverMissing(format!("q6k dequant: {e:?}")))?;
            let dev = self.drv.device_buf_from(&w)?;
            cache.insert(key, dev);
        }
        let dev = cache
            .get(&key)
            .ok_or_else(|| CudaInitError::DriverMissing("q6k cache insert failed".into()))?;
        f(dev)
    }

    #[doc(hidden)]
    pub fn q4k_device_cache_len(&self) -> usize {
        self.q4k_device_cache
            .lock()
            .map(|cache| cache.len())
            .unwrap_or(0)
    }

    #[doc(hidden)]
    pub fn q6k_f32_device_cache_len(&self) -> usize {
        self.q6k_f32_device_cache
            .lock()
            .map(|cache| cache.len())
            .unwrap_or(0)
    }

    // ── Device-resident projection helpers ────────────────────────────
    //
    // `cuda-decode-device-resident` Phase 1 — keep per-layer state on
    // the GPU through the projection chain. Each helper takes a
    // `CudaSlice<f32>` input and returns a `CudaSlice<f32>` output;
    // there is no implicit `htod` / `dtoh` and no `sync` between
    // launches.

    /// H2D copy of an f32 host slice. Thin wrapper around the
    /// crate-private `Driver::device_buf_from`.
    pub(crate) fn htod_f32(&self, host: &[f32]) -> Result<CudaSlice<f32>, CudaInitError> {
        self.drv.device_buf_from(host)
    }

    /// Cached H2D for small f32 weights (norm vectors etc.).
    /// First call htod's and stashes the device buffer; subsequent
    /// calls with the same host pointer/content return a borrow of
    /// the cached buffer via the closure. Used by the device-resident
    /// decode path to avoid re-uploading the same per-layer norm
    /// weights on every token.
    pub(crate) fn with_norm_device_buf<R>(
        &self,
        host: &[f32],
        f: impl FnOnce(&CudaSlice<f32>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        // SAFETY: f32 has no padding; reinterpreting as bytes for a
        // hash key is well-defined.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, std::mem::size_of_val(host))
        };
        let key = DeviceBytesKey::from_slice(bytes);
        let mut cache = self
            .f32_norm_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("f32 norm cache poisoned".into()))?;
        if !cache.contains_key(&key) {
            let dev = self.drv.device_buf_from(host)?;
            cache.insert(key, dev);
        }
        let dev = cache
            .get(&key)
            .ok_or_else(|| CudaInitError::DriverMissing("f32 norm cache insert failed".into()))?;
        f(dev)
    }

    /// D2H copy. Synchronises the stream first.
    pub(crate) fn dtoh_f32(&self, dev: &CudaSlice<f32>) -> Result<Vec<f32>, CudaInitError> {
        self.drv.sync()?;
        self.drv.to_host(dev)
    }

    /// Allocate an uninitialised device buffer.
    pub(crate) fn alloc_f32(&self, len: usize) -> Result<CudaSlice<f32>, CudaInitError> {
        self.drv.device_alloc(len)
    }

    /// H2D copy a host slice into an existing device buffer at the
    /// given element offset. `cuda-decode-device-resident` Phase 3
    /// uses this to write a single K/V row into the persistent
    /// device-resident KV cache slab without reallocating.
    pub(crate) fn htod_into_slice(
        &self,
        src: &[f32],
        dst: &mut CudaSlice<f32>,
        offset: usize,
    ) -> Result<(), CudaInitError> {
        if offset + src.len() > dst.len() {
            return Err(CudaInitError::DriverMissing(format!(
                "htod_into_slice OOB: offset {offset} + len {} > dst {}",
                src.len(),
                dst.len(),
            )));
        }
        let mut sub = dst.slice_mut(offset..offset + src.len());
        self.drv
            .stream
            .memcpy_htod(src, &mut sub)
            .map_err(|e| CudaInitError::DriverMissing(format!("memcpy_htod into slice: {e:?}")))
    }

    /// Q4_K matvec, device input + device output.
    pub(crate) fn q4k_matvec_device(
        &self,
        q4k_data: &[u8],
        x_dev: &CudaSlice<f32>,
        rows: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>, CudaInitError> {
        super::q4k_direct::matvec_device(self, q4k_data, x_dev, rows, hidden)
    }

    /// Q6_K matvec, device input + device output. Goes through the
    /// dequantised-f32 device cache and a cuBLAS gemv.
    pub(crate) fn q6k_matvec_device(
        &self,
        q6k_data: &[u8],
        x_dev: &CudaSlice<f32>,
        rows: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>, CudaInitError> {
        if x_dev.len() != hidden {
            return Err(CudaInitError::DriverMissing(format!(
                "q6k_matvec_device input mismatch: x_dev.len={} hidden={hidden}",
                x_dev.len(),
            )));
        }
        self.with_q6k_f32_device_buf(q6k_data, rows * hidden, |w_dev| {
            kernels::gemv_device_inout(self.driver(), w_dev, x_dev, rows, hidden)
        })
    }

    /// Q4_KF matvec, device input + device output. Q4_KF is rare on
    /// production vindexes so this path keeps the host-dequant + cuBLAS
    /// shape; the dequant trip happens once per call.
    pub(crate) fn q4kf_matvec_device(
        &self,
        q4kf_data: &[u8],
        x_dev: &CudaSlice<f32>,
        rows: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>, CudaInitError> {
        if x_dev.len() != hidden {
            return Err(CudaInitError::DriverMissing(format!(
                "q4kf_matvec_device input mismatch: x_dev.len={} hidden={hidden}",
                x_dev.len(),
            )));
        }
        let w = dequant::dequant_q4_kf(q4kf_data, rows * hidden)
            .map_err(|e| CudaInitError::DriverMissing(format!("q4kf dequant: {e:?}")))?;
        let w_dev = self.drv.device_buf_from(&w)?;
        kernels::gemv_device_inout(self.driver(), &w_dev, x_dev, rows, hidden)
    }

    /// f32 GEMV, device input + device output.
    pub(crate) fn f32_gemv_device(
        &self,
        w_dev: &CudaSlice<f32>,
        x_dev: &CudaSlice<f32>,
        rows: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>, CudaInitError> {
        kernels::gemv_device_inout(self.driver(), w_dev, x_dev, rows, hidden)
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DeviceBytesKey {
    ptr: usize,
    len: usize,
    head: u64,
    tail: u64,
}

impl DeviceBytesKey {
    fn from_slice(bytes: &[u8]) -> Self {
        fn read_u64(bytes: &[u8]) -> u64 {
            let mut out = [0u8; 8];
            let n = bytes.len().min(out.len());
            out[..n].copy_from_slice(&bytes[..n]);
            u64::from_le_bytes(out)
        }

        let tail_start = bytes.len().saturating_sub(8);
        Self {
            ptr: bytes.as_ptr() as usize,
            len: bytes.len(),
            head: read_u64(bytes),
            tail: read_u64(&bytes[tail_start..]),
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

        Array2::from_shape_vec((m, n), out).expect("CudaBackend::matmul: shape mismatch on result")
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

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn device_info(&self) -> String {
        self.drv.device_info()
    }

    fn supports(&self, cap: Capability) -> bool {
        if cap == Capability::CudaOxide {
            return cfg!(feature = "cuda-oxide");
        }
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
                | Capability::KvCompressionRotorQuant
                | Capability::DecodeToken
                | Capability::PrefillQ4
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
            assert!(b.supports(Capability::DecodeToken));
        }
    }

    #[test]
    fn supports_q4_matvec_after_q4_baseline() {
        if let Ok(b) = CudaBackend::new() {
            // Capabilities flipped on by cuda-q4-matvec.
            assert!(b.supports(Capability::QuantMatVec));
            assert!(b.supports(Capability::Q4VecMat));
            assert!(b.supports(Capability::KvCompressionRotorQuant));
            assert!(b.supports(Capability::DecodeToken));
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
            assert!(b.supports(Capability::DecodeToken));
            assert!(b.supports(Capability::PrefillQ4));
            assert!(b.supports(Capability::KvCompressionRotorQuant));
        }
    }

    #[test]
    fn supports_decode_after_cuda_decode_backend() {
        if let Ok(b) = CudaBackend::new() {
            assert!(b.supports(Capability::DecodeToken));
            assert!(b.supports(Capability::PrefillQ4));
        }
    }

    #[test]
    fn supports_cuda_oxide_when_feature_enabled() {
        if let Ok(b) = CudaBackend::new() {
            assert_eq!(
                b.supports(Capability::CudaOxide),
                cfg!(feature = "cuda-oxide")
            );
        }
    }
}
