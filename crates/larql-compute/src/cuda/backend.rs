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
use super::scratch::DecodeScratch;
use cudarc::driver::CudaGraph;

/// `cuda-decode-cuda-graph`: thin Send-marker wrapper around cudarc's
/// `CudaGraph`. The graph's internal `CUgraph` / `CUgraphExec` are
/// raw pointers which aren't `Send`/`Sync` by default; CUDA's driver
/// API is thread-safe for graph launches as long as the owning
/// context is bound to the calling thread, and `CudaBackend` already
/// follows that contract via `bind_to_thread`. We therefore mark the
/// wrapper `Send + Sync` so it can live in `Mutex<Option<…>>`.
pub(crate) struct DecodeGraph(pub(crate) CudaGraph);
unsafe impl Send for DecodeGraph {}
unsafe impl Sync for DecodeGraph {}

pub struct CudaBackend {
    drv: Arc<Driver>,
    pub(crate) kv_cache: Mutex<Option<CudaKvCache>>,
    // `cuda-decode-cuda-graph`: caches store `Arc<CudaSlice>` so the
    // captured-graph decode path can hold cheap clones of the
    // cached device pointers across the capture region without
    // serialising on the cache locks. Cache values are immutable
    // for the lifetime of the model — Arc::clone is the right
    // primitive.
    q4k_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<u8>>>>,
    q6k_f32_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<f32>>>>,
    q6k_packed_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<u8>>>>,
    q4k_f32_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<f32>>>>,
    /// Per-pointer cache of small f32 weights (norms etc.) so the
    /// per-layer norm-weight htod's collapse to a one-time upload per
    /// host buffer. Keyed by host pointer + length + content hash so
    /// distinct host buffers with identical bytes still collide
    /// safely.
    f32_norm_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<f32>>>>,
    /// `cuda-decode-cuda-graph`: pre-allocated per-decode scratch
    /// buffers. `None` until the first decode_token_device call sees
    /// a shape; reused on every subsequent same-shape call.
    pub(crate) decode_scratch: Mutex<Option<DecodeScratch>>,
    /// Captured decode graph + the warmup-call counter. The graph is
    /// captured on the call AFTER the scratch is first populated (so
    /// every weight/norm cache is already warm and the captured graph
    /// holds no htod nodes). Subsequent calls replay the graph.
    pub(crate) decode_graph: Mutex<Option<DecodeGraph>>,
    /// Number of decode_token_device calls since scratch was last
    /// (re)allocated. The capture happens at call #2.
    pub(crate) decode_warmup_count: Mutex<u32>,
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
            q6k_packed_device_cache: Mutex::new(HashMap::new()),
            q4k_f32_device_cache: Mutex::new(HashMap::new()),
            f32_norm_device_cache: Mutex::new(HashMap::new()),
            decode_scratch: Mutex::new(None),
            decode_graph: Mutex::new(None),
            decode_warmup_count: Mutex::new(0),
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
        let arc = self.arc_q4k_device_buf(host)?;
        f(&arc)
    }

    /// `cuda-decode-cuda-graph`: Arc-cloned device buffer for the
    /// graph-capture path. Holds no lock once it returns — the Arc
    /// keeps the cached buffer alive even if the cache is later
    /// re-locked.
    pub(crate) fn arc_q4k_device_buf(
        &self,
        host: &[u8],
    ) -> Result<Arc<CudaSlice<u8>>, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q4k_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q4k device cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return Ok(Arc::clone(arc));
            }
        }
        let arc = Arc::new(self.drv.device_u8_buf_from(host)?);
        let mut cache = self
            .q4k_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q4k device cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry))
    }

    /// Per-pointer cache of *packed* Q6_K weight bytes on the
    /// device. Parallel to `with_q4k_device_buf`. First call htod's
    /// the packed 210 B/super-block stream; later calls borrow.
    /// Used by the Q6_K mmvq path (`cuda-q6k-mmvq`).
    pub(crate) fn with_q6k_packed_device_buf<R>(
        &self,
        host: &[u8],
        f: impl FnOnce(&CudaSlice<u8>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let arc = self.arc_q6k_packed_device_buf(host)?;
        f(&arc)
    }

    pub(crate) fn arc_q6k_packed_device_buf(
        &self,
        host: &[u8],
    ) -> Result<Arc<CudaSlice<u8>>, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q6k_packed_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q6k packed cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return Ok(Arc::clone(arc));
            }
        }
        let arc = Arc::new(self.drv.device_u8_buf_from(host)?);
        let mut cache = self
            .q6k_packed_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q6k packed cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry))
    }

    /// Per-pointer cache of *dequantised* f32 Q4_K weights on the
    /// device. `cuda-prefill-batched-q4k` uses this for cuBLAS GEMM
    /// during prefill — re-reading 9.6 GB of f32 weights is faster
    /// than per-call dequant. Decode still uses the packed cache via
    /// `with_q4k_device_buf` + mmvq.
    pub(crate) fn with_q4k_f32_device_buf<R>(
        &self,
        host: &[u8],
        n_elements: usize,
        f: impl FnOnce(&CudaSlice<f32>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q4k_f32_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q4k f32 cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return f(arc);
            }
        }
        let w = dequant::dequant_q4_k(host, n_elements)
            .map_err(|e| CudaInitError::DriverMissing(format!("q4k dequant: {e:?}")))?;
        let arc = Arc::new(self.drv.device_buf_from(&w)?);
        let mut cache = self
            .q4k_f32_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q4k f32 cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        let arc = Arc::clone(entry);
        drop(cache);
        f(&arc)
    }

    pub(crate) fn with_q6k_f32_device_buf<R>(
        &self,
        host: &[u8],
        n_elements: usize,
        f: impl FnOnce(&CudaSlice<f32>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q6k_f32_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q6k device cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return f(arc);
            }
        }
        let w = dequant::dequant_q6_k(host, n_elements)
            .map_err(|e| CudaInitError::DriverMissing(format!("q6k dequant: {e:?}")))?;
        let arc = Arc::new(self.drv.device_buf_from(&w)?);
        let mut cache = self
            .q6k_f32_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q6k device cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        let arc = Arc::clone(entry);
        drop(cache);
        f(&arc)
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
        let arc = self.arc_norm_device_buf(host)?;
        f(&arc)
    }

    pub(crate) fn arc_norm_device_buf(
        &self,
        host: &[f32],
    ) -> Result<Arc<CudaSlice<f32>>, CudaInitError> {
        // SAFETY: f32 has no padding; reinterpreting as bytes for a
        // hash key is well-defined.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, std::mem::size_of_val(host))
        };
        let key = DeviceBytesKey::from_slice(bytes);
        {
            let cache = self
                .f32_norm_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("f32 norm cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return Ok(Arc::clone(arc));
            }
        }
        let arc = Arc::new(self.drv.device_buf_from(host)?);
        let mut cache = self
            .f32_norm_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("f32 norm cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry))
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

    /// `cuda-decode-cuda-graph` viability probe: confirms that
    /// cudarc 0.19's stream capture / graph replay mechanism actually
    /// works for our usage pattern (NVRTC-compiled kernel launched via
    /// `launch_builder`). If this test fails, the broader change is
    /// dead in the water and we'd skip the multi-day refactor.
    #[test]
    fn cuda_graph_capture_replay_smoke_test() {
        use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use cudarc::nvrtc::compile_ptx;

        if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
            return;
        }
        let Ok(backend) = CudaBackend::new() else {
            return;
        };
        // `Driver::new_with_index` already disabled event tracking
        // and switched to a non-default stream — both prerequisites
        // for CUDA Graph capture (CUDA_ERROR_STREAM_CAPTURE_ISOLATION
        // otherwise). Capture on the same stream that owns the
        // buffers.
        let ctx = &backend.driver().ctx;
        let stream = backend.driver().stream.clone();

        let src = r#"
extern "C" __global__ void axpb(const float* x, float* y, int n, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b;
}
"#;
        let ptx = compile_ptx(src).expect("compile axpb");
        let module = ctx.load_module(ptx).expect("load axpb module");
        let func = module.load_function("axpb").expect("load axpb function");

        let n = 1024_usize;
        let x_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x_dev = stream.clone_htod(&x_host).expect("htod");
        let mut y_dev = unsafe { stream.alloc::<f32>(n).expect("alloc y") };
        // Drain pending alloc/htod work BEFORE entering capture mode —
        // CUDA_ERROR_STREAM_CAPTURE_ISOLATION otherwise.
        stream.synchronize().expect("pre-capture sync");

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let a = 2.0_f32;
        let b = 1.0_f32;

        // Compute the reference on host (avoid any cross-stream dep
        // confusion from a prior on-device launch).
        let y_ref: Vec<f32> = x_host.iter().map(|&xi| a * xi + b).collect();

        // ── Capture ────────────────────────────────
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .expect("begin_capture");
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&x_dev)
                .arg(&mut y_dev)
                .arg(&n_i)
                .arg(&a)
                .arg(&b)
                .launch(cfg)
                .expect("launch in capture");
        }
        let graph = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .expect("end_capture")
            .expect("graph should not be empty");

        // Trash the output buffer to make sure replay actually does work.
        stream.memset_zeros(&mut y_dev).expect("memset");
        stream.synchronize().expect("sync");

        // ── Replay ──────────────────────────────────
        graph.launch().expect("graph launch");
        stream.synchronize().expect("sync");
        let y_replay = stream.clone_dtoh(&y_dev).expect("dtoh replay");

        let max_diff = y_ref
            .iter()
            .zip(&y_replay)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "graph replay drift {max_diff} > 1e-6 — capture mechanism is broken"
        );
    }
}
