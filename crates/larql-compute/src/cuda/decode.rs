//! CUDA decode backend integration.
//!
//! This is a correctness-first bridge from the full-pipeline trait to the
//! cudarc helpers that already exist for CUDA. It keeps KV state host-visible
//! for now. Q4_K projections route through the direct packed-weight CUDA
//! matvec path; other quant formats still use the correctness-first fallback.

use crate::backend::{DecodeBackend, QuantMatVec};
use crate::{Activation, FfnType, FullPipelineLayer, NormType, QuantFormat, QuantWeight};

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::backend::{CudaBackend, DecodeGraph};
use super::elem::Q8_1Buf;
use super::matmul as kernels;
use super::scratch::{DecodeScratch, DecodeScratchShape};
use super::{attn, dequant, elem, q4k_mmvq, q6k_mmvq};

/// `LARQL_CUDA_Q4K_MMVQ=0` disables the new Q4_K × Q8_1 mmvq path
/// and forces the existing f32-direct Q4_K matvec. Default behaviour
/// (`unset` or `=1`) routes Q4_K projections through mmvq.
fn q4k_mmvq_enabled() -> bool {
    std::env::var("LARQL_CUDA_Q4K_MMVQ").ok().as_deref() != Some("0")
}

/// `LARQL_CUDA_Q6K_MMVQ=0` disables the new Q6_K × Q8_1 mmvq path
/// and forces the existing f32-cached Q6_K GEMV. Default = enabled.
fn q6k_mmvq_enabled() -> bool {
    std::env::var("LARQL_CUDA_Q6K_MMVQ").ok().as_deref() != Some("0")
}

/// `LARQL_CUDA_DECODE_HOST_FALLBACK=1` forces the legacy
/// `decode_token_host_fallback` path that bounces every projection
/// through `Vec<f32>`. Used as a back-out and as the parity reference
/// for the new device-resident path.
fn host_fallback_enabled() -> bool {
    std::env::var("LARQL_CUDA_DECODE_HOST_FALLBACK")
        .ok()
        .as_deref()
        == Some("1")
}

/// `LARQL_CUDA_DECODE_PROFILE=1` enables per-section instrumentation
/// inside `decode_token_device`. Adds a `drv.sync()` at each section
/// boundary (so wall-clock time accounts for GPU work too) and
/// prints a one-line breakdown per token. Disabled by default; the
/// added syncs make the path slower than the unprofiled version.
fn decode_profile_enabled() -> bool {
    std::env::var("LARQL_CUDA_DECODE_PROFILE").ok().as_deref() == Some("1")
}

/// `LARQL_CUDA_DECODE_GRAPH=0` disables the CUDA Graph capture path
/// and forces every decode token through the per-call kernel-launch
/// `decode_token_device_legacy`. Default = enabled. Used as the
/// parity back-out for `cuda-decode-cuda-graph`.
fn decode_graph_enabled() -> bool {
    std::env::var("LARQL_CUDA_DECODE_GRAPH").ok().as_deref() != Some("0")
}

/// `LARQL_CUDA_PREFILL_TENSOR_CORES=1` routes the prefill projection
/// GEMM through the f16 / Tensor Core cuBLAS path (`hgemm` with
/// `CUBLAS_COMPUTE_32F` accumulator). Default = off because the
/// f16 cache is a one-time, per-session memory commitment in
/// addition to the existing f32 cache. `cuda-prefill-tensor-cores`.
fn prefill_tensor_cores_enabled() -> bool {
    std::env::var("LARQL_CUDA_PREFILL_TENSOR_CORES")
        .ok()
        .as_deref()
        == Some("1")
}

#[derive(Default, Debug, Clone)]
struct DecodeProfile {
    norm_cpu: std::time::Duration,
    htod: std::time::Duration,
    proj_qkv: std::time::Duration,
    attn_call: std::time::Duration,
    proj_wo: std::time::Duration,
    dtoh_attn_delta: std::time::Duration,
    proj_gate_up: std::time::Duration,
    dtoh_gate_up: std::time::Duration,
    proj_down: std::time::Duration,
    dtoh_ffn_delta: std::time::Duration,
    residual_cpu: std::time::Duration,
}

impl DecodeProfile {
    fn total(&self) -> std::time::Duration {
        self.norm_cpu
            + self.htod
            + self.proj_qkv
            + self.attn_call
            + self.proj_wo
            + self.dtoh_attn_delta
            + self.proj_gate_up
            + self.dtoh_gate_up
            + self.proj_down
            + self.dtoh_ffn_delta
            + self.residual_cpu
    }

    fn report(&self, layers: usize) {
        let total = self.total();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let pct = |d: std::time::Duration| {
            if total.is_zero() {
                0.0
            } else {
                d.as_secs_f64() / total.as_secs_f64() * 100.0
            }
        };
        eprintln!(
            "[cuda-decode-profile] token total={:.2}ms ({} layers)\n  norm_cpu       {:6.2}ms ({:4.1}%)\n  htod           {:6.2}ms ({:4.1}%)\n  proj_qkv       {:6.2}ms ({:4.1}%)\n  attn_call      {:6.2}ms ({:4.1}%)\n  proj_wo        {:6.2}ms ({:4.1}%)\n  dtoh_attn_d    {:6.2}ms ({:4.1}%)\n  proj_gate_up   {:6.2}ms ({:4.1}%)\n  dtoh_gate_up   {:6.2}ms ({:4.1}%)\n  proj_down      {:6.2}ms ({:4.1}%)\n  dtoh_ffn_d     {:6.2}ms ({:4.1}%)\n  residual_cpu   {:6.2}ms ({:4.1}%)",
            ms(total),
            layers,
            ms(self.norm_cpu), pct(self.norm_cpu),
            ms(self.htod), pct(self.htod),
            ms(self.proj_qkv), pct(self.proj_qkv),
            ms(self.attn_call), pct(self.attn_call),
            ms(self.proj_wo), pct(self.proj_wo),
            ms(self.dtoh_attn_delta), pct(self.dtoh_attn_delta),
            ms(self.proj_gate_up), pct(self.proj_gate_up),
            ms(self.dtoh_gate_up), pct(self.dtoh_gate_up),
            ms(self.proj_down), pct(self.proj_down),
            ms(self.dtoh_ffn_delta), pct(self.dtoh_ffn_delta),
            ms(self.residual_cpu), pct(self.residual_cpu),
        );
    }
}

/// Layer projections eligible for the device-resident hot path. Other
/// formats (FP16, etc.) hit the host fallback silently.
fn layer_supports_device_path(layer: &FullPipelineLayer<'_>) -> bool {
    use QuantFormat::*;
    let proj_ok = |fmt| matches!(fmt, Q4_K | Q4_KF | Q6_K);
    proj_ok(layer.wq.format)
        && proj_ok(layer.wk.format)
        && proj_ok(layer.wv.format)
        && proj_ok(layer.wo.format)
        && proj_ok(layer.gate.format)
        && proj_ok(layer.up.format)
        && proj_ok(layer.down.format)
}

/// `cuda-decode-cuda-graph`: every captured kernel has its weight
/// pointers, shape constants, and norm-flag baked in at capture time,
/// so the captured graph is only valid when the layer set is uniform
/// in those dimensions. Non-mmvq formats (FP16, Q4_KF, etc.) are
/// excluded — only Q4_K and Q6_K have `_into` matvec kernels. MoE
/// layers, FFN-remote, gated/standard mismatches, and per-layer
/// activation differences are all rejected here.
fn decode_graph_supports_layers(layers: &[FullPipelineLayer<'_>]) -> bool {
    use QuantFormat::*;
    if layers.is_empty() {
        return false;
    }
    let proj_ok = |fmt| matches!(fmt, Q4_K | Q6_K);
    let first = &layers[0];
    if first.moe.is_some()
        || first.ffn_is_remote
        || first.norm_type != NormType::RmsNorm
        || first.ffn_type != FfnType::Gated
        || first.has_v_norm
    {
        return false;
    }
    let head_dim = first.head_dim;
    let num_q_heads = first.num_q_heads;
    let num_kv_heads = first.num_kv_heads;
    let activation = first.activation;
    let has_post_norms = first.has_post_norms;
    layers.iter().all(|l| {
        l.moe.is_none()
            && !l.ffn_is_remote
            && l.norm_type == NormType::RmsNorm
            && l.ffn_type == FfnType::Gated
            && !l.has_v_norm
            && proj_ok(l.wq.format)
            && proj_ok(l.wk.format)
            && proj_ok(l.wv.format)
            && proj_ok(l.wo.format)
            && proj_ok(l.gate.format)
            && proj_ok(l.up.format)
            && proj_ok(l.down.format)
            && l.head_dim == head_dim
            && l.num_q_heads == num_q_heads
            && l.num_kv_heads == num_kv_heads
            && l.activation == activation
            && l.has_post_norms == has_post_norms
    })
}

const DEFAULT_CUDA_KV_CACHE_MAX_SEQ: usize = 4096;

/// `cuda-decode-cuda-graph`: per-layer Arc clones of every cached
/// device buffer the captured graph reads. Held for the duration of
/// the capture region (and pre-fetch) so the cache locks aren't
/// re-acquired inside the hot loop and the device pointers are
/// guaranteed to live until end_capture.
struct LayerArcs {
    input_norm: Arc<CudaSlice<f32>>,
    /// `None` when the layer has no post-attn norm (Llama style).
    post_attn_norm: Option<Arc<CudaSlice<f32>>>,
    ffn_norm: Arc<CudaSlice<f32>>,
    /// `None` when the layer has no post-ffn norm or the weight slice
    /// is empty.
    post_ffn_norm: Option<Arc<CudaSlice<f32>>>,
    /// QK-norm weights. Filled with a shared zero buffer when the
    /// layer disables QK-norm so the kernel still has a valid
    /// pointer (use_qk_norm == 0 prevents dereference).
    q_norm: Arc<CudaSlice<f32>>,
    k_norm: Arc<CudaSlice<f32>>,
    use_qk_norm: bool,

    wq: Arc<CudaSlice<u8>>,
    wq_format: QuantFormat,
    wk: Arc<CudaSlice<u8>>,
    wk_format: QuantFormat,
    wv: Arc<CudaSlice<u8>>,
    wv_format: QuantFormat,
    wo: Arc<CudaSlice<u8>>,
    wo_format: QuantFormat,
    gate: Arc<CudaSlice<u8>>,
    gate_format: QuantFormat,
    up: Arc<CudaSlice<u8>>,
    up_format: QuantFormat,
    down: Arc<CudaSlice<u8>>,
    down_format: QuantFormat,
}

/// Per-layer K/V cache storage. `cuda-decode-device-resident` Phase 3
/// switched these from `Vec<f32>` to `CudaSlice<f32>` so
/// `fused_decode_attention_device_kv` can read prior tokens and append
/// the new row without any per-call PCIe transfer.
///
/// `cuda-attn-wmma-f16kv` Phase 1 switched the storage type from
/// `CudaSlice<f32>` to `CudaSlice<half::f16>`, halving the K/V slab's
/// HBM footprint (≈ 1.3 GB → 660 MB on Gemma 3 4B at max_seq=4096)
/// and halving the per-step K/V read bandwidth in the fused attention
/// kernel. The kernel converts each cache element to f32 on read via
/// `cvt.f32.f16` and converts the new K-rotated/V-raw values to f16
/// on write via `cvt.rn.f16.f32`. This is independent of (and
/// complementary to) `larql_rotorquant`, which targets the
/// host-side inference cache with deeper 3-4 bit compression.
pub(crate) struct CudaKvLayer {
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) k: CudaSlice<half::f16>,
    pub(crate) v: CudaSlice<half::f16>,
}

pub(crate) struct CudaKvCache {
    max_seq: usize,
    len: usize,
    layers: Vec<CudaKvLayer>,
}

impl CudaKvCache {
    /// `cuda-decode-device-resident` Phase 3: allocate the K/V slabs
    /// directly on the device, zero-initialised. Each layer's slab is
    /// `max_seq × num_kv_heads × head_dim × f32`.
    ///
    /// Uses `device_alloc` (cuMemAllocAsync + memset_d8_async) for
    /// zero-init, NOT htod from a host zeros buffer — the latter
    /// pays a PCIe roundtrip (~38 ms per Gemma 3 4B-sized cache at
    /// PCIe 4.0). Device-side memset is HBM-bound at ~1.8 ms.
    fn new_device(
        backend: &CudaBackend,
        shapes: &[(usize, usize)],
        max_seq: usize,
    ) -> Result<Self, super::error::CudaInitError> {
        let drv = backend.driver();
        let layers = shapes
            .iter()
            .map(|&(num_kv_heads, head_dim)| {
                let n = max_seq * num_kv_heads * head_dim;
                // `cuda-attn-wmma-f16kv` Phase 1: zero-init f16 K/V
                // slabs on device. `alloc_zeros` writes 0x0000 to each
                // 2-byte element, which is the f16 representation of
                // +0.0 — the same semantic zero as the legacy f32
                // path's 0x00000000.
                let k = drv
                    .stream
                    .alloc_zeros::<half::f16>(n)
                    .map_err(|e| super::error::CudaInitError::DriverMissing(format!(
                        "alloc f16 K slab: {e:?}"
                    )))?;
                let v = drv
                    .stream
                    .alloc_zeros::<half::f16>(n)
                    .map_err(|e| super::error::CudaInitError::DriverMissing(format!(
                        "alloc f16 V slab: {e:?}"
                    )))?;
                Ok(CudaKvLayer {
                    num_kv_heads,
                    head_dim,
                    k,
                    v,
                })
            })
            .collect::<Result<Vec<_>, super::error::CudaInitError>>()?;
        Ok(Self {
            max_seq,
            len: 0,
            layers,
        })
    }

    /// Returns true if this cache's shapes match the requested
    /// `shapes` and `max_seq`. Used to make
    /// `preallocate_kv_cache_per_layer` idempotent — reuse the
    /// existing 1 GB-sized cache instead of re-allocating it on
    /// every prefill_start.
    fn matches_shape(&self, shapes: &[(usize, usize)], max_seq: usize) -> bool {
        self.max_seq == max_seq
            && self.layers.len() == shapes.len()
            && self
                .layers
                .iter()
                .zip(shapes)
                .all(|(got, want)| got.num_kv_heads == want.0 && got.head_dim == want.1)
    }

    fn ensure_for_layers(
        &mut self,
        backend: &CudaBackend,
        layers: &[FullPipelineLayer<'_>],
        max_seq: usize,
    ) -> Result<(), super::error::CudaInitError> {
        let shapes: Vec<(usize, usize)> = layers
            .iter()
            .map(|layer| (layer.num_kv_heads.max(1), layer.head_dim.max(1)))
            .collect();
        let mismatch = self.max_seq != max_seq
            || self.layers.len() != shapes.len()
            || self
                .layers
                .iter()
                .zip(shapes.iter())
                .any(|(got, want)| got.num_kv_heads != want.0 || got.head_dim != want.1);
        if mismatch {
            *self = Self::new_device(backend, &shapes, max_seq)?;
        }
        Ok(())
    }
}

fn dequant_weight(weight: QuantWeight<'_>, rows: usize, cols: usize) -> Option<Vec<f32>> {
    match weight.format {
        QuantFormat::Q4_0 => dequant::dequant_q4_0(weight.data, rows * cols).ok(),
        QuantFormat::Q4_K => dequant::dequant_q4_k(weight.data, rows * cols).ok(),
        QuantFormat::Q4_KF => dequant::dequant_q4_kf(weight.data, rows * cols).ok(),
        QuantFormat::Q6_K => dequant::dequant_q6_k(weight.data, rows * cols).ok(),
        QuantFormat::F32 => {
            if weight.data.len() != rows * cols * std::mem::size_of::<f32>() {
                return None;
            }
            let mut out = Vec::with_capacity(rows * cols);
            for chunk in weight.data.chunks_exact(4) {
                out.push(f32::from_le_bytes(chunk.try_into().ok()?));
            }
            Some(out)
        }
        QuantFormat::BF16 | QuantFormat::F16 | QuantFormat::Q8_0 => None,
    }
}

fn rms_norm_vec(x: &[f32], weight: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0_f32 / (mean_sq as f32 + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let w = weight.get(i).copied().unwrap_or(1.0 - offset);
            v * inv * (w + offset)
        })
        .collect()
}

fn add_in_place(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

fn activate(gate: &[f32], up: &[f32], activation: Activation) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let a = match activation {
                Activation::GeluTanh => {
                    0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044_715 * g * g * g)).tanh())
                }
                Activation::Silu => g / (1.0 + (-g).exp()),
            };
            a * u
        })
        .collect()
}

fn matvec(
    backend: &CudaBackend,
    weight: QuantWeight<'_>,
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Option<Vec<f32>> {
    if x.len() != cols {
        return None;
    }
    match weight.format {
        QuantFormat::Q4_K => return backend.q4k_matvec(weight.data, x, rows, cols),
        QuantFormat::Q4_KF => return backend.q4kf_matvec(weight.data, x, rows, cols),
        QuantFormat::Q6_K => return backend.q6k_matvec(weight.data, x, rows, cols),
        _ => {}
    }
    let w = dequant_weight(weight, rows, cols)?;
    kernels::gemv(backend.driver(), &w, x, rows, cols).ok()
}

/// Device-input / device-output matvec dispatch. Mirrors `matvec`
/// but stays on the GPU. Returns `None` for unsupported formats so
/// the caller can fall back to the host path.
/// `cuda-decode-device-resident` Phase 1.
fn matvec_device(
    backend: &CudaBackend,
    weight: QuantWeight<'_>,
    x_dev: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
) -> Option<CudaSlice<f32>> {
    if x_dev.len() != cols {
        return None;
    }
    match weight.format {
        QuantFormat::Q4_K => backend
            .q4k_matvec_device(weight.data, x_dev, rows, cols)
            .ok(),
        QuantFormat::Q4_KF => backend
            .q4kf_matvec_device(weight.data, x_dev, rows, cols)
            .ok(),
        QuantFormat::Q6_K => backend
            .q6k_matvec_device(weight.data, x_dev, rows, cols)
            .ok(),
        _ => None,
    }
}

/// Q4_K mmvq-aware matvec dispatch. If the weight is Q4_K and
/// `LARQL_CUDA_Q4K_MMVQ` is enabled and a `Q8_1Buf` is supplied,
/// routes through `q4k_mmvq::matvec_device` (INT8 SIMD via
/// `__dp4a`). Otherwise falls back to `matvec_device` (f32 direct).
/// `cuda-q4k-mmvq-int8` Phase 3.
fn matvec_device_mmvq(
    backend: &CudaBackend,
    weight: QuantWeight<'_>,
    x_dev: &CudaSlice<f32>,
    x_q8_1: Option<&Q8_1Buf>,
    rows: usize,
    cols: usize,
) -> Option<CudaSlice<f32>> {
    if let (QuantFormat::Q4_K, Some(q8)) = (weight.format, x_q8_1) {
        if q4k_mmvq_enabled() {
            return q4k_mmvq::matvec_device(backend, weight.data, q8, rows, cols).ok();
        }
    }
    if let (QuantFormat::Q6_K, Some(q8)) = (weight.format, x_q8_1) {
        if q6k_mmvq_enabled() {
            return q6k_mmvq::matvec_device(backend, weight.data, q8, rows, cols).ok();
        }
    }
    matvec_device(backend, weight, x_dev, rows, cols)
}

impl DecodeBackend for CudaBackend {
    fn has_kv_cache(&self) -> bool {
        true
    }

    fn reset_kv_cache(&self) {
        if let Ok(mut cache) = self.kv_cache.lock() {
            if let Some(cache) = cache.as_mut() {
                cache.len = 0;
            }
        }
    }

    fn kv_cache_len(&self) -> usize {
        self.kv_cache
            .lock()
            .ok()
            .and_then(|cache| cache.as_ref().map(|cache| cache.len))
            .unwrap_or(0)
    }

    fn truncate_kv_cache(&self, len: usize) {
        if let Ok(mut cache) = self.kv_cache.lock() {
            if let Some(cache) = cache.as_mut() {
                cache.len = len.min(cache.len);
            }
        }
    }

    fn preallocate_kv_cache_per_layer(&self, shapes: &[(usize, usize)], max_seq: usize) {
        if let Ok(mut cache) = self.kv_cache.lock() {
            // Idempotent: if the existing cache already matches the
            // requested shape, just reset `len` to 0 instead of
            // re-allocating the ~1 GB of K/V slabs. The bench harness
            // calls this on every prefill_start; without this guard
            // every prefill pays a fresh device alloc + memset for
            // the full max_seq cache — ~38 ms per Gemma 3 4B prefill.
            let needs_alloc = match cache.as_ref() {
                Some(existing) => !existing.matches_shape(shapes, max_seq),
                None => true,
            };
            if needs_alloc {
                *cache = CudaKvCache::new_device(self, shapes, max_seq).ok();
            } else if let Some(existing) = cache.as_mut() {
                existing.len = 0;
            }
        }
    }

    fn populate_kv_layer(
        &self,
        layer: usize,
        k_data: &[f32],
        v_data: &[f32],
        seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        let Ok(mut guard) = self.kv_cache.lock() else {
            return;
        };
        if guard.is_none() || guard.as_ref().is_some_and(|c| c.layers.len() <= layer) {
            let mut shapes = guard
                .as_ref()
                .map(|c| {
                    c.layers
                        .iter()
                        .map(|l| (l.num_kv_heads, l.head_dim))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            shapes.resize(layer + 1, (num_kv_heads, head_dim));
            let max_seq = seq_len.max(DEFAULT_CUDA_KV_CACHE_MAX_SEQ);
            *guard = CudaKvCache::new_device(self, &shapes, max_seq).ok();
        }
        let Some(cache) = guard.as_mut() else {
            return;
        };
        // Copy seq_len rows from the seeded host data into the
        // device-resident slabs at the start of the slab. Phase 3
        // replaced the per-element `copy_from_slice` with a single
        // htod into the device buffer at offset 0.
        let n = seq_len * num_kv_heads * head_dim;
        if k_data.len() < n || v_data.len() < n {
            return;
        }
        let Some(slot) = cache.layers.get_mut(layer) else {
            return;
        };
        if slot.num_kv_heads != num_kv_heads
            || slot.head_dim != head_dim
            || slot.k.len() < n
            || slot.v.len() < n
        {
            return;
        }
        if let Err(_e) = self.htod_f32_as_f16_into_slice(&k_data[..n], &mut slot.k, 0) {
            return;
        }
        if let Err(_e) = self.htod_f32_as_f16_into_slice(&v_data[..n], &mut slot.v, 0) {
            return;
        }
        cache.len = cache.len.max(seq_len);
    }

    fn decode_token(
        &self,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden || layers.is_empty() {
            return None;
        }
        // `cuda-decode-device-resident` Phase 1 — try the device path
        // first; fall back silently if any layer uses an unsupported
        // projection format. Setting LARQL_CUDA_DECODE_HOST_FALLBACK=1
        // forces the legacy path (parity reference / back-out).
        if !host_fallback_enabled()
            && layers.iter().all(|l| {
                l.norm_type == NormType::RmsNorm
                    && l.ffn_type == FfnType::Gated
                    && l.moe.is_none()
                    && !l.ffn_is_remote
                    && layer_supports_device_path(l)
            })
        {
            if let Some(out) = self.decode_token_device(
                layers,
                x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rope_base,
            ) {
                return Some(out);
            }
        }
        self.decode_token_host_fallback(
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
        )
    }

    fn prefill_q4(
        &self,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        if x.len() != seq_len * hidden {
            return None;
        }
        // `cuda-prefill-batched-q4k`: try the batched GEMM path first.
        // Falls back to the per-position decode loop on env-var
        // override or unsupported layer formats. Q4_K and Q6_K are
        // covered; other formats use the legacy path.
        let prefill_host_fallback = std::env::var("LARQL_CUDA_PREFILL_HOST_FALLBACK")
            .ok()
            .as_deref()
            == Some("1");
        let all_supported = layers.iter().all(|l| {
            l.norm_type == NormType::RmsNorm
                && l.ffn_type == FfnType::Gated
                && l.moe.is_none()
                && !l.ffn_is_remote
                && matches!(l.wq.format, QuantFormat::Q4_K | QuantFormat::Q6_K)
                && matches!(l.wk.format, QuantFormat::Q4_K | QuantFormat::Q6_K)
                && matches!(l.wv.format, QuantFormat::Q4_K | QuantFormat::Q6_K)
                && matches!(l.wo.format, QuantFormat::Q4_K | QuantFormat::Q6_K)
                && matches!(l.gate.format, QuantFormat::Q4_K | QuantFormat::Q6_K)
                && matches!(l.up.format, QuantFormat::Q4_K | QuantFormat::Q6_K)
                && matches!(l.down.format, QuantFormat::Q4_K | QuantFormat::Q6_K)
        });
        if !prefill_host_fallback && all_supported {
            if let Some(out) = self.prefill_q4_seq_device(
                layers,
                x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                seq_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rope_base,
            ) {
                return Some(out);
            }
        }
        self.reset_kv_cache();
        let mut out = Vec::with_capacity(x.len());
        for pos in 0..seq_len {
            let row = &x[pos * hidden..(pos + 1) * hidden];
            let h = self.decode_token(
                layers,
                row,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rope_base,
            )?;
            out.extend_from_slice(&h);
        }
        Some(out)
    }
}

impl CudaBackend {
    /// Legacy host-bouncing decode path. Used as a parity reference
    /// and as the runtime back-out via
    /// `LARQL_CUDA_DECODE_HOST_FALLBACK=1`. Every projection
    /// round-trips through `Vec<f32>`. Phase 3 made the K/V cache
    /// device-resident; the fallback dtoh's it into a temporary
    /// host slab before the host-input attention call and htod's
    /// the result back. This is intentionally slow — the path is
    /// for parity testing, not production.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_token_host_fallback(
        &self,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        _kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden || layers.is_empty() {
            return None;
        }
        let mut h = x.to_vec();
        let mut guard = self.kv_cache.lock().ok()?;
        if guard.is_none() {
            let shapes: Vec<(usize, usize)> = layers
                .iter()
                .map(|layer| {
                    (
                        layer.num_kv_heads.max(num_kv_heads).max(1),
                        layer.head_dim.max(head_dim).max(1),
                    )
                })
                .collect();
            *guard = CudaKvCache::new_device(self, &shapes, DEFAULT_CUDA_KV_CACHE_MAX_SEQ).ok();
        }
        let cache = guard.as_mut()?;
        cache
            .ensure_for_layers(
                self,
                layers,
                cache.max_seq.max(DEFAULT_CUDA_KV_CACHE_MAX_SEQ),
            )
            .ok()?;
        let pos = cache.len;
        if pos >= cache.max_seq {
            return None;
        }

        for (layer_idx, layer) in layers.iter().enumerate() {
            if layer.norm_type != NormType::RmsNorm
                || layer.ffn_type != FfnType::Gated
                || layer.moe.is_some()
                || layer.ffn_is_remote
            {
                return None;
            }

            let layer_head_dim = layer.head_dim.max(head_dim);
            let layer_num_q_heads = layer.num_q_heads.max(num_q_heads);
            let layer_num_kv_heads = layer.num_kv_heads.max(num_kv_heads);
            let layer_q_dim = layer_num_q_heads * layer_head_dim;
            let layer_kv_dim = layer_num_kv_heads * layer_head_dim;
            let layer_rope_base = if layer.rope_base != 0.0 {
                layer.rope_base
            } else {
                rope_base
            };
            let layer_rotary_dim = layer.rotary_dim;

            let h_attn = rms_norm_vec(&h, layer.input_norm, layer.eps, layer.norm_offset);
            let qkv = if layer.wq.format == QuantFormat::Q4_K
                || layer.wk.format == QuantFormat::Q4_K
                || layer.wv.format == QuantFormat::Q4_K
            {
                attn::QkvProjOutput {
                    q: matvec(self, layer.wq, &h_attn, layer_q_dim, hidden)?,
                    k: matvec(self, layer.wk, &h_attn, layer_kv_dim, hidden)?,
                    v: matvec(self, layer.wv, &h_attn, layer_kv_dim, hidden)?,
                }
            } else {
                let wq = dequant_weight(layer.wq, layer_q_dim, hidden)?;
                let wk = dequant_weight(layer.wk, layer_kv_dim, hidden)?;
                let wv = dequant_weight(layer.wv, layer_kv_dim, hidden)?;
                attn::qkv_rms_proj(
                    self,
                    &h,
                    layer.input_norm,
                    &wq,
                    &wk,
                    &wv,
                    attn::QkvProjDims {
                        hidden,
                        q_dim: layer_q_dim,
                        kv_dim: layer_kv_dim,
                    },
                    layer.eps,
                    layer.norm_offset,
                )
                .ok()?
            };

            let max_seq = cache.max_seq;
            let kv_slot = cache.layers.get_mut(layer_idx)?;
            // Phase 3: dtoh device cache → host vec for the legacy
            // host-input attention call, then htod the updated cache
            // back into the device buffers. Slow on purpose; this
            // path exists for parity correctness only.
            // `cuda-attn-wmma-f16kv` Phase 1: cache slabs are now f16
            // — round-trip through host with element-wise convert
            // so the legacy host attention wrapper still receives
            // f32 inputs.
            let kv_host_k = self.dtoh_f16_as_f32(&kv_slot.k).ok()?;
            let kv_host_v = self.dtoh_f16_as_f32(&kv_slot.v).ok()?;
            let attn_out = attn::fused_decode_attention(
                self,
                &qkv.q,
                &qkv.k,
                &qkv.v,
                &kv_host_k,
                &kv_host_v,
                layer.q_norm_weight,
                layer.k_norm_weight,
                attn::FusedDecodeAttentionOpts {
                    num_q_heads: layer_num_q_heads,
                    num_kv_heads: layer_num_kv_heads,
                    head_dim: layer_head_dim,
                    pos,
                    max_seq,
                    rotary_dim: layer_rotary_dim,
                    rope_base: layer_rope_base,
                    eps: layer.eps,
                    qk_norm_offset: layer.qk_norm_offset,
                    attn_scale: layer.attn_scale,
                    softcap: 0.0,
                },
            )
            .ok()?;
            // `cuda-attn-wmma-f16kv` Phase 1: cache is f16; convert
            // before write-back.
            self.htod_f32_as_f16_into_slice(&attn_out.k_cache, &mut kv_slot.k, 0)
                .ok()?;
            self.htod_f32_as_f16_into_slice(&attn_out.v_cache, &mut kv_slot.v, 0)
                .ok()?;

            let attn_delta =
                matvec(self, layer.wo, &attn_out.out, hidden, layer_q_dim).or_else(|| {
                    if q_dim != layer_q_dim {
                        None
                    } else {
                        matvec(self, layer.wo, &attn_out.out, hidden, q_dim)
                    }
                })?;
            let mut h_post_attn = h.clone();
            if layer.has_post_norms {
                let normed = rms_norm_vec(
                    &attn_delta,
                    layer.post_attn_norm,
                    layer.eps,
                    layer.norm_offset,
                );
                add_in_place(&mut h_post_attn, &normed);
            } else {
                add_in_place(&mut h_post_attn, &attn_delta);
            }

            let ffn_norm_weight = if layer.has_post_norms {
                layer.pre_ffn_norm.unwrap_or(layer.post_attn_norm)
            } else {
                layer.post_attn_norm
            };
            let h_ffn = rms_norm_vec(&h_post_attn, ffn_norm_weight, layer.eps, layer.norm_offset);
            let gate = matvec(self, layer.gate, &h_ffn, inter, hidden)?;
            let up = matvec(self, layer.up, &h_ffn, inter, hidden)?;
            let act = activate(&gate, &up, layer.activation);
            let ffn_delta = matvec(self, layer.down, &act, hidden, inter)?;
            let mut h_out = h_post_attn;
            if layer.has_post_norms {
                let post = layer.post_ffn_norm.unwrap_or(&[]);
                let normed = rms_norm_vec(&ffn_delta, post, layer.eps, layer.norm_offset);
                add_in_place(&mut h_out, &normed);
            } else {
                add_in_place(&mut h_out, &ffn_delta);
            }
            if layer.layer_scalar != 0.0 && layer.layer_scalar != 1.0 {
                for v in &mut h_out {
                    *v *= layer.layer_scalar;
                }
            }
            h = h_out;
        }

        cache.len = pos + 1;
        Some(h)
    }

    /// Device-resident decode path.
    /// `cuda-decode-device-resident` Phase 2: `h` stays on the device
    /// across the entire layer loop. RMSNorm, silu/gelu activation,
    /// residual add, and the per-layer scalar all run as their own
    /// kernels (`super::elem`). Only one H2D (initial input) and one
    /// D2H (final output) cross the bus per token, plus the small
    /// per-layer norm-weight htod's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_token_device(
        &self,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden || layers.is_empty() {
            return None;
        }

        // `cuda-decode-cuda-graph`: try the captured-graph path first.
        // Falls through to the legacy per-call kernel launch path on
        // unsupported layers (non-Q4_K/Q6_K weights, MoE, mixed
        // shapes), missing capabilities, or any error.
        if decode_graph_enabled()
            && !host_fallback_enabled()
            && decode_graph_supports_layers(layers)
        {
            if let Some(out) = self.decode_token_device_graph_attempt(
                layers,
                x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rope_base,
            ) {
                return Some(out);
            }
            // Fall through to legacy on None — the helper logs the
            // reason if `LARQL_CUDA_DECODE_GRAPH_DEBUG=1`.
        }
        let _ = kv_dim;

        let mut guard = self.kv_cache.lock().ok()?;
        if guard.is_none() {
            let shapes: Vec<(usize, usize)> = layers
                .iter()
                .map(|layer| {
                    (
                        layer.num_kv_heads.max(num_kv_heads).max(1),
                        layer.head_dim.max(head_dim).max(1),
                    )
                })
                .collect();
            *guard = CudaKvCache::new_device(self, &shapes, DEFAULT_CUDA_KV_CACHE_MAX_SEQ).ok();
        }
        let cache = guard.as_mut()?;
        cache
            .ensure_for_layers(
                self,
                layers,
                cache.max_seq.max(DEFAULT_CUDA_KV_CACHE_MAX_SEQ),
            )
            .ok()?;
        let pos = cache.len;
        if pos >= cache.max_seq {
            return None;
        }

        let profile_on = decode_profile_enabled();
        let mut prof = DecodeProfile::default();
        let sync_if_profile = |b: &CudaBackend| {
            if profile_on {
                let _ = b.driver().sync();
            }
        };

        // ── Initial H2D: input row → device-resident running residual ─
        let t = std::time::Instant::now();
        let mut h_dev = self.htod_f32(x).ok()?;
        sync_if_profile(self);
        prof.htod += t.elapsed();

        for (layer_idx, layer) in layers.iter().enumerate() {
            let layer_head_dim = layer.head_dim.max(head_dim);
            let layer_num_q_heads = layer.num_q_heads.max(num_q_heads);
            let layer_num_kv_heads = layer.num_kv_heads.max(num_kv_heads);
            let layer_q_dim = layer_num_q_heads * layer_head_dim;
            let layer_kv_dim = layer_num_kv_heads * layer_head_dim;
            let layer_rope_base = if layer.rope_base != 0.0 {
                layer.rope_base
            } else {
                rope_base
            };
            let layer_rotary_dim = layer.rotary_dim;

            // ── 1. Pre-attn norm: h_attn = rms_norm(h, input_norm) ──
            // norm weights are cached by host pointer so they htod
            // exactly once per session, not once per token.
            let t = std::time::Instant::now();
            let h_attn_dev = self
                .with_norm_device_buf(layer.input_norm, |w_dev| {
                    elem::rms_norm_device(
                        self,
                        &h_dev,
                        Some(w_dev),
                        hidden,
                        layer.eps,
                        layer.norm_offset,
                    )
                })
                .ok()?;
            sync_if_profile(self);
            prof.norm_cpu += t.elapsed();

            // ── 2. Q/K/V projections — shared Q8_1 input (Phase 3) ─
            // Quantize h_attn once per layer; share across q/k/v.
            let any_qkv_mmvq = (q4k_mmvq_enabled()
                && (layer.wq.format == QuantFormat::Q4_K
                    || layer.wk.format == QuantFormat::Q4_K
                    || layer.wv.format == QuantFormat::Q4_K))
                || (q6k_mmvq_enabled()
                    && (layer.wq.format == QuantFormat::Q6_K
                        || layer.wk.format == QuantFormat::Q6_K
                        || layer.wv.format == QuantFormat::Q6_K));
            let h_attn_q8_1 = if any_qkv_mmvq && hidden.is_multiple_of(32) {
                elem::quantize_q8_1_device(self, &h_attn_dev, hidden).ok()
            } else {
                None
            };
            let t = std::time::Instant::now();
            let q_dev = matvec_device_mmvq(
                self,
                layer.wq,
                &h_attn_dev,
                h_attn_q8_1.as_ref(),
                layer_q_dim,
                hidden,
            )?;
            let k_dev = matvec_device_mmvq(
                self,
                layer.wk,
                &h_attn_dev,
                h_attn_q8_1.as_ref(),
                layer_kv_dim,
                hidden,
            )?;
            let v_dev = matvec_device_mmvq(
                self,
                layer.wv,
                &h_attn_dev,
                h_attn_q8_1.as_ref(),
                layer_kv_dim,
                hidden,
            )?;
            sync_if_profile(self);
            prof.proj_qkv += t.elapsed();

            // ── 3. Fused decode attention (Phase 3 device KV cache) ─
            let max_seq = cache.max_seq;
            let kv_slot = cache.layers.get_mut(layer_idx)?;
            let t = std::time::Instant::now();
            let attn_out_dev = attn::fused_decode_attention_device_kv(
                self,
                &q_dev,
                &k_dev,
                &v_dev,
                &mut kv_slot.k,
                &mut kv_slot.v,
                layer.q_norm_weight,
                layer.k_norm_weight,
                attn::FusedDecodeAttentionOpts {
                    num_q_heads: layer_num_q_heads,
                    num_kv_heads: layer_num_kv_heads,
                    head_dim: layer_head_dim,
                    pos,
                    max_seq,
                    rotary_dim: layer_rotary_dim,
                    rope_base: layer_rope_base,
                    eps: layer.eps,
                    qk_norm_offset: layer.qk_norm_offset,
                    attn_scale: layer.attn_scale,
                    softcap: 0.0,
                },
            )
            .ok()?;
            sync_if_profile(self);
            prof.attn_call += t.elapsed();

            // ── 4. wo projection — Q8_1 quantize for single-use Q4/Q6_K ─
            let wo_mmvq = (q4k_mmvq_enabled() && layer.wo.format == QuantFormat::Q4_K)
                || (q6k_mmvq_enabled() && layer.wo.format == QuantFormat::Q6_K);
            let attn_out_q8_1 = if wo_mmvq && layer_q_dim.is_multiple_of(32) {
                elem::quantize_q8_1_device(self, &attn_out_dev, layer_q_dim).ok()
            } else {
                None
            };
            let t = std::time::Instant::now();
            let attn_delta_dev = matvec_device_mmvq(
                self,
                layer.wo,
                &attn_out_dev,
                attn_out_q8_1.as_ref(),
                hidden,
                layer_q_dim,
            )
            .or_else(|| {
                if q_dim != layer_q_dim {
                    None
                } else {
                    matvec_device_mmvq(
                        self,
                        layer.wo,
                        &attn_out_dev,
                        attn_out_q8_1.as_ref(),
                        hidden,
                        q_dim,
                    )
                }
            })?;
            sync_if_profile(self);
            prof.proj_wo += t.elapsed();

            // ── 5. h += norm(attn_delta) (or just attn_delta) ──────
            let t = std::time::Instant::now();
            if layer.has_post_norms {
                let normed = self
                    .with_norm_device_buf(layer.post_attn_norm, |w_dev| {
                        elem::rms_norm_device(
                            self,
                            &attn_delta_dev,
                            Some(w_dev),
                            hidden,
                            layer.eps,
                            layer.norm_offset,
                        )
                    })
                    .ok()?;
                elem::add_in_place_device(self, &mut h_dev, &normed).ok()?;
            } else {
                elem::add_in_place_device(self, &mut h_dev, &attn_delta_dev).ok()?;
            }
            sync_if_profile(self);
            prof.residual_cpu += t.elapsed();

            // ── 6. h_ffn = rms_norm(h, ffn_norm_weight) ────────────
            let ffn_norm_weight: &[f32] = if layer.has_post_norms {
                layer.pre_ffn_norm.unwrap_or(layer.post_attn_norm)
            } else {
                layer.post_attn_norm
            };
            let t = std::time::Instant::now();
            let h_ffn_dev = self
                .with_norm_device_buf(ffn_norm_weight, |w_dev| {
                    elem::rms_norm_device(
                        self,
                        &h_dev,
                        Some(w_dev),
                        hidden,
                        layer.eps,
                        layer.norm_offset,
                    )
                })
                .ok()?;
            sync_if_profile(self);
            prof.norm_cpu += t.elapsed();

            // ── 7. gate / up projections — shared Q8_1 input ───────
            let h_ffn_q8_1 = if q4k_mmvq_enabled()
                && (layer.gate.format == QuantFormat::Q4_K || layer.up.format == QuantFormat::Q4_K)
                && hidden.is_multiple_of(32)
            {
                elem::quantize_q8_1_device(self, &h_ffn_dev, hidden).ok()
            } else {
                None
            };
            let t = std::time::Instant::now();
            let gate_dev = matvec_device_mmvq(
                self,
                layer.gate,
                &h_ffn_dev,
                h_ffn_q8_1.as_ref(),
                inter,
                hidden,
            )?;
            let up_dev = matvec_device_mmvq(
                self,
                layer.up,
                &h_ffn_dev,
                h_ffn_q8_1.as_ref(),
                inter,
                hidden,
            )?;
            sync_if_profile(self);
            prof.proj_gate_up += t.elapsed();

            // ── 8. silu_gate_up_device(gate, up) ───────────────────
            let t = std::time::Instant::now();
            let gelu_tanh = matches!(layer.activation, Activation::GeluTanh);
            let act_dev =
                elem::silu_gate_up_device(self, &gate_dev, &up_dev, inter, gelu_tanh).ok()?;
            sync_if_profile(self);
            prof.norm_cpu += t.elapsed();

            // ── 9. down projection — Q8_1 quantize for Q4/Q6_K mmvq ─
            let down_mmvq = (q4k_mmvq_enabled() && layer.down.format == QuantFormat::Q4_K)
                || (q6k_mmvq_enabled() && layer.down.format == QuantFormat::Q6_K);
            let act_q8_1 = if down_mmvq && inter.is_multiple_of(32) {
                elem::quantize_q8_1_device(self, &act_dev, inter).ok()
            } else {
                None
            };
            let t = std::time::Instant::now();
            let ffn_delta_dev =
                matvec_device_mmvq(self, layer.down, &act_dev, act_q8_1.as_ref(), hidden, inter)?;
            sync_if_profile(self);
            prof.proj_down += t.elapsed();

            // ── 10. h += norm(ffn_delta) (or just ffn_delta) ───────
            let t = std::time::Instant::now();
            if layer.has_post_norms {
                let normed = match layer.post_ffn_norm {
                    Some(w) if !w.is_empty() => self
                        .with_norm_device_buf(w, |w_dev| {
                            elem::rms_norm_device(
                                self,
                                &ffn_delta_dev,
                                Some(w_dev),
                                hidden,
                                layer.eps,
                                layer.norm_offset,
                            )
                        })
                        .ok()?,
                    _ => elem::rms_norm_device(
                        self,
                        &ffn_delta_dev,
                        None,
                        hidden,
                        layer.eps,
                        layer.norm_offset,
                    )
                    .ok()?,
                };
                elem::add_in_place_device(self, &mut h_dev, &normed).ok()?;
            } else {
                elem::add_in_place_device(self, &mut h_dev, &ffn_delta_dev).ok()?;
            }
            if layer.layer_scalar != 0.0 && layer.layer_scalar != 1.0 {
                elem::scale_inplace_device(self, &mut h_dev, layer.layer_scalar).ok()?;
            }
            sync_if_profile(self);
            prof.residual_cpu += t.elapsed();
        }

        // ── Final D2H: device-resident `h` → host Vec<f32> ─────────
        let t = std::time::Instant::now();
        let h = self.dtoh_f32(&h_dev).ok()?;
        prof.dtoh_ffn_delta += t.elapsed();

        if profile_on {
            prof.report(layers.len());
        }

        cache.len = pos + 1;
        Some(h)
    }

    /// `cuda-decode-cuda-graph`: captured-graph variant of
    /// `decode_token_device`. Returns `Some(h)` on success, `None` on
    /// any unsupported configuration / capture failure. The dispatcher
    /// in `decode_token_device` falls back to the legacy per-call path.
    ///
    /// Lifecycle on a fresh shape:
    ///   call #0: lazy-allocate `DecodeScratch`; run kernels normally
    ///            so weight/norm caches are warmed; dtoh result.
    ///   call #1: still no graph; run kernels under `begin_capture` /
    ///            `end_capture`, store the resulting graph.
    ///   call ≥2: write `pos_dev` + `scratch.h`, `graph.launch()`,
    ///            sync, dtoh `scratch.h`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_token_device_graph_attempt(
        &self,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        _q_dim: usize,
        _kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

        if x.len() != hidden || layers.is_empty() {
            return None;
        }

        let first = &layers[0];
        let layer_head_dim = first.head_dim.max(head_dim).max(1);
        let layer_num_q_heads = first.num_q_heads.max(num_q_heads).max(1);
        let layer_num_kv_heads = first.num_kv_heads.max(num_kv_heads).max(1);
        let layer_q_dim = layer_num_q_heads * layer_head_dim;
        let layer_kv_dim = layer_num_kv_heads * layer_head_dim;
        if !hidden.is_multiple_of(32)
            || !layer_q_dim.is_multiple_of(32)
            || !inter.is_multiple_of(32)
        {
            return None;
        }

        let shape = DecodeScratchShape {
            hidden,
            q_dim: layer_q_dim,
            kv_dim: layer_kv_dim,
            inter,
            head_dim: layer_head_dim,
        };

        // ── Scratch allocation ────────────────────────────────────
        {
            let mut scratch_guard = self.decode_scratch.lock().ok()?;
            let need_realloc = match &*scratch_guard {
                Some(s) => !s.shape.matches(&shape),
                None => true,
            };
            if need_realloc {
                *scratch_guard = Some(DecodeScratch::allocate(self.driver(), shape).ok()?);
                *self.decode_graph.lock().ok()? = None;
                *self.decode_warmup_count.lock().ok()? = 0;
            }
        }

        // ── KV cache ──────────────────────────────────────────────
        let mut kv_guard = self.kv_cache.lock().ok()?;
        if kv_guard.is_none() {
            let shapes: Vec<(usize, usize)> = layers
                .iter()
                .map(|l| {
                    (
                        l.num_kv_heads.max(num_kv_heads).max(1),
                        l.head_dim.max(head_dim).max(1),
                    )
                })
                .collect();
            *kv_guard = CudaKvCache::new_device(self, &shapes, DEFAULT_CUDA_KV_CACHE_MAX_SEQ).ok();
        }
        let cache = kv_guard.as_mut()?;
        cache
            .ensure_for_layers(
                self,
                layers,
                cache.max_seq.max(DEFAULT_CUDA_KV_CACHE_MAX_SEQ),
            )
            .ok()?;
        let pos = cache.len;
        if pos >= cache.max_seq {
            return None;
        }

        // ── Pre-fetch all per-layer cached buffers as Arc clones ──
        // Holding these Arcs across the capture region keeps the
        // cached weight + norm device pointers alive AND stable, so
        // the captured graph's kernel args remain valid for replay.
        let zero_norm_host = vec![0.0_f32; layer_head_dim];
        let zero_norm = self.arc_norm_device_buf(&zero_norm_host).ok()?;
        let mut layer_arcs: Vec<LayerArcs> = Vec::with_capacity(layers.len());
        for layer in layers {
            let input_norm = self.arc_norm_device_buf(layer.input_norm).ok()?;
            let post_attn_norm = if layer.has_post_norms {
                Some(self.arc_norm_device_buf(layer.post_attn_norm).ok()?)
            } else {
                None
            };
            let ffn_norm_host: &[f32] = if layer.has_post_norms {
                layer.pre_ffn_norm.unwrap_or(layer.post_attn_norm)
            } else {
                layer.post_attn_norm
            };
            let ffn_norm = self.arc_norm_device_buf(ffn_norm_host).ok()?;
            let post_ffn_norm = if layer.has_post_norms {
                match layer.post_ffn_norm {
                    Some(w) if !w.is_empty() => Some(self.arc_norm_device_buf(w).ok()?),
                    _ => None,
                }
            } else {
                None
            };
            let q_norm = match layer.q_norm_weight {
                Some(w) => self.arc_norm_device_buf(w).ok()?,
                None => Arc::clone(&zero_norm),
            };
            let k_norm = match layer.k_norm_weight {
                Some(w) => self.arc_norm_device_buf(w).ok()?,
                None => Arc::clone(&zero_norm),
            };
            let use_qk_norm = layer.q_norm_weight.is_some() && layer.k_norm_weight.is_some();
            let wq = self.arc_qweight(layer.wq).ok()?;
            let wk = self.arc_qweight(layer.wk).ok()?;
            let wv = self.arc_qweight(layer.wv).ok()?;
            let wo = self.arc_qweight(layer.wo).ok()?;
            let gate = self.arc_qweight(layer.gate).ok()?;
            let up = self.arc_qweight(layer.up).ok()?;
            let down = self.arc_qweight(layer.down).ok()?;
            layer_arcs.push(LayerArcs {
                input_norm,
                post_attn_norm,
                ffn_norm,
                post_ffn_norm,
                q_norm,
                k_norm,
                use_qk_norm,
                wq,
                wq_format: layer.wq.format,
                wk,
                wk_format: layer.wk.format,
                wv,
                wv_format: layer.wv.format,
                wo,
                wo_format: layer.wo.format,
                gate,
                gate_format: layer.gate.format,
                up,
                up_format: layer.up.format,
                down,
                down_format: layer.down.format,
            });
        }

        // ── htod x → scratch.h, htod pos → scratch.pos ────────────
        let mut scratch_guard = self.decode_scratch.lock().ok()?;
        let scratch = scratch_guard.as_mut()?;
        self.htod_into_slice(x, &mut scratch.h, 0).ok()?;
        let pos_i = pos as i32;
        self.driver()
            .stream
            .memcpy_htod(std::slice::from_ref(&pos_i), &mut scratch.pos)
            .ok()?;

        // ── Replay path: graph already captured ───────────────────
        {
            let graph_guard = self.decode_graph.lock().ok()?;
            if let Some(graph) = &*graph_guard {
                graph.0.launch().ok()?;
                self.driver().sync().ok()?;
                let result = self.driver().to_host(&scratch.h).ok()?;
                cache.len = pos + 1;
                return Some(result);
            }
        }

        // ── Decide whether to capture this iteration ──────────────
        // Use call-counter: call #0 warms caches running normally;
        // call #1 records the graph. Subsequent calls hit the replay
        // branch above.
        let do_capture = {
            let mut w = self.decode_warmup_count.lock().ok()?;
            let count = *w;
            *w = count + 1;
            count == 1
        };

        // Drain any pre-capture stream work so begin_capture sees a
        // clean state — STREAM_CAPTURE_ISOLATION otherwise.
        self.driver().sync().ok()?;
        if do_capture {
            self.driver()
                .stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .ok()?;
        }

        // ── Run the per-layer pipeline using scratch + arcs ──────
        let pipeline_ok = self
            .run_decode_pipeline_into_scratch(
                scratch,
                cache,
                &layer_arcs,
                layers,
                hidden,
                inter,
                layer_q_dim,
                layer_kv_dim,
                layer_head_dim,
                layer_num_q_heads,
                layer_num_kv_heads,
                rope_base,
            )
            .is_ok();

        if do_capture {
            let graph = match self.driver().stream.end_capture(
                CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            ) {
                Ok(Some(g)) => Some(g),
                Ok(None) => {
                    eprintln!("[cuda-decode-cuda-graph] end_capture returned no graph");
                    None
                }
                Err(e) => {
                    eprintln!("[cuda-decode-cuda-graph] end_capture failed: {e:?}");
                    None
                }
            };
            if !pipeline_ok || graph.is_none() {
                // Capture surfaced an error or graph instantiation
                // failed. Reset warmup so the next call retries.
                *self.decode_warmup_count.lock().ok()? = 0;
                return None;
            }
            let graph = graph.unwrap();
            // Run the freshly-captured graph to actually compute the
            // output (capture only RECORDS — kernels haven't run).
            graph.launch().ok()?;
            *self.decode_graph.lock().ok()? = Some(DecodeGraph(graph));
        }

        if !pipeline_ok {
            return None;
        }

        self.driver().sync().ok()?;
        let result = self.driver().to_host(&scratch.h).ok()?;
        cache.len = pos + 1;
        Some(result)
    }

    /// Helper: arc-fetch a per-layer quant weight buffer. Q4_K /
    /// Q6_K route through their respective packed-bytes caches.
    fn arc_qweight(
        &self,
        weight: QuantWeight<'_>,
    ) -> Result<Arc<CudaSlice<u8>>, super::error::CudaInitError> {
        match weight.format {
            QuantFormat::Q4_K => self.arc_q4k_device_buf(weight.data),
            QuantFormat::Q6_K => self.arc_q6k_packed_device_buf(weight.data),
            other => Err(super::error::CudaInitError::DriverMissing(format!(
                "decode_graph: unsupported quant format {other:?}",
            ))),
        }
    }

    /// `cuda-decode-cuda-graph`: per-layer kernel pipeline that writes
    /// into `DecodeScratch`. Identical mathematically to the legacy
    /// `decode_token_device` body, but every output buffer is
    /// pre-allocated in the scratch and every cached weight/norm
    /// pointer comes from the pre-fetched `LayerArcs`. Safe to run
    /// either eagerly (call #0) or under `begin_capture` (call #1).
    #[allow(clippy::too_many_arguments)]
    fn run_decode_pipeline_into_scratch(
        &self,
        scratch: &mut DecodeScratch,
        cache: &mut CudaKvCache,
        layer_arcs: &[LayerArcs],
        layers: &[FullPipelineLayer<'_>],
        hidden: usize,
        inter: usize,
        layer_q_dim: usize,
        layer_kv_dim: usize,
        layer_head_dim: usize,
        layer_num_q_heads: usize,
        layer_num_kv_heads: usize,
        rope_base: f32,
    ) -> Result<(), super::error::CudaInitError> {
        for (layer_idx, layer) in layers.iter().enumerate() {
            let arcs = &layer_arcs[layer_idx];
            let layer_rope_base = if layer.rope_base != 0.0 {
                layer.rope_base
            } else {
                rope_base
            };
            let layer_rotary_dim = layer.rotary_dim;

            // 1. h_attn = rms_norm(h, input_norm)
            elem::rms_norm_device_into(
                self,
                &scratch.h,
                Some(&arcs.input_norm),
                &mut scratch.h_attn,
                hidden,
                layer.eps,
                layer.norm_offset,
            )?;

            // 2. h_attn_q8_1 = quantize(h_attn)
            elem::quantize_q8_1_device_into(
                self,
                &scratch.h_attn,
                &mut scratch.h_attn_q8_1.bytes,
                hidden,
            )?;

            // 3. q/k/v projections (Q4_K / Q6_K mmvq via Q8_1 input)
            self.proj_q8_1_into(
                arcs.wq_format,
                &arcs.wq,
                &scratch.h_attn_q8_1,
                &mut scratch.q,
                layer_q_dim,
                hidden,
            )?;
            self.proj_q8_1_into(
                arcs.wk_format,
                &arcs.wk,
                &scratch.h_attn_q8_1,
                &mut scratch.k,
                layer_kv_dim,
                hidden,
            )?;
            self.proj_q8_1_into(
                arcs.wv_format,
                &arcs.wv,
                &scratch.h_attn_q8_1,
                &mut scratch.v,
                layer_kv_dim,
                hidden,
            )?;

            // 4. Fused attention (device KV, device pos)
            let max_seq = cache.max_seq;
            let kv_slot = cache.layers.get_mut(layer_idx).ok_or_else(|| {
                super::error::CudaInitError::DriverMissing(
                    "decode_graph: kv slot missing for layer".into(),
                )
            })?;
            attn::fused_decode_attention_device_kv_into(
                self,
                &scratch.q,
                &scratch.k,
                &scratch.v,
                &mut kv_slot.k,
                &mut kv_slot.v,
                &arcs.q_norm,
                &arcs.k_norm,
                &scratch.pos,
                &mut scratch.attn_out,
                arcs.use_qk_norm,
                attn::FusedDecodeAttentionOpts {
                    num_q_heads: layer_num_q_heads,
                    num_kv_heads: layer_num_kv_heads,
                    head_dim: layer_head_dim,
                    pos: 0, // not consumed by `_into`; pos read from `scratch.pos`
                    max_seq,
                    rotary_dim: layer_rotary_dim,
                    rope_base: layer_rope_base,
                    eps: layer.eps,
                    qk_norm_offset: layer.qk_norm_offset,
                    attn_scale: layer.attn_scale,
                    softcap: 0.0,
                },
            )?;

            // 5. wo projection (attn_out → attn_delta).
            elem::quantize_q8_1_device_into(
                self,
                &scratch.attn_out,
                &mut scratch.attn_out_q8_1.bytes,
                layer_q_dim,
            )?;
            self.proj_q8_1_into(
                arcs.wo_format,
                &arcs.wo,
                &scratch.attn_out_q8_1,
                &mut scratch.attn_delta,
                hidden,
                layer_q_dim,
            )?;

            // 6. h += [norm(attn_delta) | attn_delta]
            // `cuda-fused-norm-add`: combine `rms_norm + add_in_place`
            // into one kernel — saves the `attn_normed` write+read.
            if let Some(post_attn_norm) = arcs.post_attn_norm.as_ref() {
                elem::rms_norm_add_device(
                    self,
                    &mut scratch.h,
                    &scratch.attn_delta,
                    Some(post_attn_norm),
                    hidden,
                    layer.eps,
                    layer.norm_offset,
                    1.0,
                )?;
            } else {
                elem::add_in_place_device(self, &mut scratch.h, &scratch.attn_delta)?;
            }

            // 7. h_ffn = rms_norm(h, ffn_norm)
            elem::rms_norm_device_into(
                self,
                &scratch.h,
                Some(&arcs.ffn_norm),
                &mut scratch.h_ffn,
                hidden,
                layer.eps,
                layer.norm_offset,
            )?;

            // 8. h_ffn_q8_1 = quantize(h_ffn); gate/up projections.
            elem::quantize_q8_1_device_into(
                self,
                &scratch.h_ffn,
                &mut scratch.h_ffn_q8_1.bytes,
                hidden,
            )?;
            self.proj_q8_1_into(
                arcs.gate_format,
                &arcs.gate,
                &scratch.h_ffn_q8_1,
                &mut scratch.gate,
                inter,
                hidden,
            )?;
            self.proj_q8_1_into(
                arcs.up_format,
                &arcs.up,
                &scratch.h_ffn_q8_1,
                &mut scratch.up,
                inter,
                hidden,
            )?;

            // 9. silu/gelu gate × up.
            let gelu_tanh = matches!(layer.activation, Activation::GeluTanh);
            elem::silu_gate_up_device_into(
                self,
                &scratch.gate,
                &scratch.up,
                &mut scratch.act,
                inter,
                gelu_tanh,
            )?;

            // 10. down projection (act → ffn_delta).
            elem::quantize_q8_1_device_into(
                self,
                &scratch.act,
                &mut scratch.act_q8_1.bytes,
                inter,
            )?;
            self.proj_q8_1_into(
                arcs.down_format,
                &arcs.down,
                &scratch.act_q8_1,
                &mut scratch.ffn_delta,
                hidden,
                inter,
            )?;

            // 11. h += [norm(ffn_delta) | ffn_delta] (× layer_scalar)
            // `cuda-fused-norm-add`: combine norm + add (+ optional
            // scalar multiply via `scale`) into one kernel. Saves
            // the `ffn_normed` write+read AND the separate
            // `scale_inplace` launch when `layer_scalar` is set.
            let layer_scale = if layer.layer_scalar != 0.0 && layer.layer_scalar != 1.0 {
                layer.layer_scalar
            } else {
                1.0
            };
            if layer.has_post_norms {
                let weight = arcs.post_ffn_norm.as_ref().map(|a| a.as_ref());
                elem::rms_norm_add_device(
                    self,
                    &mut scratch.h,
                    &scratch.ffn_delta,
                    weight,
                    hidden,
                    layer.eps,
                    layer.norm_offset,
                    layer_scale,
                )?;
            } else {
                elem::add_in_place_device(self, &mut scratch.h, &scratch.ffn_delta)?;
                if layer_scale != 1.0 {
                    elem::scale_inplace_device(self, &mut scratch.h, layer_scale)?;
                }
            }
        }
        Ok(())
    }

    /// Q4_K / Q6_K mmvq projection writing into a caller-provided
    /// output buffer.
    fn proj_q8_1_into(
        &self,
        format: QuantFormat,
        weight: &CudaSlice<u8>,
        x_q8_1: &Q8_1Buf,
        out: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), super::error::CudaInitError> {
        // `_into` matvec helpers expect host bytes for the cache key
        // lookup. Since we already hold a cached Arc, we go through
        // a back-door that takes the device pointer directly.
        match format {
            QuantFormat::Q4_K => {
                q4k_mmvq::matvec_device_into_with_dev(self, weight, x_q8_1, out, rows, cols)
            }
            QuantFormat::Q6_K => {
                q6k_mmvq::matvec_device_into_with_dev(self, weight, x_q8_1, out, rows, cols)
            }
            other => Err(super::error::CudaInitError::DriverMissing(format!(
                "decode_graph proj: unsupported format {other:?}",
            ))),
        }
    }

    /// Q-format-aware projection GEMM for batched prefill. Routes
    /// Q4_K and Q6_K through their respective f32 (or f16, with
    /// `LARQL_CUDA_PREFILL_TENSOR_CORES=1`) device caches (one-time
    /// dequant per session) and runs the projection as a cuBLAS
    /// `(seq_len, hidden) × (out_dim, hidden)^T → (seq_len, out_dim)`
    /// GEMM.
    ///
    /// f16 path (`cuda-prefill-tensor-cores`):
    /// 1. Convert `x_seq` (f32) → fresh f16 buffer.
    /// 2. cuBLAS hgemm against the cached f16 weight (Tensor Cores
    ///    on Ada/Ampere/Hopper).
    /// 3. Convert the f16 result → fresh f32 buffer for the rest of
    ///    the pipeline.
    fn gemm_proj_seq(
        &self,
        weight: QuantWeight<'_>,
        x_seq: &CudaSlice<f32>,
        seq_len: usize,
        out_dim: usize,
        hidden: usize,
    ) -> Option<CudaSlice<f32>> {
        let n_elements = out_dim * hidden;
        if prefill_tensor_cores_enabled() {
            let x_f16 = super::elem::f32_to_f16_device(self, x_seq).ok()?;
            let out_f16 = match weight.format {
                QuantFormat::Q4_K => self
                    .with_q4k_f16_device_buf(weight.data, n_elements, |w_dev| {
                        kernels::matmul_transb_device_inout_f16(
                            self.driver(),
                            &x_f16,
                            w_dev,
                            seq_len,
                            out_dim,
                            hidden,
                        )
                    })
                    .ok()?,
                QuantFormat::Q6_K => self
                    .with_q6k_f16_device_buf(weight.data, n_elements, |w_dev| {
                        kernels::matmul_transb_device_inout_f16(
                            self.driver(),
                            &x_f16,
                            w_dev,
                            seq_len,
                            out_dim,
                            hidden,
                        )
                    })
                    .ok()?,
                _ => return None,
            };
            return super::elem::f16_to_f32_device(self, &out_f16).ok();
        }
        match weight.format {
            QuantFormat::Q4_K => self
                .with_q4k_f32_device_buf(weight.data, n_elements, |w_dev| {
                    kernels::matmul_transb_device_inout(
                        self.driver(),
                        x_seq,
                        w_dev,
                        seq_len,
                        out_dim,
                        hidden,
                    )
                })
                .ok(),
            QuantFormat::Q6_K => self
                .with_q6k_f32_device_buf(weight.data, n_elements, |w_dev| {
                    kernels::matmul_transb_device_inout(
                        self.driver(),
                        x_seq,
                        w_dev,
                        seq_len,
                        out_dim,
                        hidden,
                    )
                })
                .ok(),
            _ => None,
        }
    }

    /// Batched prefill via cuBLAS f32 GEMM. Replaces the per-position
    /// `decode_token` loop in `prefill_q4` with a single GEMM per
    /// projection per layer; attention stays per-position because
    /// seq_len is bounded and the per-call kernel is already
    /// device-resident. `cuda-prefill-batched-q4k` Phase 1.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_q4_seq_device(
        &self,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        if x.len() != seq_len * hidden || layers.is_empty() || seq_len == 0 {
            return None;
        }

        // Fast path: seq_len=1 prefill is a single transformer pass —
        // delegate to `decode_token_device` to use the optimized mmvq
        // path instead of the f32-GEMM batched path (which is slower
        // for M=1). The bench harness in larql-inference calls
        // `prefill_q4` with seq_len=1 for the first token of every
        // generate, so this is hot.
        if seq_len == 1 {
            self.reset_kv_cache();
            return self.decode_token_device(
                layers,
                x,
                hidden,
                inter,
                q_dim,
                kv_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rope_base,
            );
        }

        // Reset KV cache for a fresh prefill.
        self.reset_kv_cache();
        let mut guard = self.kv_cache.lock().ok()?;
        if guard.is_none() {
            let shapes: Vec<(usize, usize)> = layers
                .iter()
                .map(|layer| {
                    (
                        layer.num_kv_heads.max(num_kv_heads).max(1),
                        layer.head_dim.max(head_dim).max(1),
                    )
                })
                .collect();
            *guard = CudaKvCache::new_device(self, &shapes, DEFAULT_CUDA_KV_CACHE_MAX_SEQ).ok();
        }
        let cache = guard.as_mut()?;
        cache
            .ensure_for_layers(
                self,
                layers,
                cache.max_seq.max(DEFAULT_CUDA_KV_CACHE_MAX_SEQ),
            )
            .ok()?;
        if seq_len > cache.max_seq {
            return None;
        }

        // Initial: htod the whole prompt as `[seq_len, hidden]`.
        let mut h_seq = self.htod_f32(x).ok()?;

        let prefill_profile =
            std::env::var("LARQL_CUDA_PREFILL_PROFILE").ok().as_deref() == Some("1");
        let mut t_norm = std::time::Duration::ZERO;
        let mut t_qkv = std::time::Duration::ZERO;
        let mut t_attn = std::time::Duration::ZERO;
        let mut t_wo = std::time::Duration::ZERO;
        let mut t_gate_up = std::time::Duration::ZERO;
        let mut t_silu = std::time::Duration::ZERO;
        let mut t_down = std::time::Duration::ZERO;
        let mut t_resid = std::time::Duration::ZERO;
        let sync_p = |b: &CudaBackend| {
            if prefill_profile {
                let _ = b.driver().sync();
            }
        };

        for (layer_idx, layer) in layers.iter().enumerate() {
            let layer_head_dim = layer.head_dim.max(head_dim);
            let layer_num_q_heads = layer.num_q_heads.max(num_q_heads);
            let layer_num_kv_heads = layer.num_kv_heads.max(num_kv_heads);
            let layer_q_dim = layer_num_q_heads * layer_head_dim;
            let layer_kv_dim = layer_num_kv_heads * layer_head_dim;
            let layer_rope_base = if layer.rope_base != 0.0 {
                layer.rope_base
            } else {
                rope_base
            };
            let layer_rotary_dim = layer.rotary_dim;

            // 1. Pre-attn rms_norm (batched).
            let t = std::time::Instant::now();
            let h_attn_seq = self
                .with_norm_device_buf(layer.input_norm, |w_dev| {
                    elem::rms_norm_batch_device(
                        self,
                        &h_seq,
                        Some(w_dev),
                        hidden,
                        seq_len,
                        layer.eps,
                        layer.norm_offset,
                    )
                })
                .ok()?;
            sync_p(self);
            t_norm += t.elapsed();

            // 2. QKV projections via cuBLAS f32 GEMM.
            let t = std::time::Instant::now();
            let q_seq = self.gemm_proj_seq(layer.wq, &h_attn_seq, seq_len, layer_q_dim, hidden)?;
            let k_seq = self.gemm_proj_seq(layer.wk, &h_attn_seq, seq_len, layer_kv_dim, hidden)?;
            let v_seq = self.gemm_proj_seq(layer.wv, &h_attn_seq, seq_len, layer_kv_dim, hidden)?;
            sync_p(self);
            t_qkv += t.elapsed();

            // 3. Batched attention. cuda-prefill-batched-attention:
            //    one launch writes all seq_len K/V to cache (with RoPE),
            //    a second launch computes causal Q×K^T softmax × V for
            //    every (qh, sp) pair. Falls back to the per-position
            //    loop when LARQL_CUDA_PREFILL_BATCHED_ATTN=0.
            let max_seq = cache.max_seq;
            let kv_slot = cache.layers.get_mut(layer_idx)?;
            let t = std::time::Instant::now();
            let attn_out_seq = if std::env::var("LARQL_CUDA_PREFILL_BATCHED_ATTN")
                .ok()
                .as_deref()
                != Some("0")
            {
                attn::fused_prefill_attention_seq_device(
                    self,
                    &q_seq,
                    &k_seq,
                    &v_seq,
                    &mut kv_slot.k,
                    &mut kv_slot.v,
                    layer.q_norm_weight,
                    layer.k_norm_weight,
                    0,
                    seq_len,
                    attn::FusedDecodeAttentionOpts {
                        num_q_heads: layer_num_q_heads,
                        num_kv_heads: layer_num_kv_heads,
                        head_dim: layer_head_dim,
                        pos: 0, // unused on the seq path; kernel uses base_pos+sp
                        max_seq,
                        rotary_dim: layer_rotary_dim,
                        rope_base: layer_rope_base,
                        eps: layer.eps,
                        qk_norm_offset: layer.qk_norm_offset,
                        attn_scale: layer.attn_scale,
                        softcap: 0.0,
                    },
                )
                .ok()?
            } else {
                // Back-out path: per-position fused_decode_attention loop.
                let mut q_pos = self.alloc_f32(layer_q_dim).ok()?;
                let mut k_pos = self.alloc_f32(layer_kv_dim).ok()?;
                let mut v_pos = self.alloc_f32(layer_kv_dim).ok()?;
                let mut attn_out_seq = self.alloc_f32(seq_len * layer_q_dim).ok()?;
                for pos in 0..seq_len {
                    let q_off = pos * layer_q_dim;
                    let kv_off = pos * layer_kv_dim;
                    self.driver()
                        .stream
                        .memcpy_dtod(&q_seq.slice(q_off..q_off + layer_q_dim), &mut q_pos)
                        .ok()?;
                    self.driver()
                        .stream
                        .memcpy_dtod(&k_seq.slice(kv_off..kv_off + layer_kv_dim), &mut k_pos)
                        .ok()?;
                    self.driver()
                        .stream
                        .memcpy_dtod(&v_seq.slice(kv_off..kv_off + layer_kv_dim), &mut v_pos)
                        .ok()?;
                    let attn_out_pos = attn::fused_decode_attention_device_kv(
                        self,
                        &q_pos,
                        &k_pos,
                        &v_pos,
                        &mut kv_slot.k,
                        &mut kv_slot.v,
                        layer.q_norm_weight,
                        layer.k_norm_weight,
                        attn::FusedDecodeAttentionOpts {
                            num_q_heads: layer_num_q_heads,
                            num_kv_heads: layer_num_kv_heads,
                            head_dim: layer_head_dim,
                            pos,
                            max_seq,
                            rotary_dim: layer_rotary_dim,
                            rope_base: layer_rope_base,
                            eps: layer.eps,
                            qk_norm_offset: layer.qk_norm_offset,
                            attn_scale: layer.attn_scale,
                            softcap: 0.0,
                        },
                    )
                    .ok()?;
                    self.driver()
                        .stream
                        .memcpy_dtod(
                            &attn_out_pos,
                            &mut attn_out_seq.slice_mut(q_off..q_off + layer_q_dim),
                        )
                        .ok()?;
                }
                attn_out_seq
            };
            sync_p(self);
            t_attn += t.elapsed();

            // 4. wo projection via batched GEMM.
            let t = std::time::Instant::now();
            let attn_delta_seq =
                self.gemm_proj_seq(layer.wo, &attn_out_seq, seq_len, hidden, layer_q_dim)?;
            sync_p(self);
            t_wo += t.elapsed();

            // 5. Residual + optional post-attn rms_norm.
            if layer.has_post_norms {
                let normed = self
                    .with_norm_device_buf(layer.post_attn_norm, |w_dev| {
                        elem::rms_norm_batch_device(
                            self,
                            &attn_delta_seq,
                            Some(w_dev),
                            hidden,
                            seq_len,
                            layer.eps,
                            layer.norm_offset,
                        )
                    })
                    .ok()?;
                elem::add_in_place_batch_device(self, &mut h_seq, &normed).ok()?;
            } else {
                elem::add_in_place_batch_device(self, &mut h_seq, &attn_delta_seq).ok()?;
            }

            // 6. Pre-FFN rms_norm (batched).
            let ffn_norm_weight: &[f32] = if layer.has_post_norms {
                layer.pre_ffn_norm.unwrap_or(layer.post_attn_norm)
            } else {
                layer.post_attn_norm
            };
            let h_ffn_seq = self
                .with_norm_device_buf(ffn_norm_weight, |w_dev| {
                    elem::rms_norm_batch_device(
                        self,
                        &h_seq,
                        Some(w_dev),
                        hidden,
                        seq_len,
                        layer.eps,
                        layer.norm_offset,
                    )
                })
                .ok()?;

            // 7. gate / up via batched GEMM.
            let t = std::time::Instant::now();
            let gate_seq = self.gemm_proj_seq(layer.gate, &h_ffn_seq, seq_len, inter, hidden)?;
            let up_seq = self.gemm_proj_seq(layer.up, &h_ffn_seq, seq_len, inter, hidden)?;
            sync_p(self);
            t_gate_up += t.elapsed();

            // 8. silu / gelu (batched element-wise).
            let t = std::time::Instant::now();
            let gelu_tanh = matches!(layer.activation, Activation::GeluTanh);
            let act_seq = elem::silu_gate_up_batch_device(
                self,
                &gate_seq,
                &up_seq,
                seq_len * inter,
                gelu_tanh,
            )
            .ok()?;
            sync_p(self);
            t_silu += t.elapsed();

            // 9. down via batched GEMM.
            let t = std::time::Instant::now();
            let ffn_delta_seq = self.gemm_proj_seq(layer.down, &act_seq, seq_len, hidden, inter)?;
            sync_p(self);
            t_down += t.elapsed();

            // 10. Residual + optional post-FFN rms_norm.
            if layer.has_post_norms {
                let normed = match layer.post_ffn_norm {
                    Some(w) if !w.is_empty() => self
                        .with_norm_device_buf(w, |w_dev| {
                            elem::rms_norm_batch_device(
                                self,
                                &ffn_delta_seq,
                                Some(w_dev),
                                hidden,
                                seq_len,
                                layer.eps,
                                layer.norm_offset,
                            )
                        })
                        .ok()?,
                    _ => elem::rms_norm_batch_device(
                        self,
                        &ffn_delta_seq,
                        None,
                        hidden,
                        seq_len,
                        layer.eps,
                        layer.norm_offset,
                    )
                    .ok()?,
                };
                elem::add_in_place_batch_device(self, &mut h_seq, &normed).ok()?;
            } else {
                elem::add_in_place_batch_device(self, &mut h_seq, &ffn_delta_seq).ok()?;
            }

            if layer.layer_scalar != 0.0 && layer.layer_scalar != 1.0 {
                elem::scale_inplace_batch_device(self, &mut h_seq, layer.layer_scalar).ok()?;
            }
            // q_dim assertion satisfied — silence unused warning.
            let _ = q_dim;
        }

        if prefill_profile {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            eprintln!(
                "[cuda-prefill-profile] seq_len={seq_len} layers={} \
                 norm={:.2}ms qkv={:.2}ms attn={:.2}ms wo={:.2}ms \
                 gate_up={:.2}ms silu={:.2}ms down={:.2}ms",
                layers.len(),
                ms(t_norm),
                ms(t_qkv),
                ms(t_attn),
                ms(t_wo),
                ms(t_gate_up),
                ms(t_silu),
                ms(t_down),
            );
            let _ = t_resid;
        }

        cache.len = seq_len;
        self.dtoh_f32(&h_seq).ok()
    }
}
