//! CUDA decode backend integration.
//!
//! This is a correctness-first bridge from the full-pipeline trait to the
//! cudarc helpers that already exist for CUDA. It keeps KV state host-visible
//! for now. Q4_K projections route through the direct packed-weight CUDA
//! matvec path; other quant formats still use the correctness-first fallback.

use crate::backend::{DecodeBackend, QuantMatVec};
use crate::{Activation, FfnType, FullPipelineLayer, NormType, QuantFormat, QuantWeight};

use cudarc::driver::CudaSlice;

use super::backend::CudaBackend;
use super::elem::Q8_1Buf;
use super::matmul as kernels;
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

const DEFAULT_CUDA_KV_CACHE_MAX_SEQ: usize = 4096;

/// Per-layer K/V cache storage. `cuda-decode-device-resident` Phase 3
/// switched these from `Vec<f32>` to `CudaSlice<f32>` so
/// `fused_decode_attention_device_kv` can read prior tokens and append
/// the new row without any per-call PCIe transfer.
pub(crate) struct CudaKvLayer {
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) k: CudaSlice<f32>,
    pub(crate) v: CudaSlice<f32>,
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
                Ok(CudaKvLayer {
                    num_kv_heads,
                    head_dim,
                    k: drv.device_alloc(n)?,
                    v: drv.device_alloc(n)?,
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
        if let Err(_e) = self.htod_into_slice(&k_data[..n], &mut slot.k, 0) {
            return;
        }
        if let Err(_e) = self.htod_into_slice(&v_data[..n], &mut slot.v, 0) {
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
            let kv_host_k = self.dtoh_f32(&kv_slot.k).ok()?;
            let kv_host_v = self.dtoh_f32(&kv_slot.v).ok()?;
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
            self.htod_into_slice(&attn_out.k_cache, &mut kv_slot.k, 0)
                .ok()?;
            self.htod_into_slice(&attn_out.v_cache, &mut kv_slot.v, 0)
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
        _kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
    ) -> Option<Vec<f32>> {
        if x.len() != hidden || layers.is_empty() {
            return None;
        }
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

    /// Q-format-aware projection GEMM for batched prefill. Routes
    /// Q4_K and Q6_K through their respective f32 device caches (one-
    /// time dequant per session) and runs the projection as a cuBLAS
    /// `(seq_len, hidden) × (out_dim, hidden)^T → (seq_len, out_dim)`
    /// GEMM. `cuda-prefill-batched-q4k` Phase 1.
    fn gemm_proj_seq(
        &self,
        weight: QuantWeight<'_>,
        x_seq: &CudaSlice<f32>,
        seq_len: usize,
        out_dim: usize,
        hidden: usize,
    ) -> Option<CudaSlice<f32>> {
        let n_elements = out_dim * hidden;
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
