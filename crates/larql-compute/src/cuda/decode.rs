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
use super::matmul as kernels;
use super::{attn, dequant};

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

#[derive(Debug, Clone)]
struct CudaKvLayer {
    num_kv_heads: usize,
    head_dim: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct CudaKvCache {
    max_seq: usize,
    len: usize,
    layers: Vec<CudaKvLayer>,
}

impl CudaKvCache {
    fn new(shapes: &[(usize, usize)], max_seq: usize) -> Self {
        let layers = shapes
            .iter()
            .map(|&(num_kv_heads, head_dim)| {
                let n = max_seq * num_kv_heads * head_dim;
                CudaKvLayer {
                    num_kv_heads,
                    head_dim,
                    k: vec![0.0; n],
                    v: vec![0.0; n],
                }
            })
            .collect();
        Self {
            max_seq,
            len: 0,
            layers,
        }
    }

    fn ensure_for_layers(&mut self, layers: &[FullPipelineLayer<'_>], max_seq: usize) {
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
            *self = Self::new(&shapes, max_seq);
        }
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
            *cache = Some(CudaKvCache::new(shapes, max_seq));
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
            *guard = Some(CudaKvCache::new(
                &shapes,
                seq_len.max(DEFAULT_CUDA_KV_CACHE_MAX_SEQ),
            ));
        }
        let Some(cache) = guard.as_mut() else {
            return;
        };
        let Some(slot) = cache.layers.get_mut(layer) else {
            return;
        };
        if slot.num_kv_heads != num_kv_heads || slot.head_dim != head_dim {
            return;
        }
        let n = seq_len * num_kv_heads * head_dim;
        if k_data.len() < n || v_data.len() < n || slot.k.len() < n || slot.v.len() < n {
            return;
        }
        slot.k[..n].copy_from_slice(&k_data[..n]);
        slot.v[..n].copy_from_slice(&v_data[..n]);
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
    /// Legacy host-bouncing decode path. Kept verbatim from the
    /// pre-`cuda-decode-device-resident` implementation so it can
    /// double as a parity reference and as the runtime back-out via
    /// `LARQL_CUDA_DECODE_HOST_FALLBACK=1`. Every projection round-
    /// trips through `Vec<f32>`.
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
            *guard = Some(CudaKvCache::new(&shapes, DEFAULT_CUDA_KV_CACHE_MAX_SEQ));
        }
        let cache = guard.as_mut()?;
        cache.ensure_for_layers(layers, cache.max_seq.max(DEFAULT_CUDA_KV_CACHE_MAX_SEQ));
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

            let kv_slot = cache.layers.get_mut(layer_idx)?;
            let attn_out = attn::fused_decode_attention(
                self,
                &qkv.q,
                &qkv.k,
                &qkv.v,
                &kv_slot.k,
                &kv_slot.v,
                layer.q_norm_weight,
                layer.k_norm_weight,
                attn::FusedDecodeAttentionOpts {
                    num_q_heads: layer_num_q_heads,
                    num_kv_heads: layer_num_kv_heads,
                    head_dim: layer_head_dim,
                    pos,
                    max_seq: cache.max_seq,
                    rotary_dim: layer_rotary_dim,
                    rope_base: layer_rope_base,
                    eps: layer.eps,
                    qk_norm_offset: layer.qk_norm_offset,
                    attn_scale: layer.attn_scale,
                    softcap: 0.0,
                },
            )
            .ok()?;
            kv_slot.k = attn_out.k_cache;
            kv_slot.v = attn_out.v_cache;

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

    /// Device-resident decode path. `cuda-decode-device-resident`
    /// Phase 1.
    ///
    /// Per-layer the projections (Q/K/V/O/gate/up/down) hold their
    /// inputs and outputs as `CudaSlice<f32>` so the seven matvec
    /// outputs no longer dtoh after each launch. `rms_norm` and the
    /// silu/gate-up activation still run on host (Phase 2 will move
    /// them onto the GPU). Returns `None` if any projection format
    /// is unsupported on the device path; callers should fall back
    /// to `decode_token_host_fallback`.
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
            *guard = Some(CudaKvCache::new(&shapes, DEFAULT_CUDA_KV_CACHE_MAX_SEQ));
        }
        let cache = guard.as_mut()?;
        cache.ensure_for_layers(layers, cache.max_seq.max(DEFAULT_CUDA_KV_CACHE_MAX_SEQ));
        let pos = cache.len;
        if pos >= cache.max_seq {
            return None;
        }

        let profile_on = decode_profile_enabled();
        let mut prof = DecodeProfile::default();
        // Forces queued GPU work to drain so the next section's
        // wall-clock measurement reflects only its own kernels.
        let sync_if_profile = |b: &CudaBackend| {
            if profile_on {
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

            // ── 1. RMSNorm (CPU) ───────────────────────────────────
            let t = std::time::Instant::now();
            let h_attn = rms_norm_vec(&h, layer.input_norm, layer.eps, layer.norm_offset);
            prof.norm_cpu += t.elapsed();

            // ── 1b. htod h_attn ────────────────────────────────────
            let t = std::time::Instant::now();
            let h_attn_dev = self.htod_f32(&h_attn).ok()?;
            sync_if_profile(self);
            prof.htod += t.elapsed();

            // ── 2. Q/K/V projections, device-resident ──────────────
            let t = std::time::Instant::now();
            let q_dev = matvec_device(self, layer.wq, &h_attn_dev, layer_q_dim, hidden)?;
            let k_dev = matvec_device(self, layer.wk, &h_attn_dev, layer_kv_dim, hidden)?;
            let v_dev = matvec_device(self, layer.wv, &h_attn_dev, layer_kv_dim, hidden)?;
            sync_if_profile(self);
            prof.proj_qkv += t.elapsed();

            // ── 3. Fused decode attention (device q/k/v in, device out) ─
            let kv_slot = cache.layers.get_mut(layer_idx)?;
            let t = std::time::Instant::now();
            let attn_out = attn::fused_decode_attention_device(
                self,
                &q_dev,
                &k_dev,
                &v_dev,
                &kv_slot.k,
                &kv_slot.v,
                layer.q_norm_weight,
                layer.k_norm_weight,
                attn::FusedDecodeAttentionOpts {
                    num_q_heads: layer_num_q_heads,
                    num_kv_heads: layer_num_kv_heads,
                    head_dim: layer_head_dim,
                    pos,
                    max_seq: cache.max_seq,
                    rotary_dim: layer_rotary_dim,
                    rope_base: layer_rope_base,
                    eps: layer.eps,
                    qk_norm_offset: layer.qk_norm_offset,
                    attn_scale: layer.attn_scale,
                    softcap: 0.0,
                },
            )
            .ok()?;
            prof.attn_call += t.elapsed();
            kv_slot.k = attn_out.k_cache;
            kv_slot.v = attn_out.v_cache;
            let attn_out_dev: CudaSlice<f32> = attn_out.out;

            // ── 4. wo projection ───────────────────────────────────
            let t = std::time::Instant::now();
            let attn_delta_dev = matvec_device(self, layer.wo, &attn_out_dev, hidden, layer_q_dim)
                .or_else(|| {
                    if q_dim != layer_q_dim {
                        None
                    } else {
                        matvec_device(self, layer.wo, &attn_out_dev, hidden, q_dim)
                    }
                })?;
            sync_if_profile(self);
            prof.proj_wo += t.elapsed();

            // ── 4b. dtoh attn_delta ────────────────────────────────
            let t = std::time::Instant::now();
            let attn_delta = self.dtoh_f32(&attn_delta_dev).ok()?;
            prof.dtoh_attn_delta += t.elapsed();

            // ── 5. Residual + post-attn rms (CPU) ──────────────────
            let t = std::time::Instant::now();
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
            prof.residual_cpu += t.elapsed();

            // ── 6. Pre-FFN rms (CPU) ───────────────────────────────
            let ffn_norm_weight = if layer.has_post_norms {
                layer.pre_ffn_norm.unwrap_or(layer.post_attn_norm)
            } else {
                layer.post_attn_norm
            };
            let t = std::time::Instant::now();
            let h_ffn = rms_norm_vec(&h_post_attn, ffn_norm_weight, layer.eps, layer.norm_offset);
            prof.norm_cpu += t.elapsed();

            // ── 6b. htod h_ffn ─────────────────────────────────────
            let t = std::time::Instant::now();
            let h_ffn_dev = self.htod_f32(&h_ffn).ok()?;
            sync_if_profile(self);
            prof.htod += t.elapsed();

            // ── 7. gate / up projections ───────────────────────────
            let t = std::time::Instant::now();
            let gate_dev = matvec_device(self, layer.gate, &h_ffn_dev, inter, hidden)?;
            let up_dev = matvec_device(self, layer.up, &h_ffn_dev, inter, hidden)?;
            sync_if_profile(self);
            prof.proj_gate_up += t.elapsed();

            // ── 7b. dtoh gate / up ─────────────────────────────────
            let t = std::time::Instant::now();
            let gate = self.dtoh_f32(&gate_dev).ok()?;
            let up = self.dtoh_f32(&up_dev).ok()?;
            prof.dtoh_gate_up += t.elapsed();

            // ── 8. Activation (CPU) ────────────────────────────────
            let t = std::time::Instant::now();
            let act = activate(&gate, &up, layer.activation);
            prof.norm_cpu += t.elapsed();

            // ── 8b. htod act ───────────────────────────────────────
            let t = std::time::Instant::now();
            let act_dev = self.htod_f32(&act).ok()?;
            sync_if_profile(self);
            prof.htod += t.elapsed();

            // ── 9. down projection ─────────────────────────────────
            let t = std::time::Instant::now();
            let ffn_delta_dev = matvec_device(self, layer.down, &act_dev, hidden, inter)?;
            sync_if_profile(self);
            prof.proj_down += t.elapsed();

            // ── 9b. dtoh ffn_delta ─────────────────────────────────
            let t = std::time::Instant::now();
            let ffn_delta = self.dtoh_f32(&ffn_delta_dev).ok()?;
            prof.dtoh_ffn_delta += t.elapsed();

            // ── 10. Residual + optional post-ffn rms (CPU) ─────────
            let t = std::time::Instant::now();
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
            prof.residual_cpu += t.elapsed();
        }

        if profile_on {
            prof.report(layers.len());
        }

        cache.len = pos + 1;
        Some(h)
    }
}
