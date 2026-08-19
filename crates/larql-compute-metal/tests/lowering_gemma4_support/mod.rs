//! CPU references and fixture glue for `test_lowering_gemma4_arms.rs`.
//!
//! Everything here is transcribed from the HF / interpreter semantics of
//! the Gemma 4 arms — weighted per-head Q/K norm, parameter-free V norm,
//! tanh-GELU gating, the hybrid dense+routed FFN — and never from the
//! lowering's encode order. The MXFP4 quantiser, page-aligned regions,
//! f16 round-trip and the dense helpers come from the routed-stack
//! support module, re-exported so the test imports one namespace.

#![allow(dead_code)]

pub mod hybrid_stack;
#[path = "../lowering_routed_support/mod.rs"]
mod routed;

pub use routed::{det, f16_matrix, matvec, rel_rms, rms_norm, AlignedRegion, Mxfp4Matrix};

use larql_compute::ffn::gelu_tanh;
use larql_compute::MoeFusedRowLayout;
use larql_compute_metal::MetalBackend;
use larql_models::quant::mxfp4::FusedHalf;

// ── shared geometry / amplitudes ─────────────────────────────────────

pub const HIDDEN: usize = 64;
pub const EPS: f32 = 1e-5;
/// Norm weights stored raw (Gemma 4 convention) — `offset 0`, weights
/// near one but not one.
pub const RAW_OFFSET: f32 = 0.0;
pub const WEIGHT_AMPLITUDE: f32 = 0.35;
pub const HIDDEN_AMPLITUDE: f32 = 1.0;
/// Spread of the near-one norm weights — wide enough that a weighted
/// norm is visibly not a weightless one.
pub const NORM_WEIGHT_AMPLITUDE: f32 = 1.0;

/// The Metal backend, or `None` (with a note) on a box without a device.
pub fn device() -> Option<MetalBackend> {
    let gpu = MetalBackend::new();
    if gpu.is_none() {
        eprintln!("no Metal device; skipping");
    }
    gpu
}

/// Run one encoder-full of work in one command buffer and wait.
pub fn run_once(gpu: &MetalBackend, encode: impl FnOnce(&metal::ComputeCommandEncoderRef)) {
    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode(enc);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// Norm weights near one: `1 + det(±amplitude/2)`.
pub fn near_one(n: usize, seed: u32, amplitude: f32) -> Vec<f32> {
    det(n, seed, amplitude)
        .into_iter()
        .map(|w| 1.0 + w)
        .collect()
}

// ── norms ────────────────────────────────────────────────────────────

/// `x · rsqrt(mean(x²) + eps)` — no weight, no offset.
pub fn rms_norm_no_weight(x: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * inv).collect()
}

/// Per-head RMS norm over `[heads, head_dim]`: weightless when `weight`
/// is `None` (Gemma 4 `v_norm`), else `× (offset + w[d])` (`q_norm` /
/// `k_norm`, weight shared across heads).
pub fn rms_norm_heads(
    x: &[f32],
    head_dim: usize,
    weight: Option<&[f32]>,
    eps: f32,
    offset: f32,
) -> Vec<f32> {
    x.chunks_exact(head_dim)
        .flat_map(|head| match weight {
            Some(w) => rms_norm(head, w, eps, offset),
            None => rms_norm_no_weight(head, eps),
        })
        .collect()
}

/// Half-split rotary at `pos` over `heads` heads, unit amplitude, from an
/// f32 inverse-frequency table (zero entries rotate by nothing — the
/// proportional plan's unrotated tail).
pub fn rope_half_split(x: &mut [f32], head_dim: usize, pos: usize, inv_freq: &[f32]) {
    let half = head_dim / 2;
    assert_eq!(inv_freq.len(), half);
    for head in x.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let angle = pos as f64 * inv_freq[i] as f64;
            let (s, c) = (angle.sin() as f32, angle.cos() as f32);
            let (x0, x1) = (head[i], head[i + half]);
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}

// ── attention reference ──────────────────────────────────────────────

/// Where the reference takes V from. `Projection` and `RawK` are the two
/// served semantics; the other two are wrong-order controls for the K≡V
/// binding (V must see K *before* the key's own norm and rotation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VSource {
    /// `Wv · normed` — a layer with its own value projection.
    Projection,
    /// The raw K projection, before `k_norm` / rope (K≡V binding).
    RawK,
    /// Control: K after its weighted norm.
    NormedK,
    /// Control: K after norm and rotation.
    RopedK,
}

/// One attention op's geometry and judged scalars.
pub struct AttnGeometry<'a> {
    pub hidden: usize,
    pub num_q: usize,
    pub num_kv: usize,
    pub head_dim: usize,
    /// Cache length including `pos`.
    pub t: usize,
    pub pos: usize,
    pub eps: f32,
    pub norm_offset: f32,
    pub qk_eps: f32,
    pub qk_offset: f32,
    pub score_scale: f32,
    pub inv_freq: &'a [f32],
}

/// One attention op's operands, all f32 host-side (the f16 arm's
/// matrices are passed already round-tripped).
pub struct AttnOperands<'a> {
    pub norm_w: &'a [f32],
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],
    pub k_cache: &'a [f32],
    pub v_cache: &'a [f32],
    /// `Some` = weighted per-head norm on Q / K (applied together, as the
    /// plan's `qk_norm` op is one op); `None` = absent.
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    /// Parameter-free per-head RMS on V.
    pub v_norm: bool,
    pub v_source: VSource,
}

/// `h + Wo · attend(q, K ∪ k, V ∪ v)` with the Gemma 4 conditioning
/// order: projections → V norm on the raw value → weighted Q/K norm →
/// rope → attention (no sinks, no gate, no post-norm, full span).
pub fn cpu_attention(h: &[f32], g: &AttnGeometry<'_>, w: &AttnOperands<'_>) -> Vec<f32> {
    let (q_rows, kv_rows) = (g.num_q * g.head_dim, g.num_kv * g.head_dim);
    let normed = rms_norm(h, w.norm_w, g.eps, g.norm_offset);
    let mut q = matvec(w.wq, &normed, q_rows, g.hidden);
    let k_raw = matvec(w.wk, &normed, kv_rows, g.hidden);
    let mut k = match w.k_norm {
        Some(kw) => rms_norm_heads(&k_raw, g.head_dim, Some(kw), g.qk_eps, g.qk_offset),
        None => k_raw.clone(),
    };
    if let Some(qw) = w.q_norm {
        q = rms_norm_heads(&q, g.head_dim, Some(qw), g.qk_eps, g.qk_offset);
    }
    let k_normed = k.clone();
    rope_half_split(&mut q, g.head_dim, g.pos, g.inv_freq);
    rope_half_split(&mut k, g.head_dim, g.pos, g.inv_freq);
    let mut v = match w.v_source {
        VSource::Projection => matvec(w.wv, &normed, kv_rows, g.hidden),
        VSource::RawK => k_raw,
        VSource::NormedK => k_normed,
        VSource::RopedK => k.clone(),
    };
    if w.v_norm {
        v = rms_norm_heads(&v, g.head_dim, None, g.qk_eps, 0.0);
    }

    let mut kc = w.k_cache.to_vec();
    let mut vc = w.v_cache.to_vec();
    kc[g.pos * kv_rows..(g.pos + 1) * kv_rows].copy_from_slice(&k);
    vc[g.pos * kv_rows..(g.pos + 1) * kv_rows].copy_from_slice(&v);

    let group = g.num_q / g.num_kv;
    let mut concat = vec![0.0f32; q_rows];
    for head in 0..g.num_q {
        let kv = head / group;
        let qh = &q[head * g.head_dim..(head + 1) * g.head_dim];
        let scores: Vec<f32> = (0..g.t)
            .map(|t| {
                let base = t * kv_rows + kv * g.head_dim;
                let kh = &kc[base..base + g.head_dim];
                qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * g.score_scale
            })
            .collect();
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let den: f32 = ex.iter().sum();
        for d in 0..g.head_dim {
            concat[head * g.head_dim + d] = (0..g.t)
                .map(|t| ex[t] / den * vc[t * kv_rows + kv * g.head_dim + d])
                .sum();
        }
    }
    let out = matvec(w.wo, &concat, g.hidden, q_rows);
    h.iter().zip(&out).map(|(a, b)| a + b).collect()
}

// ── gated FFN reference ──────────────────────────────────────────────

/// One dense gated FFN's operands and geometry, f32 host-side.
pub struct DenseFfn<'a> {
    pub norm_w: &'a [f32],
    pub gate: &'a [f32],
    pub up: &'a [f32],
    pub down: &'a [f32],
    pub hidden: usize,
    pub inter: usize,
    pub eps: f32,
    pub offset: f32,
}

/// `down · (act(gate · x) ⊙ (up · x))` for `x = rms(h)`; `gelu` selects
/// tanh-GELU, else SiLU.
pub fn gated_ffn_branch(h: &[f32], d: &DenseFfn<'_>, gelu: bool) -> Vec<f32> {
    let x = rms_norm(h, d.norm_w, d.eps, d.offset);
    let g = matvec(d.gate, &x, d.inter, d.hidden);
    let u = matvec(d.up, &x, d.inter, d.hidden);
    let act: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(g, u)| if gelu { gelu_tanh(*g) } else { silu(*g) } * u)
        .collect();
    matvec(d.down, &act, d.hidden, d.inter)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── Gemma 4 hybrid FFN reference ─────────────────────────────────────

/// Softmax over all logits → top-k by probability → renormalise over
/// the selected k → × per-expert scale. Returns `(expert, weight)`
/// pairs in selection order.
pub fn route_softmax_topk_renorm_scaled(
    logits: &[f32],
    top_k: usize,
    per_expert_scale: &[f32],
) -> Vec<(usize, f32)> {
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let ex: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
    let den: f32 = ex.iter().sum();
    let mut probs: Vec<(usize, f32)> = ex.iter().map(|e| e / den).enumerate().collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    probs.truncate(top_k);
    let sel: f32 = probs.iter().map(|(_, p)| p).sum();
    probs
        .into_iter()
        .map(|(e, p)| (e, p / sel * per_expert_scale[e]))
        .collect()
}

/// One hybrid layer's host-side operands, dequantised where the device
/// reads MXFP4.
pub struct HybridRef<'a> {
    pub hidden: usize,
    pub dense_inter: usize,
    pub moe_inter: usize,
    pub top_k: usize,
    pub pre_ffn_norm: &'a [f32],
    pub dense_gate: &'a [f32],
    pub dense_up: &'a [f32],
    pub dense_down: &'a [f32],
    /// `[E, hidden]`.
    pub router_proj: &'a [f32],
    /// `router.scale`, `[hidden]` — conditioning is `scale · hidden^-0.5`.
    pub router_scale: &'a [f32],
    pub per_expert_scale: &'a [f32],
    /// Per expert `[2·inter, hidden]` in `layout`, and `[hidden, inter]`.
    pub expert_gate_up: &'a [Vec<f32>],
    pub expert_down: &'a [Vec<f32>],
    pub layout: MoeFusedRowLayout,
    pub pre_experts_norm: &'a [f32],
    pub post_dense_norm: &'a [f32],
    pub post_experts_norm: &'a [f32],
    pub post_ffn_norm: &'a [f32],
    pub eps: f32,
    pub post_eps: f32,
    pub offset: f32,
    pub layer_scale: f32,
}

/// The router's view of the residual: `rms_no_weight(r) · scale · H^-0.5`.
pub fn hybrid_router_input(r: &[f32], h: &HybridRef<'_>) -> Vec<f32> {
    let root_inv = (h.hidden as f32).powf(-0.5);
    rms_norm_no_weight(r, h.eps)
        .iter()
        .zip(h.router_scale)
        .map(|(x, s)| x * s * root_inv)
        .collect()
}

/// The route the reference takes on residual `r`.
pub fn hybrid_route(r: &[f32], h: &HybridRef<'_>) -> Vec<(usize, f32)> {
    let logits = matvec(
        h.router_proj,
        &hybrid_router_input(r, h),
        h.per_expert_scale.len(),
        h.hidden,
    );
    route_softmax_topk_renorm_scaled(&logits, h.top_k, h.per_expert_scale)
}

/// `h' = (r + post_ffn_norm(d + e)) × layer_scale` with
/// `d = post_dense_norm(dense(pre_ffn_norm(r)))` (tanh-GELU) and
/// `e = post_experts_norm(Σ w·expert(pre_experts_norm(r)))`.
pub fn hybrid_ffn_reference(r: &[f32], h: &HybridRef<'_>) -> Vec<f32> {
    let dense = gated_ffn_branch(
        r,
        &DenseFfn {
            norm_w: h.pre_ffn_norm,
            gate: h.dense_gate,
            up: h.dense_up,
            down: h.dense_down,
            hidden: h.hidden,
            inter: h.dense_inter,
            eps: h.eps,
            offset: h.offset,
        },
        true,
    );
    let d = rms_norm(&dense, h.post_dense_norm, h.eps, h.offset);

    let x = rms_norm(r, h.pre_experts_norm, h.eps, h.offset);
    let mut e = vec![0.0f32; h.hidden];
    for (expert, weight) in hybrid_route(r, h) {
        let gu = &h.expert_gate_up[expert];
        let (g0, gs) = h.layout.row_walk(FusedHalf::Gate, h.moe_inter);
        let (u0, us) = h.layout.row_walk(FusedHalf::Up, h.moe_inter);
        let act: Vec<f32> = (0..h.moe_inter)
            .map(|i| {
                let (gr, ur) = (g0 + i * gs, u0 + i * us);
                let g: f32 = (0..h.hidden).map(|c| gu[gr * h.hidden + c] * x[c]).sum();
                let u: f32 = (0..h.hidden).map(|c| gu[ur * h.hidden + c] * x[c]).sum();
                gelu_tanh(g) * u
            })
            .collect();
        let dn = &h.expert_down[expert];
        for j in 0..h.hidden {
            let y: f32 = (0..h.moe_inter)
                .map(|i| dn[j * h.moe_inter + i] * act[i])
                .sum();
            e[j] += weight * y;
        }
    }
    let e = rms_norm(&e, h.post_experts_norm, h.eps, h.offset);

    let sum: Vec<f32> = d.iter().zip(&e).map(|(a, b)| a + b).collect();
    let post = rms_norm(&sum, h.post_ffn_norm, h.post_eps, h.offset);
    r.iter()
        .zip(&post)
        .map(|(a, b)| (a + b) * h.layer_scale)
        .collect()
}
