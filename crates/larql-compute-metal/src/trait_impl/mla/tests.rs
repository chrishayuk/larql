//! One MLA step against a scalar reference, at a hand-checkable
//! geometry.
//!
//! The real-weight gate lives in `larql-vindex`, beside the proven CPU
//! operator — but the kernels live here, and a shader this crate ships
//! needs a gate this crate runs. The reference below is a literal
//! transcription of `exec::mla::mla_forward`, written independently of
//! the shaders it scores.

use super::*;
use crate::MetalBackend;

const HEADS: usize = 3;
const LATENT: usize = 8;
const NOPE: usize = 4;
const ROPE: usize = 2;
const V_DIM: usize = 4;
const HIDDEN: usize = 6;
const POSITIONS: usize = 5;
const EPS: f32 = 1e-6;
/// A threadgroup tree against a serial sum, over a softmax — loose
/// enough for reassociation, orders below any real error.
const TOLERANCE: f32 = 1e-4;

fn shape() -> MlaShape {
    MlaShape {
        hidden: HIDDEN,
        num_heads: HEADS,
        kv_lora_rank: LATENT,
        qk_nope_head_dim: NOPE,
        qk_rope_head_dim: ROPE,
        v_head_dim: V_DIM,
    }
}

fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.29 + seed).sin() * 0.5)
        .collect()
}

/// bf16 codes as bytes, and the exact f32 values they denote — the
/// oracle must score the STORED weights, never the pre-rounding ones.
fn bf16(n: usize, seed: f32) -> (Vec<u8>, Vec<f32>) {
    let v = synth(n, seed);
    let codes: Vec<u16> = v.iter().map(|f| (f.to_bits() >> 16) as u16).collect();
    let exact = codes
        .iter()
        .map(|c| f32::from_bits((*c as u32) << 16))
        .collect();
    (codes.iter().flat_map(|c| c.to_le_bytes()).collect(), exact)
}

struct Weights {
    q_bytes: Vec<u8>,
    q: Vec<f32>,
    ka_bytes: Vec<u8>,
    ka: Vec<f32>,
    kb_bytes: Vec<u8>,
    kb: Vec<f32>,
    o_bytes: Vec<u8>,
    o: Vec<f32>,
    norm: Vec<f32>,
}

fn weights() -> Weights {
    let s = shape();
    let (q_bytes, q) = bf16(HEADS * s.q_head_dim() * HIDDEN, 0.3);
    let (ka_bytes, ka) = bf16(s.cache_stride() * HIDDEN, 1.1);
    let (kb_bytes, kb) = bf16(s.kv_row() * LATENT, 2.4);
    let (o_bytes, o) = bf16(HIDDEN * s.value_width(), 3.7);
    Weights {
        q_bytes,
        q,
        ka_bytes,
        ka,
        kb_bytes,
        kb,
        o_bytes,
        o,
        norm: synth(LATENT, 4.9).iter().map(|v| v + 1.0).collect(),
    }
}

impl Weights {
    fn device(&self) -> MlaDeviceWeights<'_> {
        MlaDeviceWeights {
            q_proj: &self.q_bytes,
            kv_a_proj: &self.ka_bytes,
            kv_a_norm: &self.norm,
            kv_b_proj: &self.kb_bytes,
            o_proj: &self.o_bytes,
            kv_a_norm_eps: EPS,
            projection_encoding: ExpertEncoding::Bf16,
        }
    }
}

fn matvec(w: &[f32], x: &[f32], out: usize) -> Vec<f32> {
    let k = x.len();
    (0..out)
        .map(|r| {
            w[r * k..(r + 1) * k]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

/// A literal transcription of `exec::mla::mla_forward`.
fn reference(w: &Weights, cache: &mut Vec<Vec<f32>>, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let s = shape();
    let scaling = (s.q_head_dim() as f32).powf(-0.5);
    let q = matvec(&w.q, x, HEADS * s.q_head_dim());
    cache.push(matvec(&w.ka, x, s.cache_stride()));
    let visible = cache.len();

    let mut kv_b = Vec::with_capacity(visible);
    for entry in cache.iter() {
        let ms = entry[..LATENT].iter().map(|v| v * v).sum::<f32>() / LATENT as f32;
        let inv = (ms + EPS).sqrt().recip();
        let normed: Vec<f32> = entry[..LATENT]
            .iter()
            .zip(&w.norm)
            .map(|(v, g)| v * inv * g)
            .collect();
        kv_b.push(matvec(&w.kb, &normed, s.kv_row()));
    }

    let mut value = vec![0.0f32; s.value_width()];
    for h in 0..HEADS {
        let q_nope = &q[h * s.q_head_dim()..h * s.q_head_dim() + NOPE];
        let q_rope = &q[h * s.q_head_dim() + NOPE..(h + 1) * s.q_head_dim()];
        let mut scores: Vec<f32> = (0..visible)
            .map(|p| {
                let kv = &kv_b[p][h * (NOPE + V_DIM)..h * (NOPE + V_DIM) + NOPE];
                let k_rot = &cache[p][LATENT..];
                let dot: f32 = q_nope.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>()
                    + q_rope.iter().zip(k_rot).map(|(a, b)| a * b).sum::<f32>();
                dot * scaling
            })
            .collect();
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in scores.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        for v in scores.iter_mut() {
            *v /= sum;
        }
        for (p, &weight) in scores.iter().enumerate() {
            let v = &kv_b[p][h * (NOPE + V_DIM) + NOPE..(h + 1) * (NOPE + V_DIM)];
            for (d, &vd) in v.iter().enumerate() {
                value[h * V_DIM + d] += weight * vd;
            }
        }
    }
    (matvec(&w.o, &value, HIDDEN), value)
}

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The whole step, and the cache it leaves behind, across positions — a
/// path that ignored its cache, or appended twice, diverges here and not
/// at position 0.
#[test]
fn the_device_step_matches_the_scalar_reference_across_positions() {
    let m = backend();
    let w = weights();
    let state = MlaDeviceState::with_capacity(&m, shape(), POSITIONS);
    let mut cache = Vec::new();

    for p in 0..POSITIONS {
        let x = synth(HIDDEN, 10.0 + p as f32);
        let (want, want_value) = reference(&w, &mut cache, &x);
        let got = m
            .mla_attention_step_traced(w.device(), shape(), &state, &x)
            .expect("device step");
        assert!(
            max_abs(&got.output, &want) < TOLERANCE,
            "pos {p} output: max|Δ| {:e}",
            max_abs(&got.output, &want)
        );
        assert!(
            max_abs(&got.attn_value, &want_value) < TOLERANCE,
            "pos {p} attn_value"
        );
        assert_eq!(state.len(), p + 1, "the cache grows by exactly one");
        let cached = state.read_back();
        assert_eq!(cached.len(), cache.len());
        for (i, (a, b)) in cached.iter().zip(&cache).enumerate() {
            assert!(max_abs(a, b) < TOLERANCE, "pos {p} cache entry {i}");
        }
        // Every head's attention weights sum to one — the property a
        // wrong softmax reduction breaks quietly.
        for h in 0..HEADS {
            let s: f32 = got.attn_weights[h * (p + 1)..(h + 1) * (p + 1)]
                .iter()
                .sum();
            assert!((s - 1.0).abs() < 1e-5, "pos {p} head {h} weights sum {s}");
        }
        assert!(got.gpu_ms >= 0.0);
    }
}

/// The plain entry point agrees with the traced one, and `reset` really
/// restarts the sequence.
#[test]
fn the_plain_step_agrees_and_reset_restarts() {
    let m = backend();
    let w = weights();
    let state = MlaDeviceState::with_capacity(&m, shape(), POSITIONS);
    assert!(state.is_empty());
    let x = synth(HIDDEN, 2.0);
    let (plain, gpu) = m
        .mla_attention_step(w.device(), shape(), &state, &x)
        .expect("plain");
    assert!(gpu >= 0.0);
    state.reset();
    assert!(state.is_empty());
    let traced = m
        .mla_attention_step_traced(w.device(), shape(), &state, &x)
        .expect("traced");
    assert_eq!(traced.output, plain, "tracing must not change the answer");
    assert_eq!(traced.q_proj.len(), HEADS * shape().q_head_dim());
    assert_eq!(traced.compressed_kv.len(), shape().cache_stride());
    assert_eq!(traced.kv_a_normed.len(), LATENT);
    assert_eq!(traced.kv_b.len(), shape().kv_row());
    assert_eq!(traced.attn_weights.len(), HEADS);
    assert_eq!(traced.attn_value.len(), shape().value_width());
}

/// Shape faults and a full cache refuse rather than reading out of
/// bounds — a Metal buffer cannot grow.
#[test]
fn shape_faults_and_a_full_cache_are_refused() {
    let m = backend();
    let w = weights();
    let state = MlaDeviceState::with_capacity(&m, shape(), 1);
    let x = synth(HIDDEN, 1.0);

    assert!(matches!(
        m.mla_attention_step(w.device(), shape(), &state, &synth(HIDDEN - 1, 0.0)),
        Err(GroupedError::OffsetOutOfRange { .. })
    ));
    let other = MlaShape {
        v_head_dim: V_DIM * 2,
        ..shape()
    };
    assert!(matches!(
        m.mla_attention_step(w.device(), other, &state, &x),
        Err(GroupedError::OffsetOutOfRange { .. })
    ));

    assert!(m
        .mla_attention_step(w.device(), shape(), &state, &x)
        .is_ok());
    assert!(matches!(
        m.mla_attention_step(w.device(), shape(), &state, &x),
        Err(GroupedError::SlotCountMismatch { .. })
    ));
}

/// `MlaShape`'s derived widths, which every binding depends on.
#[test]
fn the_shape_derives_its_widths() {
    let s = shape();
    assert_eq!(s.q_head_dim(), NOPE + ROPE);
    assert_eq!(s.cache_stride(), LATENT + ROPE);
    assert_eq!(s.kv_row(), HEADS * (NOPE + V_DIM));
    assert_eq!(s.value_width(), HEADS * V_DIM);
}
