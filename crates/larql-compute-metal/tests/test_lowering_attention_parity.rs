//! G6b-3: does the lowered attention fragment execute the plan's
//! **ordered** program?
//!
//! Attention is where a generic lowering is easiest to get subtly wrong,
//! because its steps approximately commute. A lowerer that contains every
//! operation but applies the query scale after RoPE, or normalises Q over
//! the whole vector instead of per head, produces finite, plausible,
//! wrong numbers. So agreement with a reference is necessary and not
//! sufficient, and each judged fact carries a control that must dwarf the
//! parity residual.
//!
//! | proof | what it establishes |
//! |---|---|
//! | parity | the fragment computes the program |
//! | policy | sliding / full / NoPE are read per layer, not hard-coded |
//! | gate | the judged sigmoid gate is applied |
//! | ordering | the query scale is applied **after** the QK norm |
//!
//! ## Which orderings are semantic, and which are convention
//!
//! The interpreter applies QK norm → query scale → RoPE. Only the first
//! boundary is observable, and the control set says so rather than
//! asserting all three orderings matter:
//!
//! - **query scale vs RoPE — commute exactly.** A uniform scalar and a
//!   rotation commute, so this ordering is unobservable in principle.
//!   A control for it fires at 1.0x the parity residual, which is the
//!   correct answer and not a weak fixture. The interpreter's order here
//!   is a free convention.
//! - **query scale vs QK norm — do not commute, and strongly.** RMS norm
//!   is scale-invariant, so a query scale applied *before* the norm is
//!   divided straight back out — the 3.87 multiplier simply vanishes.
//!   This is the ordering error a generic lowerer would actually make,
//!   and it is worth a factor of 3.87 in the output.
//!
//! The reference is written from the interpreter's `condition_qk_in_place`
//! and `aggregate_heads`, in f32, and consumes the **quantised** weights
//! so that quantisation error (measured separately in Q2) does not
//! dominate the comparison.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::attention::{
    AttnScratch, AttnShape, AttnWeights, LoweredPosition,
};
use larql_compute_metal::lowering::LoweredMatrix;
use larql_models::quant::nvfp4;

const HIDDEN: usize = 512;
const NUM_Q: usize = 8;
const NUM_KV: usize = 2;
const HEAD_DIM: usize = 64;
const Q_ROWS: usize = NUM_Q * HEAD_DIM;
const KV_ROWS: usize = NUM_KV * HEAD_DIM;
const T: usize = 12;
const POS: usize = T - 1;
const EPS: f32 = 1e-5;
const QK_EPS: f32 = 1e-6;
const NORM_OFFSET: f32 = 1.0;
/// Muse-Glimmer's post-block epsilon (four-norm placement).
const POST_EPS: f32 = 1e-8;
const QUERY_SCALE: f32 = 3.87;
const THETA: f64 = 500_000.0;

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(9);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * 0.5
        })
        .collect()
}

fn rel_rms(reference: &[f32], got: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(got) {
        num += (*a as f64 - *b as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    (num / den).sqrt()
}

fn inv_freq_table() -> Vec<f32> {
    (0..HEAD_DIM / 2)
        .map(|i| THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64) as f32)
        .collect()
}

fn matvec(m: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|r| (0..k).map(|c| m[r * k + c] * x[c]).sum())
        .collect()
}

fn rms_heads(v: &mut [f32], heads: usize, eps: f64) {
    for h in 0..heads {
        let off = h * HEAD_DIM;
        let sq: f64 = (0..HEAD_DIM).map(|d| (v[off + d] as f64).powi(2)).sum();
        let rms = (sq / HEAD_DIM as f64 + eps).sqrt() as f32;
        for d in 0..HEAD_DIM {
            v[off + d] /= rms;
        }
    }
}

fn rope(v: &mut [f32], heads: usize, pos: usize) {
    let half = HEAD_DIM / 2;
    for h in 0..heads {
        let off = h * HEAD_DIM;
        for i in 0..half {
            let inv = THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64);
            let a = pos as f64 * inv;
            let (s, c) = (a.sin() as f32, a.cos() as f32);
            let (x0, x1) = (v[off + i], v[off + half + i]);
            v[off + i] = x0 * c - x1 * s;
            v[off + half + i] = x0 * s + x1 * c;
        }
    }
}

/// Which order to apply QK norm / query scale / RoPE to Q. The judged
/// order is `NormScaleRope`; the others exist so the control can prove
/// the positive arm distinguishes them.
#[derive(Clone, Copy, PartialEq)]
enum Order {
    NormScaleRope,
    NormRopeScale,
    ScaleNormRope,
}

#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    h: &[f32],
    norm_w: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    wg: Option<&[f32]>,
    k_cache: &[f32],
    v_cache: &[f32],
    position: LoweredPosition,
    window: Option<usize>,
    order: Order,
    score_scale: f32,
    // (weight, eps, before_residual); None = two-norm placement.
    post: Option<(&[f32], f32, bool)>,
) -> Vec<f32> {
    // pre-attention norm
    let ms = h.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32;
    let inv = 1.0 / (ms + EPS).sqrt();
    let normed: Vec<f32> = h
        .iter()
        .zip(norm_w)
        .map(|(x, w)| x * inv * (NORM_OFFSET + w))
        .collect();

    let mut q = matvec(wq, &normed, Q_ROWS, HIDDEN);
    let mut k = matvec(wk, &normed, KV_ROWS, HIDDEN);
    let v = matvec(wv, &normed, KV_ROWS, HIDDEN);

    let do_rope = matches!(position, LoweredPosition::Rope { .. });
    match order {
        Order::NormScaleRope => {
            rms_heads(&mut q, NUM_Q, QK_EPS as f64);
            q.iter_mut().for_each(|x| *x *= QUERY_SCALE);
            if do_rope {
                rope(&mut q, NUM_Q, POS);
            }
        }
        Order::NormRopeScale => {
            rms_heads(&mut q, NUM_Q, QK_EPS as f64);
            if do_rope {
                rope(&mut q, NUM_Q, POS);
            }
            q.iter_mut().for_each(|x| *x *= QUERY_SCALE);
        }
        Order::ScaleNormRope => {
            q.iter_mut().for_each(|x| *x *= QUERY_SCALE);
            rms_heads(&mut q, NUM_Q, QK_EPS as f64);
            if do_rope {
                rope(&mut q, NUM_Q, POS);
            }
        }
    }
    rms_heads(&mut k, NUM_KV, QK_EPS as f64);
    if do_rope {
        rope(&mut k, NUM_KV, POS);
    }

    // write this position into copies of the caches
    let mut kc = k_cache.to_vec();
    let mut vc = v_cache.to_vec();
    kc[POS * KV_ROWS..(POS + 1) * KV_ROWS].copy_from_slice(&k);
    vc[POS * KV_ROWS..(POS + 1) * KV_ROWS].copy_from_slice(&v);

    // attention
    let t_start = window.filter(|w| T > *w).map(|w| T - w).unwrap_or(0);
    let mut concat = vec![0.0f32; Q_ROWS];
    let group = NUM_Q / NUM_KV;
    for head in 0..NUM_Q {
        let kv = head / group;
        let qh = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let mut scores = Vec::with_capacity(T - t_start);
        for t in t_start..T {
            let kh = &kc[t * KV_ROWS + kv * HEAD_DIM..t * KV_ROWS + (kv + 1) * HEAD_DIM];
            scores.push(qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * score_scale);
        }
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let denom: f32 = exps.iter().sum();
        for d in 0..HEAD_DIM {
            let mut acc = 0.0f32;
            for (i, t) in (t_start..T).enumerate() {
                acc += exps[i] / denom * vc[t * KV_ROWS + kv * HEAD_DIM + d];
            }
            concat[head * HEAD_DIM + d] = acc;
        }
    }

    if let Some(g) = wg {
        let gate = matvec(g, &normed, Q_ROWS, HIDDEN);
        for (c, gv) in concat.iter_mut().zip(&gate) {
            *c *= 1.0 / (1.0 + (-gv).exp());
        }
    }
    let out = matvec(wo, &concat, HIDDEN, Q_ROWS);
    let rms_norm = |v: &[f32], w: &[f32], eps: f32| -> Vec<f32> {
        let ms = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        v.iter()
            .zip(w)
            .map(|(x, wv)| x * inv * (NORM_OFFSET + wv))
            .collect()
    };
    match post {
        // Judged: normalise the attention branch, then add.
        Some((w, eps, true)) => {
            let n = rms_norm(&out, w, eps);
            h.iter().zip(&n).map(|(a, b)| a + b).collect()
        }
        // Wrong reading of "post-attention norm": add, then normalise.
        Some((w, eps, false)) => {
            let summed: Vec<f32> = h.iter().zip(&out).map(|(a, b)| a + b).collect();
            rms_norm(&summed, w, eps)
        }
        None => h.iter().zip(&out).map(|(a, b)| a + b).collect(),
    }
}

struct Fixture {
    h: Vec<f32>,
    norm_w: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    q: nvfp4::Nvfp4Matrix,
    k: nvfp4::Nvfp4Matrix,
    v: nvfp4::Nvfp4Matrix,
    o: nvfp4::Nvfp4Matrix,
    g: nvfp4::Nvfp4Matrix,
    qq: Vec<f32>,
    kq: Vec<f32>,
    vq: Vec<f32>,
    oq: Vec<f32>,
    gq: Vec<f32>,
}

fn fixture() -> Fixture {
    let (qf, kf, vf) = (
        det(Q_ROWS * HIDDEN, 1),
        det(KV_ROWS * HIDDEN, 2),
        det(KV_ROWS * HIDDEN, 3),
    );
    let (of, gf) = (det(HIDDEN * Q_ROWS, 4), det(Q_ROWS * HIDDEN, 5));
    Fixture {
        h: det(HIDDEN, 6),
        norm_w: det(HIDDEN, 7),
        k_cache: det(T * KV_ROWS, 8),
        v_cache: det(T * KV_ROWS, 9),
        q: nvfp4::quantize(&qf, Q_ROWS, HIDDEN).unwrap(),
        k: nvfp4::quantize(&kf, KV_ROWS, HIDDEN).unwrap(),
        v: nvfp4::quantize(&vf, KV_ROWS, HIDDEN).unwrap(),
        o: nvfp4::quantize(&of, HIDDEN, Q_ROWS).unwrap(),
        g: nvfp4::quantize(&gf, Q_ROWS, HIDDEN).unwrap(),
        qq: nvfp4::round_trip(&qf, Q_ROWS, HIDDEN).unwrap(),
        kq: nvfp4::round_trip(&kf, KV_ROWS, HIDDEN).unwrap(),
        vq: nvfp4::round_trip(&vf, KV_ROWS, HIDDEN).unwrap(),
        oq: nvfp4::round_trip(&of, HIDDEN, Q_ROWS).unwrap(),
        gq: nvfp4::round_trip(&gf, Q_ROWS, HIDDEN).unwrap(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_lowered(
    gpu: &larql_compute_metal::MetalBackend,
    f: &Fixture,
    position: LoweredPosition,
    window: Option<usize>,
    with_gate: bool,
    score_scale: f32,
    post_w: Option<&[f32]>,
) -> Vec<f32> {
    let h_in = gpu.lowering_upload(&f.h).unwrap();
    let norm_buf = gpu.lowering_upload(&f.norm_w).unwrap();
    let k_cache = gpu.lowering_upload(&f.k_cache).unwrap();
    let v_cache = gpu.lowering_upload(&f.v_cache).unwrap();
    let inv_freq = gpu.lowering_upload(&inv_freq_table()).unwrap();
    let h_out = gpu.lowering_scratch(HIDDEN);
    let normed = gpu.lowering_scratch(HIDDEN);
    let q = gpu.lowering_scratch(Q_ROWS);
    let gate = gpu.lowering_scratch(Q_ROWS);
    let concat = gpu.lowering_scratch(Q_ROWS);
    let gated = gpu.lowering_scratch(Q_ROWS);
    let attn_out = gpu.lowering_scratch(HIDDEN);
    let post_scratch = gpu.lowering_scratch(HIDDEN);
    let post_buf = post_w.map(|w| gpu.lowering_upload(w).unwrap());

    let proj = |m: &nvfp4::Nvfp4Matrix| LoweredMatrix::Nvfp4 {
        packed: Box::leak(Box::new(gpu.lowering_weight(&m.packed))),
        scales: Box::leak(Box::new(gpu.lowering_weight(&m.scales))),
        tensor_scale: m.tensor_scale,
    };
    let w = AttnWeights {
        q: proj(&f.q),
        k: proj(&f.k),
        v: proj(&f.v),
        o: proj(&f.o),
        gate: with_gate.then(|| proj(&f.g)),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        sinks: None,
        qk_norm: None,
        norm_weight: &norm_buf,
        post_norm: post_buf
            .as_ref()
            .map(|b| larql_compute_metal::lowering::PostNorm {
                weight: b,
                eps: POST_EPS,
                weight_offset: NORM_OFFSET,
                scratch: &post_scratch,
            }),
    };
    let s = AttnScratch {
        normed: &normed,
        q: &q,
        k_cache: &k_cache,
        v_cache: &v_cache,
        gate: &gate,
        concat: &concat,
        gated: &gated,
        attn_out: &attn_out,
        inv_freq: &inv_freq,
    };
    let shape = AttnShape {
        hidden: HIDDEN,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        head_dim: HEAD_DIM,
        norm_eps: EPS,
        norm_weight_offset: NORM_OFFSET,
        qk_norm_eps: QK_EPS,
        parameter_free_q: true,
        parameter_free_k: true,
        parameter_free_v: false,
        query_scale: Some(QUERY_SCALE),
        score_scale,
        position,
        window,
        softcap: None,
        position_index: POS,
        kv_len: T,
    };

    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    gpu.encode_attention(enc, &h_in, &h_out, &w, &s, &shape);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out = gpu.lowering_readback(&h_out, HIDDEN).unwrap();
    // `w` borrows the uploaded buffers; end its borrow before recycling.
    for b in [
        h_in,
        norm_buf,
        k_cache,
        v_cache,
        inv_freq,
        h_out,
        normed,
        q,
        gate,
        concat,
        gated,
        attn_out,
        post_scratch,
    ] {
        gpu.recycle_lowering_scratch(b);
    }
    if let Some(b) = post_buf {
        gpu.recycle_lowering_scratch(b);
    }
    out
}

fn assert_control(what: &str, perturbed: &[f32], got: &[f32], parity: f64) {
    let c = rel_rms(perturbed, got);
    let ratio = c / parity;
    eprintln!("  control `{what}`: {ratio:.0}x parity ({c:.3e})");
    assert!(
        ratio > 100.0,
        "control `{what}` moves the result only {ratio:.1}x the parity residual — \
         the positive arm cannot distinguish this defect"
    );
}

#[test]
fn lowered_attention_executes_the_ordered_program() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let f = fixture();
    let post_w = det(HIDDEN, 42);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // ── Proof 1: parity, full-attention rope layer with the gate ────
    let reference = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        Some(&f.gq),
        &f.k_cache,
        &f.v_cache,
        LoweredPosition::Rope { theta: THETA },
        None,
        Order::NormScaleRope,
        scale,
        Some((&post_w, POST_EPS, true)),
    );
    let got = run_lowered(
        &gpu,
        &f,
        LoweredPosition::Rope { theta: THETA },
        None,
        true,
        scale,
        Some(&post_w),
    );
    let parity = rel_rms(&reference, &got);
    eprintln!("attention parity (rope, full, gated): rel_rms {parity:.3e}");
    assert!(got.iter().all(|v| v.is_finite()));
    assert!(parity < 1e-4, "lowered attention disagrees: {parity:.3e}");

    // ── Proof 4: ordering ───────────────────────────────────────────
    // The observable boundary: the query scale must come *after* the QK
    // norm, or the norm's scale-invariance cancels it entirely.
    let scale_first = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        Some(&f.gq),
        &f.k_cache,
        &f.v_cache,
        LoweredPosition::Rope { theta: THETA },
        None,
        Order::ScaleNormRope,
        scale,
        Some((&post_w, POST_EPS, true)),
    );
    assert_control("query scale before QK norm", &scale_first, &got, parity);

    // The unobservable boundary, asserted as unobservable. A scalar and
    // a rotation commute, so this must *not* separate — and if a future
    // change ever made it separate, that would mean the query scale had
    // stopped being a uniform scalar or RoPE had stopped being a
    // rotation, either of which is worth failing over.
    let rope_first = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        Some(&f.gq),
        &f.k_cache,
        &f.v_cache,
        LoweredPosition::Rope { theta: THETA },
        None,
        Order::NormRopeScale,
        scale,
        Some((&post_w, POST_EPS, true)),
    );
    let commute = rel_rms(&rope_first, &got) / parity;
    eprintln!("  commutation `query scale vs RoPE`: {commute:.1}x parity (expected ~1)");
    assert!(
        commute < 10.0,
        "a uniform scalar and a rotation must commute; {commute:.1}x parity means one \
         of them is no longer what it claims to be"
    );

    // ── Proof 3: the judged gate is applied ─────────────────────────
    let ungated = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        None,
        &f.k_cache,
        &f.v_cache,
        LoweredPosition::Rope { theta: THETA },
        None,
        Order::NormScaleRope,
        scale,
        Some((&post_w, POST_EPS, true)),
    );
    assert_control("attention gate bypassed", &ungated, &got, parity);

    // ── Proof 2: policy is read per layer, not hard-coded ───────────
    // A NoPE layer: same weights, no rotation. Parity must hold *and*
    // the rope answer must be materially different, or the lowering
    // could be ignoring the policy field.
    let nope_ref = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        Some(&f.gq),
        &f.k_cache,
        &f.v_cache,
        LoweredPosition::None,
        None,
        Order::NormScaleRope,
        scale,
        Some((&post_w, POST_EPS, true)),
    );
    let nope_got = run_lowered(
        &gpu,
        &f,
        LoweredPosition::None,
        None,
        true,
        scale,
        Some(&post_w),
    );
    let nope_parity = rel_rms(&nope_ref, &nope_got);
    eprintln!("attention parity (NoPE): rel_rms {nope_parity:.3e}");
    assert!(
        nope_parity < 1e-4,
        "NoPE layer disagrees: {nope_parity:.3e}"
    );
    assert_control("NoPE vs RoPE policy", &nope_ref, &got, parity);

    // A sliding-window layer: window 4 of 12 positions.
    let win_ref = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        Some(&f.gq),
        &f.k_cache,
        &f.v_cache,
        LoweredPosition::Rope { theta: THETA },
        Some(4),
        Order::NormScaleRope,
        scale,
        Some((&post_w, POST_EPS, true)),
    );
    let win_got = run_lowered(
        &gpu,
        &f,
        LoweredPosition::Rope { theta: THETA },
        Some(4),
        true,
        scale,
        Some(&post_w),
    );
    let win_parity = rel_rms(&win_ref, &win_got);
    eprintln!("attention parity (sliding w=4): rel_rms {win_parity:.3e}");
    assert!(
        win_parity < 1e-4,
        "sliding layer disagrees: {win_parity:.3e}"
    );
    assert_control("sliding vs full span", &win_ref, &got, parity);

    // ── Post-attention norm (four-norm placement) ───────────────────
    // Omitted entirely.
    let no_post = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        Some(&f.gq),
        &f.k_cache,
        &f.v_cache,
        LoweredPosition::Rope { theta: THETA },
        None,
        Order::NormScaleRope,
        scale,
        None,
    );
    assert_control("post-attention norm omitted", &no_post, &got, parity);

    // Applied to the summed hidden state instead of the branch output.
    // "Post-attention norm" reads both ways; only one is the model.
    let after = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        Some(&f.gq),
        &f.k_cache,
        &f.v_cache,
        LoweredPosition::Rope { theta: THETA },
        None,
        Order::NormScaleRope,
        scale,
        Some((&post_w, POST_EPS, false)),
    );
    assert_control("post-norm applied after the residual", &after, &got, parity);
}
