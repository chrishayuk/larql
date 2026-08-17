//! G6b-4: does composing the two gated fragments into **one encoder**
//! compute a whole decoder layer?
//!
//! G6b-2 and G6b-3 gated the halves independently, so the arithmetic
//! inside each is already proven. What is unproven is the seam between
//! them, and the seam is where a composition bug is invisible: attention
//! and FFN both consume "a hidden state" and both emit "a hidden state",
//! so wiring the wrong one produces finite, plausible, wrong numbers of
//! exactly the right shape. Nothing crashes.
//!
//! Three distinct failures live at that seam, and they need different
//! instruments, so this file carries three proofs rather than one:
//!
//! | proof | what it establishes | bar |
//! |---|---|---|
//! | parity | the composition computes the layer program | tolerance |
//! | one-CB ≡ two-CB | no missing dependency or lifetime hazard | bitwise |
//! | shared ≡ disjoint scratch | fragments may pool intermediates | bitwise |
//!
//! ## Why two of the three are bitwise
//!
//! The parity arm changes the arithmetic realisation — GPU reduction
//! order against a CPU reference — so a tolerance is the honest bar, as
//! in the fragment tests.
//!
//! The other two do not. Both arms run the *same kernels* with the *same
//! arguments*; only scheduling and buffer identity differ. Any difference
//! at all would therefore be a real hazard — a dispatch reading a buffer
//! before its producer wrote it, or a pooled buffer aliased while still
//! live — and not float noise. Accepting a tolerance there would hide
//! precisely the class of bug the proof exists to find. This mirrors the
//! reasoning in `test_lowering_chained_encode.rs`.
//!
//! Shared scratch matters beyond tidiness: G6c runs 52 layers in one
//! command buffer, so per-layer scratch must be reused. That is only
//! sound because Metal's default `MTLDispatchTypeSerial` encoder orders
//! dispatches, and every FFN write to a shared buffer is encoded after
//! attention's last read of it. This test is what makes that argument
//! checkable rather than asserted.
//!
//! ## Controls
//!
//! Agreement alone would not show the seam is wired correctly, so each
//! boundary fact carries a negative arm modelled in the reference, and
//! each must dwarf the parity residual:
//!
//! - **FFN residual source** — the FFN must add to the *attention output*,
//!   not to the layer input. Both are hidden-shaped; swapping them drops
//!   the attention contribution from the layer's output while leaving the
//!   FFN's own arithmetic intact.
//! - **FFN norm input** — the FFN must normalise the *attention output*,
//!   not reuse attention's already-normalised scratch. This is the shared
//!   `normed` buffer, so the defect is one wrong argument away and is the
//!   specific hazard that sharing scratch introduces.
//! - **attention residual** — dropped, the FFN still runs on a plausible
//!   vector.
//! - **FFN residual** — dropped likewise.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::attention::{
    AttnScratch, AttnShape, AttnWeights, LoweredPosition,
};
use larql_compute_metal::lowering::ffn::{FfnScratch, FfnShape, FfnWeights};
use larql_compute_metal::lowering::LoweredMatrix;
use larql_models::quant::nvfp4;

const HIDDEN: usize = 512;
const INTER: usize = 1408;
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
/// Muse-Glimmer's post-block epsilon. Four-norm placement normalises each
/// branch output *before* its residual add, at an epsilon three orders of
/// magnitude below the pre-block one.
const POST_EPS: f32 = 1e-8;
const QUERY_SCALE: f32 = 3.87;
const THETA: f64 = 500_000.0;
const SCORE_SCALE: f32 = 0.125;

/// How far a control must exceed the parity residual before the positive
/// assertion can be said to distinguish the defect it models.
const CONTROL_MARGIN: f64 = 100.0;

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * 0.6
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

fn cosine(reference: &[f32], got: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(got) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs(reference: &[f32], got: &[f32]) -> f32 {
    reference
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
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

fn rms_norm(h: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let ms = h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    h.iter()
        .zip(w)
        .map(|(x, wv)| x * inv * (offset + wv))
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

/// Attention half of the reference, in the judged order
/// (QK norm → query scale → RoPE), returning both the layer-state output
/// and the normalised input — the latter only so a control can model the
/// "FFN reused attention's normed scratch" defect.
#[allow(clippy::too_many_arguments)]
fn attention_reference(f: &Fixture, residual: bool) -> (Vec<f32>, Vec<f32>) {
    let normed = rms_norm(&f.h, &f.norm_w, EPS, NORM_OFFSET);

    let mut q = matvec(&f.qq, &normed, Q_ROWS, HIDDEN);
    let mut k = matvec(&f.kq, &normed, KV_ROWS, HIDDEN);
    let v = matvec(&f.vq, &normed, KV_ROWS, HIDDEN);

    rms_heads(&mut q, NUM_Q, QK_EPS as f64);
    q.iter_mut().for_each(|x| *x *= QUERY_SCALE);
    rope(&mut q, NUM_Q, POS);
    rms_heads(&mut k, NUM_KV, QK_EPS as f64);
    rope(&mut k, NUM_KV, POS);

    let mut kc = f.k_cache.clone();
    let mut vc = f.v_cache.clone();
    kc[POS * KV_ROWS..(POS + 1) * KV_ROWS].copy_from_slice(&k);
    vc[POS * KV_ROWS..(POS + 1) * KV_ROWS].copy_from_slice(&v);

    let mut concat = vec![0.0f32; Q_ROWS];
    let group = NUM_Q / NUM_KV;
    for head in 0..NUM_Q {
        let kv = head / group;
        let qh = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let mut scores = Vec::with_capacity(T);
        for t in 0..T {
            let kh = &kc[t * KV_ROWS + kv * HEAD_DIM..t * KV_ROWS + (kv + 1) * HEAD_DIM];
            scores.push(qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * SCORE_SCALE);
        }
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let denom: f32 = exps.iter().sum();
        for d in 0..HEAD_DIM {
            let mut acc = 0.0f32;
            for (t, e) in exps.iter().enumerate() {
                acc += e / denom * vc[t * KV_ROWS + kv * HEAD_DIM + d];
            }
            concat[head * HEAD_DIM + d] = acc;
        }
    }

    let gate = matvec(&f.gq, &normed, Q_ROWS, HIDDEN);
    for (c, gv) in concat.iter_mut().zip(&gate) {
        *c *= 1.0 / (1.0 + (-gv).exp());
    }
    let out = matvec(&f.oq, &concat, HIDDEN, Q_ROWS);
    // Four-norm placement: normalise the attention branch, then add.
    let out = rms_norm(&out, &f.attn_post_w, POST_EPS, NORM_OFFSET);
    let h1 = if residual {
        f.h.iter().zip(&out).map(|(a, b)| a + b).collect()
    } else {
        out
    };
    (h1, normed)
}

/// FFN half of the reference. The norm input and the residual source are
/// separate parameters **because the two boundary defects differ in
/// exactly that way** — a reference that took one hidden state could not
/// model either.
fn ffn_reference(
    f: &Fixture,
    norm_input: &[f32],
    residual_source: &[f32],
    residual: bool,
) -> Vec<f32> {
    let normed = rms_norm(norm_input, &f.ffn_norm_w, EPS, NORM_OFFSET);
    let g = matvec(&f.gate_q, &normed, INTER, HIDDEN);
    let u = matvec(&f.up_q, &normed, INTER, HIDDEN);
    let act: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
        .collect();
    let d = matvec(&f.down_q, &act, HIDDEN, INTER);
    let d = rms_norm(&d, &f.ffn_post_w, POST_EPS, NORM_OFFSET);
    if residual {
        residual_source.iter().zip(&d).map(|(a, b)| a + b).collect()
    } else {
        d
    }
}

/// Which hidden state the FFN consumes. `Correct` is the composition
/// under test; the others are the two seam defects.
#[derive(Clone, Copy, PartialEq)]
enum Seam {
    /// FFN normalises and adds to the attention output. The layer program.
    Correct,
    /// FFN adds to the *layer input* instead of the attention output —
    /// the attention contribution is silently dropped.
    ResidualFromLayerInput,
    /// FFN normalises attention's already-normalised scratch instead of
    /// the attention output — the hazard shared scratch introduces.
    NormsAttentionScratch,
}

fn layer_reference(f: &Fixture, seam: Seam, attn_residual: bool, ffn_residual: bool) -> Vec<f32> {
    let (h1, attn_normed) = attention_reference(f, attn_residual);
    match seam {
        Seam::Correct => ffn_reference(f, &h1, &h1, ffn_residual),
        Seam::ResidualFromLayerInput => ffn_reference(f, &h1, &f.h, ffn_residual),
        Seam::NormsAttentionScratch => ffn_reference(f, &attn_normed, &h1, ffn_residual),
    }
}

struct Fixture {
    h: Vec<f32>,
    norm_w: Vec<f32>,
    ffn_norm_w: Vec<f32>,
    attn_post_w: Vec<f32>,
    ffn_post_w: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    q: nvfp4::Nvfp4Matrix,
    k: nvfp4::Nvfp4Matrix,
    v: nvfp4::Nvfp4Matrix,
    o: nvfp4::Nvfp4Matrix,
    g: nvfp4::Nvfp4Matrix,
    gate: nvfp4::Nvfp4Matrix,
    up: nvfp4::Nvfp4Matrix,
    down: nvfp4::Nvfp4Matrix,
    qq: Vec<f32>,
    kq: Vec<f32>,
    vq: Vec<f32>,
    oq: Vec<f32>,
    gq: Vec<f32>,
    gate_q: Vec<f32>,
    up_q: Vec<f32>,
    down_q: Vec<f32>,
}

fn fixture() -> Fixture {
    let qf = det(Q_ROWS * HIDDEN, 1);
    let kf = det(KV_ROWS * HIDDEN, 2);
    let vf = det(KV_ROWS * HIDDEN, 3);
    let of = det(HIDDEN * Q_ROWS, 4);
    let gf = det(Q_ROWS * HIDDEN, 5);
    let gatef = det(INTER * HIDDEN, 10);
    let upf = det(INTER * HIDDEN, 11);
    let downf = det(HIDDEN * INTER, 12);
    Fixture {
        h: det(HIDDEN, 6),
        norm_w: det(HIDDEN, 7),
        ffn_norm_w: det(HIDDEN, 13),
        attn_post_w: det(HIDDEN, 51),
        ffn_post_w: det(HIDDEN, 52),
        k_cache: det(T * KV_ROWS, 8),
        v_cache: det(T * KV_ROWS, 9),
        q: nvfp4::quantize(&qf, Q_ROWS, HIDDEN).unwrap(),
        k: nvfp4::quantize(&kf, KV_ROWS, HIDDEN).unwrap(),
        v: nvfp4::quantize(&vf, KV_ROWS, HIDDEN).unwrap(),
        o: nvfp4::quantize(&of, HIDDEN, Q_ROWS).unwrap(),
        g: nvfp4::quantize(&gf, Q_ROWS, HIDDEN).unwrap(),
        gate: nvfp4::quantize(&gatef, INTER, HIDDEN).unwrap(),
        up: nvfp4::quantize(&upf, INTER, HIDDEN).unwrap(),
        down: nvfp4::quantize(&downf, HIDDEN, INTER).unwrap(),
        // The reference consumes the quantised weights so the comparison
        // isolates the lowering from quantisation error.
        qq: nvfp4::round_trip(&qf, Q_ROWS, HIDDEN).unwrap(),
        kq: nvfp4::round_trip(&kf, KV_ROWS, HIDDEN).unwrap(),
        vq: nvfp4::round_trip(&vf, KV_ROWS, HIDDEN).unwrap(),
        oq: nvfp4::round_trip(&of, HIDDEN, Q_ROWS).unwrap(),
        gq: nvfp4::round_trip(&gf, Q_ROWS, HIDDEN).unwrap(),
        gate_q: nvfp4::round_trip(&gatef, INTER, HIDDEN).unwrap(),
        up_q: nvfp4::round_trip(&upf, INTER, HIDDEN).unwrap(),
        down_q: nvfp4::round_trip(&downf, HIDDEN, INTER).unwrap(),
    }
}

/// How the lowered layer schedules and allocates. The numbers must not
/// depend on either choice.
#[derive(Clone, Copy, PartialEq)]
enum Schedule {
    /// Both fragments in one encoder, one command buffer — the G6c shape.
    OneCommandBuffer,
    /// A command buffer each, committed and waited in turn. Nothing can
    /// race, so this is the conservative oracle.
    TwoCommandBuffers,
}

#[derive(Clone, Copy, PartialEq)]
enum ScratchPlan {
    /// Every intermediate gets its own buffer.
    Disjoint,
    /// The FFN reuses attention's hidden-sized intermediates, as 52
    /// layers sharing one pool must.
    Shared,
}

fn run_lowered(
    gpu: &larql_compute_metal::MetalBackend,
    f: &Fixture,
    schedule: Schedule,
    scratch: ScratchPlan,
) -> Vec<f32> {
    let h0 = gpu.lowering_upload(&f.h).unwrap();
    let attn_norm_buf = gpu.lowering_upload(&f.norm_w).unwrap();
    let ffn_norm_buf = gpu.lowering_upload(&f.ffn_norm_w).unwrap();
    let k_cache = gpu.lowering_upload(&f.k_cache).unwrap();
    let v_cache = gpu.lowering_upload(&f.v_cache).unwrap();
    let inv_freq = gpu.lowering_upload(&inv_freq_table()).unwrap();

    let h1 = gpu.lowering_scratch(HIDDEN);
    let h2 = gpu.lowering_scratch(HIDDEN);
    let attn_normed = gpu.lowering_scratch(HIDDEN);
    let q = gpu.lowering_scratch(Q_ROWS);
    let gate_v = gpu.lowering_scratch(Q_ROWS);
    let concat = gpu.lowering_scratch(Q_ROWS);
    let gated = gpu.lowering_scratch(Q_ROWS);
    let attn_out = gpu.lowering_scratch(HIDDEN);
    let attn_post_scratch = gpu.lowering_scratch(HIDDEN);
    let ffn_post_scratch = gpu.lowering_scratch(HIDDEN);
    let attn_post_buf = gpu.lowering_upload(&f.attn_post_w).unwrap();
    let ffn_post_buf = gpu.lowering_upload(&f.ffn_post_w).unwrap();
    let ffn_g = gpu.lowering_scratch(INTER);
    let ffn_u = gpu.lowering_scratch(INTER);
    let ffn_a = gpu.lowering_scratch(INTER);
    // Under `Shared`, the FFN's two hidden-sized intermediates alias
    // attention's. Legal only because the serial encoder orders every
    // FFN write after attention's last read — which is what this run
    // is here to check.
    let ffn_normed_own = gpu.lowering_scratch(HIDDEN);
    let ffn_down_own = gpu.lowering_scratch(HIDDEN);
    let (ffn_normed, ffn_down) = match scratch {
        ScratchPlan::Disjoint => (&ffn_normed_own, &ffn_down_own),
        ScratchPlan::Shared => (&attn_normed, &attn_out),
    };

    let proj = |m: &nvfp4::Nvfp4Matrix| LoweredMatrix::Nvfp4 {
        packed: Box::leak(Box::new(gpu.lowering_weight(&m.packed))),
        scales: Box::leak(Box::new(gpu.lowering_weight(&m.scales))),
        tensor_scale: m.tensor_scale,
    };
    let aw = AttnWeights {
        q: proj(&f.q),
        k: proj(&f.k),
        v: proj(&f.v),
        o: proj(&f.o),
        gate: Some(proj(&f.g)),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        sinks: None,
        norm_weight: &attn_norm_buf,
        post_norm: Some(larql_compute_metal::lowering::PostNorm {
            weight: &attn_post_buf,
            eps: POST_EPS,
            weight_offset: NORM_OFFSET,
            scratch: &attn_post_scratch,
        }),
    };
    let ascratch = AttnScratch {
        normed: &attn_normed,
        q: &q,
        k_cache: &k_cache,
        v_cache: &v_cache,
        gate: &gate_v,
        concat: &concat,
        gated: &gated,
        attn_out: &attn_out,
        inv_freq: &inv_freq,
    };
    let ashape = AttnShape {
        hidden: HIDDEN,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        head_dim: HEAD_DIM,
        norm_eps: EPS,
        norm_weight_offset: NORM_OFFSET,
        qk_norm_eps: QK_EPS,
        parameter_free_q: true,
        parameter_free_k: true,
        query_scale: Some(QUERY_SCALE),
        score_scale: SCORE_SCALE,
        position: LoweredPosition::Rope { theta: THETA },
        window: None,
        softcap: None,
        position_index: POS,
        kv_len: T,
    };

    let gate_buf = gpu.lowering_weight(&f.gate.packed);
    let gate_sc = gpu.lowering_weight(&f.gate.scales);
    let up_buf = gpu.lowering_weight(&f.up.packed);
    let up_sc = gpu.lowering_weight(&f.up.scales);
    let down_buf = gpu.lowering_weight(&f.down.packed);
    let down_sc = gpu.lowering_weight(&f.down.scales);
    let fw = FfnWeights {
        gate: LoweredMatrix::Nvfp4 {
            packed: &gate_buf,
            scales: &gate_sc,
            tensor_scale: f.gate.tensor_scale,
        },
        up: LoweredMatrix::Nvfp4 {
            packed: &up_buf,
            scales: &up_sc,
            tensor_scale: f.up.tensor_scale,
        },
        down: LoweredMatrix::Nvfp4 {
            packed: &down_buf,
            scales: &down_sc,
            tensor_scale: f.down.tensor_scale,
        },
        norm_weight: &ffn_norm_buf,
        post_norm: Some(larql_compute_metal::lowering::PostNorm {
            weight: &ffn_post_buf,
            eps: POST_EPS,
            weight_offset: NORM_OFFSET,
            scratch: &ffn_post_scratch,
        }),
    };
    let fscratch = FfnScratch {
        normed: ffn_normed,
        gate: &ffn_g,
        up: &ffn_u,
        act: &ffn_a,
        down: ffn_down,
    };
    let fshape = FfnShape {
        hidden: HIDDEN,
        intermediate: INTER,
        norm_eps: EPS,
        norm_weight_offset: NORM_OFFSET,
    };

    match schedule {
        Schedule::OneCommandBuffer => {
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            gpu.encode_attention(enc, &h0, &h1, &aw, &ascratch, &ashape);
            gpu.encode_gated_ffn(enc, &h1, &h2, &fw, &fscratch, &fshape);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        Schedule::TwoCommandBuffers => {
            let c1 = gpu.new_lowering_command_buffer();
            let e1 = c1.new_compute_command_encoder();
            gpu.encode_attention(e1, &h0, &h1, &aw, &ascratch, &ashape);
            e1.end_encoding();
            c1.commit();
            c1.wait_until_completed();

            let c2 = gpu.new_lowering_command_buffer();
            let e2 = c2.new_compute_command_encoder();
            gpu.encode_gated_ffn(e2, &h1, &h2, &fw, &fscratch, &fshape);
            e2.end_encoding();
            c2.commit();
            c2.wait_until_completed();
        }
    }

    let out = gpu.lowering_readback(&h2, HIDDEN).unwrap();
    for b in [
        h0,
        attn_norm_buf,
        ffn_norm_buf,
        k_cache,
        v_cache,
        inv_freq,
        h1,
        h2,
        attn_normed,
        q,
        gate_v,
        concat,
        gated,
        attn_out,
        ffn_g,
        ffn_u,
        ffn_a,
        ffn_normed_own,
        ffn_down_own,
    ] {
        gpu.recycle_lowering_scratch(b);
    }
    out
}

fn assert_control(what: &str, perturbed: &[f32], got: &[f32], parity: f64) {
    let c = rel_rms(perturbed, got);
    let ratio = c / parity;
    eprintln!("  control `{what}`: {ratio:.0}x parity ({c:.3e})");
    assert!(
        ratio > CONTROL_MARGIN,
        "control `{what}` moves the result only {ratio:.1}x the parity residual \
         ({c:.3e} vs {parity:.3e}) — the positive assertion cannot distinguish \
         this defect, so passing it proves nothing"
    );
}

#[test]
fn lowered_layer_composes_both_fragments_across_the_seam() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let f = fixture();

    let reference = layer_reference(&f, Seam::Correct, true, true);
    let got = run_lowered(&gpu, &f, Schedule::OneCommandBuffer, ScratchPlan::Disjoint);

    assert!(
        got.iter().all(|v| v.is_finite()),
        "lowered layer produced non-finite output"
    );
    let parity = rel_rms(&reference, &got);
    let cos = cosine(&reference, &got);
    eprintln!(
        "lowered layer vs CPU program: max_abs {:.3e}  rel_rms {parity:.3e}  cosine {cos:.9}",
        max_abs(&reference, &got)
    );
    assert!(
        parity < 1e-4 && cos > 0.999_999,
        "lowered layer disagrees with its own program: rel_rms {parity:.3e}, cosine {cos:.9}"
    );

    // ── Seam control 1: the FFN adds to the attention output ─────────
    assert_control(
        "FFN residual source is the attention output",
        &layer_reference(&f, Seam::ResidualFromLayerInput, true, true),
        &got,
        parity,
    );

    // ── Seam control 2: the FFN normalises the attention output, not
    //    attention's normed scratch ────────────────────────────────────
    assert_control(
        "FFN norm input is the attention output",
        &layer_reference(&f, Seam::NormsAttentionScratch, true, true),
        &got,
        parity,
    );

    // ── Control 3/4: both residuals survive composition ──────────────
    assert_control(
        "attention residual",
        &layer_reference(&f, Seam::Correct, false, true),
        &got,
        parity,
    );
    assert_control(
        "FFN residual",
        &layer_reference(&f, Seam::Correct, true, false),
        &got,
        parity,
    );
}

#[test]
fn one_command_buffer_is_bit_identical_to_two() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let f = fixture();
    let one = run_lowered(&gpu, &f, Schedule::OneCommandBuffer, ScratchPlan::Disjoint);
    let two = run_lowered(&gpu, &f, Schedule::TwoCommandBuffers, ScratchPlan::Disjoint);

    // Same kernels, same arguments, same per-row reduction order — only
    // the scheduling differs. Any difference is a dependency or lifetime
    // hazard, not float noise, so the bar is exact.
    let differing = one
        .iter()
        .zip(&two)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "one-CB and two-CB composition differ in {differing}/{HIDDEN} lanes \
         (max_abs {:.3e}) — the fragments are not correctly ordered within a \
         single encoder",
        max_abs(&two, &one)
    );
}

#[test]
fn sharing_hidden_scratch_between_fragments_changes_nothing() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let f = fixture();
    let disjoint = run_lowered(&gpu, &f, Schedule::OneCommandBuffer, ScratchPlan::Disjoint);
    let shared = run_lowered(&gpu, &f, Schedule::OneCommandBuffer, ScratchPlan::Shared);

    // G6c reuses one scratch pool across 52 layers. If aliasing
    // attention's `normed` and `attn_out` into the FFN perturbs a single
    // bit, that plan is unsound and this is where it must surface.
    let differing = disjoint
        .iter()
        .zip(&shared)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "sharing hidden-sized scratch across the seam changed {differing}/{HIDDEN} \
         lanes (max_abs {:.3e}) — an FFN write lands before attention's last read",
        max_abs(&disjoint, &shared)
    );
}
