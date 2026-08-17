//! A-9.4: the lowered attention fragment executes the three GPT-OSS
//! semantics the interpreter proved — attention **sinks**, **Q/K/V/O
//! projection biases**, and **YaRN** (scaled inverse frequencies plus a
//! `cos`/`sin` amplitude) — and each is load-bearing.
//!
//! The reference is transcribed from the interpreter's own kernels
//! (`condition_qk_in_place` biases, `yarn_frequencies`, `softmax_with_sink`),
//! in f32, consuming the same round-tripped NVFP4 weights the lowering
//! reads, so quantisation error cannot masquerade as a semantic gap. As in
//! `test_lowering_attention_parity`, agreement is necessary and not
//! sufficient: each judged fact carries a control that must dwarf the
//! parity residual.
//!
//! ```text
//! parity     lowered attention ≡ reference (biases + YaRN + sink)
//! sink       drop the sink logits              → output moves
//! bias       drop the Q bias                   → output moves
//! amplitude  YaRN amplitude → 1.0              → output moves (position 0 too)
//! frequency  YaRN inv_freq → plain rope        → output moves
//! ```

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::attention::{
    AttnScratch, AttnShape, AttnWeights, LoweredPosition,
};
use larql_compute_metal::lowering::LoweredMatrix;
use larql_models::quant::nvfp4;
use larql_models::YarnRopeScaling;

const HIDDEN: usize = 256;
const NUM_Q: usize = 8;
const NUM_KV: usize = 2;
const HEAD_DIM: usize = 32;
const Q_ROWS: usize = NUM_Q * HEAD_DIM;
const KV_ROWS: usize = NUM_KV * HEAD_DIM;
const T: usize = 10;
const POS: usize = T - 1;
const EPS: f32 = 1e-5;
const SCORE_SCALE: f32 = 0.176_776_7; // 1/sqrt(32)
const THETA: f64 = 150_000.0;

fn det(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(9);
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

/// GPT-OSS's published YaRN block.
fn yarn() -> YarnRopeScaling {
    YarnRopeScaling {
        factor: 32.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position_embeddings: 4096.0,
        truncate: false,
        mscale: None,
        mscale_all_dim: None,
    }
}

/// YaRN inverse frequencies and amplitude, transcribed from HF's
/// `_compute_yarn_parameters` (the same maths the interpreter's
/// `kernels::yarn_frequencies` and the lowering both use). Independent
/// here so the test does not depend on the vindex crate.
fn yarn_inv_freq(scaling: &YarnRopeScaling) -> (Vec<f64>, f32) {
    let d = HEAD_DIM as f64;
    let correction_dim = |rot: f64| {
        (d * (scaling.original_max_position_embeddings / (rot * std::f64::consts::TAU)).ln())
            / (2.0 * THETA.ln())
    };
    let mut low = correction_dim(scaling.beta_fast);
    let mut high = correction_dim(scaling.beta_slow);
    if scaling.truncate {
        low = low.floor();
        high = high.ceil();
    }
    let low = low.max(0.0);
    let high = high.min(d - 1.0);
    let high = if high == low { high + 0.001 } else { high };
    let inv_freq = (0..HEAD_DIM / 2)
        .map(|i| {
            let extrap = THETA.powf(-2.0 * i as f64 / d);
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            extrap / scaling.factor * ramp + extrap * (1.0 - ramp)
        })
        .collect();
    (inv_freq, scaling.attention_amplitude() as f32)
}

fn plain_inv_freq() -> Vec<f64> {
    (0..HEAD_DIM / 2)
        .map(|i| THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64))
        .collect()
}

fn matvec(m: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|r| (0..k).map(|c| m[r * k + c] * x[c]).sum())
        .collect()
}

fn rope_scaled(v: &mut [f32], heads: usize, pos: usize, inv_freq: &[f64], amplitude: f32) {
    let half = HEAD_DIM / 2;
    for h in 0..heads {
        let off = h * HEAD_DIM;
        for i in 0..half {
            let a = pos as f64 * inv_freq[i];
            let s = a.sin() as f32 * amplitude;
            let c = a.cos() as f32 * amplitude;
            let (x0, x1) = (v[off + i], v[off + half + i]);
            v[off + i] = x0 * c - x1 * s;
            v[off + half + i] = x0 * s + x1 * c;
        }
    }
}

struct Extras {
    sink: bool,
    q_bias: bool,
    inv_freq: Vec<f64>,
    amplitude: f32,
}

#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    h: &[f32],
    norm_w: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    qb: &[f32],
    kb: &[f32],
    vb: &[f32],
    ob: &[f32],
    sinks: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    e: &Extras,
) -> Vec<f32> {
    let ms = h.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32;
    let inv = 1.0 / (ms + EPS).sqrt();
    let normed: Vec<f32> = h.iter().zip(norm_w).map(|(x, w)| x * inv * w).collect();

    let mut q = matvec(wq, &normed, Q_ROWS, HIDDEN);
    let mut k = matvec(wk, &normed, KV_ROWS, HIDDEN);
    let mut v = matvec(wv, &normed, KV_ROWS, HIDDEN);
    // Biases join the projections before anything reads them.
    if e.q_bias {
        for (x, b) in q.iter_mut().zip(qb) {
            *x += b;
        }
    }
    for (x, b) in k.iter_mut().zip(kb) {
        *x += b;
    }
    for (x, b) in v.iter_mut().zip(vb) {
        *x += b;
    }

    rope_scaled(&mut q, NUM_Q, POS, &e.inv_freq, e.amplitude);
    rope_scaled(&mut k, NUM_KV, POS, &e.inv_freq, e.amplitude);

    let mut kc = k_cache.to_vec();
    let mut vc = v_cache.to_vec();
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
        let mut m = scores.iter().cloned().fold(f32::MIN, f32::max);
        if e.sink {
            m = m.max(sinks[head]);
        }
        let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let mut denom: f32 = exps.iter().sum();
        if e.sink {
            denom += (sinks[head] - m).exp();
        }
        for d in 0..HEAD_DIM {
            let mut acc = 0.0f32;
            for (i, vt) in (0..T).enumerate() {
                acc += exps[i] / denom * vc[vt * KV_ROWS + kv * HEAD_DIM + d];
            }
            concat[head * HEAD_DIM + d] = acc;
        }
    }

    let mut out = matvec(wo, &concat, HIDDEN, Q_ROWS);
    for (x, b) in out.iter_mut().zip(ob) {
        *x += b;
    }
    h.iter().zip(&out).map(|(a, b)| a + b).collect()
}

struct Fixture {
    h: Vec<f32>,
    norm_w: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    qb: Vec<f32>,
    kb: Vec<f32>,
    vb: Vec<f32>,
    ob: Vec<f32>,
    sinks: Vec<f32>,
    q: nvfp4::Nvfp4Matrix,
    k: nvfp4::Nvfp4Matrix,
    v: nvfp4::Nvfp4Matrix,
    o: nvfp4::Nvfp4Matrix,
    #[allow(dead_code)]
    qq: Vec<f32>,
    kq: Vec<f32>,
    vq: Vec<f32>,
    oq: Vec<f32>,
}

fn fixture() -> Fixture {
    let qf = det(Q_ROWS * HIDDEN, 1);
    let kf = det(KV_ROWS * HIDDEN, 2);
    let vf = det(KV_ROWS * HIDDEN, 3);
    let of = det(HIDDEN * Q_ROWS, 4);
    Fixture {
        h: det(HIDDEN, 5),
        // Norm weight near 1 (this fixture folds no offset in).
        norm_w: det(HIDDEN, 6).iter().map(|w| 1.0 + w).collect(),
        k_cache: det(T * KV_ROWS, 8),
        v_cache: det(T * KV_ROWS, 9),
        qb: det(Q_ROWS, 10).iter().map(|x| x * 4.0).collect(),
        kb: det(KV_ROWS, 11).iter().map(|x| x * 4.0).collect(),
        vb: det(KV_ROWS, 12).iter().map(|x| x * 4.0).collect(),
        ob: det(HIDDEN, 13).iter().map(|x| x * 4.0).collect(),
        sinks: det(NUM_Q, 14).iter().map(|x| x * 4.0).collect(),
        q: nvfp4::quantize(&qf, Q_ROWS, HIDDEN).unwrap(),
        k: nvfp4::quantize(&kf, KV_ROWS, HIDDEN).unwrap(),
        v: nvfp4::quantize(&vf, KV_ROWS, HIDDEN).unwrap(),
        o: nvfp4::quantize(&of, HIDDEN, Q_ROWS).unwrap(),
        qq: nvfp4::round_trip(&qf, Q_ROWS, HIDDEN).unwrap(),
        kq: nvfp4::round_trip(&kf, KV_ROWS, HIDDEN).unwrap(),
        vq: nvfp4::round_trip(&vf, KV_ROWS, HIDDEN).unwrap(),
        oq: nvfp4::round_trip(&of, HIDDEN, Q_ROWS).unwrap(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_lowered(
    gpu: &larql_compute_metal::MetalBackend,
    f: &Fixture,
    inv_freq: &[f64],
    amplitude: f32,
    with_sink: bool,
    with_q_bias: bool,
) -> Vec<f32> {
    let h_in = gpu.lowering_upload(&f.h).unwrap();
    let norm_buf = gpu.lowering_upload(&f.norm_w).unwrap();
    let k_cache = gpu.lowering_upload(&f.k_cache).unwrap();
    let v_cache = gpu.lowering_upload(&f.v_cache).unwrap();
    let inv_freq_f32: Vec<f32> = inv_freq.iter().map(|x| *x as f32).collect();
    let inv_freq_buf = gpu.lowering_upload(&inv_freq_f32).unwrap();
    let qb = gpu.lowering_upload(&f.qb).unwrap();
    let kb = gpu.lowering_upload(&f.kb).unwrap();
    let vb = gpu.lowering_upload(&f.vb).unwrap();
    let ob = gpu.lowering_upload(&f.ob).unwrap();
    let sinks = gpu.lowering_upload(&f.sinks).unwrap();
    let h_out = gpu.lowering_scratch(HIDDEN);
    let normed = gpu.lowering_scratch(HIDDEN);
    let q = gpu.lowering_scratch(Q_ROWS);
    let gate = gpu.lowering_scratch(Q_ROWS);
    let concat = gpu.lowering_scratch(Q_ROWS);
    let gated = gpu.lowering_scratch(Q_ROWS);
    let attn_out = gpu.lowering_scratch(HIDDEN);

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
        gate: None,
        q_bias: with_q_bias.then_some(&qb),
        k_bias: Some(&kb),
        v_bias: Some(&vb),
        o_bias: Some(&ob),
        sinks: with_sink.then_some(&sinks),
        norm_weight: &norm_buf,
        post_norm: None,
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
        inv_freq: &inv_freq_buf,
    };
    let shape = AttnShape {
        hidden: HIDDEN,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        head_dim: HEAD_DIM,
        norm_eps: EPS,
        norm_weight_offset: 0.0,
        qk_norm_eps: EPS,
        parameter_free_q: false,
        parameter_free_k: false,
        query_scale: None,
        score_scale: SCORE_SCALE,
        position: LoweredPosition::Scaled {
            theta: THETA,
            amplitude,
        },
        window: None,
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
    for b in [
        h_in,
        norm_buf,
        k_cache,
        v_cache,
        inv_freq_buf,
        qb,
        kb,
        vb,
        ob,
        sinks,
        h_out,
        normed,
        q,
        gate,
        concat,
        gated,
        attn_out,
    ] {
        gpu.recycle_lowering_scratch(b);
    }
    out
}

fn assert_control(what: &str, parity: f64, moved: f64) {
    assert!(
        moved > parity * 20.0,
        "control `{what}` moved the result only {:.1}x the parity residual — the lowering \
         is not executing it (parity {parity:.3e}, moved {moved:.3e})",
        moved / parity,
    );
}

#[test]
fn lowered_attention_executes_sinks_biases_and_yarn() {
    let gpu = larql_compute_metal::MetalBackend::new().expect("metal");
    let f = fixture();
    let (yarn_if, amp) = yarn_inv_freq(&yarn());
    assert!(
        amp > 1.3,
        "gpt-oss YaRN amplitude should be ~1.35, got {amp}"
    );

    let full = Extras {
        sink: true,
        q_bias: true,
        inv_freq: yarn_if.clone(),
        amplitude: amp,
    };
    let reference = cpu_reference(
        &f.h, &f.norm_w, &f.qq, &f.kq, &f.vq, &f.oq, &f.qb, &f.kb, &f.vb, &f.ob, &f.sinks,
        &f.k_cache, &f.v_cache, &full,
    );
    let got = run_lowered(&gpu, &f, &yarn_if, amp, true, true);
    assert!(got.iter().all(|v| v.is_finite()));
    let parity = rel_rms(&reference, &got);
    assert!(parity < 1e-4, "lowered attention disagrees: {parity:.3e}");

    // Control: no sink — the softmax mass the sink held returns to the keys.
    let no_sink = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        &f.qb,
        &f.kb,
        &f.vb,
        &f.ob,
        &f.sinks,
        &f.k_cache,
        &f.v_cache,
        &Extras {
            sink: false,
            ..full_like(&full)
        },
    );
    assert_control("attention sink applied", parity, rel_rms(&no_sink, &got));

    // Control: no Q bias.
    let no_qbias = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        &f.qb,
        &f.kb,
        &f.vb,
        &f.ob,
        &f.sinks,
        &f.k_cache,
        &f.v_cache,
        &Extras {
            q_bias: false,
            ..full_like(&full)
        },
    );
    assert_control(
        "Q projection bias applied",
        parity,
        rel_rms(&no_qbias, &got),
    );

    // Control: amplitude 1.0 — the YaRN cos/sin scalar is executed.
    let no_amp = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        &f.qb,
        &f.kb,
        &f.vb,
        &f.ob,
        &f.sinks,
        &f.k_cache,
        &f.v_cache,
        &Extras {
            amplitude: 1.0,
            ..full_like(&full)
        },
    );
    assert_control("YaRN amplitude applied", parity, rel_rms(&no_amp, &got));

    // Control: plain rope frequencies — the ramped blend is executed.
    let plain = cpu_reference(
        &f.h,
        &f.norm_w,
        &f.qq,
        &f.kq,
        &f.vq,
        &f.oq,
        &f.qb,
        &f.kb,
        &f.vb,
        &f.ob,
        &f.sinks,
        &f.k_cache,
        &f.v_cache,
        &Extras {
            inv_freq: plain_inv_freq(),
            ..full_like(&full)
        },
    );
    assert_control("YaRN frequency ramp applied", parity, rel_rms(&plain, &got));

    // And the lowering built from the plain table disagrees with the YaRN
    // run — the device path reads the table it is given, not a default.
    let got_plain = run_lowered(&gpu, &f, &plain_inv_freq(), amp, true, true);
    assert!(rel_rms(&got_plain, &got) > parity * 20.0);
}

fn full_like(e: &Extras) -> Extras {
    Extras {
        sink: e.sink,
        q_bias: e.q_bias,
        inv_freq: e.inv_freq.clone(),
        amplitude: e.amplitude,
    }
}
