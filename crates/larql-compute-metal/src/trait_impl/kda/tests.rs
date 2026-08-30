//! A whole KDA attention step against a scalar reference, at a
//! hand-checkable geometry.
//!
//! The real-weight gate lives in `larql-vindex`, where the proven CPU
//! operator is — but the kernels live here, and a shader this crate
//! ships needs a gate this crate runs. The reference below is a literal
//! transcription of `exec::kda::step`, written independently of the
//! shaders it scores.

use super::*;
use crate::MetalBackend;

const HEADS: usize = 3;
const DIM: usize = 4;
const HIDDEN: usize = 6;
const KERNEL: usize = 4;
const WIDTH: usize = HEADS * DIM;

/// Loose enough for a threadgroup reduction against a serial one, tight
/// enough that any real error is orders past it.
const TOLERANCE: f32 = 1e-5;

fn shape() -> KdaShape {
    KdaShape {
        hidden: HIDDEN,
        num_heads: HEADS,
        head_dim: DIM,
        conv_kernel: KERNEL,
    }
}

fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
        .collect()
}

fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// `[n, k]` bf16 codes as little-endian bytes, and the exact f32 values
/// they denote — the oracle must score the STORED weights.
fn bf16_matrix(n: usize, k: usize, seed: f32) -> (Vec<u8>, Vec<f32>) {
    let values = synth(n * k, seed);
    let codes: Vec<u16> = values.iter().map(|v| narrow(*v)).collect();
    let exact: Vec<f32> = codes
        .iter()
        .map(|c| f32::from_bits((*c as u32) << 16))
        .collect();
    (codes.iter().flat_map(|c| c.to_le_bytes()).collect(), exact)
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

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn softplus(v: f32) -> f32 {
    if v > 20.0 {
        v
    } else {
        v.exp().ln_1p()
    }
}

/// Everything one step needs, owned.
struct Weights {
    qkv_bank: Vec<u8>,
    qkv_offsets: [ExpertOffset; 3],
    qkv_exact: [Vec<f32>; 3],
    o_bytes: Vec<u8>,
    o_exact: Vec<f32>,
    conv: [Vec<f32>; 3],
    fa: Vec<f32>,
    fb: Vec<f32>,
    ga: Vec<f32>,
    gb: Vec<f32>,
    bp: Vec<f32>,
    a_log: Vec<f32>,
    dt: Vec<f32>,
    o_norm: Vec<f32>,
    eps: f32,
}

fn weights() -> Weights {
    let per = WIDTH * HIDDEN;
    let (qb, qe) = bf16_matrix(WIDTH, HIDDEN, 0.1);
    let (kb, ke) = bf16_matrix(WIDTH, HIDDEN, 1.3);
    let (vb, ve) = bf16_matrix(WIDTH, HIDDEN, 2.7);
    let (ob, oe) = bf16_matrix(HIDDEN, WIDTH, 3.9);
    let mut bank = Vec::with_capacity(3 * per * 2);
    for b in [&qb, &kb, &vb] {
        bank.extend_from_slice(b);
    }
    Weights {
        qkv_bank: bank,
        qkv_offsets: [
            ExpertOffset(0),
            ExpertOffset((per * 2) as u32),
            ExpertOffset((2 * per * 2) as u32),
        ],
        qkv_exact: [qe, ke, ve],
        o_bytes: ob,
        o_exact: oe,
        conv: [
            synth(WIDTH * KERNEL, 0.5),
            synth(WIDTH * KERNEL, 1.5),
            synth(WIDTH * KERNEL, 2.5),
        ],
        fa: synth(DIM * HIDDEN, 4.1),
        fb: synth(WIDTH * DIM, 5.2),
        ga: synth(DIM * HIDDEN, 6.3),
        gb: synth(WIDTH * DIM, 7.4),
        bp: synth(HEADS * HIDDEN, 8.5),
        a_log: synth(HEADS, 9.6),
        dt: synth(WIDTH, 10.7),
        o_norm: synth(DIM, 11.8).iter().map(|v| v + 1.0).collect(),
        eps: 1e-5,
    }
}

impl Weights {
    fn device(&self) -> KdaDeviceWeights<'_> {
        KdaDeviceWeights {
            qkv_bank: &self.qkv_bank,
            qkv_offsets: &self.qkv_offsets,
            o_proj: &self.o_bytes,
            q_conv1d: &self.conv[0],
            k_conv1d: &self.conv[1],
            v_conv1d: &self.conv[2],
            f_a_proj: &self.fa,
            f_b_proj: &self.fb,
            g_a_proj: &self.ga,
            g_b_proj: &self.gb,
            b_proj: &self.bp,
            a_log: &self.a_log,
            dt_bias: &self.dt,
            o_norm: &self.o_norm,
            norm_eps: self.eps,
        }
    }
}

/// The host state the reference carries between steps.
#[derive(Clone)]
struct RefState {
    recurrent: Vec<f32>,
    conv: [Vec<f32>; 3],
}

impl RefState {
    fn zeros() -> Self {
        let tail = WIDTH * (KERNEL - 1);
        Self {
            recurrent: vec![0.0; HEADS * DIM * DIM],
            conv: [vec![0.0; tail], vec![0.0; tail], vec![0.0; tail]],
        }
    }
}

/// A literal transcription of `exec::kda::step`, in the same order.
fn reference_step(w: &Weights, st: &mut RefState, x: &[f32]) -> Vec<f32> {
    let tail = KERNEL - 1;
    let mut streams: [Vec<f32>; 3] = Default::default();
    for (i, exact) in w.qkv_exact.iter().enumerate() {
        let p = matvec(exact, x, WIDTH);
        let mut out = vec![0.0f32; WIDTH];
        for (c, (o, pc)) in out.iter_mut().zip(&p).enumerate() {
            let cw = &w.conv[i][c * KERNEL..(c + 1) * KERNEL];
            let hist = &st.conv[i][c * tail..(c + 1) * tail];
            let mut acc = 0.0f32;
            for (j, cwj) in cw.iter().enumerate().take(tail) {
                acc += cwj * hist[j];
            }
            acc += cw[tail] * pc;
            *o = silu(acc);
        }
        for (c, pc) in p.iter().enumerate() {
            let hist = &mut st.conv[i][c * tail..(c + 1) * tail];
            for j in 0..tail - 1 {
                hist[j] = hist[j + 1];
            }
            hist[tail - 1] = *pc;
        }
        streams[i] = out;
    }
    let [mut q, mut k, v] = streams;
    for stream in [&mut q, &mut k] {
        for h in 0..HEADS {
            let head = &mut stream[h * DIM..(h + 1) * DIM];
            let n = head.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv = 1.0 / n.max(1e-12);
            for e in head.iter_mut() {
                *e *= inv;
            }
        }
    }

    let f_low = matvec(&w.fb, &matvec(&w.fa, x, DIM), WIDTH);
    let decay: Vec<f32> = (0..WIDTH)
        .map(|i| -w.a_log[i / DIM].exp() * softplus(f_low[i] + w.dt[i]))
        .collect();
    let gate = matvec(&w.gb, &matvec(&w.ga, x, DIM), WIDTH);
    let beta: Vec<f32> = matvec(&w.bp, x, HEADS)
        .iter()
        .map(|v| 1.0 / (1.0 + (-v).exp()))
        .collect();

    let scale = (DIM as f32).powf(-0.5);
    let mut out = [0.0f32; WIDTH];
    for h in 0..HEADS {
        let s = &mut st.recurrent[h * DIM * DIM..(h + 1) * DIM * DIM];
        let (qh, kh, vh) = (&q[h * DIM..], &k[h * DIM..], &v[h * DIM..]);
        let mut pred = [0.0f32; DIM];
        for kk in 0..DIM {
            let d = decay[h * DIM + kk].exp();
            for vv in 0..DIM {
                s[kk * DIM + vv] *= d;
                pred[vv] += kh[kk] * s[kk * DIM + vv];
            }
        }
        let err: Vec<f32> = (0..DIM).map(|vv| vh[vv] - pred[vv]).collect();
        for kk in 0..DIM {
            let write = beta[h] * kh[kk];
            let qv = qh[kk] * scale;
            for vv in 0..DIM {
                let cell = &mut s[kk * DIM + vv];
                *cell += write * err[vv];
                out[h * DIM + vv] += qv * *cell;
            }
        }
    }

    let mut normed = [0.0f32; WIDTH];
    for h in 0..HEADS {
        let slice = &out[h * DIM..(h + 1) * DIM];
        let ms = slice.iter().map(|v| v * v).sum::<f32>() / DIM as f32;
        let inv = (ms + w.eps).sqrt().recip();
        for (d, (sv, nv)) in slice.iter().zip(&w.o_norm).enumerate() {
            normed[h * DIM + d] = sv * inv * nv / (1.0 + (-gate[h * DIM + d]).exp());
        }
    }
    matvec(&w.o_exact, &normed, HIDDEN)
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

/// The whole step, and the state it leaves behind, across several
/// tokens — a recurrence that ignored its state, or advanced it
/// differently, diverges here and not at token one.
#[test]
fn the_device_step_matches_the_scalar_reference_across_tokens() {
    let m = backend();
    let w = weights();
    let shape = shape();
    let device = KdaDeviceState::zeros(&m, shape);
    let mut host = RefState::zeros();

    for t in 0..6 {
        let x = synth(HIDDEN, 20.0 + t as f32);
        let want = reference_step(&w, &mut host, &x);
        let (got, gpu) = m
            .kda_attention_step(w.device(), shape, &device, &x)
            .expect("device step");
        assert!(gpu >= 0.0, "the GPU window must be reported");
        assert!(
            max_abs(&got, &want) < TOLERANCE,
            "token {t}: max|Δ| {:e}",
            max_abs(&got, &want)
        );
        let (rec, conv) = device.read_back();
        assert!(
            max_abs(&rec, &host.recurrent) < TOLERANCE,
            "token {t} recurrent state: max|Δ| {:e}",
            max_abs(&rec, &host.recurrent)
        );
        for (i, (got, want)) in conv.iter().zip(&host.conv).enumerate() {
            assert!(max_abs(got, want) < TOLERANCE, "token {t} conv window {i}");
        }
    }
}

/// The traced variant must report the same output as the plain one, and
/// every plane it names must be the right length — a trace whose planes
/// were mis-sized would be read as a stage disagreement.
#[test]
fn the_traced_step_agrees_with_the_plain_one() {
    let m = backend();
    let w = weights();
    let shape = shape();
    let x = synth(HIDDEN, 3.0);

    let plain = KdaDeviceState::zeros(&m, shape);
    let (out, _) = m.kda_attention_step(w.device(), shape, &plain, &x).unwrap();
    let traced_state = KdaDeviceState::zeros(&m, shape);
    let p = m
        .kda_attention_step_traced(w.device(), shape, &traced_state, &x)
        .expect("traced step");

    assert_eq!(p.output, out, "tracing must not change the answer");
    for (name, len) in [
        ("q_proj", p.q_proj.len()),
        ("k_proj", p.k_proj.len()),
        ("v_proj", p.v_proj.len()),
        ("q_conv", p.q_conv.len()),
        ("k_conv", p.k_conv.len()),
        ("v_conv", p.v_conv.len()),
        ("q_norm", p.q_norm.len()),
        ("k_norm", p.k_norm.len()),
        ("f_lowrank", p.f_lowrank.len()),
        ("g_decay", p.g_decay.len()),
        ("recurrent_out", p.recurrent_out.len()),
        ("o_gate", p.o_gate.len()),
        ("o_norm", p.o_norm.len()),
    ] {
        assert_eq!(len, WIDTH, "{name} should be width-long");
    }
    assert_eq!(p.beta.len(), HEADS);
    assert_eq!(p.output.len(), HIDDEN);

    // The q/k planes must be L2-normalised per head, which is the one
    // property a wrong reduction would quietly break.
    for h in 0..HEADS {
        for plane in [&p.q_norm, &p.k_norm] {
            let n = plane[h * DIM..(h + 1) * DIM]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            assert!((n - 1.0).abs() < 1e-5, "head {h} norm {n}");
        }
    }
}

/// Shape faults refuse rather than reading out of bounds, and refuse
/// before the encoder opens — Metal aborts the process if a compute
/// encoder is dropped without `end_encoding`.
#[test]
fn shape_faults_are_refused_before_the_encoder_opens() {
    let m = backend();
    let w = weights();
    let shape = shape();
    let state = KdaDeviceState::zeros(&m, shape);

    let short_x = synth(HIDDEN - 1, 0.0);
    assert!(matches!(
        m.kda_attention_step(w.device(), shape, &state, &short_x),
        Err(GroupedError::OffsetOutOfRange { .. })
    ));

    let mut truncated = w.device();
    let half = &w.qkv_bank[..w.qkv_bank.len() / 2];
    truncated.qkv_bank = half;
    assert!(matches!(
        m.kda_attention_step(truncated, shape, &state, &synth(HIDDEN, 0.0)),
        Err(GroupedError::OffsetOutOfRange { .. })
    ));

    let other = KdaShape {
        head_dim: DIM * 2,
        ..shape
    };
    assert!(matches!(
        m.kda_attention_step(w.device(), other, &state, &synth(HIDDEN, 0.0)),
        Err(GroupedError::SlotCountMismatch { .. })
    ));

    // Still usable, which is the real assertion.
    assert!(m
        .kda_attention_step(w.device(), shape, &state, &synth(HIDDEN, 1.0))
        .is_ok());
}

/// `KdaShape`'s derived quantities, and that `zeroed` really is zero —
/// a recurrent state starting from a recycled buffer's leftovers would
/// produce a plausible wrong answer on token one.
#[test]
fn the_state_starts_at_zero_and_the_shape_derives_its_widths() {
    let m = backend();
    let shape = shape();
    assert_eq!(shape.width(), WIDTH);
    let state = KdaDeviceState::zeros(&m, shape);
    let (rec, conv) = state.read_back();
    assert_eq!(rec.len(), HEADS * DIM * DIM);
    assert!(
        rec.iter().all(|v| *v == 0.0),
        "recurrent state must start zero"
    );
    for c in &conv {
        assert_eq!(c.len(), WIDTH * (KERNEL - 1));
        assert!(c.iter().all(|v| *v == 0.0), "conv window must start zero");
    }
}
