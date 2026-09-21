//! PHYSICAL-1 QUALITY gate: does Q4_K on KDA's two wide projections
//! preserve the RECURRENT STATE TRAJECTORY?
//!
//! A one-step output comparison cannot close this rung. A quantisation
//! error in a feed-forward projection perturbs one token; the same error
//! in a recurrent operator enters the state and is carried forward, so
//! the question is not "how far apart is the output" but "does the
//! distance grow with sequence length". This runs a fixed token sequence
//! through the same layer carrying state, and scores the STATE at every
//! step — the output is secondary evidence.
//!
//! Pre-registered in `docs/glm5-flash-funnel.md` §8.2.2 before any of
//! this ran. The bounds below are the declared ones; a run that misses
//! them is the rung's answer, not a reason to move them.
//!
//! **Geometry is not free here.** Q4_K blocks 256 elements along the
//! reduction axis and `o_proj` transposes that axis, so BOTH `hidden`
//! and `width` must be multiples of 256 — the Q8_0 gate's 64/32 shape is
//! illegal for Q4_K and would be refused by `matrix_bytes`. `head_dim`
//! is 128 here because that is what GLM (64x128) and Kimi (32x128)
//! actually run.

use super::trajectory::{drift_ratio, kda_trajectory, report, worst};
use super::*;
use crate::MetalBackend;

const HIDDEN: usize = 256;
const HEADS: usize = 2;
const DIM: usize = 128;
const WIDTH: usize = HEADS * DIM;
const KERNEL: usize = 4;

/// Steps in the three declared arms. The long one exists because an
/// error that is harmless at 8 steps and monotone by 512 is exactly the
/// failure this gate is for.
const SHORT: usize = 8;
const MEDIUM: usize = 128;
const LONG: usize = 512;

/// Declared bounds (§8.2.2). `DRIFT_RATIO` is the one with content: it
/// asks whether divergence ACCUMULATES, which a per-step bound cannot.
const MAX_STATE_REL: f32 = 0.10;
const MIN_STATE_COS: f32 = 0.995;
const MAX_DRIFT_RATIO: f32 = 3.0;

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

/// `[n, k]` bf16 codes and the exact values they denote. Both arms are
/// built from these exact values, so the ONLY difference between them is
/// the Q4_K roundtrip — never a second RNG draw.
fn bf16_matrix(n: usize, k: usize, seed: f32) -> (Vec<u8>, Vec<f32>) {
    let values = synth(n * k, seed);
    let codes: Vec<u16> = values.iter().map(|v| narrow(*v)).collect();
    let exact = codes
        .iter()
        .map(|c| f32::from_bits((*c as u32) << 16))
        .collect();
    (codes.iter().flat_map(|c| c.to_le_bytes()).collect(), exact)
}

fn q4k(v: &[f32]) -> Vec<u8> {
    larql_compute::cpu::ops::q4_common::quantize_q4_k(v)
}

/// The known-bad control's representation: values pre-rounded to a
/// coarse grid before Q4_K, so the BYTES are legal Q4_K read by the same
/// kernel and only the codes are poor. That isolates "a much worse
/// representation" from "a different kernel".
fn coarsen(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| (x * 4.0).round() / 4.0).collect()
}

/// Every arm's weights, all from one draw.
struct Substrate {
    bf16_qkv: Vec<u8>,
    bf16_o: Vec<u8>,
    bf16_offsets: [ExpertOffset; CONV_STREAMS],
    q4_qkv: Vec<u8>,
    q4_o: Vec<u8>,
    q4_offsets: [ExpertOffset; CONV_STREAMS],
    /// Same as `q4_qkv` but with `v_proj` coarsened — `v` is written
    /// INTO the recurrent state, so this is the control that must move
    /// the state.
    bad_v_qkv: Vec<u8>,
    /// Same, but `q_proj` coarsened. `q` only READS the state, so this
    /// one must move the output and leave the state alone — kept as a
    /// diagnostic, not as the control.
    bad_q_qkv: Vec<u8>,
    conv: [Vec<f32>; CONV_STREAMS],
    fa: Vec<f32>,
    fb: Vec<f32>,
    ga: Vec<f32>,
    gb: Vec<f32>,
    bp: Vec<f32>,
    a_log: Vec<f32>,
    dt: Vec<f32>,
    /// `dt_bias` shifted — the reachable stand-in for the decay-form
    /// control (see the module note in the wrong-decay test).
    dt_shifted: Vec<f32>,
    o_norm: Vec<f32>,
}

fn substrate() -> Substrate {
    let (qb, qe) = bf16_matrix(WIDTH, HIDDEN, 0.1);
    let (kb, ke) = bf16_matrix(WIDTH, HIDDEN, 1.3);
    let (vb, ve) = bf16_matrix(WIDTH, HIDDEN, 2.7);
    let (ob, oe) = bf16_matrix(HIDDEN, WIDTH, 3.9);

    let mut bf16_qkv = Vec::new();
    for b in [&qb, &kb, &vb] {
        bf16_qkv.extend_from_slice(b);
    }
    let per_bf16 = qb.len();

    let q4: Vec<Vec<u8>> = [&qe, &ke, &ve].iter().map(|e| q4k(e)).collect();
    let per_q4 = q4[0].len();
    let mut q4_qkv = Vec::new();
    for b in &q4 {
        assert_eq!(b.len(), per_q4, "q|k|v must share one stride");
        q4_qkv.extend_from_slice(b);
    }
    let mut bad_v_qkv = Vec::new();
    bad_v_qkv.extend_from_slice(&q4[0]);
    bad_v_qkv.extend_from_slice(&q4[1]);
    let bad_v = q4k(&coarsen(&ve));
    assert_eq!(bad_v.len(), per_q4);
    bad_v_qkv.extend_from_slice(&bad_v);

    let mut bad_q_qkv = q4k(&coarsen(&qe));
    assert_eq!(bad_q_qkv.len(), per_q4);
    bad_q_qkv.extend_from_slice(&q4[1]);
    bad_q_qkv.extend_from_slice(&q4[2]);

    let dt = synth(WIDTH, 10.7);
    Substrate {
        bf16_qkv,
        bf16_o: ob,
        bf16_offsets: [
            ExpertOffset(0),
            ExpertOffset(per_bf16 as u32),
            ExpertOffset((2 * per_bf16) as u32),
        ],
        q4_qkv,
        q4_o: q4k(&oe),
        q4_offsets: [
            ExpertOffset(0),
            ExpertOffset(per_q4 as u32),
            ExpertOffset((2 * per_q4) as u32),
        ],
        bad_v_qkv,
        bad_q_qkv,
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
        dt_shifted: dt.iter().map(|v| v + 0.5).collect(),
        dt,
        o_norm: synth(DIM, 11.8).iter().map(|v| v + 1.0).collect(),
    }
}

/// Which arm to build. The encoding and the bytes always travel
/// together — a mismatch would pair bytes with another encoding's kernel.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Arm {
    Bf16,
    Q4,
    /// Q4_K bytes whose `v_proj` codes are deliberately coarse — `v`
    /// enters the state.
    BadQ4V,
    /// Q4_K bytes whose `q_proj` codes are deliberately coarse — `q`
    /// only reads the state.
    BadQ4Q,
    /// Q4_K bytes, `dt_bias` shifted — moves the decay gate.
    Q4ShiftedDecay,
}

impl Substrate {
    fn device(&self, arm: Arm) -> KdaDeviceWeights<'_> {
        let (qkv, offsets, o, enc): (&[u8], _, &[u8], _) = match arm {
            Arm::Bf16 => (
                &self.bf16_qkv,
                &self.bf16_offsets,
                &self.bf16_o,
                ExpertEncoding::Bf16,
            ),
            Arm::BadQ4V => (
                &self.bad_v_qkv,
                &self.q4_offsets,
                &self.q4_o,
                ExpertEncoding::Q4K,
            ),
            Arm::BadQ4Q => (
                &self.bad_q_qkv,
                &self.q4_offsets,
                &self.q4_o,
                ExpertEncoding::Q4K,
            ),
            _ => (
                &self.q4_qkv,
                &self.q4_offsets,
                &self.q4_o,
                ExpertEncoding::Q4K,
            ),
        };
        KdaDeviceWeights {
            qkv_bank: qkv,
            qkv_offsets: offsets,
            o_proj: o,
            projection_encoding: enc,
            q_conv1d: &self.conv[0],
            k_conv1d: &self.conv[1],
            v_conv1d: &self.conv[2],
            f_a_proj: SmallMatrix::F32(&self.fa),
            f_b_proj: SmallMatrix::F32(&self.fb),
            g_a_proj: SmallMatrix::F32(&self.ga),
            g_b_proj: SmallMatrix::F32(&self.gb),
            b_proj: SmallMatrix::F32(&self.bp),
            a_log: &self.a_log,
            dt_bias: if arm == Arm::Q4ShiftedDecay {
                &self.dt_shifted
            } else {
                &self.dt
            },
            o_norm: &self.o_norm,
            norm_eps: 1e-5,
            // Kimi's form: this substrate mirrors Kimi's geometry.
            gate_form: larql_models::config::KdaGateForm::Softplus,
        }
    }
}

/// The token sequence both arms see. Deterministic, and identical
/// across arms — a divergence must come from the weights.
fn inputs(steps: usize) -> Vec<Vec<f32>> {
    (0..steps)
        .map(|step| synth(HIDDEN, 0.3 + step as f32 * 0.11))
        .collect()
}

fn trajectory(
    m: &MetalBackend,
    reference: Arm,
    candidate: Arm,
    steps: usize,
    reset_each_step: bool,
) -> Vec<super::trajectory::StepMetric> {
    let sub = substrate();
    kda_trajectory(
        m,
        shape(),
        sub.device(reference),
        sub.device(candidate),
        &inputs(steps),
        reset_each_step,
    )
}

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device")
}

/// The gate. Q4_K on the two wide projections must hold the recurrent
/// state across short, medium and long sequences, and the drift must not
/// accumulate.
#[test]
fn q4k_projections_hold_the_recurrent_state_trajectory() {
    let m = backend();
    for steps in [SHORT, MEDIUM, LONG] {
        let t = trajectory(&m, Arm::Bf16, Arm::Q4, steps, false);
        report(&format!("Q4_K vs BF16, {steps} steps"), &t, SHORT);
        let (rel, cos, _, _) = worst(&t);
        assert!(
            rel <= MAX_STATE_REL,
            "{steps} steps: recurrent state drifted {rel} > declared {MAX_STATE_REL}"
        );
        assert!(
            cos >= MIN_STATE_COS,
            "{steps} steps: state cosine {cos} < declared {MIN_STATE_COS}"
        );
        if steps == LONG {
            let d = drift_ratio(&t, SHORT);
            assert!(
                d < MAX_DRIFT_RATIO,
                "state divergence ACCUMULATED: {d}x from step {SHORT} to {steps}, \
                 declared bound {MAX_DRIFT_RATIO}. Q4_K is not safe for a recurrent \
                 operator at this depth."
            );
        }
    }
}

/// CONTROL 1 — a deliberately coarse Q4_K representation of `v_proj`
/// must move the trajectory. Without this the gate cannot claim it would
/// notice a bad representation.
///
/// **It has to be `v`, and finding that out cost a red run.** The first
/// version of this control coarsened `q_proj` and moved the state by
/// 0.000000 — bit-identical to the honest arm — while moving the output
/// by 2.6. `q` never enters the recurrent state; it only reads out of
/// it. A control on the query path cannot qualify a state witness. See
/// `a_query_path_defect_is_invisible_to_the_state_witness` below, which
/// pins that as a property rather than leaving it as a war story.
#[test]
fn control_a_known_bad_q4_moves_the_trajectory() {
    let m = backend();
    let good = trajectory(&m, Arm::Bf16, Arm::Q4, MEDIUM, false);
    let bad = trajectory(&m, Arm::Bf16, Arm::BadQ4V, MEDIUM, false);
    report("control: known-bad Q4 (v_proj)", &bad, SHORT);
    let (g, _, _, _) = worst(&good);
    let (b, _, _, _) = worst(&bad);
    assert!(
        b > g * 3.0,
        "known-bad Q4 moved the state only {b} against the honest arm's {g} — \
         the gate cannot distinguish a poor representation from a good one"
    );
}

/// CONTROL 2 — with the state reset between steps, no step can inherit
/// the previous one's divergence, so the carried arm must diverge more.
/// This is what proves the witness is reading RECURRENCE and not just
/// per-step arithmetic.
#[test]
fn control_b_state_reset_removes_the_accumulation() {
    let m = backend();
    let carried = trajectory(&m, Arm::Bf16, Arm::Q4, MEDIUM, false);
    let reset = trajectory(&m, Arm::Bf16, Arm::Q4, MEDIUM, true);
    report("control: state reset", &reset, SHORT);
    let (c, _, _, _) = worst(&carried);
    let (r, _, _, _) = worst(&reset);
    assert!(
        c > r,
        "carried state ({c}) did not diverge more than reset state ({r}) — \
         the trajectory metric is not actually seeing the recurrence"
    );
}

/// CONTROL 3 — the decay gate is the recurrence's own dial, so moving it
/// must move the trajectory far more than quantisation does.
///
/// **This is a STAND-IN, and the reason is a finding.** The declared
/// control is GLM's decay form against Kimi's. It is not reachable here:
/// `shaders::kda`'s `kda_decay_gate` computes
/// `-exp(a_log) * softplus(f_low + dt_bias)` — Kimi's form — with no
/// `gate_lower_bound` and no family selection, on `origin/main`.
/// `KdaGateForm::{Softplus, ClampedSigmoid}` exists in `larql-models` and
/// has never reached this executor, so GLM's
/// `lower_bound * sigmoid(exp(a_log) * (f_low + dt_bias))` cannot be
/// selected. Shifting `dt_bias` moves the same gate and is the reachable
/// analogue; the missing arm is recorded in the funnel doc as PHYSICAL-1's
/// blocker for its GLM subject.
#[test]
fn control_c_moving_the_decay_gate_moves_the_trajectory() {
    let m = backend();
    let honest = trajectory(&m, Arm::Bf16, Arm::Q4, MEDIUM, false);
    let shifted = trajectory(&m, Arm::Bf16, Arm::Q4ShiftedDecay, MEDIUM, false);
    report("control: shifted decay gate", &shifted, SHORT);
    let (h, _, _, _) = worst(&honest);
    let (s, _, _, _) = worst(&shifted);
    assert!(
        s > h * 3.0,
        "a shifted decay gate moved the state only {s} against {h} — the \
         trajectory witness is insensitive to the recurrence's own dial"
    );
}

/// Q4_K's block geometry is legal at BOTH real KDA geometries, checked
/// rather than assumed: `k % 256 == 0` on the reduction axis, and
/// `o_proj` transposes that axis so it is checked at its own `k`.
#[test]
fn q4k_is_legal_at_both_real_kda_geometries() {
    // (hidden, width) for GLM-5.3-Flash and Kimi-Linear-48B.
    for (name, hidden, width) in [("GLM", 4096usize, 8192usize), ("Kimi", 2304, 4096)] {
        assert!(
            ExpertEncoding::Q4K.matrix_bytes(width, hidden).is_some(),
            "{name}: q|k|v [{width}, {hidden}] is not Q4_K-legal"
        );
        assert!(
            ExpertEncoding::Q4K.matrix_bytes(hidden, width).is_some(),
            "{name}: o_proj [{hidden}, {width}] is not Q4_K-legal"
        );
    }
    // And the gate can fail: 2880 is not a multiple of 256.
    assert!(ExpertEncoding::Q4K.matrix_bytes(256, 2880).is_none());
}

/// The state witness is BLIND to the query path, and that is why the
/// output metric is kept beside it rather than dropped as redundant.
///
/// Scored Q4 against Q4-with-a-coarsened-`q_proj`, so the ONLY
/// difference between the arms is `q` — comparing both to BF16 instead
/// would bury it under the shared Q4 floor, which on this substrate is
/// `out_rel` ~0.15 against `q`'s ~0.002 contribution.
///
/// `q` reads the recurrent state and never writes it, so the state must
/// be **bit-identical** while the output moves. KDA's counterpart to
/// MLA's "position 0 cannot witness a query-path defect": a gate scoring
/// only the state would grade a badly quantised `q_proj` as perfect.
/// The first version of CONTROL 1 coarsened `q` for exactly this reason
/// and read 0.000000 state movement against a 2.6 output movement.
#[test]
fn a_query_path_defect_is_invisible_to_the_state_witness() {
    let m = backend();
    let t = trajectory(&m, Arm::Q4, Arm::BadQ4Q, MEDIUM, false);
    report("diagnostic: coarse q_proj against honest Q4", &t, SHORT);
    let (state, cos, _, out) = worst(&t);
    assert_eq!(
        state, 0.0,
        "coarsening q_proj moved the recurrent state by {state} — q now writes \
         the state, and both CONTROL 1 and this property need rewriting"
    );
    // Not `== 1.0`: the dot and the two norms are summed in f32, so
    // identical vectors land a rounding step away from unity.
    assert!(
        cos >= 1.0 - 1e-6,
        "state cosine {cos} despite a zero delta — the two are not the same state"
    );
    assert!(
        out > 1e-3,
        "coarsening q_proj moved the output only {out} — the output metric is \
         not carrying the query path either, so the gate is blind to q entirely"
    );
}
