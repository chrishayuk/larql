//! The hermetic Gated DeltaNet gate: small, committed, no checkpoint.
//!
//! The real-checkpoint parity in `gated_delta_parity` is the gold oracle,
//! but it needs a multi-gigabyte capture and skips without it. This runs
//! everywhere, and it retains exactly what the mutation experiment showed
//! is load-bearing — no more, and no less.
//!
//! Four observables, because the experiment on the real operator proved
//! each is blind to a defect the others catch:
//!
//! | plane  | catches                                              |
//! |--------|------------------------------------------------------|
//! | `q`    | convolution order, causality, head expansion         |
//! | `core` | recurrence read path                                 |
//! | state  | recurrence write path (decay, beta)                  |
//! | output | gated norm and the output projection                 |
//!
//! And both `rel_rms` and `cosine`, because `ScaleBeforeNorm` is a pure
//! magnitude error that leaves cosine at exactly 1.0.

use super::super::continuation::RecurrentState;
use super::super::gated_delta::{layer_forward, state_geometry, GatedDeltaWeights, Mutation};
use super::qw2_tiny_fixture as fx;
use crate::format::vindex3::fixtures::lcg_values;
use crate::format::vindex3::opplan::exec::cpu::WeightRows;
use crate::format::vindex3::opplan::{GatedDeltaOp, OperandRef};

/// Generated through HF's own ops in f32, so only arithmetic order differs.
const MAX_REL_RMS: f32 = 1e-5;
const MIN_COSINE: f32 = 0.999_999;

fn operand() -> OperandRef {
    OperandRef {
        object: "target.decoder_stack".into(),
        tensor: "tiny".into(),
        dtype: "F32".into(),
        shape: vec![1],
    }
}

fn op() -> GatedDeltaOp {
    op_with_heads(fx::KEY_HEADS, fx::VALUE_HEADS)
}

/// The fixture's operator with its head counts replaced — every other
/// dimension stays the pinned fixture's. Exists so a repeat factor other
/// than the fixture's 3 (`gated_delta_repeat_one`) runs through this same
/// harness; nothing built with it is compared against pinned HF values.
pub(super) fn op_with_heads(key_heads: usize, value_heads: usize) -> GatedDeltaOp {
    GatedDeltaOp {
        num_key_heads: key_heads,
        num_value_heads: value_heads,
        key_head_dim: fx::KEY_HEAD_DIM,
        value_head_dim: fx::VALUE_HEAD_DIM,
        conv_kernel: fx::CONV_KERNEL,
        state_dtype: Some(larql_models::inventory::report::RecurrentStateDtype::Float32),
        in_proj_qkv: operand(),
        in_proj_a: operand(),
        in_proj_b: operand(),
        in_proj_z: operand(),
        conv1d: operand(),
        a_log: operand(),
        dt_bias: operand(),
        norm: operand(),
        out_proj: operand(),
    }
}

pub(super) struct Store {
    qkv: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    z: Vec<f32>,
    conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    out: Vec<f32>,
}

fn store() -> Store {
    store_with_heads(fx::KEY_HEADS, fx::VALUE_HEADS)
}

/// Weights for [`op_with_heads`]: same seeds, sized for the given heads.
/// At the fixture's own heads this is exactly [`store`].
pub(super) fn store_with_heads(key_heads: usize, value_heads: usize) -> Store {
    let key = key_heads * fx::KEY_HEAD_DIM;
    let value = value_heads * fx::VALUE_HEAD_DIM;
    let conv_dim = key * 2 + value;
    Store {
        qkv: lcg_values(conv_dim * fx::HIDDEN, fx::SEED_IN_PROJ_QKV),
        a: lcg_values(value_heads * fx::HIDDEN, fx::SEED_IN_PROJ_A),
        b: lcg_values(value_heads * fx::HIDDEN, fx::SEED_IN_PROJ_B),
        z: lcg_values(value * fx::HIDDEN, fx::SEED_IN_PROJ_Z),
        conv: lcg_values(conv_dim * fx::CONV_KERNEL, fx::SEED_CONV1D),
        a_log: lcg_values(value_heads, fx::SEED_A_LOG),
        dt_bias: lcg_values(value_heads, fx::SEED_DT_BIAS),
        norm: lcg_values(fx::VALUE_HEAD_DIM, fx::SEED_NORM),
        out: lcg_values(fx::HIDDEN * value, fx::SEED_OUT_PROJ),
    }
}

pub(super) fn weights(s: &Store) -> GatedDeltaWeights<'_> {
    GatedDeltaWeights {
        in_proj_qkv: WeightRows::F32(&s.qkv),
        in_proj_a: WeightRows::F32(&s.a),
        in_proj_b: WeightRows::F32(&s.b),
        in_proj_z: WeightRows::F32(&s.z),
        conv1d: &s.conv,
        a_log: &s.a_log,
        dt_bias: &s.dt_bias,
        norm: &s.norm,
        out_proj: WeightRows::F32(&s.out),
        norm_eps: fx::NORM_EPS,
    }
}

fn hidden() -> Vec<Vec<f32>> {
    let flat = lcg_values(fx::POSITIONS * fx::HIDDEN, fx::SEED_INPUT);
    (0..fx::POSITIONS)
        .map(|t| flat[t * fx::HIDDEN..(t + 1) * fx::HIDDEN].to_vec())
        .collect()
}

/// A longer input than the pinned fixture's three positions.
///
/// The pinned expectations cover `POSITIONS = 3`, which is SHORTER than
/// the 4-wide convolution window — so no split of it can straddle that
/// window, and a continuation test built on it would assert nothing. The
/// split tests compare one batch against a chained split rather than
/// against pinned values, so they are free to use a longer sequence and
/// must.
pub(super) fn long_hidden(positions: usize) -> Vec<Vec<f32>> {
    let flat = lcg_values(positions * fx::HIDDEN, fx::SEED_INPUT);
    (0..positions)
        .map(|t| flat[t * fx::HIDDEN..(t + 1) * fx::HIDDEN].to_vec())
        .collect()
}

pub(super) fn metrics(mine: &[f32], theirs: &[f32]) -> (f32, f32) {
    assert_eq!(mine.len(), theirs.len());
    let (mut se, mut ss, mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in mine.iter().zip(theirs) {
        se += ((a - b) as f64).powi(2);
        ss += (*b as f64).powi(2);
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    (
        (se / ss.max(f64::MIN_POSITIVE)).sqrt() as f32,
        (dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)) as f32,
    )
}

fn check(label: &str, mine: &[f32], theirs: &[f32]) {
    let (rel, cos) = metrics(mine, theirs);
    assert!(rel < MAX_REL_RMS, "{label}: rel_rms {rel:.3e}");
    assert!(cos > MIN_COSINE, "{label}: cosine {cos:.7}");
}

/// The weights are regenerated, not stored, so a change to `lcg_values`
/// would silently repoint every expectation below at different numbers.
#[test]
fn the_generator_still_produces_the_values_this_fixture_was_built_from() {
    let probe = lcg_values(4, fx::SEED_IN_PROJ_QKV);
    for (i, (got, want)) in probe.iter().zip(fx::LCG_PROBE).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "lcg_values drifted at {i}: {got:e} vs {want:e} — regenerate the fixture"
        );
    }
}

#[test]
fn the_tiny_operator_matches_the_reference_at_every_plane() {
    let op = op();
    let s = store();
    let mut state = RecurrentState::zeros(
        &state_geometry(&op).expect("the fixture declares a state precision"),
    );
    let planes = layer_forward(&op, &weights(&s), &hidden(), &mut state, Mutation::None);

    let flat = |v: &Vec<Vec<f32>>| -> Vec<f32> { v.iter().flatten().copied().collect() };
    check("q", &flat(&planes.query), &fx::EXPECTED_Q);
    check("core", &flat(&planes.core), &fx::EXPECTED_CORE);
    check(
        "final state",
        state.buffer(0).cells(),
        &fx::EXPECTED_FINAL_STATE,
    );
    check("output", &flat(&planes.output), &fx::EXPECTED_OUTPUT);
}

/// Every defect the real-operator experiment characterised, on the fixture
/// that actually runs in CI. The assertion names which plane caught it, so
/// a future change that narrows the fixture cannot quietly drop coverage.
#[test]
fn every_defect_is_still_caught_by_the_compact_fixture() {
    let op = op();
    let s = store();
    let h = hidden();
    let flat = |v: &Vec<Vec<f32>>| -> Vec<f32> { v.iter().flatten().copied().collect() };

    for mutation in [
        Mutation::ScaleBeforeNorm,
        Mutation::ReadBeforeWrite,
        Mutation::NoDecay,
        Mutation::NoBeta,
        Mutation::RawGate,
        Mutation::SiluBeforeConv,
        Mutation::CentredConv,
        Mutation::TiledHeadExpansion,
    ] {
        let mut state = RecurrentState::zeros(
            &state_geometry(&op).expect("the fixture declares a state precision"),
        );
        let planes = layer_forward(&op, &weights(&s), &h, &mut state, mutation);
        let (q_rel, _) = metrics(&flat(&planes.query), &fx::EXPECTED_Q);
        let (core_rel, core_cos) = metrics(&flat(&planes.core), &fx::EXPECTED_CORE);
        let (st_rel, _) = metrics(state.buffer(0).cells(), &fx::EXPECTED_FINAL_STATE);
        let (out_rel, _) = metrics(&flat(&planes.output), &fx::EXPECTED_OUTPUT);

        let caught = q_rel > MAX_REL_RMS
            || core_rel > MAX_REL_RMS
            || st_rel > MAX_REL_RMS
            || out_rel > MAX_REL_RMS
            || core_cos < MIN_COSINE;
        assert!(
            caught,
            "{mutation:?} slipped through every plane — the compact fixture \
             has stopped covering a defect the real operator proved it must \
             (q {q_rel:.3e}, core {core_rel:.3e}, state {st_rel:.3e}, out {out_rel:.3e})"
        );
    }
}

/// **QW-3.6.** Splitting a sequence into two batches, with state chained,
/// reproduces the single-batch result exactly.
///
/// This is the falsifier for the convolution history. Before it existed,
/// `layer_forward` left-padded the window with zeros at every batch
/// boundary, so the second batch's first `kernel-1` positions saw a
/// truncated receptive field. Nothing caught it: the whole-prefix gates
/// pass either way, because when the batch IS the prefix the zeros are
/// correct.
///
/// The control is the same test with the history suppressed — if that
/// does NOT diverge, the buffer is not load-bearing and this test proves
/// nothing.
#[test]
fn a_split_sequence_with_chained_state_matches_one_batch() {
    let s = store();
    let op = op();
    let full = long_hidden(9);
    let cut = 5;
    assert!(
        full.len() - cut >= op.conv_kernel && cut >= op.conv_kernel,
        "both sides of the split must exceed the convolution window, or the boundary \
         is not actually exercised"
    );

    // One batch.
    let mut whole = RecurrentState::zeros(&state_geometry(&op).unwrap());
    let one = layer_forward(&op, &weights(&s), &full, &mut whole, Mutation::None);

    // Two batches, state carried across.
    let mut chained = RecurrentState::zeros(&state_geometry(&op).unwrap());
    layer_forward(
        &op,
        &weights(&s),
        &full[..cut],
        &mut chained,
        Mutation::None,
    );
    let two = layer_forward(
        &op,
        &weights(&s),
        &full[cut..],
        &mut chained,
        Mutation::None,
    );

    // Relative, not absolute. This fixture's activations sit around
    // 1e-2 and its delta matrix around 1e-5, so an absolute tolerance of
    // 1e-5 would swallow a total corruption of the signal — the control
    // below measured exactly that and came in UNDER such a bound.
    let tail: Vec<f32> = one.core[cut..].iter().flatten().copied().collect();
    let got: Vec<f32> = two.core.iter().flatten().copied().collect();
    assert_eq!(tail.len(), got.len());
    let (rel, cos) = metrics(&got, &tail);
    assert!(
        rel < 1e-6 && cos > 0.999999,
        "a chained split must reproduce the single batch: rel_rms {rel:e} cos {cos:.7} \
         — the convolution history is not carried across the boundary"
    );

    // ...and the two states agree too, not just the outputs.
    let sw = whole.buffer(0).cells();
    let sc = chained.buffer(0).cells();
    let state_gap = sw
        .iter()
        .zip(sc)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        state_gap < 1e-5,
        "final delta matrices differ: {state_gap:e}"
    );
}

/// **The control for the test above.** With the convolution history
/// zeroed between batches — the pre-QW-3.6 behaviour — the split MUST
/// diverge.
///
/// Without this, `a_split_sequence_with_chained_state_matches_one_batch`
/// would also pass on a model whose convolution kernel happened not to
/// reach across the cut.
#[test]
fn suppressing_the_convolution_history_makes_the_split_diverge() {
    let s = store();
    let op = op();
    let full = long_hidden(9);
    let cut = 5;

    let mut whole = RecurrentState::zeros(&state_geometry(&op).unwrap());
    let one = layer_forward(&op, &weights(&s), &full, &mut whole, Mutation::None);

    let mut chained = RecurrentState::zeros(&state_geometry(&op).unwrap());
    layer_forward(
        &op,
        &weights(&s),
        &full[..cut],
        &mut chained,
        Mutation::None,
    );
    // Exactly the defect QW-3.6 removed: forget the window.
    chained.buffer_mut(1).cells_mut().fill(0.0);
    let two = layer_forward(
        &op,
        &weights(&s),
        &full[cut..],
        &mut chained,
        Mutation::None,
    );

    let tail: Vec<f32> = one.core[cut..].iter().flatten().copied().collect();
    let got: Vec<f32> = two.core.iter().flatten().copied().collect();
    let (rel, cos) = metrics(&got, &tail);
    println!("  history suppressed: rel_rms {rel:e}  cos {cos:.7}");
    assert!(
        rel > 1e-2,
        "forgetting the convolution history must change the answer, or the buffer is \
         not load-bearing and the equivalence test proves nothing: rel_rms {rel:e} \
         cos {cos:.7}"
    );
}
