//! The Gated DeltaNet head-expansion falsifier at repeat factor 1.
//!
//! Qwen3.5-0.8B declares as many value heads as key heads (16/16), so its
//! q/k expansion is the identity. The compact fixture's head-expansion
//! control, `TiledHeadExpansion` (`e % Hk` instead of `e / (Hv/Hk)`), is
//! built for Qwen3.8's 3x ratio and at repeat 1 computes the very same
//! index — it cannot fail there, so it is no falsifier for Qwen3.5.
//!
//! `AssumedTripleExpansion` is the arm that can: it expands as if
//! `Hv = 3·Hk` whatever the op declares, which is what an implementation
//! hard-wired to Qwen3.8 would do.
//!
//! **What this proves:** through the shipped `layer_forward` (the
//! mutation is threaded through production code, as in
//! `gated_delta_tiny`), the repeat-1 path is distinguishable from the
//! Qwen3.8-assumption defect, and the old tiled arm is provably silent.
//!
//! **What it does not prove:** that the shipped repeat-1 output is
//! CORRECT. The weights are `lcg_values`, not an HF capture; correctness
//! on the real model is established by HF parity elsewhere. Only the
//! head counts differ from the pinned `qw2_tiny_fixture` geometry.

use super::super::continuation::RecurrentState;
use super::super::gated_delta::{layer_forward, state_geometry, LayerPlanes, Mutation};
use super::gated_delta_tiny::{long_hidden, metrics, op_with_heads, store_with_heads, weights};

/// Key and value heads equal — Qwen3.5-0.8B's repeat factor of 1, at a
/// size small enough to run everywhere. Four, not two, so the wrong
/// `e / 3` mapping moves three of the four heads rather than one.
const REPEAT_ONE_HEADS: usize = 4;

/// Longer than the 4-wide convolution window, so the recurrence carries
/// state across several positions and the misrouted q/k compound.
const POSITIONS: usize = 8;

/// A misrouted head pairs a value head with an unrelated, independently
/// seeded q/k — an O(1) change — while identical arithmetic on identical
/// inputs is bit-exact. 1e-2 sits orders of magnitude above f32 rounding
/// and well below the expected error, so passing is not marginal.
const MIN_LOUD_REL_RMS: f32 = 1e-2;

fn run(mutation: Mutation) -> (LayerPlanes, Vec<f32>) {
    let op = op_with_heads(REPEAT_ONE_HEADS, REPEAT_ONE_HEADS);
    assert_eq!(
        op.num_value_heads / op.num_key_heads,
        1,
        "this file is about repeat factor 1"
    );
    let s = store_with_heads(REPEAT_ONE_HEADS, REPEAT_ONE_HEADS);
    let mut state = RecurrentState::zeros(
        &state_geometry(&op).expect("the fixture declares a state precision"),
    );
    let planes = layer_forward(
        &op,
        &weights(&s),
        &long_hidden(POSITIONS),
        &mut state,
        mutation,
    );
    let final_state = state.buffer(0).cells().to_vec();
    (planes, final_state)
}

fn flat(v: &[Vec<f32>]) -> Vec<f32> {
    v.iter().flatten().copied().collect()
}

/// **The Qwen3.5 falsifier.** A Qwen3.8-shaped 3x expansion on a
/// repeat-1 op must move the output loudly — at q and at the layer output.
#[test]
fn an_assumed_triple_expansion_is_loudly_wrong_at_repeat_one() {
    let (control, _) = run(Mutation::None);
    let control_out = flat(&control.output);
    // Nonzero, not "above some magnitude": `lcg_values` are +/-0.05, so
    // this fixture's outputs are small in absolute terms, and the
    // comparison below is relative anyway.
    let peak = control_out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("  control output peak |v| {peak:.3e}");
    assert!(
        control_out.iter().all(|v| v.is_finite()) && peak > 0.0,
        "the shipped repeat-1 path must run and produce a non-zero, finite output"
    );

    let (wrong, _) = run(Mutation::AssumedTripleExpansion);
    let (q_rel, q_cos) = metrics(&flat(&wrong.query), &flat(&control.query));
    let (out_rel, out_cos) = metrics(&flat(&wrong.output), &control_out);
    println!("  AssumedTripleExpansion  q: rel_rms {q_rel:.3e} cos {q_cos:.7}");
    println!("  AssumedTripleExpansion  output: rel_rms {out_rel:.3e} cos {out_cos:.7}");
    assert!(
        q_rel > MIN_LOUD_REL_RMS,
        "a 3x expansion of a repeat-1 op must change q: rel_rms {q_rel:e}"
    );
    assert!(
        out_rel > MIN_LOUD_REL_RMS,
        "a 3x expansion of a repeat-1 op must change the layer output: rel_rms {out_rel:e} \
         cos {out_cos:.7}"
    );
}

/// **Why the old arm cannot serve.** At repeat 1, `e % Hk == e / 1`, so
/// `TiledHeadExpansion` is bit-identical to the shipped path on every
/// plane and on the final state. Asserted, so that if this ever stops
/// holding the reasoning above is revisited rather than quietly stale.
#[test]
fn the_tiled_expansion_arm_is_silent_at_repeat_one() {
    let (control, control_state) = run(Mutation::None);
    let (tiled, tiled_state) = run(Mutation::TiledHeadExpansion);
    assert_eq!(flat(&tiled.query), flat(&control.query));
    assert_eq!(flat(&tiled.key), flat(&control.key));
    assert_eq!(flat(&tiled.core), flat(&control.core));
    assert_eq!(flat(&tiled.output), flat(&control.output));
    assert_eq!(tiled_state, control_state);
}
