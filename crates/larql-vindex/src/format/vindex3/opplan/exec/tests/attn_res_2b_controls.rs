//! **K3-ATTNRES-1 2b — what the batch witness can CATCH.**
//!
//! The companion of `attn_res_2b_batch`, which establishes what the
//! batch traversal does; this module establishes that the establishing
//! could have failed. Split out because the two answer different
//! questions and are read for different reasons: one is evidence about
//! the TRAVERSAL, the other is evidence about the WITNESS.
//!
//! Five controls run here — three positional and two at the exit —
//! plus the one-position run that shows why three separated positions
//! were necessary. Every helper they use comes from the batch module, so
//! a control is scored against exactly the harness that produced the
//! green it is testing; a second harness here could have drifted from
//! that one and made every result in both files unreadable.

use super::super::attention_residual::{self, History};
use super::super::hyper_connection::Mutation;
use super::super::observe::HcSite;
use super::super::reference::ReferenceBackend;
use super::attn_res_2b_batch::{assert_a7, batch, decode, Witness, TOKENS};
use super::attn_res_substrate::{prepare, substrate, LAYERS, POSITIONS};

/// **The exit's own controls are caught on the BATCH path.**
///
/// The batch traversal reduces at the exit in its own code, not through
/// the decode path's, so 2a's exit controls say nothing about it. Both
/// are scored here, and both against an EXACT prediction rather than
/// "the number moved":
///
/// - skipping the exit must leave the last position's final PREFIX under
///   the final norm — the reduction not merely perturbed but absent.
/// - reducing with a layer's pair instead of the shipped one must give
///   what that layer's pair predicts, which is what makes "the SHIPPED
///   pair" a claim about which operand was read rather than a
///   description of the code.
#[test]
fn the_batch_exit_controls_are_caught_against_an_exact_prediction() {
    let sub = substrate();
    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let reference = batch(&sub, &TOKENS, Mutation::None);
    let last = POSITIONS - 1;

    let normed = |v: &[f32]| match ops.final_norm() {
        Some(norm) => norm.apply(&backend, v),
        None => v.to_vec(),
    };
    let final_prefix = reference
        .witness
        .site(LAYERS - 1, HcSite::Ffn, last)
        .expect("the last layer's mlp site")
        .prefix_after
        .clone();
    let mut history = History::new(final_prefix.clone());
    for boundary in reference.witness.boundaries() {
        if boundary.position == last {
            history.push_snapshot(boundary.value.clone());
        }
    }

    // (a) the exit, skipped: the prefix alone reaches the norm.
    let skipped = batch(&sub, &TOKENS, Mutation::AttnResExitSkipped);
    assert_ne!(
        skipped.exit, reference.exit,
        "skipping the exit reduction changed nothing — the exit is not load-bearing"
    );
    assert_eq!(
        skipped.exit,
        normed(&final_prefix),
        "a skipped exit must leave the last position's final prefix, and nothing else"
    );

    // (b) the exit, reduced with a LAYER's pair.
    let swapped = batch(&sub, &TOKENS, Mutation::AttnResExitUsesALayerPair);
    assert_ne!(
        swapped.exit, reference.exit,
        "the exit pair is not load-bearing — a layer's pair gave the same answer"
    );
    let layer_pair = ops.layers()[0]
        .attention_residual
        .as_ref()
        .expect("layer 0 ships its site pairs")
        .ffn
        .pair();
    let predicted = attention_residual::reduce(
        &history,
        layer_pair,
        ops.attention_residual_exit()
            .expect("the exit pair")
            .norm_eps(),
        Mutation::None,
    )
    .expect("the shared reduction runs");
    assert_eq!(
        swapped.exit,
        normed(&predicted.mixed),
        "the mutated exit is not what layer 0's pair predicts"
    );
}

// ── The three positional controls ───────────────────────────────────

/// The three controls, and the signature each leaves behind.
const POSITIONAL_CONTROLS: [Mutation; 3] = [
    Mutation::AttnResSwapPositionHistories,
    Mutation::AttnResHistoryFromPositionZero,
    Mutation::AttnResWriteOffsetByOne,
];

/// **Every positional control breaks A7.**
///
/// The freeze is explicit that if the positional mutant passes A7 the
/// batch witness is void. All three are required to fail it, and the
/// failure is required to name a position — a control that broke the run
/// everywhere equally would not be evidence about positional identity.
#[test]
fn every_positional_control_breaks_the_parity() {
    let sub = substrate();
    let stepped = decode(&sub, &TOKENS, Mutation::None);
    for mutation in POSITIONAL_CONTROLS {
        let mutated = batch(&sub, &TOKENS, mutation);
        let verdict = assert_a7(&mutated.witness, &stepped.witness);
        assert!(
            verdict.is_err(),
            "{mutation:?} passed A7 — the batch witness cannot see positional identity \
             and this transition is not proven"
        );
        assert!(
            verdict.unwrap_err().starts_with("position "),
            "{mutation:?} must be caught at a NAMED position"
        );
    }
}

/// **The three controls are not relabellings of each other.**
///
/// Each leaves a distinct, checkable signature:
///
/// ```text
/// COLLAPSE      every position's mixed vector equal at a reducing site
///               -> the broadcast alone. The state was merged.
///
/// DROPPED WRITE position 0's prefix does not move across a site
///               -> the offset alone. A branch output went nowhere.
///
/// neither       -> the swap. The histories stayed distinct and stayed
///               per position; they were paired with the wrong rows.
/// ```
///
/// The claim this test makes is precisely that the three are
/// DISTINGUISHABLE — the pair of signatures separates all three, and no
/// control can be dropped as a duplicate of another. It is not the
/// stronger claim that each catches a defect the others cannot; proving
/// that would need two alternative traversals (one sharing state, one
/// mis-indexing it) which this rung does not build, and the module says
/// so rather than implying it.
#[test]
fn the_three_controls_leave_three_different_signatures() {
    let sub = substrate();
    let reference = batch(&sub, &TOKENS, Mutation::None);
    assert_eq!(
        (
            collapsed(&reference.witness),
            dropped_write(&reference.witness)
        ),
        (false, false),
        "the unmutated run must show neither signature, or the signatures are vacuous"
    );

    let mut seen = Vec::new();
    for mutation in POSITIONAL_CONTROLS {
        let run = batch(&sub, &TOKENS, mutation);
        let signature = (collapsed(&run.witness), dropped_write(&run.witness));
        seen.push((mutation, signature));
    }

    assert_eq!(
        seen,
        vec![
            (Mutation::AttnResSwapPositionHistories, (false, false)),
            (Mutation::AttnResHistoryFromPositionZero, (true, false)),
            (Mutation::AttnResWriteOffsetByOne, (false, true)),
        ],
        "the three controls must leave three different signatures"
    );
}

/// COLLAPSE: some reducing site produced the same mixed vector at every
/// position — the state was shared rather than per position.
fn collapsed(witness: &Witness) -> bool {
    witness.sites().into_iter().any(|site| {
        site.position == 0
            && (1..POSITIONS).all(|other| {
                witness
                    .site(site.layer, site.site, other)
                    .is_some_and(|peer| peer.mixed == site.mixed)
            })
    })
}

/// DROPPED WRITE: position 0's prefix is unchanged across some site, so
/// the branch output computed for it was written nowhere.
fn dropped_write(witness: &Witness) -> bool {
    witness
        .sites()
        .into_iter()
        .any(|site| site.position == 0 && site.prefix_after == site.prefix_before)
}

// ── The uninformed control: why three positions ─────────────────────

/// **A one-position fixture would report this transition green while
/// proving nothing about it.**
///
/// At batch size one the swap and the offset have no second row to act
/// on, and broadcasting position 0's state onto position 0 is the
/// identity — so all three controls produce a run identical to the
/// unmutated one, and A7 passes under every one of them.
///
/// This is why the witness runs three separated positions, and it is
/// evidence about the WITNESS rather than about the traversal: a green
/// here is what a void experiment looks like, recorded deliberately so
/// that the green in `every_positional_control_breaks_the_parity` can be
/// read as a result.
#[test]
fn a_single_position_batch_hides_every_positional_defect() {
    let sub = substrate();
    let one = [TOKENS[0]];
    let reference = batch(&sub, &one, Mutation::None);
    let stepped = decode(&sub, &one, Mutation::None);
    assert_a7(&reference.witness, &stepped.witness).expect("A7 at one position");

    for mutation in POSITIONAL_CONTROLS {
        let mutated = batch(&sub, &one, mutation);
        assert_eq!(
            mutated.witness.events, reference.witness.events,
            "{mutation:?} moved something at batch size one; the control's own \
             precondition has changed and the reasoning in this test is stale"
        );
        assert!(
            assert_a7(&mutated.witness, &stepped.witness).is_ok(),
            "{mutation:?} was caught at batch size one, which it must not be"
        );
    }
}
