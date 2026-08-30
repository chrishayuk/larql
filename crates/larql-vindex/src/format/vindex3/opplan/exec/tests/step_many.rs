//! CPU-7C: `step_many` is a continuation, not a second traversal API.
//!
//! The property under test is NOT "the same logits". A multi-position
//! traversal can produce correct logits for the positions it consumed and
//! still leave a wrong recurrent state behind, which then diverges on the
//! NEXT token — so every gate here ends with an ordinary
//! [`DecodeSession::step`] and compares THAT. What is being compared is
//! the state the batch left behind, observed through the only surface
//! that can see it.
//!
//! Run on the `LLLF` hybrid: three GatedDelta recurrences and one softmax
//! layer, so both kinds of continuation state are exercised. A pure
//! softmax stack would pass these while saying nothing about the
//! recurrence, which is the state most at risk.

use super::hybrid_traversal_fixture::hybrid;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

/// Relative RMS below which two logit vectors are the same answer. The
/// value the hybrid decode-vs-batch gate already uses, so this rung is
/// held to the bar its own substrate was accepted at.
const REL_RMS: f32 = 1e-5;

fn rel_rms(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit widths differ");
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = b.iter().map(|y| y * y).sum();
    (num / den.max(f32::MIN_POSITIVE)).sqrt()
}

/// Advance through `batch` in one call, then one ordinary step on
/// `probe`, and return the probe's logits.
fn many_then_step(batch: &[u32], probe: u32) -> Vec<f32> {
    let (_c, plan, store) = hybrid();
    let mut session = DecodeSession::new(&plan, &store, &ReferenceBackend).unwrap();
    session.step_many(batch).unwrap();
    assert_eq!(session.position(), batch.len());
    session
        .step(probe)
        .unwrap()
        .logits
        .expect("the fixture carries an output head")
}

/// The same, one position at a time.
fn stepped_then_step(batch: &[u32], probe: u32) -> Vec<f32> {
    let (_c, plan, store) = hybrid();
    let mut session = DecodeSession::new(&plan, &store, &ReferenceBackend).unwrap();
    for &t in batch {
        session.step(t).unwrap();
    }
    assert_eq!(session.position(), batch.len());
    session
        .step(probe)
        .unwrap()
        .logits
        .expect("the fixture carries an output head")
}

/// **P1a.** The state a multi-position advance leaves behind is the state
/// `K` ordinary steps would have left.
#[test]
fn step_many_leaves_the_continuation_that_stepping_would_have() {
    for batch in [&[1u32, 2][..], &[1, 2, 3, 4][..]] {
        let many = many_then_step(batch, 5);
        let stepped = stepped_then_step(batch, 5);
        let rel = rel_rms(&many, &stepped);
        assert!(
            rel < REL_RMS,
            "K={} : the token AFTER the batch disagrees at rel_rms {rel:e}, so the \
             continuation state the batch left behind is not the one stepping leaves \
             — correct logits for the batch itself would not have caught this",
            batch.len()
        );
    }
}

/// **Causal isolation.** A token supplied later in the batch must not
/// reach a position before it.
///
/// Observable through the last-position surface by construction: if the
/// final token's own result is the same whether it arrived INSIDE the
/// batch or as an ordinary step afterwards, then the earlier positions
/// were computed without it. A non-causal convolution or attention window
/// — the specific hazard in batching a causal model — breaks exactly this
/// and nothing else here would see it.
#[test]
fn a_later_token_does_not_reach_an_earlier_position() {
    for (batch, last) in [(&[1u32, 2][..], 3u32), (&[1, 2, 3][..], 4)] {
        let mut whole: Vec<u32> = batch.to_vec();
        whole.push(last);

        let (_c, plan, store) = hybrid();
        let mut inside = DecodeSession::new(&plan, &store, &ReferenceBackend).unwrap();
        let in_batch = inside
            .step_many(&whole)
            .unwrap()
            .logits
            .expect("the fixture carries an output head");

        let after = many_then_step(batch, last);
        let rel = rel_rms(&in_batch, &after);
        assert!(
            rel < REL_RMS,
            "K={}: the last token's own logits changed depending on whether it was \
             inside the batch (rel_rms {rel:e}) — the earlier positions saw it, so the \
             traversal is not causal",
            whole.len()
        );
    }
}

/// The planted violation. Without it, every comparison above is a claim
/// about a check that has never been shown able to fail.
#[test]
fn changing_a_token_inside_the_batch_moves_the_token_after_it() {
    let base = many_then_step(&[1, 2], 5);
    let altered = many_then_step(&[1, 6], 5);
    let rel = rel_rms(&base, &altered);
    assert!(
        rel > REL_RMS,
        "changing a token inside the batch did not move the next token's logits \
         (rel_rms {rel:e}), so the parity gates above compare nothing"
    );
}

/// An empty advance is a caller bug, not a no-op: a session that silently
/// accepted one would report a position it never reached.
#[test]
fn an_empty_advance_refuses() {
    let (_c, plan, store) = hybrid();
    let mut session = DecodeSession::new(&plan, &store, &ReferenceBackend).unwrap();
    assert!(session.step_many(&[]).is_err());
    assert_eq!(session.position(), 0, "a refused call must move nothing");
}

/// A token id outside the embedding table refuses BEFORE the continuation
/// moves. Found half way through a batch it would leave the session
/// advanced by part of it, which is a corrupted continuation rather than
/// a failed call.
#[test]
fn a_bad_token_late_in_the_batch_moves_nothing() {
    let (_c, plan, store) = hybrid();
    let mut session = DecodeSession::new(&plan, &store, &ReferenceBackend).unwrap();
    assert!(session.step_many(&[1, 2, u32::MAX]).is_err());
    assert_eq!(session.position(), 0, "a refused batch must move nothing");
}

/// **CPU-7C2 arm B vs arm C.** The two FFN shapes must compute the same
/// thing.
///
/// They differ only in WHERE the position loop sits: arm B runs positions
/// through `par_iter_mut` and calls `ffn` once each; arm C hands every
/// position to `ffn_many`, whose default is that same loop. So the
/// arithmetic is identical and the results must be BIT-identical — which
/// is what makes the timing difference between them attributable to
/// machine ownership and nothing else.
///
/// Serialised because the shape is a process-wide switch, and a parallel
/// test running `execute_layer` while this one has flipped it would be
/// measuring whichever shape won the race.
#[test]
#[serial_test::serial]
fn the_two_ffn_shapes_compute_the_same_thing() {
    use crate::format::vindex3::opplan::exec::{multi_position_ffn, set_multi_position_ffn};

    let restore = multi_position_ffn();
    let batch = [1u32, 2, 3];

    set_multi_position_ffn(false);
    let legacy = many_then_step(&batch, 5);
    set_multi_position_ffn(true);
    let raised = many_then_step(&batch, 5);
    set_multi_position_ffn(restore);

    assert_eq!(legacy.len(), raised.len());
    for (i, (a, b)) in legacy.iter().zip(&raised).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit {i}: raising the FFN surface changed the arithmetic, not only \
             the schedule ({a} vs {b})"
        );
    }
}
