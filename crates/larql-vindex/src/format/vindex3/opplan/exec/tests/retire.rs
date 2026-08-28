//! STATE-2 seam gates: retiring part of the consumed past.
//!
//! Controls before parity, in both directions. The instrument must be
//! INERT when off — a wrapper that has retired nothing decodes
//! bit-identically to the plain state — and must FIRE when on — retiring
//! a span the full-attention layer still reads changes the logits. Only
//! then do the semantic pins mean anything: retirement composes with a
//! sliding window as set intersection (retiring what the window already
//! excludes is bit-for-bit free), the decode and prefill traversals
//! answer the same retired continuation, and the backends that cannot
//! honour a declaration refuse it by name instead of attending over the
//! full prefix.
//!
//! The miniature fixture is layer 0 Sliding(3) + layer 1 Full over five
//! positions, so at the last position the window's own start is 2 —
//! which is exactly what lets one span (`0..2`) be free on the sliding
//! layer while only the full layer carries the change.

use super::super::decode::DecodeSession;
use super::super::device::DevicePlanBackend;
use super::super::kv::{validate_retired, KvState};
use super::super::production::ProductionBackend;
use super::super::reference::ReferenceBackend;
use super::super::retire::RetiringKvState;
use super::super::{backend::WeightFormat, prefill_plan};
use super::decode::{decode_logits, fixture};
use super::golden::G_TOKENS;

/// Decode every fixture token against a caller-owned provider, retiring
/// `span` (if any) before the LAST step, and return that step's logits.
fn decode_with_retirement(span: Option<std::ops::Range<usize>>) -> Vec<f32> {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let mut kv = RetiringKvState::default();
    let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
    let (last, earlier) = G_TOKENS.split_last().unwrap();
    for &token in earlier {
        session.step(token).unwrap();
    }
    drop(session);
    if let Some(span) = span {
        kv.retire(span).unwrap();
    }
    let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
    let logits = session.step(*last).unwrap().logits;
    logits.expect("plan carries an output head")
}

/// OFF-control: a wrapper with nothing retired is the plain state,
/// bit for bit — the indirection itself must not move a single ulp.
#[test]
fn an_unretiring_wrapper_is_bit_identical_to_the_plain_state() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let plain = decode_logits(&plan, &store, &backend);
    let wrapped = decode_with_retirement(None);
    assert_eq!(plain, wrapped, "an empty retirement set changed the bits");
}

/// ON-control: retiring a span the full-attention layer still reads
/// must change the logits. If it cannot fail on this known-different
/// input, none of the parity claims below are evidence.
#[test]
fn retiring_live_positions_changes_the_logits() {
    let unretired = decode_with_retirement(None);
    let retired = decode_with_retirement(Some(1..3));
    assert_ne!(
        unretired, retired,
        "retiring positions 1..3 left the logits untouched — the exclusion never reached \
         the attention arithmetic"
    );
}

/// Retirement composes with the sliding window as set intersection —
/// but the miniature also has a FULL layer that still reads positions
/// 0..2 at the last step, so this composition is only observable
/// layer-by-layer, not end to end. Pin it end to end on a plan whose
/// every layer slides: retiring exactly what the window has already
/// left behind is bit-for-bit free.
#[test]
fn retiring_behind_every_sliding_window_is_free() {
    use crate::format::vindex3::graph::policy::AttentionSpan;
    use crate::format::vindex3::opplan::LayerAttention;

    let (_c, mut plan, store) = fixture();
    for layer in &mut plan.layers {
        if let LayerAttention::Softmax(op) = &mut layer.attention {
            op.span = AttentionSpan::Sliding;
            op.window = Some(3);
        }
    }
    let backend = ReferenceBackend::new();

    let run = |span: Option<std::ops::Range<usize>>| {
        let mut kv = RetiringKvState::default();
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
        let (last, earlier) = G_TOKENS.split_last().unwrap();
        for &token in earlier {
            session.step(token).unwrap();
        }
        drop(session);
        if let Some(span) = span {
            kv.retire(span).unwrap();
        }
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
        session.step(*last).unwrap().logits.unwrap()
    };

    // At the final position (4) every window starts at 2: positions
    // 0..2 are already outside every layer's span.
    assert_eq!(
        run(None),
        run(Some(0..2)),
        "retiring positions the windows never read changed the bits"
    );
    // The same spans one position later ARE read; the instrument still
    // fires on this plan (guards against a plan edit that silently
    // stopped exercising attention at all).
    assert_ne!(run(None), run(Some(2..4)), "in-window retirement was free");
}

/// The prefill traversal answers the same retired continuation as the
/// decode step: consuming the last token by chunked prefill over a
/// retired state is bit-identical to consuming it by a decode step.
#[test]
fn prefill_over_a_retired_state_matches_the_decode_step() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let (last, earlier) = G_TOKENS.split_last().unwrap();

    let mut prefill_arm = RetiringKvState::default();
    prefill_plan(&plan, &store, earlier, &backend, &mut prefill_arm).unwrap();
    prefill_arm.retire(1..3).unwrap();
    let continued = prefill_plan(&plan, &store, &[*last], &backend, &mut prefill_arm)
        .unwrap()
        .logits
        .expect("plan carries an output head");

    let stepped = decode_with_retirement(Some(1..3));
    assert_eq!(
        continued, stepped,
        "the two traversals answered different retired continuations"
    );
}

/// Backends that cannot honour a retirement declaration refuse the step
/// by name. Attending over the full prefix instead would silently
/// answer a different continuation than the provider declared.
#[test]
fn production_and_device_backends_refuse_a_retiring_session() {
    let (_c, plan, store) = fixture();
    let (last, earlier) = G_TOKENS.split_last().unwrap();

    fn retired_state(
        plan: &crate::format::vindex3::opplan::ComponentOpPlan,
        store: &super::super::operands::OperandStore,
        earlier: &[u32],
    ) -> RetiringKvState {
        let backend = ReferenceBackend::new();
        let mut kv = RetiringKvState::default();
        let mut session = DecodeSession::with_kv_state(plan, store, &backend, &mut kv).unwrap();
        for &token in earlier {
            session.step(token).unwrap();
        }
        drop(session);
        kv.retire(1..3).unwrap();
        kv
    }

    let production = ProductionBackend::new();
    let mut kv = retired_state(&plan, &store, earlier);
    let mut session = DecodeSession::with_kv_state(&plan, &store, &production, &mut kv).unwrap();
    let err = session.step(*last).map(|_| ()).unwrap_err().to_string();
    assert!(
        err.contains("does not honour retired"),
        "production refusal missing or unnamed: {err}"
    );

    let device = DevicePlanBackend::new(
        super::device::LoopDevice,
        "loop-device-retire",
        WeightFormat::F32,
    );
    let mut kv = retired_state(&plan, &store, earlier);
    let mut session = DecodeSession::with_kv_state(&plan, &store, &device, &mut kv).unwrap();
    let err = session.step(*last).map(|_| ()).unwrap_err().to_string();
    assert!(
        err.contains("does not honour retired"),
        "device refusal missing or unnamed: {err}"
    );
}

/// The wrapper is a plain delegating provider around the state it
/// adopts: geometry announcements and recurrent-state questions land on
/// the inner state unchanged, and an adopted prefilled past is
/// retirable exactly as an accumulated one is.
#[test]
fn the_wrapper_delegates_to_the_state_it_adopted() {
    use super::super::kv::{LayerKvGeometry, RowKvState};

    let mut inner = RowKvState::default();
    inner.prepare(&[LayerKvGeometry {
        kv_dim: 3,
        window: None,
    }]);
    inner.append(0, vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
    inner.set_position(1);

    let mut wrapped = RetiringKvState::from_state(inner);
    assert_eq!(wrapped.position(), 1, "adopted position survives");
    assert_eq!(wrapped.keys(0).len(), 1, "adopted rows survive");
    wrapped.retire(0..1).unwrap();
    assert_eq!(wrapped.retired_positions(), 1);

    // A second geometry announcement is a pass-through, not a reset.
    wrapped.prepare(&[LayerKvGeometry {
        kv_dim: 3,
        window: None,
    }]);
    assert_eq!(wrapped.keys(0).len(), 1);

    // A KV-only inner state answers recurrent questions with its own
    // refusal — the wrapper adds nothing and hides nothing.
    let err = wrapped.recurrent_state(0).unwrap_err();
    assert!(
        err.to_string().contains("is not a recurrent layer"),
        "{err}"
    );
}

/// The provider's own bookkeeping refuses what cannot be a valid
/// retirement, and the interpreter's check catches a provider that
/// lies anyway.
#[test]
// One-element range arrays here are literal declarations under test,
// not mistaken `vec![start..end]` initialisers.
#[allow(clippy::single_range_in_vec_init)]
fn incoherent_retirements_are_refused_by_name() {
    let mut kv = RetiringKvState::default();
    kv.set_position(8);

    let err = kv.retire(5..5).unwrap_err();
    assert!(err.contains("empty"), "{err}");

    let err = kv.retire(6..9).unwrap_err();
    assert!(err.contains("beyond position"), "{err}");

    kv.retire(2..5).unwrap();
    let err = kv.retire(4..6).unwrap_err();
    assert!(err.contains("overlaps"), "{err}");
    kv.retire(5..7).unwrap();
    assert_eq!(kv.retired_positions(), 5);

    // The interpreter-side check, for a provider that declares what its
    // own API would have refused.
    let err = validate_retired(Some(&[3..3]), 8).unwrap_err().to_string();
    assert!(err.contains("empty"), "{err}");
    let err = validate_retired(Some(&[3..9]), 8).unwrap_err().to_string();
    assert!(err.contains("not yet behind"), "{err}");
    validate_retired(Some(&[3..8]), 8).unwrap();
    validate_retired(None, 0).unwrap();
}
