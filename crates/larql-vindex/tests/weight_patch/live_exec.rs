//! Executing pending stored-weight ops: the session computes with the patched bytes while an op is in effect,
//! with the stored bytes again once it is removed, and exactly what a fresh binding of the compiled container
//! computes in between.
//!
//! Execution is by token id through `OperandSource::overlaid`, the seam LQL's `INFER` runs over, on the
//! production CPU backend. Every comparison is of bits: all logits and the final hidden state of the last
//! position.

use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::format::vindex3::opplan::exec::backend::WeightFormat;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandSource;
use larql_vindex::format::vindex3::opplan::exec::prepared::{select_realizations, ExecutionSlice};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::weights::{load_weight, LoadedWeight};

use crate::fixture::*;
use crate::live_support::*;

/// Names of the two patches the switching tests apply.
const PATCH_A: &str = "a.vlp";
const PATCH_B: &str = "b.vlp";
/// How many times the stability test executes one patched session.
const REPEATS: usize = 3;

#[test]
fn an_applied_op_changes_what_executes_and_removing_it_restores_the_base_exactly() {
    for kind in KINDS {
        let src = source_with_pack();
        let bound = bind(&src.container);
        let base = bound.compute(None);

        let mut overlay = KnowledgeOverlay::new();
        assert_eq!(
            bound.compute_session(&overlay),
            base,
            "{kind:?}: nothing applied"
        );
        overlay
            .try_apply_patch(patch(PATCH_A, vec![op(&src.container, kind, &COLUMNS_A)]))
            .unwrap();
        overlay.refresh_live_weights(&bound.binding()).unwrap();
        let patched = bound.compute_session(&overlay);
        assert_ne!(
            patched.logits, base.logits,
            "{kind:?}: the op changes the logits"
        );

        overlay.remove_patch(0);
        assert!(
            overlay
                .refresh_live_weights(&bound.binding())
                .unwrap()
                .is_none(),
            "{kind:?}: no op is in effect"
        );
        assert!(overlay.live_weight_overrides().unwrap().is_none());
        assert_eq!(
            bound.compute_session(&overlay),
            base,
            "{kind:?}: after removal the output is the base's, bit for bit"
        );
    }
}

#[test]
fn switching_patches_leaves_nothing_of_the_first_behind() {
    for kind in KINDS {
        let src = source_with_pack();
        let bound = bind(&src.container);
        let only_b = bound.compute_session(&session_with(
            &bound,
            vec![op(&src.container, kind, &COLUMNS_B)],
        ));

        let mut overlay = KnowledgeOverlay::new();
        overlay
            .try_apply_patch(patch(PATCH_A, vec![op(&src.container, kind, &COLUMNS_A)]))
            .unwrap();
        overlay.refresh_live_weights(&bound.binding()).unwrap();
        let a = bound.compute_session(&overlay);
        overlay.remove_patch(0);
        overlay
            .try_apply_patch(patch(PATCH_B, vec![op(&src.container, kind, &COLUMNS_B)]))
            .unwrap();
        overlay.refresh_live_weights(&bound.binding()).unwrap();
        let b = bound.compute_session(&overlay);

        assert_ne!(a.logits, b.logits, "{kind:?}: A and B are different edits");
        assert_eq!(b, only_b, "{kind:?}: A then B is B alone");
    }
}

#[test]
fn a_patched_session_computes_the_same_thing_every_time() {
    for kind in KINDS {
        let src = source_with_pack();
        let bound = bind(&src.container);
        let overlay = session_with(&bound, vec![op(&src.container, kind, &COLUMNS_A)]);
        let first = bound.compute_session(&overlay);
        for _ in 1..REPEATS {
            assert_eq!(bound.compute_session(&overlay), first, "{kind:?}");
        }
    }
}

#[test]
fn two_sessions_on_one_container_do_not_see_each_others_ops() {
    for kind in KINDS {
        let src = source_with_pack();
        let (one, two) = (bind(&src.container), bind(&src.container));
        let base = one.compute(None);
        let patched = session_with(&one, vec![op(&src.container, kind, &COLUMNS_A)]);
        let clean = KnowledgeOverlay::new();

        assert_ne!(
            one.compute_session(&patched).logits,
            base.logits,
            "{kind:?}"
        );
        assert_eq!(
            two.compute_session(&clean),
            base,
            "{kind:?}: the other session is untouched"
        );
        // And over ONE store: the replacements are the overlay's, not the store's.
        assert_eq!(one.compute_session(&clean), base, "{kind:?}");
        assert_ne!(
            one.compute_session(&patched).logits,
            base.logits,
            "{kind:?}"
        );
    }
}

/// Parity with compile, at the output: every logit and the final hidden state, as bits.
#[test]
fn a_patched_session_computes_what_a_fresh_binding_of_the_compiled_container_computes() {
    for kind in KINDS {
        let src = source_with_pack();
        let bound = bind(&src.container);
        let staged = op(&src.container, kind, &COLUMNS_A);
        let compiled = compile(&src, &[staged.as_ref()], "compiled.vindex3");
        let overlay = session_with(&bound, vec![staged]);

        let live = bound.compute_session(&overlay);
        let fresh = bind(&compiled).compute(None);
        assert_eq!(live, fresh, "{kind:?}");
        assert_ne!(
            live.logits,
            bound.compute(None).logits,
            "{kind:?}: and not the base's"
        );
    }
}

/// Parity with compile, below the output: the patched operand is pinned to the realization the compiled
/// container's is, and each resident form holds the same bytes — the widened f32 image the small fixture
/// executes, and the stored-BF16 image the fused kernel takes on a matrix too large to widen.
#[test]
fn a_patched_operand_is_selected_and_loaded_exactly_as_its_compiled_twin() {
    let backend = ProductionBackend::new();
    for kind in KINDS {
        let src = source_with_pack();
        let bound = bind(&src.container);
        let staged = op(&src.container, kind, &COLUMNS_A);
        let fresh = bind(&compile(&src, &[staged.as_ref()], "compiled.vindex3"));
        let overlay = session_with(&bound, vec![staged]);
        let overrides = overlay.live_weight_overrides().unwrap().unwrap();
        let live = OperandSource::overlaid(&bound.store, &overrides);
        let stored = OperandSource::from(&fresh.store);
        let operand = bound.tensor_operand();
        assert_eq!(operand, fresh.tensor_operand());

        assert!(live.is_patched(&operand) && !live.is_overridden(&operand));
        assert_eq!(live.stored_dtype(&operand), stored.stored_dtype(&operand));
        assert_eq!(live.stored_len(&operand), stored.stored_len(&operand));
        assert_eq!(
            select_realizations(&bound.plan, live, &backend, &ExecutionSlice::Full).unwrap(),
            select_realizations(&fresh.plan, stored, &backend, &ExecutionSlice::Full).unwrap(),
            "{kind:?}: every operand is pinned alike"
        );

        let raw = (
            live.load_raw(&operand).unwrap(),
            stored.load_raw(&operand).unwrap(),
        );
        assert_eq!(
            (raw.0.dtype, raw.0.bytes),
            (raw.1.dtype, raw.1.bytes),
            "{kind:?}"
        );
        let bf16 = |source| match load_weight(source, &operand, WeightFormat::Bf16).unwrap() {
            LoadedWeight::Bf16(bytes) => bytes.as_slice()[..bytes.logical_len()].to_vec(),
            other => panic!("asked for bf16, got {:?}", other.format()),
        };
        assert_eq!(
            bf16(live),
            bf16(stored),
            "{kind:?}: the fused kernel's bytes"
        );
        let widened = |source| match load_weight(source, &operand, WeightFormat::F32).unwrap() {
            LoadedWeight::F32(values) => values.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(),
            other => panic!("asked for f32, got {:?}", other.format()),
        };
        assert_eq!(
            widened(live),
            widened(stored),
            "{kind:?}: the widened image"
        );
        let direct: Vec<u32> = live
            .load(&operand)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(direct, widened(stored), "{kind:?}: and the plain f32 load");
    }
}

/// A replaced operand is recorded the way an overlay-edited one is: resident, with no stored dtype or digest.
#[test]
fn the_residency_record_gives_a_patched_operand_no_stored_bytes() {
    let src = source_with_pack();
    let bound = bind(&src.container);
    let overlay = session_with(&bound, vec![op(&src.container, Kind::Delta, &COLUMNS_A)]);
    bound.store.record_residency();
    bound.compute_session(&overlay);
    let record = bound.store.take_residency_record();
    let (patched, others): (Vec<_>, Vec<_>) = record
        .iter()
        .partition(|r| r.object == DECODER && r.tensor == TENSOR);
    assert_eq!(patched.len(), 1);
    assert_eq!(patched[0].stored_dtype, None);
    assert_eq!(patched[0].stored_sha256, None);
    assert!(patched[0].resident_sha256.is_some() && !patched[0].manufactured);
    assert!(!others.is_empty());
    assert!(others
        .iter()
        .all(|r| r.stored_dtype.is_some() && r.stored_sha256.is_some()));
}
