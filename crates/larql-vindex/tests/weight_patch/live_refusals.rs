//! Every way a session with pending stored-weight ops is refused execution. In each, the session is handed a
//! reason and never "no overrides", which would execute the stored weights without saying so.

use std::sync::Arc;

use larql_vindex::format::vindex3::fixtures::encode_fixture_container;
use larql_vindex::format::vindex3::fixtures_kimi::kimi_per_expert_moe_f32_model;
use larql_vindex::format::vindex3::knowledge::live_weights::read_path::WHY_EXPERT_BANK;
use larql_vindex::format::vindex3::knowledge::{KnowledgeOverlay, LiveWeightRefusal};
use larql_vindex::format::vindex3::opplan::exec::lowering::LoweringIdentity;
use larql_vindex::format::vindex3::opplan::exec::operands::{
    OperandOverrides, OperandSource, RepresentationSource,
};
use larql_vindex::format::vindex3::opplan::planned::Operation;
use larql_vindex::format::vindex3::opplan::LayerFfn;
use larql_vindex::format::vindex3::weight_patch::{record_op, WeightEdit, WeightOp};
use larql_vindex::FeatureMeta;

use crate::fixture::*;
use crate::live_support::*;

/// The encoding of the fixture's compiled pack.
const PACK_ENCODING: &str = "Q8_0";
/// A norm weight: executed, but as glue, not as a matrix operand of the plan.
const NORM_TENSOR: &str = "1.input_layernorm.weight";
/// The feature slot the knowledge edits touch.
const SLOT: (usize, usize) = (0, 0);
/// Stored bytes per BF16 element.
const BF16_WIDTH: usize = 2;

fn delta(src: &Source) -> WeightOp {
    op(&src.container, Kind::Delta, &COLUMNS_A)
}

/// Stage `op`, refresh against `binding`, and return the refusal — asserting on the way that the session is
/// given an error and not `None`.
fn refused(overlay: &mut KnowledgeOverlay, bound: &Bound) -> LiveWeightRefusal {
    let refusal = overlay.refresh_live_weights(&bound.binding()).unwrap_err();
    assert_eq!(overlay.live_weight_overrides().unwrap_err(), refusal);
    refusal
}

#[test]
fn a_source_that_may_manufacture_a_representation_is_refused() {
    for (source, name) in [
        (RepresentationSource::Auto, "auto"),
        (RepresentationSource::Transient, "transient"),
    ] {
        let src = source_with_pack();
        let bound = bind_as(&src.container, None, source);
        let mut overlay = KnowledgeOverlay::new();
        stage(&mut overlay, delta(&src));
        let refusal = refused(&mut overlay, &bound);
        assert_eq!(
            refusal,
            LiveWeightRefusal::SourceNotStored { requested: name }
        );
        assert!(
            refusal.to_string().contains("only under `stored`"),
            "{refusal}"
        );
    }
}

#[test]
fn a_lowering_other_than_the_production_cpu_is_refused() {
    let src = source_with_pack();
    let bound = bind(&src.container);
    let mut overlay = KnowledgeOverlay::new();
    stage(&mut overlay, delta(&src));
    let reference = LoweringIdentity::reference();
    let refusal = overlay
        .refresh_live_weights(&bound.binding_through(reference.clone()))
        .unwrap_err();
    assert_eq!(
        refusal,
        LiveWeightRefusal::LoweringNotQualified {
            lowering: reference.to_string(),
            qualified: LoweringIdentity::cpu_production().to_string(),
        }
    );
    assert!(refusal.to_string().contains(&reference.to_string()));
    assert!(overlay.live_weight_overrides().is_err());
}

#[test]
fn knowledge_edits_and_weight_ops_together_are_refused_whichever_came_first() {
    let src = source_with_pack();
    let bound = bind(&src.container);
    let meta = || FeatureMeta {
        top_token: "t".into(),
        top_token_id: 1,
        c_score: 1.0,
        top_k: vec![],
    };

    // Weight op first, then each kind of knowledge edit.
    let edits: [fn(&mut KnowledgeOverlay, FeatureMeta); 3] = [
        |o, _| o.delete_feature(SLOT.0, SLOT.1),
        |o, m| o.update_feature_meta(SLOT.0, SLOT.1, m),
        |o, m| {
            o.insert_feature(SLOT.0, SLOT.1, vec![0.0; TENSOR_ROWS], m);
            o.set_down_vector(SLOT.0, SLOT.1, vec![0.0; TENSOR_ROWS]);
        },
    ];
    for edit in edits {
        let mut overlay = session_with(&bound, vec![delta(&src)]);
        assert!(overlay.live_weight_overrides().unwrap().is_some());
        edit(&mut overlay, meta());
        assert_eq!(
            overlay.live_weight_overrides().unwrap_err(),
            LiveWeightRefusal::KnowledgePending
        );
        assert!(LiveWeightRefusal::KnowledgePending
            .to_string()
            .contains("COMPILE refuses that mix"));
    }

    // Knowledge first: the op stages, and the refresh reports the mix.
    let mut overlay = KnowledgeOverlay::new();
    overlay.delete_feature(SLOT.0, SLOT.1);
    stage(&mut overlay, delta(&src));
    assert_eq!(
        refused(&mut overlay, &bound),
        LiveWeightRefusal::KnowledgePending
    );
}

#[test]
fn a_tensor_the_plan_does_not_project_through_is_refused() {
    let src = source_with_pack();
    let bound = bind(&src.container);
    let edit = [WeightEdit {
        row: 0,
        col: 0,
        delta: EDIT_DELTA,
    }];
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .stage_weight_op(record_op(&src.container, NORM_TENSOR, &edit).unwrap())
        .unwrap();
    let refusal = refused(&mut overlay, &bound);
    assert_eq!(
        refusal,
        LiveWeightRefusal::NotExecuted {
            tensor: NORM_TENSOR.into()
        }
    );
    assert!(refusal.to_string().contains(NORM_TENSOR));
}

/// A per-expert bank is mapped straight from the store, so a replacement would never be read: refused by
/// name before a byte of it is materialised (the bank here is F32, which compile would refuse later anyway).
#[test]
fn a_tensor_of_a_per_expert_bank_is_refused() {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        kimi_per_expert_moe_f32_model,
        checkpoint.path(),
        container.path(),
        "per-expert",
    );
    let bound = bind(container.path());
    let expert = bound
        .plan
        .planned_operands()
        .into_iter()
        .find(|p| matches!(p.operation, Operation::ExpertProject { .. }))
        .expect("the fixture carries a per-expert bank")
        .operand;
    let edit = [WeightEdit {
        row: 0,
        col: 0,
        delta: EDIT_DELTA,
    }];
    let mut overlay = KnowledgeOverlay::new();
    overlay
        .stage_weight_op(record_op(container.path(), &expert.tensor, &edit).unwrap())
        .unwrap();
    let refusal = refused(&mut overlay, &bound);
    assert!(
        matches!(
            &refusal,
            LiveWeightRefusal::OutsideSeam { tensor, why, .. }
                if tensor == &expert.tensor && *why == WHY_EXPERT_BANK
        ),
        "{refusal:?}"
    );
    assert!(refusal.to_string().contains("bypasses"), "{refusal}");
}

#[test]
fn an_object_bound_to_its_compiled_pack_is_refused() {
    let src = source_with_pack();
    let bound = bind_as(
        &src.container,
        Some(PACK_ENCODING),
        RepresentationSource::Stored,
    );
    let mut overlay = KnowledgeOverlay::new();
    stage(&mut overlay, delta(&src));
    let refusal = refused(&mut overlay, &bound);
    assert_eq!(
        refusal,
        LiveWeightRefusal::BoundToPack {
            tensor: TENSOR.into(),
            object: DECODER.into(),
            encoding: PACK_ENCODING.into(),
        }
    );
    assert!(refusal.to_string().contains(PACK_ENCODING));
}

/// What compile refuses before writing, the session refuses to execute, with compile's reason.
#[test]
fn ops_compile_would_refuse_are_refused_with_its_reason() {
    let src = source_with_pack();
    // A container that already carries a ledger: ops bind to it, and compile refuses to patch it again.
    let patched = compile(&src, &[delta(&src).as_ref()], "patched.vindex3");
    let bound = bind(&patched);
    let mut overlay = KnowledgeOverlay::new();
    stage(&mut overlay, op(&patched, Kind::Delta, &COLUMNS_B));
    let refusal = refused(&mut overlay, &bound);
    match &refusal {
        LiveWeightRefusal::Compile(why) => {
            assert!(
                why.contains("already carries a weight-patch ledger"),
                "{why}"
            )
        }
        other => panic!("{other:?}"),
    }
    assert!(refusal
        .to_string()
        .starts_with("COMPILE would refuse these ops"));

    // An op bound to another container is refused when the patches are declared.
    let foreign = bind(&src.container);
    let mut overlay = KnowledgeOverlay::new();
    stage(&mut overlay, op(&patched, Kind::Delta, &COLUMNS_B));
    assert!(matches!(
        refused(&mut overlay, &foreign),
        LiveWeightRefusal::Compile(why) if why.contains("written against another artifact")
    ));
}

#[test]
fn a_plan_that_executes_the_tensor_under_another_shape_is_refused() {
    let src = source_with_pack();
    let mut bound = bind(&src.container);
    let layer = bound
        .plan
        .layers
        .iter_mut()
        .find_map(|l| match &mut l.ffn {
            Some(LayerFfn::Dense(ffn)) if ffn.down.tensor == TENSOR => Some(ffn),
            _ => None,
        })
        .expect("the fixture's tensor is a dense down projection");
    layer.down.shape.reverse();
    let mut overlay = KnowledgeOverlay::new();
    stage(&mut overlay, delta(&src));
    let refusal = refused(&mut overlay, &bound);
    assert!(
        matches!(&refusal, LiveWeightRefusal::OperandMismatch { tensor, .. } if tensor == TENSOR),
        "{refusal:?}"
    );
    assert!(refusal.to_string().contains("is not the"), "{refusal}");
}

/// The view belongs to one op set. Any change to the set makes it stale, and a stale view refuses.
#[test]
fn a_view_not_rebuilt_after_the_ops_changed_refuses() {
    let src = source_with_pack();
    let bound = bind(&src.container);
    let mut overlay = KnowledgeOverlay::new();
    assert!(
        overlay.live_weight_overrides().unwrap().is_none(),
        "no op, nothing to refuse"
    );

    stage(&mut overlay, delta(&src));
    assert_eq!(
        overlay.live_weight_overrides().unwrap_err(),
        LiveWeightRefusal::Stale
    );
    assert!(LiveWeightRefusal::Stale
        .to_string()
        .contains("was not rebuilt"));
    overlay.refresh_live_weights(&bound.binding()).unwrap();
    assert!(overlay.live_weight_overrides().unwrap().is_some());

    // Applying a patch, staging another op and removing a patch each change the set.
    overlay
        .try_apply_patch(patch(
            "b.vlp",
            vec![op(&src.container, Kind::Delta, &COLUMNS_B)],
        ))
        .unwrap();
    assert_eq!(
        overlay.live_weight_overrides().unwrap_err(),
        LiveWeightRefusal::Stale
    );
    overlay.refresh_live_weights(&bound.binding()).unwrap();
    overlay.remove_patch(0);
    assert_eq!(
        overlay.live_weight_overrides().unwrap_err(),
        LiveWeightRefusal::Stale
    );
    overlay.refresh_live_weights(&bound.binding()).unwrap();
    assert!(overlay.live_weight_overrides().unwrap().is_some());

    // A refused apply changes nothing, the view included.
    let conflicting = patch("again.vlp", vec![delta(&src)]);
    assert!(overlay.try_apply_patch(conflicting).is_err());
    assert!(overlay.live_weight_overrides().unwrap().is_some());
}

/// The witness an execution ends with: a replacement nothing read was not executed.
#[test]
fn a_replacement_no_loader_read_fails_the_execution() {
    let src = source_with_pack();
    let bound = bind(&src.container);
    let overlay = session_with(&bound, vec![delta(&src)]);
    let overrides = overlay.live_weight_overrides().unwrap().unwrap();
    let operand = bound.tensor_operand();

    assert_eq!(
        overrides.unread_patched(),
        vec![(DECODER.to_string(), TENSOR.to_string())]
    );
    let unread = overrides.ensure_patched_read().unwrap_err().to_string();
    assert!(
        unread.contains(TENSOR) && unread.contains("were not executed"),
        "{unread}"
    );
    // An execution that bypasses the seam (the bare store) leaves it unread.
    bound.compute(None);
    assert!(overrides.ensure_patched_read().is_err());

    OperandSource::overlaid(&bound.store, &overrides)
        .load_raw(&operand)
        .unwrap();
    overrides.ensure_patched_read().unwrap();
    assert!(
        overrides.clone().ensure_patched_read().is_err(),
        "a copy has been read by nothing"
    );
}

#[test]
fn replacing_bytes_must_be_the_tensor_the_operand_declares() {
    let src = source_with_pack();
    let bound = bind(&src.container);
    let operand = bound.tensor_operand();
    let whole = TENSOR_ROWS * TENSOR_COLS * BF16_WIDTH;
    let mut overrides = OperandOverrides::new();
    assert!(overrides.is_empty() && !overrides.is_patched(&operand));

    let short = overrides
        .replace_bf16(&operand, Arc::from(vec![0u8; whole - BF16_WIDTH]))
        .unwrap_err()
        .to_string();
    assert!(short.contains("do not hold a BF16 tensor"), "{short}");
    assert!(
        overrides.is_empty(),
        "a refused replacement installs nothing"
    );

    let before = overrides.version();
    overrides
        .replace_bf16(&operand, Arc::from(vec![0u8; whole]))
        .unwrap();
    assert!(overrides.is_patched(&operand) && !overrides.is_empty());
    assert_ne!(overrides.version(), before, "a replacement is a mutation");
    let shown = format!("{overrides:?}");
    assert!(
        shown.contains(&format!("bytes: {whole}")) && shown.len() < whole,
        "a replacement is shown by its size, not its bytes: {} characters",
        shown.len()
    );
    let source = OperandSource::overlaid(&bound.store, &overrides);
    assert!(source.stamp().is_overlaid());

    // Asked for under another shape, the replacement refuses on every path that reads it.
    let mut reshaped = operand.clone();
    reshaped.shape.reverse();
    for refusal in [
        source.load(&reshaped).unwrap_err(),
        source.load_raw(&reshaped).map(|_| ()).unwrap_err(),
    ] {
        assert!(refusal.to_string().contains("shape mismatch"), "{refusal}");
    }
    assert!(source.load(&operand).unwrap().iter().all(|v| *v == 0.0));
}
