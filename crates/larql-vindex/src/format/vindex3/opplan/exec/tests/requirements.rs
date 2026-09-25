//! **The predicted requirement set equals the measured one.**
//!
//! [`required_objects`] folds over a plan to say which objects an
//! execution will need. That is a prediction, and a hydration set built on
//! a wrong prediction fails in the worst way: it succeeds on the models
//! that happen to be covered and errors mid-execution on the one that is
//! not.
//!
//! So it is not argued. A real preparation runs against a real container
//! and [`OperandStore::touched_objects`] records what it actually
//! resolved; the two sets must be equal. Not a superset — equal. A
//! prediction that is merely sufficient would let a hydration set fetch
//! objects nothing needs, which is the other half of the claim.

use std::collections::BTreeSet;

use crate::format::vindex3::fixtures::{
    dense_f32_model_with, encode_fixture_container, miniature_glimmer, HeadStorage,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::requirements::required_objects;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

const COMPONENT: &str = "target";

/// A container, its plan and an opened store.
struct Subject {
    _dir: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
    objects: BTreeSet<String>,
}

fn subject(write: impl FnOnce(&std::path::Path)) -> Subject {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, checkpoint.path(), container.path(), COMPONENT);
    let inspection = inspect_container(container.path(), false).unwrap();
    let objects = inspection
        .graph
        .objects
        .iter()
        .map(|o| o.id.clone())
        .collect();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Subject {
        _dir: container,
        plan: outcome.plan.unwrap(),
        store,
        objects,
    }
}

/// Prepare `slice` and return what the store actually resolved.
fn measure(subject: &Subject, slice: ExecutionSlice) -> BTreeSet<String> {
    let backend = ReferenceBackend;
    PreparedOperands::load(&subject.plan, &subject.store, &backend, slice)
        .expect("preparation should succeed against its own container");
    subject.store.touched_objects()
}

#[test]
fn prediction_equals_measurement_on_a_hybrid_stack() {
    // The miniature Glimmer: two layers, mixed attention policy, gated
    // attention — enough operator variety that a fold missing a field
    // would show up.
    let subject = subject(miniature_glimmer);
    let predicted = required_objects(&subject.plan, &ExecutionSlice::Full).unwrap();
    let measured = measure(&subject, ExecutionSlice::Full);
    assert_eq!(
        predicted, measured,
        "the fold and the preparation disagree about what this execution needs"
    );
    assert!(
        !predicted.is_empty(),
        "a plan requiring nothing is not a plan"
    );
}

#[test]
fn prediction_equals_measurement_with_a_tied_head() {
    // The realisation that has no object of its own. If the fold
    // reconstructed aliasing instead of reading the plan, this is where
    // it would name a head object that does not exist.
    let subject = subject(|dir| dense_f32_model_with(dir, HeadStorage::Tied));
    let predicted = required_objects(&subject.plan, &ExecutionSlice::Full).unwrap();
    assert!(
        !predicted.iter().any(|o| o.contains("output_head")),
        "a tied model requires no head object; predicted {predicted:?}"
    );
    assert_eq!(predicted, measure(&subject, ExecutionSlice::Full));
}

#[test]
fn every_required_object_exists_in_the_container() {
    // A requirement naming an object the container does not carry would
    // be a hydration set that can never be satisfied.
    for (name, subject) in [
        ("glimmer", subject(miniature_glimmer)),
        (
            "tied",
            subject(|dir| dense_f32_model_with(dir, HeadStorage::Tied)),
        ),
        (
            "separate",
            subject(|dir| dense_f32_model_with(dir, HeadStorage::Separate)),
        ),
    ] {
        let predicted = required_objects(&subject.plan, &ExecutionSlice::Full).unwrap();
        assert!(
            predicted.is_subset(&subject.objects),
            "{name}: required {predicted:?} is not contained in the container's \
             objects {:?}",
            subject.objects
        );
    }
}

#[test]
fn a_layer_range_requires_less_than_the_whole_model() {
    // The property a shard depends on, and the one that makes the set
    // worth computing at all: a slice that executes part of the program
    // must not be handed the whole model.
    let subject = subject(miniature_glimmer);
    let whole = required_objects(&subject.plan, &ExecutionSlice::Full).unwrap();
    let range = required_objects(
        &subject.plan,
        &ExecutionSlice::LayerRange { start: 0, end: 1 },
    )
    .unwrap();
    assert!(
        range.is_subset(&whole),
        "a slice cannot require an object the whole model does not"
    );
    assert!(
        whole.len() > range.len(),
        "the whole model requires the embedding, final norm and head that a \
         layer range does not: whole {whole:?} vs range {range:?}"
    );
}

#[test]
fn dense_ffn_placement_reads_only_its_declared_operands() {
    let coordinator = subject(miniature_glimmer);
    let worker = subject(miniature_glimmer);
    let local_slice = ExecutionSlice::DenseFfnCoordinator;
    let remote_slice = ExecutionSlice::DenseFfns {
        start: 0,
        end: worker.plan.layers.len(),
    };
    let local = required_objects(&coordinator.plan, &local_slice).unwrap();
    let remote = required_objects(&worker.plan, &remote_slice).unwrap();
    assert!(!local.is_empty() && !remote.is_empty());
    assert_eq!(local, measure(&coordinator, local_slice));
    assert_eq!(remote, measure(&worker, remote_slice));
    let whole = required_objects(&coordinator.plan, &ExecutionSlice::Full).unwrap();
    assert_eq!(whole, local.union(&remote).cloned().collect());
}

#[test]
fn dense_ffn_placement_never_reads_or_maps_the_other_sides_tensors() {
    use crate::format::vindex3::opplan::LayerFfn;
    let control = subject(miniature_glimmer);
    measure(&control, ExecutionSlice::Full);
    let all = control.store.touched_operand_addresses();
    let expected_ffn: BTreeSet<_> = control
        .plan
        .layers
        .iter()
        .flat_map(|l| {
            let Some(LayerFfn::Dense(op)) = &l.ffn else {
                panic!("dense fixture")
            };
            op.gate
                .iter()
                .chain([&op.up, &op.down])
                .map(|r| (r.object.clone(), r.tensor.clone()))
        })
        .collect();
    let coordinator = subject(miniature_glimmer);
    measure(&coordinator, ExecutionSlice::DenseFfnCoordinator);
    assert_eq!(
        coordinator.store.touched_operand_addresses(),
        all.difference(&expected_ffn).cloned().collect()
    );
    let worker = subject(miniature_glimmer);
    measure(
        &worker,
        ExecutionSlice::DenseFfns {
            start: 0,
            end: worker.plan.layers.len(),
        },
    );
    assert_eq!(worker.store.touched_operand_addresses(), expected_ffn);
    // A one-layer worker must not materialize another layer's FFN tensors.
    let shard = subject(miniature_glimmer);
    let Some(LayerFfn::Dense(op)) = &shard.plan.layers[1].ffn else {
        panic!("dense fixture")
    };
    let owned = op
        .gate
        .iter()
        .chain([&op.up, &op.down])
        .map(|r| (r.object.clone(), r.tensor.clone()))
        .collect();
    measure(&shard, ExecutionSlice::DenseFfns { start: 1, end: 2 });
    assert_eq!(shard.store.touched_operand_addresses(), owned);
}
