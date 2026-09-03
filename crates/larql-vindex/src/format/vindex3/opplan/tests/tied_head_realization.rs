//! **Where a logical role resolves to a physical object.**
//!
//! An output head is a logical execution requirement. Whether the
//! checkpoint serialises a distinct tensor for it is a separate,
//! independent fact — Qwen3-0.6B, 1.7B and 4B all declare
//! `tie_word_embeddings: true`, and only 4B omits `lm_head.weight`.
//!
//! This pins the answer to "who resolves that, and when": the plan
//! builder does, at plan time. A tied component's [`OutputOp`] carries an
//! [`OperandRef`] naming the **embedding object**, so by the time anything
//! holds a [`ComponentOpPlan`], every operand it names is already a
//! physically stored `(object, tensor)` pair.
//!
//! That matters beyond tidiness. Anything downstream that needs the set
//! of objects an execution requires — a hydration set for a remote
//! container, most immediately — can read it straight off the plan. A
//! second resolver that re-derived aliasing from the graph would be a
//! second authority for one decision, which is the bug class the remote
//! source was built to avoid.

use std::collections::BTreeSet;

use crate::format::vindex3::fixtures::{dense_f32_model_with, HeadStorage};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};

const COMPONENT: &str = "target";
const EMBEDDING: &str = "target.embedding";
const OUTPUT_HEAD: &str = "target.output_head";
const DECODER_STACK: &str = "target.decoder_stack";
const FINAL_NORM: &str = "target.final_norm";

/// Encode the dense fixture with the requested head storage and plan it.
fn plan_for(head: HeadStorage) -> (tempfile::TempDir, ComponentOpPlan, BTreeSet<String>) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    dense_f32_model_with(checkpoint.path(), head);
    let inventory = larql_models::inventory::build_inventory(checkpoint.path()).unwrap();
    crate::format::vindex3::encode::encode_system(
        &[(COMPONENT.to_string(), inventory)],
        container.path(),
    )
    .unwrap();

    let inspection = inspect_container(container.path(), false).unwrap();
    let objects: BTreeSet<String> = inspection
        .graph
        .objects
        .iter()
        .map(|o| o.id.clone())
        .collect();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    (container, outcome.plan.unwrap(), objects)
}

/// Every operand the plan names, in plan order.
fn operands(plan: &ComponentOpPlan) -> Vec<OperandRef> {
    let mut out = Vec::new();
    if let Some(embedding) = &plan.embedding {
        out.push(embedding.table.clone());
    }
    // The layer stack's operands are numerous and all live in one object;
    // this test is about the head, so the stack is represented by
    // whichever object its first operand names.
    if let Some(layer) = plan.layers.first() {
        if let Some(n) = &layer.pre_attention_norm {
            out.push(n.weight.clone());
        }
    }
    if let Some(norm) = &plan.final_norm {
        out.push(norm.weight.clone());
    }
    if let Some(output) = &plan.output {
        out.push(output.projection.clone());
    }
    out
}

/// The distinct objects an execution of this plan must resolve.
fn required_objects(plan: &ComponentOpPlan) -> BTreeSet<String> {
    operands(plan).into_iter().map(|o| o.object).collect()
}

#[test]
fn a_separate_head_gets_its_own_object() {
    // The control. Without this the tied assertion below could pass for a
    // build that never places an output-head object at all.
    let (_dir, plan, objects) = plan_for(HeadStorage::Separate);
    assert!(
        objects.contains(OUTPUT_HEAD),
        "a checkpoint shipping lm_head.weight should place an output-head \
         object; got {objects:?}"
    );
    assert_eq!(
        plan.output.as_ref().unwrap().projection.object,
        OUTPUT_HEAD,
        "the head op should name its own object when one exists"
    );
    assert_eq!(
        required_objects(&plan),
        [EMBEDDING, DECODER_STACK, FINAL_NORM, OUTPUT_HEAD]
            .into_iter()
            .map(str::to_string)
            .collect::<BTreeSet<_>>(),
    );
}

#[test]
fn a_tied_head_resolves_to_the_embedding_object_at_plan_time() {
    let (_dir, plan, objects) = plan_for(HeadStorage::Tied);
    assert!(
        !objects.contains(OUTPUT_HEAD),
        "a tied checkpoint serialises no head tensor, so the container \
         should place no head object; got {objects:?}"
    );

    // The logical role still exists — this is not a model that cannot
    // generate.
    let output = plan
        .output
        .as_ref()
        .expect("a tied component still has an output op");

    // And it resolves, here, to a physically stored operand.
    assert_eq!(
        output.projection.object, EMBEDDING,
        "the tied head must resolve to the embedding object at plan time"
    );
    assert_eq!(
        output.projection,
        plan.embedding.as_ref().unwrap().table,
        "tied means the SAME operand, not merely the same object"
    );
}

#[test]
fn the_required_object_set_is_the_hydration_set() {
    // The property everything downstream depends on: the objects an
    // execution needs are readable off the plan, and for a tied model
    // that set is strictly smaller than the separate-head case — three
    // objects, not four, with no head among them.
    let (_dir, tied, _) = plan_for(HeadStorage::Tied);
    let (_dir2, separate, _) = plan_for(HeadStorage::Separate);

    let tied_objects = required_objects(&tied);
    assert_eq!(
        tied_objects,
        [EMBEDDING, DECODER_STACK, FINAL_NORM]
            .into_iter()
            .map(str::to_string)
            .collect::<BTreeSet<_>>(),
    );
    assert!(
        required_objects(&separate).is_superset(&tied_objects),
        "the separate-head requirement should be the tied one plus a head"
    );
    assert_eq!(
        required_objects(&separate).len() - tied_objects.len(),
        1,
        "exactly one object should separate the two realisations"
    );
}
