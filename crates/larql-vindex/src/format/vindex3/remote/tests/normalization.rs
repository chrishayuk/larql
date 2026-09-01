//! **A logical role with no physical object of its own, on a real model.**
//!
//! Granite and Qwen3-4B are adversarial in opposite directions, and a
//! representation system needs both:
//!
//! ```text
//! granite-4.2-3b   one logical relationship → TWO physical regions
//!                  lm_head and embed_tokens tied in the source model,
//!                  serialised separately, and the shard index counts
//!                  them once
//!
//! Qwen3-4B         two logical uses → ONE physical region
//!                  tie_word_embeddings: true and no lm_head.weight at
//!                  all, so the output head has no object to be
//! ```
//!
//! Granite witnesses selective residency. This witnesses *normalisation*:
//! the alias is collapsed in `opplan/build.rs` at plan time, so the
//! container carries no head object, the plan's output projection names
//! the embedding, and everything downstream — requirement set, hydration,
//! preparation — sees an ordinary stored operand.
//!
//! `#[ignore]` because it needs a real container. Produce one with
//!
//! ```text
//! larql vindex3 encode hf://Qwen/Qwen3-4B --output <dir>
//! ```
//!
//! and point `LARQL_QWEN3_4B_CONTAINER` at it. That encode is possible
//! only since the declared-disabled sliding-window work: before it,
//! Qwen3-4B carried three blocking findings and could not be encoded at
//! all.
//!
//! The synthetic twin is `opplan::tests::tied_head_realization`, which
//! pins the same mechanism deterministically. This proves it survives a
//! real upstream checkpoint.

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::format::vindex3::encode::SEGMENTS_DIR;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::requirements::required_objects;
use crate::format::vindex3::opplan::plan_component_ops;

const CONTAINER_ENV: &str = "LARQL_QWEN3_4B_CONTAINER";
const COMPONENT: &str = "target";
const EMBEDDING: &str = "target.embedding";
const OUTPUT_HEAD: &str = "target.output_head";

fn container() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(CONTAINER_ENV).ok()?);
    path.join("index.json").exists().then_some(path)
}

#[test]
#[ignore = "needs a real Qwen3-4B container; set LARQL_QWEN3_4B_CONTAINER"]
fn a_tied_head_survives_a_real_checkpoint_into_execution() {
    let Some(root) = container() else {
        panic!("set {CONTAINER_ENV} to a real Qwen3-4B container");
    };
    let inspection = inspect_container(&root, false).unwrap();
    let objects: BTreeSet<String> = inspection
        .graph
        .objects
        .iter()
        .map(|o| o.id.clone())
        .collect();

    // 1. The container describes no head object, because the checkpoint
    //    serialises no head tensor.
    assert!(
        !objects.contains(OUTPUT_HEAD),
        "Qwen3-4B ships no lm_head.weight; the container should place no \
         head object. Got {objects:?}"
    );
    assert!(objects.contains(EMBEDDING));
    let head_segment = root.join(SEGMENTS_DIR).join(format!("{OUTPUT_HEAD}.bin"));
    assert!(
        !head_segment.exists(),
        "no head object means no head segment: {}",
        head_segment.display()
    );

    // 2. The logical role still exists, and resolves to the embedding.
    let outcome = plan_component_ops(&inspection, &root, COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let output = plan
        .output
        .as_ref()
        .expect("a tied model still generates tokens");
    assert_eq!(
        output.projection.object, EMBEDDING,
        "the tied head must resolve to the embedding object"
    );
    assert_eq!(
        output.projection,
        plan.embedding.as_ref().unwrap().table,
        "tied means the SAME operand, not merely the same object"
    );

    // 3. The requirement set contains no head, and every object it names
    //    exists — so hydration would never ask for something absent.
    let required = required_objects(&plan, &ExecutionSlice::Full).unwrap();
    assert!(
        !required.iter().any(|o| o.contains("output_head")),
        "a tied model requires no head object; required {required:?}"
    );
    assert!(
        required.is_subset(&objects),
        "requirement set {required:?} names something outside {objects:?}"
    );

    // 4. Preparation resolves the head through the embedding's bytes, and
    //    touches exactly what was predicted.
    let store = OperandStore::open(&root, &inspection).unwrap();
    let backend = ReferenceBackend;
    PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)
        .expect("a tied real model must prepare");
    assert_eq!(
        store.touched_objects(),
        required,
        "preparation touched something outside its requirement set"
    );

    eprintln!(
        "Qwen3-4B: {} objects, no output_head; head projection resolves to `{}`",
        objects.len(),
        output.projection.object,
    );
}
