//! ANE-4A0: the draft slice, and the gate it has to pass before any
//! reduced depth is believed.
//!
//! [`ExecutionSlice::Draft`] is a reduced-depth model that still owns
//! both ends of the stack — embedding in, the component's own final norm
//! and output head out — so its logits are comparable to the target's.
//! That is what makes it a drafter rather than a shard.
//!
//! **The gate is observational equivalence at full depth.** A new
//! traversal seam that changes the model's logits when it selects every
//! layer is not an acceptance experiment; it is an inference bug, and
//! every reduced-depth number measured through it would inherit the bug
//! rather than the model. So `Draft { end: L }` must reproduce `Full`
//! exactly — bit-identical logits, and the same layers executed in the
//! same order.
//!
//! Bit-identical is the right bar here, not a tolerance: both arms run
//! the same operands through the same backend in the same order, so any
//! difference at all would be the seam's doing.
//!
//! The reduced control is deliberately `L - 1`, not 4 or 8. It asks the
//! narrow structural question — does truncation work cleanly through the
//! hybrid state machinery and the final head — before the interesting
//! one. A drafter that is merely one layer short should still produce
//! well-formed logits that differ from the target's; if it does not
//! differ, the slice was ignored, and if it is malformed, the depth
//! ladder would have been measuring a broken program.
//!
//! Env-gated on the real 51 GB container, and skips LOUDLY rather than
//! reporting success over a missing subject:
//!
//! ```text
//! QW38_CONTAINER=~/chris-models/Qwen3.8-27B.vindex3 \
//!   cargo test draft_slice -- --nocapture --ignored
//! ```

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::fixtures::hybrid_lllf_f32_model;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::execute_slice;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::ExecutionSlice;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// The same LLLF hybrid fixture QW-3.6b/3.7 traverse: three recurrent
/// layers then a softmax one, so the draft slice meets durable
/// continuation state rather than a pure residual stack.
fn hybrid() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let src = tempfile::tempdir().unwrap();
    hybrid_lllf_f32_model(src.path());
    let inventory = larql_models::inventory::build_inventory(src.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("hybrid".to_string(), inventory)], container.path())
        .expect("the hybrid fixture is admissible");
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, outcome.plan.unwrap(), store)
}

/// **The gate, on a fixture.** Cheap enough to run every time, and it
/// asks exactly what the real-container version asks: selecting every
/// layer through the new seam must be the old traversal, bit for bit.
#[test]
fn a_full_depth_draft_is_the_whole_stack_on_the_hybrid_fixture() {
    let (_dir, plan, store) = hybrid();
    let depth = plan.layers.len();
    let tokens = &[1u32, 0, 2];

    let full = execute_slice(
        &plan,
        &store,
        tokens,
        &ReferenceBackend,
        ExecutionSlice::Full,
    )
    .expect("full traversal");
    let draft = execute_slice(
        &plan,
        &store,
        tokens,
        &ReferenceBackend,
        ExecutionSlice::Draft { end: depth },
    )
    .expect("full-depth draft traversal");

    assert_eq!(draft.executed_layers, (0..depth).collect::<Vec<_>>());
    assert_eq!(draft.executed_layers, full.executed_layers);
    assert_eq!(
        draft.logits, full.logits,
        "a full-depth draft must be the target bit for bit"
    );
    assert_eq!(draft.final_hidden, full.final_hidden);
}

/// Truncation must reach the head, stay finite, and actually change the
/// answer — through a stack whose earlier layers keep recurrent state.
#[test]
fn a_shorter_draft_executes_its_prefix_and_changes_the_answer() {
    let (_dir, plan, store) = hybrid();
    let depth = plan.layers.len();
    assert!(depth >= 2, "the fixture needs a layer to drop");
    let tokens = &[1u32, 0, 2];

    let full = execute_slice(
        &plan,
        &store,
        tokens,
        &ReferenceBackend,
        ExecutionSlice::Full,
    )
    .expect("full traversal");
    let short = execute_slice(
        &plan,
        &store,
        tokens,
        &ReferenceBackend,
        ExecutionSlice::Draft { end: depth - 1 },
    )
    .expect("shortened draft traversal");

    assert_eq!(short.executed_layers, (0..depth - 1).collect::<Vec<_>>());
    let logits = short.logits.as_ref().expect("the draft carries a head");
    assert_eq!(logits.len(), full.logits.as_ref().unwrap().len());
    assert!(logits.iter().all(|v| v.is_finite()));
    assert_ne!(
        short.logits, full.logits,
        "dropping a layer changed nothing — the slice was ignored"
    );
}

/// A draft is a model, so it must refuse the degenerate depths rather
/// than serve a silently wrong submodel.
#[test]
fn a_draft_refuses_zero_depth_and_overdeep_requests() {
    let (_dir, plan, store) = hybrid();
    let depth = plan.layers.len();
    let tokens = &[1u32];
    for (end, expect) in [(0usize, "at least one layer"), (depth + 1, "deeper than")] {
        let err = execute_slice(
            &plan,
            &store,
            tokens,
            &ReferenceBackend,
            ExecutionSlice::Draft { end },
        )
        .expect_err("a draft of depth {end} must be refused");
        assert!(
            err.to_string().contains(expect),
            "depth {end} refused with the wrong reason: {err}"
        );
    }
}

/// "The capital of France is" — the QW-3.7 trajectory's prompt, so a
/// divergence here can be read against a banked result rather than
/// against nothing.
const PROMPT_TOKENS: &[u32] = &[976, 6722, 315, 9822, 374];

#[test]
#[ignore = "needs the real 51 GB Qwen3.8-27B container; set QW38_CONTAINER"]
fn a_full_depth_draft_is_the_target_exactly() {
    let Ok(container) = std::env::var("QW38_CONTAINER") else {
        eprintln!("SKIP draft_slice: set QW38_CONTAINER");
        return;
    };
    let root = std::path::Path::new(&container);
    let inspection = inspect_container(root, false).unwrap();
    let plan = plan_component_ops(&inspection, root, "target")
        .unwrap()
        .plan
        .expect("the container carries a target plan");
    let store = OperandStore::open(root, &inspection).unwrap();
    let backend = ReferenceBackend::new();
    let depth = plan.layers.len();
    eprintln!("draft_slice: {depth} layers, prompt {PROMPT_TOKENS:?}");

    let full = execute_slice(&plan, &store, PROMPT_TOKENS, &backend, ExecutionSlice::Full)
        .expect("full traversal");
    let draft = execute_slice(
        &plan,
        &store,
        PROMPT_TOKENS,
        &backend,
        ExecutionSlice::Draft { end: depth },
    )
    .expect("full-depth draft traversal");

    // Which layers ran, not merely how many. A count cannot tell a
    // prefix from a planner falling back to the whole stack.
    assert_eq!(
        draft.executed_layers, full.executed_layers,
        "a full-depth draft must execute exactly the layers the target does"
    );
    assert_eq!(
        draft.executed_layers,
        (0..depth).collect::<Vec<_>>(),
        "and those layers must be the whole stack in order"
    );

    let (Some(fl), Some(dl)) = (full.logits.as_ref(), draft.logits.as_ref()) else {
        panic!("both arms must carry an output head");
    };
    assert_eq!(fl.len(), dl.len(), "logit width must match");
    let differing = fl.iter().zip(dl).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing,
        0,
        "full-depth draft diverged from the target in {differing} of {} logits — the seam \
         changes the model, so no reduced-depth measurement through it would mean anything",
        fl.len()
    );
    eprintln!("draft_slice: {} logits bit-identical", fl.len());
}

#[test]
#[ignore = "needs the real 51 GB Qwen3.8-27B container; set QW38_CONTAINER"]
fn one_layer_short_executes_the_prefix_and_changes_the_answer() {
    let Ok(container) = std::env::var("QW38_CONTAINER") else {
        eprintln!("SKIP draft_slice: set QW38_CONTAINER");
        return;
    };
    let root = std::path::Path::new(&container);
    let inspection = inspect_container(root, false).unwrap();
    let plan = plan_component_ops(&inspection, root, "target")
        .unwrap()
        .plan
        .expect("the container carries a target plan");
    let store = OperandStore::open(root, &inspection).unwrap();
    let backend = ReferenceBackend::new();
    let depth = plan.layers.len();

    let full = execute_slice(&plan, &store, PROMPT_TOKENS, &backend, ExecutionSlice::Full)
        .expect("full traversal");
    let short = execute_slice(
        &plan,
        &store,
        PROMPT_TOKENS,
        &backend,
        ExecutionSlice::Draft { end: depth - 1 },
    )
    .expect("L-1 draft traversal");

    assert_eq!(
        short.executed_layers,
        (0..depth - 1).collect::<Vec<_>>(),
        "the L-1 draft must execute exactly the first {} layers",
        depth - 1
    );

    let logits = short.logits.as_ref().expect("the draft carries a head");
    assert_eq!(
        logits.len(),
        full.logits.as_ref().unwrap().len(),
        "a draft speaks the target's vocabulary — same logit width"
    );
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "the L-1 draft produced non-finite logits — truncation is not clean through the \
         hybrid state machinery or the final head"
    );
    // Dropping a layer must change the answer. If it does not, the slice
    // was ignored and the "reduced" arm is the target wearing a label.
    assert!(
        logits
            .iter()
            .zip(full.logits.as_ref().unwrap())
            .any(|(a, b)| a != b),
        "the L-1 draft is bit-identical to the full target — the slice had no effect"
    );

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };
    eprintln!(
        "draft_slice: L-1 argmax {} vs target argmax {}",
        argmax(logits),
        argmax(full.logits.as_ref().unwrap())
    );
}
