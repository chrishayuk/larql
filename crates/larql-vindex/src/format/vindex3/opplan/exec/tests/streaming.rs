//! The streaming/resume contract: `execute_plan_streaming` with a
//! [`ResumePoint`] must continue a run **bit-identically** — a resumed
//! long execution and an uninterrupted one may not differ in a single
//! bit, or every parity claim made over a resumed dump would need an
//! asterisk. The resume state is a persisted plane, so these tests feed
//! the interpreter exactly what the CLI would read back from disk.

use super::golden::{miniature_glimmer, G_TOKENS};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{
    execute_plan, execute_plan_streaming, ExecutionTrace, PlaneEvent, ResumePoint,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// Encode the miniature fixture and hand back its plan and store, plus
/// the uninterrupted trace every resume is judged against.
fn fixture() -> (
    tempfile::TempDir,
    ComponentOpPlan,
    OperandStore,
    ExecutionTrace,
) {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let full = execute_plan(&plan, &store, &G_TOKENS, &ReferenceBackend::new()).unwrap();
    (container, plan, store, full)
}

/// What one streamed run emitted: whether plane 000 appeared, the
/// per-layer planes in arrival order, and the final outputs.
type StreamedRun = (
    bool,
    Vec<(usize, Vec<Vec<f32>>)>,
    Vec<f32>,
    Option<Vec<f32>>,
);

/// Run the streaming form from `resume`, collecting what it emits.
fn streamed(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    resume: Option<ResumePoint>,
) -> StreamedRun {
    let mut saw_embedded = false;
    let mut layers = Vec::new();
    let out = execute_plan_streaming(
        plan,
        store,
        &G_TOKENS,
        &ReferenceBackend::new(),
        resume,
        &mut |event| {
            match event {
                PlaneEvent::Embedded(_) => saw_embedded = true,
                PlaneEvent::Layer { index, trace } => layers.push((index, trace.post_layer)),
            }
            Ok(())
        },
    )
    .unwrap();
    (saw_embedded, layers, out.final_hidden, out.logits)
}

#[test]
fn resume_from_a_mid_run_plane_is_bit_identical() {
    let (_c, plan, store, full) = fixture();
    // Plane 1 = the residual leaving layer 0 = the state entering layer 1.
    let resume = ResumePoint {
        next_layer: 1,
        hidden: full.layers[0].post_layer.clone(),
    };
    let (saw_embedded, layers, final_hidden, logits) = streamed(&plan, &store, Some(resume));
    assert!(!saw_embedded, "a resumed run must not re-emit plane 000");
    assert_eq!(layers.len(), full.layers.len() - 1);
    assert_eq!(layers[0].0, 1, "resume must continue at the next layer");
    assert_eq!(
        layers[0].1, full.layers[1].post_layer,
        "resumed layer output must be bit-identical"
    );
    assert_eq!(final_hidden, full.final_hidden);
    assert_eq!(logits, full.logits);
}

#[test]
fn resume_from_the_embedding_plane_replays_every_layer() {
    let (_c, plan, store, full) = fixture();
    let resume = ResumePoint {
        next_layer: 0,
        hidden: full.embedded.clone(),
    };
    let (saw_embedded, layers, final_hidden, logits) = streamed(&plan, &store, Some(resume));
    assert!(!saw_embedded);
    assert_eq!(layers.len(), full.layers.len());
    for (got, want) in layers.iter().zip(&full.layers) {
        assert_eq!(got.1, want.post_layer);
    }
    assert_eq!(final_hidden, full.final_hidden);
    assert_eq!(logits, full.logits);
}

#[test]
fn a_fresh_streaming_run_emits_every_plane_in_order() {
    let (_c, plan, store, full) = fixture();
    let (saw_embedded, layers, final_hidden, logits) = streamed(&plan, &store, None);
    assert!(saw_embedded, "a fresh run must emit plane 000");
    let indices: Vec<usize> = layers.iter().map(|(i, _)| *i).collect();
    assert_eq!(indices, (0..full.layers.len()).collect::<Vec<_>>());
    assert_eq!(final_hidden, full.final_hidden);
    assert_eq!(logits, full.logits);
}

#[test]
fn resume_refuses_mismatched_state() {
    let (_c, plan, store, full) = fixture();
    let backend = ReferenceBackend::new();

    // Wrong position count: a plane from a different fixture length.
    let short = ResumePoint {
        next_layer: 1,
        hidden: full.layers[0].post_layer[..G_TOKENS.len() - 1].to_vec(),
    };
    let err = execute_plan_streaming(&plan, &store, &G_TOKENS, &backend, Some(short), &mut |_| {
        Ok(())
    })
    .unwrap_err();
    assert!(err.to_string().contains("positions"), "{err}");

    // Wrong hidden width: a plane from a different model.
    let mut rows = full.layers[0].post_layer.clone();
    rows[0].pop();
    let narrow = ResumePoint {
        next_layer: 1,
        hidden: rows,
    };
    let err = execute_plan_streaming(
        &plan,
        &store,
        &G_TOKENS,
        &backend,
        Some(narrow),
        &mut |_| Ok(()),
    )
    .unwrap_err();
    assert!(err.to_string().contains("hidden size"), "{err}");

    // A resume point past the plan.
    let past = ResumePoint {
        next_layer: plan.layers.len() + 1,
        hidden: full.layers[0].post_layer.clone(),
    };
    let err = execute_plan_streaming(&plan, &store, &G_TOKENS, &backend, Some(past), &mut |_| {
        Ok(())
    })
    .unwrap_err();
    assert!(err.to_string().contains("past the plan"), "{err}");
}
