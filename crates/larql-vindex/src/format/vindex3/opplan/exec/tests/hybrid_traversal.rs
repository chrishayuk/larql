//! QW-3.6b: the traversal runs a hybrid stack.
//!
//! Four gates, frozen before the refactor:
//!
//! 1. **Dispatch** — an `LLLF` stack calls three recurrent paths and one
//!    softmax path, checked by POSITION rather than by count.
//! 2. **Wrong state kind** — a provider that cannot hold recurrent
//!    buffers refuses, and refuses before committing any output.
//! 3. **State update** — a recurrent layer mutates both its buffers and
//!    appends no KV row; a softmax layer appends one KV position and
//!    touches no recurrent buffer.
//! 4. **Prefix equivalence** — `0..n` in one batch equals `0..k` then
//!    `k..n` with the state persisted. The first test that proves the
//!    whole continuation substrate composes rather than merely typechecks.

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::fixtures::hybrid_lllf_f32_model;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::kv::{
    ContinuationError, ContinuationProvider, RowKvState,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerAttention};

/// The encoded hybrid fixture, planned and ready to execute.
pub(super) fn hybrid_plan_for_tests() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    hybrid()
}

pub(super) fn hybrid() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let src = tempfile::tempdir().unwrap();
    hybrid_lllf_f32_model(src.path());
    let inventory = larql_models::inventory::build_inventory(src.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("hybrid".to_string(), inventory)], container.path())
        .expect("the hybrid fixture is admissible");
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:#?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, outcome.plan.unwrap(), store)
}

fn run(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    tokens: &[u32],
    provider: &mut dyn ContinuationProvider,
) -> Result<Vec<f32>, crate::error::VindexError> {
    let base = provider.position();
    let out = crate::format::vindex3::opplan::exec::prefill_plan(
        plan,
        store,
        tokens,
        &ReferenceBackend,
        provider,
    )?;
    assert_eq!(provider.position(), base + tokens.len());
    Ok(out.logits.expect("the fixture carries an output head"))
}

/// **Gate 1 — dispatch.** Three recurrences then one softmax, by position.
#[test]
fn an_lllf_stack_dispatches_three_recurrences_then_one_softmax() {
    let (_c, plan, _store) = hybrid();
    let kinds: Vec<&str> = plan
        .layers
        .iter()
        .map(|l| match &l.attention {
            LayerAttention::GatedDelta(_) => "L",
            LayerAttention::Kda(_) => "K",
            LayerAttention::Softmax(_) => "F",
            // Never reached — this fixture builds no MLA/Mamba2 layer.
            LayerAttention::Mla(_) => "M",
            LayerAttention::Mamba2(_) => "S",
        })
        .collect();
    assert_eq!(
        kinds,
        vec!["L", "L", "L", "F"],
        "the plan itself must carry the cadence, by position"
    );
}

/// **Gate 2 — wrong state kind, refused before any output.**
///
/// A KV-only provider meets a stack whose first three layers keep no
/// rows. It must refuse, name the semantic, and emit nothing — the
/// softmax layer is LAST precisely so a lazy refusal cannot pass.
#[test]
fn a_kv_only_provider_refuses_a_recurrence_before_committing_output() {
    /// Holds rows and says plainly that it holds nothing else.
    #[derive(Default)]
    struct KvOnly(RowKvState);
    impl ContinuationProvider for KvOnly {
        fn prepare(
            &mut self,
            layers: &[crate::format::vindex3::opplan::exec::kv::LayerKvGeometry],
        ) {
            self.0.prepare(layers)
        }
        fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
            self.0.append(layer, key, value)
        }
        fn keys(&self, layer: usize) -> &[Vec<f32>] {
            self.0.keys(layer)
        }
        fn values(&self, layer: usize) -> &[Vec<f32>] {
            self.0.values(layer)
        }
        fn position(&self) -> usize {
            self.0.position()
        }
        fn set_position(&mut self, position: usize) {
            self.0.set_position(position)
        }
        fn recurrent_state(
            &mut self,
            layer: usize,
        ) -> Result<
            &mut crate::format::vindex3::opplan::exec::continuation::RecurrentState,
            ContinuationError,
        > {
            Err(ContinuationError::RecurrentUnsupported {
                provider: "KvOnly",
                layer,
            })
        }
    }

    let (_c, plan, store) = hybrid();
    let mut provider = KvOnly::default();
    let geometry =
        crate::format::vindex3::opplan::exec::continuation::plan_continuation_geometry(&plan)
            .unwrap();
    // Refused at ANNOUNCEMENT — earlier still than the first layer. The
    // provider is told the geometry before a single token is embedded,
    // so a provider that cannot hold it says so there.
    let announced = provider
        .prepare_continuation(&geometry)
        .expect_err("a KV-only provider cannot hold this stack's state");
    assert!(
        matches!(announced, ContinuationError::RecurrentUnsupported { .. }),
        "the refusal must say the provider lacks recurrent state, not something \
         downstream: {announced:?}"
    );

    // ...and the same refusal on a traversal that USES the provider,
    // before any output is committed. The softmax layer is LAST, so a
    // lazy refusal would have emitted three layers first.
    let mut provider = KvOnly::default();
    let err = crate::format::vindex3::opplan::exec::prefill_plan(
        &plan,
        &store,
        &[1u32, 2, 3, 4, 5],
        &ReferenceBackend,
        &mut provider,
    )
    .expect_err("a KV-only provider cannot prefill this stack");
    assert!(
        err.to_string().contains("recurrent"),
        "the refusal must name the missing side: {err}"
    );
    // Nothing was committed — not even allocated. The provider never
    // reached a layer, so its logical position never advanced.
    assert_eq!(
        provider.position(),
        0,
        "refused only AFTER advancing the continuation position"
    );
}

/// **Gate 3 — state update.** Each layer touches its own kind of state
/// and only its own.
#[test]
fn each_layer_updates_its_own_kind_of_state_and_no_other() {
    let (_c, plan, store) = hybrid();
    let mut provider = RowKvState::default();
    run(&plan, &store, &[1u32, 2, 3, 4, 5], &mut provider).unwrap();

    for (index, layer) in plan.layers.iter().enumerate() {
        match &layer.attention {
            // This fixture is a Gated DeltaNet stack; a KDA layer reaching
            // here would mean the fixture changed operator, which the
            // assertion should say rather than silently skip.
            LayerAttention::Kda(_) => panic!("layer {index}: fixture is Gated DeltaNet, not KDA"),
            LayerAttention::Mla(_) => panic!("layer {index}: fixture is Gated DeltaNet, not MLA"),
            LayerAttention::Mamba2(_) => {
                panic!("layer {index}: fixture is Gated DeltaNet, not Mamba2")
            }
            LayerAttention::GatedDelta(_) => {
                let state = provider.recurrent_state(index).expect("a recurrent layer");
                let matrix = state.buffer(0).cells().to_vec();
                let conv = state.buffer(1).cells().to_vec();
                assert!(
                    matrix.iter().any(|c| *c != 0.0),
                    "layer {index}: the delta matrix never moved"
                );
                assert!(
                    conv.iter().any(|c| *c != 0.0),
                    "layer {index}: the convolution history never moved"
                );
                assert!(
                    provider.keys(index).is_empty() && provider.values(index).is_empty(),
                    "layer {index} kept KV rows it has no keys or values for"
                );
            }
            LayerAttention::Softmax(_) => {
                assert_eq!(
                    provider.keys(index).len(),
                    5,
                    "layer {index}: one KV row per position"
                );
                assert_eq!(provider.values(index).len(), 5);
                assert!(
                    provider.recurrent_state(index).is_err(),
                    "layer {index} allocated recurrent buffers for a softmax layer"
                );
            }
        }
    }
}

/// **Gate 4 — prefix equivalence.** One batch equals a persisted split.
///
/// The integration proof: it exercises the convolution history, the delta
/// matrix, the KV rows, the global position and the hybrid dispatch at
/// once. Any one of them failing to carry shows up here.
#[test]
fn a_persisted_split_prefix_matches_one_batch() {
    let (_c, plan, store) = hybrid();
    let tokens = [1u32, 2, 3, 4, 5];

    let mut whole = RowKvState::default();
    let one = run(&plan, &store, &tokens, &mut whole).unwrap();

    let mut split = RowKvState::default();
    run(&plan, &store, &tokens[..3], &mut split).unwrap();
    let two = run(&plan, &store, &tokens[3..], &mut split).unwrap();

    // Compare the LAST position, which both arms produced.
    let (a, b) = (&one, &two);
    assert_eq!(a.len(), b.len());
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f32 = a.iter().map(|x| x * x).sum();
    let rel = (num / den.max(f32::MIN_POSITIVE)).sqrt();
    assert!(
        rel < 1e-5,
        "a persisted split must reproduce the single batch: rel_rms {rel:e} — one of \
         {{conv history, delta matrix, KV rows, position}} is not carrying"
    );
    assert_eq!(
        whole.position(),
        split.position(),
        "global position drifted"
    );
}

/// **QW-3.7 — decode equals batch on a hybrid stack.**
///
/// The single-position path and the batched one must agree token for
/// token. This is the gate that makes autoregressive generation
/// meaningful: a decode step that reconstructed its convolution window
/// from the current batch would see a window of ONE and still produce
/// plausible numbers, and only a comparison against the batched
/// realisation catches it.
///
/// Both arms teacher-force the same tokens, so a per-position difference
/// is attributable to the realisation rather than to the two arms having
/// generated different text.
#[test]
fn stepping_a_hybrid_stack_matches_the_batched_traversal() {
    use crate::format::vindex3::opplan::exec::decode::DecodeSession;

    let (_c, plan, store) = hybrid();
    let tokens = [1u32, 2, 3, 4, 5, 6, 7];

    // Batched: the realisation QW-3.6b proved against HF.
    let mut batched = RowKvState::default();
    let batch_logits = run(&plan, &store, &tokens, &mut batched).unwrap();

    // Stepped: one position at a time, state carried in place.
    let mut session = DecodeSession::new(&plan, &store, &ReferenceBackend).unwrap();
    let mut last = Vec::new();
    for &token in &tokens {
        last = session
            .step(token)
            .unwrap()
            .logits
            .expect("the fixture carries an output head");
    }

    assert_eq!(last.len(), batch_logits.len());
    let num: f32 = last
        .iter()
        .zip(&batch_logits)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let den: f32 = batch_logits.iter().map(|b| b * b).sum();
    let rel = (num / den.max(f32::MIN_POSITIVE)).sqrt();
    assert!(
        rel < 1e-5,
        "stepped decode disagrees with the batched traversal on a hybrid stack: \
         rel_rms {rel:e} — the recurrent buffers are not carrying across steps"
    );

    // The recurrent buffers must have MOVED, or the comparison above
    // would also be satisfied by two runs that both did nothing.
    for (index, layer) in plan.layers.iter().enumerate() {
        if matches!(layer.attention, LayerAttention::GatedDelta(_)) {
            let state = batched.recurrent_state(index).unwrap();
            assert!(
                state.buffer(0).cells().iter().any(|c| *c != 0.0)
                    && state.buffer(1).cells().iter().any(|c| *c != 0.0),
                "layer {index} finished with untouched state"
            );
        }
    }
}
