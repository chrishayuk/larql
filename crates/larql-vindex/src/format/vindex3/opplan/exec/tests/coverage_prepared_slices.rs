//! What a prepared image that is NOT a whole model will and will not do.
//!
//! A slice is a deployment fact: an FFN worker holds FFN matrices and
//! nothing that executes a stack, a layer-range shard consumes hidden
//! states and has no embedding or head, a coordinator waits for the
//! provider it delegates to. Each refuses the part of the job it does
//! not own by name, and each still accounts for the bytes it does hold.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, DENSE_HIDDEN, DENSE_LAYERS,
};
use crate::format::vindex3::fixtures_routed::miniature_routed;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::routed_experts::{ExpertOutput, RoutedExpertProvider};
use crate::format::vindex3::opplan::exec::{
    execute_prepared_streaming, FinalState, Plane, ResumePoint,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// A prompt inside both fixtures' vocabularies.
const TOKENS: [u32; 3] = [1, 5, 9];
/// A layer index no fixture here has.
const ABSENT_LAYER: usize = 99;

struct Fixture {
    _src: tempfile::TempDir,
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

impl Fixture {
    fn build(write: fn(&std::path::Path), name: &str) -> Self {
        let src = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(write, src.path(), container.path(), name);
        let inspection = inspect_container(container.path(), false).unwrap();
        let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
        assert!(outcome.closed(), "defects: {:?}", outcome.defects);
        let plan = outcome.plan.unwrap();
        let store = OperandStore::open(container.path(), &inspection).unwrap();
        Self {
            _src: src,
            _container: container,
            plan,
            store,
        }
    }

    fn dense() -> Self {
        Self::build(dense_f32_model, "dense")
    }

    fn routed() -> Self {
        Self::build(miniature_routed, "routed")
    }

    fn prepare(&self, slice: ExecutionSlice) -> Result<PreparedOperands, VindexError> {
        PreparedOperands::load(&self.plan, &self.store, &ReferenceBackend::new(), slice)
    }

    /// An expert worker lowers the bank for the CPU production kernels,
    /// so it is prepared by that backend.
    fn prepare_expert_worker(
        &self,
        slice: ExecutionSlice,
    ) -> Result<PreparedOperands, VindexError> {
        PreparedOperands::load(&self.plan, &self.store, &ProductionBackend::new(), slice)
    }

    fn run(
        &self,
        ops: &PreparedOperands,
        resume: Option<ResumePoint>,
    ) -> Result<crate::format::vindex3::opplan::exec::FinalOutput, VindexError> {
        execute_prepared_streaming(
            &self.plan,
            ops,
            &TOKENS,
            &ReferenceBackend::new(),
            resume,
            &mut |_| Ok(()),
        )
    }
}

fn message<T>(result: Result<T, VindexError>) -> String {
    match result {
        Ok(_) => panic!("the call must be refused"),
        Err(e) => e.to_string(),
    }
}

/// A provider no test here may reach: every call below is refused before
/// a coordinator would delegate.
struct Unreachable;
impl RoutedExpertProvider for Unreachable {
    fn apply(&self, _: usize, _: &[f32], _: &[usize]) -> Result<Vec<ExpertOutput>, VindexError> {
        unreachable!("a refused call must not reach its provider")
    }
}

/// An FFN worker holds FFN matrices, accounts for them, and refuses to
/// execute a layer stack it does not hold.
#[test]
fn an_ffn_worker_accounts_for_its_matrices_and_refuses_the_stack() {
    let dense = Fixture::dense();
    let ops = dense
        .prepare(ExecutionSlice::DenseFfns {
            start: 0,
            end: DENSE_LAYERS,
        })
        .unwrap();
    assert!(ops.residency_census().ffn.total() > 0);
    assert_eq!(ops.mapped_residency().mapped_bytes, 0);
    assert!(ops.allocation_census().bytes > 0);
    let err = message(dense.run(&ops, None));
    assert!(err.contains("cannot execute a layer stack"), "{err}");
}

/// A whole dense image's allocations include every attention and FFN
/// matrix.
#[test]
fn a_whole_dense_image_counts_attention_and_ffn_allocations() {
    let dense = Fixture::dense();
    let ops = dense.prepare(ExecutionSlice::Full).unwrap();
    let census = ops.allocation_census();
    let residency = ops.residency_census();
    assert!(census.bytes >= residency.attention.total() + residency.ffn.total());
    assert_eq!(ops.mapped_residency().regions, 0);
}

/// A coordinator whose provider was never bound refuses to execute, for
/// either kind of delegated FFN; and a provider cannot be bound onto an
/// image that is not a coordinator.
#[test]
fn a_coordinator_without_its_provider_refuses_and_only_a_coordinator_binds_one() {
    let dense = Fixture::dense();
    let coordinator = dense.prepare(ExecutionSlice::DenseFfnCoordinator).unwrap();
    let err = message(dense.run(&coordinator, None));
    assert!(err.contains("dense FFN provider is not bound"), "{err}");

    let routed = Fixture::routed();
    let mut coordinator = routed
        .prepare(ExecutionSlice::RoutedExpertCoordinator)
        .unwrap();
    let err = message(routed.run(&coordinator, None));
    assert!(err.contains("routed expert provider is not bound"), "{err}");
    coordinator
        .bind_routed_expert_provider(Arc::new(Unreachable))
        .expect("a coordinator binds its provider");

    let mut whole = routed.prepare(ExecutionSlice::Full).unwrap();
    let err = message(whole.bind_routed_expert_provider(Arc::new(Unreachable)));
    assert!(err.contains("requires coordinator operands"), "{err}");
}

/// An expert worker holds its bank slice and accounts for it; a range
/// outside the plan's layers or experts is refused at preparation.
#[test]
fn an_expert_worker_accounts_for_its_bank_and_refuses_a_range_outside_the_plan() {
    let routed = Fixture::routed();
    let ops = routed
        .prepare_expert_worker(ExecutionSlice::RoutedExperts {
            start: 0,
            end: 1,
            expert_start: 0,
            expert_end: 1,
        })
        .unwrap();
    assert_eq!(ops.mapped_residency().mapped_bytes, 0);
    assert!(ops.allocation_census().bytes > 0);

    for slice in [
        ExecutionSlice::RoutedExperts {
            start: 1,
            end: 1,
            expert_start: 0,
            expert_end: 1,
        },
        ExecutionSlice::RoutedExperts {
            start: 0,
            end: 1,
            expert_start: 0,
            expert_end: ABSENT_LAYER,
        },
    ] {
        let err = message(routed.prepare_expert_worker(slice));
        assert!(err.contains("outside plan"), "{err}");
    }
}

/// A layer-range shard has no embedding: it refuses token ids, and runs
/// only from a resume plane — leaving the stack with a hidden state and
/// no logits, since it has no head either.
#[test]
fn a_layer_range_shard_consumes_hidden_states_and_produces_no_logits() {
    let dense = Fixture::dense();
    let ops = dense
        .prepare(ExecutionSlice::LayerRange { start: 0, end: 1 })
        .unwrap();
    let backend = ReferenceBackend::new();

    let err = message(ops.embed_token(&dense.plan, &backend, TOKENS[0], 0));
    assert!(err.contains("no embedding table"), "{err}");
    let err = message(dense.run(&ops, None));
    assert!(err.contains("carries no embedding table"), "{err}");
    assert!(ops
        .head_over_normed(&backend, &[0.0; DENSE_HIDDEN])
        .unwrap()
        .is_none());

    let rows = vec![vec![0.5f32; DENSE_HIDDEN]; TOKENS.len()];
    let out = dense
        .run(
            &ops,
            Some(ResumePoint {
                next_layer: 0,
                hidden: Plane::Rows(rows),
            }),
        )
        .expect("a shard runs from a resume plane");
    assert!(out.logits.is_none(), "a shard has no head");
    assert!(matches!(out.exit, FinalState::Hidden(ref h) if h.len() == DENSE_HIDDEN));
}

/// The dense image's FFN view refuses every layer it cannot answer for:
/// outside the plan, before the slice, past the slice.
#[test]
fn a_dense_ffn_image_is_refused_for_layers_the_slice_does_not_hold() {
    let dense = Fixture::dense();
    let later = dense
        .prepare(ExecutionSlice::LayerRange { start: 1, end: 2 })
        .unwrap();
    let err = message(later.dense_ffn_image(&dense.plan, ABSENT_LAYER));
    assert!(err.contains("outside the component plan"), "{err}");
    let err = message(later.dense_ffn_image(&dense.plan, 0));
    assert!(err.contains("precedes this prepared slice"), "{err}");

    let earlier = dense
        .prepare(ExecutionSlice::LayerRange { start: 0, end: 1 })
        .unwrap();
    let err = message(earlier.dense_ffn_image(&dense.plan, 1));
    assert!(err.contains("outside this prepared slice"), "{err}");
    earlier
        .dense_ffn_image(&dense.plan, 0)
        .expect("the slice's own layer answers");
}

/// A plan with no embedding op cannot run from token ids, even over an
/// image that holds a table: the op, not the table, is the declaration.
#[test]
fn a_plan_without_an_embedding_op_refuses_token_ids() {
    let dense = Fixture::dense();
    let ops = dense.prepare(ExecutionSlice::Full).unwrap();
    let mut plan = dense.plan.clone();
    plan.embedding = None;
    let err = message(execute_prepared_streaming(
        &plan,
        &ops,
        &TOKENS,
        &ReferenceBackend::new(),
        None,
        &mut |_| Ok(()),
    ));
    assert!(err.contains("has no embedding op"), "{err}");
    let err = message(ops.embed_token(&plan, &ReferenceBackend::new(), TOKENS[0], 0));
    assert!(err.contains("no embedding op"), "{err}");
}
