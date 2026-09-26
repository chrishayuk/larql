//! **Dense FFN placement, exercised from this crate.**
//!
//! A coordinator that keeps attention, norms and residuals, with every
//! dense FFN placed on workers, must decode bit-identically to the same
//! model prepared whole. The transport-level version of this gate lives
//! in `larql-inference`; this one drives the format executor's own
//! worker image ([`PreparedDenseFfns`]) and provider slot through an
//! in-process provider, so the boundary is proven where it is defined.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_HIDDEN, G_LAYERS, G_TOKENS,
};
use crate::format::vindex3::fixtures_routed::miniature_gpt_oss;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::dense_ffn::{validate_row, DenseFfnProvider};
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

const COMPONENT: &str = "target";

/// Decode steps: the prompt twice, so the sliding window is crossed
/// repeatedly rather than only at prefill.
const STEPS: usize = 2;

/// A closed plan over a freshly encoded container, and a store over it.
struct Subject {
    _dir: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

fn subject(write: impl FnOnce(&std::path::Path)) -> Subject {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, checkpoint.path(), container.path(), COMPONENT);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Subject {
        _dir: container,
        plan: outcome.plan.unwrap(),
        store,
    }
}

/// Workers that each own a half-open layer range, answering in-process.
struct Workers {
    plan: ComponentOpPlan,
    images: Vec<PreparedOperands>,
}

impl DenseFfnProvider for Workers {
    fn apply(&self, layer: usize, normalized: &[f32]) -> Result<Vec<f32>, VindexError> {
        let backend = ProductionBackend::new();
        let owner = self
            .images
            .iter()
            .find(|image| {
                let ExecutionSlice::DenseFfns { start, end } = image.slice() else {
                    unreachable!("workers are dense FFN images")
                };
                (*start..*end).contains(&layer)
            })
            .expect("every layer has an owner");
        owner
            .dense_ffns()
            .expect("a dense FFN image")
            .apply(&self.plan, &backend, layer, normalized)
    }
}

fn workers(subject: &Subject, ranges: &[(usize, usize)]) -> Workers {
    let backend = ProductionBackend::new();
    let images = ranges
        .iter()
        .map(|&(start, end)| {
            PreparedOperands::load(
                &subject.plan,
                &subject.store,
                &backend,
                ExecutionSlice::DenseFfns { start, end },
            )
            .unwrap()
        })
        .collect();
    Workers {
        plan: subject.plan.clone(),
        images,
    }
}

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn placed_dense_ffns_decode_bit_identically_to_the_whole_model() {
    for ranges in [vec![(0, G_LAYERS)], vec![(0, 1), (1, G_LAYERS)]] {
        let subject = subject(miniature_glimmer);
        let backend = ProductionBackend::new();
        let local = PreparedOperands::load(
            &subject.plan,
            &subject.store,
            &backend,
            ExecutionSlice::Full,
        )
        .unwrap();
        let mut coordinator = PreparedOperands::load(
            &subject.plan,
            &subject.store,
            &backend,
            ExecutionSlice::DenseFfnCoordinator,
        )
        .unwrap();
        assert_eq!(coordinator.residency_census().ffn.total(), 0);
        coordinator
            .bind_dense_ffn_provider(Arc::new(workers(&subject, &ranges)))
            .unwrap();
        let (mut local_kv, mut placed_kv) = (RowKvState::default(), RowKvState::default());
        let mut whole =
            DecodeSession::over_prepared(&subject.plan, &local, &backend, &mut local_kv).unwrap();
        let mut placed =
            DecodeSession::over_prepared(&subject.plan, &coordinator, &backend, &mut placed_kv)
                .unwrap();
        for id in G_TOKENS.iter().cycle().take(G_TOKENS.len() * STEPS) {
            assert_eq!(
                bits(&whole.step(*id).unwrap().logits.unwrap()),
                bits(&placed.step(*id).unwrap().logits.unwrap()),
                "ranges {ranges:?}: placed FFNs diverged at token {id}"
            );
        }
    }
}

#[test]
fn a_dense_ffn_worker_refuses_rows_and_layers_it_does_not_own() {
    let subject = subject(miniature_glimmer);
    let backend = ProductionBackend::new();
    let image = PreparedOperands::load(
        &subject.plan,
        &subject.store,
        &backend,
        ExecutionSlice::DenseFfns { start: 1, end: 2 },
    )
    .unwrap();
    let worker = image.dense_ffns().unwrap();
    let row = [0.1; G_HIDDEN];
    let refused = |layer: usize, x: &[f32], expect: &str| {
        let err = worker
            .apply(&subject.plan, &backend, layer, x)
            .unwrap_err()
            .to_string();
        assert!(err.contains(expect), "expected {expect:?}, got {err}");
    };
    refused(0, &row, "outside this dense FFN worker");
    refused(1, &row[1..], "finite values");
    let mut poisoned = row;
    poisoned[0] = f32::NAN;
    refused(1, &poisoned, "finite values");
    // A worker bound under one numerical provider may not silently answer
    // under another: the image was lowered for production.
    let err = worker
        .apply(&subject.plan, &ReferenceBackend, 1, &row)
        .unwrap_err()
        .to_string();
    assert!(err.contains("numerical provider changed"), "{err}");
    assert_eq!(
        worker
            .apply(&subject.plan, &backend, 1, &row)
            .unwrap()
            .len(),
        G_HIDDEN
    );
}

#[test]
fn only_a_coordinator_accepts_a_dense_ffn_provider() {
    let subject = subject(miniature_glimmer);
    let backend = ProductionBackend::new();
    let mut whole = PreparedOperands::load(
        &subject.plan,
        &subject.store,
        &backend,
        ExecutionSlice::Full,
    )
    .unwrap();
    let err = whole
        .bind_dense_ffn_provider(Arc::new(workers(&subject, &[(0, G_LAYERS)])))
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires coordinator operands"), "{err}");
}

#[test]
fn dense_ffn_placement_refuses_a_routed_stack() {
    // Placement is at the dense operation boundary; a routed FFN has no
    // such boundary to cut, so both sides refuse before loading a byte.
    let subject = subject(|dir| miniature_gpt_oss(dir, false));
    let backend = ProductionBackend::new();
    for slice in [
        ExecutionSlice::DenseFfnCoordinator,
        ExecutionSlice::DenseFfns {
            start: 0,
            end: subject.plan.layers.len(),
        },
    ] {
        let err = PreparedOperands::load(&subject.plan, &subject.store, &backend, slice)
            .err()
            .expect("a routed stack cannot be dense-placed")
            .to_string();
        assert!(err.contains("dense FFN placement requires"), "{err}");
    }
}

#[test]
fn row_validation_names_the_width_it_requires() {
    assert!(validate_row(&[0.0; G_HIDDEN], G_HIDDEN).is_ok());
    let err = validate_row(&[f32::INFINITY; G_HIDDEN], G_HIDDEN)
        .unwrap_err()
        .to_string();
    assert!(err.contains(&G_HIDDEN.to_string()), "{err}");
}
