//! The public hyper-connected miniature (`fixtures_hyper_connection`) is
//! what it says: the production writer admits it, its plan closes over
//! the declared topology with a head object, and the whole stack runs to
//! finite logits over the fixture's vocabulary.

use crate::format::vindex3::fixtures_hyper_connection::{
    hyper_connected_container, HC_SINKHORN_ITERS, HC_STREAMS, HC_VOCAB,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::execute_text;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::plan_component_ops;
use larql_models::config::ResidualTopology;

#[test]
fn the_public_hyper_connected_miniature_encodes_plans_and_executes() {
    let work = tempfile::tempdir().unwrap();
    let container = hyper_connected_container(work.path());
    let inspection = inspect_container(&container, false).unwrap();
    let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let plan = outcome.plan.expect("a closed outcome carries a plan");
    match plan.residual_topology {
        ResidualTopology::HyperConnection(hc) => {
            assert_eq!(hc.streams, HC_STREAMS);
            assert_eq!(hc.sinkhorn_iters, HC_SINKHORN_ITERS);
        }
        other => panic!("the miniature declares a hyper-connection, not {other:?}"),
    }
    assert!(plan.hyper_connection_head.is_some());

    let store = OperandStore::open(&container, &inspection).unwrap();
    let trace = execute_text(&plan, &store, &[1, 2, 3]).unwrap();
    let logits = trace.logits.expect("the head prices the vocabulary");
    assert_eq!(logits.len(), HC_VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
}
