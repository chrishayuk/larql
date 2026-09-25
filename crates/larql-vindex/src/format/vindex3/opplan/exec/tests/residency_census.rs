//! The census must account for the WHOLE image, and must not flatter it.
//!
//! Two ways a residency claim goes wrong, and each has a test here:
//! reporting a total that omits a site (the delta projections were f32
//! for 48 of Qwen3.8's 64 layers and no total said so), and reporting
//! geometry instead of allocation (which would agree with itself no
//! matter how much memory was really held).

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::fixtures::hybrid_lllf_f32_model;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::plan_component_ops;

/// A hybrid stack — three recurrent layers and one softmax — from an
/// **f32** checkpoint. The format matters to the assertions: there are no
/// stored compact bytes here to keep, so every site must report widened.
fn f32_container() -> (
    tempfile::TempDir,
    crate::format::vindex3::opplan::ComponentOpPlan,
    OperandStore,
) {
    let src = tempfile::tempdir().unwrap();
    hybrid_lllf_f32_model(src.path());
    let inventory = larql_models::inventory::build_inventory(src.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("hybrid".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// Every site a hybrid stack has is present and non-zero.
///
/// The delta site specifically: it is the one a census can silently omit,
/// because a recurrence contributes no operands to DEVICE placement and
/// the obvious implementation reuses that list.
#[test]
fn the_census_accounts_for_every_site_including_the_recurrence() {
    let (_c, plan, store) = f32_container();
    let backend = ProductionBackend::new();
    let session = DecodeSession::new(
        &plan,
        &store,
        &backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let census = session.residency_census();

    for site in ["embedding", "attention", "delta", "ffn", "head", "glue"] {
        let bytes = census
            .sites()
            .into_iter()
            .find(|(name, _)| *name == site)
            .map(|(_, b)| b.total())
            .unwrap_or(0);
        assert!(bytes > 0, "the census reports nothing for `{site}`");
    }
    assert_eq!(
        census.total(),
        census.sites().iter().map(|(_, s)| s.total()).sum::<usize>()
    );
}

/// An f32 checkpoint has no compact bytes to keep, and the census says so.
///
/// The control for the real-container reading: a census that reported
/// "compact" here would be describing the format the backend ASKED for
/// rather than the bytes it got.
#[test]
fn an_f32_checkpoint_reports_nothing_compact() {
    let (_c, plan, store) = f32_container();
    let backend = ProductionBackend::new();
    let session = DecodeSession::new(
        &plan,
        &store,
        &backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let census = session.residency_census();
    assert_eq!(
        census.compact(),
        0,
        "an f32 container has no stored compact bytes, so keeping some would mean the loader \
         invented them"
    );
    assert_eq!(census.widened_f32(), census.total());
}

/// The reference backend is f32 everywhere by declaration, so its image
/// weighs the same as production's over an f32 checkpoint.
///
/// Pinned because the two backends now answer `weight_format` by
/// different routes — the reference by identity, production through the
/// policy — and over a checkpoint with nothing to keep they must still
/// land in the same place.
#[test]
fn the_two_backends_agree_over_a_checkpoint_with_nothing_to_keep() {
    let (_c, plan, store) = f32_container();
    let production = DecodeSession::new(
        &plan,
        &store,
        &ProductionBackend::new(),
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap()
    .residency_census();
    let reference = DecodeSession::new(
        &plan,
        &store,
        &ReferenceBackend::new(),
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap()
    .residency_census();
    assert_eq!(production.total(), reference.total());
    assert_eq!(production.compact(), 0);
    assert_eq!(reference.compact(), 0);
}
