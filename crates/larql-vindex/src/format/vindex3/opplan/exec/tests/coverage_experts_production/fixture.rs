//! The routed miniature, its carrier variant, and the load/compare
//! helpers shared by the expert-bank and production-backend gates.

use crate::format::vindex3::opplan::OperandRef;
use std::path::Path;

use larql_models::config::ExpertFormat;

use super::super::device::LoopDevice;
use crate::format::vindex3::encode::encode_system_unenforced as encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightFormats, WeightSlice};
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::experts::FfnOperands;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::{
    plan_component_ops, ComponentOpPlan, ExpertBank, LayerFfn, RoutedFfnOp,
};

pub(super) use crate::format::vindex3::fixtures_routed::*;

/// Encode `dir` into a fresh container.
pub(super) fn encoded(dir: &Path, artifact: &str) -> tempfile::TempDir {
    let inventory = larql_models::inventory::build_inventory(dir).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[(artifact.to_string(), inventory)], container.path()).unwrap();
    container
}

/// A closed plan and the store over the same container.
pub(super) fn closed_plan(container: &Path) -> (ComponentOpPlan, OperandStore) {
    let inspection = inspect_container(container, false).unwrap();
    let outcome = plan_component_ops(&inspection, container, "target").unwrap();
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let store = OperandStore::open(container, &inspection).unwrap();
    (outcome.plan.unwrap(), store)
}

/// The routed miniature: its closed plan, its store, and layer 0's op.
pub(super) struct RoutedFixture {
    _dir: tempfile::TempDir,
    _container: tempfile::TempDir,
    pub(super) store: OperandStore,
    pub(super) op: RoutedFfnOp,
    pub(super) plan: ComponentOpPlan,
}

pub(super) fn routed_fixture() -> RoutedFixture {
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), false);
    let container = encoded(dir.path(), "mini-gpt-oss");
    let (plan, store) = closed_plan(container.path());
    let op = plan.layers[0]
        .ffn
        .as_ref()
        .expect("layer 0 has an FFN")
        .routed()
        .expect("layer 0 is routed")
        .clone();
    RoutedFixture {
        _dir: dir,
        _container: container,
        store,
        op,
        plan,
    }
}

/// The store of a container whose bank also carries the BF16 copies —
/// not closable (the copies are unclassified), so the op comes from
/// [`routed_fixture`] and is edited to point at them.
pub(super) fn bf16_carrier_store() -> (tempfile::TempDir, tempfile::TempDir, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), true);
    let container = encoded(dir.path(), "mini-gpt-oss");
    let inspection = inspect_container(container.path(), false).unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (dir, container, store)
}

/// `op` re-pointed at the BF16 copies, declared `PackedBF16`.
pub(super) fn bf16_op(op: &RoutedFfnOp) -> RoutedFfnOp {
    let mut op = op.clone();
    op.expert_format = ExpertFormat::PackedBF16;
    let ExpertBank::Packed { gate_up, down } = &mut op.bank else {
        panic!("fixture builds a packed bank");
    };
    for projection in [gate_up, down] {
        projection.weights.tensor = projection
            .weights
            .tensor
            .replace(BLOCKS_SUFFIX, BF16_SUFFIX);
        projection.scales = None;
    }
    op
}

pub(super) fn routed(op: &RoutedFfnOp) -> LayerFfn {
    LayerFfn::Routed(Box::new(op.clone()))
}

pub(super) fn load(op: &RoutedFfnOp, store: &OperandStore, format: WeightFormat) -> FfnOperands {
    FfnOperands::load(
        &routed(op),
        store.into(),
        &|_: &OperandRef| Ok(format),
        format.into(),
        &|_: &OperandRef| Ok(format),
    )
    .unwrap()
}

pub(super) fn load_err(op: &RoutedFfnOp, store: &OperandStore, format: WeightFormat) -> String {
    FfnOperands::load(
        &routed(op),
        store.into(),
        &|_: &OperandRef| Ok(format),
        format.into(),
        &|_: &OperandRef| Ok(format),
    )
    .err()
    .expect("loading must refuse")
    .to_string()
}

/// A device backend that binds the FFN class in `format` and everything
/// else in f32 (the loop device has f32, f16 and MXFP4 gemvs).
pub(super) fn loop_device(format: WeightFormat) -> DevicePlanBackend<LoopDevice> {
    DevicePlanBackend::with_formats(
        LoopDevice,
        "loop-device-coverage",
        WeightFormats {
            attention: WeightFormat::F32,
            ffn: format,
            head: WeightFormat::F32,
        },
    )
}

pub(super) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// The variant name of every matrix the operands expose.
pub(super) fn slice_kinds(operands: &FfnOperands) -> Vec<String> {
    operands
        .weight_slices()
        .iter()
        .map(|s| slice_kind(s).to_string())
        .collect()
}

/// The two MXFP4 streams of every matrix, for byte comparison.
pub(super) fn mxfp4_bytes(operands: &FfnOperands) -> Vec<(Vec<u8>, Vec<u8>)> {
    operands
        .weight_slices()
        .iter()
        .map(|s| match s {
            WeightSlice::Mxfp4 { packed, scales } => (packed.to_vec(), scales.to_vec()),
            other => panic!("expected an MXFP4 slice, got {}", slice_kind(other)),
        })
        .collect()
}

/// The representation's name, for a panic message.
///
/// Delegates rather than matching again: this used to be a second copy of
/// the same table, and a copy is one variant away from disagreeing with
/// the one the refusals print.
pub(super) fn slice_kind<'a>(slice: &WeightSlice<'a>) -> &'a str {
    slice.representation()
}
