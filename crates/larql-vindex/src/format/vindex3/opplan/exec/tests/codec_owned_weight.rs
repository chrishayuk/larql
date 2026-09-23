//! `WeightFormat::CodecOwned` on a real stored operand: the loader keeps
//! the stored bytes and the stored representation's name exactly, judges no
//! dtype, and every accessor reports those bytes and nothing else. The cost
//! model has no rate for a plan an external backend executes.

use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightSlice};
use crate::format::vindex3::opplan::exec::cpu::cost::measured_rate_gbps;
use crate::format::vindex3::opplan::exec::cpu::PhysicalProjectionPlan;
use crate::format::vindex3::opplan::exec::operands::{OperandSource, OperandStore};
use crate::format::vindex3::opplan::exec::weights::{load_weight, LoadedWeight};
use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention};

#[test]
fn a_codec_owned_load_keeps_the_stored_bytes_and_their_name() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let container = tmp.path().join("dense.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "target");
    let inspection = inspect_container(&container, false).unwrap();
    let plan = plan_component_ops(&inspection, &container, "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(&container, &inspection).unwrap();
    let LayerAttention::Softmax(op) = &plan.layers[0].attention else {
        panic!("layer 0 is a softmax layer");
    };

    let source: OperandSource<'_> = (&store).into();
    let raw = source.load_raw(&op.q).unwrap();
    let loaded = load_weight((&store).into(), &op.q, WeightFormat::CodecOwned).unwrap();

    let LoadedWeight::CodecOwned { bytes, label } = &loaded else {
        panic!("expected CodecOwned, got {:?}", loaded.format());
    };
    assert_eq!(bytes, &raw.bytes, "the stored bytes, untouched");
    assert_eq!(label, &raw.dtype, "the stored representation's own name");

    assert_eq!(loaded.format(), WeightFormat::CodecOwned);
    assert_eq!(loaded.resident_bytes(), raw.bytes.len());
    assert_eq!(loaded.mapped_bytes(), 0);
    assert_eq!(loaded.padded_allocations(), 0);
    assert!(!loaded.is_widened_f32());
    let allocations = loaded.allocations();
    assert_eq!(allocations.len(), 1);
    assert_eq!(allocations[0], (bytes.as_ptr() as usize, bytes.len()));
    match loaded.slice() {
        WeightSlice::CodecOwned { bytes: b, label: l } => {
            assert_eq!(b, raw.bytes.as_slice());
            assert_eq!(l, raw.dtype);
        }
        _ => panic!("the slice must be CodecOwned"),
    }
}

#[test]
fn the_cost_model_has_no_rate_for_a_codec_owned_plan() {
    assert_eq!(measured_rate_gbps(PhysicalProjectionPlan::CodecOwned), None);
}
