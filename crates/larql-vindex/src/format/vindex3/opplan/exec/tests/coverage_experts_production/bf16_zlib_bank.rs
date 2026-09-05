//! Rung 2, W4: a packed expert bank stored SEQUENTIALLY is refused in the
//! prepared-plan preflight by its access class — before one byte of it is
//! read — while the same op over a raw bf16 bank loads as it always did.
//!
//! Lives beside the routed fixture because that fixture is scoped here;
//! the forecast placed it in `bf16_zlib_execution.rs`, and the execution
//! notes record the move.

use super::super::bf16_zlib_execution::{transcode, Transcode};
use super::fixture::{
    bf16_carrier_store, bf16_op, encoded, miniature_gpt_oss, routed_fixture, BF16_SUFFIX, LAYERS,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::exec::experts::FfnOperands;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::{LayerFfn, OperandRef};

/// gate/up and down, per layer.
const BANKS_PER_LAYER: usize = 2;

#[test]
fn a_sequential_bank_is_refused_by_access_class_before_any_byte_is_read() {
    let op = bf16_op(&routed_fixture().op);
    let ffn = LayerFfn::Routed(Box::new(op));
    let f32_for = |_: &OperandRef| WeightFormat::F32;

    // Control: the raw bf16 carrier loads through the same preflight.
    let (_dir, _container, control) = bf16_carrier_store();
    FfnOperands::load(&ffn, (&control).into(), &f32_for, WeightFormat::F32)
        .expect("a bf16 bank is row-addressable");

    // Candidate: the same bank, each bf16 copy stored as one zlib stream.
    let dir = tempfile::tempdir().unwrap();
    miniature_gpt_oss(dir.path(), true);
    let container = encoded(dir.path(), "mini-gpt-oss");
    let transcoded = transcode(
        container.path(),
        |name, _| name.ends_with(BF16_SUFFIX),
        Transcode::Bf16Zlib,
    );
    assert_eq!(transcoded.len(), LAYERS * BANKS_PER_LAYER);
    let inspection = inspect_container(container.path(), false).unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();

    let before = store.load_count();
    let Err(err) = FfnOperands::load(&ffn, (&store).into(), &f32_for, WeightFormat::F32) else {
        panic!("a sequential bank cannot be sliced per expert");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("`BF16_ZLIB` provides sequential access; the plan requires row-random"),
        "refused by access class, not by dtype name: {msg}"
    );
    assert!(!msg.contains("expected stored dtype"), "{msg}");
    assert_eq!(store.load_count(), before, "the refusal read no bytes");
}
