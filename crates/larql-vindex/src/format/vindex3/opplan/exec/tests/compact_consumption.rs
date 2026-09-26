//! CPU-2A/2: the compact bytes reach the kernel STILL COMPACT.
//!
//! The failure this file exists to catch is the one that looks like
//! success from every other angle: a loader that keeps bf16, a census that
//! reports half the bytes, and a consumption path that widens a scratch
//! copy before computing. Residency would be halved, throughput would be
//! unchanged, and nothing else in the suite would notice.
//!
//! So these assert on what the executor RAN, not on what the loader
//! decided.

use super::super::backend::{PlanBackend, ProjectCall, WeightSlice};
use super::super::cpu::{ledger, PhysicalProjectionPlan};
use super::super::production::ProductionBackend;
use super::super::weights::{AlignedBytes, LoadedWeight};
use crate::format::vindex3::fixtures::lcg_values;

/// bf16 is the top half of f32, so both directions are exact.
fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}
fn widen(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// **The no-hidden-widening control.**
///
/// A bf16 operand handed to the production backend must be consumed by
/// the fused kernel, at bf16 byte count. A path that widened to scratch
/// would produce the same numbers and record `BlasF32` at twice the
/// bytes, which is exactly the shape of the failure.
///
/// The tallies are compared as DELTAS and asserted with `>=`: the ledger
/// is process-wide and the suite runs in parallel, so an exact equality
/// here would be pinning the absence of other tests rather than the
/// behaviour of this one. The real-container gate, which runs alone,
/// asserts exactly.
#[test]
fn a_bf16_operand_is_consumed_compact() {
    const OUT: usize = 96;
    const IN: usize = 128;
    let values: Vec<f32> = lcg_values(OUT * IN, 7)
        .iter()
        .map(|v| widen(narrow(*v)))
        .collect();
    let units: Vec<u16> = values.iter().map(|v| narrow(*v)).collect();
    let x = lcg_values(IN, 8);

    let backend = ProductionBackend::new();
    let before = ledger().get(PhysicalProjectionPlan::FusedBf16);
    let compact = backend
        .project(ProjectCall {
            weight: WeightSlice::Bf16(&units),
            out_dim: OUT,
            in_dim: IN,
            x: &x,
        })
        .expect("a bf16 operand is a representation the CPU backend runs");
    let after = ledger().get(PhysicalProjectionPlan::FusedBf16);

    assert!(
        after.calls > before.calls,
        "the projection did not reach the fused kernel at all"
    );
    assert!(
        after.bytes - before.bytes >= (units.len() * 2) as u64,
        "the fused kernel read fewer bytes than the operand holds"
    );

    // Same weight VALUES through the f32 path: bf16 widening is exact, so
    // any disagreement here is summation order and nothing else.
    let widened = backend
        .project(ProjectCall {
            weight: WeightSlice::F32(&values),
            out_dim: OUT,
            in_dim: IN,
            x: &x,
        })
        .unwrap();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in compact.iter().zip(&widened) {
        num += (*a as f64 - *b as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
    assert!(
        rel < 1e-5,
        "compact and widened consumption of the SAME values disagree at rel_rms {rel:.3e} — \
         that is more than reassociation, so a value changed somewhere"
    );
}

/// A resident slab is page-padded, and the geometry — not its length —
/// says how many rows it holds.
///
/// **Qwen3.8 cannot show this.** Every one of its matrices is an exact
/// multiple of the 16 KiB device page, so padded and logical lengths
/// coincide and a version of `WeightSlice::rows` that trusted the slice
/// would decode the model perfectly. The shape here is chosen to be
/// awkward instead: `3 x 5` is 30 bytes inside a 16384-byte allocation,
/// so a length-trusting reader sees 1638 rows where there are 3.
#[test]
fn an_awkward_shape_is_cut_to_its_geometry_not_its_allocation() {
    const OUT: usize = 3;
    const IN: usize = 5;
    let values = lcg_values(OUT * IN, 9);
    let units: Vec<u16> = values.iter().map(|v| narrow(*v)).collect();
    let bytes: Vec<u8> = units.iter().flat_map(|u| u.to_le_bytes()).collect();

    let loaded = LoadedWeight::Bf16(AlignedBytes::from_bytes(&bytes));
    let slice = loaded.slice();
    let padded = slice.as_bf16().unwrap().len();
    assert!(
        padded > OUT * IN,
        "the fixture must actually be padded, or it proves nothing: {padded} units"
    );

    let rows = slice.rows(OUT, IN).expect("the matrix fits its allocation");
    let super::super::cpu::WeightRows::Bf16(cut) = rows else {
        panic!("a bf16 slab must stay bf16 through the row view");
    };
    assert_eq!(cut.len(), OUT * IN);
    assert_eq!(rows.rows(IN), OUT, "row count follows the geometry");
    assert_eq!(rows.bytes(), OUT * IN * 2, "and so does the byte count");

    // The padding is zeros, so a length-trusting reader would not crash —
    // it would silently compute 1635 extra rows of nothing. Naming the
    // number it would have seen keeps that failure legible.
    assert_eq!(padded, 8192);
}

/// Asking for more rows than are resident refuses.
#[test]
fn a_short_slab_refuses_rather_than_reading_past_the_matrix() {
    let units = vec![0x3f80u16; 10];
    let err = WeightSlice::Bf16(&units)
        .rows(4, 5)
        .expect_err("20 weights are not resident in a 10-unit slab")
        .to_string();
    assert!(err.contains("resident"), "{err}");
}

/// Every representation names itself, and the compact accessor refuses
/// anything that is not compact.
///
/// Enumerated rather than sampled: the name appears in the refusal a
/// reader debugs from, and a variant added without a name would print
/// nothing useful on the one day it mattered.
#[test]
fn every_representation_names_itself_and_only_bf16_answers_as_bf16() {
    let f = [1.0f32; 4];
    let b = [0x3f80u16; 4];
    let bytes = [0u8; 8];
    let cases = [
        (WeightSlice::F32(&f), "f32"),
        (WeightSlice::Bf16(&b), "bf16"),
        (WeightSlice::F16(&bytes), "f16"),
        (
            WeightSlice::Mxfp4 {
                packed: &bytes,
                scales: &bytes,
            },
            "mxfp4",
        ),
        (
            WeightSlice::Nvfp4 {
                packed: &bytes,
                scales: &bytes,
                tensor_scale: 1.0,
                activation: Default::default(),
            },
            "nvfp4",
        ),
    ];
    for (slice, name) in cases {
        assert_eq!(slice.representation(), name);
        assert_eq!(
            slice.as_bf16().is_ok(),
            name == "bf16",
            "`{name}` answered the compact accessor wrongly"
        );
    }
}
