//! The NVFP4 slab path: how a compiled pack becomes rows a kernel can
//! consume, and what it refuses rather than guess.
//!
//! This is the seam the fidelity claim rests on. When a quality run says
//! it measured NVFP4, what it measured is whatever these functions
//! handed the kernel — so the partitioning has to be exact and the
//! mismatches have to refuse. A slab that silently handed back the wrong
//! scales would produce finite, plausible numbers and a fidelity figure
//! describing a representation nobody asked for.
//!
//! Two levels of scale are the whole reason NVFP4 exists and the whole
//! reason this needs its own test: the E4M3 group scales are per sixteen
//! elements along the input axis and must be cut alongside the codes,
//! while the f32 tensor scale is matrix-wide and travels with every slab
//! unchanged.

use super::super::super::backend::{WeightFormat, WeightSlice};
use super::super::physical::PhysicalProjectionPlan;
use super::super::projector::WeightRows;
use larql_models::quant::nvfp4::{NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS};

/// A pack whose bytes are positionally identifiable: code byte `i` holds
/// `i`, scale byte `i` holds `i`. Nothing here is decoded — these tests
/// ask which bytes a slab hands over, not what they denote.
fn pack(rows: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let groups = k / NVFP4_GROUP_ELEMS;
    let packed = (0..rows * groups * NVFP4_GROUP_BYTES)
        .map(|i| (i % 251) as u8)
        .collect();
    let scales = (0..rows * groups).map(|i| (i % 241) as u8).collect();
    (packed, scales)
}

const TENSOR_SCALE: f32 = 0.125;

#[test]
fn a_pack_becomes_rows_and_reports_them_from_its_own_bytes() {
    let (packed, scales) = pack(4, 32);
    let slice = WeightSlice::Nvfp4 {
        packed: &packed,
        scales: &scales,
        tensor_scale: TENSOR_SCALE,
    };
    let rows = slice.rows(4, 32).expect("a well-formed pack yields rows");
    assert_eq!(rows.rows(32), 4, "row count is derived, not asserted");
    // 32 elements = 2 groups = 16 code bytes per row.
    assert_eq!(rows.bytes(), 4 * 16 + 4 * 2, "codes plus group scales");
}

/// The tensor scale is matrix-wide. A slab that rescaled it per
/// partition would change what the weights denote depending on how the
/// work was divided.
#[test]
fn slicing_rows_cuts_codes_and_scales_together_and_keeps_the_tensor_scale() {
    let (packed, scales) = pack(4, 32);
    let all = WeightRows::Nvfp4 {
        packed: &packed,
        scales: &scales,
        tensor_scale: TENSOR_SCALE,
    };

    let slab = all.slice_rows(32, 2, 2);
    let WeightRows::Nvfp4 {
        packed: p,
        scales: s,
        tensor_scale,
    } = slab
    else {
        panic!("slicing an NVFP4 slab must yield an NVFP4 slab");
    };

    assert_eq!(slab.rows(32), 2);
    assert_eq!(tensor_scale, TENSOR_SCALE, "matrix-wide, never re-derived");
    // Row 2 starts at code byte 2*16 and scale 2*2 — the two streams cut
    // at the same row, which is the property that matters.
    assert_eq!(p, &packed[32..64], "codes start at the requested row");
    assert_eq!(s, &scales[4..8], "scales start at the SAME row");
}

/// Groups run along the input axis, so a width that is not a multiple of
/// the group size means this pack does not describe these rows. The
/// format's own constant decides, not a policy.
#[test]
fn a_width_that_is_not_a_whole_number_of_groups_refuses() {
    let (packed, scales) = pack(2, 32);
    let slice = WeightSlice::Nvfp4 {
        packed: &packed,
        scales: &scales,
        tensor_scale: TENSOR_SCALE,
    };
    let err = slice
        .rows(2, NVFP4_GROUP_ELEMS + 1)
        .expect_err("a partial group must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("not a multiple") && msg.contains("does not describe these rows"),
        "the refusal must say why, not just fail: {msg}"
    );
}

/// A pack too short for the rows asked of it refuses rather than
/// handing back the rows it happens to have.
#[test]
fn a_short_pack_refuses_rather_than_returning_fewer_rows() {
    let (packed, scales) = pack(2, 32);
    let slice = WeightSlice::Nvfp4 {
        packed: &packed,
        scales: &scales,
        tensor_scale: TENSOR_SCALE,
    };
    assert!(
        slice.rows(4, 32).is_err(),
        "four rows from a two-row pack must refuse"
    );
}

/// Scales are indexed independently of codes, so a pack with enough
/// codes and too few scales must refuse too — the arm that would
/// otherwise hand a worker another row's scales.
#[test]
fn enough_codes_but_too_few_scales_still_refuses() {
    let (packed, scales) = pack(4, 32);
    let short_scales = &scales[..scales.len() - 1];
    let slice = WeightSlice::Nvfp4 {
        packed: &packed,
        scales: short_scales,
        tensor_scale: TENSOR_SCALE,
    };
    assert!(
        slice.rows(4, 32).is_err(),
        "codes alone do not make a slab valid"
    );
}

/// **The kernel is chosen by the representation, not by the arm.**
///
/// Every other format has an arithmetic arm that can move it between
/// kernels — Q8 bytes are consumable by a widening f32 kernel and by
/// SDOT alike, so a policy decides. NVFP4 has no such choice: its two
/// scale levels are not expressible as the single per-block f32 the
/// integer paths assume, so there is exactly one kernel and no arm
/// enters into it.
///
/// This is the mechanical half of the claim a fidelity run depends on.
/// If some arm could route NVFP4 bytes to a kernel that widened them
/// first, a quality number would describe a representation nobody
/// requested — and it would still be finite, plausible, and wrong.
#[test]
fn nvfp4_has_one_kernel_and_no_arithmetic_arm_can_move_it() {
    let (packed, scales) = pack(4, 32);
    let rows = WeightRows::Nvfp4 {
        packed: &packed,
        scales: &scales,
        tensor_scale: TENSOR_SCALE,
    };
    let plan = PhysicalProjectionPlan::for_resident(rows, 32);
    assert_eq!(
        plan,
        PhysicalProjectionPlan::FusedNvfp4,
        "NVFP4 resolves to its own kernel"
    );
    assert_eq!(
        plan.format(),
        WeightFormat::Nvfp4,
        "and the plan reports the format it actually runs"
    );
}
