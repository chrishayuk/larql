//! The byte arithmetic is a pure function of the layer shape, so it is
//! tested without a Metal device. The fixtures are gpt-oss-20b's real
//! dimensions, which is what makes the padding asymmetry visible.

use super::*;

/// gpt-oss-20b under Q6_K. Both K axes pad: 2880 is not a multiple of the
/// 256-element block, so `weight_cols` AND `inter_padded` round to 3072.
/// A live run reports whole-expert amplification 1.067 for exactly this
/// reason — an earlier fixture here assumed an unpadded down axis, which
/// is unreachable at block 256.
fn gptoss_q6k() -> ExpertLayerShape {
    // Q6_K: 210 bytes per 256-element block.
    let block = 256usize;
    let bytes_per_block = 210usize;
    let padded = 2880usize.div_ceil(block) * block;
    assert_eq!(padded, 3072, "gpt-oss's 2880 pads to 3072 at block 256");
    ExpertLayerShape {
        n_slots: 4,
        inter: 2880,
        inter_padded: padded,
        hidden: 2880,
        weight_cols: padded,
        row_bytes: (padded / block) * bytes_per_block,
        down_row_bytes: (padded / block) * bytes_per_block,
        split_scales: false,
    }
}

/// The same layer served from a native MXFP4 bank: no column padding, and
/// the e8m0 exponents live in their own streams.
fn gptoss_mxfp4() -> ExpertLayerShape {
    let block = 32usize;
    let bytes_per_block = 16usize; // 32 x 4-bit payload, scale held apart
    ExpertLayerShape {
        n_slots: 4,
        inter: 2880,
        inter_padded: 2880,
        hidden: 2880,
        weight_cols: 2880,
        row_bytes: (2880 / block) * bytes_per_block,
        down_row_bytes: (2880 / block) * bytes_per_block,
        split_scales: true,
    }
}

/// Q6_K's block padding shows up as amplification above 1.0. Both axes
/// pad at gpt-oss dims, so the whole-expert figure is the full 3072/2880
/// ratio — and a live `larql run` reports 1.067, which is what pins this.
#[test]
fn q6k_block_padding_appears_as_amplification() {
    let m = gptoss_q6k().movement();
    assert!(m.physical > m.semantic);
    let ampl = m.physical as f64 / m.semantic as f64;
    assert!(
        (ampl - 3072.0 / 2880.0).abs() < 1e-3,
        "expected the 3072/2880 padding ratio, got {ampl}"
    );
}

/// The banked per-token figure for gpt-oss-20b Q6_K experts is 2.09 GB
/// (24 layers x top-4). Pinned against the shape arithmetic so a change
/// to the byte accounting cannot silently move a number that this
/// project's roofline comparisons are calibrated against.
#[test]
fn q6k_matches_the_banked_per_token_expert_figure() {
    const LAYERS: u64 = 24;
    let per_layer = gptoss_q6k().movement().physical;
    let per_token_gb = (per_layer * LAYERS) as f64 / 1.0e9;
    assert!(
        (per_token_gb - 2.09).abs() < 0.02,
        "expected ~2.09 GB/token, got {per_token_gb}"
    );
}

/// The MXFP4 arm's own banked figure, from the same live run: 1.269 GB.
#[test]
fn mxfp4_matches_the_banked_per_token_expert_figure() {
    const LAYERS: u64 = 24;
    let per_layer = gptoss_mxfp4().movement().physical;
    let per_token_gb = (per_layer * LAYERS) as f64 / 1.0e9;
    assert!(
        (per_token_gb - 1.269).abs() < 0.01,
        "expected ~1.269 GB/token, got {per_token_gb}"
    );
}

/// Native MXFP4 stores unpadded, so nothing is amplified: semantic equals
/// physical. Representation savings must NOT show up in this ratio.
#[test]
fn native_mxfp4_is_unamplified() {
    let m = gptoss_mxfp4().movement();
    assert_eq!(m.semantic, m.physical, "no padding, no amplification");
    assert_eq!(
        m.useful, m.physical,
        "the grouped kernel streams whole rows"
    );
}

/// The representation change is a PHYSICAL byte reduction — the headline
/// MXFP4 result — and it lands in the right field.
#[test]
fn mxfp4_reduces_physical_bytes_against_q6k() {
    let q6k = gptoss_q6k().movement();
    let mxfp4 = gptoss_mxfp4().movement();
    assert!(mxfp4.physical < q6k.physical);
    let cut = 1.0 - mxfp4.physical as f64 / q6k.physical as f64;
    // Q6_K is 6.5625 bpw over padded columns; MXFP4 is 4.25 bpw unpadded.
    assert!(
        (0.30..0.42).contains(&cut),
        "expected a ~1/3 physical reduction, got {cut}"
    );
}

/// Split-scale banks must count their e8m0 streams. Dropping them would
/// under-report MXFP4 by the 0.25 bpw that makes it 4.25, not 4.0 —
/// precisely the term this project's format floor arithmetic turns on.
#[test]
fn split_scale_streams_are_counted() {
    let with_scales = gptoss_mxfp4();
    let without = ExpertLayerShape {
        split_scales: false,
        ..with_scales
    };
    let a = with_scales.movement();
    let b = without.movement();
    assert!(a.physical > b.physical);

    // Payload is 4 bits/weight, e8m0 adds 1 byte per 32 weights = 0.25
    // bits/weight, so scales are exactly 1/17 of the 4.25 bpw total.
    let scale_share = (a.physical - b.physical) as f64 / a.physical as f64;
    assert!(
        (scale_share - 1.0 / 17.0).abs() < 1e-3,
        "scale share {scale_share} should be 0.25/4.25"
    );
}

/// Slot count scales the whole block linearly — top-4 reads four experts.
#[test]
fn movement_scales_linearly_with_slots() {
    let one = ExpertLayerShape {
        n_slots: 1,
        ..gptoss_mxfp4()
    }
    .movement();
    let four = gptoss_mxfp4().movement();
    assert_eq!(four.physical, one.physical * 4);
    assert_eq!(four.semantic, one.semantic * 4);
}

/// The gate/up and down axes pad independently. A shape padded only on
/// the down axis must amplify only through the down term — a single
/// whole-expert ratio would mis-attribute it.
#[test]
fn down_axis_padding_is_accounted_separately() {
    let block = 256usize;
    let bytes_per_block = 210usize;
    let shape = ExpertLayerShape {
        n_slots: 1,
        inter: 2880,
        inter_padded: 3072, // down pads, gate/up does not
        hidden: 2880,
        weight_cols: 2880,
        row_bytes: (2880 / block) * bytes_per_block,
        down_row_bytes: (3072 / block) * bytes_per_block,
        split_scales: false,
    };
    let m = shape.movement();
    assert!(m.physical > m.semantic);

    // Mirror image of the Q6_K fixture: same total padding, other axis.
    let mirrored = ExpertLayerShape {
        inter_padded: 2880,
        weight_cols: 3072,
        row_bytes: (3072 / block) * bytes_per_block,
        down_row_bytes: (2880 / block) * bytes_per_block,
        ..shape
    };
    let mm = mirrored.movement();
    // Gate/up is two matrices to down's one, so padding the gate/up axis
    // costs more physical bytes than padding the down axis. If these came
    // out equal the accounting would be collapsing the two axes.
    assert!(
        mm.physical > m.physical,
        "gate/up padding must cost more than down padding: {} vs {}",
        mm.physical,
        m.physical
    );
}

/// A zero-slot layer moves nothing rather than dividing by zero.
#[test]
fn zero_slots_move_nothing() {
    let m = ExpertLayerShape {
        n_slots: 0,
        ..gptoss_mxfp4()
    }
    .movement();
    assert_eq!(m.physical, 0);
    assert_eq!(m.semantic, 0);
    assert_eq!(m.useful, 0);
}

/// Degenerate stored widths must not panic — the semantic ratio guards
/// its denominator.
#[test]
fn zero_stored_width_does_not_divide_by_zero() {
    let m = ExpertLayerShape {
        n_slots: 1,
        inter: 0,
        inter_padded: 0,
        hidden: 0,
        weight_cols: 0,
        row_bytes: 0,
        down_row_bytes: 0,
        split_scales: true,
    }
    .movement();
    assert_eq!(m.physical, 0);
    assert_eq!(m.semantic, 0);
}
