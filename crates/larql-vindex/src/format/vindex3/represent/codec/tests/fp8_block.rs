//! `F8_E4M3` — fine-grained FP8 — decodes to what the reference decodes
//! to, and refuses a grid that is not its grid.

use super::*;
use larql_models::quant::fp8_finegrained::{dequantize, Fp8Grid};

fn fixture() -> Fixture {
    fixtures()
        .into_iter()
        .find(|f| f.label() == DTYPE_FP8_BLOCK)
        .expect("the FP8 fixture is registered")
}

#[test]
fn the_codes_decode_bit_exactly_through_the_reference_dequantiser() {
    let fixture = fixture();
    let scales = fp8_scales();
    let grid = Fp8Grid {
        rows: ROWS,
        cols: K,
        scale_rows: FP8_SCALE_ROWS,
        scale_cols: FP8_SCALE_COLS,
    };
    let expected = dequantize(&fixture.buffers[0], &scales, grid).unwrap();
    let decoded = fixture
        .codec
        .decode_all(
            &fixture.operands(),
            &fixture.shape,
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&decoded), bits(&expected));
    // And the fixture is a quantisation of the ramp, not noise: E4M3 has
    // three mantissa bits, so every value lands within a sixteenth of
    // itself (relative), and the scale grid is what puts it there.
    let values = ramp(ROWS * K);
    for (got, want) in decoded.iter().zip(&values) {
        assert!(
            (got - want).abs() <= want.abs() * 0.0625 + 1e-3,
            "{got} vs {want}"
        );
    }
}

#[test]
fn the_identity_names_the_family_and_the_label_is_the_checkpoints_dtype() {
    let id = FP8_BLOCK.identity();
    assert_eq!(FP8_BLOCK.encoding_label(), "F8_E4M3");
    assert_eq!(id.family, "fp8-block");
    assert_eq!(id.element, "e4m3");
    assert_eq!(
        FP8_BLOCK
            .required_auxiliaries(RepresentationExtent::BASE)
            .len(),
        1
    );
    assert_eq!(
        FP8_BLOCK.required_auxiliaries(RepresentationExtent::BASE)[0].name,
        FP8_SCALES
    );
    // Codes only: the grid is another operand's footprint.
    assert_eq!(
        FP8_BLOCK
            .stored_bytes(&[ROWS, K], RepresentationExtent::BASE, TENSOR)
            .unwrap(),
        (ROWS * K) as u64
    );
}

#[test]
fn a_grid_that_does_not_tile_the_matrix_is_refused_from_metadata() {
    let judge = |shape: Vec<usize>| {
        FP8_BLOCK.validate_auxiliary(
            FP8_SCALES,
            &AuxiliaryMetadata {
                object: "o".into(),
                tensor: "s".into(),
                label: "F32".into(),
                shape,
                identity: None,
            },
            &[ROWS, K],
            RepresentationExtent::BASE,
            TENSOR,
        )
    };
    judge(vec![FP8_SCALE_ROWS, FP8_SCALE_COLS]).expect("the fixture's grid tiles");
    judge(vec![1, 1]).expect("one tile over the whole matrix tiles");
    // Two tile rows over three matrix rows: no even tiling.
    let err = judge(vec![2, FP8_SCALE_COLS]).unwrap_err();
    assert!(
        matches!(&err, CodecError::AuxiliaryGeometry { name, .. } if name == FP8_SCALES),
        "{err}"
    );
    assert!(err.to_string().contains("not evenly tiled"), "{err}");
    // A one-dimensional grid is not a grid.
    let err = judge(vec![FP8_SCALE_COLS]).unwrap_err();
    assert!(
        matches!(&err, CodecError::AuxiliaryGeometry { .. }),
        "{err}"
    );
    // A zero axis would divide by zero.
    let err = judge(vec![0, FP8_SCALE_COLS]).unwrap_err();
    assert!(err.to_string().contains("zero axis"), "{err}");
}

#[test]
fn the_codes_without_their_grid_are_refused_by_name() {
    let fixture = fixture();
    let operands = CodecOperands::from_streams(
        FP8_BLOCK
            .bind_packed(&fixture.buffers[0], &fixture.shape, TENSOR)
            .unwrap(),
    );
    let err = FP8_BLOCK
        .validate(
            &operands,
            &fixture.shape,
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::MissingAuxiliary { name, .. } if name == FP8_SCALES),
        "{err}"
    );
}

#[test]
fn a_grid_short_of_its_declared_shape_is_refused_on_the_values() {
    let fixture = fixture();
    let short = fp8_scales()[..FP8_SCALE_ROWS * FP8_SCALE_COLS - 1].to_vec();
    let short_fixture = Fixture {
        codec: fixture.codec,
        shape: fixture.shape.clone(),
        buffers: fixture.buffers.clone(),
        packed: true,
        auxiliaries: vec![(FP8_SCALES, vec![FP8_SCALE_ROWS, FP8_SCALE_COLS], short)],
    };
    let err = FP8_BLOCK
        .validate(
            &short_fixture.operands(),
            &short_fixture.shape,
            RepresentationExtent::BASE,
            TENSOR,
        )
        .unwrap_err();
    assert!(
        matches!(&err, CodecError::AuxiliaryGeometry { name, .. } if name == FP8_SCALES),
        "{err}"
    );
}

#[test]
fn a_rank_other_than_two_has_no_reading() {
    let err = FP8_BLOCK
        .stored_bytes(&[2, ROWS, K], RepresentationExtent::BASE, TENSOR)
        .unwrap_err();
    assert!(matches!(&err, CodecError::Geometry { .. }), "{err}");
    assert!(err.to_string().contains("[out, in]"), "{err}");
}
