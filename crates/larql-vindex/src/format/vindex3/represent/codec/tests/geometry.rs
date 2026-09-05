//! One reading of a shape, for every rank.

use super::*;

const LABEL: &str = "X";

#[test]
fn the_last_axis_is_k_and_the_rest_fold_into_rows() {
    let of = |shape: &[usize]| RowGeometry::of(shape, LABEL, TENSOR).unwrap();
    assert_eq!(of(&[]), RowGeometry { rows: 1, k: 1 });
    assert_eq!(of(&[7]), RowGeometry { rows: 1, k: 7 });
    assert_eq!(of(&[3, 4]), RowGeometry { rows: 3, k: 4 });
    assert_eq!(of(&[2, 3, 4]), RowGeometry { rows: 6, k: 4 });
    assert_eq!(of(&[2, 3, 4]).elements(LABEL, TENSOR).unwrap(), 24);
}

#[test]
fn overflow_is_refused_rather_than_wrapped() {
    let err = RowGeometry::of(&[usize::MAX, 2, 2], LABEL, TENSOR).unwrap_err();
    assert!(
        matches!(&err, CodecError::Geometry { why, .. } if why.contains("row count")),
        "{err}"
    );
    let huge = RowGeometry::of(&[usize::MAX, usize::MAX], LABEL, TENSOR).unwrap();
    let err = huge.elements(LABEL, TENSOR).unwrap_err();
    assert!(
        matches!(&err, CodecError::Geometry { why, .. } if why.contains("element count")),
        "{err}"
    );
}

#[test]
fn a_row_range_must_lie_within_the_rows_and_be_ordered() {
    let g = RowGeometry { rows: 3, k: 8 };
    g.check_rows(&(0..3), LABEL, TENSOR).unwrap();
    g.check_rows(&(1..1), LABEL, TENSOR).unwrap();
    for bad in [0..4, 3..4] {
        let err = g.check_rows(&bad, LABEL, TENSOR).unwrap_err();
        assert!(matches!(err, CodecError::RowRange { rows: 3, .. }), "{err}");
    }
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = 2..1;
    assert!(g.check_rows(&reversed, LABEL, TENSOR).is_err());
}

#[test]
fn a_group_must_divide_k_and_a_zero_group_is_no_group() {
    let g = RowGeometry { rows: 1, k: 64 };
    assert_eq!(g.check_group(32, LABEL, TENSOR).unwrap(), 2);
    assert_eq!(g.check_group(64, LABEL, TENSOR).unwrap(), 1);
    let err = g.check_group(48, LABEL, TENSOR).unwrap_err();
    assert_eq!(
        err.to_string(),
        "tensor `layer.0.w`: shape [1, 64] cannot hold `X`: k=64 is not a whole number of 48-element groups"
    );
    assert!(g.check_group(0, LABEL, TENSOR).is_err());
}

#[test]
fn the_destination_must_hold_exactly_the_requested_rows() {
    let g = RowGeometry { rows: 3, k: 8 };
    g.check_destination(&(1..3), 16, TENSOR).unwrap();
    let err = g.check_destination(&(1..3), 17, TENSOR).unwrap_err();
    assert_eq!(
        err,
        CodecError::Destination {
            tensor: TENSOR.into(),
            need: 16,
            have: 17
        }
    );
}
