//! `StoredTensor::select` and `read`: the stored bytes exactly, whole or selected, and the exact f32 widening.

use larql_vindex::format::vindex3::stored_tensor::{locate_stored_tensor, StoredTensorError};
use larql_vindex::format::vindex3::weight_patch::qualified;

use crate::support::*;

const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;

fn bf16_at(bytes: &[u8], element: usize) -> [u8; BF16_BYTES] {
    [bytes[element * BF16_BYTES], bytes[element * BF16_BYTES + 1]]
}

#[test]
fn a_whole_read_is_the_segment_bytes() {
    let c = bf16_packed();
    let t = locate_stored_tensor(&c.root, MATRIX).unwrap();
    let selection = t.select(None, None).unwrap();
    assert_eq!(selection.shape, vec![MATRIX_ROWS, MATRIX_COLS]);
    assert_eq!(
        read_all(&t, &selection),
        segment_bytes(&c.root, CANONICAL, MATRIX)
    );
}

#[test]
fn rows_and_columns_are_read_in_the_order_given() {
    let c = bf16_packed();
    let t = locate_stored_tensor(&c.root, MATRIX).unwrap();
    let stored = segment_bytes(&c.root, CANONICAL, MATRIX);
    let (rows, cols) = ([7usize, 2, 63], [255usize, 0, 12]);
    let selection = t.select(Some(&rows), Some(&cols)).unwrap();
    assert_eq!(selection.shape, vec![rows.len(), cols.len()]);
    assert_eq!(selection.rows.as_deref(), Some(&rows[..]));
    assert_eq!(selection.columns.as_deref(), Some(&cols[..]));
    let expected: Vec<u8> = rows
        .iter()
        .flat_map(|r| cols.iter().map(move |c| (r, c)))
        .flat_map(|(r, c)| bf16_at(&stored, r * MATRIX_COLS + c))
        .collect();
    assert_eq!(read_all(&t, &selection), expected);
}

#[test]
fn rows_alone_and_columns_alone() {
    let c = bf16_packed();
    let t = locate_stored_tensor(&c.root, MATRIX).unwrap();
    let stored = segment_bytes(&c.root, CANONICAL, MATRIX);
    let row_bytes = MATRIX_COLS * BF16_BYTES;
    let rows = t.select(Some(&[3]), None).unwrap();
    assert_eq!(rows.shape, vec![1, MATRIX_COLS]);
    assert_eq!(read_all(&t, &rows), stored[3 * row_bytes..4 * row_bytes]);
    let cols = t.select(None, Some(&[1])).unwrap();
    assert_eq!(cols.shape, vec![MATRIX_ROWS, 1]);
    let expected: Vec<u8> = (0..MATRIX_ROWS)
        .flat_map(|r| bf16_at(&stored, r * MATRIX_COLS + 1))
        .collect();
    assert_eq!(read_all(&t, &cols), expected);
}

#[test]
fn a_one_dimensional_gain_has_rows_and_no_columns() {
    let c = bf16_packed();
    let t = locate_stored_tensor(&c.root, GAIN).unwrap();
    let stored = segment_bytes(&c.root, CANONICAL, GAIN);
    assert_eq!(read_all(&t, &t.select(None, None).unwrap()), stored);
    let picked = t.select(Some(&[4, 1]), None).unwrap();
    assert_eq!(picked.shape, vec![2]);
    let expected: Vec<u8> = [4, 1].iter().flat_map(|e| bf16_at(&stored, *e)).collect();
    assert_eq!(read_all(&t, &picked), expected);
    let err = t.select(None, Some(&[0])).unwrap_err();
    assert!(matches!(err, StoredTensorError::Selection { .. }), "{err}");
    assert!(err.to_string().contains("no columns"), "{err}");
}

/// Rows, columns, and a word the refusal must contain.
type Case<'a> = (Option<&'a [usize]>, Option<&'a [usize]>, &'a str);

#[test]
fn bad_selections_are_refused() {
    let c = bf16_packed();
    let t = locate_stored_tensor(&c.root, MATRIX).unwrap();
    let cases: [Case; 5] = [
        (Some(&[MATRIX_ROWS]), None, "outside"),
        (None, Some(&[MATRIX_COLS]), "outside"),
        (Some(&[1, 1]), None, "twice"),
        (None, Some(&[2, 3, 2]), "twice"),
        (Some(&[]), None, "empty"),
    ];
    for (rows, cols, why) in cases {
        let err = t.select(rows, cols).unwrap_err();
        assert!(
            matches!(err, StoredTensorError::Selection { .. }) && err.to_string().contains(why),
            "{rows:?} {cols:?}: {err}"
        );
    }
}

#[test]
fn widening_bf16_is_exact() {
    let c = bf16_packed();
    let t = locate_stored_tensor(&c.root, &qualified(EMBEDDING, AMBIGUOUS)).unwrap();
    let stored = segment_bytes(&c.root, EMBEDDING, AMBIGUOUS);
    let values = t.widen(&stored).unwrap();
    assert_eq!(values.len(), t.element_count());
    for (i, v) in values.iter().enumerate() {
        let bits = u16::from_le_bytes(bf16_at(&stored, i));
        assert_eq!(v.to_bits(), u32::from(bits) << 16, "element {i}");
    }
}

#[test]
fn an_f32_tensor_reads_its_stored_bytes_and_widens_to_itself() {
    let c = f32_miniature();
    let t = locate_stored_tensor(&c.root, F32_MATRIX).unwrap();
    let stored = segment_bytes(&c.root, F32_CANONICAL, F32_MATRIX);
    assert_eq!(read_all(&t, &t.select(None, None).unwrap()), stored);
    let values = t.widen(&stored).unwrap();
    let back: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(back, stored);
    assert_eq!(stored.len(), t.element_count() * F32_BYTES);
}

#[test]
fn a_sink_error_stops_the_read() {
    let c = bf16_packed();
    let t = locate_stored_tensor(&c.root, MATRIX).unwrap();
    for selection in [
        t.select(None, None).unwrap(),
        t.select(Some(&[0, 1]), None).unwrap(),
    ] {
        let err = t
            .read(&selection, &mut |_| Err(std::io::Error::other("sink full")))
            .unwrap_err();
        assert!(matches!(err, StoredTensorError::Io(_)), "{err}");
    }
}
