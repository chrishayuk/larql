//! `.npy` headers whose shape wraps the byte count, with no data behind them.
//!
//! Before the checked arithmetic, `2^62 × 4` wrapped to 0 in release builds, so
//! empty data passed the length check and the shape reached the allocator.

use super::archive_errors::{
    load_err, one_entry, store_with_archive, ENTRIES_NPZ, WINDOW_TOKENS_NPZ,
};
use super::raw_archive::{raw_archive, RawMember};
use super::*;

const POW_2_32: u64 = 1 << 32;
const POW_2_60: u64 = 1 << 60;
const POW_2_62: u64 = 1 << 62;
const POW_2_63: u64 = 1 << 63;
const F32_STRIDE: usize = 4;
const U32_STRIDE: usize = 4;
const TOKENS_MEMBER: &str = "0.npy";
const ENTRIES_MEMBER: &str = "entries.npy";
const BOUNDARY_FILE: &str = "boundaries/window_000.npy";
const RESIDUAL_FILE: &str = "boundary_residual.npy";

/// A v1.0 `.npy` with no data, whose descr and shape are taken as written: the
/// shape is text because `2^63` must not depend on how the test's `usize` wraps.
fn dataless_npy(descr_literal: &str, shape_literal: &str) -> Vec<u8> {
    let header = format!(
        "{{'descr': {descr_literal}, 'fortran_order': False, 'shape': {shape_literal}, }}\n"
    );
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&u16::try_from(header.len()).unwrap().to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out
}

fn entries_descr_literal() -> String {
    let fields: Vec<String> = ENTRY_DESCR_FIELDS
        .iter()
        .map(|(name, dtype)| format!("('{name}', '{dtype}')"))
        .collect();
    format!("[{}]", fields.join(", "))
}

fn tokens_store(rows: u64) -> TempDir {
    let npy = dataless_npy("'<u4'", &format!("({rows},)"));
    store_with_archive(
        WINDOW_TOKENS_NPZ,
        &raw_archive(&[RawMember::stored(TOKENS_MEMBER, npy)]),
    )
}

fn entries_store(rows: u64) -> TempDir {
    let npy = dataless_npy(&entries_descr_literal(), &format!("({rows},)"));
    store_with_archive(
        ENTRIES_NPZ,
        &raw_archive(&[RawMember::stored(ENTRIES_MEMBER, npy)]),
    )
}

fn boundary_store(rows: u64) -> TempDir {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), 1, 4, 3, &one_entry());
    let npy = dataless_npy("'<f4'", &format!("({rows},)"));
    std::fs::write(dir.path().join(BOUNDARY_FILE), npy).unwrap();
    dir
}

fn assert_shape_overflow(err: StoreLoadError, rows: u64, stride: usize) {
    match err {
        StoreLoadError::Npy {
            source:
                npy::NpyError::ShapeOverflow {
                    shape,
                    stride: reported,
                },
            ..
        } => {
            assert_eq!(shape, vec![usize::try_from(rows).unwrap()]);
            assert_eq!(reported, stride);
        }
        other => panic!("expected ShapeOverflow, got {other:?}"),
    }
}

fn assert_entries_shape_overflow(err: StoreLoadError) {
    match err {
        StoreLoadError::StructuredDtype { path, reason } => {
            assert!(path.contains(ENTRIES_MEMBER), "{path}");
            assert!(reason.contains("overflows"), "{reason}");
        }
        other => panic!("expected StructuredDtype naming the overflow, got {other:?}"),
    }
}

#[test]
fn tokens_shape_2_62_with_no_data_is_refused() {
    assert_shape_overflow(load_err(&tokens_store(POW_2_62)), POW_2_62, U32_STRIDE);
}

#[test]
fn tokens_shape_2_63_with_no_data_is_refused() {
    assert_shape_overflow(load_err(&tokens_store(POW_2_63)), POW_2_63, U32_STRIDE);
}

#[test]
fn boundary_shape_2_62_with_no_data_is_refused() {
    assert_shape_overflow(load_err(&boundary_store(POW_2_62)), POW_2_62, F32_STRIDE);
}

#[test]
fn boundary_shape_2_63_with_no_data_is_refused() {
    assert_shape_overflow(load_err(&boundary_store(POW_2_63)), POW_2_63, F32_STRIDE);
}

#[test]
fn entries_shape_2_62_with_no_data_is_refused() {
    assert_entries_shape_overflow(load_err(&entries_store(POW_2_62)));
}

#[test]
fn entries_shape_2_63_with_no_data_is_refused() {
    assert_entries_shape_overflow(load_err(&entries_store(POW_2_63)));
}

/// A shape that fits the address space still has to match the bytes present.
#[test]
fn tokens_shape_2_60_with_no_data_is_a_data_length_error() {
    let err = load_err(&tokens_store(POW_2_60));
    assert!(
        matches!(
            err,
            StoreLoadError::Npy {
                source: npy::NpyError::DataLength { .. },
                ..
            }
        ),
        "{err:?}"
    );
}

/// The residual is optional, so `load` keeps treating a malformed one as absent.
/// The store must still load rather than die on its shape.
#[test]
fn residual_shape_2_62_with_no_data_loads_as_absent() {
    let dir = boundary_store(4);
    let good_boundary = synth_f32_npy(&[0.0; 4]);
    std::fs::write(dir.path().join(BOUNDARY_FILE), good_boundary).unwrap();
    std::fs::write(
        dir.path().join(RESIDUAL_FILE),
        dataless_npy("'<f4'", &format!("({POW_2_62},)")),
    )
    .unwrap();
    let store = ApolloStore::load(dir.path()).expect("residual is optional");
    assert!(store.boundary_residual.is_none());
}

/// Two dimensions that multiply past `usize`, wrapping to a zero element count.
#[test]
fn residual_reader_refuses_a_product_that_wraps() {
    let bytes = dataless_npy("'<f4'", &format!("({POW_2_32}, {POW_2_32})"));
    let err = npy::read_f32_flat(&bytes).unwrap_err();
    assert!(
        matches!(err, npy::NpyError::ShapeOverflow { .. }),
        "{err:?}"
    );
}
