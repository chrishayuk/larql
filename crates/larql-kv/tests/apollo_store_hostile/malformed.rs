//! Oversized and malformed contents at their real paths, through the public API.

use larql_kv::apollo::npy::NpyError;
use larql_kv::apollo::{ApolloStore, StoreLimits, StoreLoadError};

use crate::fixtures::*;

const OVER_ONE_MIB: usize = (1 << 20) + 1;
const HEADER_PAST_END: &[u8] = b"\x93NUMPY\x01\x00\xFF\xFFx";

fn boundary_err(bytes: &[u8]) -> NpyError {
    let dir = control_store();
    std::fs::write(dir.path().join(BOUNDARY), bytes).unwrap();
    match cheap_refusal(dir.path()) {
        StoreLoadError::Npy { source, .. } => source,
        other => panic!("expected Npy, got {other:?}"),
    }
}

fn header_with(dict: &str) -> Vec<u8> {
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&u16::try_from(dict.len()).unwrap().to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out
}

#[test]
fn oversized_dense_manifest_is_refused_before_parsing() {
    let dir = control_store();
    std::fs::write(dir.path().join(MANIFEST), vec![b' '; OVER_ONE_MIB]).unwrap();
    let err = cheap_refusal(dir.path());
    assert!(
        matches!(err, StoreLoadError::FileTooLarge { .. }),
        "{err:?}"
    );
}

#[test]
fn deeply_nested_manifest_is_a_json_error() {
    const DEPTH: usize = 200_000;
    let dir = control_store();
    std::fs::write(dir.path().join(MANIFEST), vec![b'['; DEPTH]).unwrap();
    let err = load_watched(dir.path(), StoreLimits::default())
        .value
        .expect_err("not a manifest");
    assert!(matches!(err, StoreLoadError::Json(_)), "{err:?}");
}

#[test]
fn truncated_npy_header_is_refused() {
    assert!(matches!(
        boundary_err(b"\x93NUM"),
        NpyError::TruncatedHeader
    ));
    assert!(matches!(
        boundary_err(HEADER_PAST_END),
        NpyError::TruncatedHeader
    ));
}

#[test]
fn bad_magic_and_version_are_refused() {
    assert!(matches!(boundary_err(b"NOTANPYFILE"), NpyError::BadMagic));
    assert!(matches!(
        boundary_err(b"\x93NUMPY\x02\x00\x00\x00"),
        NpyError::UnsupportedVersion(2, 0)
    ));
}

#[test]
fn non_utf8_header_is_refused() {
    let mut bytes = b"\x93NUMPY\x01\x00\x01\x00".to_vec();
    bytes.push(0xFF);
    assert!(matches!(boundary_err(&bytes), NpyError::InvalidUtf8(_)));
}

#[test]
fn header_fields_that_do_not_parse_are_refused() {
    for dict in [
        "'descr':",
        "{'descr': '<f4', 'fortran_order': False, 'shape': )(}\n",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (é,)}\n",
        "{'descr': '<f4', 'fortran_order': False, 'shape': (18446744073709551616,)}\n",
        "{'descr': '<f4', 'fortran_order': maybe, 'shape': (1,)}\n",
    ] {
        let err = boundary_err(&header_with(dict));
        assert!(
            matches!(err, NpyError::ParseField { .. }),
            "{dict}: {err:?}"
        );
    }
}

#[test]
fn fortran_order_and_foreign_dtype_are_refused() {
    let fortran = "{'descr': '<f4', 'fortran_order': True, 'shape': (1,)}\n";
    assert!(matches!(
        boundary_err(&header_with(fortran)),
        NpyError::FortranOrder
    ));
    let foreign = "{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}\n";
    assert!(matches!(
        boundary_err(&header_with(foreign)),
        NpyError::DtypeMismatch { .. }
    ));
}

#[test]
fn boundary_shape_disagreeing_with_its_data_is_refused() {
    assert!(matches!(
        boundary_err(&npy("'<f4'", "(9,)", &[0; 4])),
        NpyError::DataLength { .. }
    ));
}

#[test]
fn zero_dimensional_boundary_is_refused() {
    assert!(matches!(
        boundary_err(&npy("'<f4'", "()", &[0; 4])),
        NpyError::ParseField { field: "shape", .. }
    ));
}

/// The header is at most 64 KiB, so the most dimensions it can list cost little.
#[test]
fn residual_with_tens_of_thousands_of_dimensions_is_cheap() {
    const DIMS: usize = 32_000;
    let dir = control_store();
    let shape = format!("({})", "1,".repeat(DIMS));
    std::fs::write(dir.path().join(RESIDUAL), npy("'<f4'", &shape, &[0; 4])).unwrap();
    let measured = load_watched(dir.path(), StoreLimits::default());
    assert!(measured.peak_bytes < REFUSAL_PEAK_BYTES);
    ApolloStore::load(dir.path()).expect("legal, if strange");
}

/// Existing behaviour, kept: a residual whose contents don't parse reads as absent.
#[test]
fn malformed_residual_contents_still_read_as_absent() {
    let dir = control_store();
    std::fs::write(dir.path().join(RESIDUAL), b"bogus").unwrap();
    let store = ApolloStore::load(dir.path()).expect("residual is optional");
    assert!(store.boundary_residual.is_none());
}
