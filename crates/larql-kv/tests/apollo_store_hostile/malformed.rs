//! Oversized and malformed contents at their real paths, through the public API.

use larql_kv::apollo::npy::NpyError;
use larql_kv::apollo::{ApolloStore, StoreLimits, StoreLoadError};

use crate::fixtures::*;

const OVER_ONE_MIB: usize = (1 << 20) + 1;
const ENTRY_ROW_BYTES: usize = 14;
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

/// Everything that isn't the writer's exact layout is one refusal, whatever it would
/// have meant to `numpy`. The list is examples of the class, not what the check looks for.
#[test]
fn any_header_not_in_the_writers_form_is_refused() {
    const AFTER: &str = "'fortran_order': False, 'shape': (1,), }\n";
    for header in [
        "'descr':".to_string(),
        // Keys repeated, spelt differently, hidden or reordered.
        format!("{{'descr': '<u4', 'descr': '<f4', {AFTER}"),
        format!("{{'descr': '<f4', 'descr' : '<u4', {AFTER}"),
        format!("{{'descr': '<f4', 'de' 'scr': '<u4', {AFTER}"),
        format!("{{'descr': '<f4', 'd\\x65scr': '<u4', {AFTER}"),
        format!("{{'descr': '<f4', u'descr': '<u4', {AFTER}"),
        format!("{{'descr': '<f4', \"descr\": '<u4', {AFTER}"),
        format!("{{'descr': '<f4', # 'shape': (9,),\n {AFTER}"),
        "{'fortran_order': False, 'descr': '<f4', 'shape': (1,), }\n".to_string(),
        "{'shape': (1,), 'descr': '<f4', 'fortran_order': False, }\n".to_string(),
        // Spacing the writer doesn't produce.
        "{ 'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\n".to_string(),
        "{'descr': '<f4','fortran_order': False, 'shape': (1,), }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1,)}\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\n\n".to_string(),
        "\n{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }\t\n".to_string(),
        // Fortran order, an extra key, and shapes that aren't a tuple's `repr`.
        "{'descr': '<f4', 'fortran_order': True, 'shape': (1,), }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': maybe, 'shape': (1,), }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1,), 'x': 1, }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': )(, }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (é,), }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (18446744073709551616,), }\n"
            .to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1), }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1, ), }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (01,), }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (+1,), }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1L,), }\n".to_string(),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (1,2), }\n".to_string(),
    ] {
        let mut bytes = header_with(&header);
        bytes.extend_from_slice(&[0; size_of::<f32>()]);
        let err = boundary_err(&bytes);
        assert!(
            // Text between the anchors is the descr, which is then compared whole.
            matches!(
                err,
                NpyError::ParseField {
                    field: "header",
                    ..
                } | NpyError::DtypeMismatch { .. }
            ),
            "{header:?}: {err:?}"
        );
    }
}

/// Padding is spaces and one newline, of any length, as numpy's versions have varied it.
#[test]
fn writers_form_loads_with_any_padding() {
    const NO_PADDING: usize = 0;
    const OLD_ALIGNMENT_PADDING: usize = 7;
    const GROWTH_PADDING: usize = 60;
    for padding in [NO_PADDING, OLD_ALIGNMENT_PADDING, GROWTH_PADDING] {
        let dict = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': (1,), }}{}\n",
            " ".repeat(padding)
        );
        let dir = control_store();
        let mut bytes = header_with(&dict);
        bytes.extend_from_slice(&[0; size_of::<f32>()]);
        std::fs::write(dir.path().join(BOUNDARY), bytes).unwrap();
        ApolloStore::load(dir.path()).expect("the writer's form");
    }
}

/// A subarray of no elements leaves the descr's fields looking right, and shifts every
/// row for `numpy`. Any other entries dtype text is refused, compared whole.
#[test]
fn entries_descr_other_than_the_canonical_text_is_refused() {
    const CANONICAL: &str = "[('token_id', '<u4'), ('coefficient', '<f4'), ('window_id', '<u2'), \
                             ('position_in_window', '<u2'), ('fact_id', '<u2')]";
    for descr in [
        CANONICAL.replace("('fact_id', '<u2')", "('fact_id', '<u2', (0,))"),
        CANONICAL.replace("('window_id', '<u2')", "('window_id', '<u2', (1,))"),
        CANONICAL.replace("'<u4'", "'>u4'"),
        CANONICAL.replace("('fact_id', '<u2')", "('fact_id', '<u2'), ('', '|V0')"),
        CANONICAL.replace(", ", ","),
    ] {
        let dir = control_store();
        let member = npy(&descr, "(1,)", &[0; ENTRY_ROW_BYTES]);
        std::fs::write(
            dir.path().join(ENTRIES_NPZ),
            npz(&[("entries.npy", &member)]),
        )
        .unwrap();
        let err = cheap_refusal(dir.path());
        assert!(
            matches!(err, StoreLoadError::StructuredDtype { .. }),
            "{descr}: {err:?}"
        );
    }
}

#[test]
fn foreign_dtype_is_refused() {
    let foreign = "{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }\n";
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
    const DIMS: usize = 20_000;
    let dir = control_store();
    let shape = format!("({})", vec!["1"; DIMS].join(", "));
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
