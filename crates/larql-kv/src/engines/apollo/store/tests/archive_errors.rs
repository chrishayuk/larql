//! Error wrapping in the two `.npz` readers (`load_window_tokens`,
//! `load_entries`): each failure must surface as the variant that names
//! the right file, never as a panic or a silently empty store.

use super::*;

const WINDOWS: usize = 1;
const HIDDEN: usize = 4;
const WINDOW_SIZE: usize = 3;
pub(super) const WINDOW_TOKENS_NPZ: &str = "window_token_lists.npz";
pub(super) const ENTRIES_NPZ: &str = "entries.npz";
/// Zip local-file-header signature (`PK\x03\x04`).
const LOCAL_HEADER_SIGNATURE: &[u8] = b"PK\x03\x04";
/// `.npy` magic prefix, used to find a stored member's payload in the archive.
const NPY_MAGIC: &[u8] = b"\x93NUMPY";

pub(super) fn one_entry() -> Vec<VecInjectEntry> {
    vec![VecInjectEntry {
        token_id: 7,
        coefficient: 1.0,
        window_id: 0,
        position_in_window: 0,
        fact_id: 0,
    }]
}

/// A one-window store whose `file` archive is then replaced by `bytes`.
pub(super) fn store_with_archive(file: &str, bytes: &[u8]) -> TempDir {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), WINDOWS, HIDDEN, WINDOW_SIZE, &one_entry());
    std::fs::write(dir.path().join(file), bytes).unwrap();
    dir
}

/// A one-window store with `file` removed.
fn store_without(file: &str) -> TempDir {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), WINDOWS, HIDDEN, WINDOW_SIZE, &one_entry());
    std::fs::remove_file(dir.path().join(file)).unwrap();
    dir
}

fn position_of(haystack: &[u8], needle: &[u8]) -> usize {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("needle present in archive")
}

/// A stored archive whose single member's payload has its last byte
/// flipped: the central directory still parses, but the CRC check fails
/// when the member is read to the end.
fn crc_corrupted_npz(name: &str, payload: Vec<u8>) -> Vec<u8> {
    let len = payload.len();
    let mut zip = synth_npz(vec![(name.to_string(), payload)]);
    let last = position_of(&zip, NPY_MAGIC) + len - 1;
    zip[last] ^= 0xFF;
    zip
}

/// A stored archive whose member's local header signature is clobbered:
/// the central directory still lists the member, but opening it fails.
fn local_header_corrupted_npz(name: &str, payload: Vec<u8>) -> Vec<u8> {
    let mut zip = synth_npz(vec![(name.to_string(), payload)]);
    let at = position_of(&zip, LOCAL_HEADER_SIGNATURE);
    zip[at] = b'X';
    zip
}

pub(super) fn load_err(dir: &TempDir) -> StoreLoadError {
    ApolloStore::load(dir.path()).unwrap_err()
}

pub(super) fn assert_archive_error(err: &StoreLoadError, expected: &ArchiveError) {
    match err {
        StoreLoadError::Archive { source, .. } => assert_eq!(source, expected),
        other => panic!("expected Archive({expected:?}), got {other:?}"),
    }
}

pub(super) fn assert_names(err: &StoreLoadError, file: &str) {
    assert!(err.to_string().contains(file), "{file} not named in: {err}");
}

// ── window_token_lists.npz ────────────────────────────────────────────

#[test]
fn missing_window_token_archive_is_an_io_error_naming_it() {
    let err = load_err(&store_without(WINDOW_TOKENS_NPZ));
    assert!(matches!(err, StoreLoadError::Io { .. }), "{err:?}");
    assert_names(&err, WINDOW_TOKENS_NPZ);
}

#[test]
fn window_token_member_that_is_not_npy_is_an_npy_error() {
    let npz = synth_npz(vec![("0.npy".to_string(), b"not npy".to_vec())]);
    let err = load_err(&store_with_archive(WINDOW_TOKENS_NPZ, &npz));
    assert!(matches!(err, StoreLoadError::Npy { .. }), "{err:?}");
    assert_names(&err, "0.npy");
}

#[test]
fn window_token_member_failing_its_crc_is_a_crc_error() {
    let npz = crc_corrupted_npz("0.npy", synth_u32_npy(&[1, 2, 3]));
    let err = load_err(&store_with_archive(WINDOW_TOKENS_NPZ, &npz));
    assert_archive_error(&err, &ArchiveError::CrcMismatch);
    assert_names(&err, "0.npy");
}

#[test]
fn window_token_member_with_a_broken_local_header_is_an_archive_error() {
    let npz = local_header_corrupted_npz("0.npy", synth_u32_npy(&[1, 2, 3]));
    let err = load_err(&store_with_archive(WINDOW_TOKENS_NPZ, &npz));
    assert_archive_error(&err, &ArchiveError::BadLocalHeader("bad signature"));
    assert_names(&err, WINDOW_TOKENS_NPZ);
}

// ── entries.npz ───────────────────────────────────────────────────────

#[test]
fn missing_entries_archive_is_an_io_error_naming_it() {
    let err = load_err(&store_without(ENTRIES_NPZ));
    assert!(matches!(err, StoreLoadError::Io { .. }), "{err:?}");
    assert_names(&err, ENTRIES_NPZ);
}

#[test]
fn entries_archive_that_is_not_a_zip_is_an_archive_error() {
    let err = load_err(&store_with_archive(ENTRIES_NPZ, b"NOT_A_ZIP"));
    assert_archive_error(&err, &ArchiveError::NoEndRecord);
    assert_names(&err, ENTRIES_NPZ);
}

#[test]
fn entries_archive_without_an_entries_member_is_missing_file() {
    let npz = synth_npz(vec![("other.npy".to_string(), synth_u32_npy(&[1]))]);
    let err = load_err(&store_with_archive(ENTRIES_NPZ, &npz));
    assert!(matches!(err, StoreLoadError::MissingFile(_)), "{err:?}");
}

#[test]
fn entries_member_failing_its_crc_is_a_crc_error() {
    let npz = crc_corrupted_npz("entries.npy", synth_structured_entries_npy(&one_entry()));
    let err = load_err(&store_with_archive(ENTRIES_NPZ, &npz));
    assert_archive_error(&err, &ArchiveError::CrcMismatch);
    assert_names(&err, "entries.npy");
}

#[test]
fn entries_member_with_a_broken_local_header_is_an_archive_error() {
    let npz = local_header_corrupted_npz("entries.npy", synth_structured_entries_npy(&one_entry()));
    let err = load_err(&store_with_archive(ENTRIES_NPZ, &npz));
    assert_archive_error(&err, &ArchiveError::BadLocalHeader("bad signature"));
    assert_names(&err, ENTRIES_NPZ);
}

#[test]
fn entries_member_with_a_foreign_dtype_is_a_structured_dtype_error() {
    let npz = synth_npz(vec![("entries.npy".to_string(), synth_u32_npy(&[1, 2]))]);
    let err = load_err(&store_with_archive(ENTRIES_NPZ, &npz));
    assert!(
        matches!(err, StoreLoadError::StructuredDtype { .. }),
        "{err:?}"
    );
    assert_names(&err, "entries.npy");
}
