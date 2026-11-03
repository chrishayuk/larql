//! Allocation failure, simulated by a thread-local allocator that returns null
//! for one size. This is the real failure path but a simulated cause: `ulimit -v`
//! is not settable on the development machine. On unfixed code each of these is a
//! process abort, since `Vec::with_capacity` does not return.

use larql_kv::apollo::npy::NpyError;
use larql_kv::apollo::{ApolloStore, StoreLoadError};

use crate::fixtures::*;
use crate::instrument::watched;

/// Bigger than anything else a load allocates here, so the size picks one buffer.
const VALUES: usize = 1 << 18;
const ENTRY_ROWS: usize = 1 << 16;

fn load_refusing(root: &std::path::Path, size: usize) -> StoreLoadError {
    let root = root.to_path_buf();
    watched(DEADLINE, Some(size..=size), move || {
        ApolloStore::load(&root)
    })
    .value
    .expect_err("allocation was refused")
}

fn assert_refused_bytes(err: &StoreLoadError, file: &str, size: usize) {
    match err {
        StoreLoadError::AllocationRefused { path, bytes } => {
            assert!(path.contains(file), "{path}");
            assert_eq!(*bytes, size as u64);
        }
        other => panic!("expected AllocationRefused for {file}, got {other:?}"),
    }
}

fn assert_npy_refused(err: &StoreLoadError, size: usize) {
    match err {
        StoreLoadError::Npy {
            source: NpyError::AllocationRefused { bytes },
            ..
        } => assert_eq!(*bytes, size as u64),
        other => panic!("expected Npy allocation refusal, got {other:?}"),
    }
}

#[test]
fn refusing_the_manifest_buffer_is_typed() {
    let dir = control_store();
    let body = manifest(1, 1);
    let padded = [body.as_slice(), &vec![b' '; VALUES]].concat();
    std::fs::write(dir.path().join(MANIFEST), &padded).unwrap();
    let err = load_refusing(dir.path(), padded.len());
    assert_refused_bytes(&err, MANIFEST, padded.len());
}

#[test]
fn refusing_the_boundary_file_buffer_is_typed() {
    let dir = control_store();
    let bytes = f32_npy(VALUES);
    std::fs::write(dir.path().join(BOUNDARY), &bytes).unwrap();
    let err = load_refusing(dir.path(), bytes.len());
    assert_refused_bytes(&err, BOUNDARY, bytes.len());
}

#[test]
fn refusing_the_decoded_boundary_is_typed() {
    let dir = control_store();
    std::fs::write(dir.path().join(BOUNDARY), f32_npy(VALUES)).unwrap();
    let decoded = VALUES * size_of::<f32>();
    assert_npy_refused(&load_refusing(dir.path(), decoded), decoded);
}

/// An allocation failure must not read as an absent residual.
#[test]
fn refusing_the_decoded_residual_is_an_error_not_absence() {
    let dir = control_store();
    std::fs::write(dir.path().join(RESIDUAL), f32_npy(VALUES)).unwrap();
    let decoded = VALUES * size_of::<f32>();
    assert_npy_refused(&load_refusing(dir.path(), decoded), decoded);
}

#[test]
fn refusing_the_residual_file_buffer_is_an_error_not_absence() {
    let dir = control_store();
    let bytes = f32_npy(VALUES);
    std::fs::write(dir.path().join(RESIDUAL), &bytes).unwrap();
    let err = load_refusing(dir.path(), bytes.len());
    assert_refused_bytes(&err, RESIDUAL, bytes.len());
}

#[test]
fn refusing_a_token_member_buffer_is_typed() {
    let dir = control_store();
    let tokens: Vec<u32> = vec![0; VALUES];
    let member = u32_npy(&tokens);
    std::fs::write(dir.path().join(TOKENS_NPZ), npz(&[("0.npy", &member)])).unwrap();
    let err = load_refusing(dir.path(), member.len());
    assert_refused_bytes(&err, "0.npy", member.len());
}

#[test]
fn refusing_the_entries_decode_is_typed() {
    const IN_MEMORY_ROW: usize = 16;
    let dir = control_store();
    let member = entries_npy(ENTRY_ROWS);
    std::fs::write(
        dir.path().join(ENTRIES_NPZ),
        npz(&[("entries.npy", &member)]),
    )
    .unwrap();
    std::fs::write(dir.path().join(MANIFEST), manifest(1, ENTRY_ROWS)).unwrap();
    let decoded = ENTRY_ROWS * IN_MEMORY_ROW;
    assert_npy_refused(&load_refusing(dir.path(), decoded), decoded);
}

/// The ZIP tail buffer is the first thing the archive reader allocates. An archive
/// of 64 KiB or more makes it exactly the end-record search distance.
#[test]
fn refusing_the_archive_tail_buffer_is_typed() {
    const TAIL_LEN: usize = 22 + u16::MAX as usize;
    let dir = control_store();
    let tokens: Vec<u32> = vec![0; VALUES];
    std::fs::write(
        dir.path().join(TOKENS_NPZ),
        npz(&[("0.npy", &u32_npy(&tokens))]),
    )
    .unwrap();
    let err = load_refusing(dir.path(), TAIL_LEN);
    assert_refused_bytes(&err, TOKENS_NPZ, TAIL_LEN);
}
