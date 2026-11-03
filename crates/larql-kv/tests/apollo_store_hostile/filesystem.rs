//! Inputs whose danger is what the path is, not what its bytes say. The sparse,
//! FIFO, device and alias cases come from the review of PR #684.

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

#[cfg(not(windows))]
use larql_kv::apollo::ArchiveError;
use larql_kv::apollo::{ApolloStore, StoreLimits, StoreLoadError};

use crate::fixtures::*;

/// The size that aborted `ZipArchive::new`: 2^44 logical bytes asked it for about
/// 78 TB. Some filesystems refuse a file this long, so 2^40 is the fallback.
#[cfg(not(windows))]
const SPARSE_LEN: u64 = 1 << 44;
const SPARSE_FALLBACK_LEN: u64 = 1 << 40;
const GIB: u64 = 1 << 30;
const ZIP64_END_FIXED_SIZE: u64 = 44;
/// A central directory record's minimum size.
const CENTRAL_RECORD_LEN: u64 = 46;
const ZIP64_END_RECORD_LEN: u64 = 56;
const ZIP64_LOCATOR_LEN: u64 = 20;
const END_RECORD_LEN: u64 = 22;
const ZIP64_TAIL_LEN: u64 = ZIP64_END_RECORD_LEN + ZIP64_LOCATOR_LEN + END_RECORD_LEN;
const VERSION_ZIP64: [u8; 2] = [45, 0];

/// The shape from the review: a 2^44 hole ending in ZIP64 records that claim as
/// many directory entries as that logical length allows.
fn zip64_tail(end_record_at: u64, directory_len: u64, records: u64, directory_at: u64) -> Vec<u8> {
    let mut tail = b"PK\x06\x06".to_vec();
    tail.extend_from_slice(&ZIP64_END_FIXED_SIZE.to_le_bytes());
    tail.extend_from_slice(&VERSION_ZIP64);
    tail.extend_from_slice(&VERSION_ZIP64);
    tail.extend_from_slice(&[0; 8]); // disks
    tail.extend_from_slice(&records.to_le_bytes());
    tail.extend_from_slice(&records.to_le_bytes());
    tail.extend_from_slice(&directory_len.to_le_bytes());
    tail.extend_from_slice(&directory_at.to_le_bytes());
    tail.extend_from_slice(b"PK\x06\x07");
    tail.extend_from_slice(&[0; 4]);
    tail.extend_from_slice(&end_record_at.to_le_bytes());
    tail.extend_from_slice(&1u32.to_le_bytes());
    tail.extend_from_slice(b"PK\x05\x06");
    tail.extend_from_slice(&[0; 4]);
    tail.extend_from_slice(&[0xFF; 12]); // record counts, directory length and offset
    tail.extend_from_slice(&[0; 2]);
    tail
}

/// A sparse archive whose only real bytes are a claiming tail. Returns its length.
fn write_sparse_zip(path: &Path, logical: u64, directory_len_is_whole_file: bool) -> u64 {
    let mut file = std::fs::File::create(path).unwrap();
    let mut logical = logical;
    if file.set_len(logical).is_err() {
        logical = SPARSE_FALLBACK_LEN;
        file.set_len(logical).unwrap();
    }
    let end_record_at = logical - ZIP64_TAIL_LEN;
    let records = end_record_at / CENTRAL_RECORD_LEN.saturating_add(1);
    let (directory_len, directory_at) = if directory_len_is_whole_file {
        (end_record_at, 0)
    } else {
        (CENTRAL_RECORD_LEN, records)
    };
    let tail = zip64_tail(end_record_at, directory_len, records, directory_at);
    file.seek(SeekFrom::Start(end_record_at)).unwrap();
    file.write_all(&tail).unwrap();
    logical
}

#[cfg(unix)]
fn replace_with_symlink(root: &Path, file: &str, target: &str) {
    let path = root.join(file);
    let _ = std::fs::remove_file(&path);
    std::os::unix::fs::symlink(target, path).unwrap();
}

#[cfg(unix)]
fn replace_with_fifo(root: &Path, file: &str) {
    let path = root.join(file);
    let _ = std::fs::remove_file(&path);
    let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    // SAFETY: `c_path` is a valid NUL-terminated string for the call's duration.
    let status = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
    assert_eq!(status, 0, "mkfifo failed");
}

fn assert_not_regular(err: &StoreLoadError, file: &str) {
    match err {
        StoreLoadError::NotRegularFile { path, .. } => {
            assert!(Path::new(path).ends_with(file), "{path}")
        }
        other => panic!("expected NotRegularFile for {file}, got {other:?}"),
    }
}

fn assert_too_large(err: &StoreLoadError, file: &str) {
    match err {
        StoreLoadError::FileTooLarge { path, .. } => {
            assert!(Path::new(path).ends_with(file), "{path}")
        }
        other => panic!("expected FileTooLarge for {file}, got {other:?}"),
    }
}

// ── Non-regular inputs ────────────────────────────────────────────────

#[cfg(unix)]
#[test]
fn symlink_to_dev_zero_is_refused_at_every_path() {
    for file in STORE_FILES {
        let dir = control_store();
        replace_with_symlink(dir.path(), file, "/dev/zero");
        assert_not_regular(&cheap_refusal(dir.path()), file);
    }
}

#[cfg(unix)]
#[test]
fn symlink_to_dev_null_is_refused_at_every_path() {
    for file in STORE_FILES {
        let dir = control_store();
        replace_with_symlink(dir.path(), file, "/dev/null");
        assert_not_regular(&cheap_refusal(dir.path()), file);
    }
}

/// With no writer, a plain open or read of a FIFO blocks forever. The watchdog
/// in `cheap_refusal` turns that into a failure.
#[cfg(unix)]
#[test]
fn fifo_is_refused_without_blocking_at_every_path() {
    for file in STORE_FILES {
        let dir = control_store();
        replace_with_fifo(dir.path(), file);
        assert_not_regular(&cheap_refusal(dir.path()), file);
    }
}

#[cfg(unix)]
#[test]
fn symlink_to_a_fifo_is_refused_without_blocking() {
    let dir = control_store();
    replace_with_fifo(dir.path(), "a_fifo");
    let target = dir.path().join("a_fifo");
    replace_with_symlink(dir.path(), BOUNDARY, target.to_str().unwrap());
    assert_not_regular(&cheap_refusal(dir.path()), BOUNDARY);
}

#[test]
fn directory_is_refused_at_every_path() {
    for file in STORE_FILES {
        let dir = control_store();
        let path = dir.path().join(file);
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert_not_regular(&cheap_refusal(dir.path()), file);
    }
}

#[cfg(unix)]
#[test]
fn symlink_loop_is_an_io_error_not_a_hang() {
    let dir = control_store();
    replace_with_symlink(dir.path(), MANIFEST, MANIFEST);
    let err = cheap_refusal(dir.path());
    assert!(matches!(err, StoreLoadError::Io { .. }), "{err:?}");
}

#[cfg(unix)]
#[test]
fn symlink_to_a_regular_file_still_loads() {
    let dir = control_store();
    let moved = dir.path().join("elsewhere.npz");
    std::fs::rename(dir.path().join(TOKENS_NPZ), &moved).unwrap();
    replace_with_symlink(dir.path(), TOKENS_NPZ, moved.to_str().unwrap());
    ApolloStore::load(dir.path()).expect("symlink to a regular file is supported");
}

#[test]
fn missing_residual_still_reads_as_absent() {
    let dir = control_store();
    std::fs::remove_file(dir.path().join(RESIDUAL)).unwrap();
    let store = ApolloStore::load(dir.path()).expect("residual is optional");
    assert!(store.boundary_residual.is_none());
}

#[cfg(unix)]
#[test]
fn dangling_residual_symlink_reads_as_absent() {
    let dir = control_store();
    replace_with_symlink(dir.path(), RESIDUAL, "no-such-file");
    let store = ApolloStore::load(dir.path()).expect("residual is optional");
    assert!(store.boundary_residual.is_none());
}

/// Behaviour change: an unreadable residual used to read as absent.
#[cfg(unix)]
#[test]
fn residual_inside_a_file_used_as_a_directory_is_an_error() {
    let dir = control_store();
    std::fs::remove_file(dir.path().join(RESIDUAL)).unwrap();
    replace_with_symlink(dir.path(), RESIDUAL, "manifest.json/nothing");
    let err = cheap_refusal(dir.path());
    assert!(matches!(err, StoreLoadError::Io { .. }), "{err:?}");
}

// ── Sparse inputs ─────────────────────────────────────────────────────

/// The review's B1. On unfixed code this is a SIGABRT inside `ZipArchive::new`.
#[test]
fn sparse_zip_of_2_44_is_refused_by_its_logical_length() {
    #[cfg(windows)]
    const LOGICAL: u64 = GIB * 2; // NTFS allocates a file not marked sparse.
    #[cfg(not(windows))]
    const LOGICAL: u64 = SPARSE_LEN;
    for file in [TOKENS_NPZ, ENTRIES_NPZ] {
        let dir = control_store();
        write_sparse_zip(&dir.path().join(file), LOGICAL, false);
        assert_too_large(&cheap_refusal(dir.path()), file);
    }
}

/// With the size limit out of the way, the claim itself must not allocate. This
/// is what proves the record count is never believed: the limit is a separate
/// protection, and this path holds without it.
#[cfg(not(windows))]
#[test]
fn sparse_zip_claiming_records_allocates_nothing_even_with_no_limits() {
    let no_limits = StoreLimits {
        max_archive_bytes: u64::MAX,
        max_store_bytes: u64::MAX,
        ..StoreLimits::default()
    };
    for directory_len_is_whole_file in [false, true] {
        let dir = control_store();
        write_sparse_zip(
            &dir.path().join(ENTRIES_NPZ),
            SPARSE_LEN,
            directory_len_is_whole_file,
        );
        let measured = load_watched(dir.path(), no_limits);
        assert!(measured.peak_bytes < REFUSAL_PEAK_BYTES);
        match measured.value.expect_err("nothing there") {
            StoreLoadError::Archive {
                source: ArchiveError::NoEndRecord,
                ..
            } => {}
            other => panic!("expected NoEndRecord, got {other:?}"),
        }
    }
}

#[test]
fn sparse_boundary_is_refused_by_its_logical_length() {
    let dir = control_store();
    make_sparse(dir.path(), BOUNDARY, &f32_npy(0), GIB);
    assert_too_large(&cheap_refusal(dir.path()), BOUNDARY);
}

#[test]
fn sparse_manifest_is_refused_by_its_logical_length() {
    let dir = control_store();
    make_sparse(dir.path(), MANIFEST, b"{", GIB);
    assert_too_large(&cheap_refusal(dir.path()), MANIFEST);
}

/// An oversized residual is a failure, not an absent residual.
#[test]
fn sparse_residual_is_refused_rather_than_read_as_absent() {
    let dir = control_store();
    make_sparse(dir.path(), RESIDUAL, &f32_npy(0), GIB);
    assert_too_large(&cheap_refusal(dir.path()), RESIDUAL);
}

/// The residual risk, pinned: a hole inside the limits really is read. The file
/// is a few KiB on disk and the load costs about its logical length in memory.
/// This is why the limits are not a claim of safety against memory exhaustion.
#[test]
fn sparse_member_within_the_limits_costs_its_logical_length_and_no_more() {
    const HOLE: u64 = 64 << 20;
    const OVERHEAD_ALLOWANCE: usize = 1 << 20;
    const CHUNK: usize = 1 << 20;
    let dir = control_store();
    // A stored member of `HOLE` zero bytes, which then fails to parse as `.npy`.
    let crc = {
        let mut hasher = crc32fast::Hasher::new();
        let chunk = vec![0u8; CHUNK];
        for _ in 0..HOLE / CHUNK as u64 {
            hasher.update(&chunk);
        }
        hasher.finalize()
    };
    let record = RawRecord {
        name: b"0.npy",
        crc,
        member_len: u32::try_from(HOLE).unwrap(),
        local_offset: 0,
    };
    let local = local_record(&record);
    let central = central_record(&record);
    let directory_at = local.len() as u64 + HOLE;
    let mut file = std::fs::File::create(dir.path().join(TOKENS_NPZ)).unwrap();
    file.write_all(&local).unwrap();
    file.seek(SeekFrom::Start(directory_at)).unwrap();
    file.write_all(&central).unwrap();
    file.write_all(&end_records(1, central.len() as u64, directory_at))
        .unwrap();
    drop(file);

    let measured = load_watched(dir.path(), StoreLimits::default());
    let err = measured.value.expect_err("zeros are not an .npy");
    assert!(matches!(err, StoreLoadError::Npy { .. }), "{err:?}");
    assert!(
        measured.peak_bytes >= HOLE as usize,
        "the read happened: {}",
        measured.peak_bytes
    );
    assert!(
        measured.peak_bytes <= HOLE as usize + OVERHEAD_ALLOWANCE,
        "more than the member cost: {}",
        measured.peak_bytes
    );
}

// ── Limits ────────────────────────────────────────────────────────────

fn control_file_len(file: &str) -> u64 {
    let dir = control_store();
    std::fs::metadata(dir.path().join(file)).unwrap().len()
}

fn limit_only_for(file: &str, limit: u64) -> StoreLimits {
    let mut limits = StoreLimits::default();
    match file {
        MANIFEST => limits.max_manifest_bytes = limit,
        BOUNDARY | RESIDUAL => limits.max_array_file_bytes = limit,
        _ => limits.max_archive_bytes = limit,
    }
    limits
}

#[test]
fn each_file_limit_is_inclusive_and_one_byte_under_refuses() {
    // The two archives share a limit, so only the longer one sits exactly at it.
    let longer_archive = if control_file_len(TOKENS_NPZ) > control_file_len(ENTRIES_NPZ) {
        TOKENS_NPZ
    } else {
        ENTRIES_NPZ
    };
    for file in [MANIFEST, BOUNDARY, longer_archive] {
        let len = control_file_len(file);
        let dir = control_store();
        ApolloStore::load_with_limits(dir.path(), &limit_only_for(file, len))
            .unwrap_or_else(|e| panic!("{file} at its limit: {e}"));
        let err = ApolloStore::load_with_limits(dir.path(), &limit_only_for(file, len - 1))
            .expect_err("one byte over");
        match err {
            StoreLoadError::FileTooLarge {
                path,
                len: reported,
                limit,
            } => {
                assert!(Path::new(&path).ends_with(file), "{path}");
                assert_eq!((reported, limit), (len, len - 1));
            }
            other => panic!("expected FileTooLarge for {file}, got {other:?}"),
        }
    }
}

#[test]
fn store_total_limit_is_inclusive_and_one_byte_under_refuses() {
    let dir = control_store();
    let total: u64 = STORE_FILES
        .iter()
        .map(|file| std::fs::metadata(dir.path().join(file)).unwrap().len())
        .sum();
    let at = |max_store_bytes| StoreLimits {
        max_store_bytes,
        ..StoreLimits::default()
    };
    ApolloStore::load_with_limits(dir.path(), &at(total)).expect("at the limit");
    let err = ApolloStore::load_with_limits(dir.path(), &at(total - 1)).expect_err("one over");
    assert!(
        matches!(err, StoreLoadError::StoreTooLarge { .. }),
        "{err:?}"
    );
}

/// The review's 1000 symlinks to one 1 MiB boundary file: each alias is charged.
#[cfg(unix)]
#[test]
fn aliases_of_one_boundary_are_each_charged_to_the_store_limit() {
    const ALIASES: usize = 1000;
    const SHARED_VALUES: usize = (1 << 20) / size_of::<f32>();
    const LIMIT_IN_ALIASES: u64 = 16;
    let dir = control_store();
    let shared = f32_npy(SHARED_VALUES);
    let limits = StoreLimits {
        max_store_bytes: shared.len() as u64 * LIMIT_IN_ALIASES,
        ..StoreLimits::default()
    };
    std::fs::write(dir.path().join("shared.npy"), &shared).unwrap();
    std::fs::write(dir.path().join(MANIFEST), manifest(ALIASES, 1)).unwrap();
    std::fs::remove_file(dir.path().join(BOUNDARY)).unwrap();
    for i in 0..ALIASES {
        let alias = format!("boundaries/window_{i:03}.npy");
        replace_with_symlink(dir.path(), &alias, "../shared.npy");
    }
    let measured = load_watched(dir.path(), limits);
    let err = measured.value.expect_err("aliases add up");
    assert!(
        matches!(err, StoreLoadError::StoreTooLarge { .. }),
        "{err:?}"
    );
    let bound = 2 * limits.max_store_bytes as usize;
    assert!(measured.peak_bytes < bound, "peak {}", measured.peak_bytes);
}
