//! `manifest.num_windows` must not size an allocation: only the boundary files
//! that actually exist may.

use super::archive_errors::{load_err, one_entry};
use super::*;

const WINDOWS: usize = 1;
const HIDDEN: usize = 4;
const WINDOW_SIZE: usize = 3;
const MANIFEST_FILE: &str = "manifest.json";
const POW_2_50: usize = 1 << 50;
const POW_2_62: usize = 1 << 62;
/// `write_minimal_store` wrote `window_000.npy`, so this is the first gap.
const FIRST_MISSING_BOUNDARY: &str = "window_001.npy";

fn store_claiming_windows(num_windows: usize) -> TempDir {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), WINDOWS, HIDDEN, WINDOW_SIZE, &one_entry());
    let path = dir.path().join(MANIFEST_FILE);
    let mut manifest: StoreManifest =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest.num_windows = num_windows;
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    dir
}

fn assert_too_many_windows(num_windows: usize) {
    match load_err(&store_claiming_windows(num_windows)) {
        StoreLoadError::TooManyRecords {
            path,
            records,
            limit,
        } => {
            assert!(path.contains(MANIFEST_FILE), "{path}");
            assert_eq!(records, num_windows as u64);
            assert_eq!(limit, StoreLimits::default().max_records);
        }
        other => panic!("expected TooManyRecords, got {other:?}"),
    }
}

#[test]
fn num_windows_2_50_is_refused_before_any_boundary_is_opened() {
    assert_too_many_windows(POW_2_50);
}

#[test]
fn num_windows_2_62_is_refused_before_any_boundary_is_opened() {
    assert_too_many_windows(POW_2_62);
}

/// Within the limit, a claim beyond the files present stops at the first missing one.
#[test]
fn num_windows_at_the_limit_reports_the_first_missing_boundary() {
    let limit = StoreLimits::default().max_records;
    match load_err(&store_claiming_windows(limit as usize)) {
        StoreLoadError::Io { path, source } => {
            assert!(path.contains(FIRST_MISSING_BOUNDARY), "{path}");
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        }
        other => panic!("expected Io naming the missing boundary, got {other:?}"),
    }
}
