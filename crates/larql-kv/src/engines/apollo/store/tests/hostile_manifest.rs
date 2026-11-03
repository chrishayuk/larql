//! `manifest.num_windows` must not size an allocation: only the boundary files
//! that actually exist may.

use super::archive_errors::{load_err, one_entry};
use super::*;

const POW_2_50: usize = 1 << 50;
const POW_2_62: usize = 1 << 62;
/// `write_minimal_store` wrote `window_000.npy`, so this is the first gap.
const FIRST_MISSING_BOUNDARY: &str = "window_001.npy";

fn store_claiming_windows(num_windows: usize) -> TempDir {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), 1, 4, 3, &one_entry());
    let path = dir.path().join("manifest.json");
    let mut manifest: StoreManifest =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest.num_windows = num_windows;
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    dir
}

fn assert_first_gap_is_reported(num_windows: usize) {
    match load_err(&store_claiming_windows(num_windows)) {
        StoreLoadError::Io { path, source } => {
            assert!(path.contains(FIRST_MISSING_BOUNDARY), "{path}");
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        }
        other => panic!("expected Io naming the missing boundary, got {other:?}"),
    }
}

#[test]
fn num_windows_2_50_reports_the_first_missing_boundary() {
    assert_first_gap_is_reported(POW_2_50);
}

#[test]
fn num_windows_2_62_reports_the_first_missing_boundary() {
    assert_first_gap_is_reported(POW_2_62);
}
