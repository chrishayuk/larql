use super::super::disk::free_bytes;

#[test]
fn a_real_directory_reports_free_space() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(free_bytes(tmp.path()).is_some_and(|b| b > 0));
}

#[test]
fn a_missing_path_reports_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    assert_eq!(free_bytes(&tmp.path().join("missing")), None);
}

#[cfg(unix)]
#[test]
fn a_path_with_an_interior_nul_reports_nothing() {
    use std::os::unix::ffi::OsStrExt;
    let bad = std::path::Path::new(std::ffi::OsStr::from_bytes(b"a\0b"));
    assert_eq!(free_bytes(bad), None);
}
