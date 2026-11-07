use std::ffi::OsString;

use super::super::super::serve_cmd::server_file_name;
use super::super::report::Status;
use super::super::server::{checks, find_on_path, locate, parse_version, ServerLocation};
use super::{healthy, home, only};

#[test]
fn version_is_read_from_the_clap_version_line() {
    assert_eq!(parse_version("larql-server 0.2.1\n"), Some("0.2.1".into()));
    assert_eq!(
        parse_version("  larql-server 1.0.0 extra"),
        Some("1.0.0".into())
    );
    assert_eq!(parse_version("something-else 0.2.1"), None);
    assert_eq!(parse_version("larql-server"), None);
    assert_eq!(parse_version(""), None);
}

#[test]
fn path_lookup_takes_the_first_directory_holding_the_file() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let c = tempfile::tempdir().unwrap();
    std::fs::write(b.path().join(server_file_name()), b"").unwrap();
    std::fs::write(c.path().join(server_file_name()), b"").unwrap();
    let path: OsString = std::env::join_paths([a.path(), b.path(), c.path()]).unwrap();
    assert_eq!(
        find_on_path(&server_file_name(), Some(&path)),
        Some(b.path().join(server_file_name()))
    );
    assert_eq!(find_on_path(&server_file_name(), None), None);
}

#[test]
fn a_directory_named_like_the_server_is_not_the_server() {
    let a = tempfile::tempdir().unwrap();
    std::fs::create_dir(a.path().join(server_file_name())).unwrap();
    let path = std::env::join_paths([a.path()]).unwrap();
    assert_eq!(find_on_path(&server_file_name(), Some(&path)), None);
}

#[test]
fn locate_prefers_the_sibling_then_path_then_reports_missing() {
    let bin = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    let exe = bin.path().join("larql");
    let path = std::env::join_paths([elsewhere.path()]).unwrap();

    assert_eq!(locate(Some(&exe), Some(&path)), ServerLocation::Missing);

    let on_path = elsewhere.path().join(server_file_name());
    std::fs::write(&on_path, b"").unwrap();
    assert_eq!(
        locate(Some(&exe), Some(&path)),
        ServerLocation::OnPath(on_path)
    );

    let sibling = bin.path().join(server_file_name());
    std::fs::write(&sibling, b"").unwrap();
    assert_eq!(
        locate(Some(&exe), Some(&path)),
        ServerLocation::Sibling(sibling)
    );
}

#[test]
fn a_matching_sibling_passes() {
    let c = only(checks(&healthy().server));
    assert_eq!(c.status, Status::Ok);
    assert!(
        c.summary.starts_with("0.2.1 (beside larql: ~"),
        "{}",
        c.summary
    );
}

#[test]
fn a_missing_server_fails() {
    let mut s = healthy().server;
    s.location = ServerLocation::Missing;
    s.version = None;
    let c = only(checks(&s));
    assert_eq!(c.status, Status::Fail);
    assert!(c.hint.unwrap().contains("larql serve"));
}

#[test]
fn a_version_mismatch_warns() {
    let mut s = healthy().server;
    s.location = ServerLocation::OnPath(home().join("bin/larql-server"));
    s.version = Some(Ok("0.1.0".into()));
    let c = only(checks(&s));
    assert_eq!(c.status, Status::Warn);
    assert!(c.summary.contains("on PATH"));
    assert!(c.summary.contains("but larql is 0.2.1"));
}

#[test]
fn a_failed_version_query_warns() {
    let mut s = healthy().server;
    s.version = Some(Err("exited with 1".into()));
    let c = only(checks(&s));
    assert_eq!(c.status, Status::Warn);
    assert!(c.summary.contains("exited with 1"));
}

#[test]
fn a_found_server_without_a_version_answer_passes() {
    let mut s = healthy().server;
    s.version = None;
    let c = only(checks(&s));
    assert_eq!(c.status, Status::Ok);
    assert!(c.summary.starts_with("beside larql"));
}
