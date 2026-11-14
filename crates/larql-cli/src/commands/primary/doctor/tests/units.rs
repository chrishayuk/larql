use std::path::Path;

use super::super::units::{display_path, format_bytes};
use super::home;

#[test]
fn bytes_use_the_largest_decimal_unit_reached() {
    assert_eq!(format_bytes(0), "0 B");
    assert_eq!(format_bytes(999), "999 B");
    assert_eq!(format_bytes(1_000), "1.0 KB");
    assert_eq!(format_bytes(1_500_000), "1.5 MB");
    assert_eq!(format_bytes(3_000_000_000), "3.0 GB");
    assert_eq!(format_bytes(2_500_000_000_000), "2.5 TB");
}

#[test]
fn a_path_under_home_is_shown_from_tilde() {
    let shown = display_path(&home().join(".cache/larql"), Some(&home()));
    assert_eq!(
        shown,
        Path::new("~").join(".cache/larql").display().to_string()
    );
    assert_eq!(display_path(&home(), Some(&home())), "~");
}

#[test]
fn a_path_elsewhere_is_shown_whole() {
    let elsewhere = Path::new("/opt/models");
    assert_eq!(display_path(elsewhere, Some(&home())), "/opt/models");
    assert_eq!(display_path(elsewhere, None), "/opt/models");
}
