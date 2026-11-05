//! The bounded read underneath every store file and member.

use super::super::super::store_file::{open_regular, read_exactly};
use super::*;

const CONTENT: &[u8] = b"twelve bytes";
const SHRUNK_TO: u64 = 5;

/// A file truncated between being measured and being read must read short and be
/// refused. Only a concurrent writer can do this to a load, so the two steps are
/// driven by hand, against a real file.
#[test]
fn file_shrinking_after_it_was_measured_is_refused() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("file");
    std::fs::write(&path, CONTENT).unwrap();
    let measured = open_regular(&path).unwrap();
    assert_eq!(measured.len, CONTENT.len() as u64);

    let writer = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    writer.set_len(SHRUNK_TO).unwrap();

    match read_exactly(measured.file, measured.len, &measured.path).unwrap_err() {
        StoreLoadError::MemberLengthMismatch { declared, read, .. } => {
            assert_eq!(declared, CONTENT.len() as u64);
            assert_eq!(read, SHRUNK_TO);
        }
        other => panic!("expected MemberLengthMismatch, got {other:?}"),
    }
}

/// A file growing after it was measured is cut at the measured length, not read to
/// its new end.
#[test]
fn file_growing_after_it_was_measured_reads_only_the_measured_length() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("file");
    std::fs::write(&path, CONTENT).unwrap();
    let measured = open_regular(&path).unwrap();
    std::fs::write(&path, [CONTENT, CONTENT].concat()).unwrap();
    let bytes = read_exactly(measured.file, measured.len, &measured.path).unwrap();
    assert_eq!(bytes.len(), CONTENT.len());
}
