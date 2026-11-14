//! The Unix FIFO surface: request FIFO creation and reply delivery.

use super::super::*;
use super::scratch;
use std::io::Read;
use std::os::unix::fs::FileTypeExt;

fn is_fifo(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|meta| meta.file_type().is_fifo())
        .unwrap_or(false)
}

#[test]
fn ensure_fifo_creates_the_fifo_and_its_parent() {
    let dir = scratch("fifo_create");
    let fifo = dir.join("nested").join("requests.fifo");
    ensure_fifo(&fifo).expect("created");
    assert!(is_fifo(&fifo));
    // A second call finds the FIFO already there.
    ensure_fifo(&fifo).expect("reused");
}

#[test]
fn ensure_fifo_refuses_a_path_that_is_not_a_fifo() {
    let dir = scratch("fifo_not_fifo");
    let file = dir.join("plain.txt");
    std::fs::write(&file, "").expect("plain file");
    let error = ensure_fifo(&file).expect_err("not a FIFO");
    assert!(error.to_string().contains("not a FIFO"), "{error}");
}

#[test]
fn make_fifo_reports_failure_and_interior_nul() {
    let dir = scratch("fifo_fail");
    let file = dir.join("taken");
    std::fs::write(&file, "").expect("plain file");
    let error = make_fifo(&file).expect_err("path exists");
    assert!(error.to_string().contains("mkfifo"), "{error}");

    let nul = PathBuf::from("bad\0name");
    let error = make_fifo(&nul).expect_err("interior NUL");
    assert!(error.to_string().contains("NUL"), "{error}");
}

/// A reader already waiting on the reply FIFO receives the line.
#[test]
fn respond_delivers_to_a_waiting_reader() {
    let dir = scratch("fifo_reply");
    let fifo = dir.join("reply.fifo");
    make_fifo(&fifo).expect("fifo");
    let reader_path = fifo.clone();
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        std::fs::File::open(&reader_path)
            .expect("open read end")
            .read_to_string(&mut text)
            .expect("read reply");
        text
    });
    respond(Some(&fifo), "ok frames=3").expect("delivered");
    assert_eq!(reader.join().expect("reader"), "ok frames=3\n");
}

/// Any open failure other than "no reader yet" is reported at once.
#[test]
fn respond_reports_an_unopenable_reply_path() {
    let dir = scratch("fifo_reply_missing");
    respond(Some(&dir.join("absent").join("reply.fifo")), "ok").expect_err("no such dir");
}
