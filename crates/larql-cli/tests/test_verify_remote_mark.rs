//! The trial's side of the relay mark (`examples/support/mark.rs`): it
//! records exactly what came back, whether the relay answered `ok`,
//! refused, never answered, or was not there.

#[allow(dead_code)]
#[path = "../examples/support/mark.rs"]
mod mark;

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::time::Duration;

const SHORT: Duration = Duration::from_millis(200);
const EXCHANGES: usize = 264;

/// A control listener that answers one connection's first line with
/// `answer`, or holds it without answering when `answer` is `None`.
fn listener(answer: Option<&'static str>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut writer = stream.try_clone().unwrap();
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).unwrap();
        assert_eq!(line, format!("{}\n", mark::MARK));
        match answer {
            Some(answer) => writer.write_all(answer.as_bytes()).unwrap(),
            None => std::thread::sleep(SHORT * 3),
        }
    });
    address
}

#[test]
fn an_ok_answer_is_recorded_as_given() {
    let control = listener(Some("ok 1\n"));
    let outcome = mark::mark(&control, SHORT, EXCHANGES);
    assert_eq!(outcome.answer.as_deref(), Some("ok 1"));
    assert_eq!(outcome.error, None);
    let json = outcome.json();
    assert_eq!(json["answer"], "ok 1");
    assert_eq!(json["exchanges_completed"], EXCHANGES);
    assert_eq!(json["control"], control);
}

#[test]
fn a_refusal_is_recorded_with_its_reasons() {
    let control = listener(Some(
        "refused 1 connection 0: 2 chunks queued on the link\n",
    ));
    let outcome = mark::mark(&control, SHORT, EXCHANGES);
    assert_eq!(
        outcome.answer.as_deref(),
        Some("refused 1 connection 0: 2 chunks queued on the link")
    );
}

#[test]
fn no_answer_within_the_wait_is_recorded_as_an_error() {
    let control = listener(None);
    let outcome = mark::mark(&control, SHORT, EXCHANGES);
    assert_eq!(outcome.answer, None);
    assert!(outcome.error.is_some());
    assert!(outcome.waited_ns >= SHORT.as_nanos() as u64 / 2);
}

#[test]
fn no_listener_is_recorded_as_an_error() {
    let unused = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = unused.local_addr().unwrap().to_string();
    drop(unused);
    let outcome = mark::mark(&address, SHORT, EXCHANGES);
    assert_eq!(outcome.answer, None);
    assert!(outcome.error.is_some());
}

#[test]
fn a_relay_that_closes_without_answering_is_an_error() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).unwrap();
    });
    let outcome = mark::mark(&address, SHORT, EXCHANGES);
    assert_eq!(outcome.answer, None);
    assert_eq!(
        outcome.error.as_deref(),
        Some("the relay closed without answering")
    );
}
