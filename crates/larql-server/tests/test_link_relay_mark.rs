//! The link relay's generation-boundary mark (VERIFY-REMOTE-1 revision 2),
//! driven through the relay's own code (`examples/support/relay.rs`, the
//! module `v3_link_shaper` runs) over loopback, against a fake upstream.
//!
//! A clean mark splits the tallies exactly at the boundary. Each negative
//! control builds a boundary that is not quiescent and must be refused.

#[allow(dead_code)]
#[path = "../examples/support/relay.rs"]
mod relay;

use relay::{Link, Marks, OK, REFUSED};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const REQUEST_BYTES: usize = 40_000;
const RESPONSE_BYTES: usize = 100_000;
const CLASS: &str = "default";
const STATS_WAIT: Duration = Duration::from_secs(10);
const STATS_POLL: Duration = Duration::from_millis(10);
/// Long enough that a relay spinning on loopback has moved every byte it
/// can, with margin for a loaded CI runner.
const SETTLE: Duration = Duration::from_millis(300);

static NEXT: AtomicU64 = AtomicU64::new(0);

fn scratch(name: &str) -> PathBuf {
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "larql-relay-mark-{}-{n}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    path
}

/// An upstream that answers every `REQUEST_BYTES` read with
/// `RESPONSE_BYTES`, after `respond_after`.
fn upstream(respond_after: Duration) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            std::thread::spawn(move || {
                let mut request = vec![0u8; REQUEST_BYTES];
                while stream.read_exact(&mut request).is_ok() {
                    std::thread::sleep(respond_after);
                    if stream.write_all(&vec![7u8; RESPONSE_BYTES]).is_err() {
                        return;
                    }
                }
            });
        }
    });
    address
}

struct Relay {
    address: String,
    control: String,
    stats: PathBuf,
    marks: Arc<Marks>,
}

/// The relay as `v3_link_shaper` wires it: one thread per connection, one
/// control listener.
fn relay(upstream: String, delay: Duration, timeout: Duration) -> Relay {
    let stats = scratch("stats.jsonl");
    let marks = Marks::new(format!("{}.marks", stats.display()), timeout);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let control = TcpListener::bind("127.0.0.1:0").unwrap();
    let control_address = control.local_addr().unwrap().to_string();
    let link = Link { delay, rate: None };
    let control_marks = marks.clone();
    std::thread::spawn(move || {
        for stream in control.incoming().flatten() {
            relay::control(stream, &control_marks);
        }
    });
    let (stats_path, serve_marks) = (stats.display().to_string(), marks.clone());
    std::thread::spawn(move || {
        for client in listener.incoming() {
            let Ok(client) = client else { continue };
            let (watched, seen) = serve_marks.admit();
            let (upstream, stats, marks) =
                (upstream.clone(), stats_path.clone(), serve_marks.clone());
            std::thread::spawn(move || {
                relay::serve(
                    client, &upstream, link, &stats, CLASS, &marks, watched, seen,
                )
            });
        }
    });
    Relay {
        address,
        control: control_address,
        stats,
        marks,
    }
}

fn exchange(client: &mut TcpStream) {
    client.write_all(&vec![1u8; REQUEST_BYTES]).unwrap();
    let mut response = vec![0u8; RESPONSE_BYTES];
    client.read_exact(&mut response).unwrap();
}

/// Send `mark` over the control listener and return the answer line.
fn mark(control: &str) -> String {
    let mut stream = TcpStream::connect(control).unwrap();
    stream.write_all(b"mark\n").unwrap();
    let mut answer = String::new();
    BufReader::new(stream).read_line(&mut answer).unwrap();
    answer.trim().to_string()
}

fn records(path: &PathBuf, wanted: usize) -> Vec<serde_json::Value> {
    let deadline = Instant::now() + STATS_WAIT;
    loop {
        let text = std::fs::read_to_string(path).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() >= wanted {
            return lines
                .iter()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
        }
        assert!(
            Instant::now() < deadline,
            "{} has {} of {wanted} records",
            path.display(),
            lines.len()
        );
        std::thread::sleep(STATS_POLL);
    }
}

fn marks_path(relay: &Relay) -> PathBuf {
    PathBuf::from(format!("{}.marks", relay.stats.display()))
}

fn closing(stats: &[serde_json::Value], direction: &str) -> serde_json::Value {
    stats
        .iter()
        .find(|r| r["direction"] == direction)
        .unwrap()
        .clone()
}

#[test]
fn a_clean_mark_splits_the_tallies_exactly_at_the_boundary() {
    let relay = relay(
        upstream(Duration::ZERO),
        Duration::from_millis(2),
        Duration::from_secs(1),
    );
    let mut client = TcpStream::connect(&relay.address).unwrap();
    for _ in 0..3 {
        exchange(&mut client);
    }
    std::thread::sleep(SETTLE);
    assert_eq!(mark(&relay.control), format!("{OK} 1"));
    for _ in 0..2 {
        exchange(&mut client);
    }
    drop(client);

    let stats = records(&relay.stats, 2);
    let marks = records(&marks_path(&relay), 2);
    let snapshot = &marks[0];
    assert_eq!(snapshot["mark"], 1);
    assert_eq!(snapshot["queued"], 0);
    assert_eq!(snapshot["unread"], false);
    assert_eq!(snapshot["last_delivered"], "response");
    assert_eq!(snapshot["read_after_snapshot"], 0);
    assert_eq!(marks[1]["summary"], true);
    assert_eq!(marks[1]["answer"], OK);
    assert_eq!(marks[1]["connections"], serde_json::json!([0]));
    assert_eq!(marks[1]["accepted_before"], 1);

    // Bytes partition exactly: three exchanges before, two after.
    for (direction, size) in [("request", REQUEST_BYTES), ("response", RESPONSE_BYTES)] {
        let before = &snapshot[direction];
        let after = closing(&stats, direction);
        assert_eq!(before["bytes"], (3 * size) as u64, "{direction}");
        assert_eq!(after["bytes"], (5 * size) as u64, "{direction}");
        assert!(after["chunks"].as_u64() > before["chunks"].as_u64());
        assert!(after["late_sum_ns"].as_u64() >= before["late_sum_ns"].as_u64());
    }
}

#[test]
fn an_unmarked_relay_records_exactly_what_it_always_did() {
    let relay = relay(
        upstream(Duration::ZERO),
        Duration::from_millis(1),
        Duration::from_secs(1),
    );
    let mut client = TcpStream::connect(&relay.address).unwrap();
    exchange(&mut client);
    drop(client);
    let stats = records(&relay.stats, 2);
    std::thread::sleep(SETTLE);
    let text = std::fs::read_to_string(&relay.stats).unwrap();
    assert_eq!(text.lines().count(), 2, "{text}");
    // Each line is exactly what the relay wrote before the mark existed.
    for (line, record) in text.lines().zip(&stats) {
        let expected = format!(
            "{{\"connection\":0,\"direction\":\"{}\",\"qos\":\"{CLASS}\",\"delay_us\":1000,\
             \"rate_bits_per_s\":\"inf\",\"bytes\":{},\"chunks\":{},\"late_sum_ns\":{},\
             \"late_max_ns\":{}}}",
            record["direction"].as_str().unwrap(),
            record["bytes"],
            record["chunks"],
            record["late_sum_ns"],
            record["late_max_ns"],
        );
        assert_eq!(line, expected);
    }
    for record in &stats {
        let keys: Vec<&String> = record.as_object().unwrap().keys().collect();
        assert_eq!(
            keys,
            [
                "bytes",
                "chunks",
                "connection",
                "delay_us",
                "direction",
                "late_max_ns",
                "late_sum_ns",
                "qos",
                "rate_bits_per_s"
            ]
        );
    }
    assert!(!marks_path(&relay).exists());
}

#[test]
fn a_mark_while_a_request_awaits_its_response_is_refused() {
    let relay = relay(
        upstream(Duration::from_millis(500)),
        Duration::ZERO,
        Duration::from_secs(1),
    );
    let mut client = TcpStream::connect(&relay.address).unwrap();
    client.write_all(&vec![1u8; REQUEST_BYTES]).unwrap();
    std::thread::sleep(SETTLE);
    let answer = mark(&relay.control);
    assert!(answer.starts_with(&format!("{REFUSED} 1")), "{answer}");
    assert!(answer.contains("a request awaits its response"), "{answer}");
    let mut response = vec![0u8; RESPONSE_BYTES];
    client.read_exact(&mut response).unwrap();
    let marks = records(&marks_path(&relay), 2);
    assert_eq!(marks[0]["last_delivered"], "request");
    assert_eq!(marks[1]["answer"], REFUSED);
}

#[test]
fn a_mark_while_a_chunk_is_queued_on_the_link_is_refused() {
    let relay = relay(
        upstream(Duration::ZERO),
        Duration::from_millis(400),
        Duration::from_secs(1),
    );
    let mut client = TcpStream::connect(&relay.address).unwrap();
    client.write_all(&vec![1u8; REQUEST_BYTES]).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let answer = mark(&relay.control);
    assert!(answer.starts_with(&format!("{REFUSED} 1")), "{answer}");
    assert!(answer.contains("queued on the link"), "{answer}");
    let marks = records(&marks_path(&relay), 2);
    assert!(marks[0]["queued"].as_u64().unwrap() > 0);
}

#[test]
fn a_mark_with_no_open_connection_is_refused() {
    let relay = relay(
        upstream(Duration::ZERO),
        Duration::ZERO,
        Duration::from_secs(1),
    );
    let answer = mark(&relay.control);
    assert_eq!(answer, format!("{REFUSED} 1 no open connection"));
    let marks = records(&marks_path(&relay), 1);
    assert_eq!(marks[0]["connections"], serde_json::json!([]));
}

#[test]
fn a_second_mark_is_a_second_epoch() {
    let relay = relay(
        upstream(Duration::ZERO),
        Duration::from_millis(1),
        Duration::from_secs(1),
    );
    let mut client = TcpStream::connect(&relay.address).unwrap();
    exchange(&mut client);
    std::thread::sleep(SETTLE);
    assert_eq!(mark(&relay.control), format!("{OK} 1"));
    assert_eq!(mark(&relay.control), format!("{OK} 2"));
    let marks = records(&marks_path(&relay), 4);
    let summaries: Vec<u64> = marks
        .iter()
        .filter(|r| r["summary"] == true)
        .map(|r| r["mark"].as_u64().unwrap())
        .collect();
    assert_eq!(summaries, [1, 2]);
}

#[test]
fn a_connection_that_never_snapshots_is_refused_by_timeout() {
    let relay = relay(
        upstream(Duration::ZERO),
        Duration::ZERO,
        Duration::from_millis(50),
    );
    // Admitted, but no loop serves it, so it never snapshots.
    let (stuck, _) = relay.marks.admit();
    assert_eq!(stuck.connection, 0);
    let answer = mark(&relay.control);
    assert!(answer.starts_with(&format!("{REFUSED} 1")), "{answer}");
    assert!(answer.contains("connection 0 gave no snapshot"), "{answer}");
}

#[test]
fn an_unknown_control_command_is_refused() {
    let relay = relay(
        upstream(Duration::ZERO),
        Duration::ZERO,
        Duration::from_secs(1),
    );
    let mut stream = TcpStream::connect(&relay.control).unwrap();
    stream.write_all(b"snapshot\n").unwrap();
    let mut answer = String::new();
    BufReader::new(stream).read_line(&mut answer).unwrap();
    assert_eq!(answer.trim(), format!("{REFUSED} 0 unknown command"));
    assert!(!marks_path(&relay).exists());
}

#[test]
fn a_chunk_read_after_its_snapshot_but_before_the_answer_is_refused() {
    let relay = relay(
        upstream(Duration::ZERO),
        Duration::ZERO,
        Duration::from_millis(800),
    );
    let mut client = TcpStream::connect(&relay.address).unwrap();
    exchange(&mut client);
    std::thread::sleep(SETTLE);
    // A connection that never snapshots holds the mark open until its
    // timeout; meanwhile the live connection, already snapshotted, carries a
    // request across the boundary.
    let (_stuck, _) = relay.marks.admit();
    let control = relay.control.clone();
    let marking = std::thread::spawn(move || mark(&control));
    std::thread::sleep(Duration::from_millis(200));
    exchange(&mut client);
    let answer = marking.join().unwrap();
    assert!(answer.starts_with(&format!("{REFUSED} 1")), "{answer}");
    assert!(
        answer.contains("connection 0: ") && answer.contains("chunks read after its snapshot"),
        "{answer}"
    );
}

#[test]
fn a_connection_admitted_after_a_mark_is_not_watched_by_it() {
    let relay = relay(
        upstream(Duration::ZERO),
        Duration::ZERO,
        Duration::from_secs(1),
    );
    let mut first = TcpStream::connect(&relay.address).unwrap();
    exchange(&mut first);
    std::thread::sleep(SETTLE);
    assert_eq!(mark(&relay.control), format!("{OK} 1"));
    let mut second = TcpStream::connect(&relay.address).unwrap();
    exchange(&mut second);
    drop((first, second));
    let marks = records(&marks_path(&relay), 2);
    assert_eq!(marks[1]["connections"], serde_json::json!([0]));
    assert_eq!(marks[1]["accepted_before"], 1);
    let stats = records(&relay.stats, 4);
    assert!(stats.iter().any(|r| r["connection"] == 1));
}
