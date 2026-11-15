//! The socket: requests answered in order over a real Unix socket, a second client turned away, an over-long
//! line answered and the connection closed, shutdown on the flag, and the socket file removed.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::json;

use super::super::server::{serve, BoundSocket, Stopped};
use super::*;

fn socket_path(fx: &Fixture) -> PathBuf {
    fx.logs.path().join("s.sock")
}

/// Run the service on the socket, with `client` driving it from another thread; the client's last act is to
/// set the shutdown flag.
fn with_client<T: Send + 'static>(
    fx: &Fixture,
    svc: &mut Service,
    client: impl FnOnce(PathBuf, Arc<AtomicBool>) -> T + Send + 'static,
) -> (T, Stopped) {
    let path = socket_path(fx);
    let socket = BoundSocket::bind(&path).unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let flag = shutdown.clone();
    let handle = std::thread::spawn(move || {
        let out = client(path, flag.clone());
        flag.store(true, Ordering::SeqCst);
        out
    });
    let stopped = serve(svc, &socket.listener, &shutdown);
    let out = handle.join().unwrap();
    assert!(socket.remove());
    assert!(!socket_path(fx).exists());
    (out, stopped)
}

fn send(stream: &mut UnixStream, reader: &mut BufReader<UnixStream>, request: Value) -> Value {
    let mut line = serde_json::to_vec(&request).unwrap();
    line.push(b'\n');
    stream.write_all(&line).unwrap();
    let mut answer = String::new();
    reader.read_line(&mut answer).unwrap();
    serde_json::from_str(&answer).unwrap()
}

fn connect(path: &Path) -> (UnixStream, BufReader<UnixStream>) {
    let stream = UnixStream::connect(path).unwrap();
    let reader = BufReader::new(stream.try_clone().unwrap());
    (stream, reader)
}

#[test]
fn a_client_is_answered_in_order_and_a_second_is_turned_away() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    let (answers, stopped) = with_client(&fx, &mut svc, |path, _| {
        let (mut stream, mut reader) = connect(&path);
        let hello = send(
            &mut stream,
            &mut reader,
            json!({"protocol": "larql-session/v1", "id": 1, "op": "hello"}),
        );
        let second = UnixStream::connect(&path).unwrap();
        // Two requests in one write: both are answered, in order.
        stream
            .write_all(
                b"{\"protocol\": \"larql-session/v1\", \"id\": 2, \"op\": \"budget\"}\nnot json\n",
            )
            .unwrap();
        let mut a = String::new();
        reader.read_line(&mut a).unwrap();
        let mut b = String::new();
        reader.read_line(&mut b).unwrap();
        let mut busy = String::new();
        BufReader::new(second).read_line(&mut busy).unwrap();
        drop(stream);
        std::thread::sleep(super::super::server::POLL * 3);
        (hello, a, b, busy)
    });
    assert_eq!(stopped, Stopped::Shutdown);
    let (hello, a, b, busy) = answers;
    assert_eq!(hello["id"], 1);
    assert_eq!(hello["ok"], true);
    let a: Value = serde_json::from_str(&a).unwrap();
    let b: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(a["id"], 2);
    assert_eq!(b["error"]["kind"], "invalid_request");
    assert!(b["seq"].as_u64() > a["seq"].as_u64());
    let busy: Value = serde_json::from_str(&busy).unwrap();
    assert_eq!(busy["error"]["kind"], "busy");
    let events: Vec<String> = fx
        .records()
        .iter()
        .map(|r| r["event"].as_str().unwrap().to_string())
        .collect();
    assert!(events.contains(&"connect".to_string()));
    assert!(events.contains(&"connection_rejected".to_string()));
    assert!(events.contains(&"disconnect".to_string()));
}

#[test]
fn an_over_long_line_is_answered_and_the_connection_closed() {
    let fx = Fixture::new();
    let mut p = fx.policy(1);
    p["limits"] = json!({"max_request_bytes": 100});
    let mut svc = fx.start_with(&p).unwrap();
    let (answer, _) = with_client(&fx, &mut svc, |path, _| {
        let (mut stream, mut reader) = connect(&path);
        stream.write_all(&[b'x'; 300]).unwrap();
        let mut answer = String::new();
        reader.read_line(&mut answer).unwrap();
        let mut rest = String::new();
        let closed = reader.read_line(&mut rest).unwrap() == 0;
        (answer, closed)
    });
    let (answer, closed) = answer;
    let answer: Value = serde_json::from_str(&answer).unwrap();
    assert_eq!(answer["error"]["kind"], "refused");
    assert!(answer["error"]["reason"]
        .as_str()
        .unwrap()
        .contains("max_request_bytes"));
    assert!(closed);
    let disconnect = fx
        .records()
        .into_iter()
        .find(|r| r["event"] == "disconnect")
        .unwrap();
    assert_eq!(disconnect["how"], "request_too_long");
}

#[test]
fn shutdown_ends_a_connected_conversation() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    let (_, stopped) = with_client(&fx, &mut svc, |path, flag| {
        let (mut stream, mut reader) = connect(&path);
        send(
            &mut stream,
            &mut reader,
            json!({"protocol": "larql-session/v1", "op": "budget"}),
        );
        flag.store(true, Ordering::SeqCst);
        let mut rest = String::new();
        // The service hangs up.
        assert_eq!(reader.read_line(&mut rest).unwrap(), 0);
    });
    assert_eq!(stopped.reason(), "shutdown");
    let how: Vec<Value> = fx
        .records()
        .into_iter()
        .filter(|r| r["event"] == "disconnect")
        .map(|r| r["how"].clone())
        .collect();
    assert_eq!(how, [json!("shutdown")]);
}

#[test]
fn a_fatal_audit_log_stops_the_loop() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    svc.mark_fatal_for_test("disk full");
    let socket = BoundSocket::bind(&socket_path(&fx)).unwrap();
    let stopped = serve(&mut svc, &socket.listener, &AtomicBool::new(false));
    assert_eq!(stopped, Stopped::Fault("disk full".into()));
    assert_eq!(stopped.reason(), "disk full");
}

#[test]
fn a_socket_path_that_exists_is_not_bound_or_removed() {
    let fx = Fixture::new();
    let path = socket_path(&fx);
    std::fs::write(&path, "someone else's").unwrap();
    assert!(BoundSocket::bind(&path)
        .err()
        .unwrap()
        .contains("already exists"));
    assert!(path.exists());
    std::fs::remove_file(&path).unwrap();
    let bound = BoundSocket::bind(&path).unwrap();
    // Replaced behind the service's back: not removed.
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, "replacement").unwrap();
    assert!(!bound.remove());
    assert!(path.exists());
    assert!(BoundSocket::bind(&fx.logs.path().join("no/dir/s.sock")).is_err());
}

/// A client that asks for far more than the socket buffers hold and never reads is hung up on when the write
/// deadline passes, the service does not block on it, and shutdown proceeds. The client never reads a byte:
/// it learns of the hang-up only from its own writes failing (EPIPE once the service has closed), so the
/// service's write can end only by its deadline, and the recorded reason is that, on every platform.
#[test]
fn a_client_that_stops_reading_is_hung_up_on() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    let path = socket_path(&fx);
    let socket = BoundSocket::bind(&path).unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let flag = shutdown.clone();
    let client = std::thread::spawn(move || {
        let mut stream = UnixStream::connect(&path).unwrap();
        stream
            .set_write_timeout(Some(std::time::Duration::from_millis(200)))
            .unwrap();
        let request = format!(
            "{}\n",
            json!({"protocol": "larql-session/v1", "op": "tensor", "args": {"name": TENSOR}})
        );
        // Ask for far more than the socket buffers (each answer is a whole tensor), and never read.
        for _ in 0..400 {
            if stream.write_all(request.as_bytes()).is_err() {
                break;
            }
        }
        let started = std::time::Instant::now();
        let mut hung_up = false;
        while started.elapsed() < std::time::Duration::from_secs(20) {
            if service_hung_up(&mut stream) {
                hung_up = true;
                break;
            }
            std::thread::sleep(super::super::server::POLL);
        }
        flag.store(true, Ordering::SeqCst);
        drop(stream);
        hung_up
    });
    let stopped = super::super::server::serve_with(
        &mut svc,
        &socket.listener,
        &shutdown,
        std::time::Duration::from_millis(200),
    );
    assert!(client.join().unwrap(), "the service never hung up");
    assert_eq!(stopped, Stopped::Shutdown);
    let how: Vec<Value> = fx
        .records()
        .into_iter()
        .filter(|r| r["event"] == "disconnect")
        .map(|r| r["how"].clone())
        .collect();
    assert_eq!(how, [json!(super::super::server::WRITE_DEADLINE_EXPIRED)]);
    assert!(socket.remove());
}

/// Whether the service has closed `stream`'s connection, found without reading: a byte that is not a
/// request (no newline) is written, and only a closed peer makes that fail outright.
fn service_hung_up(stream: &mut UnixStream) -> bool {
    match stream.write_all(b" ") {
        Ok(()) => false,
        Err(e) => !matches!(
            e.kind(),
            std::io::ErrorKind::WouldBlock
                | std::io::ErrorKind::TimedOut
                | std::io::ErrorKind::Interrupted
        ),
    }
}

#[test]
fn a_client_that_disconnects_mid_request_is_still_charged() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut svc = fx.start(3);
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    let (_, _) = with_client(&fx, &mut svc, |path, _| {
        let mut stream = UnixStream::connect(&path).unwrap();
        stream
            .write_all(
                format!(
                    "{}\n",
                    json!({"protocol": "larql-session/v1", "id": 9, "op": "generate",
                           "args": {"prompt": PROMPT, "n": 4}})
                )
                .as_bytes(),
            )
            .unwrap();
        drop(stream);
        std::thread::sleep(super::super::server::POLL * 10);
    });
    let records = fx.records();
    let admitted: Vec<&Value> = records
        .iter()
        .filter(|r| r["event"] == "admitted")
        .collect();
    assert_eq!(admitted.len(), 1, "{records:#?}");
    assert_eq!(admitted[0]["id"], 9);
    assert_eq!(svc.budget_used, 1);
}
