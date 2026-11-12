//! `--rate-limit` limits a client of the server as it is actually served.
//!
//! The middleware keys on the socket peer, and the peer address reaches a
//! request only when the listener is served with connection info. It was
//! not, so on the real listener every request had no address and was let
//! through: `--rate-limit` did nothing unless `--trust-forwarded-for` was
//! also set. This runs the real binary and counts.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Long enough for a cold debug binary to load the demo container.
const BOOT_DEADLINE: Duration = Duration::from_secs(120);
const POLL: Duration = Duration::from_millis(200);
const IO_TIMEOUT: Duration = Duration::from_secs(30);
/// One request per hour: the second request is over the limit however
/// slowly the test runs.
const LIMIT: &str = "1/hour";
const LIMITED_PATH: &str = "/v1/models";
const HEALTH_PATH: &str = "/v1/health";
const OK: u16 = 200;
const TOO_MANY_REQUESTS: u16 = 429;

/// Kills the server when the test ends, pass or fail.
struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind an ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
}

/// The status code of one `GET path`, on a connection of its own.
fn status(addr: SocketAddr, path: &str, extra_header: &str) -> std::io::Result<u16> {
    let mut stream = TcpStream::connect_timeout(&addr, IO_TIMEOUT)?;
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n{extra_header}\r\n"
    )?;
    let mut response = String::new();
    let _ = stream.read_to_string(&mut response);
    response
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| std::io::Error::other(format!("no status line in {response:?}")))
}

fn serve(extra_args: &[&str]) -> (Server, SocketAddr, tempfile::TempDir) {
    let container = tempfile::tempdir().expect("tempdir");
    let demo = Command::new(env!("CARGO_BIN_EXE_vindex3-demo"))
        .arg(container.path())
        .output()
        .expect("vindex3-demo runs");
    assert!(
        demo.status.success(),
        "{}",
        String::from_utf8_lossy(&demo.stderr)
    );

    let port = free_port();
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");
    let child = Command::new(env!("CARGO_BIN_EXE_larql-server"))
        .arg(container.path())
        .args(["--port", &port.to_string(), "--rate-limit", LIMIT])
        .args(extra_args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("larql-server starts");
    let server = Server(child);

    // The health path is exempt from the limit, so polling it spends none.
    let deadline = Instant::now() + BOOT_DEADLINE;
    while status(addr, HEALTH_PATH, "").ok() != Some(OK) {
        assert!(
            Instant::now() < deadline,
            "larql-server did not answer {HEALTH_PATH} within {BOOT_DEADLINE:?}"
        );
        std::thread::sleep(POLL);
    }
    (server, addr, container)
}

#[test]
fn a_second_request_from_one_client_is_limited_on_the_real_listener() {
    let (_server, addr, _container) = serve(&[]);

    let first = status(addr, LIMITED_PATH, "").expect("first request");
    let second = status(addr, LIMITED_PATH, "").expect("second request");

    assert_eq!(first, OK);
    assert_eq!(second, TOO_MANY_REQUESTS);
}

#[test]
fn a_client_cannot_leave_its_bucket_by_sending_forwarded_for() {
    // Behind a proxy that sets the client-address header, a client that
    // rotates X-Forwarded-For is still one client. No proxy is in front
    // here, so the header is absent and the socket peer is the key.
    let (_server, addr, _container) = serve(&["--client-ip-header", "Fly-Client-IP"]);

    let first = status(addr, LIMITED_PATH, "X-Forwarded-For: 10.0.0.1\r\n").expect("first");
    let second = status(addr, LIMITED_PATH, "X-Forwarded-For: 10.0.0.2\r\n").expect("second");

    assert_eq!(first, OK);
    assert_eq!(second, TOO_MANY_REQUESTS);
}
