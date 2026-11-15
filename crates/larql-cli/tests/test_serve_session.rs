//! `larql serve-session` as a process: it starts from a policy file, answers on its socket, and on SIGTERM
//! stops, records the stop and removes the socket. It refuses to start on an existing audit log.
#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use larql_vindex::format::vindex3::fixtures::bf16_container_with_q8_pack;
use serde_json::{json, Value};

const STARTUP: Duration = Duration::from_secs(60);
const STEP: Duration = Duration::from_millis(50);

struct Dirs {
    _root: tempfile::TempDir,
    container: PathBuf,
    work: PathBuf,
    log: PathBuf,
    policy: PathBuf,
    socket: PathBuf,
}

fn dirs() -> Dirs {
    let root = tempfile::tempdir().unwrap();
    let models = root.path().join("models");
    let work = root.path().join("work");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    let container = bf16_container_with_q8_pack(&models);
    let log = root.path().join("audit.jsonl");
    let policy = root.path().join("policy.json");
    std::fs::write(
        &policy,
        json!({
            "policy": "larql-session-policy/v1",
            "container": container, "workdir": work, "audit_log": log,
            "operations": ["tensors"], "budget": {"edited_executions": 1},
        })
        .to_string(),
    )
    .unwrap();
    let socket = root.path().join("s.sock");
    Dirs {
        _root: root,
        container,
        work,
        log,
        policy,
        socket,
    }
}

fn spawn(d: &Dirs) -> Child {
    Command::new(env!("CARGO_BIN_EXE_larql"))
        .arg("serve-session")
        .arg("--config")
        .arg(&d.policy)
        .arg("--socket")
        .arg(&d.socket)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

fn wait_for(path: &Path, child: &mut Child) {
    let start = Instant::now();
    while !path.exists() {
        assert!(
            child.try_wait().unwrap().is_none(),
            "the service exited early"
        );
        assert!(start.elapsed() < STARTUP, "no socket after {STARTUP:?}");
        std::thread::sleep(STEP);
    }
}

#[test]
fn sigterm_stops_the_service_records_the_stop_and_removes_the_socket() {
    let d = dirs();
    let mut child = spawn(&d);
    wait_for(&d.socket, &mut child);

    let mut stream = UnixStream::connect(&d.socket).unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    stream
        .write_all(b"{\"protocol\": \"larql-session/v1\", \"id\": 1, \"op\": \"hello\"}\n")
        .unwrap();
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let hello: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(hello["result"]["operations"], json!(["tensors"]));

    // SAFETY: signalling the child this test spawned.
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
    let status = child.wait().unwrap();
    assert!(status.success(), "{status:?}");
    assert!(!d.socket.exists());
    let records: Vec<Value> = std::fs::read_to_string(&d.log)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let last = records.last().unwrap();
    assert_eq!(last["event"], "stop");
    assert_eq!(last["reason"], "shutdown");
    assert!(records
        .iter()
        .any(|r| r["event"] == "request" && r["op"] == "hello"));
    assert!(d.container.is_dir() && d.work.is_dir());

    // A second start on the same log is refused, before any socket is made.
    let out = Command::new(env!("CARGO_BIN_EXE_larql"))
        .args(["serve-session", "--config"])
        .arg(&d.policy)
        .arg("--socket")
        .arg(&d.socket)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("already exists"));
    assert!(!d.socket.exists());
}
