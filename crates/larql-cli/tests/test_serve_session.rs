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
    staging: PathBuf,
}

fn dirs() -> Dirs {
    let root = tempfile::tempdir().unwrap();
    let models = root.path().join("models");
    let work = root.path().join("work");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    let container = bf16_container_with_q8_pack(&models);
    let log = root.path().join("audit.jsonl");
    let staging = root.path().join("staging");
    std::fs::create_dir_all(&staging).unwrap();
    let policy = root.path().join("policy.json");
    std::fs::write(
        &policy,
        json!({
            "policy": "larql-session-policy/v1",
            "container": container, "workdir": work, "audit_log": log,
            "operations": ["tensors"], "budget": {"edited_executions": 1},
            "staging_dir": staging,
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
        staging,
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

const WORDS: [&str; 6] = ["The", "capital", "of", "France", "is", "Paris"];
const FIRST_WORD_ID: usize = 10;
/// Long enough that the service is still generating when it is killed.
const LONG_GENERATE: u64 = 20_000;

fn tokenizer_json() -> String {
    use larql_vindex::format::vindex3::fixtures::DENSE_VOCAB;
    let token = |i: usize| match i.checked_sub(FIRST_WORD_ID) {
        Some(w) if w < WORDS.len() => WORDS[w].to_string(),
        _ => format!("[{i}]"),
    };
    let entries: serde_json::Map<String, Value> =
        (0..DENSE_VOCAB).map(|i| (token(i), i.into())).collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": entries, "unk_token": "[0]"}
    })
    .to_string()
}

fn records(log: &Path) -> Vec<Value> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

fn send(stream: &mut UnixStream, reader: &mut BufReader<UnixStream>, request: Value) -> Value {
    stream.write_all(format!("{request}\n").as_bytes()).unwrap();
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    serde_json::from_str(&line).unwrap()
}

/// Killed while an admitted edited execution runs, the service resumes with that execution spent.
#[test]
fn sigkill_during_an_admitted_execution_leaves_it_spent_on_resume() {
    let d = dirs();
    std::fs::write(d.container.join("tokenizer.json"), tokenizer_json()).unwrap();
    // A patch, recorded in-process.
    let vlp = d.work.join("edit.vlp");
    let mut s = larql_lql::Session::new();
    s.execute(&larql_lql::Statement::Use {
        target: larql_lql::ast::UseTarget::Vindex(d.container.display().to_string()),
    })
    .unwrap();
    s.begin_patch(vlp.to_str().unwrap()).unwrap();
    s.stage_weight_delta("1.mlp.down_proj.weight", 5, &[0, 1, 2], 64.0)
        .unwrap();
    s.save_patch().unwrap();
    let policy = |resume: bool| {
        json!({
            "policy": "larql-session-policy/v1",
            "container": d.container, "workdir": d.work, "audit_log": d.log,
            "staging_dir": d.staging, "resume": resume,
            "operations": ["apply_patch", "generate"], "budget": {"edited_executions": 5},
            "limits": {"max_generate_tokens": LONG_GENERATE, "max_sequence_tokens": LONG_GENERATE * 2},
        })
        .to_string()
    };
    std::fs::write(&d.policy, policy(false)).unwrap();
    let mut child = spawn(&d);
    wait_for(&d.socket, &mut child);
    let mut stream = UnixStream::connect(&d.socket).unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let applied = send(
        &mut stream,
        &mut reader,
        json!({"protocol": "larql-session/v1", "op": "apply_patch", "args": {"path": "edit.vlp"}}),
    );
    assert_eq!(applied["ok"], true, "{applied}");
    stream
        .write_all(
            format!(
                "{}\n",
                json!({"protocol": "larql-session/v1", "id": "long", "op": "generate",
                       "args": {"prompt": "The capital of France is", "n": LONG_GENERATE}})
            )
            .as_bytes(),
        )
        .unwrap();
    let start = Instant::now();
    while !records(&d.log).iter().any(|r| r["event"] == "admitted") {
        assert!(start.elapsed() < STARTUP, "never admitted");
        std::thread::sleep(STEP);
    }
    // SAFETY: signalling the child this test spawned.
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGKILL) }, 0);
    child.wait().unwrap();
    let before = records(&d.log);
    let admitted = before.iter().find(|r| r["event"] == "admitted").unwrap();
    assert_eq!(admitted["id"], "long");
    assert!(
        !before
            .iter()
            .any(|r| r["event"] == "request" && r["id"] == "long"),
        "the generate finished before the kill; lengthen it"
    );

    std::fs::remove_file(&d.socket).unwrap();
    std::fs::write(&d.policy, policy(true)).unwrap();
    let mut child = spawn(&d);
    wait_for(&d.socket, &mut child);
    let mut stream = UnixStream::connect(&d.socket).unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let budget = send(
        &mut stream,
        &mut reader,
        json!({"protocol": "larql-session/v1", "op": "budget"}),
    );
    assert_eq!(
        budget["result"],
        json!({"limit": 5, "used": 1, "remaining": 4})
    );
    // SAFETY: as above.
    unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
    assert!(child.wait().unwrap().success());
}
