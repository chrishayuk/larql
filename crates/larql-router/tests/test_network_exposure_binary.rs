//! Runs the real `larql-router` binary: a keyless grid on a network address
//! must exit nonzero before binding, and every way of supplying a key (flag,
//! environment) or the explicit opt-in must get past the check.
//!
//! The accepted cases bind a TEST-NET-1 address (RFC 5737) that no test
//! host owns, so the router gets past the exposure check, logs its listen
//! lines and then fails to bind — no socket is ever opened on a real
//! network interface.

use std::io::Read;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

const ROUTER_BIN: &str = env!("CARGO_BIN_EXE_larql-router");
const GRID_KEY_ENV: &str = "LARQL_GRID_KEY";
const INSECURE_PUBLIC_FLAG: &str = "--insecure-public";
const REFUSAL_MARKER: &str = "refusing to serve";
/// Logged only after the exposure check passes, just before the HTTP bind.
const PAST_CHECK_MARKER: &str = "HTTP listening";
/// How clap renders the `--host` default in `--help`.
const DEFAULT_HOST_IN_HELP: &str = "[default: 127.0.0.1]";
const NETWORK_HOST: &str = "0.0.0.0";
/// RFC 5737 TEST-NET-1: a non-loopback address no host is assigned.
const UNOWNED_NETWORK_HOST: &str = "192.0.2.1";
const EPHEMERAL_PORT: &str = "0";
/// `--rebalance-interval 0` disables the rebalancer task.
const REBALANCER_OFF: &str = "0";
const KEY: &str = "shared-secret";
const EXIT_TIMEOUT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(50);

fn router(host: &str, extra: &[&str]) -> Command {
    let mut cmd = Command::new(ROUTER_BIN);
    cmd.args([
        "--host",
        host,
        "--grid-port",
        EPHEMERAL_PORT,
        "--port",
        EPHEMERAL_PORT,
        "--rebalance-interval",
        REBALANCER_OFF,
    ])
    .args(extra)
    .env_remove(GRID_KEY_ENV)
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());
    cmd
}

/// Wait for the router to exit; a router still running after
/// `EXIT_TIMEOUT` is killed and the test fails — it started serving.
fn run_to_exit(mut cmd: Command) -> Output {
    let mut child: Child = cmd.spawn().expect("spawn larql-router");
    let started = Instant::now();
    loop {
        if child.try_wait().expect("poll larql-router").is_some() {
            return child.wait_with_output().expect("collect output");
        }
        if started.elapsed() > EXIT_TIMEOUT {
            child.kill().expect("kill larql-router");
            let mut stdout = String::new();
            if let Some(mut out) = child.stdout.take() {
                let _ = out.read_to_string(&mut stdout);
            }
            let _ = child.wait();
            panic!("larql-router was still serving after {EXIT_TIMEOUT:?}:\n{stdout}");
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn the_binary_binds_loopback_by_default() {
    let out = Command::new(ROUTER_BIN)
        .arg("--help")
        .output()
        .expect("run larql-router --help");
    let help = text(&out.stdout);
    assert!(help.contains(DEFAULT_HOST_IN_HELP), "help: {help}");
    assert!(help.contains(INSECURE_PUBLIC_FLAG), "help: {help}");
}

#[test]
fn a_keyless_public_grid_router_exits_nonzero_with_the_refusal() {
    let out = run_to_exit(router(NETWORK_HOST, &[]));
    let stderr = text(&out.stderr);
    assert!(!out.status.success(), "exit {:?}", out.status);
    assert!(stderr.contains(REFUSAL_MARKER), "stderr: {stderr}");
    assert!(stderr.contains(INSECURE_PUBLIC_FLAG), "stderr: {stderr}");
    assert!(!text(&out.stdout).contains(PAST_CHECK_MARKER));
}

fn assert_gets_past_the_check(out: &Output) {
    let stdout = text(&out.stdout);
    let stderr = text(&out.stderr);
    assert!(!stderr.contains(REFUSAL_MARKER), "stderr: {stderr}");
    assert!(
        stdout.contains(PAST_CHECK_MARKER),
        "stdout: {stdout}\nstderr: {stderr}"
    );
}

#[test]
fn a_key_on_the_flag_admits_a_public_bind() {
    let out = run_to_exit(router(UNOWNED_NETWORK_HOST, &["--grid-key", KEY]));
    assert_gets_past_the_check(&out);
}

#[test]
fn a_key_in_the_environment_admits_a_public_bind() {
    let mut cmd = router(UNOWNED_NETWORK_HOST, &[]);
    cmd.env(GRID_KEY_ENV, KEY);
    assert_gets_past_the_check(&run_to_exit(cmd));
}

#[test]
fn an_empty_key_in_the_environment_is_refused() {
    let mut cmd = router(NETWORK_HOST, &[]);
    cmd.env(GRID_KEY_ENV, "");
    let out = run_to_exit(cmd);
    assert!(!out.status.success());
    assert!(text(&out.stderr).contains(REFUSAL_MARKER));
}

#[test]
fn the_explicit_opt_in_admits_a_keyless_public_bind() {
    let out = run_to_exit(router(UNOWNED_NETWORK_HOST, &[INSECURE_PUBLIC_FLAG]));
    assert_gets_past_the_check(&out);
}
