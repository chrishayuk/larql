//! The remote FFN and expert-shard clients send the shard API key.
//!
//! A shard server started with `--api-key` refuses a request without
//! `Authorization: Bearer <key>`. The clients had no way to send one, so a
//! shard a client could reach had to run with no key at all.
//!
//! Each test runs a one-connection-at-a-time HTTP server on loopback that
//! records the `Authorization` header of every request it answers.

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use larql_compute::options::ScopedEnvOverride;
use larql_inference::ffn::moe_remote::{RemoteMoeBackend, RemoteMoeError, ShardConfig};
use larql_inference::ffn::remote::{RemoteFfnConfig, RemoteWalkBackend};
use larql_inference::ffn::shard_auth::{shard_api_key_from_env, SHARD_API_KEY_ENV};

const KEY: &str = "the-shard-key";
const ENV_KEY: &str = "the-key-from-the-environment";
/// Enough for the connect-time `/v1/stats` and `/v1/health` checks.
const STATS_BODY: &str = r#"{"hidden_size":4,"num_layers":1}"#;
const TIMEOUT: Duration = Duration::from_secs(10);
const EXPERTS: (usize, usize) = (0, 3);

/// The `Authorization` value of each request the server answered; `None`
/// for a request that carried none.
type Seen = Arc<Mutex<Vec<Option<String>>>>;

/// Answer every request with `200` and [`STATS_BODY`], recording its
/// `Authorization` header.
fn recording_server() -> (SocketAddr, Seen) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let seen: Seen = Arc::default();
    let recorded = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            let mut authorization = None;
            let mut got_request = false;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                got_request = true;
                if line == "\r\n" {
                    break;
                }
                if let Some((name, value)) = line.split_once(':') {
                    if name.eq_ignore_ascii_case("authorization") {
                        authorization = Some(value.trim().to_string());
                    }
                }
            }
            if !got_request {
                continue;
            }
            recorded.lock().unwrap().push(authorization);
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{STATS_BODY}",
                STATS_BODY.len()
            );
        }
    });
    (addr, seen)
}

fn seen(seen: &Seen) -> Vec<Option<String>> {
    seen.lock().unwrap().clone()
}

fn bearer(key: &str) -> Option<String> {
    Some(format!("Bearer {key}"))
}

// ── The remote FFN client ───────────────────────────────────────────────

#[test]
fn the_ffn_client_sends_the_key_it_was_given() {
    let (addr, requests) = recording_server();
    let config = RemoteFfnConfig::new(format!("http://{addr}"))
        .with_timeout(TIMEOUT)
        .with_api_key(Some(KEY.into()));

    RemoteWalkBackend::connect(config).expect("connect");

    assert_eq!(seen(&requests), [bearer(KEY)]);
}

#[test]
fn the_ffn_client_sends_no_header_without_a_key() {
    let (addr, requests) = recording_server();
    let config = RemoteFfnConfig::new(format!("http://{addr}"))
        .with_timeout(TIMEOUT)
        .with_api_key(None);

    RemoteWalkBackend::connect(config).expect("connect");

    assert_eq!(seen(&requests), [None]);
}

#[test]
fn a_config_takes_the_key_from_the_environment() {
    let _key = ScopedEnvOverride::set(SHARD_API_KEY_ENV, Some(ENV_KEY));
    let (addr, requests) = recording_server();

    assert_eq!(shard_api_key_from_env().as_deref(), Some(ENV_KEY));
    let ffn = RemoteFfnConfig::new(format!("http://{addr}")).with_timeout(TIMEOUT);
    let shard = ShardConfig::new(EXPERTS.0, EXPERTS.1, format!("http://{addr}"));
    assert_eq!(ffn.api_key.as_deref(), Some(ENV_KEY));
    assert_eq!(shard.api_key.as_deref(), Some(ENV_KEY));

    RemoteWalkBackend::connect(ffn).expect("connect");
    assert_eq!(seen(&requests), [bearer(ENV_KEY)]);
}

#[test]
fn an_empty_or_unset_environment_key_is_no_key() {
    for value in [None, Some("")] {
        let _key = ScopedEnvOverride::set(SHARD_API_KEY_ENV, value);
        assert_eq!(shard_api_key_from_env(), None, "{value:?}");
        assert_eq!(RemoteFfnConfig::new("http://x").api_key, None);
    }
}

#[test]
fn a_key_that_is_not_a_header_value_is_refused_before_connecting() {
    let (addr, requests) = recording_server();
    let config = RemoteFfnConfig::new(format!("http://{addr}"))
        .with_timeout(TIMEOUT)
        .with_api_key(Some("line\nbreak".into()));

    let err = match RemoteWalkBackend::connect(config) {
        Ok(_) => panic!("a key with a line break cannot be sent"),
        Err(err) => err.to_string(),
    };

    assert!(err.contains(SHARD_API_KEY_ENV), "{err}");
    assert!(seen(&requests).is_empty());
}

// ── The expert-shard client ─────────────────────────────────────────────

#[test]
fn the_shard_client_sends_the_key_on_its_health_check() {
    let (addr, requests) = recording_server();
    let config = ShardConfig::new(EXPERTS.0, EXPERTS.1, format!("http://{addr}"))
        .with_timeout(TIMEOUT)
        .with_api_key(Some(KEY.into()));

    RemoteMoeBackend::connect(vec![config]).expect("connect");

    assert_eq!(seen(&requests), [bearer(KEY)]);
}

#[test]
fn the_shard_client_sends_no_header_without_a_key() {
    let (addr, requests) = recording_server();
    let config = ShardConfig::new(EXPERTS.0, EXPERTS.1, format!("http://{addr}"))
        .with_timeout(TIMEOUT)
        .with_api_key(None);

    RemoteMoeBackend::connect(vec![config]).expect("connect");

    assert_eq!(seen(&requests), [None]);
}

#[test]
fn a_grpc_shard_is_refused_while_a_key_is_set() {
    // The gRPC transport cannot carry the key yet, so it must not connect
    // without it.
    let config = ShardConfig::new(EXPERTS.0, EXPERTS.1, "grpc://127.0.0.1:1")
        .with_timeout(TIMEOUT)
        .with_api_key(Some(KEY.into()));

    let err = match RemoteMoeBackend::connect(vec![config]) {
        Ok(_) => panic!("a keyed gRPC shard must be refused"),
        Err(err) => err,
    };

    assert!(matches!(err, RemoteMoeError::Client(_)), "{err}");
    assert!(err.to_string().contains(SHARD_API_KEY_ENV), "{err}");
}
