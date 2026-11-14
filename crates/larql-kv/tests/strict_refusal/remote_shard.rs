//! A failed remote expert shard ends generation with its error.
//!
//! The production `--moe-shards` CPU path drives a `StandardEngine` through
//! `generate_with_engine_resident` with `RemoteMoeFfn`. That adapter used to
//! be best-effort, so a shard answering 5xx contributed zero experts and the
//! stream carried on. This drives the same composition against a loopback
//! shard that fails every expert call, and asserts the failure is what the
//! caller of generation gets back.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::time::Duration;

use larql_inference::ffn::moe_remote::{RemoteMoeBackend, ShardConfig};
use larql_inference::ffn::RemoteMoeFfn;
use larql_inference::kv_engine::EngineError;
use larql_inference::test_utils::{
    make_test_gemma4_moe_weights, make_test_tokenizer, GEMMA4_MOE_NUM_EXPERTS,
};
use larql_kv::generation::generate_with_engine_resident_checked;
use larql_kv::StandardEngine;

use crate::entry::{empty_index, PROMPT};

const HEALTH_BODY: &str = r#"{"status":"ok"}"#;
const FAILURE_STATUS: u16 = 503;
const FAILURE_REASON: &str = "Service Unavailable";
const FAILURE_BODY: &str = "expert shard overloaded";
const TIMEOUT: Duration = Duration::from_secs(10);
/// More than one, so a stream that carried on past the failure would show.
const MAX_NEW_TOKENS: usize = 4;

/// Healthy for `GET` (the connect-time check), [`FAILURE_STATUS`] for every
/// expert `POST`. Drains the body first so the client reads the status.
fn failing_shard() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
                continue;
            }
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some((name, value)) = line.split_once(':') {
                    if name.eq_ignore_ascii_case("content-length") {
                        content_length = value.trim().parse().unwrap_or(0);
                    }
                }
            }
            let mut body = vec![0u8; content_length];
            let _ = reader.read_exact(&mut body);
            let (status, reason, payload) = if request_line.starts_with("GET") {
                (200, "OK", HEALTH_BODY)
            } else {
                (FAILURE_STATUS, FAILURE_REASON, FAILURE_BODY)
            };
            let _ = write!(
                stream,
                "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
        }
    });
    addr
}

#[test]
fn a_failing_remote_shard_ends_generation_with_its_error() {
    let weights = make_test_gemma4_moe_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = empty_index(&weights);
    let url = format!("http://{}", failing_shard());
    let config = ShardConfig::new(0, GEMMA4_MOE_NUM_EXPERTS - 1, url.clone()).with_timeout(TIMEOUT);
    let remote = RemoteMoeBackend::connect(vec![config]).expect("health check passes");
    let ffn = RemoteMoeFfn {
        weights: &weights,
        remote: &remote,
    };
    let mut engine = larql_kv::AnyEngine::Kv(Box::new(StandardEngine::new(None)));
    let mut emitted = 0usize;

    let outcome = generate_with_engine_resident_checked(
        &mut engine,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &PROMPT,
        MAX_NEW_TOKENS,
        |_, _| emitted += 1,
    );

    let err = outcome
        .error
        .expect("a 5xx shard must end generation with an error");
    assert!(
        matches!(err, EngineError::Execution(_)),
        "the shard failure arrives as the engine's execution refusal: {err:?}"
    );
    let message = err.to_string();
    assert!(message.contains(&url), "names the shard: {message}");
    assert!(
        message.contains(&FAILURE_STATUS.to_string()) && message.contains(FAILURE_BODY),
        "carries the cause: {message}"
    );
    assert!(
        outcome.tokens.is_empty() && emitted == 0,
        "prefill already needs the experts, so no token may be emitted"
    );
}
