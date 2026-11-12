//! A failed remote expert shard fails the layer; it never contributes zeros.
//!
//! The remote-MoE client used to treat a shard failure as best-effort on two
//! routes: the KV-engine adapter (`RemoteMoeFfn`) recorded the failure, left
//! the expert contribution at zero and returned the dense half, and the
//! pre-dispatch path dropped a failed shard's partial sum. Either way a
//! continuation came out of a model that had silently skipped experts.
//!
//! Each test runs a loopback HTTP shard that passes the connect-time health
//! check and answers every expert request with a server error, then asserts
//! the failure reaches the caller and names the shard and the cause.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::time::Duration;

use larql_compute::ffn::FfnBackend;
use larql_inference::ffn::moe_remote::{RemoteMoeBackend, ShardConfig};
use larql_inference::ffn::RemoteMoeFfn;
use larql_inference::test_utils::{
    make_test_gemma4_moe_weights, GEMMA4_MOE_HIDDEN, GEMMA4_MOE_NUM_EXPERTS,
};
use larql_inference::vindex::build_moe_router_weights;
use ndarray::Array2;

/// The health check only needs a 2xx with a JSON body.
const HEALTH_BODY: &str = r#"{"status":"ok"}"#;
/// What the failing shard says for every expert request.
const FAILURE_STATUS: u16 = 503;
const FAILURE_REASON: &str = "Service Unavailable";
const FAILURE_BODY: &str = "expert shard overloaded";
const TIMEOUT: Duration = Duration::from_secs(10);
/// Rows of the synthetic post-attention hidden state, and its value.
const ROWS: usize = 2;
const HIDDEN_VALUE: f32 = 0.1;
/// The MoE layer every test dispatches.
const LAYER: usize = 0;
/// Norm parameters for the direct backend calls (Gemma-style offset).
const NORM_OFFSET: f32 = 1.0;
const EPS: f32 = 1e-6;

/// A shard that is healthy for `GET` and fails every `POST` with
/// [`FAILURE_STATUS`]. The request body is drained before answering so the
/// client reads the status rather than a reset connection.
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

/// A backend whose single shard owns every expert and fails every call.
fn failing_backend() -> (RemoteMoeBackend, String) {
    let url = format!("http://{}", failing_shard());
    let config = ShardConfig::new(0, GEMMA4_MOE_NUM_EXPERTS - 1, url.clone()).with_timeout(TIMEOUT);
    let backend = RemoteMoeBackend::connect(vec![config]).expect("health check passes");
    (backend, url)
}

fn hidden_state() -> Array2<f32> {
    Array2::from_elem((ROWS, GEMMA4_MOE_HIDDEN), HIDDEN_VALUE)
}

/// The failure must say which shard and why, not just that something failed.
fn assert_names_shard_and_cause(message: &str, url: &str) {
    assert!(message.contains(url), "names the shard {url}: {message}");
    assert!(
        message.contains(&FAILURE_STATUS.to_string()),
        "carries the status: {message}"
    );
    assert!(
        message.contains(FAILURE_BODY),
        "carries the body: {message}"
    );
}

#[test]
fn a_failing_shard_fails_the_engine_layer_instead_of_contributing_zeros() {
    let weights = make_test_gemma4_moe_weights();
    let (remote, url) = failing_backend();
    let ffn = RemoteMoeFfn {
        weights: &weights,
        remote: &remote,
    };

    let err = match ffn.forward_moe_full_layer(LAYER, &hidden_state()) {
        Ok(out) => panic!(
            "a 5xx shard produced a layer output ({:?}) — its experts were skipped",
            out.map(|h| h.dim())
        ),
        Err(err) => err.to_string(),
    };

    assert_names_shard_and_cause(&err, &url);
    assert!(
        err.contains(&format!("layer {LAYER}")),
        "names the layer: {err}"
    );
}

#[test]
fn a_failing_shard_fails_predispatch_instead_of_contributing_zeros() {
    let weights = make_test_gemma4_moe_weights();
    let (remote, url) = failing_backend();
    let router = build_moe_router_weights(&weights, &*weights.arch, LAYER).expect("MoE layer");
    let h = vec![HIDDEN_VALUE; GEMMA4_MOE_HIDDEN];

    let err = match remote.forward_moe_predispatch(&[h], &[router], NORM_OFFSET, EPS) {
        Ok(h2) => panic!(
            "a 5xx shard produced {} layer(s) of expert output — its partial sum was dropped",
            h2.len()
        ),
        Err(err) => err,
    };

    assert_names_shard_and_cause(&err.to_string(), &url);
}

#[test]
fn a_failing_shard_error_names_the_shard_on_sequence_dispatch() {
    let weights = make_test_gemma4_moe_weights();
    let (remote, url) = failing_backend();
    let router = build_moe_router_weights(&weights, &*weights.arch, LAYER).expect("MoE layer");

    let err = remote
        .forward_moe_seq(LAYER, &hidden_state(), &router, NORM_OFFSET, EPS)
        .expect_err("a 5xx shard must fail the dispatch");

    assert_names_shard_and_cause(&err.to_string(), &url);
}

#[test]
fn a_failing_shard_fails_single_position_dispatch_naming_the_shard() {
    let weights = make_test_gemma4_moe_weights();
    let (remote, url) = failing_backend();
    let router = build_moe_router_weights(&weights, &*weights.arch, LAYER).expect("MoE layer");
    let h = vec![HIDDEN_VALUE; GEMMA4_MOE_HIDDEN];

    let err = remote
        .forward_moe(LAYER, &h, &router, NORM_OFFSET, EPS)
        .expect_err("a 5xx shard must fail the dispatch");

    assert_names_shard_and_cause(&err.to_string(), &url);
}
