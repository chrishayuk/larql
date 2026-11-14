//! Connection handshake
//! STATS (covers remote_stats + layer_bands + loaded branches)
//! DESCRIBE (success, no-edges, all modes)
//! WALK (hits, latency, layer-range branch, defaults, error)

use super::*;

#[test]
fn use_remote_connects_via_public_execute() {
    // `connect()` already asserts the USE handshake succeeded. Confirm
    // the session is in the Remote state by running a remote-only verb
    // (STATS) and seeing the canned server banner come back.
    let (_server, mut session) = connect();
    let joined = ok_joined(&mut session, "STATS;");
    assert!(joined.contains("Remote:"));
}

#[test]
fn use_remote_errors_on_5xx_through_execute() {
    let mut server = mockito::Server::new();
    server
        .mock("GET", ENDPOINT_STATS)
        .with_status(503)
        .with_body("upstream down")
        .create();
    let mut session = Session::new();
    let err = run(&mut session, &format!(r#"USE REMOTE "{}";"#, server.url())).unwrap_err();
    assert!(err.contains("503"), "got: {err}");
}

#[test]
fn use_remote_errors_on_unreachable_host_through_execute() {
    let mut session = Session::new();
    // Port 1 is reserved and nothing listens there.
    let err = run(&mut session, r#"USE REMOTE "http://127.0.0.1:1";"#).unwrap_err();
    assert!(err.contains("failed to connect"), "got: {err}");
}

#[test]
fn use_remote_errors_on_invalid_json_through_execute() {
    let mut server = mockito::Server::new();
    server
        .mock("GET", ENDPOINT_STATS)
        .with_status(200)
        .with_body("not actually json")
        .create();
    let mut session = Session::new();
    let err = run(&mut session, &format!(r#"USE REMOTE "{}";"#, server.url())).unwrap_err();
    assert!(err.contains("invalid response"), "got: {err}");
}

#[test]
fn stats_renders_full_summary_with_bands_and_loaded() {
    let (_server, mut session) = connect();
    let joined = ok_joined(&mut session, "STATS;");
    assert!(joined.contains("test-model"));
    assert!(joined.contains("Layers: 32"));
    assert!(joined.contains("Features: 4096"));
    // layer_bands present → "Bands:" line rendered.
    assert!(joined.contains("Bands: syntax"));
    // loaded present → "Loaded:" line rendered.
    assert!(joined.contains("Loaded: browse=true"));
    assert!(joined.contains("Remote:"));
}

#[test]
fn describe_verbose_renders_edges_and_latency() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [{
                    "target": "Paris",
                    "gate_score": 14.2,
                    "layer": 26,
                    "relation": "capital",
                    "source": "probe",
                    "also": ["French", "Europe"],
                }],
                "latency_ms": 15.0,
            })
            .to_string(),
        )
        .create();

    let joined = ok_joined(&mut session, r#"DESCRIBE "France" VERBOSE;"#);
    assert!(joined.starts_with("France"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("capital"));
    assert!(joined.contains("(probe)"));
    assert!(joined.contains("also:"));
    assert!(joined.contains("ms (remote)"));
}

#[test]
fn describe_raw_mode_drops_labels() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [{
                    "target": "Berlin", "gate_score": 8.0, "layer": 26,
                    "relation": "capital", "source": "probe", "also": ["x"],
                }],
                "latency_ms": 4.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"DESCRIBE "Germany" RAW;"#);
    assert!(joined.contains("Berlin"));
}

#[test]
fn describe_brief_mode_compact() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "edges": [{"target": "Rome", "gate_score": 7.0, "layer": 26,
                           "relation": "capital", "source": "model", "also": []}],
                "latency_ms": 2.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"DESCRIBE "Italy" BRIEF;"#);
    assert!(joined.contains("Rome"));
}

#[test]
fn describe_no_edges_friendly_line() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "latency_ms": 1.0}).to_string())
        .create();
    let joined = ok_joined(&mut session, r#"DESCRIBE "Nowhere";"#);
    assert!(joined.contains("(no edges found)"));
}

#[test]
fn describe_with_band() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(200)
        .with_body(serde_json::json!({"edges": [], "latency_ms": 1.0}).to_string())
        .create();
    // KNOWLEDGE band exercises band_str on a non-None branch.
    let joined = ok_joined(&mut session, r#"DESCRIBE "X" KNOWLEDGE;"#);
    assert!(joined.contains("(no edges found)"));
}

#[test]
fn describe_errors_on_http_500() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/describe".into()))
        .with_status(500)
        .with_body("boom")
        .create();
    let err = run(&mut session, r#"DESCRIBE "France";"#).unwrap_err();
    assert!(err.contains("500"), "got: {err}");
}

#[test]
fn walk_renders_hits_and_latency() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(
            serde_json::json!({
                "hits": [
                    {"layer": 5, "feature": 3, "gate_score": 12.5, "target": "Paris"},
                    {"layer": 7, "feature": 1, "gate_score": 9.0, "target": "France"}
                ],
                "latency_ms": 42.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"WALK "test prompt" TOP 3;"#);
    assert!(joined.contains("Feature scan"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("ms (remote)"));
}

#[test]
fn walk_with_layer_range_serialises_layers_param() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(serde_json::json!({"hits": [], "latency_ms": 1.0}).to_string())
        .create();
    // LAYERS 0-5 exercises the `layers_str` Some branch.
    let joined = ok_joined(&mut session, r#"WALK "p" LAYERS 0-5 TOP 5;"#);
    assert!(joined.contains("Feature scan"));
}

#[test]
fn walk_uses_default_top_when_omitted() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(200)
        .with_body(serde_json::json!({"hits": [], "latency_ms": 1.0}).to_string())
        .create();
    let joined = ok_joined(&mut session, r#"WALK "p";"#);
    assert!(joined.contains("Feature scan"));
}

#[test]
fn walk_errors_on_http_404() {
    let (mut server, mut session) = connect();
    server
        .mock("GET", mockito::Matcher::Regex(r"/v1/walk".into()))
        .with_status(404)
        .with_body("nope")
        .create();
    let err = run(&mut session, r#"WALK "p";"#).unwrap_err();
    assert!(err.contains("404"), "got: {err}");
}
