//! **JSON-RPC 2.0, and what happens when a client gets it wrong.**

use super::super::protocol::*;
use super::{ask, record, server};

#[test]
fn initialize_states_the_protocol_revision_and_the_tool_capability() {
    let record = record();
    let answer = ask(
        &server(&record),
        serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": METHOD_INITIALIZE}),
    );

    assert_eq!(answer["result"]["protocolVersion"], PROTOCOL_VERSION);
    assert!(answer["result"]["capabilities"]["tools"].is_object());
    assert_eq!(
        answer["result"]["serverInfo"]["name"],
        super::super::server::SERVER_NAME
    );
    assert!(answer["error"].is_null());
}

#[test]
fn a_notification_is_answered_with_silence() {
    let record = record();
    // The protocol's rule, not an omission: no `id`, no reply. MCP's
    // own `notifications/initialized` is the one every client sends.
    assert!(server(&record)
        .answer(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
        .is_none());
}

#[test]
fn a_request_with_an_id_is_always_answered() {
    let record = record();
    let server = server(&record);
    for method in [METHOD_INITIALIZE, METHOD_PING, METHOD_TOOLS_LIST] {
        let answer = ask(
            &server,
            serde_json::json!({"jsonrpc": "2.0", "id": 7, "method": method}),
        );
        assert_eq!(answer["id"], 7, "the id comes back on {method}");
        assert_eq!(answer["jsonrpc"], JSONRPC_VERSION);
    }
}

#[test]
fn an_unknown_method_is_refused_by_name() {
    let record = record();
    let answer = ask(
        &server(&record),
        serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "search/expand"}),
    );

    assert_eq!(answer["error"]["code"], ERROR_METHOD_NOT_FOUND);
    assert!(answer["error"]["message"]
        .as_str()
        .expect("a message")
        .contains("search/expand"));
    assert!(answer["result"].is_null());
}

#[test]
fn a_wrong_jsonrpc_version_is_refused_rather_than_assumed() {
    let record = record();
    let answer = ask(
        &server(&record),
        serde_json::json!({"jsonrpc": "1.0", "id": 3, "method": METHOD_PING}),
    );
    assert_eq!(answer["error"]["code"], ERROR_INVALID_REQUEST);
}

#[test]
fn one_unparseable_frame_does_not_end_the_session() {
    let record = record();
    let server = server(&record);

    let bad: serde_json::Value =
        serde_json::from_str(&server.answer("{not json at all").expect("answered")).expect("JSON");
    assert_eq!(bad["error"]["code"], ERROR_PARSE);
    assert!(bad["id"].is_null(), "there was no id to echo");

    // A client that sends one bad frame should learn that, not lose
    // everything it has established.
    let good = ask(
        &server,
        serde_json::json!({"jsonrpc": "2.0", "id": 4, "method": METHOD_PING}),
    );
    assert_eq!(good["id"], 4);
    assert!(good["error"].is_null());
}

#[test]
fn the_serve_loop_writes_one_line_per_answered_request() {
    let record = record();
    let server = server(&record);
    let input = format!(
        "{}\n\n{}\n{}\n",
        serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": METHOD_INITIALIZE}),
        serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": METHOD_TOOLS_LIST}),
    );

    let mut output = Vec::new();
    server
        .serve(std::io::BufReader::new(input.as_bytes()), &mut output)
        .expect("serve");
    let lines: Vec<&str> = std::str::from_utf8(&output)
        .expect("utf8")
        .lines()
        .collect();

    assert_eq!(
        lines.len(),
        2,
        "the blank line and the notification are not answers"
    );
    for line in lines {
        let answer: serde_json::Value = serde_json::from_str(line).expect("one JSON per line");
        assert_eq!(answer["jsonrpc"], JSONRPC_VERSION);
    }
}
