//! **What each tool answers, and that the transport added none of it.**

use larql_vindex::format::vindex3::represent::view::OptimizerView;

use super::super::protocol::ERROR_INVALID_PARAMS;
use super::super::tools;
use super::{ask, call, payload, record, server};

#[test]
fn every_tool_answers_with_the_views_own_words() {
    let record = record();
    let server = server(&record);
    let view = OptimizerView::new(&record);
    let state = record.graph().root().clone();
    let edge = record.graph().edges().next().expect("edges");

    // The theorem, made mechanical: each payload is what the view
    // renders, character for character. A transport that reshaped,
    // ordered, summarised or annotated an answer fails here.
    let cases: Vec<(&str, serde_json::Value, String)> = vec![
        (
            tools::DESCRIBE,
            serde_json::json!({}),
            serde_json::to_string_pretty(&view.describe()).expect("json"),
        ),
        (
            tools::CURRENT,
            serde_json::json!({}),
            serde_json::to_string_pretty(&view.current()).expect("json"),
        ),
        (
            tools::FRONTIER,
            serde_json::json!({}),
            serde_json::to_string_pretty(&view.frontier()).expect("json"),
        ),
        (
            tools::NEXT_EXPERIMENT,
            serde_json::json!({}),
            serde_json::to_string_pretty(&view.next_experiment()).expect("json"),
        ),
        (
            tools::EXPLAIN,
            serde_json::json!({ "state": state }),
            serde_json::to_string_pretty(&view.explain(&state).expect("held")).expect("json"),
        ),
        (
            tools::COMPARE,
            serde_json::json!({ "left": edge.parent(), "right": edge.child() }),
            serde_json::to_string_pretty(&view.compare(edge.parent(), edge.child()).expect("held"))
                .expect("json"),
        ),
        (
            tools::EVIDENCE,
            serde_json::json!({}),
            serde_json::to_string_pretty(&view.evidence(None)).expect("json"),
        ),
        (
            tools::EVIDENCE,
            serde_json::json!({ "state": state }),
            serde_json::to_string_pretty(&view.evidence(Some(&state))).expect("json"),
        ),
    ];

    for (id, (name, arguments, expected)) in cases.into_iter().enumerate() {
        let answer = ask(&server, call(id as u64, name, arguments));
        assert!(
            answer["error"].is_null(),
            "{name} errored: {}",
            answer["error"]
        );
        assert_eq!(answer["result"]["isError"], false, "{name}");
        assert_eq!(payload(&answer), expected, "{name} was reshaped in transit");
    }
}

#[test]
fn a_tool_that_does_not_exist_is_refused_by_name() {
    let record = record();
    let answer = ask(
        &server(&record),
        call(1, "optimizer.accept_candidate", serde_json::json!({})),
    );

    assert_eq!(answer["error"]["code"], ERROR_INVALID_PARAMS);
    assert!(answer["error"]["message"]
        .as_str()
        .expect("message")
        .contains("accept_candidate"));
}

#[test]
fn a_missing_required_argument_is_named() {
    let record = record();
    let server = server(&record);

    let answer = ask(&server, call(1, tools::EXPLAIN, serde_json::json!({})));
    assert_eq!(answer["error"]["code"], ERROR_INVALID_PARAMS);
    assert!(answer["error"]["message"]
        .as_str()
        .expect("message")
        .contains("state"));

    let answer = ask(
        &server,
        call(2, tools::COMPARE, serde_json::json!({ "left": "a-state" })),
    );
    assert!(answer["error"]["message"]
        .as_str()
        .expect("message")
        .contains("right"));
}

#[test]
fn a_state_the_record_does_not_hold_is_a_tool_error_and_not_a_protocol_one() {
    let record = record();
    let server = server(&record);

    // The client's frame was well formed and the server understood it.
    // What failed is the question, and the answer says which id was
    // wrong rather than that something was.
    let answer = ask(
        &server,
        call(
            1,
            tools::EXPLAIN,
            serde_json::json!({ "state": "no-such-state" }),
        ),
    );
    assert!(answer["error"].is_null(), "not a protocol failure");
    assert_eq!(answer["result"]["isError"], true);
    assert!(payload(&answer).contains("no-such-state"));

    let held = record.graph().root().clone();
    let answer = ask(
        &server,
        call(
            2,
            tools::COMPARE,
            serde_json::json!({ "left": held, "right": "no-such-state" }),
        ),
    );
    assert_eq!(answer["result"]["isError"], true);
    let not_held: serde_json::Value =
        serde_json::from_str(payload(&answer)).expect("the refusal is structured");
    assert_eq!(not_held["states"], serde_json::json!(["no-such-state"]));
}

#[test]
fn the_rung5_record_reaches_the_client_intact() {
    let record = record();
    let answer = ask(
        &server(&record),
        call(1, tools::FRONTIER, serde_json::json!({})),
    );
    let frontier: serde_json::Value = serde_json::from_str(payload(&answer)).expect("structured");

    let bytes: Vec<u64> = frontier["states"]
        .as_array()
        .expect("states")
        .iter()
        .map(|s| s["logical_bytes"].as_u64().expect("a footprint"))
        .collect();
    for expected in [
        13_684_764_800u64,
        13_682_673_664,
        13_602_484_352,
        13_600_393_216,
    ] {
        assert!(bytes.contains(&expected), "{expected} B did not survive");
    }

    assert_eq!(
        frontier["admitted"].as_array().expect("admitted").len(),
        1,
        "P alone"
    );
}

#[test]
fn the_refusal_arrives_as_a_refusal_and_not_as_an_empty_answer() {
    let record = record();
    let answer = ask(
        &server(&record),
        call(1, tools::NEXT_EXPERIMENT, serde_json::json!({})),
    );
    let refusal: serde_json::Value = serde_json::from_str(payload(&answer)).expect("structured");

    // An agent must be able to tell "nothing to try" from "cannot say",
    // and this record cannot say: it carries no physical accounting
    // authority, so no candidate can be priced. The transport is not
    // what decides that — it serialises whatever the view returns, and
    // the same call on a record carrying sealed container facts comes
    // back as `Available`. See
    // `larql_vindex::…::view::tests::next_experiment`.
    let body = &refusal["Unavailable"];
    assert!(body.is_object(), "the refusal is named in the payload");
    assert_eq!(body["reason"], "no-accounting-authority");
    assert_eq!(body["missing"].as_array().expect("missing").len(), 1);
    assert_eq!(body["accounting"]["procedure"], "logical-bytes/v1");
    assert!(
        body["accounting"]["semantics"].is_null(),
        "a procedure that did not run has no meaning to report"
    );
}
