//! The envelope: what is not a `larql-session/v1` request gets an error response, and the service goes on.

use serde_json::json;

use super::super::protocol::{ErrorKind, Op, PROTOCOL};
use super::*;

fn raw(svc: &mut Service, line: &[u8]) -> Value {
    serde_json::from_slice(&svc.handle_line(line)).unwrap()
}

#[test]
fn malformed_requests_get_errors_and_the_service_keeps_answering() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    let cases: [(&[u8], &str, Value); 8] = [
        (b"not json", "invalid_request", Value::Null),
        (b"[1, 2]", "invalid_request", Value::Null),
        (
            br#"{"protocol": "larql-session/v1", "id": 3}"#,
            "invalid_request",
            json!(3),
        ),
        (
            br#"{"protocol": "larql-session/v1", "id": "x", "op": "budget", "args": 5}"#,
            "invalid_request",
            json!("x"),
        ),
        (
            br#"{"protocol": "larql-session/v1", "op": "budget", "extra": 1}"#,
            "invalid_request",
            Value::Null,
        ),
        (
            br#"{"protocol": "larql-session/v2", "id": 4, "op": "budget"}"#,
            "unsupported_protocol",
            json!(4),
        ),
        (
            br#"{"id": 5, "op": "budget"}"#,
            "unsupported_protocol",
            json!(5),
        ),
        (
            br#"{"protocol": "larql-session/v1", "id": 6, "op": "tokenize", "args": {"txt": "x"}}"#,
            "invalid_request",
            json!(6),
        ),
    ];
    for (line, kind, id) in cases {
        let r = raw(&mut svc, line);
        assert_eq!(r["ok"], false, "{}", String::from_utf8_lossy(line));
        assert_eq!(r["error"]["kind"], kind, "{r}");
        assert_eq!(r["id"], id);
        assert_eq!(r["protocol"], PROTOCOL);
    }
    let r = raw(
        &mut svc,
        br#"{"protocol": "larql-session/v1", "op": "budget", "args": null}"#,
    );
    assert_eq!(r["ok"], true);
    assert_eq!(
        ok(&mut svc, "tokenize", json!({"text": PROMPT})),
        json!(fx.session().tokenize(PROMPT, true).unwrap())
    );
}

#[test]
fn operations_outside_the_policy_are_refused_with_the_reason() {
    let fx = Fixture::new();
    let mut p = fx.policy(1);
    p["operations"] = json!(["tokenize", "capture"]);
    let mut svc = fx.start_with(&p).unwrap();
    for op in ["update_weights", "apply_patch", "tensor"] {
        let (kind, reason) = err(&mut svc, op, json!({}));
        assert_eq!(kind, "refused");
        assert!(reason.contains("the policy does not allow"), "{reason}");
        assert!(reason.contains("tokenize, capture"), "{reason}");
    }
    for op in ["query", "use", "execute", "compile"] {
        let (kind, reason) = err(&mut svc, op, json!({"lql": "USE \"elsewhere\""}));
        assert_eq!(kind, "refused");
        assert!(
            reason.contains("is not an operation of this service"),
            "{reason}"
        );
    }
    let hello = ok(&mut svc, "hello", json!({}));
    assert_eq!(hello["protocol"], PROTOCOL);
    assert_eq!(hello["operations"], json!(["tokenize", "capture"]));
    assert_eq!(
        hello["budget"],
        json!({"limit": 1, "used": 0, "remaining": 1})
    );
    assert_eq!(hello["limits"]["max_generate_tokens"], 256);
}

#[test]
fn op_names_and_error_kinds_round_trip() {
    for op in Op::ALL {
        assert_eq!(Op::from_name(op.name()), Some(op));
    }
    assert_eq!(Op::from_name("hello"), None);
    assert_eq!(Op::names().len(), Op::ALL.len());
    let labels: Vec<_> = [
        ErrorKind::InvalidRequest,
        ErrorKind::UnsupportedProtocol,
        ErrorKind::Refused,
        ErrorKind::BudgetExhausted,
        ErrorKind::Failed,
        ErrorKind::Busy,
    ]
    .iter()
    .map(|k| k.label())
    .collect();
    assert_eq!(
        labels,
        [
            "invalid_request",
            "unsupported_protocol",
            "refused",
            "budget_exhausted",
            "failed",
            "busy"
        ]
    );
}
