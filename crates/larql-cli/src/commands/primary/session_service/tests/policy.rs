//! The policy and the start: what is refused before the service answers anything, and the audit log's
//! creation, resumption and records.

use serde_json::json;

use super::super::policy::{Limits, Policy};
use super::*;

#[test]
fn a_policy_is_checked_as_it_is_read() {
    let fx = Fixture::new();
    let mut p = fx.policy(1);
    assert!(Policy::from_bytes(p.to_string().as_bytes()).is_ok());
    assert!(Policy::from_bytes(b"{").unwrap_err().contains("not valid"));
    p["policy"] = json!("larql-session-policy/v0");
    assert!(Policy::from_bytes(p.to_string().as_bytes())
        .unwrap_err()
        .contains("this service reads"));
    let mut p = fx.policy(1);
    p["operations"] = json!(["capture", "query"]);
    assert!(Policy::from_bytes(p.to_string().as_bytes())
        .unwrap_err()
        .contains("`query`"));
    let mut p = fx.policy(1);
    p["limits"] = json!({"max_generate_tokens": 1, "max_tokens": 2});
    assert!(Policy::from_bytes(p.to_string().as_bytes())
        .unwrap_err()
        .contains("max_tokens"));
    let mut p = fx.policy(1);
    p["operations"] = json!(["capture", "capture"]);
    assert_eq!(
        Policy::from_bytes(p.to_string().as_bytes())
            .unwrap()
            .operations
            .len(),
        1
    );
    assert_eq!(fx.policy(1).get("limits"), None);
    let checked = Policy::from_bytes(fx.policy(1).to_string().as_bytes()).unwrap();
    assert_eq!(checked.policy.limits, Limits::default());

    let file = fx.logs.path().join("policy.json");
    std::fs::write(&file, fx.policy(1).to_string()).unwrap();
    assert_eq!(Policy::read(&file).unwrap().sha256, checked.sha256);
    assert!(Policy::read(&fx.logs.path().join("missing.json")).is_err());
}

#[test]
fn the_start_refuses_overlapping_or_missing_places() {
    let fx = Fixture::new();
    let refuse = |p: Value, needle: &str| {
        let e = fx.start_with(&p).err().unwrap();
        assert!(e.contains(needle), "{e}");
    };
    let mut p = fx.policy(1);
    p["workdir"] = json!(fx.container);
    refuse(p, "must not contain one another");
    let mut p = fx.policy(1);
    p["audit_log"] = json!(fx.workdir().join("audit.jsonl"));
    refuse(p, "outside the workdir");
    let mut p = fx.policy(1);
    p["staging_dir"] = json!(fx.workdir());
    refuse(p, "staging directory");
    let mut p = fx.policy(1);
    p["container"] = json!(fx.logs.path().join("none"));
    refuse(p, "the container");
    let mut p = fx.policy(1);
    let empty = fx.logs.path().join("empty");
    std::fs::create_dir(&empty).unwrap();
    p["container"] = json!(empty);
    refuse(p, "index.json");
    let mut p = fx.policy(1);
    p["audit_log"] = json!(fx.logs.path().join("no/such/dir/audit.jsonl"));
    refuse(p, "audit log's directory");
    let mut p = fx.policy(1);
    p["audit_log"] = json!("/");
    refuse(p, "names no file");
    let mut p = fx.policy(1);
    p["representation_source"] = json!("sometimes");
    refuse(p, "unknown representation source");
    let mut p = fx.policy(1);
    p["staging_dir"] = json!(fx.logs.path());
    p["representation_source"] = json!("stored");
    assert!(fx.start_with(&p).is_ok());
}

#[test]
fn a_container_that_is_not_vindex3_is_refused() {
    let fx = Fixture::new();
    let fake = fx.logs.path().join("fake");
    std::fs::create_dir(&fake).unwrap();
    std::fs::write(fake.join("index.json"), "{}").unwrap();
    let mut p = fx.policy(1);
    p["container"] = json!(fake);
    let e = fx.start_with(&p).err().unwrap();
    assert!(
        e.contains("bind the container") || e.contains("not a VINDEX3"),
        "{e}"
    );
}

#[test]
fn an_existing_log_is_refused_unless_resumed_and_a_resume_carries_on() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    {
        let mut svc = fx.start(4);
        ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
        ok(&mut svc, "infer", json!({"prompt": PROMPT}));
        svc.stop("test");
    }
    let before = fx.records();
    let e = fx.start_with(&fx.policy(4)).err().unwrap();
    assert!(e.contains("already exists"), "{e}");
    assert_eq!(fx.records(), before, "a refused start writes nothing");

    let mut p = fx.policy(4);
    p["resume"] = json!(true);
    let mut svc = fx.start_with(&p).unwrap();
    let last_seq = before.last().unwrap()["seq"].as_u64().unwrap();
    let r = call(&mut svc, "budget", json!({}));
    assert_eq!(r["result"], json!({"limit": 4, "used": 1, "remaining": 3}));
    let records = fx.records();
    let start = &records[before.len()];
    assert_eq!(start["event"], "start");
    assert_eq!(start["resumed"], true);
    assert_eq!(start["seq"], last_seq + 1);
    assert_eq!(r["seq"], last_seq + 2);
}

#[test]
fn a_resume_refuses_a_log_it_cannot_trust() {
    let fx = Fixture::new();
    let mut p = fx.policy(1);
    p["resume"] = json!(true);
    // An empty log resumes from the start.
    std::fs::write(fx.log(), "").unwrap();
    drop(fx.start_with(&p).unwrap());
    for (text, needle) in [
        ("{\"seq\": 1}\nnot json\n", "not JSONL at line 2"),
        ("{\"event\": \"start\"}\n", "without `seq`"),
        (
            "{\"seq\": 1, \"budget_used\": 0, \"event\": \"start\", \"container_index_sha256\": \"00\"}\n",
            "was written for container index 00",
        ),
    ] {
        std::fs::write(fx.log(), text).unwrap();
        let e = fx.start_with(&p).err().unwrap();
        assert!(e.contains(needle), "{e}");
    }
}

#[test]
fn every_request_is_logged_in_sequence_with_its_digests() {
    let fx = Fixture::new();
    let mut svc = fx.start(2);
    let r1 = call(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT}),
    );
    let r2 = call(&mut svc, "tokenize", json!({"text": PROMPT}));
    let r3 = call(&mut svc, "query", json!({"lql": "STATS"}));
    let raw = svc.handle_line(b"not json");
    svc.stop("test");
    let records = fx.records();
    let seqs: Vec<u64> = records.iter().map(|r| r["seq"].as_u64().unwrap()).collect();
    assert_eq!(seqs, (1..=records.len() as u64).collect::<Vec<_>>());
    assert_eq!(records[0]["event"], "start");
    assert_eq!(
        records[0]["container"],
        json!(fx.container.canonicalize().unwrap())
    );
    assert_eq!(records.last().unwrap()["event"], "stop");
    assert_eq!(records.last().unwrap()["reason"], "test");

    let capture = &records[1];
    assert_eq!(capture["seq"], r1["seq"]);
    assert_eq!(capture["op"], "capture");
    assert_eq!(capture["args"]["prompt"], PROMPT);
    assert_eq!(capture["edited"], false);
    assert_eq!(
        capture["arrays"][0]["sha256"],
        r1["result"]["array"]["sha256"]
    );
    use sha2::{Digest, Sha256};
    let line = serde_json::to_vec(&r1).unwrap();
    assert_eq!(
        capture["response_sha256"],
        format!("{:x}", Sha256::digest(&line))
    );
    assert!(capture["time_unix_ms"].as_u64().unwrap() > 0);
    assert_eq!(records[2]["seq"], r2["seq"]);
    assert_eq!(records[3]["error"]["kind"], "refused");
    assert_eq!(r3["error"]["kind"], "refused");
    let raw: Value = serde_json::from_slice(&raw).unwrap();
    assert_eq!(records[4]["seq"], raw["seq"]);
    assert_eq!(records[4]["error"]["kind"], "invalid_request");
}

#[test]
fn operand_arrays_are_logged_by_digest() {
    let fx = Fixture::new();
    let mut svc = fx.start(2);
    let rows = tensor_rows(&fx.session());
    let a = operand(&a_factor(rows, 0.5), [rows, 1]);
    call(
        &mut svc,
        "add_low_rank",
        json!({"tensor": TENSOR, "a": a, "b": {"npy_base64": "%%"}, "columns": COLUMNS, "numerics": F64_PROFILE}),
    );
    let record = fx.records().pop().unwrap();
    assert_eq!(
        record["args"]["a"]["npy_sha256"].as_str().unwrap().len(),
        64
    );
    assert!(record["args"]["a"].get("npy_base64").is_none());
    assert_eq!(record["args"]["b"], json!({"undecodable_base64_chars": 2}));
    assert_eq!(record["args"]["columns"], json!(COLUMNS));
}

#[test]
fn a_service_whose_log_fails_stops_answering() {
    let fx = Fixture::new();
    let mut svc = fx.start(2);
    svc.record("probe", json!({}));
    assert!(svc.fatal().is_none());
    // A failed append cannot be provoked on an open O_APPEND file portably; enter the state it leaves.
    svc.mark_fatal_for_test("disk full");
    let r = call(&mut svc, "tokenize", json!({"text": PROMPT}));
    assert_eq!(r["error"]["kind"], "failed");
    assert!(r["error"]["reason"].as_str().unwrap().contains("disk full"));
}
