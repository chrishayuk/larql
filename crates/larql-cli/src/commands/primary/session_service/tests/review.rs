//! One regression test (or more) per finding of the second review, each named for its finding: M1 staged
//! copies, M2 audit and memory bounds, M3 the slow reader, M4 the capture bound's default, L1 the second
//! witness, L2 the redaction oracle, L3 resume of a missing log, L4 one service per log, L5 the admitted
//! record, L6 the newline scan, and the early-stop error.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::json;

use super::super::ops::BEYOND_GENERATED;
use super::super::policy::{Limits, Policy, DEFAULT_MAX_REQUEST_BYTES};
use super::super::server::{serve_with, BoundSocket, Stopped};
use super::super::service::WITHHELD_PATH_REASON;
use super::*;

fn staged_files(svc: &Service) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(svc.staging.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// A `.vlp` in the workdir editing `row`, recorded in-process.
fn patch_on_row(fx: &Fixture, name: &str, row: usize) -> PathBuf {
    let path = fx.workdir().join(name);
    let mut s = fx.session();
    s.begin_patch(path.to_str().unwrap()).unwrap();
    s.stage_weight_delta(TENSOR, row, &[0, 1], DELTA).unwrap();
    s.save_patch().unwrap();
    path
}

// ── M1: staged copies ──

#[test]
fn m1_a_patch_larql_rejects_leaves_no_staged_copy() {
    let fx = Fixture::new();
    std::fs::write(fx.workdir().join("bad.vlp"), "not a patch").unwrap();
    let mut svc = fx.start(1);
    let (kind, _) = err(&mut svc, "apply_patch", json!({"path": "bad.vlp"}));
    assert_eq!(kind, "failed");
    assert!(staged_files(&svc).is_empty());
}

#[test]
fn m1_a_patch_removed_by_its_description_leaves_no_staged_copy() {
    let fx = Fixture::new();
    let path = fx.patch("described.vlp");
    let mut doc: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    doc["description"] = json!("capital edit");
    std::fs::write(&path, doc.to_string()).unwrap();
    let mut svc = fx.start(1);
    ok(&mut svc, "apply_patch", json!({"path": "described.vlp"}));
    assert_eq!(staged_files(&svc).len(), 1);
    assert_eq!(
        ok(&mut svc, "patches", json!({}))["applied"][0]["name"],
        "capital edit"
    );
    ok(&mut svc, "remove_patch", json!({"path": "capital edit"}));
    assert!(staged_files(&svc).is_empty());
    assert!(svc.applied.is_empty());
}

#[test]
fn m1_reapplying_a_name_keeps_the_live_copy_under_its_own_staged_name() {
    let fx = Fixture::new();
    patch_on_row(&fx, "edit.vlp", 5);
    let mut svc = fx.start(1);
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    let first = std::fs::read(&svc.applied[0].path).unwrap();
    patch_on_row(&fx, "edit.vlp", 6);
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    assert_eq!(staged_files(&svc), ["1.vlp", "2.vlp"]);
    assert_eq!(std::fs::read(&svc.applied[0].path).unwrap(), first);
    ok(&mut svc, "remove_patch", json!({"path": "edit.vlp"}));
    assert_eq!(staged_files(&svc), ["2.vlp"]);
    ok(&mut svc, "remove_patch", json!({"path": "edit.vlp"}));
    assert!(staged_files(&svc).is_empty());
}

#[test]
fn m1_staged_copies_are_flat_and_capped_in_bytes() {
    let fx = Fixture::new();
    std::fs::create_dir_all(fx.workdir().join("sub/dir")).unwrap();
    let one = patch_on_row(&fx, "sub/dir/one.vlp", 5);
    patch_on_row(&fx, "two.vlp", 6);
    let size = std::fs::metadata(&one).unwrap().len();
    let mut p = fx.policy(1);
    p["limits"] = json!({"max_staging_bytes": size + 8});
    let mut svc = fx.start_with(&p).unwrap();
    ok(&mut svc, "apply_patch", json!({"path": "sub/dir/one.vlp"}));
    // No mirrored directories: one flat file.
    assert_eq!(staged_files(&svc), ["1.vlp"]);
    let (kind, reason) = err(&mut svc, "apply_patch", json!({"path": "two.vlp"}));
    assert_eq!(kind, "refused");
    assert!(reason.contains("max_staging_bytes"), "{reason}");
    ok(&mut svc, "remove_patch", json!({"path": "sub/dir/one.vlp"}));
    ok(&mut svc, "apply_patch", json!({"path": "two.vlp"}));
}

// ── M2: audit and memory ──

#[test]
fn m2_long_arguments_are_logged_by_digest_and_length() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    let ids: Vec<u32> = (0..2000).map(|i| i % 16).collect();
    call(&mut svc, "decode", json!({"ids": ids}));
    let record = fx.records().pop().unwrap();
    assert_eq!(record["args"]["logged"], "by digest (args_sha256)");
    assert!(record["args"]["bytes"].as_u64().unwrap() > 4096);
    assert_eq!(record["args_sha256"].as_str().unwrap().len(), 64);
    assert!(record.to_string().len() < 4096);
}

#[test]
fn m2_the_request_limit_defaults_well_below_64_mib() {
    assert_eq!(DEFAULT_MAX_REQUEST_BYTES, 16 << 20);
    assert_eq!(Limits::default().max_request_bytes, 16 << 20);
}

#[test]
fn m2_text_in_and_out_is_bounded() {
    let fx = Fixture::new();
    let mut p = fx.policy(1);
    p["limits"] = json!({"max_text_bytes": 24, "max_sequence_tokens": 8});
    let mut svc = fx.start_with(&p).unwrap();
    let long = "The capital of France is Paris";
    for (op, args) in [
        ("tokenize", json!({"text": long})),
        ("infer", json!({"prompt": long})),
        ("capture", json!({"site": "logits", "prompt": long})),
        ("generate", json!({"prompt": long, "n": 1})),
        (
            "capture",
            json!({"site": "logits", "prompt": PROMPT, "continue_text": long}),
        ),
    ] {
        let (kind, reason) = err(&mut svc, op, args);
        assert_eq!(kind, "refused", "{op}");
        assert!(reason.contains("max_text_bytes"), "{op}: {reason}");
    }
    let (_, reason) = err(&mut svc, "decode", json!({"ids": vec![10; 9]}));
    assert!(reason.contains("max_sequence_tokens"), "{reason}");
    // Eight ids decode to more than 24 bytes of text.
    let (_, reason) = err(&mut svc, "decode", json!({"ids": vec![11; 8]}));
    assert!(reason.contains("max_text_bytes"), "{reason}");
    assert_eq!(
        ok(&mut svc, "decode", json!({"ids": [10, 11]})),
        "The capital"
    );
}

#[test]
fn m2_the_audit_log_refuses_past_its_record_cap() {
    let fx = Fixture::new();
    let mut p = fx.policy(1);
    p["limits"] = json!({"max_audit_records": 3});
    let mut svc = fx.start_with(&p).unwrap();
    ok(&mut svc, "budget", json!({}));
    ok(&mut svc, "budget", json!({}));
    let (kind, reason) = err(&mut svc, "budget", json!({}));
    assert_eq!(kind, "refused");
    assert!(reason.contains("max_audit_records"), "{reason}");
}

// ── M3: the slow reader ──

#[test]
fn m3_a_response_write_gives_up_on_shutdown_without_waiting_for_its_deadline() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    let path = fx.logs.path().join("s.sock");
    let socket = BoundSocket::bind(&path).unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let flag = shutdown.clone();
    let client = std::thread::spawn(move || {
        let mut stream = UnixStream::connect(&path).unwrap();
        // The client's own writes give up too, once the service stops reading its requests.
        stream
            .set_write_timeout(Some(std::time::Duration::from_millis(500)))
            .unwrap();
        let request = format!(
            "{}\n",
            json!({"protocol": "larql-session/v1", "op": "tensor", "args": {"name": TENSOR}})
        );
        for _ in 0..400 {
            if stream.write_all(request.as_bytes()).is_err() {
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
        flag.store(true, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(500));
        drop(stream);
    });
    let started = std::time::Instant::now();
    // A one-hour deadline: only the shutdown check between chunks can end the write in time.
    let stopped = serve_with(
        &mut svc,
        &socket.listener,
        &shutdown,
        std::time::Duration::from_secs(3600),
    );
    client.join().unwrap();
    assert_eq!(stopped, Stopped::Shutdown);
    assert!(started.elapsed() < std::time::Duration::from_secs(30));
    let how: Vec<Value> = fx
        .records()
        .into_iter()
        .filter(|r| r["event"] == "disconnect")
        .map(|r| r["how"].clone())
        .collect();
    assert_eq!(how, [json!("shutdown")]);
    assert!(socket.remove());
}

// ── M4: the capture bound is required by default ──

#[test]
fn m4_require_capture_bound_is_on_by_default() {
    let fx = Fixture::new();
    let checked = Policy::from_bytes(fx.policy(1).to_string().as_bytes()).unwrap();
    assert!(checked.policy.require_capture_bound);
    let mut svc = fx.start(1);
    assert_eq!(
        ok(&mut svc, "hello", json!({}))["require_capture_bound"],
        true
    );
}

// ── L1: the second witness ──

#[test]
fn l1_an_edited_execution_the_gate_missed_is_charged_even_overdrawn_and_stops_the_service() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut svc = fx.start(0);
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    svc.blind_predicate = true;
    let r = call(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": PROMPT}),
    );
    assert_eq!(r["ok"], false);
    assert_eq!(r["error"]["kind"], "failed");
    assert_eq!(
        (r["edited"].clone(), r["counted"].clone()),
        (json!(true), json!(true))
    );
    let records = fx.records();
    let admitted = records.iter().find(|x| x["event"] == "admitted").unwrap();
    assert_eq!(admitted["unpredicted"], true);
    assert_eq!(admitted["overdrawn"], true);
    assert_eq!(admitted["budget_used"], 1);
    let outcome = records.iter().find(|x| x["seq"] == r["seq"]).unwrap();
    assert_eq!(outcome["outcome"], "failed");
    assert_eq!(outcome["admitted_seq"], admitted["seq"]);
    assert!(svc.fatal().is_some());
    let next = call(&mut svc, "budget", json!({}));
    assert_eq!(next["error"]["kind"], "failed");
}

// ── L2: the redaction oracle ──

#[test]
fn l2_a_guessed_container_path_cannot_be_confirmed() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    let right = fx.container.canonicalize().unwrap().display().to_string();
    let wrong = "/models/elsewhere.vindex3".to_string();
    let mut told = Vec::new();
    for guess in [&right, &wrong] {
        for name in [guess.clone(), format!("{guess}/index.json")] {
            let (kind, reason) = err(&mut svc, "tensor", json!({"name": name}));
            assert_eq!(kind, "failed");
            assert!(!reason.contains(guess.as_str()), "{reason}");
            told.push(reason);
        }
    }
    assert!(told.iter().all(|r| r == WITHHELD_PATH_REASON), "{told:?}");
    // Nothing the client sent is rewritten: a prompt naming the path comes back as sent.
    let prompt = format!("The capital of France is {right}");
    let r = call(
        &mut svc,
        "capture",
        json!({"site": "logits", "prompt": prompt}),
    );
    if r["ok"] == true {
        assert_eq!(r["result"]["provenance"]["sequence"]["prompt"], prompt);
        assert_eq!(r["result"]["provenance"]["container"], "<bound container>");
    }
}

// ── L3, L4: the log ──

#[test]
fn l3_resuming_a_missing_log_is_refused() {
    let fx = Fixture::new();
    let mut p = fx.policy(1);
    p["resume"] = json!(true);
    let e = fx.start_with(&p).err().unwrap();
    assert!(e.contains("does not exist"), "{e}");
    assert!(!fx.log().exists());
}

#[test]
fn l4_a_second_service_on_the_same_log_is_refused() {
    let fx = Fixture::new();
    let first = fx.start(1);
    let mut p = fx.policy(1);
    p["resume"] = json!(true);
    let e = fx.start_with(&p).err().unwrap();
    assert!(e.contains("locked by another service"), "{e}");
    drop(first);
    assert!(fx.start_with(&p).is_ok());
}

// ── L5: the admitted record ──

#[test]
fn l5_the_admitted_record_names_the_request_its_arguments_and_the_patch_set() {
    let fx = Fixture::new();
    let vlp = fx.patch("edit.vlp");
    let vlp_sha = {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(std::fs::read(&vlp).unwrap()))
    };
    let mut svc = fx.start(2);
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    let r = call(&mut svc, "infer", json!({"prompt": PROMPT, "top": 3}));
    let records = fx.records();
    let admitted = records.iter().find(|x| x["event"] == "admitted").unwrap();
    let outcome = records.iter().find(|x| x["seq"] == r["seq"]).unwrap();
    // One service identifier on the response, the admitted record and the outcome.
    assert_eq!(admitted["request_uid"], r["request_uid"]);
    assert_eq!(outcome["request_uid"], r["request_uid"]);
    assert_eq!(admitted["id"], 7);
    assert_eq!(admitted["args"], json!({"prompt": PROMPT, "top": 3}));
    assert_eq!(admitted["args_sha256"], outcome["args_sha256"]);
    assert_eq!(admitted["applied_vlp_sha256"], json!([vlp_sha]));
    assert_eq!(admitted["weight_op_digests"].as_array().unwrap().len(), 1);
    assert_eq!(admitted["edits_sha256"].as_str().unwrap().len(), 64);
    // Unique across the service's lifetime and across resumes.
    let first_uid = r["request_uid"].clone();
    let again = call(&mut svc, "budget", json!({}));
    assert_ne!(again["request_uid"], first_uid);
    drop(svc);
    let mut p = fx.policy(2);
    p["resume"] = json!(true);
    let mut resumed = fx.start_with(&p).unwrap();
    let after = call(&mut resumed, "budget", json!({}));
    assert_ne!(after["request_uid"], first_uid);
    assert_ne!(after["request_uid"], again["request_uid"]);
}

// ── L6: the newline scan ──

#[test]
fn l6_a_request_arriving_a_byte_at_a_time_is_answered_once() {
    let fx = Fixture::new();
    let mut svc = fx.start(1);
    let path = fx.logs.path().join("s.sock");
    let socket = BoundSocket::bind(&path).unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let flag = shutdown.clone();
    let client = std::thread::spawn(move || {
        let mut stream = UnixStream::connect(&path).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let request = b"{\"protocol\": \"larql-session/v1\", \"id\": 3, \"op\": \"budget\"}\n";
        for byte in request {
            stream.write_all(&[*byte]).unwrap();
        }
        let mut answer = String::new();
        reader.read_line(&mut answer).unwrap();
        flag.store(true, Ordering::SeqCst);
        answer
    });
    serve_with(
        &mut svc,
        &socket.listener,
        &shutdown,
        std::time::Duration::from_secs(5),
    );
    let answer: Value = serde_json::from_str(&client.join().unwrap()).unwrap();
    assert_eq!(answer["id"], 3);
    assert_eq!(answer["ok"], true);
    assert!(socket.remove());
}

// ── The early-stop error ──

#[test]
fn early_stop_errors_carry_no_length_and_link_both_ways() {
    let fx = Fixture::new();
    fx.patch("edit.vlp");
    let mut svc = fx.start(3);
    ok(&mut svc, "apply_patch", json!({"path": "edit.vlp"}));
    for (positions, larql_reason) in [
        (json!([7]), "CAPTURE refused: position 7 is outside the 6-token sequence (positions 0 to 5)"),
        (
            json!("REPLY"),
            "CAPTURE refused: POSITIONS REPLY needs a continuation (CONTINUE or GENERATE) with at least one token",
        ),
    ] {
        svc.fail_capture_with = Some(larql_reason.to_string());
        let r = call(
            &mut svc,
            "capture",
            json!({"site": "logits", "prompt": PROMPT, "generate": 3, "positions": positions}),
        );
        let reason = r["error"]["reason"].as_str().unwrap();
        assert_eq!(reason, BEYOND_GENERATED);
        assert!(!reason.chars().any(|c| c.is_ascii_digit()), "{reason}");
        assert_eq!(r["counted"], true);
        let records = fx.records();
        let outcome = records.iter().find(|x| x["seq"] == r["seq"]).unwrap();
        let admitted = records
            .iter()
            .find(|x| x["event"] == "admitted" && x["request_uid"] == r["request_uid"])
            .unwrap();
        assert_eq!(outcome["admitted_seq"], admitted["seq"]);
        assert_eq!(admitted["request_uid"], outcome["request_uid"]);
        assert_eq!(outcome["outcome"], "failed");
        assert_eq!(outcome["error"]["internal"], larql_reason);
    }
}
