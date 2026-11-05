//! CONTINUATION-CODEC-3 W1: the run module EXECUTED end to end on a
//! fixture ladder — every arm with its guards, the checkpoint names, a
//! resume, the record and its bound verdict — before any long run (CODEC-2
//! W2's lesson: a run module that only compiles is untested).

use std::collections::BTreeMap;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::codec_2_arms::Ladder;
use super::codec_2_store::Checkpoints;
use super::codec_3_freeze::{freeze, Binding, ARMS};
use super::codec_3_freeze_arms::{record, run_arms, stored_residency_guard};
use super::*;

/// Three short rungs that cross the fixture's sliding window, and windows
/// that age rows within them.
fn ladder() -> Ladder {
    Ladder {
        rungs: vec![40, 48, 64],
        resume: 4,
        decode: 32,
    }
}

fn ids() -> Vec<u32> {
    (0..80).map(|i| ((i * 7 + 3) % 29) as u32).collect()
}

fn run_fixture(dir: &Path) -> (Checkpoints, Value) {
    let exact = subjects::fixture(miniature_glimmer, "codec3-e2e-exact");
    let yard = subjects::fixture(miniature_glimmer, "codec3-e2e-yard");
    let backend = ReferenceBackend::new();
    let (exact_ops, yard_ops) = (exact.prepare(&backend), yard.prepare(&backend));
    let stamp = json!({"implementation_git_sha": "fixture", "binary_sha256": "fixture"});
    let store = Checkpoints::open(dir, &stamp);
    run_arms(
        (&exact, &exact_ops),
        (&yard, &yard_ops),
        &backend,
        &ids(),
        &ladder(),
        &store,
    );
    (store, stamp)
}

fn snapshot(store: &Checkpoints) -> BTreeMap<String, String> {
    store
        .files()
        .into_iter()
        .map(|n| {
            let hash = format!(
                "{:x}",
                Sha256::digest(std::fs::read(store.path(&n)).unwrap())
            );
            (n, hash)
        })
        .collect()
}

#[test]
fn the_frozen_arms_run_resume_and_record_through_the_binding() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let (store, stamp) = run_fixture(dir.path());
    let before = snapshot(&store);
    for n in ladder().rungs {
        for a in ARMS {
            assert!(
                before.contains_key(&format!("n{n}-{}.json", a.key())),
                "{n} {}",
                a.name
            );
        }
        assert!(
            before.contains_key(&format!("n{n}-c.logits.f32")),
            "C's logits kept"
        );
        assert!(
            before.contains_key(&format!("n{n}-c.json")),
            "C's scores beside them"
        );
    }

    let (store, _) = run_fixture(dir.path());
    assert_eq!(
        snapshot(&store),
        before,
        "a resume reruns and rewrites nothing"
    );

    let binding = Binding::from_freeze(&freeze()).unwrap();
    let rec = record(&stamp, &binding, &store, &ladder());
    let adjudication = &rec["adjudication"];
    assert_eq!(adjudication["rule_binding"]["primary"], "CH-recent256");
    for v in [
        &adjudication["verdict"],
        &adjudication["reported_not_adjudicated"]["secondary"],
        &adjudication["reported_not_adjudicated"]["control"],
    ] {
        assert!(v["verdict"].is_string(), "{v}");
    }
    for rung in rec["rungs"].as_array().unwrap() {
        let trace = &rung["trace"];
        assert!(trace["CH"].is_array(), "{rung}");
        for a in ARMS {
            assert_eq!(
                trace[a.name], trace["CH"],
                "{} trace at {}",
                a.name, rung["n"]
            );
            assert_eq!(rung["residency"][a.name]["holds"], true, "{}", a.name);
        }
        assert!(
            rung["residency"]["CH-recent256"]["k_exact_bytes"]
                .as_u64()
                .unwrap()
                > 0,
            "the window holds exact rows on this ladder"
        );
    }
}

/// The trace guard at its call site: a resumed CH-recent arm whose
/// recorded CH trace differs from its own stops the run.
#[test]
fn a_resumed_arm_whose_trace_differs_from_chs_stops_the_run() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let (store, _) = run_fixture(dir.path());
    let n = ladder().rungs[0];
    let ch_path = store.path(&format!("n{n}-ch.json"));
    let mut ch: Value = serde_json::from_str(&std::fs::read_to_string(&ch_path).unwrap()).unwrap();
    ch["trace"][0] = json!("0");
    std::fs::write(&ch_path, serde_json::to_string(&ch).unwrap()).unwrap();
    std::fs::remove_file(store.path(&format!("n{n}-ch-recent256.json"))).unwrap();
    let stopped = std::panic::catch_unwind(|| {
        run_fixture(dir.path());
    })
    .expect_err("a trace that is not CH's must stop the run");
    let message = stopped
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(
        message.contains("CH-recent256") && message.contains("provider fault"),
        "{message}"
    );
}

/// Edit one stored checkpoint in place, leaving every checkpoint present.
fn corrupt(store: &Checkpoints, name: &str, edit: impl Fn(&mut Value)) {
    let path = store.path(name);
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    edit(&mut v);
    std::fs::write(&path, serde_json::to_string(&v).unwrap()).unwrap();
}

fn stop_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let stopped = std::panic::catch_unwind(f).expect_err("a corrupted store must be refused");
    stopped
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default()
}

/// One seeded corruption of a stored checkpoint, and the stop it must cause.
struct Corruption {
    arm: &'static str,
    what: &'static str,
    edit: fn(&mut Value),
    stop: &'static str,
}

/// Revalidation on resume and before the record: a completed store whose
/// checkpoints are all present is evidence only while their guards hold.
#[test]
fn a_completed_store_is_revalidated_not_trusted() {
    let _serial = serial();
    let n = ladder().rungs[1];
    let cases = [
        Corruption {
            arm: "ch",
            what: "CH's stored trace",
            edit: |v| v["trace"][0] = json!("0"),
            stop: "provider fault",
        },
        Corruption {
            arm: "ch-recent256",
            what: "a recorded holds=false",
            edit: |v| v["residency"]["holds"] = json!(false),
            stop: "does not hold",
        },
        Corruption {
            arm: "ch-recent128",
            what: "a stray beside holds=true",
            edit: |v| v["residency"]["strays"] = json!(1),
            stop: "does not hold",
        },
    ];
    for Corruption {
        arm,
        what,
        edit,
        stop: needle,
    } in cases
    {
        let dir = tempfile::tempdir().unwrap();
        let (store, stamp) = run_fixture(dir.path());
        let files = store.files().len();
        corrupt(&store, &format!("n{n}-{arm}.json"), edit);
        let path = dir.path().to_path_buf();
        let resumed = stop_message(move || {
            run_fixture(&path);
        });
        assert!(resumed.contains(needle), "resume after {what}: {resumed}");
        let binding = Binding::from_freeze(&freeze()).unwrap();
        let recorded = stop_message(|| {
            record(&stamp, &binding, &store, &ladder());
        });
        assert!(recorded.contains(needle), "record after {what}: {recorded}");
        assert_eq!(
            store.files().len(),
            files,
            "{what}: every checkpoint stays present"
        );
    }
}

#[test]
fn a_stored_residency_is_recomputed_from_its_fields() {
    let _serial = serial();
    let held = json!({"holds": true, "append_born_live": 10, "expected": 10, "strays": 0,
        "scratch_bytes": 4, "scratch_bound": 8});
    stored_residency_guard("CH", 1, &held).unwrap();
    for (k, v) in [
        ("expected", json!(11)),
        ("scratch_bytes", json!(9)),
        ("window_mismatches", json!(1)),
    ] {
        let mut bad = held.clone();
        bad[k] = v;
        assert!(stored_residency_guard("CH", 1, &bad).is_err(), "{k}");
    }
    let mut no_strays = held.clone();
    no_strays.as_object_mut().unwrap().remove("strays");
    assert!(
        stored_residency_guard("CH", 1, &no_strays).is_err(),
        "strays is required"
    );
}
