//! `larql doctor` through the real binary: offline, against caches in a
//! temp dir, so the machine's own caches and network cannot affect it.

use std::process::{Command, Output};

fn doctor(args: &[&str], home: &std::path::Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_larql"))
        .arg("doctor")
        .args(args)
        .env("LARQL_HOME", home.join("larql"))
        .env("HF_HOME", home.join("hf"))
        .env_remove("HF_TOKEN")
        .env_remove("HUGGING_FACE_HUB_TOKEN")
        .output()
        .expect("runs larql")
}

#[test]
fn json_is_a_checklist_whose_verdict_matches_the_exit_code() {
    let home = tempfile::tempdir().unwrap();
    let o = doctor(&["--offline", "--json"], home.path());
    let report: serde_json::Value = serde_json::from_slice(&o.stdout).expect("--json prints JSON");
    assert_eq!(report["larql_version"], env!("CARGO_PKG_VERSION"));
    let checks = report["checks"].as_array().unwrap();
    assert!(checks
        .iter()
        .all(|c| ["ok", "warn", "fail", "skip"].contains(&c["status"].as_str().unwrap())));
    let hub = checks
        .iter()
        .find(|c| c["group"] == "hugging-face" && c["name"] != "Token")
        .unwrap();
    assert_eq!(hub["status"], "skip", "--offline skips the network");
    let ready = report["ready"].as_bool().unwrap();
    assert_eq!(o.status.success(), ready);
}

#[test]
fn the_caches_it_reports_are_the_ones_the_environment_names() {
    let home = tempfile::tempdir().unwrap();
    let o = doctor(&["--offline"], home.path());
    let text = String::from_utf8_lossy(&o.stdout);
    assert!(text.contains("from $LARQL_HOME"), "{text}");
    assert!(text.contains("from $HF_HOME"), "{text}");
    assert!(text.contains("not created yet"), "{text}");
}
