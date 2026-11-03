//! **MEM-002 I2b: a record says which computation ran, through the `larql`
//! binary.** File identity says which files were read; these hold the record to
//! what executed on them, and a campaign attempt to refuse what it can't show.
#![cfg(unix)]

#[allow(dead_code)]
#[path = "support/continuation_fixture.rs"]
mod fixture;

use std::path::{Path, PathBuf};
use std::process::Command;

use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use serde_json::Value;

const NVFP4: &str = "NVFP4";
const Q4_K: &str = "Q4_K";
const TAMPER: &str = "hostile-lowering/v1";
const EVIDENCE_REFUSAL: &str = "execution evidence";
const FALLBACK: &str = "fall back";

struct Ran {
    code: Option<i32>,
    out: String,
    err: String,
    stream: Vec<Value>,
    _dir: tempfile::TempDir,
}

impl Ran {
    fn record(&self, kind: &str) -> Option<&Value> {
        self.stream.iter().rev().find(|r| r["record"] == kind)
    }
    fn admission(&self) -> &Value {
        self.record("admission").expect("an admission record")
    }
    fn execution(&self) -> &Value {
        &self.admission()["execution_identity"]
    }
    fn before_binding(&self) -> &Value {
        self.stream
            .iter()
            .find(|r| r["phase"] == "before_binding")
            .expect("a record before binding")
    }
    fn generated(&self) -> &str {
        self.out
            .lines()
            .find(|l| l.contains("generated ids"))
            .expect("generated ids on stdout")
    }
    /// Stopped before any weight was bound.
    fn stopped_before_binding(&self) -> bool {
        !self.stream.iter().any(|r| r["record"] == "observation")
    }
}

struct Attempt<'a> {
    container: &'a Path,
    campaign: bool,
    extra: &'a [&'a str],
    env: &'a [(&'a str, &'a str)],
    manifest: Option<&'a Path>,
}

fn exec(attempt: Attempt<'_>) -> Ran {
    let dir = tempfile::tempdir().unwrap();
    let (regions, sealed) = (dir.path().join("r.jsonl"), dir.path().join("s.jsonl"));
    let mut command = Command::new(env!("CARGO_BIN_EXE_larql"));
    command
        .args(["vindex3", "exec"])
        .arg(attempt.container)
        .args(["--tokens", "1,2", "--generate", "2"])
        .args(["--residency-curve", "--unquiet-ok"])
        .arg("--residency-regions")
        .arg(&regions)
        .arg("--sealed-latency")
        .arg(&sealed)
        .args(attempt.extra)
        .envs(attempt.env.iter().copied());
    if attempt.campaign {
        command.arg("--residency-campaign");
    }
    if let Some(manifest) = attempt.manifest {
        command.arg("--verification-manifest").arg(manifest);
    }
    let output = command.output().expect("run larql");
    let raw = std::fs::read_to_string(&regions).unwrap_or_default();
    Ran {
        code: output.status.code(),
        out: String::from_utf8_lossy(&output.stdout).into_owned(),
        err: String::from_utf8_lossy(&output.stderr).into_owned(),
        stream: raw
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect(),
        _dir: dir,
    }
}

fn diagnostic(container: &Path, extra: &[&str]) -> Ran {
    exec(Attempt {
        container,
        campaign: false,
        extra,
        env: &[],
        manifest: None,
    })
}

fn campaign(container: &Path, extra: &[&str]) -> Ran {
    exec(Attempt {
        container,
        campaign: true,
        extra,
        env: &[],
        manifest: None,
    })
}

fn canonical_container(root: &Path) -> PathBuf {
    let checkpoint = root.join("checkpoint");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let container = root.join("canonical");
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "target");
    container
}

fn packed_container(root: &Path, canonical: &Path, encoding: &str) -> PathBuf {
    let packed = root.join(format!("pack-{encoding}"));
    let compiled = Command::new(env!("CARGO_BIN_EXE_larql"))
        .args(["vindex3", "represent"])
        .arg(canonical)
        .arg("--output")
        .arg(&packed)
        .args(["--encoding", encoding])
        .output()
        .expect("run larql");
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    packed
}

fn objects_from_pack(execution: &Value) -> Vec<&str> {
    execution["objects"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|o| o["from_pack"] == true)
        .map(|o| o["encoding"].as_str().unwrap())
        .collect()
}

fn pinned_representations(execution: &Value) -> Vec<&str> {
    execution["realizations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["representation"].as_str().unwrap())
        .collect()
}

/// The check that started this: a plugin changing the output, with the same
/// files read. A diagnostic run may do it and its record must say so; a campaign
/// refuses it before the plugin's code loads.
#[test]
fn a_plugin_that_changes_the_output_is_named_by_the_record_and_refused_by_a_campaign() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture::glimmer_container(root.path());
    let dylib = fixture::fixture_dylib();
    let dylib = dylib.to_str().unwrap();
    let reference = ["--backend", "reference"];
    let tampered = [
        "--backend",
        "reference",
        "--plugin",
        dylib,
        "--lowering",
        TAMPER,
    ];

    let control = diagnostic(&container, &reference);
    assert_eq!(control.code, Some(0), "{}", control.err);
    let hostile = diagnostic(&container, &tampered);
    assert_eq!(hostile.code, Some(0), "{}", hostile.err);
    assert_ne!(control.generated(), hostile.generated());

    let (seen, plain) = (hostile.execution(), control.execution());
    assert_eq!(seen["provider"], TAMPER);
    assert_eq!(seen["pinned_lowering"][0], TAMPER);
    assert_eq!(seen["engine"], "tamper");
    assert_eq!(seen["requested"]["lowering"], TAMPER);
    assert_eq!(seen["requested"]["plugins"][0]["path"], dylib);
    assert!(seen["requested"]["plugins"][0]["sha256"].is_string());
    assert_ne!(seen["execution_digest"], plain["execution_digest"]);
    assert_ne!(plain["provider"], TAMPER);
    assert!(plain["requested"]["plugins"].as_array().unwrap().is_empty());

    let refused = campaign(&container, &tampered);
    assert_ne!(refused.code, Some(0));
    assert!(refused.err.contains("--plugin"), "{}", refused.err);
    assert!(refused.err.contains(EVIDENCE_REFUSAL), "{}", refused.err);
    assert!(!refused.err.contains("codecs ["), "the plugin was loaded");
    assert!(refused.stream.is_empty(), "nothing was to be recorded");
}

#[test]
fn a_campaign_refuses_a_lowering_or_representation_flag_that_a_diagnostic_run_may_use() {
    let root = tempfile::tempdir().unwrap();
    let canonical = canonical_container(root.path());
    let packed = packed_container(root.path(), &canonical, NVFP4);
    let cases: [(&str, [&str; 4]); 2] = [
        (
            "--lowering",
            ["--backend", "production", "--lowering", "reference/v1"],
        ),
        (
            "--representation",
            ["--backend", "production", "--representation", NVFP4],
        ),
    ];
    for (flag, extra) in cases {
        let refused = campaign(&packed, &extra);
        assert_ne!(refused.code, Some(0));
        assert!(refused.err.contains(flag), "{}", refused.err);
        assert!(refused.err.contains(EVIDENCE_REFUSAL), "{}", refused.err);
        assert!(refused.stream.is_empty());
        let allowed = diagnostic(&packed, &extra);
        assert_eq!(allowed.code, Some(0), "{}", allowed.err);
    }
    let lowered = diagnostic(&packed, &cases[0].1);
    assert_eq!(lowered.execution()["provider"], "reference/v1");
    let represented = diagnostic(&packed, &cases[1].1);
    assert_eq!(objects_from_pack(represented.execution()), [NVFP4]);
}

/// `production-nvfp4` over a container with no NVFP4 used to run canonical bytes
/// under a clean record.
#[test]
fn a_missing_pack_stops_a_campaign_and_a_diagnostic_run_says_what_really_ran() {
    let root = tempfile::tempdir().unwrap();
    let canonical = canonical_container(root.path());
    for backend in ["production-nvfp4", "production-q4k", "production-q8"] {
        for source in ["auto", "stored"] {
            let extra = ["--backend", backend, "--representation-source", source];
            let refused = campaign(&canonical, &extra);
            assert_ne!(refused.code, Some(0), "{backend} {source}");
            assert!(refused.stopped_before_binding(), "{backend} {source}");
            assert!(refused.err.contains(FALLBACK), "{}", refused.err);
            let before = refused.before_binding();
            assert_eq!(before["evidence_class"], "invalid");
            assert!(before["execution_identity"]["problems"][0]
                .as_str()
                .unwrap()
                .contains(FALLBACK));
            let end = refused.record("attempt_end").unwrap();
            assert_eq!(end["evidence_class"], "invalid");
            assert_eq!(end["status"], "error");
        }
        let ran = diagnostic(&canonical, &["--backend", backend]);
        assert_eq!(ran.code, Some(0), "{}", ran.err);
        let execution = ran.execution();
        assert_eq!(execution["requested"]["backend"], backend);
        assert!(execution["requested"]["wanted_encoding"].is_string());
        assert!(objects_from_pack(execution).is_empty(), "{execution}");
        assert_eq!(execution["objects_from_pack"], 0);
        assert_eq!(execution["complete"], false);
        assert_eq!(ran.admission()["evidence_class"], "diagnostic");
        let wanted = execution["requested"]["wanted_encoding"].as_str().unwrap();
        assert!(!pinned_representations(execution).contains(&wanted));
        assert_eq!(ran.record("attempt_end").unwrap()["status"], "completed");
    }
}

/// A pack that is present but widened by the process is the fallback a selection
/// can't show: the same pack is bound, and only the pins and the process differ.
#[test]
fn a_stored_pack_widened_by_the_process_stops_a_campaign_at_admission() {
    let root = tempfile::tempdir().unwrap();
    let canonical = canonical_container(root.path());
    let packed = packed_container(root.path(), &canonical, Q4_K);
    let backend = ["--backend", "production-q4k"];
    let widen = [("LARQL_KQUANT_EXEC", "widen")];

    let direct = campaign(&packed, &backend);
    assert_eq!(direct.code, Some(0), "{}", direct.err);
    assert_eq!(direct.execution()["complete"], true);

    let refused = exec(Attempt {
        container: &packed,
        campaign: true,
        extra: &backend,
        env: &widen,
        manifest: None,
    });
    assert_ne!(refused.code, Some(0));
    assert!(refused.err.contains("widens"), "{}", refused.err);
    assert_eq!(refused.admission()["evidence_class"], "invalid");
    assert_eq!(refused.execution()["complete"], false);
    assert_eq!(objects_from_pack(refused.execution()), [Q4_K]);
    assert_eq!(
        refused.record("attempt_end").unwrap()["evidence_class"],
        "invalid"
    );
    assert!(!refused.stream.iter().any(|r| r["phase"] == "after_run"));

    let recorded = exec(Attempt {
        container: &packed,
        campaign: false,
        extra: &backend,
        env: &widen,
        manifest: None,
    });
    assert_eq!(recorded.code, Some(0), "{}", recorded.err);
    assert_eq!(recorded.execution()["complete"], false);
    assert_ne!(
        recorded.execution()["execution_digest"],
        direct.execution()["execution_digest"]
    );
    assert_eq!(recorded.execution()["process"]["kquant_execution"], "Widen");
}

/// Backends that really differ differ on the record, with no flag involved.
#[test]
fn a_campaign_on_a_real_pack_records_what_each_backend_ran() {
    let root = tempfile::tempdir().unwrap();
    let canonical = canonical_container(root.path());
    let packed = packed_container(root.path(), &canonical, NVFP4);

    let fused = campaign(&packed, &["--backend", "production-nvfp4"]);
    assert_eq!(fused.code, Some(0), "{}", fused.err);
    let q8 = campaign(&packed, &["--backend", "production-nvfp4-q8"]);
    assert_eq!(q8.code, Some(0), "{}", q8.err);
    let bytes = campaign(&packed, &["--backend", "production"]);
    assert_eq!(bytes.code, Some(0), "{}", bytes.err);

    let (seen, plain) = (fused.execution(), bytes.execution());
    assert_eq!(seen["complete"], true, "{seen}");
    assert_eq!(objects_from_pack(seen), [NVFP4]);
    assert!(pinned_representations(seen).contains(&NVFP4));
    assert_eq!(seen["requested"]["wanted_encoding"], NVFP4);
    assert!(objects_from_pack(plain).is_empty(), "{plain}");
    assert!(plain["requested"]["wanted_encoding"].is_null());
    assert_ne!(seen["execution_digest"], plain["execution_digest"]);
    assert_ne!(seen["execution_digest"], q8.execution()["execution_digest"]);
    assert_ne!(seen["provider"], q8.execution()["provider"]);

    // The same request gives the same computation's digest.
    let again = campaign(&packed, &["--backend", "production-nvfp4"]);
    assert_eq!(
        seen["execution_digest"],
        again.execution()["execution_digest"]
    );

    let after = fused
        .stream
        .iter()
        .find(|r| r["record"] == "instance_identity" && r["phase"] == "after_run")
        .expect("an after-run record");
    let ran = after["execution_identity"]["projection_plans_ran"]
        .as_array()
        .unwrap();
    assert!(!ran.is_empty(), "{after}");
    assert!(ran.iter().all(|plan| plan.get("nanos").is_none()));
    assert_eq!(fused.admission()["evidence_class"], "unverified");
}

/// Only macOS can satisfy a manifest: a null volume UUID cannot match one.
#[cfg(target_os = "macos")]
#[test]
fn identity_consistent_requires_the_execution_it_describes() {
    let root = tempfile::tempdir().unwrap();
    let canonical = canonical_container(root.path());
    let backend = ["--backend", "production"];
    let seen = diagnostic(&canonical, &backend);
    let segments: Vec<Value> = seen.before_binding()["instance_identity"]["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            serde_json::json!({
                "declared": s["declared"],
                "segment_sha256_declared": s["index_sha256"],
                "segment_sha256_computed": s["index_sha256"],
                "identity": s["at_open"],
            })
        })
        .collect();
    let digest =
        larql_vindex::format::checksums::sha256_file(&canonical.join("index.json")).unwrap();
    let manifest = root.path().join("manifest.json");
    std::fs::write(
        &manifest,
        serde_json::json!({
            "schema": "larql.verification-manifest/1",
            "phase": "pre",
            "container": { "path": "/c", "volume_uuid": null, "index_json_sha256": digest },
            "verified_wall": 1.0,
            "clock": {},
            "verifier": { "command": "v", "binary": {}, "path_taken": "graph" },
            "segments": segments,
        })
        .to_string(),
    )
    .unwrap();

    let consistent = exec(Attempt {
        container: &canonical,
        campaign: true,
        extra: &backend,
        env: &[],
        manifest: Some(&manifest),
    });
    assert_eq!(consistent.code, Some(0), "{}", consistent.err);
    assert_eq!(
        consistent.admission()["evidence_class"],
        "identity_consistent"
    );
    assert_eq!(consistent.execution()["complete"], true);

    // The same verified files, under a backend whose pack doesn't exist.
    let fallback = exec(Attempt {
        container: &canonical,
        campaign: true,
        extra: &["--backend", "production-nvfp4"],
        env: &[],
        manifest: Some(&manifest),
    });
    assert_ne!(fallback.code, Some(0));
    assert_eq!(fallback.before_binding()["evidence_class"], "invalid");
    assert_eq!(
        fallback.before_binding()["instance_identity"]["identity_diagnostic"],
        "consistent_with_manifest"
    );
}
