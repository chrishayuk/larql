//! MEM-002 I2b: what a run records about the files it executed from, and
//! when a campaign attempt stops.
//!
//! The judgement is pinned on facts built by hand, because no fixture can
//! make `fstat` fail on demand. The end-to-end runs change a real file at a
//! chosen moment through a backend that delegates to the reference one.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use larql_inference::vindex3::OpenedComponent;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::operands::{
    FileIdentity, OperandStore, RepresentationSource, SegmentInstance, StoreInstance,
};
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;
use serde_json::{json, Value};

use super::super::generate::admission::{
    refuse_evidence_outside_the_curve, EvidenceRequest, RunMode,
};
use super::super::generate::instance_evidence::{
    compare_with_manifest, evidence_class, instance_block, instance_line, judge,
    post_admission_reads, scope_block, EvidenceClass, IdentityDiagnostic, InstanceJudgement,
    ManifestComparison, VerificationManifest, BINDING_IMPLEMENTATION, MANIFEST_PATH_TAKEN,
    MANIFEST_PHASE, MANIFEST_SCHEMA,
};
use super::super::generate::run_residency_curve;
use super::super::plugins::Plugins;
use super::super::prepare::prepare;
use super::super::{run, EncodeArgs, ExecArgs, ExecBackend, Vindex3Command};
use super::fixture_dir;

const DIGEST: &str = "ddd9";
const DECLARED: &str = "segments/decoder.seg";
const UUID: &str = "6DBA87E5-71E8-4161-BBBA-7B3B8914C300";
const SEGMENT_HASH: &str = "abc";
const NEW_TOKENS: usize = 2;
/// A clock step every supported filesystem's timestamps can show.
const TIMESTAMP_STEP: std::time::Duration = std::time::Duration::from_millis(50);

fn file_identity() -> FileIdentity {
    FileIdentity {
        platform: "test",
        volume_uuid: Some(UUID.to_string()),
        volume_uuid_source: Some("test"),
        inode: 7,
        size: 100,
        mtime_ns: 1_000,
        ctime_ns: 2_000,
        nlink: 1,
        device: 3,
    }
}

fn segment(
    at_open: Result<FileIdentity, String>,
    now: Result<FileIdentity, String>,
) -> SegmentInstance {
    SegmentInstance {
        object: "decoder".to_string(),
        declared: DECLARED.to_string(),
        index_sha256: SEGMENT_HASH.to_string(),
        path: PathBuf::from("/c/segments/decoder.seg"),
        mapped: true,
        copied: false,
        at_open,
        now,
    }
}

fn report(segments: Vec<SegmentInstance>) -> StoreInstance {
    StoreInstance {
        segments,
        absent_objects: Vec::new(),
        load_count: 0,
        bytes_read: 0,
        mapped_regions: 0,
        mapped_objects: 0,
    }
}

fn manifest_json(identity: Value, computed: &str) -> Value {
    json!({
        "schema": MANIFEST_SCHEMA,
        "phase": MANIFEST_PHASE,
        "container": { "path": "/c", "index_json_sha256": DIGEST, "volume_uuid": UUID },
        "verified_wall": 1.0,
        "clock": {},
        "verifier": { "command": "v", "binary": {}, "path_taken": MANIFEST_PATH_TAKEN },
        "segments": [{
            "declared": DECLARED,
            "segment_sha256_declared": SEGMENT_HASH,
            "segment_sha256_computed": computed,
            "identity": identity,
        }],
    })
}

fn identity_json(identity: &FileIdentity) -> Value {
    json!({
        "volume_uuid": identity.volume_uuid,
        "inode": identity.inode,
        "size": identity.size,
        "mtime_ns": identity.mtime_ns,
        "ctime_ns": identity.ctime_ns,
        "dev": identity.device,
    })
}

fn manifest() -> VerificationManifest {
    let text = manifest_json(identity_json(&file_identity()), SEGMENT_HASH).to_string();
    VerificationManifest::parse(Path::new("m.json"), text.as_bytes()).unwrap()
}

// ── Reading a manifest ──

#[test]
fn a_manifest_is_recorded_by_the_digest_of_the_bytes_parsed() {
    use sha2::{Digest, Sha256};
    let text = manifest_json(identity_json(&file_identity()), SEGMENT_HASH).to_string();
    let read = VerificationManifest::parse(Path::new("m.json"), text.as_bytes()).unwrap();
    let digest: String = Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(read.sha256, digest);
    assert_eq!(read.index_json_sha256, DIGEST);
    assert_eq!(read.segments.len(), 1);
}

#[test]
fn a_manifest_that_is_not_a_pre_campaign_graph_verification_is_refused() {
    let good = manifest_json(identity_json(&file_identity()), SEGMENT_HASH);
    let cases: Vec<(&str, Value)> = vec![
        ("schema", {
            let mut m = good.clone();
            m["schema"] = json!("larql.verification-manifest/9");
            m
        }),
        ("phase", {
            let mut m = good.clone();
            m["phase"] = json!("post");
            m
        }),
        ("verification path", {
            let mut m = good.clone();
            m["verifier"]["path_taken"] = json!("legacy");
            m
        }),
        ("twice", {
            let mut m = good.clone();
            let entry = m["segments"][0].clone();
            m["segments"].as_array_mut().unwrap().push(entry);
            m
        }),
        (
            "not a verification manifest",
            json!({ "schema": MANIFEST_SCHEMA }),
        ),
        ("verified_wall", {
            let mut m = good.clone();
            m.as_object_mut().unwrap().remove("verified_wall");
            m
        }),
        ("clock", {
            let mut m = good.clone();
            m.as_object_mut().unwrap().remove("clock");
            m
        }),
        ("binary", {
            let mut m = good.clone();
            m["verifier"].as_object_mut().unwrap().remove("binary");
            m
        }),
        ("command", {
            let mut m = good.clone();
            m["verifier"].as_object_mut().unwrap().remove("command");
            m
        }),
        ("path", {
            let mut m = good.clone();
            m["container"].as_object_mut().unwrap().remove("path");
            m
        }),
        ("volume_uuid", {
            let mut m = good.clone();
            m["container"]
                .as_object_mut()
                .unwrap()
                .remove("volume_uuid");
            m
        }),
    ];
    for (needle, manifest) in cases {
        let error =
            VerificationManifest::parse(Path::new("m.json"), manifest.to_string().as_bytes())
                .unwrap_err();
        assert!(error.contains(needle), "{needle}: {error}");
    }
    assert!(VerificationManifest::parse(Path::new("m.json"), b"not json").is_err());
    assert!(
        VerificationManifest::read(Path::new("/nonexistent/manifest.json"))
            .unwrap_err()
            .contains("cannot read")
    );
}

// ── One segment against the manifest ──

#[test]
fn a_comparison_matches_on_its_key_and_ignores_the_device() {
    let mut identity = file_identity();
    assert_eq!(
        compare_with_manifest(DECLARED, SEGMENT_HASH, &identity, &manifest()),
        ManifestComparison::Match
    );
    identity.device = 99;
    identity.nlink = 5;
    assert_eq!(
        compare_with_manifest(DECLARED, SEGMENT_HASH, &identity, &manifest()),
        ManifestComparison::Match,
        "neither the device nor the link count is in the manifest's key"
    );
}

#[test]
fn a_comparison_names_every_field_that_differs() {
    let mut identity = file_identity();
    identity.inode += 1;
    identity.size += 1;
    identity.mtime_ns += 1;
    identity.ctime_ns += 1;
    assert_eq!(
        compare_with_manifest(DECLARED, SEGMENT_HASH, &identity, &manifest()),
        ManifestComparison::Mismatch(vec!["inode", "size", "mtime_ns", "ctime_ns"])
    );
    assert_eq!(
        compare_with_manifest(
            "segments/other.seg",
            SEGMENT_HASH,
            &file_identity(),
            &manifest()
        ),
        ManifestComparison::Absent
    );
}

/// No UUID cannot show the same volume, so it never satisfies a manifest.
#[test]
fn a_null_volume_uuid_on_either_side_is_a_mismatch() {
    let mut no_uuid = file_identity();
    no_uuid.volume_uuid = None;
    assert_eq!(
        compare_with_manifest(DECLARED, SEGMENT_HASH, &no_uuid, &manifest()),
        ManifestComparison::Mismatch(vec!["volume_uuid"])
    );
    let text = manifest_json(identity_json(&no_uuid), SEGMENT_HASH).to_string();
    let both_null = VerificationManifest::parse(Path::new("m.json"), text.as_bytes()).unwrap();
    assert_eq!(
        compare_with_manifest(DECLARED, SEGMENT_HASH, &no_uuid, &both_null),
        ManifestComparison::Mismatch(vec!["volume_uuid"])
    );
}

/// A manifest's hash must be the hash this container's index declares: one
/// that agrees only with itself verified some other file.
#[test]
fn a_manifest_hash_the_index_does_not_declare_verifies_nothing() {
    assert_eq!(
        compare_with_manifest(DECLARED, "another", &file_identity(), &manifest()),
        ManifestComparison::Mismatch(vec!["segment_sha256"])
    );
}

/// A manifest whose own hashes disagree verified nothing, whatever the
/// identity says.
#[test]
fn a_manifest_whose_hashes_disagree_verifies_nothing() {
    for computed in ["other", ""] {
        let text = manifest_json(identity_json(&file_identity()), computed).to_string();
        let manifest = VerificationManifest::parse(Path::new("m.json"), text.as_bytes()).unwrap();
        assert_eq!(
            compare_with_manifest(DECLARED, SEGMENT_HASH, &file_identity(), &manifest),
            ManifestComparison::Mismatch(vec!["segment_sha256"])
        );
    }
}

// ── The judgement ──

#[test]
fn a_diagnostic_run_is_never_campaign_evidence_and_a_campaign_is_classed_by_its_facts() {
    let all = [
        IdentityDiagnostic::CaptureFailed,
        IdentityDiagnostic::Changed,
        IdentityDiagnostic::Mismatch,
        IdentityDiagnostic::Unverified,
        IdentityDiagnostic::ConsistentWithManifest,
    ];
    for diagnostic in all {
        assert_eq!(
            evidence_class(RunMode::Diagnostic, diagnostic),
            EvidenceClass::Diagnostic
        );
    }
    let campaign = all.map(|d| evidence_class(RunMode::Campaign, d));
    assert_eq!(
        campaign,
        [
            EvidenceClass::Invalid,
            EvidenceClass::Invalid,
            EvidenceClass::Invalid,
            EvidenceClass::Unverified,
            EvidenceClass::IdentityConsistent,
        ]
    );
    assert_eq!(
        all.map(IdentityDiagnostic::name),
        [
            "capture_failed",
            "changed",
            "mismatch",
            "unverified",
            "consistent_with_manifest"
        ]
    );
    assert_eq!(
        [
            EvidenceClass::Diagnostic,
            EvidenceClass::Unverified,
            EvidenceClass::Invalid,
            EvidenceClass::IdentityConsistent,
        ]
        .map(EvidenceClass::name),
        ["diagnostic", "unverified", "invalid", "identity_consistent"]
    );
}

#[test]
fn a_campaign_with_no_manifest_is_unverified_and_may_run() {
    let facts = report(vec![segment(Ok(file_identity()), Ok(file_identity()))]);
    let judged = judge(RunMode::Campaign, &facts, None, Some(DIGEST));
    assert_eq!(judged.diagnostic, IdentityDiagnostic::Unverified);
    assert_eq!(judged.class, EvidenceClass::Unverified);
    assert!(!judged.invalid());
    assert!(judged.reason().contains("no verification manifest"));
}

/// The worst category names the diagnostic, and no problem goes unnamed.
#[test]
fn the_reasons_name_every_problem_in_every_category() {
    let mut moved = file_identity();
    moved.ctime_ns += 1;
    let facts = report(vec![
        segment(Err("fstat failed".to_string()), Ok(file_identity())),
        SegmentInstance {
            declared: "segments/moved.seg".to_string(),
            ..segment(Ok(file_identity()), Ok(moved))
        },
        SegmentInstance {
            declared: "segments/unlisted.seg".to_string(),
            ..segment(Ok(file_identity()), Ok(file_identity()))
        },
    ]);
    let judged = judge(RunMode::Campaign, &facts, Some(&manifest()), Some(DIGEST));
    assert_eq!(judged.diagnostic, IdentityDiagnostic::CaptureFailed);
    let reason = judged.reason();
    for named in ["fstat failed", "moved.seg", "unlisted.seg"] {
        assert!(reason.contains(named), "{named}: {reason}");
    }
}

#[test]
fn a_campaign_agreeing_with_its_manifest_is_consistent_and_claims_no_more() {
    let facts = report(vec![segment(Ok(file_identity()), Ok(file_identity()))]);
    let judged = judge(RunMode::Campaign, &facts, Some(&manifest()), Some(DIGEST));
    assert_eq!(
        judged.diagnostic,
        IdentityDiagnostic::ConsistentWithManifest
    );
    assert_eq!(judged.class, EvidenceClass::IdentityConsistent);
    assert!(judged.reasons.is_empty());
    assert!(!judged.invalid());
}

/// Each failure stops a campaign, the worst one names the diagnostic, and
/// the same facts only annotate diagnostic execution.
#[test]
fn each_identity_failure_invalidates_a_campaign_and_only_annotates_a_diagnostic() {
    let mut moved = file_identity();
    moved.ctime_ns += 1;
    let mut unlinked = file_identity();
    unlinked.nlink = 0;
    let failed: Result<FileIdentity, String> = Err("fstat failed".to_string());
    let cases: Vec<(StoreInstance, Option<&str>, IdentityDiagnostic, &str)> = vec![
        (
            report(vec![]),
            Some(DIGEST),
            IdentityDiagnostic::CaptureFailed,
            "no segment",
        ),
        (
            report(vec![segment(failed.clone(), Ok(file_identity()))]),
            Some(DIGEST),
            IdentityDiagnostic::CaptureFailed,
            "fstat failed",
        ),
        (
            report(vec![segment(Ok(file_identity()), failed.clone())]),
            Some(DIGEST),
            IdentityDiagnostic::CaptureFailed,
            "fstat failed",
        ),
        (
            report(vec![segment(Ok(file_identity()), Ok(moved.clone()))]),
            Some(DIGEST),
            IdentityDiagnostic::Changed,
            "ctime_ns",
        ),
        (
            report(vec![segment(Ok(file_identity()), Ok(unlinked))]),
            Some(DIGEST),
            IdentityDiagnostic::Changed,
            "nlink",
        ),
        (
            report(vec![segment(Ok(moved.clone()), Ok(moved))]),
            Some(DIGEST),
            IdentityDiagnostic::Mismatch,
            "ctime_ns",
        ),
        (
            report(vec![SegmentInstance {
                declared: "segments/other.seg".to_string(),
                ..segment(Ok(file_identity()), Ok(file_identity()))
            }]),
            Some(DIGEST),
            IdentityDiagnostic::Mismatch,
            "not in the manifest",
        ),
        (
            report(vec![segment(Ok(file_identity()), Ok(file_identity()))]),
            Some("other"),
            IdentityDiagnostic::Mismatch,
            "index.json",
        ),
        (
            report(vec![segment(Ok(file_identity()), Ok(file_identity()))]),
            None,
            IdentityDiagnostic::Mismatch,
            "index.json",
        ),
    ];
    for (facts, digest, diagnostic, needle) in cases {
        let judged = judge(RunMode::Campaign, &facts, Some(&manifest()), digest);
        assert_eq!(judged.diagnostic, diagnostic, "{needle}");
        assert!(judged.invalid(), "{needle}");
        assert!(judged.reason().contains(needle), "{}", judged.reason());
        let annotated = judge(RunMode::Diagnostic, &facts, Some(&manifest()), digest);
        assert_eq!(annotated.diagnostic, diagnostic);
        assert_eq!(annotated.class, EvidenceClass::Diagnostic);
        assert!(!annotated.invalid(), "{needle}");
    }
}

// ── What a record says ──

#[test]
fn the_record_separates_what_is_monitored_from_what_is_not_bound() {
    let scope = scope_block();
    let not_bound: Vec<&str> = scope["not_bound"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for read in [
        "index.json",
        "system_graph.json",
        "plan_time_segment_headers",
    ] {
        assert!(not_bound.contains(&read), "{read}");
    }
    let plan = scope["plan_to_bind"].as_str().unwrap();
    assert!(plan.contains("does not show the planner read identical bytes"));
    assert!(plan.contains("does not establish complete verification-to-execution identity"));
    assert!(scope["content"]
        .as_str()
        .unwrap()
        .contains("does not prove content"));
    assert!(scope["distributed_workers"]
        .as_str()
        .unwrap()
        .starts_with("unsupported"));
}

#[test]
fn the_block_carries_every_segments_facts_and_comparison() {
    let mut facts = report(vec![segment(Ok(file_identity()), Ok(file_identity()))]);
    facts.absent_objects = vec!["elsewhere".to_string()];
    facts.bytes_read = 12;
    let manifest = manifest();
    let judged = judge(RunMode::Campaign, &facts, Some(&manifest), Some(DIGEST));
    let block = instance_block(&facts, Some(&manifest), &judged);
    assert_eq!(block["identity_diagnostic"], "consistent_with_manifest");
    assert_eq!(block["manifest"]["sha256"], manifest.sha256.as_str());
    assert_eq!(block["absent_objects"][0], "elsewhere");
    assert_eq!(block["counters"]["bytes_read"], 12);
    assert!(block["scope"]["not_bound"].is_array());
    let entry = &block["segments"][0];
    assert_eq!(entry["declared"], DECLARED);
    assert_eq!(entry["index_sha256"], SEGMENT_HASH);
    assert_eq!(entry["use"]["mapped"], true);
    assert_eq!(entry["use"]["copied"], false);
    assert_eq!(entry["manifest"], "match");
    assert_eq!(entry["at_open"]["inode"], 7);
    assert_eq!(entry["at_open"]["nlink"], 1);
    assert_eq!(entry["at_open"]["platform"], "test");
    assert_eq!(entry["now"]["volume_uuid"], UUID);

    let mut moved = file_identity();
    moved.ctime_ns += 1;
    let facts = report(vec![segment(Err("gone".to_string()), Ok(moved))]);
    let judged = judge(RunMode::Campaign, &facts, Some(&manifest), Some(DIGEST));
    let entry = &instance_block(&facts, Some(&manifest), &judged)["segments"][0];
    assert_eq!(entry["at_open"]["error"], "gone");
    assert_eq!(entry["manifest"], "mismatch");
    assert_eq!(entry["manifest_mismatch_fields"][0], "ctime_ns");

    let facts = report(vec![segment(Ok(file_identity()), Err("gone".to_string()))]);
    let judged = judge(RunMode::Campaign, &facts, Some(&manifest), Some(DIGEST));
    let entry = &instance_block(&facts, Some(&manifest), &judged)["segments"][0];
    assert_eq!(entry["manifest_mismatch_fields"][0], "capture");

    let facts = report(vec![segment(Ok(file_identity()), Ok(file_identity()))]);
    let judged = judge(RunMode::Diagnostic, &facts, None, Some(DIGEST));
    let block = instance_block(&facts, None, &judged);
    assert_eq!(block["manifest"], Value::Null);
    assert_eq!(block["segments"][0]["manifest"], "not_supplied");

    let line = instance_line("admission", &judged);
    assert!(
        line.contains("unverified") && line.contains("diagnostic"),
        "{line}"
    );
    let clean = InstanceJudgement {
        diagnostic: IdentityDiagnostic::ConsistentWithManifest,
        class: EvidenceClass::IdentityConsistent,
        reasons: Vec::new(),
    };
    assert!(!instance_line("admission", &clean).contains('—'));
}

#[test]
fn the_post_admission_counters_are_differences_and_never_underflow() {
    let mut before = report(vec![]);
    before.load_count = 5;
    before.bytes_read = 50;
    before.mapped_regions = 2;
    before.mapped_objects = 1;
    let mut after = before.clone();
    after.bytes_read = 60;
    after.mapped_regions = 3;
    let moved = post_admission_reads(&before, &after);
    assert_eq!(moved["load_count"], 0);
    assert_eq!(moved["bytes_read"], 10);
    assert_eq!(moved["mapped_regions"], 1);
    assert_eq!(moved["mapped_objects"], 0);
    assert_eq!(post_admission_reads(&after, &before)["bytes_read"], 0);
}

// ── Refusals before anything runs ──

fn exec_args(container: &Path) -> ExecArgs {
    ExecArgs {
        container: container.to_path_buf(),
        component: "target".to_string(),
        tokens: "1,2".to_string(),
        dump_layers: None,
        resume: false,
        backend: ExecBackend::Production,
        representation_source: "auto".to_string(),
        generate: Some(NEW_TOKENS),
        speculate: None,
        stop_at_eos: false,
        residency_curve: true,
        repeat: 1,
        warmup: 0,
        unquiet_ok: true,
        residency_regions: Some(PathBuf::from("r")),
        sealed_latency: Some(PathBuf::from("s")),
        residency_campaign: true,
        verification_manifest: None,
        admission_threshold_bytes: None,
        admission_basis: None,
        expert_access: "demand".to_string(),
        logit_dump: None,
        bank: None,
        dump_dir: None,
        draft_depth: None,
        profile: false,
        plugin: Default::default(),
    }
}

#[test]
fn a_campaign_refuses_a_transient_representation_source() {
    let mut args = exec_args(Path::new("c"));
    args.representation_source = "transient".to_string();
    let refusal = refuse_evidence_outside_the_curve(&args, false).unwrap_err();
    assert!(refusal.contains("transient"), "{refusal}");
    assert!(refusal.contains("cannot be bound or verified"), "{refusal}");

    // Diagnostic execution may use it, and `stored` is bound like any other.
    args.residency_campaign = false;
    assert!(refuse_evidence_outside_the_curve(&args, false).is_ok());
    args.residency_campaign = true;
    args.representation_source = "stored".to_string();
    assert!(refuse_evidence_outside_the_curve(&args, false).is_ok());
}

#[test]
fn a_manifest_on_a_run_that_never_reaches_the_curve_is_refused() {
    let mut args = exec_args(Path::new("c"));
    args.residency_campaign = false;
    args.residency_regions = None;
    args.sealed_latency = None;
    args.residency_curve = false;
    args.verification_manifest = Some(PathBuf::from("m"));
    assert!(refuse_evidence_outside_the_curve(&args, false)
        .unwrap_err()
        .contains("--verification-manifest"));
}

// ── End to end ──

/// A dense fixture's container, with the temp dir keeping it alive.
fn encoded() -> (tempfile::TempDir, PathBuf) {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    (dir, out)
}

fn opened(container: &Path) -> (ComponentOpPlan, OperandStore) {
    let OpenedComponent { plan, store, .. } = prepare(
        container,
        "target",
        ExecBackend::Production,
        RepresentationSource::Auto,
        &Plugins::load(&Default::default()).unwrap(),
    )
    .unwrap();
    (plan, store)
}

/// When a [`Hooked`] backend changes the file, in the run's own order.
#[derive(Clone, Copy, PartialEq, Eq)]
enum When {
    /// During preparation, so after the before-binding check.
    Binding,
    /// During the first pass, so after admission.
    Passes,
}

/// The reference backend, plus one action at one moment of a real run.
struct Hooked<F: Fn() + Sync> {
    when: When,
    action: F,
    fired: AtomicBool,
}

impl<F: Fn() + Sync> Hooked<F> {
    fn new(when: When, action: F) -> Self {
        Self {
            when,
            action,
            fired: AtomicBool::new(false),
        }
    }

    fn fire(&self, now: When) {
        if self.when == now && !self.fired.swap(true, Ordering::SeqCst) {
            (self.action)();
        }
    }
}

mod hooked_backend {
    use super::*;
    use larql_vindex::error::VindexError;
    use larql_vindex::format::vindex3::opplan::exec::backend::*;
    use larql_vindex::format::vindex3::opplan::exec::lowering::LoweringIdentity;
    use larql_vindex::format::vindex3::opplan::exec::realization::{
        RepresentationFacts, Selection, SelectionRefusal,
    };
    use larql_vindex::format::vindex3::opplan::planned::PlannedOperand;

    impl<F: Fn() + Sync> PlanBackend for Hooked<F> {
        fn name(&self) -> &str {
            ReferenceBackend.name()
        }
        fn identity(&self) -> LoweringIdentity {
            ReferenceBackend.identity()
        }
        fn select(
            &self,
            operand: &PlannedOperand,
            facts: &RepresentationFacts,
        ) -> Result<Selection, Box<SelectionRefusal>> {
            self.fire(When::Binding);
            ReferenceBackend.select(operand, facts)
        }
        fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
            self.fire(When::Passes);
            ReferenceBackend.embed(table, hidden, token, scale)
        }
        fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
            ReferenceBackend.norm(call)
        }
        fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
            ReferenceBackend.project(call)
        }
        fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
            ReferenceBackend.attention(call)
        }
        fn attention_step(
            &self,
            call: AttentionStepCall<'_>,
        ) -> Result<AttentionStepOut, VindexError> {
            ReferenceBackend.attention_step(call)
        }
        fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
            ReferenceBackend.ffn(call)
        }
        fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
            ReferenceBackend.routed_ffn(call)
        }
        fn output_head(
            &self,
            projection: WeightSlice<'_>,
            vocab: usize,
            hidden: usize,
            x: &[f32],
            multiplier: Option<f64>,
            softcapping: Option<f32>,
        ) -> Result<Vec<f32>, VindexError> {
            ReferenceBackend.output_head(projection, vocab, hidden, x, multiplier, softcapping)
        }
        fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
            ReferenceBackend.residual_add(acc, delta)
        }
    }
}

struct Run {
    regions: PathBuf,
    sealed: PathBuf,
    _dir: tempfile::TempDir,
}

fn read_lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .expect("the record was written")
        .lines()
        .map(|l| serde_json::from_str(l).expect("one JSON value per line"))
        .collect()
}

fn kinds(stream: &[Value]) -> Vec<(&str, &str)> {
    stream
        .iter()
        .filter(|v| v["record"] != "region")
        .map(|v| {
            (
                v["record"].as_str().unwrap(),
                v["phase"].as_str().unwrap_or(""),
            )
        })
        .collect()
}

fn last(stream: &[Value], record: &str) -> Value {
    stream
        .iter()
        .rev()
        .find(|v| v["record"] == record)
        .unwrap_or_else(|| panic!("no {record} record"))
        .clone()
}

fn sealed_kinds(path: &Path) -> Vec<String> {
    read_lines(path)
        .iter()
        .map(|v| v["record"].as_str().unwrap().to_string())
        .collect()
}

/// A manifest of what `store`'s descriptors say now, written to `dir`, as a
/// separate verification tool would have written it.
fn write_manifest(
    dir: &Path,
    container: &Path,
    store: &OperandStore,
    mutate: impl Fn(&mut Value),
) -> PathBuf {
    let digest =
        larql_vindex::format::checksums::sha256_file(&container.join("index.json")).unwrap();
    let segments: Vec<Value> = store
        .instance_report()
        .segments
        .iter()
        .map(|s| {
            json!({
                "declared": s.declared,
                "segment_sha256_declared": s.index_sha256,
                "segment_sha256_computed": s.index_sha256,
                "identity": identity_json(s.at_open.as_ref().unwrap()),
            })
        })
        .collect();
    let mut manifest = json!({
        "schema": MANIFEST_SCHEMA,
        "phase": MANIFEST_PHASE,
        "container": { "path": "/c", "volume_uuid": UUID, "index_json_sha256": digest },
        "verified_wall": 1.0,
        "clock": {},
        "verifier": { "command": "v", "binary": {}, "path_taken": MANIFEST_PATH_TAKEN },
        "segments": segments,
    });
    mutate(&mut manifest);
    let path = dir.join("manifest.json");
    std::fs::write(&path, manifest.to_string()).unwrap();
    path
}

/// Run the curve on `backend` over an already opened store.
#[allow(clippy::too_many_arguments)]
fn curve<B: PlanBackend>(
    backend: &B,
    plan: &ComponentOpPlan,
    store: &OperandStore,
    container: &Path,
    run: &Run,
    campaign: bool,
    manifest: Option<&Path>,
    threshold: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    run_residency_curve(
        backend,
        "test",
        &[1, 2],
        NEW_TOKENS,
        plan,
        store,
        1,
        0,
        true,
        "demand",
        container,
        EvidenceRequest {
            campaign,
            regions: Some(&run.regions),
            sealed_latency: Some(&run.sealed),
            manifest,
            threshold_bytes: threshold,
            basis: threshold.map(|_| "docs/frozen.md §3 @ abc123"),
        },
    )
}

fn evidence_paths() -> Run {
    let dir = tempfile::tempdir().unwrap();
    Run {
        regions: dir.path().join("regions.jsonl"),
        sealed: dir.path().join("sealed.jsonl"),
        _dir: dir,
    }
}

const FULL_RUN: [(&str, &str); 5] = [
    ("instance_identity", "before_binding"),
    ("observation", "binding"),
    ("admission", ""),
    ("instance_identity", "after_run"),
    ("observation", "after_run"),
];

fn assert_full_run(stream: &[Value]) {
    let order = kinds(stream);
    assert_eq!(&order[..FULL_RUN.len()], &FULL_RUN[..]);
    assert_eq!(order[FULL_RUN.len()], ("attempt_end", ""));
}

/// A clock reading before binding, subtracted from the binding observation's
/// start, would give away the sealed bind time.
fn assert_undated_before_the_bind(stream: &[Value]) {
    for record in stream {
        if record["record"] == "observation" {
            return;
        }
        for field in ["clock", "wall", "started_wall", "started_mono_ns"] {
            assert!(
                record.get(field).is_none(),
                "{} before the bind carries {field}",
                record["record"]
            );
        }
    }
}

/// A campaign with no manifest runs, and says at every point that it is
/// unverified: success does not promote it.
#[test]
#[cfg_attr(not(unix), ignore = "this platform has no descriptor identity")]
fn a_campaign_with_no_manifest_runs_and_stays_unverified() {
    let (_dir, container) = encoded();
    let (plan, store) = opened(&container);
    let run = evidence_paths();
    curve(
        &ReferenceBackend,
        &plan,
        &store,
        &container,
        &run,
        true,
        None,
        None,
    )
    .expect("an unverified attempt may execute");
    let stream = read_lines(&run.regions);
    assert_full_run(&stream);
    assert_undated_before_the_bind(&stream);
    let admission = last(&stream, "admission");
    assert_eq!(admission["evidence_class"], "unverified");
    assert_eq!(admission["representation_source"], "auto");
    assert_eq!(admission["binding_implementation"], BINDING_IMPLEMENTATION);
    assert_eq!(
        admission["instance_identity"]["identity_diagnostic"],
        "unverified"
    );
    let end = last(&stream, "attempt_end");
    assert_eq!(end["status"], "completed");
    assert_eq!(end["evidence_class"], "unverified");
    assert!(sealed_kinds(&run.sealed).iter().any(|k| k == "step"));

    // What execution read is in the record: copied segments, none mapped,
    // and no store read after admission.
    let after = last(&stream, "instance_identity");
    let block = &after["instance_identity"];
    let segments = block["segments"].as_array().unwrap();
    assert_eq!(segments.len(), store.selection().len());
    assert!(segments.iter().any(|s| s["use"]["copied"] == true));
    assert!(segments.iter().all(|s| s["use"]["mapped"] == false));
    assert_eq!(block["post_admission_reads"]["load_count"], 0);
    assert_eq!(block["post_admission_reads"]["bytes_read"], 0);
    assert_eq!(block["post_admission_reads"]["mapped_regions"], 0);
    assert!(segments[0]["at_open"]["inode"].as_u64().unwrap() > 0);
    assert_eq!(segments[0]["at_open"]["platform"], std::env::consts::OS);
}

/// On a platform with a volume UUID, a manifest made from the same files
/// gives the one class that may go forward, and the admission threshold
/// still decides separately: rejected, it runs and its latency is sealed.
#[test]
#[cfg(target_os = "macos")]
fn a_campaign_agreeing_with_its_manifest_is_consistent_and_a_threshold_rejection_still_runs() {
    let (dir, container) = encoded();
    let (plan, store) = opened(&container);
    let manifest = write_manifest(dir.path(), &container, &store, |_| {});
    let run = evidence_paths();
    curve(
        &ReferenceBackend,
        &plan,
        &store,
        &container,
        &run,
        true,
        Some(&manifest),
        Some(1),
    )
    .expect("a rejected attempt still completes");
    let stream = read_lines(&run.regions);
    assert_full_run(&stream);
    let admission = last(&stream, "admission");
    assert_eq!(admission["evidence_class"], "identity_consistent");
    assert_eq!(
        admission["outcome"], "witness_failed",
        "the fixture maps nothing"
    );
    assert_eq!(
        admission["instance_identity"]["manifest"]["sha256"],
        larql_vindex::format::checksums::sha256_file(&manifest)
            .unwrap()
            .as_str()
    );
    let end = last(&stream, "attempt_end");
    assert_eq!(end["status"], "completed");
    assert_eq!(end["evidence_class"], "identity_consistent");
    assert!(sealed_kinds(&run.sealed).iter().any(|k| k == "step"));
}

/// A platform with no UUID cannot satisfy a manifest, so it never reports a
/// consistent identity it cannot show.
#[test]
#[cfg(target_os = "linux")]
fn a_linux_campaign_cannot_satisfy_a_manifest() {
    let (dir, container) = encoded();
    let (plan, store) = opened(&container);
    let manifest = write_manifest(dir.path(), &container, &store, |_| {});
    let run = evidence_paths();
    let error = curve(
        &ReferenceBackend,
        &plan,
        &store,
        &container,
        &run,
        true,
        Some(&manifest),
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("volume_uuid"), "{error}");
}

/// Wrong files stop the attempt before the bind timer starts: no weight is
/// read, nothing is observed, and the reason is on the record.
#[test]
#[cfg_attr(not(unix), ignore = "this platform has no descriptor identity")]
fn a_manifest_mismatch_stops_a_campaign_before_binding() {
    let (dir, container) = encoded();
    let (plan, store) = opened(&container);
    let mutations: [(&str, fn(&mut Value)); 3] = [
        ("inode", |m| {
            m["segments"][0]["identity"]["inode"] = json!(1)
        }),
        ("not in the manifest", |m| {
            m["segments"] = json!([]);
        }),
        ("index.json", |m| {
            m["container"]["index_json_sha256"] = json!("other")
        }),
    ];
    for (needle, mutate) in mutations {
        let manifest = write_manifest(dir.path(), &container, &store, mutate);
        let run = evidence_paths();
        let error = curve(
            &ReferenceBackend,
            &plan,
            &store,
            &container,
            &run,
            true,
            Some(&manifest),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(needle), "{needle}: {error}");
        let stream = read_lines(&run.regions);
        assert_eq!(
            kinds(&stream),
            [("instance_identity", "before_binding"), ("attempt_end", "")]
        );
        let end = last(&stream, "attempt_end");
        assert_eq!(end["status"], "error");
        assert_eq!(end["evidence_class"], "invalid");
        assert!(end["error"].as_str().unwrap().contains(needle));
        assert_eq!(store.load_count(), 0, "no weight was read");
        assert_eq!(sealed_kinds(&run.sealed), ["header"]);
        assert_digest_on_the_end(&end, &run.sealed);
    }
}

/// A supplied manifest that cannot be read is refused on the record, in
/// either mode: nobody who gave evidence silently gets none.
#[test]
fn an_unreadable_manifest_refuses_the_run_on_the_record() {
    let (_dir, container) = encoded();
    let (plan, store) = opened(&container);
    for campaign in [true, false] {
        let run = evidence_paths();
        let error = curve(
            &ReferenceBackend,
            &plan,
            &store,
            &container,
            &run,
            campaign,
            Some(Path::new("/nonexistent/manifest.json")),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("cannot read"), "{error}");
        let stream = read_lines(&run.regions);
        assert_eq!(kinds(&stream), [("attempt_end", "")]);
        let class = if campaign { "invalid" } else { "diagnostic" };
        assert_eq!(stream[0]["evidence_class"], class);
        assert_digest_on_the_end(&stream[0], &run.sealed);
    }
}

/// Diagnostic execution records the same disagreement and carries on, and
/// never presents itself as campaign evidence.
#[test]
#[cfg_attr(not(unix), ignore = "this platform has no descriptor identity")]
fn a_diagnostic_run_records_a_mismatch_and_continues() {
    let (dir, container) = encoded();
    let (plan, store) = opened(&container);
    let manifest = write_manifest(dir.path(), &container, &store, |m| {
        m["segments"][0]["identity"]["inode"] = json!(1)
    });
    let run = evidence_paths();
    curve(
        &ReferenceBackend,
        &plan,
        &store,
        &container,
        &run,
        false,
        Some(&manifest),
        None,
    )
    .expect("diagnostic execution continues");
    let stream = read_lines(&run.regions);
    assert_full_run(&stream);
    let admission = last(&stream, "admission");
    assert_eq!(admission["evidence_class"], "diagnostic");
    assert_eq!(
        admission["instance_identity"]["identity_diagnostic"],
        "mismatch"
    );
    assert_eq!(last(&stream, "attempt_end")["status"], "completed");
}

/// A stopped attempt's sealed file is finished and shown untouched on the
/// record: it has no `after_run` header to carry the digest.
fn assert_digest_on_the_end(end: &Value, sealed: &Path) {
    assert_eq!(
        end["sealed_latency"]["sha256"],
        larql_vindex::format::checksums::sha256_file(sealed)
            .unwrap()
            .as_str()
    );
}

/// Rewrite one byte of the file in place, after a clock step the timestamps
/// can show.
fn write_in_place(path: &Path) {
    std::thread::sleep(TIMESTAMP_STEP);
    let mut bytes = std::fs::read(path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] = !bytes[last];
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, &bytes))
        .unwrap();
}

/// Replace the file by rename, after a clock step the timestamps can show.
fn replace_by_rename(path: &Path) {
    std::thread::sleep(TIMESTAMP_STEP);
    let replacement = path.with_extension("replacement");
    std::fs::copy(path, &replacement).unwrap();
    std::fs::rename(&replacement, path).unwrap();
}

fn first_segment(store: &OperandStore) -> PathBuf {
    store.instance_report().segments.remove(0).path
}

/// A change after the before-binding check and before admission stops the
/// attempt at admission: decided and recorded, and no timed pass runs.
#[test]
#[cfg_attr(not(unix), ignore = "this platform has no descriptor identity")]
fn a_change_found_at_admission_stops_a_campaign_before_any_pass() {
    let (_dir, container) = encoded();
    let (plan, store) = opened(&container);
    let path = first_segment(&store);
    let backend = Hooked::new(When::Binding, || write_in_place(&path));
    let run = evidence_paths();
    let error = curve(&backend, &plan, &store, &container, &run, true, None, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("changed"), "{error}");
    let stream = read_lines(&run.regions);
    assert_eq!(
        kinds(&stream),
        [
            ("instance_identity", "before_binding"),
            ("observation", "binding"),
            ("admission", ""),
            ("attempt_end", ""),
        ]
    );
    assert_eq!(last(&stream, "admission")["evidence_class"], "invalid");
    let end = last(&stream, "attempt_end");
    assert_eq!(end["evidence_class"], "invalid");
    assert_digest_on_the_end(&end, &run.sealed);
    assert!(
        !sealed_kinds(&run.sealed).iter().any(|k| k == "step"),
        "no timed pass ran"
    );
}

/// A change found after the run keeps the whole attempt, marks it invalid,
/// and exits non-zero.
#[test]
#[cfg_attr(not(unix), ignore = "this platform has no descriptor identity")]
fn a_change_found_after_the_run_invalidates_the_kept_attempt() {
    type Change = fn(&Path);
    let changes: [(Change, &str); 2] = [(write_in_place, "ctime_ns"), (replace_by_rename, "nlink")];
    for (change, field) in changes {
        let (_dir, container) = encoded();
        let (plan, store) = opened(&container);
        let path = first_segment(&store);
        let backend = Hooked::new(When::Passes, || change(&path));
        let run = evidence_paths();
        let error = curve(&backend, &plan, &store, &container, &run, true, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains(field), "{field}: {error}");
        let stream = read_lines(&run.regions);
        assert_full_run(&stream);
        assert_eq!(last(&stream, "admission")["evidence_class"], "unverified");
        let after = last(&stream, "instance_identity");
        assert_eq!(after["phase"], "after_run");
        assert_eq!(after["evidence_class"], "invalid");
        assert_eq!(after["instance_identity"]["identity_diagnostic"], "changed");
        let end = last(&stream, "attempt_end");
        assert_eq!(end["status"], "error");
        assert_eq!(end["evidence_class"], "invalid");
        assert!(
            sealed_kinds(&run.sealed).iter().any(|k| k == "step"),
            "the passes' latency is kept"
        );
        assert!(
            stream
                .iter()
                .any(|v| v["phase"] == "after_run" && v["sealed_latency"]["sha256"].is_string()),
            "the sealed digest is kept"
        );
    }
}

/// The same change on a diagnostic run is recorded and does not fail it.
#[test]
#[cfg_attr(not(unix), ignore = "this platform has no descriptor identity")]
fn a_change_on_a_diagnostic_run_is_recorded_and_does_not_fail_it() {
    let (_dir, container) = encoded();
    let (plan, store) = opened(&container);
    let path = first_segment(&store);
    let backend = Hooked::new(When::Passes, || write_in_place(&path));
    let run = evidence_paths();
    curve(&backend, &plan, &store, &container, &run, false, None, None)
        .expect("diagnostic execution carries on");
    let stream = read_lines(&run.regions);
    let after = last(&stream, "instance_identity");
    assert_eq!(after["instance_identity"]["identity_diagnostic"], "changed");
    assert_eq!(after["evidence_class"], "diagnostic");
    let end = last(&stream, "attempt_end");
    assert_eq!(end["status"], "completed");
    assert_eq!(end["evidence_class"], "diagnostic");
}

/// A real capture failure, through the real path: the platform has none to
/// capture, so a campaign stops before binding and diagnostics carry on.
#[test]
#[cfg_attr(unix, ignore = "this platform has descriptor identity")]
fn a_failed_capture_stops_a_campaign_before_binding_and_not_a_diagnostic() {
    let (_dir, container) = encoded();
    let (plan, store) = opened(&container);
    let run = evidence_paths();
    let error = curve(
        &ReferenceBackend,
        &plan,
        &store,
        &container,
        &run,
        true,
        None,
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("capture_failed"), "{error}");
    let stream = read_lines(&run.regions);
    assert_eq!(
        kinds(&stream),
        [("instance_identity", "before_binding"), ("attempt_end", "")]
    );
    assert_eq!(last(&stream, "attempt_end")["evidence_class"], "invalid");

    let run = evidence_paths();
    curve(
        &ReferenceBackend,
        &plan,
        &store,
        &container,
        &run,
        false,
        None,
        None,
    )
    .expect("diagnostic execution carries on");
    let stream = read_lines(&run.regions);
    assert_eq!(last(&stream, "attempt_end")["status"], "completed");
}

/// The whole chain on a binding that maps a bank: the mapped segment is on the
/// record, and a threshold rejection is not invalid evidence. It runs, with
/// its latency sealed, and exits zero.
#[test]
#[cfg_attr(not(unix), ignore = "this platform has no descriptor identity")]
fn a_threshold_rejection_of_a_mapped_binding_runs_and_is_not_invalid() {
    use larql_vindex::format::vindex3::fixtures_kimi::kimi_per_expert_moe_f32_model;
    use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
    let source = tempfile::tempdir().unwrap();
    kimi_per_expert_moe_f32_model(source.path());
    let dir = tempfile::tempdir().unwrap();
    let container = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![source.path().to_path_buf()],
        output: container.clone(),
    }))
    .unwrap();
    let (plan, store) = opened(&container);
    let evidence = evidence_paths();
    // Read every segment first, so the binding's witness cannot be empty and
    // a threshold of zero is a real rejection.
    for segment in store.instance_report().segments {
        std::fs::read(&segment.path).unwrap();
    }
    curve(
        &ProductionBackend::new(),
        &plan,
        &store,
        &container,
        &evidence,
        true,
        None,
        Some(0),
    )
    .expect("a rejection is not a failure");
    let stream = read_lines(&evidence.regions);
    assert_full_run(&stream);
    let admission = last(&stream, "admission");
    assert_eq!(admission["complete"], true);
    assert!(admission["resident_bytes"].as_u64().unwrap() > 0);
    assert_eq!(admission["outcome"], "above_threshold");
    assert_eq!(admission["evidence_class"], "unverified");
    let segments = admission["instance_identity"]["segments"]
        .as_array()
        .unwrap();
    assert_eq!(
        segments
            .iter()
            .filter(|s| s["use"]["mapped"] == true)
            .count(),
        1
    );
    assert!(segments.iter().any(|s| s["use"]["copied"] == true));
    assert_eq!(last(&stream, "attempt_end")["status"], "completed");
    assert!(sealed_kinds(&evidence.sealed).iter().any(|k| k == "step"));
}
