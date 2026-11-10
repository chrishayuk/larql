//! `vindex3 exec --rope-angles` (PREC-DIAG-1 E0b) on the encoded fixture:
//! the default and an explicit `larql` write byte-identical logit dumps;
//! `hf-f32` runs on the two logit arms and says so in the run's records;
//! and it is refused everywhere it would otherwise be silently ignored.

use super::super::args::{RopeAnglesArg, SiteCaptureArgs};
use super::super::prefill_dump::PrefillMethod;
use super::super::site_capture_dump::MANIFEST;
use super::super::{run, EncodeArgs, ExecArgs, ExecBackend, Vindex3Command};
use super::fixture_dir;

const TOKENS: &str = "1,2,3";

fn exec_args(container: std::path::PathBuf, backend: ExecBackend) -> ExecArgs {
    ExecArgs {
        container,
        component: "target".to_string(),
        tokens: TOKENS.to_string(),
        dump_layers: None,
        resume: false,
        backend,
        representation_source: "auto".to_string(),
        generate: None,
        speculate: None,
        stop_at_eos: false,
        residency_curve: false,
        repeat: 1,
        warmup: 0,
        unquiet_ok: true,
        residency_regions: None,
        sealed_latency: None,
        residency_campaign: false,
        verification_manifest: None,
        admission_threshold_bytes: None,
        admission_basis: None,
        expert_access: "demand".to_string(),
        logit_dump: None,
        prefill_logit_dump: None,
        prefill_lengths: None,
        prefill_method: PrefillMethod::Batched,
        gemv_witness: None,
        gemv_witness_max_operands: 1,
        capture: SiteCaptureArgs::default(),
        rope_angles: RopeAnglesArg::Larql,
        bank: None,
        dump_dir: None,
        draft_depth: None,
        profile: false,
        plugin: Default::default(),
    }
}

fn encoded_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = fixture_dir(true);
    let container = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: container.clone(),
    }))
    .unwrap();
    (dir, container)
}

fn read_json(path: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn the_default_and_explicit_larql_write_byte_identical_dumps_and_say_so() {
    let (dir, container) = encoded_fixture();
    for backend in [ExecBackend::Reference, ExecBackend::Production] {
        let default = dir.path().join(format!("{backend:?}-default.f32"));
        let explicit = dir.path().join(format!("{backend:?}-explicit.f32"));
        let mut args = exec_args(container.clone(), backend);
        args.logit_dump = Some(default.clone());
        run(Vindex3Command::Exec(args)).unwrap();
        let mut args = exec_args(container.clone(), backend);
        args.logit_dump = Some(explicit.clone());
        args.rope_angles = RopeAnglesArg::Larql;
        run(Vindex3Command::Exec(args)).unwrap();
        assert_eq!(
            std::fs::read(&default).unwrap(),
            std::fs::read(&explicit).unwrap()
        );
        let meta = read_json(&default.with_extension("meta.json"));
        assert_eq!(meta["rope_angles"], "larql");
    }
}

#[test]
fn hf_f32_runs_on_both_logit_arms_and_is_recorded() {
    let (dir, container) = encoded_fixture();
    for backend in [ExecBackend::Reference, ExecBackend::Production] {
        let cached = dir.path().join(format!("{backend:?}-hf.f32"));
        let sites = dir.path().join(format!("{backend:?}-sites"));
        let mut args = exec_args(container.clone(), backend);
        args.logit_dump = Some(cached.clone());
        args.rope_angles = RopeAnglesArg::HfF32;
        args.capture = SiteCaptureArgs {
            capture_sites: Some(sites.clone()),
            capture_positions: Some("0,2".to_string()),
            capture_budget_bytes: None,
        };
        run(Vindex3Command::Exec(args)).unwrap();
        assert_eq!(
            read_json(&cached.with_extension("meta.json"))["rope_angles"],
            "hf-f32"
        );
        assert_eq!(
            read_json(&sites.join(MANIFEST))["identity"]["rope_angles"],
            "hf-f32"
        );

        let prefill = dir.path().join(format!("{backend:?}-hf-prefill.f32"));
        let mut args = exec_args(container.clone(), backend);
        args.prefill_logit_dump = Some(prefill.clone());
        args.rope_angles = RopeAnglesArg::HfF32;
        run(Vindex3Command::Exec(args)).unwrap();
        assert!(!std::fs::read(&prefill).unwrap().is_empty());
    }
}

#[test]
fn hf_f32_is_refused_where_it_would_be_ignored() {
    let (dir, container) = encoded_fixture();
    // No logit arm at all: the summary run would never read the flag.
    let mut args = exec_args(container.clone(), ExecBackend::Reference);
    args.rope_angles = RopeAnglesArg::HfF32;
    let err = run(Vindex3Command::Exec(args)).unwrap_err();
    assert!(err.to_string().contains("--rope-angles hf-f32"), "{err}");

    // The per-prefix measurement-method control runs the shipped
    // construction only.
    let mut args = exec_args(container.clone(), ExecBackend::Reference);
    args.prefill_logit_dump = Some(dir.path().join("per-prefix.f32"));
    args.prefill_method = PrefillMethod::PerPrefix;
    args.rope_angles = RopeAnglesArg::HfF32;
    let err = run(Vindex3Command::Exec(args)).unwrap_err();
    assert!(err.to_string().contains("--rope-angles hf-f32"), "{err}");

    // A layer dump runs its own traversal, which this flag never reaches.
    let mut args = exec_args(container, ExecBackend::Reference);
    args.dump_layers = Some(dir.path().join("layers"));
    args.rope_angles = RopeAnglesArg::HfF32;
    let err = run(Vindex3Command::Exec(args)).unwrap_err();
    assert!(err.to_string().contains("--rope-angles hf-f32"), "{err}");
}

#[test]
fn the_flag_maps_to_the_executor_s_construction() {
    use larql_vindex::format::vindex3::opplan::exec::rope_angles::RopeAngles;
    assert_eq!(RopeAnglesArg::default(), RopeAnglesArg::Larql);
    assert_eq!(RopeAnglesArg::Larql.angles(), RopeAngles::Larql);
    assert_eq!(RopeAnglesArg::HfF32.angles(), RopeAngles::HfF32);
}
