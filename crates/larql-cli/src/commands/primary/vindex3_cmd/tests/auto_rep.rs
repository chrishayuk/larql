//! `vindex3 auto-rep` on the dense fixture: `init` writes a
//! characterisation-only plan-v1 record that `optimizer-mcp --snapshot`
//! loads, and `run` refuses that record, as the loop does, before
//! compiling or writing anything. The producer's and the loop's own
//! witnesses are in `represent/plan_loop_tests.rs`.

use std::path::PathBuf;

use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::represent::token_bank::{export, TOKENIZER_FILE};

use crate::commands::primary::optimizer_mcp;
use crate::commands::primary::vindex3_cmd::auto_rep::{
    run, AutoRepArgs, AutoRepCommand, InitArgs, RunArgs,
};
use crate::commands::primary::vindex3_cmd::plugins::PluginArgs;
use crate::commands::primary::vindex3_cmd::ExecBackend;

const WORDS: [&str; 8] = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran"];

struct Fixture {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    source: PathBuf,
    bank: PathBuf,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let checkpoint = root.join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let source = root.join("source.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &source, "target");
    super::write_word_tokenizer(&source, &WORDS);
    let prompts = root.join("prompts.json");
    let body = serde_json::json!({
        "bank": "cli-auto-rep-fixture",
        "prompts": [
            {"id": "prose-000", "category": "prose", "text": "the cat sat on a mat"},
            {"id": "prose-001", "category": "prose", "text": "a dog ran on the mat"}
        ]
    });
    std::fs::write(&prompts, serde_json::to_vec(&body).unwrap()).unwrap();
    let bank = root.join("bank");
    export(&prompts, &source.join(TOKENIZER_FILE), 8, &bank).unwrap();
    Fixture {
        _tmp: tmp,
        root,
        source,
        bank,
    }
}

fn init(f: &Fixture, output: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    run(AutoRepArgs {
        command: AutoRepCommand::Init(InitArgs {
            container: f.source.clone(),
            bank: f.bank.clone(),
            sequences: 2,
            encoding: "NVFP4".into(),
            output,
        }),
    })
}

#[test]
fn init_writes_a_characterisation_only_record_the_mcp_loader_accepts() {
    let f = fixture();
    let record = f.root.join("record.json");
    init(&f, record.clone()).unwrap();
    let loaded = optimizer_mcp::load(&record).unwrap();
    assert!(loaded.gate().is_none());
    assert!(!loaded.space().vocabulary.is_empty());
    assert_eq!(
        loaded.protocol().unwrap().procedure,
        larql_vindex::format::vindex3::represent::measure::plan::PROCEDURE
    );
    // Never overwrites.
    assert!(init(&f, record).is_err());
}

#[test]
fn run_refuses_a_gate_less_record_before_compiling_or_writing() {
    let f = fixture();
    let record = f.root.join("record.json");
    init(&f, record.clone()).unwrap();
    let workdir = f.root.join("candidates");
    let output = f.root.join("advanced.json");
    let campaign = f.root.join("campaign.json");
    let err = run(AutoRepArgs {
        command: AutoRepCommand::Run(RunArgs {
            snapshot: record,
            output: output.clone(),
            campaign: campaign.clone(),
            source: f.source.clone(),
            bank: f.bank.clone(),
            workdir: workdir.clone(),
            runs: f.root.join("runs"),
            budget: 1,
            node_limit: 1_000,
            component: "target".into(),
            reference_backend: ExecBackend::Production,
            candidate_backend: ExecBackend::ProductionNvfp4,
            plugins: PluginArgs {
                plugins: vec![],
                lowering: None,
                representation: None,
            },
        }),
    })
    .unwrap_err()
    .to_string();
    assert!(err.contains("characterisation-only"), "{err}");
    assert!(std::fs::read_dir(&workdir).unwrap().next().is_none());
    assert!(!output.exists() && !campaign.exists());
}

// -------------------------------------------- PR 3b: the verb executor

mod verb_executor {
    use std::collections::BTreeSet;
    use std::path::Path;

    use larql_vindex::format::vindex3::represent::actuate::executor::{
        ExecutionRefusal, ExecutorRegistry, ExperimentExecutor,
    };
    use larql_vindex::format::vindex3::represent::actuate::prepare::{PreparedExperiment, Ready};
    use larql_vindex::format::vindex3::represent::actuate::request::MeasurementRequest;
    use larql_vindex::format::vindex3::represent::actuate::DeclaredArtifacts;
    use larql_vindex::format::vindex3::represent::ingest::artifact::MeasurementArtifact;
    use larql_vindex::format::vindex3::represent::ingest::state_evidence::ArtifactStateEvidence;
    use larql_vindex::format::vindex3::represent::ingest::{ingest, IngestionSources};
    use larql_vindex::format::vindex3::represent::produce::{produce, ProduceInputs};
    use larql_vindex::format::vindex3::represent::reading::ReadingKind;
    use larql_vindex::format::vindex3::represent::{compile_representation, RepresentSpec};

    use super::{fixture, Fixture};
    use crate::commands::primary::vindex3_cmd::measure::VerbPlanExecutor;
    use crate::commands::primary::vindex3_cmd::ExecBackend;

    /// Compile the uniform candidate, run it through the verb executor
    /// against a produced gate-less record, and ingest the reading.
    fn measure_uniform(
        f: &Fixture,
        reference: ExecBackend,
        candidate: ExecBackend,
    ) -> Result<(), ExecutionRefusal> {
        let spec = RepresentSpec::nvfp4();
        let mut record = produce(&ProduceInputs {
            source: &f.source,
            spec: &spec,
            bank: &f.bank,
            sequences: 2,
        })
        .unwrap();
        let pack = f.root.join("uniform");
        compile_representation(&f.source, &pack, &spec).unwrap();
        let established = ArtifactStateEvidence::establish(&pack).unwrap();
        let key = record.standing_intent().key_for(established.established());
        let request = MeasurementRequest::of(&record, &key, &BTreeSet::new()).unwrap();
        let prepared = PreparedExperiment::Ready(Box::new(Ready {
            request,
            physical_delta: 0,
            routes: 1,
            considered: 1,
        }));
        let executor = VerbPlanExecutor {
            reference_backend: reference,
            candidate_backend: candidate,
            component: "target".into(),
            plugins: vec![],
            output_root: f.root.join("runs"),
        };
        let registry = ExecutorRegistry::new([&executor as &dyn ExperimentExecutor]).unwrap();
        let locator = DeclaredArtifacts::new()
            .container_at(&f.source)
            .corpus_at(&f.bank)
            .overlay_at(key.state(), &pack);
        let observed = registry.execute(prepared.request().unwrap(), &locator)?;
        let artifact =
            MeasurementArtifact::from_execution(&prepared, &observed, &established).unwrap();
        let sources = IngestionSources {
            container: &f.source,
            candidate: &pack,
            corpus: &f.bank,
        };
        assert!(
            ingest(&mut record, &prepared, &artifact, &sources)
                .unwrap()
                .recorded
        );
        let held = record.measurements().get(&key).unwrap();
        assert_eq!(held.kind(), ReadingKind::Plan);
        assert!(held.as_plan().unwrap().positions > 0);
        assert!(record.adjudicate(&key).is_none(), "no gate, no verdict");
        Ok(())
    }

    fn nothing_under(dir: &Path) -> bool {
        !dir.exists() || std::fs::read_dir(dir).unwrap().next().is_none()
    }

    #[test]
    fn interpreter_arms_measure_and_the_reading_is_ingested() {
        let f = fixture();
        measure_uniform(&f, ExecBackend::Production, ExecBackend::ProductionNvfp4).unwrap();
    }

    #[test]
    fn a_candidate_backend_that_would_not_read_the_pack_is_refused_before_running() {
        let f = fixture();
        let err = measure_uniform(&f, ExecBackend::Production, ExecBackend::Production)
            .unwrap_err()
            .to_string();
        assert!(err.contains("NVFP4 pack"), "{err}");
        assert!(nothing_under(&f.root.join("runs")), "nothing ran");
    }

    /// Lowered arms on the Metal device. Fails, never skips, when the
    /// device or its shader library is unavailable.
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    #[test]
    fn lowered_arms_measure_and_the_reading_is_ingested() {
        let f = fixture();
        measure_uniform(&f, ExecBackend::MetalLoweredF16, ExecBackend::MetalLowered)
            .expect("a lowered plan-v1 run on this Metal device");
    }
}
