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
    run, ArmLowering, AutoRepArgs, AutoRepCommand, InitArgs, RunArgs,
};
use crate::commands::primary::vindex3_cmd::plugins::PluginArgs;

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
            reference_lowering: ArmLowering::Production,
            candidate_lowering: ArmLowering::Production,
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
