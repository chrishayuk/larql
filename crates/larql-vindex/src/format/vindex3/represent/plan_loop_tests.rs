//! MEASURE-PLAN-2 PR 2 witnesses over a real container and a real
//! token bank: W5 (plan facts), W6 (AUTO-REP through the plan-v1
//! executor) and W8's ingestion half (a gate-less record ingests a real
//! plan-v1 reading and replays it).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

use super::actuate::executor::{ExecutorRegistry, ExperimentExecutor};
use super::actuate::prepare::{PreparedExperiment, Ready};
use super::actuate::request::MeasurementRequest;
use super::actuate::{DeclaredArtifacts, PlanTeacherForcedExecutor};
use super::auto_rep::{self, group_vocabulary, CampaignOutcome, CampaignSetup, RepresentCompiler};
use super::compiler::read_source_identity;
use super::ingest::artifact::MeasurementArtifact;
use super::ingest::state_evidence::ArtifactStateEvidence;
use super::ingest::{ingest, IngestionRefusal, IngestionSources};
use super::map::PrecisionMap;
use super::measure::outcome::VerifiedFacts;
use super::measure::plan::PROCEDURE;
use super::policy::classify_in;
use super::reading::{test_plan_gate, Gate, Observation, ReadingKind, RunFacts};
use super::state::instrument::{InstrumentSemantics, MetricSemantics};
use super::state::protocol::MeasurementProtocol;
use super::state::snapshot::{SearchSnapshot, SearchSpace};
use super::state::{
    fixtures, ActionVocabulary, EvidenceBank, LogicalBytes, PackLayoutAdmission,
    RepresentationState, RepresentationStateGraph, ResolvedState, SurfaceTensor, TensorSurface,
    TransitionPolicy,
};
use super::token_bank::{export, TOKENIZER_FILE, TOKEN_BANK_SCHEMA};
use super::{compile_representation, RepresentSpec};
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::opplan::exec::continuation_registry::{
    ContinuationFactory, ContinuationRegistry,
};
use crate::format::vindex3::opplan::exec::kv::RowFactory;
use crate::format::vindex3::opplan::exec::lowering::LoweringIdentity;

const COMPONENT: &str = "target";
/// Samples the record declares, and so the run measures.
const SAMPLES: usize = 3;
const WORDS: [&str; 16] = [
    "the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran", "far", "sun", "rose", "over",
    "hill", "we", "saw",
];

fn write_tokenizer(container: &Path) {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("[UNK]".into(), 0);
    for (i, w) in WORDS.iter().enumerate() {
        vocab.insert((*w).into(), 1 + i as u32);
    }
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tk = tokenizers::Tokenizer::new(model);
    tk.with_pre_tokenizer(Some(Whitespace {}));
    tk.save(container.join(TOKENIZER_FILE), false).unwrap();
}

struct Fixture {
    _dir: tempfile::TempDir,
    source: PathBuf,
    bank: PathBuf,
    workdir: PathBuf,
    runs: PathBuf,
    snapshot: SearchSnapshot,
    spec: RepresentSpec,
}

fn plan_instrument() -> InstrumentSemantics {
    InstrumentSemantics::new(
        "kl(reference || candidate), nats",
        "mean, p50, p99, max over positions",
        "teacher-forced, every position",
        PROCEDURE,
    )
    .with_semantics(MetricSemantics::PLAN_V1)
    .unwrap()
}

fn fixture(gate: Option<Gate>) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    std::fs::create_dir(&checkpoint).unwrap();
    let source = dir.path().join("source");
    encode_fixture_container(dense_f32_model, &checkpoint, &source, COMPONENT);
    write_tokenizer(&source);

    let prompts = dir.path().join("prompts.json");
    let bank_prompts = serde_json::json!({
        "bank": "plan-loop-fixture",
        "prompts": [
            {"id": "prose-000", "category": "prose", "text": "the cat sat on a mat and the dog ran far"},
            {"id": "prose-001", "category": "prose", "text": "the sun rose over the hill"},
            {"id": "longform-000", "category": "longform", "text": "we saw a dog on the hill"},
            {"id": "prose-002", "category": "prose", "text": "a dog sat on the hill"}
        ]
    });
    std::fs::write(&prompts, serde_json::to_vec(&bank_prompts).unwrap()).unwrap();
    let bank = dir.path().join("bank");
    export(&prompts, &source.join(TOKENIZER_FILE), 10, &bank).unwrap();

    let index: Vindex3Index =
        serde_json::from_slice(&std::fs::read(source.join("index.json")).unwrap()).unwrap();
    let mut entries = BTreeMap::new();
    for rep in index.representations.values() {
        let (header, _) = read_segment_header(&source.join(&rep.segment)).unwrap();
        for tensor in header.tensors {
            let role = classify_in(true, &rep.object, &tensor.name, &tensor.shape);
            entries.insert(
                (rep.object.clone(), tensor.name.clone()),
                SurfaceTensor::new(&rep.object, &tensor.name, role, tensor.shape),
            );
        }
    }
    let surface = TensorSurface::new(entries.into_values()).unwrap();
    let spec = RepresentSpec::nvfp4();
    let base_map =
        PrecisionMap::from_policy(spec.map_name(), &spec.encoding, &spec.roles, &spec.protect);

    let manifest = std::fs::read(bank.join("manifest.json")).unwrap();
    let evidence = EvidenceBank::new(
        TOKEN_BANK_SCHEMA,
        super::compile::hash_bytes(&manifest),
        (0..SAMPLES).map(|i| format!("seq-{i:03}")),
        0,
    );
    let protocol = MeasurementProtocol::new(evidence, plan_instrument(), PROCEDURE);
    let seed = fixtures::PricedRecord::new(&source)
        .with_protocol(protocol)
        .build();
    let model = read_source_identity(&source).unwrap();
    let root = RepresentationState::resolve(&model, &surface, &base_map, &PackLayoutAdmission);
    let mut facts = seed.facts().clone();
    facts.graph = RepresentationStateGraph::new(
        TransitionPolicy::Unconstrained,
        ResolvedState::new(root, LogicalBytes::new(0)),
    );
    let mut space = SearchSpace {
        surface,
        base_map,
        vocabulary: ActionVocabulary::new([]).unwrap(),
        applied: BTreeSet::new(),
    };
    space.vocabulary = group_vocabulary(&space).unwrap();
    let mut config = seed.config().clone();
    config.gate = gate;
    let snapshot = SearchSnapshot::new(space, config, facts);
    let workdir = dir.path().join("candidates");
    let runs = dir.path().join("runs");
    std::fs::create_dir(&workdir).unwrap();
    std::fs::create_dir(&runs).unwrap();
    Fixture {
        _dir: dir,
        source,
        bank,
        workdir,
        runs,
        snapshot,
        spec,
    }
}

fn executor(f: &Fixture) -> PlanTeacherForcedExecutor {
    let mut continuations = ContinuationRegistry::new();
    continuations.register(Box::new(RowFactory)).unwrap();
    PlanTeacherForcedExecutor {
        component: COMPONENT.into(),
        reference: LoweringIdentity::cpu_production(),
        candidate: LoweringIdentity::cpu_production(),
        continuations,
        continuation: RowFactory.identity(),
        output_root: f.runs.clone(),
    }
}

/// The compile-everything candidate, run through the real executor and
/// sealed: what a first AUTO-REP proposal would produce.
struct Run {
    prepared: PreparedExperiment,
    artifact: MeasurementArtifact,
    candidate: PathBuf,
}

fn run_uniform(f: &Fixture) -> Run {
    let candidate = f.workdir.join("uniform");
    compile_representation(&f.source, &candidate, &f.spec).unwrap();
    let established = ArtifactStateEvidence::establish(&candidate).unwrap();
    let key = f
        .snapshot
        .standing_intent()
        .key_for(established.established());
    let request = MeasurementRequest::of(&f.snapshot, &key, &BTreeSet::new()).unwrap();
    let prepared = PreparedExperiment::Ready(Box::new(Ready {
        request,
        physical_delta: 0,
        routes: 1,
        considered: 1,
    }));
    let executor = executor(f);
    let registry = ExecutorRegistry::new([&executor as &dyn ExperimentExecutor]).unwrap();
    let locator = DeclaredArtifacts::new()
        .container_at(&f.source)
        .corpus_at(&f.bank)
        .overlay_at(key.state(), &candidate);
    let observed = registry
        .execute(prepared.request().unwrap(), &locator)
        .unwrap();
    let artifact = MeasurementArtifact::from_execution(&prepared, &observed, &established).unwrap();
    Run {
        prepared,
        artifact,
        candidate,
    }
}

fn sources<'a>(f: &'a Fixture, run: &'a Run) -> IngestionSources<'a> {
    IngestionSources {
        container: &f.source,
        candidate: &run.candidate,
        corpus: &f.bank,
    }
}

// ------------------------------------------------------------------ W8

#[test]
fn w8_a_gate_less_record_ingests_a_real_plan_reading_and_replays_it() {
    let f = fixture(None);
    let run = run_uniform(&f);
    let reading = run
        .artifact
        .observation()
        .as_plan()
        .expect("a plan reading")
        .clone();
    let positions: u64 = (0..SAMPLES)
        .map(|i| {
            super::token_bank::TokenBank::open(&f.bank)
                .unwrap()
                .manifest()
                .samples[i]
                .tokens as u64
        })
        .sum();
    assert_eq!(
        reading.positions, positions,
        "positions are the token counts"
    );

    let mut snapshot = f.snapshot.clone();
    let ingested = ingest(
        &mut snapshot,
        &run.prepared,
        &run.artifact,
        &sources(&f, &run),
    )
    .unwrap();
    assert!(ingested.recorded);
    let key = run.artifact.key();
    let held = snapshot.measurements().get(key).unwrap();
    assert_eq!(held.kind(), ReadingKind::Plan);
    assert_eq!(held, &Observation::Plan(reading));
    assert!(snapshot.adjudicate(key).is_none(), "no gate, no verdict");
    assert!(snapshot.admitted().is_empty());

    // Replay: the record reopens identically and a second ingestion of the
    // same artifact is a no-op.
    let bytes = serde_json::to_vec(&snapshot).unwrap();
    let mut reopened: SearchSnapshot = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(reopened, snapshot);
    let again = ingest(
        &mut reopened,
        &run.prepared,
        &run.artifact,
        &sources(&f, &run),
    )
    .unwrap();
    assert!(!again.recorded);
    assert_eq!(serde_json::to_vec(&reopened).unwrap(), bytes);
}

// ------------------------------------------------------------------ W5

fn reseal(
    run: &Run,
    edit: impl FnOnce(&mut super::actuate::executor::Observed),
) -> MeasurementArtifact {
    let mut observed = super::actuate::executor::Observed {
        key: run.artifact.key().clone(),
        observation: run.artifact.observation().clone(),
        verified: run.artifact.verified().clone(),
        execution_note: "resealed".into(),
    };
    edit(&mut observed);
    let established = ArtifactStateEvidence::establish(&run.candidate).unwrap();
    MeasurementArtifact::from_execution(&run.prepared, &observed, &established).unwrap()
}

#[test]
fn w5_plan_facts_are_complete_or_refused() {
    let f = fixture(None);
    let run = run_uniform(&f);
    let complete = run.artifact.verified().as_plan().unwrap().clone();
    assert!(complete.complete(SAMPLES));

    for (obligation, edit) in [
        (
            "null arm",
            Box::new(|v: &mut super::measure::plan::PlanVerifiedFacts| v.null_arm_samples = 0)
                as Box<dyn Fn(&mut super::measure::plan::PlanVerifiedFacts)>,
        ),
        ("physical attribution", Box::new(|v| v.attributed_arms = 0)),
        (
            "seal/read witness",
            Box::new(|v| v.sealed_representations = 0),
        ),
        (
            "every declared sample read",
            Box::new(|v| v.bank_samples_read = 1),
        ),
        (
            "tokenizer checks",
            Box::new(|v| v.tokenizer_checked_arms = 0),
        ),
    ] {
        let artifact = reseal(&run, |o| {
            let mut facts = complete.clone();
            edit(&mut facts);
            o.verified = facts.into();
        });
        let mut snapshot = f.snapshot.clone();
        let refusal =
            ingest(&mut snapshot, &run.prepared, &artifact, &sources(&f, &run)).unwrap_err();
        let IngestionRefusal::IncompleteRun { missing, .. } = &refusal else {
            panic!("{obligation}: {refusal}");
        };
        assert!(
            missing.iter().any(|m| m == obligation),
            "{obligation}: {refusal}"
        );
        assert_eq!(snapshot, f.snapshot, "refused without change");
    }

    // A reading that disagrees with the bank's own token counts.
    let miscounted = reseal(&run, |o| {
        if let Observation::Plan(p) = &mut o.observation {
            p.positions += 1;
        }
    });
    let mut snapshot = f.snapshot.clone();
    let refusal = ingest(
        &mut snapshot,
        &run.prepared,
        &miscounted,
        &sources(&f, &run),
    )
    .unwrap_err();
    assert!(
        refusal.to_string().contains("observation positions"),
        "{refusal}"
    );

    // `top5_overlap_mean` is the mean COUNT of shared top-5 ids, in
    // [0, 5]: a real run's value above 1 is accepted (the first ingestion
    // above holds one), and a value above 5 is refused, so nobody can
    // quietly renormalise it into a share.
    assert!(
        run.artifact
            .observation()
            .as_plan()
            .unwrap()
            .all
            .top5_overlap_mean
            > 1.0
    );
    let overcounted = reseal(&run, |o| {
        if let Observation::Plan(p) = &mut o.observation {
            p.all.top5_overlap_mean = 5.5;
        }
    });
    let mut snapshot = f.snapshot.clone();
    let refusal = ingest(
        &mut snapshot,
        &run.prepared,
        &overcounted,
        &sources(&f, &run),
    )
    .unwrap_err();
    assert!(
        refusal
            .to_string()
            .contains("invalid plan observation statistics"),
        "{refusal}"
    );

    // Kimi facts on a plan reading: one run cannot produce both.
    let crossed = reseal(&run, |o| {
        o.verified = RunFacts::Kimi(VerifiedFacts {
            compiled_layers: vec![0],
            compiled_projections: vec!["q_proj".into()],
            attribution_checked_layers: vec![0],
            seal_checked_operands: 1,
            invariant_neighbour_layer: Some(1),
            positions: 1,
            gate_evaluated: "none".into(),
        });
    });
    let mut snapshot = f.snapshot.clone();
    let refusal = ingest(&mut snapshot, &run.prepared, &crossed, &sources(&f, &run)).unwrap_err();
    assert!(
        refusal.to_string().contains("one run cannot produce both"),
        "{refusal}"
    );
}

// ------------------------------------------------------------------ W6

#[test]
fn w6_auto_rep_runs_end_to_end_through_the_plan_executor() {
    let f = fixture(Some(Gate::Plan(test_plan_gate())));
    let executor = executor(&f);
    let registry = ExecutorRegistry::new([&executor as &dyn ExperimentExecutor]).unwrap();
    let mut snapshot = f.snapshot.clone();
    let record = auto_rep::run(
        &mut snapshot,
        &CampaignSetup {
            source: &f.source,
            corpus: &f.bank,
            workdir: &f.workdir,
            spec: &f.spec,
            compiler: &RepresentCompiler { source: &f.source },
            executors: &registry,
            budget: 3,
            node_limit: u64::MAX,
            pins: BTreeMap::new(),
        },
    )
    .unwrap();
    assert!(!record.entries.is_empty());
    assert!(matches!(
        record.outcome,
        CampaignOutcome::Admitted { .. } | CampaignOutcome::BudgetSpent
    ));
    assert_eq!(record.measurements_spent, record.entries.len());
    for entry in &record.entries {
        // Every reading entered through the plan executor and ingestion,
        // and the verdict is the plan gate's own.
        let held = snapshot.measurements().get(&entry.key).unwrap();
        assert_eq!(held.kind(), ReadingKind::Plan);
        let adjudication = snapshot.adjudicate(&entry.key).unwrap();
        assert_eq!(
            adjudication.admissible(),
            entry.verdict == auto_rep::Verdict::Admitted
        );
    }
    // The first proposal is uniform: it compiles every eligible group.
    assert!(record.entries[0].proposal.protected.is_empty());
    eprintln!(
        "W6 plan-v1 campaign: outcome {:?}, entries {}, first kl_p99 {:?}",
        record.outcome,
        record.entries.len(),
        snapshot
            .measurements()
            .get(&record.entries[0].key)
            .and_then(Observation::as_plan)
            .map(|p| p.all.kl_p99)
    );
}
