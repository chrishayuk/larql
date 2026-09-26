//! AUTO-REP-1b gates over a real compiled fixture container.
//!
//! The executor is scripted: it admits a candidate exactly when the map it
//! was asked to present protects every group in a hidden `required` set.
//! Everything else is production: compile, identity, request, registry,
//! artifact sealing, ingestion and the gate's adjudication.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use super::super::actuate::executor::{
    ArtifactLocator, ExecutionRefusal, ExecutorRegistry, ExperimentExecutor, Observed,
};
use super::super::compiler::read_source_identity;
use super::super::map::PrecisionMap;
use super::super::measure::outcome::VerifiedFacts;
use super::super::policy::classify_in;
use super::super::state::snapshot::SearchSpace;
use super::super::state::{
    fixtures, EvidenceBank, LogicalBytes, PackLayoutAdmission, RepresentationState,
    RepresentationStateGraph, ResolvedState, SurfaceTensor, TensorSurface, TransitionPolicy,
};
use super::*;
use crate::format::vindex3::{
    encode::segment::read_segment_header,
    fixtures::{dense_f32_model, encode_fixture_container},
    index::Vindex3Index,
};

/// Deep enough for the gate's position floor.
const POSITIONS: u64 = 8192;
/// Below the fixture gate's `kl_p99` ceiling.
const PASSING_KL: f64 = 1e-4;
/// Above it.
const FAILING_KL: f64 = 1e-2;

struct Fixture {
    _dir: tempfile::TempDir,
    source: PathBuf,
    corpus: PathBuf,
    workdir: PathBuf,
    snapshot: SearchSnapshot,
    spec: RepresentSpec,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    std::fs::create_dir(&checkpoint).unwrap();
    let source = dir.path().join("source");
    encode_fixture_container(dense_f32_model, &checkpoint, &source, "target");
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
    assert!(base_map.exceptions.is_empty());

    let corpus = dir.path().join("corpus");
    std::fs::create_dir(&corpus).unwrap();
    let rows = vec![0u8; POSITIONS as usize * 64 * 4];
    std::fs::write(corpus.join("seq_0.f32"), &rows).unwrap();
    let manifest = serde_json::to_vec(&serde_json::json!({
        "sequences":1,"positions":POSITIONS,"hidden":64,
        "token_ids":[vec![0u32; POSITIONS as usize]],"regime":"teacher-forced",
        "payload_authority":"teacher-forced-bank-payload/v1",
        "payloads":{"seq_0.f32":{"len":rows.len(),"sha256":super::super::compile::hash_bytes(&rows)}}
    }))
    .unwrap();
    std::fs::write(corpus.join("manifest.json"), &manifest).unwrap();
    let mut protocol = fixtures::protocol();
    protocol.bank = EvidenceBank::new(
        "kimi-teacher-forced/v1",
        super::super::compile::hash_bytes(&manifest),
        ["seq-000"],
        POSITIONS as u32,
    );
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
    let snapshot = SearchSnapshot::new(space, seed.config().clone(), facts);
    let workdir = dir.path().join("candidates");
    std::fs::create_dir(&workdir).unwrap();
    Fixture {
        _dir: dir,
        source,
        corpus,
        workdir,
        snapshot,
        spec,
    }
}

/// Admits a candidate exactly when its map protects every `required`
/// group; counts and remembers every experiment it is asked to run.
struct Truth {
    procedure: String,
    required: BTreeSet<GroupKey>,
    runs: RefCell<Vec<MeasurementKey>>,
}

impl Truth {
    fn new(f: &Fixture, required: impl IntoIterator<Item = GroupKey>) -> Self {
        Self {
            procedure: f.snapshot.protocol().unwrap().procedure.clone(),
            required: required.into_iter().collect(),
            runs: RefCell::new(Vec::new()),
        }
    }
}

impl ExperimentExecutor for Truth {
    fn procedure(&self) -> &str {
        &self.procedure
    }

    fn execute(
        &self,
        request: &MeasurementRequest,
        artifacts: &dyn ArtifactLocator,
    ) -> Result<Observed, ExecutionRefusal> {
        // The candidate the loop compiled is the one it asked about.
        artifacts.candidate(request)?;
        self.runs.borrow_mut().push(request.key().clone());
        let protected: BTreeSet<GroupKey> = request
            .candidate_map()
            .exceptions
            .iter()
            .filter(|e| e.encoding.is_none())
            .map(|e| GroupKey::new(e.projection.clone().unwrap(), e.layers.unwrap().0))
            .collect();
        let kl = if self.required.is_subset(&protected) {
            PASSING_KL
        } else {
            FAILING_KL
        };
        let mut observation = fixtures::authority_reading(kl, 0);
        observation.positions = POSITIONS;
        observation.routing.route_weight_mass_moved = None;
        observation.top10_mass_displaced = None;
        observation.top1_mass_displaced = None;
        Ok(Observed {
            key: request.key().clone(),
            observation,
            verified: VerifiedFacts {
                compiled_layers: vec![0],
                compiled_projections: vec!["q_proj".into()],
                attribution_checked_layers: vec![0],
                seal_checked_operands: 1,
                invariant_neighbour_layer: Some(1),
                positions: POSITIONS,
                gate_evaluated: request.gate().to_string(),
            },
            execution_note: "scripted AUTO-REP-1b truth".into(),
        })
    }
}

/// Compiles a DIFFERENT map than asked: protects nothing.
struct Liar<'a> {
    source: &'a std::path::Path,
}

impl CandidateCompiler for Liar<'_> {
    fn compile(&self, spec: &RepresentSpec, out: &std::path::Path) -> Result<(), VindexError> {
        let mut spec = spec.clone();
        spec.protect = Default::default();
        compile_representation(self.source, out, &spec).map(drop)
    }
}

fn run_with(
    f: &Fixture,
    snapshot: &mut SearchSnapshot,
    truth: &Truth,
    compiler: &dyn CandidateCompiler,
    budget: usize,
    pins: BTreeMap<GroupKey, GroupChoice>,
) -> Result<CampaignRecord, VindexError> {
    let registry = ExecutorRegistry::new([truth as &dyn ExperimentExecutor]).unwrap();
    run(
        snapshot,
        &CampaignSetup {
            source: &f.source,
            corpus: &f.corpus,
            workdir: &f.workdir,
            spec: &f.spec,
            compiler,
            executors: &registry,
            budget,
            node_limit: u64::MAX,
            pins,
        },
    )
}

/// Bytes each group adds when protected, from the record's own price
/// table: source minus compiled, summed over the group's tensors.
fn protection_cost(f: &Fixture) -> BTreeMap<GroupKey, u64> {
    let footprint = f.snapshot.footprint().unwrap();
    let base = &f.snapshot.space().base_map;
    let mut cost = BTreeMap::new();
    for t in f.snapshot.space().surface.entries() {
        if !base.roles.iter().any(|r| r == t.role.name()) {
            continue;
        }
        let (Some(p), Some(l)) = (
            super::super::policy::projection_of(&t.tensor),
            super::super::policy::layer_of(&t.tensor),
        ) else {
            continue;
        };
        let source = footprint
            .presented(&t.object, &t.tensor, None)
            .unwrap()
            .get();
        let compiled = footprint
            .presented(&t.object, &t.tensor, Some(&base.encoding))
            .unwrap()
            .get();
        *cost.entry(GroupKey::new(p, l)).or_insert(0) += source - compiled;
    }
    cost
}

/// A required group with only a few cheaper protection sets, so the
/// loop's path is short and fully checkable by enumeration.
fn required_group(cost: &BTreeMap<GroupKey, u64>) -> (GroupKey, usize) {
    let groups: Vec<(&GroupKey, u64)> = cost.iter().map(|(g, &c)| (g, c)).collect();
    assert!(
        groups.iter().all(|(_, c)| *c > 0),
        "every protection adds bytes"
    );
    let mut singles: Vec<_> = groups.clone();
    singles.sort_by_key(|(g, c)| (*c, (*g).clone()));
    for (g, c) in singles {
        // Every protection set strictly cheaper than protecting `g` alone.
        let cheaper = (0u64..(1 << groups.len()))
            .filter(|mask| {
                let total: u64 = (0..groups.len())
                    .filter(|i| mask >> i & 1 == 1)
                    .map(|i| groups[i].1)
                    .sum();
                total < c
            })
            .count();
        if (2..=20).contains(&cheaper) {
            return (g.clone(), cheaper);
        }
    }
    panic!("no group with a short, checkable path");
}

fn states(record: &CampaignRecord) -> Vec<RepresentationStateId> {
    record
        .entries
        .iter()
        .map(|e| e.proposal.state.clone())
        .collect()
}

#[test]
fn the_loop_closes_at_the_cheapest_admissible_state() {
    let f = fixture();
    let cost = protection_cost(&f);
    let (required, cheaper) = required_group(&cost);
    let truth = Truth::new(&f, [required.clone()]);
    let mut snapshot = f.snapshot.clone();
    let record = run_with(
        &f,
        &mut snapshot,
        &truth,
        &RepresentCompiler { source: &f.source },
        100,
        BTreeMap::new(),
    )
    .unwrap();

    let last = record.entries.last().unwrap();
    assert_eq!(
        record.outcome,
        CampaignOutcome::Admitted {
            state: last.proposal.state.clone()
        }
    );
    assert_eq!(last.verdict, Verdict::Admitted);
    // The cheapest admissible state protects exactly the required group:
    // every protection adds bytes, so any superset costs more.
    assert_eq!(last.proposal.protected, vec![required]);
    // Every strictly cheaper protection set was measured and refused
    // first, found independently of the solver by enumeration. Sets that
    // tie with the admitted one may precede it in the canonical order.
    let refused = &record.entries[..record.entries.len() - 1];
    assert_eq!(
        refused
            .iter()
            .filter(|e| e.proposal.bytes < last.proposal.bytes)
            .count(),
        cheaper
    );
    assert!(refused
        .iter()
        .all(|e| e.proposal.bytes <= last.proposal.bytes));
    assert!(refused
        .iter()
        .all(|e| matches!(&e.verdict, Verdict::Refused { failures } if !failures.is_empty())));
    assert!(record
        .entries
        .windows(2)
        .all(|w| w[0].proposal.bytes <= w[1].proposal.bytes));
    assert_eq!(record.measurements_spent, record.entries.len());
    assert!(record.entries.iter().all(|e| !e.reused));
    assert_eq!(truth.runs.borrow().len(), record.entries.len());
    assert_eq!(record.revision, CAMPAIGN_REVISION);
}

#[test]
fn each_refusal_is_one_exact_no_good_and_no_state_repeats() {
    let f = fixture();
    let (required, _) = required_group(&protection_cost(&f));
    let truth = Truth::new(&f, [required]);
    let mut snapshot = f.snapshot.clone();
    let record = run_with(
        &f,
        &mut snapshot,
        &truth,
        &RepresentCompiler { source: &f.source },
        100,
        BTreeMap::new(),
    )
    .unwrap();
    let distinct: BTreeSet<_> = states(&record).into_iter().collect();
    assert_eq!(
        distinct.len(),
        record.entries.len(),
        "no state is measured twice"
    );
    let cuts = no_goods(&snapshot);
    let refused: BTreeSet<_> = record
        .entries
        .iter()
        .filter(|e| e.verdict != Verdict::Admitted)
        .map(|e| Cut::ExactNoGood {
            state: e.proposal.state.clone(),
        })
        .map(|c| c.id())
        .collect();
    assert_eq!(cuts.iter().map(Cut::id).collect::<BTreeSet<_>>(), refused);
    // The entry after each refusal names that refusal among its cuts.
    for w in record.entries.windows(2) {
        let cut = Cut::ExactNoGood {
            state: w[0].proposal.state.clone(),
        }
        .id();
        assert!(w[1].proposal.cuts.contains(&cut));
    }
}

#[test]
fn a_candidate_that_is_not_the_proposed_state_is_refused_before_measuring() {
    let f = fixture();
    let (required, _) = required_group(&protection_cost(&f));
    let truth = Truth::new(&f, [required]);
    let mut snapshot = f.snapshot.clone();
    // The first proposal is compile-everything, which the liar matches;
    // the second protects a group, which it does not.
    let err = run_with(
        &f,
        &mut snapshot,
        &truth,
        &Liar { source: &f.source },
        100,
        BTreeMap::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("not the proposed"), "{err}");
    assert_eq!(
        truth.runs.borrow().len(),
        1,
        "the lie was caught before it ran"
    );
}

#[test]
fn rerunning_the_finished_record_reuses_its_reading() {
    let f = fixture();
    let (required, _) = required_group(&protection_cost(&f));
    let truth = Truth::new(&f, [required]);
    let compiler = RepresentCompiler { source: &f.source };
    let mut snapshot = f.snapshot.clone();
    let first = run_with(&f, &mut snapshot, &truth, &compiler, 100, BTreeMap::new()).unwrap();
    let runs = truth.runs.borrow().len();

    let bytes = serde_json::to_vec(&snapshot).unwrap();
    let mut reopened: SearchSnapshot = serde_json::from_slice(&bytes).unwrap();
    let again = run_with(&f, &mut reopened, &truth, &compiler, 100, BTreeMap::new()).unwrap();
    assert_eq!(again.outcome, first.outcome);
    assert_eq!(again.measurements_spent, 0);
    assert_eq!(again.entries.len(), 1);
    assert!(again.entries[0].reused);
    assert_eq!(again.entries[0].key, first.entries.last().unwrap().key);
    assert_eq!(truth.runs.borrow().len(), runs, "nothing was re-run");
    assert_eq!(
        serde_json::to_vec(&reopened).unwrap(),
        bytes,
        "the record is unchanged"
    );
}

#[test]
fn the_budget_bounds_measurements() {
    let f = fixture();
    let (required, cheaper) = required_group(&protection_cost(&f));
    let truth = Truth::new(&f, [required]);
    let mut snapshot = f.snapshot.clone();
    let budget = cheaper - 1;
    let record = run_with(
        &f,
        &mut snapshot,
        &truth,
        &RepresentCompiler { source: &f.source },
        budget,
        BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(record.outcome, CampaignOutcome::BudgetSpent);
    assert_eq!(record.measurements_spent, budget);
    assert_eq!(truth.runs.borrow().len(), budget);
}

#[test]
fn a_problem_with_no_admissible_state_is_exhausted() {
    let f = fixture();
    let groups = group_keys(&f.snapshot.space().surface, &f.snapshot.space().base_map);
    // Free exactly two groups: four states in all.
    let pins: BTreeMap<_, _> = groups[2..]
        .iter()
        .map(|g| (g.clone(), GroupChoice::Compile))
        .collect();
    let truth = Truth::new(&f, [GroupKey::new("no_such_proj", 0)]);
    let mut snapshot = f.snapshot.clone();
    let record = run_with(
        &f,
        &mut snapshot,
        &truth,
        &RepresentCompiler { source: &f.source },
        100,
        pins,
    )
    .unwrap();
    assert_eq!(record.outcome, CampaignOutcome::Exhausted);
    assert_eq!(record.measurements_spent, 4);
    let distinct: BTreeSet<_> = states(&record).into_iter().collect();
    assert_eq!(distinct.len(), 4);
}

#[test]
fn records_and_setups_the_loop_cannot_honestly_run_are_refused() {
    let f = fixture();
    let truth = Truth::new(&f, []);
    let compiler = RepresentCompiler { source: &f.source };
    let refuse = |snapshot: &SearchSnapshot, spec: &RepresentSpec| {
        let registry = ExecutorRegistry::new([&truth as &dyn ExperimentExecutor]).unwrap();
        let mut snapshot = snapshot.clone();
        let err = run(
            &mut snapshot,
            &CampaignSetup {
                source: &f.source,
                corpus: &f.corpus,
                workdir: &f.workdir,
                spec,
                compiler: &compiler,
                executors: &registry,
                budget: 100,
                node_limit: u64::MAX,
                pins: BTreeMap::new(),
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            std::fs::read_dir(&f.workdir).unwrap().next().is_none(),
            "nothing compiled"
        );
        err
    };
    let rebuild =
        |edit: &dyn Fn(&mut SearchSpace, &mut super::super::state::snapshot::SearchConfig)| {
            let mut space = f.snapshot.space().clone();
            let mut config = f.snapshot.config().clone();
            edit(&mut space, &mut config);
            SearchSnapshot::new(space, config, f.snapshot.facts().clone())
        };

    let diagnostic = rebuild(&|_, c| c.standing_intent.scale = EvidenceScale::Diagnostic);
    assert!(refuse(&diagnostic, &f.spec).contains("Authority"));

    let excepted = rebuild(&|s, _| {
        s.base_map.exceptions.push(Exception {
            projection: Some("q_proj".into()),
            layers: None,
            encoding: None,
        })
    });
    assert!(refuse(&excepted, &f.spec).contains("exceptions"));

    let narrowed = rebuild(&|s, _| {
        s.vocabulary = ActionVocabulary::new(s.vocabulary.edits()[1..].iter().cloned()).unwrap()
    });
    assert!(refuse(&narrowed, &f.spec).contains("vocabulary"));

    let mut other = f.spec.clone();
    other.encoding = "Q4_K".into();
    assert!(refuse(&f.snapshot, &other).contains("disagree"));
    assert!(truth.runs.borrow().is_empty());
}
