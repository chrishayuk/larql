//! Transition C: one paired claim over production control-plane entry points.
//! Numerical observations and historical runtime checks are explicitly synthetic.
use std::cell::Cell;
use std::path::{Path, PathBuf};

use super::super::{
    actuate::{
        artifacts::DeclaredArtifacts,
        executor::{
            ArtifactLocator, ExecutionRefusal, ExecutorRegistry, ExperimentExecutor, Observed,
        },
        prepare::PreparedExperiment,
        request::MeasurementRequest,
    },
    compile_representation,
    measurement::EvidenceScale,
    state::{
        candidate::{CandidateSet, PreMeasurementPrune},
        search_policy::Selection,
        snapshot::SearchSnapshot,
    },
    RepresentSpec,
};
use super::{
    artifact::MeasurementArtifact, ingest_bytes, state_evidence::ArtifactStateEvidence,
    tests::Fixture, IngestionRefusal, IngestionSources,
};

/// A deterministic observation provider, not a numerical execution oracle.
/// Real registry dispatch must hand it the expected request and persisted inputs.
struct FixtureExecutor<'a> {
    procedure: &'a str,
    response: Observed,
    paths: [PathBuf; 3],
    calls: Cell<usize>,
}

impl ExperimentExecutor for FixtureExecutor<'_> {
    fn procedure(&self) -> &str {
        self.procedure
    }

    fn execute(
        &self,
        request: &MeasurementRequest,
        artifacts: &dyn ArtifactLocator,
    ) -> Result<Observed, ExecutionRefusal> {
        assert_eq!(request.key(), &self.response.key);
        assert_eq!(
            [
                artifacts.container(request)?,
                artifacts.corpus(request)?,
                artifacts.candidate(request)?
            ],
            self.paths,
        );
        self.calls.set(self.calls.get() + 1);
        Ok(self.response.clone())
    }
}

/// Return only the authorisation and persisted artifact bytes. The executor,
/// observation and independent candidate-reader result are dropped here.
fn dispatch_artifact(
    f: &Fixture,
    snapshot: &SearchSnapshot,
    candidate: &Path,
    incomplete: bool,
) -> (PreparedExperiment, Vec<u8>) {
    let prepared = PreparedExperiment::of(snapshot);
    let request = prepared.request().unwrap();
    let mut response = f.observed(0.01);
    if incomplete {
        response.verified.invariant_neighbour_layer = None;
    }
    let executor = FixtureExecutor {
        procedure: request.procedure(),
        response,
        paths: [f.source.clone(), f.corpus.clone(), candidate.to_path_buf()],
        calls: Cell::new(0),
    };
    let registry = ExecutorRegistry::new([&executor as &dyn ExperimentExecutor]).unwrap();
    let locator = DeclaredArtifacts::new()
        .container_at(&f.source)
        .corpus_at(&f.corpus)
        .overlay_at(request.key().state(), candidate);
    let observed = registry.execute(request, &locator).unwrap();
    assert_eq!(executor.calls.get(), 1);
    let established = ArtifactStateEvidence::establish(candidate).unwrap();
    let artifact = MeasurementArtifact::from_execution(&prepared, &observed, &established).unwrap();
    artifact.verify_seal().unwrap();
    let bytes = serde_json::to_vec(&artifact).unwrap();
    (prepared, bytes)
}

fn candidates(snapshot: &SearchSnapshot) -> CandidateSet {
    let footprint = snapshot.footprint().unwrap();
    snapshot
        .generator(snapshot.layout().unwrap(), &footprint)
        .candidates(&snapshot.space().applied, snapshot.standing_intent())
        .unwrap()
}

#[test]
fn accepted_evidence_changes_future_selection_and_rejected_evidence_cannot() {
    let f = Fixture::new();
    let s0_bytes = serde_json::to_vec(&f.snapshot).unwrap();
    let s0: SearchSnapshot = serde_json::from_slice(&s0_bytes).unwrap();
    let initial = candidates(&s0);
    assert_eq!(initial.census().enumerated, 2);
    assert_eq!(initial.census().eligible, 2);
    assert!(initial.census().conserves());
    assert!(s0.measurements().is_empty());

    // Name both expected experiments before dispatch or ingestion, from the
    // two frozen actions. No result is used to choose the expected successor.
    let key_for = |action: &str| {
        initial
            .eligible()
            .find(|c| c.action.added == [action])
            .unwrap()
            .intended_key
            .clone()
    };
    let a = key_for("compile-all");
    let b = key_for("compile-q");
    assert_ne!(a, b);
    let selected_a = s0.next_experiment().unwrap();
    assert!(matches!(
        selected_a,
        Selection::Ranked { considered: 2, .. }
    ));
    assert_eq!(selected_a.opportunity().unwrap().key, a);

    // B/Y is a real, independently established alternative, compiled without
    // the requested key entering its writer. Layer 1 stays source-backed.
    let mut q_spec = RepresentSpec::nvfp4();
    q_spec.protect = q_spec.protect.layers(1, 1);
    for projection in [
        "down_proj",
        "gate_proj",
        "up_proj",
        "k_proj",
        "o_proj",
        "v_proj",
    ] {
        q_spec.protect = q_spec.protect.projection(projection);
    }
    let q_candidate = f.dir.path().join("candidate-q");
    drop(compile_representation(&f.source, &q_candidate, &q_spec).unwrap());
    assert_eq!(
        ArtifactStateEvidence::establish(&q_candidate)
            .unwrap()
            .established(),
        b.state()
    );

    let (prepared, artifact) = dispatch_artifact(&f, &s0, &f.candidate, false);
    assert_eq!(prepared.request().unwrap().key(), &a);
    let mut s1 = s0.clone();
    let accepted = ingest_bytes(&mut s1, &prepared, &artifact, &f.sources()).unwrap();
    assert!(accepted.recorded);
    assert_eq!(accepted.accepted.key(), &a);
    assert_eq!(s1.measurements().len(), 1);
    assert_eq!(s1.space(), s0.space());
    assert_eq!(s1.config(), s0.config());
    let mut other_facts = s1.facts().clone();
    other_facts.measurements = s0.measurements().clone();
    assert_eq!(&other_facts, s0.facts());

    let after = candidates(&s1);
    assert_eq!(after.census().eligible, 1);
    assert_eq!(after.census().already_observed, 1);
    assert!(after.census().conserves());
    assert!(after.pruned().any(|(_, reason)| matches!(reason,
        PreMeasurementPrune::AlreadyObserved { key } if key == &a)));
    let selected_b = s1.next_experiment().unwrap();
    assert!(matches!(selected_b, Selection::Sole(_)));
    assert_eq!(selected_b.opportunity().unwrap().key, b);
    assert_eq!(PreparedExperiment::of(&s1).request().unwrap().key(), &b);

    // Reopen S1 and replay A through ingestion. It remains one observation,
    // and the persisted record, not an in-memory selected-next cache, chooses B.
    let s1_bytes = serde_json::to_vec(&s1).unwrap();
    let mut reopened: SearchSnapshot = serde_json::from_slice(&s1_bytes).unwrap();
    assert_eq!(reopened.next_experiment().unwrap(), selected_b);
    assert!(
        !ingest_bytes(&mut reopened, &prepared, &artifact, &f.sources())
            .unwrap()
            .recorded
    );
    assert_eq!(serde_json::to_vec(&reopened).unwrap(), s1_bytes);
    assert_eq!(reopened.next_experiment().unwrap(), selected_b);

    // Negative twins start from the SAME S0. Both records are freshly sealed
    // and dispatched for A: one omits a validity witness; one supplies valid Y.
    for (candidate, incomplete) in [(&f.candidate, true), (&q_candidate, false)] {
        let mut unchanged: SearchSnapshot = serde_json::from_slice(&s0_bytes).unwrap();
        let frontier = unchanged.frontier();
        let promotions = format!(
            "{:?}",
            unchanged.promotion_candidates(EvidenceScale::Authority)
        );
        let (authorised, plausible) = dispatch_artifact(&f, &unchanged, candidate, incomplete);
        assert_eq!(authorised.request().unwrap().key(), &a);
        let sources = IngestionSources {
            container: &f.source,
            candidate,
            corpus: &f.corpus,
        };
        let refusal = ingest_bytes(&mut unchanged, &authorised, &plausible, &sources).unwrap_err();
        if incomplete {
            assert!(matches!(refusal, IngestionRefusal::IncompleteRun { .. }));
        } else {
            assert!(matches!(&refusal, IngestionRefusal::Authority { what, .. }
                if what == "candidate representation state"));
            assert!(refusal.to_string().contains(a.state().as_str()));
            assert!(refusal.to_string().contains(b.state().as_str()));
        }
        assert_eq!(unchanged, s0);
        assert_eq!(serde_json::to_vec(&unchanged).unwrap(), s0_bytes);
        assert_eq!(unchanged.frontier(), frontier);
        assert_eq!(
            format!(
                "{:?}",
                unchanged.promotion_candidates(EvidenceScale::Authority)
            ),
            promotions
        );
        assert_eq!(unchanged.next_experiment().unwrap(), selected_a);
        assert_eq!(PreparedExperiment::of(&unchanged), prepared);
    }
}
