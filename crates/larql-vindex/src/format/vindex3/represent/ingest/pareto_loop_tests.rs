//! **PARETO-1 through the production ingestion path.**
//!
//! The five direct worlds prove the decision KERNEL: given candidates
//! that share a tier and a physical gain, accepted quality values decide
//! the preference. They build their measurement registry by
//! `record_fixture`, which the freeze forbids for newly introduced
//! observations — "no raw registry injection".
//!
//! Here the observations are real sealed measurement artifacts, admitted
//! through `ingest_bytes` with independent source, candidate and bank
//! verification and complete `VerifiedFacts`.
//!
//! **Why this layer classifies `Unscorable` where the direct worlds
//! classify `Priced`.** The bank family is now the same — the populated
//! `authority_reading`, distributions intact. What still differs is
//! `positions`. `TailSupportPolicy::route_cal_1` needs
//! `5.0 / (1 - 0.99)` = 500 observations to support a p99, and this
//! fixture's corpus manifest declares 8. `KlP99` is therefore
//! unpriceable at AUTHORITY scale, a criterion's evidence is missing,
//! and a positive-gain move is `Unscorable`.
//!
//! It still ORDERS at diagnostic scale, because
//! `SearchCalibrationRegistry::route_cal_1` registers it as an
//! `OrderingProxy` there and that lookup hits before any tail fallback.
//! Ordering and pricing are different questions and this fixture can
//! answer only one of them.
//!
//! Closing that gap means enlarging LOOP-1's corpus from 8 positions to
//! at least 500, which would retroactively alter a closed milestone's
//! fixture. It is recorded rather than done. Both candidates share the
//! class, so stage 1 still cannot separate them and the frontier is
//! still what decides.
//!
//! **Ingestion writes only `facts.measurements`** (`ingest.rs`). It
//! creates no graph edge and no byte ledger, and `promotion_candidates`
//! skips any edge whose ends lack either. Evidence ingestion alone is
//! INTENTIONALLY insufficient to derive preference: promotion composes
//! admitted measurements with independently held graph and
//! physical-cost facts. Those facts are pinned here and identical
//! between worlds — never manufactured from the measurements to make a
//! decision come out. They are the same frozen facts the direct worlds
//! use — `state::fixtures::pareto_ledger` and `pareto_cost_model` — so
//! there is one definition of them, not two.
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
    assessment::MoveClass,
    compile_representation,
    decision::{decide_promotion, PromotionDecision},
    measurement::EvidenceScale,
    search_evidence::SearchCalibrationRegistry,
    state::{
        candidate::CandidateSet, fixtures, snapshot::SearchSnapshot, Action, MeasurementKey,
        Provenance, ResolvedState,
    },
    RepresentSpec,
};
use super::{
    artifact::MeasurementArtifact, ingest_bytes, state_evidence::ArtifactStateEvidence,
    tests::Fixture, IngestionSources,
};

use fixtures::{PARETO_BETTER as BETTER, PARETO_WORSE as WORSE};

/// **A sealed observation on the POPULATED bank family.**
///
/// `Fixture::observed_with` nulls `route_weight_mass_moved`,
/// `top10_mass_displaced` and `top1_mass_displaced`, which is LOOP-1's
/// bank family and classifies a move `Unscorable`. PARETO-1 is pinned
/// to the populated family, so the integration layer must carry it too
/// — otherwise this is a different experiment that happens to reach the
/// same comparator.
///
/// LOOP-1's helper is deliberately NOT changed: it is used by a closed
/// milestone, and editing it would retroactively alter that fixture.
/// Only the observation is replaced; the complete `VerifiedFacts`
/// authority comes from the fixture unmodified.
fn pareto_observed(f: &Fixture, kl: f64, route_flips: u64) -> Observed {
    let mut observed = f.observed_with(kl, route_flips);
    let mut bank = fixtures::authority_reading(kl, route_flips);
    // The only protocol-scale adaptation this tiny fixture needs. The
    // behavioural distributions stay POPULATED.
    bank.positions = 8;
    observed.observation = bank;
    observed.execution_note = "PARETO-1 populated authority-shaped fixture".into();
    observed
}

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

/// Dispatch the snapshot's own next experiment and seal the result at
/// the requested quality. Returns the authorisation and the artifact
/// bytes exactly as a persisted record would carry them.
fn dispatch(
    f: &Fixture,
    prepared: &PreparedExperiment,
    candidate: &Path,
    quality: (f64, u64),
    incomplete: bool,
) -> Vec<u8> {
    let request = prepared.request().unwrap();
    let mut response = pareto_observed(f, quality.0, quality.1);
    response.key = request.key().clone();
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
    let artifact = MeasurementArtifact::from_execution(prepared, &observed, &established).unwrap();
    artifact.verify_seal().unwrap();
    serde_json::to_vec(&artifact).unwrap()
}

fn candidates(snapshot: &SearchSnapshot) -> CandidateSet {
    let footprint = snapshot.footprint().unwrap();
    snapshot
        .generator(snapshot.layout().unwrap(), &footprint)
        .candidates(&snapshot.space().applied, snapshot.standing_intent())
        .unwrap()
}

/// Seed the PARENT baseline reading into the authoritative pre-state.
///
/// `promotion_candidates` skips an edge unless BOTH ends carry a
/// reading, and the root is not a candidate — no experiment is ever
/// dispatched for it. In a real search the baseline is already measured
/// and persisted before any move is proposed, so it belongs to the
/// pre-state rather than to the observations under test.
///
/// This is NOT the raw injection the freeze forbids. That arm governs
/// "newly introduced observations" — the candidate values whose effect
/// on preference is the claim. Those enter only through `ingest_bytes`.
/// The baseline is identical in every world and carries no candidate
/// value, so it cannot be what moves a preference.
fn with_baseline(snapshot: &SearchSnapshot, template: &MeasurementKey) -> SearchSnapshot {
    let mut facts = snapshot.facts().clone();
    let root = facts.graph.root().clone();
    let key = baseline_key(&root, template);
    let mut bank = fixtures::authority_reading(5.0e-3, 32);
    bank.positions = 8;
    facts
        .measurements
        .record_fixture(key, bank)
        .expect("the baseline is not already held");
    SearchSnapshot::new(snapshot.space().clone(), snapshot.config().clone(), facts)
}

/// The baseline's key, derived from a candidate key so that bank, scale
/// and instrument are the snapshot's own rather than invented here.
fn baseline_key(
    root: &super::super::state::RepresentationStateId,
    template: &MeasurementKey,
) -> MeasurementKey {
    MeasurementKey::new(
        root,
        template.bank(),
        EvidenceScale::Authority,
        template.instrument(),
    )
}

/// Supply the physical facts ingestion does not write, and the registry
/// the fixture's config omits. Measurements are carried through
/// untouched — this adds facts, it never edits evidence.
fn priced(snapshot: &SearchSnapshot, edges: &[(Action, ResolvedState)]) -> SearchSnapshot {
    let mut config = snapshot.config().clone();
    config.calibrations = SearchCalibrationRegistry::route_cal_1();

    let mut facts = snapshot.facts().clone();
    let root = facts.graph.root().clone();
    let mut ledgers = std::collections::BTreeMap::from([(
        root.clone(),
        fixtures::pareto_ledger("root", 16_384, 16_384),
    )]);
    for (i, (action, child)) in edges.iter().enumerate() {
        if facts.graph.node(child.physical_id()).is_none() {
            facts
                .graph
                .apply(
                    &root,
                    action.clone(),
                    child.clone(),
                    Provenance::new("pareto-1/opt-6"),
                )
                .expect("the generator offered it, so it is physically lighter");
        }
        // Both children remove 8,192 of 32,768 bytes, so both predict
        // gpu_ms_saved = 2.0 and stage 5 cannot separate them.
        let _ = i;
        ledgers.insert(
            child.physical_id().clone(),
            fixtures::pareto_ledger("child", 8_192, 16_384),
        );
    }
    facts.byte_ledgers = ledgers;
    facts.execution_cost = fixtures::pareto_cost_model();

    SearchSnapshot::new(snapshot.space().clone(), config, facts)
}

fn decide(snapshot: &SearchSnapshot, edges: &[(Action, ResolvedState)]) -> PromotionDecision {
    let priced = priced(snapshot, edges);
    let candidates = priced
        .promotion_candidates(EvidenceScale::Authority)
        .expect("the cost model covers this model");
    for c in &candidates {
        let s = &c.promotion.assessment.ranking_score;
        println!(
            "OPT6 row  id={:<28} class={:?} tier={} gain={}",
            c.id,
            s.class,
            s.class.tier(),
            s.gpu_ms_saved
        );
        // Pinned, with its cause stated in the module header: 8 corpus
        // positions cannot support a p99 that needs 500, so KlP99 is
        // unpriceable at authority scale.
        assert_eq!(s.class, MoveClass::Unscorable);
        assert_eq!(s.gpu_ms_saved, 2.0, "the frozen physical gain");
    }
    if candidates.len() == 2 {
        assert_eq!(
            candidates[0]
                .promotion
                .assessment
                .ranking_score
                .class
                .tier(),
            candidates[1]
                .promotion
                .assessment
                .ranking_score
                .class
                .tier(),
            "stage 1 must not separate them"
        );
    }
    decide_promotion(
        &candidates,
        &priced.config().calibrations,
        &priced.config().tail_support,
    )
}

/// Compile the second candidate independently, as LOOP-1 does: a real
/// alternative established without the requested key entering its
/// writer.
fn compile_q(f: &Fixture) -> PathBuf {
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
    q_candidate
}

/// **The accepted path.** Two observations admitted through OPT-6, and
/// the preference follows the values they carried.
#[test]
fn values_admitted_through_opt6_decide_the_preference() {
    let f = Fixture::new();
    let q_candidate = compile_q(&f);
    let s0 = f.snapshot.clone();

    // Name both edges before any observation exists.
    let initial = candidates(&s0);
    assert_eq!(initial.census().enumerated, 2);
    let edge = |added: &str| {
        let c = initial
            .eligible()
            .find(|c| c.action.added == [added])
            .unwrap();
        (c.action.clone(), c.child.clone())
    };
    let edges = [edge("compile-all"), edge("compile-q")];

    // The parent baseline is seeded into the PRE-state, before either
    // candidate artifact exists. The root is never a candidate and no
    // experiment is ever dispatched for it, yet `promotion_candidates`
    // needs a reading at BOTH ends of an edge. Seeding it here — once,
    // shared by both worlds — leaves no route for it to explain a flip.
    let template = initial
        .eligible()
        .next()
        .expect("two eligible")
        .intended_key
        .clone();
    let root = s0.facts().graph.root().clone();
    let baseline = baseline_key(&root, &template);
    let s0 = with_baseline(&s0, &template);
    assert!(
        s0.measurements().contains(&baseline),
        "the baseline must exist before any candidate artifact is ingested"
    );
    assert_eq!(s0.measurements().len(), 1, "and it is the ONLY reading");
    assert_eq!(
        candidates(&s0).census().enumerated,
        2,
        "seeding the root must not disturb the candidate set"
    );

    // Two worlds over the SAME pre-state, differing only in which
    // candidate's observation carries which accepted values.
    let run = |first: (f64, u64), second: (f64, u64)| {
        let mut s = s0.clone();
        let prepared = PreparedExperiment::of(&s);
        let artifact = dispatch(&f, &prepared, &f.candidate, first, false);
        assert!(
            ingest_bytes(&mut s, &prepared, &artifact, &f.sources())
                .unwrap()
                .recorded
        );
        let sources_q = IngestionSources {
            container: &f.source,
            candidate: &q_candidate,
            corpus: &f.corpus,
        };
        let prepared_q = PreparedExperiment::of(&s);
        let artifact_q = dispatch(&f, &prepared_q, &q_candidate, second, false);
        assert!(
            ingest_bytes(&mut s, &prepared_q, &artifact_q, &sources_q)
                .unwrap()
                .recorded
        );
        assert_eq!(s.measurements().len(), 3, "baseline plus two candidates");
        s
    };

    let w1 = run(BETTER, WORSE);
    let w2 = run(WORSE, BETTER);

    // The observed key SETS are identical; only the values differ.
    let keys = |s: &SearchSnapshot| {
        let mut v: Vec<String> = s.measurements().keys().map(|k| k.as_str().into()).collect();
        v.sort();
        v
    };
    assert_eq!(keys(&w1), keys(&w2), "identical observed key sets");
    assert_eq!(
        w1.measurements().get(&baseline),
        w2.measurements().get(&baseline),
        "the parent baseline reading must be identical in both worlds"
    );
    assert_ne!(
        serde_json::to_vec(&w1).unwrap(),
        serde_json::to_vec(&w2).unwrap(),
        "the two worlds must differ somewhere, and it must be the values"
    );

    let d1 = decide(&w1, &edges);
    let d2 = decide(&w2, &edges);
    println!("OPT6 world 1 -> {d1:?}");
    println!("OPT6 world 2 -> {d2:?}");

    let winner = |d: &PromotionDecision| match d {
        PromotionDecision::SelectForAuthority {
            candidate,
            evidence,
        } => {
            assert!(
                !evidence.decided_by_physical_gain,
                "physical gain is inert across the pair here"
            );
            assert!(!evidence.deciding.is_empty(), "a proxy must have decided");
            candidate.clone()
        }
        other => panic!("expected a selection, got {other:?}"),
    };
    let a = winner(&d1);
    let b = winner(&d2);
    assert_ne!(
        a, b,
        "swapping only the accepted values must change the preferred candidate"
    );

    // **Cross-layer equivalence.** The same frozen values, decided by
    // the same comparator, through sealed ingestion instead of a
    // directly built registry.
    //
    // Identity cannot be compared across the layers: the direct worlds
    // stand on synthetic states (`p`/`t1`/`s2`) and this one on real
    // compiled containers, so the candidate LABELS necessarily differ.
    // Everything the claim rests on is compared.
    let direct = |world: fixtures::ParetoWorld| {
        let snap = world.snapshot();
        let candidates = snap
            .promotion_candidates(EvidenceScale::Authority)
            .expect("cost");
        decide_promotion(
            &candidates,
            &snap.config().calibrations,
            &snap.config().tail_support,
        )
    };
    for (ingested, direct) in [
        (&d1, direct(fixtures::ParetoWorld::inert(BETTER, WORSE))),
        (&d2, direct(fixtures::ParetoWorld::inert(WORSE, BETTER))),
    ] {
        match (ingested, &direct) {
            (
                PromotionDecision::SelectForAuthority { evidence: i, .. },
                PromotionDecision::SelectForAuthority { evidence: d, .. },
            ) => {
                assert_eq!(i.deciding, d.deciding, "the same proxies decided");
                assert_eq!(
                    i.decided_by_physical_gain, d.decided_by_physical_gain,
                    "the same STAGE decided"
                );
                assert_eq!(
                    i.dominated.len(),
                    d.dominated.len(),
                    "the same dominance shape"
                );
            }
            other => panic!("layers disagree on the decision kind: {other:?}"),
        }
    }

    // Reopen each world and reproduce the decision and its evidence.
    for (world, expected) in [(&w1, &d1), (&w2, &d2)] {
        let bytes = serde_json::to_vec(world).unwrap();
        let reopened: SearchSnapshot = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            format!("{:?}", decide(&reopened, &edges)),
            format!("{expected:?}"),
            "a reopened snapshot must reproduce its preference and evidence"
        );
    }
}

/// **C4 — evidence that was not admitted cannot change the preference.**
///
/// Two rejection arms, each asserting the strongest transactional form:
/// the serialized snapshot, the measurement registry and the
/// `PromotionDecision` are all identical before and after the refusal.
#[test]
fn c4_rejected_evidence_changes_neither_facts_nor_preference() {
    let f = Fixture::new();
    let q_candidate = compile_q(&f);
    let s0 = f.snapshot.clone();
    let initial = candidates(&s0);
    let edge = |added: &str| {
        let c = initial
            .eligible()
            .find(|c| c.action.added == [added])
            .unwrap()
            .clone();
        (c.action, c.child)
    };
    let edges = [edge("compile-all"), edge("compile-q")];
    let template = initial
        .eligible()
        .next()
        .expect("two eligible")
        .intended_key
        .clone();
    let s0 = with_baseline(&s0, &template);

    // A world with one admitted observation, so there is a real
    // preference for a refusal to fail to disturb.
    let mut base = s0.clone();
    let prepared_a = PreparedExperiment::of(&base);
    let artifact_a = dispatch(&f, &prepared_a, &f.candidate, BETTER, false);
    assert!(
        ingest_bytes(&mut base, &prepared_a, &artifact_a, &f.sources())
            .unwrap()
            .recorded
    );

    // Admit the SECOND observation too, so the baseline preference is
    // CONTESTED — one candidate dominating another on named proxies. A
    // refusal that preserves a one-candidate answer preserves much less
    // than one that preserves a decided contest.
    let prepared_b = PreparedExperiment::of(&base);
    let artifact_b = dispatch(&f, &prepared_b, &q_candidate, WORSE, false);
    let sources_q = IngestionSources {
        container: &f.source,
        candidate: &q_candidate,
        corpus: &f.corpus,
    };
    assert!(
        ingest_bytes(&mut base, &prepared_b, &artifact_b, &sources_q)
            .unwrap()
            .recorded
    );
    assert_eq!(base.measurements().len(), 3, "baseline plus two candidates");

    let before_bytes = serde_json::to_vec(&base).unwrap();
    let before_registry = base.measurements().clone();
    let before = decide(&base, &edges);
    // Without this the refusal arms compare one vacuum to another: an
    // EmptySet preserved is not a preference preserved.
    assert!(
        matches!(
            &before,
            PromotionDecision::SelectForAuthority { evidence, .. }
                if !evidence.dominated.is_empty() && !evidence.deciding.is_empty()
        ),
        "C4 needs a CONTESTED preference to preserve, got {before:?}"
    );
    let before_decision = format!("{before:?}");
    println!("C4 baseline preference -> {before_decision}");

    // Arm 1 — an incomplete run report on the next authorised
    // experiment. Arm 2 — the SAME MeasurementKey already on the record,
    // re-sealed with different accepted values: a conflict inside one
    // history, which must refuse rather than replace.
    let arms: [(&str, (f64, u64), bool); 2] = [
        ("incomplete run", BETTER, true),
        ("same key, conflicting values", WORSE, false),
    ];

    for (name, quality, incomplete) in arms {
        let mut world: SearchSnapshot = serde_json::from_slice(&before_bytes).unwrap();
        let plausible = dispatch(&f, &prepared_a, &f.candidate, quality, incomplete);
        let refusal = ingest_bytes(&mut world, &prepared_a, &plausible, &f.sources()).unwrap_err();
        println!("C4 [{name}] -> {refusal}");

        assert_eq!(
            serde_json::to_vec(&world).unwrap(),
            before_bytes,
            "[{name}] the serialized snapshot moved"
        );
        assert_eq!(
            world.measurements(),
            &before_registry,
            "[{name}] the measurement registry moved"
        );
        assert_eq!(
            format!("{:?}", decide(&world, &edges)),
            before_decision,
            "[{name}] the preference moved"
        );
    }
}
