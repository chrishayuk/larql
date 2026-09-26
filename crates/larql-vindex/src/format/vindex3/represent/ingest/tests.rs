//! Graph-based fixture: real encoded/compiled bytes, no K3 or manifest-only authority.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use super::super::{
    actuate::{executor::Observed, prepare::PreparedExperiment},
    compile_representation,
    compiler::read_source_identity,
    map::{Exception, PrecisionMap},
    measure::outcome::VerifiedFacts,
    measurement::EvidenceScale,
    policy::classify_in,
    state::{
        self, fixtures,
        snapshot::{SearchSnapshot, SearchSpace},
        ActionVocabulary, LogicalBytes, MapEdit, PackLayoutAdmission, RepresentationState,
        RepresentationStateGraph, ResolvedState, SurfaceTensor, TensorSurface, TransitionPolicy,
    },
    RepresentSpec,
};
use super::artifact::MeasurementArtifact;
use super::state_evidence::{ArtifactStateEvidence, EstablishedState};
use super::*;
use crate::format::vindex3::{
    encode::segment::read_segment_header,
    fixtures::{dense_f32_model, encode_fixture_container},
    index::Vindex3Index,
};

pub(super) struct Fixture {
    pub dir: tempfile::TempDir,
    pub source: PathBuf,
    pub candidate: PathBuf,
    pub corpus: PathBuf,
    pub snapshot: SearchSnapshot,
    pub prepared: PreparedExperiment,
    /// The corpus depth this fixture was built at. Every observation it
    /// seals reports it, and the bank and `VerifiedFacts` declare it.
    pub positions: u64,
}

impl Fixture {
    /// The default fixture: 8 positions, as every caller before
    /// PARETO-1 expects. Unchanged, and deliberately so — LOOP-1 is a
    /// closed milestone and its fixture must stay byte-for-byte
    /// behaviourally identical.
    pub fn new() -> Self {
        Self::with_positions(8)
    }

    /// A fixture at a chosen corpus depth.
    ///
    /// Depth is not a free parameter. `TailSupportPolicy::route_cal_1`
    /// needs `5.0 / (1 - 0.99)` = 500 observations to support a p99, so
    /// below 500 positions every p99 criterion is unpriceable at
    /// authority scale and a positive-gain move classifies
    /// `Unscorable`. PARETO-1's priced witness runs at exactly 500 and
    /// its negative control at 499.
    pub fn with_positions(positions: u64) -> Self {
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
        let mut spec = RepresentSpec::nvfp4();
        spec.protect = spec.protect.layers(1, 1);
        let mut base_map =
            PrecisionMap::from_policy(spec.map_name(), &spec.encoding, &spec.roles, &spec.protect);
        base_map.exceptions.push(Exception {
            projection: None,
            layers: None,
            encoding: None,
        });
        let vocabulary = ActionVocabulary::new([
            MapEdit::new(
                "compile-all",
                Exception {
                    projection: None,
                    layers: Some((0, 0)),
                    encoding: Some(spec.encoding.clone()),
                },
            ),
            MapEdit::new(
                "compile-q",
                Exception {
                    projection: Some("q_proj".into()),
                    layers: Some((0, 0)),
                    encoding: Some(spec.encoding.clone()),
                },
            ),
        ])
        .unwrap();
        let corpus = dir.path().join("corpus");
        std::fs::create_dir(&corpus).unwrap();
        let rows = vec![0u8; positions as usize * 64 * 4];
        std::fs::write(corpus.join("seq_0.f32"), &rows).unwrap();
        let manifest = serde_json::to_vec(&serde_json::json!({
            "sequences":1,"positions":positions,"hidden":64,
            "token_ids":[vec![0u32; positions as usize]],"regime":"teacher-forced",
            "payload_authority":"teacher-forced-bank-payload/v1",
            "payloads":{"seq_0.f32":{"len":rows.len(),"sha256":super::super::compile::hash_bytes(&rows)}}
        })).unwrap();
        std::fs::write(corpus.join("manifest.json"), &manifest).unwrap();
        let mut protocol = fixtures::protocol();
        protocol.bank = state::EvidenceBank::new(
            "kimi-teacher-forced/v1",
            super::super::compile::hash_bytes(&manifest),
            ["seq-000"],
            positions as u32,
        );
        let seed = fixtures::PricedRecord::new(&source)
            .with_protocol(protocol)
            .build();
        let model = read_source_identity(&source).unwrap();
        let root = RepresentationState::resolve(&model, &surface, &base_map, &PackLayoutAdmission);
        let mut facts = seed.facts().clone();
        let source_bytes = facts
            .accounting
            .as_ref()
            .unwrap()
            .tensors()
            .map(|(_, fact)| fact.logical_bytes.get())
            .sum();
        facts.graph = RepresentationStateGraph::new(
            TransitionPolicy::StrictlyImprovingPhysical,
            ResolvedState::new(root, LogicalBytes::new(source_bytes)),
        );
        let snapshot = SearchSnapshot::new(
            SearchSpace {
                surface,
                base_map,
                vocabulary,
                applied: BTreeSet::new(),
            },
            seed.config().clone(),
            facts,
        );
        let prepared = PreparedExperiment::of(&snapshot);
        assert!(prepared.is_ready(), "{prepared:?}");
        let candidate = dir.path().join("candidate");
        drop(compile_representation(&source, &candidate, &spec).unwrap());
        let actual = ArtifactStateEvidence::establish(&candidate).unwrap();
        assert_eq!(
            actual.established(),
            prepared.request().unwrap().key().state()
        );
        let compiled: Vec<_> = actual
            .state()
            .decisions()
            .decisions()
            .iter()
            .filter(|d| matches!(d.encoding, state::ResolvedEncoding::Compiled(_)))
            .collect();
        assert_eq!(compiled.len(), 7);
        assert!(compiled.iter().all(|d| d.tensor.starts_with("0.")));
        Self {
            dir,
            source,
            candidate,
            corpus,
            snapshot,
            prepared,
            positions,
        }
    }
    pub fn evidence(&self) -> EstablishedState {
        ArtifactStateEvidence::establish(&self.candidate).unwrap()
    }
    pub fn observed(&self, kl: f64) -> Observed {
        self.observed_with(kl, 0)
    }
    /// The same observation with the ROUTE FLIP count settable.
    ///
    /// `observed` hardcoded zero, which left `Statistic::RouteFlipRate`
    /// constant across every candidate and therefore unable to order
    /// anything — one of the two statistics ROUTE-CAL-1 registers as an
    /// ordering proxy. A caller that needs two independent orderable
    /// dimensions needs this one.
    pub fn observed_with(&self, kl: f64, route_flips: u64) -> Observed {
        let mut observation = fixtures::authority_reading(kl, route_flips);
        observation.positions = self.positions;
        observation.routing.route_weight_mass_moved = None;
        observation.top10_mass_displaced = None;
        observation.top1_mass_displaced = None;
        Observed {
            key: self.prepared.request().unwrap().key().clone(),
            observation: observation.into(),
            verified: VerifiedFacts {
                compiled_layers: vec![0],
                compiled_projections: [
                    "down_proj",
                    "gate_proj",
                    "up_proj",
                    "k_proj",
                    "o_proj",
                    "q_proj",
                    "v_proj",
                ]
                .map(str::to_owned)
                .to_vec(),
                attribution_checked_layers: vec![0],
                seal_checked_operands: 7,
                invariant_neighbour_layer: Some(1),
                positions: self.positions,
                gate_evaluated: self.snapshot.gate().unwrap().id().to_string(),
            }
            .into(),
            execution_note: "tiny deterministic observation fixture".into(),
        }
    }
    pub fn artifact(&self, kl: f64) -> MeasurementArtifact {
        MeasurementArtifact::from_execution(&self.prepared, &self.observed(kl), &self.evidence())
            .unwrap()
    }
    pub fn sources(&self) -> IngestionSources<'_> {
        IngestionSources {
            container: &self.source,
            candidate: &self.candidate,
            corpus: &self.corpus,
        }
    }
}

fn refused_without_change(
    f: &Fixture,
    snapshot: &mut SearchSnapshot,
    artifact: &MeasurementArtifact,
) -> IngestionRefusal {
    let before = snapshot.clone();
    let bytes = serde_json::to_vec(snapshot).unwrap();
    let next = snapshot.next_experiment().unwrap();
    let frontier = snapshot.frontier();
    let promotions = format!(
        "{:?}",
        snapshot.promotion_candidates(EvidenceScale::Authority)
    );
    let refusal = ingest(snapshot, &f.prepared, artifact, &f.sources()).unwrap_err();
    assert_eq!(*snapshot, before);
    assert_eq!(serde_json::to_vec(snapshot).unwrap(), bytes);
    assert_eq!(snapshot.next_experiment().unwrap(), next);
    assert_eq!(snapshot.frontier(), frontier);
    assert_eq!(
        format!(
            "{:?}",
            snapshot.promotion_candidates(EvidenceScale::Authority)
        ),
        promotions
    );
    refusal
}

#[test]
fn freshly_sealed_incomplete_run_with_correct_positions_and_gate_refuses() {
    let f = Fixture::new();
    let mut observed = f.observed(0.01);
    observed.verified = VerifiedFacts {
        positions: 8,
        gate_evaluated: f.snapshot.gate().unwrap().id().to_string(),
        ..Default::default()
    }
    .into();
    assert!(!observed.verified.as_kimi().unwrap().complete());
    let artifact =
        MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence()).unwrap();
    artifact.verify_seal().unwrap();
    let refusal = refused_without_change(&f, &mut f.snapshot.clone(), &artifact);
    let shown = refusal.to_string();
    for obligation in [
        "compiled layers",
        "compiled projections",
        "attribution",
        "seal/read",
        "invariant neighbour",
    ] {
        assert!(shown.contains(obligation), "{shown} omitted {obligation}");
    }
}

#[test]
fn each_missing_run_validity_obligation_refuses_transactionally() {
    let f = Fixture::new();
    for obligation in [
        "compiled layers",
        "compiled projections",
        "attribution checks covering compiled layers",
        "seal/read witness",
        "invariant neighbour",
        "non-zero positions",
        "gate",
        "partial attribution",
        "all",
    ] {
        let mut observed = f.observed(0.01);
        assert!(observed.verified.as_kimi().unwrap().complete());
        let report = observed.verified.as_kimi_mut().unwrap();
        match obligation {
            "compiled layers" => report.compiled_layers.clear(),
            "compiled projections" => report.compiled_projections.clear(),
            "attribution checks covering compiled layers" => {
                report.attribution_checked_layers.clear()
            }
            "seal/read witness" => report.seal_checked_operands = 0,
            "invariant neighbour" => report.invariant_neighbour_layer = None,
            "non-zero positions" => report.positions = 0,
            "gate" => report.gate_evaluated.clear(),
            "partial attribution" => report.compiled_layers.push(1),
            "all" => *report = VerifiedFacts::default(),
            _ => unreachable!(),
        }
        assert!(!report.complete());
        let artifact =
            MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence()).unwrap();
        artifact.verify_seal().unwrap();
        let refusal = refused_without_change(&f, &mut f.snapshot.clone(), &artifact);
        let IngestionRefusal::IncompleteRun {
            missing,
            observed: actual,
        } = &refusal
        else {
            panic!("{obligation}: {refusal}");
        };
        assert_eq!(actual.as_ref(), &observed.verified);
        if obligation == "all" {
            assert_eq!(missing.len(), 7);
        } else {
            let expected = if obligation == "partial attribution" {
                "attribution checks covering compiled layers"
            } else {
                obligation
            };
            assert!(missing.iter().any(|m| m == expected), "{refusal}");
        }
        for absent in missing {
            assert!(refusal.to_string().contains(absent), "{refusal}");
        }
        assert!(refusal
            .to_string()
            .contains(&format!("{:?}", observed.verified)));
    }
}

#[test]
fn accepted_once_and_identical_duplicate_is_transactionally_idempotent() {
    let f = Fixture::new();
    let a = f.artifact(0.01);
    assert!(a.verified().as_kimi().unwrap().complete());
    let mut snapshot = f.snapshot.clone();
    let first = ingest(&mut snapshot, &f.prepared, &a, &f.sources()).unwrap();
    assert!(first.recorded);
    assert_eq!(snapshot.measurements().len(), 1);
    assert_eq!(first.accepted.candidate().established(), a.key().state());
    assert_eq!(first.accepted.observation(), a.observation());
    assert_eq!(first.accepted.artifact_seal(), a.seal());
    assert_eq!(
        first.accepted.source_semantic_digest(),
        a.source_semantic_digest()
    );
    assert_eq!(
        first.accepted.bank_manifest_sha256(),
        f.snapshot.protocol().unwrap().bank.manifest_sha256
    );
    let before = snapshot.clone();
    assert!(
        !ingest(&mut snapshot, &f.prepared, &a, &f.sources())
            .unwrap()
            .recorded
    );
    assert_eq!(snapshot, before);
    let mut expected = f.snapshot.facts().clone();
    expected.measurements = snapshot.measurements().clone();
    assert_eq!(snapshot.facts(), &expected);
}

#[test]
fn valid_y_is_established_then_refused_for_requested_x_without_any_scientific_change() {
    let f = Fixture::new();
    let mut source_spec = RepresentSpec::nvfp4();
    source_spec.protect = super::super::policy::Protections::default().projection("q_proj");
    let y = f.dir.path().join("valid-y");
    drop(compile_representation(&f.source, &y, &source_spec).unwrap());
    let evidence = ArtifactStateEvidence::establish(&y).unwrap();
    let requested = f.prepared.request().unwrap().key().state().clone();
    assert_ne!(evidence.established(), &requested);
    let a = MeasurementArtifact::from_execution(&f.prepared, &f.observed(0.01), &evidence).unwrap();
    a.verify_seal().unwrap();
    assert_eq!(a.key().state(), &requested);
    let mut snapshot = f.snapshot.clone();
    let wrong = Fixture { candidate: y, ..f };
    let r = refused_without_change(&wrong, &mut snapshot, &a);
    assert!(
        matches!(&r, IngestionRefusal::Authority {what,..} if what == "candidate representation state")
    );
    assert!(r.to_string().contains(requested.as_str()));
    assert!(r.to_string().contains(evidence.established().as_str()));
}

#[test]
fn conflicting_duplicate_preserves_both_observations_and_all_facts() {
    let f = Fixture::new();
    let a = f.artifact(0.01);
    let b = f.artifact(0.02);
    let mut snapshot = f.snapshot.clone();
    ingest(&mut snapshot, &f.prepared, &a, &f.sources()).unwrap();
    let r = refused_without_change(&f, &mut snapshot, &b);
    let IngestionRefusal::Conflict(c) = &r else {
        panic!("{r:?}")
    };
    assert_eq!(&c.expected, a.observation());
    assert_eq!(&c.observed, b.observation());
    assert_eq!(&c.key, a.key());
    let shown = r.to_string();
    assert!(shown.contains("0.01") && shown.contains("0.02"));
    assert!(
        shown.contains(a.key().bank().as_str()) && shown.contains(a.key().instrument().as_str())
    );
}

#[test]
fn tampered_artifact_and_malformed_observation_refuse_before_writes() {
    let f = Fixture::new();
    let mut snapshot = f.snapshot.clone();
    let a = f.artifact(0.01);
    let mut doc = serde_json::to_value(&a).unwrap();
    doc["execution_note"] = serde_json::json!("changed");
    let tampered = serde_json::from_value(doc).unwrap();
    assert!(matches!(
        refused_without_change(&f, &mut snapshot, &tampered),
        IngestionRefusal::Artifact(_)
    ));
    for kind in ["positions", "nonfinite", "count", "empty-report"] {
        let mut observed = f.observed(0.01);
        match kind {
            "positions" => observed.observation.as_kimi_mut().unwrap().positions = 0,
            "nonfinite" => observed.observation.as_kimi_mut().unwrap().logits.kl_p99 = f64::NAN,
            "count" => {
                observed
                    .observation
                    .as_kimi_mut()
                    .unwrap()
                    .logits
                    .top1_flips = 9
            }
            _ => observed.verified = VerifiedFacts::default().into(),
        }
        let a = MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence()).unwrap();
        refused_without_change(&f, &mut snapshot, &a);
    }
}

#[test]
fn protocol_source_bank_and_candidate_mutations_refuse_transactionally() {
    for kind in [
        "protocol",
        "gate",
        "source",
        "bank",
        "candidate",
        "missing-candidate",
    ] {
        let f = Fixture::new();
        let a = f.artifact(0.01);
        let mut snapshot = f.snapshot.clone();
        match kind {
            "protocol" | "gate" => {
                let mut config = snapshot.config().clone();
                if kind == "protocol" {
                    config
                        .protocol
                        .as_mut()
                        .unwrap()
                        .bank
                        .samples
                        .push("other".into());
                } else {
                    match config.gate.as_mut().unwrap() {
                        super::super::reading::Gate::Kimi(gate) => gate.positions_min += 1,
                        super::super::reading::Gate::Plan(gate) => gate.positions_min += 1,
                    }
                }
                snapshot =
                    SearchSnapshot::new(snapshot.space().clone(), config, snapshot.facts().clone());
            }
            "source" => {
                let p = f.source.join("index.json");
                let mut d: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
                d["model"] = serde_json::json!("different");
                std::fs::write(p, serde_json::to_vec(&d).unwrap()).unwrap();
            }
            "bank" => std::fs::write(f.corpus.join("manifest.json"), b"changed").unwrap(),
            "candidate" => std::fs::write(f.candidate.join("index.json"), b"changed").unwrap(),
            _ => std::fs::remove_file(f.candidate.join("candidate.json")).unwrap(),
        }
        let before = snapshot.clone();
        // A malformed protocol can itself make next_experiment unavailable;
        // compare both results instead of requiring a successful selection.
        let next = format!("{:?}", snapshot.next_experiment());
        assert!(
            ingest(&mut snapshot, &f.prepared, &a, &f.sources()).is_err(),
            "{kind}"
        );
        assert_eq!(snapshot, before, "{kind}");
        assert_eq!(format!("{:?}", snapshot.next_experiment()), next);
    }
}

#[test]
fn a_valid_sealed_artifact_for_another_authorisation_names_both_keys() {
    use super::super::actuate::{prepare::Ready, request::MeasurementRequest};
    let f = Fixture::new();
    let expected = f.prepared.request().unwrap().key();
    let other_key = state::MeasurementKey::new(
        expected.state(),
        expected.bank(),
        EvidenceScale::Diagnostic,
        expected.instrument(),
    );
    let request = MeasurementRequest::of(
        &f.snapshot,
        &other_key,
        &BTreeSet::from(["compile-all".into()]),
    )
    .unwrap();
    let other = PreparedExperiment::Ready(Box::new(Ready {
        request,
        physical_delta: 0,
        routes: 1,
        considered: 1,
    }));
    let observed = Observed {
        key: other_key.clone(),
        ..f.observed(0.01)
    };
    let a = MeasurementArtifact::from_execution(&other, &observed, &f.evidence()).unwrap();
    a.verify_seal().unwrap();
    let r = refused_without_change(&f, &mut f.snapshot.clone(), &a);
    assert!(matches!(&r,IngestionRefusal::Authority{what,..} if what=="measurement key"));
    assert!(r.to_string().contains(expected.as_str()));
    assert!(r.to_string().contains(other_key.as_str()));
}

#[test]
fn caller_verdict_fields_are_rejected_on_artifact_deserialization() {
    let f = Fixture::new();
    for field in ["verdict", "rank", "promotion", "admissible"] {
        let mut doc = serde_json::to_value(f.artifact(0.01)).unwrap();
        doc[field] = serde_json::json!("PASS");
        assert!(
            serde_json::from_value::<MeasurementArtifact>(doc).is_err(),
            "{field}"
        );
    }
}

#[test]
fn unreadable_authorities_and_invalid_distributions_cannot_change_facts() {
    for kind in ["source", "bank", "distribution", "schema", "no-protocol"] {
        let f = Fixture::new();
        let mut a = f.artifact(0.01);
        let mut snapshot = f.snapshot.clone();
        match kind {
            "source" => std::fs::remove_file(f.source.join("index.json")).unwrap(),
            "bank" => std::fs::remove_file(f.corpus.join("manifest.json")).unwrap(),
            "distribution" => {
                let mut observed = f.observed(0.01);
                observed.observation.as_kimi_mut().unwrap().top1_margin =
                    Some(super::super::quality::Distribution {
                        count: 0,
                        min: 0.0,
                        p50: 1.0,
                        p95: 0.5,
                        p99: 2.0,
                        max: 3.0,
                    });
                a = MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence())
                    .unwrap();
            }
            "schema" => {
                let mut doc = serde_json::to_value(&snapshot).unwrap();
                doc["schema"] = serde_json::json!("unknown/v9");
                snapshot = serde_json::from_value(doc).unwrap();
            }
            _ => {
                let mut config = snapshot.config().clone();
                config.protocol = None;
                snapshot =
                    SearchSnapshot::new(snapshot.space().clone(), config, snapshot.facts().clone());
            }
        }
        let before = snapshot.clone();
        assert!(
            ingest(&mut snapshot, &f.prepared, &a, &f.sources()).is_err(),
            "{kind}"
        );
        assert_eq!(snapshot, before);
    }
    let f = Fixture::new();
    let a = f.artifact(0.01);
    let mut snapshot = f.snapshot.clone();
    assert!(ingest(
        &mut snapshot,
        &PreparedExperiment::Exhausted,
        &a,
        &f.sources()
    )
    .is_err());
    assert_eq!(snapshot, f.snapshot);
}

#[test]
fn persisted_artifact_round_trip_accepts_and_missing_fields_refuse() {
    let f = Fixture::new();
    let a = f.artifact(0.01);
    let bytes = serde_json::to_vec(&a).unwrap();
    let mut snapshot = f.snapshot.clone();
    assert!(
        ingest_bytes(&mut snapshot, &f.prepared, &bytes, &f.sources())
            .unwrap()
            .recorded
    );
    for field in [
        "key",
        "observation",
        "candidate_authority_digest",
        "verified",
        "seal",
    ] {
        let mut doc = serde_json::to_value(&a).unwrap();
        doc.as_object_mut().unwrap().remove(field);
        let before = snapshot.clone();
        assert!(matches!(
            ingest_bytes(
                &mut snapshot,
                &f.prepared,
                &serde_json::to_vec(&doc).unwrap(),
                &f.sources()
            ),
            Err(IngestionRefusal::Malformed(_))
        ));
        assert_eq!(snapshot, before);
    }
}

#[test]
fn self_consistent_but_empty_or_unimplemented_protocols_are_refused() {
    for kind in ["empty-bank", "unknown-procedure"] {
        let f = Fixture::new();
        let mut config = f.snapshot.config().clone();
        let protocol = config.protocol.as_mut().unwrap();
        if kind == "empty-bank" {
            protocol.bank.samples.clear();
        } else {
            protocol.procedure = "unknown/v1".into();
        }
        config.standing_intent = protocol.intent(EvidenceScale::Authority);
        let mut snapshot = SearchSnapshot::new(
            f.snapshot.space().clone(),
            config,
            f.snapshot.facts().clone(),
        );
        let prepared = PreparedExperiment::of(&snapshot);
        assert!(prepared.is_ready());
        let observed = Observed {
            key: prepared.request().unwrap().key().clone(),
            ..f.observed(0.01)
        };
        let artifact =
            MeasurementArtifact::from_execution(&prepared, &observed, &f.evidence()).unwrap();
        let before = snapshot.clone();
        let refusal = ingest(&mut snapshot, &prepared, &artifact, &f.sources()).unwrap_err();
        assert!(matches!(refusal, IngestionRefusal::Malformed(_)));
        assert_eq!(snapshot, before);
    }
}

#[test]
fn changed_bank_rows_with_unchanged_manifest_and_key_refuse() {
    let f = Fixture::new();
    let a = f.artifact(0.01);
    let path = f.corpus.join("seq_0.f32");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] ^= 1;
    std::fs::write(path, bytes).unwrap();
    let mut snapshot = f.snapshot.clone();
    let refusal = refused_without_change(&f, &mut snapshot, &a);
    assert!(refusal.to_string().contains("bank payload"), "{refusal}");
}

#[test]
fn bank_authority_requires_seals_dimensions_and_the_executed_sample_prefix() {
    for kind in [
        "legacy",
        "version",
        "regime",
        "positions",
        "hidden",
        "inventory",
        "overflow",
        "seal",
        "length",
        "unreadable",
        "schema",
        "sample-order",
    ] {
        let mut f = Fixture::new();
        let path = f.corpus.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        match kind {
            "legacy" => {
                manifest
                    .as_object_mut()
                    .unwrap()
                    .remove("payload_authority");
            }
            "version" => manifest["payload_authority"] = serde_json::json!("unknown/v1"),
            "regime" => manifest["regime"] = serde_json::json!("free-running"),
            "positions" => manifest["positions"] = serde_json::json!(7),
            "hidden" => manifest["hidden"] = serde_json::json!(0),
            "inventory" => manifest["token_ids"] = serde_json::json!([]),
            "overflow" => manifest["hidden"] = serde_json::json!(u64::MAX),
            "seal" => manifest["payloads"] = serde_json::json!({}),
            "length" => manifest["payloads"]["seq_0.f32"]["len"] = serde_json::json!(1),
            "unreadable" => std::fs::remove_file(f.corpus.join("seq_0.f32")).unwrap(),
            _ => {}
        }
        let bytes = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(path, &bytes).unwrap();
        let mut config = f.snapshot.config().clone();
        let protocol = config.protocol.as_mut().unwrap();
        protocol.bank.manifest_sha256 = super::super::compile::hash_bytes(&bytes);
        if kind == "schema" {
            protocol.bank.schema = "unknown/v9".into();
        }
        if kind == "sample-order" {
            protocol.bank.samples = vec!["seq-001".into()];
        }
        config.standing_intent = protocol.intent(EvidenceScale::Authority);
        f.snapshot = SearchSnapshot::new(
            f.snapshot.space().clone(),
            config,
            f.snapshot.facts().clone(),
        );
        f.prepared = PreparedExperiment::of(&f.snapshot);
        assert!(f.prepared.is_ready());
        let a = f.artifact(0.01);
        let mut snapshot = f.snapshot.clone();
        let refusal = refused_without_change(&f, &mut snapshot, &a);
        assert!(refusal.to_string().contains("bank"), "{kind}: {refusal}");
    }
}

#[test]
fn bank_relocation_preserves_authority_but_reordering_does_not() {
    let mut f = Fixture::new();
    let path = f.corpus.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["sequences"] = serde_json::json!(2);
    manifest["token_ids"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!([1, 1, 1, 1, 1, 1, 1, 1]));
    manifest["payloads"]["seq_1.f32"] = manifest["payloads"]["seq_0.f32"].clone();
    std::fs::copy(f.corpus.join("seq_0.f32"), f.corpus.join("seq_1.f32")).unwrap();
    let bytes = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let mut config = f.snapshot.config().clone();
    let protocol = config.protocol.as_mut().unwrap();
    protocol.bank.manifest_sha256 = super::super::compile::hash_bytes(&bytes);
    protocol.bank.samples = vec!["seq-000".into(), "seq-001".into()];
    config.standing_intent = protocol.intent(EvidenceScale::Authority);
    f.snapshot = SearchSnapshot::new(
        f.snapshot.space().clone(),
        config,
        f.snapshot.facts().clone(),
    );
    f.prepared = PreparedExperiment::of(&f.snapshot);
    let observed = |fixture: &Fixture| {
        let mut o = fixture.observed(0.01);
        o.observation.as_kimi_mut().unwrap().positions = 16;
        o.verified.as_kimi_mut().unwrap().positions = 16;
        o
    };
    let artifact =
        MeasurementArtifact::from_execution(&f.prepared, &observed(&f), &f.evidence()).unwrap();
    let relocated = f.dir.path().join("moved-corpus");
    std::fs::rename(&f.corpus, &relocated).unwrap();
    f.corpus = relocated;
    assert!(
        ingest(
            &mut f.snapshot.clone(),
            &f.prepared,
            &artifact,
            &f.sources()
        )
        .unwrap()
        .recorded
    );
    let mut config = f.snapshot.config().clone();
    let protocol = config.protocol.as_mut().unwrap();
    protocol.bank.samples.reverse();
    config.standing_intent = protocol.intent(EvidenceScale::Authority);
    f.snapshot = SearchSnapshot::new(
        f.snapshot.space().clone(),
        config,
        f.snapshot.facts().clone(),
    );
    f.prepared = PreparedExperiment::of(&f.snapshot);
    let artifact =
        MeasurementArtifact::from_execution(&f.prepared, &observed(&f), &f.evidence()).unwrap();
    let r = refused_without_change(&f, &mut f.snapshot.clone(), &artifact);
    assert!(r.to_string().contains("bank sample order"));
}

#[test]
fn every_ingestion_refusal_preserves_its_actionable_authority() {
    use super::super::candidate_authority::CandidateAuthorityRefusal as C;
    use super::artifact::ArtifactRefusal as A;
    fn must_say(r: &IngestionRefusal) -> Vec<String> {
        match r {
            IngestionRefusal::Artifact(a) => match a {
                A::SealBroken { expected, observed }
                | A::SubstitutedExperiment { expected, observed } => {
                    vec![expected.clone(), observed.clone()]
                }
                A::Unsealable { detail } | A::NotAuthorised { detail } => vec![detail.clone()],
            },
            IngestionRefusal::Candidate(c) => match c {
                C::Invalid(detail) => vec![detail.clone()],
                C::Missing => vec!["no completed representation authority".into()],
                C::RequestedState { expected, observed } => {
                    vec![expected.to_string(), observed.to_string()]
                }
                C::Version {
                    field,
                    expected,
                    observed,
                } => vec![field.clone(), expected.clone(), observed.clone()],
                C::Binding {
                    what,
                    expected,
                    observed,
                } => vec![what.clone(), expected.clone(), observed.clone()],
                C::StoredState { stored, recomputed } => {
                    vec![stored.to_string(), recomputed.to_string()]
                }
            },
            IngestionRefusal::Authority {
                what,
                expected,
                observed,
            } => vec![what.clone(), expected.clone(), observed.clone()],
            IngestionRefusal::Malformed(detail) => vec![detail.clone()],
            IngestionRefusal::IncompleteRun { missing, observed } => {
                let mut required = missing.clone();
                required.push(format!("{observed:?}"));
                required
            }
            IngestionRefusal::Conflict(c) => vec![
                format!("{:?}", c.key),
                format!("{:?}", c.expected),
                format!("{:?}", c.observed),
            ],
        }
    }
    let f = Fixture::new();
    let x = f.evidence().established().clone();
    let y = RepresentationState::resolve(
        &read_source_identity(&f.source).unwrap(),
        &f.snapshot.space().surface,
        &f.snapshot.space().base_map,
        &PackLayoutAdmission,
    )
    .id()
    .clone();
    let mut cases = vec![
        IngestionRefusal::Authority {
            what: "bank".into(),
            expected: "expected-bank".into(),
            observed: "observed-bank".into(),
        },
        IngestionRefusal::Malformed("missing input rows".into()),
        IngestionRefusal::IncompleteRun {
            missing: vec!["compiled layers".into(), "seal/read witness".into()],
            observed: Box::new(VerifiedFacts::default().into()),
        },
        IngestionRefusal::Conflict(Box::new(state::key::MeasurementConflict {
            key: f.observed(0.01).key,
            expected: f.observed(0.01).observation,
            observed: f.observed(0.02).observation,
        })),
    ];
    cases.extend(
        [
            A::SealBroken {
                expected: "sealed".into(),
                observed: "changed".into(),
            },
            A::SubstitutedExperiment {
                expected: "requested".into(),
                observed: "restated".into(),
            },
            A::Unsealable {
                detail: "not serializable".into(),
            },
            A::NotAuthorised {
                detail: "no ready experiment".into(),
            },
        ]
        .into_iter()
        .map(IngestionRefusal::Artifact),
    );
    cases.extend(
        [
            C::Invalid("unreadable candidate".into()),
            C::Missing,
            C::RequestedState {
                expected: x.clone(),
                observed: y.clone(),
            },
            C::Version {
                field: "schema".into(),
                expected: "v1".into(),
                observed: "v99".into(),
            },
            C::Binding {
                what: "payload".into(),
                expected: "sealed-payload".into(),
                observed: "actual-payload".into(),
            },
            C::StoredState {
                stored: y,
                recomputed: x,
            },
        ]
        .into_iter()
        .map(IngestionRefusal::Candidate),
    );
    for r in cases {
        for required in must_say(&r) {
            assert!(r.to_string().contains(&required), "{r} omitted {required}");
        }
    }
}

#[test]
fn changed_baseline_segment_refuses_even_when_candidate_and_source_declaration_stay_valid() {
    let f = Fixture::new();
    let artifact = f.artifact(0.01);
    let before = read_source_identity(&f.source).unwrap();
    let entry = &before.semantic.representations[0];
    let path = f.source.join(&entry.segment);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    // Break the source/candidate hardlink before replacing baseline bytes.
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(read_source_identity(&f.source).unwrap(), before);
    assert_eq!(f.evidence().established(), artifact.key().state());
    let refusal = refused_without_change(&f, &mut f.snapshot.clone(), &artifact);
    let shown = refusal.to_string();
    assert!(shown.contains("source container payloads"), "{shown}");
    assert!(shown.contains(&entry.segment_sha256));
    assert!(shown.contains(&super::super::compile::hash_bytes(&bytes)));
}

/// MEASURE-PLAN-2 W2 at ingestion: a plan-v1 reading sealed into an
/// artifact for a Kimi-gated record is refused before anything reads its
/// values, and the record is untouched.
#[test]
fn a_plan_reading_is_refused_by_a_kimi_gated_record() {
    use super::super::measure::plan::metrics::Aggregate;
    use super::super::reading::{Observation, PlanObservation, PlanProcedure};
    let f = Fixture::new();
    let all = Aggregate {
        positions: 8,
        kl_mean: 1e-4,
        kl_p50: 1e-4,
        kl_p99: 2e-4,
        kl_max: 3e-4,
        top1_agreement: 1.0,
        top5_overlap_mean: 1.0,
        delta_nll_mean: Some(0.0),
        max_abs_delta_mean: 0.1,
        max_abs_delta_p99: 0.2,
    };
    let mut observed = f.observed(0.001);
    observed.observation = Observation::Plan(PlanObservation {
        procedure: PlanProcedure,
        sequences: 1,
        positions: 8,
        all,
        by_category: vec![],
        by_margin_band: vec![],
    });
    let artifact =
        MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence()).unwrap();
    let mut snapshot = f.snapshot.clone();
    let refusal = refused_without_change(&f, &mut snapshot, &artifact);
    assert!(
        matches!(&refusal, IngestionRefusal::Authority { what, .. } if what == "reading kind of the procedure"),
        "{refusal}"
    );
}

/// Integrity refusals the record must name: a bank manifest that cannot be
/// read, and a source container whose declared seals are intact but whose
/// payload bytes are not what they seal. Each refuses without changing a
/// fact, and names the authority that failed.
#[test]
fn an_unreadable_bank_manifest_or_corrupted_source_payload_is_refused_by_name() {
    let f = Fixture::new();
    let a = f.artifact(0.01);
    let mut snapshot = f.snapshot.clone();
    std::fs::remove_file(
        f.corpus
            .join(super::super::actuate::artifacts::BANK_MANIFEST),
    )
    .unwrap();
    let r = refused_without_change(&f, &mut snapshot, &a);
    assert!(
        matches!(&r, IngestionRefusal::Authority { what, .. } if what == "bank manifest"),
        "{r:?}"
    );

    let f = Fixture::new();
    let a = f.artifact(0.01);
    let mut snapshot = f.snapshot.clone();
    let index: Vindex3Index =
        serde_json::from_slice(&std::fs::read(f.source.join("index.json")).unwrap()).unwrap();
    let segment = f
        .source
        .join(&index.representations.values().next().unwrap().segment);
    let mut bytes = std::fs::read(&segment).unwrap();
    *bytes.last_mut().unwrap() ^= 0xff;
    std::fs::write(&segment, bytes).unwrap();
    let r = refused_without_change(&f, &mut snapshot, &a);
    assert!(
        matches!(&r, IngestionRefusal::Authority { what, .. } if what == "source container payloads"),
        "{r:?}"
    );
}

/// A run can report a complete, sealed set of facts that still disagree
/// with the record: positions measured other than the bank's, or a gate
/// evaluated other than the record's. The seal proves the report is the
/// executor's; ingestion must still refuse facts that are not this
/// record's, naming which.
#[test]
fn sealed_facts_that_disagree_with_the_record_are_refused_by_name() {
    for (what, edit) in [("reported measured positions", 0u8), ("reported gate", 1u8)] {
        let f = Fixture::new();
        let mut observed = f.observed(0.01);
        if edit == 0 {
            observed.verified.positions += 1;
        } else {
            observed.verified.gate_evaluated = "a-gate-this-record-never-named/v1".into();
        }
        assert!(
            observed.verified.complete(),
            "{what}: the report is otherwise complete"
        );
        let artifact =
            MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence()).unwrap();
        artifact.verify_seal().unwrap();
        let r = refused_without_change(&f, &mut f.snapshot.clone(), &artifact);
        assert!(
            matches!(&r, IngestionRefusal::Authority { what: named, .. } if named == what),
            "{what}: {r:?}"
        );
    }
}
