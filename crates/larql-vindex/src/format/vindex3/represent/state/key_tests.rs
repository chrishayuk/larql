//! **1c: what counts as the same experiment.**
//!
//! The reason `MeasurementKey` exists is R5-F3 — the exchange search
//! re-derived a map it had already measured and rejected, under another
//! name — and the reason it is not keyed on a bare state id is the
//! correction that came with it: a diagnostic reading and an authority
//! reading of one state are two experiments, and a key that could not
//! tell them apart would forbid the escalation the ladder is built on.

use std::collections::BTreeMap;

use super::super::compiler::SourceIdentity;
use super::super::map::{Exception, PrecisionMap};
use super::super::measurement::EvidenceScale;
use super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::policy::Role;
use super::super::quality::{Distribution, LogitEvidence, QualityBank, RoutingEvidence};
use super::*;

// ---------------------------------------------------------------- fixtures

fn model() -> SourceIdentity {
    SourceIdentity {
        manifest_hash: "manifest-aaaa".into(),
        graph_hash: "graph-1111".into(),
        segments: BTreeMap::from([("target.decoder_stack".to_string(), "seg-dddd".to_string())]),
    }
}

fn tensor(projection: &str, shape: Vec<usize>) -> SurfaceTensor {
    SurfaceTensor::new(
        "target.decoder_stack",
        format!("0.self_attn.{projection}.weight"),
        Role::DecoderLinear,
        shape,
    )
}

/// q admissible, v refused by the layout — the surface that produces two
/// realizations of one physical state.
fn surface() -> TensorSurface {
    TensorSurface::new([
        tensor("q_proj", vec![64, 64]),
        tensor("v_proj", vec![64, 24]),
    ])
    .expect("distinct tensors")
}

fn map(exceptions: Vec<Exception>) -> PrecisionMap {
    PrecisionMap {
        name: "m".into(),
        encoding: DTYPE_NVFP4.into(),
        roles: vec!["decoder-linear".into()],
        exceptions,
    }
}

fn state(m: &PrecisionMap) -> RepresentationState {
    RepresentationState::resolve(&model(), &surface(), m, &PackLayoutAdmission)
}

/// The 256 × 32 selection bank, as the Q2a report records it.
fn selection_bank() -> EvidenceBank {
    EvidenceBank::new(
        "kimi-teacher-forced/v1",
        "17d59a6b",
        (0..256).map(|i| format!("seq-{i:03}")),
        32,
    )
    .found_at("/tmp/kimi_quality_bank")
    .described("256 x 32-position teacher-forced sequences from real prose")
}

/// The 32-sequence slice the diagnostic runs over.
fn diagnostic_slice() -> EvidenceBank {
    EvidenceBank::new(
        "kimi-teacher-forced/v1",
        "17d59a6b",
        (0..32).map(|i| format!("seq-{i:03}")),
        32,
    )
}

fn instrument() -> InstrumentSemantics {
    InstrumentSemantics::new(
        "kl(baseline || candidate)",
        "distribution{min,p50,p95,p99,max}",
        "teacher-forced, all positions",
        "q2a-teacher-forced/baseline-vs-overlay",
    )
    .truncated_to(2048)
}

fn dist(p99: f64) -> Option<Distribution> {
    Some(Distribution {
        count: 1,
        min: 0.0,
        p50: 0.0,
        p95: 0.0,
        p99,
        max: p99,
    })
}

fn observation(kl_p99: f64) -> QualityBank {
    QualityBank {
        positions: 8192,
        logits: LogitEvidence {
            kl_p50: 0.0,
            kl_p95: 0.0,
            kl_p99,
            max_logit_delta: 0.0,
            top1_flips: 129,
            top10_changes: 2527,
        },
        routing: RoutingEvidence {
            route_flips: 1305,
            positions_with_route_change: 0,
            layers_with_route_change: 0,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: dist(0.1993),
        },
        min_covered_mass: Some(0.6315),
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: dist(0.0646),
        top10_rank_displacement: None,
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: dist(0.0554),
    }
}

fn key(
    s: &RepresentationState,
    bank: &EvidenceBank,
    scale: EvidenceScale,
    inst: &InstrumentSemantics,
) -> MeasurementKey {
    MeasurementKey::new(s.id(), &bank.id(), scale, &inst.id())
}

/// A registry holding one diagnostic reading of `s`.
fn registry_with(s: &RepresentationState) -> (MeasurementRegistry, MeasurementKey) {
    let k = key(
        s,
        &selection_bank(),
        EvidenceScale::Diagnostic,
        &instrument(),
    );
    let mut r = MeasurementRegistry::new();
    r.record(k.clone(), observation(2.5262e-3)).expect("record");
    (r, k)
}

// --------------------------------------------------- the five dedup cases

#[test]
fn the_same_experiment_on_the_same_state_is_reused() {
    let s = state(&map(vec![]));
    let (r, k) = registry_with(&s);

    let again = key(
        &s,
        &selection_bank(),
        EvidenceScale::Diagnostic,
        &instrument(),
    );
    assert_eq!(again, k);
    assert!(
        r.contains(&again),
        "already measured — do not spend it again"
    );
    assert_eq!(r.get(&again).expect("held").logits.kl_p99, 2.5262e-3);
}

#[test]
fn authority_is_not_deduplicated_against_diagnostic() {
    // The correction R5-F3 came with: dedup on the state alone would
    // forbid the escalation the whole ladder depends on.
    let s = state(&map(vec![]));
    let (r, _) = registry_with(&s);

    let authority = key(
        &s,
        &selection_bank(),
        EvidenceScale::Authority,
        &instrument(),
    );
    assert!(
        !r.contains(&authority),
        "a diagnostic reading is not an authority reading of the same state"
    );
}

#[test]
fn a_different_bank_is_not_deduplicated() {
    let s = state(&map(vec![]));
    let (r, _) = registry_with(&s);

    // Same corpus, same manifest — a 32-of-256 SLICE. The manifest
    // digest alone would have called this a repeat; it is the difference
    // between a diagnostic and an authority run.
    let sliced = diagnostic_slice();
    assert_eq!(
        sliced.manifest_sha256,
        selection_bank().manifest_sha256,
        "same corpus"
    );
    assert_ne!(sliced.id(), selection_bank().id(), "different samples");
    assert!(!r.contains(&key(&s, &sliced, EvidenceScale::Diagnostic, &instrument())));

    assert_eq!(selection_bank().positions(), 8192, "256 x 32");
    assert_eq!(sliced.positions(), 1024);
}

#[test]
fn different_instrument_semantics_are_not_deduplicated() {
    let s = state(&map(vec![]));
    let (r, _) = registry_with(&s);

    // `min_covered_mass` exists because a KL over a truncation covering
    // a third of the mass is a different measurement from one covering
    // all of it — top-128 saw 0.307 of a first position, top-2048 0.729.
    let narrower = instrument().truncated_to(128);
    assert_ne!(narrower.id(), instrument().id());
    assert!(!r.contains(&key(
        &s,
        &selection_bank(),
        EvidenceScale::Diagnostic,
        &narrower
    )));
}

#[test]
fn a_different_realization_of_one_physical_state_reuses_the_measurement() {
    // **The case that joins 1a, 1b and 1c.** `v_proj` is held at source
    // precision by a protection in one map and by a layout refusal in the
    // other. Different realizations, one physical state:
    //
    //     action search   →  distinguish
    //     measurement     →  collapse
    let protected = state(&map(vec![Exception {
        projection: Some("v_proj".into()),
        layers: None,
        encoding: None,
    }]));
    let refused = state(&map(vec![]));

    let a = ResolvedState::new(protected.clone(), LogicalBytes::new(1000));
    let b = ResolvedState::new(refused.clone(), LogicalBytes::new(1000));
    assert_eq!(a.physical_id(), b.physical_id(), "one physical state");
    assert_ne!(a.realization_id(), b.realization_id(), "two realizations");

    let (r, _) = registry_with(&protected);
    let from_the_other = key(
        &refused,
        &selection_bank(),
        EvidenceScale::Diagnostic,
        &instrument(),
    );
    assert!(
        r.contains(&from_the_other),
        "the same bytes were measured; the reading is the other one's too"
    );
}

// ------------------------------------------- identity, not provenance

#[test]
fn a_bank_that_moved_disk_is_the_same_bank() {
    // /tmp has proven ephemeral. A registry that treated a relocated
    // bank as a new one would re-run twenty minutes of instrument time
    // to learn what it already knew — and the same rule
    // `SourceDependency` states for containers.
    let here = selection_bank();
    let moved = EvidenceBank::new(
        here.schema.clone(),
        here.manifest_sha256.clone(),
        here.samples.clone(),
        here.positions_per_sample,
    )
    .found_at("~/chris-models/qbanks/kimi-quality-bank-256x32")
    .described("re-exported after the /tmp clear");

    assert_ne!(here.locator_hint, moved.locator_hint);
    assert_ne!(here.description, moved.description);
    assert_eq!(here.id(), moved.id(), "content, not location");

    let s = state(&map(vec![]));
    let (r, _) = registry_with(&s);
    assert!(r.contains(&key(&s, &moved, EvidenceScale::Diagnostic, &instrument())));
}

#[test]
fn an_implementation_note_is_not_measurement_semantics() {
    // A refactor that changes nothing observable must not split a
    // state's evidence.
    let refactored = instrument().implemented_by("rewritten in the new executor, 6bec0b43");
    assert_ne!(
        refactored.implementation_note,
        instrument().implementation_note
    );
    assert_eq!(refactored.id(), instrument().id());

    let s = state(&map(vec![]));
    let (r, _) = registry_with(&s);
    assert!(r.contains(&key(
        &s,
        &selection_bank(),
        EvidenceScale::Diagnostic,
        &refactored
    )));
}

#[test]
fn every_semantic_field_moves_the_instrument_id() {
    // The other direction: each of these changes what a reading IS.
    let base = instrument().id();
    let variants = [
        InstrumentSemantics::new("nll", "d", "t", "p"),
        InstrumentSemantics::new("m", "p50-only", "t", "p"),
        InstrumentSemantics::new("m", "d", "final position only", "p"),
        InstrumentSemantics::new("m", "d", "t", "single-arm"),
    ];
    for v in &variants {
        assert_ne!(v.id(), base);
    }
    // Untruncated is not a large truncation: one is a claim about the
    // whole distribution, the other about a window.
    let untruncated = InstrumentSemantics::new(
        "kl(baseline || candidate)",
        "distribution{min,p50,p95,p99,max}",
        "teacher-forced, all positions",
        "q2a-teacher-forced/baseline-vs-overlay",
    );
    assert!(untruncated.truncation.is_none());
    assert_ne!(untruncated.id(), base);
}

#[test]
fn a_bank_is_its_samples_in_order_not_a_count() {
    // A different 32 of the same 256 is a different experiment, and a
    // count cannot say which 32.
    let a = EvidenceBank::new("s", "m", ["seq-000", "seq-001"], 32);
    let b = EvidenceBank::new("s", "m", ["seq-000", "seq-002"], 32);
    let reordered = EvidenceBank::new("s", "m", ["seq-001", "seq-000"], 32);
    assert_eq!(a.sample_count(), b.sample_count());
    assert_ne!(a.id(), b.id(), "different samples");
    assert_ne!(a.id(), reordered.id(), "order is part of the bank");
    assert_ne!(
        a.id(),
        EvidenceBank::new("s", "m", ["seq-000", "seq-001"], 16).id(),
        "half the positions per sample is half the experiment"
    );
    assert_ne!(
        a.id(),
        EvidenceBank::new("other-schema", "m", ["seq-000", "seq-001"], 32).id()
    );
    assert_ne!(
        a.id(),
        EvidenceBank::new("s", "other-manifest", ["seq-000", "seq-001"], 32).id()
    );
}

// ------------------------------------------------------------- registry

#[test]
fn re_recording_a_reproduced_reading_is_a_no_op_and_a_contradiction_is_refused() {
    let s = state(&map(vec![]));
    let (mut r, k) = registry_with(&s);
    assert_eq!(r.len(), 1);

    // A control witness reproducing is exactly what a replayed round
    // should do.
    r.record(k.clone(), observation(2.5262e-3))
        .expect("a reproduced reading");
    assert_eq!(r.len(), 1);

    // A different reading under the same key says the experiment is not
    // reproducible. Silently keeping either would hide that.
    let err = r
        .record(k.clone(), observation(3.3532e-3))
        .expect_err("two readings, one key");
    assert!(format!("{err}").contains("not reproducible"), "{err}");
    assert_eq!(
        r.get(&k).expect("held").logits.kl_p99,
        2.5262e-3,
        "the held reading is unchanged"
    );
}

#[test]
fn a_registry_answers_what_is_known_about_one_state() {
    let s = state(&map(vec![]));
    let (mut r, _) = registry_with(&s);
    r.record(
        key(
            &s,
            &selection_bank(),
            EvidenceScale::Authority,
            &instrument(),
        ),
        observation(3.3532e-3),
    )
    .expect("authority");
    let other = state(&map(vec![Exception {
        projection: Some("q_proj".into()),
        layers: None,
        encoding: None,
    }]));
    r.record(
        key(
            &other,
            &selection_bank(),
            EvidenceScale::Diagnostic,
            &instrument(),
        ),
        observation(1.0e-3),
    )
    .expect("another state");

    assert_eq!(r.len(), 3);
    assert!(!r.is_empty());
    assert_eq!(r.of_state(s.id()).count(), 2, "diagnostic and authority");
    assert_eq!(r.of_state(other.id()).count(), 1);
    assert_eq!(r.keys().count(), 3);

    let scales: Vec<EvidenceScale> = r.of_state(s.id()).map(|(k, _)| k.scale()).collect();
    assert!(scales.contains(&EvidenceScale::Diagnostic));
    assert!(scales.contains(&EvidenceScale::Authority));
}

#[test]
fn a_key_names_its_four_parts_and_orders_by_its_digest() {
    let s = state(&map(vec![]));
    let bank = selection_bank();
    let inst = instrument();
    let k = key(&s, &bank, EvidenceScale::Authority, &inst);

    assert_eq!(k.state(), s.id());
    assert_eq!(k.bank(), &bank.id());
    assert_eq!(k.scale(), EvidenceScale::Authority);
    assert_eq!(k.instrument(), &inst.id());
    assert_eq!(k.as_str().len(), 64);
    assert_eq!(k.short().len(), 12);
    assert_eq!(format!("{k}"), k.as_str());

    let diagnostic = key(&s, &bank, EvidenceScale::Diagnostic, &inst);
    assert_ne!(k, diagnostic);
    assert_eq!(k.cmp(&diagnostic), k.as_str().cmp(diagnostic.as_str()));
    assert_eq!(k.partial_cmp(&diagnostic), Some(k.cmp(&diagnostic)));

    assert!(MEASUREMENT_KEY_VERSION.starts_with("measurement-key/"));
    assert!(EVIDENCE_BANK_ID_VERSION.starts_with("evidence-bank-id/"));
    assert!(INSTRUMENT_SEMANTICS_ID_VERSION.starts_with("instrument-semantics-id/"));
    assert_eq!(bank.id().short().len(), 12);
    assert_eq!(format!("{}", bank.id()), bank.id().as_str());
    assert_eq!(inst.id().short().len(), 12);
    assert_eq!(format!("{}", inst.id()), inst.id().as_str());
}

#[test]
fn a_registry_survives_serialization() {
    let s = state(&map(vec![]));
    let (r, k) = registry_with(&s);
    let json = serde_json::to_string(&r).expect("serialize");
    let back: MeasurementRegistry = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, r);
    assert!(back.contains(&k));
    assert_eq!(MeasurementRegistry::default().len(), 0);

    // A list of records, not a map: a key is four identities, and
    // flattening it to a JSON object key would throw away the parts a
    // reader needs to see WHY two observations are distinct.
    let doc: serde_json::Value = serde_json::from_str(&json).expect("json");
    let record = &doc["observations"][0];
    assert_eq!(record["key"]["state"], s.id().as_str());
    assert_eq!(record["key"]["scale"], "Diagnostic");
    assert_eq!(record["key"]["bank"], selection_bank().id().as_str());
    assert!(record["observation"]["logits"]["kl_p99"].is_number());
    assert!(
        record["key"].get("verdict").is_none() && record["key"].get("contract").is_none(),
        "a key says what was measured, never what it meant"
    );
}

#[test]
fn a_stored_registry_naming_one_experiment_twice_is_refused() {
    // The in-memory registry cannot hold a duplicate, so a file with one
    // was written by something that bypassed `record` — and loading it
    // would silently keep whichever came last.
    let s = state(&map(vec![]));
    let (r, _) = registry_with(&s);
    let mut doc: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&r).expect("serialize")).expect("json");
    let first = doc["observations"][0].clone();
    doc["observations"]
        .as_array_mut()
        .expect("array")
        .push(first);

    let err = serde_json::from_value::<MeasurementRegistry>(doc).expect_err("duplicate key");
    assert!(format!("{err}").contains("recorded twice"), "{err}");
}
