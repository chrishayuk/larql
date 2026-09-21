use super::super::super::actuate::executor::Observed;
use super::super::super::actuate::prepare::PreparedExperiment;
use super::super::super::measure::outcome::VerifiedFacts;
use super::super::super::measurement::EvidenceScale;
use super::super::super::state::fixtures::{self, PricedRecord};
use super::super::super::state::key::MeasurementKey;
use super::super::super::state::tests::container;
use super::{ArtifactRefusal, MeasurementArtifact};

/// A record that can answer AND can be prepared, over a REAL encoded
/// container — the same fixture stage 5b actuates against, so what is
/// bound here is the thing that was rendered and not a second idea of
/// what a record is. The `TempDir` is returned because it must outlive
/// the container it holds.
fn prepared_experiment() -> (tempfile::TempDir, PreparedExperiment) {
    let dir = container::glimmer();
    let snapshot = PricedRecord::new(dir.path())
        .with_protocol(fixtures::protocol())
        .build();
    let prepared = PreparedExperiment::of(&snapshot);
    (dir, prepared)
}

fn authorised_key(prepared: &PreparedExperiment) -> MeasurementKey {
    prepared
        .request()
        .expect("the rung 5 snapshot prepares a ready experiment")
        .key()
        .clone()
}

fn observed_of(key: MeasurementKey, kl_p99: f64) -> Observed {
    Observed {
        key,
        observation: fixtures::authority_reading(kl_p99, 3),
        verified: VerifiedFacts::default(),
        execution_note: "ran 8 sequences on cpu".to_string(),
    }
}

fn artifact() -> MeasurementArtifact {
    let (_dir, prepared) = prepared_experiment();
    let key = authorised_key(&prepared);
    from_execution(&prepared, &observed_of(key, 0.01))
        .expect("an observation of the authorised experiment binds")
}

#[test]
fn a_bound_artifact_verifies_against_its_own_contents() {
    let sealed = artifact();
    sealed
        .verify_seal()
        .expect("nothing has changed since it was sealed");

    // And the seal is over the contents, not a constant: the same
    // authorised experiment observed differently seals differently.
    let (_dir, prepared) = prepared_experiment();
    let key = authorised_key(&prepared);
    let other = from_execution(&prepared, &observed_of(key, 0.02)).expect("binds");
    assert_ne!(
        sealed.seal(),
        other.seal(),
        "a different observation must not seal to the same digest, or the seal \
         is not over the contents it claims to bind"
    );
}

#[test]
fn an_artifact_changed_in_round_trip_is_refused_naming_both_seals() {
    let sealed = artifact();
    let mut wire = serde_json::to_value(&sealed).expect("serialises");
    // The realistic tamper: the type forbids mutation in process, so the
    // vector is the bytes between producing an artifact and ingesting it.
    wire["execution_note"] = serde_json::Value::String("ran 8000 sequences on cpu".into());
    let tampered: MeasurementArtifact =
        serde_json::from_value(wire).expect("a tampered artifact still deserialises");

    let refusal = tampered
        .verify_seal()
        .expect_err("the contents no longer hash to the seal they carry");

    let ArtifactRefusal::SealBroken { expected, observed } = &refusal else {
        panic!("expected SealBroken, got {refusal:?}");
    };
    assert_eq!(expected, sealed.seal(), "the seal it was made with");
    assert_ne!(expected, observed);

    // Contract 5: the RENDERED form must carry both, not merely the type.
    let said = refusal.to_string();
    assert!(
        said.contains(expected.as_str()) && said.contains(observed.as_str()),
        "a rendered refusal must name the expected and the conflicting observed \
         authority, and this one said: {said}"
    );
}

#[test]
fn a_well_formed_observation_of_another_experiment_produces_no_artifact() {
    let (_dir, prepared) = prepared_experiment();
    // Nothing is wrong with this observation. It is well formed, it has a
    // real reading, it verifies what a run verifies. It is simply an
    // observation of a DIFFERENT experiment from the one authorised — which
    // is exactly the substitution a sealed artifact would later launder
    // into a record.
    let elsewhere = fixtures::key_for(&fixtures::s1(), EvidenceScale::Authority);
    let authorised = authorised_key(&prepared);
    assert_ne!(
        elsewhere, authorised,
        "the fixture must name another experiment"
    );

    let refusal = from_execution(&prepared, &observed_of(elsewhere, 0.01))
        .expect_err("an observation of another experiment may not become an artifact");

    let ArtifactRefusal::SubstitutedExperiment { expected, observed } = &refusal else {
        panic!("expected SubstitutedExperiment, got {refusal:?}");
    };
    assert_ne!(expected, observed);
    let said = refusal.to_string();
    assert!(
        said.contains(expected.as_str()) && said.contains(observed.as_str()),
        "the refusal must name the experiment authorised AND the one observed, \
         and this one said: {said}"
    );
}

#[test]
fn the_same_observation_binds_once_its_key_is_corrected() {
    // The companion to the hostile arm, and the reason it proves anything:
    // the observation above was not rejected for being malformed. Correct
    // ONLY the key and the identical reading binds.
    let (_dir, prepared) = prepared_experiment();
    let rejected = observed_of(
        fixtures::key_for(&fixtures::s1(), EvidenceScale::Authority),
        0.01,
    );
    assert!(from_execution(&prepared, &rejected).is_err());

    let corrected = Observed {
        key: authorised_key(&prepared),
        ..rejected.clone()
    };
    let bound = from_execution(&prepared, &corrected)
        .expect("the same observation, correctly keyed, binds");

    assert_eq!(bound.observation(), &rejected.observation);
    assert_eq!(bound.execution_note(), rejected.execution_note);
    bound.verify_seal().expect("and seals");
}

#[test]
fn the_artifact_carries_no_field_a_runner_could_smuggle_a_verdict_through() {
    let wire = serde_json::to_value(artifact()).expect("serialises");
    let object = wire.as_object().expect("an artifact is an object");
    let mut present: Vec<&str> = object.keys().map(String::as_str).collect();
    present.sort_unstable();

    assert_eq!(
        present,
        vec![
            "candidate_authority_digest",
            "execution_note",
            "key",
            "observation",
            "procedure",
            "seal",
            "source_semantic_digest",
            "verified",
        ],
        "arm 10 prefers structural impossibility to 'ignored': a verdict, rank, \
         promotion or admissibility field must not be expressible at all. If this \
         assertion is failing because a field was added, the question is whether a \
         runner could put a conclusion in it — not whether to widen the list."
    );
}

#[test]
fn the_procedure_and_source_identity_come_from_the_request_not_the_run() {
    // Structural, and asserted so it stays structural: the executor has no
    // way to state either, so neither can be substituted by a run.
    let (_dir, prepared) = prepared_experiment();
    let request = prepared.request().expect("ready");
    let bound = artifact();
    assert_eq!(bound.procedure(), request.procedure());
    assert_eq!(
        bound.source_semantic_digest(),
        request.model().semantic_digest()
    );
}

fn from_execution(
    prepared: &PreparedExperiment,
    observed: &Observed,
) -> Result<MeasurementArtifact, ArtifactRefusal> {
    let fixture = super::super::tests::Fixture::new();
    MeasurementArtifact::from_execution(prepared, observed, &fixture.evidence())
}

#[test]
fn a_fresh_seal_cannot_make_substituted_authority_acceptable() {
    use super::super::{ingest, IngestionRefusal};
    let f = super::super::tests::Fixture::new();
    for field in ["candidate binding", "source", "procedure"] {
        let mut a = f.artifact(0.01);
        match field {
            "candidate binding" => a.candidate_authority_digest = "another artifact".into(),
            "source" => a.source_semantic_digest = "stale source".into(),
            _ => a.procedure = "another-procedure/v1".into(),
        }
        a.seal = super::seal_of(&super::Sealed {
            key: &a.key,
            procedure: &a.procedure,
            source_semantic_digest: &a.source_semantic_digest,
            candidate_authority_digest: &a.candidate_authority_digest,
            observation: &a.observation,
            verified: &a.verified,
            execution_note: &a.execution_note,
        })
        .unwrap();
        a.verify_seal().unwrap();
        let mut snapshot = f.snapshot.clone();
        let refusal = ingest(&mut snapshot, &f.prepared, &a, &f.sources()).unwrap_err();
        assert!(
            matches!(refusal, IngestionRefusal::Authority { .. }),
            "{field}: {refusal:?}"
        );
        assert_eq!(snapshot, f.snapshot);
        assert_eq!(
            snapshot.next_experiment().unwrap(),
            f.snapshot.next_experiment().unwrap()
        );
    }
}

#[test]
fn unprepared_experiments_cannot_create_measurement_artifacts() {
    use super::super::super::actuate::request::RequestRefusal;
    let f = super::super::tests::Fixture::new();
    for prepared in [
        PreparedExperiment::Exhausted,
        PreparedExperiment::NotSelectable {
            detail: "unpriced".into(),
        },
        PreparedExperiment::NotPreparable(RequestRefusal::NoProtocol),
    ] {
        let r = MeasurementArtifact::from_execution(&prepared, &f.observed(0.01), &f.evidence())
            .unwrap_err();
        assert!(matches!(r, ArtifactRefusal::NotAuthorised { .. }));
        assert!(r.to_string().contains("no authorised experiment"));
    }
}
