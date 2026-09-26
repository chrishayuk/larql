//! Re-establish authority from persisted inputs; no scientific writes here.
use std::path::Path;

use crate::format::vindex3::inspect::inspect_container;

use super::super::{
    actuate::prepare::PreparedExperiment,
    candidate_authority::CandidateAuthorityRefusal,
    compile::hash_bytes,
    compiler::read_source_identity,
    measure::plan::{metrics::TOP_K_OVERLAP, PlanVerifiedFacts},
    measure::{outcome::VerifiedFacts, MeasurementProcedure},
    quality::QualityBank,
    reading::{
        check_binding, gate_of_any_kind, Gate, Observation, PlanObservation, ReadingKind, RunFacts,
    },
    state::{
        key::{MeasurementConflict, MeasurementKey},
        snapshot::SearchSnapshot,
    },
    token_bank::TOKEN_BANK_SCHEMA,
};
use super::artifact::{ArtifactRefusal, MeasurementArtifact};
use super::state_evidence::{ArtifactStateEvidence, EstablishedState};

/// Paths locate evidence; no supplied digest or locator result is trusted.
pub struct IngestionSources<'a> {
    pub container: &'a Path,
    pub candidate: &'a Path,
    pub corpus: &'a Path,
}

#[derive(Debug, thiserror::Error)]
pub enum IngestionRefusal {
    #[error("{0}")]
    Artifact(ArtifactRefusal),
    #[error("candidate authority refused: {0}")]
    Candidate(CandidateAuthorityRefusal),
    #[error("{what}: expected {expected}, observed {observed}")]
    Authority {
        what: String,
        expected: String,
        observed: String,
    },
    #[error("measurement artifact is malformed or incomplete: {0}")]
    Malformed(String),
    #[error("run validity requires a complete VerifiedFacts report; missing obligations: {missing:?}; observed {observed:?}")]
    IncompleteRun {
        missing: Vec<String>,
        observed: Box<RunFacts>,
    },
    #[error("{0}")]
    Conflict(Box<MeasurementConflict>),
}

/// A verified observation, not a verdict. No constructor or Deserialize path.
///
/// ```compile_fail
/// use larql_vindex::format::vindex3::represent::ingest::AcceptedMeasurement;
/// let forged: AcceptedMeasurement = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedMeasurement {
    key: MeasurementKey,
    observation: Observation,
    candidate: EstablishedState,
    source_semantic_digest: String,
    bank_manifest_sha256: String,
    artifact_seal: String,
}

impl AcceptedMeasurement {
    pub fn key(&self) -> &MeasurementKey {
        &self.key
    }
    pub fn observation(&self) -> &Observation {
        &self.observation
    }
    pub fn candidate(&self) -> &EstablishedState {
        &self.candidate
    }
    pub fn source_semantic_digest(&self) -> &str {
        &self.source_semantic_digest
    }
    pub fn bank_manifest_sha256(&self) -> &str {
        &self.bank_manifest_sha256
    }
    pub fn artifact_seal(&self) -> &str {
        &self.artifact_seal
    }
}

pub(super) fn agrees<T, U>(what: &str, expected: T, observed: U) -> Result<(), IngestionRefusal>
where
    T: std::fmt::Debug + PartialEq<U>,
    U: std::fmt::Debug,
{
    if expected == observed {
        Ok(())
    } else {
        Err(IngestionRefusal::Authority {
            what: what.into(),
            expected: format!("{expected:?}"),
            observed: format!("{observed:?}"),
        })
    }
}

pub(super) fn invalid(e: impl std::fmt::Display) -> IngestionRefusal {
    IngestionRefusal::Malformed(e.to_string())
}

/// Validate one previously authorised request against the current snapshot.
/// Selection is not rerun: an identical replay must remain idempotent even
/// after the first acceptance changes which experiment selection prefers.
pub fn validate(
    snapshot: &SearchSnapshot,
    prepared: &PreparedExperiment,
    artifact: &MeasurementArtifact,
    sources: &IngestionSources<'_>,
) -> Result<AcceptedMeasurement, IngestionRefusal> {
    artifact.verify_seal().map_err(IngestionRefusal::Artifact)?;
    if let RunFacts::Kimi(facts) = artifact.verified() {
        require_complete_run(facts)?;
    }
    snapshot.check_schema().map_err(invalid)?;
    let request = prepared
        .request()
        .ok_or_else(|| invalid("no authorised request"))?;
    if let RunFacts::Plan(facts) = artifact.verified() {
        require_complete_plan_run(facts, request.sequences())?;
    }
    agrees("measurement key", request.key(), artifact.key())?;
    agrees(
        "request canonical key",
        request.key(),
        &request.derived_key().map_err(invalid)?,
    )?;
    let protocol = snapshot
        .protocol()
        .ok_or_else(|| invalid("snapshot lacks protocol authority"))?;
    // These recompute identities from complete declarations, never names.
    agrees(
        "bank declaration",
        snapshot.standing_intent().bank.clone(),
        protocol.bank.id(),
    )?;
    agrees(
        "instrument declaration",
        snapshot.standing_intent().instrument.clone(),
        protocol.instrument.id(),
    )?;
    agrees("request bank", request.key().bank(), &protocol.bank.id())?;
    agrees(
        "request instrument",
        request.key().instrument(),
        &protocol.instrument.id(),
    )?;
    agrees(
        "request scale",
        snapshot.standing_intent().scale,
        request.key().scale(),
    )?;
    agrees(
        "request surface",
        &snapshot.space().surface,
        request.surface(),
    )?;
    agrees(
        "request layout",
        &snapshot.semantics().layout_admission,
        request.layout_admission(),
    )?;
    agrees(
        "request procedure",
        &protocol.procedure,
        request.procedure(),
    )?;
    agrees(
        "artifact procedure",
        &protocol.procedure,
        artifact.procedure(),
    )?;
    MeasurementProcedure::by_name(&protocol.procedure).map_err(invalid)?;
    // A present gate must be this build's, and bound to the instrument
    // that took the reading, before anything reads the reading's values.
    let gate = match snapshot.gate() {
        Some(declared) => {
            let implemented = gate_of_any_kind(declared.id()).map_err(invalid)?;
            agrees("gate declaration", &implemented, declared)?;
            Some(implemented)
        }
        None => None,
    };
    agrees(
        "request gate",
        snapshot.gate().map(Gate::id),
        request.gate(),
    )?;
    let kind = artifact.observation().kind();
    agrees(
        "reading kind of the procedure",
        ReadingKind::of_procedure(&protocol.procedure),
        Some(kind),
    )?;
    if let Some(gate) = &gate {
        check_binding(gate, kind, &protocol.instrument).map_err(|e| {
            IngestionRefusal::Authority {
                what: "gate binding".into(),
                expected: gate.id().to_string(),
                observed: e.to_string(),
            }
        })?;
    }

    let expected_source = snapshot.graph().model().semantic_digest();
    agrees(
        "request source",
        &expected_source,
        &request.model().semantic_digest(),
    )?;
    agrees(
        "artifact source",
        &expected_source,
        artifact.source_semantic_digest(),
    )?;
    let source =
        read_source_identity(sources.container).map_err(|e| IngestionRefusal::Authority {
            what: "source container".into(),
            expected: expected_source.clone(),
            observed: e.to_string(),
        })?;
    let source_semantic_digest = source.semantic_digest();
    agrees(
        "source container",
        &expected_source,
        &source_semantic_digest,
    )?;
    // The semantic identity records declared seals; verify their physical
    // payloads independently before accepting the baseline as evidence.
    let inspection =
        inspect_container(sources.container, true).map_err(|e| IngestionRefusal::Authority {
            what: "source container payloads".into(),
            expected: expected_source.clone(),
            observed: e.to_string(),
        })?;
    if !inspection.is_coherent() {
        return Err(IngestionRefusal::Authority {
            what: "source container payloads".into(),
            expected: expected_source.clone(),
            observed: format!("{:?}", inspection.defects),
        });
    }

    // Establish Y successfully before comparing its identity with requested X.
    let candidate =
        ArtifactStateEvidence::establish(sources.candidate).map_err(IngestionRefusal::Candidate)?;
    agrees(
        "candidate source",
        &expected_source,
        &candidate.state().model().semantic_digest(),
    )?;
    agrees(
        "candidate representation state",
        request.key().state(),
        candidate.established(),
    )?;
    agrees(
        "candidate artifact binding",
        &artifact.candidate_authority_digest(),
        &candidate.candidate_authority_digest(),
    )?;

    let bank = &protocol.bank;
    let token_bank = bank.schema == TOKEN_BANK_SCHEMA;
    if bank.schema.is_empty()
        || bank.samples.is_empty()
        || (!token_bank && bank.positions_per_sample == 0)
    {
        return Err(invalid("bank declaration is empty"));
    }
    let bytes = std::fs::read(
        sources
            .corpus
            .join(super::super::actuate::artifacts::BANK_MANIFEST),
    )
    .map_err(|e| IngestionRefusal::Authority {
        what: "bank manifest".into(),
        expected: bank.manifest_sha256.clone(),
        observed: e.to_string(),
    })?;
    let bank_manifest_sha256 = hash_bytes(&bytes);
    agrees(
        "bank manifest",
        &bank.manifest_sha256,
        &bank_manifest_sha256,
    )?;
    // Positions come from the bank as re-read here: a fixed per-sample
    // length for the Kimi rows, the manifest's token counts for a token
    // bank.
    let positions = if token_bank {
        super::token_bank_evidence::verify(bank, sources.corpus)?
    } else {
        super::bank_evidence::verify(bank, sources.corpus, &bytes)?;
        bank.positions()
    };
    // Rebuild bank identity with the digest actually read, not the executor's claim.
    let mut established_bank = bank.clone();
    established_bank.manifest_sha256 = bank_manifest_sha256.clone();
    agrees(
        "bank identity",
        request.key().bank(),
        &established_bank.id(),
    )?;

    match (artifact.observation(), artifact.verified()) {
        (Observation::Kimi(reading), RunFacts::Kimi(facts)) => {
            validate_observation(reading, positions)?;
            agrees("reported measured positions", positions, facts.positions)?;
            // The Kimi procedure evaluates a gate as part of its run, so
            // its facts name one; a record without a gate cannot have
            // produced them.
            let gate =
                gate.ok_or_else(|| invalid("Kimi facts name a gate and this record has none"))?;
            agrees("reported gate", gate.id(), facts.gate_evaluated.as_str())?;
        }
        (Observation::Plan(reading), RunFacts::Plan(facts)) => {
            validate_plan_observation(reading, positions, request.sequences())?;
            agrees("reported measured positions", positions, facts.positions)?;
        }
        (reading, facts) => {
            return Err(invalid(format!(
                "the run's facts are {} and its reading is {} — one run cannot produce both",
                facts.kind(),
                reading.kind()
            )))
        }
    }
    Ok(AcceptedMeasurement {
        key: request.key().clone(),
        observation: artifact.observation().clone(),
        candidate,
        source_semantic_digest,
        bank_manifest_sha256,
        artifact_seal: artifact.seal().into(),
    })
}

fn require_complete_run(observed: &VerifiedFacts) -> Result<(), IngestionRefusal> {
    // Preserve the measurement layer's normative predicate. A complete
    // report is necessary, not independent proof of historical execution.
    if observed.complete() {
        return Ok(());
    }
    let missing = [
        (observed.compiled_layers.is_empty(), "compiled layers"),
        (
            observed.compiled_projections.is_empty(),
            "compiled projections",
        ),
        (
            observed.compiled_layers.is_empty()
                || observed.attribution_checked_layers.len() != observed.compiled_layers.len(),
            "attribution checks covering compiled layers",
        ),
        (observed.seal_checked_operands == 0, "seal/read witness"),
        (
            observed.invariant_neighbour_layer.is_none(),
            "invariant neighbour",
        ),
        (observed.positions == 0, "non-zero positions"),
        (observed.gate_evaluated.is_empty(), "gate"),
    ]
    .into_iter()
    .filter(|(absent, _)| *absent)
    .map(|(_, obligation)| obligation.to_owned())
    .collect();
    Err(IngestionRefusal::IncompleteRun {
        missing,
        observed: Box::new(observed.clone().into()),
    })
}

/// plan-v1's completeness predicate, with each unmet obligation named.
fn require_complete_plan_run(
    facts: &PlanVerifiedFacts,
    sequences: usize,
) -> Result<(), IngestionRefusal> {
    if facts.complete(sequences) {
        return Ok(());
    }
    let missing = [
        (facts.null_arm_samples == 0, "null arm"),
        (
            facts.changed_representations.is_empty() && !facts.arm_changed,
            "candidate scope",
        ),
        (facts.attributed_arms == 0, "physical attribution"),
        (facts.sealed_representations == 0, "seal/read witness"),
        (
            facts.bank_samples_read != sequences,
            "every declared sample read",
        ),
        (facts.tokenizer_checked_arms == 0, "tokenizer checks"),
        (facts.positions == 0, "non-zero positions"),
    ]
    .into_iter()
    .filter(|(absent, _)| *absent)
    .map(|(_, obligation)| obligation.to_owned())
    .collect::<Vec<_>>();
    Err(IngestionRefusal::IncompleteRun {
        missing: if missing.is_empty() {
            vec!["plan-v1 completeness".to_owned()]
        } else {
            missing
        },
        observed: Box::new(facts.clone().into()),
    })
}

/// A plan reading's own invariants over the positions the bank holds.
fn validate_plan_observation(
    reading: &PlanObservation,
    positions: u64,
    sequences: usize,
) -> Result<(), IngestionRefusal> {
    agrees("observation positions", positions, reading.positions)?;
    agrees("observation sequences", sequences, reading.sequences)?;
    let all = &reading.all;
    agrees("aggregate positions", positions, all.positions as u64)?;
    let finite = [
        all.kl_mean,
        all.kl_p50,
        all.kl_p99,
        all.kl_max,
        all.top1_agreement,
        all.top5_overlap_mean,
        all.max_abs_delta_mean,
        all.max_abs_delta_p99,
    ]
    .iter()
    .all(|v| v.is_finite())
        && all.delta_nll_mean.is_none_or(f64::is_finite);
    if !finite
        || all.kl_p50 > all.kl_p99
        || all.kl_p99 > all.kl_max
        || !(0.0..=1.0).contains(&all.top1_agreement)
        // A mean COUNT of shared top-k ids, not a share.
        || !(0.0..=TOP_K_OVERLAP as f64).contains(&all.top5_overlap_mean)
    {
        return Err(invalid(format!(
            "invalid plan observation statistics: {all:?}"
        )));
    }
    Ok(())
}

fn validate_observation(bank: &QualityBank, positions: u64) -> Result<(), IngestionRefusal> {
    agrees("observation positions", positions, bank.positions)?;
    let logits = &bank.logits;
    if [
        logits.kl_p50,
        logits.kl_p95,
        logits.kl_p99,
        logits.max_logit_delta,
    ]
    .iter()
    .any(|v| !v.is_finite())
        || logits.kl_p50 > logits.kl_p95
        || logits.kl_p95 > logits.kl_p99
        || logits.max_logit_delta < 0.0
        || logits.top1_flips > positions
        || logits.top10_changes > positions
        || bank.routing.positions_with_route_change > positions
        || bank
            .min_covered_mass
            .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
    {
        return Err(invalid(format!("invalid observation statistics: {bank:?}")));
    }
    for dist in [
        bank.routing.route_margin,
        bank.routing.route_weight_mass_moved,
        bank.top10_margin,
        bank.top10_candidate_margin,
        bank.top10_mass_displaced,
        bank.top10_rank_displacement,
        bank.top1_margin,
        bank.top1_candidate_margin,
        bank.top1_mass_displaced,
    ]
    .into_iter()
    .flatten()
    {
        let values = [dist.min, dist.p50, dist.p95, dist.p99, dist.max];
        if dist.count == 0
            || values.iter().any(|v| !v.is_finite())
            || values.windows(2).any(|w| w[0] > w[1])
        {
            return Err(invalid(format!(
                "invalid observation distribution: {dist:?}"
            )));
        }
    }
    Ok(())
}
