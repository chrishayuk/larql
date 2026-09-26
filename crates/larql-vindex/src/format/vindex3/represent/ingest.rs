//! **Stage 6 — ingestion.**
//!
//! An executor may produce an observation. Only ingestion may decide
//! that the observation constitutes the [`MeasurementKey`] it claims,
//! and may enter `SearchFacts`.
//!
//! [`MeasurementKey`]: super::state::key::MeasurementKey
//!
//! # Why the artifact's shape is decided here
//!
//! `actuate` states, under what it deliberately does not do, that
//! whether an observation constitutes the measurement its key names is
//! answered "carefully there rather than cheaply here by a call to
//! `MeasurementRegistry::record`". This module is where that is
//! answered, so what a [`MeasurementArtifact`] must hold is decided
//! from what INGESTION needs to re-establish, not inherited from a
//! shape execution found convenient.
//!
//! [`MeasurementArtifact`]: artifact::MeasurementArtifact
//!
//! # The falsifier this module exists under
//!
//! If a correctly keyed artifact can enter `SearchFacts` without
//! independently re-establishing the authorities that make that key
//! meaningful, OPT-6 has failed. Correct keying is the claim being
//! tested, never the evidence for it — so nothing here may treat the
//! key an artifact carries as a reason to believe the artifact.
//!
//! The forecast is frozen at `docs/represent/forecasts/opt-6-ingestion.json`,
//! with its acceptance and refusal arms preregistered before any of this
//! existed.

pub mod artifact;
pub mod state_evidence;

mod bank_evidence;
mod token_bank_evidence;
mod validation;
pub use validation::{validate, AcceptedMeasurement, IngestionRefusal, IngestionSources};

/// Whether ingestion appended a new observation or reproduced one already held.
#[derive(Debug, Clone, PartialEq)]
pub struct Ingested {
    pub accepted: AcceptedMeasurement,
    pub recorded: bool,
}

/// Validate without mutation, then commit one observation atomically.
/// Every other scientific input is preserved, including on a conflict.
pub fn ingest(
    snapshot: &mut super::state::snapshot::SearchSnapshot,
    prepared: &super::actuate::prepare::PreparedExperiment,
    artifact: &artifact::MeasurementArtifact,
    sources: &IngestionSources<'_>,
) -> Result<Ingested, IngestionRefusal> {
    let accepted = validate(snapshot, prepared, artifact, sources)?;
    let recorded = !snapshot.measurements().contains(accepted.key());
    let mut facts = snapshot.facts().clone();
    super::state::MeasurementRegistry::record(&mut facts.measurements, &accepted)
        .map_err(IngestionRefusal::Conflict)?;
    if recorded {
        *snapshot = super::state::snapshot::SearchSnapshot::new(
            snapshot.space().clone(),
            snapshot.config().clone(),
            facts,
        );
    }
    Ok(Ingested { accepted, recorded })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod loop_tests;

#[cfg(test)]
mod pareto_loop_tests;

/// The persisted-artifact entry point. Decode failure cannot mutate a snapshot.
pub fn ingest_bytes(
    snapshot: &mut super::state::snapshot::SearchSnapshot,
    prepared: &super::actuate::prepare::PreparedExperiment,
    artifact: &[u8],
    sources: &IngestionSources<'_>,
) -> Result<Ingested, IngestionRefusal> {
    let artifact =
        serde_json::from_slice(artifact).map_err(|e| IngestionRefusal::Malformed(e.to_string()))?;
    ingest(snapshot, prepared, &artifact, sources)
}
