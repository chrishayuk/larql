//! **The plan-v1 executor** (MEASURE-PLAN-2): performs
//! `teacher-forced-two-arm/plan-v1` for an authorised request.
//!
//! It executes the request it is handed and nothing else. The reference
//! arm reads the source container's canonical bytes, and the candidate arm
//! reads the candidate container's stored pack in the request's encoding.
//! Both are interpreter arms over this crate's shipped lowerings (lowered
//! arms are registered by `larql-cli`, where `LoweredSession` lives).
//! The run's own refusal is forwarded whole; the reading is the run's
//! summary, and its facts are the run's own verified facts.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::super::measure::plan::{arm::interpreter_arm_for, run, PlanMeasureRequest, PROCEDURE};
use super::super::reading::PlanObservation;
use super::executor::{ArtifactLocator, ExecutionRefusal, ExperimentExecutor, Observed};
use super::request::MeasurementRequest;
use crate::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use crate::format::vindex3::opplan::exec::continuation_registry::ContinuationRegistry;
use crate::format::vindex3::opplan::exec::lowering::LoweringIdentity;

/// Performs plan-v1 with interpreter arms.
pub struct PlanTeacherForcedExecutor {
    /// The component both arms execute.
    pub component: String,
    /// The reference arm's lowering.
    pub reference: LoweringIdentity,
    /// The candidate arm's lowering.
    pub candidate: LoweringIdentity,
    /// Continuation providers the arms may be built with, and which one.
    pub continuations: ContinuationRegistry,
    pub continuation: ContinuationIdentity,
    /// Each run writes its report under `output_root/<request label>`.
    pub output_root: PathBuf,
}

fn not_instructable(detail: impl Into<String>) -> ExecutionRefusal {
    ExecutionRefusal::NotInstructable {
        procedure: PROCEDURE.into(),
        detail: detail.into(),
    }
}

impl ExperimentExecutor for PlanTeacherForcedExecutor {
    fn procedure(&self) -> &str {
        PROCEDURE
    }

    fn execute(
        &self,
        request: &MeasurementRequest,
        artifacts: &dyn ArtifactLocator,
    ) -> Result<Observed, ExecutionRefusal> {
        let source = artifacts.container(request)?;
        let candidate = artifacts.candidate(request)?;
        let bank = artifacts.corpus(request)?;
        let sequences = request.sequences();
        if sequences == 0 {
            return Err(not_instructable("the declared bank names no samples"));
        }
        let encoding = request.candidate_map().encoding.clone();
        // Arms are NAMED by their execution arm, never by their role: the
        // procedure reads a name difference as a changed arm, so role
        // labels would make two identical arms look like a changed
        // variable and let a candidate that changed nothing through.
        let reference_arm = self.reference.to_string();
        let candidate_arm_name = self.candidate.to_string();
        let mut reference = interpreter_arm_for(
            &reference_arm,
            &source,
            None,
            &self.component,
            &self.reference,
            &self.continuations,
            &self.continuation,
        )
        .map_err(not_instructable)?;
        let mut candidate_arm = interpreter_arm_for(
            &candidate_arm_name,
            &candidate,
            Some(&encoding),
            &self.component,
            &self.candidate,
            &self.continuations,
            &self.continuation,
        )
        .map_err(not_instructable)?;
        let instructed = PlanMeasureRequest {
            bank,
            sequences,
            label: request.label(),
            output: self.output_root.join(request.label()),
            provenance: BTreeMap::from([(
                "measurement_key".to_string(),
                request.key().short().to_string(),
            )]),
        };
        let receipt =
            run(&instructed, &mut reference, &mut candidate_arm).map_err(ExecutionRefusal::Plan)?;
        Ok(Observed {
            key: request.key().clone(),
            observation: PlanObservation::from_summary(&receipt.summary, sequences).into(),
            verified: receipt.facts.into(),
            execution_note: format!(
                "plan-v1 interpreter arms: reference {} on the source, candidate {} on the \
                 stored {encoding} pack, {sequences} samples",
                self.reference, self.candidate
            ),
        })
    }
}
