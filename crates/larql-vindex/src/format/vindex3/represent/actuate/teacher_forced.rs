//! **The first registered procedure — an adapter, not a second runner.**
//!
//! 5a-0 made the environment form an ADAPTER over one procedure so that
//! "the env invocation and the direct invocation agree" was true by
//! construction. This is the second adapter into the same procedure, and
//! it must not become a second way to run:
//!
//! ```text
//! environment variables  ─┐
//! a prepared experiment  ─┼─→ TeacherForcedRequest ─→ one procedure
//!                         ┘
//! ```
//!
//! Everything below builds a `TeacherForcedRequest` and hands it to
//! `measure::run`. No measurement logic lives here, and none may: the
//! teacher-forced runner is five claims at once — a measurement plus
//! four validity proofs — and a second implementation of the numbers
//! would look identical while proving none of them.
//!
//! # The mapping, and where each field's authority is
//!
//! ```text
//! source        locator, then the container's own identity checked
//! candidate     locator, by the physical state the key names
//! quality_bank  locator, then manifest.json digested against the bank id
//! sequences     the bank's declared samples          — not LARQL_Q2A_SEQUENCES
//! gate          the record's own gate id             — not a literal
//! label         derived from the experiment          — not a control
//! procedure     the record's declared procedure
//! ```
//!
//! Three of those used to come from the environment and one was a
//! literal in the runner. `measure/mod.rs` names the literal gate as the
//! defect that made the request type necessary; this is where it stops
//! being possible, because nothing in this file can reach
//! `DEFAULT_GATE`.
//!
//! # Instructed on every build, performed on some
//!
//! [`TeacherForcedExecutor::instruct`] is platform-independent and runs
//! everywhere, including in CI on Linux where the instrument cannot
//! exist. That is deliberate: the mapping above is the whole content of
//! this module, and a mapping only checkable on one machine is a mapping
//! nobody checks. The build that cannot perform the procedure still
//! instructs first and then refuses, so a malformed request is reported
//! as malformed rather than as a missing backend.

use super::super::measure::{
    self, MeasurementProcedure, TeacherForcedRequest, TEACHER_FORCED_TWO_ARM,
};
use super::executor::{ArtifactLocator, ExecutionRefusal, ExperimentExecutor, Observed};
use super::request::MeasurementRequest;

/// The teacher-forced two-arm procedure, as an executor.
#[derive(Debug, Clone, Copy, Default)]
pub struct TeacherForcedExecutor;

impl TeacherForcedExecutor {
    /// **The adapter.** An authorised experiment and a locator become an
    /// instruction, or a refusal.
    ///
    /// Separated from [`ExperimentExecutor::execute`] so the mapping can
    /// be asserted without a Metal device, and so a caller can see what
    /// a run WOULD be instructed with before spending twenty minutes of
    /// instrument time on it.
    pub fn instruct(
        &self,
        request: &MeasurementRequest,
        artifacts: &dyn ArtifactLocator,
    ) -> Result<TeacherForcedRequest, ExecutionRefusal> {
        if request.procedure() != TEACHER_FORCED_TWO_ARM {
            return Err(ExecutionRefusal::NoSuchProcedure {
                named: request.procedure().to_string(),
                implemented: vec![TEACHER_FORCED_TWO_ARM.to_string()],
            });
        }
        let not_instructable = |detail: String| ExecutionRefusal::NotInstructable {
            procedure: TEACHER_FORCED_TWO_ARM.to_string(),
            detail,
        };
        // A bank declaring no samples has no positions, and a gate
        // judging on tail statistics would be reading an empty
        // distribution. `admit` says the same thing about the built
        // request; saying it here names the DECLARATION that is wrong
        // rather than the instruction derived from it.
        if request.sequences() == 0 {
            return Err(not_instructable(
                "the declared evidence bank names no samples, so the run would have no \
                 positions to measure"
                    .into(),
            ));
        }

        let instructed = TeacherForcedRequest {
            procedure: MeasurementProcedure::TeacherForcedTwoArm,
            source: artifacts.container(request)?,
            candidate: artifacts.candidate(request)?,
            quality_bank: artifacts.corpus(request)?,
            sequences: request.sequences(),
            gate: request.gate().to_string(),
            label: request.label(),
        };
        instructed
            .admit()
            .map_err(|e| not_instructable(e.to_string()))?;
        Ok(instructed)
    }
}

impl ExperimentExecutor for TeacherForcedExecutor {
    fn procedure(&self) -> &str {
        TEACHER_FORCED_TWO_ARM
    }

    #[cfg(all(feature = "gpu", target_os = "macos"))]
    fn execute(
        &self,
        request: &MeasurementRequest,
        artifacts: &dyn ArtifactLocator,
    ) -> Result<Observed, ExecutionRefusal> {
        let instructed = self.instruct(request, artifacts)?;
        let receipt = measure::run(&instructed).map_err(ExecutionRefusal::Measurement)?;
        let execution_note = format!(
            "{TEACHER_FORCED_TWO_ARM}: {} positions over {} sequences in {:.1}s; closure \
             report at {}",
            receipt.verified.positions,
            instructed.sequences,
            receipt.wall_seconds,
            receipt.report_path
        );
        Ok(Observed {
            // The experiment as REQUESTED. The executor performed the
            // instruction derived from it and has no authority to say it
            // measured anything else; the registry checks this against
            // what it handed over, so restating it is a claim rather
            // than a formality.
            key: request.key().clone(),
            observation: receipt.bank,
            verified: receipt.verified,
            execution_note,
        })
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos")))]
    fn execute(
        &self,
        request: &MeasurementRequest,
        artifacts: &dyn ArtifactLocator,
    ) -> Result<Observed, ExecutionRefusal> {
        // Instructed first, deliberately. A build that cannot perform
        // the procedure must still tell a malformed request from an
        // absent backend, and reporting "no Metal" for a request that
        // named no corpus would send an operator to the wrong problem.
        let instructed = self.instruct(request, artifacts)?;
        let refusal = measure::run(&instructed)
            .expect_err("this build does not implement the teacher-forced procedure");
        Err(ExecutionRefusal::Measurement(refusal))
    }
}
