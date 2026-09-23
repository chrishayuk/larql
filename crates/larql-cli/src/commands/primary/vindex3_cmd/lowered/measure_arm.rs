//! MEASURE-PLAN-1's lowered arm: a [`LoweredSession`] as a
//! [`TeacherForcedArm`].
//!
//! Each sample starts from position 0 through [`LoweredSession::reset`], so
//! weights stay resident across the bank. The procedure's null arm scores
//! the reference twice and requires bit-identical logits, so a reset that
//! leaked state is refused there, on every run, not assumed away.
//!
//! Scoring is prompt-mode: every position is the caller's token, stepped
//! with the host embedding, exactly as `vindex3 exec --logit-dump` does on a
//! lowered backend.

use std::collections::BTreeMap;
use std::path::PathBuf;

use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::represent::measure::plan::arm::{
    ArmDescription, BoundObject, TeacherForcedArm,
};

use super::LoweredSession;

pub(crate) struct LoweredArm<'a> {
    session: LoweredSession<'a>,
    description: ArmDescription,
}

impl<'a> LoweredArm<'a> {
    /// Wrap a session. `store` is the one the session loaded from, read for
    /// what it bound and what it quantised at load.
    pub(crate) fn new(
        session: LoweredSession<'a>,
        store: &OperandStore,
        arm: &str,
        container: PathBuf,
        requested_pack: Option<String>,
        stored_only: bool,
    ) -> Self {
        let objects: BTreeMap<String, BoundObject> = store
            .selection()
            .iter()
            .map(|(object, selected)| {
                (
                    object.clone(),
                    BoundObject {
                        encoding: selected.encoding.clone(),
                        stored: selected.stored,
                    },
                )
            })
            .collect();
        Self {
            session,
            description: ArmDescription {
                arm: arm.to_string(),
                container,
                requested_pack,
                stored_only,
                objects,
                runtime_quantised: store.runtime_quantised(),
            },
        }
    }
}

impl TeacherForcedArm for LoweredArm<'_> {
    fn describe(&self) -> &ArmDescription {
        &self.description
    }

    fn score(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        self.session.reset();
        ids.iter()
            .enumerate()
            .map(|(i, &id)| {
                self.session
                    .step(id)
                    .map_err(|e| format!("position {i}: {e}"))?;
                self.session
                    .last_logits()
                    .ok_or_else(|| format!("position {i}: the plan carries no output head"))
            })
            .collect()
    }
}
