//! The admission gate: the one decision `encode` refuses on, shared by
//! every verb that reports it.
//!
//! `larql vindex3 encode --capability X` and `larql vindex3 plan
//! --capability X` must agree, or `plan` would answer a question `encode`
//! does not ask. Both call [`admit`]; neither re-derives the verdict.

use super::capability::{Capability, CapabilityStatus};
use super::SystemPlan;
use crate::error::VindexError;

/// Whether `plan` admits an encode scoped to `scope`.
///
/// `None` is whole-model completeness: every declared execution-semantic
/// fact must be understood. `Some(capability)` is that capability's
/// execution closure: nothing inside it may block, and this build must be
/// able to execute it. The graph encoded is the same either way; only the
/// gate is scoped.
///
/// On success, the capability's status when one was asked for — the
/// caller reports it without looking it up a second time.
pub fn admit(
    plan: &SystemPlan,
    scope: Option<Capability>,
) -> Result<Option<&CapabilityStatus>, VindexError> {
    let Some(capability) = scope else {
        if !plan.admissible {
            return Err(VindexError::Parse(format!(
                "refusing to encode an inadmissible plan: {} blocking finding(s); \
                 run `larql vindex3 plan` for the itemised reasons",
                plan.summary.blocking
            )));
        }
        return Ok(None);
    };
    let status = plan
        .capabilities
        .iter()
        .find(|c| c.capability == capability)
        .ok_or_else(|| {
            VindexError::Parse(format!("this build reports no verdict for {capability:?}"))
        })?;
    if !status.admissible {
        return Err(VindexError::Parse(format!(
            "refusing to encode for {:?}: {} blocking finding(s) inside that \
             capability's closure; run `larql vindex3 plan --capability` for the \
             itemised reasons",
            capability, status.blocking
        )));
    }
    // Understanding the semantics is not the same as this build being able
    // to run them, and a container written for a capability nothing can
    // execute would be a promise the runtime cannot keep.
    if !status.supported {
        return Err(VindexError::Parse(format!(
            "refusing to encode for {capability:?}: this build has no executor for it"
        )));
    }
    Ok(Some(status))
}
