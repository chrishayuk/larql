//! `DELETE FROM EDGES WHERE ...` — tombstone features via the session
//! overlay (V2's `PatchedVindex`, V3's `KnowledgeOverlay`).
//!
//! One logical operation, written once: candidates resolve through the
//! knowledge seam, the tombstone lands in whichever overlay the
//! binding holds, and the same `PatchOp::Delete` is recorded — so a
//! saved patch replays identically on either backend.

use crate::ast::Condition;
use crate::error::LqlError;
use crate::executor::{Backend, Session};

impl Session {
    pub(crate) fn exec_delete(
        &mut self,
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        // Validate the clause and resolve its targets with a readonly
        // borrow: a refusal here leaves the overlay and the recording
        // exactly as they were.
        let deletes = self.resolve_mutation_targets(conditions)?;

        if deletes.is_empty() {
            return Ok(vec!["  (no matching features found)".into()]);
        }

        // Only now — validated, resolved, and with something to record —
        // does the statement touch session state.
        let mut out = self.ensure_patch_session();

        match &mut self.backend {
            Backend::Vindex { patched, .. } => {
                for &(layer, feature) in &deletes {
                    patched.delete_feature(layer, feature);
                }
            }
            Backend::Vindex3 { overlay, .. } => {
                for &(layer, feature) in &deletes {
                    overlay.delete_feature(layer, feature);
                }
            }
            _ => unreachable!("browse() refused every other backend above"),
        }

        // Record to patch session
        for &(layer, feature) in &deletes {
            if let Some(ref mut recording) = self.patch_recording {
                recording.operations.push(larql_vindex::PatchOp::Delete {
                    layer,
                    feature,
                    reason: None,
                });
            }
        }

        out.push(format!(
            "Deleted {} features (patch overlay)",
            deletes.len()
        ));
        Ok(out)
    }
}
