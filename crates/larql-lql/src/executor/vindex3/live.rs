//! Stored-weight ops in a running session: keeping their live view current, and saying what it is.
//!
//! `APPLY PATCH`, `REMOVE PATCH`, `UPDATE WEIGHTS`, `REMOVE DIRECTION` and `ADD LOW RANK` change the set of
//! stored-weight ops in effect. Each ends by rebuilding the overlay's live view ([`Session::refresh_live_weights`]),
//! which materialises every patched tensor once, through COMPILE's own byte path, and holds the result until the
//! set changes again. `INFER` and `TRACE` then execute those bytes, or are refused with the view's reason; they
//! never execute the container's stored weights while an op is pending.

use larql_vindex::format::vindex3::knowledge::{ExecutionBinding, KnowledgeOverlay};
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;

use crate::executor::{Backend, Session};

/// What the status lines say executes the patched bytes.
const EXECUTED_BY: &str = "INFER and TRACE";
/// What they say still reads the container.
const READS_STORED: &str = "DESCRIBE, WALK, SELECT and DIFF";

/// The session's stored-weight ops as execution sees them, for the reply of any statement that reports them.
/// Empty when no such op is in effect.
pub(crate) fn live_weight_lines(overlay: &KnowledgeOverlay) -> Vec<String> {
    match overlay.live_weights() {
        Ok(None) => Vec::new(),
        Ok(Some(view)) => {
            let mut out = vec![format!(
                "  In effect for {EXECUTED_BY} in this session: {} tensor(s) execute from patched bytes held in \
                 memory, not from the container's stored bytes.",
                view.tensors().len()
            )];
            for tensor in view.tensors() {
                out.push(format!(
                    "    {} [{}, {}]  sha256 {}",
                    tensor.tensor, tensor.shape[0], tensor.shape[1], tensor.sha256
                ));
            }
            out.push(format!(
                "  Nothing on disk has changed: COMPILE CURRENT INTO VINDEX writes a container that stores exactly \
                 these bytes. {READS_STORED} read the stored weights."
            ));
            out
        }
        Err(refusal) => vec![
            format!(
                "  NOT executable in this session, so {EXECUTED_BY} are refused while it is pending: {refusal}"
            ),
            "  Nothing on disk has changed.".into(),
        ],
    }
}

impl Session {
    /// Rebuild the live view of the stored-weight ops in effect and describe it. Every statement that may have
    /// changed them ends with this; one that forgot would leave the view stale, and a stale view refuses.
    pub(crate) fn refresh_live_weights(&mut self) -> Vec<String> {
        let Backend::Vindex3 {
            path,
            runtime,
            overlay,
            ..
        } = &mut self.backend
        else {
            return Vec::new();
        };
        let binding = ExecutionBinding {
            container: path,
            plan: runtime.plan(),
            store: runtime.store(),
            lowering: runtime.backend().identity(),
        };
        // The outcome is read back through `live_weights`, which also sees edits made after this call.
        let _ = overlay.refresh_live_weights(&binding);
        live_weight_lines(overlay)
    }
}
