//! The session journal of a VINDEX3 knowledge overlay (FACT-ID-1,
//! `docs/fact-id-1.md` §3.3).
//!
//! `SAVE PATCH` writes the open recording and ends it, but the edits stay
//! live in the overlay without becoming an applied patch. The journal keeps
//! the knowledge ops the session applied directly, so the fact catalogue can
//! be folded from the container's own entries, the applied patches and this
//! journal, and agree with what executes.

use crate::patch::format::PatchOp;
use crate::patch::knn_store::KnnStore;

use super::overlay::KnowledgeOverlay;

impl KnowledgeOverlay {
    /// Record a knowledge op the session has just applied to this overlay
    /// directly. The caller applies it; this only records it.
    pub fn record_session_op(&mut self, op: PatchOp) {
        self.session_ops.push((self.patches.len(), op));
    }

    /// The knowledge ops applied directly in this session, in order, since the
    /// overlay last reset its session state. Each carries how many patches were
    /// applied when it ran, so a fold can interleave the journal with the
    /// patch list in the order they took effect.
    pub fn session_ops(&self) -> &[(usize, PatchOp)] {
        &self.session_ops
    }

    /// The KNN entries the bound container itself carries.
    pub fn base_knn(&self) -> &KnnStore {
        &self.base_knn
    }
}
