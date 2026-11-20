//! Re-fits of earlier installs (Gate B, B3, `docs/gate-b.md`).
//!
//! A compose INSERT does not only write its own slots. The batch refine
//! rebuilds the gate and up vectors of every install at the layer from the
//! raw captures, and the balance passes rescale down vectors. The new
//! install's own op carries its final vectors. The ops of earlier installs
//! carry the vectors they had when they were recorded, and those ops may
//! already be in a saved patch. Without more, a fresh session that applies
//! the recording reaches a different state from the live one.
//!
//! So every earlier install slot whose vectors the INSERT changed gets an
//! `Update` op with the new vectors and no metadata: a re-fit. Applied in
//! order, the recording then reproduces the live vectors exactly, whether the
//! installs share one patch or not. The FACT-ID-1 catalogue reads such an
//! op as a re-fit of the fact that owns the slot, not a physical edit
//! (`facts::fold`).

use std::collections::{BTreeMap, BTreeSet};

use larql_vindex::patch::core::encode_gate_vector;
use larql_vindex::PatchOp;

use crate::executor::{Backend, Session};

/// The gate, up and down overrides of one slot.
type SlotVectors = [Option<Vec<f32>>; 3];

/// The vectors of every earlier install slot, before an INSERT.
pub(crate) type PriorVectors = BTreeMap<(usize, usize), SlotVectors>;

impl Session {
    /// The slots of the session's earlier compose installs: every slot the
    /// refine and balance passes can rewrite.
    fn prior_install_slots(&self) -> BTreeSet<(usize, usize)> {
        self.installed_edges
            .iter()
            .map(|e| (e.layer, e.feature))
            .chain(self.raw_install_residuals.keys().copied())
            .collect()
    }

    /// The override vectors of one slot on the bound overlay.
    fn slot_vectors(&self, (layer, feature): (usize, usize)) -> SlotVectors {
        let own = |v: Option<&[f32]>| v.map(<[f32]>::to_vec);
        match &self.backend {
            Backend::Vindex { patched, .. } => [
                own(patched.overrides_gate_at(layer, feature)),
                own(patched.up_override_at(layer, feature)),
                own(patched.down_override_at(layer, feature)),
            ],
            Backend::Vindex3 { overlay, .. } => [
                own(overlay.gate_override_at(layer, feature)),
                own(overlay.up_override_at(layer, feature)),
                own(overlay.down_override_at(layer, feature)),
            ],
            _ => [None, None, None],
        }
    }

    /// Snapshot the vectors of every earlier install slot.
    pub(crate) fn prior_install_vectors(&self) -> PriorVectors {
        self.prior_install_slots()
            .into_iter()
            .map(|slot| (slot, self.slot_vectors(slot)))
            .collect()
    }

    /// The re-fit of every earlier install slot whose vectors changed since
    /// `prior`, in slot order.
    pub(crate) fn refits_since(&self, prior: &PriorVectors) -> Vec<PatchOp> {
        prior
            .iter()
            .filter_map(|(&slot, before)| {
                let after = self.slot_vectors(slot);
                (after != *before).then(|| refit_op(slot, &after))
            })
            .collect()
    }

    /// Record the re-fits since `prior` into the journal and the open
    /// recording.
    pub(crate) fn record_refits(&mut self, prior: &PriorVectors) {
        for op in self.refits_since(prior) {
            self.record_knowledge_op(op);
        }
    }
}

/// The re-fit of one slot: its current vectors, and no metadata.
fn refit_op((layer, feature): (usize, usize), [gate, up, down]: &SlotVectors) -> PatchOp {
    let b64 = |v: &Option<Vec<f32>>| v.as_deref().map(encode_gate_vector);
    PatchOp::Update {
        layer,
        feature,
        gate_vector_b64: b64(gate),
        up_vector_b64: b64(up),
        down_vector_b64: b64(down),
        down_meta: None,
    }
}
