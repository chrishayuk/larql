//! What a compose install can write to a [`PatchedVindex`], taken before it
//! runs, so a failed install can be undone exactly (Gate B, B5).
//!
//! A compose install claims slots, rewrites every gate and up row at its
//! layer in the refine pass, and scales down columns while balancing. The
//! snapshot holds every part of the overlay those steps write: the feature
//! metadata, the tombstones, the gate rows, the up and down overrides (which
//! live on the base index), and the length of the session journal, which an
//! install only appends to. It does not hold the applied patches or the KNN
//! store, which no compose install writes.

use std::collections::{HashMap, HashSet};

use super::gate_overlay::GateOverlay;
use super::overlay::PatchedVindex;
use crate::index::types::FeatureMeta;

/// The state of a [`PatchedVindex`] a compose install can write. See the
/// module note.
pub struct ComposeInstallSnapshot {
    meta: HashMap<(usize, usize), Option<FeatureMeta>>,
    gate: GateOverlay,
    deleted: HashSet<(usize, usize)>,
    up: HashMap<(usize, usize), Vec<f32>>,
    down: HashMap<(usize, usize), Vec<f32>>,
    journal_len: usize,
}

impl PatchedVindex {
    /// Takes the state a compose install can write. Restore it with
    /// [`Self::restore_compose_install`] if the install fails.
    pub fn compose_install_snapshot(&self) -> ComposeInstallSnapshot {
        ComposeInstallSnapshot {
            meta: self.overrides_meta.clone(),
            gate: self.overrides_gate.clone(),
            deleted: self.deleted.clone(),
            up: self.base.metadata.up_overrides.clone(),
            down: self.base.metadata.down_overrides.clone(),
            journal_len: self.session_ops.len(),
        }
    }

    /// Puts back the state [`Self::compose_install_snapshot`] took: every
    /// slot an install wrote is as it was, and journal entries it appended
    /// are dropped.
    pub fn restore_compose_install(&mut self, snapshot: ComposeInstallSnapshot) {
        self.overrides_meta = snapshot.meta;
        self.overrides_gate = snapshot.gate;
        self.deleted = snapshot.deleted;
        self.base.metadata.up_overrides = snapshot.up;
        self.base.metadata.down_overrides = snapshot.down;
        self.session_ops.truncate(snapshot.journal_len);
    }
}
