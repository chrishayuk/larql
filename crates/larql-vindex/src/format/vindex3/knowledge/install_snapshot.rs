//! What a compose install can write to a [`KnowledgeOverlay`], taken before
//! it runs, so a failed install can be undone exactly (Gate B, B5).
//!
//! A compose install claims a slot, rewrites every gate and up row at its
//! layer in the refine pass, and scales down columns while balancing. The
//! snapshot holds every part of the overlay those steps write: the feature
//! metadata, the tombstones, the gate, up and down overrides, and the length
//! of the session journal, which an install only appends to. It does not
//! hold the applied patches, the KNN store or the stored-weight edits, which
//! no compose install writes.

use std::collections::{HashMap, HashSet};

use super::overlay::KnowledgeOverlay;
use crate::index::types::FeatureMeta;

/// The state of a [`KnowledgeOverlay`] a compose install can write. See the
/// module note.
pub struct ComposeInstallSnapshot {
    meta: HashMap<(usize, usize), Option<FeatureMeta>>,
    deleted: HashSet<(usize, usize)>,
    gate: HashMap<(usize, usize), Vec<f32>>,
    up: HashMap<(usize, usize), Vec<f32>>,
    down: HashMap<(usize, usize), Vec<f32>>,
    journal_len: usize,
}

impl KnowledgeOverlay {
    /// Takes the state a compose install can write. Restore it with
    /// [`Self::restore_compose_install`] if the install fails.
    pub fn compose_install_snapshot(&self) -> ComposeInstallSnapshot {
        ComposeInstallSnapshot {
            meta: self.overrides_meta.clone(),
            deleted: self.deleted.clone(),
            gate: self.overrides_gate.clone(),
            up: self.overrides_up.clone(),
            down: self.overrides_down.clone(),
            journal_len: self.session_ops.len(),
        }
    }

    /// Puts back the state [`Self::compose_install_snapshot`] took: every
    /// slot an install wrote is as it was, and journal entries it appended
    /// are dropped.
    pub fn restore_compose_install(&mut self, snapshot: ComposeInstallSnapshot) {
        self.overrides_meta = snapshot.meta;
        self.deleted = snapshot.deleted;
        self.overrides_gate = snapshot.gate;
        self.overrides_up = snapshot.up;
        self.overrides_down = snapshot.down;
        self.session_ops.truncate(snapshot.journal_len);
    }
}
