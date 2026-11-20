//! INSERT atomicity (Gate B, B5): an INSERT that fails leaves the session as
//! it found it.
//!
//! A compose install writes the session in steps (the decoy and residual
//! caches, the slot, the refine pass over the layer, the balance pass, the
//! install registry, the recording) and can fail between any two of them.
//! Rather than reorder every step behind every fallible one, `exec_insert`
//! takes an [`InsertSnapshot`] of everything an install can write before it
//! starts, and puts it back if the install returns an error.
//!
//! A KNN insert writes the store once, in `knn_finalize`, after every step
//! that can fail, so it needs no overlay snapshot; the recording and journal
//! lengths still cover it.

use std::collections::HashMap;

use larql_vindex::ndarray::Array1;

use crate::executor::{Backend, Session};

enum OverlaySnapshot {
    Vindex(larql_vindex::patch::install_snapshot::ComposeInstallSnapshot),
    Vindex3(larql_vindex::format::vindex3::knowledge::install_snapshot::ComposeInstallSnapshot),
}

/// What an INSERT can write, taken before it runs. See the module note.
pub(crate) struct InsertSnapshot {
    overlay: Option<OverlaySnapshot>,
    decoy_residual_cache: HashMap<usize, Vec<Array1<f32>>>,
    raw_install_residuals: HashMap<(usize, usize), Array1<f32>>,
    installed_edges_len: usize,
    recording_len: Option<usize>,
}

impl Session {
    /// Takes what an INSERT can write. `compose` adds the overlay's slot
    /// state, which only a compose install writes.
    pub(crate) fn insert_snapshot(&self, compose: bool) -> InsertSnapshot {
        let overlay = if compose {
            match &self.backend {
                Backend::Vindex { patched, .. } => {
                    Some(OverlaySnapshot::Vindex(patched.compose_install_snapshot()))
                }
                Backend::Vindex3 { overlay, .. } => {
                    Some(OverlaySnapshot::Vindex3(overlay.compose_install_snapshot()))
                }
                _ => None,
            }
        } else {
            None
        };
        InsertSnapshot {
            overlay,
            decoy_residual_cache: self.decoy_residual_cache.clone(),
            raw_install_residuals: self.raw_install_residuals.clone(),
            installed_edges_len: self.installed_edges.len(),
            recording_len: self.patch_recording.as_ref().map(|r| r.operations.len()),
        }
    }

    /// Puts back what [`Self::insert_snapshot`] took.
    pub(crate) fn restore_insert(&mut self, snapshot: InsertSnapshot) {
        match (snapshot.overlay, &mut self.backend) {
            (Some(OverlaySnapshot::Vindex(s)), Backend::Vindex { patched, .. }) => {
                patched.restore_compose_install(s)
            }
            (Some(OverlaySnapshot::Vindex3(s)), Backend::Vindex3 { overlay, .. }) => {
                overlay.restore_compose_install(s)
            }
            _ => {}
        }
        self.decoy_residual_cache = snapshot.decoy_residual_cache;
        self.raw_install_residuals = snapshot.raw_install_residuals;
        self.installed_edges.truncate(snapshot.installed_edges_len);
        if let (Some(len), Some(recording)) = (snapshot.recording_len, &mut self.patch_recording) {
            recording.operations.truncate(len);
        }
    }
}
