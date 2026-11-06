//! Forcing a COMPOSE install to fail at a chosen point, and fingerprinting what it left behind.
//!
//! An install writes session state in several steps and can fail between them. Whether a failed
//! install leaves the session as it found it is a property that has to be measured, not assumed.
//! This module gives the measurement two parts:
//!
//! - [`InstallFaultPoint`]: arm one with [`Session::set_install_fault`] and the next install
//!   returns an error when it reaches that point. A fault fires once and disarms itself. With none
//!   armed, an install runs exactly the operations it always did.
//! - [`StateFingerprint`]: [`Session::state_fingerprint`] hashes every part of the session an
//!   install touches, by named component, so a test can say *which* part a failed install changed.
//!
//! This is instrumentation only. It adds no rollback: whatever a failed install leaves behind, it
//! leaves, and the fingerprint reports it.

use std::collections::BTreeMap;
use std::fmt;

use sha2::{Digest, Sha256};

use super::Session;
use crate::error::LqlError;

/// A step of a COMPOSE install at which a fault can be injected, in the order the install reaches
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallFaultPoint {
    /// Before the install has written anything.
    BeforeInstall,
    /// After the decoy cache and the fact's raw residual are committed, before the overlay slot.
    AfterCachesCommitted,
    /// After the slot's gate, up and down vectors are in the overlay, before the refine pass.
    AfterSlotStaged,
    /// After the refine pass has rebuilt every composed gate and up at the layer.
    AfterRefine,
    /// At the start of balance iteration `iteration`, before the controller reads anything.
    DuringBalance { iteration: usize },
    /// After the balance pass, before the cross-fact check.
    AfterBalance,
    /// After the fact is registered for later cross-fact checks, before the patch operation is
    /// recorded.
    BeforeRecord,
}

impl fmt::Display for InstallFaultPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BeforeInstall => write!(f, "before-install"),
            Self::AfterCachesCommitted => write!(f, "after-caches-committed"),
            Self::AfterSlotStaged => write!(f, "after-slot-staged"),
            Self::AfterRefine => write!(f, "after-refine"),
            Self::DuringBalance { iteration } => write!(f, "during-balance-{iteration}"),
            Self::AfterBalance => write!(f, "after-balance"),
            Self::BeforeRecord => write!(f, "before-record"),
        }
    }
}

/// The armed fault, if any.
#[derive(Debug, Default)]
pub(crate) struct FaultSlot {
    armed: Option<InstallFaultPoint>,
}

impl FaultSlot {
    pub(crate) fn arm(&mut self, point: Option<InstallFaultPoint>) {
        self.armed = point;
    }

    /// Fails once, and disarms, when the armed point is `point`.
    pub(crate) fn fire(&mut self, point: InstallFaultPoint) -> Result<(), LqlError> {
        if self.armed == Some(point) {
            self.armed = None;
            return Err(LqlError::Execution(format!("injected fault at {point}")));
        }
        Ok(())
    }
}

/// A hash of each part of the session state an install touches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateFingerprint {
    components: BTreeMap<&'static str, String>,
}

impl StateFingerprint {
    pub fn components(&self) -> &BTreeMap<&'static str, String> {
        &self.components
    }

    /// The names of the components whose hash differs from `earlier`'s.
    pub fn changed_since(&self, earlier: &Self) -> Vec<&'static str> {
        self.components
            .iter()
            .filter(|(name, hash)| earlier.components.get(*name) != Some(*hash))
            .map(|(name, _)| *name)
            .collect()
    }
}

/// Accumulates typed values into one SHA-256, length-prefixing everything so distinct states cannot
/// collide by concatenation.
struct Hasher(Sha256);

impl Hasher {
    fn new() -> Self {
        Self(Sha256::new())
    }
    fn usize(&mut self, v: usize) {
        self.0.update((v as u64).to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.usize(b.len());
        self.0.update(b);
    }
    fn floats(&mut self, v: &[f32]) {
        self.usize(v.len());
        for x in v {
            self.0.update(x.to_bits().to_le_bytes());
        }
    }
    fn finish(self) -> String {
        self.0
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

impl Session {
    /// Arms (or, with `None`, clears) a fault for the next COMPOSE install. It fires once. See
    /// [`crate::executor::install_fault`].
    pub fn set_install_fault(&mut self, point: Option<InstallFaultPoint>) {
        self.install_fault.arm(point);
    }

    pub(crate) fn inject_install_fault(
        &mut self,
        point: InstallFaultPoint,
    ) -> Result<(), LqlError> {
        self.install_fault.fire(point)
    }

    /// A hash of every part of the session an install touches: the overlay's vectors and counts,
    /// its weight edits and patches, the install bookkeeping and caches, the active patch
    /// recording, and the mutation counters. Equal fingerprints mean nothing the install writes
    /// differs; predictions are compared separately by the caller.
    pub fn state_fingerprint(&self) -> StateFingerprint {
        let mut c: BTreeMap<&'static str, String> = BTreeMap::new();

        match &self.backend {
            super::Backend::Vindex3 {
                runtime, overlay, ..
            } => {
                let layers = runtime.plan().layers.len();
                let mut vectors = Hasher::new();
                let mut counts = Hasher::new();
                for layer in 0..layers {
                    let mut gates = overlay.gate_overrides_at(layer);
                    gates.sort_by_key(|(feature, _)| *feature);
                    for (feature, gate) in gates {
                        vectors.usize(layer);
                        vectors.usize(feature);
                        vectors.floats(gate);
                        vectors.floats(overlay.up_override_at(layer, feature).unwrap_or(&[]));
                        vectors.floats(overlay.down_override_at(layer, feature).unwrap_or(&[]));
                    }
                    counts.usize(overlay.tombstones_at(layer));
                }
                counts.usize(overlay.num_overrides());
                counts.usize(overlay.weight_edit_count());
                counts.usize(overlay.patches.len());
                c.insert("overlay.vectors", vectors.finish());
                c.insert("overlay.counts", counts.finish());

                let mut weights = Hasher::new();
                for (name, op) in overlay.weight_ops() {
                    weights.bytes(name.as_bytes());
                    weights.bytes(serde_json::to_string(op).unwrap_or_default().as_bytes());
                }
                c.insert("overlay.weight_ops", weights.finish());

                let mut patches = Hasher::new();
                for patch in &overlay.patches {
                    patches.bytes(serde_json::to_string(patch).unwrap_or_default().as_bytes());
                }
                c.insert("overlay.patches", patches.finish());
            }
            _ => {
                for name in [
                    "overlay.vectors",
                    "overlay.counts",
                    "overlay.weight_ops",
                    "overlay.patches",
                ] {
                    c.insert(name, "no-vindex3-overlay".to_string());
                }
            }
        }

        let mut edges = Hasher::new();
        for e in &self.installed_edges {
            edges.usize(e.layer);
            edges.usize(e.feature);
            edges.bytes(e.canonical_prompt.as_bytes());
            edges.bytes(e.target.as_bytes());
            edges.usize(e.target_id as usize);
        }
        c.insert("session.installed_edges", edges.finish());

        let mut raw = Hasher::new();
        let mut keys: Vec<_> = self.raw_install_residuals.keys().copied().collect();
        keys.sort_unstable();
        for key in keys {
            raw.usize(key.0);
            raw.usize(key.1);
            raw.floats(self.raw_install_residuals[&key].as_slice().unwrap_or(&[]));
        }
        c.insert("session.raw_install_residuals", raw.finish());

        let mut decoys = Hasher::new();
        let mut layers: Vec<_> = self.decoy_residual_cache.keys().copied().collect();
        layers.sort_unstable();
        for layer in layers {
            decoys.usize(layer);
            for residual in &self.decoy_residual_cache[&layer] {
                decoys.floats(residual.as_slice().unwrap_or(&[]));
            }
        }
        c.insert("session.decoy_residual_cache", decoys.finish());

        let mut recording = Hasher::new();
        recording.usize(usize::from(self.patch_recording.is_some()));
        recording.usize(usize::from(self.auto_patch));
        if let Some(r) = &self.patch_recording {
            recording.bytes(r.path.as_bytes());
            for op in &r.operations {
                recording.bytes(serde_json::to_string(op).unwrap_or_default().as_bytes());
            }
        }
        c.insert("session.patch_recording", recording.finish());

        let mut counters = Hasher::new();
        counters.usize(self.epoch as usize);
        counters.usize(self.mutations_since_minor);
        counters.usize(self.mutations_since_major);
        c.insert("session.counters", counters.finish());

        StateFingerprint { components: c }
    }
}
