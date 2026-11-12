//! The map-resident half of a session: the optional patch overlay.

use std::sync::Arc;
use std::time::Instant;

use larql_vindex::{PatchedVindex, VectorIndex, VindexPatch};

use super::lease::SessionLease;

/// One session's server-side state.
///
/// The [`PatchedVindex`] overlay is **materialised lazily**. Cloning a
/// model's base index to build one is not free, and most sessions never
/// patch anything — a session bound only so its KV continuations can be
/// owned and evicted needs no overlay at all, and behaves exactly like the
/// global (unpatched) state for inference. The first patch/insert on the
/// session pays for the clone; everything else stays a few words plus a
/// [`SessionLease`].
pub struct SessionState {
    overlay: Option<PatchedVindex>,
    lease: Arc<SessionLease>,
}

impl SessionState {
    /// A session with identity but no overlay yet.
    pub(super) fn bound(lease: Arc<SessionLease>) -> Self {
        Self {
            overlay: None,
            lease,
        }
    }

    /// The shared identity record.
    pub fn lease(&self) -> &Arc<SessionLease> {
        &self.lease
    }

    /// Refresh the idle clock. Call from any code path that mutates the
    /// session through a raw map guard rather than a manager method.
    pub fn touch(&self, now: Instant) {
        self.lease.touch_at(now);
    }

    /// The patch overlay, if this session has ever been patched. `None`
    /// means "identical to the model's global state" — read paths should
    /// fall back to the model's own overlay rather than materialising one.
    pub fn patched(&self) -> Option<&PatchedVindex> {
        self.overlay.as_ref()
    }

    /// The session's applied patches, in order; empty without an overlay.
    pub fn patches(&self) -> &[VindexPatch] {
        match &self.overlay {
            Some(p) => &p.patches,
            None => &[],
        }
    }

    /// How many patches are active on this session.
    pub fn num_patches(&self) -> usize {
        self.overlay.as_ref().map_or(0, |p| p.num_patches())
    }

    /// The overlay for writing, or `None` if this session has never been
    /// patched. Lets callers that must not pay for materialisation take
    /// the fast path explicitly.
    pub fn overlay_existing_mut(&mut self) -> Option<&mut PatchedVindex> {
        self.overlay.as_mut()
    }

    /// Give a session that has never been patched its overlay. Used when the
    /// first patch was applied before the session was bound, so a refused
    /// patch leaves nothing behind.
    pub fn install_overlay(&mut self, overlay: PatchedVindex) {
        self.overlay = Some(overlay);
    }

    /// The overlay for writing, materialising it from `base` on first use.
    ///
    /// `base` is a closure so the (possibly expensive) index clone happens
    /// only for the first patch on a session.
    pub fn overlay_mut(&mut self, base: impl FnOnce() -> VectorIndex) -> &mut PatchedVindex {
        self.overlay
            .get_or_insert_with(|| PatchedVindex::new(base()))
    }
}
