//! The session map: creation, TTL eviction, patch operations, and the
//! read-only projection `/v1/sessions` renders.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use larql_vindex::{PatchedVindex, VindexPatch};
use tokio::sync::RwLock;

use crate::state::LoadedModel;

use super::clock::SessionClock;
use super::lease::SessionLease;
use super::state::SessionState;
use super::{DEFAULT_SESSION_TTL_SECS, PATCH_UNNAMED};

/// A session as the observability surface sees it: identity, lifecycle,
/// and patch identities — never overlay contents, never KV internals.
///
/// Continuation *availability* is not in here: that fact belongs to the
/// KV cache, and joining the two is the route's job, so this projection
/// stays a pure function of the session map.
pub struct SessionSummary {
    pub id: String,
    pub model: String,
    pub created_at: u64,
    pub last_used_at: u64,
    pub expires_at: u64,
    /// Names of the patches applied to this session, in application order.
    pub patch_names: Vec<String>,
    pub resumptions: u64,
    pub reused_tokens_total: u64,
}

/// Manages per-session state.
pub struct SessionManager {
    sessions: RwLock<HashMap<String, SessionState>>,
    clock: SessionClock,
    ttl: Duration,
}

/// Drop every session idle longer than `ttl` as of `now`, killing the
/// lease of each one removed. Shared by the opportunistic (in-request)
/// and periodic (maintenance sweeper) eviction paths so the policy can
/// never diverge; returns how many sessions were removed.
fn evict_expired_from(
    sessions: &mut HashMap<String, SessionState>,
    ttl: Duration,
    now: Instant,
) -> usize {
    let before = sessions.len();
    sessions.retain(|_, s| {
        let live = s.lease().idle_at(now) <= ttl;
        if !live {
            // Killing the lease is what stops an in-flight generation
            // from re-inserting a continuation for a session that has
            // just expired. See `SessionLease`.
            s.lease().kill();
        }
        live
    });
    before - sessions.len()
}

/// Get-or-create inside an already-held write guard, refreshing the idle
/// clock either way. Split out so the async and blocking entry points
/// share one creation policy — including the opportunistic eviction
/// every manager write path performs while it holds the guard anyway.
fn bind_in<'a>(
    sessions: &'a mut HashMap<String, SessionState>,
    clock: SessionClock,
    ttl: Duration,
    id: &str,
    model_id: &str,
    at: Instant,
) -> &'a mut SessionState {
    evict_expired_from(sessions, ttl, at);
    let session = sessions
        .entry(id.to_string())
        .or_insert_with(|| SessionState::bound(SessionLease::new(id, model_id, clock, at)));
    session.touch(at);
    session
}

impl SessionManager {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            clock: SessionClock::new(),
            ttl: Duration::from_secs(if ttl_secs == 0 {
                DEFAULT_SESSION_TTL_SECS
            } else {
                ttl_secs
            }),
        }
    }

    /// The configured idle TTL.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// The timebase every session record on this manager shares.
    pub fn clock(&self) -> SessionClock {
        self.clock
    }

    /// Drop every expired session. Returns how many were removed.
    ///
    /// Called periodically by the [`crate::maintenance`] sweeper; also safe
    /// to call from any async context.
    pub async fn evict_expired(&self) -> usize {
        self.evict_expired_at(Instant::now()).await
    }

    /// [`Self::evict_expired`] with an explicit clock, so tests can advance
    /// time without sleeping.
    pub async fn evict_expired_at(&self, now: Instant) -> usize {
        let mut sessions = self.sessions.write().await;
        evict_expired_from(&mut sessions, self.ttl, now)
    }

    /// Get-or-create the session `id` and return its lease.
    ///
    /// This is the cheap binding used by request paths that only need the
    /// session's *identity* — to own a KV continuation, and to appear on
    /// `/v1/sessions` — with no patch overlay materialised.
    pub async fn bind(&self, id: &str, model_id: &str) -> Arc<SessionLease> {
        self.bind_at(id, model_id, Instant::now()).await
    }

    /// [`Self::bind`] with an explicit clock, so tests can advance time
    /// without sleeping.
    pub async fn bind_at(&self, id: &str, model_id: &str, at: Instant) -> Arc<SessionLease> {
        let mut sessions = self.sessions.write().await;
        Arc::clone(bind_in(&mut sessions, self.clock, self.ttl, id, model_id, at).lease())
    }

    /// [`Self::bind`] inside a caller-held blocking write guard, for the
    /// synchronous inference paths.
    pub fn bind_in_guard<'a>(
        &self,
        sessions: &'a mut HashMap<String, SessionState>,
        id: &str,
        model_id: &str,
        at: Instant,
    ) -> &'a mut SessionState {
        bind_in(sessions, self.clock, self.ttl, id, model_id, at)
    }

    /// Get or create a session's PatchedVindex.
    ///
    /// The returned overlay is a *copy* — the session's patches replayed
    /// onto a fresh clone of the model's base — so the caller can read it
    /// without holding any session lock.
    ///
    /// # Errors
    /// When a patch the session holds no longer fits the model's base. Each
    /// was checked when it was applied, so this means the base changed under
    /// the session; the overlay is not handed back with that patch missing.
    pub async fn get_or_create(
        &self,
        session_id: &str,
        model: &Arc<LoadedModel>,
    ) -> Result<PatchedVindex, larql_vindex::VindexError> {
        // Read the base outside the sessions lock: cloning it is not free
        // and `model.patched` must never be awaited under the map guard.
        let base = model.patched.read().await.base().clone();

        let mut sessions = self.sessions.write().await;
        let now = Instant::now();
        let session = bind_in(
            &mut sessions,
            self.clock,
            self.ttl,
            session_id,
            &model.id,
            now,
        );
        let mut cloned = PatchedVindex::new(base);
        for patch in session.patches() {
            cloned.try_apply_patch(patch.clone())?;
        }
        Ok(cloned)
    }

    /// Apply a patch to a session (not global).
    ///
    /// Lock discipline: never holds the `sessions` write guard while
    /// awaiting (or blocking on) `model.patched`. An earlier
    /// implementation called `model.patched.blocking_read()` from inside
    /// `or_insert_with` while holding `sessions.write().await`, which on
    /// a multi-thread tokio runtime stalls the worker (and risks deadlock
    /// against any task acquiring those locks in the opposite order). The
    /// base is therefore read first, unconditionally — patching is rare,
    /// and one clone on the patch path is cheaper than the lock-order
    /// hazard.
    pub async fn apply_patch(
        &self,
        session_id: &str,
        model: &Arc<LoadedModel>,
        patch: VindexPatch,
    ) -> Result<(usize, usize), larql_vindex::VindexError> {
        // Fast path: the session already has an overlay, so no base clone
        // is needed and the whole operation fits under one lock.
        {
            let mut sessions = self.sessions.write().await;
            let now = Instant::now();
            if let Some(session) = sessions.get_mut(session_id) {
                session.touch(now);
                if let Some(overlay) = session.overlay_existing_mut() {
                    let op_count = patch.operations.len();
                    overlay.try_apply_patch(patch)?;
                    return Ok((op_count, overlay.num_patches()));
                }
            }
        }

        // Slow path: the session's first patch. Materialising the overlay
        // needs the model's base, read *outside* the sessions lock.
        let base = model.patched.read().await.base().clone();

        let mut sessions = self.sessions.write().await;
        let now = Instant::now();
        let op_count = patch.operations.len();

        // Another task may have given the session its overlay while the base
        // was being read. A refusal there changes nothing, as on the fast path.
        if let Some(overlay) = sessions
            .get_mut(session_id)
            .and_then(SessionState::overlay_existing_mut)
        {
            overlay.try_apply_patch(patch)?;
            return Ok((op_count, overlay.num_patches()));
        }

        // Apply onto an overlay the session does not hold yet, and bind the
        // session only once the patch is in: a refused first patch must not
        // leave a session behind with an empty overlay, which reads as
        // "patched" where an unpatched session reads as the model itself.
        let mut overlay = PatchedVindex::new(base);
        overlay.try_apply_patch(patch)?;
        let active = overlay.num_patches();
        bind_in(
            &mut sessions,
            self.clock,
            self.ttl,
            session_id,
            &model.id,
            now,
        )
        .install_overlay(overlay);
        Ok((op_count, active))
    }

    /// List patches for a session.
    pub async fn list_patches(&self, session_id: &str) -> Vec<serde_json::Value> {
        let sessions = self.sessions.read().await;
        match sessions.get(session_id) {
            Some(session) => session
                .patches()
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "name": patch_name(p),
                        "operations": p.operations.len(),
                        "base_model": p.base_model,
                    })
                })
                .collect(),
            None => vec![],
        }
    }

    /// Remove a patch from a session.
    pub async fn remove_patch(&self, session_id: &str, name: &str) -> Result<usize, String> {
        let mut sessions = self.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| format!("session '{}' not found", session_id))?;

        let idx = session
            .patches()
            .iter()
            .position(|p| patch_name(p) == name)
            .ok_or_else(|| format!("patch '{}' not found in session", name))?;

        // The position was found on an existing overlay, so this is Some.
        let overlay = session
            .overlay_existing_mut()
            .expect("a session with a matching patch has an overlay");
        overlay.remove_patch(idx);
        Ok(overlay.num_patches())
    }

    /// Blocking write access to sessions map (for use in spawn_blocking).
    pub fn sessions_blocking_write(
        &self,
    ) -> tokio::sync::RwLockWriteGuard<'_, HashMap<String, SessionState>> {
        self.sessions.blocking_write()
    }

    /// Async read access to the sessions map.
    pub(super) async fn read_guard(
        &self,
    ) -> tokio::sync::RwLockReadGuard<'_, HashMap<String, SessionState>> {
        self.sessions.read().await
    }

    /// Blocking read access to sessions map (for use in spawn_blocking).
    ///
    /// Used by `/v1/infer` and other read-only paths so concurrent
    /// sessioned inference requests do not serialize behind a single
    /// writer guard for the duration of the forward pass.  Mutations
    /// (`apply_patch`, `remove_patch`) still queue behind any
    /// outstanding readers, which is acceptable: patches are rare and
    /// single-writer-many-readers is the canonical shape.
    pub fn sessions_blocking_read(
        &self,
    ) -> tokio::sync::RwLockReadGuard<'_, HashMap<String, SessionState>> {
        self.sessions.blocking_read()
    }

    /// Number of live sessions.
    ///
    /// Expired-but-not-yet-swept entries are excluded: a session past its
    /// TTL is gone as far as every reader is concerned, whether or not the
    /// sweeper has run.
    pub async fn session_count(&self) -> usize {
        self.live_count_at(Instant::now()).await
    }

    /// [`Self::session_count`] with an explicit clock.
    pub async fn live_count_at(&self, now: Instant) -> usize {
        let sessions = self.sessions.read().await;
        sessions
            .values()
            .filter(|s| s.lease().idle_at(now) <= self.ttl)
            .count()
    }

    /// Every live session, newest access first.
    ///
    /// Read-only: this takes a *read* guard and filters expired entries
    /// out rather than evicting them. Observing must never queue behind
    /// (or in front of) an in-flight sessioned forward pass, which holds a
    /// read guard for its whole duration — a writer here would stall both
    /// inference and every other observer.
    pub async fn list(&self) -> Vec<SessionSummary> {
        self.list_at(Instant::now()).await
    }

    /// [`Self::list`] with an explicit clock.
    pub async fn list_at(&self, now: Instant) -> Vec<SessionSummary> {
        let sessions = self.sessions.read().await;
        let mut out: Vec<SessionSummary> = sessions
            .values()
            .filter(|s| s.lease().idle_at(now) <= self.ttl)
            .map(|s| self.summarize(s))
            .collect();
        out.sort_by(|a, b| {
            b.last_used_at
                .cmp(&a.last_used_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        out
    }

    /// One live session, or `None` if it is absent or expired.
    pub async fn get(&self, id: &str) -> Option<SessionSummary> {
        self.get_at(id, Instant::now()).await
    }

    /// [`Self::get`] with an explicit clock.
    pub async fn get_at(&self, id: &str, now: Instant) -> Option<SessionSummary> {
        let sessions = self.sessions.read().await;
        sessions
            .get(id)
            .filter(|s| s.lease().idle_at(now) <= self.ttl)
            .map(|s| self.summarize(s))
    }

    /// Delete a session: drop its overlay and retire its lease. Returns
    /// how many patches were freed, or `None` if there was no such
    /// session (already deleted, or already expired).
    ///
    /// Killing the lease is the half that outlives the map entry: it is
    /// what stops an in-flight generation from re-inserting a KV
    /// continuation for this session after the delete returns.
    pub async fn delete(&self, id: &str) -> Option<usize> {
        let mut sessions = self.sessions.write().await;
        let removed = sessions.remove(id)?;
        removed.lease().kill();
        Some(removed.num_patches())
    }

    /// Drop every session bound to `model_id`, killing each lease first —
    /// the same reason `delete` kills the lease before removing the map
    /// entry: it stops an in-flight generation from re-inserting a KV
    /// continuation for a session this call is retiring. Called on a
    /// successful model unload (`docs/runtime-lifecycle-design.md` §1's
    /// id-reuse trap): a session's patch overlay was built against this
    /// model's vindex, and a later load reusing the same id with
    /// different weights must not let that overlay silently apply to
    /// them. Safe to call when nothing matches. Returns how many
    /// sessions were removed.
    pub async fn drop_sessions_bound_to(&self, model_id: &str) -> usize {
        let mut sessions = self.sessions.write().await;
        let before = sessions.len();
        sessions.retain(|_, s| {
            let bound = s.lease().model_id() == model_id;
            if bound {
                s.lease().kill();
            }
            !bound
        });
        before - sessions.len()
    }

    fn summarize(&self, session: &SessionState) -> SessionSummary {
        let lease = session.lease();
        SessionSummary {
            id: lease.id().to_string(),
            model: lease.model_id().to_string(),
            created_at: lease.created_at_unix(),
            last_used_at: lease.last_used_at_unix(),
            expires_at: lease.expires_at_unix(self.ttl),
            patch_names: session.patches().iter().map(patch_name).collect(),
            resumptions: lease.resumptions(),
            reused_tokens_total: lease.reused_tokens_total(),
        }
    }
}

/// A patch's identity as every surface names it.
fn patch_name(patch: &VindexPatch) -> String {
    patch
        .description
        .clone()
        .unwrap_or_else(|| PATCH_UNNAMED.to_string())
}
