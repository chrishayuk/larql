//! What one request sees of a model: its session's overlay, or the shared one.
//!
//! The browse routes (`select`, `describe`, `walk`, `relations`,
//! `explain-infer`) used to read the shared overlay whatever the request's
//! `X-Session-Id`, so a client that patched its session read back the model
//! as if it had not. They read through [`SessionManager::view_blocking`] now,
//! as `/v1/infer` already did.

use std::collections::HashMap;
use std::ops::Deref;
use std::time::Instant;

use larql_vindex::PatchedVindex;
use tokio::sync::RwLockReadGuard;

use super::manager::SessionManager;
use super::state::SessionState;
use crate::state::LoadedModel;

/// A read guard on the overlay a request reads. Derefs to the overlay.
pub enum PatchedView<'a> {
    /// The session's own overlay. The map guard keeps it from being changed
    /// or dropped while it is read; `id` names the entry, which
    /// [`SessionManager::view_blocking`] checked has an overlay.
    Session {
        sessions: RwLockReadGuard<'a, HashMap<String, SessionState>>,
        id: String,
    },
    /// The model's shared overlay.
    Shared(RwLockReadGuard<'a, PatchedVindex>),
}

impl PatchedView<'_> {
    /// Whether this is a session's own overlay rather than the shared one.
    pub fn is_session(&self) -> bool {
        matches!(self, PatchedView::Session { .. })
    }
}

impl Deref for PatchedView<'_> {
    type Target = PatchedVindex;

    fn deref(&self) -> &PatchedVindex {
        match self {
            PatchedView::Session { sessions, id } => sessions
                .get(id)
                .and_then(SessionState::patched)
                .expect("view_blocking holds the map guard on a session it found patched"),
            PatchedView::Shared(patched) => patched,
        }
    }
}

impl SessionManager {
    /// What `session_id` sees of `model`, for a blocking read
    /// (`spawn_blocking`).
    ///
    /// The session's overlay when the session is live, was bound to this
    /// model, and has been patched. Otherwise the shared overlay: no header,
    /// an unknown or expired id, a session that never patched (it reads as
    /// the model), and a session bound to another loaded model, whose overlay
    /// was built from that model's vindex and is not this one's.
    ///
    /// Reading does not refresh the session's idle clock, as with
    /// `/v1/infer`.
    pub fn view_blocking<'a>(
        &'a self,
        model: &'a LoadedModel,
        session_id: Option<&str>,
    ) -> PatchedView<'a> {
        self.view_blocking_at(model, session_id, Instant::now())
    }

    /// Whether [`Self::view_blocking`] would read `session_id`'s own overlay
    /// of `model`, for an async caller deciding before it reads.
    pub async fn has_patched_view(&self, model: &LoadedModel, session_id: Option<&str>) -> bool {
        let Some(id) = session_id else {
            return false;
        };
        let now = Instant::now();
        self.read_guard()
            .await
            .get(id)
            .is_some_and(|session| self.reads_own_overlay(session, model, now))
    }

    fn reads_own_overlay(&self, session: &SessionState, model: &LoadedModel, now: Instant) -> bool {
        session.lease().idle_at(now) <= self.ttl()
            && session.lease().model_id() == model.id
            && session.patched().is_some()
    }

    /// [`Self::view_blocking`] with an explicit clock.
    pub fn view_blocking_at<'a>(
        &'a self,
        model: &'a LoadedModel,
        session_id: Option<&str>,
        now: Instant,
    ) -> PatchedView<'a> {
        if let Some(id) = session_id {
            let sessions = self.sessions_blocking_read();
            let readable = sessions
                .get(id)
                .is_some_and(|session| self.reads_own_overlay(session, model, now));
            if readable {
                return PatchedView::Session {
                    sessions,
                    id: id.to_string(),
                };
            }
        }
        PatchedView::Shared(model.patched.blocking_read())
    }
}
