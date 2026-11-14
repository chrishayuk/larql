//! Per-session server state.
//!
//! A session is a client-chosen identity, carried in the `X-Session-Id`
//! header, that the server hangs two kinds of state off:
//!
//! - a **patch overlay** — a private [`larql_vindex::PatchedVindex`] on top
//!   of the shared read-only base, so patches applied through the session
//!   API are isolated to that client. Materialised lazily: a session that
//!   never patches carries none and reads exactly like the global state.
//! - **KV continuations** — the resident generation states in
//!   [`crate::response_kv`], which are *owned* by the session that
//!   produced them so that deleting the session frees them.
//!
//! Requests with no `X-Session-Id` use the global (shared) state and own
//! nothing.
//!
//! # Lifetime and eviction
//!
//! The session map is bounded by an idle TTL. A session idle for longer
//! than the manager's TTL is dropped by
//! [`SessionManager::evict_expired`], which runs opportunistically inside
//! the manager's write paths and periodically from the server's
//! [`crate::maintenance`] sweeper — both through one routine so the policy
//! cannot diverge. "Idle" means no manager-mediated access refreshed the
//! clock; the read-only inference path holds only a read guard and
//! deliberately does not refresh it, so a session kept alive purely by
//! `/v1/infer` reads still expires and its patches must be re-applied.
//!
//! Read paths (`/v1/sessions`, `/v1/stats`) never evict: they take a read
//! guard and filter expired entries out. An expired session is therefore
//! unobservable the instant it expires, whether or not the sweeper has
//! run — and observing can never queue behind an in-flight forward pass.
//!
//! Deletion (explicit or by expiry) also **kills the session's lease**,
//! which is what prevents a generation still in flight from re-inserting
//! a KV continuation for a session the operator was told is gone. See
//! [`SessionLease`].

mod clock;
mod lease;
mod manager;
mod state;
mod view;

#[cfg(test)]
mod tests;

pub use clock::{unix_now, SessionClock};
pub use lease::SessionLease;
pub use manager::{SessionManager, SessionSummary};
pub use state::SessionState;
pub use view::PatchedView;

use axum::http::HeaderMap;

/// Default session TTL when the configured value is 0 (i.e. unset): one hour.
///
/// Single source of truth — `bootstrap::cli` re-exports this for its flag
/// default so the CLI and the manager can never disagree.
pub const DEFAULT_SESSION_TTL_SECS: u64 = 3600;

/// HTTP header used to scope patches, continuations, and queries to a
/// session.
pub const HEADER_SESSION_ID: &str = "x-session-id";

/// Fallback name for unnamed patches and sessions.
pub const PATCH_UNNAMED: &str = "unnamed";

/// Extract the `X-Session-Id` header value, if present.
pub fn extract_session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get(HEADER_SESSION_ID)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}
