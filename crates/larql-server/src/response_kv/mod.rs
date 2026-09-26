//! N1 — KV continuation cache for the Responses API (V3 runtimes).
//!
//! A stored response's conversation can be continued via
//! `previous_response_id`. Without this cache every chained turn
//! re-prefills the whole conversation; with it the producing turn's
//! [`V3KvHandoff`] stays resident, keyed by response id, and the next
//! link continues from it — the resumed positions cost nothing.
//!
//! Contract notes:
//!
//! - **Purely an optimisation.** The generation path
//!   ([`crate::vindex3::generate_v3_resumable`]) only reuses a handoff
//!   whose absorbed ids are a strict prefix of the new prompt; any
//!   mismatch falls back to a full prefill. Losing an entry (eviction,
//!   TTL, restart, branching) can never change produced tokens.
//! - **Take-once.** A chained request *takes* the entry (KV states are
//!   large; sharing one across concurrent chains would need cloning).
//!   Branching two continuations off one response gives the second
//!   branch a full prefill — correct, just not accelerated.
//! - **Tightly bounded.** KV states are the biggest per-conversation
//!   objects the server holds, so the cap is small and the TTL short
//!   compared to the session map; both are CLI-tunable and the
//!   maintenance sweeper drives the TTL.
//! - **Session-owned when the client asks.** A request carrying an
//!   `X-Session-Id` binds a [`crate::session`] session, and the state it
//!   retains is owned by it. That ownership is what lets
//!   `DELETE /v1/sessions/{id}` free continuations, and what lets a late
//!   insert from a generation still in flight be refused instead of
//!   resurrecting a deleted session. Requests without the header retain
//!   unowned states, governed by capacity and TTL alone.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::session::SessionLease;
use crate::vindex3::V3KvHandoff;

#[cfg(test)]
mod tests;

/// Default resident continuation states. Each can hold a whole
/// conversation's KV, so the default is deliberately small; operators
/// with RAM to spare raise `--v3-kv-cache-entries`.
pub const DEFAULT_MAX_ENTRIES: usize = 4;

/// Default idle TTL: ten minutes. Deliberately shorter than the
/// session map's one hour — an idle KV state is far more expensive
/// than an idle patch session.
pub const DEFAULT_TTL_SECS: u64 = 600;

/// What one session's continuations look like from outside — the
/// metadata `/v1/sessions` reports. Never KV bytes, never a cache key.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SessionContinuation {
    /// Resident continuation states owned by the session.
    pub entries: usize,
    /// Prompt tokens already absorbed into those states.
    pub input_tokens: u64,
}

impl SessionContinuation {
    /// Whether the session has anything to resume from.
    pub fn available(&self) -> bool {
        self.entries > 0
    }
}

struct Entry {
    handoff: V3KvHandoff,
    /// The runtime binding that produced this state. A KV state is
    /// meaningless under any other model's weights, so [`take`] with a
    /// different model id refuses — and leaves the entry for the
    /// rightful chain.
    ///
    /// [`take`]: ResponseKvCache::take
    model_id: String,
    /// The session that owns this state, when the producing request
    /// carried an `X-Session-Id`. Ownership is what makes
    /// `DELETE /v1/sessions/{id}` able to free continuations, and what
    /// makes a late insert from an in-flight generation refusable.
    /// `None` = unowned: governed by capacity and TTL alone.
    owner: Option<Arc<SessionLease>>,
    last_access: Instant,
}

impl Entry {
    /// An entry is orphaned once its owning session is gone. Orphans are
    /// unreachable state, so every path that touches one drops it.
    fn orphaned(&self) -> bool {
        self.owner.as_ref().is_some_and(|o| !o.is_alive())
    }
}

#[derive(Default)]
struct CacheInner {
    by_id: HashMap<String, Entry>,
    /// Insertion order for capacity eviction.
    order: VecDeque<String>,
}

/// Bounded, TTL-swept map: response id → the KV continuation state of
/// the generation that produced it.
pub struct ResponseKvCache {
    inner: Mutex<CacheInner>,
    max_entries: usize,
    ttl: Duration,
    /// Chained requests whose [`Self::take`] found a resident state.
    hits: AtomicU64,
    /// Chained requests whose [`Self::take`] found nothing (evicted,
    /// expired, consumed by an earlier chain, or never retained).
    misses: AtomicU64,
    /// Generations where resumption actually ENGAGED — the taken state
    /// passed the exact ids-prefix check and skipped prefill work.
    /// `hits - resumptions` is the prefix-stability gap: resident
    /// state found but unusable under exact token-id identity.
    resumptions: AtomicU64,
    /// Total prompt tokens served from resumed KV across all engaged
    /// resumptions.
    reused_tokens_total: AtomicU64,
    /// Hits whose state's recorded continuation authority refused to
    /// resume under the model's (C4) — served by a fresh prefill.
    refusals: AtomicU64,
}

impl ResponseKvCache {
    /// `max_entries == 0` disables the cache entirely (every chain
    /// re-prefills). `ttl_secs == 0` selects [`DEFAULT_TTL_SECS`].
    pub fn new(max_entries: usize, ttl_secs: u64) -> Self {
        Self {
            inner: Mutex::new(CacheInner::default()),
            max_entries,
            ttl: Duration::from_secs(if ttl_secs == 0 {
                DEFAULT_TTL_SECS
            } else {
                ttl_secs
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            resumptions: AtomicU64::new(0),
            reused_tokens_total: AtomicU64::new(0),
            refusals: AtomicU64::new(0),
        }
    }

    pub fn enabled(&self) -> bool {
        self.max_entries > 0
    }

    /// The configured idle TTL.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Retain `handoff` — produced by the runtime bound as `model_id`
    /// — under `id`, evicting the oldest entries past capacity. No-op
    /// when the cache is disabled.
    ///
    /// `owner` is the session the producing request was bound to, if
    /// any. A **dead** owner refuses the insert outright: the session
    /// was deleted or expired while this generation was in flight, and
    /// storing its state now would resurrect a session the operator has
    /// already been told is gone. The check happens under the cache
    /// lock, which is what makes it race-free against a concurrent
    /// delete — see [`SessionLease`].
    pub fn insert(
        &self,
        id: &str,
        model_id: &str,
        owner: Option<Arc<SessionLease>>,
        handoff: V3KvHandoff,
    ) {
        if !self.enabled() {
            return;
        }
        let mut inner = self.lock();
        if owner.as_ref().is_some_and(|o| !o.is_alive()) {
            return;
        }
        let entry = Entry {
            handoff,
            model_id: model_id.to_string(),
            owner,
            last_access: Instant::now(),
        };
        if inner.by_id.insert(id.to_string(), entry).is_none() {
            inner.order.push_back(id.to_string());
        }
        while inner.by_id.len() > self.max_entries {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            inner.by_id.remove(&oldest);
        }
    }

    /// Take the continuation state for `id`, removing it (take-once —
    /// see the module doc). Counts a hit or a miss for `/v1/stats`.
    ///
    /// `model_id` must match the binding that produced the state: a KV
    /// state under different weights would be garbage even when the
    /// token-id prefix happens to match (shared tokenizers make that
    /// real). A mismatched take is a miss and does NOT consume the
    /// entry — the rightful chain can still claim it.
    pub fn take(&self, id: &str, model_id: &str) -> Option<V3KvHandoff> {
        let mut inner = self.lock();
        // An orphan is dropped rather than left behind: unlike a foreign
        // model id, there is no rightful chain left to claim it.
        if inner.by_id.get(id).is_some_and(Entry::orphaned) {
            inner.by_id.remove(id);
            inner.order.retain(|e| e != id);
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        if inner
            .by_id
            .get(id)
            .is_none_or(|entry| entry.model_id != model_id)
        {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let entry = inner.by_id.remove(id).expect("presence checked above");
        inner.order.retain(|e| e != id);
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(entry.handoff)
    }

    /// Chained takes that found a resident state.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Chained takes that found nothing.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Record that a taken state passed the exact ids-prefix check and
    /// `reused_tokens` prompt positions were served from it.
    pub fn record_resumption(&self, reused_tokens: usize) {
        self.resumptions.fetch_add(1, Ordering::Relaxed);
        self.reused_tokens_total
            .fetch_add(reused_tokens as u64, Ordering::Relaxed);
    }

    /// Generations where resumption actually engaged.
    /// Count a hit whose state's recorded authority refused to resume
    /// under the model's continuation (C4). Counted apart from `misses`:
    /// a refusal is state that existed and was rejected, not state that
    /// was absent.
    pub fn record_refusal(&self) {
        self.refusals.fetch_add(1, Ordering::Relaxed);
    }

    pub fn refusals(&self) -> u64 {
        self.refusals.load(Ordering::Relaxed)
    }

    pub fn resumptions(&self) -> u64 {
        self.resumptions.load(Ordering::Relaxed)
    }

    /// Total prompt tokens served from resumed KV.
    pub fn reused_tokens_total(&self) -> u64 {
        self.reused_tokens_total.load(Ordering::Relaxed)
    }

    /// The configured capacity (0 = the cache is disabled).
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Free every continuation owned by `session_id`, whatever its
    /// incarnation. Called by `DELETE /v1/sessions/{id}` — and safe to
    /// call for a session that no longer exists, which is what makes
    /// deletion idempotent. Returns how many states were freed.
    pub fn drop_owned_by(&self, session_id: &str) -> usize {
        let mut inner = self.lock();
        let before = inner.by_id.len();
        inner
            .by_id
            .retain(|_, e| e.owner.as_ref().is_none_or(|o| o.id() != session_id));
        let removed = before - inner.by_id.len();
        if removed > 0 {
            prune_order(&mut inner);
        }
        removed
    }

    /// Free every continuation produced by `model_id`, regardless of
    /// session ownership. Called on a successful model unload
    /// (`docs/runtime-lifecycle-design.md` §1's id-reuse trap): a KV
    /// state is meaningless once the runtime that produced it is gone,
    /// and if a future load reuses the same model id with different
    /// weights, nothing must survive to be silently resumed against
    /// them. Safe to call when nothing matches. Returns how many
    /// states were freed.
    pub fn drop_owned_by_model(&self, model_id: &str) -> usize {
        let mut inner = self.lock();
        let before = inner.by_id.len();
        inner.by_id.retain(|_, e| e.model_id != model_id);
        let removed = before - inner.by_id.len();
        if removed > 0 {
            prune_order(&mut inner);
        }
        removed
    }

    /// What `session_id` currently has to resume from.
    pub fn owned_by(&self, session_id: &str) -> SessionContinuation {
        let inner = self.lock();
        let mine = inner
            .by_id
            .values()
            .filter(|e| e.owner.as_ref().is_some_and(|o| o.id() == session_id));
        let mut summary = SessionContinuation::default();
        for entry in mine {
            summary.entries += 1;
            summary.input_tokens += entry.handoff.absorbed_ids.len() as u64;
        }
        summary
    }

    /// Drop every entry idle longer than the TTL, plus every orphan whose
    /// owning session has gone. Returns how many were removed; wired into
    /// the maintenance sweeper.
    pub fn evict_expired(&self) -> usize {
        self.evict_expired_at(Instant::now())
    }

    /// [`Self::evict_expired`] with an explicit clock, so tests can
    /// advance time without sleeping.
    pub fn evict_expired_at(&self, now: Instant) -> usize {
        let mut inner = self.lock();
        let before = inner.by_id.len();
        let ttl = self.ttl;
        inner
            .by_id
            .retain(|_, e| now.duration_since(e.last_access) <= ttl && !e.orphaned());
        let removed = before - inner.by_id.len();
        if removed > 0 {
            prune_order(&mut inner);
        }
        removed
    }

    pub fn len(&self) -> usize {
        self.lock().by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CacheInner> {
        // A poisoned lock only means a panic mid-insert; recover rather
        // than wedge every subsequent chained request.
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// Re-derive the capacity-eviction order after entries were removed out
/// of band.
fn prune_order(inner: &mut CacheInner) {
    let survivors: std::collections::HashSet<&String> = inner.by_id.keys().collect();
    let kept: VecDeque<String> = inner
        .order
        .iter()
        .filter(|id| survivors.contains(id))
        .cloned()
        .collect();
    inner.order = kept;
}
