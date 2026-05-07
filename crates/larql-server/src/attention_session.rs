//! Attention session lifecycle.
//!
//! An attention session is a server-side handle for a `KvCache`.
//! Sessions are created via `POST /v1/attention/session`, advanced via
//! `prefill` and `decode`, snapshotted via `/v1/kv-cache/snapshot`,
//! and dropped via `DELETE /v1/attention/session/{id}`.
//!
//! See `openspec/changes/attention-service-routes/`.
//!
//! ## Concurrency model
//!
//! - The map itself uses a `std::sync::RwLock` — adds/removes are fast.
//! - Each session sits behind its own `tokio::sync::RwLock` so async
//!   prefill/decode handlers can hold the guard across `.await`.
//! - Prefill takes the write half (mutates the cache); decode takes
//!   the write half (also mutates — appending one column to K/V);
//!   GET takes the read half.
//!
//! ## Session id format
//!
//! 26-character Crockford-Base32-encoded ULID (128 bits): timestamp
//! (48 bits) + randomness (80 bits). Lexicographically sortable;
//! debug-friendly. Implementation: a tiny self-contained generator —
//! we don't pull in a crate just for one type.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, RwLock,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use larql_inference::attention::decode::KvCache;
use larql_rotorquant::KvFormat;
use tokio::sync::RwLock as TokioRwLock;

/// Default TTL — sessions idle longer than this are reaped.
pub const DEFAULT_SESSION_TTL_SECS: u64 = 600;

/// 26-character Crockford-Base32 ULID.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SessionId(pub String);

impl SessionId {
    pub fn new() -> Self {
        Self::from_parts(now_ms(), random_u80())
    }

    fn from_parts(timestamp_ms: u64, randomness: u128) -> Self {
        // Bit packing: top 48 bits = timestamp, low 80 bits = randomness.
        let bits: u128 = (u128::from(timestamp_ms) << 80) | (randomness & ((1u128 << 80) - 1));
        Self(crockford_base32_encode_128(bits))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One server-side attention session.
#[derive(Debug)]
pub struct AttentionSession {
    pub id: SessionId,
    pub model_id: String,
    pub cache: KvCache,
    pub created_at: Instant,
    pub last_used: Instant,
    /// Number of tokens currently materialised in the cache (post-prefill).
    pub seq_len: usize,
    /// Whether `prefill` has completed at least once. `decode` rejects
    /// the request if this is false.
    pub prefilled: bool,
}

impl AttentionSession {
    /// Construct a session with a fresh `KvCache(num_layers)` and
    /// optional KV compression format.
    pub fn new(
        id: SessionId,
        model_id: impl Into<String>,
        num_layers: usize,
        kv_format: Option<KvFormat>,
    ) -> Self {
        let mut cache = KvCache::with_layers(num_layers);
        if let Some(fmt) = kv_format {
            cache.set_kv_format(fmt);
        }
        let now = Instant::now();
        Self {
            id,
            model_id: model_id.into(),
            cache,
            created_at: now,
            last_used: now,
            seq_len: 0,
            prefilled: false,
        }
    }

    /// Record activity. Called by every endpoint that touches the session.
    pub fn touch(&mut self) {
        self.last_used = Instant::now();
    }
}

type SessionEntry = Arc<TokioRwLock<AttentionSession>>;

/// Concurrent map of session id → session.
pub struct AttentionSessionMap {
    inner: RwLock<HashMap<SessionId, SessionEntry>>,
    ttl: Duration,
    /// Soft cap on concurrent sessions. `insert` returns `Err` once
    /// the map is full. The reaper trims as TTL expires.
    cap: usize,
}

impl AttentionSessionMap {
    pub fn new(ttl_secs: u64, cap: usize) -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(ttl_secs),
            cap,
        }
    }

    /// Insert a fresh session. Returns the inserted entry handle.
    /// Errors with `SessionMapError::AtCap` when the map is full.
    pub fn insert(&self, session: AttentionSession) -> Result<SessionEntry, SessionMapError> {
        let mut g = self.inner.write().unwrap();
        if g.len() >= self.cap {
            return Err(SessionMapError::AtCap { cap: self.cap });
        }
        let id = session.id.clone();
        let entry = Arc::new(TokioRwLock::new(session));
        g.insert(id, Arc::clone(&entry));
        Ok(entry)
    }

    pub fn get(&self, id: &SessionId) -> Option<SessionEntry> {
        let g = self.inner.read().unwrap();
        g.get(id).cloned()
    }

    pub fn remove(&self, id: &SessionId) -> Option<SessionEntry> {
        let mut g = self.inner.write().unwrap();
        g.remove(id)
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop sessions whose `last_used` is older than the configured TTL.
    /// Returns the number of sessions reaped.
    pub fn reap_idle(&self) -> usize {
        let now = Instant::now();
        let mut g = self.inner.write().unwrap();
        let before = g.len();
        // Two-pass: identify candidates while holding the read of each
        // session non-blockingly via `try_read`; remove them on the
        // second pass. A session whose tokio lock is held by an
        // active handler skips reap (gets a free pass for one cycle).
        let mut to_drop: Vec<SessionId> = Vec::new();
        for (id, entry) in g.iter() {
            if let Ok(sess) = entry.try_read() {
                if now.duration_since(sess.last_used) > self.ttl {
                    to_drop.push(id.clone());
                }
            }
        }
        for id in to_drop {
            g.remove(&id);
        }
        before - g.len()
    }

    /// Snapshot the first 16 token-id prefix of every active session
    /// into a `PrefixBloom`-style `[u64; ?]` digest. Used by the
    /// heartbeat to populate the router's `cached_prefixes` bloom.
    ///
    /// **Caveat:** this iterates with `try_read`; sessions actively
    /// held in a write guard are skipped this tick (they appear in
    /// the next bloom rebuild).
    pub fn prefix_hashes(&self, prefix_len: usize) -> Vec<u64> {
        let g = self.inner.read().unwrap();
        let mut out = Vec::with_capacity(g.len());
        for entry in g.values() {
            if let Ok(sess) = entry.try_read() {
                // Without first-token visibility yet, hash the session
                // id itself as a stable per-session digest. The proto
                // wiring landing alongside the route handlers will
                // replace this with the first `prefix_len` tokens.
                let _ = prefix_len;
                out.push(fnv1a_64(sess.id.as_str().as_bytes()));
            }
        }
        out
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SessionMapError {
    #[error("session map is at its cap of {cap} concurrent sessions")]
    AtCap { cap: usize },
}

// ── helpers ─────────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// 80-bit randomness for the ULID. Uses a process-monotonic counter
/// XOR'd with the high-resolution timer; we don't need cryptographic
/// randomness, just per-process uniqueness.
fn random_u80() -> u128 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let lo = COUNTER.fetch_add(1, Ordering::Relaxed);
    let hi = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let combined = (u128::from(hi) << 64) | u128::from(lo);
    combined & ((1u128 << 80) - 1)
}

const CROCKFORD_ALPHABET: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Encode a u128 into a 26-character Crockford-Base32 string. The
/// alphabet excludes `I`, `L`, `O`, `U` to avoid common visual
/// confusion (matches the ULID spec).
fn crockford_base32_encode_128(mut value: u128) -> String {
    let mut buf = [0u8; 26];
    for slot in buf.iter_mut().rev() {
        *slot = CROCKFORD_ALPHABET[(value & 0x1f) as usize];
        value >>= 5;
    }
    String::from_utf8(buf.to_vec()).expect("crockford-base32 chars are valid utf8")
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_is_26_chars() {
        let id = SessionId::new();
        assert_eq!(id.as_str().len(), 26);
    }

    #[test]
    fn session_ids_are_unique_under_a_burst() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1024 {
            assert!(seen.insert(SessionId::new()));
        }
    }

    #[test]
    fn session_ids_lex_sort_by_creation_time() {
        let a = SessionId::new();
        std::thread::sleep(Duration::from_millis(2));
        let b = SessionId::new();
        assert!(a.as_str() < b.as_str(), "{} ≮ {}", a.as_str(), b.as_str());
    }

    #[test]
    fn insert_and_get_round_trip() {
        let map = AttentionSessionMap::new(60, 16);
        let id = SessionId::new();
        let sess = AttentionSession::new(id.clone(), "test-model", 4, None);
        let _ = map.insert(sess).expect("insert");
        let got = map.get(&id).expect("get");
        let g = got.try_read().expect("read");
        assert_eq!(g.model_id, "test-model");
        assert_eq!(g.cache.layers.len(), 4);
        assert!(!g.prefilled);
    }

    #[test]
    fn delete_makes_get_none() {
        let map = AttentionSessionMap::new(60, 16);
        let id = SessionId::new();
        let _ = map
            .insert(AttentionSession::new(id.clone(), "m", 1, None))
            .expect("insert");
        assert!(map.remove(&id).is_some());
        assert!(map.get(&id).is_none());
    }

    #[test]
    fn cap_enforced() {
        let map = AttentionSessionMap::new(60, 2);
        for _ in 0..2 {
            let id = SessionId::new();
            map.insert(AttentionSession::new(id, "m", 1, None))
                .expect("insert under cap");
        }
        let id3 = SessionId::new();
        match map.insert(AttentionSession::new(id3, "m", 1, None)) {
            Err(SessionMapError::AtCap { cap }) => assert_eq!(cap, 2),
            other => panic!("expected AtCap, got {other:?}"),
        }
    }

    #[test]
    fn reap_drops_idle_sessions() {
        let map = AttentionSessionMap::new(0, 16);
        let id = SessionId::new();
        let _ = map
            .insert(AttentionSession::new(id.clone(), "m", 1, None))
            .expect("insert");
        // ttl = 0 means anything older than 0 ns is idle.
        std::thread::sleep(Duration::from_millis(2));
        let reaped = map.reap_idle();
        assert_eq!(reaped, 1);
        assert!(map.get(&id).is_none());
    }

    #[test]
    fn iso4_session_carries_format() {
        let map = AttentionSessionMap::new(60, 16);
        let id = SessionId::new();
        let _ = map
            .insert(AttentionSession::new(
                id.clone(),
                "m",
                4,
                Some(KvFormat::Iso4),
            ))
            .expect("insert");
        let entry = map.get(&id).unwrap();
        let s = entry.try_read().unwrap();
        assert_eq!(s.cache.kv_format, Some(KvFormat::Iso4));
    }

    #[test]
    fn prefix_hashes_returns_one_per_session() {
        let map = AttentionSessionMap::new(60, 16);
        for _ in 0..5 {
            let id = SessionId::new();
            map.insert(AttentionSession::new(id, "m", 1, None))
                .expect("insert");
        }
        let hashes = map.prefix_hashes(16);
        assert_eq!(hashes.len(), 5);
    }
}
