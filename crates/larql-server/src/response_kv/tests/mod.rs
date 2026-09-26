//! Unit tests for the KV continuation cache: enable/disable, take-once
//! semantics, capacity FIFO, and TTL eviction. Session ownership lives
//! in [`ownership`].

mod ownership;

use super::*;
use larql_kv::{shipped_continuations, CanonicalFactory};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::ContinuationFactory;

/// Small capacity so eviction is observable without bulk inserts.
const TEST_MAX_ENTRIES: usize = 2;
/// Short TTL expressed through the explicit-clock entry point.
const TEST_TTL_SECS: u64 = 10;
/// The runtime binding the test states belong to.
const TEST_MODEL: &str = "m-test";

fn handoff(ids: &[u32]) -> V3KvHandoff {
    let continuation = shipped_continuations()
        .select(
            &CanonicalFactory.identity(),
            &ContinuationConfig::empty(),
            &[],
        )
        .unwrap()
        .begin();
    V3KvHandoff {
        continuation,
        absorbed_ids: ids.to_vec(),
    }
}

#[test]
fn zero_entries_disables_the_cache() {
    let cache = ResponseKvCache::new(0, TEST_TTL_SECS);
    assert!(!cache.enabled());
    cache.insert("resp_1", TEST_MODEL, None, handoff(&[1]));
    assert!(cache.is_empty());
    assert!(cache.take("resp_1", TEST_MODEL).is_none());
}

#[test]
fn zero_ttl_falls_back_to_default() {
    let cache = ResponseKvCache::new(TEST_MAX_ENTRIES, 0);
    assert_eq!(cache.ttl(), Duration::from_secs(DEFAULT_TTL_SECS));
}

#[test]
fn take_returns_the_handoff_exactly_once() {
    let cache = ResponseKvCache::new(TEST_MAX_ENTRIES, TEST_TTL_SECS);
    cache.insert("resp_1", TEST_MODEL, None, handoff(&[1, 2, 3]));
    assert_eq!(cache.len(), 1);

    let taken = cache.take("resp_1", TEST_MODEL).expect("first take hits");
    assert_eq!(taken.absorbed_ids, vec![1, 2, 3]);
    assert!(
        cache.take("resp_1", TEST_MODEL).is_none(),
        "take-once semantics"
    );
    assert!(cache.is_empty());
}

#[test]
fn insert_same_id_replaces_without_growing() {
    let cache = ResponseKvCache::new(TEST_MAX_ENTRIES, TEST_TTL_SECS);
    cache.insert("resp_1", TEST_MODEL, None, handoff(&[1]));
    cache.insert("resp_1", TEST_MODEL, None, handoff(&[1, 2]));
    assert_eq!(cache.len(), 1);
    assert_eq!(
        cache.take("resp_1", TEST_MODEL).unwrap().absorbed_ids,
        vec![1, 2]
    );
}

#[test]
fn capacity_evicts_the_oldest_entry() {
    let cache = ResponseKvCache::new(TEST_MAX_ENTRIES, TEST_TTL_SECS);
    cache.insert("resp_1", TEST_MODEL, None, handoff(&[1]));
    cache.insert("resp_2", TEST_MODEL, None, handoff(&[2]));
    cache.insert("resp_3", TEST_MODEL, None, handoff(&[3]));
    assert_eq!(cache.len(), TEST_MAX_ENTRIES);
    assert!(
        cache.take("resp_1", TEST_MODEL).is_none(),
        "oldest must be evicted"
    );
    assert!(cache.take("resp_2", TEST_MODEL).is_some());
    assert!(cache.take("resp_3", TEST_MODEL).is_some());
}

#[test]
fn ttl_evicts_idle_entries_and_keeps_fresh_ones() {
    let cache = ResponseKvCache::new(TEST_MAX_ENTRIES, TEST_TTL_SECS);
    cache.insert("resp_old", TEST_MODEL, None, handoff(&[1]));
    let now = Instant::now();
    // Past the TTL both by a margin; nothing else inserted since, so
    // the single entry goes and the cache is empty.
    let removed = cache.evict_expired_at(now + Duration::from_secs(TEST_TTL_SECS + 1));
    assert_eq!(removed, 1);
    assert!(cache.is_empty());

    // Fresh entries survive the wall-clock sweep.
    cache.insert("resp_new", TEST_MODEL, None, handoff(&[2]));
    assert_eq!(cache.evict_expired(), 0);
    assert_eq!(cache.len(), 1);
}

#[test]
fn eviction_keeps_capacity_accounting_consistent() {
    // After a TTL sweep removes entries, capacity eviction must still
    // target the true oldest survivor (the order queue is compacted).
    let cache = ResponseKvCache::new(TEST_MAX_ENTRIES, TEST_TTL_SECS);
    cache.insert("resp_1", TEST_MODEL, None, handoff(&[1]));
    let now = Instant::now();
    assert_eq!(
        cache.evict_expired_at(now + Duration::from_secs(TEST_TTL_SECS + 1)),
        1
    );
    cache.insert("resp_2", TEST_MODEL, None, handoff(&[2]));
    cache.insert("resp_3", TEST_MODEL, None, handoff(&[3]));
    cache.insert("resp_4", TEST_MODEL, None, handoff(&[4]));
    assert_eq!(cache.len(), TEST_MAX_ENTRIES);
    assert!(
        cache.take("resp_2", TEST_MODEL).is_none(),
        "resp_2 was the oldest live"
    );
    assert!(cache.take("resp_3", TEST_MODEL).is_some());
    assert!(cache.take("resp_4", TEST_MODEL).is_some());
}

#[test]
fn take_counts_hits_and_misses() {
    let cache = ResponseKvCache::new(TEST_MAX_ENTRIES, TEST_TTL_SECS);
    assert_eq!((cache.hits(), cache.misses()), (0, 0));

    cache.insert("resp_1", TEST_MODEL, None, handoff(&[1]));
    assert!(cache.take("resp_1", TEST_MODEL).is_some());
    assert_eq!((cache.hits(), cache.misses()), (1, 0));

    // Take-once: the second take on the same id is a miss, as is an
    // id that was never retained.
    assert!(cache.take("resp_1", TEST_MODEL).is_none());
    assert!(cache.take("resp_never", TEST_MODEL).is_none());
    assert_eq!((cache.hits(), cache.misses()), (1, 2));
}

#[test]
fn capacity_accessor_reports_the_configured_bound() {
    assert_eq!(
        ResponseKvCache::new(TEST_MAX_ENTRIES, TEST_TTL_SECS).max_entries(),
        TEST_MAX_ENTRIES
    );
    assert_eq!(ResponseKvCache::new(0, TEST_TTL_SECS).max_entries(), 0);
}

#[test]
fn cross_model_take_is_a_non_consuming_miss() {
    // A KV state under different weights would be garbage even when
    // the token-id prefix happens to match, so a take naming another
    // binding refuses — and must NOT consume the entry, which the
    // rightful chain can still claim.
    let cache = ResponseKvCache::new(TEST_MAX_ENTRIES, TEST_TTL_SECS);
    cache.insert("resp_1", TEST_MODEL, None, handoff(&[1, 2]));

    assert!(cache.take("resp_1", "other-model").is_none());
    assert_eq!((cache.hits(), cache.misses()), (0, 1));
    assert_eq!(cache.len(), 1, "the mismatched take must not consume");

    let taken = cache.take("resp_1", TEST_MODEL).expect("rightful chain");
    assert_eq!(taken.absorbed_ids, vec![1, 2]);
    assert_eq!((cache.hits(), cache.misses()), (1, 1));
}

#[test]
fn resumption_counters_track_engaged_reuse() {
    let cache = ResponseKvCache::new(TEST_MAX_ENTRIES, TEST_TTL_SECS);
    assert_eq!((cache.resumptions(), cache.reused_tokens_total()), (0, 0));
    cache.record_resumption(5);
    cache.record_resumption(3);
    assert_eq!((cache.resumptions(), cache.reused_tokens_total()), (2, 8));
}
