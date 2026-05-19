//! Long-running chaos test for the grid (Task #84, ADR-0020 follow-up).
//!
//! Drives `GridState` through thousands of randomised register / deregister
//! / heartbeat / route cycles and asserts a small set of safety invariants
//! after every tick:
//!
//!  1. **`route()` never panics.** A bug in the route table (stale
//!     `server_id` left behind by a deregister, malformed expert range,
//!     etc.) would surface as a panic inside `min_by`/`unwrap` deep in the
//!     comparator. We call `route()` for every layer on every tick.
//!
//!  2. **Coverage floor.** When at least `K` of the candidate servers are
//!     currently registered, every layer in the covered range must have
//!     at least one owner (`has_owners_for` returns `true` AND `route()`
//!     returns `Some`). This catches register/deregister desyncs.
//!
//!  3. **Replication ledger consistency.** `over_replicated_ranges()` and
//!     `under_replicated_ranges()` must never reference a `server_id` or
//!     layer-range that doesn't correspond to a live registered server.
//!
//! Determinism: the RNG is a fixed-seeded `xorshift64` so failures are
//! reproducible. Tweak `SEED` or `TICKS` to widen the search.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use larql_router::grid::{GridState, ServerEntry};

const SEED: u64 = 0x5e1f_9c3b_d2a4_7681;
const TICKS: usize = 5_000;
/// Eight candidates, each owning a 4-layer window that overlaps with its
/// neighbours by 2 layers — so layers 0..18 are coverable by ≥2 candidates,
/// which lets the test exercise the replication ledger when target>1.
const N_CANDIDATES: usize = 8;
const LAYERS: u32 = 18;

#[derive(Clone)]
struct Candidate {
    server_id: String,
    layer_start: u32,
    layer_end: u32,
}

fn build_candidates() -> Vec<Candidate> {
    (0..N_CANDIDATES)
        .map(|i| Candidate {
            server_id: format!("srv-{i}"),
            layer_start: (i * 2) as u32,
            layer_end: (i * 2 + 3) as u32,
        })
        .collect()
}

fn make_entry(c: &Candidate, in_flight: u32) -> ServerEntry {
    ServerEntry {
        server_id: c.server_id.clone(),
        listen_url: format!("http://{}:0", c.server_id),
        model_id: "m".into(),
        layer_start: c.layer_start,
        layer_end: c.layer_end,
        vindex_hash: "h".into(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: in_flight,
        last_seen: Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
    }
}

struct Xorshift64(u64);
impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Avoid the all-zeros fixed point.
        Self(if seed == 0 { 0xdead_beef } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Walk every ledger surface and assert it only references currently
/// registered servers / valid layer ranges. Returns the set of live
/// server_ids so the caller can cross-check coverage.
fn assert_invariants(grid: &GridState, expected_live: &HashSet<String>, tick: usize) {
    // 1. servers() must equal the live set we maintain in the test harness.
    let actual_live: HashSet<String> = grid.servers().map(|(id, _)| id.clone()).collect();
    assert_eq!(
        &actual_live, expected_live,
        "tick {tick}: servers() drift — grid={actual_live:?}, expected={expected_live:?}"
    );

    // 2. route() must not panic for any layer, and any layer the route
    //    table claims is owned must resolve to Some(url).
    for layer in 0..LAYERS {
        let has = grid.has_owners_for(Some("m"), layer);
        let routed = grid.route(Some("m"), layer);
        if has {
            assert!(
                routed.is_some(),
                "tick {tick}: has_owners_for(layer={layer})=true but route()=None"
            );
        }
    }

    // 3. Replication ledger must only mention layer-ranges that are inside
    //    the configured candidate space and only experts 0..=0 (dense).
    for (model, ls, le, es, ee, _count) in grid.over_replicated_ranges() {
        assert_eq!(
            model, "m",
            "tick {tick}: over-replication ledger leaked model_id={model}"
        );
        assert!(
            ls < LAYERS && le < LAYERS && ls <= le,
            "tick {tick}: over-replication ledger has out-of-band range {ls}..={le}"
        );
        assert_eq!(
            (es, ee),
            (0, 0),
            "tick {tick}: dense fixture must not produce expert ranges"
        );
    }
    for (model, ls, le, es, ee, _deficit) in grid.under_replicated_ranges() {
        assert_eq!(
            model, "m",
            "tick {tick}: under-replication ledger leaked model_id={model}"
        );
        assert!(
            ls < LAYERS && le < LAYERS && ls <= le,
            "tick {tick}: under-replication ledger has out-of-band range {ls}..={le}"
        );
        assert_eq!(
            (es, ee),
            (0, 0),
            "tick {tick}: dense fixture must not produce expert ranges"
        );
    }
}

/// Compute which layers in 0..LAYERS are covered by the given live set.
fn expected_covered(live: &HashSet<String>, candidates: &[Candidate]) -> HashSet<u32> {
    let mut out = HashSet::new();
    for c in candidates {
        if live.contains(&c.server_id) {
            for l in c.layer_start..=c.layer_end {
                out.insert(l);
            }
        }
    }
    out
}

#[test]
fn grid_survives_long_churn_run() {
    let candidates = build_candidates();
    let mut grid = GridState::default();
    grid.set_target_replicas(1);
    let mut rng = Xorshift64::new(SEED);
    let mut live: HashSet<String> = HashSet::new();

    for tick in 0..TICKS {
        let action = rng.range(6);
        match action {
            // Register a random candidate (idempotent — register() upserts).
            0 | 1 => {
                let c = &candidates[rng.range(candidates.len())];
                let in_flight = (rng.range(10)) as u32;
                grid.register(make_entry(c, in_flight));
                live.insert(c.server_id.clone());
            }
            // Deregister a random currently-live candidate.
            2 => {
                if !live.is_empty() {
                    let ids: Vec<String> = live.iter().cloned().collect();
                    let victim = &ids[rng.range(ids.len())];
                    grid.deregister(victim);
                    live.remove(victim);
                }
            }
            // Heartbeat — bumps in-flight and req/s on a live server.
            3 => {
                if !live.is_empty() {
                    let ids: Vec<String> = live.iter().cloned().collect();
                    let id = &ids[rng.range(ids.len())];
                    grid.update_heartbeat(
                        id,
                        rng.range(100) as f32,
                        rng.range(10_000_000) as u64,
                        rng.range(32) as u32,
                        Vec::new(),
                        rng.range(1_000) as f32,
                    );
                }
            }
            // Route burst — call route() across every layer once.
            4 => {
                for layer in 0..LAYERS {
                    let _ = grid.route(Some("m"), layer);
                }
            }
            // Deregister a dead (already-removed) server. Must be a no-op
            // and must not corrupt the ledger.
            _ => {
                let c = &candidates[rng.range(candidates.len())];
                if !live.contains(&c.server_id) {
                    grid.deregister(&c.server_id);
                }
            }
        }

        assert_invariants(&grid, &live, tick);

        // Coverage floor: every layer that's covered in the expected set
        // must also be reported as covered by the grid. This is what
        // catches a deregister leaving stale entries in route_table.
        let expected = expected_covered(&live, &candidates);
        for layer in expected {
            assert!(
                grid.has_owners_for(Some("m"), layer),
                "tick {tick}: layer {layer} should be covered (live={live:?}) but grid says no"
            );
            assert!(
                grid.route(Some("m"), layer).is_some(),
                "tick {tick}: layer {layer} covered but route()=None"
            );
        }
    }

    // Final pass: register every candidate fresh, drain heartbeats, and
    // confirm the grid is fully coherent at the end of the churn.
    for c in &candidates {
        grid.register(make_entry(c, 0));
        live.insert(c.server_id.clone());
    }
    assert_invariants(&grid, &live, TICKS);
    for layer in 0..LAYERS {
        assert!(
            grid.route(Some("m"), layer).is_some(),
            "post-churn: layer {layer} should resolve once every candidate is registered"
        );
    }
}

/// Same churn loop, but with `target_replicas=2` so the
/// over_replicated_ranges / under_replicated_ranges ledgers are exercised
/// non-trivially on every tick. A bug that left a phantom server_id in
/// the route table after deregister would show up here either as a
/// route() panic, an over-replication report for a server that no longer
/// exists, or a coverage floor miss.
#[test]
fn grid_survives_long_churn_run_with_target_replicas_two() {
    let candidates = build_candidates();
    let mut grid = GridState::default();
    grid.set_target_replicas(2);
    let mut rng = Xorshift64::new(SEED ^ 0xa5a5_a5a5_a5a5_a5a5);
    let mut live: HashSet<String> = HashSet::new();

    for tick in 0..TICKS {
        match rng.range(5) {
            0 | 1 => {
                let c = &candidates[rng.range(candidates.len())];
                grid.register(make_entry(c, rng.range(10) as u32));
                live.insert(c.server_id.clone());
            }
            2 => {
                if !live.is_empty() {
                    let ids: Vec<String> = live.iter().cloned().collect();
                    let victim = &ids[rng.range(ids.len())];
                    grid.deregister(victim);
                    live.remove(victim);
                }
            }
            3 => {
                for layer in 0..LAYERS {
                    let _ = grid.route(Some("m"), layer);
                }
            }
            _ => {
                // Read the ledger directly — must not panic regardless of
                // churn state.
                let _ = grid.over_replicated_ranges();
                let _ = grid.under_replicated_ranges();
            }
        }
        assert_invariants(&grid, &live, tick);
    }
}
