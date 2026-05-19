# ADR-0014 — Hot-Shard Load-Rate Replication

**Status:** Accepted — shipped 2026-05-15. **Amended 2026-05-16**
with two-threshold hysteresis (see §"Cool-down" below).
**Depends on:** ADR-0004 (self-assembling grid), ADR-0011
(rebalancer + replication)
**Implementation:** `crates/larql-router/src/grid/hot_shard.rs`,
`crates/larql-router/src/grid/replication.rs::effective_target_for`,
`crates/larql-router/src/tasks/rebalancer/hot_shard.rs`

---

## Context

`--target-replicas N` (ADR-0011 §Replica management) maintains a
**static** replica count per `(model_id, layer_start, layer_end)`
range. The rebalancer pulls spares from the available pool when the
count drops below `N` and drops the least-loaded replica when it
rises above. Static targets handle the **availability** dimension
(survive a server crash) but not **load**:

- A shard might be at target replica count and serving requests at
  100 req/sec per replica.
- The same shard the next day might be at target and serving 5
  req/sec per replica.

The static target can't distinguish these — but the second case has
4× the headroom of the first. Operators today have to manually bump
`--target-replicas` (and the corresponding count in the available
pool) to add capacity to a hot shard, then manually bump it back
down. That's slow and error-prone, and it scales replication for
*every* shard when only some are hot.

What we want: when a single shard saturates, add **one** replica of
**that shard** only. When it cools down, drop the extra replica back.
The rest of the grid stays at static `--target-replicas`.

---

## Decision

Introduce **per-range elevation**: a transient bump to a single
range's effective replica target driven by observed request rate.

### State: the elevation set

```rust
// in grid/mod.rs::GridState
elevated_ranges: HashSet<(String, u32, u32)>,
```

A range `(model_id, layer_start, layer_end)` is **elevated** when
its observed `req_per_sec` (max across replicas) has recently
crossed `--hot-shard-rps THRESHOLD`. The rebalancer is the only
writer; replication-side code is the only reader.

### Effective target = static target + bump

```rust
pub fn effective_target_for(&self, model_id: &str, layer_start: u32, layer_end: u32) -> u32 {
    let bump = if self.elevated_ranges.contains(&(model_id.to_owned(), layer_start, layer_end)) {
        1
    } else {
        0
    };
    self.target_replicas + bump
}
```

Both `under_replicated_ranges()` and `over_replicated_ranges()` use
`effective_target_for` instead of reading `target_replicas`
directly. That's the **entire** integration surface: a range becomes
elevated → `effective_target_for` returns N+1 → the next
under-replication tick sees the range as deficient → pulls a spare
from the available pool. When the range cools down (see below) and
gets demoted, the over-replication tick sees the surplus and drops
the least-loaded replica.

### Bump is +1, not configurable

Hard-coded. A range either is or isn't hot — a saturated shard needs
*one more* replica, not three. If a single +1 doesn't cool the load,
the next tick will still see it as hot and the next replica pull
will fire (subject to available-pool capacity). A range stays
elevated as long as the max req/sec stays above the threshold;
multiple ticks can stack additional replicas the same way the
under-replication mechanism would for any deficit, but each tick
only sends **one** assignment per range (ADR-0011
§Replica management).

### Detection: max across replicas

`hot_layer_ranges(threshold)` walks all serving servers and groups
by `(model_id, layer_start, layer_end)`, keeping the **max
`req_per_sec`** observed across that range's replicas:

```rust
let mut max_rate: HashMap<(String, u32, u32), f32> = HashMap::new();
for e in self.servers.values() {
    let key = (e.model_id.clone(), e.layer_start, e.layer_end);
    let cur = max_rate.entry(key).or_insert(0.0);
    if e.req_per_sec > *cur {
        *cur = e.req_per_sec;
    }
}
```

**Why max and not mean?** With perfect routing the per-replica rates
converge — any replica crossing the threshold means the shard's
per-replica load is at the ceiling. Mean would under-react when
routing is imperfect (one replica handling 100 req/sec while two
others handle 1 req/sec each gives mean = 34 but the shard is
clearly hot). Max reacts immediately to whichever replica saw the
spike.

### NaN-safe threshold check

`hot_layer_ranges` uses `!(threshold > 0.0)` to detect the disabled
case rather than `threshold <= 0.0`. NaN trips both > and <= to
false, so a NaN threshold should *disable* the check (treating NaN
as a config error), not match-everything. The `!(threshold > 0.0)`
form returns true for NaN, zero, and negatives — all three
correctly disable.

### Cool-down: two-threshold hysteresis (amended 2026-05-16)

The original spec used a **single threshold** — a slice was elevated
whenever its rate exceeded `T` and demoted whenever it did not.
This left a real oscillation risk at the boundary: traffic
hovering at exactly `T ± noise` would mark/demote/mark/demote on
each rebalancer tick, churning a replica pull-then-drop every 30 s
for no net load change.

The amended scheme uses **two thresholds**:

```
        rate
         ▲
         │  elevated
   T  ───┼───────────────────  ← elevate when rate > T  (rising edge)
         │                       (and not yet elevated)
         │  middle band        ← no-op: previously-elevated stays
0.8·T ───┼───────────────────    elevated; previously-non stays non
         │                       (default ratio 0.8 → 20% headroom)
         │  cool / not elevated← demote when rate < 0.8·T (falling)
         │                       (only if previously elevated)
         └─────────────────────▶ time
```

Implementation: `check_hot_shards` queries `hot_layer_ranges` **twice**
per tick — once at the elevation threshold `T`, once at the demote
threshold `T × demote_ratio`. The two-set difference gives the
elevation and demotion candidates respectively.

```rust
let hot     = hot_layer_ranges(T);
let still_hot_for_demote = hot_layer_ranges(T * demote_ratio);
let elevated = elevated_ranges_snapshot();

// rising-edge: in hot, not yet elevated
for slice in hot.difference(&elevated)               { mark_elevated(...) }
// falling-edge: was elevated, now below demote threshold
for slice in elevated.difference(&still_hot_for_demote) { demote_elevated(...) }
```

**Default ratio:** `0.8` — 20% headroom below the elevation
threshold before demotion. Trade-off:
- **High ratio (≈ 1.0)** — single-threshold behaviour, prone to
  oscillation but tracks real load closely.
- **Low ratio (≈ 0.5)** — very stable, but a slice that briefly
  hits the threshold stays elevated even after sustained cool-down.

0.8 splits the difference and matches conventional load-shedding
hysteresis (TCP's slow-start uses similar headroom).

**CLI flag:** `--hot-shard-demote-ratio <FRAC>` (default `0.8`).
Values outside `(0.0, 1.0]` clamp to the default. Setting to `1.0`
disables hysteresis entirely (reverts to single-threshold).

### Original cool-down (pre-amendment, retained for reference)

The rebalancer's hot-shard tick runs **before** the replication
ticks each interval:

```rust
async fn check_hot_shards(state: &Arc<RwLock<GridState>>, threshold: f32) {
    let mut guard = state.write().await;
    let hot: HashSet<_> = guard.hot_layer_ranges(threshold).into_iter().collect();
    let elevated: HashSet<_> = guard.elevated_ranges_snapshot().into_iter().collect();

    // Hot but not elevated → mark elevated (will pull spare on under-rep tick).
    for range in hot.difference(&elevated) { guard.mark_elevated(...); }
    // Elevated but no longer hot → demote (will drop surplus on over-rep tick).
    for range in elevated.difference(&hot) { guard.demote_elevated(...); }
}
```

The set-difference structure makes elevation/demotion **idempotent**
and **monotone**: a range that's hot for 10 consecutive ticks gets
marked once on the first tick and stays marked; the under-rep tick
acts once (one replica per tick anyway). When the range cools, it's
demoted once; the over-rep tick sees the surplus and drops one
replica.

### Tick ordering

```
rebalancer_task loop:
  1. evict_stale_heartbeats       (free any dead servers first)
  2. check_hot_shards             (update elevation set)
  3. check_under_replication      (pull spares for under-rep + newly-elevated)
  4. check_over_replication       (drop surplus for over-rep + newly-cooled)
  5. check_imbalance              (per-layer latency rebalance)
```

Hot-shard runs before the replication ticks so the new effective
targets are in effect before they're consulted. Putting it after
would mean elevation/demotion always lag one tick behind the
condition that triggered them.

---

## Alternatives Considered

### Direct replication trigger (skip the elevation set)

`check_hot_shards` could send `AssignMsg` directly instead of
marking the range elevated. Rejected because:
- Couples the rebalancer's hot-shard detection to the specifics of
  Mode B pool selection. Today, `try_replicate_from_available`
  picks the spare; coupling them would duplicate that logic.
- Loses the symmetric cool-down path. With the elevation set, the
  same over-replication tick that handles "operator dropped
  `--target-replicas` from 3 to 2" also handles "range cooled and
  was demoted." Without the set, cool-down would need its own
  bespoke replica-drop logic.
- Makes "what is the effective target for this range right now?" an
  un-answerable question. The elevation set is one source of truth.

### Multi-step elevation (+2, +3 for sustained heat)

Hard-coding `+1` means one heavy ramp can only earn one extra
replica per tick. A more aggressive scheme could track sustained
heat and bump effective target by `+2` after N consecutive elevated
ticks. Rejected for now because:
- Adds state (counter per range) and complexity (decay logic).
- The static `--target-replicas N` already handles "the shard is
  always this hot, plan for it." Hot-shard is for *transient*
  spikes.
- One additional replica typically halves per-replica load. If
  +1 isn't enough, +2 likely isn't either; the operator should bump
  the static target.

### Mean req/sec instead of max

Rejected — see Detection above.

### Per-replica rate instead of shard rate

Same per-replica rate scales with replica count. If the threshold
is "200 req/sec per replica" then a shard at 1000 req/sec across 5
replicas (200 each) wouldn't trigger; a shard at 1000 req/sec across
4 replicas (250 each) would. That's the right behavior — we care
about per-replica saturation, not absolute throughput. The current
implementation uses per-replica rate (max across replicas), which is
exactly this.

---

## Consequences

### Positive

- Hot shards get one extra replica **without operator intervention**
  and within one rebalancer tick of crossing the threshold.
- Cool-down is automatic when load subsides — no manual demote.
- The mechanism uses the existing under/over-replication machinery
  end-to-end; no separate replica-pull path.
- `effective_target_for` is one read away from anywhere in the
  codebase, so future code can ask "do I have enough replicas
  *right now*?" without reasoning about elevation directly.

### Negative

- **Oscillation risk** at a threshold boundary. A shard hovering
  around `--hot-shard-rps` could elevate / demote / elevate /
  demote on consecutive ticks, each pulling and dropping a spare
  ~30 s apart. Mitigations available if observed: hysteresis (mark
  hot above T, demote only below `0.8 × T`); minimum-stay timer
  (range can't be demoted within X ticks of being elevated).
  Neither is implemented today; the simpler scheme is shipped and
  the operator can choose `--hot-shard-rps` well clear of the
  steady-state per-replica rate.
- **One-tick reaction time.** A request spike that lasts less than
  one `--rebalance-interval` (default 30 s) won't be reacted to.
  This is acceptable for the use case (sustained per-shard load
  imbalance, not micro-bursts).
- **Available-pool dependency.** If the available pool is empty,
  elevation marks the range but `try_replicate_from_available`
  finds no spare and the elevation stays marked indefinitely (until
  the rate cools). That's the correct behavior — there's nothing
  the rebalancer can do without spares — but operators need to know
  that hot-shard elevation **requires** spare capacity in the Mode B
  pool to be effective.

---

## Implementation pointers

| File | Role |
|---|---|
| `crates/larql-router/src/grid/mod.rs::GridState::elevated_ranges` | The set itself (private field) |
| `crates/larql-router/src/grid/hot_shard.rs` | `hot_layer_ranges` + `mark_elevated` + `demote_elevated` + `elevated_ranges_snapshot` |
| `crates/larql-router/src/grid/replication.rs::effective_target_for` | The cascade `target_replicas + bump`; consumed by under/over_replicated_ranges |
| `crates/larql-router/src/tasks/rebalancer/hot_shard.rs::check_hot_shards` | The rebalancer tick — set-difference elevate/demote |
| `crates/larql-router/src/tasks/rebalancer/mod.rs` | Loop ordering: hot_shard → under_rep → over_rep |
| CLI flag | `--hot-shard-rps <FRAC>` on `larql-router` |

### Test coverage

- `grid/hot_shard.rs::tests` — 5 tests, 100% line coverage:
  threshold disable cases, max-across-replicas detection,
  elevated→under/over interaction, idempotent mark/demote, snapshot
  sort.
- `tasks/rebalancer/hot_shard.rs::tests` — 4 tests including a
  full hot→cool round trip exercising the elevation set + spare
  pull + surplus drop.

Per-file line coverage as of 2026-05-16:

| File | Line coverage |
|---|---|
| `grid/hot_shard.rs` | 100.00% |
| `tasks/rebalancer/hot_shard.rs` | 90.99% |
