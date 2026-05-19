# ADR-0016 — Router Module Organization

**Status:** Accepted — landed 2026-05-16. The shape captured here is
prescriptive for future work in `crates/larql-router/`.
**Depends on:** ADR-0004 (grid), ADR-0011 (rebalancer), ADR-0013
(routing comparator), ADR-0014 (hot-shard).

---

## Context

Two router source files had grown past the point where a single
maintainer could navigate them comfortably:

- `src/grid.rs` — 2113 lines. `ServerEntry` + `AvailableEntry` +
  `GridState` (with ~30 methods spanning state mutation, routing,
  replication, hot-shard, status, gRPC service impl) + ~600 lines of
  inline tests. The file did not consistently fail to compile or
  test in isolation; it was instead a "find anything by Cmd-F" file.
- `src/rebalancer.rs` — 861 lines. `RebalancerConfig` + the tick
  loop + five `check_*` async helpers (hot-shard, under-rep,
  over-rep, eviction, imbalance) + `send_unassign` + their tests.

Neither file's tests were failing and neither was on a critical
hot path; the split was driven by readability and future change
isolation, not by a bug or a performance regression.

The risk of splitting badly is real:

- Submodules can lose access to private fields of their parent
  struct if visibility is too tight, leading to a cascade of
  `pub(super)` or `pub(crate)` annotations that widen the
  intentional encapsulation.
- Test helpers can drift: each submodule's `#[cfg(test)] mod tests`
  block grows its own copy of a `make_entry(...)` constructor and
  then they fall out of sync when `ServerEntry` gains a field.
- Renames of files break ADR / spec references and require
  workspace-wide updates.

---

## Decision

### Folder shape per concern

```
src/
├── lib.rs              # module declarations
├── main.rs             # CLI entry, admin dispatch, server lifecycle
├── http.rs             # axum handlers
├── dispatch.rs         # multi-layer fan-out
├── shards.rs           # static --shards parser
├── admin.rs            # admin client + formatters
├── cli_helpers.rs      # small helpers
├── grid/
│   ├── mod.rs          # ServerEntry, AvailableEntry, GridState core + Default
│   ├── routing.rs      # route() / route_all() + three-tier comparator
│   ├── replication.rs  # under/over-rep, gap-fill, AssignMsg dispatch
│   ├── hot_shard.rs    # req/sec saturation + elevation set
│   ├── status.rs       # coverage_gaps, all_shard_urls, status_response
│   ├── service.rs      # gRPC GridService impl + admin RPCs
│   └── testing.rs      # #[cfg(test)] pub(crate) helpers
└── tasks/
    ├── mod.rs          # module declarations
    ├── rebalancer/
    │   ├── mod.rs      # spawn + tick loop
    │   ├── config.rs   # RebalancerConfig
    │   ├── hot_shard.rs    # elevation set updates
    │   ├── replication.rs  # under/over-rep ticks
    │   ├── eviction.rs     # stale-heartbeat eviction
    │   └── imbalance.rs    # per-layer latency tracker
    └── rtt_probe.rs    # opt-in active RTT probe loop
```

Two folders, mirroring the two large monoliths that were split:

- **`grid/`** — owns the in-memory `GridState` and all the methods
  that read or mutate it. Submodules are organised by the *concern*
  the method serves (routing, replication, hot-shard, status,
  service).
- **`tasks/`** — owns long-lived background tasks spawned at startup.
  Both submodules follow the same shape: a public
  `spawn(state, config)` entry point that registers a tokio task
  running for the process lifetime, plus an internal tick body.

### Visibility rule: rely on child-module access to parent privates

Rust's rule is that a **submodule sees its parent's private items**
(including private struct fields). This is the rule we lean on:

```rust
// grid/mod.rs
pub struct GridState {
    servers: HashMap<String, ServerEntry>,     // private field
    route_table: HashMap<(String, u32), Vec<String>>,
    elevated_ranges: HashSet<(String, u32, u32)>,
    ...
}

// grid/routing.rs — submodule of grid
impl GridState {
    pub fn route(&self, ...) -> Option<String> {
        let ids = self.route_table.get(...);    // ← reaches private field
        ...
    }
}
```

No `pub(super)` annotations on `GridState`'s fields; no
`pub(in crate::grid)`. The submodules see the privates because they
are children. This means the public surface of `GridState` is
**identical before and after the split** — split or unsplit, the
internal-vs-external distinction is the same.

Where a private *method* on `GridState` needs to be called from a
sibling submodule (e.g. `rebuild_route_table` called from
`grid/mod.rs::register` but defined in `grid/routing.rs`), the method
is marked `pub(super) fn`. Sibling submodules don't see each other
directly — they see *through* the parent. `pub(super)` makes the
parent visibility just enough to support that.

### `testing.rs` — shared `#[cfg(test)]` helpers, scoped via visibility

Five test modules across `grid/` all need to construct a
default-populated `ServerEntry`. The struct has 13 fields including
clock and HashMap types; an inline literal is 13 lines and grows
every time the struct evolves. Five copies of that literal would
drift.

Solution:

```rust
// grid/testing.rs
#![cfg(test)]

pub(crate) fn entry(server_id: &str, listen_url: &str, model_id: &str,
                    layer_start: u32, layer_end: u32) -> ServerEntry {
    ServerEntry { ... }
}

// grid/mod.rs
#[cfg(test)]
pub(crate) mod testing;
```

- `#![cfg(test)]` at the top of `testing.rs` excludes it from
  release builds entirely.
- `pub(crate)` on `mod testing;` makes the module reachable from
  sibling crates within `larql-router` (notably the rebalancer
  tests under `tasks/rebalancer/`).
- `pub(crate) fn entry(...)` is the only public item in the file.

Both `grid/*` test modules and `tasks/rebalancer/*` test modules use
`crate::grid::testing::entry`. One copy of the struct literal;
adding a `ServerEntry` field means one update site in `testing.rs`.

### No re-exports of submodule items at the parent level

When `service.rs` was carved out of `grid.rs`, the initial split
included `pub use service::GridServiceImpl;` at the top of
`grid/mod.rs` so existing call-sites (`main.rs`, the integration
tests) wouldn't need import changes. We **removed** this re-export
and updated the call-sites to use `larql_router::grid::service::GridServiceImpl`.

The reasoning:

- Re-exports make the public surface of `grid::` ambiguous — is
  `grid::GridServiceImpl` the canonical path or is `grid::service::GridServiceImpl`?
- One short rename is cheaper than carrying the re-export forever as
  a compatibility shim for a path that was never public before the
  split anyway.
- Future submodules that want to re-export at the parent level
  should justify why the path is part of the deliberate public
  surface, not just a typing-convenience.

### When to split a file

Heuristic, not a hard rule: **a file that has reached ~1000 lines
and contains multiple independent concerns is a split candidate.**
Both `grid.rs` (2113 lines, 6 concerns) and `rebalancer.rs` (861
lines, 6 concerns) qualified. A single-concern 1500-line file would
not — e.g. `http.rs` at 373 lines or `dispatch.rs` at 298 lines stay
as single files because each one does one thing.

### Test coverage policy maintained across the split

Per-file coverage floor (90% per `crates/larql-router/coverage-policy.json`)
applies to the new files individually. Pre-split, the old `grid.rs`
sat at 95.19% (the gRPC handler's lower coverage was diluted by the
state-machine methods' higher coverage). Post-split, the gRPC
handler stands alone in `grid/service.rs` and was below the floor at
79.89% — fixed by adding 4 targeted integration tests in
`tests/test_grid_service.rs` to bring it to 88.59%, with a debt
baseline at 88% (per the
[per-file 90% coverage floor](../adr/0012-grid-benchmarking.md)
policy: raise the baseline toward 90 over time, never ratchet down).

---

## Alternatives Considered

### Keep `grid.rs` as one file

Status quo. Rejected — the file was hard to navigate and PR diffs
on it were always larger than the touched concern; reviewers were
paging in unrelated context.

### Split by visibility (public API in mod.rs, private impl in a `private.rs`)

A common Rust pattern. Rejected because public/private isn't the
axis along which the file's concerns differ. Routing logic and
state-mutation logic are both `pub fn`; splitting them into a
public-only file and a private-only file wouldn't reduce the
"what's where?" problem.

### One module per method (extreme micro-split)

Each `pub fn` in its own `.rs` file. Rejected — would multiply
boilerplate (every file needs `impl GridState { fn foo(&self) {} }`)
and bury related state in different files.

### Trait-per-concern with separate impl blocks

`trait Routing { fn route(...); fn route_all(...); }` etc. Rejected
because the concerns don't fit a trait abstraction — they're not
substitutable behavior (you wouldn't swap out routing for a different
impl). Concerns share private state and call each other (`register`
calls `rebuild_route_table`); a trait split would force either
public methods or shadow private trait methods. The folder split
captures the same separation with less ceremony.

### Cross-folder shared `testing.rs` at the crate root

Considered. Rejected because the only consumers today are
`grid::*` and `tasks::rebalancer::*`, both of which work fine with
the existing `crate::grid::testing` path. Hoisting to
`crate::test_helpers` would require updating both sets of imports
and the wider-scoped visibility doesn't earn anything until a third
consumer appears.

---

## Consequences

### Positive

- Each split file owns one concern. A change to routing logic
  doesn't show up in a diff alongside the gRPC handler.
- `GridState`'s public API didn't change. External consumers
  (benches, examples, downstream crates) import the same paths.
- Test helpers are shared via `grid::testing::entry` so the
  `ServerEntry` constructor lives in one place.
- The pattern is recursive: `tasks/rebalancer/` further splits the
  rebalancer the same way `grid/` splits the state machine. New
  background tasks (e.g. a future federation task) would land as a
  sibling under `tasks/`.

### Negative

- The directory tree is deeper. `cargo doc` output for the crate
  has more modules. Acceptable cost.
- `pub use service::GridServiceImpl` would have been a 1-line shim
  to preserve old import paths during transition; we removed it,
  which meant a one-time update of `main.rs` and two integration
  tests. Worth it to keep the public surface unambiguous.
- New contributors need to be told the visibility rule (child
  modules see parent privates). This is what the ADR is for.

### Neutral

- File count went 9 → 19 in `src/`. Source line count is roughly
  unchanged (~3000); the split is a *reshape*, not a rewrite.

---

## Implementation pointers

| File | Role |
|---|---|
| `crates/larql-router/src/grid/mod.rs` | `GridState`, `ServerEntry`, `AvailableEntry`, module-level docs explaining the folder shape |
| `crates/larql-router/src/grid/testing.rs` | The shared `entry()` helper |
| `crates/larql-router/src/tasks/mod.rs` | The two-line declaration of background tasks |
| `crates/larql-router/README.md` § Source layout | Reader-facing tree diagram |

### Verification

- 132 lib + 38 integration tests pass after the split.
- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`
  are clean.
- Coverage policy: 18/19 files above the 90% per-file floor; 1 debt
  baseline (`grid/service.rs` at 88%) tracked toward 90%.
- Total line coverage 92.81% (was 91.69% pre-split — the increase
  is from the four new integration tests added to lift
  `grid/service.rs`).
