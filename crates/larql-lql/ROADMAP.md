# Roadmap — larql-lql

For shipped work, see [CHANGELOG.md](CHANGELOG.md).

## Current state (verified 2026-08-04)

**Grammar.** `ast.rs` declares 24 statement forms, all parsed:

| Group | Statements |
|---|---|
| Query | `SELECT`, `DESCRIBE`, `EXPLAIN`, `INFER`, `WALK`, `TRACE` |
| Mutation | `INSERT`, `DELETE`, `UPDATE`, `MERGE`, `REBALANCE` |
| Lifecycle | `EXTRACT`, `COMPILE`, `DIFF`, `USE`, `STATS`, `COMPACT MAJOR` |
| Patch | `BEGIN PATCH`, `APPLY PATCH`, `REMOVE PATCH` |
| Introspection | `SHOW RELATIONS`, `SHOW LAYERS`, `SHOW FEATURES`, `SHOW ENTITIES` |
| Composition | `PIPE` |

**INSERT modes.** `MODE KNN` (residual retrieval override, validated at 25K
edges) and `MODE COMPOSE` (FFN-overlay, ~5–10 facts/layer). `ALPHA` and `MODE`
clauses are accepted on `INSERT`; `ALPHA` only affects `MODE COMPOSE`.

**COMPILE targets.** `COMPILE INTO VINDEX` bakes compose patches into canonical
weight files and persists KNN entries as `knn_store.bin` — so default KNN
inserts are packaged as retrieval overlays, **not** materialized into FFN
features. `COMPILE INTO MODEL` applies MEMIT (opt-in via
`LARQL_MEMIT_ENABLE=1`).

**Routing clause.** `INFER … ROUTE VERIFY [FALLBACK] [TOPK n] [EXIT]` is a
first-class clause (lexer → ast → parser → executor), landed 2026-06-07 as the
LQL surface over the FR1/FR2 fleet-routing work. Default is `Legacy`, which is
byte-identical to the pre-clause path.

**Tests.** 1024 tests, all passing (`cargo test -p larql-lql`).

**Coverage.** Enforced: 90% per file, 93% total. Seven files carry ratcheting
debt baselines (52–81%) — all of them need either a real HuggingFace model
directory or an interactive REPL to exercise; see the `policy_note` in
`coverage-policy.json` for why each is genuinely hard rather than merely
untested.

---

## Open defects

- **P1 — four `embed.row()` sites bypass the crate's own safe helper.** Raised
  2026-05-28; **all four confirmed still open 2026-08-04** at
  `executor/query/walk.rs:38`, `executor/query/explain.rs:31`,
  `executor/mutation/insert/plan.rs:122`, and one in COMPACT MAJOR's MEMIT
  path, removed when Gate B, B6 made that statement refuse. They skip
  `average_embed_rows`, so an OOV or unbounded token id panics instead of being
  skipped. Route through `average_embed_rows` / a bounds-checked helper.
  (`relations.rs:309` looks like a fifth site but is already guarded by an
  explicit `id < embed.shape()[0]` check — it does not need changing.)

---

## P0: Review cleanup — correctness and persistence

### DELETE / UPDATE relation predicates
**Status**: Done
**Files**: `src/executor/mutation/delete.rs`, `src/executor/mutation/update.rs`,
`src/executor/tests.rs`
Parser accepts `WHERE relation = ...`, and the executor now evaluates it
through `RelationClassifier`. Vindexes without relation labels fail loudly
instead of treating relation-only mutations as broad matches.

### COMPILE path semantics
**Status**: Done
**Files**: `src/executor/lifecycle/compile/mod.rs`, `src/executor/tests.rs`
`COMPILE "<path>" INTO ...` now loads the supplied vindex in an isolated
session and compiles that source from disk. Use `COMPILE CURRENT` when active
session patches or unsaved overlays should be included.

### Balanced COMPOSE patch persistence
**Status**: Done
**Files**: `src/executor/mutation/insert/mod.rs`,
`src/executor/mutation/rebalance.rs`, `src/executor/tests.rs`
Pending compose patch ops refresh gate/up/down payloads from the overlay after
balancing and rebalance updates, so `SAVE PATCH` persists the latest vectors.

### Parser trailing input
**Status**: Done
**Files**: `src/parser/mod.rs`, `src/parser/tests.rs`, `src/repl.rs`
Single-statement parsing now requires EOF after the optional semicolon / pipe
parse. Batch splitting remains in the REPL path.

### Examples, docs, and benches drift
**Status**: Done
**Files**: `README.md`, `docs/spec.md`, `../../docs/lql-guide.md`,
`examples/*.rs`, `benches/*.rs`
Docs and benchmarks reflect KNN default, compose-only `ALPHA`, single-layer
COMPOSE behavior, and the compile benchmark now includes a down-override bake.

---

## P0: KNN journal vs committed weights

### Make retrieval overlays visible in query output
**Status**: Planned  
**Files**: `src/executor/query/infer.rs`, `src/executor/query/infer_trace.rs`,
`src/executor/trace.rs`, `src/executor/tests.rs`  
Default `INSERT MODE KNN` is a retrieval overlay over the model result. `INFER`
and `EXPLAIN INFER` should tag when `apply_knn_override` fires, include the
override layer/cosine, and show the model's unoverridden top-1. `TRACE` should
keep the residual DAG pure and add a separate `pending_retrieval_override` /
`uncommitted_overrides` section after the layer table. This makes current
semantics honest: the trace did not miss an internal edit; the edit is outside
the weights.

### Add explicit compile mode semantics
**Status**: Design  
**Files**: `src/parser/lifecycle.rs`, `src/ast.rs`,
`src/executor/lifecycle/compile/into_vindex.rs`, `src/executor/tests.rs`  
Target SQL surface:
```sql
COMPILE CURRENT INTO VINDEX "out.vindex";
COMPILE CURRENT INTO VINDEX "out.vindex" SNAPSHOT;
```
Default `COMPILE` should eventually mean commit/materialize all pending edits.
`SNAPSHOT` preserves the current behavior: bake compose overlays, then carry
`knn_store.bin` forward. Until materialization ships, keep current behavior but
surface it explicitly in output/docs as a snapshot/package operation.

### Materialize KNN entries into mechanistic edits
**Status**: Planned  
**Files**: `src/executor/lifecycle/compile/into_vindex.rs`,
`src/executor/mutation/insert/compose.rs`, `src/executor/compact.rs`,
`src/executor/tests.rs`  
Lower each `KnnEntry` into a durable FFN edit before writing the compiled
vindex, then drop or mark the sidecar entries as committed. First strategy:
compose lowering from `(entity, relation, target, layer, residual_key)` into a
free slot at the retrieval layer, reusing the canonical/decoy prompt machinery
from `INSERT MODE COMPOSE` where possible. Later strategies can route through
MEMIT or a hybrid chooser.

Acceptance criteria:
- Weak equivalence: `INFER(session_with_knn, q)` equals
  `INFER(materialized_vindex, q)` for canonical affected prompts.
- Trace conversion: pre-materialization trace reports a pending retrieval
  override; post-materialization trace shows residual/FFN contribution.
- Generalization: materialized vindex affects nearby unstored prompts without
  depending on a `knn_store.bin` lookup.

---

## P0: Phase 3 — Expert routing grammar

### `USE "..." WALK ONLY WITH EXPERTS REMOTE { ... }` grammar
**Status**: Not started  
**Files**: `src/parser/lifecycle.rs`, `src/executor/lifecycle/use_cmd.rs`  
New clause on the `USE` statement that attaches a remote expert map before
any `WALK` or `INFER` call. Syntax:
```sql
USE "gemma4-26b.vindex" WALK ONLY WITH EXPERTS REMOTE {
  "0-31":  "http://host1:8080",
  "32-63": "http://host2:8080"
};
```
Parser extension: parse the JSON-like expert map into `HashMap<ExpertRange, Url>`.
Executor: store the map on the `Session`; wire into `RemoteExpertBackend` in
larql-inference before the next `WALK` / `INFER`.

### `RESHARD EXPERTS { ... }` statement
**Status**: Not started  
**Files**: `src/parser/mutation.rs` (or new `src/parser/expert.rs`), `src/executor/`  
Allows live redistribution of experts across servers without a `USE` restart.
Useful for the demo "kill one shard, rewire on the fly" proof shot:
```sql
RESHARD EXPERTS { "0-63": "http://new-host:8080" };
```
Updates the `Session`'s expert map in place; subsequent WALK/INFER calls use
the new routing immediately.

---

## P1: INSERT quality

### Refinement rounds — `WITH refine_rounds = N`
**Status**: TODO in `mutation/insert/compose.rs`  
The `INSERT INTO EDGES … WITH refine_rounds = N` clause is parsed and stored
but the executor ignores `N` and always runs the cliff-breaker single-round
refine. Implement the loop: after the initial slot install, run up to `N`
additional refine passes that re-capture residuals under the live install
and re-orthogonalise, lifting `self_scores` when the first pass undershoots.
Validated manually in Python (`compile_facts.py refine(rounds=2)` lifts 5/5);
needs to be wired into the Rust executor path.

---

## P1: Coverage — close the remaining gaps

### `infer_trace.rs` rendering branches
**Status**: Planned
**Files**: `src/executor/query/infer_trace.rs`, `src/executor/tests.rs`
Currently 63% line coverage. The dense Backend::Weight short-circuit and
KNN-override formatter are now covered, but the per-layer rendering
loop (`render_trace_layer`) only fires when `walk_trace_from_residuals`
produces non-empty hits. The synthetic random-init fixture rarely
triggers gate-KNN matches above the threshold, so most rendering
branches are unreachable from the existing fixture. Build a richer
fixture (rich-tokenizer + has_model_weights + RICH_GATE_MAG-scale
gate vectors) so the trace produces real hits, or expose a
seed-residuals knob on the test harness.

### `lifecycle/extract.rs` happy path
**Status**: Planned
**Files**: `src/executor/lifecycle/extract.rs`, `src/executor/tests.rs`
51% line coverage; the rest is `InferenceModel::load` + `build_vindex`
+ auto-load. Needs a tiny on-disk fake-model fixture (config.json +
safetensors) cheap enough to ship in tests. ~80–100 lines for a 2-layer
4-hidden synthetic.

### `repl.rs` interactive paths
**Status**: Planned
**Files**: `src/repl.rs`, `tests/`
62% line coverage; the rest is `run_repl` and `run_repl_basic` which
need stdin simulation. A `rexpect` or `expectrl`-style harness in
`tests/` would unblock these. Lower priority since the helpers
(`split_statements`, `is_complete_statement`, `run_batch`,
`run_statement`) are fully covered.
