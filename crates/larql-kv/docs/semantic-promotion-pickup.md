# Semantic promotion — pickup notes (2026-08-05)

Written to be resumable cold. What exists, what it proves, what it
deliberately does **not** prove, and the exact next move with the reason
behind it.

The specification is `docs/specs/semantic-promotion-engine.md`. This file
is the working state around it: the evidence ledger, the scope
corrections, and the ordered queue.

---

## 1. What is built

Committed 2026-08-06 in `c1f24291`, bundled into a compute commit
rather than the five prepared ones. Test counts as of writing: 1 120 in `larql-kv`,
1 433 in `larql-inference`, clippy clean, workspace green.

### `crates/larql-kv/src/engines/semantic_promotion/`

A policy wrapper over an exact decode engine. Owns authority, visibility,
promotion lifecycle and qualification; owns no model math. The caller's
`FfnBackend` threads through all eighteen `KvEngine` methods.

| file | concept |
|---|---|
| `ids.rs` | domain-tagged `u128` identities, string-encoded on the wire |
| `authority.rs` | `SourceAuthority`, stable refs, SHA-256 span digest |
| `record.rs` | `CanonicalFact`, `DerivedClaim`, content digest |
| `qualification.rs` | `CapabilityKey`, `RegimeEvidence`, `ConsumptionMode` |
| `scope.rs` | `RetirementScope` + the gate–claim congruence rule |
| `scoring.rs` | `ScoredObject`, `DivergenceClass`, the I10 window |
| `exclusion.rs` | architecture-derived `AccessPlan`, `ExclusionJournal` |
| `checkpoint.rs` | `BoundaryCheckpoint`, `SuffixVisibility`, offset replay |
| `control.rs` | `SemanticKvControl`, the enforcing lifecycle |

### `crates/larql-kv/src/model_walk/`

A **peer** of `engines/`, not part of one. Dependency runs walk → engine
and never back; a source-scanning gate asserts no file under `engines/`
mentions `model_walk`.

```text
ModelGraph → WalkPlanner → ModelWalkPlan → WalkValidator
           → ValidatedWalkPlan → Observe | Enforce executor → WalkTrace
```

`cost.rs` carries `WalkCost` (nine terms, never collapsed to a scalar),
`RankingPolicy`, `pareto_frontier`, and separate `CostEstimate` /
`CostObservation`.

### `crates/larql-inference/`

- `kv_engine.rs`: `PerLayerKvAccess` (excise / splice / replace /
  read / set-position), `ExcisedRows`, `KvExtent`.
- `kv_row_positions.rs`: `KvRowPositions` — **built and gated, not
  wired**. See §5.

---

## 2. Gate inventory

| series | file | covers |
|---|---|---|
| G0, G8 | `semantic_promotion/tests/parity.rs` | observe-mode bit-parity; FFN call-sequence identity |
| — | `tests/qualification_gate.rs` | EXP-25 matrix as policy, incl. all four refusals |
| — | `tests/position_seam.rs` | logical position independent of resident row count |
| W0–W8 | `model_walk/tests/walk_gates.rs` | planning determinism, EXP-25 verdicts, plan/engine separation |
| B* | `model_walk/tests/enforce_gates.rs` | observe/enforce action parity, exclusion arithmetic, refusals |
| B10 | `tests/heterogeneous_gates.rs` | ragged per-layer extents; byte-identical rollback |
| O0–O8 | `tests/exclusion_oracle.rs` | logits-level excision equivalence + wrong-span control |
| C0–C10 | `tests/cost_gates.rs` | competing paths, residency changes the answer, greedy ≠ oracle |
| R0–R7b | `tests/replay_gates.rs` | checkpoint determinism, hole oracle, windowing limitation |
| — | `tests/composition_gates.rs` | one promotion per walk (provisional, evidence-gated) |

---

## 3. Evidence ledger — points, not bands

Every entry is a measured point. `RegimeEvidence::Point` covers only
itself; `ValidatedBand` requires a sweep and **none has been earned**.

### Canonical facts

```text
payload      Point(4K) PASS 3/3   Point(~49K) PASS 3/3   counterfactual
trajectory   Point(4K) 1/3        Point(~49K) 1/3
```

Independently sufficient for the **payload** in a suffix that never saw
the source, at both distances. All 12 canonical arms peak on the *first
termination token*; original history also fails 2/3, so the stop-state
defect is attributable to neither counterfactuality nor distance.

**Runtime permission: `ExactPayloadLength`, not `FirstTermination`.**

### Derived claims

```text
consumption  Point(4K)   OperandForOperation        both histories
             Point(~49K) AnswerWithoutReapplication both histories
             interval    UNMEASURED — behaviour inverts inside it
```

At ~49K the derived claim passes **payload and trajectory**, where
canonical passes payload only — so at range it is the stronger
replacement. At 4K it is actively dangerous: the operation runs twice.

Distance frontier (`exp25cf_frontier_*`), true-value arm: operand at 4K,
8K, 16K, 32K, 48K; answer at 64K. **Corrupt arm is non-monotone**
(answer at 8K, operand 16–48K, answer at 64K), so consumption is *not* a
function of geometry alone — it depends on the exact record. That is why
`CapabilityKey` carries `record_digest` **and** `materialisation_digest`.

Source-distance sweep at fixed 64K (`exp25cf_srcdist_*`), record and
query pinned at absolute 65 344: answer-mode at source→query 3 016,
7 989, 16 188, 32 047. Fifth point (~49K) re-runs an already-banked
configuration. Combined with ~2 872 tokens behind the query reading as
**operand** at 4K context: **source distance is ruled out** as the
controlling variable.

### Not established

Multi-hop composition, reached-operand binding, general-state retirement,
improved K3 prediction. See spec §7.6 — `composition_gates.rs` pins one
promotion per walk *provisionally*, and its module docs enumerate the four
outcomes and which one licenses widening.

---

## 4. Scope corrections — read before quoting any result

**Phase B is proven on the CPU per-layer executor, not on Gemma's
production attention.** `larql_inference::kv_dispatch::helpers` calls
`attention_step` with no per-layer window, then `clip_kv` with the
*engine-level* window uniformly. It never consults
`arch.is_sliding_window_layer` — that lives in `pipeline_layer.rs`, a
different path. So every layer attends over everything resident, and the
global/sliding split is a promotion-wrapper *policy*, not attention.

```text
proven      bit-exact semantic exclusion + ragged per-layer storage
            under the CPU per-layer executor
pending     those extents corresponding to production architectural
            visibility
```

**Retirement is not counterfactual absence** — and this now has a
measured answer for canonical payload (EXP-25CF), but it remains a
standing caveat on any *new* result that retires a span the suffix
already saw.

**The B1/B6 oracle is materialised, not indexed.** It compares
compact-once against compact/restore-per-step. A true indexed gather
would need a per-layer visibility parameter on `attention_step`, which
deliberately does not exist.

**Sparse position + windowing is incorrect *on the dispatch clip*.**
`clip_kv` still keeps the tail *N physical rows*; across a hole that
retains rows thousands of positions old inside a small window (`r7b`).
`clip_layer_to_logical_window` is the correct primitive and now exists,
but the decode path has not been switched onto it — that switch is
R8-policy.

---

## 5. `KvRowPositions` is wired — R8-storage LANDED

Steps 1–6 of the wiring order below are done and gated; step 7 is the
remaining R8-policy work.

### Wiring order

```text
1. StandardEngine owns KvRowPositions beside each handle          DONE
2. every append records abs_position                              DONE
3. excise returns K/V *and* positions as one value                DONE
4. splice restores both atomically                                DONE
5. BoundaryCheckpoint captures/restores positions in the same
   transaction                                                    DONE
6. clip_layer_to_logical_window(layer, window)                    DONE
7. per-layer policy: sliding → logical-age clip, global → no clip  DONE
```

### What landed

`PerLayerKvAccess` now carries the map and the atomic primitives:

```rust
fn excise_kv_rows(..)      -> Option<ExcisedKvRows>;   // rows + positions
fn splice_kv_rows(.., rows: &ExcisedKvRows) -> bool;
fn replace_layer_kv(.., positions: &[u64])  -> bool;   // checkpoint restore
fn row_positions(&self)    -> Option<&KvRowPositions>;
fn clip_layer_to_logical_window(&mut self, layer, window) -> Option<usize>;
```

`ExcisedKvRows` holds its fields **privately** behind a checked
constructor, which is stronger than the sketch below: the one-position-
per-row invariant is enforced where the value is built, so a journal or
checkpoint carrying mismatched halves is unconstructible rather than
merely wrong. `replace_layer_kv` takes positions as a parameter for the
same reason — a restore that set rows and positions in two calls would
pass through a state where they disagree.

Two refusals were added because the map made them expressible:

- a splice at an offset that would leave positions out of order is
  refused *before* the handle is rebuilt, leaving the cache untouched;
- `set_logical_next_position` refuses to move the next position behind a
  resident row, which is the internally-inconsistent state its own doc
  comment warned about but could not previously detect.

Gated in `tests/row_position_storage.rs` (10 gates), including the R7b
divergence closed end to end: after an offset replay leaves nine rows
spanning four thousand positions, a logical clip at window 4 drops all
eight pre-hole rows where the physical clip kept four of them. On a
contiguous cache the two agree, so wiring step 7 cannot change
unperforated behaviour.

**Still not load-bearing for decode.** The dispatch clip inside
`kv_decode_step_via_dispatch` is untouched; the engine *records* the
physical-tail clip faithfully rather than correcting it. Nothing in the
decode path yet consults logical age.

### R8-policy also landed — as an operation, not a decode change

`window_policy::apply_window_policy(kv, &LayerAttention)` applies the
architecture's own per-layer classification: sliding layers keep the rows
inside the declared window **by logical age**, global layers keep
everything. One call, opposite outcomes per layer class, which is exactly
what a single engine-level `window_size` cannot express.

Gated in `tests/window_policy_gates.rs`:

```text
R8a  gap 4096 > window 4   sliding layer retains 0 of 8 pre-hole rows
R8b  gap 2    < window 4   retains exactly {7}, the one still in range —
                           not 4 (physical) and not 0
R8c  same hole, global     retains all 8 pre-hole rows
R8d  decoded-then-excised == offset-replay, bit-exact in K, V and
     retained positions, at gaps 2/4/6/16
```

Plus two refusals (undeclared window on a sliding arch; layer-count
mismatch), both before any row moves, and a gate that the policy reduces
to an ordinary window on a contiguous cache.

**What this does and does not license.** R8d closes the parity claim for
the *policy applied to a cache*: heterogeneous extents built by two
different physical routes agree bit-exactly. It does **not** make
production Gemma decode logical-age-correct — `kv_decode_step_via_dispatch`
still calls `clip_kv` with the engine-level window on every layer. That
routing change is now qualified but not made, and it is deliberately a
separate step: it alters production decode behaviour, whereas everything
above is additive.

**Atomicity is the requirement, not a nicety.** K/V rows and their
positions must never be independently mutable, or a half-applied
mutation passes shape checks and fails at a later lookup. The primitive
should become

```rust
pub struct ExcisedKvRows { pub kv: ExcisedRows, pub logical_positions: Vec<u64> }
fn excise_kv_rows(..) -> Result<ExcisedKvRows, ..>;
fn splice_kv_rows(.., rows: &ExcisedKvRows) -> Result<(), ..>;
```

rather than two callers updating parallel structures.

**Do not change attention gather first.** If clipping physically removes
out-of-window rows before the next step, the existing dense path is
untouched. Indexed visibility is only needed if rows must be retained
physically while hidden logically.

### R8, split into two milestones

```text
R8-storage   metadata stays congruent through append, excise, splice,
             checkpoint, restore, offset replay          (R8e)   LANDED

R8-policy    sliding layers retain by logical age                (R8a)
             only in-range rows survive a sub-window gap         (R8b)
             global layers keep pre-hole rows across the hole    (R8c)
             decoded-then-excised == offset-replay under mixed
             policy — the gate that upgrades "Gemma-shaped
             storage" to Gemma attention parity                  (R8d)
```

Landing R8-storage first lets the row map integrate without claiming
production parity prematurely.

---

## 6. Experiment queue

```text
1. finish source-distance sweep                          [running]
2. variable-global-context regime:
      fixed source→record, source→query, record→query
      variable unrelated prefix
   — does the same semantic geometry change mode when moved later?
3. prefix-content control at fixed absolute positions
   — positional, or state-induced?
4. RoPE-offset harness (needs restore_for_offset_replay)
   — same tokens, same relative geometry, different absolute positions
5. replicate the 8K/16K corrupt-arm excursion under small protocol
   perturbations before treating it as real
6. EXP-26CF closure matrix: complete / missing-middle / wrong-edge /
   isolated-operand / stepwise, under counterfactual replay
```

Item 2 is next because 1 has already ruled out its own axis.

Item 4 must use the restore-at-offset **then regenerate** protocol.
Restoring and shifting the position without regenerating leaves rows at
phases that disagree with the geometry — silently, because nothing about
the shape looks wrong.

---

## 7. KV-reduction accounting

```text
baseline saving = qualified canonical span replacement
                − replacement-record rows
                − retained recovery authority

derived upside  = 0 by default
                  non-zero only when materialisation digest + operation
                  + placement protocol + point regime all match
                  reusable evidence

online-qualified fresh claims = 0
                  no evidence that verification costs less than
                  retaining or recomputing from source
```

Two workload regimes decide whether derived contributes at all:
recurring authorities (digest recurs → certificate may transfer) versus
fresh authorities (every value new → nothing transfers). Which holds is a
property of the workload, not the model.

**Memory saved today: zero.** Resident rollback retains the journal;
`net_bytes_saved()` is ≤ 0 and a gate asserts it. Reclamation needs
durable checkpoints outside active KV *plus* logical-age-aware clipping.

---

## 8. Deferred, with triggers

| item | trigger |
|---|---|
| B7 general logical→physical map | a cheapest plan needing >1 simultaneous or sequential exclusion, or a later query resolving a source after compaction. EXP-26's multi-hop trigger **failed** |
| multi-hop promotion planning | counterfactual closure matrix showing stepwise uniquely succeeds |
| materialised closure retrieval | same matrix showing complete closure works where stepwise does not |
| Metal / coarse-path exclusion | a need to run the regime on the coarse path; `per_layer_kv_mut` returns `None` there by design |
| `ExpertPrefetchHint` from certificates | Phase H. The *same record* takes both consumption modes at different distances, so the envelope cannot be read off `DerivedClaimActive` — it must come from the selected candidate's certificate, mode and regime included |
