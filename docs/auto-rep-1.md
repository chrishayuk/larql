# AUTO-REP-1 — proposed precision maps, admitted by joint measurement

**Class: RECORD (roadmap note) with a frozen implementation contract for
AUTO-REP-1a.** Date: 2026-09-26. Status: **AUTO-REP-1a (SEARCH mechanics)
implemented to the contract below; see its implementation record.** Sequenced after REPRESENT-CAL-1
(CAL-1.1 #548 and CAL-1.2 #563 merged); CAL-1 keeps its allocation policy
fixed.

## What is missing

REPRESENT already has the parts an allocator needs:

- `represent/constraint.rs`: each quality criterion is its own ceiling, and
  the evidence floors are separate from them;
- `represent/execution_cost.rs`: time saved per byte, derived from
  observations;
- `represent/byte_ledger.rs`: byte accounting;
- `represent/search_evidence.rs`: search evidence;
- `represent/promotion.rs`: the only path from measurement to authority.

What it lacks is the step that reads that evidence and emits a
`PrecisionMap`. Today a person writes the map.

## Rule: proposals are not authority

Per-component measurements can prune and rank maps, and they can predict a
map's quality and cost. They cannot admit a map.
[GLM-5 funnel](glm5-flash-funnel.md) measured that per-projection marginals
do not compose: the `strict` map built from the binding per-projection widths
still missed the output gate. **A proposed map has no authority until its
compiled, combined candidate passes a whole-model measurement and
promotion.**

```text
SEARCH     local evidence + execution cost → proposed maps (top-K, diverse)
VALIDATE   proposed map → compile actual bytes → joint measurement → promotion
```

If a proposal fails validation, that falsifies the prediction; the gates are
not relaxed to fit it. The search records a cut and runs again.

**Cuts are search knowledge, not quality authority.** A failed map shows only
that this whole assignment failed. It does not show which of its choices
caused the failure. So cuts come in three classes:

| Class | Excludes | Earned by |
|---|---|---|
| `ExactNoGood` | the one assignment that was measured and failed | that failed joint measurement |
| `StructuralCut` | assignments that cannot execute | codec/kernel/format impossibility, no measurement needed |
| `EvidenceCut` | a broader region, e.g. `L7=NVFP4 ∧ L8=NVFP4` | a separate, qualified experiment that isolates that interaction |

A failed map may never be widened into an `EvidenceCut` automatically.
Doing that turns an observation into a causal claim that no experiment
made.

## Solver shape

Build a small native solver in `represent/` rather than a dependency on an
external CP-SAT stack. The problem is finite-domain. The variables are
groups of tensors, split by role, projection and layer, each with a few
representation choices. Hard constraints prune most of the space before
quality enters, for example:

- no kernel for the codec on the target backend;
- role exclusions;
- byte or residency ceilings;
- known quality-ceiling violations;
- dominated choices;
- recorded cuts (see the three classes above).

Start with:

1. unary domain filtering;
2. additive resource bounds;
3. branch-and-bound;
4. top-K proposals;
5. `ExactNoGood` and `StructuralCut` sets.

Add pairwise propagation only if the search size demands it. Put the solver
behind a proposal trait, so that greedy, beam, MILP or an external CP-SAT
process can later be compared on the same problem and evidence schema.

Record each proposal with:

- the solver revision;
- the problem identity;
- the identities of the input evidence;
- the constraints and cuts used;
- the objective bound;
- whether the search completed or timed out;
- the proposal's rank.

A `.represent` lockfile can then name the search that produced its map.

The solver's output must pass the precision-map check from
REPRESENT-MAP-CHECK-1 (`PrecisionMap::check_against`, landing separately):
every exception must decide at least one eligible tensor, and a catch-all
exception may only come last. A solver that emits a dead rule has a bug,
and this check refuses it.

## Considered and deferred

These points were raised while comparing this plan with NVIDIA Model
Optimizer's `auto_quantize`:

- **Shared scales across fused operands.** Not needed now:
  `Nvfp4Segment` carries a tensor scale per segment, so fused QKV and
  gate/up do not require a shared global scale. Reopen this if a kernel
  requires one.
- **Activation-scale headroom calibration.** No target yet: the NVFP4 GEMV
  path takes f32 activations, and the int8 activation path uses per-block
  runtime scales. Neither stores a calibrated activation scale.
- **Execution-format signature.** Only as an extension of `CodecIdentity`,
  and only if lowering needs fields it cannot currently establish. Fusion
  compatibility depends on execution identity, never on the encoder recipe:
  GPTQ and nearest-rounded packs in the same format must stay fusible.
- **Per-layer KV-cache representation.** After the KV provider work
  settles. A KV error re-enters attention at every later position, so
  sensitivity measured in one layer is at most a search heuristic.
  Long-context continuation evidence is authoritative.

## AUTO-REP-1a — implementation contract (frozen 2026-09-26)

AUTO-REP-1a builds the SEARCH half only: a solver that reads a tensor
surface and a price table and proposes precision maps. It makes **no
quality claim**. VALIDATE (compile, joint measurement, promotion) and the
choice of a quality prior are later rungs.

### Why the quality side is empty

SENSITIVITY-1 falsified four cheap per-component quality scores (1A, 1B-a,
1B', 1C-suffix-replay) against Q-BANK. The best, 1B', still ranked
protecting `down_proj` second of eight while Q-BANK found that protection
worth less than nothing. There is therefore no validated per-group quality
prior to optimise against. 1a ships the mechanics with a pluggable ordering
prior that must declare its `SearchEvidence` class, and ships no prior.

### Existing authorities it builds on

`represent/state/` already holds the search stack. 1a reuses, and does not
duplicate:

- `TensorSurface` / `SurfaceTensor`: the canonical, identified tensor set.
- `PrecisionMap`, `Exception`, `PrecisionMap::check_against`.
- `RepresentationState::resolve` and `RepresentationStateId`: two maps that
  resolve identically are one state.
- `LayoutAdmission` and `SurfaceFootprint` (`CompiledBytes`): analytic
  `LogicalBytes` per tensor, without compiling.
- `Protections` / `ProtectRule`: the only exception form `compile_inner`
  compiles today.
- `SearchEvidence`: the evidence class a prior must declare.

### Problem

- **Base map.** Supplies the default encoding and the eligible roles. A base
  map that already has exceptions is refused: in 1a the solver owns every
  exception.
- **Groups (variables).** Eligible tensors (role named by the base map)
  grouped by `(projection_of, layer_of)`, both present. That is the finest
  unit one `Exception { projection, layers: (l, l) }` addresses exactly, so
  a group's decision can always be written as a map. The grouping follows
  the map grammar, which matches by tensor name across objects.
- **Fixed tensors.** An eligible tensor with no projection or no layer
  cannot be addressed without also matching other tensors. It takes the
  default and is reported as fixed. Non-eligible tensors are source.
- **Domain.** `{Compile(default), Source}` per group. This is exactly what
  `Protections` expresses, so every proposal is compilable by the existing
  compiler. Other encodings wait for a mixed-codec compile path.
- **Objective.** Minimise total `LogicalBytes` from `SurfaceFootprint`,
  including fixed and source tensors. A group's price is the sum of its
  tensors' prices; a tensor the layout refuses presents source bytes at
  either choice, as `try_logical_bytes` already prices it.
- **Constraints.** An optional byte ceiling (additive). Pins: a group forced
  to one choice by the caller.

### Cuts

- `StructuralCut { group, choice, reason }` excludes a choice for a group
  without any measurement. 1a derives one automatically: a group whose every
  tensor the layout refuses at the default has no real `Compile` choice.
- `ExactNoGood { state: RepresentationStateId }` excludes exactly the state a
  failed joint measurement measured, and nothing else. It is keyed by state,
  not by assignment, so any map that resolves to that state is excluded.
- `EvidenceCut` is not implemented in 1a, and the solver provides no way to
  widen an `ExactNoGood` into one.

### Solver

- A generic core over groups with integer costs: unary domain filtering,
  additive lower bound (partial cost plus each unassigned group's cheapest
  choice), and depth-first branch-and-bound that keeps the K best leaves.
- Leaves are checked against `ExactNoGood` by resolving the synthesised map
  to its state. Proposals are distinct states.
- A deterministic node budget stands in for a timeout. The outcome is
  `Complete` or `NodeLimit { explored }`, and the record carries the best
  lower bound reached, so a truncated search says how far it is from proven.
- Ties break deterministically: bytes, then the prior's score if one is
  supplied, then the assignment's canonical order.
- A `Proposer` trait wraps the solver so greedy, beam, MILP or an external
  CP-SAT process can later be compared on the same problem and records.

### Prior

`trait OrderingPrior { evidence() -> SearchEvidence; score(group, choice) }`.
`Unusable` is refused. A prior only breaks ties between equal-byte
assignments; it never changes the byte order and is never priced against a
budget. No prior ships in 1a.

### Output

Each proposal carries its `PrecisionMap`, its state id, its bytes, its rank,
and a record with:

- `solver_revision` (`auto-rep-bnb/v1`);
- `problem_id`: a digest of model identity, surface identity, base map,
  layout policy, price table, domain, pins and ceiling;
- the ids of the cuts and of the prior's evidence, if any;
- the outcome and the lower bound at termination.

Synthesis writes one source exception per protected group, merging
contiguous layers of one projection into one range, in deterministic order.
It also produces the equivalent `Protections`.

### Gates

1. **Brute-force agreement.** On synthetic problems small enough to
   enumerate, the solver's top-K equals exhaustive enumeration under the
   same ceiling, pins, cuts and tie-break. The reference is independent of
   the solver.
2. **Bytes agree with the footprint.** On a container fixture, each
   proposal's bytes equal `SurfaceFootprint::try_logical_bytes` of the state
   its map resolves to.
3. **Map check.** Every proposal passes `check_against` on the surface. A
   synthesis defect that emits an exception deciding no tensor is refused.
4. **ExactNoGood is exact.** Adding the rank-1 state removes that state and
   no other: the next run's rank 1 is the previous rank 2, and states that
   differ from the cut state in one group remain.
5. **StructuralCut.** An excluded choice never appears, and its reason is
   recorded.
6. **Truncation and determinism.** A node limit reports `NodeLimit` with a
   lower bound no greater than the true optimum. Identical inputs produce
   identical records.
7. **Prior discipline.** An `Unusable` prior is refused. A prior reorders
   only equal-byte proposals.
8. **Refusals.** A base map with exceptions is refused. Unaddressable
   tensors are fixed and reported.
9. **Compilable.** Each proposal's `Protections`, passed through
   `PrecisionMap::from_policy`, resolves to the same state as the proposal's
   map.

Not claimed by 1a: that any proposal is admissible, that a cheaper proposal
is better, or that any prior predicts quality.

## AUTO-REP-1a — implementation record

The code is in `represent/state/propose/`: `solver.rs` holds the generic
core, and `mod.rs` holds the REPRESENT adapter, cuts, prior, records and
synthesis. `SurfaceFootprint` gains `presented`/`admits`, which price one
tensor with the same rule `try_logical_bytes` sums.

- **Gate 1:** the core equals exhaustive enumeration on 3,000 random
  problems, with frequent cost and score ties, key collisions and
  exclusions. A node-limited search's lower bound never exceeds the true
  optimum on another 3,000.
- **Gates 2–9:** covered on the glimmer container. Draft and target share
  tensor names, so one `(projection, layer)` group spans both objects.
- **Mutations:** each was run against the tests, and each was caught.
  - Core: `>=` in the bound, no key replacement, a lost sibling bound, and
    a dropped base cost.
  - Adapter: ignored no-goods, a skipped map check, Compile priced as
    Source, ignored structural cuts, no range merging, and fixed tensors
    dropped from the total.

**Deviation found by a mutation.** With Compile priced as Source, every
group cost the same at both choices. The first bound compared cost only
and explored every equal-cost subtree to settle tie-breaks, so the test
ran for over an hour instead of failing. A real model with many equal-size
groups can present the same shape. The bound now covers the whole ranking
tuple: an equal-cost leaf scores at least the partial score plus each
remaining variable's lowest score, and an equal-score leaf's assignment is
no smaller than its fixed choices with every open variable at its lowest.
Near-equal score sums explore rather than prune, because summation order
differs. A 60-variable all-ties test settles within 100,000 nodes and
fails under the cost-only bound. Brute-force agreement still holds.
