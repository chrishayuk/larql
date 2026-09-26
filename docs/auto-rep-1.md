# AUTO-REP-1 — proposed precision maps, admitted by joint measurement

**Class: RECORD (roadmap note).** Date: 2026-09-26. Status: **planned, not
started.** Sequenced after REPRESENT-CAL-1 (open as PR #548); it adds no work
to that PR, and CAL-1 keeps its allocation policy fixed.

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
