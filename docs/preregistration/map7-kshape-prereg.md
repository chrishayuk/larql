# MAP-7 / K-SHAPE — Pre-registration

**Status:** 📋 v1.0 DRAFT — becomes FROZEN at the commit that lands it.
After that commit, frozen text is never edited; changes land only as
dated **Addendum** sections at the bottom, each stating what changed and
why, before the affected measurement runs.
**Companion:** [`map7-kshape-capture-spec.md`](./map7-kshape-capture-spec.md)
— corpus, identity, storage, and harness contract. This document owns
the science; the capture spec owns the data.
**Contains no experiment outputs.** The commit that lands this document
must contain no measured numbers from either programme.

---

## 0. The firewall

> **MAP-7 asks whether depth is locally predictable. K-SHAPE asks
> whether FFN contribution is either more selectively sparse than gate
> ranking reveals, or decomposable into a small exact innovation plus a
> predictable aggregate tail. Neither programme may use evidence from
> the other to satisfy its own kill gate.**

They share one corpus and one capture (see companion spec) and nothing
else. Each has its own registry record, its own gates, and its own
verdict.

---

## 1. Lineage

This work resumes the MAP / ZoneEngine line, paused by explicit user
decision on 2026-08-02 in favour of the K3/VINDEX3 critical path, and
deliberately unpaused by user decision on 2026-08-28. It is registered
under programme `map`; the slug `map-6` (attention-reroutes-the-
traveller) is an existing deferred experiment and is NOT reused.

Constraining prior results (all pre-existing; cited as inputs, not
re-derived here):

- **ZoneEngine zone map** (`crates/larql-inference/docs/specs/zone-engine.md`
  §3.2): Gemma 3 4B choke points L0/L4/L20/L29/L33. T2 (L4→L20) global
  rank-30 linear map: 83.8% top-1, median cosine 0.99. **T3 (L20→L29):
  global linear falsified at 41.2% top-1 with hidden cosine still
  0.96.** Spec §11 Q4 leaves nonlinear/local T3 refinement open.
  PREDICT contracts are conjunctive `top1_preserving(τ) ∧ bounded_KL(ε)`;
  hidden-state similarity is descriptive, never sufficient evidence.
- **MAP-5/5b/5c**: a single-position oracle donor-residual splice at a
  late-enough layer recovers the donor's exact top-1 (5/5 pairs);
  magnitude-matched random controls never do. T_immediate ≠ T_branch ≠
  T_persistence. The existing patching harness is stateless
  full-recompute and cannot measure persistence.
- **CellRouter (WalkFfn task #22)**: residual-cell routing matches
  gate-KNN accuracy at ~150× cheaper routing. Residual locality
  predicting the row population is ESTABLISHED and is not re-tested.
- **WalkFfn #23 / R4 zero-out**: joint faithfulness under gate-top-K
  needs K≈4096 of 10,240; with routing free, the scattered execution
  path captures only 23–40% of its row reduction; derived execution
  crossover K* ≈ 2944 on that path (to be re-priced, §4.6). R4's
  surviving lane is compiled compact-dense; R4 names the missing
  experiment: a contribution-quality oracle.
- **V1 hash-routing diagnosis** (`docs/diagnoses/v1-hash-routing.md`):
  per-layer KL screens do not compound — individually "safe" per-layer
  thresholds applied jointly produced +5.4 to +7.7 bits/token and
  78–95% argmax drift. **No verdict in either programme may rest on a
  single-layer contract.**
- **ENCODER-R4 (GPTQ)**: calibrate against the candidate path, not the
  reference path. Adopted for the K-SHAPE oracle (§4.2) and for
  K-SHAPE-B training regime 2 (§5.5).

## 2. The organizing hypothesis — marked UNPROVEN

> **Dense in contribution, sparse in innovation:** the model may be
> computationally dense but informationally low-dimensional conditional
> on its current state — at depth scale (T2's rank-30 transition) and
> possibly at FFN-row scale (a predictable aggregate tail plus a small
> exact innovation).

This is an ORGANIZING claim. It carries no evidential weight in any
gate below, and the T2 analogy is not evidence for the FFN-scale half.
Its falsifier is exact: **K-SHAPE-B fails jointly at every tested K
under BOTH training regimes (§5.5).** Until that test runs to a
verdict, this frame must not become load-bearing in any other
programme's reasoning.

---

## 3. Shared protocol elements

### 3.1 Substrate

Gemma 3 4B (`google/gemma-3-4b-it`, locally cached), 34 layers,
d_model 2560, d_ff 10240, freshly extracted dense f16 container with
digests recorded per the capture spec. All fitted/scored arithmetic in
f32. Scoring reference is the canonical dense forward of the SAME
container (self-consistent; the fp32-vs-deployment-path question is out
of scope here and noted as a limitation).

### 3.2 Boundary convention (frozen)

`i_L` := the residual stream entering layer `L` (after all layers
`< L` complete). `i_0` = embedding output; `i_34` = final pre-norm
output. Under this convention:

- **T2 map:** `i_4 → i_20` (layers 4..19).
- **T3 map:** `i_20 → i_29` (layers 20..28, nine layers).

The historical zone-map prose has an off-by-one ambiguity. If the T2
instrument gate (§4.1 of MAP-7) fails, the adjacent conventions
(`i_4→i_19`, `i_5→i_20`) are permitted **as pipeline debugging only**,
and the convention actually used is recorded in an Addendum before any
T3 result is produced.

### 3.3 Faithfulness contract (frozen)

A position is **faithful** under an intervention iff, at that position,
teacher-forced against the canonical context:

- top-1 agreement with the canonical forward, **and**
- `KL(canonical ‖ intervened) ≤ ε` on the full output distribution,
  computed live (no epsilon-clamped truncation; the rank-of-native-top1
  metric is always reported alongside as the saturation-proof check).

Primary `ε = 0.10` nats. Sensitivity reporting at `ε ∈ {0.05, 0.25}`.
These values are frozen pre-data.

### 3.4 Metrics (frozen priority order)

1. rank of native top-1 token; 2. top-1 preservation; 3. KL (nats,
full distribution); 4. top-5 / top-20 containment; 5. NLL delta.
Descriptive only, never verdict-bearing: residual rel-RMS, hidden
cosine.

### 3.5 Statistical discipline

- **Prompt is the replication unit.** All paired statistics aggregate
  per-prompt first. Positions within a prompt are never treated as
  independent.
- Paired comparisons: per-prompt deltas, Wilcoxon signed-rank at
  α = 0.05 AND a bootstrap 95% CI over prompts excluding zero. Both
  required for any "beats" claim.
- Every primary metric is reported per native-top-1-margin tertile
  (tertiles computed from measured margins on the canonical forward,
  never from topic labels). **No pooled headline without the tertile
  table.** A benefit visible only in the high-margin tertile does not
  satisfy any gate.
- Raw per-arm tables are read before any derived verdict line.
- The pre-listed comparisons in §4/§5 are the ONLY confirmatory tests;
  everything else is exploratory and labelled as such.

### 3.6 Blinding and ordering (frozen)

1. Capture bring-up + self-tests (capture spec §9).
2. **MAP-7 T2 instrument gate** (§4.1). Until its result is recorded in
   the registry, no T3 arm and no K-SHAPE verdict cell is run.
3. Then MAP-7 T3 and K-SHAPE proceed independently.

---

## 4. MAP-7 — `map-7-t3-local`

**Question:** Is T3 globally nonlinear but locally predictable? I.e.,
does any proximity-local predictor of `i_29` from `i_20` beat the best
global linear map, where the global map fails (41.2%) despite correct
neighbourhood (cosine 0.96)?

### 4.1 Instrument gate — T2 reproduction (runs first, blinds T3)

Train a global rank-30 map (delta mode, ridge, per §4.3 fitting
protocol) on the train split for `i_4 → i_20`; evaluate top-1
preservation on the dev split, general stratum.

- **PASS:** top-1 ∈ [0.78, 0.90]. The capture/fit pipeline reproduces
  the historical neighbourhood of 83.8% and T3 unblinds.
- **FAIL:** outside that band. Pipeline fault investigation (including
  §3.2 boundary variants). No T3 arm may be fitted or evaluated until a
  pass is recorded in the registry. The historical 83.8% is a
  reference, not a live baseline; all T3 comparisons use the
  re-measured global arms.

T2 is a positive control only. No MAP-7 claim is licensed by T2
results.

### 4.2 Arms (frozen)

All arms predict `i_29` from information available at the `i_20`
boundary; the canonical suffix (layers 29..33 + head) then runs from
the predicted/retrieved `i_29`. Splice floor per §4.5.

**Injection mode (frozen):** verdict-bearing evaluation is at the
**final position of each prompt** (the answer slot), **single-row
splice** — only that position's `i_29` row is replaced; all earlier
positions keep their canonical values, so suffix attention reads a
clean prefix (the MAP-5 construction). One evaluated position per
prompt also makes prompt-as-replication-unit exact. Full-matrix splice
(every position predicted at once, cross-position error compounding)
may be reported as EXPLORATORY only, never in a gate. The same rule
applies to the T2 instrument gate (§4.1): last position, single-row
splice.

| Arm | Predictor | Tests |
|---|---|---|
| G-r | Global linear, rank r ∈ {8, 16, 30, 64, 100, 128}, delta and absolute modes | re-measured global null; whether global linearity was exhausted at rank 30 |
| H1 | Nearest stored train `i_20` → copy its stored `i_29` | stored-destination retrieval |
| H2 | `i_20` + weighted-kNN displacement Δ = i_29 − i_20, k ∈ {1, 4, 16} | locally stable vector field |
| H3 | Cell-local rank-r map (k-means cells over train `i_20`, C ∈ {64, 256}; per-cell delta-mode ridge map, r ∈ {8, 16, 32}) | local operator hypothesis |
| RP | Random-partition local rank-r: identical to H3 with cells assigned by hash, matched cell-size distribution, same C and r | capacity control — locality vs parameters |
| TOK | Retrieval keyed on last-n token IDs (n ∈ {1, 4}), no residual | tokenizer/content confound |
| RND | Magnitude/tangent-matched random donor (MAP-5 control construction) | causal control |

Hyperparameters (PCA basis dim ∈ {32, 64, 128}, ridge λ ∈ log-grid
1e-4..1e1, k, C, r, mode) are selected on the dev split only. Test-split
results are computed once, for the dev-selected configuration per arm
family.

### 4.3 Fitting protocol

Delta-mode regression in a PCA basis fitted on train-split residuals
(eigenbasis, not raw 2560-D distances), ridge-regularised. Identical
machinery for global, H3, and RP arms.

### 4.4 Primary metrics and the wrong-fact gate

Primary metrics per §3.4, plus — **primary, not diagnostic** — the
**wrong-fact injection rate** on the entity-swap stratum:

- retrieval arms (H1/H2/TOK): rate at which the arm's final top-1
  equals the canonical answer of the *retrieved neighbour's* entity
  while differing from the query's canonical top-1;
- fitted arms (H3/G-r/RP): rate at which final top-1 equals the
  canonical answer of any *other* entity in the same template family
  while differing from the query's canonical top-1.

### 4.5 Splice floor (must pass before any arm is scored)

Every arm, including canonical and oracle, runs through the identical
injection harness. The oracle arm (true captured `i_29` spliced) must
be bit-identical to the unmodified forward, MAP-2b style (substitute
inside the real dispatch; `target == real` reduces to exact equality).
If the floor fails, fix the harness; no arm results exist until it
passes.

### 4.6 Decision gate (frozen)

**CONTINUE** (to retrieval-degraded causal splice, then — only then —
the KV persistence seam) iff some proximity-local arm (H1/H2/H3), on
the held-out test split, general stratum:

1. beats the best dev-selected global arm G-r by ≥ **5 percentage
   points** top-1 preservation, paired-significant per §3.5;
2. median KL ≤ **0.8×** the best global arm's (no KL regression);
3. beats its capacity-matched RP counterpart on (1)-style top-1 margin,
   paired-significant (H1/H2 compare against TOK-capacity analogues:
   H1/H2 must beat TOK on the same criteria);
4. beats TOK on criteria (1)–(2);
5. improvement point-estimate positive in the low AND mid margin
   tertiles;
6. wrong-fact injection rate on entity-swap ≤ the best global arm's.

**KILL** otherwise: T3 remains WALK; the residual-graph-as-compute
thesis closes; no `PREDICT::LocalRetrieved` zone strategy is built.

### 4.7 Out of scope for MAP-7 v1

Multi-token decode claims (needs the KV seam — ZoneEngine §4.5 refuses
Standard+PREDICT precisely because skipped layers leave K/V gaps; the
incremental KV intervention seam stays downstream of a CONTINUE
verdict). HNSW. Any ZoneEngine code change. Any architecture other
than Gemma 3 4B.

---

## 5. K-SHAPE — `kshape-a`, `kshape-b`

### 5.1 K-SHAPE-A — why is faithful K so high?

**Question:** Is K≈4096 intrinsic to FFN contribution, or partly an
artefact of ranking rows by gate activation?

**Protocol — live, self-consistent, joint.** A is a set of live corpus
runs, not an analysis of the static capture. All 34 layers are
sparsified simultaneously at the same K. At every layer the oracle
computes the full dense FFN **on the actual (drifted) candidate-path
input**, ranks rows, retains the top K, discards the rest, and
continues. Secondary arm: dense-trace ranking (rows fixed from the
canonical capture), same joint execution. The primary−secondary gap is
reported as drift sensitivity.

**Rankings (both get the full ladder):**

- gate: `score_i = g_i` (reproduces the WalkFfn selector);
- contribution: `q_i(x) = |φ(g_i·x)(u_i·x)| · ‖d_i‖` — deliberately
  undeployable; its job is the bound, not a router.

`K_min` is always stated as **K_min under ranking R**. Per-position
`K_min(R)` := the smallest ladder K such that the position is faithful
(§3.3) at that K and at every larger tested K. Greedy magnitude
ranking is not optimal subset selection; a greedy/refinement bound is
run only if contribution ranking lands within 15% of the economic gate.

**K ladder (frozen):**
`{256, 512, 1024, 1536, 2048, 2560, 2944, 3072, 3584, 4096, 6144, full}`.

**Reporting:** `P(position faithful | K, ranking)` curves and the
median/p90/p95 of per-position `K_min(R)`, stratified by margin
tertile, CellRouter cell, layer band, and corpus stratum. **Never a
single-layer threshold** (V1 discipline).

**Economic gate (frozen).** Before comparison, K* is re-priced: re-run
the R4 paired dense-vs-oracle-par protocol (AC power, rotated paired
repeats, IQR sentinel) at K ∈ {2048, 4096, 6144} on the current tree,
refit `T_sparse/T_dense = a + b·K`, K* = (1−a)/b. Then:

- **Lane A OPEN** iff contribution ranking achieves aggregate
  position-faithful rate ≥ 0.90 on the held-out test split at some
  K ≤ K*.
- **Lane A CLOSED** otherwise: better selection cannot rescue row
  escape on this execution path. (This licenses no claim about lane B
  or about compiled-layout economics — K* prices the measured scattered
  path only.)

**CellRouter economics rider** (measurement, no gate): per layer/cell —
pool-size distribution, within/across-cell support-Jaccard, union
growth vs samples, duplication factor `D = Σ_c |P_c| / d_ff`, and
coverage of contribution-ranked supports by gate-built cell pools.
Interpretation is economic (can stable broad support be compiled
without exploding model size), not existential — cells are already
established.

### 5.2 K-SHAPE-B — is the tail predictable?

**Object.** For chosen K and ranking, per layer:
`Y(x) = Y_K(x) + E_K(x)`, with `E_K(x) = D_tailᵀ a_tail(x)` — varying
tail activations through a fixed down-projection.

### 5.3 Diagnosis gate (runs before any predictor is fitted)

At probe layers (capture spec §6.3) and K ∈ {256, 512}, contribution
ranking, from recomputed intermediates:

- centred singular spectra of the `E_K` matrix AND of the tail
  activation coefficients `a_tail` (separates fixed-geometry
  compression from conditional activation structure);
- global, within-real-cell (CellRouter cells), and two nulls:
  within-random-partition-cell at matched cell-size distribution, and
  a random-K tail (complement of a random K-subset, same |tail|);
- statistics: rank@{90, 95, 99}% energy, participation ratio.

**PROCEED to fitting** iff at ≥ 1 probe layer and K ∈ {256, 512}:
median within-real-cell rank@90% ≤ **0.5×** the matched
random-partition null AND ≤ **64** absolute.
**STOP** otherwise — lane B closes at the diagnosis stage, before any
predictor exists.

### 5.4 Predictor arms (only if §5.3 passes)

| Arm | Predictor of E_K |
|---|---|
| B0 | zero tail (Y_K alone — the R4 baseline) |
| B0m | corpus-mean tail per layer (uninformed floor) |
| B0c | cell-mean tail (cell-conditional constant) |
| B1 | global rank-r linear, r ∈ {8, 16, 32, 64} |
| B2 | token-keyed predictor (last-n IDs, no residual) |
| B3 | residual-kNN tail, k ∈ {1, 4, 16} |
| B4 | cell-local rank-r, C ∈ {64, 256}, r ∈ {8, 16, 32} |
| B5 | random-partition local rank-r (capacity match to B4) |
| B6 | matched random control |

All learned corrections emit in the top-**128** left singular
directions of `D_tail` (frozen basis dim). Same PCA/ridge/dev-selection
discipline as §4.3. Same split, tertile, and pairing rules as §3.5.

**Verdict protocol:** `Y'_K = Y_K + Ê_K` applied at ALL participating
layers jointly, scored in predictive units per §3.3/§3.4.
Reconstruction RMSE and spectra are mechanism diagnostics only.

### 5.5 Two training regimes before death (frozen)

1. **Canonical-trace training:** pairs `(x_dense, E_K)` from the
   capture. Evaluate jointly.
2. If regime 1 fails jointly: **sequential candidate-path training**
   (GPTQ-style): process layers in order; layer ℓ's training pairs are
   captured from forwards where layers < ℓ already apply their trained
   correctors + exact-K; fit; continue.

**Lane B (and the §2 organizing hypothesis) is falsified only if both
regimes fail at every tested K** — i.e. no (K, arm, regime) cell
reaches aggregate position-faithful rate ≥ 0.90 on the held-out test
split.

### 5.6 Pricing (independent of lane A's K*)

Kernel-level, ledger-honest (transient traffic included):

- per-token: exact-K gather cost (measured µs/row on current tree, not
  the historical 0.39), correction matvec cost, predictor bytes read —
  which is ONE cell's map (~O(100) KB), not the artifact;
- artifact storage: all cells × all layers, reported alongside the
  compiled-row-block duplication factor D as one combined
  storage-for-bandwidth account.

Any eventual positive claim is stated explicitly as a
**storage-for-bandwidth trade**, never as free sparsity.

---

## 6. Explicitly not in scope (both programmes)

No HNSW work. No new sparse kernels. No KV intervention seam until
MAP-7 records CONTINUE. No dynamic scattered-gather execution (R4's
mechanism kill stands). No graph abstraction / `GRAPH` zone kind — a
first-class graph is considered only if a local method wins AND
exhibits stable cells, reusable edges, or multi-hop structure. No
architecture beyond Gemma 3 4B. No throughput claims from any oracle
result: a deployable claim requires the selector, the kernel, and the
fallback priced.

## 7. Registry plan

- `map-7-t3-local` under programme `map`, cross-linked to `map-5` and
  the ZoneEngine spec; `map` programme next_action updated to record
  the deliberate unpause.
- `kshape-a`, `kshape-b` as new experiments with supersession/relation
  links to the R4 zero-out and CellRouter records (relation:
  "reopens R4's surviving lane with the quality oracle R4 named", not
  supersession of R4's verdict, which stands).
- The T2 instrument-gate result is recorded on `map-7-t3-local` before
  any T3 or K-SHAPE verdict cell runs (§3.6).

## 8. Numeric choices frozen by the author — reviewed at commit

These were chosen by the authoring session, not derived from data, and
the freezing commit is the sign-off: faithfulness ε = 0.10 (sens.
{0.05, 0.25}); T2 gate band [0.78, 0.90]; MAP-7 margins (+5pp top-1,
0.8× KL); lane-A bar 0.90 at K ≤ K*; B diagnosis gate (0.5× null,
≤ 64 absolute); D_tail basis dim 128; hyperparameter grids in §4.2/§5.4;
the K ladder; corpus sizes and probe layers (capture spec §5–§6).
Amending any of these after data exists requires an Addendum with
justification that does not reference observed verdict-cell results.
