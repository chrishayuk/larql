# The LARQL Physical Optimizer — evidence-constrained physical-plan search

**Stages 1 (a-d), 2, 3 and 3b landed on `main` in #409; stage 4 is on
`worktree-optimizer-mcp-facade`, based on `origin/main` at `474a0392`.**
Everything below stage 4 is design.

The abstraction is not "quantization search". It is **evidence-
constrained physical-plan search**, and it is the query optimiser the
*model-is-a-database* thesis was always going to need:

```text
logical model  →  candidate physical plans  →  cost model
                                                   ↓
                                         evidence constraints
                                                   ↓
                                          physical model plan
```

The eventual request is one sentence — *run K3 as fast as possible on
this MacBook while satisfying `balanced-v1`* — and the search runs over
representation × residency × storage × execution strategy to produce a
measured, evidence-backed plan.

MCTS is not the architecture. It is one `SearchPolicy` among Greedy,
BestFirst, Beam and PUCT, and it is **not yet earned** (§8).

---

## 1. Epistemic responsibilities

```text
VINDEX3                 what is this model / state, exactly?
REPRESENT / RESIDENCY   what transformations are legal?
Evidence                what have we actually established?
Optimizer               which states are worth exploring?
AI                      what scientific question should we ask next?
```

Five different questions, five different authorities, and the value of
the design is that they do not leak into each other. The agent is not
handed an optimiser and told to improvise; it operates a scientific
instrument whose definitions of identity, admissibility, measurement and
evidence exist independently of it.

```text
> Diagnostic chooses what to measure next; authority decides what is
> admissible.  Extend one level: the AI chooses what question to ask
> next; the optimiser and the evidence system decide what is true.
```

REPRESENT does not know MCTS exists. Neither does RESIDENCY, and
certainly not VINDEX3. The search engine asks only
`state.actions()`, `state.apply(action)`, `state.cost()`,
`evidence.for(state)`.

---

## 2. Three objects, not one node

The mistake to avoid is a fat `MapNode` mixing immutable truth with
mutable search statistics. Split:

```text
PhysicalState {          // immutable truth, hashed
    id, representation, residency, execution,
    physical_cost, structural_properties,
}

Evidence {               // keyed BY state, never part of it
    state_id, diagnostic, authority, benchmarks, provenance,
}

SearchNode {             // the policy's business alone
    state_id, visits, value, prior, expanded_actions,
}
```

Today `PhysicalState` has exactly one component —
[`RepresentationState`](../crates/larql-vindex/src/format/vindex3/represent/state/identity.rs) —
and residency and execution join at stage 6. That widening is why the
identity contract below is written to compose rather than to be the
whole story.

---

## 3. What already existed

Grounded, all under
`crates/larql-vindex/src/format/vindex3/represent/`:

| Concern | Where |
|---|---|
| The map as policy, not transcript | `map.rs` — `PrecisionMap`, `Exception`, `resolve`, `conforms` |
| Behavioural contract + margins | `constraint.rs` — `Margin`, `ConstraintVector`, `binding()`, `admissible()` |
| The frozen contract | `quality.rs:764` — `kimi-logit-balanced-v1`; `QualityGate.id` — *"changing a threshold means a NEW id"* |
| Measurement adequacy | `measurement.rs` — `MeasurementStatus`, `EvidenceScale::{Diagnostic,Authority}` |
| How a search may *use* a statistic | `search_evidence.rs` — the four-rung ladder, `SearchCalibrationRegistry` |
| Promotion that cannot scalarise a proxy | `decision.rs` — `decide_promotion`, `PromotionDecision::Ambiguous` |
| Physical accounting | `physical.rs`, `byte_ledger.rs` |
| Causal reach of an action | `participation.rs` — `ParticipationDeclaration` |
| Container identity, three levels | `compiler.rs` — `SourceIdentity` (index, semantic graph, payload segments) |
| A worked end-to-end chain | [kimi-precision-topology.md](kimi-precision-topology.md) |

`search_evidence.rs` and `decision.rs` are what make an MCP surface safe
to expose at all: `OrderingProxy` licenses order and never magnitude, and
disagreeing proxies are refused rather than scalarised. A persuasive
agent cannot talk past either.

**Absent before this branch** (grep, not recollection): no
`MeasurementKey`, no map identity of any kind, no DAG/frontier/policy,
no MCP anywhere in the workspace. `SearchCandidate` is a *round*
abstraction with no memory across rounds.

---

## 4. Stage 1a — the identity contract (IMPLEMENTED)

`represent/state/{surface,resolved,identity}.rs`, 18 tests, 100% line
coverage on all three files.

### The normative invariant

> **If two maps cause VINDEX3 to present exactly the same representation
> decision for every tensor of the same model surface, they are the same
> search state.**

```text
RepresentationStateId = H( model_identity,
                           tensor_surface_identity,
                           effective_decision_vector )
```

```text
IN   model identity      SourceIdentity — index, semantic graph, payload
                         segments. The same map on two models is two
                         states, and the graph level catches identical
                         payloads under different semantics.
IN   tensor surface      every (object, tensor, role, shape), digested.
IN   effective decisions what is presented, per tensor.

OUT  rule syntax         ordering that changes no decision, shadowed
                         exceptions, redundant defaults, the map's
                         `name`, the recipe that built it.
OUT  evidence            a measurement DESCRIBES a state.
OUT  search history      different paths converge to one node.
OUT  the contract        a representation is the same representation
                         under another contract. Contract, bank and
                         execution belong to the MEASUREMENT key.
```

Folding `balanced-v1` into the digest would identify an *experimental
context* rather than a representation, and the same bytes measured under
a second contract would arrive as an unrelated state with no shared
physical accounting.

### Two things the spec did not anticipate

**Effective, not declared.** A map can say *compile this* and the
storage layout refuse it — NVFP4 holds 2-D matrices whose `k` is a
multiple of 16, and `represent/mod.rs` carries anything else verbatim
whatever the policy said. A state built on the *declared* decision would
believe it had compiled tensors still sitting at source precision and
would price bytes it never saved. So `ResolvedEncoding` has three
variants and the digest reads `effective()`:

```text
map says          layout says       presented       counts as
compile NVFP4     can hold it       NVFP4           Compiled
compile NVFP4     k not a multiple  source bytes    LayoutRefused
source precision  —                 source bytes    Source
```

`LayoutRefused` and `Source` are one state and two facts. Identity
collapses them; the decision vector keeps them apart, because the action
generator needs the difference — unprotecting is a legal move, and
un-refusing a layout is not a move at all.

`LayoutAdmission` is a trait, and an implementation holding no rule for
an encoding answers *not refused*. Refusal takes positive evidence from
whoever owns the layout: a refusal invented here would delete a tensor
from the action space forever, whereas an over-admission surfaces as a
resolved-decision mismatch the first time anything is compiled.

**Role is in the surface.** 4.05 B Gated DeltaNet weights once
classified `unknown` because `_proj_qkv.` is not `_proj.`. No byte on
disk moves when that is fixed, but the set of maps that would compile the
tensor does — a different search problem, so a different state.

**Tensor identity is `(object, tensor)`**, not a sorted name. That is
what `plan_roles::PlanRoles` is keyed by and what
`CompilationLedger` seals against; `weight` occurs in almost every
object, so ordering by name alone would let incidental file order reach
the digest. Aliased objects (a tied embedding and output head) stay
**two** surface entries: a map resolves per `(object, tensor)` and
REPRESENT compiles one pack per object, so collapsing them would claim an
agreement the compiler does not enforce. That the bytes are shared is
already recorded once, by the model identity's per-segment hashes.

### The tests

Six from the spec, plus the alias case, plus five the implementation
earned:

```text
same map + same model                        → same id
different recipes, identical decisions       → same id
shadowed exception added                     → same id
the map's `name` changed                     → same id
surface enumeration order reversed           → same id
exception ordering that changes a decision   → different id
same map, different model (graph; segment)   → different id
tensor reshaped, every decision unchanged    → different id
tensor added, survivors unchanged            → different id
role reclassified, every decision unchanged  → different id
aliased objects                              → two entries, independently decided
duplicate (object, tensor)                   → refused, not silently deduped
layout refusal                               → same id as protection, distinct fact
an oracle with no rule                       → does not refuse
```

**Verified they can fail.** Three mutations of the implementation, each
killing exactly its own tests and nothing else:

| Mutation | Killed |
|---|---|
| surface identity dropped from the digest | `a_reshaped_tensor…`, `a_role_reclassification…` (2) |
| model identity dropped from the digest | `the_same_map_on_a_different_model…` (1) |
| `LayoutRefused` no longer collapses to source | `a_layout_refusal_presents_source…` (1) |

The canonical form is versioned so a persisted DAG that outlives a
change to it is recognisably stale rather than silently colliding. It is
now `represent-state-id/v2`: §4i removed a false split and that changed
which containers are the same state.

---

## 4b. Stage 1b — the state graph (IMPLEMENTED)

`represent/state/{realization,transition,graph}.rs`, 12 tests, 100% line
coverage on all six state files.

### Physical identity is not search-context identity

Stage 1a collapses a protection and a layout refusal into one
`RepresentationStateId`. That is right for evidence and, left alone,
would have been a 1b information-loss bug — the two are not
*action*-equivalent:

```text
                 physical equivalence
                         │
                RepresentationStateId
                   ↑                ↑
         realization A        realization B
         X = Source           X = LayoutRefused
                   │                │
         "unprotect X" is    nothing compiles X;
         a legal move        the layout refuses it
```

So the graph carries two keys:

```text
RepresentationStateId   what is PRESENTED   → evidence, MeasurementKey
RealizationId           what is DECIDED     → the action generator
```

A node is one physical state holding a *set* of realizations. **Evidence
may deduplicate more aggressively than search may.** `ResolvedEncoding`
gained `fact()`, and the decision vector gained `canonical_full()`
alongside `canonical()` — the realization digest reads the refining
form, the physical digest the collapsed one.

### Earning the word DAG

Acyclicity is a **theorem under a declared policy**, not a property of
the domain, so the policy is a value on the graph rather than a comment:

```text
StrictlyImprovingPhysical   every edge strictly reduces LogicalBytes
                            ⇒ a cycle would need a strictly decreasing
                              u64 sequence returning to its start
                            ⇒ no cycles. This is a DAG.
Unconstrained               any edge; a general graph, caller owns
                            termination
```

The strict policy is what rung 5 already enforces — N1 pruned `−E26 + H`
at +1.39 GB and `−K25 + M23` at +2,091,136 B for being physically worse,
and Ruling 1 lists physical dominance among the three usable
pre-measurement prunes. It is nonetheless **a policy and not a law**, and
the programme's own roadmap says when it breaks: once residency joins the
state and the objective is measured tok/s, a move that *adds* logical
bytes while freeing unified memory for resident experts is exactly what
the search must be able to make. The type is therefore
`RepresentationStateGraph`, the policy is recorded and serialised, and a
graph built under one policy is recognisably not a graph built under the
other.

One consequence worth naming: a cost-neutral edge is refused too. From a
layout-refused realization, "unprotect" changes the facts and no bytes at
all — physical dominance prunes it anyway, and admitting it would put a
zero-weight edge into the structure the strict decrease is what keeps
acyclic.

### Edges own how you got there

```text
Transition { parent, child, child_realization, action, physical_delta, provenance }
```

`physical_delta` is **computed** from two footprints, never supplied —
R5-F5 read a footprint column as a saving and overstated an expert revert
by 3.39×, so `LogicalBytes` is a newtype that keeps footprint, per-token
read and delta apart. Edge identity is
`(parent, child, child_realization, action)`, with the action's removed
and added lists sorted, so `+q +k` and `+k +q` are one move while a
different *destination realization* is a different move. Re-discovering
an edge under provenance already recorded is a no-op; a genuinely
different discoverer is kept.

Round number, candidate rank, diagnostic reading and an agent's rationale
all live in `Provenance.note` as **text**, precisely so that no
comparator can compute on them — rung 5's rule that the action lists are
recorded for reproduction *and for no other purpose*.

One graph holds one model and one surface: a map's physical prize is a
property of the model it resolves against, and mixing them would hold
costs that cannot be compared.

### The tests, and that they can fail

The six that were asked for, plus five refusal guards. Four mutations,
each killing exactly its own tests:

| Mutation | Killed |
|---|---|
| realization digest reads the *collapsed* form | `one_physical_state_keeps_both_realizations`, `…serialization…` (2) |
| `observe()` always appends provenance | `rediscovering_an_edge…` (1) |
| policy admits a cost-neutral edge (`<=` for `<`) | `a_transition_that_does_not_improve_physically…` (1) |
| edge identity drops the action | `different_recipes_with_one_effective_representation…` (1) |

---

## 4c. Stage 1c — measurement identity (IMPLEMENTED)

`represent/state/{evidence_bank,instrument,key}.rs`, 14 tests, 100% line
coverage on all nine state files.

```text
MeasurementKey { state_id, bank, scale, instrument }
```

R5-F3 registered the shape; the correction that came with it is why it
is not keyed on the state alone — that would forbid the diagnostic →
authority escalation the ladder depends on.

### `QualityBank` was the wrong home, and the reason matters

The plan was to lift bank identity into `QualityBank`. It does not
belong there: `QualityBank` is *"one bank of measurements over a fixed
token sequence"* — the **observation** (measured KL, routing evidence,
margin distributions). The programme's "evidence bank" is the **corpus**
those were taken over — 256 teacher-forced sequences, a directory with a
`manifest.json`. One word, two things, and only the second identifies an
experiment. Before this stage the corpus had no type at all.

Its identity today is a `sha256` of `manifest.json` computed inline in
the Q2a harness and written to the report as `bank_manifest_sha256`.
Right instinct, **not sufficient** — because the harness *slices* the
corpus. `LARQL_Q2A_SEQUENCES` takes 32 of 256, so a 32-sequence and a
256-sequence reading share a manifest digest while being different
experiments. That slice is the difference between a diagnostic and an
authority run, and a key built on the manifest hash alone would have
called them repeats.

So `EvidenceBankId` digests schema, manifest digest, **the ordered
samples actually used**, and positions-per-sample — and excludes
location, timestamp and description. `EvidenceBank::positions()` is
computed, never multiplied by hand: `LARQL_Q2A_SEQUENCES=256` is
*sequences*, and 256 × 32 = 8192 *positions* is the confusion that once
nearly inverted an acceptance check.

### Instrument semantics, not a version string

A version string fails in both directions — it moves on a refactor that
changes nothing observable, and stays put when someone changes the
truncation. So `InstrumentSemanticsId` digests declared meaning: metric,
truncation, aggregation, token selection, procedure. `implementation_note`
is excluded, so a refactor cannot split a state's evidence.

Truncation is the case already paid for: `min_covered_mass` exists
because a KL over a truncation covering a third of the mass is a
different measurement from one covering all of it — top-128 saw 0.307 of
a first position, top-2048 saw 0.729.

**The limit, stated rather than papered over**: a declaration can lie.
Change the runner's truncation without changing what it reports and the
id does not move. The mitigation is construction — build the declaration
from the constants the runner uses — and this module can make that
possible, not mandatory.

### What is deliberately absent

No PASS/FAIL, no promotion status, no contract. The key says *what
experiment was performed against what state*, never what the result
meant. Classification stays downstream in `Margin`/`ConstraintVector` and
`decide_promotion`, so a later contract reinterprets observations
already held instead of making them disappear.

`MeasurementRegistry` stores readings only. Re-recording an identical
reading is a no-op — a control witness reproducing is what a replayed
round should do — while a *different* reading under the same key is
refused: that says the experiment is not reproducible, and quietly
keeping either would hide it.

### The case that joins 1a, 1b and 1c

```text
RealizationId differs, RepresentationStateId same
    action search  → distinguish
    measurement    → collapse, reuse the observation
```

A protected tensor and a layout-refused one present identical bytes, so
a measurement of one *is* a measurement of the other, while the moves
available from each still differ.

Five mutations, each killing exactly its own tests: key ignores scale →
2 red; bank id counts samples instead of naming them → 1; instrument id
ignores truncation → 2; bank id includes its storage location → 1;
`record` overwrites a contradiction → 1.

---

## 4d. Stage 1d — the replay gate (IMPLEMENTED)

`represent/state/{semantics,snapshot}.rs`, 9 tests, 100% on `semantics.rs`
and 99.4% on `snapshot.rs`.

```text
STORED — facts and configuration
    schema, objective, gate, tail-support policy, search semantics
    the state graph      (which states exist, how they were reached)
    the measurements     (what was observed of them)

NEVER STORED — conclusions
    admissible / refused      chosen candidate     candidate rank
    binding constraint        the frontier         "best map"
    promotion decision        an agent's recommendation
```

> **Delete every derived conclusion, deserialise the factual state, run
> the deterministic optimiser, and recover the same conclusion.**

The frontier is recomputed on every call rather than stored: a persisted
frontier is a second authority that can drift from the graph and
measurements it came from. Caching is a later optimisation; the gate
passes without one.

### The replay, on the real numbers

The snapshot is serialised, the in-memory object thrown away, and every
conclusion below derived from the reloaded facts:

```text
map   logical bytes     auth kl p99   re-derived      recorded
P     13,684,764,800    3.3532e-03    admitted        K25 survives
T1    13,682,673,664    3.6480e-03    REFUSED (kl)    kl 3.648e-3 > 3.500e-3
S2    13,602,484,352    4.0563e-03    REFUSED         refused
S1    13,600,393,216    —             MISS            never measured
```

`binding()` re-derives `KlP99` on the admitted parent — the fact the
exchange rung exists because of. S1's absence re-derives as a
*measurement miss*, which is a fact about the record rather than a
failure. Only `kl_p99`, `min_covered_mass` and the byte figures are
recorded values; the other criteria are fixtures placed well inside
their limits so that what the replay turns on is the recorded evidence.

### Semantics have identity too

Six months from now `decide_promotion` legitimately changes, an old
snapshot replays, and the answer differs while every stored measurement
is intact. **That is not corruption — the decision procedure changed.**
So `SearchSemanticsId` digests five normative version identities
(candidate generation, pre-measurement pruning, evidence interpretation,
promotion rule, physical accounting), never source hashes, and separates:

```text
observation replay          same facts, CURRENT rules
historical decision replay  same facts, the rules OF THE TIME
```

That is 1c's *observation is not meaning* applied one level up.

### What 1d does not replay, and why that is principled

`decide_promotion` takes a `SearchCandidate` carrying an
`assessment.ranking_score` — **a conclusion**. Persisting one to make
promotion replayable would be exactly the cheat this stage forbids;
re-deriving it needs the assessment layer wired to the graph, which is
the candidate generator's job at stage 2. So 1d replays the *contract*
chain — margins, binding constraint, admissibility, refusal and its
reasons — and does not claim promotion ordering.

### The anti-cheat is structural

A test walks the serialised JSON and fails on any key named
`admissible`, `admitted`, `refused`, `binding`, `frontier`, `verdict`,
`promotion`, `rank`, `chosen`, `best`, `recommendation`, `failures`,
`passed` or `adjudication` — then asserts the *facts* are present, so
the emptiness is not vacuous. It caught its first real case immediately:
`SearchSemantics.promotion` is a rule identity, not a decision, and the
field is now `promotion_rule` so the check can stay blunt instead of
growing an exemption.

Three mutations, each killing exactly its own test: storing the admitted
set → the anti-cheat; admission ignoring evidence scale → the
diagnostic-is-not-an-admission test; the frontier absorbing measurements
of states the graph never held → the foreign-state test.

---

## 4e. Stage 2 — the candidate generator (IMPLEMENTED)

`represent/state/{action_space,candidate}.rs`, 11 tests, 100% on
`action_space.rs` and 98.8% on `candidate.rs`.

```text
realization
    ↓ enumerate legal transformations
    ↓ resolve the child realization
    ↓ derive the physical state
    ↓ price it, from the footprint oracle
    ↓ apply ONLY registered pre-measurement pruning
CandidateSet
```

No ranking, no sensitivity intuition, no family heuristic, no
`decide_promotion`. One question: *what experiments are legitimately
available from this realization?*

### Three prunes, not four

Ruling 1's complete list — and my earlier summary said "four legitimate
prunes", which is wrong:

```text
1  identical MeasurementKey observed            dedup          USABLE
2  not physically better than an available map  dominance      USABLE
3  structurally impossible map                  structural     USABLE
4  a PROVED monotonicity theorem                NOT CURRENTLY HELD
```

`PreMeasurementPrune` therefore has **three** variants. The fourth is
absent by construction rather than present-and-unused, so adding it is a
visible schema change and not a quiet flag flip.

**Behavioural-superset pruning is not on the list.** Authority refusal
attaches to the measured map, is not upward-closed under action-set
inclusion, and "more low precision" is not a behavioural partial order —
the programme holds evidence against it at every level (R4-F7 sign,
R4-F2 magnitude, R5-F4 scale N=3 both directions, R5-F9 ordering, R5-F7's
2.47× between two states). The line:

```text
"this cannot produce a valid physical experiment"   → prune
"this probably will not teach us anything"          → NOT a prune
```

The second belongs downstream in assessment, where it can be argued
with. Here it would be indistinguishable from the first. A mutation that
slips it in reddens the test that records the line.

### The conservation law

```text
enumerated = eligible + already observed + dominated + structural
```

`Census::conserves()` asserts it, and every disposition is
`Eligible(candidate)` or `Pruned { action, child_state, reason }` — so
nothing disappears silently, and *why aren't we exploring E24* is
answered from a deterministic partition rather than by a language model
reconstructing a rationale.

### The vocabulary is an input, and enumeration covers it

R5-F6: neighbourhood 1 drew its in-moves from the candidates left
unpromoted at iteration 4 and never listed `{E20,E22,E23,E24,E25}` —
two moves worth ~430 MB each were invisible because the vocabulary had
been mistaken for the last round's leftovers. `ActionVocabulary` holds
the declared moves; enumeration is every unapplied edit as an addition
plus every (applied, unapplied) pair as a 1-out/1-in exchange, always
over the whole vocabulary.

The vocabulary's declaration order also supplies map order, so an
applied *set* has exactly one map — which is what makes 1a's identity
contract meaningful over search states rather than only over
hand-written maps.

### Physics is derived, never asserted

No `MapEdit` carries a byte figure. The generator applies, resolves,
asks the `Footprint` oracle what parent and child cost, and computes the
delta — guarding the R5-F5 class where a footprint column was read as a
saving and overstated a revert 3.39×. Dominance then delegates to
`TransitionPolicy::admits`, so the generator and the graph cannot give
two answers to one question.

### Dedup needs the experimental context

The generator never asks *have we measured this state*. It asks about a
`MeasurementIntent { bank, scale, instrument }`, because 1c proved
`diagnostic(child) ≠ authority(child)`. A candidate whose exact
experiment exists is pruned; one whose state carries *other* readings
arrives eligible with `prior_observations` attached, so an escalation is
visible rather than silent.

Four mutations, each killing exactly its own tests: dedup on the state
instead of the intent → 1 red; behavioural-superset pruning slipped in →
2 red, including the test that records the line; the census dropping a
category → the conservation assertion; enumeration covering part of the
vocabulary → 4 red.

---

## 4f. Stage 3 — best-first (IMPLEMENTED)

`represent/state/{assess,search_policy}.rs`, 12 tests, 100% on
`assess.rs` and 98.8% on `search_policy.rs`.

```text
CandidateSet → Assessment → BestFirst        experiment selection
Measurements → SearchEvidence → decide_promotion   admissibility
```

Two different questions, kept apart: a candidate can rank first *for
measurement* precisely because it is uncertain while being nowhere near
promotable. `decide_promotion` stays downstream and unweakened.

### Assessment carries ingredients, not a score

`CandidateAssessment` holds a `ranking_score`, and a score is a
conclusion — that is exactly why 1d could not replay promotion. So
`Assessment` holds `physical_delta`, `child_bytes`, `intended_key`,
`prior_observations` and the parent's whole `ConstraintVector`, and a
score is computed on demand under a named rule (`Assessment::score`) and
stored nowhere.

The parent's standing is kept whole rather than reduced: the binding
margin, the headroom and which criterion is scarce are all questions a
reader may want, and a scalar answers none. It is served at **authority
scale only** — a diagnostic reading prices nothing against the contract,
and handing one over as though it did is the inference R5-F4 and R5-F9
closed.

Sign convention is not flipped anywhere: `physical_delta` stays negative
for bytes removed, and ordering prefers the most negative.

### One answer, always

The complete order, stated once in `RankingSemantics::tie_break_chain`:

```text
1  registered rule
2  greater physical improvement
3  canonical child state id
4  canonical child realization id
5  canonical action identity
```

Everything after (1) exists so no answer can depend on insertion order,
map iteration, thread completion or vocabulary traversal. Tested against
reversed and rotated input, on a genuine tie, and on the case that
reaches element (4): one physical state, two realizations. Truncating
the chain reddens two tests.

`RankingRule` has one variant — `PhysicalPrizeFirst` — because one rule
has actually been used: rung 5 spent its run on the prize (−431,777,920 B
over −2,091,136 B). `RankingSemanticsId` digests the rule *and the
chain*, and `SearchSemantics` gained a `ranking_rule` field so a snapshot
can tell the same observations under best-first-v1 from v2.

### Search orders states; measurement orders experiments

```text
A --action x--> C (realization r1) ┐
                                   ├─ one MeasurementKey, run once
B --action y--> C (realization r2) ┘
```

`MeasurementOpportunity` groups by experiment, not by realization, so an
experiment is never scheduled twice in a round — while both routes
survive inside it because their future action spaces differ. When the
observation lands both inherit it, since 1c keys evidence on the physical
state while 1b keeps the realizations apart. Keying opportunities by
realization instead reddens that test.

### Ruling 3, encoded

```text
0 eligible  → Exhausted
1 eligible  → Sole — SELECT it; the diagnostic cannot veto
> 1         → Ranked — the registered rule chooses which runs first
```

The middle line is a ruling, not an optimisation: with one opportunity
"what next?" is already answered, and consulting a diagnostic there
would promote it into an admissibility screen — the accident that was
withdrawn.

### What stage 3 does not do, and what it needs

It does not order the **authority escalation** of diagnostically-measured
candidates, and it does not run promotion. Both go through
`CandidateAssessment`, which needs two facts a snapshot does not yet
hold: a per-state `ByteLedger` (per-token reads — *not* `LogicalBytes`,
which is a whole-map footprint; the newtype exists to keep them apart)
and an `ExecutionCostModel`. Those are **inputs to add**, not conclusions
to invent. Adding them is what would close the gap 1d left open.

No information-gain model either. A 40 MB experiment that would resolve
whether a family of moves is state-dependent may deserve to run before a
600 MB candidate, and *experimental value* is a real quantity distinct
from *physical value* — but nothing has measured it, and inventing it
here would put a heuristic where registered semantics belong.

Four mutations, each killing exactly its own tests: truncating the
tie-break chain → 2 red; grouping opportunities by realization → 1;
reporting a sole opportunity as ranked → 1; serving a diagnostic reading
as the parent's standing → 1.

---

## 4g. Stage 3b — the chain runs from a snapshot (IMPLEMENTED)

`represent/state/{snapshot,replay_tests}.rs`, 7 tests.

```text
snapshot facts → action space → assessment → ranking → promotion
```

with none of the intermediates serialised. 1d closed the contract half
and left promotion open on principle; 3b adds the two **facts** it
needed rather than the conclusion it lacked.

### Two gaps, both closed by adding inputs

**The snapshot could not produce an action space.** It held
`{schema, objective, gate, tail_support, semantics, graph, measurements}`
— no surface, base map or vocabulary — so nothing could build a
`Generator` from it, and the stage-3 tests assembled candidates from a
test rig rather than from reloaded facts.

**Promotion needed a per-state `ByteLedger` and an `ExecutionCostModel`.**

The snapshot is now grouped, which makes the doctrine visible in the
type:

```text
SearchSpace   surface, base map, vocabulary        what is searched over
SearchConfig  objective, gate, tail support,       how facts become
              calibrations, diagnostic policy,     conclusions
              semantics, ranking
SearchFacts   graph, measurements, byte ledgers,   what has been observed
              execution cost                       and what it costs
```

`ExecutionCostModel` already had the discipline: it stores measured
*observations*, each carrying machine, device, backend and compiler
commit, and `predict()` derives the cost while `status()` refuses to
call the model calibrated until beta has been shown across separated
breadths. Nothing about a cost is persisted.

### Promotion reads edges, not candidates

The first wiring fed `promotion_candidates` an eligible `CandidateSet`
and it returned nothing — correctly. **A measured move is exactly what
the generator prunes**, because its question is what to try *next*.
Promotion's question is about what has been built and measured, so it
reads the graph's edges. Feeding it eligible candidates would ask which
*unmeasured* move should replace the incumbent, which no evidence can
answer.

A move is a promotion candidate only when both ends carry a reading at
the scale **and** both carry a ledger. Otherwise it is skipped, not
defaulted: a marginal quantity with one end missing is not a smaller
number, it is no number.

`decide_promotion` is called unmodified, and no proxies are invented —
a `ProxyObservation` is a registered finding about an instrument, and
none exists for these statistics.

### A real defect the widening surfaced

`Role`'s `Deserialize` read `<&str>`, which demands a **borrowed**
string. It worked under `serde_json::from_str` and failed under
`from_value`, `from_reader` and every binary format, with an
`invalid type: string, expected a borrowed string` far from the cause.
A `TensorSurface` carries `Role`s, so a snapshot could not round-trip
through `serde_json::Value` — and an MCP surface will. Fixed to `Cow`,
which still borrows where borrowing is possible, and tested through
both an owning and a reader deserializer.

### The anti-cheat survived the widening

Six new fields, every one a fact or a rule, and the structural check
still holds. It found its second naming collision:
`config.calibrations[].verdict` is ROUTE-CAL-1 saying how a statistic
may be used — a registered finding about an *instrument*, not a
conclusion about a candidate. Exempted by name, so the check stays blunt
everywhere else, and `ranking_score`, `selection` and `next_experiment`
were added to the forbidden list.

### What is real and what is a fixture

The recorded rung-5 numbers — kl p99, logical bytes, covered mass — are
exercised in §4d against the register. 3b tests the *derivation chain*;
its per-token ledgers and GPU cost observation are **fixtures**, because
the record holds no measured GPU time for P, T1 or S2. Inventing one and
calling it recorded would be worse than saying so.

---

## 4h. Stage 4 — the read-only facade (IMPLEMENTED)

> **Anything an agent can learn through the facade is already derivable
> from the deterministic optimiser substrate.**

Everything an agent is shown is a projection of a stored record.
Nothing in the facade orders what the optimiser did not order, draws a
verdict the contract did not draw, or prices a byte a footprint did not
price. Two layers, and the theorem lives entirely in the first:

```text
larql-vindex  represent::view      the seven questions, as pure projections
larql-cli     optimizer_mcp        JSON-RPC on stdio: dispatch and serde
```

### The seven, and what each one is allowed to call

```text
optimizer.describe          schema, model, surface, contract, policies,
                            the six decision procedures, the vocabulary
optimizer.current           root, graph shape, the incumbent, admitted,
                            what is still dark per evidence scale
optimizer.frontier          every state's standing, every adjudication
optimizer.explain           identity  ·  discovery path  ·  standing
optimizer.compare           two standings and one physical delta
optimizer.evidence          the raw banks beside their verdicts
optimizer.next_experiment   a refusal, and what is missing (below)
```

Absent, and not "not yet": `record`, `apply`, `expand`, `promote`,
and above all `accept_candidate`. A test parses the facade's own source
and fails if an eighth method appears or if any method's name contains a
writing verb, so the surface is checked against the code rather than
against a list in a comment.

### Rendering a derived verdict is not storing one

[`Adjudication`] and [`FrontierEntry`] withhold `Serialize` on purpose:
1d forbids storing a conclusion. Rendering one is a different act — the
agent is shown a verdict the optimiser reached just now, from facts —
but it is the act with the most room to lie. So the split is:

```text
substrate type derives Serialize    render the type itself
substrate answers via a method      one view field, naming the method
```

Rendering the type is the stronger option wherever it is available: a
reshaped `Margin` is a second `Margin` that can disagree with the first.

### The anti-cheat, pointed the other way

1d walks a STORED snapshot and fails on any key that names a
conclusion. Stage 4 walks a RENDERED response and fails on any leaf that
names nothing at all. Every field declares the call behind it, and the
walk checks both directions:

```text
undeclared leaf        a value the facade invented
unreached declaration  a registry entry that has stopped describing the
                       code, and the next field added under it would
                       inherit its alibi
```

Descent stops at a declared path, so an embedded substrate type is
covered whole and the registry does not have to restate `Margin`'s
fields. An absent field or an empty collection is excused by a
declaration beneath it and by nothing else.

Eight mutations, each killed by exactly its own tests: drop an origin →
1; declare a dead one → 1; render an undeclared `recommendation` → 2;
grow a writer → 2; order the admitted set in the facade → 1; treat a
diagnostic pass as an admission → 1; subtract the wrong way round → 1;
serve a write tool → 3.

**The ordering test could not have failed.** Rung 5 admitted exactly one
state, so `first()` and `last()` were the same element and "the
incumbent is position zero" asserted nothing. It now runs against an
openly counterfactual fixture that gives S1 a passing authority reading
it never had — labelled as not the record — purely so the claim has two
elements to order.

### The transport carries; it does not derive

`Server` is handed an `OptimizerView` and never a `SearchSnapshot`, so
"derive nothing in transport" is a matter of what is reachable rather
than of what a reviewer noticed. The load-bearing test compares each
tool's payload against the view's own serialisation character for
character: a transport that reshaped, ordered, summarised or annotated
an answer fails rather than being noticed later.

An id the record does not hold comes back as a tool error carrying
WHICH id was wrong, not as a protocol failure and not as an empty
success.

### What stage 4 found: `next_experiment` cannot be served

The one tool of the seven that does not answer, and the refusal is the
stage's real finding.

`SearchSnapshot::next_experiment` derives the whole chain from stored
facts, but takes two arguments that are CODE and not data: a
`LayoutAdmission` and a `Footprint`. The first has production
implementations. **The second has none** — `Footprint`'s own contract is
*"supplied, never derived here"*, and the only implementations in the
tree are three copies of a test fixture, identical up to variable names.

Nor can the facade write one:

```text
declared   SearchSemantics.physical_accounting = "logical-bytes/v1"
held       a version id, which names a procedure and is not one
missing    a Footprint oracle the snapshot can name
missing    a source dtype on TensorSurface
```

Pricing a decision the map PROTECTS needs the source dtype, and
`TensorSurface` carries object, tensor, role and shape and no dtype at
all. The three fixtures close that gap by multiplying by two, which is
bf16 asserted rather than read. Promoting an assertion about a dtype
into production is exactly the move that makes a search price bytes it
never saved — and adding a dtype to `TensorSurface` re-opens 1a, since
the surface feeds every `RepresentationStateId`.

So the tool refuses and names both missing facts, and still serves what
needs no oracle: the move vocabulary (R5-F6 was a vocabulary failure and
cost two ~430 MB moves) and the unmeasured states, which are already in
the graph and already priced.

Closing this is a substrate question and not a transport one. A stored
price table keyed by state id is the obvious shape — an INPUT, the same
move stages 3 and 3b made twice — but `Footprint::logical_bytes` returns
`LogicalBytes` with no channel for a miss, and an unpriced candidate
that is neither eligible nor pruned breaks the census conservation law.
That is a stage of its own, not a line in this one.

### Coverage and the fixture

100 % line coverage on all nine `view/` files; `protocol.rs` and
`tools.rs` at 100 %, `server.rs` at 97.2 %. The `optimizer_mcp/` subtree
is brought UNDER the CLI coverage policy rather than left outside it,
with one baseline: `run` locks the process's own stdin and stdout, so
its six lines cannot be reached in-process. Everything below it was
split out as `dispatch`/`declare`/`serve`/`load`, which take their
reader and writer and are driven end to end from a record on disk.

The Rung 5 record is now ONE fixture, shared by the 1d replay gate and
the stage-4 views and exposed to downstream test suites behind
`larql-vindex`'s `test-utils` feature. Two copies of those numbers would
be two records, and the second would drift.

[`Adjudication`]: #4d-stage-1d--the-replay-gate-implemented
[`FrontierEntry`]: #4d-stage-1d--the-replay-gate-implemented

---

## 4i. Stage 4b — the source identity the optimizer prices from

Stage 4 could not serve `next_experiment`: pricing a PROTECTED decision
needs the source `dtype` and `len`, and `TensorSurface` carries neither.
The obvious fix — add a `source_dtype` to `TensorSurface` — re-opens 1a,
because the surface feeds every `RepresentationStateId`. 4b asks the
prior question instead: **does the container already seal those facts,
and is that seal load-bearing in the identity?**

```text
4b-a   prove the physical seal                    done
4b-b1  refuse incomplete/contradictory identity   done
4b-b2  identity semantic, not formatting-sensitive  done
4b-c   read sealed SourceStorageFacts              done
4b-d   require whole-surface accounting completeness  done
4b-e   production Footprint over a bound surface       done
4b-f   next_experiment answers, transport unchanged   done
```

### 4b-a — the seal exists, and it is the segment's own table

The facts a footprint needs live in the segment header, not the payload:

```text
SegmentTensor { name, dtype, shape, offset, len }
```

and `len` is the authority, **not** `shape × width(dtype)` — a packed or
padded tensor has a length the naive product does not predict, which is
exactly why stage 4 refused to price a `Source` decision by multiplying
by two.

Two adversarial tests on a real encoded container move one field of that
table — `dtype` in one, `len` in the other — copy the payload verbatim,
recompute `segment_sha256`, and require the state id to move while the
effective decision vector stays byte-identical. A control restates the
table unchanged and requires the id to hold. The invariant:

> A `RepresentationStateId` must never be reusable across two
> physical-accounting realities that price the same effective decision
> vector differently.

**So B holds and `TensorSurface` gains nothing.** The container seals
bytes AND table, which is strictly stronger than a dtype on the surface,
and 1a stays closed.

### 4b-b1 — identity construction is typed, total and refusing

`read_source_identity` walked `index.json` as untyped JSON and took what
it found, while the container parser refused the same document. Every
one of these returned `Ok`, over facts it had silently dropped:

```text
no `representations` key      an identity sealing NO payload at all
entry missing segment/digest  that segment left out of the seal
two entries, one segment      last writer wins, the other discarded
no `system_graph`             a filename assumed, and hashed as authority
```

Now parsed through `Vindex3Index` — the same validated schema the other
seven readers use — and refusing, naming the entry and the field.

> **An identity function may be stricter than the consumer it
> identifies. It must never be looser.**

Two representations may share a segment; what they may not do is
disagree about its digests. Eight tests on an **eight**-representation
container: a one-entry fixture cannot tell "refused" from "identified
over what was left", because nothing is left. The historical walk,
restored verbatim, is killed by six of them.

b1 changed no equivalence relation on purpose, and left the false split
below standing so it could be its own step.

### 4b-b2 — semantic identity, and an equivalence relation that moved

`manifest_hash` was `hash_bytes(index.json)`. So:

```text
same container semantics
same graph, payload and segment table
different index.json formatting
        ↓  different manifest_hash
        ↓  different RepresentationStateId
```

A re-exported container was a different search state carrying none of
its own evidence — 1a's SPLIT direction. Two digests now, and only one
of them identifies:

```text
index.json raw bytes
        → ContainerArtifactDigest      provenance; MAY move on re-export

typed Vindex3Index
+ graph authority (by content)
+ representation → segment/header/payload associations
        → SourceSemanticIdentity      what the container IS
        → RepresentationStateId
```

**The non-negotiable invariant.** Byte-hashing the index sealed the
segment header table by accident. When those bytes left, `segment_sha256`
had to enter the semantic identity **explicitly**, or b2 would have
severed exactly what 4b-a had just proved. It is
`CanonicalRepresentationAuthority::segment_sha256`, and a test asserts
the canonical form writes it.

**Associations, not a multiset.** Each authority is one record binding an
entry's identity to its own digests, ordered by that identity rather than
by input order. Hashing a sorted multiset of `segment_sha256`s would be
blind to two entries exchanging which segment file seals them — the
multiset survives and the model changes.

**A purpose-built projection, not a normalised document.** What counts is
decided by typed structures; canonical JSON is only the encoding of the
container-level tail, which is *sealed* rather than enumerated so that a
field the index gains next is IN by default. Dropping a fact is the MERGE
direction, so leaving one out takes a deliberate, named, tested removal:

```text
system_graph         a FILENAME; the graph is sealed by CONTENT
derived_from_model   an operator's hint for finding the authority
encoder              the encode RECIPE; its bytes are already sealed
compiled_from        lineage; the bytes are sealed either way
source_representation_digest
```

`codec` stays IN: `encoding` names a family, the codec names the decode
contract, and the same bytes under a revision a reader implements
differently are different bytes.

**The three siblings**, on one real encoded container:

```text
A  original
B  semantically identical index, differently serialised
C  identical index and payload, one segment-table `len` changed

artifact(A) != artifact(B)      a re-export IS a different file
semantic(A) == semantic(B)      and the same source
state(A)    == state(B)
semantic(A) != semantic(C)      a different physical reality
state(A)    != state(C)
```

C is checked to have moved in exactly one field — `segment_sha256` —
so the test cannot pass on a projection that threw itself away. The full
relation:

```text
presentation bytes changed only     SAME physical state
header/storage reality changed      DIFFERENT physical state
payload reality changed             DIFFERENT physical state
graph reality changed               DIFFERENT physical state
```

`SourceDependency::verify` moved to the same footing: a byte-different
export of the container a candidate was compiled against now verifies,
and the catch-all reads the same digest the search identifies states by,
so verification and identity cannot give two answers.

**Verified they can fail.** Seven mutations, each killed by exactly its
own tests:

| Mutation | Killed |
|---|---|
| the artifact digest reaches the state id again | `three_siblings…`, `a_provenance_only_change…` (2) |
| the authority stops writing `segment_sha256` | both 4b-a header tests, `the_semantic_identity_carries…`, `three_siblings…`, `swapping_two_authorities…`, `a_candidate_refuses_a_source…` (6) |
| the canonical form drops the catalogue | `a_catalogue_fact_no_authority_carries…` (1) |
| associations collapse to a sorted multiset | `swapping_two_authorities…` (1) |
| the projection stops removing `derived_from_model` | `a_provenance_only_change…` (1) |
| the catalogue is hashed from the index TEXT | `three_siblings…`, `a_provenance_only_change…` (2) |
| `canonical_json` sorts array items | `the_order_the_profiles_are_declared_in…` (1) |

**Versioned, because the equality changed.** `source-semantic-id/v1`
feeds `represent-state-id/v2`. v1 and v2 answer differently for
containers that already exist, which is precisely what 1a versioned the
canonical form to make visible.

> v2 removes an identified false-split failure while retaining
> sensitivity to graph, header and payload authority.

### 4b-c — the accounting facts, from the same authority

> **Accounting does not discover new source facts. It projects
> accounting facts from the same validated authority already used to
> establish source identity.**

```text
CanonicalRepresentationAuthority     segment, segment_sha256, tensor_count
        ↓ opens exactly that segment, recomputes exactly that digest
SegmentTensor { name, dtype, shape, offset, len }
        ↓
SourceStorageFact { logical_bytes = len, dtype }
```

**A dereference, not a second parse.** The per-tensor table is not in
`index.json` — it is inside the segment file, which is exactly why
`segment_sha256` had to enter `SourceSemanticIdentity` explicitly in
4b-b2. `read_source_storage` opens the file the authority names,
recomputes the digest the authority seals, and refuses before reading a
number if they differ. Without that check, accounting would be pricing
whatever happens to be on disk under a path an index mentions, next to a
state id built on something else. The header itself is parsed by the
writer's own `read_segment_header` — a second header parser here would
be 4b-b1's untyped walk one level down.

**`len` is the byte count; `dtype` explains it.** `SourceDType` is
deliberately opaque: no `width`, no `size_of`, no conversion to a
number. The moment one exists, `numel × width` is one refactor away, and
that is precisely the shape stage 4's three fixture footprints had — all
three multiplying by two and calling it bf16.

4b-a moved `dtype` and `len` in two SEPARATE adversarial tests so that
this asymmetry could be asserted:

```text
same shape, same dtype, changed len     → the price changes
same shape, changed dtype, same len     → the price does NOT change
changed shape, same len                 → the price does NOT change
```

**What it refuses**, rather than omitting: a segment that is not the
sealed one; a segment that cannot be read; a sealed segment whose header
does not parse; a table contradicting the `tensor_count` the index
declares (two sealed facts about one segment disagreeing); and one
`(object, tensor)` stored twice. That last is a container holding a
source pack and a compiled pack for one object — `compiled_from` is the
obvious discriminator and adopting it is a decision with its own
evidence, so the collision is named rather than resolved.

Aliases stay two facts: `(object, tensor)` is the optimizer's tensor
identity, now a named `TensorIdentity` because 4b-d has to hand back the
exact identities a bind is missing.

**The declaration became the code.** Stage 4 refused `next_experiment`
because `SearchSemantics.physical_accounting` named a procedure that did
not exist. `PhysicalAccountingSemantics` digests the declared MEANING —
byte authority, seal verification, and what the dtype is permitted to do
— and `read_source_storage` stamps it, with no other constructor for
`PhysicalAccountingFacts`. 1c had to state that a declaration can lie and
that the module could not enforce otherwise; here it can. The procedure
keeps the name stage 1d already declares (`logical-bytes/v1`): renaming
it would move `SearchSemanticsId` for every stored snapshot and announce
a changed procedure, and the procedure did not change — it started
existing.

`PhysicalAccountingFacts` carries the `SourceSemanticIdentity` digest it
was read from, so `describe(model)` is checkable rather than assumed —
and it resolves on the semantic digest, so a re-export does not orphan a
footprint. It carries no `TensorSurfaceId`: the facts describe what the
CONTAINER stores and the surface is what REPRESENT enumerated. Those are
different populations, and finding where they disagree is 4b-d's whole
job.

**Verified they can fail.** Six mutations, each killed by exactly its own
tests:

| Mutation | Killed |
|---|---|
| price computed as `numel × width(dtype)` | all three arithmetic tests (3) |
| the seal is not verified | `a_segment_that_is_not_the_sealed_one…`, `another_containers_identity…`, `a_missing_segment…` (3) |
| keyed on the tensor name alone | `two_objects_sharing_a_tensor_name…`, `the_facts_come_from_the_segment…` (2) |
| the declared `tensor_count` is not cross-checked | `a_table_that_contradicts_its_own_declared_count…` (1) |
| a repeated `(object, tensor)` is last-writer-wins | `one_object_stored_twice…` (1) |
| `describe()` resolves on the artifact digest | `a_reserialised_container_is_still_the_source…` (1) |

The first is the one that matters: it is the regression this step exists
to foreclose, and it dies against three independent tests.

### 4b-d — can this surface be priced authoritatively at all?

One question. No cost is computed, nothing is ranked and nothing is
pruned.

```text
PhysicalAccountingFacts
+ TensorSurface
        ↓ bind()
BoundPhysicalAccounting
or
AccountingIncomplete { missing: [TensorIdentity, …] }
```

> **READY means every tensor on the REPRESENT surface has exactly one
> authoritative source price from the sealed container facts.**

The two populations are genuinely different — `PhysicalAccountingFacts`
is what the CONTAINER stores, a `TensorSurface` is what REPRESENT
*enumerated* under one role classification — and only one direction of
disagreement matters:

```text
surface tensor with no stored fact   → cannot be priced. INCOMPLETE.
stored fact no surface tensor names  → not this surface's business
```

An extra stored fact neither satisfies nor damages completeness. Letting
it do either is how a missing price gets papered over by an unrelated
one that happened to be present.

**Incompleteness is a failure to bind, not a fourth prune.** Stage 2's
register holds exactly THREE usable pre-measurement prunes and "cannot
be priced" is not among them; an unpriceable candidate arriving neither
eligible nor pruned would break the census conservation law (§4e). So
the search does not start. `BoundPhysicalAccounting` is constructed only
by `bind`, so holding one IS the proof — which is what lets 4b-e
implement `Footprint` with no `Option`, no fallback and no missing-data
branch. `prices_for(surface)` makes ONE check, that this is the surface
that was bound, and then pairs every tensor with its price totally.

**Blind to role and shape**, asserted rather than assumed: a
reclassified role moves the surface identity and is a different search
problem (1a), and a shape is not what anything is priced by (4b-c), so
neither may change whether a model can be priced at all.

Two failures, kept apart because they call for different actions:
`ForeignSource` is the wrong facts entirely and re-reading fixes it;
`Incomplete` is a real gap. A foreign source is reported as foreign even
though every surface tensor is also, incidentally, unpriceable. The
source check resolves on the SEMANTIC digest, so a re-exported container
still binds.

**Verified they can fail.** Seven mutations, each killed by exactly its
own tests:

| Mutation | Killed |
|---|---|
| a missing tensor is skipped instead of collected | 4 |
| binding stops at the FIRST missing tensor | `several_missing…` (1) |
| keyed on the tensor NAME, so an alias is satisfied by its twin | `an_alias_is_two_entries…`, `a_surface_the_container_stores_entirely_binds` (2) |
| the source is not checked | `facts_from_another_container…` (1) |
| the source check resolves on the artifact digest | 9 |
| `prices_for` answers over any surface | `a_bound_accounting_refuses_a_surface…` (1) |
| the missing list is not ordered | `several_missing…` (1) |

**The open question 4b-d recorded**, settled in 4b-e: a container may
store tensors the surface does not enumerate, so summing the bound
prices is the SURFACE's footprint and not the container's.

### 4b-e — the production `Footprint`

> **`Footprint` is the complete footprint of the bound REPRESENT surface
> under a representation state — not the byte size of the whole
> container.**

```text
container footprint        everything physically stored in the container
representation footprint   every tensor in the bound TensorSurface,
                           priced under one resolved state        ← this
```

A map resolves over a `TensorSurface`, so its domain is that surface.
Anything outside it has no representation decision in this search
problem, and counting it would make the optimiser account for bytes it
cannot transform. 4b-d's asymmetry is the proof: an extra stored fact
neither satisfies nor damages surface completeness, so it must not
reappear in the state's footprint either. `LogicalBytes`' doc comment
said "whole-map footprint", which reads either way; it now says what it
means, and whole-container accounting gets its own type when it is
needed rather than a second meaning for this one.

```text
effective == source      → the sealed SourceStorageFact
effective == encoding    → the pack layout's stored length
```

`effective()` and not the declared encoding: a protected tensor and a
layout-refused one present the same bytes, and pricing the refusal as
compiled would book a saving the container never made. Two realizations
of one `RepresentationStateId` therefore cost the same, which is a
cross-stage invariant `physical_delta` depends on.

**Misses are made impossible, not handled.** `Footprint` returns
`LogicalBytes` with no channel for "I could not price that", and
inventing one would break stage 2's census — an unpriced candidate is
neither eligible nor pruned. So the constructor enumerates the finite
problem instead: every bound tensor × every encoding the search may
select.

```text
layout REFUSES (tensor, encoding)   → resolves to source; no price needed
layout ADMITS  (tensor, encoding)   → a compiled price is REQUIRED
```

Using the same `LayoutAdmission` the resolver uses, so the price table
and the decision vector cannot disagree about which tensors are
compiled. What remains is a state resolved against another surface,
which `try_logical_bytes` reports and the trait method — by contract,
loudly — cannot.

**A substrate fact this surfaces:** `PackCompiledBytes` prices NVFP4 from
`PackLayout::derive`, the same call the compiler and `PackLayoutAdmission`
make, and declares nothing about any other encoding. `PackLayoutAdmission`
admits Q6_K (it holds no rule for it) and nothing prices a Q6_K pack in
this build, so a search whose vocabulary names Q6_K is refused at
construction, naming the tensor and the encoding — before the first
candidate rather than at it.

**Verified they can fail.** Five mutations, each killed by exactly its
own tests:

| Mutation | Killed |
|---|---|
| prices the DECLARED encoding, not what is presented | `a_layout_refusal_is_priced_as_source…`, `two_realizations_of_one_state…` (2) |
| the compiled price is `numel × 2`, not the pack layout | `a_compiled_state_costs…`, `nothing_but_nvfp4_is_priced…` (2) |
| an unpriceable admitted encoding is skipped, not refused | 2 |
| the surface check is dropped | 3 |
| the source price is used for every decision | `a_compiled_state_costs…` (1) |

**One survived the first attempt and is worth recording.** Replacing the
pack layout with `numel × 2` passed, because the compiled-price test
asked `PackCompiledBytes` what `PackCompiledBytes` said — a
self-normalising test over the very arithmetic under test. It now
asserts against a figure derived from the FORMAT: a `[64, 64]` NVFP4
pack is `64×4×8` code bytes + `64×4` scale bytes + one f32 = 2308, and
`numel × 2` is 8192.

### 4b-f — the record answers, and the transport did not change

```text
SearchFacts.accounting          ← the only new stored input
        │ reload
        ▼
bind(surface)  →  BoundPhysicalAccounting
        ▼
SurfaceFootprint, under the record's OWN declared policies
        ▼
candidate generation → assessment → best-first
        ▼
NextExperiment::Available(…)
```

Everything below `PhysicalAccountingFacts` is derived on every call and
cached nowhere. 1d's theorem is intact rather than weakened to make a
tool answer.

**Three answers, because an agent takes three actions.** `Available`,
`Exhausted` and `Unavailable{reason, detail}`. Collapsing the middle
into the last would say "nothing to do" and "I cannot tell you" in the
same words. `Available` means the deterministic optimiser had the
factual authority to SELECT the next unresolved experiment — not that it
has been run, admitted or promoted.

**The question is stored too.** `SearchSpace.applied` (where the search
stands — a position, never a verdict) and `SearchConfig.standing_intent`
(which corpus, scale and instrument the next run would be). Without them
the caller would supply the question and the answer would stop being a
property of the record — and the tool would need arguments, which is a
transport change.

**One layout truth.** `SearchSemantics` gained a seventh field,
`layout_admission`, and it was missing: a layout refusal removes a
tensor from the action space and collapses its state onto the protected
one, so a record that did not name its layout policy could be replayed
under another and produce different states with nothing failing. The
snapshot resolves the NAME to the one implementation and hands the SAME
reference to state resolution and to price-table construction.
`compiled_bytes(procedure)` does the same on the compiled side. An
unknown name is refused, never defaulted.

The cross-test: the same `k = 24` tensor under `no-layout-constraint/v1`
is ADMITTED, so a compiled price becomes required and nothing prices it
— refused; under `pack-layout-admission/v1` it is refused by the layout,
resolves to source, and needs none — answered. Only the declared policy
differs between the two records.

**Two defects this step found:**

* `PhysicalAccountingFacts` could not be serialised at all.
  `source_storage` was a `BTreeMap` keyed by a STRUCT, which derives
  `Serialize` happily and fails at runtime — *key must be a string* —
  the moment it holds anything. Every 4b-c test read the facts in
  memory, so nothing noticed. It is `Role::deserialize`'s
  borrowed-string bug one level up. Now an ordered sequence, with a
  round-trip test through both `from_str` and `from_value`.
* The view anti-cheat caught 20 declarations no rendering reached. A
  type with alternative shapes can only describe all of them across all
  of them, so coverage is now the UNION over the three variants —
  which needs a record that can actually answer.

**The acceptance test.** Two records through ONE view method:

```text
no accounting authority   → Unavailable(no-accounting-authority)
sealed container facts    → Available(a real experiment)
```

and the answer survives a round trip through stored JSON — serialise,
drop everything, reload, ask again, same experiment. Derivation, not
recall.

**What changed under `optimizer_mcp/`: one test assertion.**
`protocol.rs`, `server.rs` and `tools.rs` are untouched. The dispatch
still calls one view method and serialises whatever it returns; the test
that asserted the payload named `NoFootprintOracle` now asserts it names
`Unavailable{reason}`. MCP was complete when it refused, and supplying
the missing substrate truth made the existing transport answer.

---

## 5. The reward, which must not be diagnostic KL

Rung 4/5 established that diagnostic KL supplies neither magnitude, sign,
authority scaling, nor parent-relative ordering. Feeding it to a policy
as a scalar reward reintroduces every one of those with a UCB formula on
top. So the value function is stratified by evidence class and only the
top stratum produces a number priced against the contract:

```text
authority-admitted   measured objective (tok/s) / physical cost
authority-refused    terminal for this identity — but the ACTION is not
                     globally poisoned (R5-F6): the same action from a
                     different parent is a different, unmeasured state
diagnostically seen  optimistic physical prize
                       × evidence confidence (SearchEvidence rank)
                       × novelty
unmeasured           prior only — physical prize is exact, the rest is a hint
```

`SearchEvidence::is_priceable()` already returns false for
`OrderingProxy`; the policy must call it. Turning a diagnostic reading
into an admission probability is legitimate only after a calibration
demonstrating magnitude transfer, and that calibration has a home:
`SearchCalibrationRegistry`.

PUCT over vanilla UCT, when it arrives, because real priors exist: exact
physical gain, depth/family history, arm and router position, structural
participation, and the set of already-explored identities. Priors say
where to look; authority says what survives.

---

## 6. The MCP surface — intent, not tree operations

The agent must not become a `for` loop. An interface of
`search.expand(node)` / `search.observe(result)` puts the LLM inside the
optimiser's inner loop and costs 40,000 round trips at K3 scale. Those
calls exist, as a debug surface. The **primary** interface is intent:

```text
optimize.search(objective = throughput,
                constraints = ["balanced-v1"],
                budget = { diagnostic: 30, authority: 5, benchmark: 2 })

optimize.status()
optimize.frontier()
optimize.explain(state)
optimize.next_experiment()
```

The deterministic optimiser performs hundreds or thousands of cheap
expansions internally. The agent is engaged where reasoning is actually
useful — *we have two competing hypotheses for E24; would an exchange
experiment distinguish state dependence; is this worth an authority run;
the frontier has collapsed onto tiny physical gains.*

Three namespaces (one server, three prefixes to start):

```text
represent.*   numerical representation search
residency.*   placement / cache / prefetch / external storage
evidence.*    banks, contracts, provenance, comparison, authority
```

`search.neighbours` is the tool that justifies the exercise: not blind
tensor mutations, but actions annotated with family, depth, arm, position
relative to the router, downstream router reach, encoding transition,
exact logical byte saving, and which structural statistics the action can
participate in (`participation.rs`).

### The tool that must not exist

```text
accept_candidate(candidate, reason = "AI judgement")     ✗ never
```

Admission stays mechanical: `measure.authority(...)` → `AuthorityResult`,
`contract.evaluate(...)` → PASS | FAIL | Unscorable. The agent chooses
what to ask; it gets no vote on what the answer means.

---

## 7. Implementation sequence

Deliberately boring, so MCTS is a policy swap and not a rewrite.

| # | Rung | State |
|---|---|---|
| 1a | Resolved-state identity contract + adversarial tests | **done** |
| 1b | State graph with **edge** provenance (§4b) | **done** |
| 1c | `MeasurementKey` + measurement dedup (§4c) | **done** |
| 1d | Persistent, replayable search state (§4d) | **done** |
| 2 | Deterministic action generator (§4e) | **done** |
| 3 | Best-first; the objective API and the search/evidence boundary (§4f) | **done** |
| 3b | Promotion-input closure — the chain runs from a snapshot (§4g) | **done** |
| 4 | MCP facade — read-only; seven intent-level tools (§4h) | **done** |
| 4b | The source identity the optimizer prices from (§4i) | **done** |
| 5 | PUCT as another `SearchPolicy`; same states, actions, evidence | |
| 6 | Extend `PhysicalState` with residency; optimise measured tok/s | |

Provenance lives on **incoming edges**, not baked into the node:

```text
A --[Q6 E24]-------------------→ C
B --[exchange K24→K25, +E24]---→ C
```

One `C`, several explanations of how it was reached. That is what makes
`optimize.explain(state)` able to separate *state identity* from
*discovery path* — the distinction the experiment ledger currently makes
by hand.

### The stage-1 exit gate

> Replay a real Rung 4/5 sequence **from the initial map and the recorded
> measurements — not the recorded decisions** — and reproduce candidate
> identity, dedup, evidence applicability and promotion decisions without
> consulting the experiment ledger.

Deriving the decisions again is the point. Replaying stored decisions
would prove serialisation; deriving them proves the optimiser state
contains enough to replace what currently lives in the ledger and in an
operator's head.

---

## 8. Where MCTS becomes earned

Not yet. State dependence alone only makes greedy ranking questionable.
The result that settles it is **path-dependent option value**:

```text
             +E24
      A ─────────────→ FAIL

      │ -K24 / +K25
      ▼
      B ─────────────→ C   PASS, −600 MB
             +E24
```

where reaching `B` required an apparently inferior intermediate move.
Beam search recovers some of that; MCTS assigns value to a state because
of its *descendants*.

The K24 state-dependence observed so far is diagnostic-scale (5.358e-3
at `{E26,M26}` against 2.172e-3 at `{E26,K25}`, 2.47×), and
`search_evidence.rs` is explicit that diagnostic scale cannot carry the
claim. So the falsifier is **an authority-scale parent-dependent
reversal**.

**That experiment is already registered and running.** Rung 5's
neighbourhood 3 — `U1 = P−M26+E24` and `U2 = P−K25+E24`, ~430 MB each —
puts the E24 action, authority-refused as E24b at 4.298e-3 under
*lighter* all-Q8 arms, into two different representation states. A PASS
is exactly the shape above: an action refused in one state proving
admissible in another. Until it returns, best-first is sufficient and
this section is a plan, not a justification.

Note what may *not* be inferred meanwhile: Ruling 1 forbids pruning
either candidate for being a superset of a refused map. Authority refusal
attaches to the measured map, is not upward-closed under action-set
inclusion, and "more low precision" is not a behavioural partial order.
The legitimate pre-measurement prunes are an identical `MeasurementKey`,
physical dominance and structural impossibility — **three**, plus a
fourth that would need a proved monotonicity theorem and is explicitly
NOT HELD. **That list is the action generator's entire contract** (§4e).

Residency will likely make the phenomenon common:

```text
representation change → frees 700 MB UMA → 6 more experts resident
                      → fewer external reads → +4 tok/s
```

There the action's value is not its local byte saving at all; it is its
effect on the future physical plan.

---

## 9. Open questions

1. **How an instrument declaration is bound to the runner.** 1c
   settled the *shape* of `InstrumentSemanticsId` and left the binding
   open: nothing forces the Q2a harness to build its declaration from
   `TOP_N` rather than restating it. Until it does, a silent semantics
   change is undetectable. The fix is a constructor at the call site,
   not a validator here.
2. **Where the DAG persists.** Evidence already lives in the experiments
   ledger; a second store risks two truths. Candidate: the DAG holds
   identities and edges and dereferences every measurement to the ledger
   rather than copying it.
3. **Crate boundary.** A thin `larql-represent-mcp` binary over public
   `represent::*` types is the obvious eventual shape, but stage 1 stays
   inside `larql-vindex` and moves only when the transport forces it.
4. **When `residency.*` joins the state.** Sharing the DAG from stage 1
   is cleaner and makes stage 1 much larger; stage 6 is the current plan.
