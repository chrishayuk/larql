# The LARQL Physical Optimizer — evidence-constrained physical-plan search

**Branch `worktree-represent-optimizer-mcp`, based on `origin/main` at
`8f647872`.** Stage 1a is implemented and green; everything below it is
design.

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
| Behavioural contract + margins | `constraint.rs` — `Margin`, `Frontier`, `binding()`, `admissible()` |
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

The canonical form is versioned (`represent-state-id/v1`) so a persisted
DAG that outlives a change to it is recognisably stale rather than
silently colliding.

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
and Ruling 1 lists physical dominance among the four legitimate
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
| 1c | `MeasurementKey` + measurement dedup | next |
| 1d | Persistent, replayable search state | |
| 2 | Deterministic action generator — exact physical deltas, participation, novelty and admissibility *before* measurement | |
| 3 | Best-first / beam; establish the objective API and the search/evidence boundary | |
| 4 | MCP facade — inspect state, hypotheses, frontier; request experiments | |
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
The only legitimate pre-measurement prunes are an identical
`MeasurementKey`, physical dominance, structural impossibility, and a
proved monotonicity theorem — and no such theorem is currently held.
**That list is the action generator's entire contract at stage 2.**

Residency will likely make the phenomenon common:

```text
representation change → frees 700 MB UMA → 6 more experts resident
                      → fewer external reads → +4 tok/s
```

There the action's value is not its local byte saving at all; it is its
effect on the future physical plan.

---

## 9. Open questions

1. **`MeasurementKey` dimensions — mostly settled already.** R5-F3
   registered them: `{map_hash, bank_manifest, evidence_scale,
   instrument_semantics}`, with the explicit correction that **dedup on
   bare `map_hash` is wrong** because it would forbid the
   diagnostic → authority escalation the whole ladder depends on. In
   this module's vocabulary `map_hash` becomes `state_id`. Still to
   settle: whether `QualityGate.id` is already implied by
   `instrument_semantics` or belongs as its own dimension, and whether
   execution/backend identity enters the key for benchmark measurements
   only or for all of them. Note that `bank_manifest` exists in the
   programme (`17d59a6b…`) but not as a field on `QualityBank`, which
   carries `positions` and evidence and no id — so bank identity has to
   be lifted into the type before 1c can key on it.
2. **Where the DAG persists.** Evidence already lives in the experiments
   ledger; a second store risks two truths. Candidate: the DAG holds
   identities and edges and dereferences every measurement to the ledger
   rather than copying it.
3. **Crate boundary.** A thin `larql-represent-mcp` binary over public
   `represent::*` types is the obvious eventual shape, but stage 1 stays
   inside `larql-vindex` and moves only when the transport forces it.
4. **When `residency.*` joins the state.** Sharing the DAG from stage 1
   is cleaner and makes stage 1 much larger; stage 6 is the current plan.
