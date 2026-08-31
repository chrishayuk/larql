# VINDEX3 ontology drill — four architectures against the schema

**Date:** 2026-08-30
**Status:** run once, against the working tree at the 3.0-candidate merge
(`7b5fb4d4` + `#341`). This is the acceptance drill the candidate spec
pins in §17.4: hostile schema review, on paper, against the real types —
not an implementation exercise. Findings are recorded before verdicts;
the verdict is at the end, where it has to answer to the evidence.

**The pass rule (brutal on purpose).** A scenario passes 3.0 if it can be
represented using existing semantics plus additive new vocabulary,
without changing what an existing field means. It exposes a 3.0 blocker
if an existing field must acquire a new meaning; an object kind secretly
mandates an operation not inherent to it; KV is structurally assumed
where arbitrary continuation state is required; model-family identity
must survive into execution; physical placement is required to express
model meaning; or a required semantic fact can only be smuggled through
opaque metadata.

**Verdict taxonomy.** Each finding is one of:

- `additive-safe` — new vocabulary, old meanings untouched;
- `implementation gap` — the schema expresses it; the code does not
  execute it (not a freeze blocker: freezing meanings, not features);
- `schema gap` — the ontology itself must change: a reinterpretation.
  These are the freeze blockers.

---

## Case 1 — pure SSM/Mamba decoder (no attention anywhere)

*Attacks: Transformer-shaped completeness. Must remain true: zero
attention surfaces can be valid; stateful operations don't require fake
attention or KV.*

### Evidence

- **`ExecutionSurface.attention` is mandatory and fabricated.**
  `graph/surface.rs:231-238`: `attention: AttentionSurface` is
  non-`Option`, no default. `surface_from_resolved`
  (`surface.rs:387-401`) unconditionally builds one from
  `resolved.num_q_heads / num_kv_heads / head_dim`, which are
  non-`Option` `usize` on `ResolvedArchitecture`
  (`larql-models/src/inventory/report.rs:128-130`). A model with no
  softmax attention still persists an `AttentionSurface` — today,
  "attention present" means "the file was written", not "this model
  attends". The in-tree witness is not hypothetical: a pure-KDA/MLA
  stack (Kimi-Linear) passes completeness carrying a softmax surface
  that **zero layers read**.
- **Completeness never inspects the surface's content.**
  `graph/complete.rs:55-63`: `implies_stack_ops(DecoderStack |
  PerceptionTower)` / `implies_head_ops(Embedding | OutputHead)` — the
  check is presence-of-surface by object kind, exactly the sentence the
  candidate pins for removal.
- **The census fails open on undeclared families.** A checkpoint that
  declares an unknown `layer_types` spelling blocks admissibly
  (`graph/policy.rs:383-390` → unexpressed → census blocks → encode
  refuses). But a checkpoint declaring **no** `layer_types` at all — the
  typical pure-SSM shape — resolves every layer to `(Softmax, Full)`
  and `matches_declaration()` returns true **vacuously**
  (`policy.rs:324-330`). The plan is admissible; **encode succeeds**;
  refusal arrives only at read time, when operand closure hits
  `UnclassifiedOperand` on every `mixer.*` tensor and `MissingOperand
  {AttnQ,K,V,O}` per layer (`opplan/build.rs:222-225, 307-312`).
- **Operand closure is a read-time gate, not a write-time one.**
  `plan_component_ops` is called by exec/ops/runtime/diff — never by
  `plan_system` or `encode`. An architecture that slips the census gets
  written to disk before its operands are ever classified.
- What held: `ObjectKind::DecoderStack` is fine for an SSM (no new kind
  needed); `LayerOperator` is a real per-layer family field; a new
  family's roles are append-only with fail-closed exact-suffix
  classification (`roles.rs:240-241, 490-517`); `LayerContinuationGeometry`
  already has a `Stateless` arm and a generic `RecurrentGeometry`
  (`opplan/exec/continuation.rs:117-131`, free-form buffer shapes).

### Verdict: **schema gap — the drill's sharpest confirmation.**

Lift 1 (§17.4) is necessary, and it is a true reinterpretation: making
`attention` optional changes what its presence means in every existing
v5 graph, which is precisely the "absence indistinguishable from a
deliberate claim" argument `graph/mod.rs:74-90` uses to forbid in-place
upgrades. Cost: `GRAPH_SCHEMA` 5 → 6, re-encode. Two adjacent defects
must move with it: the vacuous-declaration fail-open (F3), and closure
ordering (F4) — encode must not admit what closure will refuse.

The sentence the spec earns from this case: **operation families are
vocabulary, not structural requirements of a model component.**

---

## Case 2 — KDA + MLA + softmax hybrid, three continuation-state kinds

*Attacks: single-state assumptions. Must remain true: one program can
declare heterogeneous state; no operation family owns continuation
globally. Not hypothetical: this is Kimi-Linear-48B, executing in-tree.*

### Evidence

- **Lift 2 is already underway in code, under its own name.**
  `opplan/exec/kv.rs:205-213`: `pub use ContinuationProvider as
  KvState;` — "`KvState` described the runtime model when every layer
  kept rows. It no longer does … it goes away at STATE-CONSOLIDATE,
  when `ContinuationState` becomes the authoritative seam and KV
  becomes its projection." The trait already carries
  `prepare_continuation(&[LayerContinuationGeometry])` and
  `recurrent_state(layer)`, refusing by name
  (`ContinuationError::RecurrentUnsupported`).
- **But the state schema cannot yet describe two of the three kinds.**
  `plan_continuation_geometry` (`continuation.rs:186-206`) refuses KDA
  ("no state precision is declared for this operator — refusing to
  choose one") and MLA ("continuation planning has no geometry for it
  yet — refusing to invent one"). Only `GatedDelta` (with a declared
  `state_dtype`) reaches `Recurrent`. These refusals are honest — and
  they name the missing **schema facts**: a per-operator state dtype
  for KDA, a latent-KV geometry for MLA. `RecurrentStateDtype` has one
  variant (`Float32`).
- **`plan_kv_geometry` panics on recurrent layers** (`kv.rs:60-80`),
  with two production callers that would hit it on a loaded hybrid:
  LQL `STATS` (`larql-lql/src/executor/vindex3.rs:256`) and inference
  `EXPLAIN` (`explain.rs:142`).
- **The deletion invariant does not hold for KDA/MLA execution today.**
  The branch condition is clean — container-typed on the persisted
  `operator: "kda" | "mla"` (`kimi_source.rs:141-151`), no
  model_type/family dispatch anywhere in opplan/plan. But the executed
  body is a family-shaped loader that bypasses
  `ComponentOpPlan`/`OperandRole` entirely: hard-coded HF tensor
  spellings (`kimi_source.rs:297-370`), `MLA_KV_A_NORM_EPS = 1e-6`
  ("The graph carries the config value; **this one fact it cannot
  carry**"), `MLA_CACHE_POSITIONS = 64`, a `first_k_dense_replace=1`
  panic message, `kimi_*` Metal kernels — and it is reachable only from
  tests. The plan-driven path holds the invariant by **refusing** both
  operators (`prepared.rs:408-434`, `LayerOperator::has_executor()`).
- **A silent-reinterpretation vector on the policy itself.**
  `AttentionLayerPolicy.operator` carries `#[serde(default)] =
  Softmax` (`policy.rs:235`): a writer that omits the key reinterprets
  every layer as softmax, silently.
- What held: `LayerPlan.attention` is a per-layer choice ("the op is a
  choice, not a shape"); `LayerAttention` grew three times additively
  (untagged, skip-if-none); operator detection at plan time is operand
  evidence (`KdaDtBias` present → Kda), not names; heterogeneous state
  per layer is exactly what `LayerContinuationGeometry` models.

### Verdict: **lift 2 confirmed; one uncarriable fact found.**

Schema gaps, all scoped: (i) the state schema must be able to declare
KDA state precision and MLA latent-KV geometry — this *is* lift 2, and
the implementation has already named its own consolidation point;
(ii) `MLA_KV_A_NORM_EPS` is a judged semantic the container **cannot
carry** — a per-op norm-epsilon override belongs on the surface, or the
deletion invariant is unreachable for MLA even in principle;
(iii) `operator`'s absent-means-softmax default should become explicit
at the same schema bump lift 1 forces. The family-shaped executor is an
implementation gap (STATE-CONSOLIDATE + a plan-driven KDA/MLA executor),
not a freeze blocker — but the two `plan_kv_geometry` panic sites are
defects regardless.

---

## Case 3 — audio → perception → projector → text, plus a drafter

*Attacks: single-component/decoder assumptions. Must remain true:
components and interfaces compose without special multimodal topology
rules.*

### Evidence

- **As stated, this case passes today.** The composition is one
  `Perception` component (audio tower + adapter, `Modality::Audio`
  exists), one `PrimaryText` target, one `Drafter` wired by a
  `HiddenStateEdge` — structurally the in-tree Glimmer system with the
  modality swapped. Roles, object kinds and the edge cover it; nothing
  needs reinterpretation.
- **The ceilings sit just past it, and they are named.**
  `ComponentRole` is a closed three-variant enum with a binary
  derivation (`is_drafter = declared_taps.is_some()`,
  `build.rs:283-296`) — there is no fourth participant (a reranker, a
  second text model, an MTP head; the builder already refuses MTP
  honestly via `COMPONENT_EXTERNAL_NAMESPACES`, `build.rs:170-172`).
  Six call sites resolve "the text model" by `find(role ==
  PrimaryText)` — first-match semantics that go quietly wrong, not
  loudly, the day two text-shaped components exist (`plan/mod.rs:425`,
  `capability.rs:285`, `encode/mod.rs:330`, `represent/mod.rs:226`,
  `sensitivity.rs:106`, `consequence.rs:378`).
- **There is exactly one species of edge.** `HiddenStateEdge`
  (`edge.rs:13-27`) has five fields and no kind discriminator;
  `wire_edge` (`build.rs:1032-1052`) requires the producer to be
  `PrimaryText`, hard-codes the consumer object to `FeatureProjector`,
  and refuses two candidate producers ("refusing to guess").
  Perception is wired by object ownership, not by any edge.
- **Dispatch is typed, with three surviving string sites** at the
  evidence→graph boundary (tensor-prefix classification,
  `build.rs:201-211, 254-264`), capability-subject fallbacks
  (`"vision_config" / "audio_config" / "text" | "target"`,
  `capability.rs:190-200`), and a component-id prefix match in verify
  (`semantic.rs:114-116`). The de-stringification was deliberate and is
  documented (`capability.rs:133-153`); these are the residue.
- **Wire posture:** every graph enum is closed with no
  `#[serde(other)]`, and `inspect` refuses any graph whose schema ≠ 5
  exactly, by design ("each schema step added a judged semantic fact
  whose absence is indistinguishable from a deliberate claim"). So any
  vocabulary addition — a fourth role, a second edge kind — is
  wire-breaking for older readers and costs a schema bump plus
  re-encode. That is a coherent posture; it is not the additive posture,
  and the spec should say which one the graph has.

### Verdict: **passes as stated; the composition ceiling is a scope
decision Final must make explicitly.**

For 3.0, the honest position: the component algebra is `{primary_text,
perception, drafter}` with one edge species, refusals (never guesses)
at every boundary beyond it. Generalising to N participants and typed
edge kinds is post-3.0 vocabulary — but the `find(PrimaryText)`
first-match sites (F10) should become uniqueness assertions *now*,
because they are the one place the current schema fails quietly instead
of loudly.

---

## Case 4 — unknown low-bit sparse-MoE representation, permuted experts

*Attacks: physical-format and extension assumptions. Must remain true:
new codec/layout/residency vocabulary is additive; logical identity and
execution semantics unchanged; an older reader degrades, never lies.*

### Evidence

- **The §5.7 core holds.** `encoding` is a free string in both
  authorities (`index.rs:20-38`, `object.rs:84-89`), segment dtypes are
  free strings; `inspect`/`describe`/`representations` pass unknown
  encodings through untouched (`inspect.rs:136-175` never reads
  `encoding`); every execution-side consumer is a closed match with a
  **named** `VindexError::Parse` fallthrough — the arena encoder even
  documents why ("refusing rather than binding source bytes under a
  name that claims otherwise", `represent/arena.rs:167-209`); codec
  identity skew refuses by name and revision
  (`nvfp4_pack.rs:299-320`). No panic, no serde failure, anywhere on
  this path. The older-reader behaviour is exactly the drill's target:
  *I understand the object; I do not understand representation X; I can
  preserve it; I cannot execute what only X provides.*
- **COMPACT is encoding-blind by construction** — enumerates the
  index's representations, hard-links/copies byte-identically, never
  decodes (`compact/mod.rs:42-99`). Unknown-encoding representations
  survive maintenance.
- **Two preservation defects against the candidate's own §5.7:**
  1. `compile`/bake round-trips `index.json` through `Vindex3Index`
     and re-serialises (`compile/mod.rs:63-65, 197-199`) — with no
     `deny_unknown_fields` anywhere, unknown fields are read-tolerantly
     **dropped**. §5.7 says a rewriter MUST NOT drop fields it does not
     understand. COMPACT preserves them (it copies the file); COMPILE
     destroys them. The same container, baked, silently loses
     forward-compatible data.
  2. COMPACT garbage-collects any file outside `CAPABILITY_FILES` +
     the index-referenced segments — reported in `dropped`, not silent,
     but a new sidecar kind (a permutation table, a residency-facts
     file) would be collected unless registered. §5.7's "MUST NOT
     silently discard" and this behaviour need to be reconciled in one
     direction or the other.
- **Per-expert permutation has no schema home.** Nothing positional
  exists on `RepresentationEntry`, `Representation`, `SegmentHeader` or
  `SegmentTensor`; the only indirection type,
  `ProjectionAddressing::Table`, is deliberately unserialisable
  (`physical.rs:577-585`) and derived from tensor-table offsets at bind
  time. An optional permutation field is **additive on the wire**; the
  execution-side `ExpertFormat` enum and its match arms grow with it.

### Verdict: **passes — the additive claim is real — with two named
conformance defects and one missing (additive) vocabulary item.**

This case proves the argument that carried residency out of the Final
gates: representation, codec, layout and residency vocabulary genuinely
are additive 3.x evolution. The defects it found are implementation
bugs against the spec's own compatibility rules, and they are cheap:
make COMPILE preserve unknown index fields (raw-JSON merge or a
flattened extras map), and either extend COMPACT's preservation rule or
amend §5.7 to "discard only with report".

---

## Findings ledger

| # | Finding | Class | Where it lands |
|---|---------|-------|----------------|
| F1 | `ExecutionSurface.attention` mandatory + fabricated; presence means "file written", not "model attends" | **schema gap** | lift 1 · GRAPH_SCHEMA 6 |
| F2 | Completeness derives from object kinds, never surface content (`complete.rs:55-63`) | **schema gap** | lift 1 (the pinned sentence, confirmed in code) |
| F3 | No `layer_types` ⇒ every layer resolves `(Softmax, Full)`; `matches_declaration` vacuously true; census fails **open** | **schema gap** (fail-open) | lift 1: census must fail closed on undeclared families |
| F4 | Operand closure runs post-encode only — encode admits containers closure will refuse | defect (ordering) | closure (or its census projection) at encode time |
| F5 | State schema cannot declare KDA state precision or MLA latent-KV geometry; only GatedDelta expressible | **schema gap** — **closed** | lift 2: `LayerContinuationGeometry::LatentKv` (one operator-defined row per position, MLA's compressed latent — not a K/V pair, so `kv()` still answers `None`) and a KDA arm that answers `Recurrent` at the precision the reference computes at, the judgment living once in `exec::kda::state_geometry` exactly as Mamba2's does |
| F6 | `MLA_KV_A_NORM_EPS` — a judged semantic the container **cannot carry** (executor-resident constant) | **schema gap** — **closed** | `MlaSurface.kv_a_norm_eps` → `MlaOp` → the executor, sourced from `ModelArchitecture::mla_kv_a_norm_eps` (an architecture fact, not a config one: no checkpoint declares it). Additive within v6. A container carrying none is REFUSED at preparation rather than lent the layer's `rms_norm_eps`; the family-shaped loader now reads the graph too |
| F7 | `AttentionLayerPolicy.operator` `serde(default)` = Softmax — absence silently reinterprets | **schema gap** (silent default) | explicit operator at schema 6 |
| F8 | `plan_kv_geometry` panics on recurrent layers; two production callers (LQL STATS, EXPLAIN) | defect | migrate callers to `plan_continuation_geometry` |
| F9 | KDA/MLA execution is a family-shaped, test-only loader bypassing plan/roles; plan path refuses honestly | implementation gap — **closed** | `PreparedAttention::{Kda,Mla}` bind by `OperandRole` and execute on both traversals; `LayerOperator::has_executor()` true for both. The device loader remains as the ACCELERATED arm, not as the definition |
| F10 | `find(PrimaryText)` first-match sites — quiet wrongness when two text components exist (three of the six reported sites were already plural-safe `filter`s) | defect — **closed** | `SystemGraph::primary_text_component`: unique or refused naming the candidates; encode errors on ambiguity, capability/alias resolution yields none rather than first |
| F11 | One edge species; producer must be PrimaryText; consumer hard-coded FeatureProjector; two producers refuses | scope decision | declare the 3.0 component algebra; typed edge kinds post-3.0 |
| F12 | Closed graph enums + exact-schema refusal: every vocabulary addition is wire-breaking + re-encode | posture decision | spec must state the graph's strict-versioning posture explicitly |
| F13 | COMPILE drops unknown `index.json` fields (struct round-trip); COMPACT preserves them | defect vs §5.7 — **closed** | flattened preservation map on `Vindex3Index`; every struct round-trip now carries unknown fields; gated by `bake_preserves_unknown_index_fields` |
| F14 | COMPACT collects unregistered sidecar files (reported, not silent) | defect vs §5.7 wording — **closed** | §5.7 amended: reported discard of unreferenced files is sanctioned; silent discard forbidden |
| F15 | Per-expert permutation has no serialisable home; `ProjectionAddressing` deliberately runtime-only | additive-safe (missing vocab) | optional permutation field, 3.x |
| F16 | `OperandRole` append-only, exact-suffix, fail-closed; codec fallthroughs all named refusals; COMPACT byte-blind | held | the healthy core the freeze protects |

---

## Verdict

**The ontology does not freeze yet — and the drill did exactly what it
was built to do: it made the schema flinch in named places, and nowhere
else.**

The flinches are not a scattering. Every schema gap lands inside the
two lifts the candidate already pinned (F1–F3, F5–F7 are lifts 1 and 2
made concrete, down to field names and the schema-6 bump), plus one
uncarriable fact (F6) the lift must absorb, one ordering defect (F4),
and two conformance defects against §5.7 that are bugs, not design
(F13, F14). Case 3 passed as stated with its ceiling named as a scope
decision; case 4 proved the additive claim that moved residency out of
the Final gates.

What this changes in §21: gate 7 (the ontology lift) is no longer
"pinned design" — it is **drilled, confirmed necessary, and fully
enumerated**. The freeze checklist after this drill:

1. Lift 1 with F3 fail-closed and F4 ordering — `GRAPH_SCHEMA 6`,
   re-encode.
2. Lift 2 absorbing F6 and F7; retire the F8 panic sites. *(Done: F7
   and F8 at schema 6; F5, F6 and F9 in the lift-2 pass — see the
   closure log.)*
3. F13/F14 preservation fixes (small, immediate).
4. Declare the 3.0 component algebra and the graph's strict-versioning
   posture (F11, F12) in the candidate text.
5. Then the pre-existing gates: shape convergence, RFC-2119 pass,
   `vindex-core`, E8, M4.

Nothing on that list is a discovery about *what VINDEX3 is*. That
question did not flinch.

For the 3.0 release history, the drill's one-sentence result:

> **The ontology drill found defects in how VINDEX3 expressed two
> already-known abstractions; it found no missing abstraction required
> to describe the four hostile architectures.**

## The first live witness

**2026-08-30, same day** — `AntonV/mamba2-780m-hf` (the HF-format
conversion of `state-spaces/mamba2-780m`; the original repos ship
`pytorch_model.bin` only), run header-only through
`scripts/hf_metadata_checkpoint.py` → `inspect-hf` → `vindex3 plan`:

- **Finding zero, before the ontology was even consulted:** the
  transformers-written `config.json` carries a bare `Infinity`
  (`time_step_limit`), which the RFC-strict reader refuses at parse. A
  judged interpretation of non-finite bounds at exactly one boundary —
  the NoPE zero-theta discipline — is part of the schema-6 bring-up.
- **F1/F3 observed live, not just predicted:** the generic fallback
  resolved the pure-SSM stack as 48 full-attention RoPE layers with
  invented 8/4 head geometry (`generic_fallback: true` flagged; config
  says `num_heads: 48`). The census failed open exactly as the drill
  said.
- **Admission still refused — the layered defence held:** `plan: 20
  representable, 19 blocking — not admissible`. Eighteen unconsumed
  Mamba2 keys (`state_size`, `num_heads`, `expand`, `chunk_size`,
  `conv_kernel`, `n_groups`, `time_step_{floor,limit,max,min}`,
  `time_step_rank`, `rms_norm`, `layer_norm_epsilon`,
  `residual_in_fp32`, `rescale_prenorm_residual`, `use_bias`,
  `use_conv_bias`, `hidden_act`) plus `target.execution_surface`
  incomplete (norm placement underivable from `backbone.layers.N.norm`).
  Those nineteen findings **are** the schema-6 worklist for the pure
  case, itemised by the format itself. F3's sharpened statement: the
  fail-open is real, but a checkpoint only reaches encode through it if
  every declared key also grades consumed — Mamba2's do not.
- The graph builder placed `target.decoder_stack` (432 tensors, 1.4 GB),
  `target.embedding` and `target.final_norm` cleanly from the tensor
  estate — the object ontology needed nothing new, as case 1 predicted.
- Logistics for the A/B: `state-spaces/mamba2attn-2.7b` also ships
  `pytorch_model.bin` only — the hybrid witness needs a
  bin→safetensors conversion before admission.

## Closure log

- **2026-08-30, finding zero closed:** the judged non-finite boundary
  (`larql-models/src/config/nonfinite_json.rs`) — both config-parse
  sites now route through one function that quotes Python's bare
  `Infinity`/`-Infinity`/`NaN` literals as the strings they spell,
  never impersonating them with a fabricated float; strict JSON pays
  nothing, genuinely malformed JSON keeps the strict error. The raw
  witness stub parses and produces the identical 19-blocking verdict,
  with `time_step_limit` declared as `[0.0, "Infinity"]` in the
  findings.

- **2026-08-30, same day:** F13 closed (flattened preservation map on
  `Vindex3Index`, gated by `bake_preserves_unknown_index_fields`); F14
  closed (§5.7 amended to reported-discard); F10 closed
  (`SystemGraph::primary_text_component` — unique or refused naming the
  candidates; three of the six reported sites were already plural-safe
  `filter`s). The schema-6 delta was defined once, in §17.4 of the
  candidate, with the severe acceptance rule and real validation
  witnesses: `mamba2-780m` (pure SSM), `mamba2attn-2.7b` (the
  surfaces-follow-program A/B), `Falcon3-Mamba-7B-Instruct`
  (product-scale), Kimi-Linear (three-state hybrid).

- **2026-08-30, schema 6 landed — the witness admits.** Lift 1
  implemented in one intentional break (`GRAPH_SCHEMA` 5 → 6):
  `ExecutionSurface.attention` **and `.ffn`** optional, present iff the
  program runs them (the FFN turned out to be F1's twin — a mixer-only
  layer has no FFN either); explicit per-layer `operator` (F7); the
  census fails closed on generic-plus-silence — no registered family, no
  per-layer declaration, no declared attention shape (F3, scoped so a
  declared `num_attention_heads` still counts as the program declaration
  it is); operand closure enforced at encode, a failing encode removed
  (F4); `NormPlacement::PreMixer` and the nine Mamba2 operand roles,
  operator-gated. The F8 panic sites are retired (LQL `STATS` and
  `EXPLAIN` read the layers, never `plan_kv_geometry`). Witness result:
  `mamba2-780m-hf` plans **0 blocking** (was 19), encodes with 434
  operands closing, and opens through ordinary LQL with the source
  checkpoint deleted — `48 Mamba2 recurrent`, no attention surface, no
  FFN surface, `INFER` refusing by name until the executor lands. Lift
  2's state-schema facts (F5, F6) remain open, additive within the v6
  span; the Mamba2 continuation arm refuses state precision exactly as
  KDA's does, for the same reason.

- **2026-08-30, later the same day — the witness generates.** The
  generic Mamba2 executor (`exec/mamba2.rs`, transcribed stage-by-stage
  from the reference's `torch_forward`, fp32 state as a judgment
  transcribed from the reference's own `.float()` casts) runs the mixer
  through the ordinary prepared-operands / prefill / decode /
  continuation path — no family loader anywhere. Rung-4 parity against
  the banked fp32 oracle: short 5+32 max|Δ| 6.9e-4 · medium 29+32
  5.2e-4 · long 300+32 7.6e-4, argmax exact at all 430 positions,
  all three 32-token greedy trajectories token-for-token — the bound
  being pure fp32 reassociation over the oracle's own 2.1e-4
  step-vs-scan floor. The canonical KV provider grew its SERVE-HYBRID
  half (recurrent buffers beside the cache, allocated from the plan's
  declared geometry, refusing by name in both remaining directions), and
  ordinary LQL then generated the reference's greedy continuation
  word-for-word from the container alone, source checkpoint deleted:
  the deletion invariant, visible. In-tree gates: the miniature witness
  executes (bitwise determinism, prefix equivalence across the
  batch↔decode seam), the hand-computed recurrence, dt-clamp bounds,
  conv causality by impulse, and the mixed KV+recurrent provider test.

- **2026-08-31, lift 2 — the state schema stops refusing.** The two
  operators the drill's Case 2 was written around now execute through
  the ordinary plan path, and what unblocked them was not a kernel:
  `exec::kda` and `exec::mla` had been parity-proven against banked
  oracles since P3d. It was the two facts F5 and F6 named.
  - **F5, KDA's state precision.** No checkpoint declares one — Kimi
    Linear's `linear_attn_config` carries head count, head dim and conv
    width and nothing else — so the schema had nothing to carry and the
    planner refused rather than pick. The reference does declare it, in
    the only way a reference can: `fla`'s `naive_recurrent_kda`, which
    the checkpoint's own `modeling_kimi.py` calls, holds the state as
    `torch.float32` and casts q, k, v, g and beta into it every step
    (transcribed and sha-pinned at `scripts/kda_reference.py:130`). The
    judgment is therefore a transcription, and it lives in exactly one
    place — `exec::kda::state_geometry` — on the same footing as
    Mamba2's. Four buffers: the `Dk × Dv` matrix per head and three
    convolution windows, `kernel - 1` deep.
  - **F5, MLA's latent cache.** A third state species, because it is a
    third fact: `LayerContinuationGeometry::LatentKv` is ONE row per
    position of an operator-defined width, neither a K/V pair (which
    would claim two rows the model never keeps) nor a fixed-size
    recurrence (which would not grow with the prefix). `kv()` answers
    `None` for it, so every KV-only provider still fails closed; the
    provider seam grew `latent_state`, required and fail-closed, the
    same contract `recurrent_state` holds.
  - **F6, the uncarriable epsilon.** `kv_a_layernorm` runs at
    `KimiRMSNorm`'s class default `1e-6` while the layer's own norms run
    at `rms_norm_eps` `1e-5` — an architecture fact no config states,
    now read in `ModelArchitecture::mla_kv_a_norm_eps` (default `None` =
    unjudged, never "use the layer eps") and carried
    `MlaExecution` → `MlaSurface` → `MlaOp` → executor. Preparation
    REFUSES a container that carries none, before binding a single
    matrix. `MLA_KV_A_NORM_EPS` is gone from `kimi_source.rs`; that
    loader reads the graph like every other field it consumes.
  - **F9, the family-shaped executor.** `PreparedAttention::{Kda,Mla}`
    bind through `OperandRole` and run on both traversals, so
    `has_executor()` is true for both and the only operator left
    answering `false` is `Recurrent` — the one whose family was never
    identified, which is a different fact.
  - **A coincidence the fixture caught.** KDA's executor read the gate
    factorisations' inner rank as `head_dim`. That is true on both
    observed checkpoints (128 = 128) and true for no stated reason;
    `KdaOp::gate_rank` existed precisely because the config declares no
    such field. A miniature whose widths are all distinct (rank 4,
    head dim 3) failed on it immediately. `KdaWeights::gate_rank` is now
    carried from the op, and every real-weight fixture reads its own
    `f_a_proj`/`f_b_proj` shape rather than assuming the two agree.
  - **The witness, in CI:** `opplan/tests/kda_mla_exec.rs` — a
    KDA/MLA-alternating miniature admits with both operators declared
    per layer, encodes with closure held over both operand programs,
    declares both state species, and executes with the single-token
    decode path bitwise-identical to batch prefill. The pre-flight
    continuation check is now driven by the declared region rather than
    by "is this layer softmax", which is what an MLA layer needs: it
    keeps a cache and no recurrence, and the old form would have refused
    a state the provider was holding correctly.
