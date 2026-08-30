# VINDEX3 — Model-System Container Format

Status: **living spec** — the implementation-tracking companion to the
[VINDEX3 3.0 Candidate Specification](../crates/larql-vindex/docs/vindex3-format-spec.md),
which owns the normative container model and the contract stack. As of
2026-08-30: §1–§7 are implemented and gated; §8's execution contract is
implemented through 5b-3 (surface, operand closure, reference executor,
golden parity, causal controls, `larql vindex3 exec`), and the runtime
above it landed through VI3-SERVE-1 with chat/responses arms
(`docs/vindex3-runtime.md`) — five model families encode, verify and
execute end to end. The experimental programme, gates and run order live
in `docs/vindex3-experiments.md`; the LYRW v2 routed-layer physical
layout lives in `docs/lyrw-v2.md` and is incorporated by reference as one
segment codec (candidate spec §5.6, §6).

> Older `spec §N` citations in `format/vindex3/{index,lyrw2,…}` code
> comments refer to the pre-G2 draft of this spec; where they conflict,
> this document is authoritative and those comments should migrate.

---

## 1. What VINDEX3 is

A modern model release is not a weights file. It is a **system**: a target
model, a perception tower physically embedded in the same checkpoint, a
speculative drafter shipped as a sibling artifact consuming the target's
hidden states through a declared tap interface, several quantization
profiles of the same logical operands. GGUF answers *"how do I store and
run these tensors?"* VINDEX3 answers:

> What are these model objects, what operations can consume them, which
> representations are equivalent, which parts should be resident, and what
> future computation will need them?

The pipeline is deliberately compiler-shaped:

```text
HF artifacts
    ↓
inventory            (G0 — source semantics, two authorities)
    ↓
representability     (G1/G2 — type checking; fail-closed)
    ↓
system graph         (G2 — semantic IR)
    ↓
physical encoding    (G3 — materialisation)
    ↓
verification         (G4 — Declared ≡ Resolved ≡ Graph ≡ Encoded)
    ↓
execution            (G5 — from the encoded description alone)
```

### 1.1 Principles

1. **Single semantic authority.** The system graph is built once
   (`graph::build_from_inventories`) and every downstream consumer — the
   planner, the encoder, eventually the executor — consumes *it*, never a
   private re-interpretation of the checkpoint. "Representable" has
   exactly one definition: *the graph builder placed it.* There is no
   capability registry to drift out of sync.
2. **Fail-closed.** A config key nobody has judged, a tensor group no
   placement rule owns, an interface whose producer cannot be resolved
   unambiguously — each blocks the plan. Unjudged is not admissible;
   ambiguity is refused, never guessed.
3. **No silent conversion.** Regions are placed as their producer wrote
   them; generation boundaries error precisely rather than convert; a
   defaulted value never impersonates a declared one.
4. **No model-family branches.** Roles, placements and policies derive
   from evidence (declared interfaces, shape arithmetic, component
   topologies), not from `model_type` matching. Glimmer went from 51
   blocking findings to zero without a single Muse-specific branch; that
   property is the point and every future architecture must preserve it.

### 1.2 Identity distinctions (load-bearing)

```text
artifact       ≠ component
tensor name    ≠ logical object
interface      ≠ implementing tensor
NoPE           ≠ rope(theta = 0)
logical object ≠ physical representation
representable  ≠ "parser consumed the key"
```

---

## 2. The G-ladder

| Rung | Question | Gate |
|---|---|---|
| G0 | What does the source declare? | `larql inspect-hf` emits the inventory |
| G1 | Can the schema describe it? | `larql vindex3 plan` — typed findings, non-zero exit on blockers |
| G2 | Generalise the schema until reality fits | plan over the artifact set: `blocking = 0, mismatched = 0, unknown = 0` |
| G3 | Materialise the graph | `encode` then `inspect` reconstructs the system **solely from the container** |
| G4 | Prove source ≡ encoded | four-authority comparison + payload-hash equality |
| G5 | Execute from the encoded description | forward pass with zero architecture branches — §8; implemented through 5b-3 |
| G6 | Drafter parity | speculative execution discovered from the `HiddenStateEdge` — G6d (GPU-lowered plan) landed |
| G7 | Performance baseline | reference numbers on the target hardware class — recorded in `ROADMAP_STATUS.md` |
| G8 | Alternate physical/execution plans | LARQL-specific layouts/prediction over the same logical system |

G8 must not contaminate G0–G5: optimisation is an *alternate plan over the
same graph*, never a schema change.

---

## 3. G0 — the architecture inventory

`larql inspect-hf <dir>` → `ArchitectureInventory`
(`larql-models/src/inventory/`), schema version `INVENTORY_SCHEMA = 2`.

Two authorities, side by side; disagreement is the instrument:

- **`config_keys`** — every leaf of `config.json`, flattened to a dot
  path, classified `consumed` / `metadata` / `unconsumed`.
- **`resolved`** — what this build's detection would actually run: the
  full per-layer table (span, window, position policy, head geometry).

Classification honesty rules:

- Name-registry credit applies only inside containers the parser itself
  recurses into (`text_config`, `rope_scaling`, …). A leaf under any other
  container is never credited by name — `vision_config.hidden_size` shares
  a name with a consumed key but the text parser does not read it.
- Nested components (`vision_config`, any `*_config` sibling) are read by
  a generic component-topology reader whose **recorded reads are the
  consumed-credit**: a key is consumed iff a read actually stored it. The
  reader and the classification cannot diverge because there is only one
  artifact.
- A parser-sync test scans `detect/parser.rs` for key accesses and fails
  when the registry does not know one. The registry cannot rot silently.

The inventory also carries: identity, detection outcome (including the
generic-fallback flag — a model serving through the generic path with
unconsumed keys is the loudest red flag the report can raise), nested
component topologies, declared interfaces, and a tensor inventory read
from safetensors *headers only* (no payload I/O).

### 3.1 Position policy

`PositionPolicy = Rope { theta } | None`. Absence of positional rotation
is an intentional per-layer execution property, not a parameter value.
The HF `layer_rope_theta` spelling uses `0.0` as a NoPE sentinel; that
sentinel is interpreted at **exactly one boundary**
(`PositionPolicy::from_declared_theta`) and nowhere else. No zero theta
may circulate internally — `1/0^(i/d)` is degenerate, and a resolver
storing `0.0` where it means "none" has re-invented the magic value the
type exists to remove.

---

## 4. G1/G2 — the representability plan

`larql vindex3 plan <artifact>…` → `SystemPlan`
(`larql-vindex/src/format/vindex3/plan/`), schema `PLAN_SCHEMA = 2`.
Artifacts are checkpoint dirs and/or saved inventory JSONs, treated as one
model system. Exit is non-zero when the plan is inadmissible.

Findings are typed twice:

- **category** — `representable` / `mismatched` / `unrepresented` /
  `interface`;
- **semantic class** — `execution_semantic` / `tensor_semantic` /
  `interface_semantic` / `metadata_only` / `ignored_safe` / `unknown`.

Rules:

- `consumed` is never trusted. Comparators re-read declared values from
  the config facts and diff them against resolution (scalar topology,
  uniform rope θ in parser-precedence order, per-layer
  `layer_rope_theta` against per-layer policies, `layer_types`
  interleave). Equal → representable; different → `mismatched`, blocking.
- Unconsumed keys are graded by a semantic registry of *known HF field
  names*. A name the registry has never seen grades `unknown` — and
  blocks. Keep this painful; a "probably harmless" bucket is exactly how
  silent semantic loss returns.
- `ignored_safe` ships empty. Every future entry needs a per-entry
  justification comment.
- Tensor/topology/interface representability comes from the graph builder
  (§5): placed objects are representable with their graph ids as proof;
  `unplaced` groups and `unresolved_interfaces` are blocking findings.
- The plan embeds the built graph — the proof object G3 consumes.
- Verdict: `admissible ⇔ blocking == 0`.

Standing tripwire: a clean dense model (Llama-shaped fixture) plans
**admissible**. Its job is to stay that way; a regression that starts
blocking clean dense models fails there first.

---

## 5. G2 — the system graph

`larql-vindex/src/format/vindex3/graph/`, schema `GRAPH_SCHEMA = 1`.

```text
SystemGraph
├── components: [Component]        id, role, source_artifact,
│                                  num_layers, hidden_size,
│                                  attention: [AttentionLayerPolicy]?
├── objects:    [LogicalObject]    id, component, kind,
│                                  source_bindings: [SourceBinding],
│                                  representations: [Representation]
└── edges:      [HiddenStateEdge]  producer_component, producer_layers,
                                   consumer_component, consumer_object,
                                   block_size?
```

### 5.1 Components

Roles are **evidence-derived**: an artifact declaring `target_layer_ids`
is a `drafter`; a nested `*_config` component is `perception`; otherwise
`primary_text`. Ids are conceptual (`target`, `vision`, `draft`), with
numeric suffixes on collision — never directory names. The source
artifact is recorded for traceability, not identity.

`AttentionLayerPolicy { span: sliding|full, window, position }` per layer
is architectural liveness information: a KV planner reading it knows that
positions beyond `window` on a sliding layer are *architecturally* dead
before any semantic analysis runs. A component without per-layer
resolution (perception towers today) carries no table — absent is honest,
fabricated is not.

### 5.2 Logical objects

Kinds (architectural vocabulary, not familial): `embedding`,
`decoder_stack`, `final_norm`, `output_head`, `perception_tower`,
`perception_adapter`, `feature_projector`.

Identity is conceptual — `{component}.{kind}` — and **physical names may
bind objects but never define identity**. That is what later allows

```text
target.decoder_stack
    ├── canonical BF16
    ├── K-quant
    ├── Metal-native packed
    └── future derived representations
```

without unwinding a single id. `SourceBinding { artifact, tensor_prefix,
tensors, bytes }` carries the physical trace; `Representation
{ encoding, fidelity: canonical|approximate }` carries materialisations
(encodings are *observed* from shard headers, never invented).

### 5.3 Placement rules (the builder)

- Groups are classified by a name-fragment vocabulary (first match wins,
  specific before generic — a vision tower has `layers` segments too) and
  merged into `(component, kind)` objects with multi-binding.
- **The projector claim runs before name classification.** The consumer
  side of a declared interface is identified by shape evidence — a 2-D
  tensor of `len(taps)·hidden × hidden` (either orientation) — and claims
  every group sharing its first path segment by *structural adjacency*.
  Without this ordering, the projector's own norm
  (`encoder.output_norm_enc`) name-classifies into `final_norm`.
- The edge's producer must be **exactly one** other component deep enough
  to own every declared tap. Zero candidates or two candidates → the
  interface is unresolved, and blocks. Never guessed.
- Everything unplaced returns as data (`unplaced`,
  `unresolved_interfaces`); the planner converts it to blocking findings.

### 5.4 The edge is not the tensor

A `HiddenStateEdge` describes **logical flow** of residual states across a
component boundary. The fusion projector implementing its consumer side is
a tensor object referenced by id. Both are representable; they are
distinct facts and are never merged.

---

## 6. G3 — physical encoding (pinned design)

> Deliberately boring, deterministic, source-independent. No layout
> optimisation: one logical object → one canonical representation → one
> or more simple contiguous segments. Optimised layouts arrive at G8 as
> *additional* representations over the same logical objects.

```bash
larql vindex3 encode <artifact>… --output <container>/
```

Every mainline V3 producer funnels into this pipeline: `larql vindex3
encode` directly, and `larql extract --generation v3`, LQL
`EXTRACT … FORMAT VINDEX3` and the factory generation pin through
`encode_checkpoint`, which wraps it and adds the capability snapshot
(`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
`generation_config.json`, chat template) that keeps the container
servable (see `docs/vindex-generation-policy.md`).

The encoder consumes **the built graph** (via the plan pipeline), never a
private re-interpretation of the checkpoint, and refuses to encode an
inadmissible plan. Two semantic authorities immediately after eliminating
them would be a regression.

### 6.1 Container layout

```text
<container>/
├── index.json            sole root authority: format version,
│                         system_graph ref, object/representation
│                         directory (representation → segment files)
├── system_graph.json     the SystemGraph, verbatim
└── segments/
    ├── target.decoder_stack.bin
    ├── target.embedding.bin
    ├── target.output_head.bin
    ├── vision.perception_tower.bin
    ├── vision.perception_adapter.bin
    ├── draft.decoder_stack.bin
    ├── draft.feature_projector.bin
    └── …
```

- The graph references logical object ids and representation ids; the
  index's directory maps representation → segment bytes. Graph edges
  never reference safetensors names: **the HF checkpoint disappears as an
  authority once encoded.**
- Within a segment, tensors are addressed by a per-representation tensor
  table (name relative to the binding prefix, dtype, shape, offset,
  length), payloads concatenated in table order. Relative names are
  structural (`0.self_attn.q_proj.weight`), not artifact-global.
- Every canonical representation records a **source payload hash**
  (SHA-256, computed while copying) and the encoded segment records its
  own hash. These are G4's byte-equivalence inputs.
- `Vindex3Index.system_graph` is optional in the schema: absence means
  "no graph recorded", never "single-component assumed". A container with
  no graph but a `moe_manifest` is the other physical shape — the
  transitional bank-import layout the candidate spec names and ranks
  (§5.4 there). Absence of `moe_manifest` is likewise not evidence a
  model is dense: a graph-encoded routed MoE carries none.

### 6.2 The G3 gate

After encoding, the validation path gets **no access to the source**:

```bash
larql vindex3 inspect <container>/
```

must reconstruct — solely from the container, with no transformers
config, no HF filenames, no safetensors headers, no architecture
registry —

```text
components: target (52 layers, 6656, 39 sliding / 13 full-NoPE)
            vision (50 layers, 1536)
            draft  (5 layers, taps [1,13,25,37,49], block 16)
edge:       target.hidden[1,13,25,37,49] → draft.feature_projector
objects:    …with sizes, encodings and hashes
```

---

## 7. G4 — verification

```bash
larql vindex3 verify <artifact>… --container <container>/
```

Implemented in `format/vindex3/verify_system/`. Four explicit authorities,
compared structurally and semantically:

| Authority | Source |
|---|---|
| **Declared** | what HF said (`config.json`, shard headers) |
| **Resolved** | what LARQL interpreted (detection + policies) |
| **Graph** | the logical system constructed at G2 |
| **Encoded** | what the container actually contains |

Invariant: `Declared ≡ Resolved ≡ Graph ≡ Encoded` for everything
execution-semantic. Example instance:

```text
Declared  layer_rope_theta[3] = 0
Resolved  PositionPolicy::None
Graph     target.attention[3].position = none
Encoded   target.attention[3].position = none        PASS
```

Two equivalence claims are kept **separate**, because each can fail while
the other holds:

- **semantic equivalence** — the four-authority structural comparison,
  checked as three pairwise links (Declared ≡ Resolved via the plan's value
  comparators; Resolved ≡ Graph by independent re-derivation, not a call
  into the builder's own mapping; Graph ≡ Encoded element-by-element against
  the persisted graph) so a failure names the two authorities that disagree;
- **byte payload equivalence** — per representation, **both ends re-hashed
  at verify time**: the source payload hash (recomputed from the checkpoint,
  in segment order), the encoded payload-region hash (recomputed from the
  container) and the hash the directory recorded must all agree. A drifted
  checkpoint therefore fails differently (`source ≠ recorded`) from a
  corrupted container (`encoded ≠ recorded`).

"Tensor count before == after" is not verification and does not appear in
this format. The tensor *table* (name, dtype, shape, offset, length) is
compared entry by entry instead, because a dtype or shape relabel over
identical bytes is invisible to every hash in the container — the gate
includes a mutation test for exactly that case.

---

## 8. G5 — execution contract (5a, 5b-1, 5b-2 A–C and 5b-3 implemented)

Execute the system **using only semantic information present in the
container**. The executor owns *generic operations* — embedding, norms,
attention, RoPE/NoPE, sliding/full span, FFN activations, linear,
perception transformer, projection, drafter verification. It must not
own architecture:

- no `if family == X` branches — span/window/position come from
  `AttentionLayerPolicy`, scaling scalars from the persisted config
  surface;
- no layer-pattern arithmetic (`layer % 4 == 3`) — the interleave is
  already unrolled into the per-layer table;
- no hardcoded tap constants — the drafter discovers
  `target.hidden[…] → draft.feature_projector` from the
  `HiddenStateEdge`, block size included.

**The deletion invariant.** Removing the original checkpoint,
`config.json`, HF model type and architecture name must not change
execution. The runtime sees `container → system graph → operation plan →
generic kernels`, nothing else.

### 8.0 The execution surface (5a — implemented)

`Component` answers *what part of the system this is*; its
`ExecutionSurface` (`graph/surface.rs`, graph schema v2) answers *what
the generic operations need to run it* — grouped by op (`attention`,
`ffn`, `norm`, optional `head`), every value **fully resolved** at
build/judgment time: an absent `hidden_act`, a shared post-norm epsilon
and canonical 1/√d scaling are decided once, by the same detection
surface the serving path runs, and persisted. An executor reads; it
never defaults.

The **completeness contract** derives from the operations a component's
objects imply, never from what any architecture declares:

```text
DecoderStack / PerceptionTower object  →  attention + ffn + norm surface
Embedding / OutputHead object          →  head surface
```

It is enforced twice: at plan time an incomplete surface is a blocking
finding (encode refuses — a missing `intermediate_size` is a schema
refusal, not a mid-forward-pass discovery), and on the container
`larql vindex3 inspect --execution-complete` re-answers the question
from the persisted graph alone. Perception surfaces are derived from the
nested config reading plus **tensor evidence** (FFN gating from the
presence of gate weights under the tower — whoever ships them), and a
per-layer head-geometry variation is refused as a schema gap, never
averaged away. G4 covers the surface automatically: link 2 re-derives it
from today's source, link 3 compares it element-wise against the
persisted graph.

Two facts were moved out of family defaults by the first real
four-norm stack (whose generic fallback claimed `post_norms = false`
while shipping four norms per layer):

- **Norm placement is operand evidence** (`graph/roles.rs`): the norm
  roles present across a stack's layers decide `PreOnly` vs `PrePost`;
  a mixed estate refuses. When an execution-semantic structural fact can
  be established from unambiguous operand topology, tensor evidence
  outranks a family/default assumption. Count is not semantics —
  placement is: under two norms, `post_attention_layernorm` *is* the
  pre-FFN norm; under four it normalises the attention output.
- **Attention output gating is a generic primitive**
  (`AttentionSurface.output_gate` + `OperandRole::AttnOutputGate`), not
  a family feature. Its judged semantics stay `None` until a model's
  gate behaviour is actually judged — a stack shipping the operand
  meanwhile fails operand closure with the primitive named.

### 8.1 The G5 gate — five proofs

1. **Text forward from graph semantics** — the target component executes
   from the container, output-identical to the checkpoint-driven path.
2. **Position/span behaviour exclusively from `AttentionLayerPolicy`** —
   NoPE/global vs RoPE/sliding per layer is a table read; mutating one
   row of the persisted table provably changes execution (the positive
   control), and no other source of that fact exists to consult.
3. **Perception from the component, not family wiring** — the vision
   tower and adapter run because a `perception` component with its
   objects is present, not because the runtime knows Glimmer has one.
4. **Drafter capture exclusively from the `HiddenStateEdge`** — which
   hidden states are tapped, and the block protocol, are edge reads;
   editing the edge redirects the capture.
5. **Lookup by logical object id only** — operand resolution goes
   `object id → representation → segment`; no original HF tensor name
   appears anywhere on the execution path (they were already stripped at
   encode).

When the five hold, an architecture is an **instance** the container
describes, not an implementation the executor contains: an architecture
is *supported* when its execution semantics can be expressed in the
model-system IR — not when its name has been added to the runtime.

### 8.2 Operand closure and the operation plan (5b-1 — implemented)

G4 proves *consistency* across the four authorities; it cannot prove
*sufficiency* — all four can faithfully agree on an under-specified
execution model. The physical operand estate is the independent
structural witness that closes that hole:

```text
consistency  (Declared ≡ Resolved ≡ Graph ≡ Encoded)
+ operand closure  (every executable tensor ↔ a generic op)
= execution sufficiency
```

`larql vindex3 ops <container> --component <id>` (`format/vindex3/opplan/`)
emits the component's **generic operation program** solely from the
container — per-layer norm sites, attention geometry/scale/span/position
(from the policy table, never an index pattern), FFN shape, embedding
and head ops, every operand a logical-object id plus segment-relative
tensor. The closure gate requires, before any plan exists:

- every stack tensor classifies into an [`OperandRole`] — an
  unclassified operand blocks;
- every operand's op is declared by the surface — an operand implying an
  absent op blocks **naming the required primitive** (before its gate
  was judged, the real Glimmer target refused on exactly
  `self_attn.gate_proj` × 52 layers: `required primitive: attention
  output gate` — the discovery that forced the primitive into the IR);
- every op the surface implies has its operands, with the stored shapes
  the surface's geometry states;
- per-layer accounting is total (`N/N operands accounted`) — counts or
  family defaults are not sufficient.

**Sealed on the real pair**: the target closes at 52 layers × 12/12
with zero defects once its gate semantics were judged from the upstream
reference implementation (`sigmoid(gate_proj(attention_input))`
multiplied into the aggregated head output before `o_proj`), and the
assistant closes at 5 × 11/11 (two-norm placement, per-head QK-norms,
gated FFN). The NoPE interleave reads directly off the persisted table
— layers 3, 7, … 51 emit `Full / position None` with no layer
arithmetic anywhere.

Two judged facts arrived with the gate, both preserved unfolded because
strict parity is next:

- **Query and score scales are separate ops** — the declared
  `qk_scale_factor` multiplies the normalised query before position
  encoding; the canonical `head_dim^-0.5` is a score-time multiply.
  Folding them is algebra-equivalent but not fp-equivalent, and moving a
  multiply across RoPE/matmul is exactly the silent normalisation 5b-2
  would otherwise surface as a parity mystery.
- **Parameter-free QK normalisation** is a judged semantic no tensor
  evidence can reveal: the reference normalises Q and K while the stack
  ships no norm weights. It lives on the surface as
  `parameter_free_qk_norm`, distinct from weighted QK-norm.

### 8.3 The staged proof (Stages A–C implemented)

**5b-2 Stage A — implemented** (`opplan/exec/`): a reference executor
that consumes a `ComponentOpPlan` and nothing else — naive f32 loops
(deliberately sharing no arithmetic with `larql-compute`), operands
loaded only through the closure-verified
`object → representation → segment` path, every judged enum matched
exhaustively. Gate: on a dense F32 fixture, container-driven execution
matches the checkpoint-driven production forward **layer by layer**
(observed reassociation-only margins ~7e-8 hidden / ~2.4e-7 logits,
asserted at 100× the noise floor), with a corrupted-operand negative
control and a non-degeneracy check (independent arithmetic must not
agree bit-for-bit). The judged Glimmer program shape — gate, weighted +
parameter-free QK norm, four norms, hybrid span, both softcaps —
executes finite end to end; its numerical reference is Stage B.

**5b-2 Stage B — implemented.** The judged semantics against an
independent golden implementation (`exec/tests/golden.rs`): a
deliberately literal oracle sharing *nothing* with the execution stack —
its own safetensors byte reader, plain hardcoded arithmetic, no plan, no
surface, no shared enums — over a miniature Glimmer anatomy with
deliberately awkward dimensions (hidden 12, 3q/1kv, head_dim 4, ffn 20,
vocab 29, seq 5; layer 0 Sliding+RoPE+gated, layer 1 Full+NoPE+gated).
Executor matches at ~5e-8 on logits; the trace checkpoints every novel
boundary (post-pf-norm Q, post-scale Q, post-position Q, gate sigmoid,
pre/post-gate attention) and self-checks its own semantics (NoPE is
bit-identically a positional no-op; the gate strictly gates).

**5b-2 Stage C — implemented.** The three causal controls
(`exec/tests/controls.rs`), each mutating **only the persisted graph**:

```text
C1  query_scale 3.87 → 3.5   diverges from layer 0     scalar authority
C2  layer 1 None → Rope      layer 0 identical,        layer-policy
                             layer 1 diverges           authority
C3  gate judgment removed    refuses, names primitive  op-semantic
                                                        authority
```

A hidden default is precisely a fact whose mutation changes nothing;
these controls are the search for hidden defaults, and they found none.
The ladder's engineering contract, complete:

```text
four-authority consistency + operand closure   = execution sufficiency
execution sufficiency + independent parity     = execution correctness
execution correctness + causal mutation controls = semantic authority
```

Closure proves the program is complete; parity proves the program is
correct; the controls prove the IR is in charge.

**5b-3 — implemented** (surfaced as `larql vindex3 exec`, rung G5b-3c).
Parity (hidden-state and logit, not argmax) against the checkpoint-driven
fixture path, and three positive controls proving three independent
semantic pathways drive computation — a container fact must not merely
survive serialization:

```text
scalar execution fact      mutate persisted attention scale   → output changes
layer policy fact          mutate one layer's None → RoPE     → output changes
operand/op semantic fact   mutate/remove judged gate semantics → diverge or refuse
```

The third exists because of the gate discovery: the two pieces the
generic fallback failed to capture (NoPE layers, attention gating) are
exactly the two that make the strongest controls.

Parity runs **layer by layer**, not final-logit-only: per layer
`max_abs` / `mean_abs` / cosine on the hidden-state trace, then
exact/near-exact logits per what the kernels actually promise. A
divergence at layer N localises immediately to norm ordering, QK
normalisation, scale placement, RoPE/NoPE, gate placement, residual or
FFN ordering — and because `vindex3 ops` spells the generic program
out, the debug object and the execution object are the same semantic
representation. The bar 5b-2 enforces: **the operation plan is not
documentation of execution; it is execution.**

**The compiler boundary.** Some execution semantics cannot be inferred
from bytes — a gate's activation, a parameter-free QK normalisation
(there are no weights to find), the exact application point of a scale.
Some authority must tell the *compiler* what they mean, and that
authority is legitimately architecture-aware. The boundary is where
family knowledge is allowed to live:

```text
SOURCE COMPILER / FRONT END          EXECUTION / BACK END
HF model_type, upstream reference    VINDEX3 IR
        ↓                                ↓
architecture-specific judgment OK    generic op plan
        ↓                                ↓
generic VINDEX3 IR                   generic kernels
                                     — no family knowledge here
```

A `muse_glimmer` source decoder is fine; `if family == muse_glimmer`
after the vindex is produced is the violation. VINDEX3 permits
**architecture-aware ingestion** and produces an
**architecture-independent executable IR** — like language front ends
compiling to one IR for one backend, it does not need every upstream
checkpoint to be self-describing; it needs to *compile their semantics
away*.

**The naming-convention trap.** Dispatching on object-id *strings*
(`if object_id.contains("perception_tower")`) technically avoids a
family branch but is the same defect laundered through a name: the
convention becomes an undeclared schema. Object ids identify; they never
encode execution rules. Dispatch reads typed semantics only —
`ObjectKind`, `ComponentRole`, `AttentionLayerPolicy`, `HiddenStateEdge`
fields — so a renamed id changes nothing and a new architecture whose
graph carries the same kinds executes with zero runtime edits.

---

## 9. Relationship to existing formats

- **LYRW v2** (`docs/lyrw-v2.md`) is the expert-bank segment codec. The
  candidate spec's convergence rule (§5.6 there) makes it one segment
  family a representation may use; its region roles (`gate/up/down/bias/
  scales/latents`) are operand-level structure *inside* an FFN object,
  not a substitute for logical objects. Recorded honestly: today no
  graph-encoded representation emits `.lyrw` segments — LYRW files are
  written only by the bank-shape import path, whose containers carry a
  `moe_manifest.json` and no graph. The two writers are disjoint;
  unifying them is a named gate for 3.0 Final.
- The **MoE manifest** (`moe_manifest.json`) describes routed programmes
  on bank-shape containers. A graph-encoded container carries none —
  routed or not (`encode` sets `moe_manifest: null` unconditionally);
  its routing structure lives in the graph and the representations.
  "The graph locates the manifest" is the convergence configuration,
  produced by no writer today.
- **VINDEX2** containers are a different generation; loaders refuse
  cross-generation directories with a precise error naming both versions.
