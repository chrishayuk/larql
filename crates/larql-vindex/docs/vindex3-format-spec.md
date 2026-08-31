# VINDEX3 — Candidate Specification

**Version:** 3.0-candidate
**Date:** 2026-08-30 (candidate: one canonical container model — the graph shape is normative and the bank shape is a recognised transitional import (§5); the contract stack named (§0); `index.json` schema 4 and `RegionLayout` recorded (§6.4, §12); status corrected against the implementation. Predecessor draft-2 recorded three binary-layout corrections from the first lyrw2 implementation — §6.2, §6.3, §6.4, §6.5.)
**Status:** Candidate Specification. The normative model is settled; the byte-level ABI is **not yet frozen** (§21 lists exactly what stands between candidate and 3.0 Final). The implemented state: production models — gpt-oss-20b, Gemma 4 26B-A4B, Granite 4.1 3B/8B/30B and Muse-Glimmer 30B — encode, verify and execute through this format byte-identically to their HF sources (`larql vindex3 encode` → `inspect` → `verify` → `exec`), and `larql serve` serves V3 containers over `/v1/completions`, `/v1/chat/completions` and `/v1/responses` (see `docs/vindex3-runtime.md`). The LQL surface reached full V2 parity on 2026-08-22 — execution, inference, browse, mutation, patching, compose, COMPILE, logical DIFF, COMPACT (see `docs/vindex-generation-policy.md`). `extract` still writes VINDEX2 **by default**; VINDEX3 is produced today by explicit request on every surface (`--generation v3`, LQL `FORMAT VINDEX3`, the factory generation pin, `larql vindex3 encode`, and the expert-bank import path). The default flip is the M4 rung of the generation policy, deliberately not yet taken.
**Predecessor:** [format-spec.md v0.4](format-spec.md) (VINDEX2)

**Companions:** [`vindex3-format.md`](../../../docs/vindex3-format.md) (the living spec: graph/execution semantics as implemented), [`vindex3-runtime.md`](../../../docs/vindex3-runtime.md) (runtime/serving), [`vindex-generation-policy.md`](../../../docs/vindex-generation-policy.md) (the V2→V3 transition), [`vindex3-experiments.md`](../../../docs/vindex3-experiments.md) (pre-registered experimental programme), [LQL](../../../crates/larql-lql/docs/spec.md) (the operations surface), [Conformance v1](conformance-v1.md), [Operations](operations-spec.md), [Ecosystem](ecosystem-spec.md) (the last three are VINDEX2-era)
**Implementation target:** `larql-vindex` crate (Rust); format-native reader: `vindex-cli` (§18.3)

> **A note on the number 2 appearing throughout.** This document specifies
> **VINDEX3**, and its own version is therefore `3.x`. Three things nearby keep
> a `2` on purpose and are not typos:
>
> | name | why it stays |
> |---|---|
> | `V2-0`…`V2-4` gates | pre-registered identifiers with results already recorded against them; renaming would orphan the lineage |
> | registry programme `vindex2` | same — it is an external key in chuk-experiments |
> | `lyrw2` / `FORMAT_VERSION = 2` | the *bank* format's own version, on a different axis: a VINDEX2 container holds LYRW v1 files; a VINDEX3 **bank container** (§5.4) holds LYRW v2 files. A VINDEX3 **graph container** (§5.3) holds plain tensor-table segments and no LYRW files at all. |
>
> Only the container is versioned 3. An on-disk `index.json` carrying
> `"version": 2` is a **VINDEX2** file — the predecessor above, not a draft of
> this one.

---

## 0. The contract stack — how to read VINDEX3

VINDEX3 began as a storage proposal and became a **self-describing,
executable, queryable model container**. The specification is therefore a
stack of contracts, not a single byte layout. Each contract has one
normative home; this document is the root and the format authority.

| Contract | What it governs | Normative home |
|---|---|---|
| **Format** | what is stored: the container envelope, `index.json`, segments, codecs | this document, §5–§9 |
| **Graph** | what it means: SystemGraph, components, logical objects, representations | living spec §5–§6; summarised §5.1, §17 |
| **Execution** | how it runs: operation plans, operand closure, the deletion invariant | living spec §8; runtime doc; summarised §17 |
| **State** | continuation: KV state, recurrent state, sessions | runtime doc §2–§4; summarised §17.3 |
| **Query** | how it is interrogated: WALK/DESCRIBE/browse, the `vindex` reader | this document §15; LQL spec |
| **Mutation** | overlays, patches, effective model state | LQL spec §3; summarised §19 |
| **Equivalence** | verify, logical DIFF, COMPILE, COMPACT guarantees | living spec §7; LQL spec; summarised §20 |
| **Conformance** | what another implementation must do | this document §13, §16, §18, §21 |

The test that assigns a fact to its contract:

> Can an independent implementation determine this purely from the VINDEX3
> artifact? Then it belongs in the Format/Graph contracts. Does it describe
> how an engine operates on the artifact? Execution/State. Is it an
> operator interaction (TRACE, DIFF, COMPILE)? The operations contracts —
> unless it persists state another implementation must understand, in which
> case it re-enters Format.

---

## 1. What is VINDEX3?

VINDEX3 is a **self-describing, executable and queryable model container**.
A modern model release is not a weights file — it is a system: a target
model, perhaps a perception tower in the same checkpoint, a drafter
consuming declared hidden-state taps, several quantisation profiles of the
same logical operands. GGUF answers *"how do I store and run these
tensors?"* VINDEX3 answers:

> What are these model objects, what operations can consume them, which
> representations are equivalent, which parts should be resident, and what
> future computation will need them?

Concretely, a VINDEX3 container carries:

- **Structure** — a system graph of components, logical objects and
  hidden-state edges (§5.1);
- **Storage** — hashed physical segments in declared encodings (§5–§7);
- **Execution** — enough judged semantics to run the model with zero
  architecture branches, including softmax, linear/Gated-DeltaNet, KDA and
  MLA attention (§17);
- **Inspection** — the model as structured, queryable data (§15);
- **Observation, mutation, persistence, equivalence, maintenance** — TRACE,
  overlays, COMPILE, logical DIFF, COMPACT, each with stated guarantees
  (§19–§20).

**The model IS the database** remains the founding principle: no query
index is stored beside the weights; the weights are the query index (§15).

The key principles carried from VINDEX2, unchanged:

- **Each weight tensor is stored once, canonically, in its serving format.**
- **Weights are separated by function, not by file size.**
- **mmap-first.** Every physical object is independently mmap-able.
- **Loaders dispatch on declared tags, never sniff filenames.**
- **Fail closed.** A missing required operand refuses with a precise
  diagnosis; nothing silently degrades authority.

What generation 3 genuinely adds over VINDEX2:

1. **The SystemGraph as semantic authority** — components, logical
   objects, representations and hidden-state edges; the HF checkpoint
   disappears as an authority once encoded (§5.1).
2. **Per-region quantisation.** Format belongs to each weight region, not
   to a whole layer file.
3. **Multiple physical segments per logical bank**, with declared
   geometry, for layers that exceed shard caps.
4. **A validated MoE programme manifest** for routed computation (§8).
5. **Representation variants.** A region set may carry several physically
   present encodings; profiles *select* among them and never request
   formats that were not extracted (§9.1).
6. **An execution contract** — the container binds as a closed, operand-
   verified program, not as tensors to reassemble (§17).

### 1.1 K3 validates the format; it does not define it

The MoE conformance envelope is defined by real architectures plus a control:

| Model | Routing | Shared experts | Expert space | Expert programme | Native format |
| ----- | ------- | -------------- | ------------ | ---------------- | ------------- |
| Direct MoE (control) | top-2 of 8 | 0 | residual | gated MLP | any |
| GPT-OSS | top-4 of 32/128 | 0 | residual | clamped gated MLP w/ residual term | MXFP4 |
| **Inkling-Small (276B-A12B)** | top-6 of 256, sigmoid + gate bias + norm_after_topk + route_scale 8.0; **shared-expert sink** (router scores shared experts) | 2 (always active) | residual | gated MLP; dense MLP at layer index 2 (mid-stack) | BF16, with NVFP4 / MXFP8 releases |
| **Kimi-Linear-48B-A3B** | top-8 of 256, sigmoid + renormalise + 2.446 scaling | 1 | residual | gated MLP; layer 0 dense (`first_k_dense_replace=1`) | BF16 |
| K3 | top-16 of 896 | 2 | latent 3584 | SiTU-GLU latent expert | (extraction: exact Q6_K baseline) |

Inkling-Small contributes what nothing else in the set can: real
two-shared-expert reduction under a **shared-expert sink** router (shared
experts inside the scoring/normalisation, `norm_after_topk`, gate bias,
global scale — the richest router-semantics test in the envelope); real
**NVFP4/MXFP8 native regions** via its quantised releases, whose
mixed-precision convention (routed experts low-bit, shared experts and
attention BF16) is itself a per-region-format use case; a mid-stack dense
layer (`dense_mlp_idx: 2`) proving per-layer manifests handle arbitrary
dense/MoE schedules; and — decisive for the rig — it is the first
design-set model that **cannot be RAM-resident** on the M3 Max, so
partial-residency, SSD-streaming and attn-local/FFN-remote profiles get
their first *non-optional* real-model test at one-eighth K3 scale. Its MTP
heads and multimodal towers are optional auxiliary tensors — the text
backbone is the conformance target; omitting MTP never changes authority.

Kimi-Linear-48B-A3B earns its seat three ways. It is the only **real,
locally runnable** shared-expert member (98.3 GB BF16, ~3B active), so the
shared-bank rung is proven on an actual checkpoint. Its
`first_k_dense_replace=1` hybrid stack exercises per-layer manifest
heterogeneity on a real model. And it is K3's direct lineage ancestor —
the same KDA(3):MLA(1) hybrid spine, 20 KDA + 7 MLA layers of real
recurrence parameters. Its KDA and MLA execution paths are **implemented**
in the reference executor (`opplan/exec/kimi_kda_layer.rs`,
`kimi_mla_layer.rs`, `kimi_moe_block.rs`), which is also what admits
KDA-family models beyond the envelope (GLM-5.3-Flash's 34 KDA layers ride
the same operator).

K3 is the stress test — largest bank, latent expert space, shared pre/post
projections. GPT-OSS and Inkling exist in the envelope precisely to stop
K3-specific assumptions from becoming the ABI.

---

## 2. Scope and non-goals

VINDEX3 serves **one fixed checkpoint efficiently under different
inference and query policies**, and makes the resulting model system
inspectable, mutable-by-overlay and provable. Training is out of scope. It
is explicitly not:

- **A model-development store.** No optimisation for training,
  fine-tuning, gradient updates, or frequently rewritten weights.
  Mutation is overlay-then-COMPILE (§19), never in-place rewrite.
- **A general neural-graph container.** VINDEX3 does not duplicate
  ONNX/safetensors-plus-compiler. The expert-programme vocabulary is
  deliberately bounded (§8.3), and the execution IR's operations are
  generic primitives, not a graph interpreter (§17).
- **A locality store.** Hot sets, retained experts, cache allocation,
  prefetch depth, placement and reduced-top-K are **runtime metadata over
  the index**, never physical-format decisions (§9).

The supported model contract:

> Decoder systems — dense, sparse-MoE, hybrid-attention (softmax,
> linear/Gated-DeltaNet, KDA, MLA), multi-component (perception towers,
> drafters over declared hidden-state edges) — expressed as logical
> objects with judged execution semantics. Sparse MoE composes from
> routed and shared expert banks, optional pre/post transforms,
> declarative routing/reduction semantics, and a bounded expert-programme
> vocabulary; every dense model is the degenerate single-entry case.

Genuinely novel expert topologies extend via a new `programme_id` (§8.4)
without changing region storage.

---

## 3. Design principles

1. **Structure is orthogonal to quantisation — at region granularity.**
   VINDEX2 declared one `quant_format` per layer file. VINDEX3 moves the
   format tag to each weight region. Re-quantising one projection role is
   rewriting those regions (or adding a sibling representation), not
   replacing the layer.
2. **Unified for dense and MoE.** A dense layer is a bank with
   `num_entries = 1`. Binary format and dispatch path are identical.
3. **Native OS addressability.** Each segment file is independently
   mmap'd; expert sharding reads only assigned entry byte ranges.
4. **The split rule.** A component gets independent physical identity
   **only when LARQL may independently omit it, quantise it, place it,
   prefetch it, execute it — or query it.** Conceptual tensor taxonomy is
   not a reason to split. The query clause matters: WALK reads gate rows
   without up or down, so on a browse-enabled index the gate role has an
   independent access pattern by construction (§15.2).
5. **Storage aligns with dispatch.** The natural extent is the expert
   group matching the grouped kernel's dispatch width.
6. **Representable ≠ servable.** The format may describe combinations no
   kernel can yet execute. Capability checking (§10–§11) distinguishes
   representable / reference-executable / dispatched / production.
7. **Logical ownership by object.** The logical object (§5.1) is the
   stable semantic unit. Segmentation (§7) is a physical storage
   parameter, invisible to model semantics.
8. **Single semantic authority.** The system graph is built once and
   every downstream consumer — planner, encoder, executor — consumes
   *it*, never a private re-interpretation of the checkpoint.

---

## 4. The five durable weight classes

The serving vocabulary freezes exactly five classes. These are the
boundaries that inference policy may ever want to fetch, place, quantise
or omit independently:

| # | Class | Contents | Why independent |
| - | ----- | -------- | --------------- |
| 1 | **Control & router** | Embeddings, norms, LM head, router weights, routing metadata, recurrence/control parameters | Small, always resident, precision-sensitive |
| 2 | **Dense spine** | Attention / KDA / MLA projections, per layer or major projection class | Touched every token; independent quantisation ladder |
| 3 | **Shared FFN** | Shared experts and shared latent pre/post projections, per layer | Touched every token; different residency economics from routed |
| 4 | **Routed gate/up banks** | Per-layer expert-group extents | Candidate for exact Q6_K or native low-bit; grouped-dispatch aligned |
| 5 | **Routed down banks** | Per-layer expert-group extents | Independent quantisation, placement and (approximate-profile) omission policy |

The classes are a **classification vocabulary**, not a directory layout:
draft-era layouts mandated one directory per class; the candidate does not
(§5.5). Classes 4 and 5 remain separable **because their inference
treatment can differ** — not because a tensor taxonomy says so. Where a
serving policy treats them identically, a single fused
`gate_up_fused + down` bank per layer satisfies the contract (§6.5).

No sixth class. Everything else — hot sets, expert retention, cache
sizing, prefetch order, exact-vs-approximate selection — is
profile/runtime metadata (§9).

---

## 5. The container model

This section is the candidate's central clarification. Two on-disk shapes
of VINDEX3 exist in the implementation, written by two disjoint writers.
Earlier drafts each described one shape and never named the other; the
candidate names both, ranks them, and states the convergence rule.

### 5.1 The canonical layering

A VINDEX3 container is, canonically:

```text
container envelope            the directory + index.json (sole root authority)
  └── SystemGraph             the logical-model authority: components,
      │                       per-layer attention policies, hidden-state edges
      └── logical objects     identity {component}.{kind} — embedding,
          │                   decoder_stack, output_head, perception_tower, …
          └── representations physically present encodings of an object,
              │               each with recorded fidelity (canonical | approximate)
              └── segments    hashed, mmap-able byte files
                  └── codecs  the segment's encoding: plain tensor-table,
                              LYRW v2 expert banks (§6), future families
```

Meaning flows down; bytes never define meaning. The graph references
object and representation ids; the index's directory maps representations
to segment bytes; graph edges never reference safetensors names — **the
HF checkpoint disappears as an authority once encoded.** The full graph
schema is specified in the living spec (§5–§6 there); its shape:

```text
SystemGraph
├── components: [Component]        id, role, num_layers, hidden_size,
│                                  attention: [AttentionLayerPolicy]?
├── objects:    [LogicalObject]    id, component, kind,
│                                  source_bindings, representations
└── edges:      [HiddenStateEdge]  producer_component, producer_layers,
                                   consumer_component, consumer_object
```

### 5.2 `index.json` — the envelope

`index.json` is the **sole root authority**: there is no superblock, and a
second root would create competing authorities. Its schema (the struct
`Vindex3Index`) is normative:

| Field | Type | Meaning |
|---|---|---|
| `version` | u32 | container schema — the generation discriminator (§12): V3 spans **3–4**; fresh writes stamp 4 |
| `model`, `family` | string | identity |
| `hidden_size`, `num_layers` | u32 | headline geometry of the primary component |
| `system_graph` | string? | relative path of the SystemGraph (`system_graph.json`) — present on graph containers |
| `moe_manifest` | string? | relative path of the MoE programme manifest — present on bank containers |
| `representations` | map | representation id → entry (encoding, fidelity, segment refs, hashes) |
| `segments` | map | segment file → declared schema/version |
| `profiles` | array | execution profiles (§9) — **inline in the index**, not a directory |
| `variants` | catalogue | representation-variant catalogue (§9.1) |
| `authority` | enum | `canonical` \| `derived` — whether this container is the source-of-truth encode or was compiled/derived from another (§20) |
| `precision_map` | object? | the derived per-region precision map, when compiled |
| `derived_from_model` | string? | provenance link for derived containers |

Readers MUST treat `system_graph` and `moe_manifest` as independent
options. **Absence of `moe_manifest` is not evidence the model is dense**
— a graph-encoded routed MoE (gpt-oss-20b) carries `moe_manifest: null`;
read `representations` to learn what a container holds.

### 5.3 The graph shape — canonical

Written by every mainline producer: `larql vindex3 encode`,
`larql extract --generation v3`, LQL `EXTRACT … FORMAT VINDEX3`, the
factory pin — all one shared pipeline (`encode_checkpoint`).

```text
<container>/
├── index.json            envelope (§5.2); system_graph set, moe_manifest null
├── system_graph.json     the SystemGraph, verbatim
├── segments/
│   ├── target.decoder_stack.bin      one logical object → one canonical
│   ├── target.embedding.bin          representation → one or more plain
│   ├── target.output_head.bin        contiguous tensor-table segments
│   └── …                             (perception/drafter objects likewise)
├── tokenizer.json                    capability snapshot: tokenizer_config,
└── …                                 special_tokens_map, generation_config,
                                      chat_template — what keeps the
                                      container servable, not executable
```

Within a segment, tensors are addressed by a per-representation tensor
table (name relative to the binding prefix, dtype, shape, offset, length),
payloads concatenated in table order. Every canonical representation
records a source payload hash and the segment records its own hash — the
verification inputs (§20). Segment framing:
`[u64 LE header length][header JSON][payload bytes]`.

### 5.4 The bank shape — transitional import

Written by exactly one production path — the expert-bank importer
(`larql extract-index --expert-banks native --expert-banks-out DIR`) —
and by the conformance fixtures:

```text
<container>/
├── index.json            envelope (§5.2); moe_manifest set, system_graph null
├── moe_manifest.json     model + MoE programme description (§8)
└── <segment key>.lyrw    LYRW v2 bank files (§6)
```

This shape predates the graph authority. It remains **valid input**: it
carries the same `index.json` schema, and `detect_generation` binds it as
V3 like any other container. Its standing under the candidate:

- **Readers MUST accept it.** A conforming reader opens both shapes and
  reports which it found; the `vindex` reader (§18.3) does.
- **Writers SHOULD NOT extend it.** New producer surfaces target the
  graph shape; the bank writer exists for routed-bank import, kernel
  bring-up and the LYRW conformance fixtures.
- **The convergence rule (§5.6)** defines its future, and §21 gates
  3.0 Final on executing that rule.

Draft-2's five-directory layout (`control/`, `dense/`, `shared/`,
`routed/`, `query/`, a `profiles/` directory, `weight_manifest.json`) was
never produced by any V3 writer and is withdrawn as a normative layout.
The five classes survive as classification vocabulary (§4); storage
references in manifests and variant catalogues are container-relative
paths with no mandated directory taxonomy.

### 5.5 Discrimination and shape rules

- `index.json.version` is the **sole discriminator** — no filename
  sniffing, no directory-shape heuristics. V3 spans schemas **3–4**
  (§12); unknown schemas refuse by name.
- A V3 container carries **at least one** of `system_graph` /
  `moe_manifest`. Today's writers each set exactly one.
- If both are present, **the graph is the semantic authority and the
  manifest describes the routed programme the graph locates** — this is
  the convergence configuration (§5.6), currently produced by no writer.
  A reader MUST NOT treat the manifest as a rival root.

### 5.6 The convergence rule

The two shapes are not two formats; they are one format observed
mid-unification, and the direction is fixed:

> **The graph/logical representation is the format; a bank layout is an
> encoding of a representation.** LYRW v2 becomes one segment codec that
> a representation of a routed-FFN logical object may use — its region
> roles (`gate/up/down/bias/scales/latents`) are operand-level structure
> *inside* an FFN object, never a substitute for logical objects. The
> MoE programme manifest describes the routed programme; under
> convergence the graph locates it.

Honestly recorded: today no graph-shape representation emits `.lyrw`
segments and no bank container carries a graph — the two writers are
disjoint. Executing this rule (or retiring the bank writer in favour of
graph-native LYRW representations) is a named gate for 3.0 Final (§21),
not an assumption the candidate makes.

### 5.7 Compatibility rules — what a conforming reader/writer must do

For every datum, one of four fates — understood, ignored, refused,
preserved — decided by these rules:

| Encountering | A conforming implementation |
|---|---|
| unknown `index.json` schema (`version` ∉ its supported set) | MUST refuse by name, stating the version found and the versions supported — before reading any byte |
| unknown fields inside a supported `index.json` schema | MUST ignore for interpretation (additive evolution within a schema number) and MUST NOT drop them when rewriting the index |
| unknown LYRW role / format / packing / layout tags | MUST preserve at read time; refusal belongs at capability-check time (§6.5, §11): a browse-only reader must not choke on a `down` region in a codec it never touches |
| unknown files in the container directory | MUST ignore for interpretation; maintenance operations (COMPACT, §20) MAY discard files nothing references — but only as named entries in their report, never silently. Registered capability files and every index-referenced segment are always carried |
| a required operand absent for a requested operation | MUST refuse naming the operand, object, layer and segment — never best-effort execute (§11, §17) |
| a cross-generation container (V2 directory to a V3 verb or vice versa) | MUST refuse naming both generations (§12.1) — no cross-loading, no silent conversion |

---

## 6. LYRW v2 binary format

LYRW v2 is the **expert-bank segment codec**: the physical layout for
routed/shared/dense MoE banks, used by bank containers today and by
graph-container representations under convergence (§5.6). It preserves
the v1 magic and self-describing property, and generalises the fixed
four-integer offset table into banks, segments and entry-region tables.

### 6.1 Header

```
[header]
  magic:            u32   0x4C595257 ("LYRW")
  format_version:   u32   = 2
  logical_layer:    u32
  num_banks:        u16
  num_segments:     u16   (segments described by THIS file's tables; ≥1)
  flags:            u32   (bit 0: this file is one segment of a multi-segment layer)
  reserved:         u32
```

All integers little-endian. All region offsets are from the start of the
containing segment file and 64-byte aligned.

### 6.2 Bank descriptor (`num_banks ×`)

```
  bank_id:              u16
  bank_kind:            u16   0=dense, 1=routed, 2=shared
  region_schema_count:  u16   number of schema records this bank owns in the
                              schema table (§6.4) — without it a reader cannot
                              tell where one bank's schemas end  [draft-2]
  flags:                u16   bit 0-1: browse mode (00=none, 01=direct,
                              10=strided) per §15.2; rest reserved  [draft-2]
  num_entries:          u32   1 (dense) or expert count
  input_dim:            u32
  intermediate_dim:     u32
  output_dim:           u32
```

Bank descriptor is 24 bytes (4-byte aligned), not the 20 the draft-1 field list implied.

`input_dim`/`output_dim` are the expert's own operand dims — for K3's latent bank these are 3584/3584, not the 7168 residual width. Dense v1-style layers map to one bank: `bank_kind=0, num_entries=1`.

**The binary carries no programme identity.** LYRW describes storage only — banks, entries, region schemas, offsets, formats. The MoE manifest binds `bank_id → programme` (§8.4). Two authorities for the same fact ("binary says programme 4, manifest says gpt-oss-expert-v1") is a disagreement waiting to happen; the manifest is the single one, consistent with the layering: storage holds regions, the manifest gives them meaning.

### 6.3 Segment descriptor (`num_segments ×`)

```
  bank_id:          u16
  segment_index:    u16
  first_entry:      u32
  entry_count:      u32
```

A single-file layer has one segment covering `[0, num_entries)`. Multi-segment layers repeat the header in every segment file with `flags` bit 0 set; `index.json` lists the segment files per logical layer so the loader never globs. **A segment file's entry table covers only that segment's entries** (`entry_count` rows, indexed from `first_entry`) — never the whole logical bank. [clarified in draft-2]

### 6.4 Region schemas and entry table

Expert banks are homogeneous: every entry in a bank shares the same region layout. The region schema is therefore declared **once per bank**, and each entry stores only offsets and lengths. (This is the simplification that dropping LYRW v1 binary compatibility buys — see §6.6.)

```
[bank region schemas]   region_schema_count × per bank:
  schema_index:     u16
  role:             u16   (§6.5)
  format:           u16   quant enum — 0=f32 1=f16 2=bf16 3=q4_0 4=q4_k 5=q6_k
                          6=q8_0 7=fp4_larql 8=mxfp4 9=nvfp4 10=mxfp8 ...
  packing:          u16   0=row_major, 1=blocks_with_scales_inline,
                          2=blocks_values / 3=blocks_scales
  pair_id:          u16   links a blocks_values schema to its blocks_scales
                          schema; 0xFFFF = unpaired
  layout:           u16   region payload layout — 0=unspecified,
                          1=contiguous_halves, 2=interleaved; unknown values
                          preserved per §5.7. This u16 was draft-2's reserved
                          pad; claiming it is the schema 3 → 4 bump (§12) —
                          the record stays 20 bytes, which is exactly why the
                          schema version had to move rather than the size
  rows:             u32
  cols:             u32

[entry table]           entry_count × region_schema_count ×:
  offset:           u64   (from start of containing segment file, 64-B aligned)
  length:           u64
```

Consequences:

- `gate_up_fused: mxfp4` + `down: q6_k` in one file, and GPT-OSS-style separate value/scale regions, without a new container.
- Uniform expert geometry is explicit; parsing is O(schemas), not O(entries × regions).
- Per-expert codec variation — which no grouped kernel supports — is **unrepresentable**, by construction rather than by convention.
- `pair_id` makes values/scales pairing explicit; role tags alone are ambiguous once an entry carries more than one quantised tensor.
- `layout` makes fused-region payload order (contiguous halves vs interleaved rows) a declared fact rather than a convention a kernel guesses.
- Exceptional per-entry overrides are reserved behind a header flag bit, undefined in v2.0 — added only if a real model forces them.

### 6.5 Region roles

Registered roles (extensible; new roles do not bump `format_version`):

```
0  gate
1  up
2  gate_up_fused
3  down
4  bias
5  scales          (paired with a values region via packing=2/3)
6  latent_in       (shared pre-projection, when bank-local storage is preferred)
7  latent_out      (shared post-projection, likewise)
8..255   reserved-registered
256..    vendor/experimental
```

The fast-path contract is unchanged from v1: known kernels may **require** exactly `gate_up_fused + down` (or `gate + up + down`) and parse them into the same structures the grouped kernels use today. Presence of other roles does not invalidate a file; absence of a role a programme requires makes the file un-executable for that programme (§11), not invalid.

**Unknown role, format, packing and layout tags are preserved, not rejected, at read time.** Refusal belongs at capability-check time (§11): a browse-only reader must not choke on a `down` region encoded in a codec it never touches, and a future codec must not invalidate old readers' ability to serve the regions they do understand. The reader reports unknown tags; the capability check refuses the *operations* that need them. [clarified in draft-2]

### 6.6 Relationship to the v1 layer files — greenfield, deliberately

LYRW v2 owes **no binary compatibility** to the VINDEX2 `layers/*.weights` files. Those files are an internal detail of VINDEX2: they exist only inside VINDEX2 directories, are parsed only by the VINDEX2 loader path, and were never a public contract in their own right.

Consequences:

- **No synthesis adapter.** A LYRW v2 reader never opens a v1 layer file, and vice versa.
- **No in-place upgrade** of multi-hundred-GB indexes. Migration is `checkpoint → VINDEX3 encode`, or an explicit importer — a standalone tool, not a loader feature.
- **Design freedom.** The bank-level region-schema table (§6.4), explicit value/scale pairing, and segment descriptors are clean-sheet choices. The `LYRW` magic and `format_version=2` are retained purely as self-description and forensics — a v1 reader encountering a v2 file fails fast on the version field with a precise "requires VINDEX3 loader" error, never a parse error.

The compatibility obligation that **does** bind is one level up: larql must support VINDEX2 and VINDEX3 side by side (§12.1).

---

## 7. Segmentation

Motivating arithmetic (K3, exact Q6_K):

```
params per expert        = 3 × 3584 × 3072            = 33,030,144
params per routed layer  = 33,030,144 × 896           = 29,595,009,024
Q6_K bytes (210/256)     ≈ 24.28 GB  =  22.61 GiB
```

That exceeds the published 20 GiB shard cap, so `one logical layer = one physical file` cannot hold for K3 exact Q6_K. **Segment width and group width are two different scales, decided by two different measurements** — conflating them turns a 2-file layer into a 14-file layer for no read-path benefit:

| Scale | Optimises | Typical size |
| ----- | --------- | ------------ |
| **Segment file** | file count, mmap management, shard distribution, the 20 GiB cap | as large as the cap allows — for K3 exact Q6_K, **2 segments of 448 experts** (~11.3 GiB each), not 14 of 64 |
| **Group extent** (inside a segment) | SSD reads, prefetch units, grouped-kernel dispatch | 8/16/32 experts (E2/E3) |

A K3 routed layer therefore becomes:

```
routed/layer_037.seg00.weights   experts   0–447
  ├── group extent  0: experts   0– 15
  ├── group extent  1: experts  16– 31
  └── ... (28 extents of 16)
routed/layer_037.seg01.weights   experts 448–895
```

At ~92 MoE layers this is ~184 routed segment files, not ~1,288.

Rules:

- Segment boundaries **must** fall on group-extent boundaries; group width **must** divide segment width.
- Both widths are extraction-time storage parameters chosen by measurement (E2 sweeps them independently), not semantic commitments. They may differ per model and per layer.
- Physical expert order within a segment need not equal logical order — the entry table is the indirection. Permuted layouts are legal but must not be adopted without the E2/E6 evidence bar.

### 7.1 Group extents

The unit of read alignment and prefetch is the **group extent** inside a segment, sized to the grouped kernel's natural dispatch width. One grouped dispatch ≈ one group extent ≈ one read unit; the extent boundary is what the payload layout aligns to, and the entry table makes extents addressable without a separate structure. Individual-expert files are prohibited at K3 scale (896 experts × ~92 MoE layers × several roles is an operational failure, not a design).

---

## 8. The MoE programme manifest

`moe_manifest.json` describes how regions form an MoE computation. The physical index stores tensor regions; the manifest gives them meaning; the runtime selects an optimised kernel when it recognises the programme. It is the root-adjacent authority of the **bank shape** (§5.4); under convergence the system graph locates it (§5.6).

### 8.1 Per-layer shape

```json
{
  "moe_layer": {
    "layer": 12,
    "input_space": "residual",
    "router": {
      "scores": "layers.12.router.weight",
      "selection": { "kind": "top_k", "k": 16 },
      "normalisation": "k3_quantile_balanced"
    },
    "transforms": {
      "routed_input":  "layers.12.routed_expert_down_proj",
      "routed_output": "layers.12.routed_expert_up_proj"
    },
    "routed_bank": {
      "experts": 896,
      "programme": "latent-moe-v1",
      "storage": "routed/layer_012",
      "expert_dims": { "input": 3584, "intermediate": 3072, "output": 3584 }
    },
    "shared_bank": {
      "experts": 2,
      "programme": "gated-mlp-v1",
      "storage": "shared/layer_012"
    },
    "reduction": "gate_weighted_sum",
    "routed_output_norm": "layers.12.routed_out_norm",
    "combine": "residual_add"
  }
}
```

For a conventional MoE, `transforms` are null. For GPT-OSS, `routed_bank.programme = "gpt-oss-expert-v1"` and `shared_bank` is absent. For Inkling, shared and routed banks coexist in residual space. Per-layer variation (hybrid dense+MoE stacks, differing expert counts) is expressed by per-layer manifests, not global fields. `storage` values are container-relative path keys (§5.4), not a mandated directory taxonomy.

### 8.2 What stays model-specific (adapter-owned)

Router scoring/normalisation details, shared-expert participation in normalisation, activation functions, expert residual semantics, clamps/biases/scales, fusion preferences, layer-specific expert counts. The manifest names these; the adapter implements them.

### 8.3 Bounded programme vocabulary

The declarative vocabulary is inference-shaped and closed by design:

```
linear · fused linear · activation · clamp · multiply · add · scale ·
normalise · route · gather · weighted reduction · residual merge ·
pre/post transform
```

No general graph interpreter. Known arrangements compile to specialised kernels; a generic reference executor provides correctness for everything representable.

### 8.4 Programme registry

```
programme_id  0  gated-mlp-v1
              1  gated-mlp-fused-fc1-v1
              2  gpt-oss-expert-v1        (clamped gated MLP + residual term)
              3  shared-routed-mlp-v1
              4  latent-moe-v1            (K3 SiTU-GLU latent expert)
```

Each programme declares its **required region roles**. New programmes register an id, a version, required roles, and optional opaque model metadata — region storage is untouched.

The manifest is the **only** binding of `bank_id → programme_id`; LYRW files never carry programme identity (§6.2). Kernel capability entries (§10) reference programmes by registry id.

---

## 9. Execution profiles and authority

A profile selects inference behaviour over one extracted container. Profiles never trigger reslicing — and they never trigger conversion (§9.1). Profiles live **inline in `index.json`** (§5.2); they are policy, not payload, and are individually replaceable without touching the immutable artifact checksum set.

```json
{
  "profile": "routed-mxfp4",
  "base": "exact",
  "select": {
    "routed.gate_up": "native-mxfp4",
    "routed.down":    "exact-q6k"
  },
  "placement": { "routed": "local", "dense": "local" },
  "runtime_policy": {
    "resident_experts": "routing-profile-2026-08-14.json",
    "prefetch_group": 32
  }
}
```

The profile carries **no `authority` claim of its own** — authority is derived (§9.2).

### 9.1 Representation variants — profiles select bytes, they don't request formats

A profile saying `"format": "mxfp4"` cannot turn Q6_K bytes into MXFP4 bytes by declaration. Exactly one representation model is legal: **a region set may carry multiple physically present variants; a profile selects a present variant.**

```json
{
  "region_set": "layer.12.routed.gate_up",
  "variants": {
    "exact-q6k":    { "storage": "routed/layer_012.q6k",    "fidelity": "source-equivalent" },
    "native-mxfp4": { "storage": "routed/layer_012.mxfp4",  "fidelity": "source-exact" }
  },
  "baseline": "exact-q6k"
}
```

- **Selecting an absent variant fails closed**, naming the region set, the requested variant and the variants actually present — before any byte is read.
- **No runtime conversion, ever.** "No hidden decode-time repacking" (§10) holds by construction: the bytes executed are the bytes stored.
- **Incremental packs.** New variants are added beside the baseline as independent, checksummed segment files — the multi-terabyte baseline is never rewritten.
- **Single-copy, clarified.** The v1 principle forbids storing the *same* bytes twice; it does not forbid deliberate alternative encodings. The `baseline` variant is the canonical authority; additional variants are opt-in, per-component, and individually removable.

On the graph shape, the same model appears as multiple `Representation`
entries on a logical object (`canonical BF16`, `Q6_K`, `NVFP4`, …), each
with recorded fidelity, produced by REPRESENT / `vindex represent`
(§18.3) — selection semantics are identical: present variants only,
fail-closed by name.

### 9.2 Authority — graded, derived, never asserted

**Levels** (mandatory, fail-closed):

| Level | Meaning |
| ----- | ------- |
| `source-exact` | Decoded values bit-identical to the source checkpoint, in the checkpoint's own encoding family (e.g. native MXFP4 regions of a native-MXFP4 model) |
| `source-equivalent` | Different encoding whose decode reproduces the source values exactly (e.g. a lossless Q6_K container of native MXFP4 values) |
| `numerically-approximate` | Same architecture, lossy representation (e.g. Q6_K quantised from BF16) |
| `structurally-approximate` | Components omitted or replaced (reduced top-K, shared-only layers, compiled subexperts) — must list `omitted_components` / `replacement` |
| `analysis-only` | Incapable of complete forward execution (router/browse slices) |

Authority is **derived, not declared**: every variant carries a region-level `fidelity` set at extraction time from provenance, and a profile's authority is the weakest fidelity across its active selections, further capped by programme traversal (§11) when required operands are absent. This closes the loophole where a lossy extraction becomes "exact" merely by being named the baseline. A profile cannot claim above its derived level; it may voluntarily claim below it.

The container-level `authority` field (§5.2) grades the container itself: `canonical` for a source encode, `derived` for a container COMPILE produced from effective (possibly mutated or re-encoded) state — reported apart from per-representation fidelity, so a derived container never presents as the source of truth.

Standard profile names: `exact`, `native-lowbit`, `mixed-precision`, `attn-local-ffn-remote`, `partial-residency`, `reduced-top-k`, `shared-only`, `router-browse`, `compact-approximate`.

**Runtime metadata, never format:** top-K/retention %, hot/warm/cold assignment, resident experts, per-layer popularity, adaptive cache size, prefetch ordering, exact-vs-approx selection, static per-layer precision choice.

### 9.3 Omission semantics ("dropping down")

The manifest distinguishes the materially different meanings:

| Mode | Authority | Notes |
| ---- | --------- | ----- |
| Client omission (FFN remote) | inherits selection (up to source-exact) | The whole routed branch moves; the K3 latent boundary makes whole-branch RPC ~14 KB/layer f16 vs ~100 KB for projection-split — never split gate/up local from down remote absent contrary measurement |
| Analysis/router slice | analysis-only | Retains routers, gate vectors, metadata; no decode claim |
| Cheaper down representation | numerically-approximate | The production interpretation of "cheap down" |
| Down replaced by compact approximation | structurally-approximate | Must name `replacement` |
| Routed branch skipped (shared-only) | structurally-approximate | Dropping an expert's `w2` alone yields **no** expert output — the honest mode is skipping the expert/branch, not a half-expert |

### 9.4 Residency — contract vs plan (pinned direction)

Residency is the third axis of the container model: semantic identity
says what an object *is* (§5.1), a representation says which bytes
implement it (§9.1), residency says where those bytes — and execution
state — live at this moment of execution. The vocabulary is not yet
standardised; the division of responsibility is fixed now, so the wrong
kind of fact never enters the format:

- **The residency contract is persisted facts** — model- and
  representation-derived properties: access class (required-every-token
  vs demand-driven/routed), zero-copy mmap capability, an encoding's
  direct-execution capability, evictability, co-use grouping (the
  segment/extent granularity of §7), and state lifetimes (§17.3). All
  additive: under §5.7 an older reader ignores them.
- **The residency plan is runtime-derived and never persisted in the
  container** — which representation resides where, prefetch and
  eviction decisions, device placement. A `device = GPU0`-shaped fact
  in the format is a defect: placement ages with the machine, never
  with the model.
- **Residency never changes model meaning.** The same container means
  the same thing mmap-cold on NVMe, resident in unified memory, or
  split across devices; the plan changes efficiency, the graph changes
  nothing. Artifact size, resident-set size and bytes-touched-per-token
  are three independent quantities, and the container carries enough
  semantics to describe each.
- **Residency is measured, not assumed.** Routing locality in real MoEs
  is poor enough that "sparse" does not imply "small resident set" —
  the contract describes what is possible; the plan adapts to observed
  traffic.

Because the contract vocabulary is additive, standardising it is 3.x
work under the compatibility rules, not a gate for Final (§21).

---

## 10. Kernel capability registry

Kernels advertise what they can execute:

```
programme_id · region roles · formats per role · grouping widths ·
input layout · maturity
```

Maturity ladder, matching the serving-format ledger: **Representable → Reference → Grouped → Dispatched → Production.** The loader reports, per (programme, format, grouping) combination, which rung it sits on. Mixed per-region formats are either supported by a kernel or **explicitly refused** — never silently repacked at decode time.

---

## 11. Capability checking

The loader does not hard-code "down weights present" tests. It traverses the layer's MoE programme (bank shape) or the component's operand closure (graph shape, §17) and reports which required operands are absent, then:

- refuses execution profiles whose authority claim exceeds what the present operands support;
- names the missing role, bank, layer and segment precisely (`VindexError::MissingRequiredRegion { layer, bank, role, .. }`);
- distinguishes *representable-but-no-kernel* (falls back to reference executor, flagged) from *operand-absent* (hard refusal).

Programme-derived checks give the right per-architecture answers for free: routed removal on Inkling leaves shared experts contributing; on GPT-OSS it leaves no FFN contribution; on K3, a missing `routed_output` transform invalidates even completed expert computation.

---

## 12. Versioning and coexistence

The version surfaces, precisely:

| Contract | VINDEX2 value | VINDEX3 value |
| -------- | -------- | -------- |
| LYRW `format_version` | 1 (VINDEX2-internal) | **2** (self-description only; no cross-reading, §6.6) — trails the container generation by one, permanently |
| `index.json` `version` | 2 | **3–4** — the container-generation discriminator; 4 is current (the `RegionLayout` claim of §6.4's former reserved u16 — wire size unchanged, which is precisely why the bump is needed) |
| `vindex_spec_version` | 1 | **2** (programme manifest + profiles enter the validated public contract) |
| MoE manifest schema | — | **1** |
| SystemGraph schema | — | **6** (v6: presence means semantic presence — attention/FFN surfaces present iff the program runs them, explicit per-layer operator; v2 first added the execution surface) |

**On the numbering.** The container generation *is named for* `index.json.version` — VINDEX2 writes `version: 2`, VINDEX3 currently writes `version: 4` within the 3-generation's schema span. The generation is a **schema span, not a single number**: additive-but-load-bearing changes (like `RegionLayout`) bump the schema within the generation; readers accept the span. The FP4 additive-extension precedent is retained: new region formats, roles and programme ids are enum additions, not schema bumps — a schema bump is reserved for a reinterpretation of existing bytes or fields.

### 12.1 Dual-generation support in larql — the real compatibility contract

The binding obligation is not between the two on-disk formats (there is none — §6.6). It is that **one larql binary supports both vindex generations, indefinitely for reading and serving**:

- **Detection.** `index.json.version` is the sole **schema** discriminator, and the loader maps supported schema revisions to their owning container generation. No filename sniffing, no directory-shape heuristics. A missing or unknown version fails naming the version found and the schema sets this binary supports.

  The mapping is **many-to-one, not an identity**:

  | `index.json.version` | generation | note |
  | -------------------- | ---------- | ---- |
  | 1 | VINDEX2 | legacy schema; absent fields load with defaults |
  | 2 | VINDEX2 | what a fresh VINDEX2 extraction writes |
  | 3 | VINDEX3 | pre-`RegionLayout` schema; still read |
  | 4 | VINDEX3 | what a fresh VINDEX3 encode writes |

  A generation is *named* for the schema family it writes, not for the only schema it can read. Treating the version as a generation identifier rather than a generation floor refuses every legacy-schema index in existence — which E0 caught in practice, not in review. Unified dispatch routes schema 1 to the VINDEX2 loader; the VINDEX3 loader still refuses it by name.
- **One entry point.** `Vindex::open(path)` returns the generation-appropriate handle behind a common trait; `larql run / serve / verify / slice / publish / pull` all accept either generation or refuse naming the generation. Generation-specific verbs error precisely on the wrong generation rather than silently no-op.
- **No cross-loading, no silent conversion.** The VINDEX2 loader path is frozen-but-maintained: it never opens VINDEX3 directories, never gains VINDEX3 features, and VINDEX3 code never re-implements VINDEX2 parsing. Conversion is only ever an explicit importer or re-encode.
- **Hub and distribution.** `larql publish` stamps the container generation into the hub artifact metadata; `larql pull` selects the reader from that stamp and refuses a generation the installed binary lacks — before downloading terabytes, not after.
- **Wire protocols are generation-agnostic.** The expert-RPC and FFN-dispatch wire contracts carry activations and results, not container bytes; a grid may therefore mix generations. A shard's container generation is a local concern of that shard's loader.
- **Support policy.** VINDEX2 remains fully supported for read/verify/serve. The default-extraction generation is governed by `docs/vindex-generation-policy.md`: one named constant (`DEFAULT_EXTRACTION_GENERATION`), one pinned test, flipped in one commit (the M4 rung — M1–M3 are done). Explicit requests are never downgraded; a surface that cannot produce the requested generation refuses by name. After the flip, V2 becomes the explicitly-requested compatibility generation and receives no architectural expansion.

---

## 13. Conformance envelope

The MoE bank machinery freezes only after the envelope fixtures pass the generic reference executor (fixtures defined in the experiments document):

| Capability | Direct | GPT-OSS | IS-276B | KL-48B | K3 |
| ---------- | :----: | :-----: | :-----: | :----: | :-: |
| Variable expert count / top-K | ✓ | ✓ | ✓ | ✓ | ✓ |
| Routed experts | ✓ | ✓ | ✓ | ✓ | ✓ |
| Shared experts | – | – | ✓ (2) | ✓ (1) | ✓ |
| Shared-sink router (shared experts scored) | – | – | ✓ | – | – |
| Residual-space experts | ✓ | ✓ | ✓ | ✓ | – |
| Latent-space experts | – | – | – | – | ✓ |
| Hybrid dense+MoE stack | – | – | ✓ (mid-stack, idx 2) | ✓ (layer 0) | ✓* |
| Non-softmax router (sigmoid + scaling) | – | – | ✓ (+ gate bias, norm_after_topk) | ✓ | – |
| Custom expert programme | – | ✓ | – | – | ✓ |
| Native low-bit regions | ✓ | MXFP4 | NVFP4/MXFP8 (real releases) | – (BF16) | MXFP4 |
| Mixed per-role format | ✓ | ✓ | ✓ (release convention: routed low-bit, rest BF16) | ✓ | ✓ |
| Fused/decomposed tensors | ✓ | ✓ | ✓ | ✓ | ✓ |
| Grouped dispatch | ✓ | ✓ | ✓ | ✓ | ✓ |
| Auxiliary optional components (MTP, towers) | – | – | ✓ | – | ✓ (multimodal) |
| Single-segment routed layer | ✓ | ✓ | ✓ (~4.9 GiB/layer Q6_K) | ✓ (~1.4 GiB/layer Q6_K) | ✓ |
| Segmented logical layer | – | – | – | – | ✓ |
| Exceeds-RAM residency (partial/remote non-optional) | – | – | ✓ | – | ✓ |
| WALK/DESCRIBE (residual-space browse) | ✓ | ✓ | ✓ | ✓ | – |
| WALK via latent transform (§15.4) | – | – | – | – | ✓ |

\* K3's dense/MoE layer schedule is confirmed at adapter time; KL-48B's `first_k_dense_replace=1` is confirmed from the released config.

Real-model progress against the planned order **Gemma MoE → GPT-OSS → Kimi-Linear-48B-A3B → Inkling-Small → K3**: Gemma-family (Muse-Glimmer) and GPT-OSS encode, verify and execute end to end; Kimi-Linear's KDA/MLA/MoE execution operators are implemented in the reference executor (§1.1) — the adapter dress rehearsal is underway, not upcoming; Inkling-Small remains open (its admission surfaced a silent-interleave defect class the plan gate now names); K3 is extracted **once**, last, into the frozen bank ABI. Fixture C is retained purely as the tiny deterministic conformance fixture.

---

## 14. What the experiments must decide

Only these decisions genuinely belong in the on-disk bank ABI; everything else stays runtime policy:

| Decision | Experiment | Why it matters |
| -------- | ---------- | -------------- |
| Region granularity (fused vs split roles) | E1, E4 | mmap count, rewriting, read amplification |
| Expert-group / segment width | E2, E3 | couples SSD reads to grouped kernels; K3 20 GiB cap |
| Fused vs decomposed FC1 storage | E1, E7, V2-1 | checkpoint import cost, mixed precision, **and gate-only browse reads** — the serving and query answers must be reconciled here, not assumed |
| Per-region format tags | structural (E4 gates *promotion* only) | representation is justified by native values/scales, v1's existing mixed precision, and format-neutral banks; E4 decides only whether a mixed-format **profile** reaches Production |
| Physical expert ordering | E2, E6 | possible locality gain vs model-specific assumption risk |
| Profile/variant-selection mechanism | E5, V2-0 | avoids reslicing per deployment; selection-not-request semantics (§9.1) |
| Capability/authority metadata | V2-0 | approximate slices must never present as exact |

Registered prior (falsifiable): one file-set per routed layer (two segments for K3 Q6_K), one entry per expert, down independently addressable, locality as runtime metadata, omission = skip-the-branch, remote = whole-routed-branch RPC. Per-region format tags are in the ABI **structurally** (not gated on E4); the registered prior is that no mixed-format *profile* reaches Production before real-K3-layer evidence (E4 stage 3). Gate/up fusion is **no longer a prior** — it is a per-index extraction choice decided by E1/E7 (§15.2).

---

## 15. Query contract — the model IS the database

The LQL browse surface (WALK, DESCRIBE, SELECT, EXPLAIN WALK) is a first-class consumer of VINDEX3, with the same single-copy contract as v1: **no query index is stored beside the weights; the weights are the query index.** The format-native `vindex` reader (§18.3) is the second first-class consumer: inspect, describe, representations, diff, precision, verify — every answer derived from the artifact alone.

### 15.1 What replaces `gate_vectors.bin`

There is no `gate_vectors.bin` in VINDEX3. The gate rows live where the split rule puts them — as `gate` (or the gate half of `gate_up_fused`) regions inside LYRW banks, or as the gate tensors of an FFN object's representation on the graph shape. Gate KNN mmaps the segment files and walks gate regions in place:

- **f16/f32 regions:** zero-copy reinterpret, exactly the v1 fast path.
- **Block-quantised regions (FP4/FP8/Q-K):** lazy per-feature dequantisation at walk time via the existing block codecs. The v1 caveat carries over verbatim: 4-bit gate KNN is noisy; inference compensates, isolated dot products do not.
- Untouched `up`/`down` pages cost nothing under mmap, so browse over a full-fat index reads only gate bytes even when nothing was sliced.

MoE browse semantics are unchanged from v1: gate KNN selects features **across all experts, no router needed** — a bank with `num_entries = E` contributes `E × intermediate_dim` walkable features per layer. Feature numbering stays v1-flattened (`layer:feature`, experts contiguous within the layer) so `feature_labels.json` keys survive migration untouched.

### 15.2 The gate-addressability rule (resolves the fusion collision)

A browse-enabled index requires gate rows to be readable without decoding up. Two legal ways to satisfy that:

1. **Decomposed storage** (`gate` + `up` regions): clean gate-only reads; the E1/V2-1 fused-vs-decomposed parity requirement already guarantees kernels accept it.
2. **Fused storage with strided browse** (`gate_up_fused`): legal only when the packing permits striding into the gate half without decoding up rows (row-major f16 yes; interleaved quantised blocks generally no — the §6.4 `layout` tag declares which).

The choice is recorded per bank at extraction time (a `browse: none | direct | strided` tag in the bank descriptor's flags, §6.2). **Serving-only indexes may fuse freely.** A browse-enabled index defaults to decomposed unless E1/E7 shows the fused serving advantage exceeds its own promotion bar.

### 15.3 Query metadata

`down_meta.bin` (DMET, unchanged), `feature_labels.json` and `relation_clusters.json` are **derived metadata, not weight copies** — single-copy is not violated. Two notes:

- For latent MoE banks, `down_meta` is computed at extraction through the full output path — expert `w2` → `routed_output` transform → unembed — so its top-token claims describe residual-space effect, not raw latent columns.
- Query metadata is optional per profile; its absence downgrades DESCRIBE/SELECT label richness, never WALK correctness.

### 15.4 Browsing latent-space banks (the genuinely new problem)

K3's gate rows live in the 3584-dim latent space; WALK queries originate in residual space. The programme manifest already carries what browse needs: `routed_input` names the residual→latent transform. WALK against a latent bank projects the query vector through that transform **once per query**, then dot-products against latent gate rows unchanged. `EXPLAIN WALK` reports the space hop. Residual-space banks walk exactly as v1.

### 15.5 Browse profiles and slices

- **Profile:** `browse` is a standard profile at authority `analysis-only` — requires gate regions (decodable), embeddings, tokenizer; query metadata and routers optional. Capability checking (§11) derives this; no filename tests.
- **Slice:** a published browse slice is produced by copying **only gate regions** into gate-only LYRW files (absent roles are legal, §6.5) plus embeddings + query metadata. The v1 ~3 GB browse economics are preserved; the loader reports the slice as `analysis-only` automatically because the programme's required inference operands are missing.

### 15.6 Extract-level mapping

| v1 extract level | VINDEX3 equivalent |
| ---------------- | ------------- |
| Browse | `browse` profile / gate-only slice (§15.5) |
| Inference | `exact` profile over classes 1–5 |
| All / COMPILE | full container — `COMPILE … INTO VINDEX` materialises effective state today (§20); `COMPILE INTO MODEL` (container → checkpoint export) is specified but **not yet implemented** on V3 |

---

## 16. Success criteria — "done" is defined here, in advance

VINDEX3 is a successful successor when all seven hold. Each is bound to the gate or experiment that proves it, so the bar cannot drift after the fact:

| # | Criterion | Proven by |
| - | --------- | --------- |
| 1 | An existing VINDEX2 model loads, verifies, serves and publishes through the dual-generation binary with zero behavioural regression | E0 (continuous, CI) |
| 2 | Gemma and GPT-OSS run through the same LYRW2 bank machinery and the same production dispatch interface | V2-3, V2-4 rungs 1–2 |
| 3 | Routed **and** shared banks are genuinely generic — proven on a shared-expert model or fixture, not asserted | Fixture C + **KL-48B (1 shared) and Inkling-Small (2 shared, sink router)** — real, V2-4 rungs 3–4 |
| 4 | K3 is extracted once and served with no K3-specific physical layout — only a manifest and an adapter | V2-4 rung 4 |
| 5 | A new representation or placement is introduced via variants, profiles and kernel capabilities, without rebuilding unrelated weights | §9.1 mechanism + V2-0 profile-resolution acceptance |
| 6 | Unsupported or approximate configurations fail closed and report exactly why — operand, bank, role, layer, segment, variant | V2-0, §11 |
| 7 | Onboarding the **next** conventional MoE requires an importer and a programme adapter — zero format changes, zero new region roles, zero kernel-interface changes | **E8 held-out architecture** |

Criterion 7 deserves emphasis: the conformance fixtures cannot prove it, because the ABI was designed against them. Only a held-out architecture, onboarded after freeze under a no-format-changes rule, tests generalisation rather than fit. If E8 fails, the "portable sparse-serving substrate" claim is downgraded honestly, in this section.

The maturity ladder governs claims throughout: **Representable → Reference → Grouped → Dispatched → Production.** No criterion is met by a representable-only demonstration.

---

## 17. Execution contract

Normative home: living spec §8 (`docs/vindex3-format.md`) and the runtime
document. The format-level obligations, stated here because an
independent implementation must honour them:

### 17.1 The container binds as a program

A VINDEX3 binding is a closed, operand-verified **operation plan**
(`ComponentOpPlan`) plus its operand bytes — never tensors to reassemble
into an engine's own architecture type. Binding requires **operand
closure**: every executable tensor classifies into a declared operand
role; every operand's operation is declared by the component's execution
surface; every operation's operands are present with the stored shapes
the surface states. An unclosed program **refuses to open**, naming the
defects — it is never best-effort executed.

```text
four-authority consistency + operand closure     = execution sufficiency
execution sufficiency + independent parity       = execution correctness
execution correctness + causal mutation controls = semantic authority
```

### 17.2 The deletion invariant and the compiler boundary

Removing the original checkpoint, `config.json`, HF model type and
architecture name must not change execution. The runtime sees
`container → system graph → operation plan → generic kernels`, nothing
else — no family branches, no layer-pattern arithmetic, no hardcoded tap
constants, no dispatch on object-id strings. Architecture-aware judgment
is legal only in the **source compiler** (front end), which compiles
family semantics away into the generic IR; after the container exists,
family knowledge is a contract violation.

The execution surface carries every judged semantic execution needs,
fully resolved — including the facts no tensor evidence can reveal
(parameter-free QK normalisation, query-scale vs score-scale application
points, attention output gating). Attention families — softmax,
linear/Gated-DeltaNet, KDA, MLA — are first-class surfaces, present only
when the model uses them, never inferred from a model name.

### 17.3 State

Continuation state crosses the runtime as a caller-side provider whose
geometry derives from the plan (`KvState`, per-layer row width and
span/window), never from architecture inference. Recurrent-state layers
(linear attention, KDA) carry their state through the same
plan-declared discipline. Sessions, batch prefill, resume and the
serving stack are specified in the runtime document; their contract
point here is single: **state geometry is a container fact.**

### 17.4 Pinned generalisations — both lifts landed within schema 6

Two abstraction lifts were pinned for 3.0 Final, recorded here so the
freeze does not fossilise the one Transformer-shaped remnant of the
execution ontology. **Both are now implemented.**

**Lift 1** (2026-08-30): SystemGraph schema 6 carries the one
intentional reinterpretation — presence means semantic presence
(`attention` and `ffn` surface groups present iff the component's
program runs those operations, the per-layer `operator` explicit with
no absent-means-softmax default), the layer census fails closed on an
undeclared family, and operand closure runs at encode (an encoder may
not leave behind a container whose operands do not close; a failing
closure removes the output).

**Lift 2** (2026-08-31): the remaining state-schema facts — KDA state
precision, MLA latent-KV geometry, the per-op norm-epsilon override
(drill F5/F6) — landed as **additive fields within the v6 span**
(optional, not reinterpretations of existing bytes), consistent with
§12's numbering rule. The continuation schema now carries three region
species rather than two: sequence-indexed KV rows, a **latent cache**
(one operator-defined row per position — MLA's compressed latent plus
its shared rope-K, never a K/V pair), and fixed-size recurrent buffers;
`ExecutionSurface.mla.kv_a_norm_eps` carries the one judged semantic
the container previously could not (`kv_a_layernorm` runs at its
class default, not the layer's `rms_norm_eps`), and an operand set
carrying none is refused rather than lent the layer's value. A
precision no checkpoint declares is a recorded transcription of the
operator's own reference, held in one place, never a planner's choice.

The migration is **graph-only**, and that is the lift's own witness:
re-encoding Kimi-Linear-48B-A3B-Instruct under lift 2 left all five
representations' `payload_sha256` **byte-identical** to the schema-6
container while the missing semantic moved into the graph. Meaning was
corrected without touching a single physical representation — §5's
separation of logical identity from stored bytes, exercised at 92 GB.

- **The operation program replaces kind-implied completeness.** Today
  the completeness contract derives required surfaces from object kinds
  (a `DecoderStack` or `PerceptionTower` implies attention + ffn +
  norm; an `Embedding` or `OutputHead` implies head). The lift: a
  component carries an **ordered program of typed operations** —
  attention, linear-attention, KDA, MLA, SSM/convolution, MoE, norm,
  projection, gating, future kinds — and completeness means every
  declared operation closes over its operands. Object kinds keep
  identity; they stop implying operation sets. A pure-SSM decoder then
  admits by declaring its program, not by resembling attention. The
  implementation is already most of the way there — a binding *is* a
  `ComponentOpPlan`, and the attention families are surfaces present
  only when a model uses them — so the lift is normative, not
  mechanical: stop deriving the program from the kind.
- **ContinuationState replaces KV as the base state abstraction.**
  *(Landed 2026-08-31.)* The plan declares a **state schema** — typed
  regions with role (`kv | latent-kv | recurrent | convolution |
  future`), geometry, lifetime and update semantics — and KV is one
  region kind among them. Softmax attention declares KV rows; MLA
  declares a latent cache, one compressed row per position; KDA
  declares recurrent state *and* its three convolution windows;
  conv-QKV attention declares KV rows and a convolution history on the
  same layer; a stateless operation declares none. A consumer that can
  hold only rows refuses a layer it cannot serve rather than allocating
  half of one, and every reporting surface derives its account of
  continuation from this same declaration — a summary that classified
  layers itself reported a hybrid's seven growing latent caches as
  constant-size recurrent state, which is the defect this rule exists
  to prevent. §17.3's rule is unchanged and now fully general: state
  geometry is a container fact.

**Acceptance drill (paper, before freeze).** Four deliberately awkward
architectures must be describable using extension points, without
changing the meaning of any existing field:

1. a pure SSM decoder — no attention anywhere;
2. a KDA + MLA + softmax hybrid carrying three continuation-state
   kinds — not hypothetical: this is Kimi-Linear-48B, already
   executing (§1.1);
3. an audio → perception → projector → text system with a drafter —
   components and hidden-state edges under load;
4. an unknown low-bit sparse-MoE representation with permuted
   experts — §5.7 preservation under a codec nothing yet reads.

A failure names the field whose meaning would have to change; that
field is exactly what Final must fix.

**The drill ran on 2026-08-30** —
[`docs/vindex3-ontology-drill.md`](../../../docs/vindex3-ontology-drill.md)
records the evidence and the sixteen findings. Summary: every schema gap
found lands inside these two lifts (mandatory-and-fabricated attention
surfaces, the fail-open census on undeclared families, the
inexpressible KDA/MLA state, one judged fact the container cannot carry,
one silent serde default), plus two preservation defects against §5.7
and two scope decisions to state explicitly. Cases 3 and 4 passed —
the component algebra holds as scoped, and the additive-evolution claim
that carried residency out of these gates is real. The ontology
question itself did not flinch.

**The schema-6 delta, defined once.** Both lifts land in a single
intentional semantic break — `GRAPH_SCHEMA` 5 → 6, one re-encode of the
corpus — never as two migrations discovering each other halfway:

```text
operation-program-derived surfaces   completeness from the declared
                                     program; object kinds imply nothing
attention absent when absent         presence means semantic presence,
                                     not successful serialization
ContinuationState declaration        per-operation state schema
KDA state precision                  declared, never chosen by the executor
MLA latent-KV geometry               declared, never invented
per-op norm-epsilon override         the MLA kv_a_layernorm fact (F6)
explicit per-layer operator          no absent-means-softmax default (F7)
fail-closed layer census             undeclared families block (F3)
closure at encode                    encoding is the proof boundary (F4)
```

The schema-6 acceptance test is severe by design: an encoder may not
emit a v6 graph unless declared operations ↔ required surfaces ↔
complete semantic parameters ↔ required continuation-state schema ↔
closed operand estate all reconcile. The Final invariant this buys:

> **The encoded operation programme is the completeness authority, and
> an encoder may not emit a programme whose required semantics or
> operands do not close.**

Validation witnesses (real checkpoints, in order):
`state-spaces/mamba2-780m` — 48 Mamba2 layers, `attn_layer_idx: []`,
zero attention: the pure case that must encode with no attention surface
anywhere — **passed live 2026-08-30** (the `AntonV/mamba2-780m-hf`
conversion): the 19-blocking refusal fell to `0 blocking` for semantic
reasons (a registered Mamba2 operator judgment consuming the SSM keys;
the five init-only `time_step_*`/`rescale_prenorm_residual` keys graded
declaration-only), the container encoded with all 434 operands closing
at encode, carries 48 `mamba2` operators with **no attention and no FFN
surface anywhere**, and opens through ordinary LQL with the source
checkpoint deleted — `STATS` reports `0 sliding / 0 full / 48
recurrent`, continuation as recurrent state only. **The generic executor
landed the same day and paritied**: teacher-forced against the banked
fp32 reference on three prompt lengths (one crossing the SSD chunk
boundary), all 430 scored positions agree within 7.6e-4 max-abs — ~3.6×
the reference's own step-vs-scan fp bound — with argmax exact at every
position and all three 32-token greedy trajectories reproduced
token-for-token; ordinary `INFER … GENERATE` then reproduces the
reference continuation word-for-word with the source checkpoint hidden; `state-spaces/mamba2attn-2.7b` — six attention layers at
declared indices among Mamba2 blocks: the A/B proving surfaces exist
only where the declared program uses them, and that KV and SSM state
coexist per-operation; then `tiiuae/Falcon3-Mamba-7B-Instruct` as the
product-scale witness. Kimi-Linear remains the three-state hybrid
witness for the ContinuationState half.

---

## 18. Operations, observation, and the independent reader

### 18.1 The operations surface

LQL operates on V3 containers with full V2 parity: `USE` (bind),
`INFER`/`GENERATE` (execute), `WALK`/`DESCRIBE`/`SELECT` (query),
`TRACE` (observe — on V3 bindings the observational trace runs without
perturbing execution), overlay/patch statements (mutate), `DIFF`
(compare), `COMPILE` (persist), `COMPACT` (maintain). Statements that a
V3 binding cannot serve refuse explicitly, naming the generation — the
whole-language sweep guarantees no statement falls into an accidental
backend path.

### 18.2 Serving

A served V3 container answers `/v1/completions`, `/v1/chat/completions`
(including tools/structured output) and `/v1/responses`, sharing every
wire shape with the V2 path so the two runtimes cannot drift in what a
client sees. See the runtime document §5.

### 18.3 The `vindex` reader

The format-native tool (`crates/vindex-cli`, binary `vindex`) answers
**from the artifact alone** — `index.json`, the system graph, the
segment headers — with no inference runtime attached:

```
vindex inspect          the container, reconstructed from itself
vindex describe         one logical object, in full
vindex representations  the physical directory, with recorded fidelity
vindex diff             one object under two representations, value by value
vindex represent        compile a representation beside the original
vindex precision        bits per weight — derived, never asserted
vindex verify           the container against its own recorded hashes
```

Every command speaks `--json`. The doctrine: VINDEX3 is defined by its
documents, not by any tool, and an artifact must not require an engine to
be understood. Recorded honestly: this binary is the canonical *reader*,
not yet an independent *implementation* — it links the `larql-vindex`
tree; carving out a dependency-light `vindex-core` is a named gate for
Final (§21), because the independent-reader test only counts when the
reader cannot inherit the writer's assumptions.

---

## 19. Mutation contract

Mutation is **overlay, never rewrite**. Patch and overlay statements
change *effective* operands; the base container's files are not modified.
Effective state is:

- **observable** — TRACE and the query surface see the model as mutated;
- **comparable** — logical DIFF operates over effective model state, not
  file bytes (§20);
- **durable only by COMPILE** — `COMPILE CURRENT INTO VINDEX` materialises
  effective operands into a new standalone container, stamped
  `authority: derived` with `derived_from_model` provenance (§5.2, §9.2).

Nothing is destroyed: the base container remains bit-identical, and a
compiled container is a sibling artifact, never an in-place upgrade.

---

## 20. Equivalence contract

The guarantees that make transformations provable rather than asserted:

- **Verification (encode-time).** `larql vindex3 verify` compares four
  explicit authorities — Declared (HF), Resolved (detection), Graph,
  Encoded — structurally and semantically, plus byte-payload equivalence
  with both ends re-hashed at verify time, so a drifted checkpoint fails
  differently (`source ≠ recorded`) from a corrupted container
  (`encoded ≠ recorded`). "Tensor count before == after" is not
  verification and does not appear in this format.
- **Logical DIFF.** `DIFF` compares effective model state — objects,
  representations, values — between two containers or a container and
  `CURRENT`, deriving error rather than asserting it. `vindex diff` is
  the artifact-only projection of the same guarantee.
- **COMPILE equivalence.** A compiled container must answer
  `INFER / GENERATE / TRACE / WALK` equivalently to the effective state
  it materialised — equivalence is gated, not presumed, and the result
  is stamped `derived`, never `canonical`.
- **COMPACT identity.** `COMPACT` reorganises physical storage and MUST
  preserve semantic identity — same graph, same effective values, same
  answers. It carries byte-identically what it cannot decode, and it
  discards unreferenced files only as named entries in its report —
  silent discard is forbidden (§5.7).

---

## 21. Toward 3.0 Final

What the candidate settles: one canonical container model (§5), the
contract stack (§0), compatibility rules (§5.7), the schema span (§12).

**The architecture is closed.** Four questions that were open when this
document was promoted are no longer open, and they are recorded here as
settled rather than as gates, so that nothing on the remaining list is
about what VINDEX3 *is*:

| Closed | Where | What closed it |
| ------ | ----- | -------------- |
| **The ontology lift** | §17.4 | both halves landed inside one schema span — lift 1 at SystemGraph schema 6 (2026-08-30), lift 2 additively within it (2026-08-31); the four-architecture drill's findings F1–F16, each with what closed it, in [`docs/vindex3-ontology-drill.md`](../../../docs/vindex3-ontology-drill.md) |
| **State semantics** | §17.3–§17.4 | continuation is a set of declared regions with three species — KV rows, a latent cache, fixed-size recurrent buffers — sized from the plan, refused where undeclared, and consumed by execution and by every surface that reports it |
| **Schema 6** | §5, §12 | presence means semantic presence; the per-layer operator is explicit; the census fails closed; closure is enforced at encode |
| **The witness ladder** | §16 | a pure-SSM decoder, a 250M mixed-program rehearsal, a 2.7B scale witness, and a 48B three-operator hybrid — each admitting through judgments alone and executing through the generic path, the last of them re-encoded with all five representation payload hashes unchanged |

**Execution placement is not a gate.** A backend's residency policy can
make a semantically complete container infeasible to run on a given
machine — Kimi-Linear-48B's routed-expert bank expands from 94 GB stored
to roughly 188 GB resident under one CPU policy — and that is an
execution-strategy fact, not an incompleteness in the model, the
representation or the declared state. §5's separation of logical
identity from stored bytes, and §17.3's rule that state geometry is a
container fact, both hold across it.

What remains before this specification drops "candidate" is therefore
release closure — the freeze, an independent reader, the held-out test,
the maturity flip — each a named gate, none of them drift:

| # | Gate | Today |
| - | ---- | ----- |
| 1 | **Shape convergence executed** (§5.6): LYRW banks producible as a graph-container representation's segments, or the bank writer retired to importer-only status with its output re-encodable to the graph shape | two disjoint writers |
| 2 | **Required/optional freeze**: an RFC-2119 pass over §5–§9 separating normative requirements from extensions | this document's tables are the input |
| 3 | **Independent reader**: `vindex-core` carved out so the reader stops linking the writer's tree; a minimal conformance harness over published fixtures | reader exists, boundary impure (§18.3) |
| 4 | **E8 held-out architecture** (§16 criterion 7) | not yet run |
| 5 | **The M4 flip**: `DEFAULT_EXTRACTION_GENERATION = V3` per the generation policy | M1–M3 done, M4 open |
| 6 | **Bank-ABI pre-freeze rows**: the remaining V2-0..V2-4 experiment gates (profile-authority derivation, variant-selection refusal, fixtures B–D, WALK/DESCRIBE parity) | open |

Feature growth is not a gate: GENERATE, TRACE, overlays, logical DIFF,
COMPILE and COMPACT are operations over V3 containers and do not add
bytes to the format unless they persist state another implementation
must understand (§0's test) — which is exactly why the candidate can be
stable while the engine keeps moving.

---

## License

Apache-2.0
