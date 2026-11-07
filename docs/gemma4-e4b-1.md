# GEMMA4-E4B-1: per-layer input streams and cross-layer K/V in VINDEX3

**Class: FREEZE.** Frozen 2026-11-07 against main `6b62f6d2`, before any
implementation. It follows the [reconnaissance](gemma4-e4b-1-reconnaissance.md).
The machine-readable contract is
[`gemma4-e4b-1-freeze.json`](arch-conformance/forecasts/gemma4-e4b-1-freeze.json).
It is the authority for every number, pin, tolerance, test case and forecast;
this document states the contract in prose. Both change only by a new, dated
section. Execution-time findings go in `gemma4-e4b-1-execution-notes.json`;
evidence goes in `gemma4-e4b-1-evidence.json`. Both are append-only and are
created by the first rung that has something to record.

**Programme.** Gemma 4 12B, 26B-A4B and 31B already plan, encode and execute
in VINDEX3. The two "effective" checkpoints, E4B and E2B, are the only rows in
the 117-row conformance corpus that declare per-layer input embeddings or
cross-layer K/V sharing (`docs/arch-conformance/matrix.json:87-88`,
`docs/arch-conformance/README.md:996`). E4B is refused for text generation on
three findings and E2B on four. This rung adds the two missing relationships
to the graph and makes both models text-generation complete. **Scope is text
generation only.**

## The question

> Can VINDEX3 represent a dependency **between layers** as a declared graph
> relationship, with cache ownership, lifetime and memory accounting
> following from the graph rather than from Gemma flags or executor rules?
> The two cases are:
>
> - a layer's attention reading K/V that another layer produced;
> - every layer reading its slice of a per-token stream computed once
>   before the stack.
>
> And can it then execute E4B and E2B indistinguishably from the primary
> oracle, with every refusal named?

Today the answer is **no**. The graph treats layers as independent sequential
blocks. The citations are in reconnaissance §0:

- the per-layer policy names no other layer (`crates/larql-vindex/src/format/vindex3/graph/policy.rs:279-338`);
- operands name only stored tensors (`crates/larql-vindex/src/format/vindex3/opplan/ops.rs:15-22`);
- the only edge is cross-component (`crates/larql-vindex/src/format/vindex3/graph/edge.rs:5-27`);
- the planner's probe says "every layer in the graph projects its own K/V" (`crates/larql-vindex/src/format/vindex3/plan/carriage.rs:428-434`);
- an attention step is refused unless its view ends at its own position (`crates/larql-vindex/src/format/vindex3/opplan/exec/backend/step_calls.rs:108-124`).

## Decisions taken in this freeze

| # | Decision | Rejected alternative, and why |
|---|---|---|
| D1 | **GRAPH_SCHEMA 7, only for containers that use a new relationship.** v6 containers are untouched. New readers accept 6 and 7. The encoder writes 7 if and only if the graph uses `kv_source` or `layer_input_stream`, and lists each one in a non-empty `required_features`. | *Additive within v6.* The graph types do not deny unknown fields, so a v6 reader would silently ignore both relationships and compute a different function: HF's `use_cache=False` arm. *A corpus-wide v7.* Re-encoding the registry for no semantic change. |
| D2 | **Option B: typed cross-layer references**, shaped to grow into a general value graph. `KvSource::Layer { producer: ValueRef { layer, site } }` and `Component.layer_input_stream`. | *A flag* (`num_kv_shared_layers` on the surface, derived by the executor). This is the legacy design (`crates/larql-models/src/architectures/gemma4.rs:57-84`); it puts the Gemma rule in executor code and hides cache ownership. *A full value DAG.* Every executor is a fixed per-layer walk, so that would be a rewrite. |
| D3 | **Preserve the shared layers' unused `k_proj`, `v_proj` and `k_norm`** (110 MB on E4B) byte-exact and with provenance, as `retained_unreachable` tensors that cannot be bound. | *Drop them.* That loses source bytes. Google's reference says they are vestigial: "after checkpoints remove the kv_einsum from the shared layers". Under D1 they are not needed for fail-closed behaviour. |
| D4 | **Bump planner semantics** in every rung that changes admissibility or interpretation (R2, R3, R4, R5), by exactly one from whatever main carries at merge. | *One bump at first admissibility.* The R2 to R4 rungs each change what a finding means. |
| D5 | **Text generation only.** Image and audio stay inadmissible and *gain* an explicit blocker for the stream's non-text positions. | *Implicit completeness.* Clearing `vision_config.use_clipped_linears` later would otherwise admit images with an unimplemented pad-id rule. |
| D6 | **Primary oracle: transformers with `use_cache=True`, eager attention, f32.** The secondary oracle is Google's JAX reference, spot-read here and optionally spot-checked at R6. It is report-only. | *HF with default arguments.* `use_cache=False` silently disables sharing (§0). |
| D7 | **E4B first, then E2B.** E2B also needs R5. | — |
| D8 | **The graph carries producers explicitly.** The planner checks them against the HF rule and records the Google rule as a second derivation. If the two disagree, the checkpoint is refused. | *Re-derive producers in the executor.* The two oracles use different rules (§0), so an executor-side derivation would silently pick one. |
| D9 | **Cross-span sharing is refused** in this rung, by name. That covers a sliding consumer with a full producer, a full consumer with a sliding producer, and unequal windows. | *Admit it because it is well-defined.* No checkpoint needs it, and each case adds an unexercised mask or retention path (§1, K7). |
| D10 | **The pre-norm scale `hidden_size^-0.5` is a declared value in the stream**, not folded into weights. HF applies it and Google does not, which changes the effective norm epsilon (§0). | *Fold it into the weights.* That would erase the one place the oracles differ. |

## 0. Baseline facts

These were re-verified at `6b62f6d2`, or read in the pinned oracles.

**LARQL**

- `GRAPH_SCHEMA = 6` (`crates/larql-vindex/src/format/vindex3/graph/mod.rs:103`). The reader accepts exactly 6 (`crates/larql-vindex/src/format/vindex3/inspect.rs:85-99`), and the schema is read before the graph (`inspect.rs:142`).
- Two other readers bypass that check:
  - `crates/larql-vindex/src/format/vindex3/represent/pipeline.rs:490-492` parses the graph, then re-serialises it, with no schema check. A round trip through it would drop unknown fields.
  - `crates/larql-vindex/src/format/vindex3/opplan/exec/kimi_source.rs:100-105` reads the graph untyped.
- `PLANNER_SEMANTICS_VERSION = 23` (`crates/larql-vindex/src/format/vindex3/plan/report.rs:438`).
- The carriage rules for the three blocked keys have constant probes: `crates/larql-vindex/src/format/vindex3/plan/carriage/rules_surface.rs:336-349` (`num_kv_shared_layers`), `:381-390` (`hidden_size_per_layer_input`) and `:391-398` (`use_double_wide_mlp`).
- Placement uses substring matching (`crates/larql-vindex/src/format/vindex3/graph/build.rs:117-132`, `:154`). As a result:
  - `embed_tokens_per_layer` is silently filed into `target.embedding`;
  - `per_layer_projection_norm` is silently filed into `target.final_norm`;
  - `per_layer_model_projection` is unowned (`crates/larql-vindex/src/format/vindex3/graph/build/components.rs:189-191`);
  - the per-layer PLE tensors have no role, so the closure records them as `UnclassifiedOperand` (`crates/larql-vindex/src/format/vindex3/opplan/build.rs:605-608`).
- `FfnSurface.intermediate_size_by_layer` already exists (`crates/larql-vindex/src/format/vindex3/graph/surface.rs:137-144`).
- `LayerContinuationGeometry::Stateless` exists, but nothing produces it (`crates/larql-vindex/src/format/vindex3/opplan/exec/continuation.rs:159-165`).

**Primary oracle** (transformers 5.5.0, `modeling_gemma4.py`, sha-pinned)

- Sharing happens only when `past_key_values is not None` (L1198). The default forward creates a cache (L1558, L1571-1572).
- The producer is the last pre-shared layer with the same type (L1147-1159).
- PLE (L1516-1531, L1622-1678; injection L1393-1402):
  - lookup row × 16;
  - plus RMSNorm_w(projection × H^-0.5) per 256-wide slice;
  - × 2^-0.5;
  - injected after the FFN residual and before `layer_scalar`.

**Secondary oracle** (google-deepmind/gemma `9e1f7413`, `gemma/gm/nn/gemma4/`)

- Sharing is **unconditional**: the layer-sharing K/V is always returned, whether or not a decode cache exists (`_modules.py` L377-397). So Google's function equals HF with `use_cache=True`.
- Shared layers compute their own K/V and then discard it. The TODO marks those weights as vestigial (L275-287).
- The producer rule is **positional**: global layers use num_unshared−1 and local layers use num_unshared−2 (`_config.py` L55-84). It agrees with HF on E4B (22, 23) and E2B (13, 14) only because of how those layer patterns fall.
- There is **no** H^-0.5 before the PLE norm (`_modules.py` L192-193), and the HF converter does not rescale the weights. HF's effective epsilon is therefore H·1e-6 (2.56e-3 on E4B), against Google's 1e-6.

## 1. Frozen contract: cross-layer K/V

The consumer's attention is described in three separate parts:

- **(a) the producer and its site**: `ValueRef { layer, site: AttentionKv }`;
- **(b) the consumer's own attention policy**: its mask kind, window, causality, score scale and softcap, all unchanged from today's `AttentionLayerPolicy`;
- **(c) the shared history**: its row semantics, position indexing and retention.

Matching geometry and RoPE is necessary. It is **not** sufficient. Every
property below is a verification obligation with tests, listed in the JSON
under `obligations.kv_contract`.

- **K1 Producer.** The producer is an earlier layer in the same component, and its own K/V source is `Own`. Chains are refused, and so are forward and cross-component references.
- **K2 Value.** The shared rows are K after `k_norm` and RoPE, and V after the parameter-free `v_norm` (or derived from K under the producer's `v_from_k`). There is one row per absolute position, written once, by the producer.
- **K3 Positions.** Rows are indexed by absolute position in the component's token stream. **RoPE is applied once, at the producer**, using the producer's policy. The consumer rotates only Q, using its own policy at the query's absolute position. Admission requires exact equality of the position policy (kind, theta, rotary fraction, basis).
- **K4 Geometry.** `head_dim` and `num_kv_heads` must be equal.
- **K5 Consumer policy.** The consumer applies its own mask, window, scale and softcap over the producer's rows. Nothing about the producer's mask is inherited.
- **K6 Retention.** The producer keeps the **union** of the history ranges required by itself and by all its consumers. No provider evicts a row that any member of that union still requires.
- **K7 Span interactions.**
  - Same span and equal window: admitted. This covers every Gemma 4 pair.
  - **Sliding consumer, full producer: refused.** It is well-defined, but Gemma's sliding and full layers also differ in `head_dim` and RoPE, so K3 and K4 would refuse it anyway, and no model needs it.
  - **Full consumer, sliding producer: refused.** Under K6 the producer would have to keep rows beyond its own window, which changes its memory class.
  - Unequal windows: refused.
  - Widening any of these requires a new dated section.
- **K8 Ordering.** The producer runs before the consumer within each step. In batch prefill, the producer appends the whole chunk before any consumer reads.
- **K9 Step contract.** A new constructor, `AttentionStepCall::reading`, takes the producer's view, which must end at `position + 1`. It splits off the last row as the fresh K/V, so the attention kernel sees exactly the shape an own-K/V step gives it.
- **K10 Providers.** Consumers are `Stateless`. Canonical and window providers serve shared history. Codec providers refuse, by name, any program that has consumers, until a CODEC rung measures them.
- **K11 Rollback and verify-N.** Truncation acts on producers only. After a rollback to position p, a consumer reads exactly the producer's rows before p.
- **K12 Memory.** K/V bytes are the sum over producers of retained rows × row bytes. Consumers contribute 0. The expected bytes are in the JSON (E4B at 32K: 557,842,432 shared vs 976,224,256 unshared).
- **K13 Device.** Consumers bind the producer's buffers and allocate none of their own.
- **K14 Distributed.** A plan that puts a producer and its consumer on different shards is refused by name.

**Planner.** The planner gets three probes:

- the declared count, against the number of consumer layers;
- each consumer's producer, against the HF rule;
- each consumer's producer, against the Google rule.

A disagreement refuses the checkpoint (D8).

## 2. Frozen contract: the per-layer input stream

- **P1 Declaration.** `Component.layer_input_stream` declares:
  - the slice width and the number of layers;
  - two sources, each with explicit scales: `TokenLookup` (scale 16) and `EmbeddingProjection` (pre-norm scale H^-0.5, then a per-slice weighted RMSNorm with its declared epsilon);
  - the combine scale 2^-0.5;
  - the injection: site `AfterFfnResidual`, form `GatedProjection` (gelu_tanh, then a weighted post-norm).
- **P2 Token ids.** The stream is computed in the prelude, from the text token id at each position. A position without a text id is **refused by name**. That covers embedding-only entries and image or audio soft tokens. Nothing is zero-filled or reverse-mapped. This is why D5 adds an explicit image and audio blocker.
- **P3 Injection.** The injection runs after the FFN residual add and before `layer_scale`. Slice i goes to layer i.
- **P4 Residency.** The 5.6 GB lookup table belongs to a lookup-only residency class: host-mapped, read one row per position, never resident on the device, and never counted in device weight budgets.

**Placement** (R2):

- `embed_tokens_per_layer`, `per_layer_model_projection` and `per_layer_projection_norm` get their own objects. The rules that claim them are ordered ahead of `embed_tokens` and `norm`, and apply only when the stream is declared.
- Prefix matching respects `.` boundaries.
- The per-layer gate, projection and post-norm tensors get roles, again only when the stream is declared.

**Double-wide MLP** (R5):

- `intermediate_size_by_layer` is filled from the tensor shapes.
- The `use_double_wide_mlp` probe checks for exactly 2× width on every `KvSource::Layer` layer and 1× everywhere else.

## 3. Frozen contract: preservation separate from execution

- **P-PRES1.** Every tensor carries:
  - provenance: source repo, revision, name, source dtype and shape, and the sha256 of its source bytes;
  - a class: `active` or `retained_unreachable`.

  Canonical encodings are byte-exact preserved originals whichever class they are in.
- **P-PRES2.** Byte-exact recovery: every retained tensor, read back from the container, matches its source sha256.
- **P-PRES3.** Unreachability is enforced in four ways:
  - (a) Retained roles appear in no operand table.
  - (b) The closure check, **at encode and again at open**, refuses any `OperandRef` that points to a retained tensor, with `ClosureDefect::RetainedTensorBound`.
  - (c) A negative test sets a consumer's source to `Own` and must be refused, never silently bind `k_proj`.
  - (d) A negative test forges an op plan that binds a retained tensor, and that plan must be refused at open.
- **P-PRES4.** Every retained tensor carries a reason the graph still justifies, for example `ShadowedByKvSource { layer }`. A retained tensor without one is refused.
- **P-PRES5.** Retained tensors are never prepared or loaded for execution. They are counted separately in the census and stay out of execution budgets.

The negative controls for sharing and for the stream are produced **on the
oracle side** (§4). The unreachability property is never weakened to build a
control.

## 4. Frozen contract: reader compatibility (fail-closed)

| reader | v6 container | v7 container (E4B) |
|---|---|---|
| v0.2.1 release (`bb201dc1`, asset sha pinned) | **C1** works, unchanged | **C2** explicit refusal "schema v7, this build reads v6", non-zero exit, nothing on stdout, at **every** entry point that opens a container |
| new reader | **C3** works; byte-identical ids and planes vs the base commit on the gemma-3-4b-it and gemma-4-12b-it registry containers | **C4** works (V1 to V6) |
| new reader, a backend without a required feature | works | **C5** refusal by name at prepare |

Refusals that are not tied to the old release:

- **C6**: a v7 container that lists an unknown feature is refused, naming the feature.
- **C7**: a v6 container that carries any v7 field is refused.
- **C8**: a v7 container with empty or missing `required_features` is refused.
- **C9**: every graph reader goes through the check, including the represent pipeline round trip and the Kimi geometry reader.
  - A v7 container must survive the new represent pipeline byte-for-byte in its features.
  - Its output from the v0.2.1 represent pipeline must be refused by the new reader. C8 catches this, because the old round trip drops `required_features`.

**C2 is the negative test** for "an old reader never silently produces output
from an incomplete representation."

## 5. Witnesses and tolerances

Everything below is fixed in the JSON under `metric_definitions`,
`tolerances` and `test_cases`.

**Metrics**

- `max |Δ logprob|` over the **full** 262,144-entry vocabulary. Each side computes `log_softmax` in **float64** from its own f32 post-softcap logits.
- **Ordered top-5** with a pre-declared near-tie rule, τ = 0.002 nats:
  - an adjacent transposition across a margin ≤ τ is recorded as `tie_swap` together with its margin;
  - any other difference fails;
  - margins m1 to m5 are recorded at every position.
- Per-plane `rel_l2` and cosine, in float64. Plane 0 is the embedding output; plane i+1 is the residual leaving layer i after `layer_scalar`.
- A PLE tap and K/V taps.

**Tolerances**

| comparison | gate |
|---|---|
| reference vs HF | max\|Δ logprob\| ≤ 1e-3 nats, plane rel_l2 ≤ 2e-5 and cos ≥ 0.99999, top-5 under the near-tie rule |
| PLE tap | rel_l2 ≤ 2e-5 |
| K/V read by a consumer | **bit-identical** to the producer's rows, and the same allocation on device |
| LARQL cached decode vs LARQL fresh prefill | ≤ 5e-4 nats, rel_l2 ≤ 1e-5 |
| production vs reference | same gates as reference vs HF |
| lowered Metal vs reference | ≤ 1e-6, the G4.3 gate carried over verbatim; if it fails, that is recorded as a finding and the gate is not relaxed |

**Cases**

- **Chat prompts**: 8 fixed prompts (texts in the JSON), using the pinned chat template, plus 64 greedy tokens each. Greedy output is gated up to the first HF near-tie and is report-only after it.
- **Fresh prefill lengths**: 1, 7, 64, 511, 512, 513, 1023, 1024, 1025, 2048 and 4096, drawn from the pinned passage. Every position is scored.
- **Teacher-forced decode**: (7 + 64), (500 + 32) crossing 512, and (1020 + 16) crossing 1024.
- **Prefill chunkings** {whole, 1, 7, 512} at lengths 513 and 1025.
- **Comparisons**:
  - fresh prefill vs HF fresh;
  - **cached decode vs HF fresh**, at matching absolute positions;
  - LARQL cached vs LARQL fresh;
  - each chunking vs whole;
  - HF cached vs HF fresh, report-only.
- **Token ids** are chosen once, by LARQL's pinned tokenizer, and committed before any comparison.

**Controls that must fail**

- **NC-KV**: LARQL against HF with `use_cache=False`.
- **NC-PLE**: LARQL against HF with `per_layer_inputs` forced to zero.

If either control passes, every V1 pass is void.

**Pins**

- Checkpoint revisions, plus LFS and blob hashes.
- transformers 5.5.0 and torch 2.14.0, with four file hashes.
- The Google commit, with five file hashes.
- LARQL base `6b62f6d2`; previous release v0.2.1.
- `harness_version` = `gemma4-e4b-1-oracle/1`.

## 6. Forecasts (planner verdicts, scored against a uniform cache)

A precondition: all 117 rows are re-planned by one binary at the base commit
before R2 is measured.

| rung | gemma-4-E4B-it (text blockers) | gemma-4-E2B-it | every other row |
|---|---|---|---|
| base | 3 | 4 | — |
| R2 placement | 2 | 3 | unchanged |
| R3 `kv_source` | 1 | 2 | unchanged; 12B, 31B and 26B-A4B stay v6 |
| R4 stream | **0, admissible** | 1 | unchanged |
| R5 double-wide | — | **0, admissible** | unchanged |

- **F-MM.** Image- and audio-conditioned generation stay inadmissible on both rows at every rung, each with an explicit non-text-position blocker.
- **F-SCHEMA.** Re-encoding any model that does not use a v7 feature yields a v6 graph byte-identical to the base commit's.

**Falsifiers**:

- any other row moving;
- a moving row moving by a different count, or for a different subject;
- E4B becoming admissible before R4, or E2B before R5;
- any multimodal capability becoming admissible.

## 7. Rungs

| rung | content | gate | size |
|---|---|---|---|
| R1 | this freeze | committed before code | docs |
| R2 | placement, roles, tensor class + provenance, retained role | F-R2; P-PRES1, P-PRES3(a, b), P-PRES4 | ~10-15 files, +0.6-1k |
| R2b | oracle harness, token bank, NC arms (no Rust) | HF-side self-test: the NC arms differ | ~2-4 files, +0.4-0.7k |
| R3 | `kv_source` IR, planner probes, v7 writer/reader, step contract, reference | F-R3; K1-K11 fixtures; C1-C3, C5-C9 | ~40-55 files, +2.5-3.5k |
| R4 | stream IR, token-id threading, P2, P4, reference | F-R4; P2 refusal; legacy `ple.rs` as second oracle on the miniature | ~40-55 files, +2.5-3.5k |
| R5 | double-wide | F-R5 | ~5-8 files, +0.2-0.4k |
| R6 | encode E4B then E2B; V1-V4 and controls on reference; P-PRES2; C2 and C4 on real containers; Google spot-check, report-only | all gates; ladder `encoded`, `equivalent` | 1-2 days compute |
| R7 | production CPU, providers (K6, K10, K11), V6 | V4, V5, V6 | ~15-25 files, +1-1.5k |
| R8 | Metal device and lowered (K13, stream prelude); may fix #741 | V5, K13, V6 | ~15-25 files, +1-2k |
| R9 | ladder `operational`, then `published` (registry, separate PR) | stages recorded | separate |

R3 and R4 are independent after R2. R5 needs R3.

**Evidence record.** `gemma4-e4b-1-evidence.json` holds a per-model ladder:
`admissible`, `encoded`, `equivalent`, `operational`, `published`.

- Runs are **appended** and never edited.
- A failure is never overwritten. A later pass is a new run that names the failure it follows.
- A run whose pins do not match is an `instrument_failure` and proves nothing either way.
- The JSON lists the required fields.

## 8. Separately tracked defects

- **#740**: on the Metal lowered path, sliding-window layers allocate a full `max_positions` K/V cache (`crates/larql-vindex/src/format/vindex3/opplan/exec/metal_lowered/construct.rs:148-149`). It is tracked independently, with its own memory-regression benchmark, and is not absorbed here. V6's device assertion counts allocated layers (K13), not per-layer sizes, and records #740's state.
- **#741**: the Metal lowered path refuses `layer_scalar` on a dense FFN (`crates/larql-vindex/src/format/vindex3/opplan/exec/metal_lowered/construct.rs:90-100`). E4B and E2B are dense, so R8 is blocked by it. R8 may implement the fix, referencing #741. It stays a separate defect with its own test.

## 9. Out of scope

- Image and audio conditioning, including the pad-id rule for the stream at soft-token positions and `use_clipped_linears`.
- Gemma 3n: AltUp, LAuReL, and stream injection into AltUp.
- YOCO-style producers that are not attention layers (`ValueRef.site` leaves room for them).
- Quantised encodings of either model, and of the lookup table.
- Codec providers serving shared history (K10 refuses them).
- Cross-shard sharing (K14 refuses it).
- Registry publication beyond R9's own PR.

## 10. Acceptance

GEMMA4-E4B-1 is complete when:

- R2 to R8 have passed their gates on E4B and then E2B;
- every control has failed at its event;
- C1 to C9, K1 to K14, P1 to P4 and P-PRES1 to P-PRES5 have passing tests;
- F-R2 to F-R5, F-MM and F-SCHEMA are recorded as holds or as falsifications;
- the evidence ladder reaches `operational` for both models.

A falsified forecast is a finding, not a reason to change a gate. A gate
changes only through a new dated section, written before the run it governs.

## Not pinned at freeze

- The token ids (committed at R2b, before any comparison).
- The transformers wheel's own hash.
- Torch's BLAS backend (recorded per run).
- The uniform conformance cache (absent locally at freeze).
- Absolute semantics numbers.
- Whether the G4.3 1e-6 lowered gate holds at 42 layers on BF16.
- The size of the HF vs Google pre-norm epsilon divergence (X-GOOGLE-1).
- No LiteRT or gemma.cpp reading was done.
