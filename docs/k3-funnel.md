# K3 Funnel v0.1 — the three-rung adapter ladder

**Programme:** R1 (GPT-OSS-20B) → R2 (Kimi Linear 48B-A3B) → R3 (Kimi K3, 2.8T)
**Scope:** the *model-side* half of the K3 work — extraction, adapter, harvest, carve, serve. The transport half is [`dec-funnel.md`](dec-funnel.md); the production half is [`vindex-factory.md`](vindex-factory.md).
**Status:** v0.1 — draft for review. Supersedes the single-step DEC-6 plan in [`dec-funnel.md` §3 DEC-6](dec-funnel.md); DEC-6/DEC-7 keep their claims (C8) and gates, but their *content* becomes rung R3 of this ladder.
**Date:** 2026-07-28

> **Definitions flagged as new.** The phase scheme (§3) and gates G0/GA/GB/GC/GD (§4) are **defined in this document**. They are not in `dec-funnel.md` (which uses DEC-0…7 and claims C1–C8), and they were not in the experiments registry when this was written — programme `k3` was created from this document on 2026-07-28 (§10), so the registry now mirrors these definitions rather than sourcing them. Correct them here before the first rung runs: every cross-reference in §6 and every experiment `design.gates` block resolves through this section.

---

## 1. Thesis

DEC's risk is transport risk, and it is largely retired: C1 holds on both paths, the wire ladder and the drift gate are shipped, the batch curve is measured. What is *not* retired is **adapter risk** — the model-side code that has to exist before K3 is a thing you can serve at all.

The single-step plan puts the first execution of every novel component at 2.8T scale, where the only available ground truth is an API oracle you can compare top-k with. That is the weakest verification instrument in the estate, applied to the largest surface, at the highest cost per iteration.

The ladder's claim is narrow and structural:

> **Most of the K3 adapter's novel surface has an open ancestor that fits on the Mac.** Building against the ancestor first replaces API-oracle triangulation with layer-by-layer f32 diffing against a local reference implementation, for every component that has one.

What that buys, concretely: a norm-placement bug in KDA is an afternoon against a local reference and a weekend against an API oracle. The ladder does not reduce the total amount of code; it moves the debugging of that code to where ground truth is cheap.

**Cost of the detour:** ~2–3 additional weekends over the direct route. **Return:** two shippable artifacts and two demo-able results on the way up, and a bisection story at R3 — every Gate-B failure at K3 can be re-run against two working ancestors.

This is the house rule (reference tier before fast tier — see [`AGENTS.md`](../AGENTS.md)) applied to model choice rather than to kernels.

## 2. The three rungs

| Rung | Model | Size | What it is the reference for | Ground truth available |
|---|---|---|---|---|
| **R1** | `openai/gpt-oss-20b` | ~13 GB MXFP4 | MoE/format surface: expert-granular extraction, MXFP4 codec, capability tiers, harvest instrumentation, grouped scheduler | Local, bit-level — the checkpoint fits in RAM entirely |
| **R2** | `moonshotai/Kimi-Linear-48B-A3B` | ~48B total / ~3B active | KDA surface: linear-attention recurrence, ShortConv/gate stack, MLA-NoPE, 3:1 hybrid interleave, state checkpointing | Local — published reference kernels (FLA / vLLM) run on the same machine, diffable in f32 layer by layer |
| **R3** | `moonshotai/kimi-k3` | 2.8T, 16-of-896 | K3-novel delta only: block AttnRes, SiTU-GLU, QB frozen-bias router, LatentMoE W↓/W↑ wrapping, decay-parameterisation details, scale logistics | API oracle (top-k agreement at temperature 0) — plus bisection against R1 and R2 |

**R1 is small because you are already standing on it.** GPT-OSS is in the support table (`ExpertFormat::PackedMxfp4`, `crates/larql-models/src/architectures/gpt_oss.rs`), and DEC-0's routed arm already characterised expert-granular serving on a 26B MoE. R1 is mostly *productising measured behaviour*, not discovering it.

**R2 is where the risk actually lives**, and it is the rung with the strongest instrument. Kimi Linear is K3's direct architectural ancestor: KDA in the same 3:1 hybrid with NoPE MLA layers. Quantised, the whole model plus KV fits on the Mac with room to spare. It is also an MoE, so R1's expert pipeline gets its second model here for free.

**R3 becomes a delta, not a build.** With R1 and R2 passed, the K3-novel surface is three components and scale logistics.

### 2.1 Secondary value

R2 is a legitimate artifact in its own right — "Kimi Linear served decoupled on a MacBook" is the ancestor demo that sets up the K3 finale, and it lands months before K3 weights are extractable at 2.8T. Both R1 and R2 produce publishable vindexes ([`vindex-factory.md` §15.1](vindex-factory.md), which already ladders VF-0…VF-4 with VF-4 = K3).

## 3. Phases

Each rung executes the same five-phase spine, plus a sixth phase that only matters where a demo has to fit on a physical box. Phases are per-rung; a rung is complete when its phases have passed their gates.

| Phase | Name | Does | Exit gate |
|---|---|---|---|
| **P1** | Audit | read the config + reference implementation; confirm the format decoder; enumerate what the existing traits do *not* express | **G0** + **GA** |
| **P2** | Adapter | write the architecture/attention/FFN code; make one forward pass match the reference | **GB** |
| **P3** | Harvest | one instrumented forward run: routing trace, feature-activation log, up-saturation histogram, latent pools | **GA** (trace reproducibility) |
| **P4** | Extract | produce the vindex tiers — exact extents, walk carve, demo slice | **GA** |
| **P5** | Serve | route the extracted tiers through the expert tier; measure tok/s and movement ratio | **GD** |
| **P6** | Shrink | carve to what fits the demo box: hot-expert slice, skip-mode renormalisation, client requant | **GC** |

## 4. Gates

| Gate | Name | Criterion | Falsifier |
|---|---|---|---|
| **G0** | Trait absorption | every feature lands behind an existing trait with **no change to an existing method signature**. New default-impl'd methods paired with a capability flag are allowed and expected — that is the established convention in these traits, not a workaround (see §4.1). Surface: `ModelArchitecture` (`larql-models/src/config/`), `FfnBackend` (`larql-compute/src/ffn.rs:27`), `KvEngine` (`larql-inference/src/kv_engine/`), `GateIndex` (`larql-vindex/src/index/types/ffn_row/mod.rs:274`, blanket-impl over `GateLookup + PatchOverrides + FfnRowAccess`), plus the execution seams `LayerExecutor` (`larql-inference/src/layer_executor/mod.rs:70`) and `LayerGraph` (`larql-inference/src/layer_graph/mod.rs:62`) | an existing method's signature has to change, or a caller's invariant has to be silently redefined to avoid changing one → the abstraction was wrong for this model class; fix the abstraction before the rung proceeds, don't special-case the model |
| **GA** | Format fidelity | extraction round-trips: sha256 per extent, source-shard byte-range provenance map, dequantised tensors match the reference decoder on a fixed sample | any extent fails round-trip, or the dequant differs beyond fp tolerance |
| **GB** | Numerical parity | **R1/R2:** layer-by-layer f32 diff against the local reference implementation, then `larql shannon verify` ≤ 0.5 % bits/char (the repo's existing CI threshold). **R3:** top-k agreement at temperature 0 against the Kimi endpoint, preserved-thinking-mode aware | divergence outside tolerance at any layer, or bits/char gate failure |
| **GC** | Carve fidelity | the coverage curve — resident-expert fraction vs measured drift — stays inside the **C6** gate ([`dec-funnel.md` §DEC-1A](dec-funnel.md), shipped as `larql dec-bench drift`, `crates/larql-cli/src/commands/primary/dec_bench/drift.rs`), including skip-mode gate-weight renormalisation on slice misses | no resident fraction achieves the target size inside the drift gate → the demo tier does not exist at that size, and the carve is renegotiated, not the gate |
| **GD** | Serving | rung-appropriate tok/s sustained, coherent, with `dec/movement_ratio` recorded; grouped scheduler measured against per-row streaming | serving compute saturates, or movement ratio lands outside the DEC-measured band (10⁻³–10⁻⁴) without an explanation |

**G0 is checked once per rung, at P1, and it is cheap.** It is also the ladder's first result: twenty-three features across five crates absorbed with no trait signature change is a claim about the architecture, and it is falsifiable on the first afternoon of R1.

### 4.1 G0 run — R1/P1, 2026-07-28

Read-only pass over the trait surface. **Result: three pass, one was never on the path, one open decision.**

| Trait | Verdict | Notes |
|---|---|---|
| `ModelArchitecture` | **pass** | a wide, all-defaults key-pattern trait. New architectures are new impls plus, where needed, new `Option`-returning accessors that default to `None` — the established pattern (`fused_qkv_key` and `position_embed_key` for GPT-2, `packed_gate_up_blocks_key` for MXFP4). Items 7, 8 land as new modules registered in `architectures/mod.rs`. |
| `FfnBackend` | **pass** | four methods, one of them already the MoE hook `forward_moe_full_layer(layer, h_post_attn) -> Option<Array2<f32>>`. Items 14 and 15 (three per-expert serving modes, skip-mode renormalisation, dense-as-one-expert) are impl-internal; routed dispatch already has a home in `larql-inference/src/ffn/moe_remote/`. |
| `LayerExecutor` / `LayerGraph` | **pass** | these, not `Dispatcher`, are the execution seams the hybrid interleave (item 11) rides. |
| `KvEngine` | **pass, but the gate needed sharpening** | see below. |
| `GateIndex` | **open decision** | see below. |

**`KvEngine` — the pre-registered risk was misplaced.** The concern on record was that KDA's fixed-size recurrent state with checkpoint/restore would break the trait. It does not, because **there is no rollback, truncate, or checkpoint method on the trait at all** — `truncate_kv_cache` exists only on a test double (`larql-inference/src/test_utils/`), and `reset_and_preallocate_kv_cache` is a free function in the layer-graph setup path. Speculative rollback therefore needs a *new* method whether or not KDA is involved: it is the M-ladder's problem (M1) before it is R2's.

Seven of the trait's fourteen methods are already default-impl'd fallbacks (`prefill_quant`, `prefill_resident`, the four `*_via_executor` variants, `prefill_from_hidden`), and `supports_multimodal` + `prefill_from_hidden` is exactly the "capability flag guards an optional method" pair a KDA state-checkpoint method would follow. **G0 as originally written would have failed here for a boring reason** — hence the sharpened wording above: existing signatures are frozen; new default-impl'd methods are the sanctioned extension point.

**`Dispatcher` was never on this path — my error, corrected above.** `larql-inference/src/experts/session.rs:38` is `op_specs()` + `call(op, args)`: the WASM *virtual-expert* / tool-call dispatcher from the 769th-expert work. It has nothing to do with MoE expert routing and is removed from the G0 surface. The type that actually carries MoE ownership is `MoeShardConfig` (`larql-inference/src/ffn/moe_remote/config.rs`), which already models per-`(layer, expert_id)` ownership sets and per-layer expert-id ranges.

**`GateIndex` — the real finding, and it is a fork.** Every read method in the vindex API is addressed by layer, never by expert: `GateLookup` is `gate_knn(layer, residual, top_k)` / `feature_meta(layer, feature)` / `num_features(layer)` (`larql-vindex/src/index/types/gate_lookup.rs:12`), and `FfnRowAccess` is `ffn_row_dot(layer, component, feat, x)` with `component ∈ {gate, up, down}`. **There is no expert dimension in the reader.** Expert identity today lives at mount time — `expert_in_shard(e, filter)` in `format/weights/load/mod.rs:34` and `MoeShardConfig`'s `(layer, expert_id)` sets — i.e. an expert is selected by *which shard you mounted*, not addressed by the thing reading rows.

Item 1 (extent = `(layer, expert)`) therefore has two routes:

| Route | Trait change | Cost |
|---|---|---|
| **(a) synthetic layer index** — fold expert in as `layer × num_experts + expert` | none | redefines what `layer` means for every existing caller. `num_features(layer)`, `gate_knn(layer, …)` and every `layer < num_hidden_layers` assumption in the walk/KNN paths silently start addressing an expert. At K3's 896 experts the synthetic index is ~6 orders wide. |
| **(b) expert-aware methods** — `Option<usize>` expert argument or expert-addressed siblings on `GateLookup` / `FfnRowAccess` | additive, default-impl'd | honest; costs new methods on two traits |

**Decided 2026-07-28: route (b).** Route (a) buys a clean G0 by hiding the change in an invariant rather than in a signature — precisely the failure the gate's falsifier column now names. `Fp4FfnAccess` and `StorageBucket::Fp4` already exist as a first-class storage bucket, so MXFP4 expert rows have a home either way; the question was only how they are addressed.

#### 4.1.1 Shape of the expert-aware surface (route b)

Expert-addressed **siblings** of the existing methods, each default-implemented in terms of the layer-only form. An `Option<usize>` parameter on the existing methods was rejected — that is a signature change, which G0 freezes.

| Trait | Added | Default |
|---|---|---|
| `GateLookup` | `num_experts(layer) -> usize` | `1` — the shape probe; non-MoE indexes answer honestly without knowing the concept |
| `GateLookup` | `gate_knn_expert(layer, expert, residual, top_k)` | delegate to `gate_knn(layer, …)` when `expert == 0`, else empty |
| `GateLookup` | `num_features_expert(layer, expert)`, `feature_meta_expert(layer, expert, feature)` | same delegation rule |
| `FfnRowAccess` | `ffn_row_dot_expert(layer, expert, component, feat, x)` | delegate to `ffn_row_dot(layer, component, feat, x)` when `expert == 0`, else `None` |
| `FfnRowAccess` | `ffn_row_scaled_add_expert`, `ffn_row_into_expert` | same delegation rule |

**The default rule is item 15's claim, stated in code.** "Dense = one-expert MoE, formally" stops being a design slogan and becomes the literal default implementation: an index that knows nothing about experts answers every expert-addressed call at `expert == 0` and declines the rest. That makes item 15 free at R1 rather than a separate unification exercise, and it means every existing non-MoE vindex keeps working without touching a line.

**Known cost, accepted:** the method count on two traits roughly doubles. The eventual cleanup — make the `_expert` forms canonical and reduce the layer-only ones to `expert = 0` wrappers — is itself a signature-affecting refactor, so it is explicitly *not* done now and is not a prerequisite for any rung.

### 4.2 GA spot-check — MXFP4 codec (item 16), R1/P1, 2026-07-28

**Result: PASS, bit-identical, on real weights.** Decoded expert 0 of `model.layers.0.mlp.experts.gate_up_proj` from the local `openai/gpt-oss-20b` snapshot — 16,588,800 values — under both larql's decoder and the `transformers/integrations/mxfp4.py` reference. Exact equality, no tolerance needed.

**First correction: the codec under test is `quant/mxfp4.rs`, not `quant/fp4_block.rs`.** The latter is larql's *own* FP4 vindex block format (exp 26 — 256-element blocks, 137 bytes, E4M3 sub-block scales). MXFP4 is a different format (32-element groups, one E8M0 exponent-only scale per group) and has its own module. Item 16's "verify-then-extend" applies to `mxfp4.rs`; `fp4_block.rs` is the storage side and is not what reads a GPT-OSS checkpoint.

**Two edge-case divergences from the reference exist, and neither is reachable in this checkpoint.** `e8m0_to_f32` (`larql-models/src/quant/mxfp4.rs:21`) special-cases two bytes that the reference does not:

| Scale byte | larql | transformers reference | OCP MX spec |
|---|---|---|---|
| `0x00` | `0.0` | `2^-127` (`ldexp`, exp = −127) | `2^-127` — E8M0 has no zero encoding |
| `0xFF` | `NaN` | `2^+128` → `inf` | `NaN` — reserved |

One deviation each way: larql is **spec-correct on `0xFF`** and the reference is not; larql deviates on `0x00`. A scan of **all 48 scale tensors — 597,196,800 bytes** found **zero** occurrences of either value; the observed range is 115–136 (2^−12 … 2^9). So the divergence is latent, not live, and the agreement above is not hiding it.

**Re-run this scan at R3.** It is a one-line check and K3 is a different checkpoint with a 896-expert bank; "unreachable in gpt-oss-20b" is not "unreachable".

**And pre-register the decision rule now, because the obvious one is wrong.** If the R3 scan returns a nonzero count for either byte, the resolution is **match the serving stack, document the spec deviation** — *not* "be spec-correct". K3 was not trained or served against the OCP MX document; it was trained and served against some stack's kernels, and its published behaviour is whatever those kernels do. On `0xFF` in particular, larql is currently spec-correct and the reference is not — which at R1 is a pleasing catch and at R3 would be a **divergence from the model's actual behaviour**, arbitrated against us by GB's API-agreement test. The rule, stated so it cannot be relitigated under time pressure:

> Where the format specification and the deployed serving kernels disagree, the model's behaviour is defined by the kernels. Match them, and record the deviation from the specification in the extraction manifest.

This is the same posture as the rest of the programme's verification: the reference is whatever the model actually is, not whatever it ought to be.

**Decode sanity, checked rather than assumed.** The decoded tensor's exact ±0.5 bound looked suspiciously round for a format whose top code is 6.0, so the nibble distribution was checked directly: all 16 codes present, near-exact sign symmetry (e.g. code 1 `+0.5` 1,429,207 vs code 9 `−0.5` 1,431,739), scale bytes 118–124. The nibble extraction is therefore correct, and the ±0.5 bound is a property of this tensor — the highest-scale groups happen not to use the top code — not a decode artefact.

Instrument: `scratchpad/mxfp4_verify.py` (throwaway; the durable version belongs in the repo's test suite once item 1 lands).

### 4.3 Extent layout — follows source fusion, does not prescribe it

The R1 tensor shapes settle a question the extent spec (item 1) would otherwise have had to decide by argument:

```
model.layers.0.mlp.experts.gate_up_proj_blocks   (32, 5760, 90, 16)   5760 = 2 × 2880
model.layers.0.mlp.experts.gate_up_proj_scales   (32, 5760, 90)
model.layers.0.mlp.experts.down_proj_{blocks,scales}
```

**GPT-OSS ships gate and up fused**, with the expert index as the leading axis. A byte-preserving re-layout inherits that fusion, so at R1 there is no gate-vs-up ordering decision to make — the checkpoint already made it.

**The extent spec is therefore written as "follows source fusion", not as a prescribed component order.** `component ∈ {gate, up, down}` stays the *addressing* vocabulary (§4.1.1's `ffn_row_dot_expert` signature is unchanged); what varies per architecture is how many physical tensors those three components are carved from, and that is a property read out of the source, not chosen by the extractor.

**The open version of the question moves to R3**, and becomes a P1 one-liner rather than a design decision: does K3's compressed-tensors export fuse gate and up the same way, or ship the three matrices separately? Read the index, record the answer in the manifest, move on. Same check at R2 for Kimi Linear.

**Confirmed by the extract run (§4.4):** the extractor already reads the fusion correctly — the packed-MXFP4 path takes the gate as the first half of the fused rows (`gate_vectors.rs`, `&expert_data[..half * in_features]`). Neither predicted breakage occurred.

### 4.4 First extraction run — R1/P1, 2026-07-28

`larql extract <snapshot> --level browse --summary-features-per-expert 64`, release binary `0.1.0`. **Completed in 3.3 min, exit 0, 12.98 GB out.** Both predicted failure modes were absent: the fused `gate_up` tensor did not collide with any three-named-matrices assumption, and the leading `(32, …)` expert axis unbound without incident.

**Finding 1 — the vindex manifest already carries expert granularity.** This is the empirically important one, and it changes item 1's cost:

```json
{"layer": 0, "num_features": 92160, "num_experts": 32, "num_features_per_expert": 2880, …}
```

`VindexLayerInfo` already has `num_experts: Option<usize>` and `num_features_per_expert: Option<usize>`, and experts are **already flattened into the feature axis** as `feature = expert × num_features_per_expert + f`. So §4.1.1's expert-addressed methods are not a new storage layout — they are an *addressing* layer that can be default-implemented over what the format already stores and records:

```
ffn_row_dot_expert(layer, expert, component, feat, x)
    = ffn_row_dot(layer, component, expert * num_features_per_expert + feat, x)
num_experts(layer) = manifest num_experts, or 1 when absent
```

The `Option` is load-bearing and matches §4.1.1's default exactly: the PackedBF16 path (Gemma 4 26B A4B) writes `num_experts: None`, because it uses the dense FFN gate for KNN routing. So "absent ⇒ one expert" is already the format's own convention, not an invention.

**Item 1 is therefore substantially cheaper than scoped** — manifest plumbing plus addressing, not a new profile. That should be re-costed before R1's extract phase is planned.

**Finding 2 — `--summary-features-per-expert` is silently a no-op on the MXFP4 path, and that is exactly the path K3 uses.** The flag produced a full 11.87 GiB per-expert gate (2880×2880×32×24×2 B, to the byte) instead of the ~0.26 GiB a top-64 SVD summary would give. Cause: `gate_vectors.rs` branches on expert format, and the summary path exists **only** in the `else if self.is_moe && self.n_experts > 0` arm (Mixtral-style, one gate tensor per expert). The `PackedMxfp4` arm dequantises every expert and writes the full gate with no reference to `summary_k`. No warning, exit 0.

The flag's own docstring names its purpose as many-expert MoE ("DeepSeek-V4-Pro at 384 experts/layer would otherwise need ~370 GB"). **K3 is MXFP4-packed with 896 experts** — the configuration where the summary path is mandatory rather than optional, served by the branch where it does not exist.

**The K3 number, because "not tractable" undersells it.** At working K3 dims (3,072 features × 3,584 hidden × 896 experts × 92 layers, f16):

```
3072 × 3584 × 2 B × 896 × 92  =  1.82 TB
```

**That is larger than the entire source checkpoint** (~1.4 TB native), for the gate alone — one third of one component. The failure mode is not "produces a large artifact"; it is that the no-op **inflates 4-bit weights to 16-bit at 896-expert multiplicity**, so the derived index exceeds the model it was derived from. This is the difference between the K3 extraction existing and not existing.

**Action:** wire the summary path into the `PackedMxfp4` arm, at R1, before the coverage curve (`k3r1-coverage-curve`) needs it. Until then the flag should warn rather than silently no-op — silent no-ops on a size-control flag are how a 12-hour K3 build dies at hour 11.

**And note what the fix collapses into: this action and item 3 are plausibly the same work.** `--summary-features-per-expert` doing a top-K SVD per expert *is* a per-expert gate sketch — the compressed, resident-mountable selection tier item 3 specifies. Wiring the summary path into the `PackedMxfp4` arm therefore doesn't merely unblock the coverage curve; it builds item 3's substrate. The open question narrows from "design a sketch" to **"is top-K SVD the right sketch, or does it want PQ residuals on top?"** — which is a measurable question against the R1 harvest rather than a design-from-scratch. Two items become one item plus an evaluation; see the item 3 row in §6.

**Finding 3 — provenance fields are unpopulated, and the timestamp is wrong by construction. Root-caused and closed.** `source.huggingface_revision: None` and `source.safetensors_sha256: None` confirm item 4's gap on evidence.

The timestamp anomaly — manifest `2026-08-13T23:18:12Z` against a machine clock of `2026-07-28T23:19:35Z` — is **`chrono_now()` in `larql-vindex/src/extract/build_helpers/timestamp.rs`**, a hand-rolled epoch→ISO formatter that computes `years = 1970 + days/365`, `month = (days % 365)/30 + 1`, `day = (days % 365) % 30 + 1`, then clamps month to 12 and day to 31. It ignores leap days entirely and assumes 30-day months. Feeding the run's true instant through it reproduces `2026-08-13T23:18:12Z` **exactly, character for character**.

The signature was diagnostic: the time-of-day was correct to the second (`secs % 86400` arithmetic is right) while only the date was wrong, which rules out clock contamination — a monotonic/`Instant` mix-up was tested first and falsified in one command (uptime 2.28 days against a 16.00-day offset).

Two details worth carrying:

- **The repo already contains two correct implementations.** `walker/utils.rs:67` has proper `is_leap_year` + month tables, with tests covering the 2000 and 1900 century boundaries; `extract/checkpoint.rs:213` has Howard Hinnant's civil-from-days. The manifest path uses a third, wrong one. The fix is a call, not an algorithm.
- **The unit tests pass on wrong output by construction.** All three assert *plausibility* — string length, separator positions, year ≥ 2020, month ∈ 1..12, day ∈ 1..31 — and the clamps guarantee the last two. Nothing asserts a known instant maps to a known string. That is why this survived: the tests encode the format, not the semantics.

**Until it is fixed, the extractor should write `None` rather than a wrong value.** For a field item 4 is about to make load-bearing, wrong-but-plausible is strictly worse than absent — absent fails loudly at the verifier, wrong passes it.

**Also captured, useful downstream:** `moe.top_k: 4`, `moe.router_type: top_k_softmax`, `moe.shared_expert: false`, `moe.hybrid: false`, `sliding_window: 128`, `rope_base: 150000`, 64 Q heads / 8 KV heads.

**Pre-registered for the next run (`--level attention`):** the prediction is that `self_attn.sinks` **is** preserved — they are ordinary named tensors and the tier includes attention. The informative outcome is the other one: if sinks are dropped, the cause is most likely browse-tier assumptions about what "attention" comprises leaking into the attention tier, i.e. a tier-definition bug rather than a GPT-OSS-specific gap — which would be a finding about the extractor's tier model, not about this model family.

### 4.5 Fixes landed — R1/P1, 2026-07-28/29

Both defects from §4.4 are fixed, verified end-to-end, and under test.

**Finding 2 — summary path on `PackedMxfp4`.** `gate_vectors.bin` for gpt-oss-20b at `--summary-features-per-expert 64`:

| | bytes | |
|---|---|---|
| before (flag ignored) | 12,740,198,400 | 11.87 GiB |
| after | **283,115,520** | 0.264 GiB — **43×**, and exact to the byte against the predicted `64 × 2880 × 32 × 24 × 2` |

Manifest now reads `num_features_per_expert: 64`, `num_experts: 32`, consistent across all 24 layers.

**Finding 3 — timestamp.** `chrono_now()` rewritten on Hinnant's civil-from-days; `extract/checkpoint.rs`'s duplicate implementation deleted and pointed at it, so the repo went from three implementations (two correct, one not) to one. Manifest now records `2026-07-29T00:01:45Z` against a clock of `2026-07-29T00:02:49Z`. The regression test pins the exact instant that exposed the bug; the previous tests asserted only plausibility and passed on wrong output by construction.

**Structural change made necessary by the fix.** `gate_vectors.rs` was one 303-line function with four inline branches, and fixing Finding 2 would have duplicated the summarisation block into a second arm. Split into a directory: `mod.rs` (driver + dispatch), `summary.rs` (the full-vs-top-K-SVD policy, owned once), `packed_mxfp4.rs`, `standard_moe.rs`, `dense_gate.rs` — each under 180 lines. Two of the four arms turned out to be byte-identical and collapsed into `dense_gate`. The MXFP4 shape arithmetic became a `FusedGateLayout` type that validates tensor rank instead of indexing `shape[2]` blindly.

**Verification.**

| Check | Result |
|---|---|
| `larql-vindex` lib tests | 1137 pass (was 1106 — 31 added) |
| Per-file line coverage, files touched | 91.18 – 100 %, all above the 90 % floor |
| `check_coverage_policy.py` | **passed** — total 91.65 %, 156 files, 113 at the 90 % default, 43 debt baselines |
| `cargo fmt --check`, `cargo clippy --all-targets` | clean |
| Extract re-run after the refactor | `gate_vectors.bin` **SHA256-identical** to the pre-refactor run — behaviour-preserving |

**Measurement note worth keeping:** the first coverage run reported `timestamp.rs` at 85.09 % and a phantom `gate_vectors.rs` at 0/201 lines for a file that no longer existed. That was stale profile data mixed across builds — the "uncovered" line numbers pointed at source positions that no longer exist. `cargo llvm-cov clean --workspace` before measuring; a coverage number that disagrees with the source it claims to describe is measuring the wrong build.

### 4.6 Attention sinks — prediction falsified by a third mechanism

§4.4 pre-registered that `self_attn.sinks` would survive `--level attention`, with the informative alternative being a tier-definition bug. **Both were wrong, and the actual answer is worse than either.**

**`sinks` appears exactly once in the entire engine — in a doc comment.** `crates/larql-models/src/architectures/gpt_oss.rs:8` reads *"Attention has biases, sinks, and uses GQA"*, and that is the whole of it. There is no `attn_sinks_key()` on `ModelArchitecture`, no extraction path, no storage, and no use in CPU or Metal attention. The tensor class is not dropped by a tier rule; **it was never known to the extractor at all.** The architecture module documents the feature and then does not implement it.

**What the tensor does, from the reference** (`transformers/models/gpt_oss/modeling_gpt_oss.py:261-269`):

```python
combined_logits = torch.cat([attn_weights, sinks], dim=-1)   # per-head learned logit
probs  = F.softmax(combined_logits, dim=-1)
scores = probs[..., :-1]                                     # sink dropped after softmax
```

The sink is a learned per-head logit that competes in the softmax and is then discarded — so attention weights over real keys **deliberately sum to less than one**. It is a learned "attend to nothing" mass sink. Omitting it renormalises over real keys only, scaling every weight by `1/(1 − p_sink)`.

**The diverted mass is not negligible.** Reading all 24 sink tensors (64 heads each, 1,536 values, BF16):

| | |
|---|---|
| mean sink logit | **+2.45** |
| max | **+8.19** |
| min | −2.45 |
| per-layer means | +1.08 … +4.09, every layer positive |

A sink logit of +8 dominates the softmax unless real attention logits exceed it. These are large, positive, and present in every layer — this is a systematic error, not a rounding drift.

**Consequence: larql's GPT-OSS forward pass is not faithful today**, on a path the docs call supported — `README.md:609` and `AGENTS.md:125` both direct users to `INFER` for GPT-OSS. That claim needs either a fix or a caveat.

**What is *not* yet established:** the end-to-end magnitude. The mass a sink actually takes depends on the runtime attention logits it competes against, which this measurement does not capture. Sizing it is precisely a **GB** job — `larql shannon verify` against the reference — and GB now has a specific, named thing to catch instead of a general hope of noticing.

**Two work items follow, and the second matters more than the first.**

1. **Implement sinks** — `attn_sinks_key()` on `ModelArchitecture`, extraction into the attention slice, and the concat-softmax-truncate in both the CPU and Metal attention kernels. Scoped to R1, gated by GB.
2. **Add an extraction tensor-coverage audit.** The general defect is that *a tensor the checkpoint has and the extractor does not recognise is silently discarded, and nothing anywhere reports it.* Every source tensor should be classifiable as extracted, deliberately dropped by a named rule (multimodal, MTP-preserved, …), or **unrecognised — which should be loud**. This is the extraction-side sibling of `larql capabilities` in [`vindex-factory.md` §15.2](vindex-factory.md), which checks architectures at PR time; this checks *tensors* at extraction time.

   **Built 2026-07-31.** `extract::coverage` plus a `tensor_audit` stage that runs first in `build_vindex_streaming`, so an unaddressable checkpoint fails in seconds rather than after a multi-minute extraction. Reports always; fatal under `LARQL_EXTRACT_STRICT=1`, which is set in the `larql-vindex` CI workflow. Clean on ten checkpoints including Qwen3-30B-A3B at 18,867 tensors and Gemma 3's 439 SigLIP tensors (classified `non-text-tower`). It measures *naming* rather than consumption — the necessary condition, and where every silent drop found so far actually lived. Its first outputs: GPT-2 from HF safetensors is 1-of-160 addressable, and LayerNorm `β` was being dropped for GPT-2 and StarCoder2 (§4.7.9). Follow-ups in [`ROADMAP.md`](../ROADMAP.md) §"Extraction tensor-coverage audit".

**This is the ladder working exactly as designed.** A silently-dropped tensor class, on a novel architecture, discovered on a 13 GB model that runs locally — rather than at 2.8 T where the only oracle is an API and the symptom would have been "the outputs are subtly wrong and we don't know why". K3 is guaranteed to carry tensors larql has never seen (QB bias, AttnRes block state, LatentMoE projections); item 2 above is what turns that guarantee from a silent hazard into a startup error.

#### 4.6.1 It is not just sinks — 5 of 11 attention tensors are dropped

The `--level attention` run finished and made the scope precise. Per layer, the checkpoint holds **11** attention-related tensors; `weight_manifest.json` (145 entries) shows **6** written:

| Per-layer tensor | In checkpoint | Extracted |
|---|---|---|
| `input_layernorm.weight`, `post_attention_layernorm.weight` | ✅ | ✅ |
| `self_attn.{q,k,v,o}_proj.weight` | ✅ | ✅ |
| `self_attn.{q,k,v,o}_proj.bias` | ✅ | ❌ |
| `self_attn.sinks` | ✅ | ❌ |

**120 tensors silently dropped across 24 layers.** `attn_weights.bin` is 1,274,019,840 bytes — matching Q/K/V/O weights alone to the byte, with no room for the 8,000 bias floats per layer.

**Two independent causes, stacked** — worth separating because they have different fixes and different blast radii:

1. **`gpt_oss.rs` declares neither.** The word "bias" appears in that file exactly once, in the same doc comment as "sinks" (`architectures/gpt_oss.rs:8`). The `ModelArchitecture` trait *has* `attn_{q,k,v,o}_bias_key` accessors defaulting to `None` (`config.rs:244-258`), and `starcoder2.rs`, `gpt2.rs` and `qwen.rs` all override them — so the pattern exists and GPT-OSS simply doesn't follow it.
2. **Extraction never asks for attention biases, for any architecture.** No `*_bias_key` call appears anywhere in `larql-vindex` except the router bias (`router_weights.rs:38`).

**The compute side already expects them.** `attn_q_bias_key` is consumed in `larql-compute/src/attention/block.rs:355`, `attention/gpu.rs:55,226`, `attention/decode/dispatch.rs:51`, `decode/q4k_direct.rs:163` and `kquant_forward/cached/`. So the consumer is built and keyed off the architecture accessor; for GPT-OSS it asks, gets `None`, and applies nothing.

**That splits the fix cleanly.** The bias half is four accessor overrides in `gpt_oss.rs` plus writing them at extraction — the qwen.rs pattern, no new math. The sinks half needs the concat-softmax-truncate in both backends. Only the second is genuinely new work.

**Open question, deliberately not claimed:** whether a *vindex-backed* Qwen2 forward also loses its declared biases, given that extraction writes none. The compute path would ask for a key the vindex never stored. This is one experiment away and is not in this ladder's scope — but if it holds, cause 2 above is an engine-wide gap rather than a GPT-OSS one, and should be raised as its own issue.

#### 4.6.2 Extraction fixed — all 11 tensors now survive (2026-07-29)

Three changes, addressing both causes:

1. **`attn_sinks_key()` added to `ModelArchitecture`** (`config.rs`), defaulting to `None`. Its doc states the obligation explicitly — returning the key without the kernel applying it changes only the extracted bytes, not the forward pass.
2. **`gpt_oss.rs` now declares** all four projection biases plus sinks. The module header, which had claimed "attention has biases, sinks" while every accessor returned `None`, now points at the tests that assert it.
3. **Both extraction paths request them.** The f32 path (`write_f32.rs`) and the k-quant path (`write_kquant/norms.rs`) each had a 1-D vector loop that asked only for QK norms; both now also ask for the four biases and the sinks. This fixes cause 2 for *every* architecture — `qwen`, `gpt2` and `starcoder2` declared biases that extraction had never requested.

**Verified by re-extraction:**

| | before | after |
|---|---|---|
| `weight_manifest.json` entries | 145 | **265** (+120, exactly the dropped count) |
| attention tensors per layer | 6 of 11 | **11 of 11** |

`sinks` lands as `{"kind": "vector", "shape": [64], "length": 128}` — one f16 per head — and `q_proj.bias` as `shape [4096]`. Tests: `larql-models` 446 pass (6 added to a file that previously had none), `larql-vindex` 1137 pass; fmt and clippy clean.

**Still outstanding — the compute half.** Biases already have a consumer (`attention/block.rs:355`, `gpu.rs:55,226`, `decode/dispatch.rs:51`, `q4k_direct.rs:163`, `kquant_forward/cached/` all resolve `attn_q_bias_key`), so with the key declared and the tensor stored, that path closes. **Sinks have no consumer at all** — the concat-softmax-truncate does not exist in either the CPU or Metal kernel. The tensor now survives extraction, which is a precondition, not the fix. Until the kernel applies it, GPT-OSS attention remains numerically wrong, and GB is the gate that will say by how much.

#### 4.6.3 CPU softmax now supports sinks — primitive landed, data not yet flowing

**What is done.** `attention/softmax.rs` is new and owns the softmax:

```rust
pub fn softmax_in_place(scores: &mut [f32], sink: Option<f32>)
```

`None` gives an ordinary softmax summing to 1. `Some(s)` folds `exp(s − max)` into the denominator without giving the sink an output slot — algebraically identical to the reference's concat-softmax-truncate, without materialising the extra column. The sink joins the max, so a sink far above every real logit stays finite instead of overflowing.

**Thirteen tests**, the load-bearing one being `matches_concat_softmax_truncate`, which checks against a literal transcription of `modeling_gpt_oss.py`'s formulation at the sink values actually measured on the checkpoint (−2.45, 0, +2.45, +8.19). Others pin: weights sum below 1 with a sink and to 1 without; a +8.19 sink diverting >99 % of mass; a −40 sink being indistinguishable from none; ratios between real logits preserved (the sink rescales uniformly); and overflow safety at ±1000 logits.

**Why a new file.** `gqa.rs` held **three byte-identical copies** of the twelve-line softmax. Adding sinks by editing each is how a kernel ends up correct in two of three — the same duplication that produced the summary-path bug in §4.4. All three sites now call the shared function, and `sinks: Option<&[f32]>` is threaded through `gqa_attention_capture`, `gqa_attention_with_weights`, `gqa_attention_with_all_weights`, `gqa_reduced_qk_all_weights` and `gqa_attention_asym`, with the per-head lookup (`sinks.map(|s| s[h])`) inside each head loop.

**What is explicitly NOT done: no caller passes a real sink yet.** Every one of the ~30 call sites across `larql-compute`, `larql-inference`, `larql-vindex` and `larql-cli` passes `None`. The plumbing exists and the maths is verified; the tensor is not yet read from `ModelWeights` and handed to the kernel. **GPT-OSS attention is still numerically wrong on CPU** — this step makes the fix a data-wiring change instead of a kernel change.

Also untouched: the **Metal** path. The MSL kernels have their own softmax and need the same treatment before the GPU path is correct.

Workspace state: builds clean across all targets, `cargo fmt` clean, and every crate's tests pass (789 in `larql-compute`, 1309 in `larql-vindex`, 446 in `larql-models`, plus the rest).

**Remaining, in order:** (1) read the sink vector from `ModelWeights` via `attn_sinks_key` and thread it to the CPU kernel; (2) the same in the Metal kernels; (3) run GB (`shannon verify`) against the reference to measure what the whole gap was actually worth.

#### 4.6.4 CPU path complete — sinks now applied end to end

Step (1) is done. **Every CPU attention path now resolves and applies the sink.**

`attention/sinks.rs` (new) owns the lookup, because four kernels needed it and four copies of a lookup-plus-length-check is the duplication that produced two of this rung's bugs:

```rust
pub fn resolve(sinks_key: Option<String>, vectors: &HashMap<String, Vec<f32>>,
               num_q: usize, layer: usize) -> Option<&[f32]>
```

Its contract is deliberately asymmetric. A **missing key** (architecture has no sinks) or a **missing tensor** (vindex extracted before sinks were written) both degrade quietly to an ordinary softmax — old artifacts must keep loading. A **length mismatch panics**, naming the layer and both counts: a sink vector that doesn't match the head count means the tensor doesn't describe this model, and truncating or zero-padding would give a plausible-looking wrong forward pass, which is the precise failure this whole area was fixed to remove.

Wired at all four call sites — prefill (`block.rs`), and the three decode paths (`decode/dispatch.rs`, `decode/q4k_direct.rs`, `kquant_forward/cached/`).

**A fourth softmax copy turned up while doing it.** `decode/gqa_step.rs` — the single-token path that actually runs during generation — had its own twelfth-line duplicate, separate from `gqa.rs`'s three. It now calls the shared `softmax_in_place` too, so prefill and decode cannot drift.

**Tests: 21 across the three files.** Beyond the 13 pinning the maths, five cover the resolver (absent key, absent tensor, present tensor, and two panic cases), and three prove the *threading* rather than the maths:

- `sinks_change_the_attention_output` — a supplied sink actually reaches the softmax;
- `sink_weights_sum_below_one_per_position` — captured distributions sum below 1 at every query position;
- `each_head_uses_its_own_sink` — head 0 with a −40 sink is untouched while head 1 with +40 collapses, which fails if the slice is indexed wrongly or the head ignored.

**Two of those three failed when first written, and the failures were mine, not the kernel's.** The shared fixture builds values as `(i+1)·scale`, so at `scale = 0.3` the dot products reach ~45 — dwarfing a 2.45 sink, whose share then underflows and leaves the real weights summing to exactly 1.0. Physically correct; tests nothing. At the 0.01 scale the existing `run()` helper uses, both pass. Worth recording because the first failure looked exactly like "the sink isn't wired".

**Still outstanding:** the **Metal** kernels are untouched and carry their own softmax, so the GPU path remains uncorrected; and GB has not run, so the size of the original error is still unmeasured.

#### 4.6.5 CPU softmax proven correct; Metal prefill corrected

**The CPU softmax needed no correction — it is verified right.** A differential test now pins the *full attention output* (not just the softmax) against values computed independently in NumPy following `eager_attention_forward`: concatenate the per-head sink, subtract the joint max, softmax, drop the sink column, weight V. Agreement to **< 1e-7** on all 16 values, with the reference numbers hardcoded in `full_attention_with_sink_matches_reference_implementation`.

**Metal prefill is now corrected.** `fused_attention` gained two bindings — `constant float* sinks [[buffer(14)]]` and `constant uint& has_sinks [[buffer(15)]]` — with the sink joining the max reduction (or `exp(sink − max)` overflows when the sink dominates) and contributing to the denominator only. Per-layer sinks flow `build_arch_params` → `FullPipelineLayer::attn_sinks` → `stages::attention::encode` → shader, mirroring how `input_norm_bias` already travels.

**Verified on GPU against a CPU reference including the sink: max diff < 1e-4.** The two pre-existing shader tests still pass, so the `has_sinks = 0` path is unchanged.

Three implementation notes worth keeping:

- **Address space matters.** `sinks` was first declared `device const float*`, which SIGSEGV'd: `set_bytes` binds into `constant` space. Small per-head data belongs in `constant` anyway.
- **Metal has no null buffer**, so the slot always carries a real allocation and `has_sinks` gates the read. The existing tests bound only up to buffer 13, which would have left `has_sinks` reading garbage — they now bind both slots explicitly.
- **One crash was purely my test code**: `|v: &u32| *v as *const u32` casts the *value* to a pointer instead of taking its address, so `set_bytes` read from address `0x3`. Worth recording because the symptom (SIGSEGV in a GPU test right after a shader change) points hard at the shader, and the shader was fine.

#### 4.6.6 Metal decode corrected — all four softmax sites now sink-aware

The two decode shaders are done, using the template prefill proved.

| Shader | Path | Sink bindings |
|---|---|---|
| `fused_attention` | prefill | 14 / 15 |
| `attn_fused` | decode (fused QK-norm+RoPE+attend) | 18 / 19 |
| `kv_append_attend_fused` | decode (default kernel) | 12 / 13 |

Both take the sink into the max reduction and into the denominator only, and both read `layer.attn_sinks` — so the same per-layer vector now drives CPU prefill, CPU decode, Metal prefill and Metal decode from one source.

**The placeholder-and-flag convention is owned once**, in `stages::attention::sink_binding`, shared by all three Metal dispatch sites. It carries the length assertion, so a sink vector that doesn't match the head count fails loudly rather than being silently truncated — matching the CPU resolver's contract.

**Every dispatcher was audited, not assumed.** An unbound `has_sinks` reads garbage and would intermittently enable a sink read against a placeholder, so each site was checked: three test dispatchers in `test_kernel_fused_attention.rs`, one in `test_metal_shaders.rs`, the two production decode sites, and the prefill dispatch. `diag/shader_bench.rs` lists all three kernels only as `status: "inventory"` — catalogued, never dispatched — so it needs nothing.

**Full Metal suite green on GPU** (193 + every kernel test), workspace builds clean across all targets, clippy and fmt clean, and the workspace lib tests pass.

#### 4.6.7 Decode kernels now have numerical parity tests

The caveat in §4.6.6 — "same edit, same shape, but no measurement" — is closed. `tests/test_kernel_decode_attention_sinks.rs`, 8 tests, all passing on GPU.

**`kv_append_attend_fused` gets full value-level parity.** Its contract is simple enough to mirror exactly: append the new K/V row at `pos = T-1`, then attend over the cache. The CPU reference accumulates in f64 and takes the joint max over logits *and* sink, matching the reference formulation. Agreement **< 1e-4**, plus a no-sink regression check, a mass-diversion check, and a per-head indexing check.

**`attn_fused` is tested by analytic invariant instead, deliberately.** It fuses RMS-norm and RoPE, so a value-level reference would have to re-derive both — duplicating logic the norm and RoPE kernels already test, and then testing the duplicate as much as the kernel. The sink has an exact signature that sidesteps this: it changes only the denominator, so

```
w_i(sink) = e_i / (S + e_s) = w_i(no sink) · S/(S + e_s)
```

— every weight in a head is multiplied by the *same* constant, so the whole per-head output vector scales by one factor λ ∈ (0, 1). **A uniform per-head rescaling is a fingerprint no plausible bug reproduces**: applying the sink to the logits, per-element, or to the wrong head each break it. The tests assert λ is constant across the head's dimensions (not merely that the output shrank), that λ ∈ (0,1), that a larger sink gives a smaller λ, that head 0 with −40 keeps λ ≈ 1 while head 1 with +40 collapses, and that a −1e30 sink is indistinguishable from no sink.

That last one is the guard on the new bindings themselves: it proves the added buffers do not perturb the default path.

**Coverage of the sink work is now:** CPU softmax pinned to NumPy at < 1e-7; CPU threading pinned by three head-indexing tests; Metal prefill pinned to a CPU reference at < 1e-4; Metal decode — one kernel at < 1e-4, the other by exact invariant. Metal suite: 30 test binaries green.

#### 4.6.8 GB attempted and **blocked** — the gate's instrument does not cover this model class

Running GB was attempted and failed, for a reason worth more than the number would have been.

```
$ larql shannon score <gpt-oss-20b> --corpus … --bytes 384
loaded. 24 layers, hidden_size=2880 (44.0s)
scoring 86 target tokens over 384 bytes...
panicked at larql-compute/src/ffn/weight.rs:333:
  FFN weight tensor missing … (key: layers.0.mlp.up_proj.weight)
```

**`larql shannon score` hardcodes `WeightFfn`** (`shannon_cmd/` and `:1447`), which resolves the *dense* `ffn_up_key`/`ffn_down_key`/`ffn_gate_key`. GPT-OSS has no dense FFN — its experts are `mlp.experts.gate_up_proj_blocks`. The panic's suggested remedy ("this is a `--compact` vindex") is a misdiagnosis: the tensors are not missing, they never existed for this architecture.

**This is broader than GPT-OSS.** `WeightFfn` is dense-only, so the Shannon scorer cannot score *any* mixture-of-experts model. GB — the gate this ladder leans on hardest, and the one that degrades to an API oracle at R3 — currently has no instrument for the model class R1, R2 and R3 are all built on. R1 is MoE, Kimi Linear is MoE, K3 is MoE.

**And there is no local packed-MXFP4 forward at all.** `PackedMxfp4` and the MXFP4 dequantiser appear only in `larql-models` (codec, loader, architecture) and `larql-vindex` (extraction). Nothing in `larql-compute`, `larql-inference` or `larql-server` consumes them. GPT-OSS is served by *converting* MXFP4 → Q4K at extraction, so the serving path never sees MXFP4 — which is why item 16's verify mattered and why a raw-safetensors scorer cannot work here.

**What GB needs, in order:**

1. Route the Shannon scorer's FFN by architecture instead of hardcoding `WeightFfn` — MoE models need a MoE-capable backend. This is the blocking item and it is not large.
2. Failing that, score from a **Q4K vindex** rather than raw safetensors, so the MoE path that already works in serving is the one measured. This changes what GB measures (post-quantisation, not raw parity), so it is a fallback, not an equivalent.

**Consequence for the sink work:** its correctness rests on the parity tests in §4.6.5–4.6.7 — NumPy at < 1e-7 on CPU, CPU-reference at < 1e-4 on two Metal kernels, exact invariant on the third. Those are kernel-level and they are real. What remains unmeasured is the **end-to-end** effect on bits/char, which is exactly what GB exists to provide.

**The honest status line: the original error's magnitude is still unknown, and now we know why it will stay unknown until the scorer learns MoE.** That is a better outcome than a number obtained by bending the instrument, but it is not the number.

**Unrelated pre-existing flake, flagged not fixed:** `cpu::spin_pool::tests::stress_concurrent_realistic_decode_shape_no_corruption` fails roughly one run in three under `cargo test --workspace --lib`, and did so before these changes (`git diff` confirms `spin_pool.rs` is untouched). It does **not** reproduce standalone (0/15) or under synthetic CPU load (0/6), so the trigger is concurrent test *binaries* rather than CPU pressure alone. It asserts buffer *corruption*, not a timeout, in the dispatch lock of the spin-barrier pool — which is default-on and load-bearing for decode throughput. That combination deserves its own investigation rather than a shrug.

**GB is the gate the ladder exists to strengthen.** At R1 and R2 it is a local diff. At R3 it degrades to an API oracle — which is exactly why R3's remaining surface is deliberately the *smallest and simplest* three components (§7).

### 4.7 Unblocking GB found six defects in the forward pass it was going to measure (2026-07-30)

§4.6.8 recorded GB as blocked on a missing instrument. Building the instrument turned out to be the smaller half of the job: **larql's GPT-OSS MoE forward pass diverged from the reference in six independent ways**, every one of which GB would have been reporting as a bits/char number without saying why.

This is the sinks finding (§4.6) repeated one subsystem over, and it lands the same way: found on a 13 GB model that runs locally, against a reference implementation that runs on the same machine. That is the entire argument for the ladder, twice in three days.

#### 4.7.1 The findings

Verified against `transformers` 5.5.0 (`models/gpt_oss/modeling_gpt_oss.py`, `integrations/mxfp4.py`), and for the first one against the real `openai/gpt-oss-20b` snapshot.

**Finding 1 — the fused gate/up split was inverted. This is the load-bearing one.**

The reference chain is unambiguous:

```text
gate_up_proj_blocks (E, 2I, G, 16)
  -> dequant                (E, 2I, H)     row = on-disk output row
  -> .transpose(1, 2)       (E, H, 2I)     on-disk row becomes the LAST axis
  -> _apply_gate:  gate = x[..., 0::2],  up = x[..., 1::2]
```

so **gate is the even on-disk rows and up the odd ones — interleaved.** larql took the leading half as gate (`quant/mxfp4.rs`, and the extractor's `gate_vectors/packed_mxfp4.rs`), which makes each "half" a 50/50 mixture of the two projections.

Measured on layer 0, expert 0 of the real checkpoint:

| split | gate std | gate absmax | up std | up absmax |
|---|---|---|---|---|
| reference (even/odd) | 0.0287 | **0.250** | 0.0449 | 0.500 |
| larql (contiguous halves) | 0.0376 | 0.500 | 0.0377 | 0.500 |

**90.29 % of elements differ**, and the statistical signature is diagnostic rather than suggestive: the correct split separates two genuinely different distributions, while the wrong one yields two halves with matching statistics — because both are the same mixture. The true gate never exceeds 0.25 in magnitude; larql's "gate" reached 0.50 because it contained up rows.

The **bias is an independent witness**, and a cleaner one. Reference gate/up bias means separate at **−0.464 / −0.898**; larql's two halves both sat at the pooled mean, **−0.679 / −0.684**, with the standard deviation inflated from ~0.13 to 0.26 — textbook two-population variance inflation.

`gpt_oss.rs`'s module header stated the wrong belief in prose — *"gate_up_proj_blocks (first half = gate)"* — three lines above the sinks claim that §4.6 falsified. Two false statements in one comment block, both load-bearing, neither tested.

**Findings 2–4 — the expert MLP is not SwiGLU.** From `GptOssExperts._apply_gate`:

```python
g   = gate.clamp(min=None, max=limit)   # one-sided
u   = up.clamp(-limit, limit)           # symmetric
glu = g * sigmoid(g * alpha)            # alpha = 1.702
out = (u + 1) * glu                     # note the +1
```

larql computed `silu(g) * u`, missing the α, the `(up + 1)`, and both clamps. `config.json` ships `swiglu_limit: 7.0` and nothing read it. The `(up + 1)` term matters most: at `up = 0` the reference passes the GLU straight through where SwiGLU annihilates it.

**Finding 5 — the router's normalise/select order was inverted.** larql softmaxed over all 32 experts then took the top 4 (`cpu/ops/moe/mod.rs:123-124`, `MoeTopKWeightPolicy::RawSoftmax`); the reference takes the top 4 then softmaxes over **those**. The orders are not equivalent — select-then-normalise sums to exactly 1, normalise-then-select sums to whatever mass the top 4 of 32 happened to hold, so the whole expert branch was systematically attenuated. Worth recording that the fix is algebraically exact rather than approximate: `softmax(logits)_i / Σ_{j∈topk} softmax(logits)_j = exp(l_i) / Σ_{j∈topk} exp(l_j)`, so *renormalising* the selected weights **is** select-then-normalise.

**Finding 6 — three of eight per-layer MLP tensors were silently dropped.** `mlp.router.bias`, `experts.gate_up_proj_bias`, `experts.down_proj_bias`: not declared on the architecture, not extracted, no field on `MoeLayerWeights`, and `gate_up_proj_bias` / `down_proj_bias` appeared **nowhere in the workspace**. Identical mechanism to §4.6.1's 5-of-11 attention tensors.

**A seventh, found while fixing the others:** `moe_intermediate_size()` defaults to 0 and GPT-OSS never overrode it, so every expert matmul sized through that accessor would have had a zero inner dimension.

#### 4.7.2 Two things checked and cleared, so the picture isn't overstated

- **§4.2's MXFP4 codec PASS still stands.** It verified `dequantize_all_experts` — decoding blocks+scales to values — and that is genuinely bit-identical. The split happens *after* the codec. The GA spot-check was sound but scoped one function too narrow, which is the more useful lesson: a bit-exact codec test says nothing about what the caller does with the bytes.
- **`down_proj` needs no transpose.** The reference transposes it too, but on-disk `[hidden, inter]` already matches larql's matmul convention (`out[h] = Σ_i act[i]·W[h][i]`). Confirmed by derivation rather than assumed, because on the 20B `hidden == inter == 2880` makes a missed transpose **shape-invisible**. This isolates the layout defect to gate_up's row selection alone.

#### 4.7.3 Why the existing tests passed

`split_gate_up_experts` had two unit tests. Both used `out_features = 2`.

**At two rows the two conventions are indistinguishable** — row 0 is simultaneously "the first half" and "the even rows". The tests pinned the shape and the dequant scaling, and were structurally incapable of pinning the convention. Anything asserting it needs `out_features ≥ 4`.

That is the same failure mode as §4.5's timestamp tests, which asserted plausibility (year ≥ 2020, month ∈ 1..12) and passed on wrong output by construction. Two instances now, in unrelated subsystems: **a fixture too small to distinguish the candidate behaviours is not a weak test, it is an absent one.** The new tests are built on a four-row fixture and the parity test carries an explicit control that fails under the old split.

#### 4.7.4 What landed

**Loader and extractor.** The de-interleaving convention is now owned by one named function, `mxfp4::deinterleave_fused_half`, with `FusedHalf::{Gate, Up}` and the reference chain in its doc comment. Both the model loader and the vindex gate-sketch extractor call it.

**Architecture surface** (additive, default-impl'd — G0 holds): `moe_router_bias_key`, `packed_gate_up_bias_key`, `packed_down_bias_key`, and `expert_gate_policy() -> ExpertGatePolicy`, which is `Gated` for every existing architecture and `ClampedGlu { limit, alpha }` for GPT-OSS. `swiglu_limit` is now parsed from `config.json` rather than hardcoded, because a future checkpoint may pick a different bound. `GptOssArch` also overrides `moe_intermediate_size` and declares a distinct `moe_router_type` so the two routing orders can't alias.

The per-expert keys the MXFP4 loader *synthesises* at dequant time are now advertised through `expert_ffn_{gate,up,down}_key` and their spelling shared via `tensor_keys::mxfp4_dequantised` — previously the loader wrote keys under a private convention that nothing could ask for.

**`ExpertWeightFfn`** (`larql-compute/src/ffn/expert_weight/`) — the per-expert f32 MoE backend, sibling of `WeightFfn`. Every architecture-specific decision is read from `ModelArchitecture`, not branched on a family name, so a new MoE architecture that answers the accessors is served without touching the file. Split three ways: the driver, `router.rs` (select-then-normalise), `gate.rs` (the gate policies).

**The scorer routes by architecture.** `score_ffn()` replaces the two hardcoded `WeightFfn` constructions, gated on the weights actually being present so a packed-expert model still falls through rather than half-resolving.

#### 4.7.5 Verification

| Check | Result |
|---|---|
| **MoE block vs the reference**, real formulas, 3 tokens × 16 dims | **all 48 values < 1e-5**, first run |
| Control: the old contiguous-halves split against the same expectation | diverges > 1e-3 — the test would have caught the original bug |
| `ClampedGlu` vs PyTorch `_apply_gate` on four branch-covering inputs | < 1e-5 |
| Reference-vs-SwiGLU divergence on those inputs | > 30 absolute — the policies are not interchangeable |
| `larql-models` / `larql-compute` / `larql-vindex` lib tests | 471 / 839 / 1161 pass |

The parity test (`larql-compute/tests/test_moe_reference_parity.rs`) needs **no fixture file**: both sides draw weights from the same LCG in the same order, so the Python generator (`scripts/moe_reference_gpt_oss.py`) and the Rust test see bit-identical inputs. The three tokens deliberately do not all route to the same experts, so per-token routing is pinned too.

**The composition point is the reason this test exists.** Every individual piece had a unit test before and after; the bug was a *composition* error — a bit-exact decode feeding a plausible split feeding a plausible activation feeding a plausible router, each defensible alone and collectively wrong. Only an end-to-end diff against the reference catches that class, which is precisely the instrument the ladder was built to have at R1.

#### 4.7.6 GB runs — and the first thing it caught was my own bug

**`larql shannon score` now scores a MoE model.** The blocker in §4.6.8 is closed.

The first run was on OLMoE-1B-7B (a second MoE architecture, deliberately — a backend that only works on the model it was written for is not architecture-driven). It gave **2.677 bits/char** against the HF reference's **0.390** on the same corpus and the same 512/256 window. A 6.9× gap, and the cause was in the code written an hour earlier: `select_and_normalise` had **GPT-OSS's routing order baked in**, and OLMoE ships `norm_topk_prob: false` — raw softmax probabilities that sum to *less* than 1. Forcing them to sum to 1 inflated the entire expert branch.

This is worth recording rather than quietly fixing, for two reasons.

**First, it is finding 5 committed a second time, by me, immediately after documenting it.** Knowing that the normalise/select order is architecture-specific did not stop me hardcoding one of them. The fix is a typed `ExpertRoutingPolicy` on the architecture, defaulting to `SoftmaxThenSelect`, read from `norm_topk_prob` for OLMoE and overridden for GPT-OSS — the same shape as the gate policy, for the same reason. `olmoe.rs`'s module header had *already* flagged this exact dependency ("the dependency is recorded so a future default change can't quietly invert it"), which is a note that did its job and was still not enough. **A comment naming a hazard does not protect against it; only a type or a test does.**

**Second, this is the ladder's thesis demonstrated on the smallest possible scale.** A rescale-the-expert-branch bug was introduced, and caught within one run by a local reference on a 7B model. At R3 the same class of bug reaches an API oracle as "top-k agreement is a bit low."

| stage | OLMoE bits/char |
|---|---|
| HF reference | **0.390** |
| larql, routing order hardcoded | 2.677 |
| larql, routing policy read from the architecture | **1.901** |

**GPT-OSS-20B, the number this was all for:**

```
$ larql shannon score <gpt-oss-20b> --corpus data/gutenberg/frankenstein.txt --bytes 384
loaded. 24 layers, hidden_size=2880 (43.9s)
tokens scored:          84
bits/token:          3.221
bits/char:           0.708
```

86 GB peak RSS, 51 s wall — the f32 reference tier is expensive but it fits on the box, which is the R1 premise.

**What is closed and what is not.** GB has an instrument, it runs on both MoE architectures tried, and it produces numbers. It is **not passed**, and the two rungs fail for different reasons.

**OLMoE — larql is the suspect.** 1.901 against the reference's 0.390 on an identical 512/256 window: 4.9×, far outside the 0.5 % gate. The residual is *not* in the MoE block — that is pinned to the reference at < 1e-5 (§4.7.5) and the routing order now comes from config. It is elsewhere in OLMoE's forward, which has **never had an end-to-end numerical check** because the scorer could not load it until today. QK-norm placement and the MHA path are the obvious suspects. The HF reference run here is trustworthy (1.59 bits/token for a 7B on Gutenberg is exactly where it should be).

**GPT-OSS — the reference is the suspect, and all three engines failed.** This is the more consequential finding:

> **⚠️ WITHDRAWN 2026-07-31 — see §4.7.7.** The `larql f32` row below was measured on `--bytes 384`, which on this corpus is Project Gutenberg licence boilerplate rather than prose. At matched text larql gives 8.338 bits/token against HF bf16's 8.028 — the engines agree to 4%. The verdict in this subsection compared two different texts and does not stand.

| engine | result |
|---|---|
| larql f32 | 0.708 bits/char, **3.221 bits/token** |
| HF f32 | **crashes** — `RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16`, inside `transformers/integrations/moe.py::_grouped_mm_fallback`. `convert_moe_packed_tensors` dequantises MXFP4 to **bf16** regardless of the requested model dtype, so f32 activations meet bf16 expert weights. |
| HF bf16 | runs, gives **8.028 bits/token** — implausibly poor for a 20B on the opening of *Frankenstein*, so this run is not usable as ground truth either |
| MLX | **crashes** — `ValueError: [gather_qmm] The weight matrix should be uint32 but received float32` |

larql's 3.221 bits/token is the only figure in that table that is even plausible for a 20B model, which is suggestive but is *not* verification — a self-consistent number with no referee. **The GPT-OSS end-to-end comparison remains owed, and it is now blocked on the reference side rather than ours.** *(Withdrawn — §4.7.7. The comparison was not blocked on the reference side; it was never made on the same text.)*

**That is a hole in the ladder's premise, and it should be pre-registered as such.** §4 defines GB at R1/R2 as "a layer-by-layer f32 diff against the local reference implementation", and §1 justifies the whole detour on the grounds that the ancestor "fits on the Mac". *Fitting* turns out to be two claims, and only the first was checked: the weights fit in RAM (86 GB, they do), **and the reference implementation actually executes on this machine** (for `openai/gpt-oss-20b`, it does not). A reference you cannot run is not a reference.

What survives is the reference as a **transcription** rather than an executable — which is exactly what §4.7.5's differential test does, and what §4.6.5 did for sinks. That is a real instrument and it caught real bugs, but it verifies a block, not a forward pass.

**Carry this to R2 as a risk, before committing weekends to it.** Kimi Linear's advertised references are FLA and vLLM kernels; both are CUDA-first, and the ladder's cost model assumes they run locally in f32. **Verify that Kimi Linear's reference implementation executes on Apple Silicon at R2/P1, as the first task of the rung, not after the adapter is written.** If it does not, R2's GB degrades to a transcription diff too — still useful, still far better than an API oracle, but not what §1 priced.

> The instrument now exists and immediately found a defect in the first two models it was pointed at, plus a defect in itself. That is what a gate is supposed to do. GB is unblocked; it is not green.

#### 4.7.7 Correction — §4.7.6's GPT-OSS verdict was measured on the wrong text (2026-07-31)

§4.7.6 concluded that larql's 3.221 bits/token was "the only figure in that table that is even plausible" and that therefore "the reference is the suspect". **Both halves are withdrawn.** The comparison was between two different texts.

`--bytes 384` on `data/gutenberg/frankenstein.txt` does not reach the novel. The first 384 bytes are the Project Gutenberg licence header — *"This eBook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever…"* — which every engine in that table has memorised. **3.221 is a boilerplate score, not a prose score.**

| corpus slice | larql GPT-OSS-20B |
|---|---|
| first 384 B — licence boilerplate | 3.221 bits/token (reproduced exactly, so the engine is deterministic) |
| first 2048 B — boilerplate + prose | **8.338 bits/token** |

§4.7.6 records HF bf16 at **8.028** and dismisses it as implausibly poor. larql on prose gives **8.338** — a 4 % gap, not a 2.5× one. The two engines agree. The spec compared larql-on-boilerplate against HF-on-prose and read the difference as an engine disagreement.

**The control §4.7.6 never had.** Same corpus, same 2048 bytes, same `ExpertWeightFfn` reference tier:

| model | bits/token |
|---|---|
| Qwen3-30B-A3B | **1.400** |
| GPT-OSS-20B | 8.338 |
| OLMoE-1B-7B | 8.297 |

A healthy figure comes out of the same code path, so **the scorer is exonerated**: the two outliers are properties of those models or their forwards, not of the harness. That control is what makes the rest of this subsection load-bearing rather than speculative.

**What changes.**

- **OLMoE's verdict stands, and is strengthened.** larql scores 1.929 bits/char on the *same* 384-byte boilerplate where the HF reference gives 0.390. That gap is not a corpus artifact, so §4.7.6's QK-norm / MHA hunt is pointed at something real.
- **GPT-OSS's verdict is withdrawn.** There is now no evidence that larql's GPT-OSS forward is correct, and none that HF bf16 is wrong. Two readings stay live and need different follow-ups: GPT-OSS may genuinely model raw prose poorly (plausible for a heavily post-trained reasoning model scored without its harmony template), or both implementations share a defect. Scoring it on text matching its post-training discriminates.
- **GPT-OSS is demoted to a secondary rung for any measurement whose output metric is a forward-pass quantity.** Per-layer KL, bits/token, or divergence taken against a GPT-OSS forward is uninterpretable until this closes. **Qwen3-30B-A3B is the working R1-class reference rung** — validated forward, 1.400 bits/token, `qwen3_moe` served by `QwenArch`, and at 6.25 % activation it is closer to K3's 1.8 % than GPT-OSS's 12.5 %.

**Method note, because this is the third measurement-shaped error in §4.7.** §4.7.3's lesson was that a fixture too small to distinguish the candidate behaviours is not a weak test but an absent one. A corpus slice is a fixture: `--bytes 384` on a Gutenberg file cannot distinguish "the model is good" from "the text is boilerplate". **Any bits/char figure entering this document must state its byte range, and both sides of a comparison must use the same one.**

#### 4.7.8 Finding 5, third recurrence — and why the trait default was the mechanism (2026-07-31)

Finding 5 (§4.7.1) was the router's normalise/select order. §4.7.4 recorded the fix as "a typed `ExpertRoutingPolicy` on the architecture, defaulting to `SoftmaxThenSelect`, read from `norm_topk_prob` for OLMoE". §4.7.6 then recorded the *same* bug recurring an hour later in `select_and_normalise`.

It recurred a third time, and the shape is the point: the config read lived on `OlmoeArch` **alone**, while `ModelArchitecture::expert_routing_policy` returned a hardcoded `SoftmaxThenSelect`. Every other MoE architecture silently inherited it. `QwenArch` never read `norm_topk_prob`; Qwen3-30B-A3B ships `true`. Gemma 4 MoE inherited it too.

**Three recurrences is a structural problem, not three bugs.** The mechanism is: a behaviour that is a config fact, read on one architecture, with a trait default that silently answers for everyone else. Patching the affected architecture leaves the mechanism intact.

The read now lives in the trait default (`larql-models/src/config/`), which already had `config()` in scope, and `OlmoeArch`'s override is deleted rather than duplicated. GPT-OSS keeps an explicit override because its order is fixed by the architecture rather than by config — that is the legitimate use of an override.

**The test fixture was the second half of the failure.** `test_detect_qwen3_moe_30b` omitted `norm_topk_prob` entirely, so it could not distinguish the two routing orders and passed under both. Same failure as §4.5's timestamp assertions and §4.7.3's `out_features = 2` — **now three instances in unrelated subsystems.** The fixture is corrected to the real config's `true`, with the policy asserted, plus a table test over `{true, false, absent}` and one pinning GPT-OSS's override against a contradicting config.

**Carried, not done.** The remaining mechanism fix is to make silent inheritance impossible — no behavioural default at all, so a new MoE architecture fails to compile until it declares a policy — paired with a lint rejecting test configs that omit fields the policy branches on. That is a breaking change across every architecture and is deliberately not bundled with this correction.

#### 4.7.9 The coverage audit's first two findings (2026-07-31)

§4.6's work-item 2 is built: `extract::coverage` plus a `tensor_audit` stage that runs first in `build_vindex_streaming`. Reports always, fatal under `LARQL_EXTRACT_STRICT=1` (set in the `larql-vindex` CI workflow). Ten checkpoints audit clean, including Qwen3-30B-A3B at 18,867 tensors and Gemma 3's 439 SigLIP tensors classified `non-text-tower`.

It measures **naming**, not consumption — a recognised tensor is one extraction *can* reach. That is the necessary condition, and it is where every silent drop in this document actually lived: in each case the extractor asked for every key it was told about, and nobody had told it.

**Finding A — GPT-2 from HF safetensors is 1-of-160 addressable.** `gpt2.rs` matches the trait defaults only *after* the GGUF→HF normalisation, so a raw HF checkpoint (`h.N.attn.c_attn.weight`, `wte.weight`, `ln_1.*`) is unreachable. It is not silent — extraction fails late at the embeddings stage — but the message names one missing tensor rather than 159 unaddressable ones.

**Finding B — LayerNorm `β` was dropped for GPT-2 and StarCoder2.** This is the sinks pattern (§4.6) a fourth time, and the most complete instance of it:

| link | state before |
|---|---|
| trait accessor for a norm bias | **did not exist** |
| extraction | never asked, never wrote |
| `build_pipeline_layers` | hardcoded `input_norm_bias: None` |
| Metal `layer_norm` shader | **implemented** `+ bias`, and always took its no-bias variant |
| CPU dense `apply_norm` | resolved it by mangling `".weight"` → `".bias"` — so this path was correct |

LayerNorm is `γ·x̂ + β`. The consequence was that raw-safetensors CPU inference was right while **every vindex-backed and Metal path silently lost the shift term**, on two architectures in the support table.

Fixed by three additive accessors derived once from the weight key and gated on `NormType`, so RMSNorm families correctly claim nothing and an architecture that overrides its norm naming gets the matching bias for free. Extraction now writes them; `build_pipeline_layers` resolves them. Both weight writers also stopped hardcoding `"norm.weight"` in favour of `arch.final_norm_key()` — they had agreed with every reader only by coincidence.

**Status is "the tensor flows end to end", not "the output is verified."** Restoring `β` changes numerics for GPT-2 and StarCoder2 and wants a GB-shaped measurement before it is called complete. Tracked in [`ROADMAP.md`](../ROADMAP.md) §"Extraction tensor-coverage audit + silent-drop follow-ups".

#### 4.7.10 Consequences to carry

- **Every GPT-OSS vindex extracted before 2026-07-30 has mixed gate/up rows in its gate sketch** and needs re-extracting. Nothing was served from those rows (see below), but any KNN/walk routing analysis over them is void.
- **GPT-OSS was never servable from a vindex anyway.** `write_per_layer_moe_kquant` gates on `ExpertFormat::PackedBF16`, so a packed-MXFP4 model writes **no per-layer expert store at all** — extraction reports success and the expert weights simply aren't there. That bounds the blast radius of finding 1 to the gate sketch, and it is a prerequisite for item 14 nobody had noticed was missing.
- **The quantised MoE path still has findings 2–5.** `cpu_moe_forward` / `MoeLayerWeights` has no router-bias field, no expert-bias fields, and no gate policy; `RawSoftmax` is still normalise-then-select. It is untouched here deliberately — it cannot serve GPT-OSS today, and it *is* the measured, working Gemma 4 path, which uses `RenormalizedSoftmax` and is therefore correct for its own family. Bringing it up to the reference is R1/P5 work (item 14), and it now has a local f32 reference to be diffed against, which is the ordering the house rule asks for.
- **Re-run the §4.2 edge-case scan at R3 as already pre-registered**, and add finding 1's check to it: K3's fused-tensor layout is a P1 one-liner (`does the export interleave?`) that must be *read*, not inherited.

### 4.8 The layer diff gets built, and OLMoE closes — R1/P2, 2026-08-04/05

§4 defines GB at R1/R2 as "a layer-by-layer f32 diff against the local reference implementation, **then** `larql shannon verify` ≤ 0.5 % bits/char". **Only the second half existed.** `shannon verify` compares one scalar at the end of a forward pass, so it can say *that* two engines disagree and never *where* — and §4.7's entire six-defect hunt was conducted without it, by transcribing one block into a test rather than diffing the pass.

**Built:** `larql shannon layer-dump` writes each end-of-layer residual as a raw f32 plane plus a manifest; `larql shannon layer-diff` compares two dumps and names the **first** capture that drifts. First matters more than worst — downstream layers inherit an upstream difference, so the deepest disagreement is usually an echo and the shallowest is the defect. The reference side is `scripts/dump_layers_hf.py`, which takes its **token ids from the manifest** rather than re-tokenising, and `layer-diff` *refuses* to compare dumps whose token windows differ. That is §4.7.7's withdrawn verdict encoded as a precondition instead of a lesson.

**OLMoE closed, and the instrument named both defects.**

| difference | before | after |
|---|---|---|
| `rms_norm_eps` — `OlmoeConfig`'s class default is **1e-5**, and `allenai/OLMoE-1B-7B-0924-Instruct` ships no `rms_norm_eps` field, so the fallback is what actually runs. Serving it at the crate-wide 1e-6 was a systematic error in every norm of every layer | final-residual cos **0.890** | **0.991** |
| `qk_norm_scope` — `OlmoeAttention` builds `OlmoeRMSNorm(hidden_size)` and applies it to the whole projection before any reshape into heads, where `Qwen3Attention` builds `Qwen3RMSNorm(head_dim)` and applies it after | 0.991 | **1.000000000** |

**The second one is the more instructive.** The tensor *names* are identical (`self_attn.{q,k}_norm.weight`) and `transformers` marks the distinction in a Qwen3 source comment — *"unlike olmo, only on the head dim!"* — rather than in a type. **Shapes cannot discriminate it either:** OLMoE-1B-7B is MHA, so `num_heads × head_dim == hidden_size == 2048` and the stored `[2048]` weight is exactly as wide under both readings. Only the reduction differs, and treating it as per-head normalises every head to a common magnitude, discarding relative magnitude *between* heads. That is a structural change to attention, which is why it cost 1.9 bits/char rather than a rounding-sized amount.

**Verified end to end, 2026-08-05.** All 17 captures at cos 1.000000000 (max_abs 1.1e-5, rel_rms 3.3e-6 at layer 15 — f32 accumulation-order noise). On the same 384-byte slice, **larql 166.1 total bits / 1.695 bits/token / 0.435 bits/char against the HF f32 reference's 166.10 / 1.6949 / 0.4348.** GB is green on OLMoE.

**§4.7.6's OLMoE verdict is therefore closed, not carried.** Its "1.901 vs the HF reference's 0.390" is superseded: both numbers moved, and the residual gap is nil. The QK-norm/MHA hunt §4.7.6 pointed at was pointed at something real — this was it.

### 4.9 GPT-OSS: the reference *does* run, and larql's forward was wrong — R1/P2, 2026-08-05

§4.7.6 recorded a hole in the ladder's premise: the weights fit in RAM but the reference implementation would not execute, so GB degraded to a transcription diff. **Both halves of that are now resolved, and in opposite directions.**

**The reference runs.** HF f32 crashed in `transformers`' own `_grouped_mm_fallback` because `convert_moe_packed_tensors` dequantises MXFP4 to **bf16 whatever dtype you ask for**, so f32 activations met bf16 expert weights inside `torch.mm`. The fix is four lines in `scripts/dump_layers_hf.py`: cast every parameter and buffer to f32 *after* load, count the casts, and assert the model is uniformly f32 before running. It reports `cast 48 non-f32 tensors to f32`, so a silent no-op cannot be mistaken for "already f32". **A reference you cannot run is not a reference — and this one was two lines of dtype coercion away from running the whole time.**

**And the dequant is lossless, which is why this reference is sound.** MXFP4 values are ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} scaled by an E8M0 power of two — every one exactly representable in bf16's 8-bit mantissa. The attention and norm tensors are excluded from quantisation by `modules_to_not_convert` and load bf16→f32 losslessly. So the bf16 waypoint costs nothing, and §4.7.6's suspicion of the reference was misplaced in a second way.

**larql's GPT-OSS forward was wrong, from layer 0.**

| capture | before | after |
|---|---|---|
| embed | 1.000000000 | 1.000000000 |
| layer 0 | **0.977681807** (rel_rms 0.221) | **1.000000000** |
| layer 17 (worst) | 0.931985023 (rel_rms 0.363) | 1.000000000 |
| layer 23 | 0.974422198 | 1.000000000 |

**The defect is `rope_scaling: {rope_type: "yarn"}`, which larql parsed and then ignored.** `RopeScaling::scaling_type` carried the string; the forward path's only scaling hook was an `Option<Llama3RopeScaling>` parameter, a type that **cannot express YaRN**. So GPT-OSS ran plain RoPE at base 150000. Two independent errors, both quantified against `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS['yarn']` on the real config:

1. **23 of 32 rotary dimensions carried the wrong frequency** — 14 of them by the full 32×, with a 9-dimension untouched band and a ramp between.
2. **Every `cos`/`sin` was 34.7 % too small.** YaRN's `attention_factor` = `0.1·ln(32)+1` = **1.3465735902799727**, applied as `cos = emb.cos() * attention_scaling`. That is not a rotation — it rescales Q and K, so `q·k` was off by 1.81× **at every position, including position 0**.

Point 2 is why YaRN cannot be approximated by ignoring it. A model served without the amplitude is not "slightly off at long range"; it is running attention at the wrong temperature everywhere.

**What landed.** `YarnRopeScaling` + a `yarn_rope_scaling()` trait default that **reads the config** — §4.7.8's mechanism fix applied at the point of introduction rather than after the third recurrence, so any family declaring `rope_type: "yarn"` is served correctly with no per-architecture code. The `Option<Llama3RopeScaling>` parameter became a `RopeFreqScaling` enum, so a `match` is exhaustive and a future family cannot be added and silently ignored at one of the five call sites. `rope.rs` split into `rope/{mod,llama3,yarn,tests}.rs`, and the four attention paths that each reached for the llama3 resolver independently now share one `effective_rope_freq_scaling` — the same shape as §4.6's shared `softmax_in_place`.

**DeepSeek's `mscale`/`mscale_all_dim` are parsed even though no R1 checkpoint uses them,** because when both are present HF computes the amplitude as a *ratio* that collapses to 1.0, where the single-argument form gives 1.35. Honouring YaRN without reading those two fields would have introduced a fresh 35 % error into every DeepSeek layer — a regression caused *by* a fix.

#### 4.9.1 The 85-token fixture could not see the sliding window — and it did not

**GB is green on GPT-OSS at 85 tokens. That licenses a claim about 85 tokens.** GPT-OSS ships `layer_types` alternating `sliding_attention` / `full_attention` at `sliding_window: 128`. **Below 128 tokens a sliding layer and a full-attention layer compute the identical thing**, so the fixture that passed is structurally incapable of testing half the model's attention layers. This is §4.7.3's `out_features = 2` and §4.7.7's `--bytes 384` a third time — *a fixture too small to distinguish the candidate behaviours is not a weak test, it is an absent one* — and this time it was predicted from the config before the diff was run, not discovered afterwards.

The prediction paid out immediately. Two further defects, neither visible at 85 tokens:

- **`is_sliding_window_layer` defaulted to `false` and `gpt_oss.rs` never overrode it**, so all 24 layers ran full attention. `config.layer_types` was parsed *and validated* — and never consulted by anything. §4.7.8's shape a fourth time. **Fixed:** the trait default now reads `layer_types`, so a family that declares its interleave is served correctly with no per-architecture code; stride-based families (Gemma 3) still override. A useful side effect: `has_heterogeneous_attention` now correctly reports GPT-OSS as hybrid, so a backend without `Capability::HeterogeneousAttention` fails loudly instead of computing the wrong thing quietly.
- **The dense f32 path has no sliding-window support at all.** `attention/gqa.rs` slices `k[0..qi+1]` — causal, unwindowed — and the word "window" does not appear in `attention/block.rs`. Windowing exists only on the quantised `pipeline_layer` path. So the accessor fix is *necessary and not sufficient*: the reference tier GB measures still attends over the whole prefix on all 24 layers.

**Consequently the 2048-byte score was not covered by the gate.** The post-YaRN figure of 9.437 bits/token over 510 tokens crossed the window and so ran 12 layers with the wrong attention span. It is superseded by §4.9.2, and no comparison should be drawn between it and §4.7.7's 8.338.

#### 4.9.2 Sliding window implemented, and GB closes above the window — 2026-08-05

**Landed:** `attention/span.rs` — one `AttentionSpan` type resolved from `is_sliding_window_layer` + `sliding_window_size`, consumed by **both** prefill (`gqa.rs`'s two score loops, via `run_attention_block_core`) and decode (`gqa_attention_decode_step_in_span`, via the decode dispatch). One type on both paths deliberately: decode dropping different keys than prefill for the same absolute position is the divergence a KV cache hides best, and `AttentionSpan::decode_range` is pinned against `AttentionSpan::range` across 300 positions so the two cannot drift.

Captured attention distributions are written back at their **absolute** offset, so a consumer still reads key `j` at index `j` and sees exact zeros outside the window rather than a left-shifted distribution.

The span-carrying entry points are the real ones; the existing unspanned names became `AttentionSpan::Full` wrappers — the same layering `rope.rs` already uses, and it keeps 68 call sites untouched while making the windowed path explicit at the three that matter.

**Verified at 511 tokens — 4× the window.**

| capture | cos | rel_rms |
|---|---|---|
| embed | 1.000000000 | 0.000000 |
| **layer 0** (sliding) | **1.000000000** | 0.000001 |
| layer 4 | 0.999997361 | 0.002297 |
| layer 19 (worst) | 0.999976892 | 0.006798 |
| layer 23 | 0.999988494 | 0.004797 |

**Layer 0 is the proof.** It is a `sliding_attention` layer, the fixture is 4× its window, and it sits at cos 1.000000000 — a window off by even one position would show there first and unmistakably. Every capture clears the 0.9999 gate.

**The residual is larger than the 85-token run (rel_rms 4e-6 → up to 6.8e-3) and that is worth naming rather than waving through.** Direction is preserved essentially exactly (cos ≥ 0.99997) while `max_abs` reaches 451 — a few elements far out, not a broad shift. The attribution is MoE expert-selection disagreement, and §4.9.3 measures it rather than asserting it.

**Tests that the old fixture could not have been.** Deliberately run *past* the window, since below it there is nothing to see: a control pinning that windowed and full agree exactly below the window (the property that made the defect invisible), its converse above the window, a windowed position checked against an independent full-attention run over just its window's keys, absolute-offset capture, and prefill↔decode agreement on the last row. Plus `AttentionSpan`'s own suite, including the pin that the first position where sliding and full differ is *exactly* the window index.

**Still owed:** the Metal rope shaders compute their own frequencies and have not been given YaRN, and the Metal attention kernels have not been given the span. **GPT-OSS on Metal is unmeasured and should be assumed wrong until diffed.** `gqa_attention_asym` (MLA/DeepSeek) is deliberately unwindowed — that family has no sliding layers.

#### 4.9.3 The residual attribution, measured — and the test that nearly got it wrong

§4.9.2's explanation of the residual was *reasoned*: direction preserved, extremes on few elements, shallow layers flat, therefore MoE tie-breaking. That is a causal claim about which tokens diverge and why, and none of it had been measured. **Measured now**, via `LARQL_MOE_ROUTE_TRACE` on the larql side, `--router-trace` on the reference side, and `scripts/diff_moe_routing.py`.

**Expert selections disagree on 11 of 12,264 token-layer decisions (0.09 %), across just 4 of 511 tokens.**

| layer | token | router margin | rel_rms *entering* the layer |
|---|---|---|---|
| 4 | 505 | **0.00000** | 1.96e-06 |
| 8 | 505 | 0.00088 | 0.000526 |
| 14 | 420 | 0.00000 | 0.001650 |
| 16 | 420 | 0.00943 | 0.002288 |
| 17 | 420 | 0.01484 | 0.004271 |
| 18 | 420 | **0.07159** | 0.004449 |
| 19 | 420 | 0.02140 | 0.006422 |
| 20 | 510 | 0.00006 | 0.006798 |
| 22 | 428 / 505 / 510 | 0.00024 / 0.00735 / 0.05510 | 0.005765 |

**The causal loop closes on the planes themselves.** At layer 3 — before any disagreement — the largest per-token residual sits on token 509, which never flips: ordinary background noise. From layer 4 onward the largest residual is *always* on a flipped token, and at the final layer the top four positions by max-difference are exactly the four flipped tokens. **98.17 % of the final layer's squared residual sits on 0.78 % of positions.** The disagreements do not merely accompany the residual, they are it.

**The pre-registered test failed, and it was the test that was wrong.** The bar was set before looking: ≥ 90 % of disagreements at a margin ≤ 1e-2. The data gives 63.6 %, and four flips at margins up to 0.0716 — on which the first verdict read *"REFUTED — something is wrong with the router itself"*.

That verdict was an artifact of **scoring a causal chain as independent draws.** The same handful of tokens flip repeatedly, and each flip perturbs its own residual, which shifts its own router logits downstream, which lets it flip again at a margin far above a tie. Token 420 is the whole story: 0.00000 at layer 14, then 0.00943, 0.01484, 0.07159, 0.02140 as its state drifts further. A flat threshold cannot see that.

**The discriminating test is temporal, not distributional.** A defective router disagrees regardless of upstream state, so it would flip at wide margins from the *first* layer, where nothing has diverged. A cascade cannot — it has nothing to cascade from. And the data separates cleanly:

- layers 0–3, at rel_rms ~2e-6: **zero disagreements**;
- the seed at layer 4 is an **exact tie (0.00000)** entering at rel_rms 1.96e-06;
- **every** wide-margin flip enters with rel_rms ≥ 0.0043 — none occurs before the residual has drifted;
- all 4 flips below that divergence are ties.

**Verdict: CONFIRMED — a tie-break cascade seeded by one exact tie, not a router defect.** The tool now performs this test (`--layer-diff`) instead of the flat one, so the next person gets the right question rather than the right answer to the wrong one.

**Two carried lessons.** First, the general one, which is [§4.7.3]'s in a new costume: *a threshold chosen without calibrating it against the quantity it bounds is a guess wearing a number.* 1e-2 was picked without ever measuring how closely the two engines' router logits agree, and it happened to sit inside the range a cascade produces. Second: **the losslessness claim in §4.9 survives** — a direct check on a real shard confirms every dequantised MXFP4 value round-trips through bf16 exactly, so the reference's expert weights equal larql's and weight precision contributes nothing to this residual. That mattered: had it been false, the whole attribution would have had a second, unmeasured cause sitting behind it.

> The instrument's second outing found a defect in the first model it was pointed at, then its own fixture predicted two more it could not see, closing those needed a primitive the dense path had never had, and the leftover residual turned out to be a four-token cascade from a single exact tie. GB is green on OLMoE, and green on GPT-OSS **at 511 tokens, past the window** — on the CPU f32 path only.

### 4.10 Metal decode ignores *every* RoPE scaling family — and the parity harness tests the half that works (2026-08-05)

§4.9.2 carried "the Metal rope shaders have not been given YaRN" as a GPT-OSS follow-up. **The item was understated by a wide margin.** Auditing it found not a missing family but a missing *category*, live on shipped models today.

**The four Metal rope kernels take `rope_base` and nothing else.** Each recomputes `freq = 1/pow(base, 2d/rdim)` in-shader. `FullPipelineLayer` — the single construction site for per-layer params — carries `rope_base`, `rotary_dim` and `sliding_window`, and **no scaling field of any kind**. So Metal decode applies unscaled RoPE regardless of what the checkpoint asks for.

**Metal prefill is correct, which is why nothing caught it.** Prefill ropes on the *host* through `apply_rope_partial_at_full` (`attention/gpu.rs`), so it honours llama3 bands, YaRN and the linear position divisor. Decode ropes in-shader (`decode/encode_attn.rs`, `decode_hybrid.rs`) and honours none of them. **The same model therefore encodes its prompt under one rule and generates under another.**

**Who is affected, from the local checkpoint cache:**

| `rope_type` | model | status |
|---|---|---|
| `linear` (per-layer, global only) | `google/gemma-3-4b-it`, `gemma-3-12b-it`, `gemma-3-4b-pt` | **live** — the repo's primary model |
| `llama3` | `meta-llama/Llama-3.2-1B` | **live** |
| `yarn` | `openai/gpt-oss-20b` | **live** — Metal-served since 2026-08-10 (§4.11) |

**This is a known bug class, fixed once already, in the other engine.** `kquant_forward/cached/` carries the comment: *"The unscaled `apply_rope_partial_at` here was the direct-path divergence on gemma3-4b (global-layer K/Q rope'd at 8× the position the prefill cache used)."* That is this defect, on the CPU direct path, found and fixed — and the identical defect sat untouched in the Metal kernels because the fix was applied where the bug was seen rather than to every path that ropes.

**And the harness could not see it.** `examples/residual_diff.rs` is the CPU-vs-Metal instrument, and its own header says it triggers "a single prefill pass — no KV cache involvement". Prefill is the half that routes through the host and is correct. **A parity harness that exercises only the correct path returns green forever** — [R14 gate–claim congruence](dec-funnel.md) at the harness level, and the fourth instrument-shaped defect in this document.

**The fix, landed.** A `RopeFreqPlan { inv_freq, amplitude }` precomputed on the host by `rope_freq_plan(...)` — the *same* function `apply_rope_partial_at_full` uses, so there is one definition rather than a CPU one and a shader one. Base, llama3 bands, YaRN's ramp and the position divisor all fold into those two values (`angle = (pos/divisor)·inv_freq` is exactly `pos·(inv_freq/divisor)`), so the kernels take a buffer and a scalar and no longer know about scaling families at all. `inv_freq` stays `f64` on the host; narrowing happens only at the GPU boundary, because the reference tier is what Gate B measures.

**All four rope-bearing shaders converted in one change** — `rope` (4 kernels), `qk_norm_rope_fused`, `attn_fused` and `fused_attention`, eight rotation sites. **Zero `pow(rope_base, …)` remain in any shader.** The frequency table took over each kernel's existing `rope_base` slot in place and `amplitude` was appended at the end of every signature, so no other buffer index moved. `stages::rope_freq` owns the binding convention in one place, modelled on `stages::sinks`, and **verifies the table length against the layer's own `head_dim`/`rotary_dim`** — a table of the wrong width would otherwise rotate against wrong frequencies and look like a plausible forward pass.

**The GPU suite is what made this safe: 551 Metal tests pass, and they caught 16 distinct binding mistakes on the way.** Every one was a raw-dispatch site still writing a bare `rope_base` float into what had become a pointer slot; each showed up as `cos = 0.000000` against the CPU reference rather than as a compile error, because Metal bindings are untyped. Two more surfaced as `Command encoder released without endEncoding` — the geometry assert firing *inside* a live encoder, which Metal reports as an abort rather than a readable panic. One of those was a genuine latent hazard: a test mutating `layer.rotary_dim` after the plan was built, desynchronising the two. That now goes through `set_rotary_dim_unscaled`, which moves both together.

**Still owed:** `residual_diff.rs` covers prefill only, so the CPU-vs-Metal *decode* comparison that would have caught this in the first place still does not exist — tracked as **M3/M5** in [`ROADMAP.md`](../ROADMAP.md). The kernel-level parity tests now pin each rope kernel against the CPU reference across Llama-2, Gemma 3, Gemma 4 sliding and Gemma 4 global-partial geometries, which is a sharper instrument than the end-to-end diff but not a substitute for it. `AttentionSpan` is still absent from the Metal attention kernels (**M4**).

### 4.11 R1 P4/P5 closed — GPT-OSS served from its vindex, then made fast on Metal (2026-08-09/10)

Registry `k3r1-gptoss-pipeline`. **P4 extract and P5 serve are closed on both
backends.** `larql run gpt-oss-20b-q4k.vindex` generates coherent
harmony-format output on CPU, and `--metal` produces the **identical 32-token
greedy trajectory** — re-verified after every change below.

**Correctness first (2026-08-09/10, commits 04addd27…19cbefd6).** Serving took
six stacked defects, every one invisible on hidden%256==0 models (GPT-OSS is
2880): the MoE norm topology (`PreExpertsNorm`, not `Residual`), the lm_head
matvec and loader widths, the Metal QKV/O stored-row width (superblock
truncation — the Mixed route had no assert and truncated silently), missing
attention-projection biases on Metal (Q/K/V/O — no support existed at all), and
the dense-FFN branch encoded over a pure-MoE layer's empty weight slices
(NaN poisoning, found by `LARQL_DECODE_DIAG_LAYER=0`). The generic fixes:
`QuantWeight::stored_cols` (byte count is the width authority, the
`stored_gate_up_cols` contract), bias threading through `build_arch_params`
(the sinks pattern), and `has_dense_ffn()` — dense-branch presence is a
representation fact, not an architecture flag.

**Then the decode ladder (2026-08-10), each rung parity-gated:**

| ms/token | tok/s | Rung | Commit |
|---:|---:|---|---|
| 97.8 | 10.2 | staged expert copies (top-4 × ~22 MB × 24 layers ≈ 2.1 GB/token of CPU memcpy) | — |
| 25.9 | 38.7 | **zero-copy regions** — each layer-weights mmap bound once via `newBufferWithBytesNoCopy`; experts are byte offsets into the region buffer | 3ed9623f |
| 22.7 | 44.0 | **grouped experts** — K3a's `q6k_grouped_experts` (η 0.64→0.90) over the offset table; Q4_K sibling wired for Gemma | a07cf870 |
| 22.6 | — | **fused attention** — `attn_fused` gains no-QK-norm + QKV-bias slots; attention GPU 8→3.3 ms but the wall barely moved: the CBs cost ~8 ms of wall regardless of occupancy | 1d171d08 |
| 16.7 | **59.8** | **merged command buffers** — `moe_weighted_combine` moves the combine on-GPU (identity-combine class) so a layer's experts + the next layer's attention share one CB; one wait/layer, not two | f557cec9 |

The fused-attention rung is the diagnostic gem: **faster kernels ≠ faster
decode when command-buffer structure dominates the wall** — it isolated the
sync term that the merge rung then deleted. Gemma 26B A4B hybrid rode the
zero-copy + grouped rungs for free (91.3 → ~30 ms/tok); every precondition
miss falls back to the prior path byte-for-byte.

**Honest external comparison** (recorded so nobody cites the wrong ratio):
oMLX runs `gpt-oss-20b-MXFP4-Q8` — **native ~4.25 bpw experts** — at ~85-90
tok/s (best ~100) on an M3 Max 40c at 1k-4k context. LARQL carries the
lossless MXFP4→Q6_K transcode at 6.56 bpw, ~1.54× the expert bytes, at short
context. 59.8 tok/s × 1.54 ≈ 92 byte-normalised: the engines are in the same
conventional-efficiency territory; the residual difference is representation,
which is a choice, not a gap. The MXFP4-native experiment (shaders
`mxfp4_matvec` / `mxfp4_grouped_experts` already exist) is now about going
*past* the reference, not reaching it.

**Remaining budget at 16.7 ms:** ~11.5 ms MoE (genuine bandwidth — the next
MoE win is representation or η, not scheduling), ~3.3 ms attention GPU, ~2 ms
in the remaining 24 per-layer waits (GPU routing + the offset tables the
grouped kernels already read from device buffers would take this near zero).

**K3 consequence.** The substrate the residency/prediction work assumed now
exists and is measured: vindex-resident weight regions directly addressable by
GPU execution, expert offset tables, grouped sparse execution, and a
cross-layer command-buffer structure. The graph/residency work gets to be an
accelerator, not a rescue operation.

**Still open from P5:** `predict_kquant_metal` (the max-tokens-1 predict path)
panics on pure MoE — a separate dense-assuming path; GB-style bits/token
scoring of the served vindex (§4.6.8's scorer gap) remains unbuilt.

### 4.12 GPU-resident routing + native MXFP4 experts — 59.8 → 77.2 tok/s (2026-08-13/14)

Both "next MoE wins" named at the end of §4.11 landed. GPU routing (PR
#249) turned the route into an index — router matvec, top-k, and the
descriptor gather run as GPU dataflow, one command buffer per token,
route-witness counters pinned at zero (opt-in: `LARQL_GPU_ROUTE=1`) —
59.8 → 67.5 tok/s. The representation win followed on
`feat/moe-g4b3-region-registration`: `--routed-from` serves a VINDEX3
container's native MXFP4 banks (paired e8m0 streams, complete-or-refuse
admission with five tamper controls) over the Q6_K spine, and a
vectorised split kernel (`uint4` group loads, arm `a2`, oracle-exact)
recovers the skeleton deficit the K2 ceiling probes located. Paired
same-session ladder: Q6_K 68.3 → MXFP4 74.6 → **77.2 tok/s** (12.95
ms/token). Kernel-measured bandwidth predicts the expert-read delta
within ~0.1 ms of e2e — no integration tax. The remaining ~0.9 ms to
the MLX-class ~83 reference is fully attributed: ~1.09 ms lm_head
Q4_K matvec at bandwidth ceiling (irreducible as stored) + ~0.5 ms
decode↔lm_head command-buffer boundary (open: fold final norm →
lm_head → top-K into the decode CB). Full ladder + evidence: ROADMAP
"Native MXFP4 served end to end (2026-08-14)".

### 4.13 K3-RESIDENCY-VERTICAL-1 — the real plan through selection, refusal, accounting and execution (2026-09-05)

Preregistered in
[`represent/forecasts/k3-residency-vertical.json`](represent/forecasts/k3-residency-vertical.json)
after the REPRESENT codec contract closed at rung 3d (PR #425). The
executable subject is Kimi-Linear-48B-A3B, the R2 rung, because K3 proper
has no weights on this box; the machinery is what transfers. The baseline
was measured with a read-only instrument, `larql vindex3 ops <container>
--realizations`, which prepares the plan from segment headers alone: on
the real container every one of 20,081 planned operands is pinned, the
pins declare 178 GiB resident against 128 GiB, the plan silently omits the
shared expert its own graph declares (the width field postdates every
container on disk), and the per-expert bank is planned as 19,968 whole
projections the executor cannot run. The forecast fixes the interventions
(plan completeness, a budget at preparation, the mapped-bank and
shared-expert realizations through the prepared plan, observed pricing, a
three-rung parity ladder), the predictions and the falsifiers before any
of it is built.

## 5. Claims under test

| ID | Claim | Falsifier |
|----|-------|-----------|
| K1 | The existing trait layer absorbs a hybrid-linear-attention MoE with latent routing without a signature change (G0 at all three rungs) | any of the five traits needs a breaking change |
| K2 | KDA implemented against a local reference (R2) transfers to K3 with only decay-parameterisation deltas | R3's KDA needs structural rework, not parameter changes → R2 was the wrong ancestor |
| K3c | A coverage curve exists: some resident-expert fraction well below 100 % meets the C6 drift gate on a 896-expert bank | drift exceeds the gate at every fraction that fits the demo box |
| K4 | The expert-granular vindex profile serves three capability tiers (exact / walk / demo-slice) from **one** extraction pass | tiers require separate extraction runs → the profile is under-specified |
| K5 | Bisection works: an R3 Gate-B failure localises to KDA-vs-K3 by re-running the same prompt on R2 | R3 failures are not reproducible on either ancestor → the ladder bought verification but not diagnosis |

K1 is answered at R1/P1. K2 is the ladder's central bet and is answered at R3/P2. K3c feeds [`vindex-factory.md` §15.4](vindex-factory.md) (routing-stats pinning) and DEC-3. K5 is a methodology claim and is the reason R2 stays runnable after R3 starts.

## 6. Feature list

Twenty-three items, organised by crate the way the workspace carves it. `new` / `extend` is against today's code; the gate column names the gate that *proves* the item, not every gate it touches.

### larql-vindex / larql-vindex-spec

| # | Feature | Kind | Rung | Phase | Gate |
|---|---|---|---|---|---|
| 1 | **Expert-granular profile** — extent = (layer, expert), coarse sibling of the feature-major profile; manifest schema gains `expert_count`, `latent_dims`, router metadata | new | R1 | P4 | GA |
| 2 | **MoE-aware capability tiers** — existing tier machinery expresses exact-extents / walk (sketch + down payloads) / demo-slice as three `capability` values from **one** extraction | extend | R1 | P4 | GA → GC |
| 3 | **Per-expert gate sketch** — resident-mountable compressed keys with baked per-feature scalars (the up-saturation constants); the scaled-down `GateIndex` for an 82K-expert bank. **Re-scoped by §4.4 finding 2:** the substrate is the existing top-K SVD summary path, wired into the `PackedMxfp4` arm; the remaining question is *evaluative* — is top-K SVD sufficient, or does it want PQ residuals on top? — measured against the R1 harvest | extend (was new) | R1 | P4 | GC |
| 4 | **Extent integrity + provenance** — sha256 per extent, source-shard byte-range map, round-trip verifier | extend | R1 | P4 | GA |
| 5 | **Slice-cut tool** — routing-trace → demo-vindex emitter (top-N per layer, renormalisation metadata for skip-mode) | new | R1 | P6 | GC |

Items 1–5 are what make [`vindex-factory.md` §15.3](vindex-factory.md)'s three-tier table (demo ~55–65 GB Hub / walk ~450 GB R2 / exact ~1.35 TiB R2) expressible as `outputs[].carve` rather than as three hand-run extractions.

### larql-models

| # | Feature | Kind | Rung | Phase | Gate |
|---|---|---|---|---|---|
| 6 | **compressed-tensors reader** — vLLM-style `weight_packed`/`weight_scale` pair parsing over the safetensors index; HTTP range source alongside the existing mmap source | new | R1 † | P1 | GA |
| 7 | **Model adapters** — `KimiLinearArch` (R2) and `KimiK3Arch` (R3) behind `ModelArchitecture`; K3 adds LatentMoE W↓/W↑ wiring, QB frozen-bias router, MTP head preservation (the M0 rule enforced in code, not in prose) | new | R2, R3 | P2 | G0 + GB |
| 8 | **Kimi tokenizer** — carries R2 → R3 | new | R2 | P1 | GB |

Item 7 follows the pattern of the in-flight `architectures/olmoe.rs`: one module, one `ModelArchitecture` impl, registered in `architectures/mod.rs`. Item 8 gates GB because bits/char is tokenizer-sensitive — a wrong tokenizer produces a plausible-looking drift number.

**† Item 6 stays at R1, but R1's default checkpoint does not exercise it** (P1 audit finding, 2026-07-28). `openai/gpt-oss-20b` ships `quantization_config.quant_method: "mxfp4"` with OpenAI's native `experts.{gate_up,down}_proj_{blocks,scales}` naming — not vLLM's `weight_packed`/`weight_scale` pairing. The compressed-tensors convention is what Moonshot's releases use, so on architecture alone item 6 belongs at R2/R3.

It is kept at R1 deliberately, on the grounds that the reader and the HTTP-range source are better built and exercised early than at 1.4 TB. That decision carries an obligation: **R1 must additionally point at a compressed-tensors-packed GPT-OSS repo** (a vLLM-requantised mirror) so the item has a real GA gate. An item carried at a rung without a checkpoint that exercises it is worse than an item deferred — it produces confidence at R1 that R2 then discovers was never tested.

### larql-inference

| # | Feature | Kind | Rung | Phase | Gate |
|---|---|---|---|---|---|
| 9 | **KDA `KvEngine`** — naive f32 recurrence: ShortConv → Swish → L2Norm, delta update with Diag(α) decay under g_min-bounded parameterisation, gated RMSNorm out; fixed-size canonical state; new state-policy taxonomy row with checkpoint/restore at boundaries | new | R2 | P2 | G0 + GB |
| 10 | **MLA-NoPE attention** — naive materialised K/V first; latent cache path later | new | R2 | P2 | GB |
| 11 | **Hybrid layer interleave** — 3:1 KDA:MLA scheduling in the forward loop | new | R2 | P2 | GB |
| 12 | **Block AttnRes** — running block sums + softmax over pseudo-queries, embedding as b₀; α weights exposed on the trace (the zone-map instrument, free) | new | R3 | P2 | GB (oracle) |
| 13 | **SiTU-GLU** — both softcaps, in `sparse_compute`'s architecture table alongside geglu | new | R3 | P2 | GB (oracle) |
| 14 | **Routed `FfnBackend`** — router → per-expert dispatch behind the existing trait; three serving modes per expert (full-extent / walk-rescore / down-only); skip-mode with gate-weight renormalisation for slice misses | extend | R1 | P5 | G0 + GC |
| 15 | **Dense-as-walk unification** — shared experts and any dense FFN served through the same walk path; dense = one-expert MoE, formally | extend | R1 | P5 | GB |

Item 9 is a new *kind* of state, not a new state policy — the taxonomy gains a row (fixed-size recurrent) and the chaining/checkpoint contract is where it earns its GB. Item 15 gates on GB rather than GD because the acceptance criterion is that the existing dense path does not change numerically.

### larql-compute / larql-compute-metal

| # | Feature | Kind | Rung | Phase | Gate |
|---|---|---|---|---|---|
| 16 | **MXFP4 codec** — confirm the GPT-OSS path decodes vLLM's packing; streaming dequant off mmap | verify-then-extend | R1 | P1 | GA |
| 17 | **AMX/Accelerate expert tier** — thread-pool dequant-matmul over latent-space rows; DEC-0's measured thread-filling regime productised | new | R1 | P5 | GD |
| 18 | **Grouped expert scheduler** — gather-by-expert GEMM batching on Metal; already the named pre-DEC-2 priority, lands here | new | R1 (bites at P6) | P5 | GD |
| 19 | **Q4K client requant** — attention/dense client to Q4K under a bounded-KL contract | extend | R3 | P6 | GC |

Item 18 is the one item with a scheduled consumer outside this document: [`dec-funnel.md` §DEC-2](dec-funnel.md) states no tier-capacity number may be quoted before it exists (DEC-0 measured unique-expert bytes at 13.9 % of naive at B64 — ~7.2× headroom).

> **Do not carry that 7.2× to K3.** It was measured on gemma4-26b-a4b-q4k at **8-of-128 = 6.25 % activation**. K3 is **16-of-896 = 1.79 %**, where unions overlap far less: `dec8-5-k3-batch-union` puts the same quantity at **~3.0×** at B64. Union amortisation is `B·k / E·(1-(1-k/E)^B)` and depends on `k/E`, not batch alone. See [`dec-funnel.md` §1 standing rule R2](dec-funnel.md).

### Harvest / metrology (`larql dec-bench`)

| # | Feature | Kind | Rung | Phase | Gate |
|---|---|---|---|---|---|
| 20 | **Instrumented harvest mode** — one forward pass emitting routing trace, feature-activation log, up-saturation histogram, latent pools; the run that prices everything downstream | new | R1 | P3 | GA |
| 21 | **Coverage-curve runner** — resident-fraction sweep vs C6 drift; C6 itself already shipped | extend | R1 | P3, P6 | GC |
| 22 | **API-oracle gate** — temperature-0 top-k agreement harness against the Kimi endpoint, preserved-thinking-mode aware; the Gate-B tool where no local reference exists | new | R3 | P2 | GB |

Item 20's output is a **pinned** artifact, not a scratch file: [`vindex-factory.md` §15.4](vindex-factory.md) requires `source.routing_stats.artifact_sha` to hash into `build_id`, because hot-expert selection is traffic-dependent and unpinned routing statistics are the one mistake that cannot be retrofitted — the traffic that produced them will not exist again. **This applies from the very first K3 harvest, even if that harvest is hand-run.**

### M-ladder

| # | Feature | Kind | Rung | Phase | Gate |
|---|---|---|---|---|---|
| 23 | **MTP verify loop for KDA** — replay-from-projected-inputs rollback (the ReplaySSM pattern) instead of residual-truncate; slots into M1 as the recurrent-state case | extend | R2 | P5 | GB |

Existing M-ladder rows are unchanged: M0 (MTP head preservation) becomes item 7's responsibility in code; M1's greedy-acceptance/exact-match story is what makes item 23's gate GB rather than GD.

### 6.1 Weight by rung

| Rung | Items | Character |
|---|---|---|
| **R1** | 1–6, 14–18, 20–21 | heaviest by count, but roughly half is extension of proven code |
| **R2** | 7 (KimiLinear), 8–11, 23 | the risk, concentrated where the local reference lives |
| **R3** | 7 (K3), 12–13, 19, 22 + scale logistics | the smallest pile, by design |

### 6.2 Untouched — which is the point

The vindex format core, mmap zero-copy, the `GateIndex` / `FfnBackend` / `KvEngine` / `Dispatcher` traits, WalkFfn, the activation cache (expert-set keys are just new keys), `shannon verify`, C6, the wire codecs, and the state-policy taxonomy all stand unchanged. Twenty-three features, five crates, no trait signature change — **that is G0, and it is the ladder's first falsifiable claim.**

## 7. What the ladder cannot retire

**Block AttnRes (12), SiTU-GLU (13), and the QB frozen bias (inside 7)** exist in no smaller open model. They stay API-oracle-verified at R3.

This is acceptable because they are also the *simplest* components in the pile — a softmax over block sums, two tanh caps, a frozen additive bias — which is the right place for the weakest oracle. The components with the deepest state (KDA recurrence, hybrid interleave, MTP rollback) all have local references; the components with only an oracle are all stateless and inspectable in a few dozen lines.

**Scale logistics** — ~1.4 TB download, ≥3 TB NVMe, campaign-mode extraction, multi-box carve — are not de-risked by any rung. They are [`dec-funnel.md` §DEC-6](dec-funnel.md)'s infra line and [`vindex-factory.md` §15.5](vindex-factory.md)'s campaign work, and they are orthogonal to adapter correctness.

## 8. Relationship to the other two specs

| Document | Owns | Changes from this doc |
|---|---|---|
| [`dec-funnel.md`](dec-funnel.md) (v0.5) | transport: the wire, the batch curve, the router, the fleet | DEC-6's "3–6 weekends, one shot at 2.8T" becomes "R1 → R2 → R3"; C8 and DEC-7 unchanged; DEC-3 pass 2 consumes item 20's harvest |
| [`vindex-factory.md`](vindex-factory.md) (v0.1) | production: recipes, reproducibility, campaigns | items 1–5 are what §15.3's `outputs[].carve` needs to be expressible; item 20 satisfies §15.4's pinning requirement; VF-4 stays gated on R3, per §15.1 |
| this document | the adapter: what code has to exist, and what proves it | — |

Sequencing note carried from [`vindex-factory.md` §15.6](vindex-factory.md): **let the first K3 harvest be manual**, and bring the working invocation back as a recipe afterwards — with routing stats pinned from that first run.

## 9. Risks (pre-registered)

- **R2 is the wrong ancestor (K2 falsified).** If K3's KDA differs structurally rather than parametrically from Kimi Linear's, R2 buys the ShortConv/gate/interleave scaffolding but not the recurrence itself. Detection is early — R3/P2's first layer diff — and the fallback is the direct route with R1's pipeline already proven, so the loss is bounded to R2's weekends, not to the programme.
- **The API oracle is weaker than assumed.** Preserved-thinking-mode and sampling-path differences can make top-k agreement noisy enough to hide a real bug in items 12/13. Mitigation: item 22 is built and *validated against R2* — run the oracle harness against Kimi Linear's endpoint where a local reference also exists, and measure the oracle's own false-negative rate before trusting it at R3.
- **Coverage curve has no usable knee (K3c falsified).** If no resident fraction meets C6 at demo-box size, the demo tier does not exist as specified. The carve is renegotiated (more resident experts, bigger box, or a narrower prompt distribution) — the drift gate is not.
- **R1 productisation expands.** Items 14–18 are "extend proven code", which is exactly the class of estimate that doubles. Time-box R1's serving phase; items 17/18 have independent value to DEC-2 and can ship on DEC's schedule rather than this ladder's.
- **Two extra rungs delay the video.** The mitigation is that R2 is itself a shippable artifact and a demo — the ladder produces content rather than only consuming time. If the timeline compresses, R1 is non-optional (it's the pipeline) and R2 is the cut.
- **Trait absorption fails (K1 falsified) at R2, not R1.** The likely candidate is `KvEngine` meeting fixed-size recurrent state with checkpoint/restore semantics. Pre-registered as a possible finding rather than a failure: if the trait needs a new associated method, that is a real architectural result about the state-policy taxonomy and should be taken as one.

## 10. Registry conventions

Programme **`k3`**, created 2026-07-28 with five planned experiments. The ladder gets its own programme rather than living inside `dec`: DEC's claims are about transport and are measured on models it does not have to *understand*, while these are claims about adapter correctness. Different falsifiers, different instruments.

| Slug | Covers | Rung / phases | Gates |
|---|---|---|---|
| `k3r1-gptoss-pipeline` | format, expert-granular extraction, harvest, serve (items 1–6, 14–18, 20) | R1 P1–P5 | G0, GA, GB, GD |
| `k3r1-coverage-curve` | resident fraction vs C6 drift; the first coverage curve (items 3, 5, 21) | R1 P3, P6 | GC |
| `k3r2-kda-adapter` | KDA / MLA-NoPE / 3:1 interleave vs local reference (items 7, 8–11) | R2 P1–P2 | G0, GB |
| `k3r2-kimilinear-serve` | second-architecture pipeline run + MTP rollback (item 23) | R2 P3–P5 | GB, GD |
| `k3r3-k3-adapter` | AttnRes, SiTU-GLU, QB bias, LatentMoE, client requant, oracle gate (items 7, 12, 13, 19, 22) | R3 P1–P2, P6 | GB (oracle), GC |

`dec6-k3-extract` and `dec7-k3-live` stay in programme `dec` and stay as they are — DEC-6 remains the extraction execution record and DEC-7 the live-demo record; `k3r3-k3-adapter` covers the adapter delta that feeds them. `dec6-k3-extract.design.spec` should be repointed from `dec-funnel.md v0.4.2 §3 DEC-6` to this document once the gate scheme is confirmed.

Metric schema additions: `k3/coverage_frac`, `k3/drift_bits_per_char`, `k3/oracle_topk_agreement`, `k3/expert_resident_bytes`, alongside the existing `dec/*` schema.

## 11. Budget and sequencing

| Rung | Spend | Notes |
|---|---|---|
| R1 | ~£0 | 13 GB, Mac-local; the box is already the reference tier |
| R2 | ~£0–5 | ~48B total / 3B active fits quantised on the Mac; optional cheap x86 arm for item 17 |
| R3 | per `dec-funnel.md` §8 | DEC-6 ~$10–15 (download-dominated) + DEC-7 ~$10–20 |

**Order:** R1/P1 first — it is the G0 check and the compressed-tensors reader, and it costs an afternoon. Then R1 straight through P5, since items 17/18 are already on DEC-2's critical path. R2 begins at P2 (the adapter is the point; its P1 is small). R3 starts only when R2 passes GB — and R2 stays runnable afterwards, because K5 (bisection) is the instrument that makes R3's oracle survivable.
