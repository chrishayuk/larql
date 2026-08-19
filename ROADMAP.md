# LARQL Roadmap

Top-level plan. Per-crate detail lives in each crate's own `ROADMAP.md`.
This file tracks the demo narrative, the critical path, and cross-crate sequencing.

---

## Engine purpose (load-bearing — read first)

### The ultimate aim

> **Serve the largest models at blazing speed on consumer hardware, with as little GPU as possible — ideally eventually none.**

Frontier-scale models (100B–1T+ params) are physically incompatible with
consumer hardware under naïve dense matmul: a 671B Q4 model touches
~336 GB per forward pass; consumer DDR5 is ~50 GB/s; that's 6.7 sec/token.
The bandwidth wall cannot be beaten by faster compute. The *only* path
through is **touching fewer weights per token** — sparse retrieval over a
queryable weight database. Vindex was always for this.

Every invention in the codebase serves this aim:

| Invention | Role |
|---|---|
| Vindex (model-as-database) | Sparse access to weights, not dense matmul |
| LQL | Address language for sparse retrieval |
| WalkFfn (gate KNN → down lookup) | The actual sparse-FFN inference path |
| MoE expert grid (gRPC self-assembling) | Distribute models that exceed one machine across consumer machines |
| Layer sharding (`--layers`, `--shards`) | Same, by layer |
| Exp 26 (FP4 native-friendly) | 2× memory shrink without QAT (Gemma 3 4B proven) |
| Exp 27 (hash routing top-2048 mask) | 5× fewer FFN weights at KL=0.03 *at L0* — but **does NOT compound across depth (V1 FALSIFIED 2026-05-31)** |
| MEMIT / COMPOSE / AOT | Compile programs into smaller weight footprints |
| WASM-in-FFN | Replace heavy kernels with cheap programs where the math allows |
| Boundary refs / residual codec | Compress KV for long context on bandwidth-bound hardware |
| Shannon arc (1 bit/char on Frankenstein) | Theoretical compression ceiling — how far this can go |
| Mech-interp surface (M1–M8) | Discover *which* weights actually do the work; rest stays on disk |
| Cross-arch coverage | The technique stack must generalise |
| Multi-modal (vision / audio) | Accept images + audio alongside text; same sparse-retrieval story applies to the LM portion of multimodal models. **Phase 0+1 shipped** (PR #143, 2026-05-24): trait surface + Gemma 3 SigLIP + CLI `--image`. **Phase 2 shipped** (PR #144, 2026-05-25): Granite Vision SigLIP2 + MLP GELU connector + AnyRes tiling + PerTile splice stress test. Phases 3–6 (interleaving, Qwen-VL M-RoPE, audio, Llama 3.2 cross-attention) remain design-only — see `docs/multi-modal.md`. |
| KV engine trait split (KvEngine / RetrievalEngine / AnyEngine) | Uniform dispatch across production KV-cache engines + retrieval-only engines (Apollo) via typed enum |

Combined effect (rough math, ORIGINAL projection): hash routing 5× × FP4 2× × KV
compression 10× = **100× effective bandwidth reduction** on the right
corpus. **Revised 2026-05-31: hash-routing 5× FALSIFIED (V1, doesn't compound); FP4 2× confirmed (V2). The compound is smaller — see the achievability table + `docs/diagnoses/`.**
670 GB model → 6.7 GB-equivalent traffic → ~134 ms/token on
consumer DDR5. That's blazing.

### Two permanent tracks

The aim demands both competitive performance *now* and progress toward
GPU-free *eventually*. These are co-equal tracks, neither sacrifices to
the other:

1. **GPU track** — maintains competitive baseline against ollama / vLLM /
   llama.cpp on Metal (and eventually CUDA/ROCm if substrate-relevant
   experiments demand them). Permanent. Never demoted in favour of CPU
   work. Without this, every claim measured on the engine fails the
   credibility threshold below.

2. **CPU track** — drives toward "blazing big models on consumer hardware
   without GPU." The ultimate aim. Built **in addition to**, not instead
   of, the GPU track.

**Architecture rule that makes the dual-track tractable**: vindex /
WalkFfn / sparse retrieval is the shared invention. Only kernels differ.
No GPU-only paths in the core design. Every technique developed on one
track must have a path to the other, or be architected
device-agnostically from the start (the verify-loop in MTP2 is a current
example: device-agnostic decode with device-specific kernels under it).

### Why "research substrate" framing is the means, not the end

LARQL **is** a research substrate — but substrate-for-its-own-sake isn't
the goal. The substrate exists because the techniques that make the
ultimate aim possible (sparse retrieval, hash routing, FP4, KV
compression, expert sharding, AOT compilation, boundary refs) have to
be developed *somewhere*. LARQL is that somewhere.

This means:

- Adoption, OpenAI-API ergonomics, multi-tenant batched serving, MCP
  ergonomics, and other "production engine" concerns are out of scope
  **except** where they accelerate experiments or affect measurement
  credibility.
- LARQL is not a production inference engine and will not become one in
  the *commercial* sense. But it must operate at production-engine
  baseline performance on its leading device class — otherwise the
  techniques developed on it can't be credibly compared against
  state-of-the-art.

### Achievability (honest assessment 2026-05-09)

The aim is **conditionally achievable, asymmetric across model class**. The
arithmetic decides — let's run it.

**MoE frontier models (671B with ~37B active, DeepSeek-V3 class)**

Active params per token = ~37B. At Q4 = 18.5 GB touched. Consumer DDR5 = 50 GB/s
→ **370 ms/token = 2.7 tok/s, just from MoE sparsity alone**.

Stack the techniques:

| Stage | Bytes touched/token | tok/s on 50 GB/s DDR5 |
|---|---|---|
| Naïve dense over active experts | 18.5 GB | 2.7 |
| ~~+ hash-routed FFN within active experts (5×)~~ **FALSIFIED (V1) — doesn't compound** | — | — |
| + FP4 (2×, **confirmed V2**) | 9.3 GB | ~5.4 |
| + KV compression on long context | depends | further win |

**This is where the field is going** — DeepSeek-V3, Llama 4 Maverick,
Gemma 4 26B-A4B, GPT-OSS family are all MoE. The aim hits this case.

**Dense frontier models (hypothetical 2T dense)**

1 TB at Q4. ~~Hash routing 5× → 200 GB.~~ (hash-routing 5× FALSIFIED, V1). FP4 2× → 500 GB. At 50 GB/s →
**10 sec/token. Not blazing** (worse than the original 2 sec/token estimate once the
hash-routing multiplier is removed). Would need attention sparsification too, open research.

**>RAM models (e.g. 671B = 336 GB on 64 GB consumer)**

NVMe-resident vindex via mmap. Hash routing makes access sparse-but-
predictable; MoE routing has cross-token locality. Keep hot experts in
RAM, page rare ones from disk. **Untested at scale; this is the riskiest
single piece.**

**Distributed across consumer machines (C9 territory)**

Per-token cross-node bandwidth for expert-grid: ~256 KB at frontier MoE
scale. 1 GbE carries 488 tok/s of network capacity. **Network is not
the bottleneck.**

#### Tier-by-tier confidence

| Acceptance tier (from "P0 — CPU path to blazing") | Confidence | Driver |
|---|---|---|
| Short-term: Gemma 3 4B CPU within 10% of `llama.cpp -ngl 0` | **~95%** | Pure engineering |
| Medium-term: Gemma 4 26B-A4B at ≥10 tok/s on 64 GB consumer, no GPU | **~85%** (was ~80% → 70% → 62% → 70% → 75% → 85%, revised 2026-06-13: CAUGHT llama.cpp on 26B CPU MoE) | MoE active-param math works; 26B fits 64 GB (16 GB vindex). **C10 gate resolved favorably (2026-06-10):** llama.cpp-on-26B-CPU = 32 tok/s, the ≥10 target is 3× below a mature engine's proof. The gap was **byte traffic, not kernel quality** (in-process streamed ~10 GB/token f32-resident vs llama.cpp's ~2.1 GB all-quantized). **Quantized residency (2026-06-11): 7.6 → 13.9 → 15.9; int8 attn → 21.7; KV append-in-place → 27.9.** **Spin-barrier pool (2026-06-13): → ~35 tok/s — CAUGHT/EXCEEDED llama.cpp (32.1, ~9% ahead), shipped DEFAULT-ON.** The final ~1.15× was **rayon fork-join overhead** (decode driver ran outside the pool → ~211 cold-path sections/token, ~40% of thread-time parked), *not* kernel quality — exactly what the C12 roofline-crossover entry called ("target effective-bandwidth sinks — rayon fork-join gaps"); the pool closed it via scheduling. Since larql now **matches the mature reference on the same box**, any 64 GB-consumer class where llama.cpp clears 10 (all of them) clears it too. Held at 85 (not higher) only for the unmeasured M-Pro/x86 bandwidth classes + the 26B llama.cpp anchor being recorded-not-same-session. Artifact `bench/baselines/c10_gemma4-26b-a4b_cpu_reconciled.json`. |
| Long-term: 100B-class MoE at ≥5 tok/s, no GPU | **~52%** (was ~60% → 55% → 52%, revised 2026-05-31) | Four-way push: 100B@FP4 (~25–50 GB) **fits RAM** so the disk bet is moot here — *removes* a risk the original 60% priced (+); FP4 confirmed (+); lost hash multiplier makes ≥5 tok/s harder (−); and the exploitable-structure prior took a **two-probe hit** — V1 (FFN-feature sparsity doesn't compound) *and* routing locality (expert selection doesn't concentrate, ~124/128 over a sequence) both say there's less cacheable structure than the "weights-as-database" thesis assumed (−, soft but broad). The disk-risk *removal* is what keeps it off 50; **50 is the honest alternative if you weight the two-probe pattern over it.** Caveat: the uniformity is partly Gemma's load-balancing aux loss (trained-in) → may be router-specific; the cross-MoE-router check would settle 50-vs-55. |
| Ultimate: 671B-class via multi-machine grid | **~30%** (was ~40%, revised 2026-05-31) | Hit hardest. 671B even at FP4 (~335 GB) **exceeds single-machine RAM**, and the MoE-routing-locality finding (working set ≈ whole expert population, no cacheable hot subset) **closes the single-machine disk-resident escape hatch** — it would thrash. That leaves only the harder multi-machine grid (C9, demoted to P2 per ADR-019), where integration risk dominates. |
| Dense frontier (if the field stays dense at 1T+) | **~10%** (was ~15%, revised 2026-05-31) | The hash-routing 5× its arithmetic leaned on is FALSIFIED (1 TB Q4 → ~10 s/token now, not 2). Needs attention-sparsification breakthroughs outside engineering control. |

#### What could kill the aim

The "100× combined effect" assumes the techniques compound multiplicatively.
ADR-015 ("isolated kernel speedup ≠ end-to-end win") says they often don't —
and D-RMS-FUSE Phase 1 (2026-05-09) gave us a concrete falsification: predicted
~0.2 ms/tok savings collapsed to zero. So we already have one data point that
compounds *don't* always materialise. The honest path requires **falsifying the
remaining assumptions early**, before committing years to a build that rests on
them. See **"P0 — Aim-validation tests (V1–V4)"** below — these gate the
medium/long/ultimate tiers and are the highest-leverage move available
right now. Four load-bearing assumptions, four tests (V1–V3 isolated, V4
compound).

#### Known unknowns (added 2026-05-09)

The bandwidth math above assumes the architecture cooperates with sparse
retrieval. Several open questions could shift the achievability
boundaries — listed here so they don't stay silent:

| # | Unknown | Status | Where it bites |
|---|---------|--------|----------------|
| KU1 | Static-attention fraction at 31B-scale | Untested. Validated at 4B (91.7% static heads on Gemma 3 4B). | If static fraction degrades with scale, "I Killed Attention" video weakens, MTP acceptance rate also degrades, attention-replacement timeline pushes out. |
| KU2 | Softmax bottleneck phase transition above ~1,142-token RoPE distance | Characterised (Q-side drift fixable, KV-side drift at last position not, with current architecture). Not solved. | Caps long-context reliability. BR4 (boundary refs Phase 4) is the workaround; doesn't fix the underlying bottleneck. |
| KU3 | FP4 friendliness across non-Gemma archs | **RESOLVED 2026-05-31 — CONFIRMED.** V2 measured original f16 weights on Gemma 3 4B + Granite 4.1 3B + 8B (2 families, scale ladder): ≥99.8% per-feature R<16 (reproduces exp 26's 99.83% on gemma3 down exactly; `down` the only mild tail). Predictive: FP4 E2M1 within +0.116 bits/token of f32 and *beats* the shipped Q4-int baseline. See [`docs/diagnoses/v2-fp4-generality.md`](docs/diagnoses/v2-fp4-generality.md). | V2 resolved it: FP4 is a real free ~2×, no per-arch QAT for the families measured. Llama/Mistral/MoE-expert weights not yet covered (need f16 exports). |
| KU4 | Hash-routing compounding across all layers | **RESOLVED 2026-05-31 (dense) — FALSIFIED.** V1 measured 3 dense archs (Gemma 3 4B, Llama 2 7B, Mistral 7B): per-layer KL ≤ 0.05 thresholds (mean 2.7–12.2% of features) **do not compound** — applied simultaneously they give +5.4 to +7.7 bits/token NLL and 78–95% argmax drift. The per-layer screen is anti-correlated with the truth (sparser screen → worse collapse). Realisable bandwidth ~2.4–2.9× (not 5×) and catastrophic anyway. **MoE-within-expert version still OPEN** (the dense harness measures the wrong object on the 26B). See [`docs/diagnoses/v1-hash-routing.md`](docs/diagnoses/v1-hash-routing.md). | V1 resolved the dense case. The 5× *within-FFN* bandwidth multiplier is gone; MoE confidence now rests on expert active-param sparsity, not FFN hash routing. |
| KU5 | mmap thrash on disk-resident frontier models | **RESOLVED 2026-05-31 — locality is POOR (negative for the long-term tier).** Two halves: (a) V3 cold-read latency — cold scattered 16 KB read ~100µs p50/140µs p99, warm ~0.04µs (~2380×); (b) MoE routing locality (faithful in-process 26B-A4B decode): per-token routing is sparse (8/128) but the **working set saturates to ~124/128 experts over a sequence — the uniform-random expectation** (load-balanced router), so there is NO small cacheable hot subset. See [`docs/diagnoses/v3-disk-resident-mmap.md`](docs/diagnoses/v3-disk-resident-mmap.md) + [`docs/diagnoses/moe-routing-locality.md`](docs/diagnoses/moe-routing-locality.md). | 26B's full expert set (~11 GB) fits RAM → fine after warmup. But a **>RAM frontier MoE can't keep a hot fraction resident** (working set ≈ whole population) → sustained paging (~200 ms/token-class). The disk-residency bet for the long-term tier is **undermined**. Cross-MoE generality (non-Gemma router) still open. |

KU3, KU4, KU5 are scheduled to be resolved by V1–V3 below. KU1 and KU2 are
not currently scheduled — KU1 lands when 31B work matures enough to measure
head staticity; KU2 is parked behind BR4 because the workaround is on the
roadmap even if the underlying fix isn't.

### Baseline-credibility threshold (acceptance criterion)

> LARQL must be within **10% of llama.cpp / ollama** on the matching
> model + quantisation + context-length configuration **on the device
> class the claim is being made on**, before any *"+N% from technique X"*
> claim is published. CPU technique → CPU baseline. GPU technique → GPU
> baseline.

Current state (2026-05-15):

| Track | Configuration | LARQL | State-of-the-art | Gap | Threshold? |
|---|---|---|---|---|---|
| **GPU (Metal)** | Gemma 3 4B decode | 88 tok/s | ollama ~103 | 17% behind | over (defensible-with-caveat) |
| **GPU (Metal)** | Gemma 3 4B prefill (340 tok) | per-pos matvec | gemm | 14× behind | far over |
| **GPU (Metal)** | Gemma 4 + MTP (when adopted) | 88 tok/s no-MTP | ~225 with MTP | ~2.6× behind | far over |
| **CPU** | Gemma 3 4B Q4K decode | **30.9 tok/s** (residency default-on, same-session 2026-06-13; was 24.5) | llama.cpp Q4_K_M CPU ~43 | **~1.42× behind** (was 1.69×) | over — the Q4_K residency + int8 + asm + spin-pool stack is now **default-on** (2026-06-13); earlier kernels (KV-cache, direct Q4_K matvec, NEON Q4_K/Q6_K/f32_dot, Q4 lm_head, par_chunks_mut(32), Q4_K×Q8_K sdot, auto-t=8) landed 2026-05-15/16, ~86× over the original 0.36 baseline. See `bench/baselines/cpu/DIAGNOSIS.md` |
| **CPU** | Gemma 4 26B-A4B decode | in-proc KV-cached MoE **~35 tok/s** (spin pool, default-on; M3 Max t=8 warm n=256, 2026-06-13) | llama.cpp Q4_K_M CPU **32.1** (recorded, drift-bracketed) | **larql ~9% AHEAD** | ✅ **CAUGHT** — arc 7.6 → 13.9 (residency) → 21.7 (int8 attn) → 27.9 (KV append-in-place) → **~35 (spin-barrier pool)**. The final ~1.15× was **rayon fork-join overhead** (decode driver ran outside the pool → ~211 cold-path sections/token, ~40% of thread-time in waits), *not* kernel quality — closed by the spin pool (effective-bandwidth/scheduling, exactly as the C12 roofline-crossover entry predicted), shipped **default-on**. Caveat: the 26B llama.cpp anchor is the recorded 32.1 (ollama wouldn't run the HF GGUF on CPU this session); machine validated via 4B llama.cpp 44 ≈ recorded 43. `bench/baselines/c10_gemma4-26b-a4b_cpu_reconciled.json`. |
| **CPU** | Gemma 3 4B Q4K **prefill** (5-tok) | **233 ms** (q4k/q6k-direct attn+FFN + NEON dot; standard engine, 2026-06-22; was 2746 ms / ~2 tok/s) | llama.cpp pp5 ~70 ms | **~3.3× behind** (was 55×) | closing — eliminated the per-layer f32 dequant: Q/K/V/O **and** gate/up/down project straight from the Q4_K/Q6_K vindex bytes via amortised `q4k_matmul` / `q6k_matmul` (the Q6_K twin, for the default Q6_K `v_proj`/`down_proj`) with a hand-written aarch64 NEON inner dot that *beats* f32 AMX sgemm at seq=5. A Q6_K-`down_proj` mis-decode (format-tag dispatch) was caught in review and fixed before this number. Remaining gap is matmul constant-factor + batched attention, not dequant. Also de-duplicated the larql-inference↔larql-compute Q4_K forward (one substrate copy). `bench/baselines/cpu/COMPARISON.md` |

Items the threshold makes load-bearing (not optional) on the **GPU track**:
- **D-ATTN-MTG** — flash attention; without it, attention-mechanism deltas are muddied by missing baseline.
- **D-PREFILL-MM2** — `simdgroup_matrix` matmul; until landed, prefill claims fail the threshold.
- **D-METAL-PLE** — without it, every Gemma 4 E2B experiment runs CPU-fallback and any delta is unattributable.
- **MTP1–MTP6** — Gemma 4 MTP drafters are now part of the state-of-the-art baseline (Ollama supports them).
- **AI1–AI6** — cross-arch deltas need clean arch boundaries.
- **Coverage → 90%** — measurement integrity needs correctness trust.

Items the threshold makes load-bearing on the **CPU track** (see new
"P0 — CPU path to blazing" section below):
- Critical-path #4 — CPU MoE forward pass.
- WalkFfn as primary CPU decode path.
- Hash-routed FFN (exp 27 → product).
- FP4 productisation (exp 26 → product).
- mmap'd vindex with lazy disk-resident edges.
- AMX / AVX-512 / Apple AMX kernels.
- KV compression as default for long context.
- BR4 (boundary refs Phase 4).

Items the threshold makes **explicitly out of scope** (both tracks):
- **CB1, CB2** (continuous batching, PagedAttention) — concurrency-throughput, not single-stream baseline.
- **MCP1** (MCP server) — UX, doesn't change measurement.
- **TM1** (thinking-mode toggle) — UX, doesn't change measurement.
- OpenAI API compatibility beyond what experiments call.

See `docs/positioning.md` for the full framing and competitor diff.

---

## Strategic priorities (review 2026-05-28)

Layered on the achievability analysis above after the 2026-05-28 whole-codebase
review. These are **organizational / sequencing** decisions — they re-prioritise
existing roadmap items, they do not replace ADR-019 or the V1–V4 design.

**1. One gated critical path. V1–V4 is the only true P0.**
The medium/long/ultimate tiers (62 / 52 / 30%, revised 2026-05-31) are *conditional* on the compound
assumption, and we already hold two falsifications of compounds not materialising
(ADR-015, D-RMS-FUSE → 0). So everything currently labelled P0 except
aim-validation is downgraded to **"P0-conditional — unblocked by V1–V4"**:
Engine↔Backend unification, the CPU-path-to-blazing build-out, and the
best-in-class mech-interp engine. They stay important; they are not *first*.
Rationale: when seven sections are P0, the falsification gate competes with a
6–12 month refactor and loses. (ROADMAP_STATUS.md's single ordered Active
Sequence is the canonical "what's now"; this section makes the main roadmap
agree with it.)

**2. Pull a minimal V3 (disk-resident mmap) spike forward, in parallel with V1.**
V3 tests KU5 (mmap thrash on >RAM models) — named above as "the riskiest single
piece" and the gate for the long-term + ultimate tiers. It is currently queued
*behind* V1/V2, which is backwards on information value: V3 is the most likely to
fail and reshapes the most plan if it does. A throwaway spike (≥70B-class MoE
vindex on NVMe; measure page-fault rate under MoE routing locality on a single
decode stream) is worth more than a clean V1, because "models that exceed RAM"
*is* the frontier-on-consumer story. A negative result shrinks the aim to
"models that fit in RAM" — which we want to know *before* the backend rewrite,
not after.

**3. GPU track = credibility tax. Spend the minimum to stay "defensible".**
GPU's job is the baseline-credibility threshold, not parity — and it is a
treadmill (ollama shipping MTP widened the Gemma-4 gap 1.17× → 2.6× through no
change of ours). Of the load-bearing GPU items, **D-PREFILL-MM2 (the 14× prefill
gap) is the only one that actually invalidates published claims today** — any
prefill-sensitive measurement fails the threshold until it lands. Prioritise it
over further decode tok/s. Treat MTP1–6 as baseline-*matching* (don't innovate
there). D-ATTN-MTG / D-METAL-PLE stay load-bearing but sit behind D-PREFILL-MM2.

**4. MoE-first functionality; dense is for experiment velocity, not the destination.**
The sharpest fact in the achievability table is the MoE/dense asymmetry (62% vs
10%, revised 2026-05-31) and the field is all-MoE (DeepSeek-V3, Llama 4, Gemma 4, GPT-OSS). The
crown-jewel functionality is therefore **CPU MoE forward + hash-routed FFN +
disk-resident expert paging** — the three things that prove the 80% / 60% tiers.
ADR-019 making dense-31B substrate-primary is fine for velocity, but the
functionality emphasis must stay MoE-first: watch that the dense path doesn't
accrete features while the MoE path (the actual bet) stays thin.

**5. Deepen the database surface — it's the moat (see next section).**

---

## VINDEX3 — successor serving container (added 2026-08-02)

**Thesis: the format boundary is the place to make sparse serving predictable.**
VINDEX2 can *observe* which pages faulted; VINDEX3 can *state* what an operation
will read before it runs. That is the difference between paging a multi-terabyte
model and planning one.

Spec: [`crates/larql-vindex/docs/vindex3-format-spec.md`](crates/larql-vindex/docs/vindex3-format-spec.md)
(draft-2). Experimental programme: [`docs/vindex3-experiments.md`](docs/vindex3-experiments.md),
registry programme `vindex2`. Generations are named so the number equals
`index.json.version`: schemas 1–2 → VINDEX2, schema 3 → VINDEX3.

**Coexistence, not migration.** One binary serves both generations, dispatched
solely on `index.json.version`. VINDEX2 keeps its loader, its weight objects and
its production behaviour untouched; VINDEX3 keeps its catalogue, profile, route
and authority model until binding. The shared layer is execution and
orchestration, never physical storage. **`extract` must keep defaulting to
VINDEX2** until V2-1 acceptance passes — a silent default change would evaporate
E0's premise.

### Shipped

| Commit | Milestone |
|---|---|
| `f13bf385` | Reference MoE execution — fixture A matches an independent oracle below 1e-6, fused and decomposed agreeing at every checkpoint |
| `dd2017db` | Real Gemma semantic routing parity over real VINDEX2 bytes |
| `f5dd256e` | Production router kernel bound — bit-identical routing ladder |
| *(pending)* | **The container itself** — fixture A written to disk as `index.json` schema 3 + `moe_manifest.json` + a LYRW v2 bank, opened, bound and executed bit-identically; `show`/`verify` dispatch on generation |

Three properties established, each independently useful:

1. **Bound reference execution is numerically correct** (fixture A vs oracle).
2. **Resolution does not leak into decode** — 64× population costs ~1.9×, which
   is the router term and nothing more.
3. **The bound plan predicts its physical page working set exactly** — 200 pages
   predicted, 200 resident, zero overshoot, 1.63% of a 192 MiB layer after one
   token. Residency becomes computable rather than observable, which is what
   placement, prefetch and remote transfer all need.

Plus two defects fixed that were not VINDEX3's: `larql verify` rendered findings
in `HashMap` order and so disagreed with itself between runs; the separate-tensor
MoE extractor wrote no expert store.

### Two ladders, deliberately separate

VINDEX3 has an **execution** half and a **container** half, and they were built
in that order. Every parity result before the container existed bound its
operands out of a VINDEX2 file — so what was proven was the executor, not the
format:

```text
proven first   the VINDEX3 executor, fed VINDEX2 operands, matches production
proven second  a VINDEX3 container can be written, opened, bound and executed
```

Keeping the ladders apart matters because a green execution ladder says nothing
about whether a VINDEX3 *file* exists, and for a long time none did.

```text
container ladder
[x] c0  write index.json schema 3 + moe_manifest.json + LYRW v2 bank
[x] c1  detect_generation reports V3 from a real directory, not a JSON literal
[x] c2  open, validate the manifest, resolve storage keys to files
[x] c3  bind from container-resolved regions and execute — bit-identical
[x] c4  fused and decomposed storage agree under one programme id
[x] c5  structural verify with {layer, entry, role} defects; CLI dual-generation
[ ] c6  fixtures B–D (GPT-OSS, Inkling, Mini-K3) — proves nothing is hard-coded
[ ] c7  WALK/DESCRIBE parity over in-place bank regions
[x] c8  a real Gemma MoE layer written as a VINDEX3 container (CLOSED 2026-08-04, below)
[ ] c9  every Gemma layer — the first real model that *is* a VINDEX3 container
```

Gate status, stated precisely: **V2-0 and V2-1 are closed for the rows fixture
A can carry**, not in full. Outstanding on V2-0 are profile-authority
derivation and variant-selection refusal; on V2-1, the "not hard-coded" row
(needs fixtures differing on expert count, top-K and shared banks) and
WALK/DESCRIBE parity. `extract` therefore still defaults to VINDEX2, and must
until those close.

### The rung ladder to the first VINDEX3 Gemma token

```text
[x] rung 0   fixture A through the generic reference path
[x] rung 0.5 real Gemma routing parity, VINDEX3 bound over VINDEX2 bytes
[x] rung 1   production router kernel bound, bit-identical
[ ] rung 2   production Q4_K x Q8_K expert kernel bound
[ ] rung 3   full-layer residual delta parity
[ ] rung 4   every MoE layer, then final logits
[ ] rung 5   greedy token parity through normal `larql run` dispatch
```

Rung 2 begins with **Q8_K activation identity, checked before any expert runs** —
a difference there contaminates all eight expert comparisons and makes every
later diagnostic noise.

### What's next (set 2026-08-02, after PR #197)

Ordered by what unblocks what, not by size. Each item states the condition that
closes it, so "done" is not a judgement call.

**1. `layer_ffn_or_moe` — the other five engines. CLOSED 2026-08-02.**
`layer_ffn_or_moe` returns `Result<Array2<f32>, BoxRefusal>`, all ten call
sites propagate, and the gate in `larql-kv/tests/strict_refusal/engines.rs`
runs **eight** expert-routing engines × prefill/decode × three `RefusalKind`s.
The baseline tag may now say engine-wide.

The rewind half came out better than the prediction. The prediction was that
residual-canonical engines could rewind where K/V-canonical ones could not; the
answer is that **all of them can**, by two different mechanisms, and
`engine_state.rs` proves it in the strong form — the retried token is
bit-identical to never having refused, for every one of the eight:

```text
residual-canonical   markov-rs, markov-rs-codec, boundary-per-layer
                     → the step writes `stored` only after the last fallible
                       call; `hot_kv` is a droppable derivative, taken up
                       front and left None on the error path
K/V-canonical        standard, turbo-quant, windowed-checkpoint
                     → the cache grows before the FFN can refuse, so the step
                       truncates its appends: `truncate_kv` on the handle,
                       `CompressedLayer::truncate_rows` byte-exactly (rows are
                       appended at fixed offsets and never re-encoded, so the
                       codec is lossy against its *input*, not against what
                       was stored), `truncate_kv_rows` on the window shadow
```

Two engines stopped taking their store by value (`store.take()` → `as_mut()`),
which also removed a latent bug: any failure used to leave `self.store` as
`None`, so the *next* call reported "decode_step called before prefill" — a
dead engine wearing a misleading message.

Exactly one case genuinely cannot rewind, and now says so:
`windowed-checkpoint` archives a window and saves its boundary checkpoint when
the window fills, so a refusal *after* a close returns
`EngineError::StateInvalidated` rather than a retryable refusal. Same for a
`standard` window already at capacity — append-then-evict leaves the row count
unchanged while the oldest row is gone.

**1b. `no-cache` and `apollo` — the dense-only forwards. CLOSED 2026-08-02.**
Found while closing item 1: neither consulted `forward_moe_full_layer` at all,
so on a hybrid-MoE arch both ran the dense half of every layer and returned an
apparently valid answer. That is a *worse* failure than a degraded one — a
different model wearing the same answer shape, undetectable downstream — so it
was treated as a semantic disqualification rather than as missing propagation.
The two needed different corrections because the seam is in a different place:

```text
no-cache   forwards through `kv_prefill_run`, which *takes* an FfnBackend
           → gave it real dispatch. That helper is also the oracle the
             dispatch ring is compared against, so an oracle that skipped the
             expert half would have made every MoE parity comparison agree
             about the wrong answer.
apollo     forwards through `forward_from_layer` / `forward_raw_logits`, which
           live in larql-compute *below* the FfnBackend seam and construct
           their own dense `ViewFfn` — no caller-supplied backend can reach
           them → refuses the architecture up front, `RefusalKind::Unsupported`
             (operands fine, this executor cannot serve them, pick another).
```

Real dispatch stays preferable for apollo, and means threading an `FfnBackend`
through `forward_layer_range` — a change to the forward, not to the engine.
Until then it is not usable as an apparently conformant MoE engine, which is
the point.

Two more transactional bugs fell out, both of the kind only a refusal can
expose. `no-cache` pushed the decode token onto its list *before* the
re-forward could refuse, so a caller who fixed the cause and retried would
have forwarded the same token twice — the exact double-append the contract
exists to prevent; the token list is its entire continuation state, so the
push is now undone on failure. And `kv_decode_step_run` appended each layer's
K/V before the FFN could refuse, so the oracle itself is now transactional:
truncate back to the entry lengths, or report `StateInvalidated` when the
cache is windowed at capacity and eviction has already discarded a row.

*Standing:* every `EngineKind` variant is now classified and gated —
`RoutesExperts` (nine, sweeping prefill/decode × three kinds) or
`NoExpertSeam` (apollo, refusing the architecture with an executing route, so
the refusal provably comes from the engine and not the route).

**2. Variant-selection refusal. CLOSED 2026-08-04.** `index.variants`
catalogues each region set's present variants and its baseline; a `Profile`
selects per region set, and `select_profile` refuses an absent one naming the
region set, the request and what is present. `Vindex3Container::open` resolves
**every** declared profile between the index parse and the segment reads —
pinned by deleting the segment files and asserting the error still names the
variant, since a late gate would name a missing file instead.
`declares_profile` stays, documented as a name check only.

*Ceiling:* the refusal is real; **steering is not exercised end to end.** No
writer emits a multi-variant container yet (`ContainerSpec` has no variant
field) and `BankRef.storage` still names storage directly, so a selection does
not yet change which bytes the runtime binds. That wiring belongs with the
first real pack — the natural companion to item 4.

**3. WALK/DESCRIBE parity.** Closes V2-1 except shared banks. Gate KNN over
in-place bank regions must return identical top-K to a v1-style extracted
`gate_vectors.bin` control on fixture A. This is the row that keeps "the model
IS the database" true of VINDEX3 rather than only of VINDEX2.

**4. A real Gemma layer as a VINDEX3 container** (container ladder c8/c9).
**c8 CLOSED 2026-08-04**, c9 open. `format/vindex3/import.rs` imports one real
MoE layer verbatim — no transcode, no requantise, no repack — and
`examples/vindex3_import_gemma_layer.rs` drives it end to end.

Measured on `gemma4-26b-a4b.vindex` (the same index the parity rungs use),
layer 0: hidden 2816, **128 experts, top-8, intermediate 704 semantic over 768
stored**, written as a 421 MB VINDEX3 container that reopens from disk,
verifies with no structural defects, and returns **256 of 256 regions
byte-identical** to the VINDEX2 source. `larql show` reports it as
`VINDEX3 (index.json schema 3) ... bindable (no defects)`.

*Ceiling.* This licenses "these regions survive the round trip unchanged and
the container is bindable". It does **not** license "Gemma runs from VINDEX3":
the execution comparison is `vindex3_gemma_layer_parity`'s and still runs over
VINDEX2 bytes — what c8 adds is that those are demonstrably the same bytes.

c9 is all layers, at which point `extract` gaining a VINDEX3 mode becomes a
question rather than a violation. `extract` writes VINDEX2 today and has no V3
path at all (§12.1 gates the flip on the ABI freezing *and* the E0
preservation matrix passing).

**Not on the critical path, but adjacent and cheap to start: the continuation-
state intervention harness.** `larql-kv` already owns incremental decode with
real K/V continuity, explicit next-token forcing, and a state-policy taxonomy
that names exactly the question a persistence experiment asks — which parts of
a continuation are carried by the emitted token, the residual, and the K/V
history. What is missing is causal read/write access to that state during a
decode step.

PR #197 set the precedent for how that should look. `KvDispatch::truncate_kv`
is a research/recovery capability declared on the trait, implemented on CPU,
defaulting to *unsupported* rather than silently copying to host — which is the
shape a `MutableKvView` / `KvIntervention` seam should follow, at the layer
where attention appends and reads, never by exposing `KvHandle`'s
representation.

One trap is already known and should be inherited rather than rediscovered: a
checkpoint that records cache *lengths* is not a checkpoint under a sliding
window, because append-then-evict leaves the count unchanged while the oldest
row is gone. `StandardEngine::rewind_is_sound` encodes that test; a fork API
needs the same one or it will hand out silently wrong donor state under
`markov-bounded`.

### Standing method

Established by repeated failure, not preference:

- **Bind, never reconstruct.** A bridge that dequantises into an
  incumbent-shaped temporary can reach numerical parity while proving nothing
  about the binding architecture. `as_f32_slice()` hands over stored bytes or
  refuses with a typed reason.
- **Ladders, not end-to-end tolerances.** A single residual-delta tolerance
  blends router accumulation order, softmax, renormalisation, activation
  quantisation, integer rounding and reduction order; passing it establishes
  nothing and failing it identifies nothing. The router ladder localised a
  7e-4 disagreement to post-processing in one run — it was a missing bound
  operand, not the BLAS-vs-index-order accumulation it would have been blamed on.
- **Mutation-check every new test.** Several have passed for the wrong reason,
  including one that could never have caught the bug it was named for.
- **Suspect the instrument first.** This programme has produced roughly three
  measurement defects per real code defect: a process-global allocation counter
  under a parallel test runner, replay parameters defaulted instead of read from
  the record, and a `\b` in a normaliser that BSD `sed` silently ignores.

### Not discharged

- **E0-FULL.** Decode rows and all 632 WALK ranking lines match; the remaining
  12 rows need the prescribed baseline reconstruction (baseline binary at
  `6eae5ea` → baseline extraction → current reader against that artifact).
  Status stands at: E0-CI green, E0-FULL decode rows green, remaining rows not
  discharged. A runner now exists (`scripts/e0-verify-goldens.sh`); before it,
  the goldens were an assertion nobody made.
- **OLMoE goldens** pin a decode panic that the separate-tensor MoE extractor fix
  has since removed. They need a deliberate re-capture with the reason recorded.
- **CLI generation dispatch** — `detect_generation` exists and is guarded by
  E0-CI, but 15 call sites still assume VINDEX2.
- **`extract --format vindex3`** — not needed until the container round-trip
  rung, and doing it earlier would weaken the Gemma comparison by introducing
  re-extraction as a second candidate cause.
- **Mini-K3, Kimi-Linear, K3** — the conformance envelope beyond Gemma.

---
## Query / Edit / Interpret — first-class functionality track (added 2026-05-28)

**Thesis: the differentiated functionality is the database, not the tok/s.**

The performance race against ollama / vLLM / llama.cpp is a *credibility*
exercise — they will always win raw speed because that is their entire job, and
the threshold only asks us to stay within 10%. But "query, edit, and interpret
the model like a graph database" — `DESCRIBE`, `INSERT INTO EDGES`, `walk`,
MEMIT / AOT compilation — is a genuine moat with **no competitor**. This is where
LARQL is *ahead* instead of chasing.

Until now this surface has been framed as a *means* to sparsity ("discover which
weights do the work, so the rest stays on disk"). That undersells it. Promote it
to a co-equal functionality track with its own exit criteria:

- **Harden the experiment surface into LQL verbs.** A large amount of the
  differentiated capability lives in `experiments/` rather than in shipped LQL:
  vindex compilation (10/10 retrieval), MEMIT fact insertion, AOT program
  compilation (zero-drift), passage compilation, two-level routing, the
  WASM-in-FFN / VM-in-residual primitives. These are product, not just papers.
  Sequence them into first-class, tested LQL / CLI verbs at the same coverage
  floor as the rest of the workspace.
- **Make edit durable + safe.** INSERT / COMPOSE / compile paths need the
  commit-semantics + truthfulness guarantees the interpretability-truthfulness
  P0 is already chasing (TRACE parity), so an edit is verifiable and reversible.
- **Lower risk than the compound.** This track does not depend on the 100×
  compound materialising. It compounds the one durable advantage regardless of
  whether V1–V4 confirm the bandwidth math — which makes it the right hedge to
  fund *alongside* aim-validation, not after it.

**Exit criterion:** the README's `INSERT` / `DESCRIBE` / `walk` / compile demo is
backed end-to-end by tested LQL verbs (not example scripts), with edits
verifiable via TRACE parity and reversible.

### FR — Fleet routing extensions (added 2026-06-07)

Four routing/edit explorations seeded by the `chris-experiments/fleet`
native-store arc (E10–E17) and the `videos/the-mechanism` build story — the
fleet and LARQL's KNN/COMPOSE converged on the same architecture
(`fleet/SYNTHESIS.md` §9). **Measurement-experiment-first:** each item runs its
falsification probe on a real LARQL vindex, in predictive units (recall@k / NLL /
KL / drift / confident-wrong — mean-P/mean-cosine banned), **before** any build;
builds land parity-first (default off = byte-identical). Full spec + frozen
pre-registrations: [`docs/fleet-routing-extensions.md`](docs/fleet-routing-extensions.md).

The mechanism: factual memory is addressed by **(relation, entity) → value**; the
relation is a *clean* semantic index, the entity is *top-k fuzzy*; the model
*addresses*, it does not *unpack*; and operations split at linear-aggregate (rides
free) vs joint-nonlinear (walls). FR1 ⊂ FR2 (top-k is the fuzzy tier of the
two-tier router). FR3 is the cleanest standalone win; FR4 is research-first.

| # | Item | Crate | Status |
|---|------|-------|--------|
| FR1 | **Top-k fuzzy entity router + verifier.** Inference routes on top-1 cosine + a fixed 0.75 gate (`infer_patched.rs:162-163`), the brittle near-rank-1 path E11/E15 indict; `query_knn` top-k exists (`knn_store.rs:132`) but is unused. **MEASURED ✅ (2026-06-07, Gemma-3-4B N=150):** entity key real & answer-leak-free at L24-26 (L26 top1 0.89/top5 0.95, cross-rel 1.00 — beats E15's MLP under cosine-NN, no training); the live 0.75 gate fires **150/150** with **11% confident-wrong @L26 / 84% @L20**. **BUILT ✅ (2026-06-07):** `apply_knn_override_verified` — top-k + entity-in-prompt verify + abstain, resolved-layer-first (no hardcoded layer), opt-in `LARQL_KNN_VERIFY`, default off = byte-identical (14 legacy tests green). E2E on real Gemma-3-4B: legacy "Germany's capital city is"→SpainX (confident-wrong) → verified→GermanyX (fixed), Poland correct both (no regression). 5 unit tests, clippy clean. **LQL surface landed:** first-class `INFER … ROUTE VERIFY [FALLBACK] [TOPK n]` clause (`KnnRouteMode` threaded through `infer_patched`, default Legacy = byte-identical; env vars set the default when no clause). E2E no-env: `ROUTE VERIFY` → Germany fixed. [`docs/diagnoses/fr1-topk-fuzzy-router.md`](docs/diagnoses/fr1-topk-fuzzy-router.md) §"BUILD LANDED". | larql-vindex, larql-inference, larql-lql | **built ✅ (LQL clause + env)** |
| FR2 | **Two-tier router: symbolic-primary → activation-fuzzy fallback** (E16 assembled). `entries_for_entity` exact lookup exists (`knn_store.rs:172`) but isn't sequenced into routing. **MEASURED ✅ (2026-06-07, Gemma-3-4B):** symbolic exact-match **0/10** aliases, activation fallback **10/10 top-1** @L24/L26 (Persia→Iran, …) — E16 reproduced. Caveat: famous-alias easy end (general = FR1's ~0.9 top-5); FR1 verifier bounds confident-wrong. **BUILT ✅ (2026-06-07):** `apply_knn_override_two_tier` (tier-1 FR1 verify → tier-2 activation alias fallback, opt-in `LARQL_KNN_VERIFY`+`LARQL_KNN_FALLBACK`, default off = byte-identical). E2E real Gemma-3-4B: "capital of Persia" → verify-only abstains (Tehran), two-tier recovers IranX (cos 0.97), no regression on named. 4 unit tests, clippy clean. Tier-2 is the fuzzy ~0.7-0.9 route (fires only when verify missed). **LQL:** `INFER … ROUTE VERIFY FALLBACK` (E2E no-env: Persia→IranX). [`docs/diagnoses/fr2-two-tier-router.md`](docs/diagnoses/fr2-two-tier-router.md). | larql-inference, larql-vindex, larql-lql | **built ✅ (LQL clause + env)** |
| FR3 | **Relation as a clean semantic address.** Relation probe generalizes to unseen synonyms at ~1.000 (`the-mechanism/address.py`); `RelationClassifier` (`relations.rs`) is the foundation. **MEASURED ✅ (2026-06-07, Gemma-3-4B N=40):** synonym-gen **1.00 at every layer L6-L26** (train {capital,currency,language} → classify unseen {seat,money,tongue,…}, semantic not lexical; clean from **L6**, earlier than the video's L10); asymmetry stark — relation 1.00 early vs entity top-1 0.07-0.20 until L26. **BUILT ✅ (2026-06-07):** `RelationResolver` — trained residual softmax probe (not string/cosine; the near-rank-1 "proxy" trap avoided), model-agnostic probe layer (`round(0.3·num_layers)`), wired into `SELECT … FROM EDGES WHERE relation=…` as a semantic fallback (cached per vindex). E2E real Gemma-3-4B: `WHERE relation="seat"` → resolved to "capital". 2 unit tests, 717 lql green, clippy clean. [`docs/diagnoses/fr3-relation-address.md`](docs/diagnoses/fr3-relation-address.md) §"BUILD LANDED". | larql-lql, larql-vindex | **built ✅ (SELECT)** |
| FR3b | **Explicit relation rewrite — phrasing-robust fallback.** FR3's probe is synonym-robust but **phrasing-brittle**: 1.00 was synonym *words* in one template; on an unseen *phrasing* it's at **chance** at its L10 probe layer, and more training templates = **no-op** (reverted). **MEASURED ✅ (2026-06-08, Gemma-3-4B):** explicit few-shot `word→relation` classify (1 forward, `predict_kquant`) = **12/12** synonyms + unseen phrasings (head city→capital, legal tender→currency, mother tongue→language — exactly the probe's chance cases), but forced-choice confident-wrongs distractors **2/3** (weather/altitude→capital) → add a `none` escape → **0/3** (all abstain), 12/12 kept. The `none` escape = the verify/abstain (the project's recurring confident-wrong trap, cf. FR1 gate). **BUILT ✅ (2026-06-09): probe-first / explicit-classify-with-`none` fallback** in `resolve_relation_synonym` (FR2 two-tier shape) — Tier 1 probe (cheap, on confidence) → Tier 2 `resolve_relation_explicit` on abstain (few-shot+`none` frame lifted from the harness; one full forward via `InferenceWeights::predict_dense` = the INFER path's `predict_kquant`+lm_head, since `RelationResolver` only dequantises `0..=L10`; `none`-gated `match_relation_top1`). Opt-in `LARQL_FR3_EXPLICIT`, default off = byte-identical. **Real-vindex fix:** prod vindex has 2890 noisy labels; alphabetical top-64 dropped `language`/kept `food_animal` (mother-tongue failed, banana resolved — backwards) → `RelationClassifier::relation_labels_ranked` (by feature count) for Tier 2 candidates. **E2E real Gemma-3-4B:** `mother tongue`→`language` by explicit (0.97, probe abstained — the win); `weather`→abstain (none-escape); default off → no resolution. Probe stronger than the ablation implied (`head city`/`legal tender`/`altitude` ride Tier 1). 4 new tests, 726 lql lib green, clippy clean. Harnesses `examples/fr3_{template_ablation,explicit_rewrite}.rs`; [`docs/diagnoses/fr3-explicit-rewrite.md`](docs/diagnoses/fr3-explicit-rewrite.md) §"BUILD LANDED". | larql-lql, larql-inference | **built ✅ (SELECT fallback + env)** |
| FR4 | **Operation-class dispatch boundary** (E17 compute ladder). Linear-aggregate ops (COUNT/THRESHOLD/MAJORITY) ride the read free; joint-bit (PARITY) walls — **a property of the operation, not the packing**. E17's own ledger demotes the E4 bridge to a **conjecture** (G/O/T never ran). Measure first = run the real external ops (distance/argmin/optimization) on the E17 rig to close that conjecture, then map LQL aggregate verbs. **MEASURED ✅ (2026-06-07, conjecture REFINED):** ran the real external ops on the E17 rig — **DIST (geometric) + ARGMIN (selection) RIDE free at L1**, only **PARTITION (global optimization) walls like parity**. Parity was NOT a fair stand-in for "external"; E4 mis-files geometric/selection (they're internal). Real line = factors-through-reads vs global-joint. Dispatch consequence: keep count/filter/aggregate/threshold/majority/distance/argmin internal, route global-optimization+parity external. `fleet/E17_compute_ladder/E17_EXTERNAL_VERDICT.md`. Build (far): in-band eval + external dispatch per the re-cut criterion. | larql-lql, larql-router, larql-vindex | **measured ✅ (conjecture refined)** |

---

## Video pipeline (added 2026-05-09)

The roadmap is not just engineering items; many of them are gated on
producing video evidence and many videos are gated on engineering items
landing. This section maps the dependencies explicitly so neither side
drifts.

(V-prefix reserved for aim-validation tests; videos use VID-prefix to
avoid collision.)

| # | Video | Status | Engineering dependencies | Roadmap items |
|---|-------|--------|--------------------------|---------------|
| VID1 | "The Model Is a Database" (Act 1 LARQL REPL demo) | Script v3 ready | Chat template + EOS, INSERT/PATCH wired in REPL, INFER compare mode in REPL | Critical path #1, T1, C1–C3 |
| VID2 | "There Is No Context Window" (Markov RS / no KV cache) | Recorded + scheduled | Already done — uses bounded Markov RS engine | (shipped) |
| VID3 | "Navigation Map" (residual trajectory through knowledge manifold, real-time PCA projection of fact landmarks) | Planned | M1–M8 hooks shipped, depth-fraction probe API needed | M1–M8, R6 |
| VID4 | "I Added a 769th Expert to GPT-OSS (Python)" (virtual expert) | Released | n/a | (shipped, public) |
| VID5 | "No KV Cache" (full Markov RS arc + boundary refs) | Planned | BR4 (server integration), softmax bottleneck KU2 acknowledged | BR4, BR5 |
| VID6 | "Build a Fresh Model From Scratch" | Planned | n/a (research) | n/a |
| VID7 | "I Killed Attention" (decoupling attention from FFN; static/semi-static/dynamic head taxonomy) | Sketched, not drafted | Static-head taxonomy at 31B (KU1), MTP6 acceptance-rate evidence, D-ATTN-MTG flash attention baseline | KU1, MTP6, D-ATTN-MTG |

**Key cross-link**: VID7's central claim ("91% of attention heads are
static routing, not computation") is *also* what makes MTP work — MTP
exploits exactly the staticity VID7 claims. So MTP6's per-token acceptance
rate over a corpus is a direct measurement of the static-attention
fraction VID7 claims, *per architecture*. Landing MTP1–MTP6 produces both
a baseline-credibility number (Ollama parity) and substrate evidence
(VID7's central thesis at scale). Treat MTP6 as a substrate-and-baseline
item, not just a competitive-parity item.

---

## Crate roadmaps

| Crate | Owns |
|---|---|
| [larql-compute](crates/larql-compute/ROADMAP.md) | Metal GPU kernels, MoE prefill, platform expansion |
| [larql-inference](crates/larql-inference/ROADMAP.md) | Forward pass, generation quality, KV engines |
| [larql-server](crates/larql-server/ROADMAP.md) | HTTP API, gRPC grid, remote expert protocol |
| [larql-router](crates/larql-router/ROADMAP.md) | Grid routing, self-balancing, QUIC transport |
| [larql-cli](crates/larql-cli/ROADMAP.md) | CLI UX, sampling flags, streaming display |
| [larql-lql](crates/larql-lql/ROADMAP.md) | LQL grammar, INSERT/SELECT/USE extensions |
| [larql-core](crates/larql-core/ROADMAP.md) | Graph data model, algorithms, serialization |
| [larql-vindex](crates/larql-vindex/ROADMAP.md) | Vindex format, storage, extraction |
| [larql-models](crates/larql-models/ROADMAP.md) | Architecture definitions, model loading |
| larql-boundary | Confidence-gated BOUNDARY ref codec; cold-context residual storage |

---

## Current state (2026-05-16)

- **~960 tests passing** across the workspace (server 292 lib + 447 integration = 739, router 169 lib + 50 integration = 220 with `--features http3`), 0 build errors.
- **Primary CLI verbs** in place: `run`, `chat`, `pull`, `list`, `show`, `rm`, `link`, `serve`, `bench`.
- **Gemma 3 4B Metal**: **88 tok/s** (Ollama steady: ~103). **Gap: 1.17×** (was 1.18× pre QKV defuse, 1.30× pre 2026-05-02 dispatch-geometry fix). **Acceptance criterion (~85 tok/s, 1.16×) met.**
- **Gemma 4 26B A4B Metal**: **19.4 tok/s** (was 5.1 — bug-locked under the same dispatch-geometry mismatch; correct multilingual output now).
- **Cross-arch coverage validated** (2026-05-09): Gemma 3, Gemma 4 31B dense, Llama 2 7B, Mistral 7B all dispatch correctly through Metal. Gemma 4 E2B falls back to CPU (deliberate — Metal doesn't yet implement Per-Layer Embeddings; diagnosed and tracked as D-METAL-PLE).
- **Grid (CPU MoE on remote shards)**: 18.3 tok/s 1-shard / 17.3 tok/s 2-shard local-loopback. Multi-host LAN/cross-region scaling unblocked.
- **Remote FFN (dense)**: `larql run --ffn URL` + `larql serve --ffn-only` wired end-to-end.
- **gRPC grid**: 2-shard self-assembling grid live-validated on 26B A4B.
- **4 KV-cache engines**: MarkovRS (287×), WindowedCheckpoint (254×), TurboQuant (4×), Apollo (20,000×) — all at ~95 tok/s on Gemma 3 4B Metal.
- **Wire format negotiation** (2026-05-07): f16 is now the default for all grid traffic (50% bandwidth reduction). i8 symmetric quantised residuals available opt-in (`LARQL_I8_WIRE=1`, 75% reduction). Content-type negotiation via `Accept` header; f32 fallback for non-grid clients.
- **Per-layer latency routing** (2026-05-07): `HeartbeatMsg.layer_stats` carries EMA avg_ms + p99_ms per layer; router routes to the server with lowest per-layer latency (falls back to requests_in_flight when no data yet).
- **WebSocket token streaming** (2026-05-07): `WS /v1/stream` now supports `{"type":"generate","prompt":"...","max_tokens":N}` command with per-token frames and cancel support. SSE streaming on `/v1/chat/completions` was already fully wired.
- **Criterion benchmarks** (2026-05-07): `make bench-wire` (wire codec encode/decode MB/s) and `make bench-routing` (route/heartbeat/rebuild ns/op). `larql-router` now has a library crate (`larql_router::grid`) for test/bench use.
- **Dynamic rebalancing** (2026-05-08): `rebalancer.rs` background task with configurable threshold (--rebalance-interval, --rebalance-threshold). Router detects sustained per-layer latency imbalance and sends `UnassignMsg` to the slow shard; server drains in-flight requests (up to 30s), sends `DroppingMsg`, and re-enters available pool. Real `requests_in_flight` counter wired into heartbeats via `RifGuard` in walk_ffn handler.
- **CI regression gate** (2026-05-08): `scripts/bench-grid-regress.sh` + `scripts/bench_compare.py` + `bench/baselines/`. First run auto-saves baseline; subsequent runs fail if tok/s drops >5% or p99 rises >10%.
- **Shannon arc closed** (2026-05-08): Exps 42–44 prove cross-entropy is a real wire format (Exp 42: 2.0 bits/char vs 6.3 gzip), residual stream is compressible (Exp 43: int8-clip3σ, 98.7% top-1, KL=2.0 nats), gate calibrated at threshold=2.16 (Exp 44: accept=68.9%, early-div=4.8%).
- **`larql-boundary` crate shipped** (2026-05-08): Phases 1–3 of BOUNDARY_REF_PROTOCOL. int8-clip3σ + bf16 codec, per-boundary confidence metadata, calibrated confidence gate. 100% function coverage, CI on Linux/Windows/macOS, 3 examples (encode_decode, gate_decision, accuracy). Phase 4 (server integration) not started.
- **QKV defuse + cleanup pass** (2026-05-09): default flipped from fused `q4k_q6k_qkv_proj_normed` to separate `rms_norm` + non-fused `q4k_q6k_qkv_proj` (+1.6–1.8 tok/s on Gemma 3 4B, +0.4 tok/s on Gemma 4 26B A4B post-thermal-cooldown cross-arch validation, ADR-016). Cross-arch bench captured for 4 model families. Shader inventory survey (47 shaders) + retention rationale doc-blocks added to opt-in shaders. New ADRs: [017 — shader retention under model agnosticity](crates/larql-compute/docs/adr/017-shader-retention-model-agnosticity.md), [018 — architecture → shader routing](crates/larql-compute/docs/adr/018-architecture-shader-routing.md). New docs: [shader-inventory](crates/larql-compute/docs/shader-inventory.md), [architecture-shader-map](crates/larql-compute/docs/architecture-shader-map.md), [llama-cpp-comparison](crates/larql-compute/docs/llama-cpp-comparison.md). One verifiable orphan deleted (`q4k_qkv_proj_v2`).
- **`make bench-cross-arch` shipped** (2026-05-09): runs `larql bench` across the model matrix (Gemma 3 4B, Gemma 4 31B dense, Gemma 4 26B A4B MoE, Llama 2 7B, Mistral 7B). `--save-baseline` / `--compare` modes; `bench/baselines/cross-arch/`. Operationalises ADR-017 model-agnosticity check; multi-arch sweep surfaces thermal artifacts as "every arch regresses simultaneously." Run on a cool machine before saving baselines.
- **D-RMS-FUSE Phase 1 implemented + falsified end-to-end** (2026-05-09): fused post-FFN `residual_add` + next-layer input rms_norm via `residual_norm_store` for the non-Gemma path. Bit-identical parity across Llama 2 7B, Mistral 7B, Gemma 3 4B (Gemma untouched — already triple-fused). End-to-end null vs drift on Llama 2 / Mistral. Kept opt-in `LARQL_FUSED_PRELAYER_NORM=1` per ADR-017 retention. Predicted ~0.2 ms/tok savings collapsed to zero — ADR-015 magnitude-compression at the extreme. Lesson: dispatch-overhead estimates (~7 µs/dispatch) over-predict savings when the kernel being skipped is also short.
- **Gemma 4 E2B 30× anomaly diagnosed** (2026-05-09): root cause = Per-Layer Embeddings (PLE) not implemented in Metal; `gpu.rs:372-374` deliberately routes E2B to CPU. Tracked as **D-METAL-PLE** (1-2 day Metal port of `forward/ple.rs`, 80-150× expected speedup for E2B; unlocks future PLE-using arches like Gemma 4 E4B).
- **larql-compute coverage audit + improvement** (2026-05-09): `cargo llvm-cov` reports **56.03% → 64.81% line coverage** (+8.78 pp; 2,575 newly-covered lines, 22.2% reduction in uncovered LoC). Three rounds: (1) deleted `metal/prefill.rs` (591 LoC of `#[allow(dead_code)]` orphan); (2) targeted tests on small helpers — `tg_width` math (qk_norm 0% → 23%), `scale_vector` dispatch (layer_scalar 12% → 97%), `residual_norm_store` shader parity for D-RMS-FUSE; (3) synthetic end-to-end Metal decode tests (`tests/test_metal_decode_synthetic.rs`, NEW) covering Llama-style + Gemma-3-style + D-RMS-FUSE off-vs-on parity, which lifted `decode/mod.rs` 7% → 61%, `encode_attn` 0% → 46%, `encode_post_ffn` 0% → 83%, `encode_qkv` 0% → 30%, `encode_ffn` 0% → 23%. Coverage policy (`coverage-policy.json`) targets 90% per-file / 93.5% total — current is below but no longer a wide gulf. Largest remaining gaps: `metal/trait_impl/decode.rs` (627 LoC at 21% — MoE / split-profile trait methods), `metal/decode/encode_ffn.rs` (1008 LoC at 23% — Q4_KF / MoE branches), `metal/diag/*.rs` (~3000 LoC at 0% — diagnostic / dev-only).
- **Positioning vs ollama / vLLM / llama.cpp documented** (2026-05-09): [docs/positioning.md](docs/positioning.md). Three-category framing (local single-user / batched serving / research+edit); feature matrix; per-competitor gap analysis; surfaces missing items now tracked under P2 § "Competitive parity" below.
- **Google released Gemma 4 MTP drafters** (2026-05-05, 4 days ago): `google/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant` — every Gemma 4 variant LARQL supports. 0.4B BF16 ~4-layer drafter for the 26B-A4B target. Architecture: shared input embeddings + shared KV cache + target last-layer activations concatenated with token embeddings then down-projected to drafter dimension. Measured **2.2× decode speedup on Apple Silicon at speculative batch 4–8** (Google blog), up to 3× generally. Apache 2.0 / CC-BY-4.0. Supported engines: HF Transformers, MLX, vLLM, SGLang, **Ollama**, LiteRT-LM (notably not llama.cpp). Competitive implication: the LARQL gap on Gemma 4 widens from 1.17× to ~2.6× as users adopt MTP on Ollama. Red Hat AI also released an EAGLE-3 speculator for `gemma-4-26B-A4B-it` (0.9B drafter). MTP1 promoted from P2 to **P1** — see new section below.
- **ADR-019 resolved** (2026-05-09): substrate-primary is **Gemma 4 31B dense + vindex**; MoE coverage retained at single-machine scale (Gemma 4 26B-A4B for cross-arch validation, virtual-expert work). Multi-machine MoE grid (C9 productionisation, critical-path items 5–10) demoted from P0 to P2 — substantial production-engineering work with no current experiment requiring "model spans 4 consumer machines" beyond what single-machine sharding already demonstrates. C1 (CPU MoE forward pass) stays P0 because V1/V2 cross-arch sweep on 26B-A4B requires it. See full resolution in "ADR-019" section below.
- **Engine ↔ Backend unification PR shippable** (2026-05-16): three specs landed in `crates/larql-inference/docs/specs/` — (1) [`kv-engine-unification.md`](crates/larql-inference/docs/specs/kv-engine-unification.md) (Steps 1-7 implemented, all parity tests green); (2) [`compute-backend-redesign.md`](crates/larql-inference/docs/specs/compute-backend-redesign.md) (Steps 1-4 implemented — `KvDispatch` sibling trait in larql-inference, `EngineBackend` umbrella, `CpuBackend`/`MetalBackend` scaffolding, `StandardEngine` migrated to dispatch through trait); (3) [`async-compute-backend.md`](crates/larql-inference/docs/specs/async-compute-backend.md) (trait surface locked, 6 open questions resolved; A1 trait + handles, A2 `CpuBackend`, A3 `MetalBackend` scaffold, and A5 `StandardEngine` opt-in landed 2026-05-16 — A3's Metal-feature validation gate is blocked on a parallel `larql-compute-metal` extraction). Honest finding from Step 5 discovery: per-layer Metal kernels at the sync trait's granularity are *slower* than today's fused decode path because each per-layer call forces a separate GPU command-buffer commit — `AsyncComputeBackend` (intent-collector pattern, deferred dispatch) is the prerequisite for any tok/s win. That work is 6-12 months end-to-end (see new "P0 — Engine ↔ Backend unification" section below). The unification PR ships the foundation; tok/s wins land in A4 (real Metal deferred dispatch) and the multi-step Metal kernel work that compounds on top.
- **Cross-engine forward-pass correctness gate** (2026-05-16): `larql shannon verify` orchestrates LARQL Rust forward against HF/PyTorch + MLX reference scorers (subprocesses) on a shared corpus and prints a bits/char delta table. First serious application surfaced **four config-loading bugs in larql-models** — all closed in the loader (no env-var workarounds in production): (1) `rms_norm_eps` from config.json was never read by the trait default; (2) Gemma 3's per-layer-type `rope_scaling` structured form (`{full_attention: {rope_type: linear, factor: 8}, sliding_attention: {rope_type: default}}`) wasn't honoured; (3) `rope_scaling = llama3` (wavelength-dependent per-channel `inv_freq` adjustment) wasn't implemented; (4) `norm_epsilon` alias (StarCoder2's name for `rms_norm_eps`) wasn't recognised. Post-fix, all four affected models match HF F32 to <0.06% bits/char with zero env vars. `scripts/diagnose_models.py` (multi-arch sweep) reports 7/9 PASS. CI gate at `.github/workflows/shannon-verify.yml` runs SmolLM2-135M verify on every PR. Diagnostic doc: [`docs/diagnoses/shannon-cross-engine-divergence.md`](docs/diagnoses/shannon-cross-engine-divergence.md). Plus GPT-2 legacy config-key aliases (`n_embd`/`n_layer`/`n_head`/`n_inner`) parsed via new alias-list machinery in `detect/config_io.rs`.
- **larql-compute-metal coverage push closed** (2026-05-16): post-ADR-019 split, the Metal backend now lives in its own crate with **97.28% line coverage, 59/59 files at the 90% per-file floor, zero debt baselines**. Up from 75.69% (50/59 files clearing 90%, 9 debt baselines) at session start. Key techniques: (1) `MetalBackend::with_options` to bypass the env-snapshot caching that silently no-op'd flag-toggling tests on `decode_one_token_with_env`, opening the `fused_attn` / `fused_qk_norm_rope` / `fused_kv_append_attend` / `fused_post_attn_norm` branches in `decode/encode_attn.rs` (68.78% → 99.53%); (2) per-format prefill split-phase tests (Q4_K / Q4_KF / Q4_0 × gated / non-gated, `LARQL_PROFILE_SPLIT=1`) for `decode/encode_ffn.rs` (61.43% → 92.86%); (3) direct calls to the public `run_experts_prestaged_metal` / `run_experts_preselected_metal` / `run_dense_ffn_q4k` paths plus a real-MoE-layer `decode_token_q4k_moe` end-to-end test for `moe_dispatch.rs` (38.91% → 95.25%); (4) `decode_attention_layer` integration tests covering V-norm, post-norms, and `wo.format` Q4_KF/Q6_K branches for `decode_hybrid.rs` (0% baseline → 94.41%); (5) dead-code deletion of `MetalBackend::full_pipeline` (108 lines, no callers, doc said "old benchmark entry point") to clear `pipeline.rs` to 100%; (6) `Config::from_args` + JSON helper + Smoke-profile end-to-end coverage for `diag/shader_bench.rs` (4.25% → 99.36%) and `diag/kernel_profile.rs` (0% → 97.12%) — the diag scripts now smoke-run real GPU dispatches in unit tests; (7) a dedicated `tests/test_decode_diag.rs` integration binary (fresh process, fresh `CALL_COUNT`) that hits the previously-believed-structural cap on `decode/diag.rs` (85.23% → 93.75%). Coverage-policy file now an empty-baseline gate: any regression on any file breaks CI.
- **larql-router self-healing + HTTP/3 + hedged-dispatch phase** (2026-05-16): MoE expert routing (ADR-0018, per-(layer, expert-range) replication keys), Prometheus `/metrics` (ADR-0017), Phase 4 HTTP/3 shard transport behind `--http3-shards` / `--http3-port` (ADR-0019, h3 0.0.8 + h3-quinn 0.0.10 + h3-axum 0.2), hot-shard hysteresis (ADR-0014 amendment, `--hot-shard-demote-ratio` default 0.8), backpressure tier (ADR-0020 — `--saturation-ceiling N` filter in `route()` / `route_expert()`, dispatcher distinguishes 503 saturation from 400 no-owner via `has_owners_for()`, emits `Retry-After: 0.5`, bumps `larql_router_route_saturation_total`), long-running chaos test (`tests/test_grid_chaos.rs`, 5,000 random ticks × 2 variants, asserts ledger consistency + coverage floor + no `route()` panic), hedged dispatch (ADR-0021 — opt-in via `--hedge-after-ms M`, new `route_with_rank` / `route_expert_with_rank` grid APIs, `hedged_post_json` racing helper, dense + MoE fan-outs wired, `route_hedge_fires_total` / `route_hedge_wins_total` counters; supersedes the original "speculative next-layer prefetch" P1 framing — an audit falsified that framing since the router sees one batched call per token against a single input residual, so hedge-the-slow-primary is the legitimate router-layer optimisation). Concurrent-route bench (`bench_route_concurrent`, 2026-05-16) surfaced lock-contention plateau: pre-swap 1 = 5.6 → 4 = 8.7 → 8 = **4.0** → 16 = 3.6 Melem/s (8 workers *worse* than 1 — pathological). **Lock primitive swap** (2026-05-16): `tokio::sync::RwLock<GridState>` → `parking_lot::RwLock<GridState>` across larql-router and tests. Every grid critical section is short and sync (no `await` held under the lock), so synchronous is semantically correct and the compiler enforces it (parking_lot guards are `!Send`). Post-swap: 1 = 6.4 / 4 = 11.1 / 8 = 7.2 / 16 = 6.1 Melem/s — **+14% / +28% / +80% / +70%**, pathological 8-worker collapse eliminated. 220 tests still pass. Saturation-filter cost on the happy path: ~108 ns vs ~113 ns baseline (in noise); all-saturated short-circuit ~57 ns. Router test surface: 169 lib + 50 integration = **219 tests** (220 with `--features http3`). Coverage **~93%**. Five examples (`embed_grid`, `static_shards_server`, `admin_client`, `fanout_dispatch`, `saturation_backpressure`); criterion benches cover dense + MoE + saturation + concurrent-route. Multi-host deployment runbook at [`crates/larql-router/docs/multi-host-demo.md`](crates/larql-router/docs/multi-host-demo.md). Server-side `GET /v1/shard/{model}/{start}-{end}` audited + documented in [`crates/larql-server/docs/router-spec.md`](crates/larql-server/docs/router-spec.md) §4. ADRs: [0017](docs/adr/0017-router-metrics.md), [0018](docs/adr/0018-moe-expert-routing.md), [0019](docs/adr/0019-http3-shard-transport.md), [0020](docs/adr/0020-route-backpressure-tier.md), [0021](docs/adr/0021-hedged-dispatch.md).
- **Whole-codebase review** (2026-05-28): multi-agent deep review (17 crates, ~415K LOC; per-crate reader + adversarial verification). Clippy clean (2 trivial nits); exposure concentrated and thematic. ~7 verified high/medium items now tracked under "Codebase hardening (review 2026-05-28)" below and mirrored into crate-local roadmaps. Top two confirmed by hand: infallible `FfnBackend::forward` aborts serving on remote-shard blips; Metal KV append has no `pos<max_seq` clamp (GPU OOB past 4096 rows). Record: [`docs/audits/codebase-review-2026-05-28.md`](docs/audits/codebase-review-2026-05-28.md).
- **Follow-up codebase review** (2026-06-12): working-tree diff review (C10 residency + FR3) plus fresh whole-workspace sweep with adversarial verification. Numeric core verified clean (asm kernels, int8 attention, GGUF loader overflow claims all refuted); verified exposure at the edges: `model_id` path traversal in shard loader, zero GPU-error checking across 77 Metal `wait_until_completed` sites, dispatch-geometry duplication back at 2 sites despite `KernelHandle`, corrupt-vindex panics (2026-05-28 item 1 still open), GIL never released in larql-python, 145 env flags / ~18 documented. Tracked under "Follow-up review (2026-06-12)" below; maintenance-debt recommendations under "Cleanup / consolidation track (added 2026-06-12)". Record: [`docs/audits/codebase-review-2026-06-12.md`](docs/audits/codebase-review-2026-06-12.md).
- **Tagged release binaries + first tag `v0.1.0`** (2026-07-24/25, [ADR-0026](docs/adr/0026-tagged-release-binaries.md)): larql had no distribution artifact — every host started from `git clone` + a cold `cargo build --release`. `.github/workflows/release.yml` now cross-builds `larql` + `larql-server` for macOS-aarch64 / Linux-x86_64 / Windows-x86_64 on `v*` tags under a new `release-dist` profile (stripped, no line tables; the profiling-friendly `release` profile is untouched) and publishes one archive per platform to a GitHub Release. `needs: build` gates publication on all three legs, so a partial release is not reachable. **`v0.1.0` cut 2026-07-25; the workflow went green on its first run** — archives verified to contain both binaries, and the macOS one smoke-run (`larql 0.1.0`, `dec-bench` present, strip confirmed). The driver is a **hard policy, not an optimisation: GPU-provisioned hosts never build from source** — a cold build is 20–40 min of pure CPU work with the GPU idle, and the DEC funnel runs ~10 stages on ephemeral rented hosts. `scripts/lib/larql-binaries.sh` enforces it for the stage drivers (operator-supplied → reuse → fetch → build, with the build refusing and exiting non-zero when `nvidia-smi` is present unless explicitly overridden). DEC-0.5 keeps compiling only the criterion kernel bench — a bench target is not a shippable binary and that kernel is the stage's measurement object. Separately, all **18** workspace crate names are claimed on crates.io as `0.0.0` placeholders (17 + `larql-experts`, a nested workspace invisible to the `[workspace] members` sweep), verified against the registry API; this is squatting-prevention, **not** the crates.io publishing ADR-0026 still declines. (**19th name, `larql-factory`, claimed identically on 2026-07-29** — see the ADR-0026 addendum and the entry directly below.)
- **Vindex Factory G0 slice + `larql recipe estimate`** (2026-07-29, [`docs/vindex-factory.md`](docs/vindex-factory.md)): new `larql-factory` crate — recipe schema (§4), `build_id` canonicaliser (§5), a structural validator covering every §6.1 PR-check gate that doesn't need network I/O, `larql capabilities` (§15.2, sourced from a new declarative architecture registry in `larql-models` rather than a hand-duplicated list — cross-checked by a test against the real `detect_from_json` dispatch), `larql card render` (§9), and `larql recipe estimate` (§6.1 step 4, the crate's first network I/O — upstream size + a coarse per-output byte model + an executor recommendation + a cost band priced against `docs/dec-funnel-v0.2.md` §7's existing rate basis rather than a fabricated duration prediction). Also lands previously-uncommitted OLMoE + GraniteMoE architecture support (a real prerequisite for the capability registry's claims about those two families). Every source file at or above the 90% coverage floor. Full detail in [`ROADMAP_STATUS.md`](ROADMAP_STATUS.md)'s "Recently shipped" entries. PR #192.
- **`larql recipe build` — PREFLIGHT→RELEASE build driver** (2026-07-29, same spec, §7): the `larql-factory::build` module orchestrates FETCH → EXTRACT → SLICE → MANIFEST → VERIFY → PUBLISH → RELEASE as subprocess calls into this same `larql` binary, behind a `CommandRunner` trait (`SubprocessRunner` for real builds, a `MockRunner` in tests) so the whole pipeline's stage ordering and failure handling is unit-tested without spawning a process or touching credentials. FETCH scopes `HF_HUB_CACHE` per build — `resolve_model_path`'s cache lookup doesn't disambiguate by revision, so a shared cache could otherwise let EXTRACT silently build from the wrong commit. PUBLISH always goes `--private` first; RELEASE only flips a repo public once every output has verified (§8's "nothing goes public unverified"). Always returns a `BuildRecord` — JSON-printable whether the build passed or a specific stage failed, matching `dec-bench`'s `--output-file` pattern. **Scope, decided deliberately after tracing what actually exists in this codebase, not what the spec assumed**: MIRROR (R2) and REGISTER (chuk-experiments-server) aren't implemented — no R2/S3 client exists anywhere, MCP tools aren't callable from compiled Rust, and the spec's own text assumes both are the rig worker's job; `BuildRecord` is the hand-off point for an external wrapper, the way `dec0-loopback.sh` already wraps `dec-bench`'s JSON. VERIFY here is checksum integrity only (`larql verify`) — the numeric reconstruction/logit-match checks in §8.1 need per-architecture tensor-naming knowledge that isn't validatable without real model weights. Extended `larql-vindex`'s publish path with a `private: bool` option and a new `set_repo_visibility` capability (verified against the real HF OpenAPI spec) to make the private-then-public two-phase publish possible; new `larql hf visibility <repo> --public|--private` command. Wired in as `larql recipe build <FILE> [--scratch-dir DIR]`. Every new source file at or above the 90% coverage floor.

---

## Codebase hardening (review 2026-05-28)

Whole-codebase multi-agent review (17 crates, ~415K LOC; one reader per crate +
adversarial verification of every high/critical finding). Full record:
[`docs/audits/codebase-review-2026-05-28.md`](docs/audits/codebase-review-2026-05-28.md).
Verdict: mature, defensively-engineered; exposure is concentrated and thematic,
not pervasive. `cargo clippy --workspace --all-targets` is clean (2 trivial nits).
Per-crate items below are mirrored into each crate-local roadmap.

Ordered actions (✅ = also confirmed by hand):

1. **Make `FfnBackend::forward` fallible** (P0) — the trait returns an infallible
   `Array2<f32>`, forcing process-abort on served paths. Convert
   `larql-inference` `cached.rs:123,200`, `hidden.rs:38`, ✅`http.rs:519` and
   `larql-compute` `moe/forward.rs:191,211` to `?`-propagation into the existing
   `GenerateError` channel. Highest leverage — removes the top serving-abort
   class. [larql-inference, larql-compute]
2. ✅ **Bound the Metal KV cache** (P0) — `kv_attention.rs:186-187` (+ `attn_fused`,
   `kv_append_attend_fused`) write `K_cache[pos*total+tid]` with no `pos<max_seq`
   clamp; sessions exceeding the 4096-row cache write OOB on the GPU during
   normal decode. Add the position guard and extend `ensure_prompt_fits` to
   `prompt_len + max_tokens`; expose cache sizing to the caller. The only
   verified memory-corruption bug. [larql-compute-metal — no crate roadmap]
3. **Fix `larql-python` soundness gaps** (P0) — `trace_py.rs:14-28` raw
   `*const ModelWeights`/`*const Tokenizer` is use-after-free across `del model`
   (give `PyResidualTrace` a `Py<PyWalkModel>`); `walk.rs:207-223` zero-copy
   embed `Vec::from_raw_parts` lacks the length check its sibling paths use.
   [larql-python — no crate roadmap]
4. **Validate router layer ranges + wire server eviction** (P1) — `larql-router`
   `routing.rs:237` builds an unbounded route table from gRPC-announced ranges
   (clamp to model depth before `rebuild_route_table`); `larql-server`
   `session.rs:184` + `ratelimit.rs:83` never evict (dead eviction logic).
   Memory/DoS class. [larql-router, larql-server]
5. **Shared NaN-safe top-K/sort helper** (P1) — route the ~10
   `partial_cmp().unwrap()` sites (vindex router:107/lm_head:322/gate_store:330,
   core graph:278/walk:35/pagerank:19, cli parity:1119, python vindex:847,1432)
   and `larql-lql`'s four `embed.row()` callers through bounds-checked helpers.
   [larql-vindex, larql-core, larql-cli, larql-lql]
6. **SQL expert UTF-8 offset bug + typed cross-crate contracts** (P2) —
   `larql-experts/sql/src/lib.rs:161` slices the original string with offsets
   from an uppercased copy (panic on non-ASCII SQL); use `char_indices`. Then
   consider typing the `*const f32` reinterpret, positional-QKVO
   (`attn_data[1]/[2]`), and `per_layer_ffn_key` conventions to stop silent
   drift. `larql-router-protocol`: `None` fingerprint disables TLS verification.
   [larql-experts — no crate roadmap, larql-router-protocol — no crate roadmap]

Hygiene (separate from the sweep): 2 clippy nits in `larql-cli` (unused
`ProjectorWeights`, dead `total_tiles`); coverage below the ≥90% floor on
`larql-inference` (70.7%) and `larql-cli` (12.0%).

### Follow-up review (2026-06-12)

Diff review of the in-flight C10/FR3 changes + fresh whole-workspace sweep
(10 subsystem readers + adversarial verification; several headline claims
refuted — GGUF overflow, kernel release-mode bounds, `attn_fused` overflow
all died under verification). Full record:
[`docs/audits/codebase-review-2026-06-12.md`](docs/audits/codebase-review-2026-06-12.md).
Items 1 and 5 of the 2026-05-28 list were re-confirmed still open
(`cached.rs:123,200`/`hidden.rs:38` panics; python `vindex.rs:847` NaN sort)
— they stay tracked there, not duplicated here.

Ordered actions:

1. **Sanitize `model_id` in shard loader** (P0, security) —
   `larql-server/shard_loader.rs:30` joins router-supplied `model_id`
   (`announce.rs:544`) into the store path unvalidated; `../` escapes the
   shard dir (tar unpack itself is safe, tar 0.4.45). Reject path
   separators / `..`. Follow-on (P2): grid non-join RPCs (`drain_server`,
   `assign_range`, `grid/service.rs:114`) don't require the grid key.
   [larql-server, larql-router]
2. **Check Metal command-buffer status** (P0) — all 77
   `wait_until_completed()` sites read buffers with no `status()`/`error()`
   inspection (e.g. `ops/full_pipeline/dispatch.rs:456,783`); a failed GPU
   command yields stale data straight into logits. Add a `wait_and_check()`
   helper and migrate. Cheap insurance against the next phantom-drift hunt.
   [larql-compute-metal — no crate roadmap]
3. **Route the 2 hardcoded dispatches through `KernelHandle`** (P1, latent
   but a 3×-historical bug class) — `decode_hybrid.rs:388-391` hardcodes
   256 threads/TG while `q8_matvec_pipeline` is already a `KernelHandle`
   carrying the geometry; `stages/qkv_proj.rs:241` takes a raw
   `ComputePipelineState` so it can't consult one. Correct today, silently
   fast-but-wrong on any shader geometry change. [larql-compute-metal]
4. **Corrupt-vindex load robustness** (P1) — `larql-vindex
   format/load.rs:81,293` index `gate_slices[info.layer]` with
   `info.layer` straight from `index.json`, no bounds check (panic on
   corrupt manifest; validate `< num_layers` → `VindexError::Parse`);
   `load.rs:317` defaults missing manifest `offset`/`length` to 0,
   masking the real error. [larql-vindex]
5. **Validate Q4K lm_head buffer size** (P1, from the diff review) —
   `larql-kv/generation.rs:657` + `larql-inference
   forward/predict/dense.rs:189` never check buffer len vs
   `vocab_size × bytes_per_row`; truncated weights panic mid-decode,
   padded ones decode garbage logits. One length check → clean f32
   fallback. [larql-kv, larql-inference]
6. **Release the GIL in larql-python** (P1) — zero `allow_threads` in the
   crate; `predict`/`trace`/`generate_with_hooks`/`infer`/`infer_trace`
   block all Python threads for whole forward passes. Wrap compute in
   `py.allow_threads`. (NaN sort at `vindex.rs:847` already tracked as
   2026-05-28 item 5.) [larql-python — no crate roadmap]
7. **Env-flag registry** (P1) — 145 distinct `LARQL_*` flags, ~18
   documented; accepted values already diverge (`LARQL_Q4K_ASM=true` works,
   the three new C10 flags accept only `"1"` — a bench run with `=true`
   silently measures the wrong config). Route flags through the
   `larql-compute/src/options.rs` taxonomy + generate `docs/env-flags.md`.
   [workspace]
8. **Diff-review cleanups before/with the C10 commit** (P2) — fold
   `hidden == 0` into the padded-down guard (`larql-compute
   kquant_forward/cached.rs:861` + twin); extract the duplicated ~35-line
   padded-down block into one `larql-compute` helper with a reusable
   scratch buffer (kills the lockstep-comment hazard + ~69 KB/token alloc
   on 26B); drop the unnecessary `relations.clone()`
   (`larql-lql edges.rs:186`); length-check `labels`/`counts` at load
   (`relations.rs:35`); OnceLock the `LARQL_FR3_EXPLICIT` read
   (`edges.rs:279`). [larql-compute, larql-inference, larql-lql]
9. **Forward-pass loop unification** (P2, ADR first) — five parallel
   layer-step loops in `larql-inference/vindex/kquant_forward/`
   (`hidden`/`prefill`/`decode_step`/`decode_step_direct`/remote-FFN) each
   repeat the same sentinel logic; every stepping change lands 5× or
   numerics silently diverge. Big-ticket; cuts across the C10-hot files,
   so sequence behind the current residency arc. [larql-inference]
10. **Dead weight** (P2) — 4 unreferenced Metal shader modules
    (`graph_walk_knn`, `q4_sparse_matvec`, `turboquant_{encode,decode}`)
    need an ADR-017 retention rationale or deletion; `model-compute` crate
    has no second consumer (no-speculative-extraction policy); `larql-inference`
    `test_utils.rs` (1,228 lines) ships as public API. [larql-compute-metal,
    model-compute, larql-inference]
11. **Serving posture** (P2, plausible-not-verified) — document or fix:
    streaming completions serialize on the weights guard
    (`completions.rs:302`) with no per-request timeout (`:366`); no
    graceful drain on shutdown (`bootstrap.rs:1255`); grid join stream has
    no malformed-message rate limit (`grid/service.rs:121`). [larql-server,
    larql-router]

### DEC-readiness review (2026-07-22)

Targeted review of the **DEC data-plane** ahead of the DEC funnel programme
([`docs/dec-funnel.md`](docs/dec-funnel.md)) — the code the programme runs on
rented x86 marketplace hosts, against non-Gemma models, over adversarial
links. Four parallel readers (security, hardcoding/config, modularity,
performance), verified findings only. Full record:
[`docs/audits/dec-readiness-review-2026-07-22.md`](docs/audits/dec-readiness-review-2026-07-22.md).
Verdict: structurally sound and the wire decoders are mostly hardened, but a
**silent-corruption cluster** (produces a number, the number is a lie) is the
dominant risk because the whole programme is a measurement exercise. The B-row
f32/f16/i8 serving path is well-built; only the Q8K path — the wire DEC prefers
— does not batch. Work through in the order below (roughly DEC-stage sequencing).

**Batch A — silent corruption + the security HIGH (before any claim-bearing run): ✅ DONE (2026-07-22)**

1. ✅ **Q8K batched compute** (P0, corrupts C1/C6, gates DEC-0) — `walk_ffn/q8k.rs:199`
   ran B same-layer rows as B independent matvecs, each re-streaming the full
   layer's weights, so the Q8K batch curve was ~linear by construction. Fixed:
   the handler now groups request entries by layer; groups of >1 dequantise to
   f32 and run ONE batched GEMM through `kquant_ffn_forward_layer` (preserving
   the Q8K upload win, amortising weights across rows); singleton groups keep
   the existing single-row Q4K×Q8K kernel unchanged (no batching problem there,
   and it avoids dequantising gate/up on the latency-critical single-token
   decode path). Numerical equivalence between the two paths is pinned by
   `walk_ffn_kquant_layer_q8k_batched_gemm_matches_per_row_single_kernel`.
   [larql-server, larql-inference]
2. ✅ **Multi-layer decoder allocation bomb** (P0, security) — `Vec::with_capacity(n)`
   from an attacker u32 → one 16-byte packet aborts the server
   (`moe_remote/multi_layer_wire.rs:105,146,211,254,292`). Regression from the
   repo's own `max_possible_entries` guard (PR 104). Fixed: mirrored the guard
   at every task/result/expert-count allocation site, plus inside the shared
   `read_f32_slice`/`read_i16_slice` helpers so `hidden`/`nb`-derived lengths
   are bounded before any allocation, covering both `decode_multi_layer_request`
   and the client-side `decode_multi_layer_response`. 8 new
   `rejects_impossible_*_before_allocating` regression tests. [larql-inference]
3. ✅ **Shard failure zero-fills FFN output** (P0, silent generation corruption) —
   `sharded.rs:117` returned zeros on a panicked/unowned shard and decode
   continued on a corrupt hidden state. Fixed: `forward_predispatch_all` now
   panics loudly on an unowned layer or a shard's transport failure (propagating
   the worker thread's panic via `resume_unwind` instead of swallowing it),
   matching `RemoteWalkBackend::forward`'s existing panic-on-error convention.
   [larql-inference]
4. ✅ **Down-proj ignores its format tag on the Q8K fast path** (P0, gates
   DEC-4/6) — `kquant_forward/walk_ffn.rs:135` fed `ffn[2].0` into a Q4_K-only
   kernel without checking `ffn[2].1`; a non-Q4_K down slab (Inkling/K3) would
   have decoded garbage. Fixed: the fast path now additionally gates on
   `ffn[2].1 == "Q4_K"`, falling back to the format-aware `dequantize_matrix`
   path otherwise. Fixed in both the `larql-inference` copy (the live serving
   path) and the `larql-compute` twin (same bug, not yet wired to a serving
   path — see item 4g). Regression:
   `walk_ffn_kquant_layer_q8k_rejects_down_slab_with_non_q4k_format_tag`.
   [larql-inference, larql-compute]
5. ✅ **x86 scalar-fallback is silent** (P0 *observability*, gates DEC-0.5) —
   `q4k_q8k_gate_up_into` (:1377) and `q6k_q8k_matvec_into` (:2119) have no AVX2
   branch; the serving-path doc-comments falsely claimed "NEON/AVX2". Building
   the AVX2 kernels remains **C-ladder** work (not done here). Fixed: added
   `larql_compute::cpu::ops::q4k_q8k_dot::kernel_class_summary()`, logged once
   at server startup (`larql-server/bootstrap.rs`), and corrected the false doc
   comments on `q4k_q8k_gate_up_into` and the `q8k.rs` module doc — so no DEC
   number is ever recorded on an unlogged scalar path. [larql-compute, larql-server]

**Batch B — fleet/config landmines (before the x86 + Linux arms):**

6. **`127.0.0.1` announce on `--join`** (P1, breaks multi-host grid) — refuse a
   wildcard host without `--public-url`, or detect the outbound IP
   (`bootstrap.rs:1222`). [larql-server]
7. ✅ **Backend factory + capability dispatch** (P1, unblocks x86 + pre-work for
   G-ladder) — DONE 2026-07-22 (except the capture-portability doc note, folded
   into #8's script work). `larql_compute::backend::factory` adds
   `BackendKind` (+`FromStr` for a future `--backend`/`DEC0_BACKEND` string) and
   `backend_from_spec(kind, registry)` with injected constructors (ADR-019: the
   trait crate names no backend crate); `larql-cli/src/backend_select.rs` builds
   the registry once, and all 7 `if metal` cfg copy-paste sites collapse onto it
   (`run_cmd.rs` ×2, `bench/remote_ffn_runtime.rs`, `bench/local_runtime.rs`,
   `dec_bench/capture_runtime.rs`, `shannon_cmd.rs`, `walk_cmd.rs`). Semantics
   tightened: an explicit `--metal` with no usable device now errors loudly
   instead of silently benching on CPU. Dispatch de-`bool`ed: the remote-MoE
   fork probes `supports(Capability::DecodeMoe)` on the constructed instance,
   and run_cmd's experts module fixes TWO latent bugs — `metal_ready_for_q4`
   probed `default_backend()` (always CPU post-ADR-019, so the check was
   vacuous) and `Strategy::MetalQ4K` then ran `layer_graph::generate` on a
   fresh `default_backend()` (CPU) — the constructed backend is now stored in
   `Runtime` and probed via the canonical `PrefillQ4 && DecodeToken` pair.
   [larql-cli, larql-compute]
8. **`--metal` / `--backends metal` hardcoded for x86** (P1) — `DEC0_BACKEND`
   env in `scripts/dec0-loopback.sh:80,97`, platform-conditional `--backends`
   default (`bench/args.rs:26`). Couples to #7. [larql-cli]
9. ✅ **`SKIP_MOE` vs `LARQL_SKIP_MOE` name split** (P1, corrupts the anchor's
   ceiling arm) — DONE 2026-07-22. One canonical prefixed name for all three
   unprefixed vars (`LARQL_SKIP_MOE`, `LARQL_SKIP_OUTER_NORM`,
   `LARQL_DECODE_DEBUG`), read through shared accessors in
   `larql_compute::options` (`skip_moe_enabled` / `skip_outer_norm_enabled` /
   `decode_debug_enabled`) that honour the historical unprefixed names as
   deprecated aliases with a one-time stderr warning. The grid path's
   `GridRuntimeConfig` now reads the same accessor as the local path, so the
   DEC-0 ceiling arm measures one thing regardless of which name the operator
   types; dec-funnel.md DEC-0 anchor note updated to the canonical name
   (README already used it). Alias behaviour pinned by
   `unprefixed_legacy_aliases_still_enable_their_flags`.
   [larql-inference, larql-compute, larql-compute-metal, docs]
10. **DEC deployment auth posture** (P1, security) — the data plane is open
    unless `--api-key` is set (`/v1/shard` streams the whole vindex as a tar);
    router admin RPCs (`drain_server`/`assign_range`) and the grid port are
    unauthenticated (`grid/service.rs:386,397,455`; overlaps 2026-06-12 item 1
    follow-on). Decide: mandatory data-plane auth off-loopback, or private-network
    binding as a documented DEC deployment rule. Constant-time the gRPC grid-key
    compare (`service.rs:105`) while here. [larql-server, larql-router]
11. **Timeout defaults + no-op grid-LAN timeout** (P2) — the 30/60/120s defaults
    assume 26B+LAN and will 504 on Inkling cold-start (note at DEC-4/5
    provisioning); `grid_lan_runtime.rs:179` timeout is a `let _ =` no-op (wire
    it). Arch-tag the grid-regress baselines (`bench-grid-regress.sh:35`).
    [larql-cli, larql-inference]

**Batch C — structural pre-work (schedule per-ladder, not a DEC-0 blocker):**

12. **Compute admission control** (P1, protects C3/DEC-2) — ~192 concurrent
    multithreaded-BLAS `spawn_blocking` tasks at 4 clients × 48-layer fan-out
    look like tier saturation but are oversubscription. Semaphore sized to
    physical cores + `OPENBLAS_NUM_THREADS=1` for the serving build.
    [larql-server]
13. ✅ **q8k endpoint drain/heartbeat/latency blindness** (P1, breaks C7 router
    demo) — DONE 2026-07-23, extended to the whole expert surface per the
    expert-serving review (§1d): shared `track_model_request` helper
    (`RifGuard` + `requests_total`) on the q8k walk-ffn handler AND all
    expert endpoints (single/legacy-batch/layer-batch×2/multi-layer×2), with
    `layer_latency_tracker.record` on q8k walk-ffn and the expert batch
    handlers. See `docs/audits/expert-serving-review-2026-07-23.md`.
    [larql-server]
14. ✅ **dec_bench `Endpoint` seam + routing capture** (P1, gates the
    routed-experts arm that gates the C1-on-MoE verdict) — DONE 2026-07-23,
    preceded by a three-reader expert-serving review
    (`docs/audits/expert-serving-review-2026-07-23.md`) whose Phase-A server
    hardening + pre-measurement perf batch landed first (batch handlers 400
    on unresolvable experts; q8k shape validation; owned-entry
    `per_expert_bytes` probe; bulk LE codecs off the reactor thread; stale
    parallelism docs corrected). Built: `Endpoint` enum (walk-ffn ×2 +
    experts-multi-layer ×2 — path/frame/decoder/`server_ms`/denominator per
    variant); capture `--routing` flag with additive pool sidecars
    (`raw.bin`/`normed.bin`/`routing.bin`, manifest stays v1, the shipped
    330M pool still replays the dense arms); routing computed at the capture
    sink via the now-`pub` `build_moe_router_weights` + client router,
    gated by a router twin-parity test (inference `route()` ≡ compute
    policy pipeline, 4 shapes); per-point batch-aware denominators
    (`weight_bytes_tok_naive` primary — server streams per-row, no
    cross-row sharing — + `_union` as the DEC-3 bound) and
    `dec/endpoint(_code)`/`dec/experts_union_frac`/`client_rayon_threads`
    in the pulse/run record; warmup non-zero-response guard (§1a class).
    [larql-cli, larql-inference, larql-server]
15. **Server expert dispatcher** (P2, before G4 cuda-experts) — extract one
    `run_experts(state, backend, …)` from the per-handler Metal/CPU branches
    (`q8k.rs:107`, `grpc_expert.rs:178`, `expert/{layer,multi_layer}_batch.rs`).
    [larql-server]
16. ✅ **Wire consolidation** (P2) — DONE 2026-07-24, the trigger having arrived
    early (DEC-1A's asymmetric codecs + timing field, not DEC-6a). The dense
    binary frame is single-sourced in `larql-inference` `ffn/remote/codec.rs`
    (encoder+decoder+constants; server `binary.rs` is a shim; router imports);
    every CT string and `BATCH_MARKER` declared once; byte-identical wire
    pinned by encode-decode-reencode tests; all allocation-bomb guards moved
    verbatim; `call_q8k_layers` byte-counter gap fixed. Three extensions then
    landed on the consolidated seam same-night (ADR-0025): the header-gated
    `serve_us` timing trailer, independent inbound/return wire formats
    (f16/i8 REQUEST encodings — previously f32-only — with `Content-Type`=in
    / `Accept`=out decoupled), and the `dec-bench drift` C6 fidelity
    instrument. [larql-inference, larql-server, larql-router, larql-cli]
17. **MoE parity seams + hot-path cleanups** (P2) — make `build_moe_router_weights`
    `pub` and share the combine math before the DEC-6b KDA/LatentMoE port
    (`hidden.rs:93` vs `core.rs:111`); `model.patched` arc-swap so compute
    doesn't hold the read lock across FFN (C3 shared-tier landmine); drop the
    4–6 full-buffer request-lifecycle passes (`core.rs:48,235,284`,
    `binary.rs:88`); gate `--release-mmap-after-request` on `requests_in_flight`;
    persistent client fan-out pool. [larql-inference, larql-server]

### Vindex + WalkFFN review (2026-07-30)

Subsystem review of `larql-vindex` (~51K LOC) and the walk-FFN engine
(`larql-inference/src/vindex/walk_ffn/`), merged with an external strategic
review of the architecture and a kernel deep-dive. Full record:
[`docs/audits/vindex-walkffn-review-2026-07-30.md`](docs/audits/vindex-walkffn-review-2026-07-30.md).
Verdict: both subsystems structurally healthy (the storage layer and spec
crate are defensive engineering done right; the trait-dispatch refactor
paid off — FP4 cost zero kernel code), but four high-severity runtime bugs,
a silent-wrong-numerics cluster in the quantized walk paths (same
"produces a number, the number is a lie" theme as the DEC review), and
**no walk-vs-dense numerical parity test anywhere in the tree**.

**Status 2026-08-01: PROGRAMME CLOSED — 24 of 24.** Tiers 0–1 in full (2026-07-30,
incl. all four HIGHs); item 13 resolved with the finding inverted (the
exact-first gate chain is now actually wired — `enable_hnsw()` had been
leaking approximate selection into walk numerics); Tier 2 complete:
base+delta (16), forward/forward_observed split (15), runtime trace
emission (17), execution planner (18), two-stage selection (19) all
shipped; parity suite (20) landed with the per-file ≥90% coverage
pass; KnnStore unified at the retrieval-kernel level (21 — full arch-B
retirement explicitly gated in the spec, see the item); v1 conformance
contract (22) shipped 2026-08-01 (corruption suite + LE golden
vectors + `docs/conformance-v1.md`; perf benchmark protocol is a
documented follow-up in that doc); doc drift (23) closed 2026-08-01
(every number re-verified against its bench/experiment source — the
0.008 ms headline was the pre-2026-04-05 reduced-shape `vindex_bench`
example; extract-default contradiction resolved in favour of the code,
per surface; walk.md K=8092 kept — it is the literal harness constant,
now documented as such — and WalkFfn reframed as the
instrumentable/editable layer + CPU sparse path); hygiene (24) closed
2026-08-01, triaged per its own licence — done: generic-engine
vocabularies → data files behind a loud-fallback search chain, the two
deferred 16384→10240 fixes, the activation dispatch (27 sites) onto one
exhaustive helper, FFN component constants unified, 41 colocated tests
for `hnsw.rs`/`mutate`/`write_f32.rs` (97/96/93% line coverage);
documented remainder: the >250-line file splits (see the item).
Standing follow-ups carried out of the programme: server/lql
`try_apply_patch` migration, remote transport coverage harness,
logit-contribution trace field, walk-FFN thresholds surfaced into
`WalkFfnConfig`, HNSW level-0 graph fragmentation at n≳64 (new finding
from item 24's test pass — naive `add_connection` eviction orphans
nodes; recall@10 collapses to 0.16 at n=200 uniform), and the remaining
file splits (`huggingface/download/mod.rs` 1329, `patch/overlay.rs`
1071, `quant/convert.rs` 653).

Sequencing is interaction-driven: Tier 0's padded-stride fix **gates**
Tier 2's base+delta (the delta path leans on the same row-dot/sidecar
machinery, and GPT-OSS-20B hidden=2880 is K3 rung 1); the
`forward`/`forward_observed` split *is* the fix for the zero-activation
bugs (don't patch them twice); the planner enum subsumes the
wrong-capability-gate class but the live panic gets its two-line fix now.

**Tier 0 — correctness (small independent diffs, before any Q4K walk
claim on a non-256-aligned model):**

1. ✅ **Q4K cache padded-stride fix + non-aligned fixture** (DONE 2026-07-30) (P0, silent
   garbage) — `kquant_cache.rs:138-161` decodes assuming unpadded
   `[rows, cols]`; the writer pads each row's cols to 256
   (`write_kquant/ffn.rs:70`). Wrong FFN outputs, no diagnostic, on
   hidden%256≠0 models (GPT-OSS-20B 2880, Gemma3-1B 1152). The fix already
   exists in one of three copies (`kquant_forward/walk_ffn.rs:63-70`).
   Add a hidden=320 fixture — every current Q4K fixture is 256-aligned so
   the suite structurally cannot catch this class. Victims: parallel-down
   path, per-feature down accumulate, selector row norms. [larql-vindex,
   larql-inference]
2. ✅ **Q4_0 ladder gates on the wrong format → CPU panic** (DONE 2026-07-30) (P0) —
   `walk_ffn/mod.rs:405` admits Q4_0 data on `supports_quant(Q4_K)`;
   `CpuBackend` says yes but leaves `q4_matvec_pair_batch` defaulted to
   `None`, and `interleaved_q4.rs:58-62` unwraps it. Gate on Q4_0 / actual
   batch-kernel availability; unwraps → fallthrough. `interleaved_q4.rs`
   has zero tests. [larql-inference]
3. ✅ **Overlay gate cache poisoned by zero-width gate vectors** (DONE 2026-07-30) (P0,
   nondeterministic panic/wrong-scores) — `patch/overlay.rs:176-191`
   mixed-width guard misses `len==0`; `vindexfile/mod.rs:125` inserts
   `vec![]` gates on every INSERT, so the trigger is in-tree. Guard the
   zero-width case AND stop inserting empty gate vectors. [larql-vindex]
4. ✅ **Loader panics on malformed `index.json`** (DONE 2026-07-30) (P0) — `format/load.rs:81`
   and `:293` index `gate_slices[info.layer]` unchecked from parsed JSON;
   return `VindexError::Parse` per the crate's own stated standard.
   [larql-vindex]
5. ✅ **Override fallthrough** (DONE 2026-07-30 — routes to the extracted override-aware `weights_fallback` instead of erroring; step 10 honours overrides, so availability is preserved) (P1, stopgap until base+delta) —
   `mod.rs:333-339`: sparse returning `None` on an overridden layer falls
   through to override-blind whole-layer paths — the exact failure the
   module doc warns about. [larql-inference]
6. ✅ **Unaligned f32 transmutes (UB) + patch decode swallowing** (DONE 2026-07-30 — new `format/le_floats.rs`; `try_apply_patch` is the error-surfacing entry, `apply_patch` kept as an infallible wrapper that drops corrupt patches wholesale; migrating larql-server/larql-lql callers to `try_apply_patch` is a follow-up) (P1) —
   `patch/format.rs:202`, `quant/convert.rs:565`, `config/dtype.rs:60` →
   `from_le_bytes`/bytemuck (also fixes the native-endian `.vlp`
   portability gap); `overlay_apply.rs:86,122` must surface
   `decode_gate_vector` failures instead of applying meta-only half-state;
   the hand-rolled base64 decoder silently truncates trailing chars.
   [larql-vindex]

**Tier 1 — kernel-semantics campaign (one PR neighbourhood: make explicit
what's exact, approximate, observed, reconstructed):**

7. ✅ **Wire `activation_floor`** (DONE 2026-07-30 — `effective_activation_floor()` = max(user floor, named `ACTIVATION_NOISE_FLOOR`), applied on all three sparse accumulate loops, behavioral test) — documented, settable from
   `predict_cmd.rs:241`, read by nothing; the real threshold is a
   hardcoded `1e-10` ×3 (`sparse.rs:338,411,549`). [larql-inference]
8. ✅ **Name the 80% full-K threshold, align doc/code** (DONE 2026-07-30 — `walk_ffn/thresholds.rs` FULL_K_DENSITY 4/5 + PARALLEL_DOWN_MIN_HITS + GATHER_MIN_FEATURES; helper doc now states the [80%,100%) band is dense) —
   `helpers.rs:24` fires the dense gemm at `k >= intermediate*8/10` while
   docs say "K ≥ feature count"; fidelity-vs-K points above 0.8 density
   are secretly dense unless `force_walk`. Named const in config; consider
   true `k >= intermediate`. [larql-inference]
9. ✅ **`selector:fallback` trace suffix** (DONE 2026-07-30 — dispatch-trace entry + `selector_fallback_count()`) — `joint_gate_knn` silently
   degrades to GateOnly when norms/batched scores are missing; A/B sweeps
   can't currently be trusted. [larql-inference]
10. ✅ **Resolve the gather caveat** (DONE 2026-07-30 — STALE: the phrase dates from task #24's transposed-down striding; task #25's hard sidecar requirement (`down_features_q4k_layer_data(layer)?` + decline-without-sidecar pin) resolved it, validated vs dense at |err|/‖ref‖≈6e-3. Caveat deleted, history documented in `sparse_gather.rs`. Remaining issue on this path is the documented 0.15× full-forward perf collapse, not correctness) — `sparse.rs:450` says "experimental —
    not yet correct for production down" on a kernel production routing
    reaches (route-pool + sidecar). Stale comment (predates the
    feature-major sidecar?) → delete; live → opt-in flag. [larql-inference]
11. ✅ **Unify the NaN contract** (DONE 2026-07-30 — shared `selection_weight_cmp_desc` panics on NaN matching `top_k_by_abs`; 4 sites unified, `#[should_panic]` pins incl. a NaN-gate-scores mock through `joint_gate_knn`) — `top_k_by_abs` panics;
    `selector.rs:267,320` `unwrap_or(Equal)` scrambles silently. Pick one
    (also see the 2026-05-28 item 5 shared helper). [larql-inference]
12. ✅ **Delete the orphaned `larql-vindex/src/walk/` module** (DONE 2026-07-30) — no
    `mod walk;` anywhere, never compiles, stale `WalkFfnConfig` duplicate
    (left by `3944359b`). [larql-vindex]
13. ✅ **Decide the HNSW hot-path question** (DONE 2026-07-30 — the exact-first ordering is DELIBERATE (`735f570e` 2026-04-04 call-site comment; brute gemv break-even-or-better at walk N per `docs/ffn-graph-layer.md`/`benches/hnsw_decode.rs`; HNSW's 80–95% recall would break the exact-top-K selection-quality gates) **but it had never actually executed**: `impl GateLookup for VectorIndex` was missing the `gate_walk` override — the trait default's "Override in VectorIndex" comment dates to the same 2026-04-04 commit — so every `&dyn GateIndex` walk selection silently took the `None` default into `gate_knn`, and `enable_hnsw()` DID leak approximate HNSW into walk numerics (pin test caught it: exact `[1,19,30,0]` became signed-biased `[1,29,9,26]` on the f32 fixture). Fixed by wiring the intended chain, not HNSW: delegation shim in `index/core/gate_lookup.rs` + guarded `PatchedVindex::gate_walk` (declines on gate-overridden/tombstoned layers so the overlay-aware `gate_knn` merge stays authoritative); Q4K-only gates and patched layers still reach `gate_knn_q4`/`gate_knn` as before, so the MoE-expert HNSW win is preserved. `enable_hnsw()` doc now maps exactly which paths consult HNSW incl. the 2026-04→07 leak window; pinned by `gate_walk_ignores_hnsw_toggle`, `gate_walk_delegates_to_inherent_on_a_populated_index`, the 3 `PatchedVindex` gate_walk pins, and `walk_ffn_sparse_hot_path_ignores_enable_hnsw`) — verified: `gate_walk` is tried
    first (`sparse.rs:231,268`) and HNSW lives only inside the `gate_knn`
    fallback, so `enable_hnsw()` changes nothing whenever `gate_walk`
    succeeds. Intentional (brute gemv wins at these N) → document at
    `enable_hnsw()`; otherwise wire it. [larql-vindex, larql-inference]
14. ✅ **Tombstone semantics for Delete→Update + pinning test** (DONE 2026-07-30 — Update resurrects, matching Insert; pinned-None meta cleared when Update carries no replacement; oversampling named `BASE_KNN_OVERSAMPLE_FACTOR`=2 with 2×→4×→all-features escalation only on layers with tombstones; 7 regression tests) —
    Update never clears `deleted` (`overlay_apply.rs:102-138`);
    `feature_meta()` and `gate_knn()` disagree about the same feature.
    Also the 2× deletion-oversampling under-fill (`overlay.rs:426`).
    [larql-vindex]

**Tier 2 — capability (the strategic-review core, in this order):**

15. ✅ **`forward` / `forward_observed` split** (DONE 2026-07-31 —
    `FfnBackend::forward_with_activation` is GONE; the trait is
    `forward` (hot, never touches an activation buffer) +
    `forward_observed` returning `FfnActivations` (new module
    `larql-compute/src/ffn/observe.rs`): `Dense` for dense paths (the
    matrix is an intrinsic intermediate), `Sparse` per-position
    `(feature, activation)` pairs for exactly the K computed features,
    `Absent {reason}` for paths that observe nothing — the trait default,
    so unobserving backends (remote walk's fabricated `[seq,1]` zeros,
    MoE's output-as-activation, seven larql-kv/server stubs) now say so
    instead of inventing tensors. `WalkFfn` routes both entry points
    through one `forward_routed(.., Observe)` body — identical routing by
    construction; `Skip` mode threads through every walk path
    (sparse/gather/parallel/base_delta/weights_fallback +
    `sparse_compute`'s split plain/`_observed` API) so the old
    `seq_len × intermediate` zero-fill no longer exists on generation.
    The parallel Q4K down branch reports its REAL per-feature activations
    (the pinned all-zeros parity test flipped to assert bit-equality with
    the serial halves); an L1 hit serves `forward` but an observed call
    BYPASSES the cache read and recomputes (pinned); base_delta reports
    post-patch slot activations (new `base_delta_tests.rs`, incl. decline
    branches — 20%→95% file coverage). `run_ffn`'s capture arm densifies
    via `FfnActivations::into_dense()` (Absent → `None`, never zeros), so
    hooks/trace/server consumers kept their `Option<Array2>` shape;
    changed files ≥90% line coverage except the pre-existing
    network-debt pair `remote/http.rs` / `remote/sharded.rs` (12%→35%
    with new no-shard observation pins; rest needs a mock-server
    harness)) — activations
    become opt-in; sparse paths emit `(FeatureId, f32)` pairs instead of a
    dense `seq_len × intermediate` zero-fill. Subsumes (by construction)
    the parallel-path zero activations (`sparse.rs:283-371`) and the L1
    cache's fabricated-zeros hit (`mod.rs:367`), and removes the dense
    allocation from ordinary generation. [larql-inference, larql-server]
16. **Base-plus-delta patched FFN execution** (after item 1) —
    `y_patched = y_base + Σ_{i∈P}(contribᵢ_new − contribᵢ_old)` is exact
    and O(|P|) on top of the fast dense path; retires the
    override-forces-sparse cliff and makes editing production-viable.
    Exactness conditions: old-term subtraction through the SAME quantised
    row_dot bytes as the dense base (not f32-recomputed), old-down rows
    from the feature-major sidecar. Lands as a routing-ladder branch, not
    a rewrite. [larql-inference, larql-vindex]
17. ✅ **Runtime trace emission** (DONE 2026-07-31 — the post-hoc
    `gate_knn` re-run is GONE: `with_trace` upgrades every call to
    `Observe::Record` and folds the executed path's observation into
    per-(position, layer) records at the routing-ladder exit (new
    `walk_ffn/trace.rs`), riding the item-15 seam rather than a parallel
    channel. `SparseActivations` entries carry the kernels' own gate/up
    scores (`record_scored`) plus per-position kernel labels, so
    serial/gather/parallel/weights-fallback report the values they
    actually computed (gather now returns its fused gate/up dots);
    records carry gate_score/up_score/activation/rank/path +
    residual_delta_norm (`‖out_row‖`); `‖down_row‖` is served only from
    the selector's prebuilt lazy norm cache, never computed for tracing;
    dense whole-layer paths emit the layer summary and decline
    per-feature records rather than fabricating. `take_trace` rebuilds
    the public `WalkTrace` from the runtime records — hits are the
    EXECUTED features, `WalkHit` extended additively
    (up_score/activation/down_row_norm/rank; post-hoc KNN views build
    via the new `WalkHit::from_gate` and stay honestly `None`) — and
    `take_runtime_trace` exposes full fidelity. Field names follow the
    chuk-introspect snake_case vocabulary; no dependency added. Pinned
    by `take_trace_reports_executed_route_not_gate_knn`: a pool route
    vs a decoy `gate_knn` — the trace must equal the executed route,
    which the old re-run structurally cannot return. Target-logit
    contribution needs lm_head access → documented out of scope in
    `trace.rs`) — replace `take_trace`'s post-hoc
    `gate_knn` re-run (`mod.rs:281-306`, which ignores selector/pools/
    cell-router and records scores, not contributions) with emission from
    the executed path: gate, up, activation, ‖down‖, residual-delta,
    logit contribution, rank, path. Align with the chuk-introspect schema
    — no second trace format. [larql-inference]
18. ✅ **Execution planner** (DONE 2026-07-31 — path selection is an
    explicit decision value: `FfnPlan` (new `walk_ffn/plan.rs`), one
    variant per ladder destination incl. `OverrideBaseDelta` as a plan
    variant per the freeze condition, names aligned to the trace_path
    vocabulary. Every variant carries a structured `PlanReason` —
    layer/seq_len/num_features/has_overrides, the `selected`
    condition, a `skipped` list stating why EACH higher-priority rung
    did not fire (base+delta declines name the exact failed
    precondition — `base_delta_preconditions` now returns
    `Result<slots, &'static str>`), and pre-execution `ThresholdCheck`s
    (requested K vs FULL_K_DENSITY, single-sourced from
    `hits_len_ge_intermediate` so `satisfied` honours `force_walk`).
    The planner (`planner.rs` `plan_layer`) is the ladder's ONLY
    condition source: `forward_ladder` plans, then `execute_plan`
    matches condition-free, and `forward_unpatched_whole_layer`
    (base+delta's base) iterates the same `WHOLE_LAYER_RUNGS` table.
    Try-then-fallthrough handled honestly: a path returning `None`
    mid-execution re-plans with that rung in a `PlanExclusions` set,
    and the executed plan's reason records "declined at execution" —
    pinned by a test where six lying capability flags each decline and
    the ladder lands exactly where the pre-planner code did.
    Inspection: public `WalkFfn::plan_for` (pure — L1 probed via new
    stats-free `FfnL1Cache::peek`, no dispatch entries, no execution);
    the runtime trace's `LayerTraceRecord` gains `plan_reason`
    (additive — `DispatchEntry`'s literal construction is pinned by
    routing tests). Routing is decision-identical: every dispatch/
    routing/trace test passes unchanged, same trace_path strings; the
    executed forward keeps exactly one L1 `get` per eligible call so
    hit/miss accounting is preserved. 20 planner tests (one per rung +
    decline-re-plan + purity); changed/new files ≥96% line coverage.
    Thresholds stay in `thresholds.rs`, REFERENCED by reasons —
    surfacing them into `WalkFfnConfig` is a tracked follow-up) —
    `VindexFfnPlan` enum + structured reason
    (plan/reason/layer/features/overrides), config-surfaced thresholds
    replacing the magic ratios; only freeze once base+delta exists as a
    plan variant. The ladder's trace_path names + routing tests are the
    seed; add the reason field. [larql-inference]
19. ✅ **Two-stage selection: shortlist top-M by gate, exact rerank**
    (DONE 2026-07-31 — opt-in `WalkFfnConfig::shortlist_m:
    Option<usize>` (+ `with_shortlist_m`; `None` = single-stage,
    default everywhere), consumed on the selector-dispatch route (new
    `walk_ffn/shortlist.rs`): stage 1 takes the top-M through the
    production `gate_walk` → `gate_knn_q4` → `gate_knn` chain (now
    factored as `production_gate_chain`, shared with the `GateOnly`
    route and the joint fallback — no new projection code); stage 2
    evaluates the configured criterion for ONLY those M candidates —
    per-candidate up dots via the per-row `ffn_row_dot`, norms from
    the existing lazy caches, O(M·d), never a full projection — and
    fully sorts to the final top-K (`rerank_cmp`: weight desc, feature
    asc on ties; the runtime trace's `rank` field is therefore the
    FINAL rerank order, and `joint_gate_knn` sorts its top-K by the
    same comparator so the two paths report identical order). The
    weight formulas are single-sourced in `criterion_weight` /
    `criterion_inputs` — `joint_gate_knn`'s inline per-variant
    closures were extracted onto them, so the full-projection and
    two-stage paths cannot drift. Hits keep the
    `(feat_idx, raw_gate_score)` contract; `shortlist_m` forces the
    per-position walk (like pools — the full-K gemv rewrite would
    bypass the structure); the Sparse plan reason records a
    `SHORTLIST_M` `ThresholdCheck` (actual=M, cutoff=K, satisfied =
    two-stage actually runs). M < K, `Random` (no criterion), or
    missing stage-2 inputs decline to single-stage OBSERVABLY — a
    `shortlist:declined` dispatch-trace entry +
    `shortlist_decline_count`, the M10 `selector:fallback` precedent.
    Pinned by 13 tests (`shortlist_tests.rs`): M=N two-stage ==
    `joint_gate_knn` (same features, same order, raw scores) for every
    scored selector; a huge-‖down‖/tiny-gate decoy the full-projection
    rerank picks but the top-M gate shortlist structurally excludes;
    observable declines; default-off bit-identical to single-stage;
    and the cost pin — a delegating index that PANICS on
    `gate_scores_batch`/`gate_scores_batch_backend`/
    `kquant_matmul_transb` runs a full two-stage forward clean, while
    its counting twin shows single-stage joint pays ≥2 full
    projections. Changed/new files ≥94% line coverage) — the
    rerank criterion already exists as
    `FeatureSelector::ActXUpScoreXDownNorm`; add the shortlist structure
    so it stops paying full projections. Production-cost shape of the
    existing experiment harness. [larql-inference]

**Tier 3 — productization:**

20. ✅ **Walk-vs-dense parity suite** (DONE 2026-07-30 — landed with the per-file 90% coverage pass: serial-vs-parallel, gather-vs-serial on a real sidecar, walk-vs-dense WeightFfn parity for gemv + exact/full_mmap/interleaved, dispatch-trace assertions against the REAL ladder in the moved dispatch_tests.rs; every walk_ffn file >= 90% line coverage) — the four tests that would have caught
    the four worst bugs: non-aligned Q4K fixture through cache + serial
    walk vs dequant baseline; CpuBackend + Q4_0 forward; serial-vs-parallel
    parity at hits ≥ 512 asserting output AND activation; dispatch-trace
    assertions against the REAL ladder (routing_tests.rs currently tests a
    hand-copied replica that can drift without failing). No test anywhere
    compares walk output against dense ground truth on a served vindex.
    [larql-inference, larql-server]
21. ✅ **KnnStore unification** (DONE 2026-07-31 — unified at the
    RETRIEVAL-KERNEL level; honestly short of full arch-B retirement,
    which is now explicitly gated in the spec rather than silently
    pending. The parallel scoring implementation is GONE: `KnnStore`'s
    private `key_matrices` GEMM + `dirty`-flag rebuild machinery is
    deleted, and its L2-normalized keys now live as rows in the new
    shared `patch/gate_overlay.rs::GateOverlay` — the same structure
    that holds `PatchedVindex`'s gate overrides — so `gate_knn` and
    every KNN query score through ONE kernel carrying the campaign's
    hardening (H3 zero-width guard, mixed-width slow-path fallback,
    per-layer snapshot cache). Mutators invalidate their own layer's
    snapshot, retiring the manual `invalidate_gate_cache*` calls (a
    forgotten-invalidation hazard class). What stays KnnStore-specific
    is POLICY, not machinery: entity/relation/target entry metadata,
    normalize-on-insert, rank-by-raw-cosine (vs `gate_knn`'s `|score|`
    merged with base hits — match the statistic to the operation).
    Public API, `.vlp` `InsertKnn`/`DeleteKnn` ops and the
    `knn_store.bin` format are unchanged; all five consumer crates
    (inference/lql/server/python/engine) compile untouched. Full
    "FFN = KNN index = vindex" (spec §3: appended-slot
    `AppendFeature`, delete the post-logits override) is NOT done —
    the FR1/FR2/early-exit routers (2026-06/07) shipped ON the
    post-logits override after the spec was written, and the α
    calibration (spec Q2) plus the 189-fact parity benchmark are
    unvalidated empirical work; `FFN_VINDEX_UNIFICATION_SPEC.md`
    rewritten to describe the post-unification reality and the
    remaining gate. Regression pins: query correctness after entity
    removal renumbers indices, clone-preserves-retrieval; changed
    files ≥90% line coverage in-crate) — still exported and live in
    `patch/knn_store_io.rs`/`overlay.rs`/`overlay_apply.rs`; the
    unification spec still describes it. Until removed, "FFN = KNN index =
    vindex" is partly aspiration. [larql-vindex]
22. ✅ **Vindex v1 conformance contract** (DONE 2026-08-01 —
    `crates/larql-vindex/docs/conformance-v1.md` + the pinning suite
    `tests/conformance_v1_{index,kquant,patches,down_meta,golden_le}.rs`
    (38 tests over the shared `tests/common/` fixture): every v1
    artifact × corruption class asserts error-not-panic-not-garbage —
    index.json (malformed/missing/wrong-typed fields, unknown
    dtype/quant tags, the H4 out-of-range-layer fix pinned as
    contract), interleaved_kquant slab+manifest (unknown format tag →
    Err; truncated slab / offset-length overflow / short manifest →
    checked_view decline; H1 padded-stride pinned at the writer),
    down_features sidecar (bin-without-manifest and missing shape[1]
    → Err, OOB → decline), .vlp (corrupt/truncated base64 → wholesale
    rejection, zero half-applied ops — M4/M5 pinned), .lknn
    (magic/version/truncation/absurd-count), down_meta.bin (truncation,
    checked-arithmetic overflow, allocation-bomb regression on both
    readers). Cross-platform: byte-level LE golden vectors for
    le_floats, .vlp base64, down_meta.bin, .lknn — exact bytes, not
    round-trip equality; no BE runner exists, the goldens are the
    guard. The two §3 LOW conformance violations fixed: legacy
    `down_meta::read_binary` now bounds every allocation by the real
    file size with checked arithmetic (mirrors `mmap_binary`; module
    split into `down_meta/{mod,read}.rs`), and the Vindexfile parser
    got quote-aware tuple splitting (`INSERT ("Acme, Inc", …)`),
    hard errors on missing/unknown/duplicate DELETE condition keys,
    and `find_free_feature().unwrap_or(0)` → error instead of
    silently overwriting feature 0; `.lknn` capacity hints bounded by
    remaining bytes as part of the same pass. Perf benchmark protocol
    is a documented follow-up in conformance-v1.md §4 (walk-vs-dense
    parity exists as item 20; no numbers faked). [larql-vindex,
    larql-vindex-spec]
23. ✅ **Doc drift** (DONE 2026-08-01 — every number traced to its
    source before editing. The `0.008 ms/layer` + `0.3 ms` 34-layer
    walk headline (repo README, vindex `operations-spec.md`) was the
    pre-2026-04-05 `vindex_bench` example at its reduced 1024×256
    synthetic shape ("reduced from 10240/2560/34 for bench speed"),
    scaled to 34 layers — replaced with the current criterion
    `vindex_ops` numbers at BOTH shapes (22.7 µs at 1024×256, 2.64 ms
    at the Gemma 10240×2560 production shape) plus an explicit
    exact-brute-gemv note: the walk hot path never consults HNSW,
    `enable_hnsw` is gate-KNN-consumers-only (item 13 inversion), now
    also stated in the crate README's interpretability recipe.
    Extract-level default contradiction resolved IN FAVOUR OF THE
    CODE: `larql extract` defaults to `--level inference`
    (`extract_index_cmd.rs:46`) while bare LQL `EXTRACT MODEL`
    defaults to browse (`lql parser/lifecycle.rs:17`) — the README
    table now says which default belongs to which surface, and the
    stale "add `--f16`" footer became "f16 is the default, `--f32`
    opts out". walk.md "Lossless at K=8092": NOT fixed by swapping
    8092→8192 — the 2026-04-03 boundary sweep, sparse.md and the
    remote-codec tests all literally ran K=8092 (the typo is baked
    into the harness), so the doc now says exactly that, notes
    8092 = 79% of 10240 stays genuinely sparse while K≥8192 hits the
    80% full-K dense rewrite (`thresholds.rs`), and date-qualifies
    the 97.91% figure (LQL-spec INFER example run, not the sweep).
    walk.md/ffn-README "production" framing reframed: WalkFfn =
    instrumentable/editable execution layer + CPU sparse path, Q4K
    GPU decode (~88 tok/s vs ~1.9 tok/s CPU INFER walk) is the perf
    centre; historical results kept, date-qualified. Campaign-sweep
    fixes: runtime trace emission + `new_with_trace` in walk.md
    (item 17), base+delta-first for patched layers in walk.md + the
    crate README W2 note (item 16), `gate_overlay.rs`/KnnStore
    GateOverlay-backed scoring in the crate README tree (item 21),
    `walk_ffn.rs` → `walk_ffn/` paths. No code changes.) [docs]
24. ✅ **Hygiene** (DONE 2026-08-01 — triaged per the item's own
    licence: worked in priority order, each piece fully or not at all,
    remainder documented. **(1) Generic-engine violations:** the
    English word lists (countries/languages/months/numbers + the
    148-word stop list) and the Wikidata category vocabulary are OUT
    of `clustering/` engine code and into `data/entity_patterns.json`
    + `data/stop_words.json` (+ the existing
    `data/wikidata_categories.json`), loaded through the new
    `clustering/data_files.rs` search chain — `LARQL_DATA_DIR` env dir
    → compile-time workspace `data/`, explicit config path via the
    `*_from(path)` loaders, NEVER cwd; `load_reference_databases`'s
    identical cwd-probe (`data`/`../data`/`../../data`) fixed with the
    same resolver; fallbacks are minimal built-in core sets and LOUD
    (stderr `warning:`). Bare `0.25` floor → `MIN_CATEGORY_SIMILARITY`;
    the "60%+" doc-vs-`0.5`-code pattern threshold resolved in favour
    of the code as `PATTERN_MATCH_FRACTION` (+ `MORPHOLOGICAL_MAX_LEN`);
    class order is data (language before country), pinned. Tests cover
    data-file loading, env-dir precedence, missing/invalid/empty-file
    loud fallbacks, threshold boundaries, and a behavioural
    similarity-floor pair through `auto_label_clusters_from_embeddings`;
    clustering files 93–100% line coverage. **(2) The two deferred
    item-23 16384 fixes:** Gemma 3 4B intermediate is 10240 (verified
    against `larql-models` `gemma3.rs:195`) — `docs/ffn-cache.md:46`
    now states the real sparse gate (below the 4/5 `FULL_K_DENSITY`
    rewrite ⇒ `top_k < 8192`, 8092 qualifies) and lql
    `insert/capture.rs:99` says 10240. **(3) Activation dispatch:**
    27 copies of the GeluTanh|Gelu → gelu-tanh-else-SiLU match (10
    `walk_ffn/` files, `sparse_compute.rs` ×3, `layer_graph/template.rs`,
    `kquant_forward/walk_ffn.rs` ×4 across larql-inference AND
    larql-compute, `cached.rs`, `ffn/weight.rs` ×4,
    `expert_weight/gate.rs`, 3 examples) now route through ONE helper,
    `larql_models::Activation::uses_gelu_tanh_gate_up()` — a
    wildcard-free exhaustive match (a hypothetical new variant is a
    compile error, not a silent SiLU landing; pinned by tests incl. a
    `#[should_panic]` for `Relu`, which has no kernel and no in-tree
    arch). Two silently-drifted copies found en route (`weight.rs`
    gated arms and `cached.rs` matched `GeluTanh` only, dropping exact
    `Gelu` to SiLU) are now consistent. **(4) Component constants:**
    `FFN_GATE`/`FFN_UP` added beside `FFN_DOWN` +
    `FFN_COMPONENTS_PER_LAYER`, pub in larql-vindex (crate-root
    export) and mirrored in `larql_compute::kv_index` with
    compile-time equality pins in `kv_index_impl.rs`; every bare
    `0/1/2` walk/kquant call site replaced (selector norms, sparse
    row-dot/scaled-add, sparse_parallel, `interleaved_q4`'s `* 3` →
    `FFN_COMPONENTS_PER_LAYER` + component-slice helper, both
    `kquant_forward/walk_ffn.rs`, and `base_delta.rs`'s local consts
    unified on the vindex ones). **(5) Colocated tests, 41 new:**
    `index/compute/hnsw.rs` 0 → 12 tests at 97.3% line coverage
    (insert/search, recall@10 = 0.97 vs brute force on clustered
    synthetic, level-RNG determinism with the LCG constants pinned);
    `index/mutate/mod.rs` 14 tests at 96.0% (meta/gate/override
    mutation, INSERT/DELETE-then-query, save→load round trips incl.
    mmap→heap promotion); `format/weights/write_f32.rs` 15 tests at
    92.6% (round trip through the f32 loader, MoE/MLA/BitNet writer
    branches, error paths). New finding pinned honestly rather than
    papered over: HNSW's level-0 graph FRAGMENTS as n grows — naive
    `add_connection` eviction orphans nodes (~33/200 BFS-reachable,
    recall@10 0.16 at n=200 uniform even with ef=n; fully connected
    ≤~64) — production gate-KNN at 10K+ features may be silently
    degraded; carried as a standing follow-up. **REMAINDER (documented,
    not done): (6) file splits** — `huggingface/download/mod.rs` 1329,
    `patch/overlay.rs` 1071, `quant/convert.rs` 653 still exceed the
    250-line rule (94/196 vindex src files over; `walk_ffn/mod.rs` is
    already down to 553 and `sparse.rs` to 861 via the Tier-2 sibling
    decompositions). Verification: larql-vindex 1296 lib tests +
    integration suites, larql-inference 1423 lib tests, larql-models /
    larql-compute / larql-lql all green; clippy + fmt clean on changed
    files; changed/new files ≥90% line coverage except
    `larql-models/src/config.rs` (63% file-wide pre-existing
    trait-default debt; the added helper's lines are 100% covered))
    — file splits (`walk_ffn/mod.rs` 926 → timings/ladder/
    builders; `sparse.rs` 842 → gemv/route/parallel/gather;
    `overlay.rs` 959; `huggingface/download/mod.rs` 1329;
    `quant/convert.rs` 655 — 88/186 vindex files exceed the 250-line
    rule); dedupe the 8-site GeluTanh/SiLU activation dispatch (new
    activations silently land in the SiLU arm); English word lists +
    Wikidata categories out of `clustering/` into data files (+ fix
    cwd-relative probing); colocated tests for `hnsw.rs` (455L, zero
    tests), `index/mutate/mod.rs`, `write_f32.rs` (777L); bare `0/1/2`
    component indices → `FFN_DOWN` et al. [larql-vindex, larql-inference]

### Extraction tensor-coverage audit + silent-drop follow-ups (2026-07-31)

Built the audit §4.6 work-item 2 asked for: every source tensor is classified
as **recognised** (an architecture accessor names it), **dropped by a named
rule**, or **unrecognised** — and the third bucket is loud.
`extract::coverage` + the `tensor_audit` stage, which runs *first* in
`build_vindex_streaming` so an unaddressable checkpoint fails in seconds
rather than after a multi-minute extraction. Reports always; fatal under
`LARQL_EXTRACT_STRICT=1`, which is now set in the `larql-vindex` CI workflow.

The case for it was five silent drops in one week, none caught automatically:
5 of 11 attention tensors (§4.6.1), 3 of 8 MLP tensors (§4.7), the
`gate_walk` trait default silently `None` (review item 13), a
`moe_intermediate_size()` defaulting to 0, and LayerNorm `β` — see item 3.

Validated on ten checkpoints: Qwen3-30B-A3B (18,867 tensors), OLMoE (3,219),
gpt-oss-20b, Gemma 3 4B (439 SigLIP tensors correctly classified
`non-text-tower`), all clean. **GPT-2 from HF safetensors: 1 of 160
recognised** — see item 2.

1. **Migrate `residual_diff` off process-global env vars onto the
   thread-local override.** `larql_compute::options::set_env_override`
   exists precisely to replace `std::env::set_var`, "which races concurrent
   `getenv` on the decode path and SIGSEGVs libc" — and all three dump sites
   already read through `options::env_value`, which consults it first.
   `run_with_dump_dir` / `run_with_two_env_vars` never adopted it; they were
   fixed on 2026-07-31 with a shared mutex, which is correct but serialises
   four ~110 s captures. Thread-local removes the shared state instead of
   guarding it. **Prerequisite:** the dump hook must read the var on the same
   thread that set it — true for the CPU path (`hidden.rs`'s own test relies
   on it), plausible for Metal encoding, but a read inside a rayon worker
   would silently stop dumping. Verify against the 4-model parity suite
   (~7 min) before switching. Needs an additive `clear_env_override(name)`;
   today only `clear_fast_path_overrides()` (clears all) exists.
   [larql-inference, larql-compute]
2. **GPT-2: rename or add an HF-safetensors variant.** `gpt2.rs` matches the
   trait defaults only *after* the GGUF→HF normalisation, so a raw HF
   checkpoint (`h.N.attn.c_attn.weight`, `wte.weight`, `ln_1.*`) is
   unaddressable. It fails late at the embeddings stage with a one-tensor
   message rather than silently, but 159 of 160 tensors are unreachable.
   Needs the `h.N.` prefix + `c_attn`/`c_fc`/`c_proj` spellings, and a drop
   rule for `h.N.attn.bias` (the causal mask — a derived constant, not a
   weight, so it belongs in `coverage::rules`). Unblocked by item 3.
   [larql-models]
3. **Verify the restored LayerNorm `β` numerically.** No accessor named a
   norm bias until 2026-07-31, so extraction never wrote one and
   `build_pipeline_layers` hardcoded `input_norm_bias: None` — while the
   Metal `layer_norm` shader implemented `+ bias` and always took its
   no-bias variant. The CPU dense path got away with it by mangling the
   weight key, which is why raw-safetensors inference was right and every
   vindex-backed path dropped the shift, for **GPT-2 and StarCoder2**. Now
   declared, extracted and resolved; the honest status is "the tensor flows
   end to end", not "the output is correct". Wants a GB-shaped measurement.
   [larql-models, larql-vindex, larql-compute]
4. **Consumption-level coverage audit.** The current audit measures
   *naming*, not consumption — a recognised tensor is one extraction *can*
   reach, not one it wrote. Naming is where all five drops actually lived,
   so this closes the bug class that has bitten; recording `WeightSource`
   reads would subsume it and also catch "named but never asked for".
   [larql-vindex]
5. **`capture.rs` cannot reach the 90 % floor in Linux CI.** ~250 of its 408
   lines are `metal_decode` / `metal_decode_steps` / `metal_prefill`, which
   need a Metal device by construction — which is why the file sits outside
   the crate's `include_globs`. Raised 34 % → ~50 % by testing `cpu_prefill`
   for the first time, plus a macOS-gated `metal_prefill` test so the
   constructor is exercised somewhere. Either accept the exclusion
   permanently and say so in the policy note, or split the GPU dispatch from
   the dump-readback logic so the latter is testable everywhere.
   [larql-inference]
6. **`named_keys` is hand-maintained against 62 trait accessors.** A new
   `*_key` accessor not wired into `collect()` makes its tensors report as
   *unrecognised* — noisy, never quiet, and the pin test catches it at the
   source. It has already fired twice for real (`moe_post_ffn1_norm_key`;
   then the three norm-bias accessors). If the accessor count keeps growing,
   consider deriving the list rather than pinning a count.
   [larql-vindex]


### K3 R1 P4/P5 CLOSED — GPT-OSS served from its vindex; Metal decode 10.2 → 59.8 tok/s (2026-08-09/10)

Write-up [`docs/k3-funnel.md`](docs/k3-funnel.md) §4.11. Registry
`k3r1-gptoss-pipeline`. Branch `feat/k3-r1-p5-moe-bias-gate-policy`, commits
04addd27…f557cec9.

**Serve is real on both backends.** `larql run gpt-oss-20b-q4k.vindex`
generates coherent harmony-format output on CPU at ~60 ms/token, and `--metal`
produces the **identical 32-token greedy trajectory at 16.7 ms/token
(59.8 tok/s)** — parity re-verified after every rung. Correctness took six
stacked hidden%256 defects (MoE norm topology, lm_head widths ×2, Metal QKV/O
stored-row width, missing Metal attention biases, dense-FFN encode over empty
pure-MoE slices); the fixes are all generic — `QuantWeight::stored_cols`
(byte count is the width authority), bias threading via `build_arch_params`,
`has_dense_ffn()` as a representation fact.

**The decode ladder, each rung parity-gated:** 97.8 ms (staged expert
memcpys, ~2.1 GB/token) → 25.9 (zero-copy mmap regions:
`BufferCache::register_region` + experts as byte offsets) → 22.7 (K3a grouped
expert kernels, η 0.64→0.90 as measured) → 22.6 (fused no-QK-norm attention +
folded QKV biases — attention GPU 8→3.3 ms with the wall unmoved, isolating
the sync term) → **16.7** (GPU `moe_weighted_combine`; a layer's experts and
the next layer's attention share one command buffer — one wait/layer).
Gemma 26B A4B hybrid rode the first two rungs free (91.3 → ~30 ms/tok).

**Benchmark framing (pinned):** oMLX's ~85-90 tok/s reference is **native
MXFP4** (~4.25 bpw experts); LARQL carries the lossless Q6_K transcode at
6.56 bpw — ~1.54× the expert bytes. 59.8 × 1.54 ≈ 92 byte-normalised: same
conventional-efficiency territory; the residual is representation, not
runtime. The MXFP4-native experiment (shaders exist) now measures how far
*past* the reference the engine goes.

**Remaining budget at 16.7 ms:** ~11.5 MoE (bandwidth), ~3.3 attention GPU,
~2 sync residue (24 waits; GPU routing + device-side offset tables — which
the grouped kernels already read — take it near zero).

**Standing lesson earned:** faster kernels ≠ faster decode when
command-buffer structure dominates the wall; measure the wall against the
CB-window sum before optimising a kernel.

**Follow-ups (tracked here, not just in PR #241's notes):**

| # | Item | Crate | Status |
|---|---|---|---|
| P5-F1 | **Native-MXFP4 expert execution — DONE (2026-08-14).** Served composed: `--routed-from` a VINDEX3 container carrying the checkpoint's own MXFP4 payload + paired e8m0 streams; expert-bank authority override is generic (no format branch in the plumbing). Paired same-session ladder: Q6_K 68.3 → MXFP4 74.6 → vectorised kernel **77.2 tok/s**. The first-order ~78 budget was honest: the ~2.6 ms byte-projection assumed equal kernel η, and the measured MXFP4 η deficit (skeleton, not decode — fixed for ~0.46 ms by `uint4` loads) accounts for the rest. See "Native MXFP4 served" below. | larql-compute-metal, larql-vindex | **done** |
| P5-F2 | **GPU routing + device-side offset tables — DONE (2026-08-13, PR #249).** Router + top-k + descriptor gather as GPU dataflow; one command buffer per token; route-witness counters (host_resolves / bias_copies / weight_binds / offset_binds) pinned at zero by gates and printed by the composed serve. Opt-in via `LARQL_GPU_ROUTE=1`. 59.8 → 67.5 tok/s at landing. | larql-compute-metal | **done** |
| P5-F3 | **`predict_kquant_metal` panics on pure MoE** ("ffn Q4K slices missing for layer") — the max-tokens-1 predict path is a separate dense-assuming forward; either route it through the MoE-aware decode or refuse with the typed capability error. | larql-inference | open |
| P5-F4 | **`metal_decode_synthetic` parallel-run flake** — pre-existing (clean-tree reproducible, ~2/5 parallel runs, victim varies, prefill NaN; single-threaded always green; CI blind — runners have no Metal device). Cross-test interaction under concurrent GPU load, mechanism unidentified; suspect list starts at pooled-buffer recycling vs in-flight command buffers under parallel backends. Until fixed, local red on this suite means: re-run single-threaded before diagnosing. | larql-compute-metal | open |
| P5-F5 | **GB-style bits/token scoring of the served vindex** — `shannon score` still can't load vindexes (raw-model only), so the serve path's quality gate is greedy-trajectory parity, not measured bits/token (§4.6.8's scorer gap, still open). | larql-cli, larql-inference | open |

---

### Native MXFP4 served end to end — 77.2 tok/s, every remaining millisecond named (2026-08-14)

Branch `feat/moe-g4b3-region-registration` (commits `ba7418b9`,
`1f0a5909`, `e9af233d`). Three results, one paired same-session A/B/A
ladder on GPT-OSS-20B / M3 Max (warmup 16, n 256, long prompt; battery,
idle — an AC-protocol repeat is owed for the books):

| ms/token | tok/s | What changed |
|---:|---:|---|
| 14.63 | 68.3 | Q6_K control through the GPU route (reproduces the recorded 67.5) |
| 13.41 | 74.6 | **native MXFP4 expert banks** — `--routed-from` composes a VINDEX3 container's payload + paired e8m0 streams over the Q6_K spine; expert reads 1 959 → 1 269 MB/token |
| 12.95 | **77.2** | **vectorised split kernel** (`mxfp4g_split_lut16_vec`, arm `a2`, now default) — `uint4` group loads replace 16 single-byte loads; oracle-exact; 16-byte-alignment precondition checked per bank with loud scalar demotion |

**What each step proved, and what it didn't:**

- **G4b.3 (serve seam).** VINDEX3 segments are now mmapped (page-aligned
  by construction) and registered with the Metal buffer cache beside the
  spine's packed mmaps. Fired evidence, all in one run: 24 layers
  overridden as MXFP4/SplitE8M0, `GPU_ROUTE_LAYERS = 24 × forwards`
  remainder 0, all four route-witness counters zero, cmd_bufs/token 1,
  output byte-identical to the legacy arm. Five tamper controls (wrong
  declared format / missing scale pair / short scale pair / payload one
  group short → refuse; untampered + the real 9.5 GB container → admit)
  — composed open now derives paired scale-partner sizes from the
  container's declared format, so moving byte authority did not weaken
  validation. Trap worth recording: the GPU route is opt-in
  (`LARQL_GPU_ROUTE=1`); unset, the run falls back healthy-looking at
  cmd_bufs=25.
- **G5 attribution.** Measured grouped-kernel bandwidth at production
  geometry predicts the expert-read delta within ~0.1 ms of e2e — no
  integration tax; the old −2.6 ms projection had assumed equal η.
  Instrument caveat, learned the honest way: isolated tournament η at
  17–35 MB working sets is SLC-confounded (arm-A down η read
  0.36/0.57/0.65 across three runs); within-run ordering is stable, the
  paired e2e numbers are the claim-bearers.
- **TOKEN-B1 rung 1 (lm_head).** `q4k_matvec_topk` fuses the Q4_K
  matvec with the partial top-K reduction in one submission (8 KB back
  instead of ~800 KB + a 201K CPU scan). e2e-neutral — readback was
  never the cost — but it pinned the stage: **~1.09 ms is Q4_K matvec
  at the bandwidth ceiling; ~0.5 ms is the CB boundary/host round trip.**

**Open next (in order):**

| # | Item | Crate | Status |
|---|---|---|---|
| B1-2 | **Fold final norm → lm_head → top-K into the decode CB** — single wait per token, partials-only readback; the measured ~0.5 ms boundary is the prize (77.2 → ~80; MLX ≈ 83 sits ~0.9 ms away in total). Acceptance: token parity, fired-path witness, no hidden readback before lm_head, then the paired bench. | larql-compute-metal, larql-inference | open |
| K1-2 | **Cold-fixture kernel instrument** — rotate expert banks ≫ SLC (the `bench_mxfp4_dequant` discipline) so isolated MXFP4 η is claim-bearing; then ask whether `a2` leaves more on the table at the down shape. | larql-compute-metal | open |

---

### Attention execution ladder — #229 closed, KV-B1 licensed, VINDEX3 carries it, geometry planner (2026-08-16)

One day, three rungs closed and one bug that had been impersonating five
others. Full record: [`docs/kv-attention-scaling.md`](docs/kv-attention-scaling.md)
(§"The fault is on main and predates seqpar", §"Glimmer geometry — the policy
becomes a planner"). PRs #262 #263 #261 #260 #264 #265 #266.

**#229 closed (#263).** The intermittent startup-integrity fault (EOS at token
1, token ids past the vocabulary, "decode" at 0.5 ms/token over a full budget,
wrong-but-plausible trajectories, hangs, one NaN row) was one bug: the routed
decode path sized sliding-layer KV at `window × 2 = 256` rows — a residency
policy that presumes compaction — and never compacts, so from position 256
every sliding layer wrote and read past its buffers by a margin growing with
position; the corruption stayed self-consistent until another allocation
shared the memory, then NaN → router `~0u` ids → `moe_descriptor_gather` page
fault → failed command buffers that nothing checked. Fix: every layer at full
`max_seq` on that path; `encode_kv_append` refuses a full cache. Containment
kept regardless: `cb_status::wait_checked` on all 101 waits (a test forbids
naked waits), the MoE entry seam refuses a step whose buffer failed, the
prompt pass propagates a failed step instead of substituting zeros, the
gather bounds its ids. Acceptance 16/16 fresh processes clean, `run` output
byte-identical across processes. **Lesson: a residency policy and the
operation that makes it valid must be pinned together.**

**KV-B1 licensed (#264).** Bracketed A/B/C ladder on one binary, one session,
`off / default / off`, 300 s rests: A 12.04 → 10.80 ms/token (+11.5%, brackets
0.08%), B 13.71 → 11.01 (+24.5%, 0.51%), C 18.11 → 11.88 (+52.4%, 0.55%).
Serial 12.0 → 13.7 → 18.1 with context; seqpar 10.8 → 11.0 → 11.9 — the slope
claim. `SEQPAR_DEFAULT_ON_HEAD_DIMS = &[64]` shipped.

**VINDEX3 carries it (#265) and the policy became a planner (#266).** The
lowering's attention now tiers short/long (it had pinned the short kernel and
overflowed its threadgroup scratch past span 1024) and dispatches KV-B1
seqpar behind the shared policy, with a route witness (`serial N / seqpar M`)
printed by `larql vindex3 exec --generate`. The hd-64 policy did **not**
transfer to Glimmer (32 q-heads, 2 KV heads, head_dim 128), so the policy
moved into `ops/attention_geometry.rs`: `(head_dim, q_heads, kv_heads, span)
→ Serial | SeqPar { slices }` over measured rows, no model names, serial
where unmeasured; the gpt-oss row reproduces the KV-B1 tiers (pinned by test).
Glimmer's row, from the rested `off/2/4/8/off` surface (ms/token, serial
brackets, identical ids on every arm, witness on every row):

```text
ctx    serial   2     4     8    serial   bracket   verdict
512      73    70    69    68      77      5.5%    direction only → serial
1024     88    83    83    79      89      1.1%    near-valid: 8 slices +12%
2048    116    96    92    85     113      2.6%    lower bounds +18 / +23 / +33%
4000    144     -   136   134     145      0.7%    VALID: +6.2% / +7.8%
```

→ serial below 1024, `SeqPar { 8 }` (the 1024-thread ceiling at head_dim 128)
from 1024. Default policy at 2K: witness splits at position 1024, **116 → 92
ms/token**, same trajectory. Glimmer serial curve: 61 / 73 / 88 / 116 / 144
ms/token at 128 / 512 / 1K / 2K / 4K.

**Method notes that paid for themselves today:** fresh-process sampling
(`--repeat` is N correlated draws of one process); a per-process watchdog (a
hang prints no row); rest before every arm (unrested arms drift monotonically
in *time* and read as a slice-count cliff: 76 → 159 across nine minutes);
route witnesses so performance and routing are separate assertions; and no
timing claim from an evening's drifted machine — the planner's decode-path
query was verified as `{64,64,8,1025} → 16 slices` and pinned by test instead.

**What's next (order agreed 2026-08-16):**

| # | Item | Condition that closes it | Status |
|---|---|---|---|
| A-1 | **What dominates Glimmer past ~2K?** B1's gain collapses (+33% → +8%) although 39/52 layers stay at their 2048 window. Hypothesis: 16:1 GQA read amplification — 16 query-head threadgroups per KV head each stream that head's whole K/V (~2 GB/token at 4K), which no intra-TG slicing touches. Discriminating experiment: synthetic attention bench, 32 per-head TGs vs 2 KV-head groups serving 16 queries from one K/V stream, at 2K and 4K. | The grouped form barely moves 2K and opens 4K (hypothesis holds) — or does not (falsified cheaply, before any kernel machinery). | open |
| A-2 | **GQA-group attention kernel** if A-1 holds: one threadgroup per KV head serving its query group, then composed with seqpar / B2. Planner gains a `GqaGroup` geometry. | Bracketed Glimmer ladder 1K/2K/4K, same oracle trajectory, witness. | blocked on A-1 |
| A-3 | **Re-certify Glimmer under the planner** — same oracle fixture (argmax 13796; interpreter parity), same guards; that becomes the engine baseline. | 32/32 id parity, no `[metal]` lines, tok/s on a rested bracket. | open |
| A-4 | **Close the quant chain through the planner-aware lowerer** — Meta fixed KQuant and KQuant-dynamic alongside NVFP4 and the MXFP4 control. | Representation/Pareto branch declared closed. | open |
| A-5 | **Small projection geometry** (Q/K/V/O GEMV) — the remaining conventional kernel weakness; the 20–21 → 22–23+ tok/s move. | Kernel-level then e2e bracket. | open |
| A-6 | **(layer, role) → representation policy** from the Meta recipes + quality matrix. | Bytes down without the coarse FFN-only cost. | open |
| A-7 | **Speculative decoding last**, entering at ~22–24 tok/s native. | — | open |
| A-8 | **B2 (sequence tiles across threadgroups)** once B1's intra-TG width is spent — at head_dim 128 that is 8 slices. | Planner rung. | after A-2 |
| A-9 | **gpt-oss through the same lowerer** — the proof it is an engine architecture, not a Glimmer implementation. Mapped 2026-08-17: `larql vindex3 plan` on the HF checkpoint is inadmissible on four facts, and that is only rung 0 of six (below). Oracle banked: `bench/prompts/gpt-oss/vindex3-oracle-2026-08-17.txt` (65 chat-wrapped prompt ids → 16 generated ids, via `larql run --emit-ids`, #268). | Same oracle discipline on gpt-oss: byte-identical greedy ids to the served path through `vindex3 exec --backend metal-lowered`. | **closed 2026-08-18** (parity chain, rungs A-9.0–A-9.5 below); the bracketed ladder is A-10 |
| A-10 | **Cost of abstraction — gpt-oss bracketed perf ladder** through the generic plan. Arms: served routed Metal path (`larql run --routed-from`) · `vindex3 exec --backend metal-lowered` f16-attention · the same with NVFP4 attention; contexts short / ~512 / ~1K / ~2K / ~4K; baseline/candidate/baseline brackets, full decode budget, route/plan witness on every row, same trajectory fingerprint, rests between arms (the KV-B1 hygiene). The question is the delta between the generic lowering and the mature served path — single-digit % says the execution architecture generalised without discarding the optimised backend; larger says there is a clean, correctness-free profiling problem. | A rested bracketed table with witnesses; a stated delta per context. | open, next |

A-9's rungs, each with its file coordinates (from the 2026-08-17 map; the
generic system-graph path is **dense-only today** — Glimmer encodes because it
is dense on that path, and the two container shapes, `system_graph.json` vs
`moe_manifest.json`, are mutually exclusive: `encode/mod.rs` sets
`index.moe_manifest = None`):

```text
A-9.0  DONE 2026-08-17 (branch feat/vindex3-gpt-oss-rung0): `larql vindex3 plan` on the HF
       checkpoint is admissible — 50 representable / 0 mismatched / 4 unrepresented (all
       alias or training_only) / 0 blocking; PositionPolicy::Yarn{theta 150000, factor 32,
       β 32/1, orig 4096} carried, amplitude derived by ONE authority
       (YarnRopeScaling::attention_amplitude); ffn.gate_policy = ClampedGlu{7, 1.702};
       decoder_stack encodings BF16+MXFP4. Executors and the lowering REFUSE Yarn and
       ClampedGlu (typed, before any bytes) rather than serve the wrong model. Two finds
       the gate forced: the parser read theta from transformers-5 `rope_parameters` but
       SCALING only from legacy `rope_scaling` (a 5.x YaRN block was dropped at parse —
       §4.7.8 again; fixed, pinned); and the YaRN leaves had carriage rules but were not
       classified execution-semantic, so the rules were never reached. As mapped:
       plan admissibility as REPRESENTATION, not suppression:
       PositionPolicy gains a YaRN variant carrying {theta, factor, beta_fast, beta_slow,
       truncate, original_max_position_embeddings, amplitude} — amplitude is the part that
       moves every logit and lowering/mod.rs:397 fakes 1.0 today; probe_rope_type reads it
       (plan/carriage.rs:281, config/position.rs:25). swiglu_limit → an explicit gate policy
       on FfnSurface/FfnOp mirroring ExpertGatePolicy::ClampedGlu{limit, alpha}
       (graph/surface.rs:91, opplan/mod.rs:116, carriage rule at carriage.rs:110; the test
       at plan/tests/carriage.rs:86 already says "MOE1 is where it gets one").
       quantization_config.{quant_method, modules_to_not_convert} classified (semantics.rs
       registries / config_keys.rs:116) as the representation fact they are.
A-9.1  DONE 2026-08-17 (branch feat/vindex3-gpt-oss-rung0): judged AttentionSinkSpec::
       SoftmaxDenominator (models crate, derived from attn_sinks_key so the schema fact and
       the served kernel cannot disagree) + `attention_bias` on AttentionSurface; roles
       Attn{Q,K,V,O}Bias/AttnSinks; closure paired both ways (declared ⇒ all four operands,
       operand ⇒ declaration; sink operand ⇒ judgment, judgment ⇒ operand; shapes pinned);
       AttentionOp.{q,k,v,o}_bias/sinks → BiasCall/SinkCall → reference (append-and-drop
       softmax), production + device (served `softmax_in_place(scores, sink)`), lowering
       REFUSES until A-9.4. Gates: golden parity (denominator form, third transcription),
       3-backend parity, 5 causal controls (each operand moves layer-0 attention 1e-2…6e-1,
       propagates to logits), absence (bias-free plan serialises byte-identically; four
       fail-closed refusals). gpt-oss `vindex3 ops`: 384 → 264 defects, ZERO attention
       defects, every survivor MoE (A-9.2). As mapped: attention sinks + q/k/v/o biases in
       AttentionSurface / AttentionCall and every backend — the reference interpreter had
       neither (reference.rs:165 softmax without the sink, :92 projection without bias); the
       lowering still binds has_sinks=0 with a placeholder (lowering/attention.rs:344).
A-9.2  DONE 2026-08-17 (branch feat/vindex3-gpt-oss-rung0): ObjectKind::ExpertBank, carved out
       of the stack by binding SPECIFICITY (`most_specific_owner`: longest binding prefix owns
       a tensor — the graph's one membership rule, consulted by encode/verify/closure; no
       exclusion lists, no manifest); FfnSurface.moe (MoeSurface lifted from the family
       judgment: experts, top_k, router_kind, routing_policy, router_bias, expert_format,
       gate_up_layout, …); roles MoeRouter{Weight,Bias} (stack) + Expert{GateUp,Down}
       {,Scales,Bias} (bank); LayerFfn::{Dense,Routed} untagged so dense plans serialise
       byte-identically; RoutedFfnOp → RoutedFfnCall → reference (literal transcription),
       production (served router::select + MoeGateRule), device (per-expert gemv, NATIVE
       MXFP4 bytes bound when declared); lowering refuses (A-9.4). Gates on a miniature
       gpt_oss-FAMILY fixture (packed MXFP4 + biases + sinks + router bias + clamped GLU):
       served CPU forward ≡ reference ≡ production ≡ device(mxfp4) at 2e-5; four causal
       controls (router / gate_up / down / bias → 3e-2…5e-1); closure fail-closed both ways
       (no judgment ⇒ 8 stray operands/layer; wrong expert count ⇒ 8 geometry/layer; expert
       operand in the stack ⇒ misplaced). REAL gpt-oss: `vindex3 ops` 264 → 0, "plan closed:
       24 layer(s), every executable operand accounted"; container = decoder_stack@BF16 1.28 GB
       + expert_bank@BF16+MXFP4 10.17 GB (24 bindings) + embedding/final_norm/output_head, NO
       moe_manifest.json, `inspect` coherent; `vindex3 exec` now runs to the executor and
       refuses at YaRN — the last unexecuted semantic (A-9.3). As mapped:
       MoE in the generic system graph: an expert-bank ObjectKind (graph/object.rs:11),
       router/expert/bias OperandRoles (graph/roles.rs:22,58), an MoE FfnOp; encode writes
       banks into the system container (or exec composes system + routed containers the way
       `run --routed-from` composes VINDEX2 + banks).
       SUCCESS CRITERION (2026-08-17, stronger than "zero MoE defects"): an MoE is
       expressible ENTIRELY inside the object/operand/op-plan universe dense attention and
       FFN use — no second container semantics, no side-loaded expert manifest execution
       must reinterpret; `moe_manifest.json` disappears or shrinks to physical layout.
       Shape: DecoderLayer → FfnOp::Moe { RouterWeight, RouterBias, ExpertBank(ObjectKind)
       { gate/up repr, down repr, biases, quant auxiliaries } }. The judged MoE vocabulary
       already exists in format/moe_manifest (Router{activation, selection, post}, Reduction,
       Combine, Programme, BankRef{experts, expert_dims}) — A-9.2 lifts it into the graph,
       it does not re-judge it. Gates mirror A-9.1: bidirectional closure (declared MoE ⇒
       every operand; stray MoE operand ⇒ the op), shape/count closure (router [E,H], E,
       top-k, packed geometry gate_up [E,2I,K/32,16]+[E,2I,K/32] scales, down [E,H,I/32,16]),
       causal controls (perturb router weight, one expert gate/up, one down, one bias →
       output moves), absence (dense FFN plan byte-identical), container universality (no
       executor reaches outside the generic graph to find a bank), real gpt-oss closure
       264 → ~0 (only genuinely later-rung semantics may survive).
A-9.3  DONE 2026-08-17 (branch feat/vindex3-gpt-oss-rung0): the interpreter executes YaRN —
       kernels::rope_rotate_scaled + kernels::yarn_frequencies (reference transcription) and
       the served rope_freq_plan(Yarn) on production/device; MoE execution landed with A-9.2.
       Gates: fixture with GPT-OSS's YaRN block, served ≡ reference ≡ production ≡ device at
       2e-5; persisted `factor` 32→4 moves position 0 (amplitude) and layer 0. REAL gpt-oss:
       `vindex3 exec` runs end-to-end (24 layers, 65-id prompt: 20.8 s single forward, 148
       ms/token prefill, 7.5 tok/s CPU decode); argmax = oracle's first id; `--generate 16`
       matches a 3-id prefix then diverges at step 4 and re-syncs; TEACHER-FORCED over the
       oracle's 81 ids: 14/16 per-position argmax agreement, both misses at the two lowest
       margins (+0.47, +0.12 logits; oracle id ranked #2). CONGRUENCE: the banked oracle is a
       Q4_K-SPINE served run; the container is BF16 attention + native MXFP4 — so this is
       "same semantics, different representation", NOT the A-9.5 byte-identical claim, which
       needs a same-representation oracle (served path over the same BF16 spine bytes).
       As mapped: MoE execution in the interpreter (exec/mod.rs, reference.rs, production.rs) with
       gpt-oss routing (top-k then softmax over the selected — MoeRouterKind::TopKThenSoftmax,
       ExpertRoutingPolicy::NormalisedOverSelected) and ClampedGlu (pipeline/moe.rs:43).
A-9.4  DONE 2026-08-18 (branch feat/vindex3-gpt-oss-rung0): the real gpt-oss-20b container
       lowers END-TO-END through the generic Metal path — routed MXFP4 MoE + YaRN + sinks +
       biases, no family-name branching. Routed FFN: LayerFfnLowering::{Dense,Routed} in the
       stack encoder; lowered.rs registers the container's expert-bank segments as zero-copy
       regions (AlignedBytes page-aligned == PAGE_SIZE 16384) and builds a MoeLayerWeights
       (router from decoder_stack, experts from expert_bank, top_k_then_softmax, Interleaved,
       Paired MXFP4 scales, ClampedGlu via MoeGateRule) → the SERVED descriptor MoE encode
       (encode_moe_layer_gpu_route) — ClampedGlu executes through the existing clamped_glu_bias
       kernel, so no new gate. `vindex3 exec --backend metal-lowered` on the 65-id oracle
       prompt: argmax 200005 = oracle id 1; `--generate 16` = 15/16 IDENTICAL to the production
       interpreter's own trajectory (one flip at position 6, NVFP4 attention vs f32, re-syncs
       immediately), 18.5 tok/s. Fragment gate: tests/test_lowering_attention_extras.rs (YaRN+
       sinks+biases <1e-4 + 4 controls); routed FFN reuses the tested served MoE encode. 821
       Metal tests green. REMAINING for a per-layer congruent gate: wire lowered `--dump-layers`
       (encode_stack Checkpoints, per-position [seq,hidden] planes) so `shannon layer-diff` can
       localise the lowered vs interpreter/served dumps to a first-differing layer (whole-model
       argmax+trajectory already agree within NVFP4-attention noise). Was:
       ATTENTION HALF DONE 2026-08-17 (branch feat/vindex3-gpt-oss-rung0): sinks, Q/K/V/O
       biases and YaRN (scaled inv_freq + amplitude) lowered in lowering/attention.rs +
       mod.rs, reusing the existing bias_add / sinks-slot(10,11) / rope-amplitude(slot 6)
       kernels; lowered.rs feeds them from the plan and lifts the three refusals (LoweredMatrix
       gains Scaled{theta, amplitude}; per-(theta,yarn) inv_freq table; resident bias/sink
       vectors). Gate: tests/test_lowering_attention_extras.rs — lowered ≡ reference (biases +
       YaRN + sink) < 1e-4, with four controls (drop sink / drop Q bias / amplitude→1 / ramp→
       plain rope) each moving output ≥20× the parity residual; all 421+ Metal lowering tests
       green. REMAINING (routed FFN): the lowered stack still refuses a routed layer, so
       gpt-oss does not yet lower e2e. moe_zero_copy.rs / moe_gpu_route encode the served MoE
       into one CB but resolve experts through registered mmap REGIONS (BufferCache::
       register_region → resolve_region); the lowering loads operands as owned bytes, so the
       routed rung must register the container's expert-bank segment as a region and thread a
       MoeLayerWeights-shaped call (router in stack, experts in bank) through encode_stack —
       then layer-diff the lowered dump against the served/interpreter dumps from A-9.5.
       As mapped: moe_zero_copy.rs is the served path's machinery and is not plan-reachable.
A-9.5  the parity chain: ops closure → exec reference → production → lowered, byte-identical
       to the banked oracle; then the bracketed ladder.
       CLOSED 2026-08-18: the three-way chain holds on the real gpt-oss-20b. `vindex3 exec
       --dump-layers` now works on the Metal-lowered path (encode_stack Checkpoints capture
       every layer per position into [seq,hidden] planes, byte-compatible with the interpreter
       dump). Over the same 81 ids, `shannon layer-diff` metal-lowered-ffn (f16 attention/head,
       native MXFP4 experts) vs the production interpreter: cos 1.000000000 and rel_rms ≤ 1e-6
       through ALL 24 layers, "no capture drifts" — the same congruence as served-vs-interpreter.
       This MEASURES the earlier generated-token flip: with all-NVFP4 attention the layer-0
       rel_rms is 0.058 (cos still 0.998, direction preserved), and swapping attention to f16
       collapses it to 0.000000 — the drift was the NVFP4 attention weight representation, not a
       semantic defect. Stated at the strength the measurement licenses: any divergence
       attributable to the common MXFP4 expert path is BELOW the ~1e-6 residual of the
       congruent f16-attention run — the end-to-end measurement bounds it, not only the fact
       that both paths read the same expert codes). CPU interpreter and Metal now execute the represented GPT-OSS semantics —
       YaRN, sinks, biases, routed MXFP4 MoE, ClampedGlu — numerically identically to f32 noise.
       Remaining: the bracketed performance ladder. Earlier note kept:
       INTERPRETER HALF CLOSED 2026-08-17: `shannon layer-dump --tokens` (new; given ids) over
       the HF checkpoint = the served CPU forward on the exact BF16+MXFP4 bytes, vs `vindex3
       exec --dump-layers` on the container, same 81 oracle ids → `layer-diff`: cos
       1.000000000, rel_rms 1e-6 (layers 0–16) to 3e-6 (17–23), max_abs ≤ 0.08, "no capture
       drifts". SAME weights, SAME semantics, two engines, f32-reassociation agreement through
       all 24 layers. Corollary: the 14/16 vs the Q4_K-spine oracle is the spine's
       representation (final residual within 3e-6 rel-RMS ⇒ logits within ~1e-3 ≪ the 0.12/
       0.47 miss margins). Remaining: the lowered arm (A-9.4) against the same dumps, and a
       BF16-spine served id oracle if a byte-identical id claim is wanted.
```


The framing to keep: **mechanism is portable; optimal schedule is
geometry-dependent.** VINDEX3 carries semantic geometry; the Metal planner
owns the execution choice; the KV engines belong under the same planner.
Individual techniques (kernel selection, seqpar) are not novel; the claim is
integration from semantics down to execution.

### VINDEX3 evolution — versioned by capability, not by model architecture (2026-08-18)

The organising rule, adopted 2026-08-18: **freeze the VINDEX3 ontology soon;
everything after that evolves through opsets, representation sets, profiles
and packaging — not schema redesign.** Stages are defined by the capability
they add, each with an exit criterion; what may land in each stage is
disciplined. The gauntlet (F1–F7) and standard-grade backlog (P0–P2) below
are the *detail* of the first two stages and are tagged with their stage.

| Stage | Goal | Defining capability | Status 2026-08-18 |
|---|---|---|---|
| **V3-F0 — Saturation** | prove the ontology | awkward architectures fit without changing a foundational relationship | **2 of 3 witnesses**: Glimmer (dense) and gpt-oss (routed MoE + sinks + biases + YaRN + clamped GLU, A-9) both went graph → plan → interpreter → Metal with new *vocabulary* only (F1 held). Third not started through the generic chain: Gemma 4 hybrid MoE exists only via the V2→V3 *import* path (c8), Kimi Linear/KDA/MLA only via the K3 adapter, multimodal only served. |
| **V3.0 — Stable Core** | freeze the model-database ABI | Model · Object · Representation · Segment/Region · Operation · Operand · Graph · Profile · Reference, and their relationships; separate versioning (container 3 / semantic IR 1 / opsets / representation sets); formal unknown-field rules; canonical serialisation + deterministic ids; corruption fixtures; independent reference reader | **~half**: generic MoE graph done (A-9.2, no `moe_manifest.json`); multi-representation *binding* not done (refusal real, selection does not yet steer bytes — F3); WALK/DESCRIBE against V3 authority not done (V2-1/c7 — F5); opset versioning not started (one integer today); criticality classes exist in code but not in the spec; F7 mutation sweep unwritten; canonical serialisation only as "dense plans serialise byte-identically under absence"; reference reader/conformance not started. |
| **V3.1 — Immutable Distribution** | composable, distributable models | content-addressed segments (OCI descriptor: type + digest + size + locations, manifests as Merkle DAGs), overlays/deltas/adapters (`parent: sha256:…; replace object 391 MXFP4 → sha256:…; add adapter`), provenance DAG per representation (tool, version, args, input/output/calibration digests) | not designed; Hub publish/slice contract (larql-factory, ADR-0026) is the packaging start |
| **V3.2 — Adaptive Representation** | runtime-selectable physical model | one semantic object with several *legitimate* representations; declarative profiles (`prefer: MXFP4`, `allow_remote`) that carry no kernel names; quality as metadata per representation (reference, rel_rms, cosine, kl_delta, shannon_delta, calibration digest) so a profile can say `max_shannon_loss: 0.01` and the planner chooses | the representation algebra exists in the lowering (F16/Q4_K/MXFP4/NVFP4 priced under identical scheduling) and Shannon scoring exists; nothing is *carried* as representation metadata yet |
| **V3.3 — Location Independence** | model bigger than machine | `segment == local file` removed: segment identity → resolver (GPU-resident / RAM / mmap / NVMe / HTTP-range / S3 / remote expert node); `PLAN operation|layer|token|prompt` returns inputs, selected experts, representations, required regions, expected bytes, residency, missing bytes — before execution | groundwork only: segments are page-aligned, mmapped and registered straight into Metal buffers ("bind, never reconstruct"); the working-set instrument (200 predicted / 200 resident / 0 overshoot) is `PLAN operation` before it has a name |
| **V3.4 — Ecosystem Standard** | no dependency on LARQL | `vindex-spec/` + `vindex-conformance/` + `vindex-reference/` (tiny C reader, Rust reader with no engine, Python reader); conformance corpus (dense, GQA, MoE, multi-rep, overlay, remote segment, unknown optional/mandatory extension, corruption); `vindex pull/inspect/verify/diff/compat`; registries (HF, OCI, S3, HTTP, LAN peer) equivalent | not started |
| **V4** | only if the ontology fails | a foundational relationship must change (e.g. object identity genuinely dynamic/context-dependent in a way operations/state/relations cannot express) | — VINDEX4 means "our ontology was wrong", not "there is a new model" (no FP3, no KDA2, no video) |

**Exit criterion for V3-F0 (the freeze trigger):** *three consecutive
structurally different architectures require new operators or
representation types, but no new foundational relationship.* Not "N tests
pass", not "all models work" — ontology saturation. Candidates for the
third witness, in the order they attack different parts of the ontology:
hybrid MoE (dense + routed/shared experts in one layer — cheapest, the
vocabulary already exists), Kimi Linear (recurrent/KDA state + MLA — the
most informative, because *state as an object* is where the V4 trigger
would show if it exists), another MLA model (attention/KV independently),
one multimodal architecture (towers/projectors/decoder graph); later, a
Mamba/Jamba/RWKV-type model to test "neural model" versus "transformer".

**Discipline per stage.** V3-F0: attack the ontology, add no ecosystem
feature. V3.0: freeze, publish the spec, ship the reference reader and
conformance suite, add no architectural idea; remote execution, cost models
and Shannon *selection* stay out of the frozen core. After ~V3.2, resist
3.5/3.6/…: the semantic IR stays at 1, and innovation happens in
`vindex.moe:3`, `vindex.kda:2`, `vindex.ssm:1`, `vindex.audio:2`, quant
representation sets, placement, quality contracts. Engine work (A-1…A-10)
continues throughout but is **not permitted to move schema** — a schema
change is a V3-F0/V3.0 event, argued as such.

**V3-F0 witness 3 — Gemma 4 26B-A4B (hybrid MoE), mapped 2026-08-18.**
Rung 0 measured: `larql vindex3 plan` on `google/gemma-4-26B-A4B-it` (HF
snapshot, BF16, 2 shards) is **inadmissible: 68 representable / 3 mismatched
/ 39 unrepresented → 42 blocking** (gpt-oss started at 4). Objects already
place: `decoder_stack` 3.3 GB, `embedding` 1.5 GB, `expert_bank` 45.7 GB
(the A-9.2 bank machinery recognises the 128-expert packed BF16 banks
unaided), `final_norm`, `perception_tower` 1.1 GB. The 42 sort into one root
cause, four semantic families, and classification debt:

```text
ROOT CAUSE  per-layer attention geometry. graph/surface.rs:238 requires ONE head geometry per
            component ("a per-layer variation is a schema gap to surface, not to average away") and
            Gemma 4 has head_dim 256 / kv 8 on 25 sliding layers, global_head_dim 512 / kv 2 on the
            5 full layers → target.execution_surface incomplete → the text component never builds →
            every probe that reads a BUILT component reports "no built component answered":
            rms_norm_eps, hidden_activation, attention_bias, final_logit_softcapping, both rope_thetas.
            These clear together once geometry is per-layer.
FAMILY 1    position. rope_parameters.{full,sliding}_attention is per LAYER TYPE (transformers-5
            shape, the same trap A-9.0 hit): full theta 1e6 resolved as 10000 (mismatched — a
            resolution bug, not a schema gap); rope_type "proportional" (mismatched vs default) with
            partial_rotary_factor 0.25 (unrepresented): rotate the first 0.25·head_dim dims, inverse
            frequencies over the FULL head_dim (global_head_dim on full layers), zero frequency on the
            rest — HF _compute_proportional_rope_parameters. Needs a PositionPolicy variant.
FAMILY 2    K ≡ V. attention_k_eq_v=true: v_proj is ABSENT on the 5 full layers (present on the 25
            sliding) — the V operand is the K object. Parsed but unclassified today.
FAMILY 3    hybrid FFN. EVERY layer has dense mlp.{gate,up,down} (inter 2112) AND experts.{gate_up,
            down}_proj (128 × inter 704, top-8) with router.{proj,scale,per_expert_scale} and FIVE
            FFN norms. HF Gemma4TextDecoderLayer: h = pre_ffn_norm(r); d = post_ffn_norm_1(mlp(h));
            router over the RESIDUAL r (norm WITHOUT scale, × scale × H^-0.5, softmax, top-k,
            renormalise, × per_expert_scale[idx]); m = post_ffn_norm_2(experts(pre_ffn_norm_2(r)));
            out = r + post_ffn_norm(d + m); then × layer_scalar. enable_moe_block / num_experts /
            top_k_experts / moe_intermediate_size are parsed but unclassified.
FAMILY 4    output. final_logit_softcapping 30 (OutputOp.softcapping EXISTS — opplan/mod.rs:271,
            surface.rs:211 — will answer once the component builds); tie_word_embeddings (no
            output_head object placed — the head must bind the embedding object; verify).
CLASSIFY    num_kv_shared_layers 0 (rule missing; 0 = none — but >0 on E2B/E4B is attention
            reading ANOTHER LAYER's KV state, a cross-op state dependency: THE F0 question this
            family poses to the ontology; refuse until represented), hidden_size_per_layer_input 0
            and vocab_size_per_layer_input (PLE; 0 = absent), use_double_wide_mlp, global_head_dim,
            num_global_key_value_heads, use_bidirectional_attention "vision"; root: audio_config
            null, {audio,boa,boi,eoa,eoi,image,video}_token_id, vision_soft_tokens_per_image
            (interface_semantic); vision_config: hidden_activation gelu_pytorch_tanh vs gelu_tanh
            is an ALIAS mismatch (plan/compare.rs must compare through Activation), attention_bias /
            rope_theta 100 / rope_type / global_head_dim 72 / pooling_kernel_size /
            position_embedding_size / default_output_length / standardize / use_clipped_linears are
            tower execution facts to carry on perception_tower (not executed by the text plan);
            id2label / label2id / problem_type / return_dict / output_* / is_encoder_decoder /
            chunk_size_feed_forward / _name_or_path are metadata_only.
```

Rungs, mirroring A-9 (each closes with its gate; served-side authority for
every fact is `architectures/gemma4.rs`, which already judges all of it):

```text
G4.0  DONE 2026-08-19 (branch feat/vindex3-gemma4): `larql vindex3 plan` on the real 26B-A4B is
      ADMISSIBLE — 111 representable / 0 mismatched / 0 unrepresented / 0 blocking (from 42). What
      landed, all as vocabulary in existing categories (F1 still holding): AttentionLayerPolicy
      gains per-layer HeadGeometry{head_dim, num_kv_heads} and v_from_k; the surface no longer
      requires uniform geometry and the op plan judges each layer's shapes against ITS geometry
      (kv_rows 24 ≠ 16 in the fixture — a coincident product hid the K/V check once);
      PositionPolicy::PartialRope{theta, rotary_fraction, basis: RotaryWidth|HeadWidth} with
      ROPE_TYPE_PROPORTIONAL, judged by the gemma4 arch on its full layers; carriage probes now
      take a ProbeContext (the attention span the fact's PATH names + the declared value for alias
      resolution) so per-layer-TYPE facts are judged against their own layers, and the resolution
      comparator does the same; ParameterFreeQkNorm.v (Gemma 4's scale-less v_norm) and
      AttentionOp.v_from_k (V = the K operand, closure-paired: no V operand on such a layer, a
      stray one refused); rules for attention_k_eq_v, enable_moe_block, top_k_experts,
      global_head_dim, num_global_key_value_heads, num_kv_shared_layers / hidden_size_per_layer_input
      / use_double_wide_mlp / use_clipped_linears (represented as ABSENT only — 0/false agree,
      anything else blocks); two new recorded readers (inventory/interfaces.rs for the root
      multimodal join incl. audio_config:null and use_bidirectional_attention; the PLE-vocab /
      double-wide knobs are ModelConfig fields read by the main parser — landed as a stopgap
      reader in #271, moved to ModelConfig the same day); label-map
      containers (id2label/label2id) and HF return/chunk plumbing classified metadata; a nested
      tower with no layer_types gets a Full × N table so its rope facts are judged; the tower's
      attention_bias reaches its surface. Executors (reference/production/device, the Metal
      lowering) REFUSE PartialRope, v_from_k and the V norm, typed, naming G4.2/G4.3. Pinned by
      plan/tests/gemma4.rs (9, each re-dropping its fact), opplan/tests/gemma4_closure.rs (K≡V both
      ways), exec/tests/gemma4_refusals.rs, and the invariant that every execution-semantic leaf
      has a carriage rule. TWO FINDS the gate forced: (1) the comparator treated
      rope_parameters.full_attention.rope_theta as the checkpoint-wide base and compared it to
      every layer — a false mismatch on any per-layer-type checkpoint; (2) HF `proportional`
      takes inverse frequencies over the FULL head width (base^(2i/512) on the global layers)
      while the served path's partial rotary takes them over the rotary width (base^(2i/128)) —
      different angles on the 5 global layers; the served gemma4 forward has NOT been certified
      against HF at attention level, and G4.2's layer-dump parity gate will arbitrate it.
      As mapped: admissibility as REPRESENTATION: AttentionLayerPolicy (graph/policy.rs:74) gains per-layer
      {head_dim, num_kv_heads}; surface_from_resolved stops requiring uniformity and AttentionOp reads
      the layer's geometry (KV allocation follows); PositionPolicy::Proportional{theta, rotary_fraction}
      (config/position.rs) + carriage rules for partial_rotary_factor and rope_type=proportional
      (plan/carriage.rs; the leaves are already in plan/semantics.rs:35) + the per-layer-type
      rope_parameters resolution fix pinned; AttentionLayerPolicy.v_from_k (closure: v_from_k ⇒ no V
      operand, else required); the CLASSIFY list judged (config_keys.rs / semantics.rs registries);
      vision alias compare. Executors REFUSE Proportional / v_from_k / hybrid until they execute them.
      Gate: plan admissible, 0 blocking; every dropped fact pinned by a test that re-drops it.
G4.1+G4.2  DONE 2026-08-19 (branch feat/vindex3-gemma4-exec) — GEMMA 4 RUNS THROUGH VINDEX3, HF-IDENTICAL.
      Graph: LayerFfn::Hybrid{dense, routed, pre_experts_norm, post_dense_norm, post_experts_norm}
      (the HF Gemma4TextDecoderLayer program, one place); RoutedFfnOp gains Gemma 4's router
      conditioning (router_scale, router_per_expert_scale, router_norm_eps — paired with
      MoeRouterKind::Gemma4Hybrid); LayerPlan.layer_scale (`layer_scalar`, by evidence); roles
      MoeRouterScale / MoeRouterPerExpertScale / PreExpertsNorm / PostDenseFfnNorm /
      PostExpertsNorm / LayerScalar and the checkpoint's spellings (`experts.*`, `router.proj.
      weight`, `router.scale`, `router.per_expert_scale`, `layer_scalar`, `*_layernorm_{1,2}`);
      HeadSurface.tied_to_embedding → OutputOp binds the EMBEDDING operand (tie_word_embeddings
      re-classified execution-semantic with a rule — it was "metadata" until a family with no
      head operand made the drop visible); `embed_vision` is a PerceptionAdapter (its `embedding`
      fragment had landed the vision→text projector in the text embedding object). Closure both
      ways on the miniature; on the REAL container `vindex3 ops`: 30 layers, every operand
      accounted (22 sliding / 21 K≡V full), output tied to target.embedding, softcap 30.
      Execution (reference / production / device): PositionPolicy::PartialRope both bases
      (reference: kernels::partial_rotary_frequencies — HF proportional = rotate-half pairs over
      the FULL head with the top frequencies zero, NOT a contiguous prefix; production: the new
      served planner larql_compute::attention::rope::rope_freq_plan_proportional, so the served
      gemma4 path can adopt it); V from the K operand + the parameter-free V norm
      (condition_v_in_place, shared production/device glue); the hybrid block via
      FfnOperands::apply_from_residual (router reads the RAW residual, experts read
      pre_experts_norm(residual), dense reads pre_ffn_norm(residual); post-norms; sum; post_ffn_norm;
      residual; × layer_scalar) used by BOTH the batch driver and the decode session;
      Gemma4Hybrid routing (softmax → top-k → renormalise → × per_expert_scale) literal in the
      reference, served primitives in production; GeluTanh gated FFN on production/device via the
      served gelu_tanh. Gates: miniature closes + reference ≡ production ≡ device ≤ 2e-5 + four
      causal operands + decode ≡ batch (exec/tests/gemma4.rs; note: a UNIFORM gain on K is a null
      perturbation under k_norm/v_norm — the fixture perturbs with a different matrix);
      partial-rotary bases at parity and pairwise distinct (exec/tests/gemma4_refusals.rs).
      REAL 26B-A4B: `vindex3 encode` 52.75 GB in 4m19s; HF transformers f32 (scripts/
      dump_layers_hf.py, 6 ids) ≡ `vindex3 exec --backend production --dump-layers` layer-diff
      cos 1.000000000 / rel_rms 1e-6…8e-6 ALL 30 layers incl. the five proportional-rope K≡V
      global layers, "no capture drifts"; chat-templated "What is the capital of France?" →
      greedy ids 818,5279,529,7001,563,5213,50429,84750,106 = HF's greedy BYTE-IDENTICAL
      ("The capital of France is **Paris**.<turn|>"). CPU interpreter 2.7 s/token decode
      (full re-forward per step; no KV cache on `--generate` yet) — a correctness instrument,
      not a serving number. F0 verdict so far: vocabulary only, again — no new foundational
      relationship for the 26B (num_kv_shared_layers = 0 remains the open question for E2B/E4B).
      FIND #3 for the served path: the served MoeRoutingPolicy::gemma4_hybrid routes on
      pre_experts_norm(residual) through a scale-less norm, where HF routes on the RAW residual
      through that norm (rms(w⊙rms(r)) ≠ rms(r)); together with the proportional-rope find, the
      served gemma4 forward is now KNOWN to differ from HF at two points — VINDEX3 is the
      certified path; served re-certification is its own item.
      As mapped: hybrid FFN in the generic graph: LayerFfn::Hybrid{dense, routed, combine} (opplan/mod.rs:200) —
      or Dense+Routed as two ops in the layer program with an explicit combine — plus roles for
      router.scale, per_expert_scale, layer_scalar, pre_ffn_norm_2, post_ffn_norm_1/_2 (graph/roles.rs),
      MoeRouterKind::SoftmaxThenTopK + NormalisedOverSelected + per-expert scale, the scale-less router
      norm, ExpertFormat packed BF16 [E,2I,H]/[E,H,I]. Closure both ways on a miniature gemma4-family
      fixture; `vindex3 ops` on the real container: N → 0 defects.
G4.2  interpreter executes it (reference / production / device): per-layer geometry, proportional rope,
      K≡V, hybrid FFN + router scales + layer_scalar, softcapped tied head. Gate: served CPU forward
      (`shannon layer-dump --tokens` on the HF checkpoint) ≡ `vindex3 exec --dump-layers`, all 30
      layers, plus causal controls per new operand.
G4.3  DONE 2026-08-19 (branch feat/vindex3-gemma4-exec) — GEMMA 4 RUNS ON METAL THROUGH THE
      GENERIC LOWERING, CERTIFIED, HF-IDENTICAL IDS, 10 tok/s. Lowering crate: weighted per-head
      Q/K norm (encode_weighted_qk_norm — the served qk_norm kernel; the lowering had NO weighted
      path and did not refuse one, a latent hole Glimmer/gpt-oss never exercised), parameter-free
      V norm on the raw projection in its cache slot (AttnShape.parameter_free_v), K≡V by binding
      the K matrix as V, per-LAYER rope tables (LayerLowering.inv_freq; the stack had shared one
      table and refused plans with two — Gemma 4 has plain θ=1e4 over 256-wide heads and
      proportional over 512-wide), FfnShape.activation (SiLU | GeluTanh kernels),
      encode_gated_ffn_branch (branch without post-norm/residual) and LayerFfnLowering::Hybrid —
      dense branch + router (rms_no_weight(r)·scale·H^-0.5 folded into ONE weighted rms_norm
      dispatch with weight router.scale·H^-0.5, served router logits/select with renormalise +
      per-expert scale on-GPU) + descriptor experts over pre_experts_norm(r) combined onto a ZERO
      residual (bare expert sum) + post_experts_norm + branch sum + post_ffn_norm + residual +
      layer_scale, all in the one command buffer. CLI session: expert banks declared packed BF16
      are QUANTISED TO MXFP4 AT LOAD with the interpreter's own quantiser (the descriptor kernels
      serve Q6_K and MXFP4 only; Q6_K needs the padded-row machinery since down's K=704 ∤ 256 —
      next representation rung), router conditioning / branch norms resident, hybrid scratch
      (slots 18..24), new arms `metal-lowered-mxfp4-ffn` (= the interpreter's `metal-mxfp4`
      representation) and `metal-lowered-f16` (f16 attention/dense/head, experts MXFP4 only).
      CERTIFICATION on the real 26B-A4B: lowered(mxfp4-ffn) vs interpreter(metal-mxfp4), SAME
      representation: cos 1.000000000 / rel_rms ≤ 1e-6 ALL 30 layers, "no capture drifts" — the
      A-9.5 instrument, no new plumbing. Greedy ids on the chat prompt = HF's (…**Paris**.) on
      BOTH arms: 98 ms/token (10.2 tok/s) mxfp4-ffn, 107 ms/token f16; weights resident in 86 s
      incl. quantising 45.7 GB of bf16 experts; route witness serial 870 / seqpar 0 (no planner
      row for (256,16,8)/(512,16,2) yet → serial, correctly). REPRESENTATION COST vs HF f32
      (priced, not argued): mxfp4-ffn rel_rms 0.084 (L0) → 0.476 (L29, cos 0.897); f16 with
      experts-only MXFP4 0.045 → 0.247 (cos 0.970) — the expert bank is about half the loss, the
      dense MLP the other half; the argmax trajectory survived 9 tokens on this prompt but this
      is the lossy end of the Q2 frontier and a Q6_K bank arm is the next representation rung.
      Pinned: tests/test_lowering_gemma4_arms.rs (see PR). As mapped: Metal lowering: per-layer KV geometry (a (512, 16, 2, span) planner row — serial where
      unmeasured), proportional rope table, K bound as V, hybrid encode = gated-ffn + descriptor MoE +
      norms + sum + layer_scalar in one CB, softcap head. Gate: layer-diff lowered vs interpreter
      ≤ 1e-6 (the A-9.5 instrument, no new plumbing).
G4.4  banked id oracle (`larql run --emit-ids` on the served path) + bracketed ladder; then the F0
      verdict: did Gemma 4 add vocabulary only? (Expected yes for 26B; num_kv_shared_layers>0 on
      E2B/E4B is the open relationship question and is scored separately.)
```

**Immediate queue this implies:** (1) the third hostile architecture through
the generic chain; (2) multi-representation binding, WALK/DESCRIBE on V3
authority, the opset/version model, refusal semantics in the spec (the
V3.0 pre-freeze list); (3) freeze. A-10 runs alongside as engine work.

### VINDEX3 stabilisation — freeze gauntlet, positioning, standard-grade backlog (2026-08-17)

**Judgement: the architecture is converged; the ABI had one adversarial pass
left, and that pass was A-9 — closed 2026-08-18 (below): gpt-oss went through
the same graph → plan → CPU interpreter / Metal lowering as Glimmer, no
family branch, interpreter ≡ Metal at rel_rms ≤ 1e-6 across all 24 layers, and
every A-9 rung added vocabulary to existing categories rather than a new
category (F1 held).** Distinguish *architectural stability* from
*freezing schema 3*. The question is no longer "is the VINDEX3 abstraction
right?" but "have we discovered every semantic noun and relationship the
frozen ABI must carry?" — and gpt-oss is the last high-value attack on it,
because every A-9 blocker is a **schema-vocabulary** fact (YaRN incl.
amplitude, clamped-GLU policy, representation-level quantisation policy,
sinks, Q/K/V/O biases, expert-bank objects, router/expert/bias roles, MoE
FFN op, over-selected normalisation), not a kernel. Freeze before A-9.0–A-9.3
close and schema 3.0 immediately fails to describe a production architecture.

| Layer | Assessment |
|---|---|
| Core VINDEX3 idea (state the working set before execution) | ~95 % settled — 200 predicted / 200 resident / 0 overshoot |
| Object / region / addressability model | ~90 % settled |
| Plan / execution architecture (reference → production → lowered) | ~90 % settled |
| Dense / Glimmer execution model | proven (real-model parity, GPU-resident multi-layer, KV through the plan) |
| Routed-bank serving | proven in production machinery (native MXFP4, GPU routing, one CB/token) |
| Generic MoE graph semantics | settled 2026-08-18 — A-9.2/A-9.3 closed; MoE is an object/operand/op arrangement inside the graph, `moe_manifest.json` absent from the gpt-oss container |
| Representation / profile machinery | conceptually settled, plumbing incomplete (selection does not yet steer bytes) |
| V3 extraction / default lifecycle | not ready to freeze |
| On-disk ABI / schema 3 | one serious architecture pass from freeze |

The pattern that says the centre is stable: *new architecture → missing
semantic fact exposed → added to graph/plan → the existing lowerer can reason
about it.* The 2026-08-16 representation algebra is the milestone —
attention policy stopped being a model-name decision and became
`(head_dim, q_heads, kv_heads, span) → execution geometry`, unmeasured cases
falling back rather than guessing. VINDEX3 knows *what the operation means,
what representations exist, what semantic geometry exists*; the backend
planner knows *how this hardware executes it*. **Protect that boundary.** The
Metal work strengthened it rather than distorting it: VINDEX3 segments are
mmap'd and registered straight into Metal buffers, expert selection stays
GPU-side, host resolution/binding witnesses stay zero, output byte-identical
— the bytes remain authoritative, so **bind, never reconstruct** graduates
from standing method to a design principle of the format.

**One smell to resolve before freeze:** `system_graph.json` and
`moe_manifest.json` are mutually exclusive container shapes. The frozen
answer must be *MoE is another operation/object arrangement inside a VINDEX3
system*, not a different container type — `moe_manifest.json` may survive as
a physical/indexing artefact, but MoE must not live outside the graph
universe. A-9.2 is that closure.

```text
VINDEX3
   ├── objects          tensor · expert_bank · embedding · norm · …
   ├── representations  F16 · Q4_K · MXFP4+E8M0 · NVFP4 · …
   ├── operations       attention · dense_ffn · routed_ffn · norm · vocab_projection · …
   └── plans / profiles
```

**Freeze gauntlet — seven gates, all required before "schema 3.0 is frozen":**

```text
F1  ontology closure      A-9.0/1/2/3 close without a NEW CATEGORY of semantic fact
                          → HELD 2026-08-18 (YaRN, ClampedGlu, sinks, biases, expert bank,
                          MoE roles/op all landed as vocabulary in existing categories)
F2  cross-family witness  Glimmer (dense, local/global) · gpt-oss (pure routed MoE +
                          biases/sinks/YaRN) · Gemma 4 (hybrid MoE); recurrent/KDA later,
                          non-blocking if unsupported ops fail loudly
                          → Glimmer + gpt-oss WITNESSED 2026-08-18; Gemma 4 open
F3  representation witness ONE container genuinely carries ≥2 representations of the same
                          semantic role; profile choice changes the BOUND bytes; a tamper
                          control proves it cannot silently hit the other
F4  independent execution reference → production → lowered parity (the discipline in hand)
                          → exercised on gpt-oss 2026-08-18: served HF ≡ interpreter ≡ lowered
F5  database parity       WALK/DESCRIBE run against V3 authority, not a V2 shadow — the
                          exact expert-bank bytes execution binds ARE the addressable
                          objects; no second KNN index unless declared a derived repr
F6  E0                    VINDEX2 untouched: full golden matrix + legacy CLI dispatch
F7  mutation test         corrupt every load-bearing relationship (wrong role, dims,
                          representation, missing partner scale, impossible profile,
                          illegal programme, wrong bank ownership, overlapping ranges,
                          unsupported policy) → typed refusal, every one
```

After F1–F7: *"VINDEX3 schema 3.0 is frozen. Additive extensions preserve the
3.x compatibility rules; semantic breaking changes require VINDEX4."*
Extraction default is a **later, separate** ladder — semantic freeze → schema
freeze → conformance fixtures → `--format vindex3` opt-in → soak → default →
VINDEX2 readable indefinitely. Not on the same day.

Housekeeping the freeze implies: this file has stopped being the right
authority for schema status (c8 was `[ ]` in the ladder above and CLOSED in
the prose below it). At freeze the status moves to one canonical
requirement × status × test × fixture × since matrix in the spec, and the
roadmap points at it.

**Where VINDEX3 sits.** Safetensors stores tensors; GGUF packages an
inference model; ONNX describes a computation; ExecuTorch/MLC/TensorRT
describe or compile an execution. VINDEX3 describes the model, its physical
representations, and how those may be *selectively bound* into execution
while the model stays addressable as data. Ticks below are *what the format
represents*, not maturity — GGUF's ecosystem, stability, tooling and
architecture coverage, ONNX's independent normative standard, and
TensorRT/TVM's compiler depth all beat VINDEX3 today, and none of those is the
battle.

| Format | Semantic graph | Quantised layout | mmap/segmented | Exec planning | Multi-repr / placement | Queryable model data | HW-neutral |
|---|---|---|---|---|---|---|---|
| Safetensors | ✗ | limited | ✓ | ✗ | ✗ | tensor-level | ✓ |
| GGUF | metadata/naming | ✓✓ | ✓✓ | ✗ | limited | tensor-level | mostly |
| ONNX | ✓✓ | some | external data | ✗ | limited | graph-level | ✓✓ |
| OpenVINO IR / Core ML | ✓ | some/✓ | separate/package | runtime/✓ | some | limited | mostly / Apple |
| ExecuTorch PTE | ✓ | ✓ | ✓ segmented | ✓✓ | delegates/mem plan | limited | ✓-ish |
| MLC/TVM | ✓ | ✓✓ | ✓ | ✓✓✓ | target compile | ✗ | source-portable |
| TensorRT engine | compiled | ✓✓✓ | internal | ✓✓✓ | HW-selected | ✗ | ✗ |
| **VINDEX3** | ✓ | ✓✓ | ✓✓✓ | ✓✓ | **✓✓✓** | **✓✓✓** | designed to be |

Safetensors is a *source* format for VINDEX3, not a competitor. GGUF's unit
of authority is the tensor at an offset; VINDEX3's is the semantic object
with representations and regions — the moat is that *operation → working set*
is part of the format contract, not a clever runtime's external policy.
ONNX is the closest to the semantic graph but deliberately declines to
prescribe physical representation; VINDEX3 keeps meaning + representations +
locations + bindability together, and encodes a higher-level neural algebra
(attention, routed_ffn, norm, …) rather than an SSA op graph, so one semantic
op can lower differently by length, residency, representation, hardware,
selected experts, prefill vs decode. ExecuTorch is the closest cousin
(graph + data + lowering + memory plan) but its output is a *prepared
program*; VINDEX3 is a *datastore from which an execution is constructed at
serve time*. TensorRT is the far extreme VINDEX3 must not become — an opaque
compiled artefact cannot answer "where are expert 37's up-projection bytes"
or "which pages will this op touch". Combined: GGUF's physical storage +
ONNX's versioned semantic contracts + OCI's content-addressed packaging +
Arrow/Parquet's random-access engineering + a database's addressability +
LARQL's representation-aware planning, under **one authority model whose
bottom does not sever from its top**. "Format" undersells it — it is a
neural-database storage-engine format.

**Standard-grade backlog** — optimise against the format we would design from
scratch if models were large, queryable, heterogeneous databases, not against
GGUF:

| Pri · stage | Improvement | Why |
|---|---|---|
| P0 · V3.0 | **Separate container / semantic-IR / opset / representation-set versioning** (ONNX's model: `container 3`, `semantic_ir 1`, `org.larql.core:1`, `org.larql.moe:2`, `org.larql.attn:3`, `org.larql.quant:4`) | KDA arrives as `org.larql.kda:1`, not VINDEX4; a reader says "cannot execute ops 46–91" instead of branching on architecture name |
| P0 · V3.1 | **Content-address every physical segment** (OCI descriptor: media type + digest + size + locations; manifests as Merkle DAGs) | dedup across fine-tunes, one-segment publishes, local/remote indistinguishable, per-region verification, model identity = hash of the semantic manifest |
| P0 · V3.0 (binding) / V3.2 (algebra) | **Finish genuine multi-representation objects** — representation as an independently describable entity (encoding, layout, scale encoding, block geometry, alignment, companion streams, accuracy contract, source repr, transformation, HW compat); profiles select a *physical representation* | the signature VINDEX3 capability; F3 above |
| P0 · V3.0 | **Formal unknown-field semantics** — every extension is `annotation` (ignore) · `execution_metadata` · `semantic_required` (refuse) · `representation_required` (fine if another admissible repr exists) · `interface_required` | criticality work already invented the answer; put it in the spec, avoid GGUF's accumulate-conventions-forever fate |
| P0 · V3.0 (reader) / V3.4 (ecosystem) | **Conformance suite + tiny independent reader** (`vindex-spec/`: spec, schema, test-vectors, reference-{c,rust,python}; open/validate/list objects+ops/resolve reprs/map segments/verify hashes, no LARQL dependency) | if only `larql-vindex` can read it, it is a good LARQL format; if a compatible reader is a weekend, it can be a standard |
| P1 · V3.1 | Transformation/provenance DAG per representation (tool, version, args, input/output digest, calibration corpus digest) | reproducible builds; "where did these 64 bytes come from" has an answer |
| P1 · V3.1 | First-class overlays/deltas/adapters (`parent: sha256:… ; replace object 391 MXFP4 → sha256:… ; add adapter`) | LoRA, patches, expert replacement, LARQL INSERTs, spec heads as composable artefacts — no 30 GB duplicate, no base mutation |
| P1 · V3.3 | Remote/range-addressable segments — objects refer to segment *identity*, a resolver yields RAM/mmap/NVMe/HTTP-range/S3/peer/GPU | model-as-database at its logical conclusion |
| P1 · V3.1 | Compact binary navigation index alongside (or canonical under) `index.json` (`index.vxb`; Arrow footer / Parquet metadata model) | opening K3 must not mean parsing 300 MB of JSON |
| P1 · V3.3 | Compatibility/capability query — `vindex compat model --runtime metal-m3-max` → per-opset ✓/✗, executable layers, no model data touched | complete-or-refuse, before touching weights |
| P2 · V3.2 | Per-region measured quality contracts (reference, max_abs, rel_rms, cosine, shannon_delta, fixture digest) | representation choice becomes a correctness decision — the Shannon work meets the representation algebra |
| P2 · V3.1 | Optional signatures / encryption / access policy | distributed commercial models |
| P2 · V3.4 | Standard packaging profile (HF, OCI registries, S3, local disk) | natural homes |

Never embed executable code (no Python/dylib/CUDA/wasm-with-host-access to
*load* a model): operators are declarative contracts, unknown ones fail
closed, backend code belongs to runtimes — downloading a VINDEX stays closer
to downloading data than a program.

If only three land before 1.0: **independent IR/opset/representation
versioning, content-addressed immutable segments, conformance suite + tiny
reader.** Target end state:

```text
vindex describe model      → operators, representations by count, local/remote/hot-set bytes,
                             per-runtime FULL/PARTIAL, derivation digests
vindex plan model --hardware this-machine --memory 12GB
                           → a valid physical plan WITHOUT loading the weights
```

Then K3/Kimi stop being about discovering VINDEX3 and become the stronger
test: *can the frozen algebra express exotic architectures without changing
its ontology?*

---

### K3 expert transport codec — CLOSED, cross-expert redundancy is nil (2026-08-10)

Registry: `dec8-13-cross-expert-conditional-census` (programme `dec`), completed
and refuted. Rule **R15** added to [`docs/dec-funnel.md`](docs/dec-funnel.md) §1.

The previous rung measured each expert's MXFP4 symbols *on their own* (3.7525 of
4.0 bits — near-optimal). It conditioned on nothing, leaving open whether the 896
experts in a layer share structure that per-expert coding discards. They do not.
`larql k3-ledger cond-census` conditions one expert's **untouched packed
nibbles** on a leave-one-out bank prototype, an out-of-sample-selected coding
parent, a random parent, and a permutation-invariant per-input-channel profile:
**mutual information 0.0000 bits on all three GLU branches**, every arm sitting
on top of its own marginal-preserving shuffled null. Agreement is 0.086–0.109
against a 1/16 chance floor, so the runnable match/escape code prices at
**4.56–4.66 bpw — worse than shipping the raw 4.25**.

**The controls are the result.** A self control reads exactly 0.0000 with
agreement 1.0000 on the same bytes; the marginal reproduces the banked 3.7525 to
1e-4; and an adjacent-row control *within one expert* also reads 3.7527 — so
there is nothing aligned to find, not merely no correspondence between experts.
Mechanism: MXFP4's per-32 e8m0 scale already absorbs per-channel magnitude,
leaving residual nibbles near-i.i.d. A format that spends its bits well leaves
no cross-object redundancy for a dictionary to collect.

**Closed with it:** entropy coding MXFP4 symbols, dictionary coding aligned
expert symbols, resident-parent lossless delta coding, adaptive symbol
alphabets, and **ETC-0B** (a specific parent adds 0.0007 bits over the
prototype; selection beats random by −0.0000, so there are no edge weights to
route around and the DEC-0 traces need not be opened for coding parents). R4 had
capped the whole idea at **1.87×** before the fetch anyway; the measured floor is
worth **1.06×** end-to-end.

**Still open, deliberately deprioritised:** a permutation-*aligned* comparison
(assignment over 3072 rows — poor prior from the adjacent-row control), and
lossy value-space decomposition, which is **approximate expert factorisation,
not compression** — approximate lane, scored on induced bits/token and route
stability, and this lossless result must not be cited for or against it.

Consequence: the routing graph and the compression graph are different objects.
Effort returns to access structure — residency, owner grouping, prefetch,
avoidance — where the route-aware hot-cache rung already measured **1.80×** from
grouping work by physical owner.

### K3 serving-format ladder + efficiency re-bank (2026-08-01)

Two rungs closed and one measurement corrected. Registry: `dec8-11`, `dec8-12`
(programme `dec`); rules R7/R8 added to [`docs/dec-funnel.md`](docs/dec-funnel.md) §1.

**The exact-format search is finished.** K3's experts are MXFP4, so a group
reconstructs at most 15 distinct values — 4 payload bits is the floor by
counting, not by search, and MXFP4 already spends exactly it. Doubled, the
alphabet is not an arithmetic progression, so the smallest affine grid
containing it needs 25 levels: **Q4_K can never be exact** (9 levels short),
Q5_K can but is dominated, and **Q6_K is the cheapest exact container that can
actually serve today**. The variable-rate loophole is closed too — measured
entropy 3.75 bits over 7.86 M real weights, and **0.0000%** of tiles hold ≤8
symbols at any block size ≥64, so palettes and escape codes are dead. Exact
floor **4.06731 bpw**. `larql k3-ledger formats` / `symbol-census`.

**MXFP4's low kernel efficiency is the container, not a defect.** Seven crossed
arms at the real expert shape decomposed the winner into skeleton 76% / fp4
decode 22% / input gather 2%, with the skeleton already streaming at 0.95 of
attainable bandwidth. Four decoders tried; the ordering is monotone in table
size and a table-free bit-manipulation decoder is worst by 37%. **The
expert-side kernel line closes at single-token width.**

**Numbers, and they moved down twice — both times because a measurement got
honest, never because anything got slower:**

| claim | status |
|---|---|
| **3.02 tok/s** | controlled healthy-regime exact-Q6_K composed ceiling |
| **2.79–3.18** | observed, composed **paired per run** over 7 accepted runs |
| **3.65 tok/s** | + grouped routed experts — clean measurement, integration still required |
| **4.15 tok/s** | + routed MXFP4 — a **kernel projection**, maturity `Grouped`, below `is_servable()` |
| **5.49 tok/s** | density-only **upper bound**; reuses Q6_K efficiencies at MXFP4 density, which R7 forbids |
| *unmeasured* | **sustained** laptop throughput under the degradation regime below |

**Two harness bugs, both silent, both now guarded.** `BufferCache::get_bytes`
keys on `(pointer, length)`, so same-length *temporaries* aliased and returned
each other's buffers — which meant the cold-rotation loop feeding every
efficiency figure was handing back **one buffer eight times**. The composed
ledger survived it (3.70 → 3.68), because the dominant term is also the
steadiest. And a 16-run promotion campaign found that **more repeats make it
worse**: 9 runs were unusable as the machine degraded under sustained load and
the attention control fell 0.89 → 0.06. Runs are a time series, not
exchangeable draws.

1. **Run the sustained end-to-end decode, and name the degradation.** The nine
   rejected runs are a second scoreboard nobody has measured: report throughput
   by time window (startup / healthy / late / steady-state floor) over 20–30
   minutes with system telemetry. Thermal, power management, memory pressure
   and paging are all still live candidates. **The demo number is this one, not
   3.02** — and it may be lower.
   [larql-compute-metal]

2. **Promote `gate/up` and the ungrouped expert shape across independent
   cool-start sessions.** Both sit at 2.2–2.4% relative standard error against a
   1% bar, and both feed DEC-8.7b's target row — which is the only live
   throughput rung now that kernel efficiency is closed as a lever. The R4 lever
   ordering *refuses to print* until they clear. Not another same-session
   campaign; that reproduces the artifact. Check the histogram for bimodality
   before banking a mean.
   [larql-cli]

3. **Finish the grouped-down integration A/B on a loaded model.** DEC-8.9's
   kernel risk is retired and its `next_action` carries the six-step order;
   this is what converts 3.02 → 3.65 from projection into result, and it is the
   nearest end-to-end milestone.
   [larql-compute-metal, larql-inference]

4. **Resolve `A_log` before any KDA numerics.** K3's checkpoint ships `[128]`
   where the reference module allocates `num_heads` = `[96]`, and two readings
   of the geometry each explain the large tensors while breaking one small one
   — **shapes cannot decide it**. `kda_a_log` fails closed and ships a
   deliberately rectangular discriminating fixture, because the two readings
   coincide on the diagonal and a square fixture would pass vacuously.
   [larql-cli, larql-models]

5. **Build a sentinel with a working set ≥ the largest class it gates.**
   Attention is currently both a banked class and the control, so its 0.876 is
   self-selected and biased upward. The obvious cheap fix is *worse*: a 21 MB
   sentinel admitted two runs where the 72 MB attention cell had already
   collapsed. Degradation is size-dependent; the K2 weights-only probe (89.5 MB)
   is the candidate.
   [larql-compute-metal]

6. **`prefill_q4_seq4_synthetic_smoke` is flaky at ~3–5%, all-NaN output.**
   Found by the new commit gate, which runs `--all-targets` rather than the
   `--lib` subset. Failure mode is the *entire* prefill output NaN, not a
   drifted value. **Bisect did not resolve it and n=16 per commit was
   underpowered**: pooled 3 failures in 88 runs, with the failures landing on
   two non-adjacent commits and 0/16 on the commits between them — at an 8%
   true rate, `P(0 in 16) = 0.26`, so a clean 16 proves nothing and ~36 runs
   per candidate are needed. Not attributable to any one change on the
   evidence available. Same family as the threadgroup-scratch reuse race fixed
   earlier in fused attention, so treat it as a real race rather than noise;
   localising it wants a proper campaign, not another bisect.
   [larql-compute-metal]

7. **Attention E/F ceiling probes — parked, bar pre-registered.** R7 means
   attention's 0.87–0.89 may describe its container rather than a fixable
   kernel. Same harness, needs Q6_K variants. **If the skeleton returns ≥ 0.93
   the class is closed and no decoder work is licensed.** Run it when preparing
   dense-format work or DEC-8.7b, not before the integration above.
   [larql-compute-metal]

---

### K3 R1 Gate B — forward parity closed on CPU, open on Metal (2026-08-04/05)

Write-up [`docs/k3-funnel.md`](docs/k3-funnel.md) §4.8–4.10. Registry
`k3r1-gptoss-pipeline` (programme `k3`). **P2 is closed on the CPU f32 path for
both R1-class models; the remaining work is Metal and the P3–P6 phases.**

**Closed.** GB's missing half — the layer-by-layer diff — is built
(`larql shannon layer-dump` / `layer-diff` + `scripts/dump_layers_hf.py`) and
immediately closed two models. OLMoE: `rms_norm_eps` class default 1e-5 with
the field absent from the checkpoint, and a QK-norm applied over the whole
projection rather than per head (cos 0.890 → 0.991 → **1.000000000**; bits/char
0.435 vs the reference's 0.4348). GPT-OSS: `rope_type: "yarn"` parsed and then
ignored because the only scaling hook was an `Option<Llama3RopeScaling>`, a type
that could not express it — 23 of 32 rotary dims at the wrong frequency and
every cos/sin 34.7 % small (cos 0.9777 → **1.000000000** at layer 0). Sliding-
window attention, absent from the dense path entirely, now exists as one
`AttentionSpan` shared by prefill and decode; verified at 511 tokens (4× the
window) with layer 0 — a sliding layer — at cos 1.000000000. The leftover
residual is measured, not assumed: a **four-token tie-break cascade** carrying
98.17 % of the final squared residual, seeded by one exact tie.

| # | Item | Crate | Status |
|---|---|---|---|
| M1 | **Metal decode ignored every RoPE scaling family.** Prefill roped on the host and honoured llama3 / YaRN / Gemma 3's linear divisor; decode roped in-shader from `rope_base` alone and honoured none — live on `gemma-3-4b/12b-it` and `Llama-3.2-1B`. **FIXED**: `RopeFreqPlan` computed once by the same `rope_freq_plan` the CPU uses, bound as a buffer + amplitude. | larql-compute-metal | **done** |
| M2 | **All four rope-bearing shaders converted atomically** — `rope` (4 kernels), `qk_norm_rope_fused`, `attn_fused`, `fused_attention`; eight rotation sites, zero `pow(rope_base, …)` left. `stages::rope_freq` owns the binding and checks the table width against the layer's geometry. **551 Metal tests green; the suite caught 16 binding mistakes**, each surfacing as `cos = 0.0` rather than a compile error, since Metal bindings are untyped. | larql-compute-metal | **done** |
| M3 | **A decode-pass diff exists. DONE (2026-08-06), re-homed 2026-08-07 as `larql shannon decode-diff`.** The example it originally landed in was deleted by main's examples reorganisation, so the pass now lives in the CLI beside `layer-dump`/`layer-diff`, driving `residual_diff::ResidualCapture` rather than reimplementing it. It is a *different axis* from `layer-diff` and the doc says so: `layer-diff` compares this engine to an external HF reference over a prefill, which by construction cannot see a decode-only defect. Verified on `gemma-3-4b-it`, 34/34 layers, `--steps 2`. **Original entry:** The example now runs a fourth section: Metal `prefill(N-1) + decode_token(N)` against CPU `prefill(N)` projected to its last row, per layer, reusing `residual_diff::ResidualCapture` rather than re-spelling the dump plumbing. Note the finding along the way: the *library* already had `metal_decode` / `metal_decode_steps` and `tests/test_decode_consistency.rs` already compared them against a CPU reference — the gap was only in the interactive tool, so "the decode diff does not exist" was too strong. | larql-inference | **done** |
| M4 | **Metal prefill now honours the sliding window. DONE (2026-08-07).** The defect, measured before fixing: Metal *decode* windowed correctly but Metal *prefill* took no window at all — `stages::attention::encode` had no such argument — while CPU prefill windowed via `effective_attention_window_for_layer`. So every sliding layer attended the whole prefix on GPU (Gemma 3: 29 of 34, window 1024; GPT-OSS: 12). **The M1 asymmetry inverted.** **Fix:** `fused_attention` gains `window_size` at buffer 17 and a `k_start` that mirrors the CPU rule (`causal_len.saturating_sub(w)`) exactly; the score, softmax and V-weighted loops all start there, and the two threadgroup reductions now count `active_len` rather than `causal_len`. The per-layer window was already resolved and already on `FullPipelineLayer` — `build_pipeline_layers` computes it through the shared rule with `0` as the no-window sentinel — so global layers arrive as 0 and stay unwindowed. **Evidence:** `tests/test_prefill_sliding_window.rs` compares Metal prefill against the production CPU `gqa_attention_windowed` (not a hand-rolled reference — the claim is that the two *backends* agree). `seq_len=48, window=8` puts 40 of 48 queries outside the window, and a fixture-adequacy test asserts windowed and unwindowed CPU actually differ on it, so the suite cannot pass vacuously. Verified discriminating: with `k_start` forced to 0 exactly one test fails and the no-window control still passes. Full Metal suite green; real-model decode-consistency green on gemma3-4b/llama2-7b/mistral-7b. **Left open:** a model-level long-prompt parity fixture. The existing suites still prompt with ~16 tokens against a 1024 window, so they remain blind to this class — the kernel test is what guards it today. | larql-compute-metal | **done** |
| M5 | **Prove the M1 fix end to end on `gemma-3-4b-it`. DONE (2026-08-06), and it took a detour.** First run passed at cos 1.000000 across all 34 layers — but the vindex the test loads has **no `rope_scaling` at all**, so `rope_position_divisor_for_layer` returned 1.0 on every layer and the 8× divisor M5 names was never exercised. That is a gate–claim congruence failure, not a result. Re-run with `LARQL_ROPE_POS_DIVISOR_GLOBAL=8`, which drives the same `effective_rope_position_divisor_for_layer` → `rope_freq_plan` both backends read: **34/34 layers at cos 1.000000, 1 and 2 decode steps**. Knob verified to bite, not silently no-op: outputs are identical for L00–L04 and diverge from **L05 — the first global layer** — with final ‖h‖ 21424.07 (divisor 1) vs 21878.96 (divisor 8). | — | **done** |
| M6 | **Every Metal Q6_K and Q4_0 kernel decoded a private nibble layout. FIXED (2026-08-07).** PR #207 moved the CPU side of both formats onto ggml's planar layout and changed **no file** under `larql-compute-metal`, so the two halves silently disagreed and `main` went red. Q6_K planar packs a super-block as two 128-element halves where one `l` column yields four elements at *stride 32* from three bytes (`ql[64h+l]`, `ql[64h+l+32]`, `qh[32h+l]`); Q4_0 packs byte `j` as elements `j` and `j+16`. The shaders read the pre-ggml `ql[i/2]`/`qh[i/4]` and `2j`/`2j+1` forms. **Six Q6_K kernels** (`q6k_matvec`, `q6k_matvec_8sg`, `q6k_grouped_experts`, `q4k_q6k_qkv_proj` ×2 kernels, `q6k_geglu_down` ×2, `q6k_geglu_gelu_tanh_down_cached`) and **four Q4_0 kernels** (`q4_matvec_v4`, `q4_f32_matvec`, `q4_vecmat`, `q4_sparse_matvec`) converted. This is a **served-model** defect, not just a test one: #207 also moved `larql-models`' GGUF readers, so Q4_0/Q6_K weights loaded from disk decoded wrong on GPU. `q6k_grouped_experts` is K3's expert-dispatch kernel. | larql-compute-metal | **done** |
| M7 | **Only one test in the tree could see M6, and the others were blind by construction.** `stage_quant_matvec_routes_format_to_correct_shader` caught it because it compares against a **true f32 gemv**. The rest did not: `q6k_matvec_8sg_matches_4sg_bit_equal` compares two shaders *to each other*, so a shared defect keeps it green; and the five `q6k_geglu_down` parity tests did compare against the planar CPU backend but their fixture was `cos(seed + 0.001·i) + 0.3·sin(i >> 8)` — a super-block spanned 0.26 rad of a smooth curve, and the second term was **constant across a whole super-block so it survived any permutation exactly**. A layout error permutes elements; a fixture too smooth to notice a permutation is an *absent* test, the same class as §4.9.1's 85-token window and §4.7.3's `out_features = 2`. **Fixed:** the generator is now hash-decorrelated, `q6k_matvec_both_geometries_match_cpu_reference` anchors both TG geometries to `CpuBackend`, and `fixture_can_distinguish_planar_from_interleaved_layout` asserts the *counterfactual* — that decoding this fixture the old way breaches the very threshold the parity tests enforce — so the property cannot silently regress. All verified discriminating by reverting each shader and confirming the tests fail. | larql-compute-metal | **done** |
| M8 | **The x86 Q4_0 reference test carried the same stale layout.** `tests/test_q4_x86_correctness.rs`'s `dequantize_q4_0_row` still wrote `2j`/`2j+1` while #207 moved `csrc/q4_dot.c` to planar. It is `heavy_tests`-gated so it never ran in the failing CI job, and it is x86-only so an aarch64 box does not reach it by default. Fixed and verified discriminating: 2 of its 3 tests fail against the old reference. | larql-compute | **done** |

**Phases.** R1/P1 (audit) and P2 (adapter) are closed. **P3 harvest, P4 extract,
P5 serve, P6 shrink are not started** — but P5's named blocker is gone.

**P5 expert-store blocker CLEARED (2026-08-07).** The diagnosis in §4.7.10 was
half the story: GPT-OSS fell through *both* writers, not one.
`write_per_layer_moe_kquant` requires `PackedBF16` and
`write_per_layer_moe_per_expert` required `PerExpert`, so `PackedMxfp4` matched
neither and no expert store was written at all — extraction reporting success,
checksums verifying, and the model unservable. That is the **third** appearance
of the silent-0-byte expert store this file's lineage has documented, and each
time the cause was a gate testing an *enum value* rather than the *capability*
the writer needs. The gate is now `arch.is_moe() && arch.expert_ffn_gate_key(0,
0).is_some()` — "does this arch expose per-expert tensors", which is exactly
what the writer consumes. Packed models still decline correctly, because the
trait default for that key is `None` and Gemma 4 does not override it.

**Format:** MXFP4 experts transcode to **Q6_K, not Q4_K.** An MXFP4 group
reconstructs at most 15 distinct values and Q6_K represents every one exactly,
so the transcode is lossless — the K3 kernel-ladder result, applied. Q4_K would
re-quantise an already-quantised tensor and discard the checkpoint's own values
for no benefit the serving path can use.

**Still to verify:** this is pinned by unit tests on a synthetic GPT-OSS-shaped
source (both verified to fail against the old gate), *not* by a real extraction.
`openai/gpt-oss-20b` is present locally; running P4 extract against it end to end
and then serving it is the next step, and until that is done "GPT-OSS is
servable" remains a claim about the writer, not about the model. Item 14 (routed
`FfnBackend`) is still the other half.

**Standing rules earned here** (see [`AGENTS.md`](AGENTS.md)): diff the forward
before theorising about it; a fixture too small to distinguish the candidate
behaviours is an *absent* test, not a weak one; a config fact belongs in the
trait default, not in one architecture; and a threshold chosen without
calibrating it against the quantity it bounds is a guess wearing a number.

---

### Compute-layer hygiene review — `larql-compute` / `larql-compute-metal` / `larql-models` (2026-08-05)

Scanned for the four standing rules: architecture-driven rather than
model-hardcoded, no magic strings/numbers, modular and decoupled, no large
files. **The architecture-independence story is much better than the file-size
one**, and the one real hardcoding leak is a stringly-typed protocol.

| # | Finding | Where | Priority |
|---|---|---|---|
| H1 | **`moe_router_type()` was a `&str` protocol between models and compute.** `pipeline_layer::moe_routing_policy` matched the literal `"gemma4_top_k_softmax"` and fell through to a default for everything else — so `gpt_oss`'s `"gpt_oss_topk_then_softmax"`, a genuinely different rule, **silently took the ordinary policy**. That is the mechanism behind §4.7.10's open quantised-MoE defect, not merely a style issue. **FIXED**: typed `MoeRouterKind` with the string kept as the vindex wire form (`as_str`/`from_wire`); compute now matches exhaustively, so a new variant fails to compile rather than defaulting. The predecessor test called the function twice and asserted nothing — replaced with one that pins each kind to a distinct policy. | `larql-compute`, `larql-models` | **done** |
| H2 | **`diag/shader_bench.rs` — 1 759 non-test lines**, and it hardcodes `"gemma3"` profiles and `"gemma3-4b"` labels. Diagnostics may name models, but not at this size in one file. **Attempted and reverted:** a line-based carve into config / shapes / measure / benches kept cutting across item boundaries (a trailing `#[derive]`, a truncated function body, the `mod tests {` wrapper). It wants an AST-aware split or a careful manual one, not a `sed` pass — and it is the lowest-value item here, so it was not worth finishing badly. | `larql-compute-metal` | low (was medium) |
| H3 | **`kquant_forward/cached.rs` — tests. DONE (2026-08-06).** "Zero tests" was half right: the sibling `kquant_forward/mod.rs` suite already drove most of the public surface, but **every one of those tests asserts a shape, not a value** (`h.shape() == [1, hidden]`, "must complete without panic"). That is exactly the hole §4.10 fell through — a RoPE defect keeps the shapes correct. 16 new tests in `cached/tests.rs` built on *agreement*: prefill-vs-decode on the rope-scaled Q4_K fixture (CPU analogue of M5), the padded-intermediate refusal in `layer_supports_direct_matvec`, and the guard clauses of `matvec_q4k_or_q6k_q8k`. Also replaced mod.rs's `let _: bool = supports_direct_matvec_decode(...)` — a test that asserted nothing — with a real assertion. | `larql-compute` | **done** |
| H4 | **`decode/mod.rs` — tests. DONE (2026-08-06), and it found a bug.** "Zero tests" again described coverage, not correctness: `tests/test_metal_decode_synthetic.rs` already drives `decode_token` end to end and says so in its own header ("smoke tests, not numerical-parity tests"). What nothing touched was the **KV cache geometry** layer — `kv_shapes_for_layers` / `ensure_kv_cache_for_{layers,shapes}` — where every failure mode is silent: decode still runs, still returns finite numbers of the right shape, and is simply wrong past some position. Six tests in `decode/tests.rs` on the Gemma-4 sliding(16×256)/global(4×512) pair, GPU-guarded. **`grow_to_shapes` ignored its `max_seq` argument** — see below. | `larql-compute-metal` | **done** |
| H5a | **`lm_head` silently tied to the embedding matrix.** `unwrap_or_else(\|\| embed.clone())` fired whenever `lm_head.weight` was absent, and **`tie_word_embeddings` was never parsed at all** despite appearing in every checkpoint config and several fixtures. A model declaring `false` (GPT-OSS, OLMoE) that lost the tensor to a key mismatch or skip filter would have served a wrong output projection and still produced fluent text. **FIXED**: field parsed (outer *and* `text_config`), and untied-but-missing is now a `MissingTensor` error naming the conflict. Absent stays `None` — not a claim either way — so tie-on-absence is unchanged for models that really are tied. | `larql-models` | **done** |
| H5b | **Split `loading/safetensors.rs`. DONE (2026-08-06).** 1 205 lines → `safetensors/{mod,mxfp4,dtype,paths}.rs`, largest 491. Moved whole functions with the compiler as the check (the H2 lesson), then moved each concern's tests and helpers to sit with it. Seams: MXFP4 packed-expert expansion (+ the seven `MXFP4_*` name constants, which now live with the layout they describe), raw-dtype/FP8 decode, model-path resolution; the shard walk and key normalisation stay in `mod.rs`. 649 `larql-models` tests green, clippy clean. | `larql-models` | **done** |
| H6 | `attention/gqa.rs` was 1 308 lines but **377 non-test** — the bulk was its (good) test suite. **FIXED**: split to `gqa/mod.rs` (379) + `gqa/tests.rs` (934), the same pattern `rope/` uses. | `larql-compute` | **done** |

**What is already right, and worth not regressing.** Architecture behaviour is
genuinely trait-driven: model-type strings appear almost exclusively in
`detect/mod.rs` (the dispatcher, where they belong) and in test fixtures. The
`stages::sinks` / `stages::rope_freq` modules are the pattern to copy — each
owns one binding convention in one place, with the reason it exists documented
against the defect that motivated it. Numeric constants are named
(`ROPE_BASE_DEFAULT`, `DEFAULT_NORM_EPS`, `YARN_BETA_FAST`,
`UNIT_AMPLITUDE`, `LAYER_TYPE_*`, `ROPE_TYPE_*`) rather than inline.

**Status after the second pass (2026-08-06), updated 2026-08-07:** H1, H3, H4,
H5a, H5b and H6 are done; M3, M4 and M5 are done, and M6/M7/M8 landed on
2026-08-07. **H2 is the only listed item still open**, and it is low by choice.

The second pass also turned up **three** defects that were not on the list —
two from the standing scan, one from writing H4's tests. All three are the same
family as H1/H5a: a value that answers a question nobody asked it. Two are
`_ =>`/omission defaults; the third is an argument accepted and ignored, which
is a pattern the scan did not previously look for and now should.

#### Next actions for the open items

**H2 is the only one left**, and it is deliberately last.

**H2 — split `diag/shader_bench.rs`.** Unchanged and still lowest value. Do
**not** repeat the line-range carve: it cut across a trailing `#[derive]`,
truncated a function body, and orphaned the `mod tests {` wrapper. Move items
one at a time with the compiler as the check — that is how H5b was done this
pass and it worked without incident — or leave it; it is diagnostics code with
no known defect behind it.

#### Standing follow-ups from the same pass

- **M4** — **done 2026-08-07**, and the defect was the mirror of what this line
  described: Metal *decode* windowed correctly while Metal *prefill* took no
  window at all. See the M4 row above. **Still open from it:** a model-level
  long-prompt parity fixture. The existing suites prompt with ~16 tokens
  against a 1024 window, so they remain blind to this class; the kernel test is
  what guards it today.

##### Ninth instance — an argument accepted and ignored (H4, 2026-08-06)

Not found by grepping for `_ =>` or `unwrap_or`: this one is a **parameter
that is taken and then not used**, which the scan's three patterns do not
catch. Worth adding as a fourth thing to look for at the boundary.

`KVCache::grow_to_shapes(bufs, shapes, max_seq)` only ever grew the *layer
count*; it never looked at `max_seq` for layers that already existed. Its
caller `ensure_kv_cache_for_shapes` rebuilds only on a **shape** mismatch — so
a second, longer prompt with the same attention geometry kept buffers sized
for the first one, while the caller had just asked for more room and had no
way to learn it did not get it. `encode_kv_append` then writes at
`current_len` and bumps it with **no bound check** against `max_seq`, so the
appends run off the end of a buffer allocated as
`max_seq * num_kv_heads * head_dim * 4`.

Reachable on a real path, not just in theory:
`vindex::kquant_forward::metal` sizes the cache as
`token_ids.len().max(MIN_KV_CACHE_SEQ)`, so it varies with prompt length
across calls on one backend. The uniform call sites (`kv_cache_mut*`) pass the
constant `DEFAULT_KV_CACHE_MAX_SEQ` and never take the branch, which is why
nothing had hit it.

Fixed by reallocating undersized layers in `grow_to_shapes`; regrowing drops
that layer's cached K/V, which matches what a shape mismatch already does, and
the one caller that varies `max_seq` calls `reset_kv_cache()` immediately
after. `ensure_kv_cache_grows_max_seq_for_a_longer_prompt` pins it — verified
to **fail** with the fix reverted while the other five geometry tests still
pass, so it discriminates the defect rather than merely covering the line.

##### The standing scan found two more (2026-08-06)

The scan works. Run it: grep the model→compute boundary for `_ =>`,
`unwrap_or`, and `&str` parameters that carry a behavioural choice — and now
also for **parameters that are accepted and never read** (the H4 instance
above, which none of the first three patterns would have caught). Two hits
from this run, both fixed, both worth reading as a pair because they fail in
opposite directions.

**Seventh instance — `Activation` collapsed to SiLU at the pipeline boundary.**
`pipeline_layer.rs` translated `arch.activation()` into the compute enum with
`match { GeluTanh => GeluTanh, _ => Silu }`, re-spelled at three construction
sites. `larql_models::Activation` has four variants, so **`Relu` and `Gelu`
both became `Silu`** — and the compute enum already had `ReLU` and `GeluExact`
waiting to receive them, so nothing was lost for lack of a destination. The
damning part: `larql-compute-metal`'s `assert_metal_activation_supported`
exists precisely to "fail loud rather than silently routing GeluExact / ReLU
layers to SiLU (the prior behaviour, which produced wrong logits with no
signal)". That guard is correct, tested, and **was unreachable** — the
wildcard upstream guaranteed it could never be handed either variant. The fix
was applied at the consumer and missed at the producer. Now three `From` impls
in `pipeline/enums.rs` are the one definition, exhaustive so a new variant
fails to compile; the five CPU MoE expert loops that carried the same
`_ => silu` wildcard route through one `gate_up_is_gelu_tanh()`. No in-tree
architecture returns `Gelu` or `Relu`, so this is behaviour-preserving today —
it converts a latent silent-wrong into the loud refusal that already existed.

**Eighth instance — the vindex format could not carry `rope_scaling`.** This
one is an *omission*, not a wildcard: `VindexModelConfig` simply had no such
field, so `from_arch` dropped it and every vindex-served model read back
`rope_scaling: None`. `google/gemma-3-4b-it` declares
`{"factor": 8.0, "rope_type": "linear"}`; served from a vindex it ran with a
position divisor of **1.0 on its five global layers**, rotating them eight
times faster than the checkpoint asks. This is a served-model correctness
defect, not a test gap — and it is why M5 above needed an env override to mean
anything.

Note *why it was invisible*: CPU and Metal both read the same `index.json`, so
both were wrong identically and `test_decode_consistency` stayed green.
**A parity gate cannot see a defect in a config that both of its arms share.**
That belongs next to R14 (gate–claim congruence) as a standing rule.

Fixed by `RopeScaling::to_config_json` (an inverse of the detector's parser,
with per-family `parse(emit(x)) == x` round-trip tests, because an inverse
that drifts from its forward is worse than none) plus five previously-dropped
fields — `rope_scaling`, `attn_logit_softcapping`, `swiglu_limit`,
`norm_topk_prob`, `tie_word_embeddings`. That last one is H5a's field: the
H5a fix could not reach a vindex-served model. All are `#[serde(default)]`, so
**existing vindexes still load and still answer `None` — they must be
re-extracted to pick the values up.** The two production writers that had
open-coded their own copy of `from_arch` now call it.

Two guards so the class cannot recur: `model_config_persists_every_forward_
affecting_field` scrapes both structs and fails on any `ModelConfig` field
with no home (a deliberate not-persisted list carries the reasons — MLA
geometry and `has_vision_config` are named as real gaps, `embedding_multiplier`
as already carried via `embed_scale`), and
`gemma3_global_rope_divisor_survives_the_vindex_round_trip` asserts the
divisor end to end with a precondition that the source arch had one.

---

## Cleanup / consolidation track (added 2026-06-12)

Standing recommendations from the 2026-06-12 review, distinct from the
hardening bug-fixes above: this is the maintenance-debt layer. The repeated
observation across both reviews is that bugs in this codebase come back from
the dead through **duplication** — parallel paths created to avoid
destabilising a parity-verified one, then maintained in lockstep by comment
("keep in lockstep" twins, `KernelHandle` bypassed at 2 new sites, 6 copies
of the env-flag helper with diverging semantics). The corrective habit, made
policy:

> **Prefer a parameter on the existing path over a parallel path.** A new
> code path needs the same justification as a new crate: a reason the
> existing one cannot be parameterised. Opt-in experiment paths are fine,
> but they get a removal-or-promotion condition when added, not after.

Themes, in leverage order (concrete first steps live in hardening items
7–10 above; this section tracks the policy-level work):

1. **One forward-pass spine** — the five parallel layer-step loops in
   `larql-inference/vindex/kquant_forward/` are the canonical instance.
   ADR first (what is the shared layer-step contract: sentinels, MoE
   detection, KV dispatch, capture hooks), then fold
   `hidden`/`prefill`/`decode_step`/`decode_step_direct`/remote-FFN onto
   it. Sequenced behind the C10 residency arc (same hot files). The
   padded-down twin extraction (hardening item 8) is the cheap pilot for
   the same move one level down. [larql-inference, larql-compute]
2. **Flags → config** — beyond the registry (hardening item 7): any
   `LARQL_*` flag that changes numerics and has survived its experiment
   (e.g. the Q4K residency trio once C10 lands) gets promoted to real
   config/CLI surface or deleted; env vars stay for diagnostics and
   short-lived experiments only. Uniform parsing through the
   `options.rs` taxonomy so `=true` vs `=1` can never again silently
   change what a bench measured. [workspace]
3. **Experiment-path lifecycle** — opt-in paths that lost their A/B keep
   accumulating (ADR-017 covers shaders; nothing covers CPU/env paths).
   Extend the ADR-017 rule workspace-wide: every opt-in path carries a
   retention rationale + revival story, and reviews may delete any that
   lack one. Current deletions/decisions owed: 4 unreferenced Metal shader
   modules, `model-compute` (no second consumer), `larql-experts`
   integration status, `test_utils.rs` out of larql-inference's public
   API. [workspace]
4. **API surface honesty** — `larql-inference/vindex` re-exports ~28
   implementation-named functions (`predict_kquant_*` variants); external
   callers choose forward paths by fuzzy naming. After (1), expose one
   facade that dispatches internally; deprecate the variants. Pairs with
   the Engine/StatePolicy framing already proposed. [larql-inference]
5. **Coverage debt** — per-file ≥90% floor policy vs reality:
   `larql-inference` 70.7%, `larql-cli` 12.0% (snapshot 2026-05-16).
   Raise toward the floor opportunistically as files are touched by (1)
   and (4) rather than as a standalone sweep; new/split files land at
   ≥90% (existing policy). [larql-inference, larql-cli]
6. **Scratch-artifact hygiene** — underscore-prefixed bench baselines
   (`bench/baselines/_*.json`) are scratch by convention but accumulate
   untracked/half-tracked; adopt the rule that `_`-prefixed artifacts are
   gitignored, and reconciled baselines get real names + a RUNBOOK line.
   [bench]

---

## BitNet b1.58 integration hardening (added 2026-06-20)

Native-ternary BitNet (`microsoft/bitnet-b1.58-2B-4T`) landed on
`feat/quant-ternary-a8`. The **W1.58·A8 kernel work in `larql-compute`
is the strongest part and fits the architecture cleanly**: ternary matvec
lives in `cpu/ops/ternary_matvec.rs` alongside the q4k/q6k kernels, follows
the `_into` allocation-free convention, and is parity-disciplined —
dequant-reference parity, **bit-exact NEON-vs-scalar across `cols % 16`
tails**, and shape-guard rejection tests. NEON gives ~12–13× the f32
reference on BitNet shapes; x86_64 has the scalar-A8 ~2.4× today.

The **system-level integration is a deliberate parallel stack** — a fresh
instance of the consolidation-track theme above (parallel path created to
avoid destabilising a parity-verified one). BitNet bypasses every shared
seam: `QuantFormat`/`FormatRoute` dispatch, the `KvEngine` trait,
`larql-kv::KvCache`, the models arch registry, and the vindex build
pipeline. This is documented as intentional pending the FormatRoute
roadmap and is a defensible MVP posture for a narrowly-scoped path. The
module comment that gated folding-in on "once the quantised-activation
kernel exists" now has its precondition met (this branch), so the
promotion-or-isolation decision the policy box demands is no longer
blocked — it is made explicit below (the G1–G4 structural items), with the
2026-06-20 hardening pass clearing the quick wins first.

Per-crate status:

- **`larql-compute`** — fits. Kernel + tests land in the right module with
  the right discipline. The earlier doc inaccuracy (header implied the path
  already routes through `FormatRoute`) is **reconciled**: `QuantFormat`
  (`pipeline.rs:25-34`) still has no ternary variant, `from_registry_tag`
  (`pipeline.rs:104-116`) maps no ternary tag, and the `QuantMatVec` dispatch
  trait has no ternary arm — `BitLinearWeight` is reachable only by direct
  call, and the docstring now says so plainly (dispatch integration is the
  open structural item, not done).
- **`larql-inference`** — bespoke parallel stack. `BitnetModel` is not a
  `KvEngine`; `BitnetKvCache` is a hand-rolled `Vec<Array2<f32>>` rather
  than `larql-kv::KvCache`, so none of the KV append-in-place / windowing /
  surgery work applies. Entry is direct (`load_bitnet_model` → `generate`),
  no unified dispatch picks between dense and ternary. (The hot path now
  quantises the shared activation once per Q/K/V and gate/up, and the header
  comment reflects the now-met kernel precondition — the structural
  KvEngine/shared-cache fold remains open.)
- **`larql-vindex`** — `bitnet_writer`/`bitnet_loader` write a `bitnet/`
  sidecar (`*.i2s` + `scales.f32` + `bitnet_layout.json`) and patch
  `index.json` with a `bitnet_layout` field, independent of the
  `quant: QuantFormat` enum field (two parallel quant-tag mechanisms).
  Writer is a post-build patch from `convert_cmd.rs`, not part of the build
  pipeline.
- **`larql-models`** — was the fragile seam; **FIXED 2026-06-20**.
  `"bitnet-*"` is now recognised explicitly in `detect/mod.rs` and routes to
  a thin named `BitnetArch` (`architectures/bitnet.rs`, `family() ==
  "bitnet"`, Llama-style defaults, `norm_eps` honoured from config) instead
  of silently collapsing to `GenericArch`. Native-ternary inference is still
  served by the `larql-inference` ternary path, not this trait; `BitnetArch`
  is the home for first-class overrides when BitNet graduates. Covered by
  `test_detect_bitnet_is_explicit_not_generic`.

### Completed — hardening pass (2026-06-20)

The quick-win review items landed; all touched crates build clean and
clippy-clean (`--all-targets`), tests green (compute ternary 19/19,
inference ternary 28/28 incl. the FFN A8-vs-f32 parity gate, models detect
59/59):

1. ✅ **[larql-models] Killed the silent `GenericArch` fallback** — explicit
   `bitnet-*` recognition → thin named `BitnetArch`; `norm_eps` honoured;
   `test_detect_bitnet_is_explicit_not_generic`. *(was P1)*
2. ✅ **[larql-compute] Reconciled the `ternary_matvec.rs` docstring** — no
   longer implies the path routes through `FormatRoute`; states that dispatch
   integration is the open item and the kernel is reached by direct call.
3. ✅ **[larql-inference] Reuse one activation quant** — Q/K/V and gate/up
   quantise the shared activation once (`quantize_activation_i8` +
   `matvec_i2s_a8_into`) across all five forward sites. Bit-exact (parity
   tests unchanged), saves the repeat int8 quantise per projection.
4. ✅ **[larql-inference] Refreshed the `ternary.rs` header comment** — the
   "fold in once the quant-activation kernel exists" precondition is now met;
   the comment frames the fold as live roadmap work, not a missing dependency.
5. ✅ **[larql-compute] x86_64 gap documented** — verified already clear at
   the dispatch entry (`matvec_i2s_a8_into`: "scalar int8 elsewhere — AVX2
   twin is the x86_64 follow-up") and the status block.

Owed back to the user (not a code change):

6. **[git hygiene] Split the `pipeline_layer.rs` refactor** — the
   `attn_str_to_format`/`ffn_str_to_format` → `from_registry_tag` dedup is a
   sound single-source-of-truth cleanup but is **orthogonal to BitNet**
   (BitNet never flows through `resolve_ffn_weights`). Land it as its own
   "refactor: dedupe tag→format mapping" commit, not inside the feature.

### Remaining — graduation to first-class (status 2026-06-20 after scoping)

Close contact with the code (two scoping passes) revised this list: only G1
was cleanly doable on the current machine. G2 and G4 hit genuine blockers and
G3's framing was falsified. Detail per item:

- **G1 — `QuantFormat` ternary variant + dispatch — ✅ DONE 2026-06-20.**
  Added `QuantFormat::I2S` (+ `registry_tag`/`from_registry_tag` round-trip,
  `is_ternary`), a dedicated `QuantMatVec::ternary_matvec` method (a
  `BitLinearWeight` carries the per-channel scales the `&[u8]` `quant_matvec`
  signature can't), and a `CpuBackend` impl on the best-available A8 kernel.
  `quant_matvec` returns `None` for I2S (loud, like Q8_0); Metal panics (no
  ternary shader). Registry-reachable now, not only by direct call. Tested +
  clippy-clean.
- **G2 — `KvEngine` impl + shared cache — BLOCKED on a breaking trait change
  (your call).** Scoping (kv_engine.rs / larql-kv) found `KvEngine::prefill`
  and `decode_step` are typed `(&ModelWeights, &dyn FfnBackend, …)` — dense
  f32 weights. `BitnetModel` holds ternary `BitLinearWeight`s, an incompatible
  container. Making BitNet a first-class `KvEngine` needs EITHER a
  workspace-wide generalisation of the weight parameter to a `dyn` trait (real
  breaking change, hot-path dyn dispatch — heavy for an example-only feature)
  OR a type-lie (accept `&ModelWeights`, ignore it, route to an owned
  `BitnetModel`) — rejected as exactly the parallel-path anti-pattern the
  policy box forbids. The cache-only sub-part (`BitnetKvCache` →
  `larql-kv::KvCache`) is shape-compatible but marginal (the shared cache
  doesn't append-in-place for this path either) and carries hot-path parity
  risk, and it does NOT reach "first-class". → Decision: take the breaking
  trait generalisation, or leave BitNet isolated-but-explicit (recommended
  until a second consumer exists).
- **G3 — vindex quant-tag unification — GOAL FALSIFIED; struck.** Scoping
  found BitNet vindexes are **mixed**: the dense scaffold (`embed`, `lm_head`,
  `output_norm`) is f32 and loaded by the standard loader with
  `skip_attn`/`skip_ffn`, while only attn/ffn are ternary. `quant:
  QuantFormat::None` is therefore *correct* for the dense loader — setting
  `quant: I2S` would mislead it into decoding the embedding as ternary. A
  single `quant` tag cannot represent a mixed model; the two-field design
  (`quant` for the dense scaffold + `bitnet_layout` manifest for the ternary
  tensors) is the right shape. The only survivor is a modest mechanical
  cleanup — move `bitnet_writer` from a `convert_cmd` read-modify-write
  post-patch into the build pipeline so `index.json` is written once — low
  payoff, invasive through the shared build path. Not pursued.
- **G4 — AVX2 `_mm256_sign_epi8` twin — BLOCKED on an x86 build/test box.**
  Design is clear (decode trit codes → a `{-1,0,+1}` int8 control, one
  `_mm256_sign_epi8(x, control)`, widen-accumulate; bit-identical to scalar).
  But this aarch64 machine can neither runtime-validate the SIMD NOR even
  compile-check it (cross-`check` fails in the C-FFI build script — no
  `x86_64-linux-gnu-gcc`). Committing unbuilt, unvalidated intrinsics violates
  the parity discipline. → Defer to an x86 dev box / Linux CI runner, where
  the scalar-vs-AVX2 bit-exact test (already the pattern for the NEON twin)
  can gate it. x86_64 keeps the correct scalar-A8 path (~2.4×) meanwhile.

### Productization plan (decision: PRODUCTIZE, 2026-06-20)

Direction chosen: make BitNet a real served path, not a validated experiment.
Scoping fixed the magnitude — BitNet has **zero CLI/server hookup today**
(`load_bitnet_model` is called only from the example); the dense run path is
`layer_graph::generate_streaming` over the engine dispatch; `run_cmd::run()`
is a chain of early-return mode branches (experts / ffn / moe / image). Three
stages, smallest-blast first:

- **P-A — Serve BitNet from `larql run` (CLI) — ✅ BEHAVIOUR-VERIFIED
  2026-06-20.** `run_cmd::run()` branches on `config.bitnet_layout.is_some()`
  and drives `ternary::generate_streaming_bitnet` (greedy stream + chat REPL),
  bypassing the dense `walk_cmd` path. Smoke-tested against
  `~/larql-vindex/bitnet-2b.vindex`:
  `larql run <vindex> "The capital of France is" -n 16` →
  `" Paris. Paris is a city that is known for its rich history, culture,"` —
  deterministic across runs. **This greedy output is the P-B regression
  oracle** (saved local-only at `bench/oracles/bitnet_2b_capital_of_france.txt`;
  not committed — depends on the >1 GB vindex; repro = the command above).
  Bridges at the run layer; does NOT make BitNet a `KvEngine`.
  *Remaining (deferred to AFTER P-B, deliberately): server stream-route wiring
  + chat-template/sampling parity — wiring the server now would thread
  `&ModelWeights` through the hot path B1 is about to strip; wire once, after.*
- **P-B — First-class `KvEngine` (the structural refactor).** Blast radius
  measured: **8 production engine impls + ~171 `prefill`/`decode_step` call
  sites + `EngineKind`/`AnyEngine`**. The one-way-door is the trait shape;
  pick before the breaking change:
  - **B1 (CHOSEN 2026-06-20): engines own their weights.** Move `&ModelWeights`
    out of `prefill`/`decode_step` into engine construction (engines hold
    `Arc<ModelWeights>`); `BitnetEngine` holds `Arc<BitnetModel>`.
    **Read-only check (done):** dense `prefill`/`decode_step` and the
    `*_resident` path take `&ModelWeights` (read-only — B1-clean); BitNet
    weights are final (no mutation). BUT the **quant-resident path
    (`prefill_quant`/`decode_step_quant`) takes `&mut ModelWeights`** — it
    memoizes resident-quant buffers back into the struct. So B1 is NOT pure
    mechanical churn; it bundles one design sub-decision for that path:
    **(a)** relocate the resident-quant memoization out of `ModelWeights` into
    engine-owned derivative state (recommended — lands on the StatePolicy
    split: canonical weights immutable, derived caches are engine state), or
    **(b)** `Arc<ModelWeights>` + interior mutability (`OnceCell`/`RwLock`) on
    just the resident-quant fields (smaller, keeps derived state in the
    canonical struct). Cost = ~171 mechanical sites + this sub-decision.
  - **B2: `&dyn ModelSource` param.** New trait; `&ModelWeights` auto-coerces
    so most call sites are untouched, but the trait must mirror the slice of
    `ModelWeights` engines use, and BitNet panics on dense-only methods.
  - **B3: `ModelWeights` gains a ternary representation.** Smallest type diff
    but leaks ternary-awareness into the dense engines — rejected.
  Do P-B as its own PR after P-A proves the path; hold parity per the 7-spec
  `resident_identity_tests` discipline.

  **Grounded execution stages (B1a chosen, real-code scope 2026-06-20).** The
  `&mut` has a single chokepoint:
  `larql_inference::vindex::dequant::ensure_attn_tensors_dequantised(&mut weights, index)`
  (`vindex/dequant.rs:35`) — it dequantises Q4K Q/K/V/O into `weights.tensors`
  (a `HashMap`) keyed by `arch.attn_{q,k,v,o}_key(layer)`, idempotent, and the
  forward reads them back from that map. Pure derivative state. Stages, each
  compilable + checked against the captured greedy oracle:
  - **P-B.1 — relocate the dequant cache (HOME LOCKED: engine, not
    `ModelWeights`; concurrency evidence 2026-06-20).** Move the dequantised-
    attention `HashMap` out of `weights.tensors` into **engine-owned** state and
    consult it at the forward's tensor-read sites (resolver: engine cache →
    canonical weights). Drops the `&mut` from `prefill_quant`/`decode_step_quant`.
    *Why engine, not an interior-mutable `RwLock` field on `ModelWeights`:* the
    scratch is **transient** (per-layer evicted for the memory bound) → per-
    forward state, not a persistent cache. The server holds one
    `weights: OnceLock<RwLock<ModelWeights>>` and **serializes every generation
    behind an exclusive write lock** (`state.rs:186 lock_weights_for_gen`,
    used by all OpenAI gen routes) *specifically because* this dequant mutation
    makes weights non-immutable ("concurrent reads block while a generation is
    in flight"). An interior-mut `RwLock` field can't lift that — two forwards
    sharing one `Arc` would clobber each other's evicting scratch, so gen would
    still have to serialize (and the dense 117 tok/s path would pay a per-resolve
    read-lock + `ArcArray` clone for a scratch that's always empty for it).
    Engine-owned scratch makes `ModelWeights` **truly immutable** →
    `Arc<ModelWeights>` shared across **concurrent** generations, each engine
    its own cache (no lock, no race, no tax) — the actual payoff of the refactor.
    Resolver threads as `&mut self.dequant_cache` from the engine (it's already
    the `&mut self` forward context). Touches the Q4K residency path —
    `resident_identity_tests` + the oracle guard it. *(A provisional
    `RwLock`-in-`ModelWeights` impl was tried this session and reverted on this
    evidence before the read-site/trait sweep could cement it.)*

    **P-B.1 status (2026-06-20): signature stages DONE+committed, relocation
    set up + reverted-to-green.** Done behavior-identical: `WeightsView`/
    `DequantScratch` foundation; **Stage 1** (`run_attention_with_kv_backend` →
    `WeightsView`, ~22 `dense()` wraps); **Stage 2a** (`dense_ffn_forward` →
    `WeightsView`, `WeightFfn`/`BackendFfn` wrap `dense()` internally so the 326
    `WeightFfn` construction sites stay untouched). The workspace-spanning
    cross-crate signature diff is banked, decoupled from any behavior change,
    each proven byte-identical by parity tests.

    **Stage 2b (the relocation, behavior-changing) was reverted, and the reason
    is categorical, not cost.** The first four blast-radius escalations
    (RwLock→engine, cross-crate, 326-`WeightFfn`, decode reader) were all
    *compiler-visible*: change a signature, the compiler enumerates callers. The
    fifth is *type-system-invisible* — a reader that resolves `weights.tensors`
    via `Deref` (canonical) while the scratch sits in an unconsulted
    `DequantScratch` compiles clean, runs, and is wrong **only on the decode
    path under a real Q4K vindex**. Holding that on a red tree across a session
    boundary strands a miscompilation `cargo check` can't recover, so revert to
    Stage 2a green was the only correctness-preserving move.

    **Silent-break closure = make the miss LOUD, not enumerate readers.** The
    grep inventory (`tensors.get(&arch.attn_*/ffn_*`) is *current, not complete*
    — blind to precomputed-key reads, prefix iteration, accessor methods that
    `.get` internally. The design fix: for a quant model those dequant keys were
    **never** in canonical `tensors` (they only ever existed as the forward-time
    mutation target being relocated), so if the relocation inserts only into
    scratch and leaves canonical untouched, a missed reader resolves `None` →
    the existing `.unwrap_or_else(panic)` / `?`-bail fires on first decode, on
    any vindex. **Design property to enforce: leave canonical genuinely empty of
    dequant keys (not shadowed)** → misses are loud by construction. Grep scopes
    the conversion; the runtime catches its misses.

    **Stage 2b entry conditions (all met — no upstream gap this time):**
    (1) a Q4K vindex — **`~/larql-vindex/qwen3-0.6b-q4k.vindex` exists**;
    (2) a **multi-token DECODE** oracle captured at Stage 2a (NOT prefill-only /
    single-token — the decode reader is exactly the one Stage 1 missed, so a
    prefill-heavy capture has a blind spot the shape of the bug); byte-identical
    decode vs Stage 2a is the regression spine; (3) the canonical-empty shaping
    above. With these, the reader conversion is mechanical and the silent-break
    class is closed by construction.

    **Stage 2b progress + the reader-family finding (2026-06-20).** Done +
    committed behavior-identical: all THREE "primary" quant-path readers now
    take `WeightsView` — `run_attention_with_kv_backend` (Stage 1),
    `dense_ffn_forward` (Stage 2a), `run_attention_block_decode_step_backend`
    (Stage 2b-pre). The Q4K **decode** oracle is captured
    (`bench/oracles/q4k_qwen3_history_of_computing.txt`, 24-token greedy on
    `qwen3-0.6b-q4k.vindex`). The relocation proper (inserters→scratch,
    `ViewFfn`, wire the cached prefill+decode loops, drop `&mut`) was drafted
    and reverted to green when the **secondary** loops (`hidden.rs`,
    `interventions.rs`) surfaced that the reader set is *still* expanding on
    contact: they reach attention through `run_layer_with_ffn` →
    `run_attention_inner` / `run_attention_with_kv_cache` →
    `run_attention_block_core` (block.rs) + `run_attention_block_gpu` (gpu.rs)
    — **un-converted readers the grep never surfaced**, exactly the
    "current-not-complete" inventory. So the true relocation scope is "convert
    the **whole attention-reader family**" (with_kv_backend✓ / decode✓ /
    block_core / block_gpu / inner / with_kv_cache), each a Stage-1-style
    cascade through a widely-used fn — several more passes, not one. The cached
    decode path (the oracle path) wired cleanly; the secondary loops need the
    rest of the family first. **Loud-break makes this safe to do incrementally**
    (canonical empty of dequant keys → a missed reader gets `None` → the
    existing `.unwrap_or_else(panic)` fires on first decode, loud not silent),
    so each remaining reader can be converted + the loop wired + validated
    against the oracle without a silent miscompilation risk.

    **✅ DONE (2026-06-20, `9650582e` + `f0da87cc`).** The whole-family
    conversion + the relocation both landed. (1) `9650582e` converted the
    entire attention-reader family to `WeightsView` — `block_core`, `block_gpu`,
    `run_attention_inner`, `run_attention_with_kv_cache`, `run_layer_with_ffn`,
    `run_layer_with_capture[_hooked]`, `run_attention_public` + the block.rs
    family — ~100 callers `dense()`-wrapped across compute/inference/kv/cli/
    server/examples, behavior-identical (the compiler enumerated the family for
    me; the cascade bottoming out *is* the proof the inventory is now complete).
    (2) `f0da87cc` did the relocation: the production decode path
    (`predict_kquant_prefill/decode_step` + `hidden` + `interventions`)
    dequantises into a forward-local `DequantScratch` resolved via
    `WeightsView::with_scratch` + `ViewFfn` — **`weights` is `&ModelWeights`
    (immutable, Arc-able) on the decode path, `&mut` dropped.** Bulk
    f32-fallback + dev drivers (KvEngine `*_quant` trait defaults, all larql-kv
    quant-engine overrides, apollo, ov_rd CLI, the lql relation resolver, the
    vision/image CLI, examples) keep in-`weights` behaviour via `*_resident`
    shims (dequant → scratch → merge into `weights.tensors`). Validated:
    workspace `--all-targets` green, clippy 0 warnings, 50 kquant + 13 dequant +
    resident_identity tests pass, decode **byte-identical to the oracle** both
    after the family conversion and after the relocation. **Follow-up:** the
    `*_resident` bulk path is still `&mut` — dropping it needs engine-owned
    scratch state (folds into P-B.2/P-B.3, not a blocker); loud-break guards it.

    **P-B.1b — "no shims" full sweep (scoped 2026-06-20; WIP stashed).** Going
    for zero `weights.tensors.extend` shims surfaced a **second `kquant_forward`
    implementation**: the production `larql run` decode dispatches via KvEngines
    → `coarse_prefill` → **larql-compute's** `kquant_forward` (1005 lines), NOT
    the larql-inference copy (1772 lines) that P-B.1 relocated. The
    larql-inference copy serves the direct-`predict_kquant`/AVE/hidden paths and
    is validated by the 50 kquant unit tests; **the e2e oracle actually
    exercises larql-compute's copy** (so the family conversion — shared
    `run_attention_*` — was oracle-validated, but the larql-inference relocation
    was unit-test-validated, not oracle). The full no-shim change is large and
    interconnected:
    (1) relocate **larql-compute's** `kquant_forward` too (cached/decode loops →
        forward-local scratch + `ViewFfn`) — DONE in the stash, the real oracle
        path now no-shim;
    (2) `KvDispatch` (5 methods) + the 7 dispatch helpers + `AsyncComputeBackend`
        + cpu/metal impls → `WeightsView` — DONE in the stash;
    (3) coarse path (`coarse_prefill`/`coarse_decode_step`) drops `&mut` (delegates
        to the now-`&` `predict_kquant_*`) — DONE;
    (4) `KvEngine`/`RetrievalEngine` trait quant methods → `&ModelWeights`;
        `RetrievalEngine::prefill_quant` default → loud error (apollo overrides;
        the `ffn`-less `prefill` can't thread a scratch) — DONE;
    (5) **engine-scratch design** (validated on `StandardEngine`): each engine
        owns a `dequant_scratch: DequantScratch`; `do_prefill`/`do_decode_step`
        build `WeightsView::with_scratch(weights, &self.dequant_scratch)` — the
        view borrows `self.dequant_scratch` while `self.handles`/`self.backend`
        are borrowed disjointly, so **no take/restore dance**; `prefill_quant`
        dequants into the field, no merge. StandardEngine compiles clean.
    Remaining (the stash is mid-sweep, ~29 errors): the **other 7 engines**
    (no_cache, boundary×2, markov×2, turbo, unlimited, apollo) each need the
    same field + view-thread + `&mut`-drop, plus their forward helpers
    (`kv_prefill_run` + the `generate_cached_*` loops in larql-kv/generation.rs,
    apollo's `forward_raw_logits`) converted to `WeightsView`, then the dev
    drivers (ov_rd/lql/vision/examples) + delete the `*_resident` shims. The
    pattern is mechanical but keeps surfacing forward helpers on contact (the
    reader-family expansion, now in the engine layer) — a focused dedicated pass.
    `git stash list` → "no-shims WIP".

    **Convergence measurement (2026-06-21).** Resumed the sweep and pushed into
    the engines. Additional shared decode/recompute readers converted to
    `WeightsView` (these are real foundation, beyond the StandardEngine work):
    `run_attention_block_decode_step_auto` + `_auto_inplace` (the resident-decode
    switcher used by **5 engines** — q4k-direct branch reads native index bytes
    via `.canonical()`, f32 branch threads the view), `kv_prefill_run`
    (no_cache + standard), `recompute_kv` + `attn_kv_projection_weights`
    (boundary + markov), and **NoCacheEngine** fully (field + view-thread +
    `&mut` drop, clean). **But the larql-kv error count DIVERGED as I converted:
    28 → 45 → 67.** Each engine's `walk.rs`/`compute.rs` forward module is a deep
    chain (engine → walk → recompute → projection → `weights.tensors`), and
    converting one helper exposes its callers + their internal reads. This is the
    reader-family expansion at its widest — **converting every engine's full
    forward/recompute internals** (~30-50+ functions across 6 engine modules +
    apollo's `forward_raw_logits` + the dev drivers). The diverging count is the
    decision signal: this is a **staged multi-session refactor**, best done one
    engine module at a time (convert its walk+compute internals, validate that
    engine against a per-engine oracle, commit), not a single grind. The shared
    helpers above are the foundation already laid; the per-engine internals are
    the remaining bulk. WIP re-stashed.

    **✅ DONE (2026-06-21, `379885ed`).** The diverging count (28→45→67)
    **converged to 0** as each engine module got the same template — the
    "diverging" was the compiler enumerating the work, not the work being
    unbounded. Every KvEngine (standard, no_cache, markov_residual,
    markov_residual_codec, boundary_per_layer, boundary_kv, turbo_quant,
    windowed_checkpoint, apollo) now owns a `dequant_scratch` field; quant methods
    dequant into it and the forward resolves through `WeightsView::with_scratch`
    — **0 `&mut ModelWeights` quant methods, 0 `weights.tensors.extend` merges on
    the engine/serving path.** Per-engine pattern: bulk-convert the engine's
    `walk.rs`/`compute.rs`/`executor.rs`/`cold_tier.rs`/`dispatch.rs` to
    `WeightsView` (canonical reads — `embed`/`run_ffn`/`layer_ffn_or_moe`/
    `BackendFfn`/`WalkFfn`/native-q4k — via `.canonical()`/`&weights`; attn reads
    via the view), then the engine adds the field + threads `with_scratch` to its
    forward calls + drops `&mut`. Also converted: `LayerExecutor` trait +
    local_walk, `recompute_kv` + `attn_kv_projection_weights` (explicit lifetime),
    `auto`/`auto_inplace` (5 engines), `kv_prefill_run`, `forward_raw_logits`/
    `forward_from_layer` + raw.rs internals (`ViewFfn`; `hidden_to_raw_logits`/
    `apply_logits_transform` stay `&ModelWeights` = lm_head canonical). The
    `*_resident` helpers (`ensure_attn_..._resident` etc.) deliberately **remain**
    for ~58 dev/research call sites (ov_rd CLI, lql resolver, vision CLI,
    examples) that own a `&mut ModelWeights` and run one-off forwards against
    canonical `weights.tensors` — a documented separate API, not serving-path
    shims. Validated: workspace `--all-targets` green, clippy 0, **766 larql-kv +
    40 kquant + resident_identity + 4 dispatch_parity (cross-engine bit-parity)**
    tests pass, decode **byte-identical to the oracle**, and
    markov-rs/unlimited/turbo-quant/no-cache smoke-tested coherent at runtime.
    **Engine/serving path is now fully `&ModelWeights` — P-B.2 (Arc-owned) is
    unblocked with no remaining `&mut` to chase in the engines.**
  - **P-B.2 — Arc-owned weights.** Every weight param is now `&ModelWeights`;
    move it into engine construction (engines hold `Arc<ModelWeights>`) and drop
    the param from prefill/decode/quant/resident/executor variants. ~171 call
    sites (all production, 0 in test files) + `EngineKind`/`AnyEngine`.
    Compiler-driven — the safe kind of large churn.
  - **P-B.3 — `BitnetEngine` + dispatch.** New engine holding
    `Arc<BitnetModel>`, `impl KvEngine` over the ternary forward; add the
    `EngineKind`/`AnyEngine` arm so unified dispatch picks ternary vs dense.
  - **P-B.4 — validate** against the oracle (greedy "Paris…" byte-identical) +
    the engine parity suite.
  Best run in an isolated worktree so `main` stays stable through the change.
- **P-C — G4 AVX2 + x86 CI.** Add a Linux-x86 CI job; land the AVX2 twin gated
  by the scalar-vs-AVX2 bit-exact test (the NEON-twin pattern). Independent of
  P-A/P-B; unblocks G4's environment blocker.

---

## Demo narrative

### Act 1 — "The model is the database"
Run Gemma 3 4B or 4 26B locally. The vindex is the model; `larql run` queries it.
Show: latency, footprint, `larql walk` tracing a fact through layers.

**Status**: Works end-to-end. Needs chat-template + EOS fix so it doesn't loop.

### Act 2 — "The experts live elsewhere" (reframed per ADR-019)
**Original framing** (multi-machine grid for 671B-class MoE): demoted to P2.
The "elsewhere" was always a stretch for a substrate, and multi-machine
production-engineering doesn't accelerate any current experiment.

**Reframed**: single-machine expert dispatch on Gemma 4 26B-A4B. The shipped
gRPC grid (1-shard local) demonstrates expert routing; the demo can show
expert-by-expert activation tracing on one box, which is closer to the
substrate story (mechanism transparency) than to the production-engine
story (distributed inference). Replace "experts live elsewhere" framing
with "experts are addressable" framing.

**Status**: Server-side grid works (single-machine). Multi-machine items
(critical-path 5–10, RemoteExpertBackend, `/v1/expert/*`, reliability)
are P2 per ADR-019.

### Act 3 — "Replace an expert"
Swap expert 42 at layer 18 for a custom one. Observe the model's behaviour change.

**Status**: Works on single-machine via VID4-style approach (already
shipped publicly as VID4). Unaffected by ADR-019.

### Act 4 — "I killed attention" (future, video VID7)
Profile attention heads on a static template. Show 91% of heads produce
identical outputs across entity substitutions. Replace those with cached
lookups; remaining 9% run normally. Same outputs, fewer matmuls.

**Status**: Sketched in chat, not drafted. Gated on KU1 (static fraction at
31B-scale) and MTP6 (acceptance-rate evidence). See Video pipeline above.

---

## P0 — Mechanistic surface (lazarus parity)

Driver: replace the chuk-mlx engine in `chuk-mcp-lazarus` with larql. Lazarus
exposes ~77 inference-time MCP tools (capture, ablate, patch, steer, probe,
DLA, KV-surgery). Larql is currently strong on weight-level edits (MEMIT, KNN,
LQL) and weak on inference-time inspection/intervention. The 77 tools collapse
to one missing primitive: a **programmatic forward-hook system**. Once that
lands the rest is mostly Python wrappers.

| # | Item | Crate | Status |
|---|------|-------|--------|
| M1 | `LayerHook` trait + CPU plumbing (read + write) | larql-inference | shipped |
| M2 | `RecordHook`, `ZeroAblateHook`, `SteerHook`, `CompositeHook` | larql-inference | shipped |
| M3 | Activation patching (cross-prompt residual swap) | larql-inference | shipped |
| M4 | Full logit lens — `logit_lens_topk`, `track_token`, `track_race` | larql-inference | shipped |
| M5 | `KvCache::{get_layer, set_layer, clear_layer, clone_layer_from, clone_layer_position_range}` | larql-inference | shipped |
| M6 | Hooks during multi-token generation (`generate_cached_hooked` on CPU; Metal `generate` stays fast by design) | larql-inference | shipped |
| M7 | `W_E` / `W_U` + `embedding_neighbors` + `project_through_unembed` | larql-inference | shipped |
| M8 | pyo3 `PyWalkModel` mech-interp methods (capture / ablate / steer / patch / lens / generate_with_hooks) | larql-python | shipped |

Detail in `larql-inference/ROADMAP.md` § Mechanistic hooks (lazarus parity).

---

## P0 — Best-in-class mechanistic interpretability engine

Driver: make LARQL's executed mechanisms queryable, attributable, patchable,
and reproducible. This is the layer above lazarus parity: not just hooks, but
evidence-grade traces and causal operators over the actual vindex-backed
inference path.

| # | Item | Crate | Status |
|---|------|-------|--------|
| MI0 | Faithful residual DAG: TRACE uses the canonical layer runner and pins additive reconstruction | larql-inference | shipped |
| MI1 | Python `WalkModel.trace()` / `patch_activations()` use `WalkFfn` instead of dense fallback | larql-python + larql-inference | shipped |
| MI2 | Backend-parametric donor capture and activation patching | larql-inference | shipped |
| MI3 | Strict trace artifacts: complete ordered chains, exact file length, `TRACE SAVE` requires `POSITIONS ALL` | larql-inference + larql-lql | shipped |
| MI4 | Golden parity: TRACE final residual/logits match canonical forward; extend to WalkFfn, patched vindex, Q4K, MoE | larql-inference | partial — dense/custom backend pinned |
| MI5 | Rich attribution objects: attention-head writes, FFN feature activations, router/expert decisions, provenance | larql-inference + larql-python | planned |
| MI6 | Causal operators beyond residual replacement: head/feature/router/expert/KV patching | larql-inference + larql-python | planned |
| MI7 | Q4K/MoE trace and patch parity with explicit precision caveats | larql-inference + larql-vindex | planned |
| MI8 | Python experiment ergonomics: batched prompts, donor/recipient alignment, causal metrics, reproducibility metadata | larql-python | planned |

Near-term order: finish MI4 parity coverage, then add attribution records where
the forward path already exposes data, then expand patching operators one
mechanism at a time.

---

## P1 — Research stack promotion: OV/RD → engine primitives

Driver: make LARQL one of the strongest practical mechanistic
interpretability stacks by promoting reusable experiment plumbing into
stable engine APIs, while leaving fast-moving hypotheses in
`larql dev ov-rd` and Python artifact analysis.

| # | Item | Crate | Status |
|---|------|-------|--------|
| R1 | Promote Q4K per-layer tensor insertion/removal from `ov_rd` into `larql-inference::vindex` | larql-inference | shipped |
| R2 | Add Q4K hidden forward with `LayerHook`/intervention support | larql-inference | shipped |
| R3 | Add pre-W_O capture/replacement hook adapters so experiments stop manually driving full layer loops | larql-inference | shipped |
| R4 | Define a compact research trace artifact contract for prompt ids, tokens, layer inputs, pre-W_O rows, oracle codes, logits, and metrics | larql-inference + larql-cli | planned |
| R5 | Keep PQ/address/codebook experiments in `larql dev ov-rd`; move only stable runtime contracts into engines | larql-cli | ongoing |
| **R6** | **Promote depth-fraction-law probe API into a stable engine primitive: `Model::probe_at_depth_fraction(f) -> Probe`. Probe consumes residual at the requested fractional depth (15% / 25% / 38% verified on Gemma/Llama/Mistral) and returns a 32-dim PCA + logistic regression classifier output. Single API consumed by MTP3 (drafter activation extraction), virtual-expert dispatch (Act 3 demo), and grammar-mask routing.** | **larql-inference + larql-models** | **planned — MUST land before MTP3 begins (MTP3's layer-choice validation depends on R6)** |

Rule of thumb: engine code owns reusable capture/intervention/runtime
primitives; `ov_rd` owns experiment orchestration, PQ variants, address
probes, and report schemas until a runtime contract survives repeated
experiments.

**R6 rationale (added 2026-05-09)**: depth-fraction probes have been
validated across three architectures with a 32-dim PCA + logistic
regression at 0.3% inference-time overhead. They currently live in
`larql dev ov-rd` as experiment code. Three downstream items implicitly
need this API: MTP3's drafter-input extraction layer choice, the Act 3
expert-swap demo's routing decision, and grammar-mask construction for
constrained generation. Promoting once removes duplicate implementations
in three places.

**Sequencing (added 2026-05-09)**: R6 must land **before** MTP3 begins.
MTP3 explicitly depends on R6 for layer-choice validation ("if R6 says
discriminative information matures at 0.85·N, there is potentially free
quality improvement available"). Without R6, MTP3 ships with Google's
default layer choice and the validation has to be redone after R6 lands —
duplicate work. Insert R6 between MTP2 and MTP3 in the implementation
order.

---

## P1 — Grid transport, self-balancing & benchmarking

Driver: minimum latency across on-device/LAN/WAN; elastic scaling without
manual shard pre-loading; reproducible, architecture-agnostic performance
evidence. All work is model-family-neutral — no hardcoded layer counts,
hidden sizes, or architecture assumptions.

Spec: ADR-0009 (wire format), ADR-0010 (QUIC), ADR-0011 (self-balancing),
ADR-0012 (benchmarking).

| # | Item | Crates | Status |
|---|------|--------|--------|
| GT1 | f16 wire default for all grid traffic; `LARQL_F16_WIRE_DISABLE` opt-out; Accept header negotiation | larql-server + larql-inference | **shipped 2026-05-07** |
| GT2 | i8 symmetric quantised residuals on wire; `LARQL_I8_WIRE=1` opt-in; per-position scale | larql-server + larql-inference | **shipped 2026-05-07** |
| GT3 | `LayerLatency` in `HeartbeatMsg` (proto + EMA tracker in server + per-layer routing in router) | larql-router-protocol + larql-server + larql-router | **shipped 2026-05-07** |
| GT4 | WebSocket token streaming (`generate` cmd + cancel); SSE for `/v1/chat/completions` confirmed wired | larql-server | **shipped 2026-05-07** |
| GT5 | Mode B gap-fill: `AvailableMsg → AssignMsg → download → ReadyMsg`; new `shard_loader.rs` | larql-router + larql-server | planned |
| GT6 | Dynamic rebalancing: `UnassignMsg` drain protocol + `rebalancer.rs` background task | larql-router + larql-server | **shipped 2026-05-08** |
| GT7 | QUIC transport for grid (`quinn` feature-gated); 0-RTT reconnect; per-stream independence for expert fan-out | larql-router + larql-server | planned |
| GT8 | `larql bench --bench-grid / --wire / --transport / --concurrent / --output json`; arch-agnostic from vindex config | larql-cli | planned |
| GT9 | Criterion micro-benchmarks: `wire_codec.rs` (encode/decode MB/s) + `routing.rs` (route/heartbeat/rebuild ns/op) | larql-inference + larql-router | **shipped 2026-05-07** |
| GT10 | CI regression gate: `scripts/bench-grid-regress.sh` + `bench/baselines/` committed JSONs | scripts/ | **shipped 2026-05-08** |

**Implementation order** (each step is a shippable increment):
~~GT3~~ → ~~GT1~~ → ~~GT2~~ → ~~GT4~~ → ~~GT9~~ → ~~GT5~~ → ~~GT6~~ → ~~GT8~~ → ~~GT10~~ → GT7

---

## P0 — Interpretability truthfulness + commit semantics

Driver: make the current edit model honest before the demo, then earn the
stronger "INSERT commits into weights" story. Today default `INSERT MODE KNN`
is a retrieval overlay persisted in `knn_store.bin`; `COMPILE INTO VINDEX`
bakes compose/MEMIT overlays but carries that KNN sidecar forward. That is a
snapshot/package operation, not a mechanical commit of the journal into FFN
features.

| # | Item | Crate | Status |
|---|------|-------|--------|
| T1 | Tag KNN overrides visibly in `INFER`, `EXPLAIN INFER`, and `TRACE` as post-logits retrieval events, including the model's unoverridden top-1 | larql-lql + larql-inference | planned |
| T2 | Fix decomposed `TRACE` to route through the shared layer sequence, including PLE/layer-scalar deltas or equivalent captured intermediates | larql-inference | shipped |
| T3 | Make Python `WalkModel.trace()` use the vindex `WalkFfn`/patch overlay rather than dense `WeightFfn` | larql-python + larql-inference | shipped |
| T4 | Replace gate-KNN absolute-dot feature ranking in interpretability displays with post-activation magnitude, or filter ghost negative gates after activation | larql-vindex + larql-inference | planned |
| T5 | Fix L1 FFN cache activation capture: cache activations with outputs or bypass cache when activations are requested | larql-inference | planned |
| T6 | Rename residual-capture embedding-neighbor fields (`top_token`) or add separate true logit-lens fields | larql-inference + larql-models | planned |
| T7 | Pin TRACE evidence with final residual/logit parity tests across dense, custom backend, WalkFfn, patched vindex, Q4K, and MoE paths | larql-inference | partial |
| C1 | Add explicit compile modes: default commit/materialize semantics vs `SNAPSHOT` preserving `knn_store.bin` | larql-lql + larql-vindex | design |
| C2 | Implement KNN materialization by lowering retrieval entries into compose/MEMIT/FFN edits, then dropping or marking committed sidecar entries | larql-lql + larql-vindex + larql-inference | planned |
| C3 | Add acceptance tests: session KNN equivalence, trace conversion, and generalization beyond stored prompts | larql-lql + larql-inference | planned |

Acceptance target for materialization:

```text
INFER(session_with_knn, q) == INFER(materialized_vindex, q)
```

for affected canonical prompts, plus a stronger trace/generalization check:
session trace reports pending retrieval; materialized trace shows residual/FFN
evidence; nearby unstored prompts behave through the materialized edit rather
than through a lookup sidecar.

Until C1-C3 ship, video language should distinguish three mechanisms:
KNN journal/retrieval overlay, compose FFN overlay, and compiled/baked weights.

---

## P1 — Model architecture independence hardening

Driver: keep LARQL from becoming "Gemma-shaped with exceptions." The core
`ModelArchitecture` trait is the right boundary, but several production paths
still infer family from strings, pass scalar attention geometry through
per-layer pipelines, or advertise architectures whose extraction/inference
contracts are incomplete.

| # | Item | Crate | Status |
|---|------|-------|--------|
| AI1 | Gate supported architecture families by executable contracts: extraction, vindex weight writing, forward/decode, trace, and prompt rendering | larql-models + larql-vindex + larql-inference | planned |
| AI2 | Implement or explicitly reject MLA architectures in vindex writers and inference; DeepSeek is detected today but `mla_*` tensors are not consumed outside `larql-models` | larql-models + larql-vindex + larql-inference | planned |
| AI3 | Remove scalar attention-geometry fallbacks from backend decode APIs; allocate KV/cache/scratch from `FullPipelineLayer` per-layer shapes everywhere | larql-compute + larql-inference | planned |
| AI4 | Replace vector-only extraction's model-name family guesses with explicit metadata or validated architecture input | larql-vindex | planned |
| AI5 | Roll validated loading/detection through inference, extraction, CLI, and server entry points where missing config should fail fast | larql-models consumers | planned |
| AI6 | Harden vindex extraction/write paths with explicit capability gates, named manifest/tensor tags, and tests proving unsupported attention layouts fail before writing partial indexes | larql-vindex + larql-models | next |

Acceptance target: adding a new transformer architecture should require changes
inside `larql-models::architectures/*` and explicit capability decisions at
storage/forward boundaries, not incidental string matches or hidden Gemma/Llama
defaults in extraction and decode.

---

## Critical path (P0 — what blocks the demo)

Items in order. Each depends on the one above it. Truly P0 only — items
that were #5–#10 in the previous version (multi-machine grid) demoted
to P2 per ADR-019 (2026-05-09); see new section "P2 — Multi-machine MoE
grid" below for the demoted items.

| # | Item | Crate | Status |
|---|------|-------|--------|
| 1 | Chat template + EOS stop | larql-inference + larql-cli | not started |
| 2 | Token streaming | larql-inference + larql-cli | not started |
| 3 | **Per-layer FFN format** (`layers/`, GPU dispatch) Phase 2: pre-alloc buffers | larql-vindex + larql-compute | shipped — `MoeScratch` pre-allocates once per decode call; combined with the 2026-05-02 dispatch-geometry fix, 26B A4B Metal now runs at **19.4 tok/s** (was bug-locked at 5.1) |
| 4 | MoE-aware CPU forward pass (non-Metal fallback) | larql-inference | not started — **promoted to P0 of the CPU track as C1; see "P0 — CPU path to blazing"** |

Items 1–2 are needed for Act 1. Item 3's MoE performance gate landed
2026-05-02. Item 4 = C1 (CPU MoE forward pass) in the CPU-track section.

---

## P2 — Multi-machine MoE grid (deferred per ADR-019)

The items below were critical-path #5–#10 before ADR-019 (resolved
2026-05-09). They build the multi-machine MoE grid for "model spans
multiple consumer machines." Demoted because they are
production-engineering work with no current experiment requiring
multi-machine expert dispatch — single-machine sharding (already
shipped) covers all current substrate needs.

**Re-promotion conditions** (any one triggers re-promotion to P0):
1. A specific experiment requires multi-machine expert dispatch.
2. A frontier model release (671B-class or larger) becomes substrate-relevant.
3. The Ultimate acceptance tier in "P0 — CPU path to blazing" becomes a near-term goal rather than a stretch.

| # | Item | Crate | Status |
|---|------|-------|--------|
| MMG1 | Wire `RouterIndex` client-side (was critical-path #5) | larql-inference | not started |
| MMG2 | `POST /v1/expert/{layer}/{expert_id}` (was critical-path #6) | larql-server | not started |
| MMG3 | `POST /v1/expert/batch` (was critical-path #7) | larql-server | not started |
| MMG4 | `--experts 0-31` flag on `larql serve` (was critical-path #8) | larql-server | not started |
| MMG5 | `RemoteExpertBackend` client (was critical-path #9) | larql-inference | not started |
| MMG6 | Reliability pass — timeouts, retries (was critical-path #10) | larql-server | not started |
| MMG7 | C9 (multi-machine grid productionisation) (was P0 in CPU track) | larql-router + larql-server | shipped (grid + rebalancer); needs production polish |

Detail on the original framing in `larql-server/ROADMAP.md` (F-COLLECT,
F-LOCAL-MOE, G-SCALE) and `larql-vindex/ROADMAP.md` P0.

---

## P0 — Aim-validation tests (V1–V4)

Driver: the achievability analysis (see "Engine purpose" above) rests on
**four** load-bearing assumptions (three isolated + one compound). ADR-015
says isolated wins don't always compose — and D-RMS-FUSE Phase 1 (2026-05-09)
gave us a concrete falsification: predicted ~0.2 ms/tok savings collapsed to
zero. So we already have one data point that compounds *don't* always
materialise. The framing itself needs falsification tests before committing
years of engineering. Until V1–V4 land, the medium/long/ultimate acceptance
tiers in "P0 — CPU path to blazing" are aspirational, not engineering-targets.
**These are the highest-leverage items on the entire roadmap right now**:
each is relatively cheap (days to ~2 weeks) and each can collapse a large
downstream investment.

**Important framing**: V1, V2, and V4 are **extensions of work that's
already 60–80% done**, not open research. Read the prior-evidence column
before committing engineering time — these are not months-of-risk items.
V3 is the genuinely-new-territory item.

| # | Test | Prior evidence | What it falsifies | What it produces | Effort |
|---|------|----------------|-------------------|------------------|--------|
| **V1 ✅ DONE 2026-05-31 — FALSIFIED (dense)** | Hash routing across all layers (extend exp 27) | **Exp 27 Gemma 3 4B L0 at top-2048/d_ffn (20% mask) → KL=0.030.** Walk boundary sweep (April 2026) progressively pushed the walk down through layers on Gemma 3 4B. **One-layer one-model evidence in hand.** | "5× FFN bandwidth reduction holds at end-to-end output, not just one layer" → **FALSIFIED.** Per-layer KL ≤ 0.05 thresholds DON'T compound: applied together they give +5.4 to +7.7 bits/token NLL and 78–95% drift on all 3 dense archs. The per-layer screen is anti-correlated with the truth. Deployable bandwidth ~2.4–2.9× (gate projection still paid), not 5×, and catastrophic anyway. | **DELIVERED:** per-layer threshold tables + compounding NLL/drift + cheap-route realizability + honest bandwidth, 3 dense archs (`bench/aim-validation/v1_*.json`), harness `chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_v1_hash_routing.rs`, writeup [`docs/diagnoses/v1-hash-routing.md`](docs/diagnoses/v1-hash-routing.md). **MoE-within-expert version OPEN** (dense harness measures the wrong object on the 26B → needs expert-aware tooling). | ~1 week (done) |
| **V2 ✅ DONE 2026-05-31 — CONFIRMED** | FP4 generality (extend exp 26 across archs) | **Exp 26: gemma3-4b-f16.vindex is 99.83% FP4-friendly per-feature without QAT (down is the tail at 99.65%).** Single-arch evidence in hand. | "FP4-friendliness is universal, not Gemma-3-4B specific" → **CONFIRMED.** ≥99.8% per-feature R<16 across Gemma 3 4B + Granite 3B/8B (reproduces exp 26's 99.83% exactly; down the tail). Predictive E2M1 +0.116 bits/tok vs f32, beats Q4-int. No QAT. | **DELIVERED:** static scan (`fp4_q1_scan`, generalized) + predictive NLL (`walk_ffn_v2_fp4_nll`, real E2M1 codec), artifacts `bench/aim-validation/v2_*_scan.json`, writeup [`docs/diagnoses/v2-fp4-generality.md`](docs/diagnoses/v2-fp4-generality.md). Llama/Mistral/MoE-expert weights not covered (need f16 exports). | ~1 week (done) |
| **V3 ~ PARTIAL 2026-05-31** | mmap'd vindex with sparse access on disk-resident frontier MoE | **None.** This is the genuinely-new-territory item. Risk dominates the long-term tier confidence (~52%, revised 2026-05-31). | "Disk locality + page-fault behaviour is acceptable when only top-k experts fire" → **partial:** cold scattered read ~100µs p50/140µs p99, warm ~0.04µs (~2380× gap). Steady-state hinges on cache hit rate. | **DELIVERED (feasibility):** cold-read probe (`mmap_cold_read_probe`, F_NOCACHE + verified-cold mmap faults), artifact `bench/aim-validation/v3_granite-30b.json`, writeup [`docs/diagnoses/v3-disk-resident-mmap.md`](docs/diagnoses/v3-disk-resident-mmap.md). **DEFERRED:** steady-state fault-rate + end-to-end tok/s on a >RAM model — needs >128 GB-class vindex or Linux/cgroup box (128 GB machine can't force RAM-pressure paging). | ~2 weeks |
| **V4** | **Compound test** (V1+V2+V3 stacked end-to-end on a real MoE model) | **D-RMS-FUSE Phase 1 (2026-05-09)**: predicted ~0.2 ms/tok savings collapsed to zero. ADR-015 has a concrete instance. | "Independent wins compound multiplicatively, not destructively" — per ADR-015. The framing's central claim. | End-to-end tok/s on Gemma 4 26B-A4B (or larger if available) with hash routing + FP4 + mmap'd disk-resident vindex active simultaneously. Measure perplexity degradation, tok/s, and compare to product-of-individual-speedups prediction. | ~1 week (after V1–V3) |

**Interpretation rule**: V1, V2, V3 each collapse a tier of the
acceptance ladder if they fail.

- V1 fails (hash routing doesn't compound across layers, or output diverges
  too much) → medium-term and below acceptance shrinks; ultimate aim
  needs different sparsity mechanism.
- V2 fails (FP4 needs QAT for non-Gemma archs) → still workable but FP4
  becomes a per-model retraining concern, not a free 2×; multiplies
  long-term build cost.
- V3 fails (mmap'd vindex thrashes) → ultimate aim shrinks to
  "models that fit in RAM"; rules out 671B on 64 GB consumer.
- V4 fails (techniques don't compound) → re-derive the achievable
  envelope from measurement, not from the multiplicative product;
  re-tier confidence accordingly. Note D-RMS-FUSE has already given us
  one such data point at the small-magnitude end; V4 measures the
  large-magnitude case.

**Sequencing**: V1 and V2 are independent and cheap — run in parallel.
V3 takes longer and depends on V1 (hash routing creates the sparse
access pattern V3 measures). V4 runs once V1–V3 are done.

**Output artifact**: `experiments/V1-V4_aim_validation/` directory with
results, plus an updated "Achievability" subsection in this roadmap
with measured numbers replacing predicted ones, plus a memory entry per
test (per the user's falsification-log convention).

**This is the work to do next.** Everything else in the long-term roadmap
either gates on these tests or is engineering on assumptions these tests
verify.

---

## P0 — Engine ↔ Backend unification (specs landed 2026-05-16)

Driver: today's `KvEngine` (in `larql-kv`) and `ComputeBackend` (in
`larql-compute`) are unaware of each other. The four research KV engines
(MarkovRS, WindowedCheckpoint, TurboQuant, Apollo) live in research-only
bench paths; the production decode loop bypasses them. And every backend
(CPU, Metal, future Vulkan/CUDA) hides under a single trait that doesn't
let engines express *intents* (windowed attention, K/V recompute,
boundary upload) — only flat compute primitives (matmul, softmax). The
net effect: engine-aware kernel fusion, compute-aware engine selection,
and per-engine prefill graphs are all foreclosed today.

Three landed specs in `crates/larql-inference/docs/specs/`:

- [`kv-engine-unification.md`](crates/larql-inference/docs/specs/kv-engine-unification.md)
  — KvEngine trait + dispatch in `larql-inference`; `larql-kv` ships
  six engines (`Standard`, `NoCache`, `MarkovResidual`,
  `WindowedCheckpoint`, `TurboQuant`, `Apollo`).
- [`compute-backend-redesign.md`](crates/larql-inference/docs/specs/compute-backend-redesign.md)
  — `KvDispatch` sibling trait in `larql-inference` (intent-based
  per-layer surface); `EngineBackend: ComputeBackend + KvDispatch`
  umbrella; engines hold `Box<dyn EngineBackend>`.
- [`async-compute-backend.md`](crates/larql-inference/docs/specs/async-compute-backend.md)
  — `AsyncComputeBackend: ComputeBackend + KvDispatch` sibling trait
  (deferred dispatch / intent-collector / handle-based). Required for
  any GPU performance at per-layer intent granularity. Trait surface
  locked; implementation pending.

Honest scope: the unification PR is shippable today. The tok/s wins
require the multi-month AsyncComputeBackend implementation (Steps A1–A8
in the spec). Expect 6–12 months end-to-end before per-layer Metal
beats today's fused `decode_token` path.

| ID | Item | Crate(s) | Status | Notes |
|----|------|----------|--------|-------|
| U1 | KV engine unification — Steps 1–7 | larql-inference, larql-kv, larql-cli | **shipped 2026-05-16** | `KvEngine` trait + EngineInfo + DecodeStageSummary in `larql-inference::kv_engine`; `larql-kv` re-exports. `Standard` + `NoCache` engines added. `larql run` / `larql walk` route through engine dispatch (default `--kv-cache standard` = `Standard { window_size: None }`, bit-parity gated). `--engine SPEC` + `LARQL_KV_ENGINE` env var on run/walk. Server wiring deferred to U7 (server uses fused `decode_token` and would silently downgrade to CPU under sync dispatch). |
| U2 | ComputeBackend redesign — Steps 1–4 | larql-inference, larql-compute | **shipped 2026-05-16** | `KvDispatch` trait in `larql-inference` (per-layer intents: cache, attention, engine-specific). `EngineBackend: ComputeBackend + KvDispatch` umbrella with blanket impl. `CpuBackend::KvDispatch` real implementation; `MetalBackend::KvDispatch` CPU-fallback scaffolding. `cpu_engine_backend()` / `default_engine_backend()` factories. 6 new `Capability` flags (`FusedAttentionStep`, `WindowedAttentionStep`, `NativeKvCodec`, `PipelinedBoundaryUpload`, `FusedResidualNorm`, `KvHandleNative`). |
| U3 | ComputeBackend redesign — Step 3c (engine migration) | larql-kv, larql-inference | **shipped 2026-05-16** (partial); follow-up in U8 | All six engines accept `Box<dyn EngineBackend>` in constructors. `KvDispatch` widened with `Option<&VectorIndex>` on attention intents + new `coarse_prefill` / `coarse_decode_step` (quantization-agnostic, backends inspect index format internally). `StandardEngine` fully migrated: routes Q4K through `coarse_prefill` on `CpuBackend` (which calls production `predict_q4k_prefill` / `predict_q4k_decode_step_direct`). **27.6 tok/s on Gemma 3 4B Q4K, M3 Max, 8 threads — slightly faster than the legacy `larql-cpu` path (24.0 tok/s).** `NoCache` migrated (slow on purpose: O(N²) debug fallback). Others (`MarkovResidual`, `WindowedCheckpoint`, `TurboQuant`, `Apollo`) still carry their bespoke `prefill_q4k` overrides — they work correctly but run at ~0.4 tok/s through f32-dequant fallback. Migration to fast Q4K kernels via the dispatch trait is **U8** below. Spec: [`kv-dispatch-quantization.md`](crates/larql-inference/docs/specs/kv-dispatch-quantization.md). |
| U4 | AsyncComputeBackend impl — Steps A1–A5 (the trait + foundation) | larql-inference, larql-compute, larql-compute-metal, larql-kv | **A1–A3 + A5 (StandardEngine) shipped 2026-05-16; A4 next** | A1 ✅ trait + handle types in `larql-inference/src/async_compute_backend.rs` (per-handle inner traits, `read(self: Box<Self>)` — stable-Rust translation of spec's `Arc<dyn AsyncHandleInner>` pattern). A2 ✅ `CpuBackend` async impl as degenerate `Ready*` wrapper, 6 bit-parity tests vs sync. A3 ✅ `MetalBackend` scaffold via CPU-delegation, feature-gated; 4 Metal-aware bit-parity tests pass under `--features metal`. A5 ✅ for `StandardEngine`: `with_async_backend` constructor + internal `BackendSlot` enum + async dispatch helpers + 8 new parity tests (`larql-inference`: 1002 lib tests; `larql-kv`: 221 lib tests). A4 next: real `MTLCommandBuffer` deferred dispatch (4–8 weeks). Remaining engines' A5 slices (`MarkovResidual`, `WindowedCheckpoint`, `TurboQuant`, `NoCache`, `Apollo`) compose on the same pattern (~1–2 weeks each). |
| U5 | AsyncComputeBackend impl — Step A6 (per-engine specialised shaders) | larql-compute, larql-kv | **spec'd, not started** | This is the tok/s payoff. Priority order: `attention_step_windowed` (the `standard:window=N` win), then engine-specific intents in order of impact — `markov-rs` Metal K/V recompute, `apollo` pipelined boundary upload, `turbo-quant` codec kernel. Each shader paired with a real-model bench. Ongoing — months of iterative work. |
| U6 | AsyncComputeBackend impl — Step A7 (VulkanBackend) | larql-compute | **spec'd, not started — blocked on U9-U12** | Same trait shape as Metal, different primitives (`VkCommandPool`, semaphores, SPIR-V). Validates the multi-backend story is real, not Metal-shaped. 6–10 weeks **once U9-U12 unblock the engine layer**. Today the substrate trait is drop-in but `larql-inference` still has 30+ `cfg(feature = "metal")` gates and 2 `downcast_ref::<MetalBackend>()` sites that conflate "Metal" with "GPU pipeline" — landing Vulkan against today's tree would force per-backend cfg explosion across the inference crate. |
| U7 | AsyncComputeBackend impl — Step A8 (CudaBackend) + server wiring | larql-compute, larql-server | **spec'd, not started — blocked on U9-U12** | CUDA streams map naturally to the deferred-dispatch shape — designed against it. Server wiring (deferred from `kv-engine-unification.md` §10.6) lands here: `larql-server`'s `handle_stream_generate` switches from direct `generate_streaming` to `generate_with_engine` against an `AsyncComputeBackend`, finally honouring `LARQL_KV_ENGINE` server-side. 6–10 weeks Cuda + 1–2 weeks server. Same engine-layer blockers as U6. |
| U8 | Engine migration — bespoke `prefill_q4k` paths onto dispatch trait | larql-kv, larql-inference | **specced, not started** | `MarkovResidual`, `WindowedCheckpoint`, `TurboQuant`, `Apollo` each carry an engine-side `prefill_q4k` override that bypasses the dispatch trait's `coarse_prefill` / `coarse_decode_step` intents and uses slower CPU code paths (dequant-to-f32 + f32 sgemv) instead of the production `predict_q4k_*` kernels. Result: ~0.4 tok/s vs `StandardEngine`'s 27.6 tok/s on the same hardware. Each engine has legitimate specialisation (RsStore residuals, per-window K/V checkpoints, WHT+Lloyd-Max codec, boundary residual injection) — the migration keeps that engine-side logic but routes the per-layer matvec through `larql_compute::QuantMatVec::q4k_matvec` instead of dequant-then-f32. Per-engine: ~2-5 days. See [`kv-dispatch-quantization.md`](crates/larql-inference/docs/specs/kv-dispatch-quantization.md) Phase 2. |
| U9 | De-Metal the inference-side GPU cfg gates | larql-inference, larql-cli | **not started — compute-refactor branch** | 23 `cfg(all(feature = "metal", target_os = "macos"))` sites in `larql-inference/src` + 8 in `larql-cli/src` use "metal" as a synonym for "GPU pipeline available." Two options: (a) rename `feature = "metal"` → `feature = "gpu"` on `larql-inference` with `larql-compute-metal` as one optional backend inside it, so the same flag turns on Metal today and Vulkan/CUDA tomorrow without per-call-site flag matrix; (b) replace cfg gates with `Capability::FullPipelineQ4` / `Capability::DecodeToken` probes on `&dyn ComputeBackend`. Mechanical search/replace + targeted refactor; ~1-2 days. **Prerequisite for U6/U7.** |
| U10 | Move `prepare_ple_inputs` (Per-Layer Embeddings upload) onto a trait method | larql-compute, larql-compute-metal, larql-inference | **not started — compute-refactor branch** | Kills the 2 `downcast_ref::<larql_compute_metal::MetalBackend>()` sites (`layer_graph/hybrid.rs:78`, `layer_graph/generate/gpu/mod.rs:261`) and the `metal_ple: Option<&MetalBackend>` typed parameter that flows through `generate/gpu/decode_loop.rs:60-67`. Add `fn prepare_ple_inputs(&self, flat: &[f32], num_layers: usize, ple_dim: usize)` to `ComputeBackend` (default no-op) plus `Capability::PerLayerEmbeddings`. Spec at `compute-backend-redesign.md` §6.3 explicitly says "Engines do **not** check `backend.name()` to decide behaviour" — this is the residual gap. ~1 day. **Prerequisite for U6/U7.** |
| U11 | Move `take_last_split_timings()` onto a trait method | larql-compute, larql-compute-metal, larql-inference | **not started — compute-refactor branch** | `larql_compute_metal::take_last_split_timings()` is reached directly as a free function from `decode_loop.rs:194-200`. Replace with `fn take_split_timings(&self) -> Option<ProfileTimings>` on a sub-trait (or `ComputeBackend` with a default `None`) so Vulkan/CUDA can expose the same instrumentation hook. Also folds the `ProfileTimings` type down into `larql-compute`. ~0.5 day. **Prerequisite for U6/U7.** |
| U12 | Backend-agnostic `predict_hybrid_gpu` | larql-inference | **not started — compute-refactor branch** | `layer_graph/hybrid.rs:65-91` (`predict_hybrid_metal`) downcasts to `MetalBackend` then dispatches the hybrid attention-only-on-GPU + FFN-on-walk path. Rewrite as `predict_hybrid_gpu` that gates on `Capability::FullPipelineQ4` (or a new `Capability::AttentionOnly` if the attention-only entry point needs its own probe) and dispatches through the trait. Co-lands with U10 (the PLE method is one of the inputs hybrid needs). ~1-2 days. **Prerequisite for U6/U7.**

**Implementation order**: U1 ✅ → U2 ✅ → U3 ✅ → U4 (A1–A3 + A5
StandardEngine slice ✅; A4 real Metal deferred dispatch next; A5
remaining engines compose on the same pattern) → U5 (highest tok/s
leverage, run continuously alongside U6/U7) → **U9 → U10 → U11 → U12
(engine-layer de-Metal-ing — compute-refactor branch)** → U6 → U7. U4's
A4 is the next critical-path commitment; until it lands, U5/U6/U7 are
blocked. U9-U12 close the residual "Metal-as-GPU" coupling in
`larql-inference` so U6 (Vulkan) and U7 (CUDA) land as pure sibling
crates without inference-side cfg explosion.

**Acceptance**:
1. **Short-term** (U4 lands): engines that opt into async on Metal see
   decode at ≥ today's fused-path tok/s (1 GPU sync per token, matched
   cadence). No regression on default `StandardEngine` user-visible
   behaviour.
2. **Medium-term** (U5 lands `attention_step_windowed`): `standard:window=N`
   decode at ≥ 1.5× today's `standard` Metal decode on Gemma 3 4B at
   window=512. Per-shader bench artifact in `bench/baselines/cpu/` (or
   `metal/` once we add it).
3. **Long-term** (U5 covers `apollo` + `markov-rs`): long-context
   workloads where Apollo's compressed path applies decode at ≥ 8×
   today's Metal `standard` on Gemma 3 4B at 32k context. Requires
   offline boundary-store preprocessing — separate work item.
4. **Ultimate** (U6 + U7): same engine catalog runs on Vulkan
   (consumer NVIDIA/AMD/Intel GPUs without Apple Silicon) and CUDA
   (datacenter NVIDIA) with the same per-engine perf cliffs.

---

## P0 — CPU path to blazing (the ultimate-aim track)

Driver: the ultimate aim ("largest models at blazing speed on consumer
hardware, ideally without GPU") demands a permanent CPU track in
parallel with the GPU competitive-baseline track. CPU work is built
**in addition to** Metal work, not instead of it. Every item here is
either device-agnostic by construction (sparse retrieval) or has a
matched GPU twin (so the technique stack stays portable).

The bandwidth math is the gating constraint: 50 GB/s consumer DDR5
means a 671B Q4 model is 6.7 sec/token under naïve dense matmul.
Combined sparse-retrieval techniques (hash routing 5× × FP4 2× × KV
compression 10× = ~100×) make this ~134 ms/token — the actual
"blazing on consumer hardware" target. **(Revised 2026-05-31: hash-routing 5×
FALSIFIED by V1 — doesn't compound; FP4 2× confirmed by V2. The realistic
compound is smaller and rests on MoE active-param sparsity + FP4. See
achievability table + `docs/diagnoses/`.)**

| # | Item | Crate | Status | Notes |
|---|------|-------|--------|-------|
| C1 | Critical-path #4 — MoE-aware CPU forward pass (non-Metal fallback) | larql-inference | not started | **Promoted from critical path #4 to P0 of this track**. Currently CPU MoE has no production path; everything routes through Metal or grid. Without C1, CPU track has no decode loop to measure. **Stays P0 under ADR-019** because V1/V2 cross-arch sweep on 26B-A4B requires CPU MoE. |
| C2 | WalkFfn as **primary** CPU decode path (not research-only mode) | larql-inference | partial — exists, not productionised | Currently `WeightFfn::forward` is the dense fallback; switch the default for vindex-loaded models to `WalkFfn`. Bench numbers required. Cross-references CPU MoE work in C1. |
| C3 | ~~Hash-routed FFN (exp 27 → product)~~ — top-k mask on gate scores | larql-inference + larql-vindex | **DO NOT BUILD — FALSIFIED (V1, 2026-05-31)** | Exp 27's L0 top-2048 → KL=0.030 is real *at one layer*, but V1 measured all layers on 3 dense archs: per-layer KL ≤ 0.05 thresholds **do not compound** (+5–8 bits/token NLL, 78–95% drift when stacked), and cheap routing can't even realise the oracle sparsity. Deployable bandwidth ~2.4–2.9× (gate projection paid), not 5×, and catastrophic anyway. See `docs/diagnoses/v1-hash-routing.md`. Drop this item. |
| C4 | FP4 productisation (exp 26 → product) — native FP4 quantisation tier (`Q4_K → FP4`) | larql-vindex + larql-compute | research only → **V2-validated, greenlit** | Exp 26 + **V2 (2026-05-31, confirmed)**: ≥99.8% FP4-friendly per-feature across Gemma 3 / Granite (no QAT, `down` the tail); predictive E2M1 +0.116 bits/tok vs f32, beating Q4-int. The FP4 codec already exists (`larql-models/src/quant/fp4*.rs`). Add `Quantisation::FP4` variant; CPU-first kernel; Metal twin. ~2× shrink vs Q4_K. See `docs/diagnoses/v2-fp4-generality.md`. |
| C5 | mmap'd vindex with lazy disk-resident edges — only resident pages for active edges per token | larql-vindex + larql-inference | not started | Today vindex loads whole layer tensors into RAM. For models bigger than RAM, mmap the vindex file and let the OS page in only the gate-KNN-resolved edges. Pairs with C2 and C3: when only 20% of edges fire, only those pages are read. |
| C6 | AMX / AVX-512 / Apple AMX kernels for residual compute | larql-compute (CPU side) | partial — Accelerate BLAS, AMX through it | Current CPU path uses ndarray + Accelerate; promote to direct AMX intrinsics on Apple Silicon, AVX-512 on x86. Compute that *does* happen needs to be as good as it gets, since bandwidth is what's left over. |
| C7 | KV compression as **default** for long context (Apollo / MarkovRS / WindowedCheckpoint / TurboQuant) | larql-inference | engines reachable on `run`/`walk` (CPU) via `--engine` / `LARQL_KV_ENGINE`; default still `standard` (production K/V cache); GPU performance on opt-in engines requires AsyncComputeBackend (see U-series below) | Unification spec at [`kv-engine-unification.md`](crates/larql-inference/docs/specs/kv-engine-unification.md) — all 7 steps landed. MarkovRS / WindowedCheckpoint / TurboQuant opt-in via `--engine` (CPU-correct, Metal works via CPU-fallback delegation). Apollo bench-only. Promoting any of these as default for long context requires `AsyncComputeBackend` Step A6 (engine-specific Metal shaders) to land — see U5 below. Server engine wiring also blocked on AsyncComputeBackend (U7); without it the server would silently downgrade Metal decode to CPU. |
| C8 | BR4 (Boundary refs Phase 4 — bounded KV eviction + durability-first capture) | larql-server + larql-inference | not started | See § "P1 — Boundary refs and cold-context storage" below. The CPU track makes BR4 load-bearing because long-context CPU inference can't keep raw KV in RAM. |
| C9 | Distributed-load-balancing for "model spans 4 consumer machines" | larql-router + larql-server | shipped (grid + rebalancer) | **DEMOTED to P2 per ADR-019 (2026-05-09)** — substantial production-engineering with no current experiment requiring multi-machine. Single-shard grid (already shipped) sufficient for substrate. Re-promote if a specific experiment needs multi-machine. |
| C10 | CPU bench harness — `larql bench --cpu` with per-stage breakdown matched against `llama.cpp -ngl 0` | larql-cli + bench/ | **DISCREPANCY RESOLVED 2026-06-02 — no regression; true gap ~1.6–1.8×.** The 1.50× (05-16) vs 1.93× (05-31) split was **two stacked measurement confounds**, not a real change: (1) **larql path mismatch** — 27.6 was the `StandardEngine` path, 23.6 the legacy `larql bench --cpu` (`predict_kquant_decode_step`) path; a stable ~12% delta (26.4 vs 23.5 today), so comparing one date's StandardEngine against the other's legacy path manufactured a phantom "regression"; (2) **llama.cpp harness artifact** — the 45.5 was an unwarmed/short-n ollama `num_gpu=0` fluke; warmed + n=128 it converges to **42.8–43.0 = llama-bench's 42.99** (both harnesses, both dates agree at ~43). Reconciled like-for-like (M3 Max, t=8, warm): **larql 23.5 legacy / 26.4 StandardEngine vs llama.cpp 43.0 → 1.6–1.8×.** Gap is C12 (both attn AND FFN already use the int8 Q8_K SDOT kernel via `attention_decode_step_native`). **Free wins landed (2026-06-02):** `larql bench --cpu` now also reports the production StandardEngine row; new `--ollama-cpu` forces `num_gpu=0`+`num_thread` so `--ollama` is a true CPU baseline (was silently Metal-GPU). Reconciled artifact `bench/baselines/c10_gemma3-4b_cpu_reconciled.json`. **26B-A4B baseline LANDED 2026-06-10** (`c10_gemma4-26b-a4b_cpu_reconciled.json`): llama.cpp **32.1** vs larql in-process **7.1** default / **9.7** with `LARQL_Q4K_DIRECT_ATTN=1` / loopback 7.3 (t=8, warm, n=128, drift-checked). The 26B gap (4.5×) is **f32-residency byte traffic** (attn 4.15 GB + dense slab 2.14 GB + lm_head 2.95 GB per token vs llama.cpp ~2.1 GB all-quantized; every leg bandwidth-saturated ~62–71 GB/s), NOT the C12 kernel (experts already int8 SDOT, ~8% of bytes). Medium-term tier 62%→70% per the gate rule. Method addition: **pmset AC check + cross-engine drift bracket are now mandatory** — the first session was invalidated by a silent battery drain (llama.cpp itself collapsed 34→1 tok/s at 31% battery; far beyond the 1.5–3× thermal class). | CPU-track baseline-credibility threshold can't be enforced without this. First acceptance test: Gemma 3 4B Q4_K on M3 Max CPU vs quant-matched `llama.cpp -ngl 0`. Then Llama 2 7B + Mistral 7B for cross-arch CPU + the 26B-A4B MoE baseline. Major improvement 2026-05-15→05-16 (2.78× → 1.50×) — see `bench/baselines/cpu/COMPARISON.md` and `DIAGNOSIS-2026-05-16-thread-scaling.md`; reconciliation `bench/baselines/c10_gemma3-4b_cpu_reconciled.json`. |
| C11 | Architecture rule enforcement — CI check for "no GPU-only paths in core" | scripts/ + crate boundaries | not started | Static check: anything in `larql-inference` core (not `metal/`, not `cpu/`) must compile and pass tests with Metal feature off. Prevents the dual-track from drifting into Metal-locked code. |
| C12 | Q4K decode kernel — hand-asm aarch64 to close the 1.50× gap to llama.cpp | larql-compute | **v1 asm landed opt-in 2026-06-02 (`LARQL_Q4K_ASM=1`); roofline reframed the work.** Two 2026-06-02 results: (a) **Roofline microbench** (`benches/q4k_q8k_matvec.rs`) shows the kernel is **compute/issue-bound, NOT DRAM-bandwidth-bound** — scalar 9.3 vs NEON 17.7 GiB/s on identical data, size-invariant — which **overturns the `DIAGNOSIS-2026-05-16` "memory-system-level" conclusion** and confirms hand-asm scheduling is a real lever (17.7 GiB/s ↔ ~33 cyc/super-block, exactly as specced). (b) **`q4k_q8k_matvec_asm`** (whole super-block dot in one `asm!` block, 8 scales as vector lanes killing the 8 scalar `ldrb`) — **bit-exact** (`q8k_matvec_asm_matches_scalar_bit_exact`), **+3.7–4.9% isolated**, ~+1–2% e2e (diluted: opt-in covers `matvec_into` callers — attention Q/K/V/O + `down` — but NOT the fused `gate_up`). **Finding: latency-hiding has low headroom** — a 4-accumulator variant showed no reliable gain (the inlined row loop lets the OoO core already overlap super-blocks), so **the two-super-block interleave is deprioritized**; the real lever to reach ~28 GiB/s is **instruction-count reduction** (perf-counter-guided, llama.cpp-style vectorized scale path) + **asm-ifying `gate_up`** (lifts the e2e ceiling). See spec §"2026-06-02 roofline measurement". | Per-core gap is **1.73× constant across thread counts** (5.7 vs 9.88 tok/s single-threaded on M3 Max). Same algorithm (Q4K × Q8K with NEON SDOT), same `vdotq_s32` instructions — llama.cpp uses hand-written inline aarch64 asm with two-super-block interleaving + explicit prefetch hints, we use Rust intrinsics lowered by LLVM. Effective bandwidth: ~63 GB/s vs ~95 GB/s. **Per-stage profile (`LARQL_INSTRUMENT_UNLIMITED=1` on Gemma 3 4B 8-thread, 2026-05-16): FFN 26.0 ms (74%) + Attention 9.3-11.0 ms (26%, grows with ctx) + Embed ~0 ms = 35-37 ms/step.** FFN matvec on gate/up/down (4608 × 9216) is the dominant target; attention matvec is the same kernel on smaller matrices. The 38 tok/s asymptote (FFN-alone) sets the floor any engine can reach on the current kernel — Standard and WindowedCheckpoint both hit 26.6 tok/s on Gemma 3 4B Q4K CPU (8-thread, 40-token prompt, 64 decode tokens) because both route through the same `attention_decode_step_native` + `ffn_decode_step_native` hot paths. Phases: (1) hand-asm Q4K matvec on the FFN tile shapes (gate/up/down) — closes ~95% of the gap, 1-2 weeks; (2) pre-formatted block layout — 1.1-1.2× on top, 3-5 days; (3) Q6K kernel for `ffn_down` — 1.05×, 2-3 days; (4) reduce rayon launch overhead — 1.04×, 2-3 days. Acceptance: ≥9.5 tok/s single-core, ≥39 tok/s 8-thread on Gemma 3 4B Q4K. Spec: [`crates/larql-compute/docs/q4k-decode-kernel.md`](crates/larql-compute/docs/q4k-decode-kernel.md). Per-stage measurement protocol: see "C12 per-stage measurement" below. |

**Implementation order** (post ADR-019): C10 → C1 → C2 → C7 → C12 → C3 → C4 → C5 → C6 → C8 → C11.

(C12 — Q4K decode kernel — slots in mid-sequence: after the dispatch trait is stable and StandardEngine is matching the legacy `larql-cpu` path through it (both now true), the hand-asm kernel is the next high-leverage CPU performance win. Single-threaded gain ~1.73× from a focused 1-2 week effort, scaling cleanly to ~1.7× at 8 threads.)
(C9 dropped from P0 sequence per ADR-019; re-add only if re-promoted.)

C10 first because the threshold can't be enforced without measurement.
C1+C2+C7 give you a working CPU decode path with bearable long-context.
C3+C4+C5 are the bandwidth-shrinking techniques that make the ultimate
aim possible. C6 squeezes the compute that remains. C8 unblocks
long-context. C11 prevents architectural drift.

**Acceptance**:
1. **Short-term** (C10 + C1 + C2): CPU Gemma 3 4B Q4_K decode within 10% of `llama.cpp -ngl 0` on M3 Max CPU.
2. **Medium-term** (+C3 + C4 + C7): CPU Gemma 3 4B FP4 + hash-routed decode at ≥2× the dense Q4_K CPU baseline.
3. **Long-term** (+C5 + C8): Gemma 4 26B-A4B (or larger) decode on a single 64GB consumer machine at ≥10 tok/s, no GPU.
4. **Ultimate** (full stack + frontier model): 100B-class model on consumer hardware at ≥5 tok/s, no GPU. Stretch goal: 671B-class via multi-machine grid (gated on re-promoting C9 per ADR-019).

### C12 per-stage measurement

Two instruments measure the kernel-bound nature of CPU decode and let you isolate which sub-kernel the asm should target first:

- `LARQL_INSTRUMENT_UNLIMITED=1` — prints `embed / attention / ffn` per `extend_q4k` call from `larql_kv::engines::windowed_checkpoint::rs_extend_from_checkpoint_q4k`. Captures the per-token, per-layer-aggregated breakdown. Source: `crates/larql-kv/src/engines/windowed_checkpoint/extend.rs`.
- `LARQL_INSTRUMENT_MARKOV=1` — same shape for `markov-residual`, kept for cross-engine sanity that both substrate paths agree. Source: `crates/larql-kv/src/engines/markov_residual/q4k.rs`.

Reproducer (Gemma 3 4B Q4K, M3 Max, default 8 threads):

```
cargo build --release -p larql-cli
LARQL_INSTRUMENT_UNLIMITED=1 ./target/release/larql bench \
  ~/.cache/larql/local/gemma3-4b-q4k-v2.vindex \
  --backends cpu --engine windowed-checkpoint -n 32
```

Recorded baseline (2026-05-16, 8-thread, ~70-token ctx after warmup):

```
embed       ≈ 0.0 ms    ( 0%)
attention   ≈ 11.0 ms   (30%)  ← grows linearly with ctx
ffn         ≈ 26.1 ms   (70%)  ← flat regardless of ctx
total       ≈ 37.1 ms          ↔ 26.9 tok/s decode steady-state
```

**Acceptance for C12 Phase 1 (FFN hand-asm)**: at the same prompt/ctx, FFN drops from 26 → ≤15 ms (Phase 1 spec predicts ≥1.7× on gate/up/down, which would put FFN-alone at ~15 ms). Attention is the second-tier target after FFN is profile-clear; pre-Phase-1 it accounts for too little of the budget to bother with.

**Cool-machine protocol**: the M3 Max throttles on sustained Q4K matvec; a hot-bench reading can show 1.5-3× regressions that aren't real. Treat any kernel-A vs kernel-B comparison as inconclusive unless both runs start from a >5 min idle, and both `attention` and `ffn` rows move in the predicted direction (kernel work that improves only one should explain why).

---

## ADR-019 — MoE substrate decision (resolved 2026-05-09)

**Status**: **Resolved 2026-05-09** — Option A-modified. Substrate-primary
is dense (Gemma 4 31B); MoE coverage retained at single-machine scale;
multi-machine MoE grid demoted from P0 to P2.

### Resolution

**Decision**: Substrate-primary model is **Gemma 4 31B dense + vindex**.
MoE coverage is retained at single-machine scale (Gemma 4 26B-A4B for
cross-arch validation, virtual-expert work on existing MoE models, V1/V2
cross-arch sweeps). The multi-machine MoE grid (C9 productionisation,
critical-path items 5–10) drops to P2.

**Why not pure Option A** (drop MoE entirely): VID4 (virtual expert on
GPT-OSS) is already shipped publicly; the field is MoE (DeepSeek-V3,
Llama 4 Maverick, GPT-OSS family); V1/V2 must measure both dense and
MoE for honest cross-arch claims. Dropping MoE would forfeit
substrate-relevant ground.

**Why not pure Option B** (keep grid at P0): Multi-machine MoE grid is
substantial production-engineering work with no current experiment
requiring "model spans 4 consumer machines" beyond what single-machine
sharding already demonstrates. Critical-path items 5–10
(RemoteExpertBackend, `/v1/expert/*` endpoints, `--experts` flag,
reliability pass) are production-engine concerns the substrate framing
explicitly excludes.

### Forcing factors that drove the decision

- **Video pipeline MoE-specificity**: VID4 already shipped. VID7 ("I
  killed attention") needs static-attention measurement that works on
  any arch — not MoE-specific. No upcoming video requires multi-machine.
- **V1–V4 grid-dependency**: Single-machine is sufficient for V1, V2, V3
  on Gemma 4 26B-A4B (3.8B active params fits in 64 GB consumer RAM
  comfortably). V4 (compound test) does not need multi-machine for the
  acceptance bar. Multi-machine becomes relevant only at the *Ultimate*
  acceptance tier (671B-class), which is the ~30%-confidence stretch (revised 2026-05-31).
- **MCP / lazarus parity**: Arch-neutral. No MoE dependency.
- **Vindex framing**: "vindex is MoE taken to its logical extreme, every
  fact is its own expert" (April 2026 thread). Multi-machine MoE
  engineering doesn't accelerate the dense + vindex experimental program.

### Demotions effective immediately

| Item | Was | Now | Reason |
|------|-----|-----|--------|
| C9 (multi-machine grid productionisation) | P0 in CPU track | **P2** | Production engineering; no current experiment needs it |
| Critical-path #5 (Wire RouterIndex client-side) | P0 | **P2** | Multi-machine grid client; same reason as C9 |
| Critical-path #6 (`POST /v1/expert/{layer}/{expert_id}`) | P0 | **P2** | Remote expert endpoint; same reason |
| Critical-path #7 (`POST /v1/expert/batch`) | P0 | **P2** | Batched remote expert; same reason |
| Critical-path #8 (`--experts 0-31` flag on `larql serve`) | P0 | **P2** | Multi-machine deployment ergonomics |
| Critical-path #9 (`RemoteExpertBackend` client) | P0 | **P2** | Multi-machine client |
| Critical-path #10 (Reliability pass) | P0 | **P2** | Production reliability for multi-machine |
| Demo Act 2 ("experts live elsewhere") | P0 narrative | **Reframed** | "Elsewhere" was always a stretch for a substrate; reframe as single-machine expert dispatch (works on Gemma 4 26B-A4B locally with shipped grid) |

### Promotions effective immediately

| Item | Was | Now | Reason |
|------|-----|-----|--------|
| Gemma 4 31B dense as substrate-primary | implicit | **explicit** | Largest dense model in the supported set; vindex showcase target |
| Loose-end "Fix `dispatch_full_pipeline` layer_scalar (dense)" | "non-urgent: Gemma 3 4B has scalar=0" | **needs verification on Gemma 4 31B (substrate-primary per ADR-019)** | If Gemma 4 31B has scalar≠0, this loose end becomes urgent |

### Stays at original priority (not affected by ADR-019)

- **C1 (MoE-aware CPU forward pass)** — required by V1/V2 cross-arch sweep on Gemma 4 26B-A4B. Stays P0 in CPU track.
- **Critical-path #1, #2, #3, #4** — chat template/EOS, CLI streaming, per-layer FFN format, CPU MoE forward pass. Items 1–2 unblock Act 1; #3 shipped; #4 = C1.
- **VID4 (virtual expert)** — already shipped publicly; demonstrates single-machine expert dispatch.
- **Demo Act 3 ("replace an expert")** — works on single-machine via VID4-style approach.
- **MTP1–MTP6** — Gemma 4 MTP drafter work spans both dense (31B) and MoE (26B-A4B) targets.
- **All V1–V4 aim-validation tests** — unaffected; cross-arch coverage was always part of the design.

### Re-opening clause

C9 and critical-path #5–10 re-promote to P0 if any of:
1. A specific experiment requires multi-machine expert dispatch (none currently).
2. A frontier model release (671B-class or larger) becomes substrate-relevant.
3. The Ultimate acceptance tier in "P0 — CPU path to blazing" becomes a near-term goal rather than a stretch.

---

## P1 — Gemma 4 MTP drafter support (promoted from P2 2026-05-09)

Driver: Google released MTP drafters for every Gemma 4 variant on
**2026-05-05** (see Current state bullet above). Apple Silicon decode
speedup measured at **~2.2× at speculative batch 4–8**. Ollama already
supports MTP out-of-the-box; without this, the LARQL gap on Gemma 4
widens from 1.17× to ~2.6× as users adopt the drafters.

The drafters are the *exact* models LARQL is built around:
`google/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant`. Apache 2.0 (code) +
CC-BY-4.0 (weights). The 26B-A4B drafter is 0.4B BF16 (~4 layers).

Architecture (from Google blog + ai.google.dev/gemma/docs/mtp + the X
explainer thread):

1. Drafter shares the **input embedding table** with the target model.
2. Drafter consumes the target's **last-layer activations** at each
   accepted position, concatenates them with the next token embedding,
   and **down-projects to drafter dimension**.
3. Drafter cross-attends to the target's **global-layer KV cache** —
   specifically the *final* layer's KV, which is always global in
   Gemma 4 (the architecture interleaves local sliding-window attention
   with global attention, sliding window is 512 for E2B/E4B, 1024 for
   26B-A4B/31B). Local-sliding-window layer KVs are NOT shared.
4. E2B/E4B variants add an "Efficient Embedder" clustering layer that
   restricts drafter computation to selected token clusters.

**Substrate connection (added 2026-05-09)**: MTP exploits exactly the
attention-staticity that the "I Killed Attention" video (VID7) claims.
Per-token acceptance rate over a corpus is a direct measurement of the
static-attention fraction VID7 claims, *per architecture*. So MTP1–MTP6
produces both:
- A baseline-credibility number (Ollama parity on Gemma 4)
- Substrate evidence (VID7's central thesis at scale, per-arch)

Treat MTP6 as a substrate-and-baseline item, not just a competitive-parity
item. See Video pipeline section above.

| # | Item | Crate | Status | Notes |
|---|------|-------|--------|-------|
| MTP1 | `gemma-4-*-it-assistant` HF safetensors loader + `MtpDrafter` arch in larql-models | larql-models + larql-vindex | not started | New arch trait variant `MtpDrafter`; vindex extraction must handle the embedding-sharing reference (drafter doesn't carry its own embed table). Decide vindex layout: separate `*.assistant.vindex` sidecar vs unified `*.with-mtp.vindex` |
| MTP2 | Verify-loop decode (`generate_speculative`) — draft k tokens with drafter, verify k+1 with one target forward, accept longest matching prefix, rollback rejected positions | larql-inference | not started | Needs k as runtime param (default 4–8 per Google's batch-size sweet spot); reuse existing KV management; rollback logic touches `KvCache::clear_layer_position_range` (already shipped under M5) |
| MTP3 | Last-layer-activation feedback path — capture target's final residual at accepted positions, feed into drafter's input projection, down-project to drafter hidden | larql-inference + larql-compute | not started | **Sequencing: R6 must land before MTP3 begins** (MTP3's layer-choice validation depends on R6 depth-fraction probes; without R6 it ships with Google's default and validation has to be redone). CPU path reuses M1–M4 capture infrastructure. Metal path needs a dedicated lightweight last-residual tap during verify forward (M6 explicitly excludes Metal `generate` from hooks for performance reasons). The tap is one read from the residual buffer at the end of the last layer, before unembed — cheaper than full M1 plumbing. New Metal kernel: concatenate-and-project (or two separate dispatches if fusion regresses, ADR-015 lesson). **Activation extraction layer choice: validate against R6 depth-fraction probes** — Google reads from layer N by architectural choice; if R6 says discriminative information matures at 0.85·N, there is potentially free quality improvement available. |
| MTP4 | Shared KV cache between target and drafter — single cache, separate write heads | larql-inference | not started | **Drafter cross-attends to target's *global*-layer KV (Gemma 4 final layer is always global per the architecture). Local-sliding-window layer KVs are not shared.** May need `KvCache::view_global_only_for_drafter` or similar. Verify against Gemma 4 hybrid attention: 512-token sliding window for E2B/E4B, 1024-token for 26B-A4B/31B. Implementing as "single cache, drafter writes its own K/V into all slots" will silently corrupt local-window layer KV; do not. |
| MTP5 | Efficient Embedder clustering layer (E2B/E4B only) | larql-models + larql-compute | not started | Restrict drafter computation to top-N token clusters; smaller-model-only optimisation; defer until MTP1–MTP4 prove out on 26B-A4B |
| MTP6 | `larql bench --mtp` — measure speculative-batch sweep (k=1..16), token-acceptance rate, end-to-end tok/s vs no-MTP baseline | larql-cli + bench/ | not started | Confirms the 2.2× number on M3 Max before promoting to default. **Per-token acceptance rate is also the VID7 substrate measurement** — treat the bench output as evidence for the "I Killed Attention" video, not just a tok/s number. |
| SD1 | Generic speculative-decoding framework (n-gram draft / EAGLE / external draft model) — share MTP2's verify loop | larql-inference | not started | Broader machinery; promoted from P2 alongside MTP1. Build MTP2 first (concrete spec, immediate users); generalise to SD1 once the verify loop pattern is stable |
| SD2 | EAGLE-3 speculator support — Red Hat AI released `gemma-4-26B-A4B-it` EAGLE-3 (0.9B drafter); same machinery as MTP, different drafter loading | larql-models + larql-inference | not started | Validates SD1 generality on a non-Google drafter for a model we already support |

**Implementation order**: MTP1 → MTP2 → **R6** → MTP3 → MTP4 → MTP6 (validate
2.2× number AND collect VID7 evidence) → MTP5 (E2B/E4B optimisation)
→ SD1 (generalise) → SD2 (EAGLE-3 drop-in).

**Acceptance**: Gemma 4 26B-A4B Metal decode goes from 19.4 tok/s to
≥35 tok/s at speculative batch 4–8 with bit-identical token output vs
no-MTP baseline (Google guarantees identical-quality output; verify with
parity test across the existing cross-arch corpus).

**Why P1 not critical-path**: doesn't block the demo (Acts 1–3) — but
it *does* block any future tok/s comparison with Ollama on Gemma 4. If
the comparison story matters, MTP1–MTP4 should land before any public
benchmark refresh.

---

## P1 — Boundary refs and cold-context storage

Driver: replace unbounded KV retention in long-context and multi-host scenarios
with compact, contract-bearing residual checkpoints. Hot KV window stays bounded;
older context is represented as 2564-byte compressed residual frames.

```
KV for the present. Residual boundaries for memory.
```

Foundation: `crates/larql-boundary/` (Phases 1–3 shipped).
Protocol spec: `~/chris-source/chris-experiments/shannon/43_residual_stream_codec/BOUNDARY_REF_PROTOCOL.md`.
Calibration data: `~/chris-source/chris-experiments/shannon/44_boundary_gate_calibration/`.

The existing `BoundaryStore` in `larql-inference/src/trace/boundary.rs` stores raw
bf16 residuals. `larql-boundary` adds the 2× compressed path on top of it. Phase 4
connects them to the running server.

| # | Item | Crate | Status |
|---|------|-------|--------|
| BR1 | int8-clip3σ + bf16 codec (Phase 1) | larql-boundary | **shipped** |
| BR2 | Per-boundary metadata + calibrated gate at threshold=2.16 (Phase 2–3) | larql-boundary | **shipped** |
| BR3 | BoundaryFrame wire format + A/B/C/D/E contract taxonomy | larql-boundary | **shipped** |
| BR4 | Phase 4: bounded KV eviction + durability-first capture (Option A) | larql-server + larql-inference | not started — **see CR10** in the context-retirement track below: promoting a compact record into *already-built* state is untested, and it is the operation this eviction path assumes |
| BR5 | Phase 4: boundary archive (disk/remote) + restore path | larql-server + larql-inference | not started |
| BR6 | Phase 5: boundary frames over gRPC grid (protobuf schema defined) | larql-router + larql-server | not started |
| BR7 | Track B: per-channel codec (int4 + outlier side-channel, ≤1024 bytes) | larql-boundary | not started |
| BR8 | Gate calibration n≥300 to tighten 95% CI below 1.6%–10.7% | ~/chris-source/chris-experiments/shannon/44_boundary_gate_calibration | not started |

**What D-@high actually contracts:** first ~5 continuation tokens safe at 4.8%
early-div (95% CI 1.6%–10.7%, n=62). Total 20-token divergence is ~20% regardless
of threshold — cascade compounds past step 5. Use for boundary-to-fresh-decode; not
for long uninterrupted continuation. See BOUNDARY_REF_PROTOCOL §6.

**Connection to KU2 (softmax bottleneck)**: BR4 is the workaround for the
softmax bottleneck phase transition at ~1,142-token RoPE distance.
Q-side drift is fixable; KV-side drift at last position is not, with
current architecture. BR4 evicts hot KV before the bottleneck triggers
and falls back to compressed residual frames for older context.

**Immediate unblocking item:** BR4 (Phase 4 server integration). The eviction
ordering decision (durability-first Option A: capture → gate → fsync → evict KV)
is specified in the protocol; implementation in `larql-server` can start from it
directly.

---

## P1 — Context retirement & route-aware hot tier (research track, opened 2026-08-04)

Driver: BR4 assumes older context can be *represented* compactly. This track asks
the prior question — **when does historical state stop being needed at all, and
what cheaper form can it be promoted into?** Directly upstream of BR4/BR5 eviction
policy and of VINDEX3's route metadata.

Registry: `rsl-rp` (closed), `rsl-exp21` (open, write-ups v1–v12),
`rsl-exp24`/`rsl-exp25` (CR7/CR10/CR13..CR16), `rsl-exp26`..`rsl-exp38` — the
**authority control plane**, whose layer-mechanism branch closed 2026-08-05 and
whose bankable results are CR17–CR19. Full state, open questions and instrument
rules in `docs/authority-control-plane.md`; read EXP-36 and EXP-37 first.
Instruments: `~/chris-source/chris-experiments/rsl/exp14`–`exp38`.

```
Which old tokens should this query attend to?   <- sparse attention
Why do those tokens still exist as attention state at all?   <- this track
```

### What is measured and bankable

| # | Finding | Status |
|---|---------|--------|
| CR1 | **Route-aware hot tier.** At equal semantic cardinality, a route touching 1 vs 3 distinct FFN layers differs **2.11×** in replay latency (0.748 → 1.576 ms, 150 → 450 MiB). Fit R² 0.9966 at 385.6 GB/s ≈ 96% of M3 Max spec peak. Grouping by physical owner adds **1.80×** where reuse exists, ~1.0× where it does not. `argmin \|R\|` sees neither. | **measured** |
| CR2 | **Gemma-3 is 29 sliding + 5 global layers** (window 1024; MLX uses 29 `RotatingKVCache` + 5 `KVCache`). Long-context cost lives in 5 layers only; architecture-aware KV is **6.8× smaller** than a naive full-KV projection (20.1 GiB vs 136 GiB at 1M). | **measured** |
| CR3 | **Bounded-history decode**, 64K, frozen state, paired: **1.700× [1.670, 1.717]** best, ~1.6× at the representative planned budget point. Allocation shape, page topology and recent-window size are all **null** — active positions alone predict latency. | **measured** |
| CR4 | **Retirement refused for exact copy.** Post-question exclusion of the source span fails whole-answer qualification (maxKL 9.04 / 15.51, n=2). Causal origin **confirmed** — ABSENT fails and CORRUPT follows the replacement. | **measured** |
| CR5 | **Frontier**: k\*_value = 4, k\*_trajectory = 5. Source stays live through the value **and the first termination decision**. Mechanism is **progressive prefix handoff with a late direct-dependence tail**, not per-token reattendance (3 of 5 k values diverge *beyond* k). | **measured** |
| CR6 | **Compact record reproduces the payload at 1.14e-4 bits**; the entire 0.770-bit discrepancy sits on the single first-`<end_of_turn>` token. Semantic compressibility demonstrated. **Narrowed by EXP-24:** sufficiency is *prompt-mode and model-size dependent* — reproduced in raw mode on 4B (maxKL 0.7466 at 64K, same boundary structure), but in **chat** mode the same record makes 4B answer *"the text doesn't provide the vault access code"*, while on **12B/chat it qualifies outright at 0.0179**. | **measured; scope narrower than originally banked** |
| CR6b | **The compact record is operation-conditioned: it serves reads and fails computation.** 12B/chat/64K, one span, identical padding and token length: copy **0.0179 QUALIFIES**, `>7000?` 0.0655 (payload right), binding 10.06 (payload right) — but `+1` **12.98** and digit-reversal **15.23**, in both cases the model reverting to emitting the raw literal `7431` instead of the computed value. Copy is a **positive control** carrying the identical `Noted. Noted` padding, so padding cannot be the cause; this **resolves the CR9 confound for this comparison** (CR9 still needed for CR6's boundary term). | **measured** |

### The engineering diagnosis

> The verbose wording is **not inherently required** to represent the fact. It stays
> required in ordinary execution because the model has not promoted the fact into a
> sufficiently complete replacement state.

Ordinary computation *does* promote partially (first answer token survives full
exclusion; prefix carry extends beyond the granted interval) — it just does not
complete before transcription finishes. **The untested middle operation is injecting
or promoting a record into already-built state**; neither experiment touches it, and
it is what BR4-style eviction would actually need.

### Open items

| # | Item | Blocking |
|---|------|----------|
| CR7 | **Operation-conditioned frontiers** — **EXP-24 COMPLETE, primary contrast INCONCLUSIVE** (`rsl-exp24`, 4 runs at 64K: 4B raw + chat, 12B chat, + capability pre-screen). On 12B the two *monotone* length-matched arms are copy k\*=3 vs digit-reversal k\*=4 — spread 1, the instrument's quantisation floor. A spread-2 reading exists only via `+1`, whose frontier is **non-monotone** (passes k=3 at 0.0481, *fails* k=4 at 0.0816, passes k=5), so its k\* is decided by one point straddling KL_TOL=0.05 and was not banked. On 4B the contrast is unavailable outright — the only length-matched transformation it performs is `+1`, which shares `743`, so its identical frontiers are exactly what the confound predicts. **Not blocked on instrument design; blocked on finding a de-confounded transformation the model performs *at range*.** Signal that did survive: `derived` retires at **k\*=0** on both models (12B maxKL 0.00019), answering `Yes` — not the `No` that ABSENT gives — so the boolean is fully resolved into question state before the first output token. Length-confounded, so it cannot carry the claim alone. | complete, inconclusive |
| CR8 | **Step-local transfer matrix** `M[j,t]` — access enabled/disabled only at step *j*, scored at every later position. Diagonal ⇒ per-token; triangular ⇒ progressive. | nothing |
| CR9 | **Padding factorial** on the record arm (record × padding A/B/C, padding-only, corrupted record + same padding) to attribute the boundary-token KL. | nothing |
| CR10 | **Promotion into built state — DEMONSTRATED (EXP-25, `rsl-exp25`).** A 64K context built from the *verbose* source; a 13-token canonical record injected ~49K tokens downstream; the raw span excluded. Copy row, all three legs: pad-only → `7249` (**fails**, source genuinely necessary), canonical → `7431` at maxKL **0.0441**, payload **and** trajectory qualifying, corrupt → `5824` (**follows the corruption**). This is hot-state *replacement*, not the ingestion exp23/exp24 measured. Margin is narrow (0.0441 vs 0.05), so the claim rests on payload recovery + causal steering, with trajectory equivalence supporting. | **measured** |
| CR14 | **Read-only-ness was a property of DISTANCE, not of records.** Same record, model, framing, question and 64K context; only the record's position relative to the active boundary differs: `+1` under *ingestion* (49K back) → maxKL **12.98**, recites `7431`; under *late injection* (at the boundary) → **0.1472**, answers `7432`. Reversal likewise 15.23 → 0.0610. Causal control confirms real computation on the injected value (corrupt `5824` → **`5825`**, an answer appearing nowhere in the context). **Promotion is therefore a capability-restoring operation, not merely a memory operation** — refreshing a fact near the boundary buys back computation unavailable while it sat distant. Canonical passes reads 1/1 and compute 2/2 on payload. | **measured** |
| CR15 | **Prose cannot encode `discharged=true`; the record store needs a semantic type.** A discharged-result injection works for `+1` (*"The Meridian code plus one is 7432"* → `7432`, maxKL **0.0218**, the *only* compute arm passing trajectory) and fails for reversal: *"The Meridian code reversed is 1347"* → model answers **`7431`**, having reversed 1347 *again*. Systematic, not noise — the corrupt twin (`4285`) likewise returns `5824`. The earlier key-value form (`code_plus_one=5825` → `5826`) failed the same way, so **rewriting as prose did not fix it and whether it bites depends on the operation**. Requires `CanonicalFact{subject,relation,value}` vs `DerivedResult{operation,operands,result,discharged}` as typed objects, not text. | **measured** |
| CR16 | **Payload and state authority are distinct retirement permissions, and the difference is entirely the stopping decision.** Per-position KL decomposition: canonical promotion reproduces the **payload** at 0.0001 / 0.0010 / 0.0223 bits (copy / `+1` / reversal) and **100% of its residual sits on position 4, the first `<end_of_turn>`** — a reachable token, not an unreachable post-EOT probe, so the trajectory failures are real and not a scoring artifact. Copy's 0.0001-bit payload matches exp23's ingested record (1.14e-4), generalising CR6's termination-local signature from copy to all three operations. **The discharged result fixes exactly what the canonical fact does not:** `+1` derived is 0.0008 payload / **0.0003 termination** (its 0.0218 max is a post-termination probe). Design rule: **canonical promotion licenses `answer_scoped_retirement`; only discharged-result promotion licenses `general_state_retirement` for a compute operation.** Diagnostic worth keeping: peak *position* separates failure modes — wrong answers (pad-only, corrupt, derived-reversal) peak on a **digit** at 6.9–18.8 bits; right answers peak on **termination**. | **measured** |
| CR13 | **Operation capability is an estate precondition, and it decays with distance.** At 64K, 4B answers the literal `7431` to *four of five* query forms; 12B computes `+1111` correctly at 2K but returns `74311111` (concatenation) at 64K. A frontier over an operation the model never performed measures nothing, so capability is now an admission gate that runs first — `exp24_capability_prescreen.py`, ~30s vs ~15min per 64K arm. Constrains what an operation-conditioned semantic planner may assume: at range, models converge on retrieve-and-transcribe. | **measured** |
| CR17 | **Source authority is a query-time, per-LAYER gate — but only on the fact it was measured on.** Retiring global layer 29 *alone* flips a live source at 6.3298 bits with the source readable in all seven other global layers; no other singleton comes close (0.0008–0.2312), and count-matched triples behave oppositely (`[17,29,41]` 7.5827 vs `[5,23,47]` 0.0013). Authority is acquired at QUERY time in a 2–4 token window at the **chat turn boundary**, not at entity resolution — every prefix window fails, including one hiding the queried entity itself, and masking idx 22 flips it while idx 23 (*closer* to generation) does not. At least two distinct routes exist (`[29]` vs `[35,41,47]`), differing on layers, phases and window width. **Scope limit, load-bearing:** EXP-36 shows the regime this was measured in (contested source that wins by default) does not exist for the second fact in the same document, so none of these values is a transferable certificate. The 256-subset lattice is cancelled. | **measured, one cell** |
| CR18 | **The authority mechanism licenses no KV deletion — measured bit-exactly, not inferred.** Scoped and unscoped caches after a masked query are **bit-identical over the source span at every global layer** (max\|ΔK\| = max\|ΔV\| = 0.0, all eight), with the placebo rows also identical as a span-arithmetic cross-check. The source sits 55.6K tokens from the query, far outside the 1024 sliding window, so global layers are the *only* path to it and coverage is **complete** for this claim. Attention-time masking changes a layer's *output*, never its K/V *write* — so no amount of scoping can free the source, and this is architectural rather than statistical. Independently doubled by EXP-35's `B_norecord`, where the demoted source still answers correctly once uncontested. **"The decision persists" is not "the old evidence is dead."** | **measured** |
| CR19 | **The persisted decision is a ~49 KB object that rides in V, not K.** A layer-29 mask's entire global-layer divergence footprint is `{35,41,47}` — exactly the second authority route, re-derived by an independent instrument. Transplanting rows between arms at fixed positions (nothing added or removed, RoPE untouched, both row sets genuine model outputs at those positions) flips the later answer: boundary rows and model-turn rows are **each independently sufficient and neither is necessary**, and `Vonly` flips it while `Konly` does not. Sufficient object = 4 positions × 3 layers × 8 kv-heads × 256 dims, V only. **Not** a discrete revocable record: substituting either carrier region alone leaves the effect intact, so it is the leading edge of a contaminated continuation and a runtime cannot revoke it by rewriting boundary rows. Direct hit on `rsl-4`'s channel-specific-liveness premise. | **measured** |
| CR20 | **A live readable source CAN be overridden by promotion alone — the "never" tier is refuted, and the engine question moves to the promotion side.** Two independent refutations at 64K with nothing hidden and no attention intervention: two of four planted facts fall to a terse `key=value` record, and the *original* fact — the one declared un-overridable across k=0..3 — falls to a natural-language corroborator that works **alone**, with no terse record present. A caller can supply the winning material at promotion time: a record asserting a value verified absent from the whole 64K corpus wins outright, so nothing has to have been present when the context was built. **For this class `RESOLVE → PROMOTE → EXECUTE` needs no `DEAUTHORIZE`, no per-span certificate and no masked forwards** — which removes the entire CR17 layer machinery from the critical path for it. **Not yet a runtime rule:** an in-context type-correct ally correlates perfectly with the regime across the usable facts (depths 15–41%, both digit and word answers, so neither explains it) but fails necessity, and what rescues the resistant fact differs in *form* (natural-language vs `key=value`) as well as provenance. EXP-39 is the 2×2 that separates them, holding ally-presence at zero. | **measured, predictor open** |
| CR11 | **Physical compression** — the record was padded to equal token length, so CR6 shows semantic compressibility only. A production record should occupy fewer active slots or live outside the token stream. | CR10 |
| CR12 | **Predictive resident-envelope benchmark** — policies (no-prefetch / LRU / record-lookup / operation-conditioned / oracle) scored on *envelope coverage*, not exact-route match; see the objective change below. Needs CR7's oracle lifetime traces to define the target. | CR7 |

### Design consequence: predict residency, qualify execution

The CR-track findings change what the predictive hot cache is *for*. The original
framing made one predictor answer two questions at once — what should be
physically available, and what the model should actually compute against — so a
false positive could alter semantics and a false negative could damage the
answer. Separating them makes the prediction problem strictly easier and removes
its correctness burden:

```
resident set  ⊇  execution set
```

A prefetched page, expert or record does **not** participate in computation merely
by being resident. False positives then cost capacity, I/O and eviction pressure
but **cannot change model behaviour**; false negatives cost a cold replay or late
fetch, not a wrong answer. Prediction controls residency; qualification controls
execution. This is R13 Corollary A stated as an architecture rather than as a
measurement rule.

Four physical tiers follow: **active execution set** (semantically conservative —
recent KV, answer prefix, qualified source pages) / **predicted resident envelope**
(where prediction operates, legitimately larger) / **warm canonical records**
(CR6 makes these excellent cache objects — far cheaper than token-level K/V) /
**cold source and replay authority** (exact quotation, reinterpretation, record
failure, operations the promoted state does not cover). A miss follows a
provenance pointer rather than re-searching history.

**The experiment objective changes accordingly.** The question is no longer "did
the predictor choose the exact future route?" but "**did the predicted resident
envelope contain the eventually qualified route?**" — minimising
`late-miss + unused-prefetch + eviction` cost subject to the execution route
qualifying semantically, with behavioural parity held fixed. CR7 supplies the
second predictive input: not just *which* page will be needed but *for how long*,
which is what makes lifetime-based eviction possible where LRU and frequency
cannot see. Cache entries must be keyed by stable logical identity (record id,
logical token range, layer, page identity, epoch, expert owner) and resolved to
physical locations at execution time — the R14 mutating-state corollary, which a
snapshot-index defect in this track already established the hard way.

Attention-side and expert-side prediction can share one semantic planner but
**must stay separate caches** — different costs, different routes, and their
speedups must not be multiplied until end-to-end overlap is measured.

**Estate caveat, carried:** 2 of 5 values admitted — 1 genuine retrieval failure,
**2 rejected by a global answer-length cap, not by retrieval**. Per-value answer
budgets are a **new estate**, not a re-blend. Everything in CR4–CR6 rests on one
query form over two digit strings, and transcription is plausibly the worst case
since every output token is a literal.

**Not licensed by any of the above:** speculative SSD prefetch of route state. On
this estate the whole schedulable pool is 1.76 GiB and fits in RAM; with it
preloaded, storage stalls are exactly zero for every policy including no-prefetch.
K3 is the first estate where that question is even askable.

### Method rules earned here (now standing, see [`dec-funnel.md`](docs/dec-funnel.md))

- **R12** — name the metric's SPACE and the search's GUARANTEE; record an execution fingerprint (per-operation precision, trajectory, batching).
- **R13** — route membership is selected and validated jointly; isolated marginal importance is not a safe pruning criterion. Corollary A: resident ⊃ execution set, qualification attaches to the activation mask. Corollary B: a family's core is objective-relative — plan `argmin C_physical(R | residency)`, never `argmin |R|`.
- **R14** — **gate–claim congruence**: a gate licenses only claims over the object, trajectory and counterfactual relation it tests. Five instrument defects in this track were caught only by gates spanning the same object as the claim; two were missed by gates that did not.

---

## P1 — Spec'd implementations, sequenced behind P0 validation

Driver: two implementation tracks have shipped specs and review cycles but
are deliberately queued behind V1–V4 / R6 / MTP / BR4. Recording the
sequencing here so they don't drift to the front of the queue on momentum
alone, and so the gating preconditions are written down in one place.
Both specs live at `crates/larql-inference/docs/specs/`.

| # | Item | Crate(s) | Status | Gating preconditions | Effort |
|---|------|----------|--------|----------------------|--------|
| SQ1 | **Markov-residual engine migration** — ✅ **shipped**. Production impl in `larql_kv::engines::markov_residual` (Q4K hot-path routed via `attention_decode_step_native` + `ffn_decode_step_native`; `KvDispatch`/`KvEngine` wired). The `kv-cache-benchmark` reference impl was retired with the crate (2026-05-16). See [`markov-residual-engine.md`](crates/larql-inference/docs/specs/markov-residual-engine.md) for the contract it honours. | larql-kv | shipped | (a) ✅ V1/V2 measurement infra landed; (b) ✅ trait shape resolved via `KvEngine`+`KvDispatch`; (c) ✅ Q4K fixture in `larql-kv/benches/engine_decode.rs`. | done. |
| SQ2 | **Vindex-as-FFN compiled-fact lookup** — implement the cosine-thresholded FFN backend per `vindex-as-ffn.md`, with §5.4 cost-model refusal rule (`N > 2 * h_ref * K_layer`, `h_ref = 0.20`) at engine construction. | larql-inference + larql-vindex + larql-server (+ larql-router for /v1/ffn-lookup endpoint) | spec shipped, review-passed (incl. WalkFfn-substrate framing + corrected break-even algebra); impl not started | (a) **R6 must land first** — the spec's per-arch layer-policy table (§7) currently has TBD entries for gemma-3-1b/llama-2-7b/mistral-7b; with R6 these become probe calls instead of three separate Exp 52 re-runs. (b) **A video script or research workflow that needs paraphrase-reach compiled facts above the L1 i16 cos≥0.999 threshold.** None currently does — VID1/Act 3/VID4 all use different mechanisms. (c) Optionally: a deployment scenario where K_layer is large enough that the §5.4 break-even is comfortable (current decode K is 256–1024; at K=1024, h=0.20, crossover is N<410 — admissible but not a clear wall-clock win on small fact corpora). | ~2 weeks once unblocked. Greenfield (decorator + cache + endpoint + COMPILE wiring). |

**Why these are queued, not P0/P1-active**

- **SQ1 (Markov)**: contract is sound, reference impl already works, but
  it's engineering not research — and the open trait-shape question
  means migrating Markov first risks forcing
  `WindowedCheckpointEngine`/`ApolloEngine` into a shape that doesn't fit.
  Designing the trait once across all three engines (or at least
  resolving sibling-vs-trait before SQ1 lands) is cheaper than migrating
  one and refactoring twice. V1/V2 also produce the measurement
  infrastructure that lets the migration prove parity.
- **SQ2 (FFN lookup)**: the §5.4 cost model says it's a wash on the
  configurations LARQL actually runs at typical K (256–1024) without
  large compiled-fact corpora (>410 entries at K=1024, h=0.20). Building
  it now means it sits unused until a future video script needs
  paraphrase-reach. R6 also unblocks the per-arch layer-policy table —
  building SQ2 before R6 means re-doing the layer calibration manually
  for each architecture.

**Re-promotion conditions** (any one promotes that item to active P1):

- SQ1: V1/V2 land **and** trait-vs-sibling decision recorded in an ADR.
- SQ2: R6 lands **and** a specific video/experiment requires
  paraphrase-reach compiled facts (i.e. the L1 cos≥0.999 cache is
  measurably leaving paraphrases on the floor in that workflow).

**The fact that the specs are written is itself the work**

Both specs went through 2–3 review cycles and caught real issues that
would otherwise have surfaced as wall-clock surprises (the §5.4 algebra
error in particular: a refusal rule of `N > 2K` instead of
`N > 2*h*K` would have green-lit configurations that are net-negative
by ~5× at typical hit rates). The remaining work is implementation
under contract, not design — so when SQ1 or SQ2 do become active P1,
they start from a much better place than typical greenfield work.

---

## P1 — Generation UX (parallel to critical path)

Details in `larql-inference/ROADMAP.md` and `larql-cli/ROADMAP.md`.

- Sampling: `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`
- Multi-turn state: running KV across `larql chat` turns
- Long context: `--max-context N`, dynamic KV buffer growth
- OpenAI-compatible `/v1/chat/completions` (after streaming lands)
- Auto-extract on `larql run hf://owner/name`
- Gemma 3 4B regression smoke test (gate on `CI_INTEGRATION=1`)

---

## P1 — Voice bank: voices as first-class data (added 2026-08-09)

Gated on TTS funnel step 5 (green) — the speech engine is real enough
that maintaining `aru-12.tokens` as a manually prepared magic file is
beneath the abstraction level of the rest of the system.

**The design choice that matters: the asset is the *voice*, not "a MOSS
token file."** One logical voice accumulates model-specific
representations — MOSS conditions on spliced RVQ reference tokens,
Qwen3-TTS on a pooled ECAPA speaker vector, a future model on whatever
it requires. Same `--voice aru-12` resolves the representation the
target model needs. This is the CLI face of the voice-as-data ladder
(`docs/tts-funnel.md` §4): `clone` is the user's goal; *materialising a
voice identity into a representation usable by a particular model* is
what LARQL actually does (`voice derive` may become the truer verb).

```bash
larql voice clone reference.wav --name aru-12 --model moss-realtime
larql voice list
larql voice inspect aru-12
larql voice clone reference.wav --name aru-12 --model qwen3-tts-1.7b  # second representation
larql speak --voice aru-12 "Good evening."
larql voice compare aru-12 aru-03        # eventually
```

Voice package: derived representations + provenance, NOT the source
recording (originals stay where they belong):

```text
voices/aru-12/
├── voice.toml            # name, source sha256/duration/rate, representation manifest
└── representations/
    └── moss-realtime.tokens
```

Boundaries, pinned now: voice identity is user/runtime data; model
weights are model data. Voices are **never** bundled into a model's
vindex — 1 speech model × 100 local voices with no duplication. VINDEX3
interaction is resolution only: speech model + voice bank → resolved
conditioning representation → speech session.

Sequencing: `voice clone --model moss-realtime` is mechanically almost
available (WAV → MOSS codec encode → 16-channel token rows → package);
the Qwen3-TTS representation is the abstraction's first real test.
A practical dividend: `larql speak --voice aru-N` across a bank makes
the EXP-V ladder experiments (and eventual `voice compare`) one-liners.

---

## Standing execution rule — physical planning (extracted 2026-08-10)

> **A logical operator must be physically planned from
> `(format, operation, shape, hardware, workspace lifetime)`, never
> selected from tensor format alone.**

Extracted after the TTS funnel's TTFA work found the same pathology
three times in one day — a decode-shaped primitive applied repeatedly
to a prefill-shaped workload (FFN row-at-a-time; the blocked integer
GEMM in three loop structures; attention position-at-a-time gemv +
scalar softmax) — and measured the correct plans diverging by 1.6-4x
(`docs/tts-funnel.md`, 2026-08-10 entries). One logical operator,
multiple physical plans selected by execution phase:

```text
ATTENTION / FFN (logical)
├─ decode:  packed Q4K×Q8K matvec (bandwidth/issue-oriented)
├─ prefill: dequant-once + GEMM, batched softmax (compute-oriented)
└─ future:  long-context, resident-KV, speculative variants
```

Applies beyond speech: K3, dense vindex prefill, speculative branches,
multimodal prompt ingestion. Corollary for placement: not "a GPU
model" but per-phase operator routing (CPU attention + Metal FFN GEMM
is a legitimate plan). This is the query-optimizer half of the
model-as-database thesis, now empirical.

The *workspace lifetime* term earned its place empirically too
(2026-08-10): the production prefill FFN ran ~250 ms over its own
bench prediction because each layer allocated three fresh ~50 MB
dequant buffers (~4 GB of page traffic per prefill the warm-allocator
bench never paid). Kernel choice and shape were right; the workspace
policy wasn't. Measurement instrument for all of this:
`LARQL_PHASE_TIMING=1` (`larql_compute::phase_timing`) — the
production-path phase split that repeatedly outperformed arithmetic
estimates at finding the real bottleneck.

---

## P1 — Model-to-model fusion: the FUSE ladder (added 2026-08-09)

The principle to lock in now: **text is one interoperability layer, not
LARQL's model-to-model ABI.** Two models in the same runtime should
exchange the cheapest useful materialisation of a computation —
generated ids, residual state, KV state — with English text reserved
for when text genuinely is the cheapest interchange format.

The abstraction is model-to-model fusion, NOT "Qwen token sharing."
Qwen→MOSS is only the first proving ground (shared tokenizer lineage
makes the experiments easy); the architecture is:

```text
producer model → intermediate state / token domain / residual
              → binding (model-specific; LARQL owns the mechanism)
              → consumer model
```

Compatibility levels, weakest binding first:

```text
1. token-compatible      reuse ids directly
2. vocabulary-mappable   cheap token-domain translation
3. hidden-state          reuse residuals directly
4. projectable           small learned/fixed projection
5. state-composable      semantic state + target-specific state
```

The runtime consequence: a decode step exposes more than its final
token — `GeneratedToken { id, hidden, kv_position, .. }` — and the
consumer takes the view it needs (ids → text protocol, residual →
conditioning). Generated text becomes one *view* of the computation.
VINDEX3 eventually describes interfaces, not pairings: a model declares
`output_domain` (token ids, semantic hidden) and `input_domain` (text
tokens, hidden state, acoustic context); the binding
(mapping/projection/adapter) is a separate, inspectable object.

### The ladder (speech instance; each rung gated on the last)

- **FUSE-0 — token pipe.** LLM-generated ids feed MOSS's text channel
  directly (MOSS has no text head; text is already an input stream, and
  the 12-token lead maps onto a generated-token queue naturally). Gate:
  identical speech tokens to the encode(decode(ids)) round trip.
  Stated precisely: *zero-copy token-domain forwarding when producer
  and consumer domains happen to be compatible* — not "speech fusion
  requires a Qwen LLM". Mostly an engineering cleanup; the value is the
  primitive it installs: token-domain piping between models.
- **FUSE-1 — residual comparison.** Same text prefix through a generic
  Qwen LLM and the MOSS backbone; compare hiddens layer-by-layer.
  Cosine is not enough (the voice ladder's lesson) — behavioural
  probes and linear mappings too.
- **FUSE-2 — direct residual substitution.** Replace a MOSS final
  backbone hidden with a shape-compatible LLM residual; run the proven
  depth transformer. Ask only: plausible codebooks? terminates? how far
  do logits move? Cheap falsification — MOSS's backbone state carries
  semantics + acoustic history + previous frame + conversation KV,
  while an LLM residual carries semantics + LLM state, so straight
  substitution *should* fail informatively.
- **FUSE-3 — small bridge.** `H_llm → projection P → depth stage`,
  acoustic conditioning preserved separately. The thesis test: how
  little MOSS backbone computation is required once semantic state
  already exists upstream?
- **FUSE-4 — acoustic residual injection.**
  `H_llm + A(previous audio tokens) → P → speech decoder`. If this
  works, MOSS's backbone has been decomposed into semantic and
  acoustic operands — and the steady-state frame stops paying for a
  second full language-model pass.

Prior art, tracked honestly: PRIME-Speech (HF 2606.30944) already
drives a causal speech decoder from intermediate hidden states of a
frozen LM — one specifically *trained* architecture. TADA (arXiv
2602.23068) aligns text/acoustic representations. LARQL's differentiated
claim is the **runtime composition primitive**: arbitrary producer →
declared interface → binding → arbitrary consumer, across models that
were never trained together. K3 → SpeechBinding → MOSS (or a small fast
planner → speech model) is the long-term Jarvis pipeline this enables.

---

## Speech track — competitive position & the three proofs (added 2026-08-09)

Position audit (2026-08-09, external claims are vendors'/leaderboards',
not our measurements): LARQL's speech-token generator at Q4 on a laptop
CPU (~RTF 0.63 conventional) sits in the same throughput order as the
vendor's own MOSS figure on an L20 GPU (RTF 0.51, 180 ms TTFB) — as an
inference-engine result, unusually strong. But the frontier is faster on
cold latency (Fish S2 ~100 ms TTFA; Qwen3-TTS ~97 ms e2e claims;
ElevenLabs Flash ~75 ms model inference), clone quality is externally
unproven (top of Artificial Analysis is closed models; best open-weight
~Fish S2 Pro), and `voice clone` as a product surface is table stakes.
The moat is NOT "local TTS in Rust" (VoxCPM2 has GGUF/ONNX/ANE/Rust
ports; Chatterbox Nano claims 3x realtime on 8-core CPU): it is **one
execution system that understands multiple generative architectures,
their physical representation, their state, and their composition** —
plus the voice-as-data research (a 2026 study argues commercial
"cloning" behaves like style transfer; the V-series asks what identity
state actually is, which is better-timed than another clone API).

The three proofs that change the story:

1. **TTFA < 500 ms** while retaining the CPU steady-state class
   (in flight, 2026-08-10: TTFA 2.0 → **~1.25 s** and steady state
   1.6 → **~1.9x** via three CPU replanning steps, all token-exact
   through the dump oracle; the prefill budget is 99%-accounted by
   phase split — FFN block 904 ms + attention projections 131 ms are
   the Metal `simdgroup_matrix` scope, and the measured operators
   alone are sufficient to cross the gate at ~460 ms projected.
   #242 was falsified — no build fix was needed).
2. **Controlled blind clone comparison** — aru-12 versus Sonic 3.5,
   Eleven v3, Fish S2, VoxCPM2, Qwen3-TTS, and MOSS-reference, scored
   blind. Until this runs, no claim about voice quality, only about
   engine performance.
3. **A second, structurally different TTS architecture** through
   LARQL/VINDEX3 — the model-independence claim made real (pairs with
   the voice bank's Qwen3-TTS representation).

Landed, they upgrade "MOSS runs very fast in Rust" to "a local
generative speech runtime competitive with specialized stacks,
model-independent, exposing model state hosted systems hide." The
caveat to keep repeating until the audio-device path exists: current
numbers are the token-generation path; end-to-end comparisons against
vendor stacks wait for codec + ring + device integration.

---

## P2 — Film checklist

- [ ] Confirm Gemma 4 26B A4B public config (expert count, top-K, active-param figure, GQA ratio). Replace every `~` in `docs/demo-script-gemma4-moe.md`.
- [ ] Measure real footprint + latency on `google/gemma-4-31b-it` for Act 1.
- [ ] Reliability pass on `RemoteWalkBackend` (timeouts, retries, partial shard outage). **(P2 per ADR-019.)**
- [ ] `RemoteExpertBackend` same reliability pass. **(P2 per ADR-019.)**
- [ ] Decide repo-public date. `cargo install larql-cli && larql serve` must be live the week the video drops.
- [ ] Pick expert IDs for the Act 3 swap shot — one that fires on medical prompts, one that doesn't.
- [x] ~~Resolve ADR-019 before final Act 2 / Act 3 commitments.~~ Resolved 2026-05-09.

---

## P2 — Competitive parity (positioning analysis 2026-05-09)

Driver: items surfaced by [docs/positioning.md](docs/positioning.md) that the
ollama / vLLM / llama.cpp comparison treats as table stakes but LARQL doesn't
yet ship.

**Re-evaluated 2026-05-09 under the substrate framing** (see "Engine purpose"
above). Each item is now scored by *"does this affect the credibility of
measured technique deltas, or accelerate experiments?"* Items that only serve
"becoming a production engine" are explicitly **dropped or deferred** — LARQL
will never be a production engine, so spending engineering on production-engine
features that don't tighten the experiment loop is scope creep.

| # | Item | Crate | Substrate verdict | Notes |
|---|------|-------|-------------------|-------|
| CB1 | Continuous batching engine — iteration-level scheduler | larql-inference + larql-server | **DROPPED** | Pure concurrency-throughput; doesn't affect single-stream baseline; doesn't accelerate any experiment. Re-open only if a future experiment needs concurrent decode. |
| CB2 | PagedAttention KV allocator | larql-inference | **DROPPED** | Pairs with CB1; useless without it. |
| CB3 | Concurrent stress benchmark | larql-server + bench/ | **DROPPED** | Measures a property the substrate framing doesn't care about. |
| MCP1 | MCP client + server in `larql serve` | larql-server | **DEFERRED** | Re-open only if a research workflow needs LARQL as an MCP-callable tool from inside an agent loop. Otherwise UX. |
| TM1 | Thinking-mode toggle | larql-inference + larql-server | **DEFERRED** | Re-open only if reasoning-trace structure becomes part of an experiment (e.g. probing thinking tokens). |
| RD1 | RMS-norm + scalar-mul pre-fusion shader (ADR-016 follow-up) | larql-compute | **KEEP** (small) | Affects baseline by ~0.1 ms/layer × 34 = ~3.4 ms; below baseline-credibility threshold floor but pure win. |
| (MTP1–MTP6 promoted to P1 — see "P1 — Gemma 4 MTP drafter support" above) | | | KEEP | Both substrate (new mechanism to study) and baseline (Ollama supports it on Gemma 4). |
| (SD1–SD2 promoted to P1) | | | KEEP | Reusable verification machinery; supports any future drafter-based technique. |
| Multi-machine MoE grid (former critical-path 5–10 + C9) | larql-router + larql-server + larql-inference | **DEMOTED 2026-05-09 per ADR-019** | Items now individually tracked as MMG1–MMG7 in dedicated section "P2 — Multi-machine MoE grid (deferred per ADR-019)" above. |

**Decision recorded 2026-05-09**: multi-tenant batched serving is out of
scope. LARQL will never be a production engine; the substrate framing's
"engine purpose" section above makes the call explicit. CB1, CB2, CB3 are
dropped. Re-open only if a specific *experiment* needs concurrent decode
(currently none does).

---

## Loose ends (shipped features with open follow-ups)

| Item | Crate | Detail |
|---|---|---|
| `KernelHandle` spread to 9 remaining tiled shaders | larql-compute | Mechanical, same pattern as q4_matvec_v4 |
| `dispatch_full_pipeline` 30+ params | larql-compute | Bundle into `FullPipelineRefs<'_>` context |
| `QuantFormat` match spread (14 files) | larql-compute | Introduce `FormatRoute` enum |
| `ProfileTimings` producer | larql-compute | Wire commit/wait boundaries into decode_token |
| Benches in CI | larql-compute | GHA workflow written, needs trigger merged |
| `--compact` loader for non-MoE models | larql-vindex | `WeightFfn::forward` panics on compact vindex |
| MoE compact mode | larql-vindex | Blocked on per-expert feature-major files |
| Fix `dispatch_full_pipeline` layer_scalar (dense) | larql-compute | **Was: "Non-urgent: Gemma 3 4B has scalar=0". Now: needs verification on Gemma 4 31B (substrate-primary per ADR-019). If 31B has scalar≠0, this becomes urgent.** |
| Cross-vindex dedup (tokenizer, down_meta) | larql-vindex | Low priority, ~200 MB duplicated at 7 vindexes |
| `BaseVindex` trait + `PatchedVindex` composition (ADR-worthy) | larql-vindex | `patch/{overlay.rs, overlay_apply.rs, format.rs, knn_store.rs}` ≈ 2.6k LOC mirrors `format/load.rs` (~640 LOC). Introduce a `BaseVindex` trait so the read-only loader and the overlay path share dtype/quant decode; today both reimplement it. Targets ~1k LOC reduction in `patch/` and one source of truth for weight decode. |
| Codebase-review hardening (2026-05-28) | workspace | ~7 verified high/medium items from the whole-codebase review — see §"Codebase hardening (review 2026-05-28)" above and [`docs/audits/codebase-review-2026-05-28.md`](docs/audits/codebase-review-2026-05-28.md). |
