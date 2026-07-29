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
- **4 KV-cache engines**: MarkovRS (287×), UnlimitedContext (254×), TurboQuant (4×), Apollo (20,000×) — all at ~95 tok/s on Gemma 3 4B Metal.
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
- **larql-router self-healing + HTTP/3 + hedged-dispatch phase** (2026-05-16): MoE expert routing (ADR-0018, per-(layer, expert-range) replication keys), Prometheus `/metrics` (ADR-0017), Phase 4 HTTP/3 shard transport behind `--http3-shards` / `--http3-port` (ADR-0019, h3 0.0.8 + h3-quinn 0.0.10 + h3-axum 0.2), hot-shard hysteresis (ADR-0014 amendment, `--hot-shard-demote-ratio` default 0.8), backpressure tier (ADR-0020 — `--saturation-ceiling N` filter in `route()` / `route_expert()`, dispatcher distinguishes 503 saturation from 400 no-owner via `has_owners_for()`, emits `Retry-After: 0.5`, bumps `larql_router_route_saturation_total`), long-running chaos test (`tests/test_grid_chaos.rs`, 5,000 random ticks × 2 variants, asserts ledger consistency + coverage floor + no `route()` panic), hedged dispatch (ADR-0021 — opt-in via `--hedge-after-ms M`, new `route_with_rank` / `route_expert_with_rank` grid APIs, `hedged_post_json` racing helper, dense + MoE fan-outs wired, `route_hedge_fires_total` / `route_hedge_wins_total` counters; supersedes the original "speculative next-layer prefetch" P1 framing — an audit falsified that framing since the router sees one batched call per token against a single input residual, so hedge-the-slow-primary is the legitimate router-layer optimisation). Concurrent-route bench (`bench_route_concurrent`, 2026-05-16) surfaced lock-contention plateau: pre-swap 1 = 5.6 → 4 = 8.7 → 8 = **4.0** → 16 = 3.6 Melem/s (8 workers *worse* than 1 — pathological). **Lock primitive swap** (2026-05-16): `tokio::sync::RwLock<GridState>` → `parking_lot::RwLock<GridState>` across larql-router and tests. Every grid critical section is short and sync (no `await` held under the lock), so synchronous is semantically correct and the compiler enforces it (parking_lot guards are `!Send`). Post-swap: 1 = 6.4 / 4 = 11.1 / 8 = 7.2 / 16 = 6.1 Melem/s — **+14% / +28% / +80% / +70%**, pathological 8-worker collapse eliminated. 220 tests still pass. Saturation-filter cost on the happy path: ~108 ns vs ~113 ns baseline (in noise); all-saturated short-circuit ~57 ns. Router test surface: 169 lib + 50 integration = **219 tests** (220 with `--features http3`). Coverage **~93%**. Five examples (`embed_grid`, `static_shards_server`, `admin_client`, `fanout_dispatch`, `saturation_backpressure`); criterion benches cover dense + MoE + saturation + concurrent-route. Multi-host deployment runbook at [`crates/larql-router/docs/multi-host-demo.md`](crates/larql-router/docs/multi-host-demo.md). Server-side `GET /v1/shard/{model}/{start}-{end}` audited + documented in [`crates/larql-server/docs/router-spec.md`](crates/larql-server/docs/router-spec.md) §4. ADRs: [0017](docs/adr/0017-prometheus-metrics.md), [0018](docs/adr/0018-moe-expert-routing.md), [0019](docs/adr/0019-http3-shard-transport.md), [0020](docs/adr/0020-route-backpressure-tier.md), [0021](docs/adr/0021-hedged-dispatch.md).
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
    unlimited_context, apollo) now owns a `dequant_scratch` field; quant methods
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
| **V1 ✅ DONE 2026-05-31 — FALSIFIED (dense)** | Hash routing across all layers (extend exp 27) | **Exp 27 Gemma 3 4B L0 at top-2048/d_ffn (20% mask) → KL=0.030.** Walk boundary sweep (April 2026) progressively pushed the walk down through layers on Gemma 3 4B. **One-layer one-model evidence in hand.** | "5× FFN bandwidth reduction holds at end-to-end output, not just one layer" → **FALSIFIED.** Per-layer KL ≤ 0.05 thresholds DON'T compound: applied together they give +5.4 to +7.7 bits/token NLL and 78–95% drift on all 3 dense archs. The per-layer screen is anti-correlated with the truth. Deployable bandwidth ~2.4–2.9× (gate projection still paid), not 5×, and catastrophic anyway. | **DELIVERED:** per-layer threshold tables + compounding NLL/drift + cheap-route realizability + honest bandwidth, 3 dense archs (`bench/aim-validation/v1_*.json`), harness `examples/walk_ffn_v1_hash_routing.rs`, writeup [`docs/diagnoses/v1-hash-routing.md`](docs/diagnoses/v1-hash-routing.md). **MoE-within-expert version OPEN** (dense harness measures the wrong object on the 26B → needs expert-aware tooling). | ~1 week (done) |
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
(MarkovRS, UnlimitedContext, TurboQuant, Apollo) live in research-only
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
  `UnlimitedContext`, `TurboQuant`, `Apollo`).
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
| U3 | ComputeBackend redesign — Step 3c (engine migration) | larql-kv, larql-inference | **shipped 2026-05-16** (partial); follow-up in U8 | All six engines accept `Box<dyn EngineBackend>` in constructors. `KvDispatch` widened with `Option<&VectorIndex>` on attention intents + new `coarse_prefill` / `coarse_decode_step` (quantization-agnostic, backends inspect index format internally). `StandardEngine` fully migrated: routes Q4K through `coarse_prefill` on `CpuBackend` (which calls production `predict_q4k_prefill` / `predict_q4k_decode_step_direct`). **27.6 tok/s on Gemma 3 4B Q4K, M3 Max, 8 threads — slightly faster than the legacy `larql-cpu` path (24.0 tok/s).** `NoCache` migrated (slow on purpose: O(N²) debug fallback). Others (`MarkovResidual`, `UnlimitedContext`, `TurboQuant`, `Apollo`) still carry their bespoke `prefill_q4k` overrides — they work correctly but run at ~0.4 tok/s through f32-dequant fallback. Migration to fast Q4K kernels via the dispatch trait is **U8** below. Spec: [`kv-dispatch-quantization.md`](crates/larql-inference/docs/specs/kv-dispatch-quantization.md). |
| U4 | AsyncComputeBackend impl — Steps A1–A5 (the trait + foundation) | larql-inference, larql-compute, larql-compute-metal, larql-kv | **A1–A3 + A5 (StandardEngine) shipped 2026-05-16; A4 next** | A1 ✅ trait + handle types in `larql-inference/src/async_compute_backend.rs` (per-handle inner traits, `read(self: Box<Self>)` — stable-Rust translation of spec's `Arc<dyn AsyncHandleInner>` pattern). A2 ✅ `CpuBackend` async impl as degenerate `Ready*` wrapper, 6 bit-parity tests vs sync. A3 ✅ `MetalBackend` scaffold via CPU-delegation, feature-gated; 4 Metal-aware bit-parity tests pass under `--features metal`. A5 ✅ for `StandardEngine`: `with_async_backend` constructor + internal `BackendSlot` enum + async dispatch helpers + 8 new parity tests (`larql-inference`: 1002 lib tests; `larql-kv`: 221 lib tests). A4 next: real `MTLCommandBuffer` deferred dispatch (4–8 weeks). Remaining engines' A5 slices (`MarkovResidual`, `UnlimitedContext`, `TurboQuant`, `NoCache`, `Apollo`) compose on the same pattern (~1–2 weeks each). |
| U5 | AsyncComputeBackend impl — Step A6 (per-engine specialised shaders) | larql-compute, larql-kv | **spec'd, not started** | This is the tok/s payoff. Priority order: `attention_step_windowed` (the `standard:window=N` win), then engine-specific intents in order of impact — `markov-rs` Metal K/V recompute, `apollo` pipelined boundary upload, `turbo-quant` codec kernel. Each shader paired with a real-model bench. Ongoing — months of iterative work. |
| U6 | AsyncComputeBackend impl — Step A7 (VulkanBackend) | larql-compute | **spec'd, not started — blocked on U9-U12** | Same trait shape as Metal, different primitives (`VkCommandPool`, semaphores, SPIR-V). Validates the multi-backend story is real, not Metal-shaped. 6–10 weeks **once U9-U12 unblock the engine layer**. Today the substrate trait is drop-in but `larql-inference` still has 30+ `cfg(feature = "metal")` gates and 2 `downcast_ref::<MetalBackend>()` sites that conflate "Metal" with "GPU pipeline" — landing Vulkan against today's tree would force per-backend cfg explosion across the inference crate. |
| U7 | AsyncComputeBackend impl — Step A8 (CudaBackend) + server wiring | larql-compute, larql-server | **spec'd, not started — blocked on U9-U12** | CUDA streams map naturally to the deferred-dispatch shape — designed against it. Server wiring (deferred from `kv-engine-unification.md` §10.6) lands here: `larql-server`'s `handle_stream_generate` switches from direct `generate_streaming` to `generate_with_engine` against an `AsyncComputeBackend`, finally honouring `LARQL_KV_ENGINE` server-side. 6–10 weeks Cuda + 1–2 weeks server. Same engine-layer blockers as U6. |
| U8 | Engine migration — bespoke `prefill_q4k` paths onto dispatch trait | larql-kv, larql-inference | **specced, not started** | `MarkovResidual`, `UnlimitedContext`, `TurboQuant`, `Apollo` each carry an engine-side `prefill_q4k` override that bypasses the dispatch trait's `coarse_prefill` / `coarse_decode_step` intents and uses slower CPU code paths (dequant-to-f32 + f32 sgemv) instead of the production `predict_q4k_*` kernels. Result: ~0.4 tok/s vs `StandardEngine`'s 27.6 tok/s on the same hardware. Each engine has legitimate specialisation (RsStore residuals, per-window K/V checkpoints, WHT+Lloyd-Max codec, boundary residual injection) — the migration keeps that engine-side logic but routes the per-layer matvec through `larql_compute::QuantMatVec::q4k_matvec` instead of dequant-then-f32. Per-engine: ~2-5 days. See [`kv-dispatch-quantization.md`](crates/larql-inference/docs/specs/kv-dispatch-quantization.md) Phase 2. |
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
| C7 | KV compression as **default** for long context (Apollo / MarkovRS / UnlimitedContext / TurboQuant) | larql-inference | engines reachable on `run`/`walk` (CPU) via `--engine` / `LARQL_KV_ENGINE`; default still `standard` (production K/V cache); GPU performance on opt-in engines requires AsyncComputeBackend (see U-series below) | Unification spec at [`kv-engine-unification.md`](crates/larql-inference/docs/specs/kv-engine-unification.md) — all 7 steps landed. MarkovRS / UnlimitedContext / TurboQuant opt-in via `--engine` (CPU-correct, Metal works via CPU-fallback delegation). Apollo bench-only. Promoting any of these as default for long context requires `AsyncComputeBackend` Step A6 (engine-specific Metal shaders) to land — see U5 below. Server engine wiring also blocked on AsyncComputeBackend (U7); without it the server would silently downgrade Metal decode to CPU. |
| C8 | BR4 (Boundary refs Phase 4 — bounded KV eviction + durability-first capture) | larql-server + larql-inference | not started | See § "P1 — Boundary refs and cold-context storage" below. The CPU track makes BR4 load-bearing because long-context CPU inference can't keep raw KV in RAM. |
| C9 | Distributed-load-balancing for "model spans 4 consumer machines" | larql-router + larql-server | shipped (grid + rebalancer) | **DEMOTED to P2 per ADR-019 (2026-05-09)** — substantial production-engineering with no current experiment requiring multi-machine. Single-shard grid (already shipped) sufficient for substrate. Re-promote if a specific experiment needs multi-machine. |
| C10 | CPU bench harness — `larql bench --cpu` with per-stage breakdown matched against `llama.cpp -ngl 0` | larql-cli + bench/ | **DISCREPANCY RESOLVED 2026-06-02 — no regression; true gap ~1.6–1.8×.** The 1.50× (05-16) vs 1.93× (05-31) split was **two stacked measurement confounds**, not a real change: (1) **larql path mismatch** — 27.6 was the `StandardEngine` path, 23.6 the legacy `larql bench --cpu` (`predict_kquant_decode_step`) path; a stable ~12% delta (26.4 vs 23.5 today), so comparing one date's StandardEngine against the other's legacy path manufactured a phantom "regression"; (2) **llama.cpp harness artifact** — the 45.5 was an unwarmed/short-n ollama `num_gpu=0` fluke; warmed + n=128 it converges to **42.8–43.0 = llama-bench's 42.99** (both harnesses, both dates agree at ~43). Reconciled like-for-like (M3 Max, t=8, warm): **larql 23.5 legacy / 26.4 StandardEngine vs llama.cpp 43.0 → 1.6–1.8×.** Gap is C12 (both attn AND FFN already use the int8 Q8_K SDOT kernel via `attention_decode_step_native`). **Free wins landed (2026-06-02):** `larql bench --cpu` now also reports the production StandardEngine row; new `--ollama-cpu` forces `num_gpu=0`+`num_thread` so `--ollama` is a true CPU baseline (was silently Metal-GPU). Reconciled artifact `bench/baselines/c10_gemma3-4b_cpu_reconciled.json`. **26B-A4B baseline LANDED 2026-06-10** (`c10_gemma4-26b-a4b_cpu_reconciled.json`): llama.cpp **32.1** vs larql in-process **7.1** default / **9.7** with `LARQL_Q4K_DIRECT_ATTN=1` / loopback 7.3 (t=8, warm, n=128, drift-checked). The 26B gap (4.5×) is **f32-residency byte traffic** (attn 4.15 GB + dense slab 2.14 GB + lm_head 2.95 GB per token vs llama.cpp ~2.1 GB all-quantized; every leg bandwidth-saturated ~62–71 GB/s), NOT the C12 kernel (experts already int8 SDOT, ~8% of bytes). Medium-term tier 62%→70% per the gate rule. Method addition: **pmset AC check + cross-engine drift bracket are now mandatory** — the first session was invalidated by a silent battery drain (llama.cpp itself collapsed 34→1 tok/s at 31% battery; far beyond the 1.5–3× thermal class). | CPU-track baseline-credibility threshold can't be enforced without this. First acceptance test: Gemma 3 4B Q4_K on M3 Max CPU vs quant-matched `llama.cpp -ngl 0`. Then Llama 2 7B + Mistral 7B for cross-arch CPU + the 26B-A4B MoE baseline. Major improvement 2026-05-15→05-16 (2.78× → 1.50×) — see `bench/baselines/cpu/COMPARISON.md` and `DIAGNOSIS-2026-05-16-thread-scaling.md`; reconciliation `bench/baselines/c10_gemma3-4b_cpu_reconciled.json`. |
| C11 | Architecture rule enforcement — CI check for "no GPU-only paths in core" | scripts/ + crate boundaries | not started | Static check: anything in `larql-inference` core (not `metal/`, not `cpu/`) must compile and pass tests with Metal feature off. Prevents the dual-track from drifting into Metal-locked code. |
| C12 | Q4K decode kernel — hand-asm aarch64 to close the 1.50× gap to llama.cpp | larql-compute | **v1 asm landed opt-in 2026-06-02 (`LARQL_Q4K_ASM=1`); roofline reframed the work.** Two 2026-06-02 results: (a) **Roofline microbench** (`benches/q4k_q8k_matvec.rs`) shows the kernel is **compute/issue-bound, NOT DRAM-bandwidth-bound** — scalar 9.3 vs NEON 17.7 GiB/s on identical data, size-invariant — which **overturns the `DIAGNOSIS-2026-05-16` "memory-system-level" conclusion** and confirms hand-asm scheduling is a real lever (17.7 GiB/s ↔ ~33 cyc/super-block, exactly as specced). (b) **`q4k_q8k_matvec_asm`** (whole super-block dot in one `asm!` block, 8 scales as vector lanes killing the 8 scalar `ldrb`) — **bit-exact** (`q8k_matvec_asm_matches_scalar_bit_exact`), **+3.7–4.9% isolated**, ~+1–2% e2e (diluted: opt-in covers `matvec_into` callers — attention Q/K/V/O + `down` — but NOT the fused `gate_up`). **Finding: latency-hiding has low headroom** — a 4-accumulator variant showed no reliable gain (the inlined row loop lets the OoO core already overlap super-blocks), so **the two-super-block interleave is deprioritized**; the real lever to reach ~28 GiB/s is **instruction-count reduction** (perf-counter-guided, llama.cpp-style vectorized scale path) + **asm-ifying `gate_up`** (lifts the e2e ceiling). See spec §"2026-06-02 roofline measurement". | Per-core gap is **1.73× constant across thread counts** (5.7 vs 9.88 tok/s single-threaded on M3 Max). Same algorithm (Q4K × Q8K with NEON SDOT), same `vdotq_s32` instructions — llama.cpp uses hand-written inline aarch64 asm with two-super-block interleaving + explicit prefetch hints, we use Rust intrinsics lowered by LLVM. Effective bandwidth: ~63 GB/s vs ~95 GB/s. **Per-stage profile (`LARQL_INSTRUMENT_UNLIMITED=1` on Gemma 3 4B 8-thread, 2026-05-16): FFN 26.0 ms (74%) + Attention 9.3-11.0 ms (26%, grows with ctx) + Embed ~0 ms = 35-37 ms/step.** FFN matvec on gate/up/down (4608 × 9216) is the dominant target; attention matvec is the same kernel on smaller matrices. The 38 tok/s asymptote (FFN-alone) sets the floor any engine can reach on the current kernel — Standard and UnlimitedContext both hit 26.6 tok/s on Gemma 3 4B Q4K CPU (8-thread, 40-token prompt, 64 decode tokens) because both route through the same `attention_decode_step_native` + `ffn_decode_step_native` hot paths. Phases: (1) hand-asm Q4K matvec on the FFN tile shapes (gate/up/down) — closes ~95% of the gap, 1-2 weeks; (2) pre-formatted block layout — 1.1-1.2× on top, 3-5 days; (3) Q6K kernel for `ffn_down` — 1.05×, 2-3 days; (4) reduce rayon launch overhead — 1.04×, 2-3 days. Acceptance: ≥9.5 tok/s single-core, ≥39 tok/s 8-thread on Gemma 3 4B Q4K. Spec: [`crates/larql-compute/docs/q4k-decode-kernel.md`](crates/larql-compute/docs/q4k-decode-kernel.md). Per-stage measurement protocol: see "C12 per-stage measurement" below. |

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

- `LARQL_INSTRUMENT_UNLIMITED=1` — prints `embed / attention / ffn` per `extend_q4k` call from `larql_kv::engines::unlimited_context::rs_extend_from_checkpoint_q4k`. Captures the per-token, per-layer-aggregated breakdown. Source: `crates/larql-kv/src/engines/unlimited_context/extend.rs`.
- `LARQL_INSTRUMENT_MARKOV=1` — same shape for `markov-residual`, kept for cross-engine sanity that both substrate paths agree. Source: `crates/larql-kv/src/engines/markov_residual/q4k.rs`.

Reproducer (Gemma 3 4B Q4K, M3 Max, default 8 threads):

```
cargo build --release -p larql-cli
LARQL_INSTRUMENT_UNLIMITED=1 ./target/release/larql bench \
  ~/.cache/larql/local/gemma3-4b-q4k-v2.vindex \
  --backends cpu --engine unlimited-context -n 32
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
| BR4 | Phase 4: bounded KV eviction + durability-first capture (Option A) | larql-server + larql-inference | not started |
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
  `UnlimitedContextEngine`/`ApolloEngine` into a shape that doesn't fit.
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
