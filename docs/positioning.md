# LARQL positioning vs ollama, vLLM, llama.cpp

**Class: RECORD.** This dated positioning and aim-validation record keeps its original claims and results. Start with [current architecture](architecture-stack.md) and [VINDEX3 status](vindex3/status.md) for the present system.

**Date**: 2026-05-09
**Audience**: anyone framing what LARQL *is* and *is not* relative to the dominant inference engines.
**Companion docs**:
- `crates/larql-compute/docs/llama-cpp-comparison.md` — kernel-level Metal diff vs llama.cpp.
- `ROADMAP.md` — top-level critical path; § "Engine purpose" for the load-bearing framing; § "ADR-019" for the MoE-substrate decision; § "Video pipeline" for the engineering↔video dependency map.
- `crates/larql-server/ROADMAP.md` § N0 — OpenAI API compatibility plan (scope reduced — see roadmap implications below).

---

## TL;DR — the ultimate aim

> **Serve the largest models at blazing speed on consumer hardware, with as little GPU as possible — ideally eventually none.**

Frontier-scale models (100B–1T+ params) are physically incompatible with consumer hardware under naïve dense matmul: a 671B Q4 model touches ~336 GB per forward pass; consumer DDR5 is ~50 GB/s; that's 6.7 sec/token. The bandwidth wall cannot be beaten by faster compute. The *only* path through is **touching fewer weights per token** — sparse retrieval over a queryable weight database. Vindex was always for this.

Combined sparse-retrieval techniques were projected at ~100× (hash routing 5× × FP4 2× × KV compression 10×) → the same 671B model at ~134 ms/token on consumer DDR5. **Aim-validation has since revised this (2026-05-31):** the **hash-routing 5× is falsified** — per-layer FFN sparsity that individually clears KL ≤ 0.05 does *not* compound across depth (V1, three dense archs: +5–8 bits/token NLL, 78–95% argmax drift); **FP4 2× holds** (V2, confirmed cross-arch, near-lossless). So the realistic compound is smaller, and the dominant bandwidth lever is MoE **active-parameter** sparsity (37B of 671B), not within-FFN hash routing. Detail: [`docs/diagnoses/v1-hash-routing.md`](diagnoses/v1-hash-routing.md), [`v2-fp4-generality.md`](diagnoses/v2-fp4-generality.md).

### Two permanent tracks

The aim demands competitive performance *now* and progress toward GPU-free *eventually*. These are co-equal tracks, neither sacrifices to the other:

1. **GPU track** — maintains competitive baseline against ollama / vLLM / llama.cpp on Metal. Permanent. Never demoted in favour of CPU work. Without this, every claim measured on the engine fails the credibility threshold below.
2. **CPU track** — drives toward "blazing big models on consumer hardware without GPU." The ultimate aim. Built **in addition to**, not instead of, the GPU track.

**Architecture rule**: vindex / WalkFfn / sparse retrieval is the shared invention. Only kernels differ. No GPU-only paths in the core design — every technique developed on one track has a path to the other.

### Why "research substrate" framing is the means, not the end

LARQL is a research substrate — but substrate-for-its-own-sake isn't the goal. The substrate exists because the techniques that make the ultimate aim possible (sparse retrieval, hash routing, FP4, KV compression, expert sharding, AOT compilation, boundary refs) have to be developed *somewhere*. LARQL is that somewhere. Adoption, OpenAI-API ergonomics, multi-tenant batched serving, and other "production engine" concerns are out of scope except where they accelerate experiments or affect measurement credibility.

**LARQL is not a production inference engine and will not become one** in the commercial sense. But it must operate at production-engine baseline performance on its leading device class — otherwise the techniques developed on it can't be credibly compared against state-of-the-art.

### Achievability — conditionally yes, asymmetric across model class

The aim is **achievable for MoE frontier models on consumer hardware**. That's where the field is going (DeepSeek-V3, Llama 4 Maverick, Gemma 4 26B-A4B, GPT-OSS family — all MoE).

The arithmetic on a 671B MoE with 37B active params:

| Stack | Bytes touched/token | tok/s @ 50 GB/s consumer DDR5 |
|---|---|---|
| Naïve dense over active experts | 18.5 GB | 2.7 |
| ~~+ hash-routed FFN within active experts (5×, exp 27)~~ — **FALSIFIED (V1)**: doesn't compound across depth | — | — |
| + FP4 (2×, exp 26 → **confirmed V2**) | 9.3 GB | ~5.4 |

(The 5× row is struck: V1 showed within-FFN hash routing doesn't compound. The realistic post-active-sparsity stack is FP4 2× alone unless a different bandwidth lever lands. MoE active-param sparsity — the 18.5 GB starting point — remains the dominant win.)

Tier-by-tier honest confidence:

| Acceptance tier | Confidence |
|---|---|
| Short-term (Gemma 3 4B CPU within 10% of `llama.cpp -ngl 0`) | ~95% — pure engineering |
| Medium-term (Gemma 4 26B-A4B at ≥10 tok/s on 64 GB consumer, no GPU) | **~62%** (was ~80%→70%→62%, 2026-05-31) — V2 confirms FP4 & 26B fits RAM (+); but measured **~4.4 tok/s, ~2.3× short of 10**, easy lever gone, and Q4K-direct already washed out at representative context (GQA O(N²) is the real wall). Gated on C10 (match-vs-beat llama.cpp): rises to ~70 if "match," drops to ~55 if "beat" |
| Long-term (100B-class MoE at ≥5 tok/s, no GPU) | **~52%** (was ~60%→55%→52%, 2026-05-31) — 100B@FP4 fits RAM so the disk bet is moot here (+); but a **two-probe hit** to the exploitable-structure prior (V1 + routing locality) trims it; 50 is defensible if you weight that over the disk-risk removal. Cross-MoE-router check would settle it |
| Ultimate (671B-class via consumer multi-machine grid) | **~30%** (was ~40%, 2026-05-31) — 671B@FP4 (~335 GB) exceeds one machine and routing locality closes the disk-resident escape hatch → only the harder multi-machine grid (C9/P2) remains; integration risk dominates |
| Dense frontier (if field stays dense at 1T+) | **~10%** (was ~15%, 2026-05-31) — hash-routing 5× falsified (1 TB Q4 → ~10 s/token); needs attention-sparsification breakthroughs |

The "100× combined effect" assumes techniques compound multiplicatively. ADR-015 ("isolated kernel speedup ≠ end-to-end win") said they often don't — and aim-validation (2026-05-31) bore that out: of the four load-bearing assumptions, **V1 (hash-routing compounding) is FALSIFIED**, **V2 (FP4 generality) is CONFIRMED**, and **V3 (disk-residency) is undermined** by poor MoE routing locality. V4 (full compound on a >RAM model) remains, blocked on hardware. Net: the compound is smaller than 100× and the ultimate-aim arithmetic now rests on MoE active-param sparsity + FP4, not within-FFN hash routing or disk-spill. See **"P0 — Aim-validation tests (V1–V4)"** in `ROADMAP.md` and `docs/diagnoses/`.

#### Known unknowns (from ROADMAP.md § "Engine purpose")

The bandwidth math assumes the architecture cooperates with sparse retrieval. Five open questions could shift the achievability boundaries (full table in `ROADMAP.md`):

- **KU1** — static-attention fraction at 31B scale (validated at 4B: 91.7%; if degrades, VID7 weakens and MTP acceptance drops).
- **KU2** — softmax bottleneck above ~1,142-token RoPE distance (Q-side fixable; KV-side at last position not, with current architecture; BR4 is the workaround, not a fix).
- **KU3** — FP4 friendliness across non-Gemma archs. **RESOLVED 2026-05-31 (V2): CONFIRMED** — ≥99.8% per-feature R<16 on Gemma 3 / Granite, FP4 near-lossless, no QAT.
- **KU4** — hash-routing compounding across all layers. **RESOLVED 2026-05-31 (V1): FALSIFIED** — doesn't compound on 3 dense archs.
- **KU5** — mmap thrash on disk-resident frontier models. **RESOLVED 2026-05-31 (V3 + routing-locality): locality POOR** — no cacheable hot subset, >RAM MoE thrashes.

KU3/KU4/KU5 now resolved (see `docs/diagnoses/`). KU1, KU2 are parked.

#### ADR-019 resolved (2026-05-09)

Substrate-primary model is **Gemma 4 31B dense + vindex**. MoE coverage retained at single-machine scale (Gemma 4 26B-A4B for cross-arch validation, virtual-expert work). Multi-machine MoE grid demoted to P2. Reasoning: vindex is MoE taken to its logical extreme (every fact is its own expert), and multi-machine production-engineering doesn't accelerate any current experiment. Full resolution + demotions/promotions tables in `ROADMAP.md` § "ADR-019".

### Baseline-credibility threshold (methodological, not commercial)

> LARQL must be within **10% of llama.cpp / ollama** on the matching model + quantisation + context-length configuration **on the device class the claim is being made on**, before any *"+N% from technique X"* claim is published. CPU technique → CPU baseline. GPU technique → GPU baseline.

Current state (2026-05-09):

| Track | Configuration | LARQL | State-of-the-art | Gap | Threshold? |
|---|---|---|---|---|---|
| **GPU (Metal)** | Gemma 3 4B decode | 88 tok/s | ollama ~103 | 17% behind | over (defensible-with-caveat) |
| **GPU (Metal)** | Gemma 3 4B prefill (340 tok) | per-pos matvec | gemm | 14× behind | far over |
| **GPU (Metal)** | Gemma 4 + MTP (when adopted) | 88 tok/s no-MTP | ~225 with MTP | ~2.6× behind | far over |
| **CPU** | Gemma 3 4B decode | not measured | llama.cpp `-ngl 0` | unknown | not measurable yet (needs C10) |
| **CPU** | Gemma 4 26B-A4B decode | grid 18.3 tok/s | unknown | unknown | not measurable yet |

The kernel projects on the GPU track (D-ATTN-MTG, D-PREFILL-MM2, D-METAL-PLE, MTP1–MTP6) are load-bearing because they buy methodological standing for every claim measured on Metal. The CPU-track items (C1–C11 in `ROADMAP.md`) are load-bearing because they're the ultimate-aim path itself.

### The 2026 inference-engine landscape (for context, not competition)

The market settled into a "hybrid stack" pattern: **prototype in Ollama → scale in vLLM → embed with llama.cpp**. LARQL doesn't compete in any of those three roles. It's a fourth thing: **a research substrate at production-engine baseline, building toward blazing inference of frontier models on consumer hardware without GPU.**

---

## Four categories of "inference engine"

| Category | Representative | What it optimises | LARQL position |
|---|---|---|---|
| **Local single-user runtime** | Ollama, llama.cpp, LM Studio, MLX, vllm-mlx | Single-stream latency on local hardware; easy install; broad model library | **not competing — but baseline must match within 10%** so deltas measured on top are credible |
| **Batched serving framework** | vLLM, TGI | Multi-tenant throughput at concurrency; KV memory efficiency; multi-GPU tensor parallelism | **out of scope** — CB1, CB2, CB3 dropped in `ROADMAP.md` |
| **Research / edit Python tooling** | TransformerLens, nnsight, pyvene | Hooks, capture/patch, weight surgery in research notebooks | **adjacent** — LARQL pushes these primitives into the engine itself rather than wrapping a separate inference path |
| **Research substrate at production baseline** | (LARQL — no other player) | Plug-in surface for new mechanisms / arches / kernels / edits, on real weights, with real performance, so technique deltas are credible | **the actual category** |

The comparison literature ([decodesfuture](https://www.decodesfuture.com/articles/llama-cpp-vs-ollama-vs-vllm-local-llm-stack-guide), [aimadetools](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/), [Towards AI stress test](https://pub.towardsai.net/i-tested-ollama-vs-vllm-vs-llama-cpp-the-easiest-one-collapses-at-5-concurrent-users-d4f8e0e84886?gi=275236a76e13)) treats the first two categories as the whole field. The fourth doesn't appear because no existing player occupies it — Python research tooling cedes performance, production engines cede mechanism. LARQL refuses both trades.

---

## Feature matrix

✅ shipped · 🟡 partial / planned · ❌ missing · n/a not applicable

| Feature | LARQL | Ollama | vLLM | llama.cpp |
|---|---|---|---|---|
| **Inference perf** | | | | |
| Apple Silicon Metal | ✅ (88 tok/s Gemma 3 4B) | ✅ (~103 tok/s — wraps llama.cpp) | 🟡 (`vllm-mlx` fork) | ✅ baseline |
| CUDA | ❌ | ✅ | ✅ | ✅ |
| ROCm / Vulkan | ❌ | ✅ | partial | ✅ |
| Flash attention | ❌ (D-ATTN-MTG planned) | ✅ via llama.cpp | ✅ FA4 default on Blackwell | ✅ |
| `simdgroup_matrix` prefill matmul | ❌ (D-PREFILL-MM2 planned) | ✅ via llama.cpp | n/a (CUDA path) | ✅ |
| **Concurrent serving** | | | | |
| Continuous batching | ❌ | ❌ (collapses at ~5 concurrent users) | ✅ (485 tok/s aggregate at 10 concurrent) | ❌ |
| PagedAttention | ❌ | ❌ | ✅ (~4× KV memory waste reduction) | ❌ |
| Multi-GPU tensor parallelism | ❌ | ❌ | ✅ | partial |
| Multi-host MoE expert sharding | ✅ (gRPC self-assembling grid) | ❌ | ❌ | ❌ |
| Multi-Token Prediction (Gemma 4 MTP drafters, released 2026-05-05) | ❌ (MTP1–MTP6 P1) | ✅ supported | ✅ supported | ❌ (notably absent from Google's launch list) |
| Speculative decoding (n-gram / EAGLE / draft-model) | ❌ (SD1–SD2 P1) | partial | ✅ all three | ✅ |
| **API surface** | | | | |
| OpenAI `/v1/chat/completions` | 🟡 SSE wired, no chat template | ✅ | ✅ | ✅ |
| OpenAI `/v1/embeddings` | 🟡 server has embed surface | ✅ | ✅ | ✅ |
| WebSocket token streaming | ✅ | ❌ | ✅ | ✅ |
| Constrained decoding (JSON / GBNF) | ❌ | partial (format json) | ✅ | ✅ GBNF |
| Multimodal (vision) | ❌ | ✅ | ✅ | ✅ |
| **Tooling / UX** | | | | |
| MCP client built into server | ❌ | ❌ (external tools) | ❌ | ✅ (`llama-server`, Mar 2026) |
| Thinking-mode toggle | ❌ | ✅ | ✅ | partial |
| Chat template + EOS | ❌ (critical-path #1) | ✅ | ✅ | ✅ |
| Hot model swap | ✅ (vindex) | ✅ (auto-unload) | ❌ (locks one model in VRAM) | ✅ |
| Model library / pull command | ✅ (`larql pull`, hf://) | ✅ | n/a | partial |
| **Quantisation** | | | | |
| Q4_K / Q6_K | ✅ | ✅ | ❌ (FP16/FP8/AWQ/GPTQ) | ✅ |
| AWQ / GPTQ | ❌ | ❌ | ✅ | partial |
| FP8 / FP4 | 🟡 (FP4 vindex exp 26) | ❌ | ✅ | partial |
| **What only LARQL does** | | | | |
| Vindex (model-as-database) | ✅ | ❌ | ❌ | ❌ |
| LQL query language over weights | ✅ | ❌ | ❌ | ❌ |
| Mech-interp hooks (M1–M8) | ✅ (lazarus parity) | ❌ | ❌ | ❌ |
| MEMIT / KNN weight edits | ✅ | ❌ | ❌ | ❌ |
| AOT compilation (residual programs) | ✅ | ❌ | ❌ | ❌ |
| WASM-in-FFN compute primitives | ✅ | ❌ | ❌ | ❌ |
| Boundary refs (residual codec) | ✅ Phase 1–3 | ❌ | ❌ | ❌ |
| KV cache engines (MarkovRS, WindowedCheckpoint, Apollo) | ✅ | ❌ | ❌ | ❌ |
| Cross-arch dispatch as first-class concern | ✅ (ADR-018) | n/a | n/a | n/a |

---

## vs Ollama (apples-to-apples, single-user local Metal)

Same niche. Direct measurements:

| Workload | LARQL | Ollama | Gap |
|---|---|---|---|
| Gemma 3 4B decode (M3 Max) | 88 tok/s | ~103 tok/s | **1.17×** behind |
| Gemma 3 4B prefill (18 tok) | per-position matvec | gemm | **3.9×** behind |
| Gemma 3 4B prefill (340 tok) | per-position matvec | gemm | **14×** behind |
| Gemma 4 26B A4B MoE | 19.4 tok/s | not supported | LARQL ahead by virtue of arch coverage |
| Concurrent users | not stress-tested | collapses at ~5 | comparable weakness |

**Where LARQL wins**: vindex, LQL, edit/compile primitives, MoE grid sharding, hot model swap (parity but model-as-database is a stronger story), Gemma 4 / Gemma 4 26B A4B coverage.

**Where Ollama wins today**: 1.17× decode tok/s, ~14× prefill on long prompts (kernel gap), broader CUDA/ROCm/Vulkan coverage, vision models, larger production model library, polished install UX, thinking-mode toggle.

**Closeable items in the LARQL roadmap**: D-ATTN-MTG (flash attention, +5–8 tok/s), D-PREFILL-MM2 (`simdgroup_matrix` matmul, closes prefill gap entirely), critical-path items #1–#2 (chat template, CLI streaming).

---

## vs vLLM (different category — batched serving)

vLLM v0.17.1 (March 2026) ships Model Runner V2 (+56% throughput on GB200) and FlashAttention 4 default on Blackwell SM100/SM103. The [Towards AI stress test](https://pub.towardsai.net/i-tested-ollama-vs-vllm-vs-llama-cpp-the-easiest-one-collapses-at-5-concurrent-users-d4f8e0e84886?gi=275236a76e13) measured vLLM sustaining ~485 tok/s aggregate at 10 concurrent users on Llama 3.1 8B via continuous batching, while Ollama collapsed at ~5.

| Dimension | LARQL | vLLM |
|---|---|---|
| Hardware | Metal + CPU (Mac/Linux/Win) | CUDA-first; `vllm-mlx` Mac fork emerging |
| Concurrency model | Single-stream + multi-host expert sharding | PagedAttention + continuous batching (1000s concurrent) |
| Throughput at scale | not designed for it | 10–100× LARQL at high concurrency on a single H100 |
| Single-user latency | competitive (88 tok/s tier on M3 Max) | overkill — vLLM wins throughput, not single-stream latency |
| Multi-GPU | ❌ tensor parallelism | ✅ tensor parallelism |
| Multi-host MoE | ✅ self-assembling gRPC grid (different shape from TP) | ❌ (TP only) |
| Quantisation | Q4_K / Q6_K / FP4 vindex | AWQ / GPTQ / FP8 |
| Hot model swap | ✅ | ❌ (locks one model in VRAM) |

**Where LARQL wins**: anywhere that isn't max-throughput multi-tenant batched serving on NVIDIA datacenter GPUs. Hot model swap. Multi-host MoE grid. Mac/Linux/Windows portability without CUDA.

**What it would take to enter vLLM's lane**: continuous batching engine + PagedAttention. Quarters of work; not currently on any roadmap entry. See P2 additions below.

---

## vs llama.cpp (kernel substrate of Ollama)

Detailed in `crates/larql-compute/docs/llama-cpp-comparison.md`. Short version: three architectural Metal-kernel gaps explain the entire 1.17× decode + 4–14× prefill picture:

1. **Flash attention** — llama.cpp's `kernel_flash_attn_ext_vec_reduce` vs LARQL's 3-dispatch attention. D-ATTN-MTG.
2. **`simdgroup_matrix` prefill matmul** — Apple 8×8 register-tile intrinsics. D-PREFILL-MM2.
3. **RMS-norm pre-fusion shape** — different fusion direction than ours. ~0.1 ms/layer, low priority.

**New as of March 2026**: `llama-server` shipped a built-in MCP client with full tool/resource/prompt support. This closed a major gap vs Ollama (which still requires external tooling). LARQL has no MCP surface — see P2 additions.

**New as of March 2026**: parallel multi-GPU model loading across GPU contexts.

---

## Critical: Gemma 4 MTP drafters (Google, 2026-05-05)

Released 4 days before this doc was written. Drops the timing of every other gap analysis here.

**What Google released**: official MTP drafter checkpoints for every Gemma 4 variant LARQL supports — `google/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant`. The 26B-A4B drafter is 0.4B BF16 (~4 layers, vs 25.2B total / 3.8B active in target). Apache 2.0 (code) + CC-BY-4.0 (weights).

**Architecture** (from [ai.google.dev/gemma/docs/mtp](https://ai.google.dev/gemma/docs/mtp/overview)):
1. Drafter shares the **input embedding table** with the target.
2. Drafter consumes the target's **last-layer activations**, concatenates with token embeddings, and **down-projects to drafter dimension**.
3. Drafter and target share the **KV cache**.
4. E2B/E4B add an "Efficient Embedder" clustering layer.

**Measured speedups**:
- **Apple Silicon: ~2.2× at speculative batch 4–8** (Google blog) — directly relevant to LARQL on M3 Max.
- ~2× on RTX PRO 6000 (26B target).
- Up to 3× generally.

**Engines supported out-of-the-box**: HF Transformers, MLX, vLLM, SGLang, **Ollama**, LiteRT-LM. **llama.cpp is conspicuously absent** from Google's launch list — meaning Ollama's MTP support comes through a different path than its usual llama.cpp wrapping (likely a grafted spec-decode implementation), and llama.cpp users don't get this for free.

**Independent ecosystem signal**: Red Hat AI shipped an EAGLE-3 speculator for `gemma-4-26B-A4B-it` (0.9B drafter) at the same time — the spec-decode space for our exact supported MoE is hot.

**Competitive impact**:

| Scenario | LARQL | Ollama+MTP | Gap |
|---|---|---|---|
| Today (no MTP anywhere) | 88 tok/s Gemma 3 4B | ~103 tok/s | 1.17× behind |
| Once Gemma 4 + MTP becomes default on Ollama | 88 tok/s (no MTP path) | ~225 tok/s on Gemma 4 | **~2.6× behind** |
| LARQL ships MTP1–MTP6 | ~190 tok/s decode, 26B-A4B 19.4 → ~43 tok/s | same ~225 tok/s | back to ~1.2× |

**Roadmap response**: MTP1 was promoted from P2 to **P1** in the same pass that produced this doc. New P1 section in `ROADMAP.md` ("P1 — Gemma 4 MTP drafter support") tracks MTP1–MTP6 + SD1–SD2.

The launch list omitting llama.cpp is itself a positioning data point: it means LARQL adding MTP isn't "catching up to llama.cpp" — it's **landing on the same launch tier as MLX and vLLM** for a Google-blessed feature on Google's own model family. That's a stronger pitch than chasing tok/s parity.

---

## Where LARQL is uncontested

None of the comparison articles surface anything matching:

- **Vindex / model-as-database / LQL** — no equivalent in any of Ollama, vLLM, llama.cpp, MLX, LM Studio, TGI.
- **In-process mech-interp surface** (M1–M8 hooks, MEMIT, KNN edits, AOT compilation, WASM-in-FFN) — these aren't "inference engine" features at all in the comparison literature. The closest analogues are scattered Python research codebases (TransformerLens, nnsight) — not productionised, not query-language-backed, not running on the actual production inference path.
- **Self-assembling MoE expert grid over gRPC** — vLLM does tensor parallelism (different shape); nobody else does multi-host expert sharding for MoE.
- **Boundary refs / residual codec** — no analogue. Closest concept is the academic literature on KV compression; nothing shipped as a contract-bearing wire format.
- **Cross-arch routing as a first-class engine concern** (ADR-017 / ADR-018) — Ollama and llama.cpp have model-family conditionals scattered through the codebase; vLLM has architecture classes but doesn't treat shader retention rationale as a first-class artifact.

**Implication**: every minute spent comparing tok/s vs Ollama is a minute not spent telling the story of what LARQL does that nobody else does. The inference perf needs to be "good enough not to embarrass the rest of the stack" — not "category-leading." The category we lead is a different category.

---

## Roadmap implications

Re-tiered 2026-05-09 under the substrate framing. Each item is scored by *"does this affect baseline credibility, or accelerate experiments?"* — items that only serve "becoming a production engine" are dropped.

| ID | Item | Tier | Substrate verdict |
|---|---|---|---|
| CB1 | Continuous batching engine | ~~P2~~ | **DROPPED** — concurrency-throughput, not single-stream baseline. Re-open only if a future experiment needs concurrent decode. |
| CB2 | PagedAttention KV allocator | ~~P2~~ | **DROPPED** — pairs with CB1. |
| CB3 | Concurrent stress benchmark | ~~P2~~ | **DROPPED** — measures a property the substrate framing doesn't care about. |
| MCP1 | MCP server built into `larql serve` | ~~P2~~ | **DEFERRED** — UX, doesn't change measurement. Re-open if a research workflow needs LARQL as an MCP-callable tool. |
| TM1 | Thinking-mode toggle | ~~P2~~ | **DEFERRED** — UX. Re-open if reasoning-trace structure becomes part of an experiment. |
| RD1 | RMS-norm + scalar-mul pre-fusion | P2 | **KEEP** — small baseline win (~3.4 ms). |
| MTP1–MTP6 | Gemma 4 MTP drafter support | **P1** | **KEEP — load-bearing**. Both substrate (new mechanism to study) and baseline (Ollama supports it on Gemma 4; without MTP, LARQL fails the 10% threshold on Gemma 4 by ~2.6×). |
| SD1–SD2 | Generic spec-decode + EAGLE-3 | **P1** | **KEEP** — reusable verification machinery for any future drafter-based technique. |
| D-ATTN-MTG | Flash attention multi-TG retry | P0 | **KEEP — load-bearing**. Without it, attention-mechanism deltas are muddied by missing baseline. |
| D-PREFILL-MM2 | `simdgroup_matrix` matmul rewrite | P0 | **KEEP — load-bearing**. Until landed, all prefill-touching technique claims fail the 10% threshold (currently 4–14× behind). |
| D-METAL-PLE | Gemma 4 E2B Per-Layer Embeddings on Metal | P0 | **KEEP — load-bearing**. Without it, every Gemma 4 E2B experiment runs CPU-fallback and any delta is unattributable. |
| AI1–AI6 | Architecture independence hardening | P1 | **KEEP — load-bearing**. Cross-arch deltas need clean arch boundaries or they're arch-specific accidents. |
| T1–T7, C1–C3 | Interpretability truthfulness + commit semantics | P0 | **KEEP — central**. Substrate that lies to you is worse than no substrate. |
| MI4–MI8 | Rich attribution, causal operators, Q4K/MoE trace parity | P0 | **KEEP — central**. The substrate's plug-in surface lives here. |
| R1–R5 | OV/RD → engine primitive promotion | P1 | **KEEP — exemplar pattern**. This *is* the substrate development model: experiments harden into engine APIs. |
| R6 | Depth-fraction-law probe API as engine primitive | P1 | **KEEP — sequencing-critical**. Must land before MTP3; consumed by MTP3 (drafter activation extraction layer choice), virtual-expert dispatch (Act 3), grammar-mask routing. |
| BR4–BR8 | Boundary refs (residual codec) server integration | P1 | **KEEP** — itself a research direction (Shannon-arc continuation). |
| Coverage → 90% | larql-compute test coverage | P1 | **KEEP — load-bearing**. Measurement integrity needs correctness trust. |
| Acts 1/2/3/4 demo narrative | Demo plan | n/a | **REFRAME** — not a product demo. Act 1 = "model is a database" (substrate pitch); Act 2 = "experts are addressable" (reframed per ADR-019 — single-machine, not multi-machine); Act 3 = replace an expert (single-machine via VID4-style); Act 4 = "I killed attention" (gated on KU1 + MTP6, becomes video VID7). |
| Critical-path #1–#2 (chat template + EOS, CLI streaming) | Demo unblocker | P0 | **KEEP only as needed by experiments**. If experiments operate at residual/weight level, EOS handling is downstream. If demos are the communication channel, ship #1 to make Act 1 work. |
| OpenAI API surface (server N0) | OpenAI compat | n/a | **REDUCE SCOPE** — keep only what experiments call. Drop `/v1/completions`, `/v1/responses`, multimodal until an experiment needs them. |

Items already on the roadmap that the comparison validates as load-bearing:

- **Critical-path #1 (chat template + EOS)** — without this, OpenAI-compat is non-functional and the demo loops.
- **Critical-path #2 (CLI token streaming)** — server has SSE/WS; CLI side is the gap.
- **D-ATTN-MTG** (flash attention multi-TG retry) — closes ~5–8 tok/s of the Ollama decode gap.
- **D-PREFILL-MM2** (`simdgroup_matrix` matmul rewrite) — closes the 4–14× prefill gap entirely.
- **N0 (OpenAI API compatibility, larql-server)** — `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, constrained decoding.
- **AI1–AI6 (architecture independence hardening)** — required before claiming arch-agnosticity in marketing material.

---

## Sources

- [llama.cpp vs Ollama vs vLLM: 2026 Comparison](https://www.decodesfuture.com/articles/llama-cpp-vs-ollama-vs-vllm-local-llm-stack-guide)
- [Fastest Local LLM Setup: Ollama vs vLLM vs llama.cpp Real Benchmarks](https://insiderllm.com/guides/llamacpp-vs-ollama-vs-vllm/)
- [Ollama vs llama.cpp vs vLLM — Which Should You Use? (2026)](https://www.aimadetools.com/blog/ollama-vs-llama-cpp-vs-vllm/)
- [llama.cpp vs MLX vs Ollama vs vLLM: Apple Silicon 2026](https://contracollective.com/blog/llama-cpp-vs-mlx-ollama-vllm-apple-silicon-2026)
- [I Tested Ollama vs vLLM vs llama.cpp — collapses at 5 concurrent users](https://pub.towardsai.net/i-tested-ollama-vs-vllm-vs-llama-cpp-the-easiest-one-collapses-at-5-concurrent-users-d4f8e0e84886?gi=275236a76e13)
- [vLLM vs Ollama vs llama.cpp vs TGI — Engines Compared (2026)](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Ollama vs vLLM Performance Benchmark 2026 (SitePoint)](https://www.sitepoint.com/ollama-vs-vllm-performance-benchmark-2026/)
- [Ollama vs LM Studio vs llama.cpp vs vLLM (CraftRigs)](https://craftrigs.com/comparisons/ollama-vs-lm-studio-vs-llama-cpp-vs-vllm-2026/)
- [Performance vs Practicality: vLLM and Ollama (Robert McDermott)](https://robert-mcdermott.medium.com/performance-vs-practicality-a-comparison-of-vllm-and-ollama-104acad250fd)
- [2026 Mac Inference: vllm-mlx vs Ollama vs llama.cpp (MACGPU)](https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html)
- [Inside vLLM: Anatomy of a High-Throughput Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [vLLM PagedAttention + Continuous Batching explained (RunPod)](https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching)
- [Local LLM Benchmarks on Apple Silicon (ModelPiper)](https://modelpiper.com/blog/local-llm-benchmarks-apple-silicon/)
- [Performance of llama.cpp on Apple Silicon (GitHub discussion)](https://github.com/ggml-org/llama.cpp/discussions/4167)
- `crates/larql-compute/docs/llama-cpp-comparison.md` — kernel-level Metal architecture diff.

### Gemma 4 MTP (2026-05-05)

- [Accelerating Gemma 4: faster inference with multi-token prediction drafters (Google blog)](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
- [Speed-up Gemma 4 with Multi-Token Prediction (ai.google.dev overview)](https://ai.google.dev/gemma/docs/mtp/overview)
- [Gemma 4 MTP using Hugging Face Transformers](https://ai.google.dev/gemma/docs/mtp/mtp)
- [google/gemma-4-26B-A4B-it-assistant (HuggingFace)](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant)
- [Google AI Releases MTP Drafters for Gemma 4: Up to 3x Faster Inference (MarkTechPost)](https://www.marktechpost.com/2026/05/06/google-ai-releases-multi-token-prediction-mtp-drafters-for-gemma-4-delivering-up-to-3x-faster-inference-without-quality-loss/)
- [Gemma 4 MTP Drafter: 3x Faster Inference (BuildFastWithAI)](https://www.buildfastwithai.com/blogs/gemma-4-mtp-drafter-faster-inference)
- [Red Hat AI: EAGLE-3 speculator for gemma-4-26B-A4B-it](https://x.com/RedHat_AI/status/2044078615789224255)
