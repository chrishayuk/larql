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
| Exp 27 (hash routing top-2048 mask) | 5× fewer FFN weights touched at KL=0.03 |
| MEMIT / COMPOSE / AOT | Compile programs into smaller weight footprints |
| WASM-in-FFN | Replace heavy kernels with cheap programs where the math allows |
| Boundary refs / residual codec | Compress KV for long context on bandwidth-bound hardware |
| Shannon arc (1 bit/char on Frankenstein) | Theoretical compression ceiling — how far this can go |
| Mech-interp surface (M1–M8) | Discover *which* weights actually do the work; rest stays on disk |
| Cross-arch coverage | The technique stack must generalise |

Combined effect (rough math, conservative): hash routing 5× × FP4 2× × KV
compression 10× = **100× effective bandwidth reduction** on the right
corpus. 670 GB model → 6.7 GB-equivalent traffic → ~134 ms/token on
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
| + hash-routed FFN within active experts (5×) | 5 GB | 10 |
| + FP4 (2×) | 2.5 GB | 20 |
| + KV compression on long context | depends | further win |

**This is where the field is going** — DeepSeek-V3, Llama 4 Maverick,
Gemma 4 26B-A4B, GPT-OSS family are all MoE. The aim hits this case.

**Dense frontier models (hypothetical 2T dense)**

1 TB at Q4. Hash routing 5× → 200 GB. FP4 2× → 100 GB. At 50 GB/s →
**2 sec/token = 0.5 tok/s. Not blazing.** Would need attention
sparsification too, which is open research.

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
| Medium-term: Gemma 4 26B-A4B at ≥10 tok/s on 64 GB consumer, no GPU | **~80%** | MoE math works; gated on V1+V2 below |
| Long-term: 100B-class MoE at ≥5 tok/s, no GPU | **~60%** | Adds disk-locality bet (V3) |
| Ultimate: 671B-class via multi-machine grid | **~40%** | Integration risk dominates; gated on ADR-019 (resolved 2026-05-09 — multi-machine MoE grid demoted to P2; ultimate tier becomes a P2 stretch) |
| Dense frontier (if the field stays dense at 1T+) | **~15%** | Needs research breakthroughs outside engineering control |

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
| KU3 | FP4 friendliness across non-Gemma archs | Tested on Gemma 3 4B only (99.83% per-feature). | V2 covers this. If V2 fails, FP4 becomes per-model retraining work, not a free 2×. |
| KU4 | Hash-routing compounding across all layers | Tested on Gemma 3 4B L0 only (KL=0.030 at top-2048). | V1 covers this. If V1 fails per-layer at deep layers, the 5× claim shrinks. |
| KU5 | mmap thrash on disk-resident frontier models | Untested. | V3 covers this. If V3 fails, ultimate aim shrinks to "models that fit in RAM." |

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
| **CPU** | Gemma 3 4B Q4K decode | 24.5 tok/s | llama.cpp Q4_K_M CPU 42.53 tok/s | ~1.69× behind | over (KV-cache + direct Q4_K matvec + NEON Q4_K/Q6_K/f32_dot + Q4 lm_head + 4-way acc + par_chunks_mut(32) + Q4_K×Q8_K sdot + auto-t=8 on Apple silicon landed 2026-05-15/16; ~68× over original 0.36 tok/s baseline. Per-core ratio 1.73× (kernel inner-loop vs llama.cpp's hand-asm); remaining gap needs prefetch + scheduling. See `bench/baselines/cpu/DIAGNOSIS.md`) |
| **CPU** | Gemma 4 26B-A4B decode | currently grid 18.3 tok/s | unknown | unknown | not measurable yet |

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
| **V1** | Hash routing across all layers (extend exp 27) | **Exp 27 Gemma 3 4B L0 at top-2048/d_ffn (20% mask) → KL=0.030.** Walk boundary sweep (April 2026) progressively pushed the walk down through layers on Gemma 3 4B. **One-layer one-model evidence in hand.** | "5× FFN bandwidth reduction holds at end-to-end output, not just one layer" | Per-layer top-k threshold table (KL ≤ 0.05 per layer; end-to-end perplexity within X% of dense). Measured on Gemma 3 4B (existing exp 27 baseline) + Gemma 4 26B-A4B + Llama 2 7B + Mistral 7B. | ~1 week |
| **V2** | FP4 generality (extend exp 26 across archs) | **Exp 26: gemma3-4b-f16.vindex is 99.83% FP4-friendly per-feature without QAT (down is the tail at 99.65%).** Single-arch evidence in hand. | "FP4-friendliness is universal, not Gemma-3-4B specific" | FP4-friendliness fraction per arch + per layer, with QAT-required threshold flagged. Cross-arch corpus. | ~1 week |
| **V3** | mmap'd vindex with sparse access on disk-resident frontier MoE | **None.** This is the genuinely-new-territory item. Risk dominates the long-term tier confidence (~60%). | "Disk locality + page-fault behaviour is acceptable when only top-k experts fire" | Page-fault rate, p50/p99 disk-read latency, end-to-end tok/s on a model that doesn't fit RAM. Compare cold-start vs warm-conversation locality. Use existing 26B-A4B vindex on a 32 GB-RAM machine to force paging. | ~2 weeks |
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
| U6 | AsyncComputeBackend impl — Step A7 (VulkanBackend) | larql-compute | **spec'd, not started** | Same trait shape as Metal, different primitives (`VkCommandPool`, semaphores, SPIR-V). Validates the multi-backend story is real, not Metal-shaped. 6–10 weeks. |
| U7 | AsyncComputeBackend impl — Step A8 (CudaBackend) + server wiring | larql-compute, larql-server | **spec'd, not started** | CUDA streams map naturally to the deferred-dispatch shape — designed against it. Server wiring (deferred from `kv-engine-unification.md` §10.6) lands here: `larql-server`'s `handle_stream_generate` switches from direct `generate_streaming` to `generate_with_engine` against an `AsyncComputeBackend`, finally honouring `LARQL_KV_ENGINE` server-side. 6–10 weeks Cuda + 1–2 weeks server. |
| U8 | Engine migration — bespoke `prefill_q4k` paths onto dispatch trait | larql-kv, larql-inference | **specced, not started** | `MarkovResidual`, `UnlimitedContext`, `TurboQuant`, `Apollo` each carry an engine-side `prefill_q4k` override that bypasses the dispatch trait's `coarse_prefill` / `coarse_decode_step` intents and uses slower CPU code paths (dequant-to-f32 + f32 sgemv) instead of the production `predict_q4k_*` kernels. Result: ~0.4 tok/s vs `StandardEngine`'s 27.6 tok/s on the same hardware. Each engine has legitimate specialisation (RsStore residuals, per-window K/V checkpoints, WHT+Lloyd-Max codec, boundary residual injection) — the migration keeps that engine-side logic but routes the per-layer matvec through `larql_compute::QuantMatVec::q4k_matvec` instead of dequant-then-f32. Per-engine: ~2-5 days. See [`kv-dispatch-quantization.md`](crates/larql-inference/docs/specs/kv-dispatch-quantization.md) Phase 2. |

**Implementation order**: U1 ✅ → U2 ✅ → U3 ✅ → U4 (A1–A3 + A5
StandardEngine slice ✅; A4 real Metal deferred dispatch next; A5
remaining engines compose on the same pattern) → U5 (highest tok/s
leverage, run continuously alongside U6/U7) → U6 → U7. U4's A4 is the
next critical-path commitment; until it lands, U5/U6/U7 are blocked.

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
"blazing on consumer hardware" target.

| # | Item | Crate | Status | Notes |
|---|------|-------|--------|-------|
| C1 | Critical-path #4 — MoE-aware CPU forward pass (non-Metal fallback) | larql-inference | not started | **Promoted from critical path #4 to P0 of this track**. Currently CPU MoE has no production path; everything routes through Metal or grid. Without C1, CPU track has no decode loop to measure. **Stays P0 under ADR-019** because V1/V2 cross-arch sweep on 26B-A4B requires CPU MoE. |
| C2 | WalkFfn as **primary** CPU decode path (not research-only mode) | larql-inference | partial — exists, not productionised | Currently `WeightFfn::forward` is the dense fallback; switch the default for vindex-loaded models to `WalkFfn`. Bench numbers required. Cross-references CPU MoE work in C1. |
| C3 | Hash-routed FFN (exp 27 → product) — top-k mask on gate scores, ~5× bandwidth reduction at KL ≤0.05 | larql-inference + larql-vindex | research only | Exp 27 proved Gemma 3 4B L0 at top-2048/d_ffn (20% mask) gives KL=0.030. Productise as a vindex flag (`hash_route_topk: usize`) + per-layer threshold table; mask threshold per-layer (L0 vs late layers behave differently). Device-agnostic. |
| C4 | FP4 productisation (exp 26 → product) — native FP4 quantisation tier (`Q4_K → FP4`) | larql-vindex + larql-compute | research only | Exp 26 proved gemma3-4b-f16.vindex is 99.83% FP4-friendly per-feature without QAT (down is the tail at 99.65%). Add `Quantisation::FP4` variant; CPU-first kernel; Metal twin in larql-compute. ~2× memory shrink vs Q4_K. |
| C5 | mmap'd vindex with lazy disk-resident edges — only resident pages for active edges per token | larql-vindex + larql-inference | not started | Today vindex loads whole layer tensors into RAM. For models bigger than RAM, mmap the vindex file and let the OS page in only the gate-KNN-resolved edges. Pairs with C2 and C3: when only 20% of edges fire, only those pages are read. |
| C6 | AMX / AVX-512 / Apple AMX kernels for residual compute | larql-compute (CPU side) | partial — Accelerate BLAS, AMX through it | Current CPU path uses ndarray + Accelerate; promote to direct AMX intrinsics on Apple Silicon, AVX-512 on x86. Compute that *does* happen needs to be as good as it gets, since bandwidth is what's left over. |
| C7 | KV compression as **default** for long context (Apollo / MarkovRS / UnlimitedContext / TurboQuant) | larql-inference | engines reachable on `run`/`walk` (CPU) via `--engine` / `LARQL_KV_ENGINE`; default still `standard` (production K/V cache); GPU performance on opt-in engines requires AsyncComputeBackend (see U-series below) | Unification spec at [`kv-engine-unification.md`](crates/larql-inference/docs/specs/kv-engine-unification.md) — all 7 steps landed. MarkovRS / UnlimitedContext / TurboQuant opt-in via `--engine` (CPU-correct, Metal works via CPU-fallback delegation). Apollo bench-only. Promoting any of these as default for long context requires `AsyncComputeBackend` Step A6 (engine-specific Metal shaders) to land — see U5 below. Server engine wiring also blocked on AsyncComputeBackend (U7); without it the server would silently downgrade Metal decode to CPU. |
| C8 | BR4 (Boundary refs Phase 4 — bounded KV eviction + durability-first capture) | larql-server + larql-inference | not started | See § "P1 — Boundary refs and cold-context storage" below. The CPU track makes BR4 load-bearing because long-context CPU inference can't keep raw KV in RAM. |
| C9 | Distributed-load-balancing for "model spans 4 consumer machines" | larql-router + larql-server | shipped (grid + rebalancer) | **DEMOTED to P2 per ADR-019 (2026-05-09)** — substantial production-engineering with no current experiment requiring multi-machine. Single-shard grid (already shipped) sufficient for substrate. Re-promote if a specific experiment needs multi-machine. |
| C10 | CPU bench harness — `larql bench --cpu` with per-stage breakdown matched against `llama.cpp -ngl 0` | larql-cli + bench/ | started — `larql bench --cpu --output json` is now a first-class shorthand; Gemma 3 4B Q4K vs llama.cpp Q4_K_M CPU baseline is recorded in `bench/baselines/cpu/`; current gap **1.50× decode** (27.6 vs 41.4 tok/s on Gemma 3 4B Q4K via `StandardEngine`), **55× prefill** (kernel-level, see C12) | CPU-track baseline-credibility threshold can't be enforced without this. First acceptance test: Gemma 3 4B Q4_K on M3 Max CPU vs quant-matched `llama.cpp -ngl 0`. Then Llama 2 7B + Mistral 7B for cross-arch CPU. Major improvement since the 2026-05-15 baseline (2.78× → 1.50×) — see `bench/baselines/cpu/COMPARISON.md` and `DIAGNOSIS-2026-05-16-thread-scaling.md`. |
| C11 | Architecture rule enforcement — CI check for "no GPU-only paths in core" | scripts/ + crate boundaries | not started | Static check: anything in `larql-inference` core (not `metal/`, not `cpu/`) must compile and pass tests with Metal feature off. Prevents the dual-track from drifting into Metal-locked code. |
| C12 | Q4K decode kernel — hand-asm aarch64 to close the 1.50× gap to llama.cpp | larql-compute | **specced 2026-05-16, not started** | Per-core gap is **1.73× constant across thread counts** (5.7 vs 9.88 tok/s single-threaded on M3 Max). Same algorithm (Q4K × Q8K with NEON SDOT), same `vdotq_s32` instructions — llama.cpp uses hand-written inline aarch64 asm with two-super-block interleaving + explicit prefetch hints, we use Rust intrinsics lowered by LLVM. Effective bandwidth: ~63 GB/s vs ~95 GB/s. **Per-stage profile (`LARQL_INSTRUMENT_UNLIMITED=1` on Gemma 3 4B 8-thread, 2026-05-16): FFN 26.0 ms (74%) + Attention 9.3-11.0 ms (26%, grows with ctx) + Embed ~0 ms = 35-37 ms/step.** FFN matvec on gate/up/down (4608 × 9216) is the dominant target; attention matvec is the same kernel on smaller matrices. The 38 tok/s asymptote (FFN-alone) sets the floor any engine can reach on the current kernel — Standard and UnlimitedContext both hit 26.6 tok/s on Gemma 3 4B Q4K CPU (8-thread, 40-token prompt, 64 decode tokens) because both route through the same `attention_decode_step_native` + `ffn_decode_step_native` hot paths. Phases: (1) hand-asm Q4K matvec on the FFN tile shapes (gate/up/down) — closes ~95% of the gap, 1-2 weeks; (2) pre-formatted block layout — 1.1-1.2× on top, 3-5 days; (3) Q6K kernel for `ffn_down` — 1.05×, 2-3 days; (4) reduce rayon launch overhead — 1.04×, 2-3 days. Acceptance: ≥9.5 tok/s single-core, ≥39 tok/s 8-thread on Gemma 3 4B Q4K. Spec: [`crates/larql-compute/docs/q4k-decode-kernel.md`](crates/larql-compute/docs/q4k-decode-kernel.md). Per-stage measurement protocol: see "C12 per-stage measurement" below. |

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
  acceptance tier (671B-class), which is the ~40%-confidence stretch.
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
