# kv-cache-benchmark

An inference-memory ladder for the LARQL project. The framing here is not
"compress KV better" — it is that **correctness can be stratified**, and each
stratum admits a radically different state representation. The table below is
the ladder: each rung pairs a storage regime with the specific correctness
target it is obligated to hit.

| # | Strategy | Contract | Breaks at | Memory @ 370K | Compression |
|---|----------|----------|-----------|---------------|-------------|
| 1 | **Standard KV** | bit-exact baseline | — (this *is* the baseline) | 25.8 GB | 1× |
| 2 | **TurboQuant** | top-1 preserved under quantisation noise | tasks that need full-distribution fidelity — temperature sampling, beam search with tight margins, RLHF reward scoring | 6.6 GB | ~4× |
| 3 | **Markov RS (W=512)** | bit-exact via cold-replay reconstruction | architectures that carry state outside the residual stream — explicit memory modules, retrieval-augmented decoders, external attention sinks | ~193 MB | ~134× |
| 4 | **Tier 2 — `UnlimitedContextEngine`** | bit-exact within-window (per-window replay) | anything outside the replayed window | ~30 MB | ~2,000× |
| 5 | **Tier 3 — `ApolloEngine`** | first-token factual correctness | continuation (lossy — one vector can't uniquely ground suffix text) | ~2.8 MB | ~20,000× |
| 6 | **RS Graph Walk** † | graph-level semantic recall | queries the extracted graphs don't cover | 1.5 MB † | ~17,200× † |

† Row 6 is **projected, not measured**. It requires cracked attention, which is not yet implemented. `GraphWalk::decode()` returns zero vectors as a sentinel — do not pipe it through `run_strategy_benchmark` for fidelity metrics.

Rows 1–5 are measured on Gemma 3 4B via the `real-model` feature; see the **Latest measured run** section below for the most recent trace.

**The `Contract` column is not a total order.** Row 4 (bit-exact within window) and Row 5 (first-token factual) make incomparable promises: Row 4 reproduces the window's logits exactly but says nothing outside it; Row 5 lands the right first token over arbitrary context but drifts on continuation. Neither dominates the other — they answer different questions. Pick by which contract your workload actually needs.

### The correctness ladder

The rungs are not interchangeable — they answer different questions:

1. **Bit-exact continuation** (Standard KV) — identical logits, identical decode.
2. **Top-1 preserved under quantisation noise** (TurboQuant) — 4-bit Lloyd-Max perturbs the hidden state (cos ≈ 0.99 on decoded residuals) but the argmax survives; KL is small, not zero.
3. **Bit-exact via cold-replay reconstruction** (Markov RS) — same next-token distribution under the implemented cold-replay path on the benchmarked setup.
4. **Bit-exact within a bounded replay window** (Tier 2) — K/V checkpoint + token archive reproduces the window exactly; behaviour outside the window is not claimed.
5. **First-token factual correctness** (Tier 3) — the right fact lands; continuation is lossy because a single boundary vector cannot uniquely ground arbitrary suffix text.
6. **Graph-level semantic recall** (Graph Walk, target) — answers recoverable from the extracted graphs; not a literal replay of the original forward pass.

## Implementation status

All engines now live in `larql_inference::engines::kv_engines/`. This crate
re-exports from there; the implementations are no longer duplicated here.

| Strategy | Lives in | End-to-end real | Synthetic |
|---|---|---|---|
| Standard KV | `real_model::kv_capture` | ✓ | ✓ `standard_kv` |
| TurboQuant | `larql_inference::engines::kv_engines::turbo_quant` | ✓ (~95 tok/s Metal) | ✓ |
| Markov RS | `larql_inference::engines::kv_engines::markov_residual` | ✓ (~95 tok/s Metal, bit-perfect) | ✓ |
| UnlimitedContext | `larql_inference::engines::kv_engines::unlimited_context` | ✓ (~94 tok/s Metal) | ✓ |
| ApolloEngine | `larql_inference::engines::kv_engines::apollo` | ✓ (compressed path via `forward_from_layer`) | ✓ |
| Graph Walk | `graph_walk::GraphWalk` (memory accounting only) | partial | — |

### Speed (Gemma 3 4B, Metal Q4K, 2026-04-26)

All engines use `prefill_q4k`/`decode_step_q4k` → Metal `decode_token` pipeline:

```
Backend                           prefill    ms/tok   tok/s
larql-metal (standard)            58ms       13ms     76.7
markov-rs (Q4K Metal)             294ms      10.5ms   95.2
unlimited-context (Q4K Metal)     208ms      10.6ms   94.3
turbo-quant 4-bit (Q4K Metal)     203ms      10.6ms   94.8
turbo-quant 3-bit (Q4K Metal)     201ms      10.6ms   94.3
```

Apollo runs on the CPU compressed path (4 layers via `forward_from_layer`).

### Criterion benchmarks

```
cargo bench -p kv-cache-benchmark --bench kv_strategies
```

38 benchmarks across 7 groups: encode, decode, wht, memory_sweep, accuracy, engine_kind, engine_memory.

### Latest measured run — 2026-04-23, Gemma 3 4B (q4k vindex)

Reproduced locally via `cargo run -p kv-cache-benchmark --example real_model_bench --features real-model --release -- google/gemma-3-4b-it <vindex-path>`. Full trace in `results/real_model.json`.

Top-1 match vs Standard KV baseline across five factual prompts (France → Paris, Mozart → Salzburg, Japan currency → the, water freezes → ' ', largest planet → Jupiter):

| Strategy                       | Top-1 matches | Hidden cosine vs baseline | Per-prompt memory (Jupiter prompt, 9 toks) |
|--------------------------------|:-:|:-:|:-:|
| Standard KV (FP16)             | 5 / 5 | 1.000 (baseline) | 1.3 MB  |
| TurboQuant 4-bit               | 5 / 5 | cos ≈ 0.9915 (quantisation noise, top-1 preserved) | 323 KB (4.0×)  |
| Markov RS (W=512 cap)          | 5 / 5 | **1.000000** (bit-perfect) | 3.1 MB populated (floor; rises to ~193 MB ceiling as context fills W) |
| RS Graph Walk (MarkovFallback) | 5 / 5 | 1.000 (fallback to Markov path) | 36 B per-conversation |

KL is not reported separately because `cos = 1.000000` on the hidden state forces logit identity (final_norm + lm_head are deterministic), which is strictly stronger than `KL = 0.0` on the output distribution. Wall-clock per prompt ~560–1020 ms; Markov RS is slightly faster than Standard KV at these short contexts because it doesn't carry a growing K/V tensor. Graph Walk stays in the `MarkovFallback` tier for all five prompts — the Tier A / Tier B template-cache and dynamic-graph paths are not yet wired, so these results demonstrate the fallback-correctness floor, not the target-architecture speed.

**What fraction of real queries graduate out of fallback?** Unmeasured — that is the open question for Row 6. The current `real_model::graph_walk_layer` FFN walk covers the 348K features in the vindex (factual retrieval over Gemma 3 4B's training distribution: capitals, elements, dates, named entities, well-known relations). Queries that decompose to one of those features should move to Tier A (cached template) or Tier B (dynamic graph) once the routing path is wired; everything else — free-form generation, multi-hop reasoning, novel domain compositions — stays in the Markov RS fallback. Quantifying coverage on a realistic query distribution is a future work item; don't read the five factual prompts above as a coverage claim, they're a correctness floor.

**Compression numbers in the headline table above**:

- Rows 1–3: measured end-to-end on Gemma 3 4B via the `real-model` feature. **Row 3 validation:** the most recent run (2026-04-23, trace above) measured `Markov RS hidden cosine vs baseline = 1.000000` on all five factual prompts, re-confirming the bit-perfect claim. The full `#[ignore]`'d suite (`tests/test_real_model.rs::test_accuracy_markov_rs_bitperfect` and adjacent) still requires weights + vindex on disk and does not run in default `cargo test`; the example above is the lightweight reproduction.
- Row 4 (Tier 2): measured via `tests/test_unlimited_context::test_compression_ratio`. Within-window K,V is bit-exact via model-forward replay from the prior window's per-layer K,V checkpoint.
- Row 5 (Tier 3): the Rust `ApolloEngine` loads `apollo-demo/apollo11_store/` end-to-end (2.13 MB in RAM) and runs the full pipeline: tf-idf-lite routing → positional-proximity entry retrieval → forward-with-injection (answer-tokens-only, step-0 only) at L30 coefficient 10× → greedy decode.

  Four entry points, all measured end-to-end on Gemma 3 4B. The axes are orthogonal: **decode length** (single-token top-1 vs iterative decode) × **context representation** (raw window tokens vs 10 KB boundary vector):

  |                      | Uncompressed (raw window tokens, ~519 tok) | Compressed (10 KB boundary + query, ~9 tok) |
  |----------------------|-----|-----|
  | **Single-token top-1** (`query_greedy*`) | `" John"` @ logit 24.0 | `" John"` @ logit 31.1 |
  | **Iterative decode** (`query_generate*`) | **`" John Coyle.\n\n02 05 5"`** — correct answer, then drifts into the transcript's time-stamp structure | `" John and Mary.\n\nJohn and Mary won the porridge eating"` — first token correct (injection lands), hallucinates "Mary" because the single-vector boundary can't uniquely identify the "Coyle" continuation |

  **The gap between compressed and uncompressed outputs is exactly the fidelity/compression trade-off the four-rung ladder predicts**: uncompressed forwards have the raw window text to ground on, compressed forwards rely on the ~10 KB boundary (variant-ii-class) + injection — which lands the first-token fact via amplification but can't carry detailed continuation info. A Tier 2-style per-layer K/V checkpoint (~139 KB per window) would reproduce "Coyle" exactly at the cost of ~14× more storage per boundary.

  Python reference: `chuk-mlx/src/chuk_lazarus/inference/context/research/unlimited_engine.py` + `vec_inject/`.
- Row 6: see the † footnote under the headline ladder. The 1.5 MB figure is projected conditional on the cracked-attention path landing; FFN graph walk is proven (`real_model::graph_walk_layer`), but queries outside the factual graph currently fall back to Markov RS.

## Quick start

```bash
# Phase 1: Synthetic benchmark (no model needed)
cargo test -p kv-cache-benchmark

# Run the shader/CPU benchmark
cargo run -p kv-cache-benchmark --example shader_bench

# Run the multi-turn simulation
cargo run -p kv-cache-benchmark --example multi_turn_demo

# Criterion benchmarks
cargo bench -p kv-cache-benchmark

# Phase 2: Real model (requires Gemma 3-4B weights + a vindex on disk)
cargo run -p kv-cache-benchmark --example real_model_bench --features real-model --release -- \
    google/gemma-3-4b-it <path-to-vindex-dir>
```

## Architecture

```
kv-cache-benchmark/
  src/
    lib.rs              KvStrategy trait, run_strategy_benchmark()
    standard_kv.rs      Row 1: raw FP16 encode/decode
    turboquant/         Row 2: WHT + Lloyd-Max + bit packing
    markov_residual/    Row 3: bounded window + cold tier
    graph_walk/         Row 6: routing table + vindex lookup (projected; memory accounting only, no KvStrategy impl)
    benchmark.rs        Sweep runner, multi-turn sim, table formatter
    shader_bench.rs     CPU/Metal operation benchmarks
    metrics.rs          MSE, cosine, inner product error
    model_config.rs     Gemma 4B / Llama 8B / 70B dimensions
    real_model/         Phase 2: wired into larql-inference (feature-gated)
    unlimited_context/  Row 4 (Tier 2): per-window K,V checkpoint + model-forward replay
    apollo/             Row 5 (Tier 3): single-vector boundary + vec_inject
  tests/                unit + integration tests
  benches/              Criterion benchmarks
  examples/             Demo runners
  docs/                 Benchmark spec v3
```

## Strategies in detail

### Standard KV (baseline)
What llama.cpp, vLLM, and MLX use. FP16 keys and values stored per-token,
per-layer, per-head. Memory grows linearly with context length.

### TurboQuant (Google, ICLR 2026)
Compresses KV cache to 3-4 bits per coordinate using Walsh-Hadamard rotation
followed by Lloyd-Max scalar quantization. 4-6× compression at the Shannon
limit. Still grows O(context_length).

### Markov RS (W=512)
**Markov RS turns the KV cache from the memory into a view of the memory.**
The residual stream is the source of truth; K/V becomes a computed view —
recomputed at decode time from the residual state, not persisted. In
database terms: K/V is no longer a table, it's a query over the residual
table. Stores a bounded hot window of up to 512 residuals per layer (f32)
plus cold-tier token IDs (4 bytes each). At full
occupancy the hot window dominates: 512 × 34 layers × 2560 dim × 4 bytes
≈ 178 MB (the ~193 MB figure includes bookkeeping + checkpoints). Cold tier
adds 4 bytes/token — effectively flat vs the 5,120 bytes/token of hot
residual state (a 1,280× ratio), but not literally zero. Net effect: memory
is bounded by the W=512 hot window, with only a small linear tail past it.

**Why short-prompt measurements look smaller than the headline 193 MB:** the
headline is the *allocation ceiling* at full-W occupancy (i.e. once the hot
window is actually holding 512 rows per layer). At the 9-token Jupiter
prompt in the latest-run table above, only 9 rows × 34 layers × 2560 dim ≈
3.1 MB are actually populated — the cap hasn't been reached yet. The floor
rises monotonically until context ≥ W and then plateaus. Use the headline
for steady-state planning; the per-prompt number for short-context
behaviour.

**Correctness claim (precise form):** under the implemented cold-replay
reconstruction path — `[cold_token_ids ‖ hot_residuals]` recomposed before
`recompute_kv` at each decode step — the stored state is sufficient to
reproduce the next-token distribution bit-for-bit on the benchmarked setup
(Gemma 3-4B, KL = 0.0 vs. Standard KV). This is a statement about the
reconstruction path under the benchmarked conditions, not a general claim
that residuals are context-free Markov states across all architectures.

### Boundary residual architecture — delivered by Tier 2 / Tier 3, not a standalone strategy

The "32-token hot window + boundary vector + cold token IDs" idea from
the Python experiments (`unlimited_engine.py`, `rs_generator.py`) is
delivered by two concrete engines in this crate:

- **`unlimited_context::UnlimitedContextEngine`** (Tier 2) — per-window
  K,V checkpoint (174 KB on Gemma 3 4B) + token archive + model-forward
  replay. Bit-exact within-window. Reference: `chuk-mlx/.../unlimited_engine.py`.
- **`apollo::ApolloEngine`** (Tier 3) — single-vector boundary at
  crystal_layer (10 KB per window) + token archive + `vec_inject`
  retrieval index + injection-at-L30 amplification. Task-level correctness
  on queries routable via the injection index. Reference:
  `chuk-mlx/.../vec_inject/` + `apollo-demo/apollo11_store/`.

An earlier synthetic-accounting-only `BoundaryResidual` `KvStrategy` was
removed because its cold-tier `decode` was a placeholder (cold slots
reconstructed as `boundary.clone()`), so fidelity metrics produced by
piping it through `run_strategy_benchmark` were not meaningful.

### RS Graph Walk _(target architecture — not yet fully operational)_
The endgame once attention is cracked. The forward pass would be a walk over
three composed graphs (FFN, attention, residual). Extract the graphs, walk
them directly.

**Current status:**

- FFN graph walk is proven (348K features in vindex, 34 layers, zero accuracy
  loss on factual queries).
- Attention elimination requires cracked attention — not yet implemented.
- Until then, queries outside the factual graph fall back to Markov RS for the
  full forward pass.

Treat this rung as a target architecture, not a delivered system. The 1.5 MB
figure is a projected steady-state footprint under the assumption that the
cracked-attention path lands; it is not a current end-to-end measurement.

**Shared infrastructure:** ~1.5 GB q4k vindex covering 348K FFN features
across 34 layers (gate + down codebooks, feature metadata, embeddings) +
352 KB routing table. This is the same artifact pointed to by
`<vindex-path>` in the Quick Start — one copy per host, amortised across
all conversations, not per-conversation. The economic argument for the
architecture rests on that amortisation: `Graph Walk @ 370K = 1.5 MB` is a
per-conversation number, conditional on the shared 1.5 GB already being
resident.

## Memory scaling

| Metric | Standard KV | TurboQuant 4b | Markov RS (W=512) | Graph Walk |
|--------|------------|---------------|-----------------|------------|
| Memory @ 4K | 285 MB | 74 MB | 193 MB | 16 KB |
| Memory @ 32K | 2.24 GB | 580 MB | 193 MB | 130 KB |
| Memory @ 370K | 25.8 GB | 6.6 GB | 193 MB | 1.5 MB |
| Grows O(N)? | yes | yes | cold only (+4B/tok) | cold only (+4B/tok) ‡ |
| Hot window fixed? | no | no | ~178 MB | — ‡ |

‡ Graph Walk has no hot window in the K/V sense — the per-conversation growth is cold-tier token IDs (4 B each) plus a small per-step routing trace. Shared infrastructure (~1.5 GB vindex + 352 KB routing table) is reported separately and is not per-conversation.

## Compute per token

| Operation | Standard KV | TurboQuant | Markov RS | Graph Walk |
|-----------|------------|------------|-----------|------------|
| Attention matmul | 34 layers | 34 layers | window only | **ELIMINATED** |
| FFN matmul | 34 layers | 34 layers | 34 layers | **ELIMINATED** |
| Logits matmul | 1× | 1× | 1× | **ELIMINATED** |
| KV cache write | 34L | 34L + quant | none | none |
| Graph lookup | none | none | none | 3 per hop |

**Key insight: this row is the punchline.** The rungs above trade memory
for fidelity. This one trades the forward pass itself for graph lookups.

"3 per hop" means: for a factual query that hits the FFN graph, the decode
step is `gate KNN → feature → down KNN → token` — three keyed lookups,
no matmul. Empirically the FFN graph walk has been validated end-to-end on
factual queries (348K features in vindex, 34 layers, zero accuracy loss
vs the forward pass). The remaining work for the full target architecture
is the **attention** graph: the projected "one lookup per graph traversed"
per-decode figure is conditional on cracked attention landing and a
routing-table lookup taking constant time. Until then, queries outside the
factual graph fall back to Markov RS for the full forward pass (the
`MarkovFallback` tier you see in the latest-run table).

Markov RS (row 3 above) still runs the full 34-layer FFN but replaces K/V
matmuls with residual recompute, so it's bounded in memory but not in
compute. Graph Walk in the target configuration is bounded in *both*.

## Feature flags

- Default: synthetic benchmark only (zero LARQL dependencies)
- `real-model`: enables Phase 2 integration with larql-inference, larql-vindex, etc.

## Spec

Full benchmark specification: [docs/kv-cache-benchmark-spec-v3.md](docs/kv-cache-benchmark-spec-v3.md)
