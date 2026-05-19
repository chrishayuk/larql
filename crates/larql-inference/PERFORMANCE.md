# Performance — larql-inference

Machine: M3 Max, macOS. Gemma 3 4B (34 layers, hidden=2560, vocab=262K).

## CPU headline (2026-05-15)

`larql bench output/gemma3-4b-q4k-v2.vindex --cpu --tokens 16 --warmup 1 --profile`:

```
Backend       prefill    ms/tok    tok/s   steps
larql-cpu     2517 ms    67-70 ms  14.5-14.9 15
llama.cpp     47 ms      24.2 ms   41.37     16  (Q4_K_M, BLAS, ngl=0, 12 threads)
```

**Decode 2.78× behind llama.cpp; prefill 55× behind.** ~40× over the
pre-branch 0.36 tok/s baseline. Detailed side-by-side:
`bench/baselines/cpu/COMPARISON.md`.

Per-step CPU decode breakdown (post-branch):

| Stage | ms/tok | % | What runs |
|---|---:|---:|---|
| CPU fwd | 58-70 | 85-88% | 33 layers × 6 Q4_K matvec (Q/K/V/O/gate/up) + 1 Q6_K matvec (down). NEON inner with 4-way f32x4 accumulators; rayon-parallel via `par_chunks_mut(32)` across rows; K+V and gate+up share inputs via `q4k_dual_matvec` |
| LM head | 9-10 | 12-15% | Q4_K matvec against the vindex's synthesised `lm_head_q4` view (262K vocab); same NEON kernel |
| **Total** | **67-78** | **100%** | **= 14.5-14.9 tok/s** |

### Six CPU optimisations on this branch

| Change | Decode (tok/s) | Δ vs prior | Δ vs baseline |
|---|---:|---:|---:|
| Baseline (legacy O(N²) per-step path)  | 0.36  | —      | 1.0× |
| + KV-cached decode (`predict_q4k_prefill` + `predict_q4k_decode_step`) | ~1.5 | 4.2× | 4.2× |
| + Direct Q4_K matvec, no per-step dequant (`predict_q4k_decode_step_direct`) | 2.6 | 1.7× | 7.2× |
| + Row-parallel f32 lm_head sgemv (`parallel_lm_head_logits`) | 5.4 | 2.1× | 15× |
| + NEON Q4_K / Q6_K / f32_dot kernels | 9.9 | 1.8× | 28× |
| + Q4_K lm_head (synth from f16 embed, `logits_to_predictions_q4_lm_head`) | 12.6 | 1.27× | 35× |
| + 4-way acc NEON + fused gate+up / K+V dual-matvec | 13.1 | 1.04× | 36× |
| + `par_chunks_mut(32)` outer rayon (q4k_matvec + dual + q6k_matvec) | 14.5–14.9 | 1.10× | 40× |

The fused gate+up and K+V dual-matvec changes are bit-exact w.r.t. two
sequential calls but didn't move the bench needle — both
`h_norm` (10 KB) and `h_in_post_norm` (10 KB) fit in L1 across separate
matvec calls, so the predicted re-stream saving wasn't real. Kept the
code (one fewer Vec alloc + one fewer rayon launch per pair) but it was
a wrong hypothesis.

### Where the 2.78× decode gap to llama.cpp lives

Effective Q4_K weight read bandwidth on M3 Max: **~33 GB/s** for us,
**~95 GB/s** for llama.cpp. Both well below LPDDR5 peak (~400 GB/s).
The 2.78× ratio matches the bandwidth ratio almost exactly.

Closing it is hand-tuned kernel work:
1. **Hand-rolled aarch64 asm** matching llama.cpp's per-cycle byte
   throughput — they hit ~95 GB/s effective, we top out at ~33 GB/s
   with NEON intrinsics + chunked rayon.
2. **Multi-superblock interleaving in NEON Q4_K** — process 2+
   super-blocks of one row concurrently to hide M3's load latency
   between successive Q4_K block reads. *Tried 2026-05-15 with a
   split-accumulator outer unroll; null result. LLVM already
   schedules the row loop fine. Future: lower-level asm to force the
   exact interleaving.*
3. **Prefetch hints** ahead of the inner FMA loop (`vld1q_u8` to a
   discarded register or inline `prfm pldl1keep` asm).

Previously-tried-and-killed micro-opts:
* Multi-accumulator unroll across super-blocks (null on this branch —
  helper-function inlining already produces the schedule LLVM wants).
* Fused gate+up dual-matvec (bit-exact but no speedup — `h_in` is
  10 KB and stays in L1 across separate matvecs).
* Fused K+V dual-matvec (same reason as gate+up — kept for the
  one-fewer-rayon-dispatch saving).

### Where the 55× prefill gap lives

Prefill is still on the legacy `predict_q4k_prefill` path:
dequantise each layer's Q/K/V/O/gate/up/down to f32 once per layer
(~75 ms × 33 layers ≈ 2.5 s), then run per-position attention + FFN
over the prompt. llama.cpp does batched gemm: weights read once,
applied to all 5 prompt positions in a single Accelerate sgemm call.

The fix is to route prefill through a CPU `q4k_matmul` (multi-row
matvec). Backend trait declares it; CPU backend returns `None` today.
A working implementation would amortise weight reads across `seq_len`
positions — estimated prefill ~300 ms (vs 2 593 ms now), gap drops from
55× to ~6×. The remaining 6× is the same kernel bandwidth gap as
decode.

**Caveat** — see `project_prefill_matmul_falsified` memory. That entry
documents a failed Metal-side experiment with the same idea, but Metal
already runs per-position Q4_K matvec without dequant; the bottleneck
there was different. CPU still pays full dequant cost today, so the
matmul-amortisation argument applies.

## Real-vindex headline (2026-05-02)

`larql bench output/gemma3-4b-q4k-v2.vindex --tokens 30 --warmup 8`:

```
Backend       prefill    ms/tok    tok/s    steps
larql-metal   ~67 ms    13.5–13.9   72–75     29
Ollama        ~10 ms/tok = 96–104 tok/s (reference, same model)
```

Per-stage breakdown of one decode step (with all five fusions default-on):

| Stage | ms/tok | % | What runs |
|---|---:|---:|---|
| GPU forward | 11.5–12.0 | 79% | `dispatch_full_pipeline` per-token Metal compute (Q4_K matvecs, fused QKV proj + input_norm, fused QK_norm + RoPE, fused KV append + attend, fused post_attn norm + residual + store, fused gate + up, fused GEGLU + down, fused post_ffn norm + residual_add) |
| LM head    | 2.9–3.0 | 20% | Q4 matvec on `lm_head_q4.bin` + GPU argmax reduction (256K vocab) — stride-32 reduction tree (lm_head v5) |
| Embed / final norm / detok / sample / EOS |  0.05 | <1% | Per-step CPU work outside the Metal compute path |
| **Total** | **13.5–14.0** | **100%** | **= 72–75 tok/s** |

### Shipped Metal dispatch fusions (2026-05-01 → 2026-05-02)

Five default-on fusions; `LARQL_FUSED_*=0` opt-out flags wired for diagnostics.
Cumulative GPU forward saving ~0.99 ms vs. unfused baseline (10.45 ms → 9.46 ms
isolated kernel time; end-to-end 71.5 → 72–75 tok/s).

| Fusion | Δ GPU | Mechanic |
|---|---:|---|
| `qk_norm_rope_fused` | -0.10 ms | One TG/head does RMS-norm + RoPE in one pass; replaces qk_norm_qk + rope_at_pos_batched_qk |
| `residual_norm_store` (always-on) | -0.38 ms | Single 1-TG kernel writes both `ffn_norm_out` and `h_post_attn` |
| `post_attn_residual_norm_store_pipeline` | -0.43 ms | Triple-fused post_attn norm + residual + h_post_attn store + ffn_norm; replaces a 3-dispatch chain on the `has_post_norms` path |
| `post_ffn_norm_residual_add_pipeline` | -0.78 ms | 1-TG kernel: RMS over down_out + residual sum into `new_h` (next-layer input) in one pass |
| `kv_append_attend_fused_pipeline` | -0.99 ms | Per-Q-head TG cooperatively writes new K/V row at pos, `mem_device` barrier, then standard attention |

### Failed fusion attempt — `attn_fused` (kept opt-in)

Merging `qk_norm_rope_fused` (12 TGs) + `kv_append_attend_fused` (8 TGs) into one
kernel regressed 74 → 64 tok/s (-1.45 ms). Diagnosis: collapsing to 8 TGs lost
parallelism that 12 TGs had given the standalone kernel; dispatch-overhead saving
(~30 µs) was dwarfed by the parallelism cost. Kernel registered as opt-in
`LARQL_FUSED_ATTN=1` for any future multi-TG-per-head retry that preserves
parallelism. **Lesson**: dispatch fusions only win when they don't reduce TG
count for an already parallelism-bound stage. See `crates/larql-inference/ROADMAP.md`.

### Headline-vs-reality reading guide

The number you measure depends on **how the run is timed**:

| Run shape | tok/s | Why |
|---|---:|---|
| `larql bench --warmup 8 --tokens 30` (steady-state, post-fusion) | **72–75** | Drops the 54-ms cold token, averages over enough steps for variance to wash out. **Use this for any speed comparison.** Variance is ~3 tok/s between cold/warm GPU; multi-run average is the honest number. |
| Short bench (`--max-tokens 20`, no warmup) | ~67 | Cold token 1 (54 ms) dragged into the average; the per-token decode after warmup is still ~13 ms (= 75 tok/s) but the average reports higher. |
| Compute `PERFORMANCE.md` 78.7 tok/s claim | 78.7 | Pre-correctness-fix snapshot, on the buggy Q4_K → Q4_KF dispatch path. **Not a real reference** — see `project_metal_decode_81_was_buggy` memory. |

## LM head path matters

Four lm_head paths exist; which one fires is determined by what the loader finds:

| Path | When it fires | ms/tok | Note |
|---|---|---:|---|
| **Q4 matvec (Metal)** | `lm_head_q4.bin` present + `vocab_size > 0` | **~1.9** | Production fast path. Saves the 1MB readback + 262K-element CPU sort by computing argmax on the GPU. |
| f16 gemv (tied embed) | Tied-embedding model + embeddings adopted as lm_head | ~3-5 | Half the bandwidth of f32, 2× of Q4. |
| f32 KNN (`lm_head.bin`) | Separate untied lm_head shipped at f32 | ~2 | Untied models only. |
| f32 BLAS gemv (slow) | None of the above — falls back to `weights.lm_head` full gemv | ~8 | What you hit when `vocab_size = 0` silently bails the Q4 path. |

`larql diag <vindex>` prints which path will fire and surfaces the silent-slowdown classes (stale 148-byte stride, `vocab_size = 0`) at a glance.

## Production Benchmark: "The capital of France is"

Real vindex (`output/gemma3-4b-v2.vindex`), 6-token prompt.

| Strategy | Output | Time | tok/s | Notes |
|----------|--------|------|-------|-------|
| Dense (baseline) | Paris (80.47%) | 552ms | 1.8 | CPU BLAS, all 34 layers |
| Full pipe (CPU) | Paris | 224ms | 4.5 | Cached L0-12 + WalkFfn L13-33 |
| **Honest (production)** | **Paris (88.41%)** | **203ms** | **4.9** | **Cached L0-12, CPU L13-33, GPU logits** |
| Split cached | Paris (88.41%) | 3ms | 311 | Pre-computed residuals (one-time build) |
| Prefill logits | Paris (88.41%) | 4.0ms | — | Logits only (from prefilled hidden state) |
| Ollama | Paris | 144ms + 8.5ms/tok | 117 | Full GPU pipeline |

## Honest Path Breakdown

```
predict_honest("The capital of France is"):
  Phase 0 (L0-12): CachedLayerGraph          ~5ms  (template-fixed, 0.999 cosine)
  Phase 1 (L13-33): CPU attention + WalkFfn  ~195ms (GELU-tanh activation, post-norms)
  Phase 2: GPU logits KNN                     ~4ms  (vindex lm_head Q4 via Metal)
  Total:                                     ~203ms = 4.9 tok/s
```

## GPU Decode Path

### Synthetic (compare_ollama, random weights, 2026-04-09)

| Engine | ms/tok | tok/s | Notes |
|--------|--------|-------|-------|
| **LARQL Q4_KF decode (34L, KV)** | **8.5ms** | **117** | **Synthetic ceiling** |
| LARQL Q4_K decode (21L, KV) | 11.6ms | 86 | |
| LARQL Q8 decode (21L, KV) | 19.3ms | 52 | |
| Ollama (34L) | 10.3ms | 98 | |
| **vs Ollama (synthetic)** | **0.83x** | — | **17% faster** |

### Real vindex (larql bench, gemma3-4b-q4k-v2.vindex, 2026-05-02)

Prompt: "The capital of France is" (5 tokens), 30 tok, 8 warmup, all
five Metal dispatch fusions default-on.

| Engine | prefill | ms/tok | tok/s | Notes |
|--------|---------|--------|-------|-------|
| **LARQL Metal** | **~67ms** | **13.5–13.9ms** | **72–75** | Five default-on fusions; lm_head v5 stride-32 reduction tree |
| Ollama gemma3:4b | ~15ms | ~10ms | ~96–104 | |
| **vs Ollama (real)** | — | ~1.40x slower | — | GPU fwd 79% of decode; lm_head 20% |

Per-stage: embed 0.002ms · GPU fwd 11.5–12.0ms · final_norm 0.006ms · lm_head 2.9–3.0ms · detok 0.04ms

Progress:
- 2026-04-07: 28.0ms / 36 tok/s (34L synthetic) = 2.84x Ollama
- 2026-04-08: 18.3ms / 55 tok/s (34L synthetic) = 1.79x Ollama
- 2026-04-09: 8.5ms / 117 tok/s (34L synthetic) = 0.83x Ollama (synthetic ceiling)
- 2026-04-19: 15.6ms / 64 tok/s (34L real vindex) — lm_head Q4 synthesis, KV cache fix
- 2026-05-01: 13.6ms / 73 tok/s (34L real vindex) — 4 dispatch fusions default-on (qk_norm+rope, residual_norm_store, post_attn_norm, post_ffn_norm)
- 2026-05-01: 13.4ms / 74 tok/s — 5th fusion default-on (kv_append + kv_attend)
- 2026-05-02: 13.5–13.9ms / 72–75 tok/s — `attn_fused` merger attempt regressed and was reverted to opt-in; lm_head v5 stride-32 holds. Path-to-80 lever search documented in ROADMAP G-3

## Layer Graph Strategies

| Strategy | What it does | When used |
|----------|-------------|-----------|
| CachedLayerGraph | Returns pre-computed residual | L0-12 (template-fixed) |
| DenseLayerGraph | Matmul attention + pluggable FFN | Baseline/fallback |
| WalkLayerGraph | Dense attention + sparse WalkFfn | CPU walk path |
| PipelinedLayerGraph | CPU attention + Metal Q4 FFN | GPU acceleration |
| PerLayerGraph | Per-layer strategy selection | Adaptive routing |

## Component Breakdown (CPU BLAS, seq=6, Gemma 3 4B, `bench_components`)

| Component | µs/layer | % | Notes |
|-----------|---------|---|-------|
| FFN gate+up (2× BLAS) | 6,008 | 44.5% | Dominant cost |
| FFN down (BLAS) | 3,475 | 25.7% | |
| QKV projection (3× BLAS) | 2,896 | 21.4% | |
| O projection (BLAS) | 789 | 5.8% | |
| Attention (scores+softmax+V) | 143 | 1.1% | Small at seq=6 |
| GEGLU SiLU | 105 | 0.8% | Element-wise |
| RoPE | 56 | 0.4% | |
| RMSNorm (×2) | 30 | 0.2% | |
| Residual add (×2) | 3 | 0.0% | |
| **Layer total** | **13,504** | | |
| **34-layer model** | **513ms** | | **2 tok/s CPU** |

97% of time is BLAS matmul. GPU Q4_K pipeline replaces these: 513ms → 17.5ms (29x speedup).

### Norm comparison

| Norm | µs (seq=6, hidden=2560) | vs RMSNorm |
|------|------------------------|-----------|
| RMSNorm | 14.9µs | baseline |
| LayerNorm | 28.4µs | 1.91x |

### RoPE comparison

| Variant | µs (8 heads) | Notes |
|---------|-------------|-------|
| Full (hd=256) | 56.0µs | Standard |
| Partial 25% (hd=512) | 16.8µs | Gemma 4 global, 3.3x faster |

## Activation Function Support

| Model | Activation | FFN Type | GPU Path | CPU Path |
|-------|-----------|----------|----------|----------|
| Llama 2/3 | SiLU | Gated | geglu_silu | ✅ |
| Gemma 2/3/4 | GELU-tanh | Gated | geglu_gelu_tanh | ✅ |
| Mistral | SiLU | Gated | geglu_silu | ✅ |
| Qwen 2/3 | SiLU | Gated | geglu_silu | ✅ |
| StarCoder2 | GELU-tanh | Standard | gelu_tanh (standalone) | ✅ |
| GPT-2 | GELU | Standard | gelu_tanh (standalone) | ✅ |
| Granite | SiLU | Gated | geglu_silu | ✅ |

## Post-Norm Architecture

Gemma3 uses post-norms (norm after attention/FFN, not before):
- CPU path: fully correct (tested, "Paris" output)
- GPU decode_token: correct (activation + post-norm handled)
- GPU prefill_q4: **not yet correct** for post-norm models → falls to CPU
- See larql-compute ADR-009

## Connection to Compute and Vindex

```
larql-inference orchestrates:
  predict_honest()
    → CachedLayerGraph (pre-computed residuals from vindex)
    → FullPipelineLayer (weights from vindex, format tags from vindex)
    → ComputeBackend.decode_token() (GPU Metal kernels)
    → finalize_logits() (vindex lm_head KNN via backend.q4_matvec)
```

Quantization format flows: vindex Q4_K bytes → FullPipelineLayer.format → compute shader dispatch.

## Cross-Crate Performance Comparison

All measurements on M3 Max, Gemma 3 4B (34 layers, hidden=2560).

| Path | Component | Crate | Time | Notes |
|------|-----------|-------|------|-------|
| **CPU forward** | Matmul (BLAS) | inference | 13.5ms/layer | 97% of layer time |
| **CPU forward** | Attention | inference | 0.14ms/layer | 1.1% — negligible |
| **CPU forward** | RMSNorm + GEGLU + RoPE | inference | 0.19ms/layer | 1.4% — element-wise |
| **GPU decode** | Q4_K QKV (fused) | compute | 0.044ms/layer | 6.3x faster than Ollama's layer |
| **GPU decode** | Q4 FFN (gate+up+geglu+down) | compute | 0.38ms/layer | 36% of GPU time |
| **GPU decode** | KV cache attend | compute | 0.31ms/layer | 29% of GPU time |
| **GPU decode** | Norms | compute | 0.16ms/layer | Actual GPU compute |
| **Vindex** | Gate KNN (f32 BLAS) | vindex | 3.0ms/layer | Production dims |
| **Vindex** | Gate KNN (Q4 CPU) | vindex | 1.0ms/layer | 3x faster |
| **Vindex** | Gate KNN (Q4 Metal) | vindex | 0.5ms/layer | 6x faster |
| **Vindex** | Walk (14 layers) | vindex | 14ms | Mmap zero-copy |
| **Ollama** | Full layer | external | 0.30ms/layer | Metal GPU, merged dispatches |

## Sampling Overhead (2026-04-26)

Per-call cost of `Sampler::sample` over realistic vocab sizes. Measured
1000 iterations after 50 warmup, M3 Max release build. Reference: Metal
Q4K decode budget ≈ 10ms/tok = 10,000 µs.

### Sparse top-K path — `sample_from_topk` (production hot path)

`generate_with_sampling` requests `K=5` for greedy or `K=64` for sampling
from the LM-head KNN, then calls `sample_from_topk`. This is the only
sampling path that runs per generated token in the inference loop.

| Config | Hits | µs/call | % of decode budget |
|--------|-----:|--------:|-------------------:|
| greedy | 5 | <0.01 | 0.00% |
| temperature=0.8 | 64 | 0.28 | 0.003% |
| temperature=1.0 + top_p=0.9 | 64 | 1.67 | 0.017% |
| temperature=1.0 + top_k=40 | 64 | 0.63 | 0.006% |

Sparse-path sampling is effectively free — well below the per-step decode
budget across every config. Switching from greedy to non-greedy moves the
needle on tok/s by less than 0.02%.

### Full-vocab path — `sample` (reserved for OpenAI-API logprobs)

Sampling from a dense logit vector. Not on the inference hot path today
— used by the planned OpenAI-compatible HTTP API for `logprobs` and
likelihood-class evals (HellaSwag, MMLU, ARC).

| Config | Vocab=32K | Vocab=128K | Vocab=256K |
|--------|----------:|-----------:|-----------:|
| greedy | 181 µs | 748 µs | 1.5 ms |
| temperature=0.8 | 134 µs | 572 µs | 1.2 ms |
| temperature=1.0 + top_p=0.9 | 2.5 ms | 5.4 ms | 8.0 ms |
| temperature=1.0 + top_k=40 | 104 µs | 423 µs | 820 µs |

The top-p path is ~10× slower than the others at 256K vocab — it does a
full sort + HashSet membership rather than a partial nth-element. Not
hot-path-relevant today; revisit if/when full-vocab sampling moves to
the decode loop.

Reproduce with `cargo run --release -p larql-inference --example bench_sampling`.
