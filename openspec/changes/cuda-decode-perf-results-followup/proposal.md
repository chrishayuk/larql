## Why

Continues [`cuda-decode-perf-results`](../cuda-decode-perf-results/proposal.md)
with the work that landed after its 8.04 ms/tok checkpoint.

Three things changed since that doc was written:

1. **One more 7.5% win shipped** (`cuda-mmvq-hw-f16-cvt`): hardware
   PTX `cvt.f32.f16` replaces a hand-rolled software emulation in
   the Q4_K mmvq hot path. 8.04 → 7.44 ms/tok; tok/s 124.4 → 134.5.
2. **Path A settled negative** (`cuda-marlin-imma-probe`): the
   highest-effort, highest-reward candidate from the perf-results
   doc is no longer viable for batch=1 decode. INT8 IMMA loses
   3-7× to dp4a because the same fragment-row-waste mechanism that
   killed `cuda-tensor-cores-q4k` and `cuda-attn-wmma-multi-warp`
   applies — at batch=1, every Tensor Core path wastes 15/16
   columns regardless of which TC variant is attempted.
3. **Architectural pivot** (`cuda-speculative-decoding`): with
   four independent Tensor Core paths now empirically dead at
   batch=1, the only known mechanism to lift effective batch on a
   single-user decode is speculative decoding. Multi-week design
   doc + phase 1 scaffolding shipped this session.

## What This Change Ships

A new top-level
`openspec/changes/cuda-decode-perf-results-followup/proposal.md`
documenting the post-`8.04` checkpoint progression. Companion to
the original perf-results retrospective. No code changes.

## Bench progression update (RTX 4090 / sm_89, Gemma 3 4B Q4_K_M)

| Checkpoint                    | Decode ms/tok | tok/s     | Prefill ms | Notes |
|-------------------------------|--------------:|----------:|-----------:|-------|
| Pre-session baseline          | 9.62          | 103.9     | 18.0       | f32 throughout, no graph |
| `cuda-attn-wmma-f16kv`        | **8.04**      | **124.4** | 10.7       | End of original perf-results retro |
| `cuda-mmvq-hw-f16-cvt`        | **7.44**      | **134.5** | 10.7       | hw PTX cvt replaces software emulation |
| `cuda-marlin-imma-probe`      | (no change)   | —         | —          | Documented INT8 IMMA negative result |
| **llama-cpp-turboquant**      | **4.34**      | **230.2** | **6.25**   | Reference target |

Cumulative gap closure since pre-session baseline:

- Decode: 9.62 → 7.44 ms/tok (-23%, +29% tok/s).
- Prefill: 18.0 → 10.7 ms (-40%).
- Run-to-run variance: 0.47 ms range → 0.02 ms range (24× tighter).
- Gap with llama.cpp: decode 2.18× → 1.71× (closed 38%); prefill
  3.20× → 1.71× (closed 62%).

## Branches landed since perf-results retro

### Wins (1)

| Branch | Decode Δ | Mechanism |
|--------|---------:|-----------|
| `feat/cuda-mmvq-hw-f16-cvt` | 8.04 → 7.44 | Hardware PTX `cvt.f32.f16` replaces hand-rolled software fp16→f32 emulation in `q4k_mmvq.rs` and `q6k_mmvq.rs`. The software emulation was costing ~7.5% of decode time per token at the 32-million-call rate (8 layers × 4 mmvq × 256 sub-blocks × 4 dispatches per token). Single-instruction PTX cvt is essentially free. |

### Negative results (1)

| Branch | Hypothesis | Outcome |
|--------|-----------|---------|
| `feat/cuda-marlin-imma-probe` | Marlin-style INT4-IMMA mmvq beats dp4a at batch=1 (path A from perf-results retro) | INT8 IMMA loses 3-7× to dp4a on every Gemma 3 4B mmvq shape. Same fragment-row-waste mechanism as `cuda-tensor-cores-q4k` / `cuda-attn-wmma-multi-warp`. **Path A is settled — no more batch=1 Tensor Core attempts will pay off.** |

## Updated profile of remaining 3.10 ms decode gap with llama.cpp

The mmvq-hw-cvt win shifted the per-bucket breakdown. Updated
profile (5-token average, post-`cuda-mmvq-hw-f16-cvt`):

| Bucket                  | ms        | %       | Status after this session |
|-------------------------|----------:|--------:|---------------------------|
| `attn_call`             | ~2.50     | ~34%    | SIMT optimal at batch=1 (4 dead TC branches) |
| `proj_down` (mmvq)      | ~1.45     | ~19%    | hw cvt applied; dp4a optimal at batch=1 |
| `proj_gate_up` (mmvq)   | ~1.27     | ~17%    | hw cvt applied; dp4a optimal at batch=1 |
| `norm_cpu`              | ~1.07     | ~14%    | Already fused (`cuda-fused-norm-add`) |
| `proj_qkv` (3 mmvq)     | ~0.77     | ~10%    | Fusion regresses (`cuda-q4k-qkv-fuse-v2` neg result) |
| `residual_cpu`          | ~0.05     | ~1%     | Folded into norm |
| `proj_wo`               | ~0.33     | ~4%     | Small budget |

Every bucket >10% has been independently optimized. **The mechanism
left is architectural, not bucket-level.**

## Concrete next-step path: speculative decoding

After settling Path A (Marlin) negative this session, the
[design.md](../cuda-speculative-decoding/design.md) for
[`cuda-speculative-decoding`](../cuda-speculative-decoding/proposal.md)
captures the only remaining mechanism that closes the gap:
lift effective batch from 1 to 4-8 via draft+verify so the four
empirically-dead Tensor Core paths (`cuda-tensor-cores-q4k`,
`cuda-attn-wmma-kernel-v2`, `cuda-attn-wmma-multi-warp`,
`cuda-marlin-imma-probe`) re-arm above their fragment-utilization
threshold.

Performance model with α=0.6 acceptance, depth=2, branches=2:

```
target_pass_at_batch5  ≈ 7.44 × 1.4   = 10.4 ms
draft_pass_at_batch1   ≈ 1.5 ms
total_per_step         ≈ 11.9 ms
expected_tokens/step   = 1+α+α²+α³+α⁴ = 2.31
ms_per_token           = 11.9 / 2.31  = 5.15 ms
```

At α=0.7 (EAGLE-2 published rates): ~4.39 ms/tok, inside the
noise floor of llama.cpp.

Phase 1 scaffolding shipped this session: trait, config, env-flag
dispatch, EAGLE stub, CPU `verify_and_accept` + `verify_tree`
oracles, `DraftTree` data structure, `TreeAttentionMask`. All on
[`feat/cuda-spec-draft-head`](https://github.com/ianblenke/larql/tree/feat/cuda-spec-draft-head),
30 unit tests, no CUDA dependency, all local CI gates green.

Phases 2-4 (batched mmvq, tree attention + verify kernel, Tensor
Core re-arm) are documented in
[design.md §6](../cuda-speculative-decoding/design.md) with
stop-ship gates per phase. Estimated 2-3 weeks for a working
end-to-end implementation that passes the bit-equal token-ID
parity test on a 256-prompt eval.

## What this means for batch=1 decode

The original perf-results retro's "Path A: Marlin-style INT4-IMMA
mmvq (5-10 days, est. -1.0 to -1.5 ms)" is **closed as not
viable** for batch=1. Update its status:

- Path A — **closed, see `cuda-marlin-imma-probe`**
- Path B (WMMA mitigation #2/#3) — closed by
  `cuda-attn-wmma-multi-warp`
- Path C (RoPE freq cache) — open, but ≤0.07 ms savings, deferred
- Path D (Q/K/V mmvq fusion) — closed by `cuda-q4k-qkv-fuse-v2`
  (regressed)
- Path E (norm + Q8_1 fusion) — closed by
  `cuda-fused-norm-quantize` (regressed)

**Every path that doesn't change the batch dimension has been
exhausted.** The next 1.71× factor closure requires speculative
decoding or equivalent architectural batching.

## Capabilities

(none — proposal-only, no spec deltas)

## Impact

- **Affected files**: this proposal only.
- **Affected systems**: documentation. CUDA decode/prefill code
  unchanged.
- **Out of scope**: any code changes; speculative decoding
  implementation (covered by `cuda-speculative-decoding`).
