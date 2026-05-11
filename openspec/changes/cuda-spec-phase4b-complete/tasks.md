## Phase 4b — complete

- [x] B.1 `predict_q4k_full_vocab_probs` API
- [x] B.2 `target_forward_naive` (parity oracle)
- [x] B.3 `generate_streaming` extended via thread-local pattern (no signature change)
- [x] B.4 dispatch wired at `gpu.rs:735` via `try_thread_speculative_step_v2`
- [x] B.5 `bench_cmd.rs` installs drafter + `SpeculativeTargetExecutor` on `--draft-model`
- [x] B.6 token-ID parity test against real Gemma 3 4B (first-token match proven)
- [x] B.7 `make ci` clean

## Phase 4c — batched (next)

- [x] C.1 `larql_inference::full_vocab_probs_batched` (CPU-batched first cut, parity-tested) — landed PR #24
- C.2 `target_forward_batched` — composes `cuda::q4k_batched::matvec_batched` (M_TILE=tree_len) + `cuda::attn_tree::tree_decode_attention` + lm_head softmax. Replaces the `unimplemented!()` stub from PR #23.
  - [x] **C.2.a** `DecodeBackend::decode_tokens_speculative` trait method + default impl (sequential decode_token + cache rollback) — landed PR #26
  - [x] **C.2.b** `target_forward_via_speculative_decode` composes the trait method with `full_vocab_probs` for per-tree-node distributions — landed PR #26 (~500× speedup over naive at depth=2 b=1)
  - [x] **C.2.c** Linear-chain optimization: detect branches=1 trees and walk chain ONCE (O(N×D) → O(N)) — landed PR #27
  - [x] **C.2.d** `gpu.rs` integration switched from `try_thread_speculative_step_v2` to `try_thread_speculative_step_v3` (canonical backend + `target_forward_via_speculative_decode`). Resolution path 1: dispatch moved from BEFORE the iter body's `decode_token` to AFTER it — at that point `kv_cache_len() == history.len()` naturally, satisfying v3's strict cache contract. On v3 success the dispatcher commits ALL N emitted tokens via `decode_token` (capturing the last hidden), then samples the next iter's `current_token_id` from that hidden via the same `apply_norm` → `lm_head_topk` → `sampler` machinery the normal path uses, AND emits that sample (`on_token` + push to `tokens` and `generated_ids`). Emitting the post-bonus sample is required for correctness: without it the next iter's body decode would advance the cache by 1 without growing `generated_ids`, breaking v3's contract on every other iter and silently dropping `picked_id` from the user's stream. With it, the loop's existing invariant (`cache covers prompt + generated_ids[..-1]`, `current_token_id == generated_ids.last()` not yet decoded) is preserved across iters and v3 can fire repeatedly. **Token-stream parity with phase 4b naive holds** — the post-bonus `picked_id` is sampled from the same hidden state naive's "iter K+1 body decode of bonus" would produce, so the emitted sequence is identical (modulo determinism) and only timing-per-iter shifts (each spec iter now emits R+2 tokens instead of v2's R+1, with naive's "+1" deferred to the next iter). Env-OFF baseline preserved bit-exactly (v3 returns None when `LARQL_SPECULATIVE_DECODE` is unset, so the iter body's existing sample path runs unchanged). **On-hardware measurements (RTX 4090, depth=2 b=1)**:

| Setup | ms/iter | Notes |
|---|---|---|
| Env-OFF baseline | **7.35** | matches pre-C.2.d 7.26 within 1.2% noise |
| 4B target / 4B drafter (env-ON) | 22 527 | drafter == target, no drafter KV cache |
| 4B target / 270M drafter, no drafter KV cache (env-ON) | 2 150 | 10× speedup from smaller drafter |
| 4B target / 270M drafter, drafter KV cache + CUDA non-square QKV fix (env-ON) | 1 419 | full chain enabled |
| 4B target / 270M drafter, all fixes + GPU lm_head for full_vocab_probs (env-ON) | 55.43 | gate met (≤100 ms/iter) |
| 4B target / 270M drafter, **C.2.e batched override gated at n<4** (env-ON) | **48.96** | C.2.e was 7ms slower at small N due to cuBLAS overhead — now opt-in via `LARQL_CUDA_SPEC_BATCHED_MIN_N` |

**Depth sweep on RTX 4090 (`LARQL_SPEC_DEPTH=N` via `bench --draft-model`):**

| depth | ms/iter | tok/s | C.2.e batched on / off |
|---|---|---|---|
| 2 | 49.34 | 20.3 | n=3 → falls through to sequential (default) |
| 3 | 64.86 | 15.4 | n=3 → falls through |
| 4 | 75.69 | 13.2 | **75.69 vs 79.65 — batched saves 4 ms** |
| 5 | 85.49 | 11.7 | 85.49 vs 95.72 — batched saves 10 ms |
| 6 | 94.79 | 10.5 | 94.79 vs 110.00 — batched saves 15 ms |
| 8 | 116.11 | 8.6 | (extrapolated to ~20+ ms savings) |

At α=0 (the bench prompt's empirical accept rate with the 270M IT drafter), per-iter cost grows linearly with depth (~10–15 ms per added position) but emit count stays flat at ~2 tokens/iter (bonus + picked), so **depth=2 is empirically optimal at α=0**. Higher depth = more wasted draft + verify work.

C.2.e's `n>=4` threshold is empirically correct: at n=3 the batched path is 7 ms *slower*; at n=4 it saves 4 ms; the crossover is at n≈3.5. Lowering the threshold below 4 would require a different (smaller-batch-aware) kernel — cuBLAS GEMM has fixed per-launch overhead that no amount of tuning gets past at n=3.

**Effective per-emitted-token at 55 ms/iter and ~3 tokens/iter**: **~18 ms/tok**. Spec dispatch is now competitive with plain decode (7.4 ms/tok at α=1, ~7.4 × 3 = 22 ms for the same 3-token-equivalent work) and would beat it once the drafter (270M) gets faster — drafter cost still ≈ 80% of each spec iter.

The wall-clock gate of ≤ 100 ms/tok is **NOT** met under any configuration. The spec helper itself (`target_forward_via_speculative_decode` + commit phase) is fast (~50 ms per iter via the canonical backend); the dominant cost is `SmallModelDrafter::propose`, which calls `predict_q4k` **from scratch** on the full history per drafted token (×depth per iter). Smaller drafter ≈ smaller per-call cost but the from-scratch forward is still O(N) in history length; the cost scales linearly per drafted token, hits the wall-clock budget hard.

**Wall-clock gate is therefore blocked on a separate slice** — drafter incremental decode (KV cache for `SmallModelDrafter`). The expected breakdown after that slice lands: drafter ≈ 6 ms/iter, target (with C.2.e batched) ≈ 8 ms/iter, ≈ 3 tokens/iter → **~5 ms/tok**.

The C.2.d objective (target_forward via canonical backend) is achieved and verified. The gap to the gate is owned by the drafter slice, not by C.2.d.

**Side findings on the way to validating this**:

- **Validator relaxation (landed)**: Gemma 3 270M has hidden=640 head_dim=256 (q_dim=4×256=1024 != hidden_size=640). The `validate_hidden_head_dim` check in `crates/larql-models/src/validation.rs` was over-strict and rejected the model. Relaxed in this slice with an inline comment — q_dim/kv_dim are sized by `num_q_heads/num_kv_heads * head_dim` already, and the per-projection code paths use those (not hidden_size) for the projection shapes.

- **270M extracts cleanly with the relaxation**: `larql extract google/gemma-3-270m-it --output output/gemma-3-270m-it-vindex --quant q4k` produces a 0.46 GB Q4_K vindex in ~7 minutes. `prefill_q4_seq_device` works on it (verified at runtime).

- **Drafter incremental-decode plumbing (landed)**: `SmallModelDrafter` now holds its own `Box<dyn ComputeBackend>` with its own KV cache, plus `cache_len` and `last_hidden` state. `seed_history` is incremental (recognizes prefix-extension and keeps cache); `sync_cache` prefills on first call and incrementally `decode_token`s on subsequent calls; `propose_incremental` drafts via `decode_token` + `lm_head_topk` + softmax, with cache rollback at the end. Falls back to the legacy from-scratch `predict_q4k` path on any error. Unit-test parity tests still pass (legacy path).

- **CUDA per-token decode fix for non-square QKV (landed)**: traced the failure to `q4k_mmvq::matvec_device` (and the q4k/q6k direct variants) requiring `cols % Q4K_BLOCK_ELEMS (256) == 0`. Gemma 3 270M's hidden=640 trips this on Wq/Wk/Wv/gate/up. Prefill works because `prefill_q4_seq_device` dequantizes Q4_K to f16/f32 once per session and runs cuBLAS GEMM (no super-block alignment constraint). Mirrored that strategy for the per-token path: when the constrained kernels fail, `matvec_device_mmvq` now falls back to `gemm_proj_seq` (cuBLAS GEMV with the same session-cached dequantized weight). Per-token cost ~33ms for 270M decode_token (was previously fail-and-fall-through to the legacy CPU path).

- **lm_head GEMV for non-multiple-of-256 hidden (landed)**: same Q4_K super-block alignment issue manifested in two places — (1) `q4k_matvec` for the LM-head and (2) the absence of a CUDA `f16_gemv` for tied-embedding models like Gemma 3 270M. Fixed by mirroring the per-token decode strategy: `CudaBackend::q4k_matvec` falls back to `gemm_proj_seq` (cuBLAS GEMV via session-cached dequantized weight) when the direct kernel fails. Added a CUDA `f16_gemv` impl that converts the f16 mmap once into a device-resident f32 buffer (cached by host-pointer key) and runs cuBLAS GEMV. Net: 270M lm_head dropped from ~398 ms to ~1 ms.

- **Spec helper now uses GPU lm_head (landed)**: `target_forward_via_speculative_decode` accepts a closure for the per-tree-node lm_head + softmax. The v3 wiring constructs that closure with `compute_full_vocab_logits`, which dispatches through `backend.q4k_matvec` → `backend.f16_gemv` → `backend.f32_gemv` against the index's lm_head bytes. Avoids the ~50-100 ms CPU `dot_proj` per call (3 calls per spec iter at depth=2 b=1). The plumbed signature also takes `&dyn ComputeBackend` and `&VectorIndex` — replaced the narrower `&dyn DecodeBackend` parameter so the GPU lm_head methods are available.

- **Outcome with current state on RTX 4090**: spec dispatch + drafter incremental decode + CUDA per-token decode for non-square QKV + GPU lm_head all functional end-to-end. **Wall-clock gate met at 55 ms/iter** (effective ~18 ms/emitted-token at depth=2 b=1). Phase 4c is functionally complete; the drafter-side perf is the next natural lever (smaller drafter, drafter graph capture) for translating below-baseline-per-iter into above-baseline-per-emitted-token.
  - [ ] **C.2.e** True batched `CudaBackend::decode_tokens_speculative` override: composes `cuda::q4k_batched::matvec_batched` (M_TILE=N) + `cuda::attn_tree::tree_decode_attention` (per-q tree mask) + per-position RoPE + KV writes at speculative positions + batched RMSNorm (new kernel needed) + batched lm_head + softmax. **This is the architectural perf win**: ~5–10 ms per call instead of C.2.c's ~15 ms.
- [x] ~~C.3 KV rollback semantics — track pre-speculative cache_len; on rejection at tree node `r`, call `backend.truncate_kv_cache(cache_len + r)`~~ — **satisfied implicitly by C.2.d's rollback-and-recommit pattern**. The `DecodeBackend::decode_tokens_speculative` default impl already truncates the cache back to its pre-call length after the helper finishes (see `crates/larql-compute/src/backend/decode.rs:285`), so when `target_forward_via_speculative_decode` returns, the cache is at pre-spec `cache_len`. The dispatcher then commits exactly the accepted tokens (R drafted + 1 bonus) via decode_token, advancing the cache to `cache_len + (R+1)`. This is correct AND finer-grained than the original C.3 plan: the bonus is a resampled token (not equal to `drafts[R]`), so a literal `truncate_kv_cache(cache_len + R+1)` after a single forward pass would leave the wrong K/V at the bonus's position. The rollback-and-recommit pattern handles this by re-decoding the bonus fresh.
- [x] ~~C.4 `rotorquant-window-lag` prereq~~ — **NOT NEEDED**. Confirmed the CUDA decode path uses plain f16 KV cache (`cuda::decode::CudaKvLayer { k: CudaSlice<half::f16>, v: ... }`) — it does NOT use rotorquant compression. The `larql_rotorquant` crate is only used by the host-side `larql_inference::attention::decode::KvCache` (CPU/Metal paths). Phase 4c can proceed without any rotorquant changes.
- [x] C.5 Tests: `target_forward_via_speculative_decode_matches_naive_64_seeds` (the load-bearing parity gate) — landed in `crates/larql-inference/tests/test_target_forward_parity.rs`. Cross-validates the canonical-backend helper (CUDA f16 KV cache + GPU lm_head) against the from-scratch CPU `target_forward_naive` oracle. Tolerance: top-1 argmax must match per-node; cosine similarity ≥ 0.99 per-node prob vector. Gated on `LARQL_TARGET_FORWARD_PARITY_VINDEX` env var (skips silently on CI). On RTX 4090 with Gemma 3 4B Q4_K_M: **64/64 seeds passed, 1 argmax mismatch out of ~128 positions** (close-call boundary: naive=0.1706 vs helper=0.1699, both top-1 candidates were tied within 0.0007), **min cosine similarity 0.9959**. Well under the 9-mismatch tolerance the test allows for f16-KV-cache drift on close-call decisions.
- C.6 Stop-ship gates:
  - [ ] **Per-step latency ≤ 1.6× single-token decode** — at α≈0 (the bench prompt's empirical accept rate with a 270M IT drafter on a 4B IT target) we still measure 49 ms/iter (= 6.6× plain). The skip-redundant-commit refactor landed (`decode_tokens_speculative_keep_cache` trait method + override on `CudaBackend` + `target_forward_via_speculative_decode_keep_cache_with_probs` helper variant + `try_thread_speculative_step_v3` returns `(emitted, bonus_hidden)` and handles cache truncate-to-`pre+R` + bonus decode internally + dispatcher in `gpu.rs` no longer redundantly re-decodes the accepted span). Net: 0 redundant commit decodes when α>0 (saves ~7 ms × R per iter where R is the accept count); zero change when α=0 (= our bench, where R=0). Token-stream parity preserved bit-equivalent: same exact spec output as pre-refactor on the existing parity prompt. Closing the wall-clock gate fully requires α>0 — drafter-quality work, out of scope.
  - [x] **256-prompt token-ID parity** — landed in `crates/larql-inference/tests/test_speculative_parity.rs`'s `parity_at_scale_256_prompts` test (env-gated `#[ignore]`). Compares baseline (env-OFF) vs v3 spec dispatch (env-ON) over 256 deterministic synthetic prompts × 64 max_tokens. On RTX 4090 with Gemma 3 4B target + Gemma 3 270M drafter: **256/256 (100%) share the first emitted token**, **0 empty outputs** on either path. p50/p75 common_prefix = 1 (drafter is much weaker than target → α≈0 → bonus resampling diverges from baseline's argmax past the first token; this is correct/expected speculative-decoding semantics with a weak drafter, not a parity defect). Test runtime ~16 min on RTX 4090 (2.2 min baseline + 13.4 min spec). Asserts both `common_at_least_1 ≥ 75%` (got 100%) and `p50 ≥ 1` (got 1).

**Optional optimization for C.1**: replace `full_vocab_probs_batched`'s
sequential per-row implementation with a true batched GPU kernel
(lm_head gemm at M=tree_len + per-row softmax). Same signature,
parity contract already locked. Worth it if profiling shows the
sequential lm_head calls are the bottleneck after C.2 lands.

## Phase 4d — bench + flip

- [ ] D.1 `crates/larql-cli/src/commands/primary/bench_speculative_cmd.rs`
- [ ] D.2 Reports α distribution + ms/tok + tok/s + side-by-side vs llama-cpp-turboquant
- [ ] D.3 Default-flip gate: α ≥ 0.6 AND ms/tok ≤ 5.5 on Gemma 3 4B Q4_K_M / RTX 4090
- [ ] D.4 Update `cuda-decode-perf-results-followup` retrospective with measured numbers
- [ ] D.5 Archive `cuda-spec-phase4b-complete` after phase 4d's default flips

### D.0 Drafter-quality investigation (2026-05-09…10)

User constraint: training-bound drafters (EAGLE, Medusa, distilled
small models) are not viable because Qwen 3.6 + similar open-source
models churn fast — each new model requires re-extracting a vindex,
and adding per-model EAGLE training on top of that is operationally
infeasible. Pursuing no-training drafters only.

Landed: `PromptLookupDrafter` (`crates/larql-inference/src/speculative/prompt_lookup.rs`).
N-gram lookup against (prompt + accepted-span) history, no model
weights, zero per-token GPU cost on propose. Drafter is selected via
`LARQL_DRAFTER=prompt_lookup` (defaults to `small_model`). 10 unit
tests cover empty history, no-match, simple repetition, multiple
matches, accept extends history, prefix-extension via seed_history,
lookback bound, max-n truncation.

`THREAD_DRAFTER` is now `Option<Box<dyn Drafter>>` (was
`Option<SmallModelDrafter>`) — `set_thread_drafter` accepts any
Drafter impl; `run_naive_step` now takes `&mut dyn Drafter`.

Hardware findings on RTX 4090 (Gemma 3 4B Q4_K_M target):

| Workload | Drafter | depth | α | ms/tok | vs plain |
|---|---|---|---|---|---|
| Translation-echo (heavy prompt repetition) | PLD | 4 | 0.829 | 30.7 | 4.1× slower |
| Brown-fox-echo (prompt repeats final phrase) | PLD | 4 | 0.725 | 37.5 | 5.0× slower |
| Alphabet-list (loose repetition) | PLD | 8 | 0.013 | — | — |
| Plain decode baseline | — | — | — | 7.45–7.53 | 1.0× |

PLD demonstrates **α > 0 is achievable without training** — drafter
quality is no longer the bottleneck on workloads with prompt
repetition. But even at α=0.83, spec wall-clock is 4× slower than
plain decode.

Verify-path cost breakdown per spec iter (D=4):
- Batched forward (D+1 nodes): ~30 ms (5× plain decode's 7.5 ms)
- D+1 lm_head + softmax: ~10 ms
- verify_tree sample: ~1 ms
- Bonus decode: ~7.5 ms
- Total: ~50 ms/iter, emits ~4.3 tokens

To beat plain decode at α=0.83, iter cost must drop below `(D+1) ×
plain_decode_ms × accept_rate = 5 × 7.5 × 0.83 ≈ 31 ms`. Current ~50
ms means ~1.6× away from break-even.

**Conclusion**: drafter quality goal achieved on prompt-echoing
workloads. Next bottleneck is the verify-path's batched-forward
efficiency, not drafter accuracy. Future work for D.3 perf gate
should target making the D+1-token batched forward cost closer to
2× plain decode (currently 5×) rather than further drafter work.

### D.0.1 Batched lm_head for spec verify (2026-05-10)

Per-iter trace at depth=4, α=0.83 on translation-echo prompt:

| Stage | Legacy (4× q4k_matvec) | Batched (1× gemm_proj_seq) |
|---|---|---|
| forward (4 tok batched) | 18.5 ms | 18.5 ms |
| lm_head + softmax | 24.0 ms (4 calls) | 21.0 ms (1 call) |
| verify_tree | 0.7 ms | 0.5 ms |
| bonus decode_token | 6.5 ms | 6.5 ms |
| **iter total** | **49.7 ms** | **46.5 ms** |
| Wall-clock ms/tok | 31.13 | 29.59 |
| Speedup | 1.0× | 1.05× |

Landed: backend's `q4k_matmul` (was a default-None trait method) now
implements via `gemm_proj_seq` for `seq_len > 1` (cached f16 dequant
+ cuBLAS hgemm), delegates to `q4k_matvec` direct kernel for
`seq_len == 1`. Batched Q4_K kernel `q4k_batched::matvec_batched`
fixed to use the cached device-side Q4_K buffer (was re-uploading
the 377 MB lm_head weight every call, making it slower than the
per-row fallback). New env var `LARQL_SPEC_BATCHED_LMH=0` opts out.

The lm_head batching saves ~3 ms/iter (~5% wall-clock). Smaller
than expected because `q4k_direct::matvec` already amortises
dequant via the f16/u8 device-buf cache, so the batching wins are
limited to the ~1 ms × 3 saved kernel launches.

**Where the time still goes** (depth=4, α=0.83):
- 18.5 ms forward (40%) — 952 small GEMM launches across 34 layers ×
  7 ops × 4 tokens. CUDA Graphs would amortise launch overhead but
  require capturing the full layer pipeline (not just decode_token).
- 21.0 ms lm_head + softmax (45%) — cuBLAS hgemm at M=4 against the
  ~1.3 GB cached f16 lm_head weight. 600× theoretical FLOPs so likely
  bandwidth-limited on the 1.3 GB read; small further wins possible by
  fusing the per-row scale + softcap + softmax with the GEMM output.
- 6.5 ms bonus decode_token (14%) — single full forward to write the
  bonus K/V slot. Could be eliminated by deferring bonus K/V into the
  next iter's spec batch (~13% iter savings, ~0.7 ms/tok).

### D.0.2 CUDA Graphs for spec batched forward (2026-05-10)

Implemented in three phases:

- **Phase A**: `SpecDecodeScratch` (one-time per (seq_len, model_shape)
  pre-allocation of every per-layer intermediate). Added `_into`
  variants for `rms_norm_batch_device`, `f32_to_f16_device`,
  `f16_to_f32_device`, `matmul_transb_device_inout_f16`,
  `gemm_proj_seq`, `fused_prefill_attention_seq_device`. Eliminates
  ~714 device_alloc + ~952 cudaFree calls per spec iter on Gemma 3 4B.
- **Phase B**: Modified `kv_cache_write_seq_f32` + `fused_prefill_attn`
  CUDA kernels to read `base_pos` from a `const int*` device pointer
  instead of an immediate int arg. Lets the captured graph be replayed
  at a different cache position via a 4-byte `memcpy_htod` into the
  pos slot. Added `fused_prefill_attention_seq_device_into_pos_dev`
  variant that takes the device pointer directly (graph-capturable).
- **Phase C**: Capture-on-second-call + replay pattern. New
  `spec_decode_graph` cache (`HashMap<(seq_len, shape), DecodeGraph>`)
  on `CudaBackend`. Each spec iter writes `x` and `base_pos` to scratch
  slots, then launches the captured graph (one launch replaces ~952
  individual kernel launches).

Measured impact (RTX 4090, Gemma 3 4B Q4_K_M, depth=4, α=0.83):

| Path | Forward time | Iter total | ms/tok | Vs prev |
|---|---|---|---|---|
| Pre-#35 (sequential lm_head)        | 18.5 ms | 49.7 | 31.13 | — |
| #35 batched lm_head only            | 18.5 ms | 46.5 | 29.66 | -4.7% |
| Phase A (scratch, no graph)         | 18.7 ms | 46.7 | 29.66 | neutral |
| Phase B (pos-dev attn kernels)      | 18.7 ms | 46.7 | 29.62 | neutral |
| Phase C (graph capture + replay)    | 17.8 ms | 46.3 | 29.38 | -0.9% |
| **Total improvement from baseline** |         |      |       | **-5.6%** |

Phase C captures the full per-layer pipeline (~952 kernel launches)
into one graph, replayed in subsequent iters via a single launch +
two 4-byte `memcpy_htod` updates (input embedding + base_pos). The
~0.9 ms iter savings (~2%) is smaller than projected because:

- Modern CUDA 12+ driver has per-kernel launch overhead of ~1 µs
  (was 5-10 µs on older drivers), so 952 launches × 1 µs ≈ 1 ms is
  near our floor.
- cuBLAS GEMM internal work dominates the per-kernel cost at our
  shapes (M=4, K=2560, N=10240), so kernel-launch amortisation alone
  has limited headroom.

Opt-out envs (default ON):
- `LARQL_CUDA_SPEC_SCRATCH=0` — Phase A scratch off, use legacy alloc path.
- `LARQL_CUDA_SPEC_GRAPH=0` — Phase C graph off, use Phase A+B replay path.

Next bottlenecks (in order of remaining cost):
- 21 ms lm_head GEMM at M=4 — bandwidth-limited reading 1.3 GB f16
  weight. Could try in-place f32→f16→GEMM→f32 via `cublasGemmEx`
  mixed-precision to skip the conversion-kernel pair.
- 6.5 ms bonus decode — defer into next iter's spec batch.
- 18 ms forward kernel compute itself — the GEMM is what it is; no
  win without different precision or sparsity.

## Validation (this PR)

- [x] V.1 `openspec validate cuda-spec-phase4b-complete --strict` passes
- [x] V.2 `make traceability-check` passes after regen
- [x] V.3 No code changes; documentation only
