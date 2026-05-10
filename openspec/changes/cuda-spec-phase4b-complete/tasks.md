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
  - [ ] **C.2.d** `gpu.rs:735` integration switch from `try_thread_speculative_step_v2` (separate ModelWeights + naive) to a new `try_thread_speculative_step_v3` that uses canonical backend + `target_forward_via_speculative_decode`. Measurement gate: end-to-end ms/tok must drop from naive's ~25 s/tok to ≤ 100 ms/tok at depth=2 b=1 (the ~250× speedup the function provides should propagate through to wall time).
  - [ ] **C.2.e** True batched `CudaBackend::decode_tokens_speculative` override: composes `cuda::q4k_batched::matvec_batched` (M_TILE=N) + `cuda::attn_tree::tree_decode_attention` (per-q tree mask) + per-position RoPE + KV writes at speculative positions + batched RMSNorm (new kernel needed) + batched lm_head + softmax. **This is the architectural perf win**: ~5–10 ms per call instead of C.2.c's ~15 ms.
- [ ] C.3 KV rollback semantics — track pre-speculative cache_len; on rejection at tree node `r`, call `backend.truncate_kv_cache(cache_len + r)` (API already exists on `CudaBackend`)
- [x] ~~C.4 `rotorquant-window-lag` prereq~~ — **NOT NEEDED**. Confirmed the CUDA decode path uses plain f16 KV cache (`cuda::decode::CudaKvLayer { k: CudaSlice<half::f16>, v: ... }`) — it does NOT use rotorquant compression. The `larql_rotorquant` crate is only used by the host-side `larql_inference::attention::decode::KvCache` (CPU/Metal paths). Phase 4c can proceed without any rotorquant changes.
- [ ] C.5 Tests: `target_forward_batched_matches_naive_64_seeds` (the load-bearing parity gate)
- [ ] C.6 Stop-ship: per-step latency ≤ 1.6× single-token decode; 256-prompt token-ID parity vs phase 4b naive

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

## Validation (this PR)

- [x] V.1 `openspec validate cuda-spec-phase4b-complete --strict` passes
- [x] V.2 `make traceability-check` passes after regen
- [x] V.3 No code changes; documentation only
