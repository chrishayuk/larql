## Phase 1 — Draft model integration

- [ ] 1.1 New crate module `crates/larql-inference/src/speculative/mod.rs`
      with `Drafter` trait, `SpecConfig`, and `SpeculativeDecoder` skeleton
      that returns `Vec<TokenId>` from a single `step()` call.
- [ ] 1.2 `EagleDraftHead` impl: load a checkpoint matching the target's
      LM head dim, run one transformer block + LM head on the target's
      last hidden state, return top-K candidates.
- [ ] 1.3 Draft KV cache lives separately from target KV cache. Same
      `KvCache` trait, smaller depth (1 layer).
- [ ] 1.4 CLI `--draft-model <path>` flag in `larql-cli`.
- [ ] 1.5 Tests: `eagle_draft_head_loads_checkpoint`,
      `eagle_draft_head_proposes_deterministic_on_seed`.
- [ ] 1.6 Branch: `feat/cuda-spec-draft-head`. Land behind
      `LARQL_SPECULATIVE_DECODE=1` (still draft-only; verification is
      stubbed to "always accept first draft").

## Phase 2 — Batched mmvq + attention

- [ ] 2.1 `cuda::q4k_mmvq::mul_mat_q4_K_q8_1_batched<M_TILE>` template,
      with M_TILE ∈ {1, 2, 4, 8}. Dispatcher selects on `q_tokens`.
      M_TILE=1 SHALL produce bit-identical output to current cooperative
      kernel.
- [ ] 2.2 `cuda::elem::rms_norm_q8_1_batch` for `q_tokens` rows. One
      block per row (no across-SM fusion — avoids the regression in
      `feat/cuda-fused-norm-quantize`).
- [ ] 2.3 `cuda::attn::fused_decode_attention` gains `q_tokens: u32`
      parameter, defaults to 1, broadcasts cleanly across the new dim.
      Kernel changes: score buffer becomes `[q_tokens, kv_len]`; output
      becomes `[q_tokens, n_q_heads, head_dim]`. Bit-exact at q=1.
- [ ] 2.4 `cuda::scratch::DecodeScratch` grows `tree_q`, `tree_mask`,
      `accept_buf` slots. Sized for `max_tree_nodes = 64` worst-case.
- [ ] 2.5 Microbench
      `cuda_q4k_mmvq_batched_vs_single_threshold` measuring crossover
      point of cooperative-dp4a vs cuBLAS-hgemm by `M_TILE`. Document.
- [ ] 2.6 Tests: `q4k_mmvq_batched_matches_unbatched_at_m1`,
      `fused_decode_attention_q_tokens_eq_5_parity`.
- [ ] 2.7 Branch: `feat/cuda-spec-batched-mmvq`. Stop-ship gate:
      bit-exact at q=1; cost at q=5 ≤ 1.6× cost at q=1.

## Phase 3 — Tree attention + verification

- [ ] 3.1 `cuda::attn_tree::tree_decode_attention(q_tree, k_cache,
      v_cache, tree_mask, opts)` — fused tree attention, single
      launch, per-q-token causal mask uploaded as bitmask.
- [ ] 3.2 `cuda::sampling::verify_tree(target_logits, draft_probs,
      tree_layout, rng_state, temperature) -> AcceptedSpan`.
      One launch produces accepted prefix, corrected rejection token,
      bonus token.
- [ ] 3.3 `larql_inference::speculative::verify_and_accept` host-side
      coordinator: call verify_tree, walk accepted span, return
      `AcceptedSpan { accepted: Vec<TokenId>, corrected: TokenId,
      bonus: Option<TokenId> }`.
- [ ] 3.4 Hook `verify_and_accept` into `SpeculativeDecoder::step`
      replacing the stub from phase 1.
- [ ] 3.5 KV cache rollback: extend `larql_rotorquant::compress` with
      `compress_with_window_lag(slot, lag)`. Default lag=8, widened on
      speculative-depth growth. (See design.md §5 option A.)
- [ ] 3.6 Tests: `tree_attention_depth2_branch2_parity_cpu`,
      `verify_tree_matches_cpu_reference_64_seeds`,
      `rotorquant_lag_window_rollback_round_trip`.
- [ ] 3.7 Stop-ship eval: `bench/decode_speculative.rs --verify --eval
      256_prompts.jsonl`. Token IDs SHALL match non-speculative path
      bit-exactly. Any mismatch = revert.
- [ ] 3.8 Branch: `feat/cuda-spec-tree-verify`.

## Phase 4 — Tensor Core re-arm + production rollout

- [ ] 4.1 Re-enable the `feat/cuda-tensor-cores-q4k` cuBLAS hgemm path
      with a `batch ≥ 4` guard in the dispatcher. Microbench at batch
      ∈ {1,4,8} confirms it wins above the threshold and loses below.
- [ ] 4.2 Re-enable WMMA attention scores from
      `feat/cuda-attn-wmma-multi-warp` at `q_tokens ≥ 4`.
- [ ] 4.3 CLI: `--speculative-tree-depth N --speculative-branches K`
      (defaults: depth=2, branches=2 → 5-node tree).
- [ ] 4.4 `bench/decode_speculative.rs` reports ms/tok, tok/s,
      acceptance rate `α`, and gap vs llama-cpp-turboquant baseline.
- [ ] 4.5 If `α ≥ 0.6` and ms/tok ≤ 5.5 on Gemma 3 4B Q4_K_M,
      flip default to `LARQL_SPECULATIVE_DECODE=1`. Otherwise leave
      as opt-in and document the gap.
- [ ] 4.6 Update `openspec/changes/backfill-specs/proposal.md` to add
      the new capability `inference-speculative-decoding` to the
      catalogued list.
- [ ] 4.7 Branch: `feat/cuda-spec-tensor-core-rearm`.

## Validation

- [ ] V.1 `openspec validate cuda-speculative-decoding --strict` passes.
- [ ] V.2 `make ci` clean (fmt + clippy + tests + traceability +
      openspec-validate).
- [ ] V.3 All four phase branches squash-merge to main behind the env
      flag. `LARQL_SPECULATIVE_DECODE` unset = bit-exact current path.
- [ ] V.4 Final result row added to
      `crates/larql-cli/bench/results/decode_perf.md` showing the
      LARQL-vs-llama.cpp gap closure.
- [ ] V.5 If phase 3 stop-ship fails: document in
      `openspec/changes/cuda-speculative-decoding/RETROSPECTIVE.md`
      and revert the speculative entry-point. Keep the batched
      kernels (phase 2) — they are useful on their own.
