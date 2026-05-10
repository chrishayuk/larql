## Phase 4a — design + spec (this PR)

- [x] D.1 Write proposal.md with cross-references to PRs #1–#15
- [x] D.2 Write design.md with integration map, target_forward design, KV semantics, full-vocab probs path, phasing
- [x] D.3 Add 5 spec scenarios in inference-speculative-decoding/spec.md
- [x] D.4 Regenerate openspec/coverage/traceability.{md,json}
- [x] D.5 `openspec validate cuda-spec-phase4-integration --strict` passes

## Phase 4b — naive sequential target_forward (next PR)

Branch: `feat/cuda-spec-naive-target-forward`

- [ ] B.1 Add `predict_full_vocab_probs(weights, tokenizer, token_ids, index) -> Vec<f32>` to `crates/larql-inference/src/forward/predict/dense.rs`
- [ ] B.2 Implement `target_forward_naive(tree, history, weights, tokenizer, index) -> Vec<Vec<f32>>` in `larql_inference::speculative` (new submodule `target_forward.rs`)
- [ ] B.3 Modify `generate()` signature in `crates/larql-inference/src/layer_graph/generate/gpu.rs` to accept `Option<&mut SmallModelDrafter>`. Update all existing callers to pass `None`.
- [ ] B.4 Wire dispatch at `gpu.rs:735`:
  - if `speculative::enabled() && drafter.is_some()`: call `maybe_speculative_step` with `target_forward_naive` closure
  - on `Some(tokens)`: emit each, advance cache, call `drafter.accept(&tokens)`
  - on `None`: fall through to existing `decode_token`
- [ ] B.5 Update `bench_cmd.rs` to pass the loaded drafter into `generate()` (currently `_draft` is held but unused)
- [ ] B.6 Tests:
  - `predict_full_vocab_probs_normalizes_to_one`
  - `predict_full_vocab_probs_argmax_matches_predict_q4k`
  - `target_forward_naive_linear_tree_matches_per_position_predict`
  - `generate_with_drafter_env_off_matches_legacy` (256 prompts, bit-exact)
  - `generate_with_drafter_env_on_naive_matches_legacy` (256 prompts, parity gate)
- [ ] B.7 `make ci` clean

## Phase 4c — batched target_forward (PR after 4b)

Branch: `feat/cuda-spec-batched-target-forward`

Prerequisite: `rotorquant-window-lag` change (separate proposal) for
`compress_with_window_lag` API.

- [ ] C.1 Implement `target_forward_batched(tree, ...)` composing the 3 GPU kernels from main:
  - `cuda::q4k_batched::matvec_batched` for projections (M_TILE = tree_len)
  - `cuda::attn_tree::tree_decode_attention` for attention with the tree mask
  - Batched lm_head + softmax over vocab for the per-node distributions
- [ ] C.2 KV cache rollback path: track pre-speculative cache_len; on rejection at tree node `r`, call `backend.truncate_kv_cache(cache_len + r)`
- [ ] C.3 Switch dispatch at `gpu.rs:735` to use batched closure; keep naive available behind `LARQL_SPECULATIVE_FORWARD=naive` for parity testing
- [ ] C.4 Tests:
  - `target_forward_batched_matches_naive_64_seeds`
  - `kv_rollback_after_rejection_restores_cache_position`
  - `generate_with_batched_drafter_matches_naive_256_prompts` (parity vs phase 4b oracle)
- [ ] C.5 Perf: per-step latency ≤ 1.6× single-token decode at depth=2 b=2 tree

## Phase 4d — bench + default-flip eval (PR after 4c)

Branch: `feat/cuda-spec-bench-and-eval`

- [ ] D.1 New `crates/larql-cli/src/commands/primary/bench_speculative_cmd.rs` (or extend existing `bench_cmd.rs`)
- [ ] D.2 Reports: prefill_ms, ms/tok, tok/s, **acceptance rate α**, draft model name + size
- [ ] D.3 Side-by-side comparison row vs `llama-cpp-turboquant` if available
- [ ] D.4 Acceptance-rate eval on a fixed 256-prompt set: emit `α` distribution histogram
- [ ] D.5 Default-flip decision: if α ≥ 0.6 AND ms/tok ≤ 5.5 on Gemma 3 4B Q4_K_M / RTX 4090, change `LARQL_SPECULATIVE_DECODE` default in `dispatch.rs::enabled()` from `unset = off` to `--draft-model implies on`
- [ ] D.6 Update `openspec/changes/cuda-decode-perf-results-followup` with measured numbers
- [ ] D.7 Archive `cuda-spec-phase4-integration` change after default flips

## Validation (this PR)

- [x] V.1 `openspec validate cuda-spec-phase4-integration --strict` passes
- [x] V.2 `make traceability-check` passes after regen
- [x] V.3 No code changes; documentation only
