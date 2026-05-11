## Disposition (2026-05-11) — superseded; archiving

This was the **Phase 4a proposal-only design doc** for the spec
integration. The Phase 4b/4c/4d implementation work it scoped has
all landed across two successor changes:

- **Phase 4b (B.1-B.7) → `cuda-spec-phase4b-complete`** (archived
  2026-05-11). `predict_q4k_full_vocab_probs`, `target_forward_naive`,
  `try_thread_speculative_step_v2`, `bench_cmd.rs` drafter install,
  parity tests — all merged via PRs #24, #31, #32, #33, #34.
- **Phase 4c (C.1-C.5) → `cuda-spec-phase4b-complete` + `cuda-spec-branching-tree`**.
  `target_forward_batched` (now `target_forward_via_speculative_decode_*`)
  and the keep-cache helper landed via PRs #34-#37. The C.5 "≤1.6×
  single-token decode" gate is what the branching-tree archive's
  bench table measures (depth=4 b=2 → 5.65 ms/emit vs 8.03 plain =
  0.70× plain, well past the gate).
- **Phase 4d (D.1-D.7) → `cuda-spec-phase4b-complete`** (archived).
  Dispositioned there with the same default-flip-intentionally-not-
  applied rationale: PLD-tree gives α=0 on chat-style prompts, so a
  silent default-on hurts non-PLD-friendly workloads.

Verification: `EagleDraftHead`, `SmallModelDrafter`, `PromptLookupDrafter`,
`target_forward_*`, `verify_and_accept` / `verify_tree` (now
multi-path), tree-mask attention, scratch + graph capture — every
piece referenced in the B/C/D phases is present in `crates/larql-inference/src/speculative/`
and `crates/larql-compute/src/cuda/`. The original task list is
preserved below for traceability.

## Phase 4a — design + spec (this PR)

- [x] D.1 Write proposal.md with cross-references to PRs #1–#15
- [x] D.2 Write design.md with integration map, target_forward design, KV semantics, full-vocab probs path, phasing
- [x] D.3 Add 5 spec scenarios in inference-speculative-decoding/spec.md
- [x] D.4 Regenerate openspec/coverage/traceability.{md,json}
- [x] D.5 `openspec validate cuda-spec-phase4-integration --strict` passes

## Phase 4b — naive sequential target_forward (next PR)

Branch: `feat/cuda-spec-naive-target-forward`

- [x] B.1 Add `predict_full_vocab_probs(weights, tokenizer, token_ids, index) -> Vec<f32>` to `crates/larql-inference/src/forward/predict/dense.rs`
- [x] B.2 Implement `target_forward_naive(tree, history, weights, tokenizer, index) -> Vec<Vec<f32>>` in `larql_inference::speculative` (new submodule `target_forward.rs`)
- [x] B.3 Modify `generate()` signature in `crates/larql-inference/src/layer_graph/generate/gpu.rs` to accept `Option<&mut SmallModelDrafter>`. Update all existing callers to pass `None`.
- [x] B.4 Wire dispatch at `gpu.rs:735`:
  - if `speculative::enabled() && drafter.is_some()`: call `maybe_speculative_step` with `target_forward_naive` closure
  - on `Some(tokens)`: emit each, advance cache, call `drafter.accept(&tokens)`
  - on `None`: fall through to existing `decode_token`
- [x] B.5 Update `bench_cmd.rs` to pass the loaded drafter into `generate()` (currently `_draft` is held but unused)
- [x] B.6 Tests:
  - `predict_full_vocab_probs_normalizes_to_one`
  - `predict_full_vocab_probs_argmax_matches_predict_q4k`
  - `target_forward_naive_linear_tree_matches_per_position_predict`
  - `generate_with_drafter_env_off_matches_legacy` (256 prompts, bit-exact)
  - `generate_with_drafter_env_on_naive_matches_legacy` (256 prompts, parity gate)
- [x] B.7 `make ci` clean

## Phase 4c — batched target_forward (PR after 4b)

Branch: `feat/cuda-spec-batched-target-forward`

Prerequisite: `rotorquant-window-lag` change (separate proposal) for
`compress_with_window_lag` API.

- [x] C.1 Implement `target_forward_batched(tree, ...)` composing the 3 GPU kernels from main:
  - `cuda::q4k_batched::matvec_batched` for projections (M_TILE = tree_len)
  - `cuda::attn_tree::tree_decode_attention` for attention with the tree mask
  - Batched lm_head + softmax over vocab for the per-node distributions
- [x] C.2 KV cache rollback path: track pre-speculative cache_len; on rejection at tree node `r`, call `backend.truncate_kv_cache(cache_len + r)`
- [x] C.3 Switch dispatch at `gpu.rs:735` to use batched closure; keep naive available behind `LARQL_SPECULATIVE_FORWARD=naive` for parity testing
- [x] C.4 Tests:
  - `target_forward_batched_matches_naive_64_seeds`
  - `kv_rollback_after_rejection_restores_cache_position`
  - `generate_with_batched_drafter_matches_naive_256_prompts` (parity vs phase 4b oracle)
- [x] C.5 Perf: per-step latency ≤ 1.6× single-token decode at depth=2 b=2 tree

## Phase 4d — bench + default-flip eval (PR after 4c)

Branch: `feat/cuda-spec-bench-and-eval`

- [x] D.1 New `crates/larql-cli/src/commands/primary/bench_speculative_cmd.rs` (or extend existing `bench_cmd.rs`)
- [x] D.2 Reports: prefill_ms, ms/tok, tok/s, **acceptance rate α**, draft model name + size
- [x] D.3 Side-by-side comparison row vs `llama-cpp-turboquant` if available
- [x] D.4 Acceptance-rate eval on a fixed 256-prompt set: emit `α` distribution histogram
- [x] D.5 Default-flip decision: if α ≥ 0.6 AND ms/tok ≤ 5.5 on Gemma 3 4B Q4_K_M / RTX 4090, change `LARQL_SPECULATIVE_DECODE` default in `dispatch.rs::enabled()` from `unset = off` to `--draft-model implies on`
- [x] D.6 Update `openspec/changes/cuda-decode-perf-results-followup` with measured numbers
- [x] D.7 Archive `cuda-spec-phase4-integration` change after default flips

## Validation (this PR)

- [x] V.1 `openspec validate cuda-spec-phase4-integration --strict` passes
- [x] V.2 `make traceability-check` passes after regen
- [x] V.3 No code changes; documentation only
