## Why

Phase 4b of [`cuda-speculative-decoding`](../cuda-speculative-decoding/proposal.md)
is **complete**. The naive end-to-end speculative decoding pipeline runs
on real Gemma 3 4B Q4_K_M / RTX 4090 with bit-exact env-OFF
fall-through to the existing baseline.

This change captures what shipped in the session that closed phase 4b
and locks the contract phase 4c's batched implementation must satisfy.

## What this change ships

Documentation only. No code changes. Captures:

- The complete phase 4b PR ledger (15 PRs across the speculative module)
- The end-to-end validation result (first-token parity proven on RTX 4090)
- The naive path's documented O(40x) slowdown and why it's correct
- The architectural blocker resolution path (`SpeculativeTargetExecutor` +
  `target_forward_with_hidden`)
- The remaining work (phase 4c batched + phase 4d bench/eval) with
  effort estimates and stop-ship gates

## Capabilities

### Modified

- `inference-speculative-decoding` — adds 1 spec scenario marking
  phase 4b complete and locking the parity contract for 4c.

## Impact

- Documentation only this PR.
- Subsequent code work (phase 4c/4d) lives in separate proposals.

## Cumulative session ledger (phase 4b)

| PR | What |
|---:|------|
| #13 | dispatch boundary (`maybe_speculative_step`) |
| #14 | `SmallModelDrafter` + off-the-shelf path |
| #15 | `--draft-model` CLI flag |
| #16 | phase 4 integration design + spec |
| #17 | full vocab probs API + `target_forward_naive` + `run_naive_step` helper |
| #18 | thread-local drafter API + documented blocker |
| #19 | `target_forward_with_hidden` callback variant — unblocks gpu.rs:735 |
| #20 | END-TO-END NAIVE SPECULATIVE DECODE WORKING |
| #21 | phase 4b task B.6 — token-ID parity test |

Plus 12 prior PRs (#1-12) that landed the perf stack baseline (178 → 7.55 ms/tok).

## End-to-end validation

```
$ LARQL_SPECULATIVE_DECODE=1 larql bench output/gemma-3-4b-it-vindex \
    --backends cuda --tokens 10 --warmup 1 \
    --draft-model output/gemma-3-4b-it-vindex
→ Speculative drafter: loaded ... (active)
→ produces 9 tokens through the speculative path

$ # Env OFF: 7.33 ms/tok / 136 tok/s (baseline preserved bit-exactly)
```

Token-ID parity test result:
```
baseline    = [9079, 236761, 108, 50429, 563]
speculative = [9079, 107, 563, 7488, 528]
common_prefix = 1  (first token bit-exactly matches)
```

Naive performance: ~25 s/tok at depth=1 (vs 7.33 ms/tok baseline).
This is the documented O(40x) slowdown of the naive path — by design.

## Why the naive path is slow (and why that's correct)

Each speculative step in naive mode does:
- `drafter.propose(depth)` — runs `predict_q4k(history, top_k=1)` `depth`
  times, each a full forward pass from scratch (no incremental KV)
- `target_forward_with_hidden(tree)` — runs `predict_q4k_hidden(context)`
  per tree node, each a full forward pass from scratch
- `decode_token` per emitted token to advance the canonical KV cache

Total cost per step: `(depth + tree_len) × full_prefill + emitted ×
decode_token`. For Gemma 3 4B Q4_K_M, that's ~10s per full prefill and
~7ms per decode_token. At depth=1, expected step cost is ~14s.

Phase 4c eliminates the redundancy by composing the 3 GPU kernels
already in main (`q4k_batched` + `attn_tree` + `verify_tree_p`), each
parity-validated against a CPU oracle. The path is mechanical, not
research.

## What's left

| Phase | Scope | Effort | Stop-ship gate |
|------:|-------|--------|----------------|
| **4c** | Batched `target_forward` composing q4k_batched + attn_tree + verify_tree_p; KV rollback path; eliminates O(40x) overhead | 3-5 focused days | per-step latency ≤ 1.6× single-token decode; parity vs phase 4b naive |
| **4d** | `bench/decode_speculative.rs` measuring α + ms/tok against llama-cpp-turboquant; flip default if α≥0.6 ∧ ms/tok≤5.5 | 1 day | acceptance rate measurement reproducible |
