## Why

End-to-end speculative decoding has every kernel and CPU primitive
landed (PRs #1–#15). The remaining work is **integration** — wiring
the off-the-shelf `SmallModelDrafter` through `generate()` and
implementing the `target_forward` closure that runs the target on
N candidate tokens at once. This is the slice that converts
"bench shows 'drafter loaded'" into "decode 7.55 → 4.2 ms/tok".

This proposal captures the integration shape, decisions, and
phasing so the next session(s) can execute without re-deriving
the design.

## What's already in main

- `cuda::sampling::verify_tree_kernel` (serial + parallel)
- `cuda::q4k_batched::matvec_batched` (M_TILE up to 8, 7× speedup at M=8)
- `cuda::attn_tree::tree_decode_attention` (per-q bitmask, GQA)
- `larql_inference::speculative::{Drafter, SpecConfig, SpeculativeStep, StepOutcome}`
- `larql_inference::speculative::dispatch::maybe_speculative_step`
- `larql_inference::speculative::SmallModelDrafter` (off-the-shelf, vindex-loaded)
- `larql bench --draft-model <vindex>` flag (loads drafter, doesn't dispatch yet)

## What this change ships

This is a **proposal-only** change documenting the integration plan.
No code changes. The actual implementation lands across 3 follow-on
PRs:

- **Phase 4a (this proposal)** — design doc + spec scenarios for the integration contract
- **Phase 4b** — naive sequential `target_forward` + dispatch wiring at `gpu.rs:735`. Correctness first; **not faster than baseline.**
- **Phase 4c** — batched `target_forward` using the 3 GPU kernels. Where the actual perf win lives.
- **Phase 4d** — 256-prompt token-ID parity eval + `bench/decode_speculative.rs`.

## Capabilities

### Modified

- `inference-speculative-decoding` — adds 5 spec scenarios for the
  integration contract: dispatch decision tree, KV cache rollback
  semantics, full-vocab probability path, naive vs batched
  target_forward equivalence, perf gate.

## Impact

- Documentation only this PR.
- Subsequent PRs touch: `crates/larql-inference/src/layer_graph/generate/gpu.rs`, `crates/larql-inference/src/layer_graph/generate/sampling.rs`, `crates/larql-inference/src/forward/predict/dense.rs` (full-vocab probs path), and a new `bench/decode_speculative.rs`.
- Out of scope: EAGLE draft head training, custom checkpoint formats.
