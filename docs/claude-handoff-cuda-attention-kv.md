# Claude Handoff: CUDA Attention + KV / OpenSpec Follow-Up

You are continuing work in `/home/ianblenke/github.com/ianblenke/larql`.

Current branch: `main`.
Current worktree at handoff was clean.
Latest pushed commits:

- `f1a24ab [cuda-q4k-device-cache] keep quant weights resident`
- `67ff38c [cuda-resident-q4k-matvec] add direct q4k cuda matvec`
- `c29f4ad [cuda-oxide-migration] port softmax to cuda-oxide`

## Important Workflow

- Follow `AGENTS.md`.
- This repo is OpenSpec-driven. Use OpenSpec changes for behavior/performance work.
- Use `openspec-apply-change` style workflow for existing changes.
- Run focused validation after each slice.
- Commit signed changes and push to `main` only when green.
- Do not revert unrelated user work.

## What Just Happened

The goal was to make `larql bench --backends cuda` meaningfully fast on the local Q4K Gemma vindex:

```bash
output/gemma-3-4b-it-vindex
```

Initial CUDA bench was unexpectedly slow:

- prefill: `43546.1ms`
- decode: `9249.36ms/token`
- throughput: `0.1 tok/s`
- GPU fwd: `7758.175ms`
- LM-head: `1527.103ms`

Root cause at that point:

- CUDA Q4/Q6 paths dequantized weights on CPU.
- They uploaded expanded f32 matrices to GPU for each matvec.
- LM-head was also not using `lm_head_q4.bin` in the bench path.

## Completed OpenSpec Changes

### `cuda-resident-q4k-matvec`

Status: complete, committed and pushed as `67ff38c`.

What it did:

- Added direct packed Q4_K CUDA matvec kernel:
  - `crates/larql-compute/src/cuda/q4k_direct.rs`
- `CudaBackend::q4k_matvec` now uses direct packed Q4_K kernel by default.
- Preserved old host-dequant fallback:

```bash
LARQL_CUDA_Q4K_HOST_DEQUANT=1
```

- CUDA decode now routes Q4_K projections through quant matvec.
- Added Q4_K direct parity/fallback/decode tests.

Validation that passed:

```bash
openspec validate cuda-resident-q4k-matvec --strict
cargo fmt --all --check
LARQL_CUDA_AVAILABLE=1 cargo test -p larql-compute --features cuda --test test_cuda_q4 -- --test-threads=1 --nocapture
LARQL_CUDA_AVAILABLE=1 cargo test -p larql-compute --features cuda --test test_cuda_decode -- --test-threads=1 --nocapture
cargo build --release -p larql-cli --features cuda
make traceability-check
```

Benchmark after this change:

- prefill: `20192.6ms`
- decode: `5102.07ms/token`
- throughput: `0.2 tok/s`
- GPU fwd: `3600.307ms`
- LM-head: `1510.047ms`

Conclusion: direct Q4_K helped, but LM-head was still wrong/slow.

### `cuda-q4k-device-cache`

Status: complete, committed and pushed as `f1a24ab`.

What it did:

- Added backend-local packed Q4_K device cache in:
  - `crates/larql-compute/src/cuda/backend.rs`
- Q4_K direct matvec now reuses cached `CudaSlice<u8>`.
- Added Q6_K dequantized f32 device cache:
  - Q6_K now dequantizes once, uploads f32 once, then reuses device f32 buffer.
  - Debug fallback:

```bash
LARQL_CUDA_Q6K_HOST_DEQUANT=1
```

- Added cached-device GEMV helper:
  - `crates/larql-compute/src/cuda/matmul.rs`
  - `gemv_device_w`
- Fixed `larql bench` Q4K path to load `lm_head_q4.bin`:
  - `crates/larql-cli/src/commands/primary/bench_cmd.rs`
- Added tests:
  - `q4k_matvec_reuses_device_cache`
  - `q6k_matvec_reuses_device_cache`
- Added temporary-ish diagnostic env vars in Q4_K direct matvec:

```bash
LARQL_CUDA_Q4K_TRACE=1
LARQL_CUDA_Q4K_TRACE_MIN_ROWS=10000
```

These print large Q4_K call timings. They are useful but may be worth cleaning up or converting to proper tracing later.

Validation that passed:

```bash
openspec validate cuda-q4k-device-cache --strict
LARQL_CUDA_AVAILABLE=1 cargo test -p larql-compute --features cuda --test test_cuda_q4 -- --test-threads=1 --nocapture
cargo build --release -p larql-cli --features cuda
make traceability-check
```

Final benchmark:

```bash
LARQL_CUDA_AVAILABLE=1 ./target/release/larql bench output/gemma-3-4b-it-vindex --backends cuda --tokens 20 --warmup 3 --verbose
```

Result:

- prefill: `1155.1ms`
- decode: `162.72ms/token`
- throughput: `6.1 tok/s`
- GPU fwd: `160.820ms`
- LM-head: `1.888ms`

This is a major improvement from the original `9249.36ms/token`, but still slower than where a mature CUDA backend should be.

## Key Findings From Diagnostics

1. LM-head was not using Q4_K until bench loaded `lm_head_q4.bin`.
   After fixing this, LM-head dropped from about `1510ms` to about `2-3ms`.

2. Q4_K cached direct kernel is no longer the major bottleneck.
   Trace showed cached gate/up Q4_K calls around `0.09-0.10ms` each.

3. Q6_K down projection was the real remaining artificial bottleneck.
   Caching dequantized Q6_K f32 device buffers moved decode from about `3640ms/token` to `162.72ms/token`.

4. The remaining bottleneck is now real forward-pass work and/or residual avoidable host round trips, not LM-head or repeated quant-weight upload.

## Current OpenSpec Status

Relevant completed changes:

- `cuda-resident-q4k-matvec`: `17/17`, complete.
- `cuda-q4k-device-cache`: `13/13`, complete.

Relevant still-open changes:

- `cuda-oxide-migration`: `33/34`, only burn-in remains.
- Several older CUDA-related changes are still listed as in-progress (`cuda-f32-baseline`, `cuda-q4-matvec`, `cuda-fused-attention`), but their effective work may already have landed in newer commits. Do not blindly continue them without checking tasks and current code.

Use:

```bash
openspec list --json
openspec instructions apply --change <change> --json
```

## Suggested Next Work

Start by profiling the remaining `~160ms/token` GPU forward stage.

Likely next targets:

1. Reduce per-matvec host readback/sync in CUDA decode.
   Current quant matvec APIs return `Vec<f32>`, forcing device-to-host after each projection. Decode then sends the next input back to GPU. This is probably now the biggest architectural problem.

2. Fuse or keep device-resident FFN path:
   - gate/up Q4_K -> activation -> down Q6_K currently likely crosses host between stages.
   - A fused CUDA FFN path should keep gate/up outputs and activation on GPU, then run Q6_K/f32 down on GPU.

3. Device-resident QKV/O path:
   - QKV still has mixed paths and may dequant/cache differently.
   - Inspect `crates/larql-compute/src/cuda/decode.rs`.

4. Cleanup/decide on `LARQL_CUDA_Q4K_TRACE` diagnostics.
   They are useful but noisy.

5. Consider archiving completed OpenSpec changes after review:

```bash
openspec archive cuda-resident-q4k-matvec
openspec archive cuda-q4k-device-cache
```

Only archive if the project wants these merged into canonical specs now.

## Files To Inspect First

- `crates/larql-compute/src/cuda/backend.rs`
- `crates/larql-compute/src/cuda/q4k_direct.rs`
- `crates/larql-compute/src/cuda/quant_matvec.rs`
- `crates/larql-compute/src/cuda/matmul.rs`
- `crates/larql-compute/src/cuda/decode.rs`
- `crates/larql-cli/src/commands/primary/bench_cmd.rs`
- `crates/larql-compute/tests/test_cuda_q4.rs`
- `openspec/changes/cuda-resident-q4k-matvec/`
- `openspec/changes/cuda-q4k-device-cache/`

## Commands To Reproduce

Build:

```bash
cargo build --release -p larql-cli --features cuda
```

Focused tests:

```bash
LARQL_CUDA_AVAILABLE=1 cargo test -p larql-compute --features cuda --test test_cuda_q4 -- --test-threads=1 --nocapture
```

Benchmark:

```bash
LARQL_CUDA_AVAILABLE=1 ./target/release/larql bench output/gemma-3-4b-it-vindex --backends cuda --tokens 20 --warmup 3 --verbose
```

Optional Q4_K trace:

```bash
LARQL_CUDA_AVAILABLE=1 \
LARQL_CUDA_Q4K_TRACE=1 \
LARQL_CUDA_Q4K_TRACE_MIN_ROWS=10000 \
./target/release/larql bench output/gemma-3-4b-it-vindex --backends cuda --tokens 2 --warmup 0 --verbose
```

OpenSpec validation:

```bash
openspec validate cuda-q4k-device-cache --strict
openspec validate cuda-resident-q4k-matvec --strict
make traceability-check
```

## Prompt To Continue

Continue optimizing the CUDA attention + KV-cache inference backend from the current clean `main`.

Do not restart the Q4_K or Q6_K work from scratch. The latest pushed state already has:

- direct packed Q4_K CUDA matvec,
- Q4_K packed device cache,
- Q6_K dequantized f32 device cache,
- `larql bench` loading `lm_head_q4.bin`,
- benchmark at `162.72ms/token`, `6.1 tok/s`.

Your next goal is to reduce the remaining `GPU fwd ~160ms/token` by identifying and removing device-host-device round trips in `cuda::decode`, especially around FFN gate/up/activation/down and QKV/O projections. Use OpenSpec for any behavior changes, run focused GPU tests, run the real benchmark, commit signed changes, and push to `main` when green.
