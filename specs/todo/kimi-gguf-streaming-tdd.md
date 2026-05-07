# Kimi GGUF streaming extraction TDD

## Problem
`larql extract <kimi-gguf-dir>` gets past `tokenizer.json` but fails with `no safetensors files`, even when the directory contains sharded GGUF weights. The existing `convert gguf-to-vindex` path opens a single GGUF and eagerly dequantizes huge tensors, which is not viable for Kimi K2 Q4_K_M.

## Root-cause hypothesis
The extraction path unconditionally calls `build_vindex_streaming`, whose first weight-source discovery only searches for `.safetensors`. GGUF directories are therefore misclassified as missing weights. Kimi's GGUF layout is also not immediately extractable by the current GGUF loader because it is `deepseek2`, split across 13 shards, and contains packed 3D tensors.

## RED plan
1. Add a unit test that a directory containing `Q4_K_M/*.gguf` is classified as a GGUF source, not `NoSafetensors`.
2. Add a unit test that a Kimi-like GGUF source produces a clear unsupported streaming error mentioning `GGUF`, `deepseek2`, `split`, and `3D`, instead of `no safetensors`.

## GREEN plan
1. Add GGUF source discovery/preflight before the existing safetensors empty check.
2. Open GGUF headers only; do not dequantize tensors.
3. Return a specific unsupported error for GGUF streaming until sharded/pcked tensor extraction is implemented.

## Acceptance criteria
- [x] Focused RED tests fail before production changes.
- [x] Focused GREEN tests pass.
- [x] `cargo test -p larql-vindex gguf -- --nocapture` passes.
- [x] CLI smoke reports the GGUF/Kimi unsupported preflight rather than `no safetensors`.
- [x] Default workspace lib/bin tests still pass.

## Result
The tokenizer blocker remains fixed via sidecar. `larql extract` now performs header-only GGUF source discovery before falling through to safetensors, recognizes nested sharded GGUF directories, and returns a Kimi/DeepSeek2-specific unsupported-layout diagnostic with file/split/tensor/3D counts. This avoids the misleading `no safetensors` error and avoids the eager single-shard dequantization path.
