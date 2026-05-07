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

## Next blocker: sharded GGUF tensor catalog

### Problem
The preflight knows only aggregate counts. The next extraction step needs a mmap/header-only catalog mapping GGUF tensor names to shard index, shape, tensor type, and byte offsets before any Kimi 3D expert tensor can be streamed.

### RED plan
1. Add a synthetic two-shard GGUF test where each shard has a different tensor.
2. Assert a wished-for `build_gguf_catalog(&files)` returns architecture/split metadata plus lookup entries preserving shard index, dimensions, tensor type, and 3D detection.

### GREEN plan
1. Implement a header-only catalog from `GgufFile::open` outputs.
2. Keep it private to streaming extraction for now; no tensor data reads or dequantization.
3. Reuse catalog totals in the current unsupported preflight so real Kimi diagnostics still work.

### Acceptance criteria
- [x] RED catalog test fails before implementation.
- [x] GREEN catalog test passes.
- [x] Existing GGUF preflight tests still pass.
- [x] Real Kimi smoke still returns the same actionable unsupported-layout diagnostic, now backed by catalog data.

### Result
Added a private header-only `GgufCatalog` in streaming extraction. It indexes tensor names across sharded GGUF files with shard index, dimensions, tensor type, architecture, split count, and 3D tensor totals. Existing unsupported Kimi diagnostics now reuse this catalog, which is the next seam for streaming packed 3D Kimi expert tensors without eager dequantization.
