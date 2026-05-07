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

## Next blocker: classify DeepSeek2/Kimi GGUF tensor roles

### Problem
The catalog can find tensor names, but extraction still cannot tell which Kimi GGUF tensors are packed MoE gate/up/down projections, router weights, or shared expert projections. Without that role map, the streaming extractor cannot select the right tensor and slice/expert layout for each vindex output.

### RED plan
1. Add a synthetic GGUF catalog with Kimi-style names:
   - `blk.0.ffn_gate_exps.weight`
   - `blk.0.ffn_up_exps.weight`
   - `blk.0.ffn_down_exps.weight`
   - `blk.0.ffn_gate_inp.weight`
   - `blk.0.ffn_gate_shexp.weight`
2. Assert a wished-for `classify_deepseek2_layout(&catalog)` returns role maps for packed expert components, router, and shared expert components by layer.

### GREEN plan
1. Implement a header-only DeepSeek2/Kimi tensor-role classifier over catalog names.
2. Preserve dimensions in catalog entries; classification only names roles and references entries, no data reads.
3. Keep the classifier private until the streaming reader consumes it.

### Acceptance criteria
- [x] RED role-classification test fails before implementation.
- [x] GREEN role-classification test passes.
- [x] Existing GGUF catalog/preflight tests still pass.
- [x] Real Kimi smoke still returns the same unsupported-layout diagnostic.

### Result
Added a private DeepSeek2/Kimi GGUF layout classifier over the header-only catalog. It maps `blk.N.ffn_*_exps.weight` packed MoE tensors to gate/up/down expert components, maps `blk.N.ffn_gate_inp.weight` as the router, and maps `blk.N.ffn_*_shexp.weight` shared-expert projections. The classifier performs no tensor-data reads; it only classifies catalog entries by name and layer, setting up the next TDD slice for packed 3D tensor slicing/dequantization.

## Next blocker: packed 3D expert slice geometry

### Problem
The role classifier can point to packed Kimi tensors, but extraction still cannot address an individual expert inside a packed 3D tensor. Kimi GGUF stores expert projections as `[cols, rows, experts]` in GGUF dimension order, so the extractor needs a deterministic per-expert 2D geometry before it can stream/dequantize expert rows.

### RED plan
1. Add a synthetic Kimi packed expert tensor entry with dims `[7168, 2048, 384]`.
2. Assert a wished-for `packed_expert_slice(entry, expert_idx)` returns conventional 2D matrix shape `[rows=2048, cols=7168]`, expert count, element span, and element offset for the requested expert.
3. Assert the down projection dims `[2048, 7168, 384]` map to `[rows=7168, cols=2048]`.
4. Assert an out-of-range expert index returns an error.

### GREEN plan
1. Implement header-only packed expert slice geometry over `GgufTensorEntry`.
2. Do not read tensor bytes or dequantize yet.
3. Keep the slice geometry private until the streaming reader consumes it.

### Acceptance criteria
- [x] RED packed-slice test fails before implementation.
- [x] GREEN packed-slice test passes.
- [x] Existing GGUF catalog/classifier/preflight tests still pass.
- [x] Real Kimi smoke still returns the same unsupported-layout diagnostic.

### Result
Added private `PackedExpertSlice` geometry and `packed_expert_slice()` over catalog entries. It maps Kimi GGUF packed expert tensors from `[cols, rows, experts]` into conventional 2D expert matrices `[rows, cols]`, records expert count, per-expert element span, and element offset, and rejects out-of-range expert indices. This still performs no tensor-data reads or dequantization; the next slice can use this geometry to stream/dequantize one packed expert projection.

## Next blocker: mmap/dequant one packed expert slice

### Problem
The extractor can compute which elements belong to a packed expert, but it still cannot read tensor bytes from a GGUF shard and materialize one expert projection. Query smoke requires browse extraction to build gate/up/down matrices without eager loading a whole Kimi shard.

### RED plan
1. Add a miniature GGUF fixture with one F32 packed 3D tensor `blk.0.ffn_gate_exps.weight` and dims `[4, 3, 2]`.
2. Store two experts with distinct values in GGUF contiguous order.
3. Assert a wished-for `read_packed_expert_slice_f32(&catalog, name, expert_idx)` returns expert 0 and expert 1 as separate conventional `3 x 4` matrices.
4. Assert out-of-range expert indices return an error.

### GREEN plan
1. Add offset/data-offset metadata to the GGUF catalog entries.
2. Implement a minimal mmap reader for F32 packed expert slices.
3. Keep quantized Q4_K/Q6_K packed slice support as the next slice after the F32 seam is green.

### Acceptance criteria
- [x] RED packed-slice reader test fails before implementation.
- [x] GREEN packed-slice reader test passes.
- [x] Existing GGUF catalog/classifier/preflight tests still pass.
- [x] Real Kimi smoke still returns the same unsupported-layout diagnostic until quantized packed slices are wired.

### Result
Added a minimal F32-only packed expert slice reader. `GgufCatalog` now records each tensor's shard-local tensor offset and shard data offset; `read_packed_expert_slice_f32()` mmaps the containing shard, reads only the requested expert byte range, and returns a conventional `[rows, cols]` `Array2<f32>`. This proves the tensor-data slicing seam without eager whole-tensor loads. The next blocker is quantized packed slice support for Kimi's Q4_K/Q6_K tensors.
