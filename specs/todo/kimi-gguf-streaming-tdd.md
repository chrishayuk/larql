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

## Next blocker: mmap/dequant one packed Q4_K expert slice

### Problem
Real Kimi Q4_K_M packed expert tensors are quantized, not F32. The extractor needs to read only the selected expert's block-aligned Q4_K byte range, dequantize that slice, and return a conventional `[rows, cols]` matrix without decoding the whole packed tensor.

### RED plan
1. Add a tiny GGUF fixture with one packed Q4_K tensor `blk.0.ffn_gate_exps.weight` and dims `[256, 1, 2]`.
2. Store two real Q4_K blocks: expert 0 decodes to all `1.0`, expert 1 decodes to all `2.0`.
3. Assert a wished-for `read_packed_expert_slice_q4_k(&catalog, name, expert_idx)` returns the correct `1 x 256` matrix for each expert.
4. Assert a non-block-aligned expert slice is rejected with an actionable error.

### GREEN plan
1. Factor common mmap byte-range validation if useful, but keep scope minimal.
2. Implement Q4_K block-aligned slice byte mapping: `(element_offset / 256) * 144` and `(element_len / 256) * 144`.
3. Dequantize only those blocks with the existing GGML Q4_K decoder.

### Acceptance criteria
- [x] RED Q4_K packed-slice reader test fails before implementation.
- [x] GREEN Q4_K packed-slice reader test passes.
- [x] Existing F32 packed-slice and GGUF catalog/classifier/preflight tests still pass.
- [x] Real Kimi smoke still returns the same unsupported-layout diagnostic until the extractor bridge is wired.

### Result
Added `read_packed_expert_slice_q4_k()` for block-aligned packed expert slices. The reader maps expert element offsets to Q4_K block byte ranges, mmaps only the containing shard range, dequantizes those blocks with the existing GGML Q4_K decoder, and returns a conventional `[rows, cols]` matrix. Shared byte-range validation now backs both F32 and Q4_K readers. The next blocker is either Q6_K packed-slice support or the first extractor bridge that consumes the classifier + slice readers for one Kimi layer.

## Next blocker: mmap/dequant one packed Q6_K expert slice

### Problem
Kimi Q4_K_M mixes can store down projections and other high-precision tensors as Q6_K. The extractor needs the same selected-expert, block-aligned mmap/dequant path for Q6_K before a browse-level Kimi vindex can materialize reliably.

### RED plan
1. Add a tiny GGUF fixture with one packed Q6_K tensor `blk.0.ffn_down_exps.weight` and dims `[256, 1, 2]`.
2. Store two real Q6_K blocks: expert 0 decodes to all `1.0`, expert 1 decodes to all `2.0`.
3. Assert a wished-for `read_packed_expert_slice_q6_k(&catalog, name, expert_idx)` returns the correct `1 x 256` matrix for each expert.
4. Assert a non-block-aligned expert slice is rejected with an actionable error.

### GREEN plan
1. Reuse the shared mmap byte-range validation from the F32/Q4_K readers.
2. Implement Q6_K block-aligned slice byte mapping: `(element_offset / 256) * 210` and `(element_len / 256) * 210`.
3. Dequantize only those blocks with the existing GGML Q6_K decoder.

### Acceptance criteria
- [x] RED Q6_K packed-slice reader test fails before implementation.
- [x] GREEN Q6_K packed-slice reader test passes.
- [x] Existing F32/Q4_K packed-slice and GGUF catalog/classifier/preflight tests still pass.
- [x] Real Kimi smoke still returns the same unsupported-layout diagnostic until the extractor bridge is wired.

### Result
Added `read_packed_expert_slice_q6_k()` for block-aligned packed expert slices. The Q4_K and Q6_K readers now share block-aligned mmap byte-range validation/dequant shaping; Q6_K maps expert element offsets to `Q6_K_BLOCK_BYTES` ranges, reads only the selected expert blocks from the containing shard, dequantizes with the existing GGML Q6_K decoder, and returns a conventional `[rows, cols]` matrix. The next blocker is the first extractor bridge that consumes the DeepSeek2 classifier plus F32/Q4_K/Q6_K packed slice readers for one Kimi layer.

## Next blocker: read one DeepSeek2/Kimi packed expert layer

### Problem
The extractor now has independent readers for F32, Q4_K, and Q6_K packed expert slices, but no bridge that consumes the DeepSeek2 role classifier and returns the gate/up/down projections for a single layer/expert. Without this bridge, the GGUF extractor still cannot materialize layer data for browse-level vindex construction.

### RED plan
1. Add a synthetic three-shard DeepSeek2 GGUF fixture with packed expert role tensors for one layer:
   - `blk.0.ffn_gate_exps.weight` as F32
   - `blk.0.ffn_up_exps.weight` as Q4_K
   - `blk.0.ffn_down_exps.weight` as Q6_K
2. Classify the catalog with `classify_deepseek2_layout(&catalog)`.
3. Assert a wished-for `read_deepseek2_packed_expert_layer(&catalog, &layout, 0, 1)` returns the expert-1 gate/up/down matrices and dispatches by tensor type.

### GREEN plan
1. Implement a small private bridge struct holding gate/up/down `Array2<f32>` matrices.
2. Look up role tensor names via `GgufDeepseek2Layout::packed_experts`.
3. Dispatch each tensor to the existing F32/Q4_K/Q6_K packed slice readers by GGUF tensor type.

### Acceptance criteria
- [x] RED bridge test fails before implementation.
- [x] GREEN bridge test passes.
- [x] Existing F32/Q4_K/Q6_K packed-slice and GGUF catalog/classifier/preflight tests still pass.
- [x] Real Kimi smoke still returns the same unsupported-layout diagnostic until the full extractor phase is wired.

### Result
Added a private `GgufPackedExpertLayer` bridge plus `read_deepseek2_packed_expert_layer()`. The bridge consumes the DeepSeek2/Kimi role classifier, looks up one layer's packed gate/up/down expert tensors, dispatches each tensor by GGUF dtype through the existing F32/Q4_K/Q6_K packed slice readers, and returns one expert's gate/up/down matrices as conventional `[rows, cols]` arrays. Real Kimi smoke still stops at the intentional unsupported GGUF streaming boundary; the next blocker is wiring this bridge into the actual browse extraction phase so it can write gate/up/down vindex artifacts.
