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
Added a private `GgufPackedExpertLayer` bridge plus `read_deepseek2_packed_expert_layer()`. The bridge consumes the DeepSeek2/Kimi role classifier, looks up one layer's packed gate/up/down expert tensors, dispatches each tensor by GGUF dtype through the existing F32/Q4_K/Q6_K packed slice readers, and returns one expert's gate/up/down matrices as conventional `[rows, cols]` arrays. Real Kimi smoke still stops at the intentional unsupported GGUF streaming boundary; the next blocker is wiring these per-expert matrices into browse extraction artifact writing.

## Next blocker: stream Kimi packed gate vectors into a vindex artifact writer

### Problem
Browse extraction needs to move from per-expert readers to actual on-disk vindex artifacts. The first artifact seam is `gate_vectors.bin` plus a `VindexLayerInfo`: for one DeepSeek2/Kimi packed layer, stream each expert's gate matrix in expert order and report layer offset/length/feature counts without loading the full packed tensor.

### RED plan
1. Add a synthetic one-shard DeepSeek2 GGUF fixture with `blk.0.ffn_gate_exps.weight` as a tiny F32 packed tensor with dims `[4, 2, 2]`.
2. Call a wished-for `write_deepseek2_packed_gate_vectors(&mut writer, &catalog, &layout, 0, offset, dtype)`.
3. Assert it writes both expert slices in order and returns `VindexLayerInfo { num_experts: Some(2), num_features_per_expert: Some(2), num_features: 4, offset, length }`.

### GREEN plan
1. Look up the classified packed gate tensor for the layer.
2. Infer expert count and features per expert from Kimi packed dims `[cols, rows, experts]`.
3. Iterate experts, dispatch each gate slice by GGUF dtype, write encoded floats to the provided writer, and return the matching `VindexLayerInfo`.

### Acceptance criteria
- [x] RED gate-writer test fails before implementation.
- [x] GREEN gate-writer test passes.
- [x] Existing packed-slice/bridge/GGUF tests still pass.
- [x] Real Kimi smoke still returns the same unsupported-layout diagnostic until the full extraction phase calls the writer.

### Result
Added `write_deepseek2_packed_gate_vectors()`, a private artifact-writer seam that consumes the DeepSeek2/Kimi role classifier, infers packed dims `[cols, rows, experts]`, streams each expert gate slice through the dtype-dispatch reader, writes encoded floats to a caller-provided writer, and returns the corresponding `VindexLayerInfo` with expert counts and byte offset/length. This gets GGUF/Kimi one step closer to `gate_vectors.bin`; the remaining blocker is calling this from the real GGUF extraction branch and adding corresponding down/up artifact seams.

## Next blocker: route real GGUF extraction through the packed gate writer

### Problem
`build_vindex_streaming()` still stops all GGUF inputs at the generic unsupported preflight boundary. We need the real GGUF branch to route DeepSeek2/Kimi catalogs into a bounded gate-vector phase, while protecting real Kimi smokes from accidentally writing enormous gate artifacts before the rest of browse extraction is budgeted.

### RED plan
1. Add a synthetic GGUF model directory with one DeepSeek2 packed gate tensor.
2. Call a wished-for private `build_gguf_streaming(&catalog, output_dir, dtype, callbacks)` route.
3. Assert it writes `gate_vectors.bin` for the tiny fixture, then returns the next explicit unsupported blocker: embeddings/down-meta wiring after gate vectors.

### GREEN plan
1. In `build_vindex_streaming()`, branch `WeightSource::Gguf(catalog)` before safetensors mmap setup.
2. Implement `build_gguf_streaming()` for DeepSeek2/Kimi gate phase:
   - classify layout;
   - estimate gate output size and refuse huge real Kimi writes with an actionable unsupported diagnostic;
   - for small/bounded catalogs, call `write_deepseek2_packed_gate_vectors()` per layer;
   - return the next unsupported blocker after writing gate vectors.
3. Keep non-DeepSeek2 GGUFs on the existing unsupported diagnostic.

### Acceptance criteria
- [x] RED real-branch test fails before implementation.
- [x] GREEN real-branch test passes.
- [x] Existing packed-slice/bridge/gate-writer/GGUF tests still pass.
- [x] Real Kimi smoke advances to the new bounded gate-budget/next-blocker diagnostic without large artifact writes.

### Result
Added the first real GGUF extraction branch in `build_vindex_streaming()`: GGUF sources now dispatch to `build_gguf_streaming()` instead of stopping at generic preflight. For DeepSeek2/Kimi catalogs, the branch classifies packed expert roles, estimates the gate-vector output size, refuses enormous real Kimi writes above a conservative in-process budget, and for bounded synthetic catalogs writes `gate_vectors.bin` through `write_deepseek2_packed_gate_vectors()` before returning the next explicit unsupported blocker (`embeddings/down_meta artifact wiring remains pending`). The real Kimi smoke now advances from generic unsupported to a safe gate-vector budget diagnostic without creating a huge artifact.


## Next blocker: compact GGUF gate manifest for over-budget Kimi gates

### Problem
Real Kimi Q4_K_M reaches the DeepSeek2 GGUF gate phase, but a naive dense `gate_vectors.bin` materialization estimates at hundreds of GB. Browse extraction needs a compact, mmap-friendly description of the original GGUF packed gate tensors so extraction can advance without writing the dense artifact.

### RED plan
1. Add a synthetic DeepSeek2 GGUF fixture whose packed gate dims make the dense gate estimate exceed the 128 MiB in-process budget.
2. Call `build_gguf_streaming()`.
3. Assert it does **not** create `gate_vectors.bin`, but does write `gguf_gate_manifest.json` with architecture, dense estimate, source tensor, dims, rows/cols/experts, and offset metadata.

### GREEN plan
1. Add a centralized `GGUF_GATE_MANIFEST_JSON` filename constant.
2. Add a compact manifest writer that records original shard/tensor geometry and byte offsets instead of materializing dense gate vectors.
3. In the over-budget DeepSeek2/Kimi branch, write the manifest and return the next unsupported blocker instead of failing before creating any useful artifact.

### Acceptance criteria
- [x] RED compact-manifest test fails before implementation.
- [x] GREEN compact-manifest test passes.
- [x] Existing GGUF real-branch/gate-writer tests still pass.
- [x] Real Kimi smoke writes a small `gguf_gate_manifest.json`, does not write `gate_vectors.bin`, and advances the diagnostic past the dense-gate budget blocker.

### Result
Added a compact over-budget GGUF gate manifest path. When DeepSeek2/Kimi packed gate tensors would exceed the dense `gate_vectors.bin` budget, `build_gguf_streaming()` now writes `gguf_gate_manifest.json` containing the original shard/tensor geometry and dense-size estimate, then returns the next explicit blocker (`embeddings/down_meta wiring remains pending`). The real Kimi smoke produced a ~26 KiB manifest with 60 gate layers, preserved the 676,457,349,120-byte dense estimate, and did not create `gate_vectors.bin`.


## Next blocker: GGUF embeddings before down-meta wiring

### Problem
After the compact gate manifest, the GGUF branch still had no browse-level embeddings artifact. Browse queries need embeddings; real Kimi embeddings are also large enough that dense materialization should stay budgeted.

### RED plan
1. Add a synthetic two-shard DeepSeek2 GGUF fixture with:
   - `token_embd.weight` as a tiny F32 `[hidden, vocab]` tensor.
   - one tiny packed gate tensor so the branch reaches the existing next blocker.
2. Call `build_gguf_streaming()`.
3. Assert the branch writes `embeddings.bin` before returning the current down-meta blocker, and that the bytes match the original embedding values.

### GREEN plan
1. Add a GGUF embedding tensor lookup (`token_embd.weight`, `model.embed_tokens.weight`, etc.).
2. For bounded embeddings, read/dequantize the 2D GGUF tensor and write `embeddings.bin` in the requested storage dtype.
3. For over-budget embeddings, write `gguf_embeddings_manifest.json` instead of creating a huge dense file.
4. Keep down-meta generation as the next explicit unsupported blocker.

### Acceptance criteria
- [x] RED embeddings test fails before implementation because `embeddings.bin` is absent.
- [x] GREEN embeddings test passes.
- [x] Existing GGUF branch/gate tests still pass.
- [x] Real Kimi smoke writes `gguf_embeddings_manifest.json` and `gguf_gate_manifest.json`, does not write dense `embeddings.bin` or `gate_vectors.bin`, and advances to the remaining down-meta blocker.

### Result
Added GGUF embedding wiring for the DeepSeek2/Kimi branch. Small F32 GGUF embeddings now materialize to `embeddings.bin`; over-budget real Kimi embeddings write a compact `gguf_embeddings_manifest.json` with tensor/source geometry instead. The real Kimi smoke produced an embeddings manifest for `token_embd.weight` (`vocab_size=163840`, `hidden_size=7168`, dense estimate `2348810240`) plus the existing gate manifest, while avoiding both dense `embeddings.bin` and dense `gate_vectors.bin`. The remaining blocker is now down-meta generation/manifesting from packed down projections.


## Next blocker: GGUF down_meta generation/manifesting

### Problem
After GGUF embeddings, the DeepSeek2/Kimi branch still did not produce `down_meta.bin` for small browse fixtures or a compact `gguf_down_meta_manifest.json` for real Kimi. Without this, browse-level feature labels cannot be loaded or deferred safely.

### RED plan
1. Add a synthetic three-shard DeepSeek2 GGUF fixture with:
   - tiny `token_embd.weight` dims `[hidden=2, vocab=3]`;
   - tiny packed gate tensor so the branch reaches gate output;
   - tiny packed down tensor `blk.0.ffn_down_exps.weight` dims `[features=2, hidden=2, experts=1]`.
2. Run the real GGUF branch helper.
3. Assert `down_meta.bin` exists and its binary header/records contain the expected top token ids from the embedding × down-projection dot products.

### GREEN plan
1. Add a compact `gguf_down_meta_manifest.json` artifact for over-budget real Kimi down-meta work.
2. For bounded synthetic fixtures, read dense embeddings and packed down slices, compute top-k token ids per down feature, and write the existing binary `down_meta.bin` format.
3. Route the GGUF extraction branch through down-meta before gate-vector/manifest handling.
4. Keep real Kimi bounded by writing the manifest rather than materializing the impossible dense down-meta projection.

### Acceptance criteria
- [x] RED failed because `down_meta.bin` was absent.
- [x] GREEN synthetic test writes `down_meta.bin`; feature 0 maps to token id 2 and feature 1 maps to token id 1.
- [x] Real Kimi smoke writes `gguf_down_meta_manifest.json` and does not write `down_meta.bin`.
- [x] Existing GGUF focused tests still pass.

### Result
Added DeepSeek2 GGUF down-meta wiring. Small fixtures now compute down-meta from dense GGUF embeddings and packed down projections and write the existing `down_meta.bin` format. Real Kimi writes compact `gguf_down_meta_manifest.json` instead: 60 packed down layers, first layer `blk.1.ffn_down_exps.weight`, dims `[2048, 7168, 384]`, features `786432`, estimated dot ops `55415386039910400`. The smoke now has embeddings, down-meta, and gate manifests while still avoiding dense `embeddings.bin`, `down_meta.bin`, and `gate_vectors.bin` for Kimi-scale artifacts. The remaining blocker is config/tokenizer/index wiring for a loadable browse-level manifest-backed vindex.


## Next blocker: loadable GGUF browse-level vindex wiring

### Problem
After embeddings, down-meta, and gate manifest slices, the GGUF branch still returned an unsupported error and did not write `index.json` or `tokenizer.json`. That meant real Kimi produced useful manifests but was not a loadable `.vindex` for LQL smoke tests.

### RED plan
1. Add a synthetic DeepSeek2 GGUF browse fixture with tiny embeddings, gate, and down tensors.
2. Call the GGUF streaming branch with a real tiny tokenizer and wished-for metadata parameters.
3. Assert extraction succeeds, writes `index.json`, `tokenizer.json`, `embeddings.bin`, `down_meta.bin`, and `gate_vectors.bin`, and can be loaded by `VectorIndex::load_vindex`.

### GREEN plan
1. Thread tokenizer, model name, and down-top-k into `build_gguf_streaming`.
2. Write tokenizer JSON for GGUF outputs using the same tokenizer serialization as the safetensors path.
3. Synthesize browse-level `index.json` from the GGUF catalog/layout:
   - family `deepseek2`;
   - num layers from max packed layer + 1;
   - hidden/vocab from `token_embd.weight` or packed gate fallback;
   - per-layer feature geometry from packed gate tensors;
   - zero-length layer entries when gate vectors are manifest-backed.
4. For small fixtures, keep dense `gate_vectors.bin`/`down_meta.bin`; for real Kimi, keep compact manifest artifacts and make the directory loadable.

### Acceptance criteria
- [x] RED failed at the intended missing production seam: `build_gguf_streaming` did not accept tokenizer/model/index metadata and did not produce loadable browse artifacts.
- [x] GREEN synthetic fixture writes all browse artifacts and loads via `VectorIndex::load_vindex`.
- [x] Real Kimi extraction exits `0` and writes `index.json`, `tokenizer.json`, `gguf_embeddings_manifest.json`, `gguf_down_meta_manifest.json`, and `gguf_gate_manifest.json` without dense Kimi-scale artifacts.
- [x] LQL can `USE` the real Kimi manifest-backed vindex and execute `SELECT entity, target FROM EDGES LIMIT 10` without loader failure.

### Result
The Kimi GGUF smoke now produces a loadable manifest-backed browse vindex at `/home/bkearns/data/larql-smoke/kimi-q4km-loadable-smoke.vindex`. The real smoke exits `0`; `index.json` reports 61 layers, hidden size 7168, vocab size 163840, and 60 packed gate layer entries with zero byte lengths because gate vectors are manifest-backed. LQL load smoke exits `0` and reports the index shape; edge results are empty until manifest-backed gate/down query execution is implemented.

## Query-time manifest-backed browse execution slice

- RED: `load_vindex_uses_gguf_down_meta_manifest_for_feature_scans` showed manifest-backed Kimi browse vindexes loaded with zero gate/down metadata, so `SELECT ... FROM EDGES` returned no rows.
- GREEN: loader now attaches compact GGUF down-meta manifests and exposes layer/feature metadata without materializing `down_meta.bin`; real Kimi `SELECT entity, target FROM EDGES LIMIT 10` now returns manifest-backed rows.
- RED: `load_vindex_embedding_rows_reads_gguf_manifest_selected_tokens` covered the missing query-time embedding path for over-budget GGUF embeddings.
- GREEN: `load_vindex_embedding_rows()` can read only selected token rows from dense `embeddings.bin` or from `gguf_embeddings_manifest.json` using GGUF byte offsets and Q4_K/Q6_K block alignment checks.
- RED/GREEN: `load_vindex_uses_gguf_gate_manifest_for_bounded_gate_knn` covers bounded manifest-backed gate KNN from compact `gguf_gate_manifest.json`.
- Real smoke: `WALK "The capital of France is" TOP 10` against `/home/bkearns/data/larql-smoke/kimi-q4km-loadable-smoke.vindex` now loads tokenizer + selected embedding row, scans a bounded prefix of GGUF gate rows per layer, and emits non-empty hits with GGUF down-meta labels.
- Bound: real manifest gate scans default to the first 64 features per layer; override with `LARQL_GGUF_MANIFEST_GATE_SCAN_FEATURES=N` for deeper smoke scans without materializing dense gates.
- Remaining limitation: labels are manifest coordinates (`gguf:<tensor>:E<expert>:F<feature>`) until query-time down projection against `gguf_down_meta_manifest.json` computes semantic token top-k for selected hits.

## Selected-hit GGUF down-projection semantic labels

Status: GREEN + real smoke.

Problem: manifest-backed Kimi `WALK` could query compact gate/down manifests, but displayed only coordinate placeholder labels such as `gguf:blk.N.ffn_down_exps.weight:E0:Fk` because `down_meta.bin` is intentionally absent for real Kimi.

RED:
- Added `load_vindex_gguf_feature_meta_projects_selected_down_feature`, a tiny manifest-backed F32 fixture that stores compact embeddings plus a packed down tensor and expects selected feature metadata to be produced by projecting the selected down feature against candidate token embeddings.
- Verified the test failed first because `load_vindex_gguf_feature_meta(...)` did not exist.

GREEN:
- Added `load_vindex_gguf_feature_meta(dir, layer, feature, candidate_token_ids, top_k)`.
- It reads `gguf_down_meta_manifest.json`, loads only the selected expert slice/feature column, reads only candidate embedding rows via `gguf_embeddings_manifest.json`, and returns `FeatureMeta` top-k token ids/logits without materializing dense `down_meta.bin` or full embeddings.
- `WALK` now tries this resolver for a bounded number of selected hits and decodes token ids through the vindex tokenizer; it falls back to coordinate placeholders if resolution is unavailable.
- Bounds:
  - `LARQL_GGUF_MANIFEST_DOWN_META_TOKENS` default `256`, plus prompt token ids are always included.
  - `LARQL_GGUF_MANIFEST_DOWN_META_HITS` default `12` selected displayed hits.

Verification:
```bash
cargo fmt --all -- --check
PATH=/tmp/larql-cmake-venv/bin:$PATH cargo test -p larql-vindex load_vindex_ -- --nocapture
PATH=/tmp/larql-cmake-venv/bin:$PATH cargo check -p larql-lql
PATH=/tmp/larql-cmake-venv/bin:$PATH LARQL_GGUF_MANIFEST_DOWN_META_TOKENS=64 LARQL_GGUF_MANIFEST_DOWN_META_HITS=3 timeout 180s cargo run -q -p larql-cli -- lql 'USE "/home/bkearns/data/larql-smoke/kimi-q4km-loadable-smoke.vindex"; WALK "The capital of France is" TOP 10 LAYERS 1-2;'
```

Real Kimi smoke result:
- Exit `0`.
- First selected hits now show decoded tokenizer labels, e.g. `top="X" down=[X, of, ^]`, while hits beyond the configured semantic-hit budget still show coordinate placeholders.
- Log: `/tmp/larql-kimi-walk-semantic-downmeta-small.log`.

Remaining limitations:
- Semantic labels are bounded candidate-token labels, not exhaustive vocab labels.
- `SELECT`/`DESCRIBE` still primarily use coordinate fallback unless they are wired to pass a vindex path and candidate set into the resolver.
