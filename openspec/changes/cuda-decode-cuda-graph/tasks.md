# cuda-decode-cuda-graph — tasks

## 1. DecodeScratch + write-into infrastructure

- [ ] 1.1 New `crates/larql-compute/src/cuda/scratch.rs`
      module. `DecodeScratch` struct holding all the
      per-call intermediate buffers (h_dev, q_dev, etc.),
      Q8_1 scratches, and `pos_dev`.
- [ ] 1.2 `CudaBackend::ensure_decode_scratch(shape)` —
      lazy-allocate, reuse on shape match.
- [ ] 1.3 `_into` variants of every kernel wrapper used in
      decode_token_device:
      - `q4k_mmvq::matvec_device_into`
      - `q6k_mmvq::matvec_device_into`
      - `q4k_direct::matvec_device_into`
      - `elem::rms_norm_device_into`
      - `elem::silu_gate_up_device_into`
      - `elem::quantize_q8_1_device_into`
      - `attn::fused_decode_attention_device_kv_into`
      The existing return-fresh-buffer variants stay as
      thin wrappers around the `_into` form.

## 2. Device-side pos

- [ ] 2.1 Modify `FUSED_DECODE_ATTN_SRC` kernel signature:
      `int pos` → `const int* pos_dev`. Read
      `int pos = *pos_dev` at kernel entry.
- [ ] 2.2 Update `attn::fused_decode_attention_device_kv`
      to take `pos_dev: &CudaSlice<i32>` instead of
      `pos: usize`.
- [ ] 2.3 Caller updates `*pos_dev` via
      `htod_into_slice(&[pos as i32], pos_dev, 0)` before
      each call.

## 3. decode_token_device refactor

- [ ] 3.1 Refactor to use `DecodeScratch` for all
      intermediate buffers.
- [ ] 3.2 Initial htod into `scratch.h_dev`; final dtoh
      from same.
- [ ] 3.3 `LARQL_CUDA_DECODE_NO_SCRATCH=1` env var falls
      back to the legacy fresh-alloc path.

## 4. CUDA Graph capture + replay

- [ ] 4.1 `DecodeGraph` cache on `CudaBackend`, keyed by
      shape + layer-set fingerprint.
- [ ] 4.2 First decode call after scratch alloc: run
      normally + `begin_capture`/`end_capture` to record
      the graph.
- [ ] 4.3 Subsequent calls: `htod` new pos + h, then
      `graph.launch()`, then `dtoh` final h.
- [ ] 4.4 `LARQL_CUDA_DECODE_GRAPH=0` env var falls back
      to the per-call launch path.

## 5. Tests

- [ ] 5.1 `decode_token_phase1_matches_host_fallback` MUST
      pass with `LARQL_CUDA_DECODE_GRAPH=1` (the default).
- [ ] 5.2 Multi-step parity:
      `decode_token_graph_matches_per_call_over_5_steps` —
      runs 5 decode tokens with graph and again with
      `LARQL_CUDA_DECODE_GRAPH=0`; assert per-step
      max-element ≤ 1e-3.

## 6. Bench gate

- [ ] 6.1 `LARQL_CUDA_AVAILABLE=1 ./target/release/larql
      bench output/gemma-3-4b-it-vindex --backends cuda
      --tokens 20 --warmup 3 --verbose`.
- [ ] 6.2 Acceptance: `decode ms/token ≤ 7` AND `tok/s ≥
      140`. If `decode > 8.5 ms`, profile and document.

## 7. Documentation + archive

- [ ] 7.1 Final bench numbers in proposal.md.
- [ ] 7.2 Document `LARQL_CUDA_DECODE_GRAPH=0` and
      `LARQL_CUDA_DECODE_NO_SCRATCH=1` env vars.
- [ ] 7.3 Archive when acceptance cleared.
