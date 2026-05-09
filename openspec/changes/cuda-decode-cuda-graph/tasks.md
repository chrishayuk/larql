# cuda-decode-cuda-graph — tasks

## 0. Foundation (driver / event tracking)

- [x] 0.1 Add `cuda_graph_capture_replay_smoke_test` viability
      probe. Confirms cudarc 0.19's begin_capture / launch /
      end_capture / graph.launch loop works on this stack.
- [x] 0.2 Driver: switch to `ctx.new_stream()` and
      `unsafe { ctx.disable_event_tracking() }`. Required to
      avoid `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` (default
      stream rejected) and `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`
      (cross-stream `stream.wait` injected by cudarc inside the
      capture region).

## 1. DecodeScratch + write-into infrastructure

- [x] 1.1 New `crates/larql-compute/src/cuda/scratch.rs`
      module. `DecodeScratch` struct holding all the
      per-call intermediate buffers (`h`, `q`, `k`, `v`,
      `attn_out`, `attn_delta`, `attn_normed`,
      `h_attn`, `h_ffn`, `gate`, `up`, `act`, `ffn_delta`,
      `ffn_normed`), the four Q8_1 scratches
      (`h_attn_q8_1`, `attn_out_q8_1`, `h_ffn_q8_1`,
      `act_q8_1`), and `pos: CudaSlice<i32>`.
- [x] 1.2 `CudaBackend::decode_scratch: Mutex<Option<...>>`
      lazy-allocates on first use; reuses on shape match
      via `DecodeScratchShape::matches`.
- [x] 1.3 `_into` variants of every kernel wrapper used by
      the captured decode pipeline:
      - `q4k_mmvq::matvec_device_into` /
        `matvec_device_into_with_dev`
      - `q6k_mmvq::matvec_device_into` /
        `matvec_device_into_with_dev`
      - `elem::rms_norm_device_into`
      - `elem::silu_gate_up_device_into`
      - `elem::quantize_q8_1_device_into`
      - `attn::fused_decode_attention_device_kv_into`
      The existing return-fresh-buffer variants stay as
      thin wrappers around the `_into` form.

## 2. Device-side pos

- [x] 2.1 `FUSED_DECODE_ATTN_SRC` signature changed:
      `int pos` → `const int* pos_dev`. Reads
      `int pos = *pos_dev` at kernel entry.
- [x] 2.2 `fused_decode_attention_device_kv_into` takes
      `pos_dev: &CudaSlice<i32>` directly. The legacy
      wrappers `fused_decode_attention[_device][_kv]`
      keep their `usize` API by allocating a one-shot
      `pos_dev` internally.
- [x] 2.3 Captured-graph path writes `*pos_dev` via
      `stream.memcpy_htod(&[pos as i32], &mut scratch.pos)`
      before each replay.

## 3. decode_token_device refactor

- [x] 3.1 New `decode_token_device_graph_attempt` runs the
      captured pipeline using `DecodeScratch` exclusively.
      Existing `decode_token_device` stays untouched as the
      legacy back-out.
- [x] 3.2 Initial htod into `scratch.h`; final dtoh from
      same.
- [x] 3.3 `LARQL_CUDA_DECODE_GRAPH=0` env var forces the
      legacy fresh-alloc per-call path. The graph path
      also auto-falls-back when the layer set is
      unsupported (MoE, non-mmvq formats, mixed shapes,
      LayerNorm, Standard FFN, FFN-remote).

## 4. CUDA Graph capture + replay

- [x] 4.1 `DecodeGraph` (Send/Sync wrapper around
      `cudarc::CudaGraph`) cached in
      `CudaBackend::decode_graph: Mutex<Option<...>>`.
      Invalidated on scratch shape mismatch.
- [x] 4.2 Two-phase warmup: call #0 runs the pipeline
      eagerly so the cache lookups warm every weight + norm
      buffer; call #1 captures with `begin_capture` /
      `end_capture` and stores the graph.
- [x] 4.3 Subsequent calls: `htod` new pos + h, then
      `graph.launch()`, then `dtoh` final h.
- [x] 4.4 `LARQL_CUDA_DECODE_GRAPH=0` env var falls back
      to the per-call launch path.

## 5. Tests

- [x] 5.1 `decode_token_phase1_matches_host_fallback`
      passes with the default `LARQL_CUDA_DECODE_GRAPH=1`
      (the captured graph produces bit-exact-within-1e-3
      output vs the host fallback).
- [x] 5.2 Multi-step parity: the full integration suite
      (139 lib + 56 integration tests) passes with the
      graph path on for runs of 30+ tokens.

## 6. Bench gate

- [x] 6.1 `LARQL_CUDA_AVAILABLE=1 ./target/release/larql
      bench output/gemma-3-4b-it-vindex --backends cuda
      --tokens 20 --warmup 3`.
- [x] 6.2 Result: 8.52 ms / 117.4 tok/s vs legacy 9.62 ms /
      103.9 tok/s — 11% improvement. Falls short of the
      original ≤ 7 ms / ≥ 140 tok/s target; the remaining
      gap with llama-cpp-turboquant (4.40 ms / 227.5 tok/s)
      is in compute, not launch overhead. Tensor Cores
      (Q4_K via WMMA) is the next step.

## 7. Documentation + archive

- [x] 7.1 Final bench numbers documented in `proposal.md`.
- [x] 7.2 `LARQL_CUDA_DECODE_GRAPH=0` env var documented in
      `proposal.md`.
- [ ] 7.3 Archive when this change is reviewed.
