# cuda-attn-rope-hoist — tasks

## 1. Kernel modification

- [x] 1.1 `q_rot` shared-memory region added at
      `smem + max_seq + bdim`, size `head_dim` floats.
- [x] 1.2 One-pass pre-rotation runs after `k_inv` reduction;
      each thread handles `d = tid + k * bdim`. Same
      arithmetic as the prior inline block.
- [x] 1.3 `__syncthreads()` after the pre-rotation.
- [x] 1.4 Score loop simplified — inline Q rotation block
      replaced with `float qv = q_rot[d];`. K rotation
      (only `j == pos`) unchanged.
- [x] 1.5 All three attention launch sites in `attn.rs`
      (`fused_decode_attention`, `fused_decode_attention_device`,
      `fused_decode_attention_device_kv`) extended
      `shared_mem_bytes` by `head_dim * sizeof(float)`.

## 2. Tests

- [x] 2.1 `decode_token_phase1_matches_host_fallback` passes
      (≤ 1e-3 vs host fallback, 3 decode steps).
- [x] 2.2 `fused_decode_attention_matches_cpu_reference`
      passes (existing test in `test_cuda_attn.rs`, same
      tolerance).
- [x] 2.3 Full CUDA test suite (33 files, 192 tests) passes.

## 3. Bench gate

- [x] 3.1 Bench run on dev box. Recorded:
      `decode 12.88 ms/tok`, `GPU fwd 10.898 ms`,
      `77.6 tok/s`, `prefill 141.2 ms`.
- [x] 3.2 `LARQL_CUDA_DECODE_PROFILE=1` confirms
      `attn_call: 6.35 → 3.68 ms` (–42%, –2.67 ms).
      Cleared the `≤ 4 ms` gate.
- [x] 3.3 No further profiling needed — the hoist landed
      exactly as predicted.

## 4. Documentation + archive

- [x] 4.1 Bench numbers recorded in `proposal.md`'s
      acceptance table (every metric cleared).
- [ ] 4.2 Archive:
      `openspec archive cuda-attn-rope-hoist`. Pending the
      final cuda-rotorquant-status update co-located with
      cuda-q4k-mmvq-int8.
- [x] 4.3 cuda-q4k-mmvq-int8 originally targeted ≤ 10 ms/tok
      and missed at 15.55. After this change, decode is at
      12.88 — still ≥ 10, so cuda-q4k-mmvq-int8 stays
      unarchived; the natural close-out for both is the next
      mmvq port (`cuda-q6k-mmvq`, where `proj_down` is the
      4.06 ms top bucket).
