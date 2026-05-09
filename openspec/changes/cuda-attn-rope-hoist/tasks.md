# cuda-attn-rope-hoist — tasks

## 1. Kernel modification

- [ ] 1.1 In `crates/larql-compute/src/cuda/attn.rs`'s
      `FUSED_DECODE_ATTN_SRC` NVRTC string, add a `q_rot`
      shared-memory region of size `head_dim` floats (located
      at `smem + max_seq + block_dim`).
- [ ] 1.2 After the `q_inv` reduction and before the K/V
      append + score loop, add a one-pass pre-rotation that
      writes `q_rot[d]` for `d ∈ [0, head_dim)`. Each thread
      handles `d = tid + k * block_dim` for `k = 0, 1, …`
      until exhausted.
- [ ] 1.3 Add `__syncthreads()` between the pre-rotation and
      the score loop.
- [ ] 1.4 Inside the score loop, replace the inline Q
      rotation block with `float qv = q_rot[d];`. The K
      rotation logic (only when `j == pos`) is unchanged.
- [ ] 1.5 Update the Rust-side launch config in
      `fused_decode_attention_device_kv` (and the older
      `fused_decode_attention_device` and
      `fused_decode_attention` if they share the launch
      config builder) to extend `shared_mem_bytes` by
      `head_dim * sizeof(float)`.

## 2. Tests

- [ ] 2.1 `decode_token_phase1_matches_host_fallback` MUST
      still pass with `≤ 1e-3` tolerance. Run with
      `LARQL_CUDA_AVAILABLE=1`.
- [ ] 2.2 `fused_decode_attention_matches_cpu_reference` (in
      `test_cuda_attn.rs`) MUST still pass. Same tolerance.
- [ ] 2.3 No new tests required — the change is numerically
      equivalent to the prior code.

## 3. Bench gate

- [ ] 3.1 Run the standard bench:
      `LARQL_CUDA_AVAILABLE=1 ./target/release/larql bench
      output/gemma-3-4b-it-vindex --backends cuda --tokens 20
      --warmup 3 --verbose`. Record `decode ms/token`,
      `GPU fwd ms/token`, `tok/s`.
- [ ] 3.2 Run with `LARQL_CUDA_DECODE_PROFILE=1` to confirm
      `attn_call` dropped. Acceptance:
      `attn_call ≤ 4 ms` AND `decode ms/token ≤ 13`.
- [ ] 3.3 If `attn_call` did not drop by ≥ 1 ms, profile with
      `nvprof`/`nsys` to determine the actual bottleneck.
      Document in the proposal before merging.

## 4. Documentation + archive

- [ ] 4.1 Record final bench numbers in `proposal.md` (the
      "Acceptance bar" table updated with `actual` column).
- [ ] 4.2 If acceptance hit, archive:
      `openspec archive cuda-attn-rope-hoist`.
- [ ] 4.3 If `decode ms/tok` is now ≤ 10 (the
      `cuda-q4k-mmvq-int8` original target), update
      `cuda-q4k-mmvq-int8`'s archive note accordingly.
