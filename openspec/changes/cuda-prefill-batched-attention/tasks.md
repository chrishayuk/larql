# cuda-prefill-batched-attention — tasks

## 1. Two new NVRTC kernels

- [x] 1.1 `kv_cache_write_seq_f32` in
      `crates/larql-compute/src/cuda/attn.rs`. One CUDA block per
      `(seq_pos, kv_head)`, 256 threads. Writes RoPE-rotated K
      and raw V to `cache[base_pos + sp, kvh, :]`.
- [x] 1.2 `fused_prefill_attention_f32` —
      `grid_dim = (num_q_heads, seq_len, 1)`, `pos = base_pos +
      blockIdx.y`. Q-RoPE pre-rotation hoist (same pattern as
      `cuda-attn-rope-hoist`). Causal score loop reads
      `k_cache[0..pos+1]`.

## 2. Rust-side wrapper

- [x] 2.1 `attn::fused_prefill_attention_seq_device` shipped:
      compiles both kernels lazily, dispatches in sequence,
      returns the `[seq_len, q_dim]` output as
      `CudaSlice<f32>`.

## 3. Decode wiring

- [x] 3.1 `decode::prefill_q4_seq_device` calls the new
      function once per layer in place of the per-position
      loop.
- [x] 3.2 `LARQL_CUDA_PREFILL_BATCHED_ATTN=0` reverts to the
      per-position loop (back-out path).

## 4. Tests

- [x] 4.1 `decode_token_phase1_matches_host_fallback` still
      passes — the prefill path is invoked indirectly via
      `prefill_q4` → `decode_token` for synthetic 1-token
      decode tests, and the bench shows greedy parity (19/20
      tokens before EOS, same as host-fallback control).
      A dedicated batched-vs-per-position parity test was
      deferred — the bench is the integration gate.

## 5. Bench gate

- [x] 5.1 Bench measured: `prefill 97.3 ms` (≤ 100 ✓),
      `attn 0.86 ms` (≤ 5 ✓, 26× drop),
      `decode 10.36 ms/tok` (≤ 11 ✓).
- [x] 5.2 No miss; profile recorded in proposal.md.

## 6. Documentation + archive

- [x] 6.1 Bench numbers in proposal.md.
- [x] 6.2 `LARQL_CUDA_PREFILL_BATCHED_ATTN=0` documented.
- [ ] 6.3 Archive together with `cuda-prefill-batched-q4k`
      once their consolidated bench progression doc lands
      (deferred follow-up).
