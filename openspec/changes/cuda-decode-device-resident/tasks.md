# cuda-decode-device-resident — tasks

## Phase 1 — Device-resident projections

### 1. Backend API

- [ ] 1.1 `CudaBackend::q4k_matvec_device(weight, x_device,
      rows, cols) -> Result<CudaSlice<f32>, CudaInitError>` —
      mirrors `q4k_matvec` but takes a device input slice and
      returns a device output slice. The existing
      `q4k_matvec(&self, ..., x: &[f32], ...)` becomes a thin
      `htod → q4k_matvec_device → dtoh` wrapper.
- [ ] 1.2 Same for `q6k_matvec_device`, `q4kf_matvec_device`,
      and `f32_gemv_device`. The existing `gemv_device_w` helper
      in `cuda/matmul.rs` is the template.
- [ ] 1.3 Cached weight handle on the matvec — `q4k_matvec_device`
      consults the existing per-backend Q4_K device cache (added
      in `cuda-q4k-device-cache`) so the first call uploads,
      subsequent calls reuse.

### 2. fused_decode_attention device entry point

- [ ] 2.1 `attn::fused_decode_attention_device(q_dev, k_dev,
      v_dev, kv_k_dev, kv_v_dev, …) -> AttnOut<CudaSlice<f32>>` —
      symmetric to the existing host entry. Internally the kernel
      already runs on device; this just changes the input/output
      shape.
- [ ] 2.2 Keep the host-input variant as a wrapper that does
      H2D for K/V slabs and D2H for the result.

### 3. decode_token rewrite

- [ ] 3.1 Split the existing function into
      `decode_token_host_fallback` (current code, unchanged) and
      `decode_token_device_resident` (new path). The trait impl
      dispatches based on `LARQL_CUDA_DECODE_HOST_FALLBACK` or a
      `Cell<bool>` toggle on the backend.
- [ ] 3.2 In the new path: hold `h_dev: CudaSlice<f32>` across
      layers. Each projection chains
      `q4k_matvec_device → q4k_matvec_device → …`. Only `h_post_attn`
      and final `h` cross D2H per layer (Phase 1 still does CPU
      rms_norm/silu/add).
- [ ] 3.3 Result `Vec<f32>` from a single final `dtoh_sync_copy`
      after the layer loop.

### 4. Tests

- [ ] 4.1 `q4k_matvec_device_returns_same_as_host` — random Q4_K
      packed weight + random input, both paths, byte-equal output.
- [ ] 4.2 `decode_token_phase1_matches_host_fallback` — synthetic
      pipeline layer (the existing test infra builds these), one
      decode step both paths, max-element diff ≤ 1e-3.
- [ ] 4.3 `decode_q4k_gemma3_20_tokens_match_host` —
      `#[ignore]`'d, gated on `LARQL_CUDA_AVAILABLE=1` and the
      real vindex. Asserts greedy token ids agree across 20 steps.

### 5. Bench gate

- [ ] 5.1 Run `larql bench output/gemma-3-4b-it-vindex --backends
      cuda --tokens 20 --warmup 3 --verbose`. Record
      `decode ms/token`, `GPU fwd ms/token`, `tok/s`.
- [ ] 5.2 Acceptance: `decode ms/token ≤ 100` AND
      `GPU fwd ms/token ≤ 95`. If miss > 25%, abort and document
      in the PR description.

## Phase 2 — GPU rms_norm / silu / add

### 6. New kernels

- [ ] 6.1 `cuda/kernels/rms_norm.cu` (or NVRTC-string in
      `cuda/matmul.rs`-style module): single-block
      reduction-then-scale; 1024 threads.
- [ ] 6.2 `cuda/kernels/silu_gate_up.cu`: element-wise; one launch
      over `inter` elements; supports the existing `Activation`
      enum (Silu, Gelu, …).
- [ ] 6.3 `cuda/kernels/add_in_place.cu`: trivial element-wise.
- [ ] 6.4 NVRTC compile + cache them via the existing cudarc
      cache directory pattern. PTX persists across server boots.

### 7. Wire the new kernels into decode_token_device_resident

- [ ] 7.1 Replace `rms_norm_vec(...)` calls inside the device path
      with `rms_norm_vec_device(...)`. Drop the per-layer D2H +
      H2D pair around the residual adds.
- [ ] 7.2 Replace `activate(gate, up, ...)` with
      `silu_gate_up_device`.
- [ ] 7.3 Replace `add_in_place(h_post, delta)` with
      `add_in_place_device`.
- [ ] 7.4 Final `h` is the only D2H per token after Phase 2.

### 8. Tests + bench

- [ ] 8.1 `rms_norm_vec_device_matches_cpu` — random input,
      max-element ≤ 1e-3.
- [ ] 8.2 `silu_gate_up_device_matches_cpu` — random input, ≤ 1e-3.
- [ ] 8.3 `add_in_place_device_matches_cpu` — bit-equal.
- [ ] 8.4 Re-run the Phase 2 parity test on the synthetic decode
      pipeline (same 1e-3 bound).
- [ ] 8.5 Bench gate: `decode ms/token ≤ 80` AND
      `GPU fwd ms/token ≤ 75`.

## Phase 3 — Device-resident KV cache

### 9. Type swap

- [ ] 9.1 `CudaKvLayer::k: Vec<f32>` → `k: CudaSlice<f32>`. Same
      for `.v`. Allocate once at `preallocate_kv_cache_per_layer`
      time.
- [ ] 9.2 `populate_kv_layer(layer, k_data: &[f32], …)` becomes a
      `htod_sync_copy_into` of the K/V slabs into the
      pre-allocated `CudaSlice`s.

### 10. fused_decode_attention_device contract

- [ ] 10.1 Accepts `&CudaSlice<f32>` for K/V cache instead of
      `&[f32]`. The internal H2D-copy of K/V slabs is removed.
- [ ] 10.2 Kernel writes the new K/V row directly into the
      `CudaSlice<f32>` buffer at `pos * num_kv_heads * head_dim`.

### 11. Tests + bench

- [ ] 11.1 `kv_cache_device_roundtrips_through_populate_kv_layer`
      — populate then read back via a temporary `dtoh_sync_copy`,
      assert bit-equality.
- [ ] 11.2 Re-run the gated `decode_q4k_gemma3_20_tokens_match_host`
      smoke; tokens MUST still agree.
- [ ] 11.3 Bench gate: `decode ms/token ≤ 60` AND
      `GPU fwd ms/token ≤ 55`. If we hit the gate, archive the
      change.

## 12. Documentation

- [ ] 12.1 Update `docs/cuda-rotorquant-status.md` with the
      bench-progress table after each phase.
- [ ] 12.2 Document `LARQL_CUDA_DECODE_HOST_FALLBACK=1` in the
      same doc (alongside the existing
      `LARQL_CUDA_Q4K_HOST_DEQUANT=1` and
      `LARQL_CUDA_Q6K_HOST_DEQUANT=1` env vars).
- [ ] 12.3 Note in `docs/claude-handoff-cuda-attention-kv.md`
      (or its successor) that the device-resident path is the
      default; host-fallback is for parity tests and debugging.

## 13. Archive

- [ ] 13.1 Once the bench acceptance hits and CI is green, archive
      this change: `openspec archive cuda-decode-device-resident`.
