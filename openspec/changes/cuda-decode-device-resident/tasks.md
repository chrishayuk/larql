# cuda-decode-device-resident — tasks

## Phase 1 — Device-resident projections

### 1. Backend API

- [x] 1.1 `CudaBackend::q4k_matvec_device(weight, x_device,
      rows, cols) -> Result<CudaSlice<f32>, CudaInitError>` —
      mirrors `q4k_matvec` but takes a device input slice and
      returns a device output slice. The existing
      `q4k_matvec(&self, ..., x: &[f32], ...)` becomes a thin
      `htod → q4k_matvec_device → dtoh` wrapper.
- [x] 1.2 Same for `q6k_matvec_device`, `q4kf_matvec_device`,
      and `f32_gemv_device`. The existing `gemv_device_w` helper
      in `cuda/matmul.rs` is the template.
- [x] 1.3 Cached weight handle on the matvec — `q4k_matvec_device`
      consults the existing per-backend Q4_K device cache (added
      in `cuda-q4k-device-cache`) so the first call uploads,
      subsequent calls reuse.

### 2. fused_decode_attention device entry point

- [x] 2.1 `attn::fused_decode_attention_device(q_dev, k_dev,
      v_dev, kv_k_dev, kv_v_dev, …) -> AttnOut<CudaSlice<f32>>` —
      symmetric to the existing host entry. Internally the kernel
      already runs on device; this just changes the input/output
      shape. (Phase 1 still takes host K/V cache; Phase 3 swaps
      those for `&CudaSlice<f32>`.)
- [x] 2.2 Keep the host-input variant as a wrapper that does
      H2D for K/V slabs and D2H for the result. (existing
      `fused_decode_attention` retained verbatim.)

### 3. decode_token rewrite

- [x] 3.1 Split the existing function into
      `decode_token_host_fallback` (current code, unchanged) and
      `decode_token_device` (new path). The trait impl
      dispatches based on `LARQL_CUDA_DECODE_HOST_FALLBACK` env
      var; the new path is the default and falls back to the
      host path silently for unsupported projection formats.
- [x] 3.2 In the new path: hold `h: CudaSlice<f32>` across
      projection chains within a layer. Each projection chains
      `q4k_matvec_device → q4k_matvec_device → …`. Phase 1 still
      does CPU rms_norm/silu/add, so per-layer crossings drop
      from 7-8 D2H to 4 D2H (gate, up, attn_delta, ffn_delta).
- [x] 3.3 Result `Vec<f32>` from a single final `dtoh_sync_copy`
      after the layer loop.

### 4. Tests

- [ ] 4.1 `q4k_matvec_device_returns_same_as_host` — random Q4_K
      packed weight + random input, both paths, byte-equal output.
      (Implicit in `decode_token_phase1_matches_host_fallback`;
      explicit unit will land alongside the Phase 2 GPU norm tests.)
- [x] 4.2 `decode_token_phase1_matches_host_fallback` — synthetic
      pipeline layer with Q4_K weights, three decode steps both
      paths, max-element diff ≤ 1e-3 per step.
- [ ] 4.3 `decode_q4k_gemma3_20_tokens_match_host` —
      `#[ignore]`'d, gated on `LARQL_CUDA_AVAILABLE=1` and the
      real vindex. Asserts greedy token ids agree across 20 steps.

### 5. Bench gate

- [x] 5.1 Run `larql bench output/gemma-3-4b-it-vindex --backends
      cuda --tokens 20 --warmup 3 --verbose`. Recorded:
      `decode 152.73 ms/token`, `GPU fwd 151.024 ms/token`,
      `tok/s 6.5`. Host-fallback control:
      `decode 158.88 ms/token`, `GPU fwd 157.166 ms/token`.
      ~6 ms/token (3.8%) improvement.
- [ ] 5.2 Acceptance: `decode ms/token ≤ 100` AND
      `GPU fwd ms/token ≤ 95`. **MISS** — Phase 1 alone is
      ~52% over target. Per the change's own decision gate
      ("if `decode ms/token > 120`, inspect for residual sync"),
      Phase 2 work needs the GPU rms_norm / silu / add kernels
      to remove the remaining 4 D2H per layer (gate / up /
      attn_delta / ffn_delta). Sync overhead is not the
      dominant cost at this scale; per-call cuBLAS launch +
      kernel arithmetic plus the inevitable K/V cache D2H per
      layer is. See PR description for the bench numbers.

## Phase 2 — DROPPED

After Phase 1, profiling under `LARQL_CUDA_DECODE_PROFILE=1`
showed the targeted ops (rms_norm + silu_gate_up + add_in_place)
sum to <6 ms/tok — not on the critical path. Phase 2 work is
out of scope; the time saved in those ops would not have
cleared the ≤80 ms/tok target.

## Phase 3 — Device-resident KV cache (now the only post-P1 work)

### 9. Type swap

- [x] 9.1 `CudaKvLayer::k: Vec<f32>` → `k: CudaSlice<f32>`. Same
      for `.v`. Allocate once via `CudaKvCache::new_device` at
      `preallocate_kv_cache_per_layer` time.
- [x] 9.2 `populate_kv_layer(layer, k_data: &[f32], …)` uses a
      new `CudaBackend::htod_into_slice` helper that copies into
      the pre-allocated `CudaSlice` at offset 0.

### 10. fused_decode_attention_device_kv contract

- [x] 10.1 New `attn::fused_decode_attention_device_kv` accepts
      `&mut CudaSlice<f32>` for K/V cache instead of `&[f32]`.
      No internal H2D / D2H of K/V slabs.
- [x] 10.2 Kernel writes the new K/V row directly into the
      device buffer at `pos * num_kv_heads * head_dim`. Existing
      kernel reused — only the wrapping function changed.
- [x] 10.3 Existing `fused_decode_attention_device` retained for
      one release; unused on the device path now (only the
      host-fallback bridges through it). Will be deleted after
      a follow-up cleanup change once external callers (none in-
      tree today) confirm migration.

### 11. Tests + bench

- [x] 11.1 `decode_token_phase1_matches_host_fallback` (3-step
      device-vs-host parity, ≤ 1e-3 max-element) is unchanged
      after Phase 3 — still passes. The host-fallback path now
      dtoh's the device K/V cache once per layer to feed the
      legacy host-input attention call, then htod's the result
      back; this is parity-only and intentionally slow.
- [x] 11.2 Bench gate cleared: `decode 27.37 ms/tok` AND
      `GPU fwd 25.416 ms/tok` (vs target ≤ 60 / ≤ 55).
      Throughput 36.5 tok/s vs target ≥ 16. Archive on next
      pass.

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
