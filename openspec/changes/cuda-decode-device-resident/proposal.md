## Why

After `cuda-resident-q4k-matvec` (`67ff38c`) and
`cuda-q4k-device-cache` (`f1a24ab`) the CUDA decode benchmark on
the local Q4K Gemma-3-4B vindex sits at:

```
prefill 1155.1 ms  /  decode 162.72 ms/token  /  6.1 tok/s
GPU fwd 160.820 ms  /  LM-head 1.888 ms
```

The remaining ~160 ms/token is **GPU forward time** that is mostly
synchronisation overhead, not arithmetic. Per-layer in
`crates/larql-compute/src/cuda/decode.rs::decode_token`:

```text
rms_norm_vec(h, …)             ← CPU
matvec(wq, …) -> Vec<f32>      ← GPU launch + D2H sync
matvec(wk, …) -> Vec<f32>      ← GPU launch + D2H sync
matvec(wv, …) -> Vec<f32>      ← GPU launch + D2H sync
fused_decode_attention(…)      ← H2D for K/V slices, GPU, returns Vec
matvec(wo, …) -> Vec<f32>      ← GPU launch + D2H sync
rms_norm_vec(attn_delta, …)    ← CPU
add_in_place(…)                ← CPU
matvec(gate, …) -> Vec<f32>    ← GPU launch + D2H sync
matvec(up, …)   -> Vec<f32>    ← GPU launch + D2H sync
activate(gate, up)             ← CPU
matvec(down, …) -> Vec<f32>    ← GPU launch + D2H sync
add_in_place(…)                ← CPU
```

That's **7 D2H syncs per layer × 34 layers = ~240 round-trips per
decode token**. At ~0.5 ms each (cudaMemcpy + driver enqueue + CPU
trip) we burn ~120 ms/token in sync alone — close to the observed
160 ms/token total, leaving only ~40 ms of actual compute.

`fused_decode_attention` already runs on-device. The K/V cache,
norms, activation, and add are still host-bound. Lifting them
onto the device collapses the round-trip count to **one D2H per
token** (the final hidden state to feed lm_head).

This change is **performance-only** — output bit-equivalent within
1e-3 max-element vs the current path. No spec or API surface
change beyond the new `*_device` matvec variants on `CudaBackend`,
which are additive.

## What Changes

### Phase 1 — Device-resident projections

- ADD `CudaBackend::q4k_matvec_device(weight, x_dev, rows, cols)
  -> CudaSlice<f32>` and the symmetric `q6k_matvec_device`,
  `q4kf_matvec_device`, and `f32_gemv_device`. All return a
  `CudaSlice<f32>` instead of `Vec<f32>`; the existing
  host-returning variants stay as thin wrappers (call the device
  variant, then `dtoh_sync_copy`).
- REWORK `decode.rs::decode_token` so per-layer state stays on the
  device through the projection chain. Only `h_post_attn` (for
  the residual add path) and the final `h` after the layer loop
  come back to host. Per-layer host crossings drop from 7 to 0 in
  the GPU fwd hot loop; `fused_decode_attention` either accepts
  device pointers directly (preferred) or moves the H2D inside.
- ADD a `LARQL_CUDA_DECODE_HOST_FALLBACK=1` env var that forces
  the existing host-bouncing path. Default is the new path.
- KEEP the existing `matvec` CPU-bound helper for the non-Q4_K
  rare-path (fp16 etc.); the bottleneck is Q4_K-quantised models,
  which is where every shipped Gemma vindex lives.

### Phase 2 — GPU kernels for rms_norm / activate / add

- ADD `kernels::rms_norm_vec_device(x_dev, weight_dev, eps,
  norm_offset) -> CudaSlice<f32>`. Single-block kernel: 1024
  threads, parallel reduction for sum-of-squares, then scale +
  weight in-place.
- ADD `kernels::silu_gate_up_device(gate_dev, up_dev) ->
  CudaSlice<f32>` (and the other Activation variants). One launch
  per layer instead of three CPU traversals.
- ADD `kernels::add_in_place_device(target_dev, delta_dev)` — a
  trivial element-wise add. Pair-wise add of [hidden] vectors at
  hidden = 2560 is bandwidth-bound; on GPU it's ~10 µs vs ~50 µs
  on CPU plus the round trip.
- KEEP `rms_norm_vec` / `activate` / `add_in_place` host helpers
  for parity tests and the host-fallback path.

### Phase 3 — Device-resident KV cache

- REPLACE `CudaKvCache::layers[*].k: Vec<f32>` and `.v: Vec<f32>`
  with `CudaSlice<f32>` allocated once at
  `preallocate_kv_cache_per_layer` time. The host-side `populate_kv_layer`
  path (used by the prefill bridge in larql-inference) becomes a
  `htod_sync_copy_into` of the K/V slabs.
- UPDATE `fused_decode_attention` to take `&CudaSlice<f32>` for
  K/V cache (not `&[f32]`). The kernel already operates
  device-side; this avoids the H2D it currently does internally.
- KEEP `populate_kv_layer` API (it's how
  `larql_inference::predict_honest`'s post-norm CPU path seeds the
  cache for Gemma 3 4B); just back it with `htod_sync_copy_into`.

## Capabilities

### New Capabilities

(none — extends `compute-cuda-kernels`.)

### Modified Capabilities

- `compute-cuda-kernels` — adds requirements for the
  device-resident projection API, the GPU rms_norm/silu/add
  kernels, the device-resident KV cache, and the
  host-fallback escape hatch.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/q4k_direct.rs` — gains the
    `*_device` variant of the existing direct Q4_K kernel.
  - `crates/larql-compute/src/cuda/quant_matvec.rs` — adds
    `*_device` methods to the `QuantMatVec`-CUDA impl on
    `CudaBackend`.
  - `crates/larql-compute/src/cuda/matmul.rs` — `gemv_device`
    already exists per `cuda-q4k-device-cache`; verify.
  - `crates/larql-compute/src/cuda/decode.rs` — major rewrite of
    the per-layer body. Old code stays under
    `decode_token_host_fallback` for env-var bypass.
  - `crates/larql-compute/src/cuda/attn.rs` — `fused_decode_attention`
    learns a `*_device` variant that accepts device pointers.
  - `crates/larql-compute/src/cuda/kernels/` — new `rms_norm.cu` /
    `silu.cu` / `add_in_place.cu` (or NVRTC strings;
    cuda-oxide is mid-pilot but not the canonical path yet).

- **Affected systems**: GPU container only. CPU FFN container
  unaffected. Metal backend on macOS unaffected (the `*_device`
  helpers are CUDA-only; a parallel Metal version is out of
  scope for this change).

- **Provenance**: bottleneck identified by manual code audit
  after the `cuda-q4k-device-cache` (`f1a24ab`) commit. The
  ~160 ms/token figure is from the bench command in
  `docs/claude-handoff-cuda-attention-kv.md`.

- **Out of scope**:
  - Multi-GPU / tensor parallelism — single device, single stream.
  - Flash-attention-2-style fused attention rewrite — the
    existing `fused_decode_attention` is already on-device; we
    just stop bouncing K/V around it.
  - Migrating these new kernels to cuda-oxide — that's the
    parent change `cuda-oxide-migration`. This change uses
    NVRTC strings consistent with the rest of the cudarc path;
    the migration of these new kernels to cuda-oxide is a
    follow-up under `cuda-oxide-migration` Phase 3.
  - Q4_KF and Q4_0 paths — they're rare on production vindexes
    (Gemma 3 4B / Gemma 4 26B both ship Q4_K). The existing
    host-fallback covers them; we don't add `*_device` variants
    in this change.

## Risks and back-out

- **Numerical drift.** Moving rms_norm and activate from CPU to
  GPU introduces small fp32 reduction-order differences. Bound:
  max-element diff ≤ 1e-3 vs the old path on 100 random tokens.
  Mitigation: parity test in `tests/test_cuda_decode.rs` runs
  both paths and asserts the bound.
- **K/V cache size.** Device-side per-layer slabs at max_seq=4096
  × num_kv_heads × head_dim × f32 = ~5 MB per layer × 34 = ~170 MB
  VRAM for Gemma 3 4B. Tractable on the RTX 4090's 24 GB.
  Mitigation: document in `docs/cuda-rotorquant-status.md`;
  larger contexts will benefit from RotorQuant compression
  (existing `rotorquant-attention-integration` change).
- **Back-out:** `LARQL_CUDA_DECODE_HOST_FALLBACK=1` reverts to
  the current path at runtime. The new code is additive; the
  old `decode_token` stays as `decode_token_host_fallback` and
  is reachable via the env var.

## Acceptance bar

Phase 1 is "ship it" on these numbers, measured on the dev box
(RTX 4090, CUDA 13.1, Gemma 3 4B Q4_K vindex, 20 tokens after 3
warmup):

| Metric | Pre-Phase 1 | Phase 1 actual | Phase 1 target | Phase 3 target |
|---|---:|---:|---:|---:|
| `decode ms/token` | 162.72 | **152.73** | ≤ 100 | ≤ 60 |
| `GPU fwd ms/token` | 160.820 | **151.024** | ≤ 95 | ≤ 55 |
| `tok/s` | 6.1 | **6.5** | ≥ 10 | ≥ 16 |
| Bit parity vs host fallback | — | **≤ 1e-3 max-element** | ≤ 1e-3 max-element | ≤ 1e-3 max-element |

`LARQL_CUDA_DECODE_HOST_FALLBACK=1` control on the same build:
`decode 158.88 ms/token`, `GPU fwd 157.166 ms/token` — the new
default beats the legacy path by ~6 ms/token (3.8%) at parity.

**Phase 1 misses the target.** Per the proposal's own decision
gate ("if `decode ms/token > 120`, inspect for residual sync"),
Phase 2 (GPU `rms_norm` / `silu_gate_up` / `add_in_place`)
remains needed to clear the bar. Sync overhead is not the
dominant cost at this scale; the remaining ~150 ms/token is
spread across cuBLAS launch + arithmetic on the projection
GEMVs, the K/V cache D2H per layer (Phase 3), and per-call
allocation. Phase 2 will eliminate 4 of the 5 remaining D2H
per layer (gate, up, attn_delta, ffn_delta) and is expected
to land most of the remaining headroom.

Phase 1 ships as-is on the strength of: (a) parity correctness,
(b) measured improvement vs the host-fallback control, and
(c) being a strict prerequisite for Phase 2 — without the
device-resident projection API the GPU norm/activate kernels
have nowhere to read their inputs from on the device.
