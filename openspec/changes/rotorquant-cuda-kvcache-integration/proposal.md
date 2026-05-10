## Why

The CUDA decode path holds K/V in plain `half::f16`
(`cuda::decode::CudaKvLayer { k: CudaSlice<half::f16>, v: ... }`),
unlike the host-side `larql_inference::attention::decode::KvCache`
which gained a rotorquant compression seam in
[`rotorquant-attention-integration`](../rotorquant-attention-integration/proposal.md).
The asymmetry means:

- Long-context CUDA serving is VRAM-bound. A 7B-class model at
  128k tokens carries ~4 GB of f16 KV alone — non-negligible on a
  24 GB 4090 once weights + activations + workspace land on top.
  Rotorquant's Iso3/Planar3 formats compress K/V ~10× over f16
  with cosine ≥ 0.95 round-trip (see
  `kv-cache-rotorquant/spec.md` §"Compression ratios").
- The CUDA quantize/dequantize PTX kernels shipped in
  [`rotorquant-cuda-kernels`](../rotorquant-cuda-kernels/proposal.md)
  are **stranded** — they live in `larql_rotorquant::cuda` and
  pass round-trip parity tests, but the canonical decode path
  never invokes them.
- Code duplication: two KV-cache implementations (host-side
  `KvCache` with the rotorquant seam, CUDA-side `CudaKvCache`
  without it) drift independently. The upcoming
  `engine-rotorquant-auto-compress` story has no story for the
  CUDA path.

The C.4 note on `cuda-spec-phase4b-complete` deferred this work
("CUDA decode path uses plain f16 KV cache — Phase 4c can proceed
without any rotorquant changes"). That deferral was correct for
phase 4c's throughput-first scope; this change picks up the
follow-up.

## What Changes

This change adds the **data-layer seam** on `CudaKvCache`,
mirroring the host-side `KvCache` integration. It deliberately
stops at the seam — it does NOT auto-compress on every decode
write. That policy decision (immediate compress vs deferred-K vs
window-lag) is the natural follow-up after the seam exists.

- ADD `larql-rotorquant` as a dep of `larql-compute` (currently
  only `larql-inference` depends on it).
- ADD `CudaKvCache::kv_format: Option<KvFormat>` + `set_kv_format`
  setter. Default `None` preserves bit-exact f16 behavior at all
  existing call sites.
- ADD `CudaKvCache::quantized_kv: Vec<Option<(DeviceQuantizedKv,
  DeviceQuantizedKv)>>` parallel side-table for compressed K/V
  slabs. New `DeviceQuantizedKv` mirrors the host-side
  `QuantizedKv` but stores codes/scales/rotation-tables in
  `CudaSlice` device buffers.
- ADD `CudaKvCache::quantize_layer(layer)` — moves a layer's f16
  slab into `quantized_kv[layer]` via `cuda::rotorquant_kernels::
  planar_quantize_kernel` / `iso_quantize_kernel` (depending on
  `kv_format`). Returns `false` if format unset or layer empty.
- ADD `CudaKvCache::dequantize_layer(layer)` — non-destructive
  read: launches the inverse kernel into a temp f16 buffer
  without disturbing `quantized_kv[layer]`. Returns
  `Option<(CudaSlice<f16>, CudaSlice<f16>)>`.
- ADD `CudaKvCache::promote_layer_to_fp32(layer)` — inverse of
  `quantize_layer`: dequantizes into the f16 slab and clears the
  compressed entry.
- ADD `CudaKvCache::is_layer_compressed(layer)`.
- MODIFY `cuda::attn::decode_attention` (the per-layer read path)
  to call `dequantize_layer` into a scratch f16 buffer when a
  layer is compressed, then proceed with the existing attention
  kernel. Cost: ~30 µs per compressed layer per token (per the
  perf numbers in `rotorquant-cuda-kernels`'s proposal).
- ADD parity tests in `crates/larql-compute/tests/test_cuda_kv_rotorquant.rs`
  (env-gated by `LARQL_CUDA_AVAILABLE=1`):
  - `cuda_kvcache_quantize_layer_no_op_when_format_unset`
  - `cuda_kvcache_iso3_quantize_then_dequantize_roundtrip_preserves_direction`
  - `cuda_kvcache_planar3_quantize_then_dequantize_roundtrip_preserves_direction`
  - `cuda_kvcache_promote_layer_to_fp32_restores_f16_slot`
  - `cuda_decode_attention_with_compressed_layer_matches_uncompressed_within_cosine_0_95`

This is non-breaking. All existing `CudaKvCache` construction
sites are untouched; new fields default-initialise. Existing call
sites that read `CudaKvLayer` directly keep working — they just
won't see compressed layers (those are in the parallel
side-table, behind `dequantize_layer`).

## Capabilities

### New Capabilities

(none — extends scenarios already on `kv-cache-rotorquant`.)

### Modified Capabilities

- `kv-cache-rotorquant`: new requirements for the CUDA-side
  KvCache integration. The existing "CUDA backend supports
  RotorQuant KV compression" requirement was about kernel
  round-trip; this change adds the cache-level API surface
  that exposes those kernels through `CudaKvCache`.

## Impact

- **Affected files**:
  - `crates/larql-compute/Cargo.toml` (+1 dep: `larql-rotorquant`).
  - `crates/larql-compute/src/cuda/decode.rs` (~250 LOC additions
    on `CudaKvCache`).
  - `crates/larql-compute/src/cuda/attn.rs` (~30 LOC for
    dequantize-on-read in `decode_attention`).
  - `crates/larql-compute/tests/test_cuda_kv_rotorquant.rs` (new
    test file, ~200 LOC).
- **Affected systems**: CUDA backend only. No host-side / Metal
  / CPU changes. No spec-decode dispatch changes — the
  `decode_tokens_speculative` helper continues to write/read f16
  slabs (compressed layers are static, read-only views of past
  committed K/V; spec rollback only truncates the f16 slab).
- **VRAM**: with format set + auto-compress wired, expected
  ~10× reduction on compressed layers. Without auto-compress
  wired (this change's scope), VRAM unchanged.
- **Hot-path perf**: zero-cost when `kv_format` is `None` (new
  fields are read-only checks). When set, compressed layers add
  ~30 µs per attention call per layer for dequant. Auto-compress
  adds quant cost on writes — sized in the follow-up.
- **Out of scope**:
  - Auto-compress policy on writes (immediate vs deferred-K vs
    window-lag) — separate change.
  - Speculative decode interaction with compressed cache —
    separate change once the policy is picked. The spec helper
    can stay f16-only because committed accepted spans go through
    the policy layer, not directly through `decode_tokens_speculative`.
  - `Capability::KvCompressionRotorQuant` is already advertised
    by `CudaBackend` per `rotorquant-cuda-kernels`; no flip needed.
