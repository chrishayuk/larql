## Storage strategy: parallel side-table vs unified compressed

The host-side change (`rotorquant-attention-integration`) chose a
**parallel side-table**: the existing FP32 `layers` slot keeps
working for uncompressed layers; compressed layers move into
`quantized_kv[layer]: Option<(QuantizedKv, QuantizedKv)>` and
`layers[layer]` becomes `None`. This change adopts the same
structure on the CUDA side for three reasons:

1. **Backwards compatibility.** Every call site that reads
   `CudaKvLayer` directly today (the attention kernel,
   `populate_kv_layer`, `decode_token`, the spec helper's
   `decode_tokens_speculative`) keeps working unchanged when no
   layer is compressed. New behaviour is opt-in via
   `set_kv_format`.

2. **Spec-decode compatibility.** Phase 4c's
   `target_forward_via_speculative_decode` writes to the f16 K/V
   slabs and rolls back via `truncate_kv_cache`. If we replaced
   the f16 storage with compressed-only storage, the spec helper
   would need to either (a) decompress the spec window into f16
   on every probe (expensive — D × decompress per spec call), or
   (b) write directly into compressed format (requires fused
   compress-on-write which doesn't exist yet). The parallel
   side-table sidesteps both: compressed slots are static
   read-only views of past committed K/V; the spec helper writes
   to f16 as today.

3. **Promote symmetry.** `promote_layer_to_fp32` is a real
   operation (e.g., for layers that fall back to high-precision
   re-attention on rejection or for fine-tuned analysis). The
   side-table makes promotion cheap: dequantize into the empty
   f16 slot, drop the compressed entry. With unified compressed
   storage, promote would have to allocate fresh f16 storage on
   every promote.

## DeviceQuantizedKv layout

The host-side `QuantizedKv` (in `larql_rotorquant`) bundles
`codes: Vec<u8>`, `scales: Vec<f16>`, and optionally `rotations:
Vec<RotationTable>`. The CUDA mirror keeps the same logical
fields but in device memory:

```rust
pub(crate) struct DeviceQuantizedKv {
    pub(crate) codes: CudaSlice<u8>,
    pub(crate) scales: CudaSlice<half::f16>,
    pub(crate) rotations: Option<CudaSlice<f32>>, // 2x2 or 4x4 packed
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) seq_len: usize,
    pub(crate) format: KvFormat,
}
```

`rotations` is `Option<>` because rotation tables exist only for
Iso* formats (per the upstream RotorQuant design); Planar*
formats use scalar-only quant. The host-side `QuantizedKv`
already encodes this via the format enum; the device mirror
follows.

## Dequantize-on-read

When `decode_attention` reads a layer that is compressed, it
allocates a transient f16 scratch buffer of `[max_seq,
num_kv_heads, head_dim]` shape, launches the inverse kernel
(`planar_dequantize_kernel` or `iso_dequantize_kernel`) to
populate it, and proceeds with the existing attention kernel.

**Why scratch and not a permanent f16 mirror**: a permanent
mirror would defeat the VRAM savings (the whole point of
compressing). Scratch lives for the duration of one
`decode_attention` call. With CUDA streams, the dequant launch
overlaps the prior layer's attention compute; net latency add
is ~30 µs per compressed layer (kernel launch + dequant of one
layer's worth of K/V).

**Allocation pattern**: a single per-`CudaKvCache` scratch buffer
(largest layer's `[max_seq, num_kv_heads, head_dim]`) is
preallocated at `set_kv_format` time and reused across reads.
This avoids per-call cudaMalloc.

## Quantize-on-write: deliberately deferred

This change does NOT auto-compress on every K/V write. The
canonical decode path's `decode_token` continues to write f16 to
the `CudaKvLayer` slab as today. Triggering `quantize_layer` is
the caller's responsibility (or, in the follow-up, the engine
decorator's).

The reasoning mirrors the host-side: phase 4c's spec dispatch
does many writes per step, and forcing each to round-trip
through quantize would tank ms/tok. The "deferred-K" / "window-
lag" patterns from upstream RotorQuant let the engine decide
WHEN to compress (e.g., compress everything older than the last
N tokens). Picking a policy is a separate scope.

## Capability advertisement

`CudaBackend` already advertises
`Capability::KvCompressionRotorQuant` per
`rotorquant-cuda-kernels`. No flip needed. The capability claim
goes from "kernels exist and round-trip on synthetic inputs" to
"the canonical KV cache exposes the seam to use them" — a
strengthening, not a change in declared support.

## Why not a separate `CompressedCudaKvCache` type?

A new struct that lives alongside `CudaKvCache` would avoid
mutating the existing one. Rejected because:

- Every call site (~10 places) that takes `&CudaKvCache` would
  need a generic over `CudaKvCacheKind` or two parallel paths.
- The `DecodeBackend` trait's `kv_cache_len` / `truncate_kv_cache`
  / `populate_kv_layer` methods would need to be implemented on
  both, doubling the surface area for divergence.
- The host-side precedent is in-place mutation of `KvCache`, and
  symmetry is valuable for cross-backend reasoning.

## Test strategy

All tests are gated on `LARQL_CUDA_AVAILABLE=1` (existing
pattern, see `crates/larql-rotorquant/tests/cuda_round_trip.rs`).
The test file is new — `crates/larql-compute/tests/
test_cuda_kv_rotorquant.rs` — and exercises the
`CudaKvCache`-level API (the underlying CUDA quant/dequant
kernels are already covered by the round-trip tests in
`larql_rotorquant`).

The end-to-end attention test
(`cuda_decode_attention_with_compressed_layer_matches_uncompressed_within_cosine_0_95`)
seeds a `CudaKvCache`, populates one layer with synthetic FP32
data, runs attention twice (once with the layer left as f16,
once with it compressed via `quantize_layer`), and asserts the
two attention outputs match within cosine ≥ 0.95. This is the
load-bearing test that proves dequantize-on-read is wired
correctly, not just that quantize/dequantize round-trips.

## Migration / rollback

This change is purely additive at the data layer. There is no
flag day. If the new requirements turn out to be wrong, the
change can be reverted by deleting the new fields/methods and
the test file; no existing call site has to change.
