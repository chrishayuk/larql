## Phase 1 — Data-layer seam (this change)

Branch: `feat/rotorquant-cuda-kvcache-integration`

### Setup

- [ ] D.1 Add `larql-rotorquant` to `crates/larql-compute/Cargo.toml`
      `[dependencies]` (no feature gate; the CUDA-specific code
      paths are already gated on `cfg(feature = "cuda")` inside
      `larql-compute`).
- [ ] D.2 Add `pub use larql_rotorquant::{KvFormat, QuantizedKv};`
      re-export to `crates/larql-compute/src/cuda/mod.rs` so
      downstream callers don't need a direct `larql-rotorquant`
      dep.

### CudaKvCache fields + setter

- [ ] D.3 In `crates/larql-compute/src/cuda/decode.rs`, add to
      `CudaKvCache`:
  ```rust
  pub(crate) kv_format: Option<KvFormat>,
  pub(crate) quantized_kv: Vec<Option<(DeviceQuantizedKv, DeviceQuantizedKv)>>,
  pub(crate) dequant_scratch: Option<DequantScratch>,
  ```
  `quantized_kv` is sized to match `layers` at `new_device` time.
- [ ] D.4 Define `DeviceQuantizedKv` (per design.md "DeviceQuantizedKv layout").
- [ ] D.5 Define `DequantScratch { k: CudaSlice<f16>, v: CudaSlice<f16> }`
      with capacity for the largest layer's `[max_seq, num_kv_heads, head_dim]`.
- [ ] D.6 Add `pub fn set_kv_format(&mut self, format: KvFormat)`.
      First call also allocates `dequant_scratch`.

### Quantize / dequantize / promote

- [ ] D.7 `pub fn quantize_layer(&mut self, layer: usize) -> bool`.
      Returns `false` if `kv_format` is `None`, layer is empty
      (`self.len == 0` or `layers[layer].k.len() == 0`), or layer
      is already compressed. On success, launches the appropriate
      CUDA quantize kernel (`larql_rotorquant::cuda::launcher`)
      against `layers[layer].k` and `layers[layer].v`, stores the
      result in `quantized_kv[layer]`, and replaces the f16 slabs
      with empty `CudaSlice`s (or a sentinel zero-len slab — the
      goal is to free the device memory).
- [ ] D.8 `pub fn dequantize_layer(&self, layer: usize) -> Option<(&CudaSlice<f16>, &CudaSlice<f16>)>`.
      Non-destructive: launches the inverse kernel into
      `dequant_scratch.k` / `.v` and returns references to the
      scratch slabs. Returns `None` if the layer is not
      compressed.
- [ ] D.9 `pub fn promote_layer_to_fp32(&mut self, layer: usize) -> bool`.
      Dequantizes into a fresh `CudaSlice<f16>` of the original
      shape, replaces the f16 slab in `layers[layer]`, clears
      `quantized_kv[layer]`. Returns `false` if not compressed.
- [ ] D.10 `pub fn is_layer_compressed(&self, layer: usize) -> bool`.

### Attention read path

- [ ] D.11 In `crates/larql-compute/src/cuda/attn.rs`'s
      `decode_attention` (the per-layer attention call site —
      grep for the existing `cache.layers[layer]` reads), wrap
      each layer access:
  ```rust
  let (k_slice, v_slice) = if cache.is_layer_compressed(layer) {
      cache.dequantize_layer(layer)?
  } else {
      (&cache.layers[layer].k, &cache.layers[layer].v)
  };
  ```
- [ ] D.12 Verify the `decode_tokens_speculative` path is
      unaffected (it writes via `decode_token` which goes through
      the f16 slab; compressed layers are read-only). Add a
      comment block at the top of `decode_tokens_speculative`
      stating the contract: "compressed layers are read-only;
      this method writes to the f16 slab only."

### Tests

- [ ] D.13 Create `crates/larql-compute/tests/test_cuda_kv_rotorquant.rs`
      with the following tests, all gated by
      `LARQL_CUDA_AVAILABLE=1`:
  - [ ] `cuda_kvcache_quantize_layer_no_op_when_format_unset`
  - [ ] `cuda_kvcache_iso3_quantize_then_dequantize_roundtrip_preserves_direction`
  - [ ] `cuda_kvcache_planar3_quantize_then_dequantize_roundtrip_preserves_direction`
  - [ ] `cuda_kvcache_promote_layer_to_fp32_restores_f16_slot`
  - [ ] `cuda_decode_attention_with_compressed_layer_matches_uncompressed_within_cosine_0_95`

### Validation

- [ ] V.1 `openspec validate rotorquant-cuda-kvcache-integration --strict` passes.
- [ ] V.2 `make traceability-check` passes after regen.
- [ ] V.3 `make test-cuda` clean on RTX 4090 (host with
      `LARQL_CUDA_AVAILABLE=1`).
- [ ] V.4 Bit-exact f16 baseline preserved when `kv_format` is
      `None` — verified by running the existing
      `make test-cuda` suite, which constructs `CudaKvCache`
      without setting a format.
- [ ] V.5 `make ci` clean (workspace build, fmt, clippy, test-fast,
      traceability, openspec-validate).

## Phase 2 — Auto-compress policy (separate change)

Out of scope for this proposal. Once the seam exists, a
follow-up change picks the auto-compress policy. Candidate
policies:

- **Immediate**: `quantize_layer` after every `decode_token`
  write. Simple but expensive; ~30 µs × num_layers per step.
- **Deferred-K + window-lag**: keep the most recent N tokens
  (the speculative window plus a safety margin) as f16; compress
  older tokens. The historical `rotorquant-window-lag` proposal
  (the prereq C.4 deferred) maps directly to this.
- **On-demand from engine**: callers (`UnlimitedContextEngine`,
  etc.) decide. Mirrors `engine-rotorquant-auto-compress`'s
  decorator pattern.

The engine-side `engine-rotorquant-auto-compress` change blocks
on `engine-kvcache-unification`; the CUDA-side equivalent should
NOT be similarly blocked because `CudaKvCache` is the canonical
type that all CUDA decode paths already use.
