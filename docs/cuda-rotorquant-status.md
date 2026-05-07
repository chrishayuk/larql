# CUDA + RotorQuant — status snapshot

> _Tracks progress against the parent OpenSpec change
> [`cuda-and-rotorquant-kv`](../openspec/changes/cuda-and-rotorquant-kv/proposal.md)._

This is the **machine-checked truth** of where the CUDA backend and
RotorQuant KV-cache compression stand. Anything not listed below as
shipped is either explicitly out of scope or left for a follow-up
sub-change.

## Shipped

### Phase 1 — Inventory and scaffolding

- ✅ Parent OpenSpec change `cuda-and-rotorquant-kv` (proposal +
  design + 10 capability deltas + tasks).
- ✅ `deploy/docker/` — `Dockerfile.ffn` (CPU FFN), `Dockerfile.gpu`
  (CUDA 13.1 base), `docker-compose.yml` (ffn + attention + router),
  `docker-compose.cpu.yml` (single-binary fallback), `start.sh`,
  `README.md` with topology + VRAM/RAM budget tables.
- ✅ Makefile targets: `docker-ffn`, `docker-gpu`, `docker-up`,
  `docker-down`, `docker-logs`, `test-cuda`, `cuda-status`.
- ✅ `larql-cli` `--features metal` no longer default — `cargo check
  --workspace` passes on Linux without macOS-only deps.
- ✅ Pre-existing `larql-vindex` `pub mod build` breakage repaired
  (file restored from commit `fbb5a70`).
- ✅ `rust-toolchain.toml` pins workspace to stable (was nightly,
  pre-edition2024).

### Phase 2 — CUDA kernel surface

- ✅ `cuda-f32-baseline`: `CudaBackend::matmul`, `matmul_transb`,
  `f32_gemv` via cuBLAS through cudarc 0.19. **9/9** parity tests
  pass on RTX 4090 in 8 s. Capability::F32Gemv on.
- ✅ `cuda-q4-matvec`: Q4_0 / Q4_K / Q6_K matvec via host dequant +
  cuBLAS gemv. **5/5** parity tests on Gemma 4B FFN gate (10240×2560)
  and Llama LM head (128256×4096). Capability::QuantMatVec +
  Q4VecMat on.
- ✅ `cuda-fused-attention`: scaled-softmax PTX kernel via NVRTC
  with optional causal mask + softcap; `decode_attention` helper
  chains cuBLAS GEMM → softmax → cuBLAS GEMM in one host roundtrip.
  **6/6** parity tests including Gemma 4B head_dim=320, n_kv=2048.
  Capability::FlashAttentionV2 on.

### Phase 3 — RotorQuant

- ✅ `rotorquant-kernels`: new `larql-rotorquant` workspace member
  (zero LARQL deps; mirrors model-compute's "extract later" pattern).
  CPU reference for all four formats (Planar3 / Planar4 / Iso3 /
  Iso4). **9/9** round-trip tests + 1 doctest pass; cosine ≥ 0.95
  including Gemma 4B head_dim=320. CUDA module is a feature-flagged
  stub today; PTX kernels land in a follow-up.
- ✅ `rotorquant-strategy`: `RotorQuantStrategy` joins the
  `KvStrategy` trait family in `kv-cache-benchmark`. Four
  constructors (`iso3`, `planar3`, `iso4`, `planar4`) plumb
  `larql-rotorquant`'s CPU reference into the same harness used by
  Standard / TurboQuant / Markov / Apollo. **3/3** strategy tests
  pass.

### Phase 4 — Router topology

- ✅ `router-heterogeneous-shards`: `ServerEntry` carries a
  `capabilities: Vec<String>` set; `GridState::route_for_capability`
  filters by capability + layer range. Backwards-compat default
  for legacy shards is `["attention", "expert"]`. **4/4** new grid
  tests pass alongside the existing 7. Proto extension to carry
  capabilities on the announce wire ships with
  `attention-service-routes`.

### Phase 5 — Attention KvCache integration

- ✅ `rotorquant-attention-integration`: `KvCache` gains a
  `kv_format: Option<KvFormat>` parameter and a parallel
  `quantized_kv: Vec<Option<(QuantizedKv, QuantizedKv)>>`
  side-table. New methods: `set_kv_format`, `quantize_layer`
  (FP32 → compressed; takes the FP32 slot to avoid memory doubling),
  `dequantize_layer` (non-destructive readback;
  `dequantize_v_with_inverse_rotation` for V), `promote_layer_to_fp32`,
  `is_layer_compressed`. Round-trip cosine ≥ 0.95 on synthetic
  Gemma-shaped data. **18/18** attention::decode tests pass
  including 3 new for the compressed side-table.

## Not yet shipped (known follow-up sub-changes)
- **`attention-service-routes`** — new HTTP + gRPC endpoints on
  `larql-server` (`/v1/attention/{session,prefill,decode}`,
  `/v1/kv-cache/{snapshot,restore,free}`); session lifecycle;
  binary KV-snapshot wire format; extends `AnnounceMsg` proto with
  the capability set so `route_for_capability` (shipped) gets real
  data. Depends on test-fixture drift in
  `larql-server/tests/test_expert_endpoint.rs` being repaired.
- **`rotorquant-cuda-kernels`** — replaces the CPU reference in
  `larql-rotorquant` with PTX kernels for planar3 / iso3
  quantize+dequantize; flips
  `Capability::KvCompressionRotorQuant` on `CudaBackend`.
- **`engine-rotorquant-auto-compress`** — proposal-only.
  Decorator-pattern `RotorQuantEngine { inner: Box<dyn KvEngine>,
  format: KvFormat }` that wraps any underlying engine and calls
  `cache.quantize_layer` post-decode. Adds `cache_mut()` to the
  `KvEngine` trait. Spec string: `iso3:inner=unlimited-context`.
  See `openspec/changes/engine-rotorquant-auto-compress/`.
- **`deploy-compose-end-to-end`** — `docker compose up` boots
  Gemma 4B end-to-end through the router; `make demo` target
  produces a one-shot inference; PERFORMANCE.md gets the measured
  tok/s + VRAM column.

### SMG-derived backlog (revisit after CUDA stabilises)

Three sub-changes drafted as **proposal-only** after analysing the
PyTorch / LightSeek SMG blog post — not on the critical path,
spec'd for later pickup:

- **`server-tokenizer-cache`** — L0 exact-match + L1 prefix-aware
  trie cache in front of `Tokenizer::encode`. SMG measured 23%
  TTFT reduction at 256 concurrency.
- **`router-prefix-aware-routing`** — `ServerEntry` carries a
  Bloom filter of cached prefix hashes; routing prefers shards
  that already have the request's prefix in KV cache. SMG: 23%
  TTFT win + 10–12× faster cache routing.
- **`attention-service-prefill-decode-split`** — extends the
  planned `attention-service-routes` design to support optional
  PD disaggregation: prefill stateless, decode session-bound,
  KV snapshot is the handoff. SMG / Sarathi-Serve / DistServe:
  20–30% TTFT improvement.

## Snapshot of capability bits today

```rust
// On RTX 4090 / Linux + cuda feature, after this branch:
backend.supports(Capability::Cuda)                     // true
backend.supports(Capability::F32Gemv)                  // true (cuda-f32-baseline)
backend.supports(Capability::QuantMatVec)              // true (cuda-q4-matvec)
backend.supports(Capability::Q4VecMat)                 // true (cuda-q4-matvec)
backend.supports(Capability::FlashAttentionV2)         // true (cuda-fused-attention)
backend.supports(Capability::KvCompressionRotorQuant)  // false — flips on with rotorquant-attention-integration
backend.supports(Capability::DecodeToken)              // false — trait-level decode_token still None
```

## What `make test-cuda` exercises

```
cargo test -p larql-compute --features cuda --test test_cuda_f32   →  9 tests
cargo test -p larql-compute --features cuda --test test_cuda_q4    →  5 tests
cargo test -p larql-compute --features cuda --test test_cuda_attn  →  6 tests
cargo test -p larql-rotorquant                                     →  9 tests + 1 doctest
cargo test -p kv-cache-benchmark --lib rotorquant                  →  3 tests
cargo test -p larql-router --bin larql-router grid::tests          → 11 tests
cargo test -p larql-inference --lib attention::decode               → 18 tests
                                                                   ────
                                                                    62 tests
```

All require `LARQL_CUDA_AVAILABLE=1` for the GPU-gated subset; the
RotorQuant tests run anywhere (CPU reference only today).

## What `cargo check --workspace` reports today

Clean. 94 warnings in `larql-cli` are pre-existing dead-code /
unused-mut style; not related to this work.

## Known pre-existing breakage I did NOT fix

These are workspace issues that predate the CUDA workstream and are
not on the critical path for it:

- `crates/larql-server/tests/test_expert_endpoint.rs` — fails to
  compile because `MoeLayerWeights` API drifted (added
  `expert_data_format`; `experts_gate_up`/`experts_down` moved from
  `&[u8]` to `Vec<&[u8]>`). Out of scope until
  `attention-service-routes` lands.

## Ledger of commits (most recent last)

| Commit | Subject |
|---|---|
| `b8a4301` | propose CUDA backend + split CPU/GPU topology (parent change) |
| `0bb4923` | unblock workspace builds for cuda follow-ups |
| `ccbc0ce` | [cuda-f32-baseline] real cuBLAS f32 GEMM/GEMV via cudarc |
| `876b42c` | [cuda-q4-matvec] Q4_0 / Q4_K / Q6_K matvec on cuBLAS via host dequant |
| `0cdbf1b` | [cuda-fused-attention] scaled-softmax PTX kernel + decode_attention helper |
| `dbf57e7` | [rotorquant-kernels] new larql-rotorquant crate with CPU reference |
| _post-wrapup_ | [rotorquant-strategy] RotorQuantStrategy joins kv-cache-benchmark |
| _post-wrapup_ | [router-heterogeneous-shards] capability-tagged routing in larql-router |
| `5cb199d` | [rotorquant-attention-integration] KvFormat side-table on KvCache |

## Bring-up

```bash
# CPU-only sanity (no GPU box required)
cargo check --workspace

# With CUDA enabled
cargo check --workspace --features 'larql-cli/cuda'

# Full GPU parity sweep (requires nvidia driver + CUDA 12.5 SDK)
LARQL_CUDA_AVAILABLE=1 \
LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
  make test-cuda

# Two-container topology (requires nvidia-container-toolkit)
make docker-up
```
