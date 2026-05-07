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

## Not yet shipped (known follow-up sub-changes)

These are designed and have spec scenarios attached to the parent
change but aren't implemented yet. Each is a contained piece of work
that benefits from explicit human review before launching.

- **`rotorquant-attention-integration`** — wires `KvFormat` into
  `larql_inference::attention::KvCache`; deferred-K behaviour during
  prefill; KV-surgery operations transparently quantise on insert /
  dequant on read; `RotorQuantStrategy` joins
  `kv-cache-benchmark::strategies`. ~3–4 days of careful
  inference-path work.
- **`attention-service-routes`** — new HTTP + gRPC endpoints on
  `larql-server` (`/v1/attention/{session,prefill,decode}`,
  `/v1/kv-cache/{snapshot,restore,free}`); session lifecycle;
  binary KV-snapshot wire format. Depends on test-fixture drift in
  `larql-server/tests/test_expert_endpoint.rs` being repaired (the
  `MoeLayerWeights` API moved from single buffers to per-expert
  `Vec<&[u8]>`).
- **`router-heterogeneous-shards`** — `larql-router` accepts
  `capabilities: ["attention" | "expert"]` on shard registration;
  routes by capability + layer range; per-hop deadline timeout
  prevents heterogeneous deadlocks.
- **`deploy-compose-end-to-end`** — `docker compose up` boots
  Gemma 4B end-to-end through the router; `make demo` target
  produces a one-shot inference; PERFORMANCE.md gets the measured
  tok/s + VRAM column.

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
                                                                   ────
                                                                    30 tests
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
