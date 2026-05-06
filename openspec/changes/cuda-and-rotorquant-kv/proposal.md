## Why

LARQL today targets Apple Silicon Metal as its only GPU backend. The
development host moved to Linux + RTX 4090 + CUDA 13.1, and the deployment
target is shifting from a single Fly.io CPU box to a split topology: a CPU
container that serves the FFN expert bank from RAM, and a GPU container
that runs attention plus a compressed KV-cache. RotorQuant
(<https://github.com/scrya-com/rotorquant>) is the chosen KV-cache
compression — its `iso3` / `planar3` block-diagonal rotations beat
Walsh-Hadamard TurboQuant on perplexity, decode tok/s, and prefill tok/s,
and it ships production CUDA kernels via a llama.cpp fork.

The Mac-only assumption now blocks every workstream: extraction, COMPILE,
inference goldens, and CI. Adding a real CUDA path also unlocks
RotorQuant, which has no Metal port at production quality. This change
lands the framework, the deployment topology, and a working CUDA
scaffold; subsequent changes fill in the kernel surface.

## What Changes

- ADD `compute-cuda-kernels` capability — CUDA backend in `larql-compute`
  paralleling the existing Metal one. cuBLAS for f32 GEMM/GEMV, custom
  CUDA kernels (or vendored `llama.cpp` kernels) for Q4_0 / Q4_K /
  Q4_KF / Q6_K matvec, fused decode-time attention, GEGLU, RoPE.
  Built behind a `cuda` feature flag (off by default).
- ADD `kv-cache-rotorquant` capability — RotorQuant K/V cache
  compression integrated into the attention path. Two production modes
  (`planar3`, `iso3`) plus 4-bit variants (`planar4`, `iso4`).
  Deferred-K to avoid prefill-time error compounding; explicit inverse
  rotation on V dequantize.
- ADD `server-attention-service` capability — new HTTP and gRPC routes
  on `larql-server` that expose attention prefill and decode as a
  remote service, including a KV-cache snapshot/restore protocol so
  the GPU container can hold session state across calls.
- ADD `deploy-cpu-gpu-split` capability — `deploy/docker/` directory
  with one Dockerfile per role (CPU FFN server, GPU attention server),
  a `docker-compose.yml` that wires them together with shared vindex
  storage, and a README explaining the topology, VRAM budget, and how
  to run end-to-end on a single RTX 4090 / 24 GB box.
- MODIFY `compute-backend-traits` to expose new capability bits
  (`KvCompressionRotorQuant`, `FlashAttentionV2`, `Cuda`) and a
  `default_backend()` that prefers CUDA on Linux+CUDA, Metal on macOS,
  CPU otherwise.
- MODIFY `inference-attention-and-kv` to allow the KV-cache type to be
  RotorQuant-compressed; surgery operations (`get_layer`, `set_layer`,
  `clone_layer_position_range`) take an optional rotation table.
- MODIFY `inference-residual-engine` to add a `RotorQuantEngine` /
  `IsoQuantEngine` engine alongside the existing TurboQuant /
  Markov / Apollo / UnlimitedContext set, so the engine selection
  hooks naturally pick it up.
- MODIFY `router-grid` to allow heterogeneous shards: an "attention
  shard" advertises `attention` capability and accepts attention RPCs;
  an "expert shard" advertises `expert` capability and accepts FFN
  batch RPCs. The router routes by capability and layer range.
- MODIFY `kv-cache-benchmark-strategies` to add `RotorQuantStrategy`
  (the iso3 + planar3 variants) so the existing accuracy / compression
  / decode-tok-s harness measures the new compression alongside the
  baselines.
- MODIFY `server-vindex-loading` to gate FFN-only vs attention-only
  loading by container role, so the FFN container does not pay the
  cost of loading attention weights and vice versa.

This is **non-breaking** for existing Metal / CPU users. The CUDA path
is opt-in via `--features cuda`. The split topology is opt-in via the
new compose stack — the existing single-binary mode keeps working.

## Capabilities

### New Capabilities

- `compute-cuda-kernels`: CUDA backend in `larql-compute`. cuBLAS for
  f32 paths, vendored / hand-rolled CUDA C++ kernels for Q4_0 / Q4_K /
  Q4_KF / Q6_K, fused QKV+norm and KV+attend kernels, partial-RoPE.
  Compiles via `build.rs` against `nvcc` when the `cuda` feature is on
  and a CUDA toolkit is detected; otherwise the feature gate is a
  build error with a clear install hint.
- `kv-cache-rotorquant`: K/V cache compression via block-diagonal
  rotations (`planar3` 2D Givens, `iso3` 4D quaternion, plus 4-bit
  variants). Production CUDA kernels are vendored from the
  `feature/planarquant-kv-cache` branch of llama.cpp; the API surface
  exposes `quantize_planar3`, `quantize_iso3`, and matching
  `dequantize_*_with_inverse_rotation` entrypoints. A small Python
  reference shim lives under `crates/larql-rotorquant/ref/` to verify
  the Rust path against the upstream Triton implementation in CI.
- `server-attention-service`: New HTTP + gRPC surface on
  `larql-server` that runs the attention block remotely. Endpoints:
  `/v1/attention/prefill`, `/v1/attention/decode`, `/v1/kv-cache/snapshot`,
  `/v1/kv-cache/restore`, `/v1/kv-cache/free`. Designed to be loaded
  with attention weights only (no FFN, no embeddings) so the GPU
  container's VRAM budget is dominated by KV cache.
- `deploy-cpu-gpu-split`: `deploy/docker/` deliverables. Two
  Dockerfiles (CPU FFN, GPU attention), one compose file, one README.
  Single-box and split-box modes. Documents the VRAM budget on a
  24 GB GPU for Gemma 4B, Llama 3 8B, and Qwen 2.5 14B with iso3 KV.

### Modified Capabilities

- `compute-backend-traits`: adds `Capability::Cuda`,
  `Capability::FlashAttentionV2`, `Capability::KvCompressionRotorQuant`;
  `default_backend()` prefers CUDA on Linux when the `cuda` feature is
  enabled.
- `inference-attention-and-kv`: KV cache becomes parameterised by a
  `KvFormat` (Fp16, Iso3, Planar3, Iso4, Planar4); attention surgery
  operations transparently quantize on insert and dequantize on read.
- `inference-residual-engine`: `RotorQuantEngine` joins the engine
  registry alongside `TurboQuantEngine`. `Apollo` and `MarkovResidual`
  remain unchanged.
- `router-grid`: shards declare a capability set
  (`{attention, expert}`); routes go to a shard that advertises the
  required capability for the requested layer.
- `kv-cache-benchmark-strategies`: `RotorQuantStrategy { variant:
  Iso3 | Planar3 | Iso4 | Planar4 }` joins the existing strategy
  enum, with measurements anchored on the same Gemma 3 4B / wikitext
  fixture.
- `server-vindex-loading`: a new `--role` flag (`ffn` | `attention` |
  `all`) constrains which weight families the bootstrap loads.

## Impact

- **New deps**: `cudarc` (high-level CUDA bindings), gated to Linux +
  `cuda` feature. Vendored llama.cpp CUDA kernels under
  `crates/larql-rotorquant/cuda/` (MIT, header-only style — no rebuild
  of the whole library).
- **New crates**: `larql-rotorquant` (FFI + safe Rust API around the
  vendored kernels). Joins the workspace as a new member.
- **Build prerequisites**: CUDA 12+ toolkit on the build host for the
  GPU container; the CPU container build is unchanged. `nvcc` invoked
  from `build.rs`; `cc-rs` orchestrates compile flags.
- **Dockerfiles**: `deploy/docker/Dockerfile.ffn` (Linux,
  `cargo build --release -p larql-server`, no GPU deps),
  `deploy/docker/Dockerfile.gpu` (`nvidia/cuda:13.1-devel-ubuntu24.04`
  base, `cargo build --release -p larql-server --features cuda`).
- **Compose**: `deploy/docker/docker-compose.yml` defines `ffn` and
  `attention` services, a shared `vindex_data` volume, and a
  `larql-router` service that routes between them.
- **CI**: a new GitHub Actions matrix entry for `cargo check
  --features cuda` on a CUDA-enabled runner; the existing build
  matrix is unchanged. Coverage thresholds for the new
  `compute-cuda-kernels` capability start at 0% and ratchet up as the
  kernels land.
- **Risk**: kernel maintenance burden. Mitigation — vendor llama.cpp's
  RotorQuant kernels rather than hand-rolling, and keep the surface
  small enough that a single engineer can audit it.
- **Out of scope**: macOS dropping. Metal stays as a first-class
  backend; this proposal makes it not the *only* one.
