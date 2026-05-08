## Context

LARQL's compute layer is built around a `ComputeBackend` trait family
(`MatMul`, `QuantMatVec`, `DecodeBackend`, `Capability`) with two
implementations today:

- **CPU**: BLAS f32 (Accelerate / OpenBLAS), hand-rolled C kernels for
  Q4_0 / Q4_K / Q4_KF / Q6_K / Q8 matvec, GEGLU, fused attention,
  vector ops. ~30k lines.
- **Metal**: Apple Silicon GPU. Tiled f32, simdgroup Q4 / Q6, fused
  QKV+norm, fused QK+rope+softcap, fused KV+attend, FFN gate+up+GEGLU+down,
  hybrid CPU/GPU dispatch. ~30k lines, MSL shaders + dispatch
  orchestration.

KV-cache strategies live in `larql-inference::engines` and
`kv-cache-benchmark`: Standard FP16, TurboQuant (Walsh-Hadamard +
Lloyd-Max, 3 / 4 bit), Markov Residual, UnlimitedContext, Apollo. The
benchmark crate is the source of truth for accuracy / compression
numbers.

The user moved to a Linux + RTX 4090 dev box with CUDA 13.1 driver and
docker / podman available. The deployment goal is to split the inference
loop across two containers so we can put attention + KV-cache on the
24 GB GPU and the FFN expert bank in CPU RAM (Gemma 4 4B's expert bank
is ~10 GB even quantised, more for larger models). RotorQuant
(<https://github.com/scrya-com/rotorquant>) is the chosen KV
compression — its `iso3` / `planar3` block-diagonal rotations beat
Walsh-Hadamard TurboQuant on perplexity, decode tok/s, and prefill tok/s
on Llama 3.1 8B, with a 10.3× compression ratio.

## Goals / Non-Goals

**Goals:**

- A working CUDA backend that boots, declares its capability set, and
  runs at least one end-to-end forward pass on real weights through
  the `ComputeBackend` interface — even if the kernel surface is
  initially small (cuBLAS f32 GEMM only). Subsequent changes fill in
  the quantised paths.
- RotorQuant integrated as a `KvFormat` variant the attention path
  understands. Both production variants (`iso3`, `planar3`) work
  through the existing KV-surgery API. The benchmark harness measures
  it.
- A two-container deployment topology runnable on this RTX 4090 box
  via `docker compose up`. CPU container holds FFN experts; GPU
  container holds attention weights and the KV-cache.
- The router supports heterogeneous capabilities so a single inference
  request can hop between the GPU container (attention layers) and
  CPU container (FFN experts).
- The Mac / Metal path keeps working unchanged. CI on macOS keeps
  passing.

**Non-Goals:**

- Beating llama.cpp on raw decode tok/s. We're after correctness +
  reproducibility + integration with vindex.
- Multi-GPU. RTX 4090 24 GB is the target. Multi-GPU is a future
  change once the single-GPU path is solid.
- Removing TurboQuant. Existing benchmarks reference it; the engine
  registry stays additive.
- Production hardening of the deploy artifacts — the compose stack is
  for development and small deployments. K8s manifests are out of
  scope.

## Decisions

### D1 — `cudarc` over manual `bindgen`

Two ways to talk to CUDA from Rust:

- **`cudarc`** (high-level wrapper): safe APIs over driver / runtime /
  cuBLAS / cuDNN, runtime PTX compilation via NVRTC, builds with no
  CUDA SDK at compile time (only at runtime).
- **`bindgen` against `cuda.h` + manual FFI**: total control, every
  warning is yours, no runtime PTX compile.

Chose **cudarc** because: (a) it dramatically shortens the path to a
first working f32 GEMM, (b) NVRTC lets us ship custom kernels as
strings without an `nvcc` build-time dep, (c) it tracks recent CUDA
versions, (d) it has a maintained cuBLAS wrapper. Trade-off: we depend
on `cudarc` keeping pace with CUDA toolkit updates, and PTX
compile-at-startup adds ~50 ms to first-kernel latency. Both
acceptable.

For the RotorQuant kernels we vendor pre-built `.cu` source from the
llama.cpp fork rather than runtime-compiling — those kernels are tuned
and we want to compile them with `nvcc` for max performance. So the
GPU container's Dockerfile installs the CUDA toolkit; non-RotorQuant
paths run with cudarc + NVRTC and need only the runtime.

### D2 — Vendor RotorQuant CUDA kernels under a new crate

Three integration paths considered:

- **Submodule the upstream llama.cpp fork**: full library, maintained
  upstream, but pulls in 200k+ lines we don't need.
- **Vendor only the kernel `.cu` files** (and a small C++ shim) under
  `crates/larql-rotorquant/cuda/`: ~2k lines, manual upstream sync,
  full control over compile flags.
- **Re-implement in Rust + cudarc PTX strings**: tempting, but the
  reference accuracy numbers come from the upstream kernels and we
  don't want to chase the same bug RotorQuant's commit `6e5a4aa`
  caught (V dequantize must use the *inverse* rotation; PPL went from
  15K to 7.05 after that fix).

Chose **vendoring**. License is MIT, the kernel surface is small, and
upstream sync is a manual diff a couple of times a year. Source lives
under `crates/larql-rotorquant/cuda/{planar3,iso3,planar4,iso4}.cu`
with `crates/larql-rotorquant/UPSTREAM.md` recording the source
commit hash and any local patches.

### D3 — Two-container topology over single-container with `--gpus=all`

A single container with both attention and FFN simplifies dev. We
chose split because:

- Attention container needs CUDA runtime + `nvidia-container-toolkit`
  + a base image like `nvidia/cuda:13.1-devel-ubuntu24.04` — large
  (~3 GB compressed) and only relevant where there's a GPU.
- FFN container is plain `ubuntu:24.04` + OpenBLAS — small (~250 MB)
  and runs anywhere, including the existing Fly.io deployment.
- Future: an attention shard might run on a different host than the
  FFN shard. The split forces us to design the wire format from day
  one.

Compose orchestrates the two services on the dev box. Production
deployments (Fly, K8s, bare metal) reuse the Dockerfiles independently.

### D4 — Attention service wire format: gRPC primary, HTTP/JSON debug fallback

Mirror the existing `expert-service` design. gRPC for production
(streaming-friendly, binary, codegen via `larql-router-protocol`),
HTTP/JSON for debugging (`/v1/attention/decode`, etc.). The KV-cache
snapshot RPC is gRPC-only because the payload is binary and large
(tens of MB per snapshot).

KV-cache transport: an opaque `KvHandle` (server-issued u128) passed
between calls. The client never sees the cache contents; the server
holds them in VRAM. Snapshots are reified blobs the client can store
to disk and restore later (e.g., for session resumption). Format:
`{ format: KvFormat, layers: [LayerKv { K: bytes, V: bytes, norms: bytes, rotation_indices: bytes }] }` — same shape across `Iso3`, `Planar3`, `Iso4`, `Planar4`, with size-per-element governed by the format.

### D5 — KV format selection lives at session level, not request level

Quantising K/V costs CPU+GPU cycles. Switching format mid-session
would require re-quantising the entire cache. Therefore: the format
is fixed at session-create time (`POST /v1/attention/session`), and
all subsequent prefill/decode/snapshot calls use it.

### D6 — `default_backend()` precedence: CUDA → Metal → CPU

On Linux with `cuda` feature enabled and a working CUDA runtime:
return `CudaBackend`. On macOS with `metal` feature: return
`MetalBackend`. Otherwise: return `CpuBackend`. Override via env:
`LARQL_BACKEND=cpu|metal|cuda`. Tests that need a specific backend
construct it directly.

### D7 — `larql-rotorquant` is a top-level workspace member, not a crate inside `larql-compute`

We considered making RotorQuant an internal module of `larql-compute`.
Rejected because:

- The vendored CUDA kernels are 2k lines that recompile slowly. Putting
  them in their own crate means edits to other compute kernels don't
  trigger a rebuild.
- A future "attention service" wants RotorQuant without pulling all of
  `larql-compute`'s Metal / CPU baggage.
- Crate-level licensing: RotorQuant is MIT (matches LARQL Apache-2.0,
  but the directory boundary makes the vendored license clearer).

### D8 — Phased roll-in via OpenSpec sub-changes

This change is a framework change. Real kernel work is a series of
follow-ups:

| Sub-change | Scope | Estimated complexity |
|---|---|---|
| `cuda-f32-baseline` | cuBLAS GEMM/GEMV via cudarc; default_backend wiring; first golden test passes | medium |
| `cuda-q4-matvec` | Q4_0 / Q4_K kernels; parity vs CPU on the existing Q4_K parity tests | high |
| `cuda-fused-attention` | Fused QKV+norm + KV+attend on FP16 KV | high |
| `rotorquant-kernels` | Vendor + build planar3/iso3 kernels; round-trip tests | medium |
| `rotorquant-attention-integration` | Hook into the attention forward; KV-cache surgery uses KvFormat | medium |
| `attention-service-routes` | HTTP + gRPC routes; session lifecycle | medium |
| `router-heterogeneous-shards` | Capability-tagged shards; capability-aware routing | medium |
| `deploy-compose-end-to-end` | docker compose up runs Gemma 4B end-to-end | low |

Each sub-change references this change's specs as source of truth and
its own delta narrows. This keeps PR review tractable.

## Risks / Trade-offs

- **VRAM budget on 24 GB**: Llama 3 8B with iso3 KV at 128k context is
  ~4 GB KV cache + ~5 GB attention weights + ~1 GB workspace ≈ 10 GB.
  Comfortable. Qwen 2.5 14B is ~12 GB attention + ~7 GB KV ≈ 20 GB —
  tight. Larger models need MIG or multi-GPU; out of scope. **Mitigation:**
  document supported model sizes on a 24 GB GPU in the README; gate
  larger models behind a capability check at session-create.
- **Rotation-table mismatch corruption**: applying the wrong rotation
  inverse on V dequantize silently produces garbage outputs (this is
  exactly what the upstream commit `6e5a4aa` fixed). **Mitigation:**
  the rotation table is part of the `KvHandle`'s server-side state;
  clients cannot construct one. Round-trip tests in
  `crates/larql-rotorquant/tests/` exercise quant→dequant against
  the upstream Triton reference for `iso3` and `planar3`.
- **CUDA toolkit drift**: cudarc tracks CUDA versions but we'll
  occasionally hit a regression on a new toolkit. **Mitigation:** pin
  `cudarc` to a known-good minor version; the GPU Dockerfile pins the
  CUDA base image tag so dev / CI / prod resolve to the same toolkit.
- **Two-container coordination latency**: each FFN call from GPU
  container to CPU container is a network round-trip. Even on
  loopback that's ~0.5–1 ms per layer. For a 32-layer model in decode
  mode that's 16–32 ms per token of pure transport. **Mitigation:**
  same as the existing remote FFN path — batch FFN calls per layer,
  pipeline ahead by one layer, use HTTP/2 or gRPC streaming. The
  benchmark harness will report the cost so we know what we're paying.
- **Kernel maintenance**: vendoring CUDA `.cu` files means upstream
  improvements are not free. **Mitigation:** `UPSTREAM.md` records
  the upstream commit; a `make rotorquant-sync` target diffs the
  vendored copy against the upstream, surfacing changes a human can
  review.
- **`cudarc` PTX startup cost**: ~50 ms first time we compile a
  custom kernel. **Mitigation:** compile all custom kernels at
  backend init, not lazily on first dispatch; cache the resulting
  modules in `~/.cache/larql/cudarc/` keyed by toolkit version + GPU
  arch + source hash.
- **Heterogeneous router can deadlock**: if the GPU container blocks
  on FFN result and the CPU container blocks on KV-cache update, you
  can manufacture a deadlock. **Mitigation:** the router enforces a
  one-way data flow per layer (attention output → FFN, FFN output →
  next-layer attention) and times out individual hops.
- **Container build times**: the GPU image is ~3 GB and takes minutes
  to build. **Mitigation:** multi-stage Docker builds with aggressive
  cache, separate builder image; document `make docker-gpu` and
  `make docker-ffn` Makefile targets.

## Migration Plan

1. Land this change. No behavioural changes — just inventory + scaffold.
2. Sub-change `cuda-f32-baseline`: real cudarc + cuBLAS GEMM. First
   end-to-end inference run on RTX 4090 in the GPU container.
3. Sub-change `rotorquant-kernels`: vendor + compile + round-trip
   tests.
4. Sub-change `rotorquant-attention-integration`: KV cache becomes
   format-parameterised; tests pass.
5. Sub-change `attention-service-routes`: GPU container exposes the
   service.
6. Sub-change `router-heterogeneous-shards`: router routes by
   capability.
7. Sub-change `deploy-compose-end-to-end`: docker compose up runs
   Gemma 4B end-to-end with iso3 KV.

Rollback: each sub-change is independent. Reverting any of them does
not break preceding ones.

## Open Questions

- **Q1: cudarc version pin**. The current latest is 0.13. Do we pin a
  minor version or take patch updates? Recommendation: pin to
  `=0.13.x` and bump deliberately.
- **Q2: Where do KV snapshots live**? Server-side temp files? Client
  responsibility? Recommendation: client-driven — the server returns
  bytes on snapshot, the client stores them, and the server frees the
  in-VRAM cache on `kv-cache/free`. Avoids the server needing
  persistent storage.
- **Q3: Should we support GPU arch < 89 (Ada Lovelace)?** The 4090 is
  sm_89. Older cards (3090, 3060) are sm_86. Most kernels we vendor
  work on sm_70+. Recommendation: target sm_70 minimum but tune for
  sm_89; let users override via `LARQL_CUDA_ARCH`.
- **Q4: Flash-Attention vs hand-rolled fused attention?** The
  `flash-attn` library is 100k+ lines and not trivially vendorable.
  llama.cpp's fused attention is simpler and good enough for our
  scale. Recommendation: start with llama.cpp's, leave the door open
  to flash-attn via a feature flag.
