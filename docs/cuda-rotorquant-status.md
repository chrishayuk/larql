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
  including Gemma 4B head_dim=320. CUDA module exposes the upstream
  packed FP16 → RotorQuant block-copy kernels and the matching
  RotorQuant → FP32 dequantize kernels under `--features cuda`.
- ✅ `rotorquant-strategy`: `RotorQuantStrategy` joins the
  `KvStrategy` trait family in `kv-cache-benchmark`. Four
  constructors (`iso3`, `planar3`, `iso4`, `planar4`) plumb
  `larql-rotorquant`'s CPU reference into the same harness used by
  Standard / TurboQuant / Markov / Apollo. **3/3** strategy tests
  pass.
- ✅ `cuda-decode-backend` RotorQuant wrapper completion:
  `larql-rotorquant` now round-trips all four active packed CUDA
  formats on-device. 3-bit packed formats preserve direction at
  ~0.985 cosine on synthetic Gemma-shaped KV rows; 4-bit packed
  formats reach ~0.994. `CudaBackend` now advertises
  `Capability::KvCompressionRotorQuant` after the CUDA round-trip
  test passes.

  RTX 4090 benchmark, 16,384 × 320 values, 20 iterations:

  | Format | KV bytes vs FP16 | cosine | quantize+dequantize ms/iter | logical FP16 GiB/s |
  | --- | ---: | ---: | ---: | ---: |
  | Planar3 | 19.53% | 0.984766 | 0.3920 | 24.92 |
  | Planar4 | 26.56% | 0.994617 | 0.3007 | 32.48 |
  | Iso3 | 19.53% | 0.984643 | 0.3933 | 24.83 |
  | Iso4 | 26.56% | 0.993546 | 0.3021 | 32.33 |

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
- **`attention-service-routes`** — partially shipped on the
  `feat/attention-service-routes` branch (proposal + design +
  tasks + capability deltas validated). What's already on disk:
  - `larql_server::kv_snapshot` — versioned binary wire format
    with magic `0x4C415141` ('LAQA'). 8/8 round-trip tests pass
    (FP32 byte-identical, Iso4 field-by-field, magic/version
    rejection, truncation cleanly).
  - `larql_server::attention_session::{SessionId,
    AttentionSession, AttentionSessionMap}` — ULID ids, per-
    session tokio::RwLock, std-RwLock map, configurable cap +
    TTL, reaper hook. 9/9 tests.
  - HTTP routes — session create / get / delete + KV-cache
    snapshot / restore / free (12/12 tests). Prefill / decode
    are slot-reserved; their handlers wire in alongside the
    model-side attention runner.
  - `--role attention | expert | both` CLI flag drives
    `AnnounceMsg.capabilities`. 5/5 tests.
  - Proto extension: `AnnounceMsg.capabilities`,
    `HeartbeatMsg.cached_prefixes`. 4/4 router tests round-trip
    the bloom through the wire.
  - 30 s tokio reaper task spawned from bootstrap.
  - Heartbeat closure rebuilds the bloom from
    `SessionMap::prefix_hashes(16)` on every tick.
  - `docs/attention-service-protocol.md` — wire-format reference
    with payload diagrams + curl/httpx examples.
  - `deploy/docker/start.sh` translates `ROLE` to the new
    `--role` flag; compose env documents
    `ATTN_SESSION_TTL` / `MAX_ATTN_SESSIONS`.
- **`rotorquant-cuda-kernels`** — proposal-only. Four PTX kernels
  (Iso/Planar × quantize/dequantize) compiled via cudarc NVRTC,
  cached on disk. Round-trip cosine ≥ 0.99 vs CPU reference;
  ≥ 10× speedup target on RTX 4090. Flips
  `Capability::KvCompressionRotorQuant` on `CudaBackend`. See
  `openspec/changes/rotorquant-cuda-kernels/`.
- **`engine-rotorquant-auto-compress`** — ⚠ **BLOCKED** during
  implementation. Discovery: existing engines
  (UnlimitedContext, Markov) don't hold a `KvCache` struct —
  they have their own per-engine state. The proposed
  `cache_mut() -> Option<&mut KvCache>` trait method would
  return `None` for every existing engine. Prerequisite
  identified: `engine-kvcache-unification` (refactor
  UnlimitedContextEngine to use the shared `KvCache` type).
  See `openspec/changes/engine-rotorquant-auto-compress/proposal.md`'s
  "Status: blocked" section.
- ✅ **`rotorquant-promote-on-read`** — `KvCache::get_layer`
  signature changed from `&self` to `&mut self` and now
  auto-promotes compressed layers via `dequantize_layer`. Added
  `get_layer_lazy(&self)` no-promote variant for snapshot
  callers. `promote_on_read_count: u64` metric increments per
  successful auto-promote (not per cache hit). `clear_layer` now
  clears both FP32 + compressed slots symmetrically.
  **22/22 attention::decode tests pass** including 4 new for
  promote-on-read.
- **`deploy-compose-end-to-end`** — `docker compose up` boots
  Gemma 4B end-to-end through the router; `make demo` target
  produces a one-shot inference; PERFORMANCE.md gets the measured
  tok/s + VRAM column.

### SMG-derived backlog (now partly implemented)

After analysing the PyTorch / LightSeek SMG blog post we drafted
three sub-changes; two are now shipped, one stays as proposal:

- ✅ **`server-tokenizer-cache`** — L0 exact-match + L1 prefix-aware
  cache in front of `Tokenizer::encode`. `larql_server::tokenizer_cache::TokenizerCache`
  with 7/7 tests passing. SMG-derived: 23% TTFT reduction at 256
  concurrency. Wire-up to route handlers is the next bite-sized
  follow-up; the cache type itself is ready.
- ✅ **`router-prefix-aware-routing`** — `ServerEntry::cached_prefixes:
  PrefixBloom` (256-bit, splitmix64-derived k=4 hash positions);
  `GridState::route_for_prefix` picks the shard with most matches
  and falls back to least-loaded. 5/5 new grid tests pass (30/30
  total). FP rate bound at design load (n=16) is ≤ 1.5%; degrades
  to ~16% at n=64 (proposal corrected — earlier 1.5%@n=64 was
  numerically wrong).
- **`attention-service-prefill-decode-split`** — proposal-only;
  extends the planned
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
backend.supports(Capability::KvCompressionRotorQuant)  // true (cuda-decode-backend)
backend.supports(Capability::DecodeToken)              // true (cuda-decode-backend)
backend.supports(Capability::PrefillQ4)                // true (cuda-decode-backend)
```

## What `make test-cuda` exercises

```
cargo test -p larql-compute --features cuda --test test_cuda_f32   →  9 tests
cargo test -p larql-compute --features cuda --test test_cuda_q4    →  5 tests
cargo test -p larql-compute --features cuda --test test_cuda_attn  →  6 tests
cargo test -p larql-rotorquant                                     →  CPU reference tests
cargo test -p larql-rotorquant --features cuda --test cuda_round_trip
                                                                 →  4 CUDA RotorQuant tests
cargo test -p kv-cache-benchmark --lib rotorquant                  →  3 tests
cargo test -p larql-router --bin larql-router grid::tests          → 30 tests (+19: prefix-aware + bloom)
cargo test -p larql-inference --lib attention::decode               → 22 tests (+4: promote-on-read)
cargo test -p larql-server --lib tokenizer_cache                    →  7 tests (new)
                                                                   ────
                                                                    92 tests
```

Plus shipped on the `fix/encode-cached-ids-sync` PR branch:
`server-tokenizer-cache` is now wired into all nine
`Tokenizer::encode` call sites in `larql-server`
(`routes/insert.rs`, `routes/stream.rs`,
`routes/openai/{chat,completions}.rs`, `routes/patches.rs`,
`grpc.rs`). The `LoadedModel::encode_cached_ids(&str, bool)`
helper is sync; cache hits skip BPE for the chat-template
prefix shared across requests.

All require `LARQL_CUDA_AVAILABLE=1` for the GPU-gated subset; the
RotorQuant tests run anywhere (CPU reference only today).

## What `make ci` reports today

**Green** on `fix/encode-cached-ids-sync` (PR pending; main is
currently broken from commit `4ffdc89` which migrated
`routes/insert.rs` before `encode_cached_ids` had been changed
from `async fn` to `fn`).

The PR also catches the workspace up to Rust 1.95's tightened
clippy: a `[workspace.lints]` block in the root `Cargo.toml`
allows the most pedantic of the newly-promoted lints
(`too_many_arguments`, `manual_is_multiple_of`,
`unnecessary_sort_by`, `manual_div_ceil`, etc.) which would
otherwise demand churning every dev/exploratory module
(`larql-cli/.../ov_rd/`, `parity.rs`, `shannon_cmd.rs`). New
code should still satisfy them; this just keeps `cargo clippy
--workspace -- -D warnings` from drowning out real issues.

## Known pre-existing breakage I did NOT fix

These are workspace issues that predate the CUDA workstream and are
not on the critical path for it:

- ~~`crates/larql-server/tests/test_expert_endpoint.rs` —
  `MoeLayerWeights` API drift~~. **Repaired** on
  `fix/encode-cached-ids-sync` (4/4 tests now pass; this also
  unblocks `attention-service-routes`).

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
| `becb5cf` | [rotorquant-promote-on-read] auto-promote on cache read |
| `263bf6d` | [server-tokenizer-cache] [router-prefix-aware-routing] |
| `1cedd8e`+`4ffdc89` | wire TokenizerCache into LoadedModel + first call site |
| `0fc21a8` (PR) | make encode_cached_ids sync to match call sites |
| `a00f28c` (PR) | migrate remaining 8 encode call sites |
| `07d2399`+`cd23265` (PR) | rust 1.95 clippy + workspace lint policy |

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
