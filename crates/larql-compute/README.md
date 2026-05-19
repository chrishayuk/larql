# larql-compute

Hardware-accelerated compute backends for LARQL. CPU (BLAS + NEON Q4), Metal GPU, and future CUDA.

## What it does

Provides a `ComputeBackend` trait that abstracts all hardware-specific matrix operations. Every LARQL crate (inference, vindex) uses this trait — the caller never knows whether the operation runs on CPU or GPU.

The trait is split into four sub-traits, each with its own focus:

| Sub-trait | What's there |
|---|---|
| [`MatMul`](src/backend/matmul.rs) | f32 / f16 matmul, `matmul_transb`, `f32_gemv`, `f16_gemv`, batch matmul |
| [`QuantMatVec`](src/backend/quant_matvec.rs) | unified `quant_matvec(format, …)` + per-format pre-quantised fast paths |
| [`DecodeBackend`](src/backend/decode.rs) | KV-cached decode + multi-position prefill + MoE hook |
| (umbrella) `ComputeBackend` | `name`, `device_info`, `Capability`-based feature probe |

Most callers stay typed against `&dyn ComputeBackend`; `use larql_compute::prelude::*;` brings every sub-trait in scope at once.

## Adding a new quant format

Adding e.g. FP4 = one `QuantFormat` enum variant + one match arm in `QuantMatVec::quant_matvec`'s default impl + one CPU kernel + one Metal shader. The Metal shader gets a `Kernel` marker (impl `metal::kernel::TiledKernel`) so its name + dispatch geometry travel with it — no separate constants importing.

## Backends

| Backend | Feature flag | f32 matmul | Quantized ops | Pipeline |
|---------|-------------|------------|---------------|---------|
| **CPU** | (always) | BLAS (Accelerate AMX) | C kernel (ARM vdotq_s32) | Sequential |
| **Metal** | `--features metal` | Tiled shaders | Simdgroup Q4/Q4_K/Q6_K/Q8 | One command buffer |
| **CUDA** | (planned) | — | — | — |

## Performance vs Ollama

Live `larql bench gemma3-4b-q4k-v2 --ollama gemma3:4b`
on M3 Max (2026-05-09, post QKV defuse):

```
  larql-metal  88 tok/s     11.4ms/tok   (GPU fwd ~11.40ms, lm_head ~1.85ms)
  ollama       ~103 tok/s   ~9.7ms/tok
  gap          1.17×        ~1.66ms/tok
```

Reproduce: `larql bench <vindex> --backends metal --ollama <tag>`.
**Acceptance criterion (~85 tok/s, ~1.16×) effectively met.** See
`PERFORMANCE.md` for the full breakdown, [ADR-015](docs/adr/015-isolated-vs-batched-kernel-perf.md)
(iso-vs-batched diagnostic order), [ADR-016](docs/adr/016-defused-rms-norm-qkv.md)
(QKV defuse rationale), [ADR-017](docs/adr/017-shader-retention-model-agnosticity.md)
(shader retention under model agnosticity), [ADR-018](docs/adr/018-architecture-shader-routing.md)
(architecture → shader routing), [docs/llama-cpp-comparison.md](docs/llama-cpp-comparison.md)
(kernel-architecture comparison vs llama.cpp), and
[docs/architecture-shader-map.md](docs/architecture-shader-map.md)
(per-architecture shader dispatch map).

### Key optimisations

**2026-05-09 — QKV defuse (+1.6–1.8 tok/s on Gemma 3 4B; +0.4 on 26B MoE)**

| Optimization | Savings | Technique |
|---|---|---|
| Default flipped from fused `q4k_q6k_qkv_proj_normed` to separate `rms_norm` + non-fused `q4k_q6k_qkv_proj` | **+1.6–1.8 tok/s, −0.30 ms/tok GPU fwd** on Gemma 3 4B; **+0.4 tok/s** confirmed on Gemma 4 26B A4B (MoE) post-thermal-cooldown cross-arch validation | The fused kernel rolled RMS norm into Phase 1 of the matmul to save 1 dispatch/layer (~0.24 ms/tok), but each TG's 4 simdgroups independently re-traverse H + norm_w in Phase 2 (different stride patterns for Q4_K Q/K vs Q6_K V), dropping the kernel from 287 → 199 GB/s. The 1.46 ms/tok kernel cost exceeds the 0.24 ms/tok dispatch saving. ADR-016. `LARQL_QKV_FUSED=1` opts back in to the old fused path. |
| D-RMS-FUSE Phase 1 — `residual_norm_store` at post-FFN→next-input boundary on non-Gemma | **End-to-end NULL** (within drift). Predicted ~0.2 ms/tok; measured 0.0. | Fuses post-FFN `residual_add` + next layer's input rms_norm into one `residual_norm_store` dispatch. Bit-identical parity ✓ across Llama 2 7B / Mistral 7B / Gemma 3 4B (Gemma path untouched — already triple-fused via `post_ffn_norm_residual_add`). Kept opt-in `LARQL_FUSED_PRELAYER_NORM=1`. ADR-015 magnitude-compression at the extreme — the dispatch saving in layer N+1 is offset by `residual_norm_store` doing the RMS reduction inline where `residual_add` was just element-wise. |
| `make bench-cross-arch` — multi-arch decode bench infrastructure | Infrastructure (no perf change) | `scripts/bench-cross-arch.sh` + `bench/baselines/cross-arch/`. Operationalises ADR-017's model-agnosticity check. Multi-arch sweep automatically surfaces thermal artifacts (every-arch-regresses-simultaneously signature). |

**2026-05-02 — dispatch geometry fix (+8 tok/s on Gemma 3 4B, +14 tok/s on Gemma 4 26B A4B)**

| Optimization | Savings | Technique |
|---|---|---|
| `q4k_matvec` dispatch geometry from bound pipeline | **+7.7 tok/s on 4B / +14.3 tok/s on 26B** | Use `pipeline.rows_per_tg` / `threads_per_tg` instead of hardcoded 4sg shader-module constants; the 8sg pipeline (default since 2026-04-28) was being under-dispatched, leaving simdgroups 4..7 idle and half the rows unwritten. **Same family as 077884b's "81–84 tok/s on broken Q4_K dispatch"** — second confirmed instance. ADR-015 § "Lesson — diagnostic order for 'fast but wrong' results" |
| Promoted `lm_head_knn_backend` (q4k_matvec first) to default | (within above) | Stride-32 was the workaround for the pre-fix argmax drift; production now goes through the now-correct, faster q4k_matvec → f16 → f32 chain. `LARQL_LM_HEAD_SKIP_Q4K=1` for diagnostic A/B |

**Earlier optimisations (2026-04-25 → 2026-05-01)**

| Optimization | Savings | Technique |
|---|---|---|
| `q6k_matvec` ROWS_PER_TG 4→2 | +1-2 tok/s | 2× concurrent TGs → better DRAM latency hiding |
| `q6k_matvec` inter-superblock interleaving | +3 tok/s | Adjacent lanes read alternate superblocks; X preloaded; deferred scaling |
| `q6k_matvec` 4-element batching | +7 tok/s | Compile-time hi2 shifts, preloaded scales |
| Fused QK-norm Q+K (`qk_norm_qk`) | −0.17ms | One dispatch instead of two per layer |
| Fused RoPE Q+K (`rope_at_pos_batched_qk`) | −0.17ms | One dispatch instead of two |
| Fused residual+norm (`residual_norm_store`) | −0.17ms | Writes both normed and raw sum in one pass |
| Fused norm+QKV (`q4k_q6k_qkv_proj_normed`) | −0.17ms | Norm computed cooperatively inside QKV TGs |
| Cooperative SIMD norms | −10ms | O(N²)→O(N) reads (2026-04-09) |
| Q4_KF FFN routing | −8ms | llama.cpp-exact kernel (2026-04-09) |
| Buffer pre-allocation | −2ms | Eliminated 550 allocs/decode (2026-04-08) |

### Bottleneck analysis (from `diag_shader_bench`, post 2026-05-02)

| Kernel | Batched GB/s | ms/tok | Bound by |
|---|---|---|---|
| q6k_matvec (FFN down, K=10240) | ~312 GB/s | 2.35ms | bandwidth (84% of LPDDR5X peak) |
| q4k_ffn_gate_up_8sg (gate+up, K=2560) | ~275 GB/s | 3.64ms | bandwidth (74% of peak) |
| q4k_matvec (lm_head, 262K×2560) | (Q4_K, post fix) | 1.85ms | bandwidth + dequant |
| f32_gemv (legacy lm_head fallback) | ~387 GB/s | — | bandwidth (at peak) |

Both big FFN kernels are bandwidth-bound at 74–84% of LPDDR5X peak; no
single-kernel headroom remains. The remaining ~1.17× gap to ollama is
distributed across dispatch overhead + the ~30 ms/tok of CPU-side ops
(routing, KV append, sampling) — not a hot kernel waiting to be tuned.

### Architecture

Single command buffer + single global encoder for all 34 layers. Pre-allocated scratch
buffers. Format-aware FFN: Q4_KF routes through llama.cpp kernel, Q4_K through fused
gate+up, Q4_0 through legacy Q8 path. All norms use cooperative SIMD reduction.

## Shaders

Production kernels are in **bold**; the rest are either dispatched only by
diagnostic / fallback paths or compiled-but-unwired (kept around because
the shader source is small and the bench harness still exercises them).

| Category | Kernels | Notes |
|----------|---------|-------|
| f32 matmul | sgemm, sgemm_transb | Tiled 32×32 |
| f32/f16 gemv | **f32_gemv**, **f16_gemv** | LM head (large vocab × hidden) |
| Q4_0 matvec | **q4_matvec_v4** (prod), q4_f32_matvec, q4_vecmat | v4: uint32 wide loads, sub-block stride |
| Q4_K / Q4_KF | **q4k_matvec**, **q4k_qkv_proj**, **q4k_q6k_qkv_proj**, **q4k_q6k_qkv_proj_normed**, **q4kf_qkv_proj**, **q4kf_proj** | `_normed` variant computes RMS norm inline (saves 1 dispatch) |
| Q4_K fused FFN | **q4k_ffn_gate_up**, **q4kf_ffn_gate_up** | Fused gate+up with inter-superblock interleaving |
| Q4_K GEGLU+down | **q4k_geglu_silu_down**, **q4k_geglu_gelu_tanh_down** | Fused activation+down for all-Q4_K models |
| Q6_K | **q6k_matvec** | 2-way inter-superblock interleaving, X preload, deferred scaling |
| Q8 | **q8_matvec**, **q8_qkv_proj**, **quantize_q8** | Fused QKV, simdgroup reduction |
| Attention | **fused_attention** (RoPE+GQA+softcap), **kv_attention** (decode), **kv_cache_append** | SIMD reductions, float4 dot |
| Normalization | **rms_norm**, **layer_norm** / **layer_norm_no_bias**, **v_norm_batched**, **qk_norm**, **qk_norm_qk** | `qk_norm_qk` fuses Q+K heads in one dispatch |
| Activation | **geglu_silu**, **geglu_gelu_tanh**, **silu**, **gelu_tanh** | Gated + standalone |
| Element-wise | **residual_add**, **scale_vector** | |
| RoPE | **rope_apply** (prefill), **rope_at_pos** (single-head), **rope_at_pos_batched** (all heads), **rope_at_pos_batched_qk** (Q+K fused) | `_qk` saves 1 dispatch/layer |
| Fused residual+norm | **rms_norm_q8**, **residual_norm**, **residual_norm_q8**, **residual_norm_store** | `_store` writes both normed output AND raw sum in one dispatch |
| Experimental / unwired | causal_attention, q4_sparse_matvec, q6k_geglu_silu_down, q6k_geglu_gelu_tanh_down, v_norm (singleton), turboquant_encode/decode, graph_walk_knn | Kept compiled; not dispatched in production |

## Safe Buffer Access

All Metal buffer reads go through one audited function with null/size checks:

```rust
// Replaces 13 previous unsafe { from_raw_parts } sites
pub fn read_buffer_f32(buf: &metal::Buffer, len: usize) -> Vec<f32>
```

## Quick Start

```rust
use larql_compute::prelude::*;
use larql_compute::{default_backend, QuantFormat};

let backend = default_backend();
println!("Using: {} ({})", backend.name(), backend.device_info());

// f32 matmul
let c = backend.matmul_transb(a.view(), b.view());

// Unified quant matvec — dispatches on format. Q4_K / Q4_KF / Q6_K
// take f32 input directly; Q4_0 / Q8_0 internally re-quantise.
let scores = backend.quant_matvec(QuantFormat::Q4_K, &q4k_data, &x, rows, hidden);

// Pre-quantised fast path for hot decode loops (avoid re-quantising
// the layer's input on every gate/up matvec):
let scores = backend.q4_matvec(&q4_0_data, &q8_x, &q8_scales, rows, hidden);

// Capability probe — branch on what the backend accelerates instead
// of pattern-matching on `Option<…> = None`.
if backend.supports(Capability::F32Gemv) {
    let logits = backend.f32_gemv_force(lm_head.view(), &h_last);
}

// KV-cached decode (one token through all layers).
let h = backend.decode_token(&layers, &x, hidden, inter, q_dim, kv_dim,
    num_q_heads, num_kv_heads, head_dim, rope_base);

// GPU prefill (seq>1, populates KV cache).
let h = backend.prefill_q4(&layers, &x, hidden, inter, q_dim, kv_dim,
    seq_len, num_q_heads, num_kv_heads, head_dim, rope_base, qk_norm, softcap);
```

## KernelHandle and ShaderKernel: no raw strings at binding sites

Two traits in `metal::kernel`:

**`TiledKernel`** — for kernels dispatched with `dispatch_thread_groups` that need row geometry. Each shader file exports a `Kernel` marker implementing `TiledKernel { KERNEL_NAME, ROWS_PER_TG, THREADS_PER_TG }`. `KernelHandle::from_kernel::<…::Kernel>(device, library)` bundles the pipeline + geometry. Dispatchers read `kernel.rows_per_tg` — no parallel constants that can drift.

**`ShaderKernel`** — for flat-dispatch kernels (`dispatch_threads` or fixed-geometry `dispatch_thread_groups`) that don't need row geometry. Each shader file exports a marker implementing `ShaderKernel { KERNEL_NAME }`. `get_shader_pipeline::<T>(device, library)` looks up the kernel by that constant. All 31 previously magic-string `library.get_function("...")` calls in `MetalBackend::new()` now go through one of these two typed paths — renaming a shader without updating its marker is a compile error, not a silent runtime `None`.

Construction asserts `pipeline.maxTotalThreadsPerThreadgroup() >= threads_per_tg` (TiledKernel) so silent simdgroup drop is caught at startup. (See the `q4_matvec_v4` 75 %-row drop entry in `ROADMAP.md`.)

## Linear algebra primitives (`cpu/ops/linalg.rs`)

Beyond the matmul/quantization backends, `larql-compute` ships a small set
of pure-CPU f64 linear algebra primitives used by the higher crates:

| Primitive | Signature | Used by |
|-----------|-----------|---------|
| `cholesky(a, ridge)` | `(N,N) → L (N,N)` lower-triangular factor with optional ridge | MEMIT covariance solve, vindex MEMIT |
| `cholesky_solve(L, B)` | solves `L L^T X = B` for any `(N,m)` RHS | as above |
| `cholesky_inverse(L)` | A⁻¹ = `cholesky_solve(L, I)` | covariance whitening |
| `ridge_decomposition_solve(K, T, λ)` | closed-form `ΔW = T^T (K K^T + λI)⁻¹ K`, returns `(d,d)` | `larql_vindex::memit_solve` (COMPACT MAJOR) |

The N×N Cholesky runs in f64 — `K K^T` becomes ill-conditioned in f32 when
keys share a dominant direction (canonical-form templates, exp 8). Inputs/
outputs of `ridge_decomposition_solve` are f32 for caller convenience; the
solve is f64 internally.

Bench: `cargo bench -p larql-compute --bench linalg`
Demo:  `cargo run --release -p larql-compute --example demo_ridge_solve`

> The MEMIT-flavoured wrapper (`memit_solve` returning `MemitSolveResult`
> with per-fact reconstruction quality) lives in `larql-vindex` next to
> `MemitStore`. The production weight-edit pipeline with covariance
> whitening is in `larql-inference/forward/memit.rs`.

## Architecture

```
src/
  lib.rs              Re-exports + `prelude` module
  pipeline.rs         QuantFormat, QuantWeight, NormType, FfnType, Activation, FullPipelineLayer

  backend/            (folder, one file per concern)
    mod.rs            Umbrella `ComputeBackend` (name/device_info/supports)
    matmul.rs         `MatMul` — f32 / f16 matmul + gemv
    quant_matvec.rs   `QuantMatVec` — unified `quant_matvec(format, …)` + per-format helpers
    decode.rs         `DecodeBackend` — KV-cached decode + prefill + MoE hook
    capability.rs     `Capability` enum — what a backend accelerates
    helpers.rs        `dot_proj_gpu` / `matmul_gpu` (free functions)

  cpu/
    mod.rs            CpuBackend
    ops/              f32_matmul, q4_matvec, q4_vecmat, q4k_matvec, q6k_matvec,
                      q4_common (quantizers: Q4_0, Q4_K, Q4_KF, Q6_K, GGUF Q4_K),
                      q8_matvec, vector, attention, geglu, linalg

  metal/              (feature-gated: --features metal)
    mod.rs            MetalBackend (~30 pipeline handles + KV cache)
    kernel/           `KernelHandle` + `TiledKernel` trait
      handle.rs       Pipeline + geometry, bundled
      traits.rs       The trait shader files implement to expose constants
    trait_impl/       (one file per sub-trait)
      mod.rs          Umbrella ComputeBackend impl + Capability mapping
      matmul.rs       MatMul impl + f32_gemv / f16_gemv encoders
      quant_matvec.rs QuantMatVec impl
      decode.rs       DecodeBackend impl
    decode/           KV-cached decode (norm→QKV→attend→O→FFN per layer)
      mod.rs          decode_token + decode_token_with_moe_fn
      encode_qkv.rs   Step 1 — input norm + format-aware fused QKV
      encode_ffn.rs   Step 6 — format-aware FFN (Q4_KF / Q4_K / Q4_0)
      moe_combine.rs  Hybrid-MoE outer combine (Gemma 4 26B A4B)
      diag.rs         Per-stage / residual / NaN dump helpers
    prefill.rs        GPU prefill for seq>1
    buffers.rs        GPU buffer cache + read_buffer_f32
    shaders/          Metal kernel sources (one file per shader; each
                      tiled shader has a `Kernel` marker for KernelHandle)
    stages/           Reusable stage encoders (qkv_proj, rope, qk_norm,
                      ffn, residual, layer_scalar, quant_matvec, …)
    ops/              GPU dispatch helpers
      full_pipeline/  `dispatch_full_pipeline` + `LayerBuffers` + dump + kv_copy
      …               kv_cache, q4_matvec, q4_batched, …

  csrc/q4_dot.c       ARM NEON Q4 kernel
```

## Tests

**Coverage (2026-05-16)**: this crate is CPU-side only after the
ADR-019 Metal split — the trait, the CPU backend, pipeline types, and
env-option helpers. **`coverage-policy.json` enforces 95%-total / 90%-
per-file with no debt baselines.**

The Metal backend lives in `larql-compute-metal` and has its own
coverage policy (97.28%-total / 90%-per-file / 0 debt baselines as of
2026-05-16 — see [`crates/larql-compute-metal/coverage-policy.json`](../larql-compute-metal/coverage-policy.json)).
Make targets: `make larql-compute-metal-coverage` (full report +
policy gate) and `make larql-compute-metal-coverage-summary` (fast
summary; reuses the last profile).

```bash
# Fast inner loop: library/unit tests only, no integration-binary crawl.
make larql-compute-test-fast

# Heavy integration sweep: every test binary under crates/larql-compute/tests.
make larql-compute-test-integration

# Cross-platform compile checks used before CI handoff.
cargo check -p larql-compute --all-targets

# Metal backend lives in a sibling crate (post-ADR-019).
make larql-compute-metal-test  # lib tests
make larql-compute-metal-coverage  # coverage + policy gate (macOS only)

# Multi-arch perf regression gate (run on a cool machine first to save baselines)
make bench-cross-arch                       # report current numbers
make bench-cross-arch ARGS=--save-baseline  # save current as baseline
make bench-cross-arch ARGS=--compare        # diff vs saved baseline
```

`cargo test -p larql-compute` remains valid; this crate is CPU-only
post-ADR-019, so the test set is small (`test_correctness.rs`,
`test_q4_x86_correctness.rs`, `test_q4k_parity.rs`,
`test_backend_matmul_quant.rs`). Per-shader kernel parity tests live
in `crates/larql-compute-metal/tests/` and run via
`make larql-compute-metal-test`.

**Cross-model parity coverage**: integration tests in
`crates/larql-inference/tests/test_logits_goldens.rs` cover Gemma 3 4B,
Gemma 4 31B dense, Llama 2 7B, Mistral 7B end-to-end. Per-shader tests
in `crates/larql-compute-metal/tests/` are still mostly Gemma-3-4B-only
— extending them to multiple model families is **D-CROSS-PARITY
Phase 2** on the ROADMAP.

The cross-backend / cross-stage parity layer lives in `larql-inference`:

- `larql-inference/tests/test_cpu_metal_parity.rs` — full prefill,
  CPU vs Metal at every layer, all four production architectures.
- `larql-inference/tests/test_decode_consistency.rs` — Metal decode
  vs CPU prefill at the same effective sequence length.
- `larql-inference/tests/test_decode_stage_bisect.rs` — per-stage L0
  divergence localiser (closed the Gemma 4 31B parity gap; ship log
  2026-04-25).
- `larql-inference/tests/test_logits_goldens.rs` — pinned top-5 +
  top-1 logit per (architecture × backend) on a fixed prompt. Catches
  *correlated* drift (CPU and Metal regressing in the same direction)
  that the parity tests can't detect.

## Examples

Eleven examples in three groups — see [`examples/README.md`](examples/README.md) for a one-line description of each.

```bash
# Demos (teach the API)
cargo run --release --features metal -p larql-compute --example demo_basic
cargo run --release --features metal -p larql-compute --example demo_architecture
cargo run --release --features metal -p larql-compute --example demo_ridge_solve

# Compares (full-pipeline benchmarks — distinct from kernel-level criterion suite)
cargo run --release --features metal -p larql-compute --example compare_decode      # Q4_K decode latency
cargo run --release --features metal -p larql-compute --example compare_formats     # Q4_KF vs Q4_K vs Q8
cargo run --release --features metal -p larql-compute --example compare_generation  # End-to-end tok/s
cargo run --release --features metal -p larql-compute --example compare_pipeline    # Q4_K fused vs Q8 fused
cargo run --release --features metal -p larql-compute --example compare_ollama      # Head-to-head vs Ollama

# Diagnostic
cargo run --release --features metal -p larql-compute --example diag_decode_pipeline
cargo run --release --features metal -p larql-compute --example diag_profile_kernels
cargo run --release --features metal -p larql-compute --example diag_shader_bench
```

The headline tok/s vs Ollama uses the CLI's `bench` subcommand against a real vindex:

```bash
larql bench gemma3-4b-q4k-v2 --backends metal --tokens 50 --ollama gemma3:4b
```

## Benchmarks

Three Criterion benches — see [`benches/README.md`](benches/README.md):

| Bench | Surface |
|---|---|
| `quant_matvec` | Q4_0/Q4_K/Q4_KF/Q6_K × 3 shapes × cpu/metal — the regression-detector |
| `matmul` | f32/f16 matmul + lm-head gemv at three shapes |
| `linalg` | Cholesky + ridge solve |

```bash
make bench-compute   # quant_matvec Criterion bench with Metal
cargo bench -p larql-compute --bench matmul
cargo bench -p larql-compute --bench linalg
```

## Diagnostics: parity bisect

When a forward path drifts (CPU vs Metal, or Metal decode vs a fresh
prefill), the per-stage bisect tool localises the divergence to a
single sub-stage of a single layer. This is the diagnostic that
closed the open Gemma 4 31B parity gap (2026-04-25 ship log) — every
attention-side stage at L0 matched at `cos=1.0`, the first
divergence appeared at `ffn_out_raw` / `down_out`, pointing at the
`q4k_ffn_gate_up` shader.

```bash
# Per-layer end-of-layer diff: CPU prefill vs Metal prefill
cargo run --release --features metal -p larql-inference \
    --example residual_diff -- <vindex> "The capital of France is"

# Per-stage L0 diff: CPU prefill vs Metal KV-cached decode
cargo run --release --features metal -p larql-inference \
    --example stage_bisect -- <vindex> "The capital of France is" 0
```

`stage_bisect` exposes the public `larql_inference::residual_diff::stages`
API; the same calls back the regression suite at
`larql-inference/tests/test_decode_stage_bisect.rs`.

## Documentation

| Doc | Content |
|-----|---------|
| [PERFORMANCE.md](PERFORMANCE.md) | Benchmark data, component profiling, optimization history |
| [ROADMAP.md](ROADMAP.md) | Planned optimizations, performance targets, open tracks (D-METAL-PLE, D-ATTN-MTG, D-PREFILL-MM2) |
| [docs/adr/](docs/adr/) | 18 architectural decision records — recent additions: 015 (iso-vs-batched kernel perf), 016 (defused RMS norm + QKV), 017 (shader retention under model agnosticity), 018 (architecture → shader routing) |
| [docs/shader-inventory.md](docs/shader-inventory.md) | **Per-shader retention rationale + applicability** — surveyed 2026-05-09. Read this before adding/deleting any shader |
| [docs/architecture-shader-map.md](docs/architecture-shader-map.md) | **Per-architecture Metal dispatch map** — bridges `larql-models/architectures/` to which shaders each model family fires |
| [docs/llama-cpp-comparison.md](docs/llama-cpp-comparison.md) | **Kernel-architecture comparison vs llama.cpp** — three concrete gaps (flash attention, simdgroup_matrix matmul, RMS-norm pre-fusion) explain remaining decode + prefill perf delta |
| [docs/shaders.md](docs/shaders.md) | Metal kernels with origin, performance, parameters (may lag the source — see the Shaders table above for the current production set) |
| [docs/quantization-formats.md](docs/quantization-formats.md) | Q4_0, Q4_K, Q4_KF, Q6_K, Q8_0 format specs |
| [docs/decode-pipeline.md](docs/decode-pipeline.md) | Decode data flow, dual-path architecture, KV cache |
| [`bench/baselines/cross-arch/README.md`](../../bench/baselines/cross-arch/README.md) | Cool-machine baseline workflow for `make bench-cross-arch` (post-D-CROSS-PARITY) |

## Design Principles

1. **Trait-based dispatch** — callers use `ComputeBackend` exclusively
2. **One file per kernel family** — ~38 shader files under `src/metal/shaders/`, each containing related kernels
3. **Zero-copy mmap** — `newBufferWithBytesNoCopy` for weight buffers
4. **Safe by default** — `read_buffer_f32` with bounds checking
5. **Feature-gated** — Metal with `--features metal`, CPU always available
6. **Auto-calibration** — benchmarks CPU vs GPU at startup for routing threshold
7. **Dual-path decode** — auto-detects Q4_K vs Q8 weights, uses optimal pipeline
8. **GGUF-compatible** — Q4_K/Q6_K formats match Ollama's quantization

## License

Apache-2.0
