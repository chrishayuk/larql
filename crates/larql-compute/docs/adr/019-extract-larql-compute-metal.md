# ADR-019: Extract `larql-compute-metal` (and future GPU backends) into Sibling Crates

**Status**: Proposed
**Date**: 2026-05-16
**Supersedes**: Partially supersedes ADR-001's "single crate, feature-gated backends" model. Trait split (`MatMul` / `QuantMatVec` / `DecodeBackend` / `ComputeBackend`) is preserved unchanged.

## Context

`larql-compute` currently contains the `ComputeBackend` trait + four backend-shaped concerns in one crate:

1. CPU impl (always compiled) — `cpu/**` and `pipeline.rs` types
2. Metal impl (`#[cfg(feature = "metal", target_os = "macos")]`) — `metal/**`, ~50 files, ~12K LOC
3. The `default_backend()` resolver — branches on `cfg(feature = "metal")`
4. Re-exports of Metal-specific symbols (`MetalBackend`, `MoeScratch`, `MetalBuffer`, `BackendOptions`, `DecodeFlags`, `metal_take_last_split_timings`) so downstream callers don't depend on `::metal` directly.

This worked while Metal was the only GPU backend. Pain points have accumulated:

- **Coverage policy hack**: 42 of ~70 source files are GPU-shader code that needs real model weights + commit/wait cycles to test. Today's `coverage-policy.json` lists every metal file in `exclude_globs` (43 entries). Vulkan/CUDA would add another 80+ exclude entries.
- **Compile-time blast on non-Mac hosts**: `cargo build -p larql-compute` on Linux compiles cfg-gated declarations that immediately discard ~12K LOC. With Vulkan + CUDA piling on, every host would build every other host's dead code.
- **Cfg-spaghetti in `lib.rs`**: `default_backend()` has paired `#[cfg(all(feature = "metal", target_os = "macos"))]` and `#[cfg(not(...))]` arms; `default_backend_with_options` exists in two variants. Adding Vulkan would require a third cfg arm in every such function.
- **CI matrix coupling**: Linux CI runs `cargo check -p larql-compute --features metal` purely to keep the cfg-disabled code compiling. With Metal in a separate crate, Linux just doesn't build it.
- **Trait coverage measurement is asymmetric**: ADR-001 split `ComputeBackend` into four sub-traits so callers can branch on capability. With Metal in-crate, the trait + its sole GPU impl share the same coverage policy file, mixing "trait contract" coverage with "GPU dispatch" coverage that requires hardware.

CPU stays put because:
- It's the always-available fallback. Pulling it into its own crate would force every caller to depend on two crates for the default case.
- It compiles to plain Rust + BLAS — no hardware preconditions, fits the unit-test model. The 90% per-file floor is meetable.
- ADR-001's "callers never know which backend" promise is preserved by keeping the trait + the default impl together.

## Decision

Extract one new crate per GPU backend. Start with Metal:

```
crates/
├── larql-compute/                 ← trait + CPU impl + pipeline types + default_backend()
│   ├── src/backend/               (unchanged: trait + sub-traits)
│   ├── src/cpu/                   (unchanged)
│   ├── src/pipeline.rs            (unchanged: Activation, QuantFormat, FullPipelineLayer, MoeLayerWeights, …)
│   ├── src/options.rs             (unchanged: env helpers)
│   └── src/lib.rs                 (slimmed: no more cfg(metal), no MetalBackend re-exports)
└── larql-compute-metal/           ← NEW
    ├── Cargo.toml                 (depends on larql-compute, metal, objc, blas-src/accelerate)
    └── src/
        ├── lib.rs                 (re-exports: MetalBackend, MoeScratch, BackendOptions, DecodeFlags, MetalBuffer)
        ├── backend.rs             ← was crates/larql-compute/src/metal/mod.rs
        ├── attention_kernels.rs
        ├── buffers.rs
        ├── calibrate.rs
        ├── decode/                ← entire subtree
        ├── decode_hybrid.rs
        ├── diag/
        ├── direct_ops.rs
        ├── f32_ops.rs
        ├── ffn_kernels.rs
        ├── flags.rs
        ├── kernel/
        ├── moe_dispatch.rs
        ├── norm_kernels.rs
        ├── ops/
        ├── pipeline.rs
        ├── quant_kernels.rs
        ├── shaders/
        ├── stages/
        └── trait_impl/            ← impls ComputeBackend (from larql-compute) for MetalBackend
```

Future `larql-compute-vulkan`, `larql-compute-cuda` follow the same shape. No further changes to `larql-compute` once the pattern is in place.

### Trait + type ownership

| Type / fn                              | Lives in                       | Notes                                                                 |
|----------------------------------------|--------------------------------|-----------------------------------------------------------------------|
| `ComputeBackend`, `MatMul`, `QuantMatVec`, `DecodeBackend`, `Capability` | `larql-compute`                | Unchanged. The contract every backend implements.                     |
| `CpuBackend`                            | `larql-compute`                | Unchanged.                                                            |
| `cpu_backend()`                         | `larql-compute`                | Unchanged. Always returns CPU.                                        |
| `default_backend()`                     | `larql-compute`                | **Returns CPU only.** No cfg branches. Callers who want GPU call the GPU crate's constructor explicitly. |
| `MoeLayerWeights`, `FullPipelineLayer`, `Activation`, `QuantFormat`, etc. | `larql-compute`                | Pipeline types stay — they're the trait's argument vocabulary.        |
| `Q8KActivation`, `quantize_x_to_q8k`    | `larql-compute`                | CPU primitive used cross-crate (server's expert routes).              |
| `MetalBackend`                          | `larql-compute-metal`          | Implements `larql_compute::ComputeBackend` for itself.                |
| `MoeScratch`, `BackendOptions`, `DecodeFlags`, `MetalBuffer` (Buffer re-export) | `larql-compute-metal`          | Today re-exported through `larql-compute`; move to the Metal crate.   |
| `metal_take_last_split_timings`         | `larql-compute-metal`          | Renamed to plain `take_last_split_timings` since the crate prefix already disambiguates. |
| `metal_backend()` / `metal_backend_with_options()` | `larql-compute-metal`          | New constructors — return `Option<MetalBackend>`. Replace `MetalBackend::new()` at call sites if desired. |

### `default_backend()` semantics change

**Before** (cfg-branching, hidden Metal preference):
```rust
let backend = larql_compute::default_backend();  // returns Metal if available, else CPU
```

**After** (explicit, caller composes preference):
```rust
// Mac binary that wants Metal:
let backend: Box<dyn ComputeBackend> =
    larql_compute_metal::metal_backend()
        .map(|m| Box::new(m) as Box<dyn ComputeBackend>)
        .unwrap_or_else(|| larql_compute::default_backend());

// Linux binary or CPU-only test:
let backend = larql_compute::default_backend();  // always CPU, no cfg involved
```

This is more verbose at call sites, but:
- Backend preference becomes explicit code, not a compile-time cfg. A developer reading `bench/local_runtime.rs` sees the fallback chain instead of inferring it from `default_backend()`'s contract.
- A new backend crate (Vulkan, CUDA) doesn't have to teach `larql-compute` about itself. Callers just add another `.or_else(|| larql_compute_vulkan::vulkan_backend().map(...))`.
- Removes the existing `default_backend_from_optional_metal` test seam (ADR introduced in this session) — the seam was needed only because the cfg-gating made the fallback path untestable. With the seam gone, no Metal-aware code remains in `larql-compute`.

For the common case, ship a tiny convenience crate:

```toml
# larql-compute-runtime/Cargo.toml
[features]
metal = ["dep:larql-compute-metal"]
```
```rust
// larql-compute-runtime/src/lib.rs
pub fn best_available() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "metal")]
    if let Some(m) = larql_compute_metal::metal_backend() {
        return Box::new(m);
    }
    larql_compute::default_backend()
}
```

Callers who want one-line setup use this; callers who want explicit control use the backend crates directly. The cfg-spaghetti is concentrated in one ~10-line file instead of spread across `larql-compute::lib.rs`.

### Downstream impact

Audit of `larql_compute::` imports across the workspace (today):

- **CPU-only imports** (no churn): `larql-kv`, `larql-router`, `larql-router-protocol`, `larql-boundary`, `larql-lql`, `larql-models`. They use `ComputeBackend`, `CpuBackend`, `cpu_backend()`, pipeline types. All stay in `larql-compute`.

- **Metal-specific imports** (need crate flip): 26 files across 4 crates:
  - `larql-cli` — 4 files (bench runners, diagnostics, ov_rd dev tools)
  - `larql-inference` — 14 files (gpu/* decode loops, hybrid, tests, examples)
  - `larql-server` — 4 files (state.rs cache, walk_ffn.rs warmup, expert/metal.rs, expert/warmup.rs)
  - `larql-vindex` — 1 file (cpu_vs_gpu bench)

  Each replaces `use larql_compute::MetalBackend;` with `use larql_compute_metal::MetalBackend;`. The `larql_compute::metal::MetalBackend::new()` path (currently used in cli) flattens to `larql_compute_metal::MetalBackend::new()`. Mechanical change.

- **`larql_compute::metal::*` deep imports** (currently `pub mod metal`): the cli reaches into `larql_compute::metal::MetalBackend` directly. After the split, this is just `larql_compute_metal::MetalBackend` — the deep `metal::` prefix disappears, which is a small ergonomic win.

### Coverage policy

**Today**:
- `larql-compute/coverage-policy.json` has `exclude_globs` covering 43 metal files.
- TOTAL coverage on `--features metal` is dragged down to ~67% by GPU shader files that have no realistic unit-test path.
- Per-file 90% floor passes for the included (CPU) tree at ~97% today.

**After split**:
- `larql-compute/coverage-policy.json` drops the `exclude_globs` entirely. Pure CPU + trait crate; per-file 90% floor applies to everything. Expected TOTAL: ~97% (already there).
- `larql-compute-metal/coverage-policy.json` gets its own floor set realistically (e.g., 70% with per-file baselines for the kernel-dispatch files that genuinely need integration tests). The "this file needs a GPU to cover" property is local to the Metal crate, not contaminating the CPU policy.
- Future `larql-compute-vulkan/coverage-policy.json` doesn't have to negotiate with Metal's floor.

### CI implications

`larql-compute.yml` becomes Linux-friendly without the `cargo check -p larql-compute --features metal` step (which is currently checking that cfg-gated code still compiles on Linux even though it can't run). The `--features metal` check moves to a new `larql-compute-metal.yml` that only runs on the macOS runner.

The macOS runner already exists (the project's perf benches and parity tests run on it). One new workflow file, one new entry in the matrix.

## Migration order

This is a multi-PR refactor. Each step is independently mergeable and leaves the workspace green.

1. **PR 1 — Create empty `larql-compute-metal` crate.** Add `crates/larql-compute-metal/Cargo.toml` + empty `src/lib.rs`. Adds the workspace member, depends on `larql-compute`. No code moves. Validates the workspace layout works.
2. **PR 2 — Move `crates/larql-compute/src/metal/**` into `crates/larql-compute-metal/src/`.** Files move bit-for-bit; intra-file imports rewrite `crate::` → `crate::` (intra-crate stays the same) and `crate::cpu::` / `crate::backend::` / `crate::pipeline::` / `crate::options::` → `larql_compute::cpu::` / etc. Add the four submods (`stages`, `ops`, `shaders`, `decode`, `diag`, `trait_impl`) to `larql-compute-metal/src/lib.rs`. Delete `metal/` from `larql-compute`. Re-export the same symbols from `larql-compute-metal/src/lib.rs` that used to be re-exported from `larql-compute/src/lib.rs`.
3. **PR 3 — Flip downstream imports.** 26 files. `larql_compute::MetalBackend` → `larql_compute_metal::MetalBackend`. Mechanical. Each downstream crate's `Cargo.toml` adds `larql-compute-metal` as a (likely optional, feature-gated) dep.
4. **PR 4 — Slim `default_backend()`.** Remove the `#[cfg(feature = "metal")]` branch from `larql-compute/src/lib.rs`. `default_backend()` now always returns CPU. Drop the `default_backend_from_optional_metal` test seam. Add the `metal_backend()` helper to `larql-compute-metal/src/lib.rs`. Update the callers that used `default_backend()` to expect-Metal behaviour — they pick a fallback chain explicitly. (Optional: introduce `larql-compute-runtime` here.)
5. **PR 5 — Update coverage policies.** `larql-compute/coverage-policy.json` drops `exclude_globs`. New `larql-compute-metal/coverage-policy.json` gets realistic baselines.
6. **PR 6 — Update CI.** `larql-compute.yml` drops `--features metal` jobs. New `larql-compute-metal.yml` runs on the macOS runner with the metal feature.

Each PR is small enough for incremental review and bisect.

## Consequences

- **Good — Coverage policy honesty**: CPU policy stops carrying 43 exclude entries it can't realistically meet. Each backend owns its own floor + per-file debt.
- **Good — Linux compile time**: `larql-compute` shrinks from ~17K LOC to ~5K. Linux/CUDA host builds skip Metal entirely instead of cfg-gating its declarations.
- **Good — Adding Vulkan / CUDA is additive**: New crate, no edits to `larql-compute`. Pattern is established once.
- **Good — `lib.rs` cfg-spaghetti gone**: `default_backend()` becomes a 3-line function. The test seam introduced this session (`default_backend_from_optional_metal`) goes away because the underlying problem (untestable cfg branch) is gone.
- **Trade-off — Caller verbosity for backend selection**: `default_backend()` no longer auto-picks Metal. Callers either compose a fallback chain explicitly, or use `larql-compute-runtime`. Net wash; the explicit form is more debuggable.
- **Trade-off — Refactor churn**: 26 files in downstream crates flip imports. Mechanical but tedious. Each migration PR (esp. PR 3) needs careful review to ensure no Metal-coupled code lingers in `larql-compute`.
- **Trade-off — Cargo dep graph deepens**: `larql-server` gains a direct dep on `larql-compute-metal` (currently inherits Metal through `larql-compute`'s feature flag). That's the point — server's Metal coupling becomes visible at the crate-graph level instead of hidden behind a feature flag.

## Non-decisions (out of scope for this ADR)

- **Whether to introduce `larql-compute-runtime`**: Recommended but not required. PRs 1-5 work without it; callers can compose backends inline. Add later if the inline pattern shows up in 5+ binaries.
- **Whether to split `cpu/ops/moe/*` into a `larql-compute-moe` crate**: No. MoE is CPU compute logic that benefits from sharing `ExpertScratch` + `cpu_moe_forward` across server (expert routes) and inference (dispatch path). Splitting it adds a coordinate problem (router policy types live in `pipeline.rs`) for no obvious win.
- **Renaming `MetalBackend::new()` to `metal_backend()`**: The `MetalBackend::new()` constructor stays — it's the canonical way to build one. The free function `metal_backend()` is sugar that returns `Option<MetalBackend>` for the common "try and fall back" path. Both compile.
