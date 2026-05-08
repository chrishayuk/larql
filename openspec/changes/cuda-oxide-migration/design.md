# cuda-oxide pilot — design

## What cuda-oxide actually is

[NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) (README
rev 2026-05-08) is **a custom rustc backend that compiles GPU
kernels in pure Rust**. The pipeline:

```
Rust source
   │  cargo oxide build
   ▼
Rust MIR
   │  rustc-codegen-cuda
   ▼
Pliron IR (MLIR-like)
   │  pliron pass pipeline
   ▼
LLVM IR
   │  llc-21 --march=nvptx
   ▼
PTX  ──►  cudarc / cuda-host driver  ──►  GPU
```

Single-source: device kernels live in the same `.rs` file as
host code, marked `#[kernel]`. The cuda-oxide workspace ships a
host runtime (`cuda-core`, `cuda-async`, `cuda-host`,
`cuda-bindings`) that's a candidate drop-in for the cudarc
driver API; we don't migrate to it in Phase 1.

## What this proposal is NOT

- Not a Rust-CUDA debate. The existing `cudarc` path stays for
  cuBLAS. cuda-oxide isn't trying to replace BLAS.
- Not a kernel-rewrite for performance. The pilot kernel
  (RotorQuant Iso3 dequantize) is the missing device path needed
  by `cuda-decode-backend`; both cuda-oxide and the eventual
  cudarc-NVRTC variant are greenfield. We pick whichever delivers
  the better author experience at acceptable performance.
- Not a Metal migration. cuda-oxide is Linux+CUDA only today.
  Metal kernels stay in the existing Objective-C++ pipeline.

## Pilot kernel: RotorQuant Iso3 dequantize

The smallest meaningful kernel is the one currently blocking
RotorQuant KV-cache inference on CUDA. The vendored upstream CUDA
source exposes FP16-to-packed quantize/copy helpers, but no CUDA
dequantize helper. From
`crates/larql-rotorquant/src/cpu_ref.rs::dequantize` (Iso3 path):

1. For each row, read the per-row absmax scale from `norms`.
2. For each block of 4 coordinates (Iso = 4D quaternion):
   - Unpack 4 LSB-first 3-bit codes from the packed `codes`
     buffer.
   - Look up the 8-codeword Lloyd-Max values.
   - Read the block's `rotation_idx` and apply the inverse
     4x4 Iso rotation.
   - Multiply by the row scale and write row-major `f32` output.

This is ~150 lines of CPU Rust today. The cuda-oxide port:
- One `#[kernel]` function: `iso3_dequantize_block(codes,
  norms, rotation_indices, out, n_rows, head_dim)`.
- Host wrapper that launches it across `(n_rows × n_blocks_per_row)`
  threads, reading packed `DeviceBuffer`s and writing one
  `DeviceBuffer<f32>`.
- Parity check against `cpu_ref::dequantize` on CPU-quantized
  input — max-element diff ≤ 1e-3 and cosine ≥ 0.99 against the
  original input row.

```text
┌──────────────────────────────────┐
│  larql_rotorquant::cuda_oxide    │
│  ────────────────────────────    │
│   #[kernel]                      │
│   fn iso3_dequantize_block(...)  │
│   ────────────────────────────   │
│   pub fn dequantize_iso3_oxide(  │
│     ctx: &CudaContext,           │
│     qkv: &QuantizedKv,           │
│   ) -> Vec<f32>                  │
└──────────────────────────────────┘
              │
              │ same f32 reconstruction as the CPU reference
              ▼
┌──────────────────────────────────┐
│  larql_rotorquant::cpu_ref       │
│  ────────────────────────────    │
│   pub fn dequantize(qkv) -> Vec  │
└──────────────────────────────────┘
              │
              │ cosine ≥ 0.99 vs the original input
              ▼
              parity test passes
```

## Cargo feature interaction

Today:

```toml
# crates/larql-rotorquant/Cargo.toml
[features]
default = []
cuda = ["dep:cudarc"]      # NVRTC + cuBLAS via cudarc
```

After Phase 1:

```toml
[features]
default = []
cuda = ["dep:cudarc"]                                 # unchanged
cuda-oxide = ["dep:cuda-core", "dep:cuda-host", "dep:cuda-device"] # new, pilot
```

The two features are **mutually exclusive at compile time**.
A workspace-level `compile_error!` macro enforces it, and the
GPU Dockerfile picks one or the other.

The host code paths are independent: each format has a
`Format::backend(BackendKind)` accessor that returns the
selected implementation. `BackendKind::Cudarc` and
`BackendKind::CudaOxide` flow through different module trees.

## Toolchain split

cuda-oxide pins **`nightly-2026-04-03`** (per the upstream
README's `rust-toolchain.toml` example). The LARQL workspace
pins stable. We don't change the workspace default. The pilot
also targets CUDA Toolkit **13.1**, because the pinned
cuda-oxide commit calls driver symbols exposed by CUDA 13 headers
(`cuEventElapsedTime_v2`) that are not generated from the dev
box's CUDA 12.x headers.

The pilot uses an **adjacent toolchain entry** in the GPU
Dockerfile only:

```dockerfile
RUN rustup toolchain install nightly-2026-04-03 && \
    rustup component add rust-src rustc-dev --toolchain nightly-2026-04-03 && \
    cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git \
        --rev <PINNED_COMMIT> cargo-oxide
```

Local contributors building the pilot run:

```bash
make cuda-oxide-pilot   # uses +nightly-2026-04-03 internally
```

The default `make ci` target stays on stable and ignores the
cuda-oxide feature entirely.

## Build dependency: CUDA 13.1 + LLVM 21

cuda-oxide emits PTX that requires `llc` from LLVM 21 (the
README explicitly calls out TMA / tcgen05 / WGMMA intrinsics
that LLVM 20 can't handle). On Ubuntu 24.04 (the GPU image
base), LLVM 18 ships by default; LLVM 21 comes from
`apt.llvm.org`.

The GPU container base is `nvidia/cuda:13.1.0-devel-ubuntu24.04`.
Local CUDA 12.x installations are useful for the existing cudarc
path but do not satisfy the pinned cuda-oxide pilot.

```dockerfile
RUN curl -sSf https://apt.llvm.org/llvm.sh | bash -s -- 21 && \
    apt-get install -y libclang-common-21-dev clang-21
```

The image grows by ~400 MB. Acceptable for the GPU container
(currently ~3 GB).

## Performance expectations

Given the kernel is greenfield, we have no perf baseline. The
table below is the **acceptance bar** we'd hold cuda-oxide to
in Phase 2 — measured on RTX 4090, Gemma 4B head shape
(hidden=2560, head_dim=320, n_kv_heads=4):

| Metric | Cudarc-NVRTC reference | cuda-oxide target |
|---|---|---|
| Iso3 dequantize on 8 × 320 block | (TBD; build it first) | ≤ 1.25× cudarc |
| Cold-start kernel-load | 200 ms (cudarc-cached) | ≤ 100 ms (PTX is cargo-built) |
| PTX size | (TBD) | ≤ 1.5× cudarc |
| Round-trip cosine vs CPU | ≥ 0.99 | ≥ 0.99 (same bound) |

If cuda-oxide hits "≤ 1.25× cudarc on throughput" we ship Phase
3. If it's > 1.5× slower, we abort and write up findings.

## Tests

- Unit: cuda-oxide-built `iso3_dequantize_block` on
  synthetic CPU-quantized 64×320 input → max-element diff ≤ 1e-3
  against CPU dequantize and cosine ≥ 0.99 against the original
  row (gated on `LARQL_CUDA_AVAILABLE=1` + `--features cuda-oxide`).
- Parity: cuda-oxide vs cpu_ref vs (when shipped) cudarc-NVRTC
  dequantize on the same quantized input — all three within
  1e-3 max-element diff.
- Build doctor: `make cuda-oxide-doctor` runs `cargo oxide
  doctor` on the dev box and reports missing prerequisites
  (LLVM 21, clang-21, nightly toolchain, CUDA Toolkit 13.1).

## Documentation

- `docs/cuda-rotorquant-status.md` — add a "cuda-oxide pilot"
  subsection that links to this change.
- `deploy/docker/README.md` — feature matrix expanded:

  | Feature flag | Backend host runtime | Custom kernels |
  |---|---|---|
  | `cuda` (today) | cudarc 0.19 | NVRTC PTX strings |
  | `cuda-oxide` (pilot) | cudarc 0.19 (cuBLAS) + cuda-core (custom) | rustc-codegen-cuda |

  Both features keep cudarc for cuBLAS — the difference is
  custom-kernel authoring.
- `crates/larql-rotorquant/UPSTREAM.md` (when this proposal
  ships its Phase 1 commit) — record the cuda-oxide commit hash
  pinned, the date imported, and any local patches.

## Decision criteria — written down so we don't move the goalposts

This pilot is a **time-boxed experiment**, not a commitment.
Phase 2 evaluation must answer:

1. **Author experience**: did writing the kernel in Rust feel
   meaningfully better than CUDA C in NVRTC strings?
2. **Build experience**: did contributors successfully build
   the pilot on first try (with `cargo oxide doctor`)?
3. **Performance**: does the kernel hit the acceptance table
   above on RTX 4090?
4. **Stability**: did the pinned cuda-oxide commit produce zero
   hard failures over 2 weeks of CI?

If any answer is "no", the rollback is `git revert` of the
Phase 1 PR. The `cuda-oxide` feature flag and module stay
gone; the cudarc path that ships under `cuda` is unchanged.
