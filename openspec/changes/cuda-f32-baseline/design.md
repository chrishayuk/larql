## Context

The parent change `cuda-and-rotorquant-kv` landed:

- the `cuda` Cargo feature with `cudarc 0.19` pinned (driver + cuBLAS
  + dynamic-loading), so the same binary runs on hosts without a CUDA
  toolkit and falls back to CPU,
- `larql_compute::cuda::CudaBackend` as a stub returning
  `unimplemented!()` from every kernel dispatch path,
- the `Capability::Cuda` bit so `default_backend()` can pick the
  CUDA backend when available.

The cuBLAS f32 GEMM/GEMV is the natural "first real kernel" because:

1. It's the most-exercised path — every linear projection in the
   model touches it.
2. It validates the entire host↔device plumbing (alloc, copy, kernel
   launch, copy back, free) without the additional complexity of
   custom kernels.
3. The CPU implementation produces a strong reference oracle, making
   parity tests cheap and high-signal.

cudarc gives us the cuBLAS handle directly. The work is a thin
wrapper: ndarray slice → device buffer → cuBLAS GEMM → device buffer
→ ndarray result. The trickiest part is row-major / column-major
reconciliation: cuBLAS expects column-major, ndarray hands us
row-major slices; the standard fix is the transposed-product identity
plus the `Op::T` flag.

## Goals / Non-Goals

**Goals:**

- `CudaBackend::matmul`, `matmul_transb`, and `f32_gemv` all return
  results that match CPU within 5e-4 absolute on Gemma 4B-shaped
  inputs (`hidden=2560`, `intermediate=10240`, `vocab=256000`).
- `CudaBackend::new()` performs a real CUDA driver probe and returns
  a typed error when the driver is missing.
- The implementation is simple enough that a future contributor can
  read it end-to-end in 15 minutes.
- Parity tests are gated so CPU-only CI keeps passing.

**Non-Goals:**

- Performance tuning. We're chasing correctness; tuning is later.
- Async streams. Single default stream, single host-blocking copy.
- f16 / bf16 matmul. Phase-2 lands these once Q4 lands first.
- Multi-GPU. Single GPU at index 0.

## Decisions

### D1 — Reconcile row-major / column-major via the transposed-GEMM identity

For row-major inputs `A` (M×K) and `B` (K×N) computing `C = A·B`:

- Treat the row-major buffers as column-major buffers of the
  transposed shapes: `A` becomes a K×M column-major matrix; `B`
  becomes a N×K column-major matrix.
- Compute `C^T = B^T · A^T` in column-major. cuBLAS GEMM with
  `Op::T` on both inputs gives `C^T` directly.
- Read `C^T` back as a row-major M×N — which is exactly `C`.

This is the standard cuBLAS-from-row-major trick; codified in a
helper `gemm_f32(driver, op_a, op_b, a, b, m, n, k) -> Vec<f32>`.

`matmul_transb` (where the second matrix arrives already
transposed in row-major) collapses one transpose: pass `Op::N` for
the second operand.

### D2 — `Driver` owns the context and the cuBLAS handle, lazy-init

```rust
pub struct Driver {
    ctx: Arc<CudaContext>,
    blas: CudaBlas,
}
```

Creating either is non-trivial (driver init ~30 ms, cuBLAS handle
~10 ms). We do both in `CudaBackend::new()` and stash the driver in
the backend. Subsequent kernel launches reuse them. `Drop` for
`Driver` frees the cuBLAS handle and the context handle in order.

Thread safety: cudarc 0.19's `CudaContext` is `Send + Sync` by way
of an internal mutex; we don't add our own.

### D3 — Host↔device copies are blocking, single default stream

cudarc's high-level `htod_sync_copy` / `dtoh_sync_copy` block the
host until done. For correctness work this is fine and dramatically
simplifies the surface. Async streams + pinned memory are a
performance-tuning concern for a later sub-change.

### D4 — Device buffer helpers are local to `cuda::matmul`, not exported

Two helpers (`Driver::device_buf_from(slice)` and `Driver::to_host(buf)`)
live in `cuda/driver.rs`. They are `pub(crate)` — the moment a
caller outside `cuda::` touches device buffers we want to reconsider
the boundary.

### D5 — Cache directory is XDG-compliant

PTX modules and cuBLAS workspace caches go under
`$XDG_CACHE_HOME/larql/cudarc/<arch>/<hash>/` (default
`~/.cache/larql/cudarc/`). The directory is created lazily on first
write and respects the `XDG_CACHE_HOME` environment variable. This
sub-change creates the directory and writes a `.version` marker; the
first user is `cuda-q4-matvec`.

### D6 — Driver probe failure → typed error, no panic

`CudaContext::new(0)` returns a `cudarc::DriverError` on init failure
(missing driver, no devices, version mismatch, etc.). We map it to
the existing `CudaInitError` enum:

| cudarc condition | `CudaInitError` |
|---|---|
| `DriverError::CUDA_ERROR_NOT_INITIALIZED` etc. | `DriverMissing(_)` |
| `DriverError::CUDA_ERROR_NO_DEVICE` | `NoDevices` |
| `DriverError::CUDA_ERROR_INVALID_DEVICE` (no device 0) | `NoDevices` |
| `DriverError::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` etc. | `ToolkitMismatch { found, need }` |
| anything else | `DriverMissing(format!("{e}"))` (best-effort) |

No `unwrap` / `panic` on cudarc results. `default_backend()` falls
back to CPU on `Err`.

### D7 — Parity tests are env-gated, not feature-gated

`#[cfg(feature = "cuda")]` is too aggressive — the test compiles
fine on a non-GPU host, it just can't run. Better: compile when
`feature = "cuda"`, but skip with `if std::env::var("LARQL_CUDA_AVAILABLE").is_err() { return; }`
at the top of each test fn. CI sets the env var on GPU runners.
Local devs see the test get reported as "ok" (no-op) by default.

## Risks / Trade-offs

- **Risk: blocking copies hide perf cliffs.** A real workload pipelines
  copies behind kernel launches, sometimes via pinned host memory.
  → Mitigation: tests are correctness-only; performance work is its
  own sub-change. Document the synchronous behaviour explicitly in
  the `Driver` doc-comment so future contributors don't read it as a
  recommendation.
- **Risk: cuBLAS API surface drift across cudarc versions.** cudarc
  0.19 → 0.20 has rearranged enum variants (`Op::T` → `cublasOperation_t::CUBLAS_OP_T`).
  → Mitigation: we wrap all cuBLAS calls in our own `cuda::matmul`;
  upgrading cudarc requires touching only that module.
- **Risk: row/column-major confusion silently produces wrong outputs.**
  Easy to get the transpose flags wrong; cuBLAS doesn't error, you
  just get garbage. → Mitigation: the parity tests trip on max
  absolute element diff > 5e-4, which catches any wrong-transpose
  bug at the first run.
- **Risk: NVIDIA driver mismatch.** Host has a CUDA 13.1 driver;
  cudarc is built against the 13.1.0 ABI. A driver/runtime mismatch
  manifests as cudaErrorInsufficientDriver at first dispatch.
  → Mitigation: `CudaInitError::ToolkitMismatch` carries the found
  driver version in the error so users see what to update.

## Migration Plan

Land. Run `cargo test -p larql-compute --features cuda
LARQL_CUDA_AVAILABLE=1` on a GPU box; expect 4–6 new green tests.
Spot-check a `larql-cli predict` run with `LARQL_BACKEND=cuda` —
confirm logits match a CPU run within tolerance.

Rollback: revert the commit. The backend reverts to the Phase-1
stub. No data path or storage format changes.

## Open Questions

- **Q1: float-precision policy for matmul comparisons.** 5e-4
  absolute is the threshold I propose; we may want relative
  tolerance for very large matrices. Re-anchor after the first run
  and adjust if numbers exceed it benignly.
- **Q2: Should we eagerly call `CudaContext::synchronize()` on every
  matmul boundary?** cudarc's blocking copies imply it; explicit sync
  is a defense-in-depth. Recommendation: yes, at the end of each
  `gemm_f32` call. Cheap, removes a class of "test passes locally,
  fails in CI" bugs.
