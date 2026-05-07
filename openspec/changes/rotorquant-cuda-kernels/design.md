## Context

The CPU reference in `larql-rotorquant::cpu_ref` does:

1. Per row: compute absmax, normalise to [-1, 1].
2. Per block (size 2 for Planar, 4 for Iso): brute-force search
   the rotation table for the rotation that minimises codebook
   error. Apply rotation; quantise to nearest codeword; record
   rotation index.
3. Per block on dequantize: look up codes → codeword values,
   apply inverse rotation, multiply by row absmax.

All steps map cleanly to CUDA: rows are independent, blocks
within a row are independent. The natural decomposition is one
thread per block × `(rotations × block_size)` flops per block —
~50–200 ops per block depending on format. With 32-thread warps
and rows of typical width (head_dim ≥ 128), each row fits in a
single warp and the kernel is bandwidth-bound on host→device
transfer rather than compute-bound.

## Goals / Non-Goals

**Goals:**
- Four PTX kernels: `planar_quantize`, `planar_dequantize`,
  `iso_quantize`, `iso_dequantize`.
- Each kernel cosine-matches the CPU reference within 1e-3 on
  synthetic data.
- Compilation cached on disk (~50 ms first call, ~1 ms warm).
- ≥ 10× wall-clock speedup vs CPU reference at Gemma 4B head
  dim = 320, batch = 32.
- Capability bit flips on the CUDA backend.

**Non-Goals:**
- f16 / bf16 coded variants (separate sub-change).
- Multi-stream pipelining. Single default stream.
- Per-batch dispatch tuning. Defaults that work for 4090 + Ada
  archs.
- Vendoring upstream `feature/planarquant-kv-cache` `.cu` sources.
  We keep our from-scratch reference; the PTX implementations
  match our CPU oracle.

## Decisions

### D1 — Inline PTX via NVRTC, not nvcc-compiled cubin

cudarc NVRTC was already proven by the softmax kernel
(`cuda-fused-attention`). We reuse the same plumbing:
`cudarc::nvrtc::compile_ptx` then `ctx.load_module(ptx)`. No
build.rs nvcc invocation, no cross-compilation concerns.

### D2 — One block per row, each thread handles one of the row's blocks

Standard pattern for row-decorrelated work. Block dim = 256
threads (covers head_dim up to 1024 with one block-of-256
threads handling 4 head_dim positions each). Grid dim = num_rows.

For long rows the strided loop pattern (each thread sees
`stride = blockDim.x` blocks) handles up to head_dim ≈ 4096.

### D3 — Codebook + rotation tables in `__constant__` memory

Both tables are < 1 KB total per format (8-quaternion table for
Iso = 16 floats × 16 = 1024 bytes; 8-Givens table for Planar = 4
floats × 8 = 128 bytes; codebook = 16 × 4 bytes = 64 bytes).
Constant memory is cached and broadcast-friendly across warps;
ideal for these read-mostly tables.

### D4 — Brute-force rotation search per block

Same as CPU reference. 8 rotations × 4 multiply + 4 quantize ×
4 components ≈ 200 fast ops per block on Iso. With one block
per warp lane this is 6 µs per million blocks — bound by memory
bandwidth, not compute.

### D5 — Capability flip is conditional on PTX compile success

`CudaBackend::supports(Capability::KvCompressionRotorQuant)`
returns `true` only after the kernels have successfully
compiled (or been loaded from cache). A failed PTX compile
leaves the capability `false` and the dispatch path falls back to
CPU. This way a host with cudarc but a buggy PTX environment
still works.

## Risks / Trade-offs

- **Risk: numerical drift between CPU and PTX.** Reductions in
  different orders, float-vs-double in intermediate
  computations, etc. → Mitigation: 1e-3 absolute tolerance vs
  CPU reference (matches the existing CPU round-trip threshold);
  the codebook values are exactly representable, so error
  sources are limited to the rotation matmul.
- **Risk: NVRTC compile cost on first call.** ~150 ms.
  → Mitigation: compile at backend init, cache to disk.
- **Risk: register pressure on Iso (16-float rotation matrix per
  block).** → Mitigation: block-shared memory holds the rotation
  table; threads read on demand. No per-thread copy.
- **Risk: VRAM double-buffering during quantize-then-write.**
  → Mitigation: in-place quantize writes the codes array in
  parallel with norms / rotation_indices arrays in different
  buffers; no double-allocation.

## Migration Plan

Land. The crate's public API (`quantize_k`, `dequantize_k`, etc.)
unchanged; consumers don't see whether CPU or CUDA executed.

Rollback: revert. CPU reference stands.

## Open Questions

- **Q1: Should the kernel cache live alongside the softmax cache
  under `larql/cudarc/` or get its own subdir?** Same dir.
  Distinct filenames per kernel name + arch.
- **Q2: f16 codes?** Future change. Today norms are f32; halving
  saves 4 bytes per row.
- **Q3: Custom block dim for very small head_dim?** head_dim < 64
  doesn't fill 256 threads. Either block dim = 64 for those, or
  pack multiple small rows per block. **Recommendation**:
  defer; start with 256, measure.
