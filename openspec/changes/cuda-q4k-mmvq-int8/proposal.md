## Why

After `cuda-decode-device-resident` (Phases 1+3+2) and the small
tightening pass (`cuda-decode-tightening`), CUDA decode on the local
Q4_K Gemma 3 4B vindex sits at:

```
prefill 181.7 ms  /  decode 19.49 ms/token  /  51.3 tok/s
GPU fwd 17.491 ms  /  LM-head 1.964 ms
```

The `LARQL_CUDA_DECODE_PROFILE=1` breakdown shows the residual
cost is now ~82% raw projection compute:

```
proj_wo         6.54 ms (31%)   ← cuBLAS GEMV (M=1)
proj_gate_up    4.82 ms (23%)   ← Q4_K direct matvec, f32 path
proj_down       4.20 ms (20%)   ← Q6_K cached f32 GEMV
proj_qkv        1.70 ms ( 8%)   ← Q4_K direct matvec, f32 path
norm_cpu        1.37 ms ( 7%)
residual_cpu    1.25 ms ( 6%)
htod            0.65 ms ( 3%)
attn_call       0.54 ms ( 3%)
dtoh            0.01 ms ( 0%)
```

Side-by-side vs llama-cpp-turboquant on the same hardware (RTX
4090, CUDA 12.5, identical Gemma 3 4B Q4_K_M GGUF, llama-bench
`-p 0 -n 20 -ngl 99 -t 1`):

| | LARQL today | llama-cpp-turboquant | Gap |
|---|---:|---:|---:|
| decode tok/s | 51.3 | 227.5 | 4.43× |
| decode ms/tok | 19.49 | 4.40 | 4.43× |

**The dominant cost difference is the Q4_K matvec kernel itself.**
LARQL's current `q4k_direct.rs` kernel uses single-precision
multiply-add throughout. llama.cpp routes Q4_K through a
`vec_dot_q4_K_q8_1` kernel that:

1. Quantizes the input vector to Q8_1 (32-element blocks, fp16
   scale + fp16 sum-of-squares) once per layer-input.
2. Computes the matvec via `__dp4a` — a single-instruction
   four-way INT8 dot product that returns an INT32 accumulator.
   On sm_89 `__dp4a` runs at **4× the rate of fp32 fused
   multiply-add**, and the dedicated DP4A pipeline is largely
   independent of the f32 SIMT pipeline.
3. Folds Q4_K's `(scale × value − dmin × min)` decode + the
   per-block fp16 input scale into the final fp32 accumulator,
   so numerical error stays in the same band as the existing
   f32 path.

Porting this to LARQL is the single biggest remaining lever for
decode throughput. The kernel pattern is well-established (the
upstream is the *reference* Q4_K implementation), the primitives
(`__dp4a`) work via NVRTC inline calls, and the parity contract
is the same one we already use elsewhere (≤ 1e-3 max-element
vs the host fallback).

## What Changes

### Phase 1 — Q8_1 quantization kernel for input vectors

- ADD `cuda::elem::quantize_q8_1_device(x_dev, n) ->
  CudaSlice<u8>` (or a typed wrapper). One block per 32-element
  group, 32 threads per block. Each block computes a fp32
  amax/2 → fp16 scale, divides each element by the scale to
  s8, and emits `[n_blocks × 32 bytes (s8)] + [n_blocks × 4
  bytes (fp16 scale, fp16 sum)]`. Memory layout matches
  llama.cpp's `block_q8_1`.
- ADD a small per-backend cache so the Q8_1-quantized form of
  the per-layer input vectors (`h_attn_dev`, `h_ffn_dev`) is
  computed **once per layer** and shared across the q/k/v
  matvecs (and across gate/up). Cuts the quantize cost per
  layer from 5 calls to 2.

### Phase 2 — Q4_K × Q8_1 mmvq kernel

- ADD `cuda::q4k_mmvq::matvec_device(weight_q4k, x_q8_1, rows,
  hidden) -> CudaSlice<f32>`. NVRTC kernel, structure modelled
  on llama.cpp's `mmvq.cu` + `vec_dot_q4_K_q8_1_impl_vmmq`:
  - One row per warp (32 threads).
  - `__dp4a` four-way INT8 dot for the per-256-element
    super-block accumulators.
  - Per-sub-block scale (Q4_K's 6-bit packed scales) and min
    decoded inline; final accumulator scaled by `dm4f.x` and
    `dm4f.y` (fp16 → fp32).
- ADD a runtime dispatch flag on `CudaBackend`:
  `LARQL_CUDA_Q4K_MMVQ=0` forces the existing direct f32 kernel
  (back-out path); default after parity verification is `1`.
- ADD `q4k_matvec_device_mmvq` as the new entry point. The
  existing `q4k_matvec_device` becomes a thin dispatcher.

### Phase 3 — Wire into the decode pipeline

- UPDATE `decode_token_device` to:
  - Quantize `h_attn_dev` to Q8_1 once per layer, share across
    q/k/v projections.
  - Quantize `h_ffn_dev` to Q8_1 once, share across gate/up.
  - Pass the cached Q8_1 form into the new mmvq matvec.
- KEEP the existing f32-input matvec entry point for the wo and
  down projections initially. wo's input is `attn_out_dev`
  (transient, single-use) and down's input is `act_dev` (also
  single-use after Phase 2's GPU silu); they don't benefit from
  the share-across-projections optimization. We can adopt mmvq
  for them in a follow-up if the bench shows it pays off.
- KEEP `LARQL_CUDA_DECODE_HOST_FALLBACK=1` and
  `LARQL_CUDA_DECODE_PROFILE=1` env vars as-is.

### Out of scope

- Q4_K mmq (matrix-matrix-quantized) for prefill — that is the
  Phase 1 of the planned `cuda-prefill-batched-q4k` change.
- Q6_K mmvq — the down projection is already memory-bound thanks
  to the f32 device cache (`cuda-q4k-device-cache`), so the
  expected speedup is small. Reconsider only if the post-Phase-2
  profile shows `proj_down` is still material.
- True Tensor Core (`mma.sync.m16n8k32.s32.s8.s8`) batched matvec
  for multi-token decode. Single-token decode doesn't fill the
  16×8 mma shape; the win is bigger when batched. Out of scope
  here.
- F16/BF16 weight storage. Separate change.
- Multi-GPU / tensor parallelism. Single device, single stream.

## Capabilities

### New Capabilities

(none — extends `compute-cuda-kernels`.)

### Modified Capabilities

- `compute-cuda-kernels` — adds requirements for the Q8_1
  quantize kernel, the Q4_K × Q8_1 mmvq kernel, the per-layer
  input-vector Q8_1 cache, and the runtime dispatch flag.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/elem.rs` — adds
    `quantize_q8_1_device`.
  - `crates/larql-compute/src/cuda/q4k_mmvq.rs` (new) — the
    `__dp4a` Q4_K mmvq kernel + launch wrapper.
  - `crates/larql-compute/src/cuda/q4k_direct.rs` — unchanged
    semantically; remains the back-out path.
  - `crates/larql-compute/src/cuda/backend.rs` —
    `q4k_matvec_device` becomes a dispatcher; new
    `q4k_matvec_device_mmvq` plumbs through to the new kernel.
  - `crates/larql-compute/src/cuda/decode.rs` — wires the
    once-per-layer Q8_1 quantize and shares the result across
    q/k/v and gate/up projections.
  - `crates/larql-compute/tests/test_cuda_q4.rs` — adds the
    mmvq vs direct parity test.

- **Affected systems**: GPU container only. CPU FFN container
  unaffected. Metal backend unaffected. Hardware requirement:
  `__dp4a` works on sm_61 (Pascal) and later — every CUDA
  device LARQL targets satisfies this.

- **Provenance**: bottleneck identified by direct comparison
  against `johndpope/llama-cpp-turboquant` at commit
  `08e025c06ab521e4fa9e5c08b80af57614543e53` (the same
  upstream LARQL imported its RotorQuant CUDA kernels from).
  Reference implementation lives at
  `ggml/src/ggml-cuda/{mmvq.cu, vecdotq.cuh}` in that repo.

- **Out-of-scope notes**: `__dp4a` does not provide
  Tensor-Core-rate throughput for batch=1. True Tensor Core
  acceleration would require either batching multiple decode
  positions or doing matvec via `mma.m16n8k32` with output
  reduction across the 16 rows of the mma tile. The latter is
  an open question for a follow-up change.

## Risks and back-out

- **Numerical drift.** Quantizing the input to Q8_1 introduces a
  per-block fp16 scale + s8 quantization. Worst-case absolute
  error per accumulator element is bounded by `scale * 0.5 *
  hidden`. On Gemma 3 4B (`hidden=2560`) the empirical bound
  measured upstream is ≈ `5e-3` for the largest layer; greedy
  token IDs nonetheless agree on long runs because the
  per-element noise is unbiased and uncorrelated with the
  argmax. Mitigation: parity test asserts max-element ≤ 1e-3 vs
  the existing f32 path; smoke test asserts greedy token IDs
  agree across 20 decode steps.
- **Kernel correctness.** Q4_K's super-block layout (144 bytes,
  packed 6-bit scales + mins) is fiddly and easy to get wrong;
  the upstream kernel has had years of bug-fixing. Mitigation:
  the `vec_dot_q4_K_q8_1_impl_vmmq` body is small (~30 lines)
  and we port it close-to-verbatim into NVRTC source rather
  than rewriting from scratch. License (MIT/llama.cpp) is
  compatible; the kernel source gets a `// from llama.cpp
  vecdotq.cuh, MIT` provenance comment.
- **Back-out:** `LARQL_CUDA_Q4K_MMVQ=0` reverts to the existing
  direct-f32 Q4_K kernel at runtime. The new code is purely
  additive; the old kernel stays compiled and reachable.

## Acceptance bar

Measured on the dev box (RTX 4090, CUDA 12.5, Gemma 3 4B Q4_K
vindex, 20 tokens after 3 warmup):

| Metric | Today (post-tightening) | Phase 3 target | llama-cpp-turboquant |
|---|---:|---:|---:|
| `decode ms/token` | 19.49 | ≤ 10 | 4.40 |
| `GPU fwd ms/token` | 17.491 | ≤ 8 | — |
| `tok/s` | 51.3 | ≥ 100 | 227.5 |
| Bit parity vs host fallback | ≤ 1e-3 | ≤ 1e-3 | — |
| Greedy 20-token id parity | not gated | bit-equal vs f32 path | — |

If Phase 3 misses by > 25% (i.e. `decode ms/token > 12.5`),
abort and document the residual cost in the PR description
before opening a follow-up. The most likely miss mode is
`__dp4a` throughput being capped by something other than what
we modelled (e.g., scale/min decode being on the critical
path), in which case the remediation is kernel-level and
benefits from `nvprof` profiling, not more architecture work.

A 2× speedup (≤ 10 ms/tok) would close roughly 60% of the gap
with llama-cpp-turboquant, leaving the rest in: (a) CUDA Graph
launch-overhead amortization, (b) batched mmq for prefill,
(c) true Tensor Core paths once batched decode is in scope.
