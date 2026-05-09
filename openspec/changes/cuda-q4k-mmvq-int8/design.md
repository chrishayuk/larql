# cuda-q4k-mmvq-int8 — design

## Why dp4a, not full mma

`mma.sync.m16n8k32.s32.s8.s8` (Tensor-Core INT8 matmul) requires
filling at least a 16×8 output tile per warp — meaning at least
16 input rows of Q. Single-token decode produces one Q row.
Without batching, mma is leaving the other 15 rows idle, and
the dispatch overhead exceeds the dot-product time.

`__dp4a` (single-instruction 4-way INT8 dot → INT32 accumulator)
is the correct primitive for batch=1. It runs on a separate
INT pipeline at 4× the rate of fp32 multiply-add and doesn't
need a tile shape. llama.cpp's `vec_dot_q4_K_q8_1_impl_vmmq` is
the well-tuned reference; we port it close-to-verbatim.

When we eventually want batched decode (multi-token speculative
or batched serving), revisit with `mma.sync` — but that's a
different change.

## Q8_1 layout

Each 32-element Q8_1 block is:

```c
struct block_q8_1 {
    half2 ds;     // .x = scale (fp16), .y = sum-of-elements * scale (fp16)
    int8_t qs[32];
};
```

Total: 4 + 32 = 36 bytes per block.

For a 2560-element input vector that's 80 blocks = 2880 bytes
device-side per quantize call. Cheap to compute, cheap to keep
on device.

The `ds.y` term is the `sum(x_i)` — needed because Q4_K stores
weights as `(scale × q4 - dmin × m)`, so the output dot is

```
dot = scale * sum(q4_i * q8_i) - dmin * m * sum(q8_i)
                                       ^^^^^^^^^^^^^^
                                       comes from ds.y / ds.x
```

This is what gets the `m[i]` term its sum-of-input multiplier
without re-reading the input.

## Quantize kernel

```cuda
extern "C" __global__ void quantize_q8_1_f32(
    const float* x,
    int n_blocks,        // n_elements / 32
    int8_t* qs,          // [n_blocks * 32]
    half2* ds            // [n_blocks]
) {
    int b = blockIdx.x;
    if (b >= n_blocks) return;
    int t = threadIdx.x;             // 0..31
    int i = b * 32 + t;

    float v = x[i];

    // amax = max |v| within block (warp reduction)
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    float scale = amax / 127.0f;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    int q = __float2int_rn(v * inv_scale);
    q = max(-128, min(127, q));
    qs[i] = (int8_t)q;

    // sum_x = sum(v) within block (warp reduction)
    float sum_x = v;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sum_x += __shfl_xor_sync(0xffffffff, sum_x, o);
    }

    if (t == 0) {
        ds[b] = __floats2half2_rn(scale, scale * sum_x);
    }
}
```

One block per 32-element group, 32 threads (= 1 warp), pure
warp-shuffle reductions, no shared memory. Trivial occupancy.

## Q4_K × Q8_1 mmvq kernel

Skeleton (close-to-verbatim port of llama.cpp's mmvq):

```cuda
#define QK_K     256
#define QI8_1    8                   // 32-byte block / 4 = 8 ints
#define QR4_K    2                   // i.e. each Q4_K iqs spans 2 Q8_1 blocks

extern "C" __global__ void mul_mat_vec_q4_K_q8_1(
    const uint8_t* __restrict__ vbq,   // packed Q4_K weights
    const int8_t*  __restrict__ q8_qs, // Q8_1 input quants
    const half2*   __restrict__ q8_ds, // Q8_1 input scales
    float*         __restrict__ dst,
    int rows, int n_super_blocks       // n_super_blocks = hidden / 256
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    int tid = threadIdx.x;             // 0..31
    extern __shared__ float smem[];
    float* row_smem = smem + threadIdx.y * 32;

    float sumf = 0.0f;
    const uint8_t* row_base = vbq + (size_t)row * n_super_blocks * 144;

    // Each warp lane owns 4 of the 16 sub-blocks (16 sub-blocks / 32 lanes
    // doesn't tile cleanly; the upstream solves this by having each lane
    // step through (n_super_blocks * 16 / 32) iterations of QR4_K=2 sub-
    // blocks, accumulating sumf_d and sumf_m, then folding dm4f at the end.

    // Inline body lifted from vec_dot_q4_K_q8_1_impl_vmmq:
    //   for each sub-block i in [0, QR4_K):
    //     v0i = (v[0] >> (4*i)) & 0x0F0F0F0F
    //     v1i = (v[1] >> (4*i)) & 0x0F0F0F0F
    //     dot1 = __dp4a(v1i, u[2i+1], __dp4a(v0i, u[2i+0], 0))   // SIMD INT8 dot
    //     dot2 = __dp4a(0x01010101, u[2i+1], __dp4a(0x01010101, u[2i+0], 0))
    //     sumf_d += d8[i] * (dot1 * sc[i])
    //     sumf_m += d8[i] * (dot2 * m[i])
    //   sumf += dm4f.x * sumf_d - dm4f.y * sumf_m

    // Warp-reduce sumf, lane 0 writes:
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, o);
    }
    if (tid == 0) dst[row] = sumf;
}
```

`__dp4a(a, b, c)` = `c + (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)`
where each component is interpreted as INT8 inside a packed
INT32. Single SASS instruction on sm_61+. We invoke it via the
NVRTC built-in (no inline asm needed).

## Where to put the kernel source

We've consolidated NVRTC sources in `q4k_direct.rs` and
`elem.rs`. The mmvq kernel is large enough (~80 lines C) that
embedding it inline would crowd the file. Plan: a new
`crates/larql-compute/src/cuda/q4k_mmvq.rs` that mirrors
`q4k_direct.rs` shape — module-level `Q4K_MMVQ_SRC` const,
`OnceLock` for the loaded function, `matvec_device` entry point.

## Sharing Q8_1 across projections

`decode_token_device` currently does:

```text
h_attn_dev = rms_norm_device(h_dev, input_norm)
q_dev      = q4k_matvec_device(wq, h_attn_dev, ...)
k_dev      = q4k_matvec_device(wk, h_attn_dev, ...)
v_dev      = q4k_matvec_device(wv, h_attn_dev, ...)
```

Three Q4_K matvecs share the same input. With mmvq the input
needs to be Q8_1-quantized. The win is computing the Q8_1 once:

```text
h_attn_dev   = rms_norm_device(h_dev, input_norm)
h_attn_q8_1  = quantize_q8_1_device(h_attn_dev)             ← once
q_dev        = q4k_matvec_device_mmvq(wq, h_attn_q8_1, ...)
k_dev        = q4k_matvec_device_mmvq(wk, h_attn_q8_1, ...)
v_dev        = q4k_matvec_device_mmvq(wv, h_attn_q8_1, ...)
```

Same idea for gate/up:

```text
h_ffn_q8_1   = quantize_q8_1_device(h_ffn_dev)              ← once
gate_dev     = q4k_matvec_device_mmvq(gate, h_ffn_q8_1, ...)
up_dev       = q4k_matvec_device_mmvq(up,   h_ffn_q8_1, ...)
```

5 mmvq calls per layer, 2 quantize calls per layer. Hidden=2560
quantize is ≈ 80 blocks × 32 threads = 2560 threads of trivial
work — should be < 50 µs per call.

## What about wo and down

`wo` reads `attn_out_dev` (q_dim ≈ 2048 elements) — single use.
`down` reads `act_dev` (inter ≈ 10240 elements) — single use,
Q6_K weight (we have a cached f32 device buffer for it already).

For wo, the quantize cost (~q_dim / 32 = 64 blocks) is a one-time
per-layer cost, and the matvec compute is ~half the cost of
gate/up. Whether the win covers the quantize is unclear at the
spec stage — the bench will tell. Plan: ship wo on the existing
direct f32 path in Phase 3, measure, and only adopt mmvq for it
in a follow-up if the bench confirms a win.

For down, the input is the FFN intermediate (inter elements);
the weight is Q6_K. Q6_K mmvq is a separate kernel pattern, and
we already cache the dequantized f32 weight on device. Out of
scope for this change; may be revisited.

## Numerical contract

Per-row error model:

- Each Q8_1 block contributes a quantization error bounded by
  `scale_q8_1 * 0.5` per element.
- The 6-bit Q4_K scales are exact (they're stored as integers
  and decoded; the inexactness is in the f16 `d` and `dmin`
  super-block scales).
- The accumulator is fp32 throughout; the final `dm4f.x*sumf_d -
  dm4f.y*sumf_m` is fp32.
- Across `hidden=2560` accumulations the per-output element
  error is bounded by:
  `2560 × (scale_q8_1 × 0.5 × |q4_K_value|_max × dm4f_max)
  ≈ 5e-3` worst-case, typical ~1e-3.

This matches the upstream tolerance and matches what we already
require for Phase 1's parity test — no contract change.

## Phase boundaries

| Phase | Deliverable | Test gate | Bench gate |
|---|---|---|---|
| 1 | `quantize_q8_1_device` + parity test | `q8_1_quantize_roundtrips_to_within_quant_noise` | n/a (no decode wiring) |
| 2 | `q4k_matvec_device_mmvq` | `q4k_mmvq_matches_q4k_direct` (≤ 1e-3) | n/a (env var off by default) |
| 3 | Wire into `decode_token_device`; share Q8_1 across q/k/v and gate/up | existing `decode_token_phase1_matches_host_fallback` still passes; new `decode_q4k_gemma3_20_tokens_match_host` (gated) | `decode ms/token ≤ 10` AND `GPU fwd ms/token ≤ 8` |

## Decision gates

- After Phase 1: if quantize takes > 100 µs/call at hidden=2560,
  the share-across-projections design needs revisiting. Likely
  cause: shfl_xor pattern wrong for non-warp-aligned tail.
- After Phase 2: if mmvq matvec is slower than `q4k_direct.rs`
  on a microbench (`gate × hidden` = 10240 × 2560), abort and
  inspect `cuobjdump` SASS. Almost certainly a mistake in the
  unroll factor or the dot accumulation order.
- After Phase 3: if `decode ms/token > 12.5`, profile residual
  cost. Possibilities: (a) Q8_1 quantize on critical path —
  fuse into rms_norm via a `rms_norm_quantize_q8_1_device`
  kernel; (b) wo/down still f32 — port them to mmvq.

If we hit ≤ 8 ms/tok we're done; archive.

## Test plan

| Layer | Test | Status |
|---|---|---|
| Unit | `q8_1_quantize_roundtrips_to_within_quant_noise` | new |
| Unit | `q4k_mmvq_matches_q4k_direct` (small + Gemma-shape) | new |
| Unit | `q4k_mmvq_dispatch_via_env_var` (LARQL_CUDA_Q4K_MMVQ toggle) | new |
| Integration | `decode_token_phase1_matches_host_fallback` | unchanged, must still pass |
| Smoke (gated) | `decode_q4k_gemma3_20_tokens_match_host` | new |

## Bench plan

```bash
LARQL_CUDA_AVAILABLE=1 \
./target/release/larql bench output/gemma-3-4b-it-vindex \
    --backends cuda --tokens 20 --warmup 3 --verbose
```

Phase 3 acceptance: decode ms/token ≤ 10. Side-by-side comparison
recorded in the PR description against the current llama-cpp-
turboquant baseline of 4.40 ms/token.
