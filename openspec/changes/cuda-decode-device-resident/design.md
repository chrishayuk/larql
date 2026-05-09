# cuda-decode-device-resident — design

## Where the time goes today

Per-layer call shape in `decode.rs::decode_token` (commit
`f1a24ab`):

```text
                                         crossing
─────────────────────────────────────────────────
rms_norm_vec(h, …)         CPU
matvec(wq, h_attn, …)      GPU launch  →  D2H sync     (1)
matvec(wk, h_attn, …)      GPU launch  →  D2H sync     (2)
matvec(wv, h_attn, …)      GPU launch  →  D2H sync     (3)
fused_decode_attention     H2D K/V slabs → GPU → D2H   (4 H2D + D2H bound)
matvec(wo, attn_out, …)    GPU launch  →  D2H sync     (5)
rms_norm_vec(attn_delta)   CPU
add_in_place(h_post_attn)  CPU
matvec(gate, h_ffn, …)     GPU launch  →  D2H sync     (6)
matvec(up,   h_ffn, …)     GPU launch  →  D2H sync     (7)
activate(gate, up)         CPU (silu)
matvec(down, act, …)       GPU launch  →  D2H sync     (8)
rms_norm_vec(ffn_delta)    CPU
add_in_place(h_out)        CPU
```

8 cudaMemcpy device-to-host calls per layer, 5 of which are tiny
(per-vector `[hidden_dim]` = 2560 floats = 10 KB). Each carries
implicit `cudaStreamSynchronize` cost: bus signalling + driver
serialisation + CPU re-enqueue ≈ 0.4–0.8 ms each on the dev box.
At 34 layers × 8 = 272 round-trips per token, a 0.5 ms median
implies ~136 ms of pure overhead — the dominant share of the
observed 160 ms.

## Phase 1 — keep projections on the device

The smallest meaningful win. We keep the kernel implementations
unchanged but change what they return.

### New API surface

```rust
impl CudaBackend {
    /// Q4_K matvec, device-resident. Same packed weight bytes,
    /// same Q8_K quantised input scratch, but the output stays
    /// on the GPU as a `CudaSlice<f32>` of length `rows`.
    pub fn q4k_matvec_device(
        &self,
        weight: &[u8],
        x_device: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<CudaSlice<f32>, CudaInitError>;
    // Symmetric: q6k_matvec_device, q4kf_matvec_device, f32_gemv_device.
}
```

The existing host-returning `q4k_matvec(...)` is kept as a thin
wrapper:

```rust
pub fn q4k_matvec(&self, weight: &[u8], x: &[f32],
                  rows: usize, cols: usize)
    -> Option<Vec<f32>>
{
    let x_dev = self.htod_copy(x).ok()?;
    let out = self.q4k_matvec_device(weight, &x_dev, rows, cols).ok()?;
    self.dtoh_sync_copy(&out).ok()
}
```

This preserves callers that genuinely want host output (tests,
prefill bridge for the FP32 path, `LARQL_CUDA_DECODE_HOST_FALLBACK=1`).

### Decode loop — phase 1 shape

```text
h_dev = htod_copy(x)                                      ← 1 H2D per token
for layer in layers:
    h_norm_dev = rms_norm_vec_dev(h_dev, …)               ← still CPU in P1
    qkv_dev = (q_dev, k_dev, v_dev) =
        backend.q4k_matvec_device(wq/k/v, h_norm_dev, …)  ← stays on GPU
    attn_out_dev = fused_decode_attention_device(
        q_dev, k_dev, v_dev,
        kv_slot.k_dev, kv_slot.v_dev, …,
    )                                                     ← already on GPU
    delta_dev = backend.q4k_matvec_device(wo, attn_out_dev, …)
    h_post_attn = h + dtoh_copy(rms_norm(delta_dev))      ← rare D2H
    h_ffn_dev = rms_norm_vec_dev(h_post_attn, …)
    gate_dev = backend.q4k_matvec_device(gate, h_ffn_dev)
    up_dev   = backend.q4k_matvec_device(up,   h_ffn_dev)
    act_dev  = silu_gate_up_dev(gate_dev, up_dev)         ← still CPU in P1
    ffn_delta_dev = backend.q4k_matvec_device(down, act_dev)
    h = h_post_attn + dtoh_copy(rms_norm(ffn_delta_dev))  ← rare D2H
return h                                                   ← 1 D2H per token
```

In Phase 1 the rms_norm and activate are still on CPU, so we
**still pay 2 D2H per layer** (around the residual adds), but
that's a 4× drop from 8.

Expected impact on Gemma 3 4B Q4_K: 160 → ≤ 95 ms/token GPU fwd.

## Phase 2 — pull rms_norm / silu / add onto the GPU

Three small kernels:

### rms_norm_vec_device

NVRTC string, single block, 1024 threads. Parallel reduction for
`sum(x²)` → broadcast `inv_rms` → `out[i] = (x[i] *
(weight[i] + norm_offset)) * inv_rms`.

```cuda
extern "C" __global__
void rms_norm_vec(const float* x, const float* w,
                  float eps, float norm_offset, int n, float* out)
{
    __shared__ float sq[1024];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        acc += x[i] * x[i];
    }
    sq[tid] = acc;
    __syncthreads();
    // tree-reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sq[tid] += sq[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(sq[0] / n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * (w[i] + norm_offset) * inv_rms;
    }
}
```

### silu_gate_up_device

Element-wise. The current CPU code does
`out[i] = silu(gate[i]) * up[i]` for the Silu activation; one
launch over `inter` elements.

### add_in_place_device

Pair-wise add. Trivial; bandwidth-bound. Useful as a primitive,
even though its win per-call is small — it's one fewer D2H per
residual.

After Phase 2 the per-layer host crossings drop to **0** in the
non-fallback path. Only the final hidden state comes back.

Expected impact: 95 → ≤ 80 ms/token GPU fwd.

## Phase 3 — device-resident KV cache

`CudaKvCache::layers[*].{k, v}` go from `Vec<f32>` to
`CudaSlice<f32>`. Pre-allocated once at session start (or first
`populate_kv_layer`).

`fused_decode_attention` already operates device-side internally;
the entry point gains a `*_device` variant that accepts
`&CudaSlice<f32>` for K/V instead of `&[f32]`. The internal H2D
copy disappears.

`populate_kv_layer` (used by `larql_inference::predict_honest`'s
post-norm CPU path on prefill) becomes a `htod_sync_copy_into` of
the seeded K/V — same wire shape, just an explicit upload instead
of a per-decode internal one.

Expected impact: 80 → ≤ 60 ms/token GPU fwd.

## Numerical parity

Each phase ships a parity test in
`crates/larql-compute/tests/test_cuda_decode.rs` that compares
the new path to the host-fallback path on the same input,
asserting max-element absolute difference ≤ 1e-3. The bound is
loose enough to absorb fp32 reduction-order differences but
tight enough to catch a real bug.

A separate `test_cuda_decode_q4k_gemma3_smoke` (gated on
`LARQL_CUDA_AVAILABLE=1` and a real vindex on disk) drives 20
decode steps against `output/gemma-3-4b-it-vindex` and compares
each generated token id to the host-fallback path. Tokens MUST
agree exactly under greedy sampling.

## Test plan

| Layer | Test |
|---|---|
| Unit (Phase 1) | `q4k_matvec_device_returns_same_as_host` |
| Unit (Phase 1) | `decode_token_phase1_matches_host_fallback` |
| Unit (Phase 2) | `rms_norm_vec_device_matches_cpu` |
| Unit (Phase 2) | `silu_gate_up_device_matches_cpu` |
| Unit (Phase 3) | `kv_cache_device_roundtrips_through_populate_kv_layer` |
| Smoke (gated) | `decode_q4k_gemma3_20_tokens_match_host` |

## Bench plan

Reuse `larql bench` exactly as the handoff doc lays it out:

```bash
LARQL_CUDA_AVAILABLE=1 \
./target/release/larql bench output/gemma-3-4b-it-vindex \
    --backends cuda --tokens 20 --warmup 3 --verbose
```

Phase-by-phase: record `decode ms/token`, `GPU fwd ms/token`,
`tok/s`. The bench output table goes into the change's PR
description and `docs/cuda-rotorquant-status.md` once the change
archives.

## Decision gates

- After Phase 1: if `decode ms/token > 120` (vs target ≤ 100),
  inspect `nvprof` for residual sync overhead. Don't ship Phase
  2 until Phase 1 hits its target.
- After Phase 2: if `decode ms/token > 90`, the bottleneck is no
  longer sync — it's compute. Phase 3 is unlikely to help much;
  pivot to kernel-fusion work.
- After Phase 3: bake-in. Archive the change.

## Phase 1 profile (2026-05-08, RTX 4090, Gemma 3 4B Q4_K)

Per-token breakdown from `LARQL_CUDA_DECODE_PROFILE=1` averaged
across 5 decode steps (3 tokens after 1 warmup, total 168 ms/tok):

```
attn_call      144.7 ms  86.5%   ← fused_decode_attention_device
norm_cpu         5.7 ms   3.4%   rms_norm + activate + add (CPU)
proj_gate_up     5.0 ms   3.0%   gate + up matvec_device
proj_down        4.8 ms   2.8%   down matvec_device
proj_wo          3.0 ms   1.8%   wo matvec_device
proj_qkv         1.8 ms   1.1%   q + k + v matvec_device
htod             1.0 ms   0.6%   h_attn + h_ffn + act H2D
dtoh_gate_up     0.6 ms   0.4%   gate + up D2H
dtoh_attn_delta  0.3 ms   0.2%
residual_cpu     0.3 ms   0.2%
dtoh_ffn_delta   0.2 ms   0.1%
```

Each `attn_call` is 4.26 ms/layer. The K/V cache slabs at
`max_seq=4096 × num_kv_heads=4 × head_dim=256 × f32` = 16 MB per
slab. `fused_decode_attention_device` does 2 H2D + 2 D2H of
those slabs per call → 64 MB × 34 layers = **2.2 GB of PCIe
traffic per token**. At PCIe 4.0 x16's ~28 GB/s effective that's
~78 ms/token — accounting for ~54% of the `attn_call` cost. The
remaining ~66 ms is the kernel arithmetic itself.

### Strategic implication

**Phase 2 is no longer worth doing.** The proposal allocated
Phase 2 to remove ~6 ms of host crossings (rms_norm, silu_gate_up,
add_in_place). The profile shows those crossings sum to <6 ms.
Even fully eliminated, decode ms/token drops from 153 to ~147 —
still a long way from the ≤80 Phase 2 target.

**Phase 3 is the only path to the proposal's targets.** Replacing
`CudaKvLayer::{k, v}: Vec<f32>` with `CudaSlice<f32>` removes the
2 H2D + 2 D2H inside every `fused_decode_attention_device` call.
Predicted decode ms/token after Phase 3: ~70-75 (PCIe traffic
goes to zero; kernel time stays). That clears the ≤80 ms Phase 2
target *and* approaches the ≤60 ms Phase 3 target without any
norm/activate kernel work.

The plan therefore changes:
- **Skip Phase 2** entirely. The targeted ops are not on the
  critical path.
- **Phase 3 becomes Phase 2** — fold the device-resident KV
  cache type swap into the next merge.
- If post-Phase-3 numbers still miss ≤ 60 ms/tok, the residual
  cost is kernel-internal and warrants either a fused-FA2-style
  rewrite (separate change) or just bench-archiving at whatever
  the new floor turns out to be.
