## Why

Decode on RTX 4090 has reached the practical floor of single-token,
batch-1 dispatch. Side-by-side measurement against
`llama-cpp-turboquant` (Gemma 3 4B Q4_K_M, 4090, same prompt):

| Path     | LARQL (ms) | llama.cpp (ms) | Ratio  |
|----------|-----------:|---------------:|-------:|
| Decode   | 7.44       | 4.34           | 1.71×  |
| Prefill  | 10.7       | 6.25           | 1.71×  |

Across `feat/cuda-mmvq-hw-f16-cvt`, `feat/cuda-attn-wmma-f16kv`,
`feat/cuda-attn-grid-split`, `feat/cuda-attn-rope-hoist`,
`feat/cuda-decode-cuda-graph`, `feat/cuda-q4k-mmvq-warp-cooperative`,
`feat/cuda-fused-norm-add`, `feat/cuda-prefill-tensor-cores` and
`feat/cuda-sfu-intrinsics` we closed the decode gap from 2.18× to
1.71× (38% of the gap closed) and prefill from 3.20× to 1.71× (62%
closed). The remaining gap is structural, not micro-optimisable.

Four independent Tensor Core paths were attempted and empirically
killed by 15/16 column-tile waste at batch=1:

- `feat/cuda-tensor-cores-q4k` — cuBLAS hgemm at `m=1`
- `feat/cuda-attn-wmma-kernel-v2` — WMMA single-warp attention
- `feat/cuda-attn-wmma-multi-warp` — WMMA multi-warp attention
- `feat/cuda-marlin-imma-probe` — Marlin-style INT4-IMMA mmvq

All four lost 3-7× to the dp4a / SIMT path because Tensor Core
fragments are 16×16; at batch=1 we drive one column and waste 15.
**The only known mechanism to lift the effective batch dimension on
a single-user decode path is speculative decoding.**

This change proposes integrating EAGLE-style speculative decoding
into the LARQL inference loop, bringing batch ≥ 4 to the verification
step where the dead Tensor Core paths above become dominant wins.

## What Changes

This is a **multi-phase** change spanning 2-3 implementation weeks.
It is gated behind an env flag (`LARQL_SPECULATIVE_DECODE=1`) for the
entire rollout so the production path stays untouched until phase 4.

### Phase 1 — Draft model integration (week 1)

- ADD `larql-inference::speculative` module hosting the
  `SpeculativeDecoder` trait + `EagleDraftHead` impl.
- ADD a small draft head trained jointly with target (LM-head re-use
  + 1-layer transformer block, ~80M params for Gemma 3 4B target);
  training is out of scope, we ship a checkpoint loader.
- ADD `--draft-model <path>` CLI flag and config plumbing.
- ADD device-side draft KV cache (separate from target KV cache).

### Phase 2 — Batched attention + mmvq path (week 2)

- MODIFY `cuda::attn::fused_decode_attention` to accept
  `q_tokens: u32` (currently hard-coded 1) and broadcast over a
  per-token Q dimension, materialising scores in
  `[q_tokens, n_kv]` instead of `[n_kv]`.
- ADD `cuda::q4k_mmvq::mul_mat_q4_K_q8_1_batched` parameterised on
  `batch ∈ {1,2,4,8}`, dispatching to the same kernel-template with
  `M_TILE` constexpr; batch=1 keeps the cooperative path bit-exactly.
- ADD batched RMSNorm + Q8_1 quantize path (single launch per batch,
  no per-token launch sprawl).
- KEEP the batch=1 path as the default dispatch when
  `LARQL_SPECULATIVE_DECODE` is unset.

### Phase 3 — Tree attention + verification kernel (week 2-3)

- ADD `cuda::attn::tree_decode_attention` — a fused kernel that
  computes attention for a draft *tree* (prefix-shared parent path
  + N branches) in one launch. Mask is uploaded per-batch.
- ADD `cuda::sampling::verify_tree` — given target logits over the
  tree and draft logits, returns the longest accepted prefix per the
  exact distribution-matching rule (rejection sampling on ratio
  `min(1, p_target/p_draft)`, fall through to corrected residual on
  rejection). Numerically verified bit-equal to the CPU reference.
- ADD `larql-inference::speculative::accept_tokens`.

### Phase 4 — Tensor Core unlock + production rollout (week 3)

- RE-ENABLE `feat/cuda-tensor-cores-q4k` cuBLAS hgemm path, but
  gated on `batch ≥ 4` (the four dead branches all become wins
  above the 16-column threshold).
- RE-ENABLE WMMA attention scores at `q_tokens ≥ 4`.
- ADD `--speculative-tree-depth N --speculative-branches K` CLI.
- Flip default to `LARQL_SPECULATIVE_DECODE=1` once acceptance
  rate ≥ 60% on Gemma 3 4B (measured on a fixed eval set).
- ADD `bench/decode_speculative.rs` measuring tok/s and acceptance
  rate against the same `llama-cpp-turboquant` baseline.

## Capabilities

### New Capabilities

- `inference-speculative-decoding` — declares the API surface,
  the verification-correctness contract, and the rollout knobs.

### Modified Capabilities

- `compute-cuda-kernels` — adds batched mmvq, batched attention,
  tree-attention, and tree-verification kernel scenarios.
- `inference-attention-and-kv` — adds the multi-token decode KV
  write path (current write is single-token).
- `kv-cache-rotorquant` — note: rotorquant compression is per-token;
  speculative decode rejects up to N draft tokens, so the rotorquant
  promote/demote logic SHALL roll back any speculative writes that
  the verification step rejects. This is the **single most invasive
  bit** of the change for the rotorquant subsystem and is called
  out separately in `design.md §5`.

## Impact

- **Affected files**: ~25 files, ~3500 LOC. New modules:
  `larql-inference/src/speculative/`,
  `crates/larql-compute/src/cuda/sampling.rs`,
  `crates/larql-compute/src/cuda/attn_tree.rs`. Modifications to
  `cuda/attn.rs`, `cuda/q4k_mmvq.rs`, `cuda/decode.rs`,
  `cuda/scratch.rs`, `larql_rotorquant::compress` rollback API.
- **Affected systems**: CUDA decode + inference loop. CPU + Metal
  unchanged (no draft-model checkpoint required for them; flag is a
  no-op outside `cuda` feature).
- **Performance target**: at acceptance rate 0.6 with depth-2 tree
  (3 candidates), expected end-to-end decode improvement is
  ~1.5×-1.8× on the 4090, putting LARQL at ~4.5-5.0 ms/token vs
  llama.cpp 4.34 ms — **inside the noise floor**.
- **Numerics**: verification SHALL be bit-equal to the
  non-speculative path on an exhaustive 256-prompt eval. Any
  divergence is a stop-the-line bug.
- **KV-cache memory**: doubles temporarily during a draft window
  (target cache + draft cache). For Gemma 3 4B at seq=4096 this is
  ~80MB extra — fits comfortably on the 4090's 24GB.
- **Out of scope**: training the EAGLE draft head (we load a
  pre-trained checkpoint); multi-batch serving (this change only
  lifts batch via speculation, not via concurrent users; that's
  the existing `attention-service-routes` / `router-*` work);
  CPU/Metal speculative paths.
