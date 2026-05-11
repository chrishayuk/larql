## Context

LARQL's batch=1 decode path on RTX 4090 is at 7.44 ms/token after
the audit pass (`feat/cuda-mmvq-hw-f16-cvt` + 8 sibling branches).
llama.cpp serves the same Gemma 3 4B Q4_K_M at 4.34 ms/token. The
1.71× gap is **architectural**, not a missing micro-optimisation:

- **Tensor Cores are dead at batch=1.** Four independent attempts
  (cuBLAS hgemm, WMMA single-warp, WMMA multi-warp, INT8 IMMA) all
  lost 3-7× to the dp4a SIMT path because their fragment shape
  (16×16) wastes 15/16 columns when only one query token is in
  flight. Empirically settled on
  `feat/cuda-tensor-cores-q4k`, `feat/cuda-attn-wmma-kernel-v2`,
  `feat/cuda-attn-wmma-multi-warp`, `feat/cuda-marlin-imma-probe`.
- **Memory bandwidth is the floor.** Q4_K weights at 4.5 bits/elem
  for Gemma 3 4B (~2.4GB) + KV cache reads dominate decode time.
  At batch=1 we read every weight once per token; there is no
  arithmetic intensity to amortise.

The only known mechanism that lifts effective batch on a single-user
decode path without serving more users is **speculative decoding**:
draft N candidate tokens cheaply, then verify them in a single
batched forward pass through the target model. The verification pass
sees a batch of N+1 tokens, which is enough to make Tensor Core
fragments productive again.

llama.cpp ships speculative decoding (`-md` flag). EAGLE-2
(<https://arxiv.org/abs/2406.16858>) and Medusa
(<https://arxiv.org/abs/2401.10774>) are the published designs we'll
reference. We pick EAGLE because (a) draft head is small (~80M for a
4B target), (b) acceptance rate published as 0.6-0.8 on
chat-completion workloads, (c) draft + target can share weights
except the LM head and one transformer block.

## Goals / Non-Goals

**Goals:**

- End-to-end decode 1.5-1.8× faster than current LARQL non-speculative,
  bringing LARQL within noise of llama.cpp on Gemma 3 4B Q4_K_M.
- Verification step SHALL be **bit-equal** to non-speculative decode
  on a fixed 256-prompt eval. Any deviation is a stop-the-line bug.
- Tensor Core paths previously empirically dead at batch=1 SHALL be
  re-enabled at batch ≥ 4.
- Rollout is gated behind `LARQL_SPECULATIVE_DECODE=1` for the entire
  development cycle. The non-speculative path stays the default and
  bit-exactly unchanged.
- The rotorquant KV cache subsystem keeps its current compression
  invariants — speculative writes that get rejected SHALL be rolled
  back cleanly without leaving partial-rotor state.

**Non-Goals:**

- Training the EAGLE draft head from scratch. We document the
  checkpoint loader and ship one or two pre-trained heads (Gemma 3 4B,
  optionally Llama 3 8B). Training pipelines belong in a separate
  change (`research-eagle-training` or similar, not yet proposed).
- Multi-user serving / batched concurrent requests. That's the
  existing `attention-service-routes` work. This change only lifts
  batch via speculation for one user at a time.
- CPU and Metal speculative paths. Flag is a no-op outside `cuda`.
- Tree-of-thoughts or beam-style speculation; we only do prefix-tree
  speculation that an EAGLE-style draft proposes.
- Runtime draft-model swapping. Pinned at session start.

## 1. Algorithm overview

### 1.1 Single-step speculative decode (depth-1)

```
1. Run draft head once on target's last hidden state h_t  →  draft tokens d_1..d_K
2. Append d_1..d_K to target sequence (provisionally; will be rolled back if rejected)
3. Run target model **once** on tokens [d_1, ..., d_K] in parallel  →  target logits L_1..L_K
4. For k = 1..K, accept d_k with probability min(1, p_target(d_k) / p_draft(d_k))
5. On first rejection at k=r, sample a corrected token from the residual distribution
   max(0, p_target - p_draft) (renormalised) and emit accepted prefix d_1..d_{r-1} + d_r'
6. On all-accept, emit d_1..d_K plus one bonus token sampled directly from L_K
```

The acceptance rule guarantees the output distribution is
**identical** to greedy/temperature sampling from the target model.
This is the load-bearing correctness property.

### 1.2 Tree speculation (depth ≥ 2, our default)

EAGLE-2 ranks each draft token by its expected contribution to
acceptance and grows a small (typically 26-64 node) tree where each
node is a candidate continuation. The target model runs **once** with
a custom causal mask that respects the tree's parent-child structure.
Verification walks the tree depth-first and emits the longest
accepted path.

We will start with depth=2, branches=2 (5 nodes) for simplicity
and tune up after acceptance-rate measurement.

### 1.3 Why this lifts batch

The target model's verification pass sees `q_tokens = N` (e.g. N=5
for our default tree). The mmvq kernel and attention score kernel
become batched along Q, which fills 5/16 columns of a Tensor Core
fragment instead of 1/16. At N≥4 the cuBLAS hgemm path becomes
faster than dp4a; at N≥8 the WMMA paths win.

## 2. Where the boundaries cut

### 2.1 New module: `larql-inference::speculative`

```rust
// Trait + EAGLE impl.
pub trait Drafter {
    fn propose(&self, h_target: &[f32], n: usize) -> Vec<DraftToken>;
}
pub struct EagleDraftHead { /* layer + lm_head + draft KV cache */ }
impl Drafter for EagleDraftHead { ... }

// Verification + accept/reject. Uses target probabilities.
pub fn verify_and_accept(
    target_logits: &[Logits],
    draft_probs: &[Probs],
    rng: &mut Rng,
    temperature: f32,
) -> AcceptedSpan;

pub struct SpeculativeDecoder<'m, D: Drafter> {
    target: &'m TargetModel,
    drafter: D,
    cfg: SpecConfig,
}
impl SpeculativeDecoder<'_, _> {
    pub fn step(&mut self, kv: &mut KvCache) -> Vec<TokenId>;
}
```

`step()` returns 1..=K+1 tokens per call. Caller (the inference
loop) iterates until stop-token / max-len.

### 2.2 New CUDA kernels: `cuda::attn_tree`, `cuda::sampling`

- `cuda::attn_tree::tree_decode_attention` — the existing
  `fused_decode_attention` lifted to `q_tokens > 1`, with a
  per-q-token causal mask uploaded as a `q_tokens × kv_len + q_tokens`
  bitmask buffer. The mask covers the cache portion (causal: every q
  sees all of cache) plus the tree portion (q_i sees only its
  ancestors).
- `cuda::sampling::verify_tree` — fused softmax + acceptance loop +
  residual sampling. One launch produces accepted prefix + corrected
  rejection token + bonus token.

### 2.3 Modified CUDA kernels

- `cuda::q4k_mmvq` gets a `M_TILE` constexpr template parameter
  (1, 2, 4, 8) and a dispatcher. The cooperative path stays the
  default at `M_TILE=1`. At `M_TILE ≥ 4` the dispatcher goes through
  the dormant cuBLAS hgemm hot path since the column fragment is
  no longer mostly waste.
- `cuda::attn::fused_decode_attention` gains `q_tokens` parameter,
  defaulting to 1 (preserving the bit-exact current path).
- `cuda::elem::rms_norm_q8_1` gets batched form
  `rms_norm_q8_1_batch(x: [M, hidden])` — important: this DOES NOT
  fuse across SMs (one block per row), avoiding the regression seen
  in `feat/cuda-fused-norm-quantize`.
- `cuda::decode::DecodeScratch` grows a `tree_q: CudaSlice<f16>` of
  size `max_tree_nodes × hidden`, a `tree_mask: CudaSlice<u32>`, and
  a `accept_buf: CudaSlice<u32>` of size `max_tree_nodes`.

### 2.4 Modified rotorquant API

Speculative writes to the KV cache must be rolled back on rejection.
`larql_rotorquant::compress` currently has fire-and-forget semantics:
`compress(slot, k, v)` writes a permanent compressed entry. We need:

```rust
pub struct ProvisionalWrite { slot: usize, undo_token: u64 }
pub fn compress_provisional(...) -> ProvisionalWrite;
pub fn commit(write: ProvisionalWrite);
pub fn rollback(write: ProvisionalWrite);  // restores previous compressed state
```

The `undo_token` is a generation counter on the slot; rollback is
O(1) and idempotent. Implementation: keep one prior-state
shadow-copy per active speculative window (≤ N slots ≤ 64).

This is the **single most invasive change** for the rotorquant
subsystem. It's why this design doc exists — without thinking
through rotorquant rollback we'd ship a quietly-incorrect KV cache.

## 3. Numerical correctness

### 3.1 The acceptance rule guarantees distributional equivalence

The standard rejection-sampling rule
`accept with prob min(1, p_target/p_draft)` and sample residual
`max(0, p_target - p_draft) / Z` on rejection is mathematically
proven (Leviathan et al. 2022, "Fast Inference from Transformers via
Speculative Decoding") to produce samples from `p_target` exactly.
This is the load-bearing claim — but **only at exact arithmetic**.

In practice, finite-precision softmax at f32 can shift the
distribution by ~1e-7 per element. Over 256k vocab × 5 tree positions
this is ~10x less than the smallest numeric tolerance we hit on
existing fused-attention parity tests (1e-3). We assert bit-equal
**token IDs** on the eval set, not bit-equal probabilities.

### 3.2 Verification approach

- Inline microbench: `cuda_verify_tree_matches_cpu_reference` —
  deterministic RNG seed, fixed logits, assert accepted span equals
  CPU reference's accepted span exactly.
- End-to-end: `bench/decode_speculative.rs --verify` runs 256
  prompts through both `LARQL_SPECULATIVE_DECODE=0` and `=1` with
  the same seed and asserts identical token IDs. Failure here is
  a stop-the-line bug — do not ship.

### 3.3 Why temperature/top-k matter

At `temperature → 0` (greedy), `p_target` is a one-hot, so acceptance
becomes "draft must match argmax(target)". Acceptance rates fall.
At `temperature ≥ 0.7` (typical chat default) acceptance is in the
0.6-0.8 range published by EAGLE-2. We measure both and document.

## 4. Performance model

Decoder cost at depth=2, branches=2 (5-node tree), acceptance rate `α`:

```
expected_tokens_per_step  = 1 + α + α² + α³ + α⁴       (geometric)
target_pass_cost          = T_target_at_batch5
draft_pass_cost           = T_draft_at_batch1
total_per_step            = target_pass_cost + draft_pass_cost
ms_per_token              = total_per_step / expected_tokens_per_step
```

With our measured numbers:

```
T_target_at_batch5  ≈ T_target_at_batch1 × 1.4   (cuBLAS hgemm dominates,
                                                  not 5× because shared
                                                  weight reads amortise)
                    ≈ 7.44 × 1.4 = 10.4 ms
T_draft_at_batch1   ≈ 1.5 ms                     (80M params, ~10× smaller)
total_per_step      ≈ 11.9 ms
α = 0.6:    expected_tokens = 1+0.6+0.36+0.22+0.13 = 2.31
            ms_per_token    = 11.9 / 2.31 = 5.15 ms
α = 0.7:    expected_tokens = 2.71
            ms_per_token    = 11.9 / 2.71 = 4.39 ms       (~ llama.cpp)
α = 0.8:    expected_tokens = 3.36
            ms_per_token    = 11.9 / 3.36 = 3.54 ms       (faster than llama.cpp)
```

**Sensitivity:** if `T_target_at_batch5` grows to 1.6× single-batch
(weight-bound stays dominant) the picture is similar; if it grows to
2.5× we lose the win. Phase 2 measurement of batched mmvq is the
go/no-go gate.

## 5. Rotorquant interaction (the load-bearing risk)

`larql_rotorquant` currently:

1. Writes f16 K/V into the device KV cache.
2. After write, runs the rotor quantizer to compress to 3-4 bits.
3. Replaces the f16 entry with the compressed entry on the next
   read or after a sliding-window threshold.

This is a one-way transformation. Speculative decoding requires a
**rollback** because rejected draft tokens leave KV writes that
must be undone. Three options were considered:

- **A — defer compression for the speculative window**: keep f16
  for the last ≤ 8 KV positions. Simple. Costs ~2x KV memory in the
  active window only (~1 MB at Gemma 3 4B head shape). Roll-back is
  pointer move. **Selected — this is what the design specifies.**
- **B — compress provisionally and shadow-copy**: keep one
  pre-compression snapshot per provisional slot. ~50% memory of A
  but doubles the rotor work per token. Worse: rotor compression is
  one of the cheaper ops, so saving it is not worth the complexity.
- **C — recompute on rollback**: re-run forward to position k after
  rollback to reconstruct the f16 K/V. Saves all memory but adds
  a full forward per rollback, killing the speculative win.

Option A is selected. The rotorquant change is small:
`compress_with_window_lag(slot, lag=8)` postpones compression by 8
positions. Existing readers already tolerate mixed f16/compressed
slots since the demote/promote path was added in
`rotorquant-promote-on-read`.

## 6. Phasing and rollback

Each phase is independently shippable and behind the env flag:

| Phase | Branch                                  | Goal                                      | Stop-ship gate                               |
|------:|-----------------------------------------|-------------------------------------------|----------------------------------------------|
| 1     | `feat/cuda-spec-draft-head`             | Load EAGLE checkpoint, propose 1 token    | Draft tokens deterministic on fixed seed     |
| 2     | `feat/cuda-spec-batched-mmvq`           | Batched mmvq + attn at q_tokens ∈ {1,4,8} | Bit-exact at q=1; ≤ 1.6× cost at q=5         |
| 3     | `feat/cuda-spec-tree-verify`            | Tree attention + verify_tree kernel       | Bit-equal token IDs on 256-prompt eval       |
| 4     | `feat/cuda-spec-tensor-core-rearm`      | Re-enable hgemm + WMMA at batch ≥ 4       | End-to-end ≤ 5.5 ms/tok at α=0.6             |

Each phase lands as its own PR with its own openspec sub-change.
This proposal is the umbrella; the four sub-changes will reference
back to it via `[parent]: ../cuda-speculative-decoding/proposal.md`.

If any phase fails its stop-ship gate, we revert the branch and
**leave the env flag undefaulted**. The non-speculative path
remains the production default.

## 7. Testing

- `crates/larql-compute/tests/test_cuda_attn_tree.rs` — tree
  attention parity vs CPU reference (depth ≤ 4, branches ≤ 4).
- `crates/larql-compute/tests/test_cuda_verify_tree.rs` — verify
  kernel parity vs CPU reference, 64 fixed seeds.
- `crates/larql-inference/tests/test_speculative_parity.rs` —
  end-to-end token-ID parity vs non-speculative on 256 prompts.
- `bench/decode_speculative.rs` — measures ms/tok and acceptance
  rate. Reports the same headline numbers we've been tracking
  against `llama-cpp-turboquant`.

## 8. Open questions

- **EAGLE checkpoint distribution.** We need a Gemma 3 4B EAGLE head
  checkpoint. SafeCoder/EAGLE-2 release a Llama-3 head; for Gemma 3
  we may need to train one (out of scope here) or fall back to a
  smaller separate Gemma 2B as draft (lower acceptance rate, no
  training required). **Decision deferred to phase 1.**
- **Tree shape tuning.** Start at depth=2, branches=2. EAGLE-2's
  paper picks 26-node trees with adaptive branching; we'll measure
  acceptance × wall time for a small grid in phase 4.
- **Sliding window interaction.** Gemma 3 has sliding-window layers;
  speculative window must not extend past the SWA boundary or the
  target verification will see a different attention mask than the
  draft. We bound speculative depth ≤ min(8, swa_window - cache_len).
- **rotorquant lag override.** Default lag=8 may be too aggressive
  if speculative depth grows. Make it
  `max(8, max_speculative_depth + 2)` and document.

## 9. Why not just …

- **… buy more bandwidth.** RTX 4090 is 1 TB/s; we're already pulling
  ~80% of theoretical on Q4_K reads. Hardware ceiling, not a tuning
  problem.
- **… use a 2-bit quant.** Q2_K loses 2-3 PPL points on Gemma 3 4B
  vs Q4_K_M. LARQL's positioning is "loseless decoding", so this is
  off the table.
- **… batch concurrent users.** That's the existing
  `attention-service-routes` work, orthogonal to single-user latency.
  Speculative decoding lifts batch *for one user*; concurrent users
  is an additional multiplier.
- **… port everything to FlashInfer.** FlashInfer is sm_90+ optimised
  (H100/H200); 4090 is sm_89. Their kernels run but don't outperform
  cudarc's. Re-evaluate when sm_90 hardware is the target.
