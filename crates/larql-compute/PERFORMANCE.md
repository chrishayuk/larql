# Performance — larql-compute

Machine: M3 Max, macOS 24.6.0, Gemma 3 4B (34 layers, hidden=2560, inter=10240, vocab=262K)
Vindex: `gemma3-4b-q4k-v2` (Q4_K attn/gate/up, Q6_K V/down — Ollama convention)

> **Note on the historical "81–84 tok/s"**: an earlier ROADMAP table cited
> 81–84 tok/s for this same vindex on 2026-04-26. Bisect (2026-04-28)
> traced that to a silent dispatch bug fixed in commit `077884b "working
> on performance"`: Q4_K weights were routed through the **Q4_KF kernel**
> with the wrong threadgroup geometry (4 rows/TG instead of 8), leaving
> ~75% of output rows unwritten. The 81–84 was real wall-clock
> throughput on broken (wrong-output) code. **78.7 tok/s is the correct
> baseline for valid output.** Reverting 077884b would re-introduce the
> bug.

> **Profiler note (2026-04-28)**: an earlier per-kernel diagnosis claimed
> q4k_ffn_gate_up was "ALU-limited at 103 GB/s, compute-bound on Q4_K
> dequant". That was a profiler bug — `measure_batched` was creating a
> fresh cmd buffer per kernel call (with commit+wait per call) instead
> of running `n_layers` dispatches in one cmd buffer, so per-call
> dispatch overhead dominated the measurement. Fixed via
> `measure_single_cmdbuf_batched`. Corrected numbers: q4k_ffn_gate_up at
> **274 GB/s = 74% of LPDDR5X peak (bandwidth-bound)**, not 103 GB/s
> compute-bound. Both big FFN kernels are at bandwidth saturation; the
> remaining ~1.17× decode gap to ollama is distributed across the
> pipeline, not concentrated in any single kernel.

---

## CPU kernels (2026-05-15)

Per `crates/larql-compute/src/cpu/ops/`. NEON-accelerated on aarch64
(always available on Apple Silicon); scalar fallback elsewhere. All
NEON kernels have parity tests against the scalar reference.

### Production CPU matvecs

| Kernel | File | Purpose | NEON inner | Parity |
|---|---|---|---|---|
| `q4k_matvec_into` | `ops/q4_common.rs::535` | Q4_K weights × f32 vec | `q4_dual_dot_32_neon` — 16-byte chunk, 8 f32x4 accumulators (4 lo + 4 hi), tree reduce | `q4k_matvec_matches_dequant_then_matmul` |
| `q4k_dual_matvec_into` | `ops/q4_common.rs` | Two Q4_K matrices × one shared x; gate+up and K+V both use it | Inner shares `q4_dual_dot_32` with above | `q4k_dual_matvec_into_matches_two_sequential_calls` (bit-exact) |
| `q6k_matvec::dispatch` | `ops/q6k_matvec.rs::42` | Q6_K weights × f32 vec (FFN down) | `q6_subblock_dot_16_neon` — 16-element sub-block, `vqtbl1q_u8` broadcast + per-lane `vshlq_u8` for hi2 bits, 4 mul + tree reduce | `q6_subblock_matches_scalar_full_6bit_range` |
| `parallel_lm_head_logits` + `f32_dot_neon` | (in `larql-inference`) | f32 sgemv over 262K-vocab lm_head, row-parallel | `f32_dot_neon` — 4 independent f32x4 accumulators, 16 f32/iter | `f32_dot_matches_scalar_on_aligned_length` |

### Why a single 4-way accumulator matters

M3's NEON FMA is 4-cycle latency, 1-per-cycle throughput. A single
accumulator chain (`acc = vfmaq(acc, ...)` four times) serialises on
the destination register at one FMA every 4 cycles = 25% of peak. Four
independent accumulators let the FMAs pipeline at 1/cycle = 100% of
peak.

Empirically the 4-way restructure (2026-05-15) bought a modest 4% on
end-to-end decode — well short of the theoretical 4×, because the
inner loop is now bandwidth-limited on Q4_K reads (~30 GB/s per
thread × 11 threads, well below LPDDR5 peak but ~3× below llama.cpp's
effective rate on the same hardware).

### Why fusion didn't help as predicted

Both `q4k_dual_matvec` callers (gate+up in the FFN, K+V in attention)
operate on a `[1, hidden] = 10 KB` input vector. The hypothesis was
that fusing two matvecs would keep `x` hot in L1 across the two weight
decodes, saving ~80 MB of re-stream per layer. **The saving wasn't
real**: `x` is 10 KB, L1 on M3 is 128 KB, and a single matvec doesn't
evict it. Two sequential matvecs already hit L1 on the x reads. The
fused version still gives correctness-equivalent output via a single
shared `sum_x` precompute + one fewer rayon launch, but the
end-to-end Δ is within bench variance.

Documented because the inverse error (assuming a bandwidth win where
there isn't one) is the kind of thing a future maintainer might want
to re-test rather than re-derive.

### What's still scalar

`q4k_matvec::dispatch` (separate from `q4k_matvec_into`) and the
correctness-oracle paths under `#[cfg(test)]` remain scalar — they're
references the NEON kernels are validated against. Production callers
hit `q4k_matvec_into` via `CpuBackend::q4k_matvec`.

---

## Current state (2026-05-09, post QKV defuse)

```
larql-metal  gemma3-4b-q4k-v2     87.9–88.1 tok/s 11.35–11.41 ms/tok (post QKV defuse, quiet GPU)
larql-metal  gemma3-4b-q4k-v2     85.0–86.3 tok/s 11.71–11.79 ms/tok (pre QKV defuse, post dispatch-geometry fix)
larql-metal  gemma3-4b-q4k-v2     76.1–76.7 tok/s 13.06–13.14 ms/tok (pre dispatch fix; stride-32 lm_head workaround)
larql-metal  gemma3-4b-q4k-v2     74.6–75.6 tok/s 13.22–13.41 ms/tok (post O-proj routing fix only)
larql-metal  gemma3-4b-q4k-v2     72–75 tok/s      13.5–13.9 ms/tok  (pre O-proj routing fix)
Ollama       gemma3:4b            101.5–115.9 tok/s ~8.6–9.9 ms/tok (steady-state, same harness; ±15% session-to-session)
Gap          1.17×                ~1.66 ms/tok                 (was 1.18× before defuse, 1.30× before dispatch fix)

larql-metal  gemma4-26B-A4B        19.0–19.8 tok/s ~52ms/tok  (post 2026-05-02 moe_dispatch geometry fix)
larql-metal  gemma4-26B-A4B          5.1 tok/s   ~194ms/tok   (pre-fix; broken dispatch was masking ~3.8× perf AND degrading output)
LARQL_SKIP_MOE=1 ceiling           56.8 tok/s   ~15ms/tok    (attention + dense FFN only)
```

Per-stage (Gemma 3 4B, 100-token run, 8 warmup, 2026-05-09 post QKV defuse):

| Stage | ms/tok | % |
|---|---|---|
| GPU fwd | ~11.41–11.52 ms | 85–86% |
| lm_head | **~1.84–1.97 ms** | 14% |
| embed + norm + detok | ~0.04 ms | <1% |

The QKV defuse change (2026-05-09) cuts decode from 11.79 → 11.35 ms/tok
(−0.30 ms/tok GPU fwd, +1.6–1.8 tok/s end-to-end) by skipping the fused
`q4k_q6k_qkv_proj_normed` kernel and using a separate `rms_norm` dispatch
+ non-fused `q4k_q6k_qkv_proj` instead. The fused kernel rereads H + norm_w
3× per TG (4 simdgroups, different stride patterns for Q4_K Q/K vs Q6_K V),
dropping it from 287 → 199 GB/s — a 1.46 ms/tok kernel cost that exceeds
the 0.24 ms/tok dispatch saving the fusion gave. End-to-end magnitude was
18% of what the per-kernel diag predicted; see ADR-015 § "Magnitude can
compress 4×" for why bandwidth-headroom wins compress in the live decode
pipeline. `LARQL_QKV_FUSED=1` opts back in.

The dispatch-geometry fix (2026-05-02) cuts lm_head from 2.95 → 1.85 ms
(−1.14 ms/tok, +7.7 tok/s end-to-end) by making `MetalBackend::q4k_matvec`
and the three sibling sites in `moe_dispatch.rs` + `decode/encode_ffn.rs`
use `pipeline.rows_per_tg` / `pipeline.threads_per_tg` instead of hardcoding
`shaders::q4k_matvec::ROWS_PER_TG`. Production has bound the 8sg pipeline
since 2026-04-28; the hardcoded 4sg constants left simdgroups 4..7 of
each TG unscheduled, corrupting half the lm_head output rows. See
"Decision: lm_head dispatch order" below for full root-cause analysis.

The 78.7 / 80.3 tok/s headlines below are preserved for context but
predate (a) the v5 lm_head stride-32 correctness *workaround*, (b) the
2026-05 dispatch-fusion wave, (c) the 2026-05-02 dispatch-geometry
*fix* that obviated the workaround, and (d) the 2026-05-09 QKV defuse.
The honest current number is **88 tok/s** with correct output, gap to
ollama 1.17×.

---

## Decision log

Canonical reference for **what is the production default and why**. Each
entry is self-contained: options measured, data, chosen path, rationale,
opt-out env vars. The "Recent changes" table below remains the chronological
log; this section is the by-topic reference.

Decision blocks added here when (a) a path was chosen between ≥2 measured
candidates, OR (b) a candidate looked promising but was deliberately not
promoted. Both are the kind of context that tends to evaporate from PRs and
flat changelogs.

### Decision: lm_head dispatch order (2026-05-02, revised)

**Question:** which Metal lm_head kernel runs by default for a non-CPU
backend on a Q4_K vindex with tied embeddings (`gemma3-4b-q4k-v2`)?

**The "broken-fast" `q4k_matvec` was a dispatch bug, not a kernel bug.**
Earlier write-up (preserved in git history) attributed the argmax drift
to `q4k_matvec`'s 32-lane simdgroup reduction tree. **Wrong root cause.**
The actual bug: `MetalBackend::q4k_matvec` (and three sibling sites in
`moe_dispatch.rs` + the non-gated FFN path) hardcoded the 4sg shader's
`THREADS_PER_TG=128` while dispatching the 8sg `q4k_matvec_pipeline`
(production default since 2026-04-28). With only 128 threads dispatched,
simdgroups 4..7 of each 8sg TG never executed — half the rows in each
8-row TG were left unwritten. Same family as the historical 2026-04-26
`077884b` "81–84 tok/s on broken Q4_K dispatch" trap.

**Fix:** dispatch with the actually-bound pipeline's geometry —
`pipeline.rows_per_tg` / `pipeline.threads_per_tg` instead of the static
4sg constants. Once fixed, `q4k_matvec_matches_cpu` parity test passes
on the same shape that previously failed by 182.89.

**Options measured** (Gemma 3 4B v2, M3 Max, quiet GPU, mean of 3 runs):

| Path | lm_head ms | tok/s | Correct? | Bytes read/token |
|---|---|---|---|---|
| **Default: `q4k_matvec` (post-dispatch-fix)** | **1.85** | **83.3** | ✓ "**Paris**" | 327 MB |
| `LARQL_LM_HEAD_SKIP_Q4K=1` → stride-32 Q4_K | 2.98 | 76.0 | ✓ "**Paris**" | 327 MB |
| stride-32 → f16 GEMV (within `_skip_q4k` fallback) | 3.88 | 71.2 | ✓ "**Paris**" | 1.31 GB |
| f32 BLAS fallback (last resort) | (slow) | — | ✓ | 2.62 GB |

**Chosen:** `q4k_matvec` (now correct) first → f16 GEMV / f32 fallback chain.

**Why:**
- `q4k_matvec` is now correct AND the fastest option. After the dispatch
  fix it produces identical top-1 to the CPU reference and runs at
  1.85 ms/tok lm_head. **+8 tok/s end-to-end vs the stride-32 workaround**.
- Stride-32 was the workaround for the dispatch-bug-disguised-as-
  reduction-tree-drift. Now redundant on production paths but kept on
  the `_skip_q4k` fallback chain for vindexes lacking Q4_K lm_head bytes
  and as a diagnostic A/B.
- f16 GEMV remains in the fallback chain only — bandwidth math makes it
  4× more expensive than Q4_K (1.31 GB vs 327 MB), so it never wins on
  throughput when the Q4_K path is healthy. Where f16 matters is **memory
  footprint on 31B models**: the f32 fallback would allocate a 5.6 GB
  clone of the lm_head matrix on load. f16 avoids that one-time setup
  cost. See `f16_gemv_wiring_todo` memo for the original motivation.

**Env vars:**
- `LARQL_LM_HEAD_SKIP_Q4K=1` — diagnostic A/B; routes to
  `lm_head_knn_backend_skip_q4k` (stride-32 first, then f16, then f32).
- `LARQL_LM_HEAD_STRIDE32=0` — only meaningful inside the `_skip_q4k`
  chain; disables stride-32 there too.

(The legacy `LARQL_METAL_LM_HEAD=1` env var was removed 2026-05-02 —
the path it used to enable IS the default now, so the override has no
purpose.)

**Lesson for future kernel bring-ups:** when an "isolated" or "broken-fast"
kernel result looks too good — particularly when the kernel produces
correct output on some prompts but flips on others — **suspect a dispatch
geometry mismatch first** before blaming reduction trees or numerical
precision. Two confirmed instances now (077884b 4-rows-vs-8-rows on Q4_K
dispatch; this 4sg-constants-on-8sg-pipeline). Both signatures: hardcoded
shader-module constants while the bound pipeline has different geometry.
**Always dispatch through `pipeline.rows_per_tg` / `pipeline.threads_per_tg`.**

**Related:**
- `crates/larql-compute/docs/adr/015-isolated-vs-batched-kernel-perf.md`
  — broken-fast pattern; this entry corrects the 4th instance from
  "kernel-level drift" to "dispatch-geometry mismatch."
- `crates/larql-compute/src/metal/trait_impl/quant_matvec.rs::q4k_matvec`
  — fixed dispatch site.
- `crates/larql-compute/src/metal/moe_dispatch.rs` — three sibling sites
  fixed in the same pass.

---

**Recent changes (2026-05-09):**

| Change | Model | Effect | Notes |
|---|---|---|---|
| **D-RMS-FUSE Phase 1 — `LARQL_FUSED_PRELAYER_NORM=1` opt-in** (post-FFN residual_add + next-layer input rms_norm fused via `residual_norm_store`) | Llama 2 7B / Mistral 7B / Qwen / etc. (non-Gemma); no-op on Gemma 3/4 (already triple-fused) | **End-to-end NULL** (within drift). Predicted ~0.2 ms/tok savings; measured 0.0 ± drift on Llama 2 7B and Mistral 7B. | Bit-identical parity preserved across all 3 archs tested. ADR-015 magnitude-compression at the extreme — the dispatch-overhead saving from skipping rms_norm in layer N+1 is offset by the heavier `residual_norm_store` (does the RMS reduction inline) replacing the lighter `residual_add` in layer N. Kept opt-in per ADR-017 retention rules. **Lesson**: dispatch-overhead estimates of ~7 µs/dispatch over-predict savings when the kernel being skipped is also short. See `project_rms_fuse_implemented.md` memory entry. |
| **`make bench-cross-arch`** — multi-arch decode bench script + baselines dir + Makefile target | All archs in matrix (Gemma 3 4B, Gemma 4 31B dense, Gemma 4 26B A4B MoE, Llama 2 7B, Mistral 7B) | Infrastructure (no perf change) | Operationalises ADR-017 model-agnosticity check + surfaces thermal artifacts (`feedback_thermal_perf_artifacts.md`). `scripts/bench-cross-arch.sh` runs `larql bench --output json` per arch; `--save-baseline` captures, `--compare` diffs. Baselines in `bench/baselines/cross-arch/<shorthand>.json`. **Run on a cool machine** before saving baselines. |
| **QKV defuse cross-arch validation** | Gemma 4 26B A4B (60L, hidden=5376, MoE) | **+0.4 tok/s, −2.19 ms GPU fwd** (defused vs fused) | Confirms ADR-016 promotion holds on the larger MoE arch too. Smaller win than 4B (close to noise floor: drift 1.6 ms, signal 2.19 ms) but same direction. No arch-specific dispatch path needed. |
| **QKV defuse — default flipped to `rms_norm` + `q4k_q6k_qkv_proj` (non-fused)** (was: fused `q4k_q6k_qkv_proj_normed`; opt back in via `LARQL_QKV_FUSED=1`) | Gemma 3 4B v2 | **+1.6 tok/s, −0.30 ms/tok GPU fwd** (warmup 8, n=100, drift = 0.02 ms) | The fused kernel rolled the RMS norm into Phase 1 of the matmul to save 1 dispatch/layer (~0.24 ms/tok) but each TG's 4 simdgroups independently re-traverse H + norm_w in Phase 2 with different stride patterns (Q4_K Q/K vs Q6_K V), dropping it from 287 → 199 GB/s vs the non-fused kernel. **Diag predicted −1.22 ms/tok end-to-end; measured −0.22 ms/tok** — direction matched but magnitude was 18% of prediction. Pinned in [ADR-016](docs/adr/016-defused-rms-norm-qkv.md) (defuse decision) and [ADR-015](docs/adr/015-isolated-vs-batched-kernel-perf.md) § "Magnitude can compress 4×" (case study on per-kernel batched diag over-predicting when the candidate's upside is bandwidth headroom that gets reabsorbed by the surrounding LPDDR5X-bound pipeline). Real win nonetheless; correctness preserved (Paris ✓ on `larql run ... -n 8 --metal`). Fused kernel + dispatcher kept reachable as opt-in fallback. |

**Recent changes (2026-05-01 → 2026-05-02):**

| Change | Model | Effect | Notes |
|---|---|---|---|
| **Q4_K O-proj routes through `q4k_matvec_pipeline`** | Gemma 3 4B v2 | **+3–4 tok/s, -0.7 to -0.9 ms GPU fwd** | Decode O-projection was still passing `q4k_proj_pipeline` into the format-aware matvec helper, bypassing the selected 8sg `q4k_matvec_pipeline`. Initial three bench runs after fix: 74.6, 75.6, 75.4 tok/s; follow-up quiet-GPU runs: 76.1, 76.6, 76.3 tok/s; side-by-side steady Ollama: 99.5–100.6 tok/s, 1.30× gap. Correctness smoke: `larql run ... "The Capital of France is" -n 8 --metal` emits Paris. Hybrid decode now uses selected Q4_K/Q6_K KernelHandle geometry too. |
| **`q4k_ffn_gate_up_nr2` candidate** (was opt-in `LARQL_GATE_UP_NR2=1`; **removed 2026-05-09**) | Gemma 3 4B v2 | **REGRESSED** 75.9 → 72.9 tok/s, GPU fwd 11.19 → 11.80 ms (+0.62 ms); re-bench 2026-05-09 confirmed (8sg 86.3 tok/s / 11.71 ms vs NR2 83.4 tok/s / 12.05 ms = **−0.34 ms/tok**) | Profiler showed iso 0.401 ms / 76.8 GB/s vs 8sg's 0.591 ms / 51.4 GB/s — **1.47× isolated win**. But batched: 0.110 ms / 267 GB/s vs 8sg's 0.106 ms / 279 GB/s — NR2 is *worse* in the production geometry. The iso win was dispatch-overhead amortisation that disappears once n_layers calls share one cmd buffer. **Third confirmed instance of the iso-vs-batched pattern** (after `f16_acc` and `attn_fused`); pinned in `docs/adr/015-isolated-vs-batched-kernel-perf.md`. Re-benched 2026-05-09 alongside back-to-back baseline runs (drift = 0.07 ms, well under signal): direction matched the batched diag, magnitude was ~3× the diag delta but the ordering held — so the candidate was deleted rather than left as a dangling opt-in. Shader, pipeline field, env-var, dispatch branch, diag-profile entry, and shader-bench inventory entry all gone. |
| **`LARQL_LM_HEAD_STRIDE32=0` A/B** | Gemma 3 4B v2 | **REGRESSED** 75.9 → 69.5 tok/s, lm_head 2.99 → 4.08 ms (+1.09 ms) | Tested whether the v5 stride-32 lm_head was paying a perf tax for correctness. It is not — disabling it costs +1.09 ms vs the default. The "+0.7 ms cost" line in the v5 row below is relative to the *pre-fix broken-output* kernel (which produced gibberish), not the current fallback path. **The v5 stride-32 lm_head is both correct AND the fastest available path.** The correctness/perf tradeoff is settled; no further A/B needed here. |
| **lm_head v5 stride-32 Q4_K matvec** | Gemma 3 4B v2 | **correctness — model now emits "Paris"** | Each lane accumulates over `i % 32 == lane` elements (mirrors `f16_gemv` reduction tree). Same Q4_K bytes, same bandwidth, but reduction tree matches CPU rankings. End-to-end argmax flips to the correct token. ~0.7 ms slower than the prior (incorrect) kernel; held as the production lm_head path. See `shaders/q4k_matvec_stride32.rs`. |
| **`qk_norm_rope_fused` shader** (default-on; opt-out `LARQL_FUSED_QK_NORM_ROPE=0`) | Gemma 3 4B | -0.10 ms GPU | One TG/head: RMS-norm + RoPE in one kernel. Replaces qk_norm_qk + rope_at_pos_batched_qk. |
| **`kv_append_attend_fused` shader** (default-on; opt-out `LARQL_FUSED_KV_APPEND_ATTEND=0`) | Gemma 3 4B | -0.21 ms GPU | Per-Q-head TG cooperatively writes new K/V row at pos, then standard attention. Absorbs the kv_cache_append dispatch. |
| **`post_attn_residual_norm_store` shader** (default-on; opt-out `LARQL_FUSED_POST_ATTN_NORM=0`) | Gemma 3 4B | cumulative -0.43 ms GPU | Triple fusion on the `has_post_norms` path: post-attn RMS + residual + ffn-norm RMS + h_post_attn store, two sequential RMS reductions in one 1-TG kernel. |
| **`post_ffn_norm_residual_add` shader** (default-on; opt-out `LARQL_FUSED_POST_FFN_NORM=0`) | Gemma 3 4B | cumulative -0.78 ms GPU | 1-TG fused RMS over `down_out` + per-element norm + residual sum into next-layer input. Bit-equivalent to the unfused chain. |
| **`attn_fused` shader** (opt-in only, `LARQL_FUSED_ATTN=1`) | Gemma 3 4B | **REGRESSED** -1.45 ms GPU | Tried merging `qk_norm_rope_fused` + `kv_append_attend_fused` into one kernel (per-Q-head TG normalises+ropes Q+K, writes cache, attends). Standalone qk_norm_rope ran 12 TGs in parallel; the merger collapses to 8 TGs. Dispatch saving (~30 µs) dwarfed by parallelism loss. Kept registered for a future multi-TG-per-head retry. **Lesson saved**: dispatch fusions only win when they don't reduce TG count for an already parallelism-bound stage. |

**Recent changes (2026-04-26 → 2026-04-28):**

| Change | Model | Effect | Notes |
|---|---|---|---|
| **lm_head Q4_K vs Q4_0 dispatch fix** | Gemma 3 4B v2 | correctness — output was gibberish | Writer produced Q4_K, reader dispatched Q4_0 (same byte rate so file size matched). Now dispatches q4k_matvec. |
| **MoE combine helper unification** (CPU + Metal share `outer_combine.rs`) | Gemma 4 26B-A4B | **correctness — was multilingual gibberish** | 4 silent divergences between CPU/Metal MoE combine logic (f32/f64 RMS, identity-scale-on-missing-norm, etc.) collapsed into one helper. Verified via `larql parity --component layer`: 30/30 layers cos=1.0. |
| **Q4_K dispatch correctness fix** (commit 077884b) | Gemma 3 4B | **−5 tok/s** (84 → 79) | Q4_K was routed through Q4_KF kernel, leaving 75% of output rows unwritten; 81-84 was on broken code, 79 is correct baseline |
| **`q6k_matvec` ROWS_PER_TG=4 correctness fix** | Gemma 3 4B | **78.7 tok/s, GPU fwd 10.8ms** | Silent bug: rows 1282-2559 were zeros; fixed to ROWS_PER_TG=4 everywhere |
| **Profiler harness fix** (`measure_single_cmdbuf_batched`) | profiling tool | corrects per-kernel GB/s by 2-4× | Old harness ran each kernel call in its own cmd buffer; per-call dispatch overhead dominated the measurement. Fixed numbers: q6k_matvec 311 GB/s (was 74), q4k_ffn_gate_up 274 GB/s (was 103). |
| **`q4k_matmul` Metal kernel** + parity tests | prefill | kernel 1.79× isolated; **end-to-end falsified twice** | Wiring into O proj + FFN gate+up was attempted and reverted 2026-04-28: short-prompt prefill within noise, long-prompt prefill regressed ~10% (FFN gate+up: 2933 → 3268 ms on 340 tokens). **Re-benched 2026-05-09** under post-dispatch-fix + post-QKV-defuse state to test whether the diagnosis still held: matmul still 5-7% slower across 10/50/150-token prompts (e.g. 1392 → 1469 ms at 150 tokens). Same failure mode as f16 acc — kernel is bandwidth-near-peak and matmul's [seq_len × hidden] X working set thrashes L1 on long prompts. Kernel + parity tests remain shipped (callers can use `MetalBackend::q4k_matmul` directly) but the production prefill path stays per-position matvec. **Closing the prefill gap needs a different matmul kernel** (K-dim tiled or `simdgroup_matrix`-based), not a wiring change. ROADMAP D-PREFILL-MM closed 2026-05-09; falsification record at `project_prefill_matmul_falsified.md`. |
| **Encoder coalescing** in 3 dispatch sites (O proj, QKV f32, QKV Q8) | prefill | <5% on long prompts | Below noise on short prompts. Real win is the matmul kernel above; coalescing was the cheap risk-free first move. |
| **`q4k_ffn_gate_up_f16acc` shader** (opt-in, `LARQL_F16_ACC=1`) | Gemma 3 4B | kernel 1.79× isolated; **end-to-end at parity** | Numerical parity perfect (10-prompt greedy bit-identical), but kernel was already bandwidth-bound — freed ALU cycles get absorbed by surrounding kernels. Initial +23% measurement was thermal-throttle artifact. Kept as opt-in. |
| **`q4k_ffn_gate_up_8sg` shader** (now default; opt-out `LARQL_GATE_UP_8SG=0`) | Gemma 3 4B | **+2.1% end-to-end** (77.2 → 78.9 tok/s) | 8 simdgroups per TG (256 threads, 8 rows/TG) instead of 4/128/4. Same per-thread register footprint (`nr0=1`). Bit-identical output. First positive end-to-end perf this session. |
| **`q6k_matvec_8sg` shader** (opt-in only, `LARQL_Q6K_8SG=1`) | Gemma 3 4B | kernel **1.96× isolated**, end-to-end **at parity** | Q6_K was already at 84% of LPDDR5X peak — too little headroom for 8sg to recover; larger TGs cause schedule contention with 8sg gate+up. Kept opt-in. |
| **`q4k_matvec_8sg` shader** (now default; opt-out `LARQL_Q4K_MATVEC_8SG=0`) | Gemma 3 4B | **+5.2% end-to-end** (76.3 → 80.3 tok/s) | Profiler showed q4k_matvec at 220 GB/s = 55% of LPDDR5X peak (most under-utilised matvec). 8sg gives biggest single-shader win this session — touches Wo + QKV fallback + other call sites, gains compound. Bit-equal parity ✓. |
| **Pattern observation (2026-04-28)**: 8sg geometry helps proportionally to bandwidth headroom: 55% util (q4k_matvec) → +5.2%; 68% util (gate+up) → +2.1%; 84% util (q6k_matvec) → 0% (regressed). When considering 8sg for a new kernel, profile its production-batched GB/s first — only worth it if utilisation is below ~75% of LPDDR5X peak. | | | |
| `f32_gemv_topk1` GPU argmax | any | 0 in bench (KNN fires first) | Saves 0.33ms for top_k=1 non-KNN callers |
| Q4_K float4 dual-sub-block | Gemma 3 4B | **REGRESSED** (reverted) | K=2560 — added addressing overhead |
| Batched MoE prefill | Gemma 4 26B A4B | **+35% tok/s, −31% prefill** | 130 → 26 GPU commits for 5-token prompt |
| Q4_K `sumy` precompute | Gemma 3 4B | neutral (within noise) | Compiler already hoisting; FMA chain unchanged |
| Per-layer Q4K format + GPU expert dispatch | Gemma 4 26B A4B | **+75% overall (2.9 → 5.1 tok/s)** | Expert FFNs on GPU; see §26B A4B below |

### Per-kernel batched throughput (refreshed 2026-05-02)

`diag_shader_bench --profile gemma3`, M3 Max, gemma3-4b-q4k-v2 (warmup 5, iters 30):

| Kernel | Batched ms/call | GB/s | Per-token (×34) | Notes |
|---|---|---|---|---|
| q6k_matvec_active / 4sg (down, K=10240) | 0.069 ms | **312 GB/s** | 2.3 ms | bandwidth-bound, ~84% of LPDDR5X peak; production default |
| q6k_matvec_8sg | 0.069 ms | 311 GB/s | 2.4 ms | tied with 4sg at this granularity; opt-in only |
| q4k_ffn_gate_up_8sg (production gate+up) | 0.107 ms | **275 GB/s** | 3.6 ms | bandwidth-bound; +2.1% end-to-end vs 4sg (not visible at batched-GB/s level) |
| q4k_ffn_gate_up (4sg, original) | 0.107 ms | 276 GB/s | 3.6 ms | statistically tied with 8sg at the per-kernel level — the 8sg promotion was an end-to-end win, not a per-kernel one |
| q4k_ffn_gate_up_f16acc (opt-in) | 0.110 ms | 268 GB/s | 3.7 ms | slower batched; do not promote (ADR-015 instance #1) |
| q4k_ffn_gate_up_coop (opt-in) | 0.119 ms | 248 GB/s | 4.0 ms | slower batched; do not promote |
| q4k_matvec_8sg (Wo, K=8192) | 0.026 ms | 144 GB/s | 0.9 ms | lower util but small per-token cost |
| q4k_q6k_qkv_proj (mixed Q/K Q4_K + V Q6_K) | 0.092 ms | 287 GB/s | 3.1 ms | **production QKV path since 2026-05-09** (paired with separate `rms_norm` dispatch) |
| q4k_q6k_qkv_proj_normed (fused norm + QKV) | 0.135 ms | 194 GB/s | 4.6 ms | rereads H + norm 3× per TG; was production default until 2026-05-09; now opt-in via `LARQL_QKV_FUSED=1` |
| f32_gemv (lm_head, 262K×2560) | 0.866 ms | **387 GB/s** | 0.87 ms (×1) | near LPDDR5X peak; production lm_head uses Q4_K stride32 path |

**No headroom in any single kernel.** The remaining ~1.17× decode gap to ollama is distributed across dispatch overhead + sustained-clock effects + the cumulative inefficiency of running fewer-fused kernels than llama.cpp. (Was 1.30× before the 2026-05-02 dispatch-geometry fix and 2026-05-09 QKV defuse; both moved larql closer to its hardware ceiling.)

**Architecture gap to llama.cpp** is documented in [docs/llama-cpp-comparison.md](docs/llama-cpp-comparison.md): three concrete kernel differences (flash attention, `simdgroup_matrix` prefill matmul, scalar-op RMS-norm pre-fusion) explain the remaining decode gap and the entire 4–14× prefill gap. All three are kernel-research projects (ROADMAP D-ATTN-MTG, D-PREFILL-MM2, D-RMS-FUSE).

**Promotion rule (2026-05-02):** isolated kernel speedups are not promotion
evidence for decode. Promote only when production-batched GB/s improves AND
`larql bench --warmup 8 -n 30 --profile` improves with correct output. False
positives now include `q4k_ffn_gate_up_f16acc`, `attn_fused`,
`q4k_ffn_gate_up_coop`, and the removed `q4k_ffn_gate_up_nr2` (deleted
2026-05-09 after a second-confirming bench). Canonical workflow below.

### How to A/B a shader candidate

Two commands. The save-then-compare flow is the contract for promoting a new
shader to default — it implements the three-step diagnostic pinned in
[ADR-015](docs/adr/015-isolated-vs-batched-kernel-perf.md).

**Step 1 — capture a baseline** (commit `main` or whatever `HEAD` you trust):

```bash
cargo run --release --features metal -p larql-compute --example diag_shader_bench -- \
  --profile gemma3 \
  --json /tmp/larql-shaders-baseline.json
```

**Step 2 — change the shader, then compare:**

```bash
cargo run --release --features metal -p larql-compute --example diag_shader_bench -- \
  --profile gemma3 \
  --compare /tmp/larql-shaders-baseline.json \
  --json /tmp/larql-shaders-current.json \
  --threshold 5
```

Reads `--compare` first, prints per-kernel `improved` / `flat` / `regressed`
(with the threshold percent), then writes the new JSON. `--threshold` defaults
to 5%; tighten for noise-sensitive comparisons.

**Step 3 — end-to-end bench A/B with correctness smoke:**

```bash
./target/release/larql run output/gemma3-4b-q4k-v2.vindex "The capital of France is" -n 8 --metal
./target/release/larql bench output/gemma3-4b-q4k-v2.vindex --warmup 8 -n 30 --profile
```

The bench is the final word; the run output must still emit "Paris".

**When step 2 says regressed or flat, do not run step 3.** Three sessions have
been spent re-confirming that an isolated-only win does not carry — see
ADR-015. The exception is the 8sg geometry pattern: kernels under ~75% of
LPDDR5X peak have headroom to convert isolated wins into batched wins; above
~80% peak the headroom is gone.

---

## Gemma 4 26B A4B — MoE model (2026-04-26, updated 2026-05-02)

Machine: M3 Max, 5-token prompt, 5 warmup / 30 measured tokens  
Vindex: `gemma-4-26B-A4B-it.vindex` (30 layers, 128 experts/layer, top-K=8, inter=704, hidden=2816)

### Progress log

| Optimisation | Decode tok/s | GPU fwd | Δ |
|---|---|---|---|
| BF16 blob baseline | 2.9 | 334ms | — |
| Batched MoE prefill | 3.9 | 246ms | +35% |
| Q4K per-layer format + GPU expert dispatch | 5.1 | ~194ms | +75% from baseline |
| **moe_dispatch geometry fix** (2026-05-02) | **~19.4** | **~52ms** | **+3.8× from prior** |
| GPU-only ceiling (`LARQL_SKIP_MOE=1`) | 56.8 | 15ms | theoretical max |

### What the 2026-05-02 moe_dispatch fix changed

Same root cause as the Gemma 3 4B lm_head fix: three sites in
`metal/moe_dispatch.rs` (per-expert down projection) hardcoded the legacy
4sg `q4k_matvec` shader's `THREADS_PER_TG=128` while dispatching the
`q4k_matvec_pipeline` (bound to the 8sg variant since 2026-04-28).
Per token, that meant:

- 30 MoE layers × top_k=8 = **240 broken expert dispatches**.
- Each dispatched `ceil(hidden/4)` TGs × 128 threads — DOUBLE the TG
  count the 8sg kernel needed, but only the first 4 of 8 simdgroups
  per TG actually ran.
- Net: **2× dispatch overhead × half the work-per-TG = ~140ms/tok of
  wasted GPU time**, plus half the down-projection rows in each TG
  left unwritten (silently degrading output: short truncated
  responses, missed continuations).

Fixed by reading `pipeline.rows_per_tg` / `pipeline.threads_per_tg`
from the bound `KernelHandle` instead of hardcoding shader-module
constants. Output went from "Paris." (truncated) to "1. Paris (France)
2. Berlin (Germany) 3. Rome (Italy)" (coherent, multilingual-capable),
and tok/s went from 5.1 → 19.4.

**The 5.1 tok/s baseline was lying** — it logged as "post Phase 1 GPU
dispatch" as if it were the new floor; it was actually bug-locked. The
prior assumption that "Metal buffer allocation overhead is the
bottleneck" was reading a corrupted measurement: ~140ms of the supposed
194ms GPU-fwd was the broken-dispatch waste, not the buffer allocation.

### Phase 2: pre-allocated scratch buffers — DONE (already shipped, attribution corrected 2026-05-02)

`MoeScratch::new` pre-allocates all expert staging buffers (gate, up,
per-expert down × top_k, activation, output) once per model shape and
caches by `(top_k, hidden, intermediate_size)` on the backend. Per-layer
`gpu_moe_dispatch_with_scratch` calls only memcpy expert bytes into the
existing buffer contents — no `bufs.output(...)` calls in the hot path.
Confirmed by audit: every `bufs.output(...)` in `moe_dispatch.rs` is in
`MoeScratch::new` (one-shot), never per-layer.

The 19.4 tok/s baseline measured 2026-05-02 includes both Phase 2 AND
the dispatch geometry fix from the same day. Pre-2026-05-02 the 5.1
tok/s "Phase 1" headline was Phase 2 *infrastructure was wired* but
the dispatch geometry was bug-locking the perf — the broken-dispatch
2× TG overhead was being attributed to "Metal buffer allocation
overhead" in the prior write-up. Both diagnoses turned out to be
reading the same corrupted measurement.

### Remaining 26B headroom: 19.4 → 56.8 tok/s ceiling

The `LARQL_SKIP_MOE=1` ceiling (56.8 tok/s, 15 ms GPU fwd) is "attention + dense
FFN only" — what 26B would do if the experts cost zero. Current MoE
overhead: 52 - 15 = **37 ms/tok of expert work** spread across 30
layers × top_k=8 = 240 expert dispatches (~155 µs/dispatch) plus 30
per-layer commit/wait syncs.

Real next levers (in rough EV order):

1. **Batched expert dispatch** — fuse the 8 separate gate+up + 8
   activation + 8 down dispatches per layer into one or two batched
   calls with per-expert offsets. Reduces dispatch count from ~24/layer
   to ~3/layer, ~21 saved × 30 layers × ~10 µs = up to 6 ms/tok.
2. **Reduce per-layer sync count** — current pipeline commits + waits
   between attention/dense-FFN and experts so CPU can read `h_post_attn`,
   route, and stage expert weights. Folding the routing into a small
   GPU kernel would let the experts launch on the same cmd buffer.
   ~30 syncs × ~50 µs = ~1.5 ms/tok.
3. **Larger TG geometry for expert matmuls** — each expert is a small
   N=2816 matmul; bigger TGs may amortize dispatch better.

These are real shader work, not the cheap "audit dispatch geometry"
class of fix.

---

## Per-kernel profiling (2026-04-26, M3 Max, Gemma 3 4B shapes)

Run: `cargo run --release --features metal -p larql-compute --example diag_profile_kernels`

Two measurement modes:
- **Isolated**: one commit+wait per call (includes ~20µs GPU spin-up overhead)
- **Batched**: 34 calls per command buffer, single commit+wait (matches real decode pipeline)

| Kernel | Data/layer | Batched GB/s | Batched ms/layer | ms/tok×34L | Bottleneck |
|---|---|---|---|---|---|
| q6k_matvec (FFN down, K=10240) | 21.5 MB | **312 GB/s** | 0.069ms | 2.34ms | bandwidth-bound |
| q4k_ffn_gate_up (gate+up, K=2560) | 29.5 MB | **272 GB/s** | 0.108ms | 3.68ms | **compute-bound** |
| f32_gemv (lm_head, 262K×2560) | 2680 MB | **370 GB/s** | — | 7.4ms | bandwidth-bound (near peak) |

**These two kernels (down + gate+up) account for 6.01ms of the ~11.7ms GPU fwd.**

### Why gate+up is compute-bound

Q4_K at K=2560 has the lowest bytes-per-element ratio (0.5625 B/elem) of any kernel.
The GPU spends more cycles on nibble dequant than waiting for LPDDR5X. Ollama closes
this gap via vectorized `float4` accumulation in their `kernel_mul_mv_q4_K_f32_impl`,
but that kernel assumes a transposed nibble layout (GGUF format: lo=elem b, hi=elem b+32)
incompatible with LARQL's linear layout (lo=elem 2b, hi=elem 2b+1).

### Projected impact of closing each gap

| Gap | Current | Target (Ollama est.) | Savings |
|---|---|---|---|
| q6k_matvec: 312→390 GB/s | 2.34ms | 1.87ms | 0.47ms |
| q4k_ffn_gate_up: 272→390 GB/s | 3.68ms | 2.57ms | 1.11ms |
| lm_head overhead | 2.45ms | ~1.3ms | 1.15ms |
| Dispatch overhead | ~1.87ms | ~1.36ms | 0.51ms |
| **Total projected savings** | | | **~3.24ms** → ~85 tok/s |

---

## llama.cpp / Ollama gap analysis (2026-04-25)

### Bandwidth budget

Gemma 3 4B weight data read per token (34 layers):

| Matrix | Format | Size/layer | Total 34L |
|---|---|---|---|
| Wq (8192×2560) | Q4_K | 11.8 MB | 401 MB |
| Wk (4096×2560) | Q4_K | 5.9 MB | 201 MB |
| Wv (4096×2560) | Q6_K | 8.6 MB | 292 MB |
| Wo (2560×8192) | Q4_K | 11.8 MB | 401 MB |
| W gate+up (10240×2560 ×2) | Q4_K | 29.5 MB | 1003 MB |
| W down (2560×10240) | Q6_K | 21.5 MB | 731 MB |
| **Total** | | **89.1 MB** | **3029 MB** |

Theoretical minimums at M3 Max GPU bandwidth:

| Bandwidth | Min time | Max tok/s |
|---|---|---|
| 400 GB/s (peak) | 7.6ms | 132 |
| 300 GB/s (practical) | 10.1ms | 99 |

Measured effective bandwidth (kernel time only, subtracting dispatch overhead):

| Engine | GPU fwd | Dispatch est. | Kernel time | Eff. BW |
|---|---|---|---|---|
| LARQL | 11.8ms | ~2.4ms (476 dispatches×5µs) | ~9.4ms | ~322 GB/s |
| Ollama | 10.1ms | ~1.4ms (272 dispatches×5µs) | ~8.7ms | ~348 GB/s |

LARQL kernels are at ~322 GB/s vs Ollama's ~348 GB/s — a 8% kernel efficiency
gap. The larger gap (1.33×) is dominated by dispatch overhead.

### Dispatch count gap

LARQL has ~14 dispatches per layer × 34 = **476 dispatches/token** = ~2.4ms overhead.
Ollama groups ops more aggressively: estimated ~8 dispatches/layer × 34 = ~272 dispatches.
Dispatch savings alone: **~1.0ms/token**.

### Three specific things llama.cpp does in Q6_K that we've now partially adopted

Comparing `kernel_mul_mv_q6_K_f32_impl` (llama.cpp) vs `q6k_matvec` (LARQL):

| Technique | llama.cpp | LARQL (post 2026-04-25) | Impact |
|---|---|---|---|
| Inter-superblock interleaving | `ix = tiisg%2` → 2 banks in parallel | ✅ `ix = lane & 1u` | Better DRAM utilization |
| X preloading | `yl[16]` loaded before compute loop | ✅ `xl[16]` preloaded | Hides L2 latency |
| Deferred scaling | `float4 sums` → scale once/group | ✅ `acc += d*sc*(...)` | 4× fewer multiplications |
| TG size | 64 threads (2 rows/TG) | 128 threads (4 rows/TG) | Lower register pressure |
| Block format | GGUF transposed layout | LARQL linear layout | Different algorithms needed |

The format mismatch (LARQL uses linear Q6_K, GGUF uses transposed) means
llama.cpp's exact inner loop can't be ported directly — the element ordering
is different. The inter-superblock interleaving + preload + deferred scale
improvements were adapted to the linear layout.

### What remains

1. **Dispatch overhead** (~1ms): 14→8 dispatches/layer through fusion
   - Fused input norm + QKV projection (saves 34 dispatches)
   - Combined QK-norm Q+K (saves 34 dispatches)
   - Combined RoPE Q+K dispatch (saves 34 dispatches)
   Together: ~102 fewer dispatches = ~0.5ms

2. **Q4_K kernel** (~0.5ms): gate+up (Q4_K, 29.5 MB/layer) runs the old sub-block
   stride kernel. llama.cpp's `kernel_mul_mv_q4_K_f32_impl` uses:
   - 4 parallel block groups (`ix=tiisg/8`, 4 groups at once)
   - `yl[]/yh[]` preloading of X values + `sumy[]` for the min correction
   - `float4 acc1/acc2` vectorized accumulation
   Adapting these to LARQL's GGUF-compatible Q4_K format should close another
   ~0.5ms.

3. **lm_head** (~0.5ms overhead over 1.55ms kernel): async readback + heap
   top-k already reduced the CPU-side cost; GPU-side quantize still CPU-bound.

---

## Optimization history

| Date | Change | Before | After | Delta |
|---|---|---|---|---|
| 2026-04-09 | Full kernel + norm rewrite, Q4_KF, fused ops | 29ms (34 tok/s) | 8.5ms (117 tok/s) | −20ms |
| 2026-04-19 | FFN Q4K + Q6K correctness, decode KV cache | — | 14.7ms (68 tok/s) | baseline |
| 2026-04-25 | `q6k_matvec` 4-element batching (compile-time hi2 shifts) | 14.7ms | 13.7ms | −1.0ms |
| 2026-04-25 | Q6K inter-superblock interleaving + X preload + deferred scale | 13.7ms | 11.8ms | −1.9ms |
| 2026-04-25 | lm_head min-heap top-k (avoids 2MB Vec allocation) | 2.40ms | 2.35ms | −0.05ms |
| 2026-04-25 | Dispatch fusions (QK-norm Q+K, RoPE Q+K, residual_norm_store, normed QKV) | 72ms | ~13ms | +1–2 tok/s |
| 2026-04-26 | `f32_gemv_topk1` GPU argmax (gemv + argmax, 8KB readback vs 1MB) | — | — | 0.33ms/tok for top_k=1 |
| 2026-04-26 | Diagnostic: `diag_profile_kernels` (per-kernel GB/s, isolated+batched) | — | — | tooling |
| 2026-04-26 | **q6k_matvec ROWS_PER_TG=4 correctness fix** (shader+dispatch mismatch; rows 1282-2559 were zeros) | 68-75 tok/s (wrong) | **78.7 tok/s, 10.8ms** | +0.2ms vs wrong fast path; correct output |
| 2026-04-26 | Batched MoE prefill (dispatch_full_pipeline moe_fn callback) | 2.9 tok/s, 334ms | 3.9 tok/s, 246ms | −31% prefill, +35% decode |
| 2026-04-26 | Per-layer Q4K expert format + GPU dispatch (Phase 1) | 3.9 tok/s | **5.1 tok/s, 194ms** | +31% decode; Phase 2 open |

---

## Historical context

```
2026-04-09 — synthetic Q4_KF (random weights):  8.5ms = 117 tok/s (17% FASTER than Ollama)
           The 117 tok/s number used synthetic weights; Q4_KF fast-path doesn't
           fire on production GGUF extracts which use Q6_K for down projection.

2026-04-19 — first real-vindex decode:  ~14.7ms = 67.9 tok/s  (Ollama ~100 tok/s)
           Real model uses Q4_K gate/up + Q6_K down (Ollama convention).
           Q6_K was the bottleneck: 79 GE/s effective vs Q4_K's 105 GE/s.

2026-04-25 — Q6_K rewrite session:  62 → 72 tok/s over three shader iterations.
           Root cause of original gap: runtime hi2 shift + sequential superblock
           access + register pressure from sc_f[16] preload (paradoxically hurt
           by occupancy reduction).
```

---

## Key data points for future work

- M3 Max GPU practical bandwidth: ~300-400 GB/s (system-shared LPDDR5X)
- Ollama effective bandwidth: ~390 GB/s (measured, not estimated — inferred from kernel gap)
- LARQL effective bandwidth: ~315-330 GB/s
- Metal dispatch overhead: ~5µs per `dispatch_thread_groups` call
- Current: 374 dispatches/tok ≈ 1.9ms overhead (vs Ollama ~272 = 1.4ms → 0.5ms gap)
- **Gate+up is ALU-limited at K=2560**: 272 GB/s despite L1-cached input; dequant ops dominate
- **q6k_matvec is bandwidth-limited at K=10240**: 315 GB/s; ROWS_PER_TG=4 (640 TGs × 128 threads, 4 rows/TG, no overlap) is both correct and fast (78.7 tok/s)
- `f32_gemv_topk1` GPU argmax: fires for top_k=1 callers; main decode uses KNN lm_head (top_k=5), so bench gain = 0. Value for non-KNN model paths.
- To close the kernel compute gap: need format-compatible vectorized Q4_K dequant (no solved approach yet)
