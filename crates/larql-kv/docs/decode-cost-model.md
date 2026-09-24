# Decode Cost Model — what actually limits each KV engine

**Status:** 📊 Results v0.1 (2026-08-03). Registry: `larql/kvperf-1`.
**Audience:** anyone choosing an engine, or optimising one.
**Scope:** CPU decode throughput vs context length, all nine `EngineKind`
variants. Says nothing about accuracy — see
[`state-policy.md`](state-policy.md) for what each engine treats as canonical.

---

## 1. The measurement design is the first result

The engines differ almost entirely in **how K/V is materialised for attention
each decode step** — read from a cache, recomputed from residuals,
decompressed, windowed, or re-forwarded. That is an *O(context)* term.

A fixed short prompt therefore cannot rank them. At ctx≈7 every engine is
pinned to its fixed per-step cost and they land within noise of each other,
which is exactly what the default `larql bench` prompt produces. Any
engine comparison taken at a short prompt is measuring the intercept and
calling it the engine.

**Sweep context, or do not compare engines.**

## 2. Method

```text
machine     Apple M3 Max, 12 P + 4 E, 128 GB, AC power
            no thermal / performance / CPU-power warning recorded
backend     cpu (BLAS + C Q4 kernel), no concurrent build or benchmark
model       qwen3-0.6b-q4k.vindex
            28 layers, hidden 1024, 16 q-heads / 8 kv-heads, head_dim 128
            → kv_dim 1024, GQA reps = 2, vocab 151936
command     larql bench <vindex> --cpu -n 8 --warmup 2 --prompt <N×9 words>
            --engine <list>
contexts    ≈60, 260, 1050, 2100 tokens
statistics  mean and p50 both kept — mean carries amortised periodic work
            (buffer doubling, window close, cold-tier encode), p50 the
            steady state
```

Page cache warmed before measurement. A cold cache on a large model inflates
the *first* engine in a run by an order of magnitude; an earlier 26B run
showed `standard` at 503 ms/token cold and 31.8 ms warm. Warm every model
before believing any number.

## 3. Results — slope is the engine's real cost

Linear fit over the four context points, dense model:

| engine | µs / ctx-token (p50) | R² | intercept | µs / ctx-token (mean) |
|---|---|---|---|---|
| turbo-quant | 7.72 | 1.000 | 5.30 ms | 8.48 |
| markov-rs | 8.08 | 0.998 | 4.34 ms | 8.57 |
| boundary-per-layer | 8.12 | 0.996 | 4.35 ms | 8.74 |
| markov-rs-codec | 8.20 | 0.999 | 4.25 ms | 8.37 |
| standard | 8.23 | 1.000 | 4.18 ms | 8.35 |
| windowed-checkpoint (w=256) | 8.30 | 0.999 | 4.26 ms | 8.78 |
| no-cache | 11581 | 0.996 | — | 11584 |

**Every cached engine has the same marginal cost per context token** — 7.72 to
8.30 µs, a 7.5% spread across six mechanisms that could hardly be more
different. The representation choice buys nothing at the margin. What the
engines actually differ in is **intercept** (4.18–5.30 ms) and, at specific
context lengths, **variance**.

`no-cache` is 1400× the slope of the others: 11.58 ms per context token is a
full forward pass per token, which is its documented design (O(N²) overall,
correctness fallback only). Cross-check: prefill at ctx=2100 measured 23875 ms
= 11.4 ms/token, and its decode step at that context measured 24144 ms ≈ one
whole prefill. The two agree, so the number is the engine, not the harness.

## 4. Where the time goes

Per-stage split of the reference forward, ctx=60 → ctx=1050:

```text
lm_head     7.49 ms  →   7.20 ms      constant
CPU fwd     5.26 ms  →  16.06 ms      +10.9 µs / ctx-token
```

lm_head is a large **fixed** cost, and the whole of the context dependence
lives in the forward — i.e. attention over K/V. At short context lm_head is
59% of the step; by ctx=1050 it is 31%. Engines that swap the full vocab
matmul for a KNN lm_head win a flat ~5 ms and nothing else; that win does not
scale with context and is invisible in the slope.

> **Measurement note (added after this fit).** The intercepts in §3 are
> **forward-only**: at the time of the sweep the engine harness stopped its
> timer before `pick_next`, so lm_head sat outside the measured step while the
> reference row included it. That is why the engine intercepts (4.18-5.30 ms)
> sit close to the reference's *forward* stage (5.26 ms) rather than to its
> full step (~12.8 ms). The slopes are unaffected — lm_head is constant in
> context, so it cancels out of a marginal fit, and §3's conclusion stands
> unchanged. The absolute intercepts and any engine-vs-reference tok/s
> comparison from that era do not: they omit ~5-7 ms of real per-token work.
> The harness now times the whole token and reports the `fwd=` / `head=`
> split per row, so a re-run of this sweep will show intercepts roughly one
> lm_head higher. Re-fitting is not required to trust §3.

### The K/V read is at ~44% of attainable bandwidth

```text
K/V bytes per context token = 28 layers × 1024 kv_dim × 2 (K,V) × 4 B (f32)
                            = 229 376 B = 0.229 MB
standard slope              = 8.23 µs
                            → 27.9 GB/s on logical bytes
                            → 55.7 GB/s counting the GQA reps=2 re-read
attainable (this machine)   ≈ 127 GB/s
```

The decode attention kernel reads K/V **head-minor**:

```rust
// attention/decode/gqa_step.rs
let k_block = k_full.slice(s![.., kv_off..kv_off + head_dim]);
let raw = k_block.dot(&q_row);
```

`k_full` is `[L, kv_dim]`, so each head's gemv reads a 512 B window with a
4096 B stride, and because `reps = 2` every KV head's block is streamed twice
— once per q-head sharing it.

**Hypothesis (untested):** a head-major layout, `[kv_head][L][head_dim]`,
would make each head's read contiguous *and* let the `reps` q-heads share one
pass, roughly halving the O(context) term that dominates every engine.
This is the single highest-leverage change the data points at, because it is
below all six engines rather than inside one.

### turbo-quant: a reduction is a kernel claim

turbo-quant stores K/V **8× smaller** (4-bit WHT + Lloyd-Max vs f32) and gets
the *same* slope (7.72 vs 8.23) plus the **worst intercept** (5.30 ms, +27%
over standard). The bandwidth saving is entirely consumed by decompression:
it decompresses the full prior K/V, every layer, every step, and then runs the
identical attention over the identical f32 bytes. Compression that must be
undone before use is not a bandwidth reduction.

### Variance is transient, not a standing tax

At ctx=1050 the three residual-canonical engines showed mean 19–24% above p50
(markov-rs 15.75 vs 13.24; codec 16.46 vs 13.22; boundary-per-layer 16.05 vs
13.56). At ctx=2100 the gap collapsed to 3–6%. So it is a transient at
particular context lengths, not a persistent cost — the doubling-capacity
buffers in `helpers::append_row` are the obvious suspect, but this is **not
established** and the sample (8 steps) is small.

## 5. Defect found and fixed: `windowed-checkpoint` did not window its attention

**Fixed 2026-08-03** — the diagnosis is kept in full because the reasoning is
reusable and because the fix had to invert an earlier one.

`windowed-checkpoint:window=N` reported `window=N` from `info()` and paid
full-context attention cost.

Evidence:

1. Its slope matches `standard` across all four context points (8.30 vs 8.23).
2. `window=32`, `256` and `4096` at ctx=1050 are indistinguishable in decode
   cost; a 32-row window should cost ~1/33 of the attention.
3. `LARQL_W10_DISABLE=1` restores the engine-side shadow (`hot` 0.0 MB → 5.5
   MB, i.e. genuinely 32 rows) and the cost does **not** change (13.42 →
   13.13 ms).
4. `decode_step_via_dispatch` calls `coarse_decode_step_with_state_masked`,
   whose trait signature has **no window parameter**, against
   `self.kv_handle` — a backend cache spanning the whole stream — and
   `windowed_checkpoint/dispatch.rs` never clips that handle. `window_size` is
   used only to segment the prompt into archived windows and to size the
   engine-side shadow.
5. `StandardEngine::prefill_quant` documents this exact hazard and guards
   against it:

   > the coarse trait surface has no window parameter, so the coarse path
   > always attends over the FULL context. A windowed engine must therefore
   > decline coarse and take the per-layer path … otherwise the same CLI flag
   > gives windowed behaviour on one backend and full-context on another
   > while `info()` reports `window=N`.

`StandardEngine` declines coarse when windowed. `WindowedCheckpointEngine`,
whose entire identity *is* the window, does not.

> **Superseded 2026-08-04.** The quoted guard no longer exists. Declining the
> fused path was always the expensive half of this fix — on a host-delegating
> backend the per-layer route runs the whole forward on the CPU, measured at
> ~9x on Gemma 3 4B — so the trait grew the window instead:
> `coarse_prefill_windowed` / `coarse_decode_step_windowed`, which fail closed
> when a backend cannot bound both attention and K/V. `StandardEngine` now asks
> rather than declines, and both `CpuBackend` and `MetalBackend` answer. The
> correctness reasoning below stands unchanged — an engine must not report a
> window it does not enforce; only the remedy moved from "refuse the fast path"
> to "make the fast path honour it".

This was a correctness finding before it was a performance one: on the dispatch
path the engine was not the engine it reported being, its boundary checkpoints
and archive were maintained but unused for attention, and any accuracy measured
on that path was not the windowed engine's accuracy.

The obvious fix was `StandardEngine`'s — decline the coarse path when windowed
— but that would have cost this engine the fast path entirely, since it is
*always* windowed. Clipping the handle keeps both.

### The fix

Two parts, at the layer each belongs to.

`CpuBackend::clip_kv` only understood the per-layer `CpuKvHandle`, not the
coarse `CpuQ4kCacheHandle` the dispatch path allocates — so a windowed engine
on that path could not clip even if it tried (it panicked as a "foreign
handle"). It now handles both shapes.

`WindowedCheckpointEngine` now clips the backend handle to the window: after
prefill, down to the open window plus the boundary row; after each auto-close,
down to the boundary row alone. `clip_kv` keeps the *tail*, which is exactly
the row the checkpoint was taken from — so the dispatch path now reproduces
what `extend_current` does on the per-layer path when it seeds a fresh window
from `checkpoints.load`.

One coupling had to move with it. `close_window` under HOnly read the boundary
row back from the handle by **absolute** stream position — correct only while
the handle spanned the stream, and itself a fix for an older bug where a
window-relative read re-checkpointed the *first* window's row every time. With
the handle clipped, absolute runs off the end and the boundary row is simply
the tail. The two tests that pinned the absolute indexing now pin the claim
both mechanisms share: distinct windows checkpoint distinct rows, each recorded
at its own absolute position.

### Verification

The write-up's own prediction was that a correctly windowed engine shows a
near-zero slope, and that this is both the fix and its test. Measured at
`window=64`:

| engine | ctx 60 | ctx 260 | ctx 1050 | slope |
|---|---|---|---|---|
| windowed-checkpoint (fixed) | 4.63 ms | 5.21 ms | **5.01 ms** | **~0.4 µs** |
| standard | 4.53 ms | 6.29 ms | 12.97 ms | 8.5 µs |

Flat, and **2.6× faster than `standard`** at ctx=1050 — the saving the window
was supposed to deliver all along. Pinned by two regression tests in
`windowed_checkpoint/dispatch.rs` that assert the backend cache never exceeds
`window_size + 1` rows, through prefill and across repeated auto-closes.

### The siblings are not defective

`markov-rs:window=64` (slope 9.0 µs) and `boundary-per-layer:window=64` (24 µs)
both track or exceed `standard`, but that is their contract, not a bug: they
retain a **cold tier** and attend over the full history, so their window bounds
hot-tier memory rather than attention. They are exact-under-contract engines.
`windowed-checkpoint` was the only one claiming to be lossy beyond its window,
which is why it was the only one whose attention had to be bounded.

Worth noting separately: windowed `boundary-per-layer` costs **2.2× `standard`**
at ctx=1050 (28.4 ms vs 13.0) and 35% more prefill, because its per-layer
encoded cold tier is decoded every step. That is a cost, not a defect, but it
is large enough that the engine should not be reached for casually.

## 6. What this says about engine choice

```text
short context (< ~200 tokens)   every cached engine is within noise;
                                choose on memory and accuracy, not speed
long context                    every cached engine costs the same per token;
                                choose on memory and accuracy, not speed
never                           no-cache, outside correctness debugging
```

The uncomfortable summary is that on CPU **none of the K/V representations
currently buys decode throughput**. They buy memory (turbo-quant 8×,
markov-rs's residual store, windowed-checkpoint's checkpoints) and they differ
in accuracy contract. The shared bottleneck is one layer below all of them, in
how the attention kernel streams K/V — which is where optimisation effort
should go.

## 7. Gaps in the instrument

- `EngineProfiler` covers four of the nine engines of the time (ten now) (markov-rs, markov-rs-codec,
  turbo-quant, windowed-checkpoint). `standard`, `no-cache`, `boundary-kv`,
  `boundary-per-layer` and `apollo` have no per-stage split, so their costs
  are inferred from slope rather than attributed. `bench --profile` says
  "markov-rs only for now" and in practice prints the *reference* forward's
  split, not the engine's.
- Single machine, single dense model, CPU only. The 26B MoE spot-check showed
  the same *ordering* but was taken at ctx≈7 and so measures intercepts only.
- 8 decode steps per point. Enough for p50 on a linear fit (R² ≥ 0.996), not
  enough to characterise the tail.
