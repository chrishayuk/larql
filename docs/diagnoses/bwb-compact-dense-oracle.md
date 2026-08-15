# BW-B — the compiled compact-dense oracle vs. sparse-gather vs. dense

**Date:** 2026-08-14 · **Box:** M3 Max, AC power, `--release`
**Instrument:** `crates/larql-inference/examples/bwb_compact_dense_oracle.rs`
(new) · `qwen3-0.6b-q4k-v2.vindex` (28 layers, hidden=1024,
intermediate=3072, Q4K, retrofitted with the `down_features_q4k.bin`
feature-major sidecar via `larql convert add-feature-major-down`)

## The question

R4 found sparse-gather capturing only 23–40% of its own row-count
reduction, worsening as retained width grew (`docs/diagnoses/
walk-ffn-r4-zeroout.md`). LA-6/LA-7 independently found scattered
magnitude-selected columns touching 29–89× their logical byte count
under block-quantized layouts, then closed the dynamic fine-grained
selection family — while leaving one door explicitly open
(`docs/vindex2-la.md`, branch `worktree-vindex2-la`):

> an offline-compiled, cross-input-STABLE contiguous representation —
> neither experiment tested this.

BW-B tests it. Three arms, same oracle mask, same input, same
underlying Q4K bytes:

1. **dense** — `WalkFfnConfig::dense`, routes to
   `interleaved_kquant:native` (`kquant_matmul_transb` over the full
   layer — the production ceiling).
2. **gather** — `WalkFfnConfig::sparse(..).with_pool_per_layer(mask)
   .with_precomputed_routing(true)`, routes to `sparse:gather_q4k` —
   gathers the mask's Q4K rows into contiguous buffers **fresh on every
   call**. This is what R4 measured; BW-B does not attempt to reproduce
   R4's exact numbers on a different model, it reproduces R4's SHAPE as
   a control that the instrument and methodology are sound.
3. **compact** — `CompactDenseLayer::materialize` (new,
   `crates/larql-inference/src/vindex/walk_ffn/compact_dense.rs`)
   gathers the SAME mask's rows **once**, outside the timed loop.
   `WalkFfn::compact_dense_forward` then runs `score_and_accumulate`,
   the identical fused Q4K row-dot / scaled-add kernel
   `gather_q4k_accumulate` uses, with zero gather cost paid per call.

Arms 2 and 3 share a refactored-out `gather_kquant_rows` (byte
selection) and `score_and_accumulate` (the kernel) — pinned bit-exact
by `compact_dense_forward_matches_gather_q4k_accumulate_bit_exact`
(`compact_dense.rs` tests). A measured gap between them can therefore
only be the **timing** of the gather — never a different kernel, a
different byte selection, or a different quantisation path. R4's own
naive "dequant the gather then call BLAS" variant already lost (0.12×,
alloc-dominated, `examples/walk_ffn_gather_gemm.rs`), so the open
question was never "can any compact form win" — it was specifically
"does paying the gather cost every call explain the loss, once that
cost is taken out of the critical path."

## Method

- Oracle mask: the REAL top-K feature ranking `WalkFfnConfig::sparse(..,
  K_MAX=2048)`'s production `FeatureSelector::GateOnly` selects — real
  Q4K gate rows, real row-dot kernel, captured via `WalkFfn::with_trace`
  + `take_runtime_trace` on a direct `WalkFfn::forward` call per layer.
  Nested: the top-K prefix at any K ≤ 2048 is a real top-K route.
- Input: a fixed, deterministic, per-layer synthetic vector, not a
  residual captured from a live `generate()` pass — `larql-inference`'s
  CPU reference attention pipeline (`predict_with_ffn_trace`) expects
  full f32 attention weights this Q4K-only-loaded vindex doesn't carry
  (confirmed: `run_layer_with_ffn` returned `None` for every layer,
  empty dispatch/runtime traces, before the harness settled on calling
  `WalkFfn::forward` directly). This does not affect bytes or wall time
  — Q4K row-dot / scaled-add kernels are data-value-independent — only
  which features the real gate weights rank highest for that
  direction, which is what a non-trivial oracle mask needs, not what
  generation would have selected. Fixing the CPU attention pipeline for
  Q4K-only loads is out of BW-B's scope.
- All 28 layers, K ∈ {256, 512, 1024, 1536, 2048} (8.3%–66.7% of
  intermediate=3072). 2 warmup + 7 timed blocks of 20 calls per (layer,
  K, arm) cell; the block MEDIAN is kept, arms run block-interleaved
  (R4's "paired/interleaved arm rotation"). A dispatch-trace guard
  confirms `gather` actually lands on `sparse:gather_q4k` every cell —
  0 fallbacks across 28 × 5 = 140 cells.
- Bytes and the roofline join reuse `larql_compute::movement_ledger`
  (BW10) directly, declared against the CPU-cluster attainable DRAM
  roofline (127 GB/s, `docs/diagnoses/memory-bandwidth-roofline.md`) —
  distinct from BW-A's 367 GB/s GPU figure; this is CPU kernel work.

## Result — mean over all 28 layers

| K | frac | phys bytes | dense | gather | compact | gather/dense | compact/dense | compact/gather |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 8.3% | 509,952 | 584.0µs | 287.8µs | 228.7µs | **2.03×** faster | **2.55×** faster | 1.26× faster |
| 512 | 16.7% | 1,019,904 | 586.1µs | 388.6µs | 286.8µs | **1.51×** faster | **2.04×** faster | 1.35× faster |
| 1024 | 33.3% | 2,039,808 | 583.1µs | 624.0µs | 382.6µs | 0.93× (7% **slower**) | **1.52×** faster | 1.63× faster |
| 1536 | 50.0% | 3,059,712 | 584.3µs | 857.5µs | 481.0µs | 0.68× (47% **slower**) | **1.22×** faster | 1.78× faster |
| 2048 | 66.7% | 4,079,616 | 582.5µs | 1131.4µs | 558.4µs | 0.51× (94% **slower**) | **1.04×** faster | 2.02× faster |

(ratio columns: dense-time / arm-time — >1 means the arm beats dense)

**gather reproduces R4's crossover.** Faster than dense below ~20%
retained, crosses over to SLOWER between K=512 (16.7%, still 1.51×
faster) and K=1024 (33.3%, already 7% slower), and keeps degrading —
94% slower than dense at 66.7% retained. R4 found the same qualitative
shape on Gemma3-4B (faster at 20%, 0.837× at 40%, 0.670× at 60%): a
different model, a different crossover point, the SAME signature. This
is the control — it says the instrument and methodology are sound
before trusting what they say about the new arm.

**compact never crosses over, anywhere in the tested range.** Still
1.04× FASTER than dense at K=2048 (66.7% of the layer) — the point
where gather is already 94% slower. Byte-for-byte identical to gather
at every K, compact's advantage over gather widens monotonically with
K: 1.26× → 1.35× → 1.63× → 1.78× → 2.02× faster, because compact's
gather cost is paid once (outside every timed cell) while gather's
grows with K inside every one.

**η explains it, not just describes it.** Roofline utilisation at
K=2048: dense η=0.083 (10.5 GB/s), gather η=0.028 (3.6 GB/s), compact
η=0.058 (7.3 GB/s) — compact's kernel streams at ~2× gather's effective
rate and ~70% of dense's own rate, over a matrix 33% the size. The same
fused kernel approaches dense's own streaming efficiency once the
gather cost is removed from the timed window — it does not need a
different kernel to do it.

## Conclusion

> **The problem was representation, not sparsity.**

R4/LA-6/LA-7 correctly killed *dynamic per-call* fine-grained selection
— the family that pays a gather (or scattered-touch) cost inside the
critical path can't win once retained width crosses roughly a third of
the layer, on this evidence as on Gemma3-4B's. But an **offline-compiled,
cross-call-stable** compact representation of the identical oracle
selection is a different animal: it wins across the entire tested range,
and the margin over the dynamic-gather kernel *grows* with K rather than
shrinking.

This reopens exactly the door LA-6/LA-7 left open, and closes it in the
affirmative: **if the useful subset is known and stable, materialising
it into a genuinely contiguous representation is a real kernel-level
win, distinct from and larger than the win logical sparsity alone would
predict.** The unresolved half — whether a REAL selector can produce a
mask stable enough, for long enough, to amortise the one-time
materialise cost in a live decode loop — is exactly what a VINDEX3-
shaped "materialise a derived representation when a usage pattern
proves stable" mechanism would need to answer, and is not yet tested
here (this harness compiles the mask once and reuses it for every
timed call, deliberately isolating the kernel-cost question from the
stability/staleness question).

## What this does NOT show

- **Not accuracy.** No output-correctness scoring here — `compact` and
  `gather` are pinned bit-exact against each other by construction
  (same kernel, same bytes); neither is scored against the dense
  reference's actual numerical output. The oracle mask's *routing
  quality* (does top-K-by-real-gate-score actually preserve generation
  behaviour) is R4/LA-6's question, already answered elsewhere, not
  re-litigated here.
- **Not mask-staleness cost.** `materialize` is called once per (layer,
  K) and reused for every timed call — this measures the kernel win
  GIVEN a stable mask, not how often a real decode loop's mask would
  need re-materialising, nor what that re-materialise costs amortised
  over a realistic reuse window.
- **Not this specific model's number, generalised.** qwen3-0.6b
  (hidden=1024, intermediate=3072) is not Gemma3-4B (R4's model,
  hidden=3072, intermediate=10240). The CROSSOVER SHAPE reproduces; the
  exact crossover K and the exact compact/gather ratio should be
  expected to shift with model dims — re-run on a larger model before
  citing an absolute number as a production estimate.

## BW-B1 — materialization/break-even closure

**Date:** 2026-08-14 · **Instrument:**
`crates/larql-inference/examples/bwb1_materialize_breakeven.rs` (new)

BW-B measured the compact kernel's per-call cost GIVEN an
already-materialized layer. The one question that decides whether the
result is operationally useful: how many reuses does `materialize +
N×compact` need to beat `N×dense` and `N×gather`?

```text
N* = T_materialize / (T_dense - T_compact)     (break-even vs dense)
N* = T_materialize / (T_gather - T_compact)    (break-even vs gather)
```

Bounded to 3 K points (BW-B's headline numbers: 8.3%/33.3%/66.7% of
intermediate=3072), validated empirically at N ∈ {1,2,4,8,16,32}, plus
one realistic-control arm. **Methodology note**: an early run showed
`T_dense` varying with K (1720→1225→617µs), which is impossible — dense
reads the same bytes regardless of what K other arms use. Root cause:
insufficient per-layer warmup let the first K processed in each layer's
loop pay a cold-cache/CPU-clock-ramp tax the later K's didn't. Fixed
with an 8-call throwaway warmup pass per layer before any timed cell;
re-run confirmed `T_dense` flat (589.6/590.6/588.2µs) and consistent
with BW-B's own independently-measured 582–586µs — banked only after
that check passed.

| K (retained) | T_materialize | N* vs dense | N* vs gather |
|---:|---:|---:|---:|
| 256 (8.3%) | 20.1µs | 0.06 | 0.31 |
| 1024 (33.3%) | 80.3µs | 0.38–0.39 | 0.34–0.38 |
| 2048 (66.7%) | 274.7µs | ~9.2 | 0.50 |

At K=256 and K=1024, N* < 1: `materialize + 1×compact` already beats a
single dense (and gather) call — the "compile" overhead is small enough
relative to the per-call saving that you don't need multi-token reuse
to win, a materialize-once-use-once pattern already pays for itself. At
K=2048, compact still beats gather from the first call (N*=0.50) but
needs ~9–16 genuine reuses to beat dense, because the per-call margin
over dense narrows sharply as K approaches the point BW-B already
showed gather crossing over (dense's own kernel is close to its
streaming ceiling here, so there's less room left to beat).

**Realistic-control arm**: a 32-position smooth-drift synthetic
trajectory (NOT a captured real `generate()` trace — same CPU-attention
limitation as BW-B; disclosed proxy, deliberately smooth rather than
i.i.d. noise, since a real residual stream drifts incrementally, it
doesn't jump), K=1024, 7 layers sampled (every 4th of 28):

- consecutive-position mask Jaccard overlap: mean 0.996, min 0.992
- mean run length before ANY feature swap (strict): 1.06 positions
- mean run length before >5% mask drift (tolerant, Jaccard≥0.95): 32.00
  positions — never dropped below tolerance across the whole trajectory

Since even the STRICT run length (1.06) exceeds N* (0.34–0.39) at
K=1024, this is **DYNAMIC-VIABLE** at that K under either reading: real
selector masks don't need to survive more than about one position on
average for materialize-once-use-once to already be worth it. At
K=2048 the picture would need genuine multi-position mask stability
(N*≈9) to clear break-even against dense specifically — not measured
here (the control arm ran at K=1024 only, the cleanest point, per the
brief's "keep it bounded").

**Separately preserved as its own fact, independent of anything above**:
even if a live decode mechanism later proves too expensive to
amortise, BW-B already falsified the claim that fine-grained logical
sparsity intrinsically loses to dense execution. The R4/LA-6 failure
was a physical-layout/gather-timing problem, not a sparsity problem —
that stands regardless of BW-B1's answer.

**Caveat**: the realistic-control arm's mask-stability numbers are a
property of the SYNTHETIC smooth-drift trajectory, not a captured real
decode. A real trajectory could churn faster (if real per-token
residual deltas are larger than this proxy's 15%-over-32-positions
drift) or slower. Re-run against a real captured trajectory once the
CPU attention pipeline supports Q4K-only loads before treating the
DYNAMIC-VIABLE verdict as load-bearing for a production design.

## Not yet done

- Re-run BW-B and BW-B1 on Gemma3-4B directly for a head-to-head R4
  number, once the CPU reference attention pipeline is fixed for
  Q4K-only loads (or the harness is ported to drive real per-layer
  residuals some other way) — would also let the realistic-control
  arm run against an actually-captured trajectory instead of a
  synthetic proxy.
- BW-C (whole-expert compute-skip oracle), BW-D (permutation-aligned
  expert redundancy), BW-E (residency horizon measured against the
  ledger) remain open from the registered BW programme.
