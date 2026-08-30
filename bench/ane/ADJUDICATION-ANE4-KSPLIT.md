# ANE-4 — the down_proj k-split. REJECTED as a production lowering.

Banked 2026-08-26, battery regime, coremltools 9.0.
Harness: `bench/ane/ane4_ksplit.py`. Synthetic pseudorandom weights,
one common f64 reference for every arm.

```
    y = W x        W : [5120, 17408]
    W = [W0 W1]    y = W0 x0 + W1 x1     both reductions ANE-admissible
```

---

## The three questions, answered independently

```
arm            piece k    linear dev   sum dev    min ms    p50 ms     rel_rms
full-req-ANE     17408           CPU         -     1.523     1.555   3.486e-03
CPU-full         17408           CPU         -     1.524     1.559   3.486e-03
ANE-split-2       8704       ANE+ANE       ANE     1.603     1.726   1.517e-02
CPU-split-2       8704       CPU+CPU       CPU     2.721     2.779   2.451e-03
ANE-split-4       4352     ANE x 4       ANE x 3   1.649     1.754   1.518e-02
CPU-split-4       4352     CPU x 4       CPU x 3   2.549     2.592   1.384e-03
```

### Placement: the split WORKS

`ANE-split-2` puts **2/2 projections and the sum on the ANE**. The
lowering does what it was designed to do — an ANE-ineligible operation
becomes ANE-resident with exact semantics.

### Latency: and it is still SLOWER than the CPU

```
CPU-full        1.524 ms    baseline (what the FFN tail does today)
ANE-split-2     1.603 ms    1.05x   slower
ANE-split-4     1.649 ms    1.08x   slower
```

**Full ANE residency costs 5–8% more wall-clock than simply running
`down_proj` on the CPU.** The ANE runs this projection at ~111 GB/s
against the CPU's ~117 GB/s on the same bytes, and the partial sum adds a
little on top. There is no latency case for the split.

### Accuracy: and it is 4.4x worse

The CPU-split arm is what makes this readable:

```
CPU-full     3.486e-03   unsplit arithmetic, CPU precision
CPU-split-2  2.451e-03   split arithmetic,  CPU precision   -> SPLITTING alone
ANE-split-2  1.517e-02   split arithmetic,  ANE precision   -> + DEVICE

splitting alone   0.70x   (splitting IMPROVES accuracy)
device on top     6.19x
```

Without that control this would have read as "splitting costs accuracy".
It does the opposite: **splitting alone improves accuracy 1.4x**, because
CPU error grows with reduction depth and the split halves it. Splitting
into four is better still (1.384e-03, 0.40x of unsplit). The entire loss
is the device.

## Two independent confirmations of the flat-floor model

1. **ANE-split-2 and ANE-split-4 are identical**: 1.517e-02 vs 1.518e-02.
   Halving the reduction again changes nothing. The ANE floor does not
   move with reduction depth — measured here by a completely different
   route than the depth ladder.
2. **The CPU arms move exactly as the ladder said they should.** The
   ladder found CPU error growing with k; here, cutting k in half or
   quarters lowers it monotonically (3.486e-3 -> 2.451e-3 -> 1.384e-3).

The mechanism-based prior held for the ANE — no recovery — and the
CPU-split arm caught the effect that would otherwise have been
misattributed to it.

## A pre-registration I got wrong, and what it reveals

I predicted `ANE-split-4`'s 4352-wide pieces would **not** place on the
ANE, since ANE-2A put the f16 preference edge at ~4992 and measured
k=4864 as CPU-preferred.

**They placed on the ANE, 4/4.**

ANE-2A measured single-op graphs. Here the same shape sits in a graph
whose other operations are already ANE-resident, and Core ML places it
there too — presumably because two device transitions would cost more
than the op saves. So:

> **The k <= 16384 admissibility limit is hard. The ~4992 preference edge
> is SOFT and graph-contextual.** A shape below it can still land on the
> ANE when its neighbours are there.

A VINDEX3 capability model must therefore treat admissibility and
preference differently: the first is a per-op constraint that can be
checked in isolation, the second is a whole-graph cost decision that
cannot.

## Verdict: reject the split, keep the mixed plan

```
                        residency   latency        accuracy
ANE gate/up + CPU tail    8/12      4.69 ms        1.27e-03 (block)
full ANE via k-split     12/12      +5-8% worse    6.2x worse
```

**Maximum accelerator residency is not the optimal physical plan.** The
production FFN plan is the mixed one:

> **ANE gate/up -> CPU tail.**

The CPU handles the one operation the ANE is bad at, while the ANE
handles the two largest weight reads — and ANE-3's bubble-recovery result
says the two engines running together deliver 1.20–1.24x what the GPU
alone does. The heterogeneous plan wins on residency-where-it-helps,
latency, and accuracy simultaneously.

That is an evidence-backed plan rather than an ideology of "put
everything on the accelerator".

## What this does NOT license

- Synthetic pseudorandom tensors, not real Qwen weights. Placement and
  latency should carry; the error magnitudes are fixture-dependent.
- f16 only. An int8 k-split was not run, and ANE-2C showed int8 moves the
  preference edge — though not the 16384 admissibility limit.
- The split was measured as an isolated projection, not inside the full
  FFN block. Re-integrating it could change the seam cost, though it
  would have to overturn a 5-8% deficit *and* a 6.2x accuracy gap.
- One session, battery regime.
