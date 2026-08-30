# ANE-4 step 1 — does a realistic Qwen FFN block stay ANE-resident?

Banked 2026-08-25, battery regime, coremltools 9.0. Synthetic
pseudorandom weights, no real Qwen tensors yet. Harness:
`bench/ane/ane4_ffn_block.py`.

Block: `RMSNorm -> gate/up -> SiLU(gate)*up -> down -> residual add`.

```
A   intermediate 17408   down_proj k = 17408   INELIGIBLE (real Qwen)
B   intermediate 16384   down_proj k = 16384   ELIGIBLE   (control)
```

---

## Verdict: the useful outcome. The block does NOT collapse.

```
                        A (k=17408)        B (k=16384)
rms_square  mul              ANE                ANE
rms_mean    reduce_mean      ANE                ANE
rms_eps     add              ANE                ANE
rms_rsqrt   rsqrt            ANE                ANE
rms_scale   mul              ANE                ANE
rms_weight  mul              ANE                ANE
gate_proj   linear           ANE                ANE
up_proj     linear           ANE                ANE
silu                         CPU  (sup: CPU)    ANE
gate_up_mul mul              CPU                ANE
down_proj   linear           CPU  (sup: CPU)    ANE
residual    add              CPU                ANE
                          8 / 12             12 / 12
```

`gate_proj` and `up_proj` — the two biggest weight reads in the block —
**stay on the ANE in the real Qwen shape.** The fracture is confined to
the tail.

## The A/B control fired cleanly

B differs from A only in `down_proj`'s eligibility, and B is fully
resident where A is 8/12. **That is causal evidence that the k > 16384
rule is what fractures the graph**, not some unrelated limitation on
norms or elementwise ops.

## Two findings the isolated benchmarks could not have produced

### 1. The 16384 limit is per-op and applies to tensor WIDTH, not only to a linear's reduction depth

In A, `silu` on a 17408-wide tensor reports `supported: [CPU]` — the ANE
is absent, exactly as `down_proj` is. In B at 16384 the same op reports
`CPU+ANE`.

Yet `gate_proj` (5120 -> 17408) is ANE-**preferred** in A. So:

```
linear   may PRODUCE a 17408-wide output on ANE,
         but may not REDUCE over k > 16384
silu     may not even CONSUME a 17408-wide tensor
```

ANE-2A framed the rule as "reduction depth k" because k was what varied.
The sharper statement is that **16384 is a per-op tensor-dimension limit,
and different ops apply it to different axes.** A VINDEX3 backend
capability model must therefore be per-operator, not one global rule.

### 2. The fracture propagates to ANE-capable neighbours

`gate_up_mul` and `residual_add` are `CPU+ANE`-supported in A and are
still placed on **CPU**. Core ML pulled them across to sit with the ops
that had no choice, rather than pay two device transitions. So an
ineligible op costs more residency than itself — here, two extra ops.

## The fracture is cheap

A's latency is almost exactly the sum of its parts:

```
gate + up on ANE   356.5 MB at ~112 GB/s   =  3.18 ms
down on CPU        178.3 MB at ~127 GB/s   =  1.40 ms
                                    predicted 4.58 ms
                                     observed 4.69 ms
```

**~0.11 ms, about 2.4%, is unexplained — that is the transfer and
synchronisation cost of splitting the block across two engines.** Not
ruinous. In equivalent-rate terms A runs at 114.0 GB/s against B's
121.7 GB/s: full residency buys ~7%.

Both figures sit right on ANE-1's isolated 111.9 GB/s, so a whole block
runs at about the rate a lone projection does.

**This licenses a real heterogeneous FFN physical plan.**

## The finding that cuts the other way: full ANE residency costs ACCURACY

> **SUPERSEDED IN PART — see `ADJUDICATION-ANE4-ACCURACY.md`.** The
> *conclusion* below (ANE residency costs accuracy) was confirmed by a
> proper control: one artifact loaded twice, differing only in
> `computeUnits`, gives **110.6x**, not 26x. But the *attribution* below
> was uncontrolled — A and B differ in device and in every matrix — and
> the *mechanism* named below is **refuted**: ANE error is FLAT in
> reduction depth (0.97x across a 3.2x increase in k), so it is a fixed
> precision floor of ~1.5e-2, not fp16 accumulation. Read that file.

```
A   8/12 on ANE, down_proj on CPU    rel_rms 1.272e-03
B  12/12 on ANE                      rel_rms 3.302e-02      26x worse
```

Both are measured against the same f32 reference computed from the same
f16 weights, by the same code. The two variants differ in intermediate
size by 6%, which cannot explain a 26x error difference — the difference
is **where `down_proj` ran**. fp16 accumulation across a 16384-long
reduction on the ANE loses roughly a decimal digit and a half relative to
the CPU's wider accumulator.

Consequences:

- **For ANE-5 / target-side work this is a real cost.** Moving target
  projections onto the ANE would degrade the model's output, not merely
  its speed. Any such plan needs a quality gate, in predictive units, of
  the kind CPU-5 used.
- **For ANE-4's drafter this may be perfectly acceptable** — a drafter
  only proposes tokens, and wrong proposals are rejected by the target
  rather than emitted. But "acceptable" must be *measured* as an
  acceptance rate, not assumed.
- It also reframes the `down_proj` k-split: splitting would move
  `down_proj` onto the ANE and therefore **into the less accurate
  regime**. The split is not purely a capability win.

## What this does NOT license

- Synthetic weights, not Qwen's. Placement is structural so it should
  carry, but the accuracy numbers are fixture-dependent and real weight
  distributions may behave differently.
- One block, not a stack. Nothing here says how per-block transfer costs
  compose across 64 layers, and the 2.4% could be per-block.
- No attention or GatedDelta block yet — the FFN is the easy half.
- One session, battery regime.

## Next

The natural graph fractured exactly where predicted, so the k-split
experiment is now clean to run and interpret:

```
down_proj 17408 -> 5120   lower to   8704 -> 5120
                                     8704 -> 5120
                                     partial sum
```

Question: does that convert the FFN into an effectively ANE-hosted block
at acceptable cost — and now, explicitly, at acceptable *accuracy*, since
step 1 showed those pull in opposite directions.
