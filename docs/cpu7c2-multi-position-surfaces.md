# CPU-7C2 — restore machine ownership, then measure stationarity on top

Pre-registered 2026-08-26, before implementation. Gates FROZEN. Earned by
CPU-7C1 (`docs/cpu7c-two-position-layer.md`), which established both the
mechanism and the defect this rung exists to remove.

---

## What C1 left

```
p = 0.904   projection share of a 262.2 ms serial token
g = 0.209   share currently groupable   (GatedDelta dense projections)
p - g = 0.695   trapped behind FfnCall and attention_into_kv
```

and one defect:

```
B / (2A) = 1.422    batching is 42% SLOWER than serial
                    slabs/call 5.03 -> 2.81
```

`execute_layer` runs the FFN and attention inside `h.par_iter_mut()`, so
each position's projections execute inside a rayon worker, where
`caller_owns_the_machine()` collapses them to a single worker. Positions
in parallel, each invoking another parallel projection — the nested
structure C1's measurement falsified.

---

## The shape

**Rows own the machine, not positions.**

```
    positions
        |
    multi-position logical operator
        |
    CpuExecutor
        |
    partition OUTPUT ROWS across workers
        |
    each worker consumes all K activations
```

For the FFN:

```
old   h.par_iter_mut()  ->  up(x_t), gate(x_t), down(x_t)
new   ffn_many([x0..xK]) ->  up_many, gate_many,
                             activation PER POSITION,
                             down_many
```

and the analogous change for attention. This is also the shape the
eventual speculative verifier needs, so it is not a benchmark fix.

---

## FOUR arms, because the change does two things at once

Raising the surfaces removes the fan-out collapse AND makes those
projections groupable. Those must not be reported as one number.

```
A  serial                 K x step. Known-good machine utilisation.
B  current batched        today's nested parallelism. Reproduces 1.422x.
C  raised, stationary OFF FAN-OUT RESTORATION only.
D  raised, stationary ON  WEIGHT-STATIONARY gain, on a healthy schedule.
```

```
C / (K·A)   did restoring row ownership recover serial parity?
D / C       the clean production transfer of the 7B mechanism
D / (K·A)   what the programme buys
B / (K·A)   retained so the defect stays visible after it is fixed
```

### FROZEN gate — restoration is adjudicated BEFORE stationarity

```
C / (K·A)   <= 1.05   RESTORED. Stationarity may then be credited.
            >  1.15   NOT RESTORED. `D/C` is uninterpretable and the
                      rung stops here: something else owns the loss.
```

**Stationarity gets no credit until `C ~= K·A`.** A single arm going from
747 ms to ~300 ms, decomposed afterwards, is worth much less than this.

---

## Prediction, frozen in FORM before `g` is remeasured

```
R_K = 1 - g + (m_K / K)·g          m_2 = 1.02, m_4 = 1.27   (CPU-7B)

N=2   R_2 = 1 - 0.490·g
N=4   R_4 = 1 - 0.6825·g
```

If raising both surfaces takes `g` toward `p`:

```
  g      R_2    target-side      R_4    target-side
0.80    0.608        1.64x      0.454        2.20x
0.85    0.584        1.71x      0.420        2.38x
0.90    0.559        1.79x      0.386        2.59x
0.904   0.557        1.80x      0.383        2.61x
```

**`g = 0.90` is NOT assumed.** The C1 order is repeated exactly:

```
1  implement the surfaces
2  parity — the same hierarchy: C vs D bit-identical; A vs C numerical
3  K=1: measure the NEW g, before any timing
4  freeze R_2 and R_4 from that g
5  adjudicate C/(K·A) for restoration
6  only then read D/C and D/(K·A)
7  order-reversed clock; worse ratio adjudicates, never the mean
```

Ceiling note: these use `p = 0.904` measured in production, not the 0.930
of the first calibration, so the N=4 figure is 2.61x rather than the 2.74x
quoted before C1 ran.

---

## Controls carried forward from C1, unchanged

- `slabs/call` per arm. It is what turned "the fan-out probably collapsed"
  into a measurement, and it is the first thing to read if a ratio misses.
- Counters snapshotted BEFORE the probe step.
- Arm A timed over the SAME K positions as the others — C1's first run
  timed the prefill too and printed a fourfold win made of nothing.
- `grouped` is the kernel's own answer, never inferred from `positions > 1`.
- Machine quiet and verified by instantaneous sampling, not by load
  average; one contaminated repeat already poisoned a worse-of-orders
  ratio in C1.

---

## What a result licenses

```
C/(K·A) <= 1.05 and D/C tracking R_K
    -> the cost model is predictive across TWO independent increments,
       which is a planner rule and not an optimisation
D/(K·A) at N=4 near 0.39
    -> the target side of speculative decoding is no longer the risk,
       and CPU-7D is earned
```

Still unmeasured and still not claimed by any of this: ACCEPTANCE RATE,
which is the other factor in every speculative-decoding estimate.

---

# AMENDMENT 1 — pre-implementation, pre-measurement

A code audit corrected the scope before anything was built. The body above
frames C2 as raising FFN **and** attention, and says one edit both removes
the `1.422x` regression and moves `g` toward `p`. Only the first half of
that is true of attention.

## What the audit found

`execute_layer`'s parallel regions, and what runs projections inside them:

```
h.par_iter()      pre-attention norm        no projections
attention block   attention_into_kv         TOP LEVEL — full row fan-out
h.par_iter_mut()  residual add              no projections
h.par_iter()      pre-FFN norm              no projections
h.par_iter_mut()  ffn.apply_from_residual   <- the ONLY projections inside
                                               a parallel region
```

`attention_into_kv` is per-position and sequential, but it is called at the
top level and keeps the executor's row fan-out. **The FFN's `par_iter_mut`
is the sole source of `slabs/call` 5.03 -> 2.81.**

## The scope, restated

```
CPU-7C2   FFN multi-position ONLY
          1. remove the measured nested-parallelism fan-out collapse
          2. make FFN up/gate/down stationary-eligible

CPU-7C3   attention multi-position
          purely additive stationarity coverage.
          NO fan-out-restoration claim — there is nothing to restore.
```

**Every arm and every gate above is unchanged.** A/B/C/D stand,
`C/(K·A) <= 1.05` PASS and `> 1.15` FAIL stand, and the ordering stands.
Only the edit's extent narrowed.

## Why this makes C2 cleaner rather than smaller

C1 measured two phenomena at once — nested FFN scheduling collapsing the
fan-out, and GatedDelta grouping amortising traffic. C2 now separates them
across one structural seam:

```
B -> C   FFN surface raised, stationary OFF   pure machine-ownership repair
C -> D   same healthy schedule, stationary ON  pure FFN stationarity gain
```

And because attention never caused the collapse, **C3 becomes an
INDEPENDENT REPLICATION of the cost model rather than part of the repair.**
Three operator classes, introduced separately, each following

```
T_K ~= K(1-g)T_1 + m_K·g·T_1
```

is much harder to dismiss as Qwen-specific luck than one fitted curve.

## `g` is instrumented in TWO pieces, in ONE build

At K=1, before any clock:

```
g_GD        already-groupable GatedDelta time      (C1 measured 0.209)
g_FFN       newly groupable FFN time
g_total     g_GD + g_FFN
p - g_total remaining attention tranche, for C3
```

Both measured in the SAME binary. Taking `g_FFN` as `g_total` minus C1's
banked 0.209 would be a difference across builds, and a rebuild has
already moved an untouched function 14% on this codebase (CPU-2D).

## The INCREMENTAL prediction, frozen

Beyond the total `R_K`, C2 predicts its own increment. At N=2 the FFN's
expected saving is

```
dT_FFN = (2 - m_2)·g_FFN·T_1 = 0.98·g_FFN·T_1
```

so, before D is measured:

```
D predicted = C measured - dT_FFN
```

compared against D measured, exactly as C1 compared 693.2 against 696.2.
That is an independent replication of C1's 0.4%, not merely a check that
the total lands in a band.

## The fan-out control must be brutally obvious

```
A slabs/call   ~5.03
B slabs/call   ~2.81      the defect, still visible
C slabs/call   ~A         RESTORED
D slabs/call   ~A         stays restored
```

**If `C/(2A)` reaches ~1 while `slabs/call` does NOT restore, something
else compensated and the claimed mechanism is unproven.** The ratio alone
must not be accepted as evidence for the mechanism; the counter is what
attributes it.

## K=4 stays held until N=2 earns it

```
N=2   parity -> g -> restoration C/(2A) -> D/C model transfer
      all four hold, and only then:
N=4
```

Unlike C1, N=4 is now worth running: it will not be dominated by a known
scheduling defect. What it would then establish is that multi-position
execution retains ordinary CPU utilisation AND that the stationary
primitive scales from two positions to four inside the full model — which
together are the target-side substrate speculation needs.

---

# AMENDMENT 2 — the class split, measured BEFORE implementation

K=1 on the real container, AC, quiet. Frozen here so C2's prize is a
number the rung was aimed at rather than one it produced.

```
class        time      share    groupable today
other       11.1 ms    0.039        0.0 ms      head, embedding
recurrent   59.0 ms    0.210       59.0 ms      C1, already grouped
ffn        166.7 ms    0.593        0.0 ms      C2's tranche
attention   18.7 ms    0.066        0.0 ms      C3's tranche
-----------------------------------------------
           255.5 ms    0.909  =  p             sums to the whole
```

**The sum-check is the point of printing it.** The four classes account
for `p` to rounding, so nothing else is issuing projections and the split
can be predicted from. Had they not summed, the split would have been
unusable however plausible each row looked.

`g` across four independent calibrations: 0.209, 0.210, 0.209, 0.210.
Absolute wall drifted 263 -> 281 ms between runs on the same machine; the
SHARES did not. Read shares, not milliseconds — the same lesson CPU-7B
learned when its ratios held to 0.04 while rates moved 5%.

## What this does to the roadmap

**Attention is small.** 0.066, not the large remaining tranche `p - g`
implied when it was lumped with the FFN. So:

```
after C2   g = 0.210 + 0.593 = 0.803    R_2 0.607 (1.65x)   R_4 0.452 (2.21x)
after C3   g = 0.803 + 0.066 = 0.869    R_2 0.574 (1.74x)   R_4 0.407 (2.46x)
ceiling    g = p     = 0.909            R_2 0.555           R_4 0.380
```

C2 alone reaches ~92% of the way from today's 0.209 to the ceiling.

**That makes C3 a better experiment than it would have been as a prize.**
A small, precisely predicted increment is a sharper test of the cost model
than a large one: 0.066 leaves nowhere for a mis-specified model to hide.

## C2's INCREMENTAL prediction, frozen with the number in it

```
dT_FFN = (2 - m_2)·g_FFN·T_1 = 0.98 x 166.7 ms = 163.4 ms

D predicted = C measured - 163.4 ms
```

against `D measured`, exactly as C1 compared 693.2 against 696.2.

## Instrument defect this measurement found, recorded because it nearly passed

The first attempt returned `ffn 0.0, attention 0.0` and 182.6 ms in
`other`. The guards had been placed in `execute_layer`, but the K=1
calibration runs `step_observed` — the hand-written serial path, with its
own FFN and attention call sites. `layer_forward_with` is shared by both
paths, which is why `recurrent` alone reported correctly and the output
looked like a result rather than a bug.

**Inferring `g_FFN` from `p - g` would have given a number that was
roughly right for entirely the wrong reason.** The class sum-check is what
makes that distinguishable, and it is now a standing control.

---

# AMENDMENT 3 — the prediction was wrong twice; both fixed before building

## 1. It anchored on absolute milliseconds from ANOTHER run

Amendment 2 froze `dT_FFN = 0.98 x 166.7 ms = 163.4 ms` — and Amendment 2
itself had just established that the stable quantity is the SHARE, not the
time: wall moved 263 -> 281 ms across runs while `g` held at 0.209/0.210.

That is CPU-5's G1 defect exactly: **an anchor measured elsewhere, carried
into a gate.** Every prediction below is a RATIO, converted to
milliseconds using the adjudicating run's OWN arm A and nothing else.

## 2. Arm D turns on ALL stationarity, not just the FFN's

`D` is "raised surface, stationary ON", so `C -> D` re-enables the
recurrent tranche as well as the new FFN one:

```
recurrent   g_GD  = 0.210   (already proven by C1)
ffn         g_FFN = 0.593   (what C2 adds)
                    -----
            g_D   = 0.803
```

So `D = C - FFN saving` is simply wrong; D also collects the recurrent
saving. Corrected and frozen, conditional on `C/(K·A) ~= 1`:

```
N=2   recurrent   0.4900 x 0.210 = 0.1029  of 2A
      ffn         0.4900 x 0.593 = 0.2906  of 2A
                                   ------
      total                        0.3935  ->  D/(2A) = 0.607

N=4   recurrent   0.6825 x 0.210 = 0.1433  of 4A
      ffn         0.6825 x 0.593 = 0.4047  of 4A
                                   ------
      total                        0.5480  ->  D/(4A) = 0.452
```

## A fifth arm, because it turned out to be trivial

The decomposition above INFERS the FFN contribution from a mechanism C1
established separately. A direct measurement is better, and the `Site`
thread-local added in Amendment 2 makes class-selective stationarity about
fifteen lines — so it is built rather than reasoned around.

```
C   raised, all stationary OFF
E   raised, RECURRENT stationary only
D   raised, recurrent + FFN stationary
```

```
E / C   replication of C1's result on a HEALTHY schedule
        predicted 1 - 0.1029/1.0 = 0.897 of C, at N=2
D / E   the PURE FFN transfer
        predicted 1 - 0.2906/0.897 = 0.676 of E, at N=2
D / (K·A)  the total, as above
```

`E/C` is worth more than it looks: C1 measured the recurrent tranche on a
schedule that had collapsed the row fan-out. `E/C` measures the same
tranche on a healthy one. If the model survives that change of substrate,
it is not describing an artefact of the broken schedule.

## The accounting invariant, promoted

`other` closing the accounting is now a standing planner invariant, not a
one-off check:

> Every projection site belongs to a declared class. A new site must make
> `other` go NONZERO rather than silently contaminating a derived `g`.

The first class-split attempt returned `ffn 0.0, attention 0.0` with
182.6 ms in `other` — and inferring `g_FFN` from `p - g` would have given
0.699 against the true 0.593: wrong by 18%, and wrong in a way that would
have propagated into every prediction on this page.

---

# AMENDMENT 4 — predictions are ADDITIVE in normalized K·A space

Amendment 3 stated `E/C` and `D/E` as fixed ratios, which silently assumes
`C/(K·A) = 1.000`. Restoration is allowed to land anywhere at or under
1.05, and at `C/(K·A) = 1.04` a perfectly behaved tranche would look like
it missed — a gate failing for a reason that is not the finding, which is
the same error the A-vs-B parity contract was written to avoid.

**The authoritative predictions are normalized SAVINGS, anchored on the
run's own measured `C`.** `E/C` and `D/E` are derived, never frozen.

```
c = C / (K·A)                          measured, not assumed

              N=2                    N=4
s_recurrent   0.4900 x 0.210 = 0.1029   0.6825 x 0.210 = 0.1433
s_ffn         0.4900 x 0.593 = 0.2906   0.6825 x 0.593 = 0.4047

E/(K·A) predicted = c - s_recurrent
D/(K·A) predicted = c - s_recurrent - s_ffn
```

At perfect restoration this reproduces Amendment 3 exactly
(`c = 1.000 -> E 0.897, D 0.607` at N=2); at `c = 1.04` it gives
`E 0.937, D 0.647`, and `E/C 0.901`, `D/E 0.690`.

## Which makes C2 a replication ladder, not one result

```
A -> C   machine-ownership restoration
C -> E   does the recurrent tranche save exactly its pre-measured share,
         now on a HEALTHY row-owned substrate?
E -> D   does the FFN tranche save exactly its pre-measured share?
```

Each effect predicted before timing, each adjudicated separately, in that
order. C1 established the recurrent tranche on a substrate whose fan-out
had collapsed; `C -> E` re-establishes it on one that has not. If all
three land, the cost model has survived an OPERATOR-CLASS change and a
SCHEDULER change — which is what a planner rule needs, and what a single
large speedup number would not have supplied.

## Why the dense-FFN geometry supports the prediction

```
FFN wall share    59.3%
FFN byte share    ~63%      (3 x 17408 x 5120 x 64 layers at Q8 ~ 17 of 27 GB)
```

Those agreeing to a few points is evidence the object is the expected
bandwidth-dominated one rather than a timing-accounting artefact. All
three projections should contribute: `up` and `gate` share one activation
across two enormous `[17408, 5120]` traversals; `down` takes new `[17408]`
activations but is an equally large traversal. The activation itself stays
per-position and is small.

---

# AMENDMENT 5 — restoration needs BOTH halves, and the mechanism half needs a number

"`C/(K·A) <= 1.05` AND the row fan-out restored" was stated without a bar
for the second half. Frozen now, before the harness runs:

```
timing      C/(K·A)                    <= 1.05
mechanism   C slabs/call  >=  0.90 x  A slabs/call
```

Both, or restoration FAILS and no stationarity verdict is printed.

The mechanism clause is not decoration. `C/(K·A)` could come back near 1
because something unrelated compensated — a cache effect, a different
allocator path, a scheduling accident — while the row partition stayed
collapsed. The claim being made is specifically that machine ownership was
restored, and only `slabs/call` can attest to it. CPU-7C1 measured
`A 5.03` against `B 2.81`; C is expected to come back to A's figure,
because it removes the nesting rather than tuning around it.
