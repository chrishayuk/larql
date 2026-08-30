# CPU-7B2 — does the amortisation multiplier transfer across matrix geometry?

Pre-registered 2026-08-26, before measurement. Gates FROZEN. Earned by
CPU-7C2, which falsified the transfer of a GLOBAL `m_K`.

---

## What C2 established, stated at the width the evidence supports

CPU-7C2 grouped every eligible projection in a real Qwen3.8 layer on a
repaired execution schedule. Logical weight-load demand fell 53.55 ->
29.48 GB, 45%, exactly as designed. Wall time did not move: `C 510.7 ms`,
`D 517.5 ms`.

So this is falsified:

```
T_K ~= K(1-g)T_1 + m_K·g·T_1        with a GLOBAL m_2 = 1.02 from CPU-7B
```

**And this is NOT established**, though an earlier draft of the C2 write-up
claimed it:

> "traffic fell 45%, therefore the workload became compute-bound"

The ledger counts LOGICAL weight-load demand — bytes the program asked
for. It does not count DRAM transactions. Cache can satisfy some nominal
second reads, and "apparent GB/s fell" is a statement about the ledger's
numerator, not about the memory system. No instrument in this programme
currently distinguishes the two.

The defensible conclusion is exactly:

> Weight stationarity reduces logical weight traffic as designed, but on
> the restored production schedule that reduction does not reduce wall
> time. CPU-7B's amortisation multiplier does not transfer unchanged to
> this execution substrate.

---

## The question, and why it is the cheapest next move

CPU-7B measured `m_2 = 1.02` on ONE shape: `5120 x 5120`. CPU-7C2 is
dominated by shapes it never tested:

```
FFN up      17408 x 5120
FFN gate    17408 x 5120
FFN down     5120 x 17408      a much deeper reduction
```

So before any planner model is changed, ask whether `m_K` is a property of
the KERNEL or of the GEOMETRY. That is a synthetic probe on the same
instrument that produced 7B, and it costs one quiet machine-hour rather
than another 51 GB run.

---

## Arms

```
shape            out x in        role
5120 x 5120      CONTROL — must reproduce CPU-7B's m_2 = 1.02 +- 0.10.
                 Outside that the PROBE changed, not the geometry, and
                 no other row is readable.
17408 x 5120     FFN up / gate
5120 x 17408     FFN down
```

`N` in {1, 2, 4}, production worker count, `Q8[64] x asym-Q8[16]`,
DRAM-resident bank, and every control CPU-7B already carries: C1 rate
reproduction, C2 cache/DRAM discriminator, C3 bit-identity plus its
planted violation, C4 order reversal.

---

## FROZEN outcome bands, on `m_2` for the FFN shapes

```
m_2 <= 1.35    TRANSFERS. Geometry is not the explanation, and C2's
               missing saving is an execution-system effect at the layer
               or executor level. Next step is per-class projection wall
               time inside arms C/E/D.

m_2 >= 1.70    DOES NOT TRANSFER. CPU-7B was real but not general, and
               the planner rule becomes an amortisation curve PER MATRIX
               GEOMETRY rather than one scalar.

1.35 - 1.70    partial. Report both halves; neither branch is earned.
```

The bands are CPU-7B's own N=2 bands, unchanged, so the two runs are
adjudicated on one bar.

---

## What each branch licenses

**Does not transfer.** The model becomes

```
T_K = sum_i  T_i · m_K(geometry_i) / K
```

which is richer than a global `g` and is a genuinely useful compiler
result: the physical planner would learn an amortisation curve per matrix
geometry and group positions only where the machine has unused arithmetic
headroom.

**Transfers.** The isolated kernel amortises on the real shapes and the
whole layer does not, so something at the layer or executor level absorbs
the saving. The next instrument is per-class projection WALL TIME inside
arms C, E and D — now meaningful, because C2 removed the position-parallel
nesting that made those intervals overlap. The question becomes whether
`C ffn projection wall ~= D ffn projection wall`, or whether projection
time falls while another component expands.

---

## Retraction carried forward

CPU-7C1 banked:

> "CPU-7 now has a predictive cost model. `g` measured independently,
> `m_K` imported from 7B, production effect obtained without fitting."

CPU-7C2's arm E is what tested that, and it fails: the same recurrent
tranche that C1 predicted to 0.4% on a fan-out-collapsed substrate saves
approximately nothing on a healthy one. **`m_K` is therefore not a
property of the stationary kernel alone.** More likely

```
m_K = f(matrix geometry, worker geometry, memory state,
        instruction throughput, K)
```

which is still planner-usable and is not the scalar C1 hoped for. Arm E
existed precisely to ask this, and it earned its fifteen lines.

---

## RUN LOG

### 2026-08-26 — TRANSFERS. Geometry is NOT the explanation.

Battery, machine 96% idle, synthetic probe (no container). Provisional by
the standing power rule — but every control passed, the order-reversal
control passed, and the CONTROL ROW reproduced CPU-7B inside its own run,
so the within-run contrast this rung turns on is not one an absolute-rate
drift can manufacture.

```
C1  126.1 GB/s against 118.0 banked        within 10%
C2  198.6 cache / 126.1 dram = 1.58x       >= 1.5x
C3a 3.773e-5   C3b bit-identical   C3c planted violation fired
C4  worst forward/reversed drift 0.07      <= 0.15
```

```
shape                  N=1 ms    m_2     m_4    wt GB/s
control 5120x5120       16.73   1.005   1.337     128.2
ffn up/gate             16.58   0.976   1.275     125.7
ffn down                16.48   0.994   1.331     126.5
gd qkv 10240x5120       15.38   0.995   1.357     130.4
```

The control reproduces CPU-7B's banked `m_2 = 1.02` as 1.005 and `m_4 =
1.27` as 1.337, so the probe did not change. **Every production geometry
amortises exactly as the square one did**: `m_2` spans 0.976 to 1.005, all
far inside the `<= 1.35` TRANSFERS band, and `m_4` spans 1.275 to 1.357.

The deep-reduction shape — `5120 x 17408`, the FFN `down` projection that
was the most plausible geometric culprit — is 0.994. It amortises.

### Verdict

```
m_2 <= 1.35   TRANSFERS      <- taken
```

**CPU-7B's amortisation multiplier IS general across the matrix geometries
a real Qwen3.8 layer is made of.** So CPU-7C2's missing saving is not a
shape effect, and the planner rule does not need a per-geometry curve.

The saving is absorbed at the LAYER or EXECUTOR level: the isolated kernel
amortises on exactly these shapes at exactly this worker count, and the
whole layer does not.

### The next instrument, per this protocol's own branch

Per-class projection WALL TIME inside arms C, E and D:

```
C ffn projection wall  ~=  D ffn projection wall    ->  the projections
                                                       never got faster
C ffn projection wall  >   D ffn projection wall    ->  they did, and
                                                       something else grew
```

`ProjectionLedger::site()` already records this. It is readable at K>1 now
ONLY because CPU-7C2 removed the position-parallel FFN nesting — the same
overlap that made `nanos` unusable at K>1 before. That is worth stating:
the fix under test is what made its own diagnostic possible.

**Not run yet, and deliberately not built yet.** The CPU-7C2 AC
replication runs FIRST, on the unmodified binary `1fb4e87d...`. Adding a
printout changes the binary, and a rebuild has already moved an untouched
function 14% on this codebase — so the replication must not be the run
that also carries a new diagnostic.
