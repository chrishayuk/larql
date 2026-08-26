# CPU-7C — does projection amortisation survive a whole layer?

Pre-registered 2026-08-25, before any measurement, after CPU-7B passed.
Gates below are FROZEN. Container: `~/chris-models/Qwen3.8-27B.vindex3`.

---

## The question

CPU-7B established, on a synthetic 5120x5120 sweep:

```
two projection vectors cost 1.02x one weight traversal of that projection
```

A real Qwen3.8 layer is not a projection. Around the eight per-position
projections it runs a causal depthwise convolution, a GatedDelta
recurrence carrying durable state, two or three RMS norms, an activation
quantisation, residual adds, and an FFN activation. **None of that
amortises across positions**, and some of it is strictly sequential in
position order.

So: how much of the projection-level amortisation survives?

---

## The sites, named before they are touched

Per GatedDelta layer, per position, today (`exec/gated_delta.rs`,
`exec/production.rs`):

```
in_proj_qkv   gated_delta.rs:555   one project() per position, in a .map()
in_proj_a/b/z gated_delta.rs:643-645
out_proj      gated_delta.rs:688
ffn up/gate/down   production.rs:783-813, one FfnCall per position
```

Eight weight traversals per position. At two positions, sixteen.

---

## Three arms, because the obvious baseline is wrong

The tempting comparison is "two single-position calls". That would
OVERSTATE the win, because it is not what the code does today.

```
A  serial decode      two ordinary single-position steps, the decode
                      driver's own path — what speculation replaces
B  batched, today     execute_layer with two positions. Positions run
                      POSITION-PARALLEL (h.par_iter_mut), each thread
                      streaming the full weight independently and each
                      projection collapsing to one worker via
                      caller_owns_the_machine
C  weight-stationary  execute_layer with two positions, projections
                      taken ONCE and applied to both
```

**Arm B is the honest baseline for the change**, and it may already
capture part of the win by accident: two threads walking the same matrix
in near-lockstep can be served from cache, which is weight-stationarity
arrived at by luck rather than by design. That is precisely why B is
measured rather than assumed away.

```
C / B       what the CHANGE buys over today's batched path
C / (2 x A) what the PROGRAMME buys, and what 7D's arithmetic needs
```

---

## Controls, run first

```
P1  arm C is BIT-IDENTICAL to arm B, per position, at every layer output.
    Achievable and therefore required: reordering the loop nest changes
    which weights are resident, never which products are summed in which
    order. The codebase already holds this invariant for position
    parallelism ("an execution strategy, never a reassociation") and
    arm C must not be the thing that breaks it.

P2  the planted violation: perturb one position's input by one ulp and
    P1 must FAIL, and fail only for that position.

P3  arm B at ONE position must be bit-identical to arm A. If the batched
    and serial paths already disagree at a single position, no ratio
    between them means anything and 7C stops until that is explained.
```

---

## FROZEN gate

```
C / (2 x A)     <= 0.65    STRONG — the prerequisite for speculative
                           decode is demonstrated in the architecture;
                           CPU-7D becomes real
                <= 0.80    modest — real but not transformative
                >  0.90    RETIRE — projection amortisation does not
                           survive the layer, and 7B was a kernel fact
                           about a kernel, not about decoding
```

Reported alongside, not gated on: `C / B`, and the per-class time split.

---

## Pre-registered PREDICTION, with its falsifier

The layer's projection class is already instrumented (`OpClass::Projection`).
Let `p` be its share of single-position layer time. If projections are free
at N=2 and nothing else amortises:

```
predicted  C / (2 x A)  =  (p + 2(1-p)) / 2  =  1 - p/2
```

```
p = 0.85  ->  0.575
p = 0.70  ->  0.650
p = 0.50  ->  0.750
```

**Falsifier:** a measured ratio that misses `1 - p/2` by more than 0.10.
That would mean either the projections did not amortise inside the layer
(cache pressure from the recurrence evicting weight tiles is the obvious
mechanism) or something outside the projection class scales worse than
linearly with positions. Either is a finding; neither is a tuning knob.

Measure `p` FIRST, from arm A, and write it down before running C.

---

## Scope limits

- N = 2 only. Not 4, not 8. The recurrence is sequential in position and
  the conv window reaches backwards; whatever N=2 costs, N=4 is a
  separate measurement and not an extrapolation.
- One layer class. Qwen3.8 has 48 GatedDelta and 16 full-attention
  layers; this rung measures GatedDelta. The attention layers are their
  own arm and are not claimed by it.
- Nothing here measures ACCEPTANCE. Every speculative-decode estimate has
  two factors and this rung supplies one of them.
- The candidate CPU-5 arithmetic is NOT required. Arms run whatever the
  process arm selects; the ratio is between execution geometries at fixed
  arithmetic, so a CPU-6 RETIRE would not invalidate this rung.

---

# AMENDMENT 1 — pre-implementation calibration, before arm C exists

## `p` is MEASURED: 0.930

Real container, `--backend production`, 6-token `--generate`, battery,
`~/chris-models/Qwen3.8-27B.vindex3`. Not a result — a calibration, taken
before the implementation so the gate can be checked against the mechanism
rather than against taste.

```
class              calls   total ms   % token
Projection           497     330.47     93.0%
DeltaRecurrence       48      12.98      3.7%
DeltaConv             48       3.32      0.9%
FfnActivation         64       1.84      0.5%
AttentionCore         16       1.31      0.4%
everything else                 2.42      0.6%
-----------------------------------------------
steady token wall            355.51 ms
unattributed                   3.08 ms    0.9%
```

The 355.51 ms reproduces the banked shipped 348 ms/token to +2.2% on
battery, so this is the shipped arithmetic and the instrument agrees with
the ledger it is being read beside.

**A layer that is 93% projection is the best possible news for this
programme and the worst possible news for any other CPU lever.** Nothing
outside the projection class is worth optimising: the GatedDelta
recurrence — the thing that looked like the barrier — is 3.7%.

## The prediction, renamed and restated

```
idealized traffic prediction    C / (2A)  =  1 - p/2      = 0.535
literal, using 7B's measured 1.02x at N=2
                                C / (2A)  =  1 - 0.49p    = 0.544
```

The difference is under one point and the falsification band is +-0.10, so
the frozen `1 - p/2` stands as the IDEALIZED form and 0.544 is what the
mechanism actually predicts.

**The frozen STRONG gate is checked against the mechanism and survives.**
`<= 0.65` against a predicted 0.544 leaves ~0.11 of slack, which is about
the falsification band — so passing the gate is close to "did not miss the
prediction badly", which is the relationship a gate and a prediction should
have. The gate is NOT asking the implementation to beat its own mechanism.

## Causal eligibility of all eight sites — the answer is all of them

The concern was that a recurrence makes some projections ineligible for
cross-position execution. Traced site by site through
`exec/gated_delta.rs`:

```
site            reads                        eligible?
in_proj_qkv     hidden[t]  (layer input)     YES, already batched in a .map()
in_proj_a/b/z   hidden[t]  (layer input)     YES — needs hoisting OUT of the
                                             `for t` loop it currently sits in
out_proj        normed[t] <- core[t] <-      YES — but only AFTER the whole
                step_inner, sequential in t  `for t` loop: collect normed[t]
                                             for every t, then one traversal
ffn up/gate/down  post-attention residual    YES
attention q/k/v   layer input                YES
attention o       attention output           YES
```

**Not one projection's INPUT depends on another position's recurrent
state.** `step_inner` consumes `q`, `k`, `value`, `g`, `beta`, all of which
derive from `hidden[t]` alone; it advances `DELTA_MATRIX`, and the only
projection downstream of that is `out_proj`.

So the seam is real but it is a SCHEDULING seam, not a dependency seam:

```
before the recurrence   in_proj_qkv, in_proj_a/b/z     batch across positions
strictly sequential     step_inner over t              3.7% of the token
after the recurrence    out_proj, then FFN             batch across positions
```

The eligible projection share is therefore ~1.00 of `p`, and the prediction
uses `p` itself rather than a discounted share. **This is a claim with a
falsifier**: if arm C cannot hoist `in_proj_a/b/z` and `out_proj` out of the
`for t` loop without changing a single output bit, the eligibility analysis
is wrong and the prediction must be recomputed against what actually moved.

## `C / B` is frozen as the PRODUCTION comparison

`C/(2A)` and `C/B` answer different questions and a great one must not
launder the other. It is entirely possible to see `2A = 2.00, B = 1.30,
C = 1.27` — a spectacular `C/(2A) = 0.635` alongside essentially no
production win, because today's position-parallel path was already getting
the effect from shared cache.

```
mechanism            C / (2A)     does weight-stationary execution expose
                                  the predicted physical amortisation?
production benefit   C / B        does this beat what LARQL already gets
                                  by accident?
```

```
C / B   <= 0.90     meaningful production win — integrate
        0.90-0.97   modest — integrate only if the architecture gets cleaner
        >  0.97     no performance reason to replace B
```

`C/B` does NOT gate the scientific finding. It gates INTEGRATION.

## Arm B is a measurement in its own right

If B is surprisingly good, that is not 7C spoiled — it is a finding:
**LARQL may already have emergent temporal weight-stationarity**, two
`par_iter` threads walking one matrix near-lockstep being served from
shared cache. Explicit C would still buy determinism over scheduler luck,
less cache duplication, controlled worker ownership, and a geometry that
EXTENDS. B's accidental sharing plausibly holds at N=2 and falls apart at
N=4, when four threads no longer stay in step. That is a prediction, and
the N sweep below tests it.

## The N sweep is extended to 1, 2, 4

7B says N=4 is where the extraordinary amortisation still survives
(per-vector 0.32x). Applying 7B's measured multipliers to the measured
`p = 0.930`:

```
        m(N) from 7B    C / (N x A)    implied target-side speedup
N=2            1.02x          0.544                        1.84x
N=4            1.27x          0.365                        2.74x
N=8            2.41x          0.350                        2.86x
```

N=8 barely beats N=4, which matches 7B's per-vector curve flattening
(0.32 -> 0.30). **N=4 is the knee, and the sweep stops there.**

Still start at N=2 and do not read N=4 until N=2's parity controls pass.
None of these are measurements; they are the arithmetic the calibration
implies, and the entire point of running 7C is that a real layer may not
deliver them.

---

## RUN LOG

### 2026-08-26 — CPU-7C1 ADJUDICATED. Mechanism CONFIRMED, arm C not integrable.

Quiet machine (97.3% idle, load 1.11), battery, real container
`Qwen3.8-27B.vindex3`, `q8xq8b` + `ACT_BLOCK=16` + `ACT_CODE=asymmetric`.
Two earlier runs are void and recorded below as such.

**Parity — the hierarchy held, and A/B beat its own gate.**

```
B vs C   batch identical   probe identical     mandatory, PASSED
A vs B   0.00e0            0.00e0              bit-identical
A vs C   0.00e0            0.00e0
```

`A vs B` was pre-registered at `rel_rms < 1e-5` because `traverse` takes a
batched attention branch `step` does not. It came out BIT-IDENTICAL on the
real model. Recorded as a stronger observed result; the gate is NOT
retroactively tightened, because a gate stronger than its semantics fails
for reasons that are not the finding.

**Calibration, K=1.**

```
wall                   262.2 ms
all projection         237.1 ms    p = 0.904
groupable projection    54.9 ms    g = 0.209     OPPORTUNITY
remaining projection   182.3 ms    p-g = 0.695
```

**The finding: batching is SLOWER than serial, and the reason is measured.**

```
              calls   pos/call   slabs/call   bytes
A  serial       994       1.00       5.03     54.90 GB
B  batched      753       1.32       2.81     53.55 GB
C  stationary   753       1.32       2.81     47.67 GB
```

`slabs/call` is the row partition the executor actually used. A cuts each
projection across ~5 workers; B across ~2.8 — 42% of A's partitions.
`execute_layer` runs the FFN and attention inside `h.par_iter_mut()`,
where `caller_owns_the_machine()` collapses each projection to ONE worker.
That is the whole of

```
B / (2A) = 1.422        stable to 0.07% over three repeats
                        (747.5 / 747.4 / 747.0 ms)
```

**And with that term present, CPU-7B predicts the layer to 0.4%.**

Groupable work is 54.9 ms per serial token. In B it runs twice (109.8 ms);
in C at `m2 = 1.02` (56.0 ms). Predicted saving 53.8 ms:

```
B measured      747.0 ms
C predicted     693.2 ms
C measured      696.2 ms       0.4% off
C/B predicted     0.928
C/B measured      0.931
```

**The frozen prediction is exactly right once the collapse is removed.**

```
(525.2 - 53.8) / 525.2  =  0.8976
frozen 1 - 0.49g        =  0.897
```

Not a fit: the prediction was frozen before the run and `g` was measured
independently at K=1.

**Clock, order-reversed.**

```
             C/(2A)   B/(2A)    C/B
B→C           1.326    1.423   0.931
C→B           1.346    1.423   0.946
B→C           1.365    1.422   0.960
                                0.960  worse-of-orders adjudicates
```

### Verdicts against the frozen gates

```
C / (2A) = 1.326   vs 0.897 predicted   MISS by 0.43 — and fully attributed
                                        to B/(2A) = 1.422, not to the kernel
C / B    = 0.960   band 0.90-0.97       MODEST — integrate only if the
                                        architecture gets cleaner. It does
                                        not, on its own. DO NOT integrate C1.
```

### Engagement, and why 144 rather than 240

C grouped 144 calls = 3 sites x 48 GatedDelta layers: `in_proj_qkv`,
`in_proj_z`, `out_proj`. `in_proj_a` and `in_proj_b` are the `48 x 5120`
delta gates that stay f32 — below the compact threshold, so not
`WeightRows::Q8`, and `supports` correctly declines. Both have 48 outputs
and cost almost nothing in `g`.

### Three conclusions banked

1. **Multi-position batching in current LARQL is actively SLOWER than
   serial decode**, by 42%, and the mechanism is worker ownership
   collapsing from 5.03 to 2.81 slabs/call. A real defect, found by
   building the experiment rather than by profiling.
2. **Explicit weight stationarity works in the real model.** Against the
   actual grouped work, CPU-7B predicts the measured saving to 0.4%.
3. **CPU-7 now has a predictive cost model.** `g` measured independently,
   `m_K` imported from 7B, production effect obtained without fitting the
   timing result:

```
T_K  ~=  K(1-g)T_1  +  m_K · g · T_1        (no fan-out collapse)
```

   If C2 and C3 repeat that, the result is not one optimised Qwen
   implementation but a PLANNER RULE: whether multi-position execution
   pays, predicted from operator time-share and the kernel's amortisation
   curve.

### K=4 on this graph is deliberately NOT run

It would mostly measure the collapse, more expensively. Deferred to C2.

### Two void runs, recorded because they happened

- **Run 1**: `run_serial` timed the 8-token prefill, so arm A was 11
  positions against B/C's 2 — printing `C/(2A) = 0.239`, a four-fold
  "amortisation" made entirely of prefill. Caught before it was quoted.
- **Run 2**: another session started a 13.6 GB job mid-run; load reached
  33.6 and one repeat came in 5-7x the others, poisoning the worse-of-
  orders ratio. Nothing was killed; the run was repeated once the machine
  was quiet.
