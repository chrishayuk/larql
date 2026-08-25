# CPU-5 — Q4 x Q8: the quality gate, pre-registered

**Frozen before any arm was built or run.** Nothing below was chosen
after seeing a number. Where a band references a measured quantity that
is not yet known (A2's bank KL), the band is written as a RATIO to it and
the anchor is measured, never assumed.

## The question

> Can Qwen3.8-27B tolerate **4-bit weights x 8-bit activations**?

CPU-4Y priced the mechanics and stopped there deliberately:

```
BF16 x F32   420.84 ms/token   51.20 GB   121.7 GB/s   1.00x
Q8   x F32   332.97 ms         27.20 GB    83.4 GB/s   1.26x
Q8   x Q8    224.75 ms         27.20 GB   118.0 GB/s   1.87x
Q4   x Q8    135.10 ms         14.40 GB   106.6 GB/s   3.12x
```

With the MEASURED synthetic->real correction (x1.047) and this build's
17-24 ms non-projection floor, that projects ~158-166 ms/token =
**6.0-6.3 tok/s**, against the 2.797 tok/s Q8 ships today.

None of it is claimable until the arithmetic is shown to preserve the
model. This document is what "preserve" means, written down first.

## Arms

Every arm is teacher-forced over Q-BANK-1 (69 prompts, 7 categories)
against the same BF16 reference bank. One resident model per arm.

| arm | weights | activation | bpw | what it ISOLATES |
|-----|---------|------------|-----|------------------|
| R   | bf16, exact | f32 | 16.0 | the reference bank |
| A1  | bf16, exact | Q8  | 16.0 | **activation quantisation ALONE** |
| A2  | Q8          | f32 | 8.5  | what SHIPS today (CPU-3B) |
| A3  | Q8          | Q8  | 8.5  | both levers, at 8 bits |
| A4  | Q4 int4 blk64 | Q8 | 4.5 | **the CPU-4Y format** |
| A5  | mixed       | Q8  | —    | determined by A4's localisation |

**A1 is the arm the programme would otherwise be missing.** Q4 x Q8
moves two things at once, and CPU-4A already cost this ladder a wrong
conclusion by testing one coupled lever alone. If A4 fails, A1 says
immediately whether the weights or the activations did it. A programme
that only ran A4 would have to guess.

A2 is not decoration: it is the **accepted-cost anchor**. Q8 x F32 is in
production and its numerical cost has already been judged acceptable, so
"how much worse than A2" is the only calibrated question available. An
absolute KL threshold picked from nothing would be a number chosen before
seeing the distribution, which Q-BANK-1's own README refuses to do.

## Instrument controls — these run FIRST and gate everything

The instrument must be shown to work before any arm is interpreted.

1. **Null.** R vs R must be *bitwise* identical, KL exactly 0.000000.
   A harness that reports nonzero KL on identical arms is measuring
   itself. (`project_vindex3_represent_programme` recorded a KL that read
   exactly 0.000000 for the opposite reason — a saturated softmax — so a
   zero must be shown to be a real zero.)
2. **The instrument must FAIL on known-different input.** A2 is run
   first; if A2's bank KL is not clearly above the null, the bank cannot
   resolve a quantisation this size and no A4 result means anything.
3. **A2 must reproduce CPU-3B.** On the 5-token fixture
   (`760,6511,314,9338,369`): rel_rms 5.78e-03, cos 0.999983, KL max
   1.6e-04 nats, 48/48 greedy ids. If the shipped format's own numbers
   have moved, the instrument changed and not the format.
4. **Every kernel proves it computes what its format DENOTES** against a
   portable scalar definition, before it is allowed into an arm — the
   existing `the_qN_kernel_computes_what_the_format_denotes` pattern.
   Against the format's definition, never against the original f32
   weights: at 4.5 bits the quantiser's error would hide almost any
   kernel bug inside a tolerance chosen for it.

## Pre-registered bands

Primary metric is **KL(BF16 || arm) in bits/token** over the bank —
predictive units, not cosine. Let `K2` be A2's measured bank KL mean.

| band | condition | verdict |
|---|---|---|
| NEGLIGIBLE | KL mean <= K2 | no worse than what already ships |
| **ACCEPT** | KL mean <= 2 x K2 | 2x the accepted cost for 2.2x the decode |
| **MIXED REQUIRED** | KL mean <= 5 x K2 | blanket Q4 rejected; search the exception set |
| **REJECT** | KL mean > 5 x K2 | this quantiser is not viable at 4 bits |

Hard gates, independent of K2 — an arm fails if ANY is missed:

- top-1 agreement over the bank **>= 99%**
- top-1 flips at BF16 margin **>= 0.10: exactly zero**
- p99 KL **<= 10 x** A2's p99 KL

The margin-conditioned flip gate is the one that matters. A flip where
the reference separated its first two choices by 0.001 is a different
event from one where it was certain, and reporting a mean hides both.
Uniform NVFP4 was called an unsafe default on Granite on exactly this
evidence: top-1 83.54%, 267 flips, 189 of them at BF16 margin >= 0.01.

## Trajectory and state gates (secondary)

The bank is primary because it is a distribution. These are confirmatory:

- 48-token greedy continuation on the standard fixture (the CPU-3B gate)
- **then 256-token continuations on 4 prompts** — 48 tokens is the
  MINIMUM, not the programme, because Q4 x Q8 is a much larger
  perturbation than Q8 weights alone
- recurrent state rel_rms / cosine across all **48 GatedDelta layers**,
  tracked per step, since a recurrent state accumulates what a logit
  comparison at one position cannot see
- per-layer residual rel_rms vs BF16, split **softmax (16) / GatedDelta
  (48) / FFN**, to localise where error enters

**`48/48 ids identical` is the WEAKEST of these gates and must never be
the headline.** CPU-3B said so about Q8 and it is more true at 4 bits.

## Pre-registered PREDICTION

Stated in advance so that a pass is informative and a failure is not
retro-fitted into a hypothesis.

Q4's quantisation step is `peak/7`; Q8's is `peak/127` — **18.1x at the
same block size**. Logit error tracked weight step closely from bf16 to
Q8 (a dot of random-sign terms preserves the ratio). If that holds, A4
lands near 18x A2, i.e. logit rel_rms ~0.10 and KL far outside every band
above.

> **I expect blanket A4 to FAIL, and the deliverable of this rung to be
> A5.** Block size cannot rescue it: shrinking blk64 -> blk16 lowers the
> expected block peak by only ~1.25x, against an 18.1x deficit.

## The corollary that must travel with a failure

**The 4.5 bpw price tag is not tied to this quantiser.** CPU-4Y priced
BYTES and ARITHMETIC — 14.4 GB/token through integer dots. *Any* 4.5-bpw
format whose codes are small integers with a per-block scale runs at that
same ~135 ms.

NVFP4 is 4.5 bpw, is already built and measured in this repo (PR #299),
and halved MXFP4's layer drift at equal bit budget — E8M0's power-of-two
scale was the Q2 culprit, not 4 bits. Its e2m1 alphabet is

```
{0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}
  = {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12} / 2
```

— **exactly integers**, so it is SDOT-realisable with the x1/2 folded
into the block scale. A 16-element block of e2m1 codes against int8
activations peaks at `12 x 127 x 16 = 24384`, comfortably inside i32.

So if A4 fails on ALPHABET rather than on BIT BUDGET, the response is a
better 4.5-bpw code, not a retreat to 8 bits. That is a different rung
(A6) and it is named here so that it cannot look like a post-hoc rescue.

## Scope limits, stated up front

- `MatrixOperand` carries `class`, `elements`, `stored_bf16` — **not
  depth**. An exception set keyed on layer index cannot be expressed by
  today's planner. If localisation says depth is the axis (and PR #299's
  knee in the last ~5 layers' FFN says it might), that is a planner
  change and it is out of scope for this rung.
- The existing L2 rule ALREADY produces most of the mixed precision this
  programme would otherwise have to invent: `k_proj`/`v_proj` (10.5 MB
  bf16) stay bf16 and the `48 x 5120` delta gates stay f32, because their
  images fit L2. Those are not Q4 candidates and never were.
- One machine, one model. The bands are about Qwen3.8-27B on this M3 Max.

---

# AMENDMENT 1 — after the smoke test, before the Qwen arms

**Nothing above this line has been edited.** A pre-registration that gets
rewritten as results arrive is a lab notebook with the inconvenient pages
torn out. Everything learned since is recorded here instead, with what
prompted it.

## A1 fired, and it changed the programme

The control arm — exact bf16 weights against a per-tensor int8
activation — came back at rel_rms **4.8e-01** on the Granite smoke test.
That is not a quantisation cost; it is a destroyed activation, and it
happens with the weights held EXACT. Measured cause, from the residual
stream itself:

```
layer   peak      rms     peak/rms   effective bits
000     1.758   0.5085        3.5         5.20
036    36.326   1.0059       36.1         1.81
040    78.687   2.7750       28.4         2.16
```

One scale over a vector whose peak is ~30x its RMS spends most of the
int8 range on a handful of outlier channels. Blocking the activation on
the weights' own boundaries confines an outlier to its own block:

```
                    per-tensor   per-block
exact weights          0.476       0.047
Q8 weights             0.471       0.029
Q4 weights             0.709       0.439
```

**Consequence for the arms.** "Q8 activation" was never a sufficient
description of the arithmetic. The pre-registered arms A1/A3/A4 measured
the per-TENSOR variant; the blocked variants A1b/A3b/A4b are added
BESIDE them, not in place of them, and the per-tensor numbers stand as
measured. The blocked arms cost one extra multiply per 64 elements and
leave `SDOT` untouched, so CPU-4Y's 135.10 ms price survives.

## Two confounds found and fixed in the instrument itself

- **A1 was not matched to A4's operand set.** `for_resident` reads
  bytes, and an operand is bf16 either because its image fits L2 or
  because A1 swapped a streaming Q8 operand back — indistinguishable.
  A1 was therefore quantising the activation on MORE operands than the
  arm it explains, and read WORSE than Q8 x Q8 while holding exact
  weights. The size rule is now re-applied in the observation; ledgers
  confirm both arms route 121 calls.
- **Granite is MoE.** Quantisation flips router decisions, a discrete
  high-variance event that dominates a 5-position sample and made the
  arm ordering non-monotonic. Granite is a SMOKE TEST only. Qwen3.8-27B
  is dense — no router, no experts — so the subject does not have this.

## DISCOVERY and VALIDATION are now separate banks

The original spec named one bank. That is unsafe for what comes next: if
the same 69 prompts are used to locate where Q4 fails, to choose which
operands to protect, AND to declare the result acceptable, the
acceptance number is partly a training number. The effect is not small
when the search is over operator families and the metric is a tail
statistic.

```
quality-bank-1   DISCOVERY   look freely; order families; derive a candidate
  cpu5 screen      20% subset, fixed by rule, for the family sweep only
quality-bank-2   VALIDATION  frozen 2026-08-25, scored ONCE, on the final candidate
```

`quality-bank-2` is 69 prompts in the same seven categories, 1946
positions, **frozen before any exception set was searched**. Its
disjointness from bank 1 and from the already-spent SENSITIVITY-1B'
calibration set is verified by `freeze.py`, not asserted in a comment —
exact id, whitespace/case-normalised text, and a 40-character prefix
near-duplicate scan. It refused the first draft: two exact text
collisions and five near-duplicates, now replaced.

```
text  digest a3fb510b9c8945199d78c9bc50e1d1aa40f26e8d678238f716d38434b2e7181e
token digest a757240ce0e083fd6eb0ecde32e7c5a81c3b6e6179570d53d9b2d765e8afa868
```

## The rescue rungs, and their one-variable rule

Blanket Q4 is a hypothesis. If it fails, the question becomes the
smallest set of operands that must be RESTORED, and the answer is
searched ONE AXIS AT A TIME rather than by trying combinations until
something passes:

```
R0  all          blanket Q4
R1  attn,ffn     output head restored
R2  ffn,head     attention restored
R3  attn,head    FFN restored
```

**A restored class falls back to Q8 in the SAME integer domain** — same
activation, same accumulator, only the weight bits change. A rescue that
also reverted the activation would move two variables, which is exactly
how CPU-4A concluded Q4 was dead when it had only shown Q4 x F32 was.

The reported order above is NOT the order to read them in. Each rung's
marginal value is what is being measured, and the ordering comes from
the numbers.

**Scope limit, stated before the search.** The only axis today's seam can
express is the matrix CLASS: `prepared::resolve` deliberately refuses to
hand the policy an `OperandRef`, because resolving operands by name is
the one thing the seam forbids and a name-keyed exception set would be a
per-model recipe rather than a policy. So this search cannot say "the
last five layers' FFN" — which is where a 4-bit knee has already been
found once, on another model. If class is too coarse, the answer is a
planner change (depth and shape are physical facts and could be carried
honestly), and that is a separate rung.

## Every rung reports BOTH halves

A quality result alone cannot choose an exception set, because the whole
question is what the restored bytes cost:

```
bytes restored     Q4 -> Q8 traffic added
predicted ms       from the byte census, at CPU-4Y's measured rates
KL / top-1 / flips at margin
continuation drift
```

The cost model is gated on reproducing CPU-4Y's own table from CPU-4Y's
own byte counts to within 1%, and on moving proportionally when the
bytes move. It predicts 140.6 ms real for a 14.40 GB Q4 plan and
6.0-6.3 tok/s against the 17-24 ms floor, which is CPU-4Y's own
extrapolation recovered from the model rather than restated.

## The invariant this rung produced

> **A resident representation CONSTRAINS which plans are possible. It
> does not DETERMINE which plan executes.**

Identical Q8 bytes are consumed by a widening f32 GEMV and by `SDOT`, at
81.7 and 121.0 GB/s and with different numerics. `PhysicalProjectionPlan`
therefore names `WeightRep x ActivationRep -> AccumulatorRep`, and the
activation carries its scale GEOMETRY because `Q8[tensor]` and `Q8[64]`
differ by an order of magnitude in logit error. Pinned as a test, because
the tempting simplification is to infer arithmetic from residency again
and it would pass every gate on a machine running the default arm.

---

# AMENDMENT 2 — the selection rule, frozen before the family sweep

Written while `bf16xq8b` was at 5/69 and `q8xq8b` and `q4xq8b` had not
started. **No rescue rung has been run.** The rule below therefore
cannot have been chosen to favour a plan that had already been seen,
which is the only thing that makes a rule worth writing down.

## The anchor is now MEASURED

```
A2 — shipped, Q8 x F32, 1,740 positions over 69 prompts

  KL bits/token   mean 0.00016   median 0.00011   p95 0.00048   p99 0.00097
  top-1           99.60%   7 flips
    at BF16 margin >= 0.01      1
    at BF16 margin >= 0.10      0
  max |dlogit|    mean 0.1326    p99 0.3615
```

So `K2 = 0.00016 bits/token`, and the bands frozen before any arm ran
resolve to absolute numbers for the first time:

```
NEGLIGIBLE        KL mean <= 0.00016
ACCEPT            KL mean <= 0.00032      (2 x K2)
MIXED REQUIRED    KL mean <= 0.00080      (5 x K2)
REJECT            KL mean >  0.00080
```

The anchor clears its own hard gates — zero flips above a 0.10 margin,
worst flip margin 0.0134 — so "as good as what ships" is a real bar and
not a permissive one.

## Quality gates a candidate must clear (Bank 1, FULL, not the screen)

All four, jointly. Any miss disqualifies the plan.

```
G1  KL mean          <= 0.00032 bits/token     (the ACCEPT band)
G2  KL p99           <= 0.00970 bits/token     (10 x A2's p99)
G3  top-1 agreement  >= 99.00%
G4  flips at BF16 margin >= 0.10  ==  0
```

G4 is the one that matters and the one a mean would hide. A flip where
the reference separated its first two choices by 0.001 is a different
event from one where it was certain.

## THE SELECTION RULE

Applied mechanically, in order, with no discretion at any step:

```
1. ELIGIBLE   = plans clearing G1-G4 on the FULL Bank 1.
2. CHOOSE     = the eligible plan with the lowest PREDICTED REAL ms.
3. TIE (<1%)  = fewer restored classes (a simpler policy wins).
4. TIE still  = fewer physical bytes per token.
5. TIE still  = REFUSE to choose. Report the tie and escalate.
```

Step 5 exists so that "this one looks nicer" has nowhere to enter.

**Dominance check, applied after selection.** The chosen plan must beat
`Q8 x blocked-Q8` on predicted real ms. If it does not, the exception
set has eaten the entire Q4 win and the honest result is that **Q8 x Q8
is the frontier** — 27.20 GB at 121.02 GB/s, ~235 ms real, ~3.9-4.0
tok/s — which is still 1.4x what ships today and must be reported as the
outcome rather than buried as a failed Q4 programme.

**If no plan is eligible**, the class axis is exhausted and there are
exactly two honest continuations, named here so neither looks like a
rescue invented afterwards:

- **CPU5-B**: extend the policy's facts to operator class + depth +
  shape. Justified by evidence that class is too coarse, not by
  expressiveness being nice to have.
- **REJECT**: 4-bit weights are not viable for Qwen3.8 on this axis.

## Bank 2 is a ONE-SHOT instrument

```
Bank 1  full         all diagnostic evidence; look freely
Bank 1  20% screen   the family rescue sweep ONLY
Bank 1  full         confirm the ONE selected candidate
Bank 2               ONE acceptance evaluation, on that candidate alone
```

**If the candidate fails Bank 2, Bank 2 is SPENT.** It is not re-run
against a tuned plan, and no candidate is selected using it. The
permitted responses are to reject CPU-5, or to freeze a Bank 3 *before*
the next design iteration begins. Re-scoring a second candidate on Bank 2
would make it a discovery bank retroactively and destroy the only
untouched evidence the programme has.

## What every rescue rung reports

Absolute numbers do not order families; marginal ones do.

```
d bytes/token        Q4 -> Q8 traffic added
d predicted ms       from the byte census, CPU-4Y rates
d KL mean, d KL p99
d dNLL
d flips at margin >= 0.01 and >= 0.10
d continuation drift
```

and, as EXPLANATORY rather than selection quantities:

```
KL recovered per added GB
KL recovered per added ms
```

These describe numerical criticality as a physical-planning fact — which
operands buy the most behaviour per byte — and that is the durable
output. The selection rule above deliberately does not use them: a ratio
is a good explanation and a bad tie-breaker, because it rewards tiny
denominators.

## Interaction, tested rather than assumed

The three Qwen arms are also read INCREMENTALLY, not just absolutely:

```
activation cost           A1b - A2ref     BF16 x Q8[64]  vs  BF16 x F32
Q8 weight cost            A3b - A1b       under one activation
Q4 weight cost            A4b - A1b       under one activation
```

KL is not additive, so agreement is not expected to be exact. The
question is whether the combined arm is FAR worse than its isolated
components predict. If it is, the weight and activation perturbations
interact numerically, and mixed-precision selection cannot be reasoned
about one operand at a time — which would be a finding in its own right,
and would invalidate the per-class marginal analysis above rather than
merely complicate it.

---

# RESULTS — the four pre-registered arms, Q-BANK-1 full

1,740 positions, 69 prompts, reference = exact bf16 weights x f32
activation. Gates and bands are AMENDMENT 2's, frozen before any of this
was run.

```
arm             pos   KL mean    KL p95    KL p99    top-1  flips  >=.01  >=.10     dNLL
shipped        1740   0.00016   0.00048   0.00097  99.60%      7      1      0  +0.00141
bf16xq8b       1740   0.00061   0.00190   0.00324  99.02%     17      6      0  +0.00099
q8xq8b         1740   0.00074   0.00231   0.00487  98.91%     19      7      0  +0.00262
q4xq8b         1740   0.05384   0.18074   0.35731  90.98%    157    130     38  +0.04618

                G1    G2    G3    G4
shipped         ok    ok    ok    ok   PASS — NEGLIGIBLE
bf16xq8b      FAIL    ok    ok    ok   FAIL — MIXED REQUIRED
q8xq8b        FAIL    ok  FAIL    ok   FAIL — MIXED REQUIRED
q4xq8b        FAIL  FAIL  FAIL  FAIL   FAIL — REJECT
```

## 1. The pre-registered prediction was right, and for the stated reason

AMENDMENT-free prediction, written before any arm ran: *"Q4's step is
`peak/7`, Q8's is `peak/127` — 18.1x. If logit error tracks weight step,
A4 lands near 18x A2."* Subtracting the activation floor:

```
Q8 weight cost                     0.00016
Q4 weight cost (by subtraction)    0.05323     332.7x
step ratio 127/7 = 18.1x, SQUARED  329x
```

**KL scales as the SQUARE of the quantisation step**, to within 1%. That
is worth more than the verdict, because it turns the exception search
from an empirical sweep into arithmetic.

## 2. Which is why no exception set rescues this format

If a class's contribution is roughly linear in the fraction of weights
left at Q4 — which it is, since independent operand errors add in
quadrature and KL goes as the square:

```
blanket            166x over G1
FFN only   (~70%)  116x
attention  (~28%)   47x
head only   (~9%)   15x
Q4 fraction that fits the WHOLE budget:  0.60%
```

### This is a PREDICTION, and it has a named falsifier

The table above assumes damage is **uniform per byte** — that a class
carrying 70% of the bytes carries 70% of the KL. That assumption is
exactly what this repo's SENSITIVITY ladder found to be false in
general: *local consequence is not output sensitivity, and `down_proj`
is the counterexample.* If damage were instead CONCENTRATED — say 99% of
it in 5% of the bytes — an exception set would reopen.

So the claim is stated as a prediction with a one-run test:

```
PREDICT   R3 (FFN restored to Q8, ~70% of bytes) lands near
          0.70 x 0.05323 = 0.037 on the screen.

If it does        uniform-sensitivity holds, the arithmetic above
                  applies, and no exception set rescues this format.
If it is FAR
lower (<0.005)    damage is concentrated, the class axis is alive, and
                  R0-R3 must be run properly.
```

One rescue rung on the 20% screen settles it, which is why exactly one
is worth running rather than none or four.

### R3 RAN. The prediction was directionally right and quantitatively wrong.

```
screen, activation floor (block 64)        0.00047
R0  blanket Q4         KL 0.05013   weight damage 0.04966   top-1 91.48%
R3  FFN -> Q8          KL 0.02770   weight damage 0.02723   top-1 92.33%
```

Restoring the FFN — about 70% of the bytes — removed only **45%** of the
damage, not the 70% uniform sensitivity predicts:

```
damage remaining after restoring FFN     0.548
uniform prediction (attn+head ~30%)      0.300
=> attention+head carry 1.83x the damage PER BYTE that uniform assumes
```

So sensitivity is NOT uniform, and it is concentrated in the opposite
place from the one PR #299 found on Granite — there the knee was the
last few layers' FFN; here it is the attention class, which on Qwen3.8
contains the 48 GatedDeltaNet projections. Different architecture,
different knee: a recipe carried across models would have been wrong.

**The conclusion survives.** The falsifier's threshold was
`< 0.005 => concentrated, axis alive`; measured 0.02770 is **6x** that.

### What is MEASURED, and what is inferred

Two points are measurements. Everything else below is arithmetic on
them, under an additivity assumption that the interaction test supports
but does not prove for class subsets:

```
MEASURED    blanket Q4          KL 0.05013   weight damage 0.04966
MEASURED    FFN restored        KL 0.02770   weight damage 0.02723
            => FFN's own damage 0.02243 (70% of bytes)
INFERRED    attention ~0.01906, head ~0.00817  (split by byte share, NOT measured)
```

Every class split, with what each BUYS — because a rescue that clears
nothing is not interesting, and neither is one that saves nothing:

```
                     KL       x gate   traffic saved
blanket (all Q4)   0.05013     157x        50.0%
FFN+attn Q4        0.04196     131x        45.5%
FFN only Q4        0.02290      72x        35.0%
attn+head Q4       0.02770      87x        15.0%
attention only Q4  0.01953      61x        10.5%
head only Q4       0.00864      27x         4.5%
```

**The honest statement is that class-level rescue is decisively
implausible under the measured bracket** — not that every row above is a
measurement. Only the first two are. But the smallest class that could
carry Q4 at all (the head, 9% of bytes) is still ~27x the gate while
saving 4.5% of projection traffic, and the two ends of the range are
measured, so no plausible split of the middle changes the verdict.

R1 and R2 were not run because at 27-157x no additional measurement
changes the decision — only the mechanism, and the mechanism question is
now about the CODE.

My own pre-registered number was 0.037 against a measured 0.02770 — a
25% over-prediction. The uniform-sensitivity model is good enough to
bracket the question and not good enough to price an operand.

The lever being the CODE rather than the exception set is the same
statement: at 332.7x, an exception set has to remove 99.4% of the damage,
and that is a lot to ask of any axis.

## 3. The binding constraint is the ACTIVATION, not the weights

`bf16xq8b` holds the checkpoint's own weights and still fails G1 by
1.9x. Every integer arm inherits that floor, so **no weight format can
clear the gate until the activation does**, and a programme that had
gone looking for a better 4-bit code first would have been optimising
the wrong operand. This is what A1 was added for, and it is the second
time it has changed the programme's direction.

## 4. Interaction: additive, mildly cancelling

```
activation alone (BF16 x Q8[64])   0.00061
weights alone    (Q8 x F32)        0.00016
sum of parts                       0.00076
measured together (Q8 x Q8[64])    0.00074    ratio 0.97x

error-vector cosine, raw logit space        -0.2176
error-vector cosine, softmax TANGENT space  -0.2237
```

The cosine survives projecting out the shift-invariant direction, so the
mild anti-correlation is real rather than common-mode. Consequence: the
perturbations are near-independent, isolated arms slightly OVERSTATE
combined damage, and per-operand marginal reasoning is valid and
conservative. Mechanism unexplained; banked, not claimed.

## 5. A third of the activation's logit error is invisible to the softmax

```
common-mode share of the activation error: 30.7%
```

Logits are shift-invariant, so that share changes no probability at all.
Any claim resting on raw `|dlogit|` overstates this representation's
damage by about a third — which is why the gates are in KL and the
margin-conditioned flip count, not in logit distance.

## 6. What this rung did NOT establish

- **Nothing about a better 4.5-bpw code.** The corollary registered in
  the original spec still stands untested: CPU-4Y priced BYTES and
  ARITHMETIC, and NVFP4's e2m1 alphabet is
  `{0,+-1,+-2,+-3,+-4,+-6,+-8,+-12}/2` — integers, so SDOT-realisable at
  the same 4.5 bpw. That is arm A6 and it has not been run.
- **Nothing about the activation at any geometry but 64.** The block
  sweep is a separate measurement.
- **Nothing on quality-bank-2.** It remains untouched and unspent.

---

# AMENDMENT 3 — CPU5-K, the kernel ladder, and K3's gate

Frozen BEFORE K3 was written. The representation is settled and is not
in question here; only its implementation is.

## The accepted representation

```
Q8 weights x asymmetric Q8[16] activation -> I32 -> F32

KL mean     0.000286992   against G1 0.000316368   (9.3% headroom)
KL p99      0.00194       top-1 99.5402%
flips at BF16 margin >= 0.10: 0   (worst flip margin 0.04157)
```

First plan in this programme to clear all four frozen gates. It is not
shippable: it runs at 757 ms/token against production's 348 ms.

## The ladder so far

```
asym16 original          757 ms   3.39x the memory wall
K1 precomputed index     868 ms   3.89x   FALSIFIED
K2 batched reductions    632 ms   2.83x   bit-identical, banked
target (beat shipped)    348 ms
memory wall              223 ms   (sym64 reaches 1.19x wall)
```

**K1 is a clean negative.** Precomputing `SUM(q)` removes a second
`SDOT` but adds ~12% compact traffic and a THIRD memory stream. A
`vdotq` against a ones-vector reads codes already in registers; the index
does not. Corroborated by the cost surface: at block 64, doubling the
SDOTs costs only +14%, so `SDOT` throughput was never the bottleneck.
Kept behind `LARQL_CPU_WEIGHT_INDEX`, off by default.

## K3 must attack the BLOCK GEOMETRY, not the asymmetry

The decisive arithmetic, before writing anything:

```
sym64        266 ms
sym16        484 ms      <- the block-16 tax, on symmetric arithmetic
asym16+K2    632 ms
shipped      348 ms
```

Removing ALL asymmetric-only overhead would land at sym16's 484 ms —
still worse than production. **So K3 cannot succeed by making asymmetry
cheaper; it has to remove the block-16 execution pathology itself.**

The design does: replace the per-block vector→scalar crossing with
vector-domain accumulation, one reduction per ROW instead of per block.

```
was:  4 blocks -> packed i32 -> 8 lane extracts -> 8 scalar ops -> scalar acc
now:  4 blocks -> packed i32 -> cvt to f32 -> vector scales/mids
                             -> vector FMA acc -> ONE reduction per row
```

## K3 is run as TWO arms, and sym16 is the mechanism check

Same binary, same implementation, measured immediately and mechanically:

```
K3-sym16    Q8 x symmetric  Q8[16]
K3-asym16   Q8 x asymmetric Q8[16]
```

```
sym16 collapses 484 -> ~300     the block-geometry tax is genuinely gone
asym16 then close to sym16      the asymmetric correction is cheap enough
sym16 barely moves              K3 cannot rescue asym16 at any quality
```

This costs minutes and gates a 50-minute bank run.

## FROZEN performance gate — K3 asym16, real decode

```
<348 ms       GO — beats shipped; spend a full Bank 1
348-400 ms    interesting, NOT a production candidate
400-500 ms    insufficient
>500 ms       the K3 mechanism largely failed

~300 ms       strong
~270 ms       essentially solves the block-16 overhead
~250 ms       exceptional
```

**Bank 1 is not re-run above 348 ms** unless the kernel timing points at
an obvious one-step cleanup, which must be named before it is taken.

## K3 leaves the bit-identical regime, deliberately

K1 and K2 were bit-identical, which is why neither needed new numerical
evidence. K3 reassociates the accumulation of already-computed block
contributions, so it cannot be.

Two instruments, and only one of them is acceptance:

```
K2-vs-K3 direct control   max/rel movement on the same inputs.
                          Expect ~1e-7. A 1e-3 reading is an
                          IMPLEMENTATION BUG, caught before the bank.
                          NOT an acceptance substitute.

full Bank 1               authoritative. Re-establishes G1-G4 for the
                          arithmetic that will actually run.
```

Reassociating the sum of block contributions is a far smaller
perturbation than the quantisation that produced the KL in the first
place — but that is a reason to expect a pass, not a reason to skip
measuring one.

## Sequence, and Bank 2's continued silence

```
K3 implementation -> mechanical gate -> (if <348) full Bank 1
                  -> (if PASS) candidate selected under the frozen rule
                  -> Bank 2, ONCE
```

If K3 reaches 320 ms and Bank 1 fails, Bank 2 stays pristine. Bank 2 is
spent only on something that would actually ship if independently
validated.

## The lesson this ladder is producing

```
activation block geometry  <->  numerical quality  <->  kernel geometry
```

Block 16 was numerically NECESSARY and physically disastrous under a
scalarised implementation. So `ActivationRep::Q8 { span: Block(16) }` is
not sufficient for a planner: its COST depends on whether a kernel can
consume that geometry efficiently. **Representation geometry and kernel
geometry have to be co-designed**, and a cost model indexed on the
representation alone will keep being wrong — as this one was, by 3.2x.

---

# AMENDMENT 4 — K3 measured, and K4's frozen gate

## K3, both arms

```
arm                          before  after K3   gain
sym16 (mechanism check)         484       279   1.73x
asym16 (the candidate)          632       509   1.24x
```

**The mechanism check passes decisively.** `sym16` is now within 5% of
`sym64` (266 ms) and at 1.25x the memory wall, so the block-16 execution
pathology is solved: removing the per-block vector-to-scalar crossing was
the right diagnosis.

**The candidate arm fails its frozen band.** 509 ms is in
`>500 — the K3 mechanism largely failed`. Both readings stand. What the
pair says is that the asymmetric correction is now the DOMINANT term:
230 ms, +82% on top of sym16, up from +56% before K3 — the base got
faster and the correction did not.

## K4 — the named one-step cleanup

AMENDMENT 3 permits a Bank-1 run above 348 ms only after "an obvious
one-step cleanup, which must be named before it is taken". Named:

**K1 was not wrong about WHAT to remove, only about HOW to fetch it.**
It loaded the weight-sum index as 320 scalar `i16` reads a row — a third
memory stream of tiny dependent loads — and lost 111 ms to buy 26 ms of
traffic. Under K3's four-block geometry the same index is ONE 64-bit
vector load:

```
now:   4 useful SDOT + 4 correction SDOT + 6 vpaddq + 2 cvt + 2 vfma
K4:    4 useful SDOT               + 3 vpaddq + 2 cvt + 2 vfma
                                   + 1 vld1_s16 + 1 vmovl_s16
```

K1 could not have been evaluated properly before K3, because the noise it
was hiding under was larger than the effect.

**K4 is expected to be BIT-IDENTICAL to K3**, not merely close: the index
holds exactly the integers the correction SDOTs compute, and every float
operation after that is unchanged. That is asserted, not assumed — and it
means one Bank-1 run covers both.

## FROZEN gate — K4 asym16, real decode

```
<=300 ms      exceptional
300-330       PASS — production candidate
330-348       PASS — beats shipped, Bank 1 justified
348-400       mechanism helped, candidate still dominated
>400          K4 insufficient
```

**The hard line stays 348 ms.** A 365 ms result is not "close enough";
it is a candidate that loses to what already ships.

## Control the A/B tightly

```
asym16, index OFF   (K3)
asym16, index ON    (K4)
```

One binary, and the LEDGER checked — the K4 arm must be seen consuming
weight codes, weight scales, the sum index, and the activation, rather
than quietly falling back through another path. K1's lesson is precisely
that a mathematically smaller operation can lose through plumbing, so
the plumbing is what gets inspected.

`sym16` is not perturbed: it needs no index, and changing two things at
once is how CPU-4A got its wrong answer.

## If K4 misses badly

At ~390 ms the conclusion is that asymmetry's bookkeeping — not an
avoidable SDOT — is the cost, and this formulation is finished. The
response would then be on the REPRESENTATION side (block-32 asymmetric,
mixed activation geometry by operator class, offline scale migration),
and none of it is started now. **K4 has earned exactly one shot**,
because it follows from the measured decomposition rather than from a
list of things that might help.

## What a passing K4 would make the physical format

```
Q8WeightBlock64 {
    codes[64]
    scale
    subblock_sum[4]        <- exists only to serve one legal consumer:
}                             Q8[64] x asymmetric-Q8[16]
```

A materialised auxiliary index for an execution plan, discovered
empirically rather than designed in — and only justified where a
consumer that needs it is the one being run.

---

# AMENDMENT 5 — K4 falsified, and K5's frozen gate

## K4: the index is consumed, and it changes nothing

```
K3 index-OFF   518 ms   27.02 GB   residency 32.55 GB
K4 index-ON    516 ms   30.20 GB   residency 35.72 GB
```

Traffic and residency both moved, so the index is genuinely being read —
this is not the silent fallback K1 taught us to check for. Removing four
correction `SDOT`s and three `vpaddq` per group, and paying 12% more
traffic, bought **0.4%**.

So the asymmetric correction's 230 ms is NOT the correction `SDOT`s.

## Where it actually is

`fold_scales` runs PER OUTPUT ROW, materialising a 320-entry `f32`
buffer; asymmetric materialises two.

```
write 2560 B (scales+mids), read back 2560 B
= 5120 B/row x 5120 rows = 26.2 MB per 5120x5120 matrix
against 26.2 MB of Q8 codes -> +100%
```

**But the framing "hidden traffic" is wrong, and the correction matters.**
Those buffers are 2560 B — they live in L1, not DRAM. The cost is
load/store OP THROUGHPUT: ~320 vector load/store operations against ~320
`SDOT`s per row, roughly DOUBLING the inner loop's op count.

Which is exactly why K4 did nothing. It removed vector ALU operations
and left the load/store operations untouched, so the binding port
pressure never moved. Two hypotheses failed for one reason: **the
arithmetic was never the cost.**

Also corrected: `fold_scales` does NOT allocate per row. The `Vec`s are
hoisted out of the row loop and `clear()` retains capacity, so
allocation churn is not the mechanism — it is the round trip itself.

## K5: delete the buffers

`ascale` and `amid` are **row-invariant** — they depend only on the
activation. Only the weight scale varies by row. And at `ablock` 16 with
`block` 64 a group of four activation blocks IS exactly one weight
block, so the weight scale is a CONSTANT within a group:

```
was:  per row, materialise fold[320] = wscale[b/4] * ascale[b], then reload it
K5:   vmulq_n_f32(vld1q_f32(&ascale[b]), wscale[b/4])   — one broadcast, in register
```

**Expected to be BIT-IDENTICAL** to K3/K4: `ws * ascale[b]` is the same
f32 whether it is computed into a buffer first or in a register. So no
new Bank-1 evidence is needed beyond the run K3 already requires.

## FROZEN gate — K5 asym16, real decode

```
<348 ms      PASS — beats shipped, Bank 1 justified
348-400      useful, still dominated
400-475      mechanism partly confirmed
>475         hypothesis largely falsified
```

Two arms again, and the signatures are diagnostic:

```
sym16 modest gain, asym16 large gain    asymmetric's SECOND buffer is the cost
both improve similarly                  materialisation is general
neither moves                           fold_scales was not the cause
```

`sym16` already proves the underlying Q8[16] kernel runs at **279 ms**,
so asym16 does not need a new integer kernel. It needs to stop paying
~230 ms for something sym16 does not do.

## What the ladder has established

```
K1  scalar precomputed sums    SLOWER   -> removing SDOTs was not enough
K2  batch integer reductions   +20%     -> vector->scalar crossing mattered
K3  stay vector through floats  1.73x   -> block-16 geometry SOLVED (sym16 279)
K4  vector precomputed sums     ~0%     -> asymmetric SDOTs were not the cost
K5  delete folded buffers       ???     -> tests load/store op pressure
```

Three falsified hypotheses, each excluding arithmetic further and
pointing at execution-generated memory movement. The cost ledger counts
STORED MODEL BYTES and nothing else, so a kernel that reads 27 GB of
weights and privately round-trips a scratch buffer per row is priced as
a 27 GB kernel. A physical planner's cost model needs at least:

```
persistent representation traffic
+ transient/materialisation traffic
+ arithmetic and port pressure
```

That is the second time this rung has caught the same class of error —
the first was the cost model being indexed on representation without
kernel geometry, and wrong by 3.2x.
