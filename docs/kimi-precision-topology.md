# Kimi Linear 48B — the PRECISION-1 topology

**The first complete REPRESENT chain: a per-layer precision frontier,
a composed map earned against a frozen behavioural contract at
authority scale, and an in-session decode benchmark.** Closed
2026-08-30 on Kimi-Linear-48B-A3B-Instruct
(`~/chris-models/Kimi-Linear-48B-A3B-Instruct.aligned.vindex3`).

PRECISION-1 mapped one byte family (routed experts). The second
chapter below — **the whole-decoder topology** — extends the map to
every byte family the decoder reads, under a contract calibrated
afterwards from measured consequences.

Contract: **kimi-logit-v3, frozen at `ce6e87a3`** — six authority
criteria (positions ≥ 4096, KL p99 ≤ 1e-3, covered mass ≥ 0.6, top-1
mass displaced ≤ 5e-2, top-10 mass p99 ≤ 1e-1, route mixture mass
p99 ≤ 0.15 / max ≤ 0.25). Evidence bank: 256 × 32-position
teacher-forced sequences from real prose, identity
`a73f437aeb1d…` (manifest + per-file SHA256 in
`~/chris-models/kimi-quality-bank-provenance/`).

## The earned topology

```text
layers 0–23   BF16      (Q8_0 refused at L19 and below; see frontier)
layers 24–26  Q8_0      composed map, PASSED v3 at 8192 positions
```

Composed authority (8192 positions): **kl p99 4.153e-4** (2.4×
headroom), covered 0.631, top-1 mass max 1.56e-2, top-10 mass p99
6.15e-2, route mass p99 0.113. Report
`kimi_map-l24-26q80-8192_report.json`. Counts at that scale — 23
top-1 flips, 1041 top-10 changes, 163 route flips — would have
REFUSED under the count-based v1/v2 gates; the consequence contract
is what makes 8192-position authority reachable at all.

Bytes: expert banks for L24–26 drop 3 × 3.62 GB (BF16) → 3 × 1.93 GB
(Q8_0), **5.1 GB saved**, all other tensors untouched.

Decode (in-session, interleaved blocks, min-of-N floors, two
sessions, AC power): baseline BF16 **35.46 / 35.52 tok/s** vs
candidate **36.44 / 36.27 tok/s** → **1.021–1.028×**; GPU time
27.0 → 26.4–26.5 ms/token (−2.2%, consistent with the roofline for
the byte reduction). Harness:
`opplan/exec/tests/q2a_decode_bench.rs`.

## The per-layer Q8_0 frontier (256-position diagnostics)

```text
layer  kl p99      verdict
 25    2.305e-4    PASS   (confirmed at 8192: 1.446e-4)
 24    8.804e-5    PASS
 23    7.049e-4    PASS
 22    1.569e-4    PASS
 21    8.988e-4    PASS   (non-monotone inside the band)
 20    4.258e-4    PASS   ← frontier
 19    1.144e-3    FAIL   kl alone (14% over)
 18    8.906e-4    FAIL   top10_mass alone (0.1042) — kl passes
 16    4.962e-3    FAIL   kl + route_mass
 13    8.001e-3    FAIL   kl + route_mass
  6    1.665e-2    FAIL   kl + top10_mass + route_mass
  1    4.665e-2    FAIL   kl + top10_mass + route_mass
 26    1.161e-4    PASS   (Q8_0; Q6_K also passed individually at
                           8.53e-4 @ 8192 — see below for why the
                           map still prefers Q8_0 here)
```

The failure mode rotates at the edge (L18/L19 fail single criteria)
and enriches with depth. L18/L19 are the balanced-profile seeds.

## Composition is the finding

Individually admissible layers do not compose, and no scalar
correction fixes that:

```text
map                       measured    raw Σ      α       verdict
V1 {20-25:Q8, 26:Q6}      2.489e-3    3.2e-3     0.77    FAIL kl
V2 {22-25:Q8, 26:Q6}      1.110e-3    ~2.0e-3    0.55    FAIL kl
V3 {22,24,25,26:Q8}       1.124e-3    5.92e-4    1.90    FAIL kl  (super-additive)
V4 {24,25,26:Q8}          2.262e-4    4.35e-4    0.52    PASS → 8192 PASS
```

* V2 and V3 land nearly identical composed KL with raw sums 3.4×
  apart: the composed tail is dominated by interaction, not member
  costs.
* V3 − V4 attributes it: **adding L22 to the late band costs
  ~9.0e-4 composed against 1.57e-4 solo (~6×)**. The mechanism is
  the routing cascade — L22's hidden-state displacement flips
  routes from L23 onward in every map containing it.
* **L26 composes almost free** (last routed layer — no downstream
  router to amplify it), which is why the map takes Q8_0 there
  (1.16e-4) over the individually-cheapest Q6_K (8.53e-4): *the
  best representation for a scope in isolation need not belong to
  the best admissible map.* A per-layer recipe cannot discover
  this; only a composed gate can.
* Diagnostic→8192 drift on the winner was +84% (2.26e-4 →
  4.15e-4): the ~7e-4 promotion margin for a 1e-3 gate is not
  conservatism theatre.

## Reproduction

```text
bank      scripts/kimi_quality_bank_export.py  (deterministic; verify
          per-file SHA256 against provenance SHA256SUMS.json)
compile   LARQL_Q6_MAP="24-26:Q8_0" → represent/compile_real_tests.rs
quality   q2a_teacher_forced (LARQL_Q2A_SEQUENCES=8 diagnostic,
          256 authority)
speed     q2a_decode_bench (both arms in-process, interleaved)
evidence  ~/chris-models/kimi-quality-bank-provenance/
```

---

# The whole-decoder topology — MLA-SHARED-1

**Every byte family in the decoder measured against one frozen
behavioural contract, and the first family REFUSED on economics
rather than on capability.** Closed 2026-08-31 on the same model and
the same two banks.

PRECISION-1 above answered *which layers* of one family (routed
experts) tolerate a cheaper representation. This chapter answers the
next question: *which families*, and at what price per byte. Five
byte families make up the decoder's active read per token; four are
now admitted under one composed map and one is supported by the
engine but declined by the economics.

## The active byte ledger, BF16

Per decoded token, at BF16, geometry from the container
(27 layers, layer 0 dense; 256 experts of 3 × 2304 × 1024 per MoE
layer, 8 routed per token; MLA at layers {3,7,11,15,19,23,26}, KDA at
the other 20; vocabulary 163,840):

| family | per token | share |
|---|---|---|
| routed experts (8 of 256 × 26 layers) | 2.944 GB | 49.2 % |
| KDA projections (20 layers × 75.5 MB) | 1.510 GB | 25.2 % |
| output head (2304 × 163840) | 0.755 GB | 12.6 % |
| MLA projections (7 layers × 58.2 MB) | 0.408 GB | 6.8 % |
| shared experts (26 layers × 14.2 MB) | 0.368 GB | 6.1 % |
| **total** | **5.985 GB** | |

The checkpoint is 97 % experts; the *token* is not. That asymmetry is
why a whole-decoder representation map beats an expert-only one, and
it is the reason this chapter exists.

## The five families, measured

Every cell below is one scope re-encoded to Q8_0 against a BF16
baseline arm otherwise byte-identical, judged on the 256 × 32
selection bank, `kl p99` from the frozen consequence metrics:

| family | cheapest cell | costliest cell measured | shape with depth |
|---|---|---|---|
| routed experts | L26 1.16e-4 | L1 4.67e-2 | late plateau, enriches with depth |
| KDA projections | L22 8.07e-5 | L13 1.49e-2 | late plateau, NON-MONOTONE teens |
| output head | 7.13e-4 (no depth axis) | — | direct consequence, no mediation |
| MLA projections | L23 5.75e-5 | tower 2.61e-2 | late plateau, **sharp interior cliff** |
| shared experts | L25 1.24e-3 | — | (economically refused before mapping) |

The KDA row's qualifier is measured, not hedging: its band from L16
to L18 does not order with depth (L16 3.98e-4 sits BELOW both L17
2.36e-3 and L18 1.25e-3), the same non-monotonicity the expert
frontier showed at L21. The plateau and the eventual cliff are robust;
the exact edge inside the band is not, which is why a frontier is
found by measuring cells rather than by bisection.

**The same shape recurs in four independent families.** Late scopes
are nearly free; the cost enriches with depth; the composed map takes
late members only. That is now a property of the model, not of a
family — and it is what makes a search over scopes tractable.

### The MLA depth curve

MLA is not intrinsically sensitive — it is intrinsically *positional*:

```text
layer   kl p99      bytes saved   reading
 23     5.751e-5    27.3 MB       the cheapest cell of ANY family at any depth
 26     1.237e-4    27.3 MB       plateau — the last MLA layer
 19     8.433e-4    27.3 MB       15x jump: the cliff edge
 15     1.064e-3    27.3 MB       marginal refusal
 11     3.699e-3    27.3 MB       boundary characterisation
 all 7  2.614e-2    191.1 MB      450x the L23 cell — early MLA is brutal
```

The tower cell's token-distance curve ACCUMULATES (2.6e-4 → 2.9e-3
across positions) where the late cells' does not: early-MLA error is
carried forward, not absorbed. By the jump rule — take the plateau,
stop at the first order-of-magnitude step — the admitted scope is
**MLA{23,26}**, and the topology is allowed to be non-contiguous.

## The shared experts: a useful rejection

The shared branch is fully supported: its dispatch has been
encoding-aware since the expert rung, so admitting it costs no kernel
work at all. It is declined anyway, and the reason is the point:

| scope | behavioural cost | bytes saved | cost per MB saved |
|---|---|---|---|
| MLA L23 | 5.75e-5 | 27.3 MB | **2.1e-6 / MB** |
| shared L25 | 1.240e-3 | 6.6 MB | **1.9e-4 / MB** |
| shared {24,25} | 1.092e-3 | 13.3 MB | 8.2e-5 / MB |

**~90x worse per byte than MLA.** The shared pair even composes
sub-additively — {24,25} costs LESS than L25 alone, a genuine
absorption result — and it still loses, because the budget it consumes
buys far more elsewhere. The engine can produce this representation;
it should not.

This is the first refusal in the programme that is not a capability
statement. A per-layer "quantise what passes" recipe would have taken
the shared branch. An optimizer with a byte model declines it.

## The composed four-family map

```text
routed experts   L20-26            Q8_0   (compiled candidate banks)
KDA projections  L20,21,22,24,25   Q8_0   (transient requant)
MLA projections  L23,26            Q8_0   (transient requant)
output head                        Q8_0   (transient requant)
shared experts                     BF16   REFUSED on economics
everything else                    BF16
```

Bytes removed per token: experts 371.6 MB + KDA 176.9 MB + head
353.9 MB + MLA 54.6 MB = **957 MB, 16.0 % of the BF16 ledger.**

Composition is again sub-additive, and the newest family is the
clearest case:

```text
map                                 diagnostic kl p99   delta    solo cost of the addition
experts L20-26 + KDA x5                    2.4321e-3    —
  + output head                            2.4006e-3    -1.3 %   7.13e-4
  + MLA{23,26}                             2.4762e-3    +3.1 %   5.75e-5 + 1.24e-4
```

Both additions are ABSORBED. The head's 7.13e-4 solo cost moved the
composed p99 by −1.3 % — a rank statistic over 256 positions does not
resolve a change that small, so the honest reading is *no measurable
composed cost*, not a negative one. MLA{23,26} added 3.1 % to a map
already at 2.4e-3 while removing another 117 MB of BF16 scope.

Late-scope perturbations across families do not stack. The mechanism
is the one PRECISION-1 identified: composed cost is dominated by
whether a perturbation reaches a downstream router, and by L20+ there
is little routing left to disturb. It is why the map can keep gaining
families almost for free, and why the same families refuse flatly when
taken early.

*(The 256-position diagnostic is the first 8 sequences of the same
bank the 8,192-position run uses — a SCREEN, not an independent
sample. Only the authority runs below are verdicts.)*

## Authority: both banks, 8,192 positions, `kimi-logit-balanced-v1`

The composed four-family map judged against the frozen contract, on
the selection bank AND the held-out bank (zero window overlap, never
used to choose any scope), with the three-family map beside it so the
price of the fourth family is legible:

| criterion | limit | three-family | four-family | budget used |
|---|---|---|---|---|
| **selection bank** | | | | |
| kl p99 | 3.5e-3 | 2.4006e-3 | **2.3791e-3** | 68.0 % |
| top-1 mass displaced (max) | 0.12 | 9.415e-2 | **5.546e-2** | 46.2 % |
| top-10 mass displaced p99 | 0.12 | 7.134e-2 | **6.465e-2** | 53.9 % |
| route mixture mass p99 | 0.15 | 0.1258 | **0.1248** | 83.2 % |
| route mixture mass max | 0.25 | 0.1993 | **0.1993** | 79.7 % |
| covered mass | ≥ 0.55 | 0.6315 | **0.6315** | — |
| **held-out bank** | | | | |
| kl p99 | 3.5e-3 | 2.6378e-3 | **2.6174e-3** | 74.8 % |
| top-1 mass displaced (max) | 0.12 | 5.8313e-2 | **5.8313e-2** | 48.6 % |
| top-10 mass displaced p99 | 0.12 | 6.904e-2 | **6.881e-2** | 57.3 % |
| route mixture mass p99 | 0.15 | 0.1262 | **0.1257** | 83.8 % |
| route mixture mass max | 0.25 | 0.2099 | **0.2099** | 84.0 % |
| covered mass | ≥ 0.55 | 0.5773 | **0.5773** | — |

**PASS on both banks, no failed criterion.** Counts move by at most
2 %: route flips 1296→1305 and 1260→1270, top-1 flips 130→129 and
124→122, top-10 changes 2515→2527 and 2322→2341.

The strongest statement is on the held-out bank, where **both max
statistics are identical to the three-family map** — `top1_mass`
5.8313e-2 and `route_max` 0.2099 to four decimals. The worst single
overturn and the worst single routing displacement in the whole
8,192-position measurement are the same events, unchanged, after
adding a family. On the selection bank the worst top-1 overturn's
severity fell 0.094 → 0.055 while the flip count went 130 → 129; with
counts flat that is one position reshuffling, not an improvement.

**MLA{23,26} is behaviourally absorbed at authority scale while
removing 117 MB of BF16 scope per token.** That is what free
composition looks like when it is real.

### The route budget is the binding constraint

The criteria do not deplete together:

```text
kl p99                68-75 % of limit    plenty
top-1 mass displaced  46-49 % of limit    plenty
top-10 mass p99       54-57 % of limit    plenty
route mixture p99     83-84 % of limit    BINDING
route mixture max     80-84 % of limit    BINDING
```

Route mixture mass is the only criterion measuring a DISCRETE
consequence — which experts were selected. KL and the mass-displacement
metrics degrade smoothly and average over 8,192 positions; a route
either moves or it does not. That is a structural reason to expect the
route budget to keep binding as breadth grows, and it is why a search
over representations should rank candidates against the vector of
remaining margins rather than a scalar bytes-per-KL: a candidate that
is cheap on logits but moves routing spends the scarce resource.

### Promotion drift

```text
map                                256-position     8192-position    drift
PRECISION-1 narrow (experts 24-26)      2.262e-4         4.153e-4      +84 %
four-family broad                       2.4762e-3        2.3791e-3     -3.9 %
```

A broad perturbation's diagnostic estimate was well calibrated where a
narrow one's was not — plausibly because a broadly distributed
perturbation populates the consequence tail densely enough that 256
positions already sample it. **Recorded as a search heuristic only**
(N = 2, and the diagnostic is a subset of the authority bank, not an
independent sample). No gate changes on it; full-bank composed
authority remains the only admission.

## Decode: the cost model scored against the machine

The prediction was REGISTERED before the run, from the byte ledger
alone: 957 MB of the 5.985 GB per-token BF16 read removed (16.0 %),
predicted **1.15-1.16x**. Two sessions, both arms in one process,
interleaved blocks with alternating order, per-arm minimum:

| session | baseline | candidate | wall speedup | GPU ms/token | GPU speedup |
|---|---|---|---|---|---|
| 1 | 35.81 tok/s | **40.79 tok/s** | 1.139x | 26.87 → 23.43 | 1.147x |
| 2 | 35.16 tok/s | **40.52 tok/s** | 1.153x | 26.97 → 23.56 | 1.145x |

**Measured 1.139-1.153x wall, 1.146x GPU. The registered prediction
was 1.15-1.16x — accurate to about 1 %, and on the optimistic side:
the GPU measure sits just BELOW the predicted band, not inside it.**
The model is a good planner and not yet a calibrated one.

Read the GPU column, not the wall column, when scoring the byte model.
GPU time is the stable instrument here — four blocks per arm land
within 0.05 ms in session 1 — while wall time carries 1-6.6 ms/token
of intermittent non-GPU overhead. The min-of-N floor exists to strip
exactly that, and it works: both arms' floor blocks carry ~1.05 ms of
overhead, so the floors compare like with like.

The conversion is the reusable number:

```text
bytes removed        16.0 %
GPU time removed     12.7 %
conversion           0.79   (0.80 session 1, 0.79 session 2)
```

**About 80 % of removed bytes became GPU time.** A pure-bandwidth
roofline would have predicted 1.190x; the decode step is not purely
bandwidth bound, and ~0.8 is the discount this machine applies. That single
number is what makes the next search's predictions cheap: bytes are
computable from a map without running anything.


## Instrument hazards — two catalogued flat signatures

Two four-family authority runs on 2026-08-31 returned numbers that are
NOT verdicts. Both were refused by the `covered_mass` criterion, which
is precisely what it exists for: a run that could only "pass" by being
blind must fail.

| signature | reading |
|---|---|
| `kl_p99` exactly 0.0, `covered_mass` exactly `TOP_N/vocab` (2048/163840 = 0.0125) | the GPU completed command buffers without executing them — both arms constant, coverage at the uniform-distribution floor |
| `covered_mass` 0.0297 with `top10_mass_displaced` p99 exactly 1.0 | a MIXED bank: the instrument degraded PART WAY THROUGH a ~25-minute run |

Root cause: the probe held a THIRD full stack (`null_partner`, ~17 GB
of wired attention banks) needed only by the 2-minute null arm, through
the entire measurement. At four-family scale that tipped the process
over the wired-collector wall mid-run. It is now dropped immediately
after the null arm.

Two lessons are load-bearing:

* **An up-front guard cannot protect a long run.** The second episode
  passed a diagnostic-scale guard at t0 and degraded 25 minutes later.
  Instrument health is checked after EVERY stage, and a contaminated
  stage refuses to be promoted to provenance.
* **`covered_mass` is not a formality.** Without it, the flat run
  reports `kl_p99 = 0.0` — a perfect score — and the map would have
  been admitted on a measurement of nothing.

Quarantined artifacts and their signatures are kept under
`~/chris-models/kimi-quality-bank-provenance/QUARANTINE.md`.

## Reproduction

```text
banks      selection ~/chris-models/qbanks/kimi-quality-bank-256x32
           held-out  ~/chris-models/qbanks/kimi-quality-bank-heldout-256x32
source     ~/chris-models/Kimi-Linear-48B-A3B-Instruct.lift2.vindex3
experts    LARQL_Q6_MAP="20-26:Q8_0" -> represent/compile_real_tests.rs
           (/tmp/kimi-map-l20-26q80-lift2.vindex3, ~2.5 min, resumable)
quality    kda_q8_real, scopes by env:
             LARQL_KDA_Q8_LAYER=20,21,22,24,25
             LARQL_MLA_Q8_LAYER=23,26
             LARQL_LMHEAD_Q8=1
             LARQL_KIMI_Q6_CANDIDATE=<expert candidate>
           LARQL_Q2A_SEQUENCES=8 diagnostic / 256 authority
speed      q2a_decode_bench, same scope environment, two sessions
evidence   ~/chris-models/kimi-quality-bank-provenance/
             kimi_full4-{guard-256,selection-8192,heldout-8192}_report.json
```

Every report carries its bank's `manifest.json` SHA-256, the gate it
was judged by, AND the `balanced-v1` verdict — a claim of
admissibility never has to be re-derived from the bank by hand.

## What this closes, and what it opens

Four byte families are now admissible under one frozen behavioural
contract, and the fifth is refused by an economic argument the engine
could have ignored. Q8_0 was the INSTRUMENT that mapped the topology,
not the destination: every family showed the same late-plateau /
interior-cliff shape, which is what makes a search tractable.

The manual phase ends here. What follows is an optimization problem,
not an exploration one: a candidate option is
`(scope, representation, bytes saved, solo cost, measured interaction
terms, kernel economics)`, and the objective is to maximise predicted
decode rate subject to a composed `balanced-v1` PASS — ranked against
the vector of remaining margins, with the full-bank composed gate
staying authoritative. Q6 and Q4 are where the remaining budget gets
spent.
