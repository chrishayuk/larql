# H1 — the single hypothesis `holdout-v1` is spent on

**Frozen 2026-09-07, BEFORE `holdout-v1` was executed even once.**
sha256 of the bank under test: `c518b5ec40426242` (`holdout-v1.json`).

## The hypothesis

> **H1: `strict-4.25` (q=3, k=4, v=5, o=5; 4.25 RTN bpw) behaviourally
> DOMINATES uniform Q4_K (4.50 RTN bpw) — no material regression on any
> required behavioural boundary — while costing fewer bits.**

Dominance is control-relative and vector-valued. H1 holds only if
`strict-4.25` is no worse than `q4` on **all** of:

| boundary | metric |
|---|---|
| winner preservation | top-1 agreement with BF16; BF16-winner rank movement |
| distribution preservation | KL mean **and** p95 |
| neighbourhood preservation | top-5 overlap |

with the recurrent-state admission already passed (it is: 0.0582 ≤ 0.10).

## Calibration basis

`calibration-v1` (12 prompts, 198 positions, unsaturated at BF16 median
confidence 0.601) found `strict-4.25` better than `q4` on every dimension:
KL 0.0201 vs 0.0438, KL p95 0.0680 vs 0.2492, top-1 93.5 % vs 91.8 %,
top-5 95.1 % vs 92.7 %, rank moved 6.5 % vs 8.2 %.

## Rules binding this run

1. **Only H1 is under test.** `q8` and `q6` run as controls and are
   reported; they are not candidates.
2. **If H1 fails, the hypothesis fails.** `holdout-v1` may NOT then be
   used to select a different candidate. The search resumes on
   calibration or on future, separately frozen evidence.
3. **`aggr-3.75` is NOT taken to holdout.** Its calibration claim already
   failed — 3.1x Q6's KL, and worse than `q4` on the tail.
4. **No tail claim may be made from this run.** "Worst prompt" on 20
   prompts is still a handful of samples; the calibration tail figure was
   driven by a single 15-token prompt. Tail qualification is a separate
   future rung (TAIL-1) needing a bank sized for tails.
5. **`holdout-v1` is spent after this run.** Any further use requires a
   new frozen bank.

## Recorded procedural debt

The calibration candidate columns were read at n=6 before the full bank
completed, so the "derive the rule before looking at candidates"
discipline was partly spent. H1 is stated in dominance form specifically
because dominance is hard to fit to numbers already seen: it requires
`strict-4.25` to win on *every* dimension, not on a threshold chosen after
the fact.
