# PATCH-TRAIN-2 calibration plan (DRAFT r2, 2026-11-14; direction approved, nothing run)

Companion to `PATCH-TRAIN-2.md` r3 §7. **Calibration is not started** and waits for review of this plan, the data plan and the generator.
The frozen v2 numerics contract is untouched. Results are emulation-consistent, not verified.

## 1. Inputs

Generated CAL split (4 pairs, 8 entities) and RET split (24 entities) from `PATCH-TRAIN-2-DATA-PLAN.md`. Same fixed configuration as
the protocol §4 (rho 0.40, rank 1, 256 columns, 200 steps, lambda 1.0 for R and D, `f = 0.5`). `retain1.py` supplies the arms
unchanged; no change to objective, optimiser or cap.

## 2. Steps

| Step | What | Output |
|---|---|---|
| C1 isolated learning | for each of the 4 CAL pairs, primary order: isolated A, isolated B, then composition C, N, R, D | per-prompt results for all arms; check the configuration learns on untouched entities (a failure is a negative result, no re-tuning) |
| C2 null patches, family N1 | 200 draws of random `a` (norm matched to the learned `a` of each CAL patch, same footprint, rank 1) with the **true address `b`** of a CAL entity (address fires, payload random) | distribution of every metric under "address fires, no learning", applied on W0 and on a learned W1 |
| C3 null patches, family N2 | 200 draws of random `a` and random `b` (norm matched) | the same metrics under "nothing fires" |
| C4 floors | for each metric take the **directional 99th percentile** (§3.1), per family, and use the **larger** of N1 and N2 as the floor; **report** the order statistics, their uncertainty and the N1-vs-N2 difference (§3.1) | table `T_null_*` |
| C5 derived thresholds | see §3 | frozen threshold table |
| C6 feasibility report | whether R and D, N and C meet each floor on CAL (**report only; never loosen**) | CAL report |

## 3. Thresholds (statistical gates from the null distributions; none from method results or discovery)

| Gate | Statistic | Source |
|---|---|---|
| `Delta_retain` | per draw, the largest **deterioration** (decrease) in log p of A's retained prompts, `max(0, -change)` | **upper tail** of that deterioration under C2/C3 applied after A |
| `T_null_gain` (verdict P) | per draw, mean **gain** (increase) in log p on B's 3 paraphrases | **upper tail** of the null gain from C2 (random payload, B's real address); the observed gain must exceed it |
| `f` | fraction of isolated gain | **0.5, fixed**, not calibrated |
| X_stat | per draw, the largest **increase** in log p of a patched answer over the uncovered unrelated prompts; `S` | **upper tail** per metric; decreases are not leakage |
| X_func (a) | unrelated prompt whose full-vocab top-1 becomes a patched answer | **zero tolerance**, definitional |
| X_func (b) | F2 in-set argmax flip rate over the uncovered unrelated prompts | **upper tail** of the null flip rate |
| X_func (c) | F3 largest margin **increase** of a patched answer over the uncovered unrelated prompts | `T_null_margin`, **upper tail** of the null margin increase |
| G | real-fact top-1 >= 23/24; neutral NLL <= base + 0.15 | fixed by `DESIGN.md` §7 |

**X_func (a) and, as stated, the whole X_func gate are independent of the null-derived X_stat floor and bind regardless of what the null distribution permits.** A broad null floor makes X_stat lenient by construction; X_func (a) and the margin gate are independent of it and are reported
beside it. If a null floor looks implausibly large (it permits a changed answer), that is reported and X_func still binds.

### 3.1 Estimating and disclosing the tail

With 200 draws per family the 99th percentile is an extreme order statistic of few observations, so it is reported, not hidden:

- the estimator is the order statistic of rank `ceil(0.99 * (n + 1)) = 199` of 200 (ascending); the 198th, 199th and 200th values are all
  reported;
- uncertainty: the exact distribution-free 95% confidence interval for the 0.99 quantile from binomial order statistics (reported as
  the ranks and values, with "upper limit = maximum observed" stated when the interval is unbounded above by the sample), plus a seeded
  bootstrap (2000 resamples, seed fixed in the plan) interval for the same estimate;
- the **difference between the N1 and N2 values** is reported per metric; the floor is the larger, so a large gap is visible and is
  not averaged away;
- the draws are per-draw summaries (max over the unrelated prompts, or mean over B's paraphrases, as in the table), so the quantile is over
  independent draws, not over correlated prompts;
- the direction of each metric is fixed in the table above and is not chosen after seeing the distributions.

## 4. Decision rules written before running

- Thresholds are the table above and nothing else; no value is adjusted after seeing CAL method results.
- If isolated learning fails on CAL at rho 0.40, calibration stops and reports; the protocol is not rescued by changing rho, rank,
  footprint or optimiser inside this protocol (a new protocol would be needed).
- No loss-weight grid, no D0, no new arm.
- The primary endpoint, verdict definitions and analysis plan in `PATCH-TRAIN-2.md` are not changed after CAL; only the numeric floors
  are inserted.

## 5. Cost (estimate, not measured here)

RETAIN-1 ran 6 cases x 4 arms in about 50 minutes. C1 is about 4 pairs x 6 learning runs; C2/C3 are 400 forward-only evaluations per
patch over roughly 40 prompts. Order of one to a few hours on this machine, CPU only.

## 6. After CAL

Review the frozen threshold table with the CAL report, freeze `PATCH-TRAIN-2.md` (hash recorded), only then generate CONF
(data plan step 5). Nothing here compiles natively or runs the verifier; that remains a separate milestone.
