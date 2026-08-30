# CPU-6 — paired independent validation of the CPU-5 candidate

**Frozen before Bank 3A or 3B existed.** Written after CPU-5 failed Bank
2, and deliberately: the candidate is UNCHANGED, and what is being
corrected is the protocol that tested it. Changing the candidate to chase
a failed validation set is the thing this document exists not to do.

## The candidate, unchanged

```
freeze SHA    df36ca9fc6553b7e636416886ccc51e09a2142d3
arithmetic    Q8[64] x asymmetric-Q8[16] -> I32 -> F32
kernel        K5 register-folded vector path
performance   264 ms/token, 3.79 tok/s, 1.32x shipped
```

No numerical change. No performance change. A successor SHA is
acceptable only if `crates/` is byte-identical to the freeze.

## What CPU-5 got wrong

```
intended     candidate degradation <= 2 x shipped degradation
implemented  candidate KL on bank N <= 2 x shipped KL measured on BANK 1
```

The gate carried an absolute number across prompt sets, so it was never a
ratio in practice. It was NOT underpowered: Bank 1's paired statistic
needed ~10 prompts at 95% and had 69. **Nothing numerical travels from
Bank 1 into CPU-6 except the 2x rule itself.**

## Every bank carries its own anchor

Three arms per bank, same prompts, same positions, same build, same
environment:

```
BF16 reference          the baseline both lossy arms are scored against
shipped Q8 x F32        the ANCHOR — what production already costs
candidate               Q8[64] x asym-Q8[16]
```

## Primary statistic — paired, per prompt

```
D_p = mean_position( KL(candidate_p || BF16_p) )
    - 2 * mean_position( KL(shipped_p   || BF16_p) )
```

**The prompt is the sampling unit**, not the token position. Positions
within a prompt are correlated observations of one sampled object, and
position-weighting would let a 128-token longform prompt outvote ten
factual ones. So a 12-token prompt and a 128-token prompt each count
ONCE.

A DIFFERENCE, never a per-prompt ratio: some shipped KLs are tiny and
`candidate_p / shipped_p` is pathologically conditioned there.

## Primary gate — each bank independently

```
one-sided 95% upper confidence bound on mean_p(D_p)  <=  0
```

By **prompt bootstrap, stratified by the seven categories**, 20,000
resamples — not the normal `1.645 x SE` approximation, because the
prompt-level D distribution may be skewed or heavy-tailed and there is
no reason to assume otherwise.

**Resampling is WITHIN each category, preserving that category's frozen
count**, so every replicate carries 44 factual, 29 prose, 29 code, 29
arithmetic, 29 uncertain, 23 structured and 17 longform. Those weights
are part of the frozen target population; letting the composition itself
fluctuate would bootstrap a different population than the one being
validated.

**Both 3A and 3B must pass.** This is an intersection-union decision —
each bank must independently establish non-inferiority — so the two
tests are NOT Bonferroni-corrected to 97.5%.

## Hard gates, each bank

```
top-1 agreement >= 99%
zero flips at BF16 margin >= 0.10
```

## Secondary, descriptive, NOT gates

```
position-weighted aggregate KL      "degradation per generated token"
KL p95 / p99
per-category paired D
aggregate ratio  mean_p(candidate_p) / mean_p(shipped_p)
```

The aggregate ratio is the interpretable headline. It is a ratio OF
MEANS, never a mean of per-prompt ratios.

## What a failure means, stated in advance

> **Failing to demonstrate `upper95(mean D) <= 0` is a failure to
> VALIDATE the candidate. It is not proof that the candidate is
> materially worse.**

The contract is 2x. If the truth sits near exactly 2.00x, an arbitrarily
large bank will eventually resolve the boundary in one direction or the
other, and that would say nothing about quality. 200 prompts exists to
make a useful distinction with reasonable precision — not to convert
statistical significance into a quality metric.

## Bank sizing, from measured prompt-level variance

Bank 1's paired D had mean -3.146e-05, sd 6.131e-05. Prompts required
for the 95% upper bound to clear 0, if the true effect is smaller than
Bank 1 suggested:

```
effect vs bank-1   n @95%
          100%         10
           50%         41
           33%         94
           25%        164
```

**200 prompts per bank**, which tolerates the true paired effect being a
QUARTER of Bank 1's. Not larger: two independent banks passing is
already a strong replication standard, and chasing 99% on each buys less
than the second bank does.

## Category targets — frozen, per bank

Fixed BEFORE authoring so 3A's composition cannot be influenced by 3B's,
and neither by results:

```
factual     44      prose       29      code        29
arithmetic  29      uncertain   29      structured  23
longform    17
                                        total      200
```

## Disjointness

Each bank verified disjoint from ALL of:

```
quality-bank-1        (69, discovery, spent)
quality-bank-2        (69, spent on CPU-5)
sensitivity-1B' calibration (12, spent)
and from EACH OTHER
```

by exact id, whitespace/case-normalised text, and a 40-character prefix
near-duplicate scan. Both frozen SIMULTANEOUSLY, before either runs.

## Authoring is blind to candidate-specific results

The protocol prevents tuning the CANDIDATE. It cannot prevent the author
knowing how Banks 1 and 2 went — but it can bound what may inform the
writing:

```
ALLOWED      the frozen category definitions
             general Qwen use cases
             spent-bank prompt TEXT, solely for deduplication

NOT ALLOWED  candidate KL by prompt
             which Bank-1 prompts hurt asym-Q8 most
             candidate flip locations
             any prompt characteristic derived from candidate results
```

The banks represent the frozen categories. They are not steered toward
"easy" or "hard" quantisation cases in either direction.

Each prompt carries provenance — hand-authored, drawn from a source, or
produced by a generator — so a category accidentally dominated by one
authoring process is detectable afterwards rather than invisible.

### Author provenance is a LIMITATION, not metadata

All 400 prompts are hand-authored by a single agent. 3A and 3B are
statistically independent in MODEL EXECUTION — different prompts,
different positions — but they share a human-generation prior, and no
amount of care removes that.

The honest mitigation would be a genuinely independent second author.
That is not available here, and a "second generation process" run by the
same agent would share the same prior while LOOKING independent, which
is worse than the limitation stated plainly. So it is stated plainly:

> **Two banks written by one author are not two independent samples of
> the prompt space. They bound execution-side variation, not authorship
> bias.**

Consequence for interpretation: if both banks pass, that is strong
evidence the candidate meets the 2x contract ON PROMPTS OF THIS KIND. It
is weaker evidence about prompts a different author would have written.
A future bank from an independent source would strengthen it, and
nothing in this protocol substitutes for that.

**No templated prompts.** Generating a category from a template makes the
bank's statistics describe the template rather than the category —
that error once moved a headline robustness rate from 68.8% to 34.8%.
Every prompt is authored to vary in structure, not only in content.

## Decision

```
3A PASS  +  3B PASS    VALIDATED — productionise
3A FAIL  +  3B PASS    RETIRE
3A PASS  +  3B FAIL    RETIRE
3A FAIL  +  3B FAIL    RETIRE

No Bank 4.
```

**Failure on EITHER of two independently powered validation banks
retires the representation; there is no third appeal bank.**

An earlier draft said "two properly-powered independent failures"
retires it — a different and weaker protocol than the table. The table
governs.

**Both banks are run even if 3A fails.** 3B is already frozen and cannot
rescue the candidate, but it remains replication evidence about the
representation and about this protocol; stopping early would discard it
to save an afternoon.

---

# STATUS: PARKED, NOT ABANDONED — 2026-08-25

```
protocol     frozen   a87f441d
candidate    frozen   df36ca9f   crates/ unchanged
bank 3A      frozen   200 prompts, 4859 positions, token cbdaa46111a475ef
bank 3B      frozen   200 prompts, 4912 positions, token e3913dab3e536e83
adjudicator  frozen   selftest PASS
verdict      NOT RUN / NOT KNOWN
```

The six-arm run was started and deliberately cancelled about twenty
minutes in, to free the machine for CPU-7. **No partial arm was
retained and nothing was computed from one.** Both output directories
were deleted so that a future run cannot silently mix a stale arm with a
fresh one.

Measured rate on a quiet machine: **0.65 s/position**, so the full six
arms are ~4.6 h, not the ~15 h first estimated. That first estimate was
extrapolated from Bank-2 throughput which was contaminated by a
duplicate process — the same discipline that applies to a performance
number applies to a plan built from one.

## To resume

Nothing is re-authored and nothing is re-frozen. Check out this state and
run exactly what is already specified:

```
git checkout a87f441d          # or any HEAD with crates/ == df36ca9f
python3 bench/prompts/cpu6_freeze.py verify <container>
python3 bench/prompts/cpu6_run.py <container> <out3a> --bank 3a --sha df36ca9f...
python3 bench/prompts/cpu6_run.py <container> <out3b> --bank 3b --sha df36ca9f...
python3 bench/prompts/cpu6_adjudicate.py <out3a> --bank 3a
python3 bench/prompts/cpu6_adjudicate.py <out3b> --bank 3b
```

Run all six together. A completed provenance-verified arm could in
principle be kept across sessions, but the arms of one bank are meant to
sit close together in time, and the cost of redoing an hour is smaller
than the cost of reasoning about whether a months-old arm is comparable.

**Parking does not weaken the experiment.** Every degree of freedom —
prompts, protocol, gate, seed, adjudicator — was frozen before any
CPU-6 number existed, so the verdict is exactly as trustworthy whenever
it is finally computed.
