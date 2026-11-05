# GW-CAR-1B — prospective carrier-source construction

**Status: executable source-design draft; not sealed; no CAR-1B model outcomes.**

The [CAR-1A result](gw-car-1a-results.md) established a 192-byte sufficient
carrier substitution on its registered surface. Its top-32 support was chosen
from each already computed natural carrier, and the measured contrast was
amplified. It did not establish a fixed support, natural-state fidelity, or
cheap construction. The historical CAR-1A specification and seal are unchanged.

CAR-1B asks: **what is the cheapest declared source that produces a causally
adequate carrier substitution, and which natural-prefix dependencies remain
elsewhere in execution?** The first wave separates support selection, values,
template reuse, indexed reuse, and an earlier-state transition. Negative results
constrain these constructions; they cannot establish an intrinsically necessary
transformer prefix.

## Scope and population decision

The original GW-V2 train split (44 subjects, 396 rows) is the proposed training
bank. No GW-V2 validation/test carrier may enter fitting, retrieval, mean/support
selection, or hyperparameter selection. Prior CAR-1A results motivated this
design; the old 666-row cohort is not fresh confirmation.

Fresh subject confirmation must exclude all 74 GW-V2 subjects and all 53
earlier factual GW-0 subjects, using resolved country identities. Validation and
test must also be subject-disjoint from one another. Exact prompt rewordings
on these subjects would test prompt generalization, not fresh-subject transfer.

There is a real population constraint: GW-V2 selected **all** countries in its
pinned source satisfying independence, UN membership, one capital, one currency,
and one official language after GW-0 exclusion. Reapplying those criteria after
GW-V2 exclusion gives no new subjects. A new factual scope/source must therefore
be declared before this draft can be sealed. Neither silently admitting ambiguous
facts nor relabeling old held-out rows is an acceptable default.

The machine draft records this population decision as unresolved. The included
freshness checker rejects overlapping subjects/rows and inconsistent splits;
it does not establish source truth, tokenization, balanced relation coverage, or
adequate statistical power. Those require the eventual population contract.

An outcome-free [capital-only proposal](../bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-capital-population.proposal.json)
is now concrete: **67 fresh subjects, 201 prompts, 34 validation subjects and
33 test subjects**, with 57 distinct first-answer token IDs. It retains all
remaining source entries with one capital, without restricting their currencies
or languages. All three existing capital templates tokenize successfully with
the pinned tokenizer; standalone subject tokens are recorded separately.
The split is a deterministic subject hash order, with no model outcomes read.
These are historical source annotations, not a new audit of political status
or capital ambiguity. This proposal would support a capital-relation claim only.
The [reproducible builder](../scripts/gwcar1b_population.py) verifies the pinned
country-source and tokenizer hashes and refuses to overwrite an existing proposal.
The proposal remains separate from the unsealed experiment protocol.

## Fixed local interface and remaining dependencies

The intended interface is the STATE-1 context-128 composition: entering L24
carrier, exact target H1 versus matched-donor H1, seven matched-donor non-H1
heads in both arms, and the target-natural KV/continuation. Only the carrier
varies between candidates; both H1 arms receive the identical carrier candidate.
Heads pass through effective prepared W_O, post-attention normalization, and
residual addition. L24 FFN onward supplies the terminal continuation.

On a new population the controls and context-255 denominator must be captured
and replayed anew. Historical 142-token readouts, 15-subject tensor shapes,
donor maps, and STATE-1 replay files are not portable population constants.
The proposed readout construction is the sorted union of frozen first-answer
token IDs, fixed without model outcomes. Donors and same/different-fact controls
must be defined on the new rows before capture.

Every result must distinguish these dependencies:

| Input | Source in this local experiment | Consequence |
|---|---|---|
| Candidate carrier | Declared source below | The tested construction claim |
| Target H1 | Natural target execution/capture | Still an oracle semantic operation |
| Seven other heads | Matched donor execution/capture | Offline bank and sourcing costs remain |
| Target KV and continuation | Natural target prefix and ordinary suffix | No end-to-end prefix-elimination claim |
| Earlier-state feature | Normal blocks 0–11 or 0–15 | Prefix depth retained, including attention/KV work |

A positive result establishes source substitution at this interface. A separate
execution experiment must supply H1, KV, and continuation without hidden target
prefix work before claiming compiled inference or an economical graph walk.

## First-wave candidates

All supports have 32 coordinates; no support-size search is proposed here.
`template` means the **relation × prompt-family** train mean. `relation` means
the train mean pooled across that relation's prompt families. Training is
balanced: exactly one row per subject × relation × prompt family. Query
templates must occur in training; unknown groups refuse rather than falling back.

| ID | Carrier construction | Online source | Claim class |
|---|---|---|---|
| 0 | Exact natural carrier | Target L24 | Mandatory oracle control |
| 1 | Matched-donor carrier | Donor L24 | Historical negative-control analogue |
| 2 | Target top-absolute 32 | Target L24 | CAR-1A oracle replication |
| 3 | Train-fixed support, target values | Target L24 | Support-versus-value diagnostic |
| 4 | Template + target top-absolute delta 32 | Target L24 | Adaptive delta oracle |
| 5 | Template + train-fixed-support target delta 32 | Target L24 | Fixed delta oracle |
| 6 | Global mean | Metadata | Shared scaffold |
| 7 | Relation mean | Relation ID | Relation reuse |
| 8 | Template mean | Relation and prompt-family IDs | Template reuse |
| 9–10 | Same-template nearest train carrier; 1 or 4 neighbors | Subject embedding | Indexed reuse |
| 11 | Template + fixed-support predicted delta | Subject embedding | Entity-conditioned composition |
| 12–13 | Template + fixed-support predicted delta | Entering L12 / L16 carrier | Earlier-state transition |
| 14–15 | Same-template nearest train carrier; 1 neighbor | Entering L12 / L16 carrier | Indexed state transition |
| 16 | Zero carrier | None | Scaffold diagnostic |
| 17–19 | Corresponding 11–13 with shuffled train delta targets | Same features as 11–13 | Entity correspondence controls |

Only IDs 6–15 are source-construction candidates. Oracle, zero, and shuffled
controls cannot become winners. Exact-query cache lookup is deferred to a
separate repeated-query experiment: a fresh-subject miss must not trigger
natural-prefix execution disguised as retrieval. Recurrent models and graph
propagation are also later prospective families, not hidden fallbacks here.

## Deterministic source algorithms

All carrier/features/parameters are stored as little-endian f32; accumulation,
distances, regression fitting, and regression application use f64, with the
explicit f32 roundings in `gwcar1b_sources.py`. Sort training rows by row ID.
Compute template/relation means in f64 and round to f32.

The carrier support (ID 3) ranks the train mean absolute carrier values. The
delta support (IDs 5, 11–13, 17–19) ranks the train mean absolute residual after
subtracting its f32 template mean and rounding the residual to f32. Ties use
ascending coordinate index; serialize selected indices in ascending order.
Adaptive oracle supports use the target carrier/delta. All omitted coordinates
are zero for IDs 2–3 and retain the template value for delta candidates.

The subject feature is the mean of effective, scaled model input embeddings
for the standalone subject-name tokenization without special tokens. The
eventual capture manifest must bind exact token IDs, operand identity, scaling,
and pooling. This feature must be bit-identical across a subject's relations
and prompt families. It does not require an attention or FFN block.

`enter_l12` and `enter_l16` mean the final prompt position's residual immediately
before that layer's attention. They are not the subject token's V vector or the
earlier layer's post-attention residual. Extraction must bind observer address,
prompt token IDs, canonical execution identity, and all normal prefix work.
The old V2 `source-depth-v.f32` artifact is not this feature.

Nearest neighbors minimize unnormalized squared Euclidean distance within the
same relation/template group. Equal distances break by lexical training row
ID. Average the selected 1 or 4 full carriers uniformly. Search is exhaustive;
the ledger must not claim an approximate index or sublinear search.

Ridge regression predicts just the 32 fixed-support residual values. Center X
and Y by their f32 train means; use
`alpha = max(0.01 * trace(X_centered X_centered^T) / n_train, 1e-12)` and
`W = X_centered^T solve(X_centered X_centered^T + alpha I, Y_centered)`.
Round W to f32. Apply `(x - mean_X) W + mean_Y`, round the predicted delta to
f32, and add it at the template's selected coordinates with a final f32 round.
There is no validation-selected rank, penalty, norm correction, or clipping.

For shuffled controls, sort training subject IDs and cyclically move each
subject's entire target-delta bundle to the next subject, preserving relation
and prompt family. Features stay fixed. This excludes identity mappings and
preserves the paired structure across the nine executions per subject.

## Prospective analysis requirements

These requirements guide the future executable adjudicator; this source-design
draft does not yet implement causal replay or adjudication.

- Replay both H1 arms for every candidate and every registered row. Compute
  the same subject-clustered, control-adjusted raw/z candidate-JS proximal and
  terminal contrasts as STATE-1. Define subject aggregation and z normalization
  in the population-bound analysis, without importing old population counts.
- Use fresh context 255 as the proximal denominator. Require positive point
  estimates, all bootstrap denominators, and pointwise 2.5% lower bounds.
- Proposed absolute gate: both proximal point retentions ≥0.80, both
  simultaneous lower bounds ≥0.50, and both terminal-effect simultaneous lower
  bounds >0, independently on validation and test. Exact control 0 must clear
  both splits or invalidate the experiment. ID 2 is diagnostic, not a validity
  requirement; failure would limit transfer of CAR-1A's compactness result.
- Proposed bootstrap: 10,000 subject-clustered draws, NumPy PCG64 seed 27022034
  reset per split; ddof=1, linear quantiles, one-sided 95% studentized max-t band.
  Freeze handling of degenerate standard errors in the executable before seal.
- The simultaneous family must include all 20 candidates × four absolute
  readouts, plus raw/z proximal and terminal **effect differences** for 11−8,
  12−8, 13−8, 11−17, 12−18, and 13−19. A claim that a learned correction adds
  entity information requires positive lower bounds for both its template and
  shuffle comparisons on all four readouts and both splits, as well as its
  absolute gate. Passing the absolute gate alone supports source adequacy only.
- Report the full registered surface, including failed controls and amplified
  contrasts. Do not interpret a ratio above one as fidelity. Report vector norm,
  support overlap, and carrier reconstruction error descriptively; these cannot
  select candidates or change gates. No best-validation-cell selection.

## Cost accounting

Report offline training/capture cost separately from per-query cost. Record
numeric shared bytes, actual serialized bank/metadata/index bytes, query feature
bytes, output bytes, source depth, feature extraction, exhaustive search, decode,
and regression work. A dense replay tensor remains 10,240 bytes even when its
representation is a template plus a 128-byte fixed-support value vector.

The source harness reports numeric array bytes only and labels omitted metadata
and extraction costs explicitly. Its ridge application has 2,560 × 32 = 81,920
multiply-add pairs, plus centering and additions. The complete library bank holds
more arrays than any single candidate needs; per-candidate numeric bytes are
not a measurement of process memory. Freeze a real serialization before a storage
claim. No scalar count is a measured latency or avoided-work claim.

No single cheapest winner is defined across depth, storage, and computation.
Report eligible candidates as a cost/causal-retention surface; any amortization
population must be declared before adjudication. Performance measurements need
the repository's exclusive baseline/candidate/baseline protocol separately.

## Implementation and execution boundary

The [source harness](../scripts/gwcar1b_sources.py) implements the 20 candidate
definitions with a separate oracle API. Its tests exercise support/value
separation, known additive structure, deterministic retrieval, shuffle controls,
and leakage refusals using synthetic data only. The API cannot prove that a
supplied early feature was captured honestly; a hash-bound capture runner must
provide that provenance.

The [draft checker](../scripts/gwcar1b_preregister.py) binds the specification,
source code, tests, historical population, and result authorities. It deliberately
has no seal or model-run command. It reports the remaining requirements rather
than presenting a source-design hash as an executable experiment freeze:

1. Resolve the fresh population scope, fact source, rows, split, power/sample
   sizes, templates, tokenization, donor/control maps, and readout vocabulary.
2. Implement and bind the feature/reference capture, candidate artifact IO,
   paired replay, and population-shaped adjudicator. Check identities on the
   new population; preserve historical CAR-1A/STATE-1 bytes and executables.
3. Jointly review prose and executable gates, bind executable and model hashes,
   and seal before new candidate fitting/replay or outcome inspection.
4. Fit on the declared train bank, materialize all candidate sources, replay
   controls and candidates, then adjudicate validation and test once.

Inspect the current draft and run its synthetic checks with:

```bash
python3 scripts/gwcar1b_preregister.py validate bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-protocol.draft.json
python3 -m unittest discover -s scripts -p 'test_gwcar1b*.py'
```
