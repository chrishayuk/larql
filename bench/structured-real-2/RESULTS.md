# STRUCTURED-REAL-2 results (2026-11-12)

Run once, as pre-registered in `PROTOCOL.md` (committed before the run), with no change to the protocol or harness afterwards.
Raw record: `results.json`. Same code base as REAL-1 (verifier PR #806 containing engine PR #805). CPU only.

**All nine acceptance criteria held.**

| # | Criterion | Result |
|---|---|---|
| 1 | Source binding | Patch identities equal the registry container's |
| 2 | Native compile | Succeeded, v3 ledger, 1.6 s, 2.3 GB peak RSS |
| 3 | Independent verification | SATISFIED, 0 of 268,098,176 elements differing, 8 expected to change; 18.6 s, 15.4 GB peak RSS |
| 4 | Preservation (bytes) | No other tensor changed, embedding and final-norm segments identical, nothing changed outside the declared columns |
| 5 | Mutation sensitivity (re-sealed copies) | A flipped bit in a changed element, an untouched element and another tensor: each VIOLATED exactly once, located correctly |
| 6 | Reproducibility | Second compile byte-identical |
| 7 | Registry unchanged | Yes |
| 8 | Unreachable by any f32 delta | All 8 changed elements are outside what the delta search reaches |
| 9 | Closed form | All 8 new values equal the BF16 of `-2^-30 * old`, derived in numpy without either implementation, all in row 7 |

## Recorded

8 elements changed (row 7, the 8 declared columns, all with nonzero base). Changed magnitudes 8.5e-13 to 5.0e-11; the largest
relative change was 1 + 9.3e-10 (each element went from `old` to `-2^-30 * old`, so the sign flipped and the magnitude
collapsed by about 1e9). The three mutated copies each gave exactly one violation, e.g. `first at (7, 0): expected ad23, found ad22`.

## What this adds to REAL-1

REAL-1 showed the pipeline on real weight layout; here the same pipeline produces, on real weights, results that
`UPDATE WEIGHTS` cannot write, and the independent verifier reconstructs them, along with a closed form that was derived
without reference to either implementation. This is the representability advantage the structured form exists for.

## What it does not show

The edit is deliberately degenerate: it makes eight weights almost exactly zero with the opposite sign, to hit the
representability case. It is not a meaningful intervention, nothing was generated or evaluated, and nothing says such a
result is useful. "Unreachable by a delta" is shown by a bounded search (8 f32 ulps each way, exact arithmetic) and by the
argument in the protocol, not by an exhaustive proof over all f32 deltas. The interpreter and verifier still implement the
same profile definition, and the verifier's 15.4 GB peak memory is unchanged and unaddressed.
