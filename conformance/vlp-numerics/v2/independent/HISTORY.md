# How v2 was independently reproduced, in order, with its limits

**What `evaluate.py` is.** A 245-line standard-library implementation (`fractions.Fraction` only) written by a fresh agent
that was given only `v2/SPEC.md` and `v1/SPEC.md` and blind inputs (`inputs.json`: expected values removed, case names
replaced by opaque ids, purposes dropped). It was told not to read `generate.py`, `vectors.json`, `FREEZE.json` or any other
repository file, and it reports that it read nothing else. Its only self-check was its rounder against native f32
conversion on 20,000 random values. `compare.py` is the referee and is not part of the evaluator.

## Sequence

1. **Draft spec** (commit `6ae11718`) and 89 vectors. The evaluator reproduced **89 of 89** blind, with the 38 v1-equivalence
   cases splitting 37 identical and exactly 1 different (`eq-lr-negzero-plus-poszero`).
2. **Its ambiguity list** (it could not know which cases depended on each item). Three exposed real defects, not just
   unclear wording: the generator accepted a **negative column index** (it wrapped to the last column) and a **base with the
   wrong word count**, and a malformed operand either crashed or was silently accepted. The generator was fixed, two vectors
   were added (`ref-column-negative`, `ref-base-word-count`) and the spec states what the evaluator had to guess.
   **Those two vectors were written to match the evaluator's choices, so they are NOT independent predictions.** The other 89
   are.
3. **Re-run** of the unchanged evaluator (same sha256) on the 91 inputs: **91 of 91**.
4. **Re-read of the revised spec** by the same agent: no sentence contradicted its evaluator. It found wording defects
   ("verbatim" no longer true; "byte-identical" stated without the NaN precondition; "zero-valued op" undefined; schema and
   JSON-type gaps). These were fixed as **wording only**, with the schema and encoding gaps declared out of contract.
5. **Diff check** of the wording-only changes by the same agent: no contradictions with its evaluator, nothing on
   well-formed documents left undetermined.

## Known residual wording notes (reported at step 5; none can change any outcome; not edited, to keep the frozen text the reviewed text)

- "The parser refuses it before this op runs" could be read as a refusal code the op must produce; the text says no
  behaviour is defined, so either is conformant.
- A float `rank` such as `2.0` is out of contract, which the `rank_invalid` wording does not itself decide.
- "A zero-valued op" is defined for whole-tensor zero operands only; a zero in some rows only is not a zero-valued op (those
  rows' elements are individually untouched).
- "Declared precondition" and "refuses with `non_finite_operand`" describe the same fact from two sides.
- §1's "wording of a few refusals" does not list them: they are `columns_invalid` (adds non-negative integers) and
  `shape_invalid` (adds the base word count), plus the touched set of fixups (`J`, not every element).

## Limits of this evidence

- **One evaluator, one author-independent implementation, same model family.** It is a fresh context with a restricted
  reading list, not a different team, language or arithmetic library. It shares whatever the specification itself gets wrong,
  and v2 inherits v1's profile definitions, which v1's own review covered.
- **The spec was revised in response to the evaluator**, which is the point of the exercise, but it means the frozen text is
  not the text it first saw. Both later passes confirmed the revisions against its code.
- **Agreement is only on the 91 cases.** Schema and encoding violations, resource limits, the source binding and the verifier
  behaviour are outside it (SPEC §8).
- The comparison itself was shown able to fail: 546 deliberate corruptions of expected or result values (a flipped bit, a
  changed kind, a changed length), every one detected.

## To reproduce

```
python3 -I evaluate.py inputs.json results.json      # writes results.json (identical to the committed one)
python3 -I compare.py results.json mapping.json inputs.json   # exit 0; rewrites RECORD.json
```
