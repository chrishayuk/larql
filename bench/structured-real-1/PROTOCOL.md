# STRUCTURED-REAL-1: one real BF16 tensor through native structured VLP and the independent verifier

Written before the first run. Branch `research/structured-real-1` (worktree `larql-structured-real`), based on the verifier
PR (#806, which contains the engine PR #805). No changes to any frozen campaign code, Gate A branch or evaluation fixture.
CPU only. **No generation, no behavioural evaluation, no claim about what the edit means.**

## Question

Can the native compiler apply a structured `remove_direction` to a real Gemma 3 270M BF16 weight tensor, and can the
independent verifier reconstruct the exact resulting bits from the operands and the base weights?

This is not the 270M inference-equivalence investigation (PREC-DIAG) and does not touch its containers: the registry
container is opened read-only and the compile writes elsewhere.

## Fixed inputs

| | |
|---|---|
| Base | `~/chris-models/larql-registry/gemma-3-270m-it.vindex3` (BF16 only: decoder stack, embedding, final norm) |
| Tensor | `5.mlp.down_proj.weight`, shape (640, 2048) |
| Direction `u` | 640 f64 values from `numpy.random.default_rng(20261112).standard_normal(640)`, normalised to unit length in f64; the operand file records the exact f64 bits |
| Columns | 0, 1, 2, 3, 1000, 1001, 2046, 2047 |
| Scale | 1.0 |
| Profile | `f64-seq-rne1/v1` |
| Verifier | `verifier/` from this branch, run as `larql-verifier patch ...` via `LARQL_VERIFIER`; the statement is LQL `VERIFY PATCH` |

`u` is arbitrary on purpose. It is not a discovered feature; nothing here says it means anything.

## Acceptance criteria (all must hold)

1. **Source binding.** The patch's `index_sha256` equals the SHA-256 of the registry container's `index.json`, and its
   payload digest equals that representation's `payload_sha256`.
2. **Native compile.** `COMPILE` succeeds, the ledger is `weight-patches/v3`, and the container opens under the strict
   reader (`inspect` coherent).
3. **Independent verification.** `VERIFY PATCH` is SATISFIED with 0 differing elements, elements compared = every BF16
   element of the base (100,325,376 decoder + 167,772,160 embedding + 640 final norm), and at least one element changed.
4. **Preservation.** The embedding and final-norm segments are byte-identical to the base; in the decoder segment every
   tensor other than the edited one is byte-identical; within the edited tensor every element outside the 8 columns is
   unchanged. Checked by byte comparison, not by the verifier.
5. **Mutation sensitivity on the real container** (copies, digests re-sealed so the container reads as intact): one flipped
   bit in a changed element, one in an untouched element of the edited tensor, and one in another tensor are each VIOLATED.
6. **Reproducibility.** Compiling the same patch twice gives byte-identical segments and ledger.
7. **The registry container is unchanged**: `index.json` and the segment digests before and after are equal.

## Recorded, not judged

How many elements changed; the largest relative change; wall time and peak memory of the compile and of the verifier;
whether any changed element is one an f32 delta could not have written (computed with the engine's own `patched_bits`).

## Stop conditions

Any criterion failing is a finding to report as it is, not a reason to change the protocol and rerun. If the verifier cannot
finish on this tensor set for resource reasons, that is recorded as a resource limit and not worked around silently.
