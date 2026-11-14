# STRUCTURED-REAL-2: a real-weight result that no f32 delta can write

Pre-registered and committed before the first run. Branch `research/structured-real-1`, worktree `larql-structured-real`,
based on the verifier PR (#806, containing the engine PR #805). Follows STRUCTURED-REAL-1 (sealed, commit `be1d41b5`), whose
harness is reused unchanged in method; `run.py` of REAL-1 is not modified. CPU only. **No generation, no behavioural
evaluation, no claim about what the edit means.**

## Question

REAL-1 showed the pipeline on real weights, but every changed element was also reachable by an f32 delta, so it did not show
the one thing the structured form exists for. Here the intervention is chosen so that the exact BF16 result is one that
**no f32 delta can write**. Does the native compiler produce exactly that result on a real Gemma 3 270M tensor, and does the
independent verifier reconstruct it?

## Why the result is unreachable (stated before the run, checked by the harness)

With `u = e_r` (a basis vector) and scale `a = 1 + 2^-30`, column `j` becomes `W[r,j] - a*u_r*(u.W[:,j])`. The dot product
is exactly `W[r,j]` (the other terms are zero), so the result is `-2^-30 * W[r,j]` computed exactly in f64, and that is
exactly a BF16 value (a power-of-two multiple of one). To write it with `UPDATE WEIGHTS` a delta `d` needs
`RNE_bf16(W[r,j] + d) = t`, `t = -2^-30 * W[r,j]`: `d` must lie within half a BF16 ulp of `t` of `-W[r,j] + t`. The f32 grid
near `W[r,j]` has spacing about `2^-23 * |W[r,j]|`, far larger than that window (about `2^-39 * |W[r,j]|`), and the nearest
grid point to `-W[r,j] + t` is `-W[r,j]`, which gives 0, not `t`. So no f32 delta reaches it.

## Fixed inputs

| | |
|---|---|
| Base | `~/chris-models/larql-registry/gemma-3-270m-it.vindex3` (opened read-only) |
| Tensor | `5.mlp.down_proj.weight`, shape (640, 2048) |
| Direction `u` | the unit basis vector `e_7` (1.0 at row 7), f64 |
| Columns | 0, 1, 2, 3, 1000, 1001, 2046, 2047 |
| Scale | `1.000000000931322574615478515625` (= 1 + 2^-30 exactly) |
| Profile | `f64-seq-rne1/v1` |
| Verifier | as REAL-1 (`LARQL_VERIFIER`, the LQL `VERIFY PATCH` statement, and directly) |

Row 7 and the columns were fixed without looking at the weights. Changed elements are those declared-column elements of row 7
whose base value is nonzero (a zero stays zero and is not a change).

## Acceptance criteria (all nine must hold)

1-7. As REAL-1: source binding; native compile with a v3 ledger; independent verification SATISFIED with 0 differing over
every BF16 element and at least one expected change; preservation by byte comparison (no other tensor, no element outside
the declared columns, embedding and final-norm segments identical); three re-sealed mutations each VIOLATED exactly once;
byte-identical recompilation; registry container unchanged.

8. **Unreachable by any f32 delta.** Every changed element is outside what the harness's delta search can reach (the f32
neighbours of `new - old`, 8 ulps either side, with `RNE_bf16(f64(old) + f64(delta))` reproduced exactly on rationals, as
the engine's own golden does).

9. **Closed form.** Every changed element's new bits equal the BF16 of `-2^-30 * old`, computed here in numpy float64 with no
reference to either implementation; and no changed element is outside row 7.

## Recorded, not judged

How many elements changed; the smallest and largest changed magnitudes; wall time and peak memory as in REAL-1.

## Stop conditions

Any criterion failing is a finding to report as it is, not a reason to change the protocol and rerun.
