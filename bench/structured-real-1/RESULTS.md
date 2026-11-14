# STRUCTURED-REAL-1 results (2026-11-12)

Run once, as pre-registered in `PROTOCOL.md`, with no change to the protocol or the harness after the first run. Raw record: `results.json`.
Code: branch `research/structured-real-1` on the verifier PR #806 (which contains the engine PR #805). CPU only.

**All seven acceptance criteria held.**

| # | Criterion | Result |
|---|---|---|
| 1 | Source binding | The patch's `index_sha256` and payload digest equal the registry container's |
| 2 | Native compile | `COMPILE` succeeded, ledger `weight-patches/v3`, 1.5 s, 2.3 GB peak RSS |
| 3 | Independent verification | SATISFIED, 0 of 268,098,176 elements differing (every BF16 element of the base), 4,663 expected to change; through LQL `VERIFY PATCH` and directly |
| 4 | Preservation (by byte comparison) | No other tensor changed; embedding and final-norm segments byte-identical; 0 changed elements outside the 8 declared columns |
| 5 | Mutation sensitivity (re-sealed copies) | A flipped bit in a changed element, in an untouched element of the edited tensor, and in another tensor: each VIOLATED with exactly 1 violation, located correctly |
| 6 | Reproducibility | A second compile gives byte-identical segments and ledger (`candidate.json` excluded: it carries the output's temp-path hint) |
| 7 | Registry container unchanged | `index.json` and every file digest equal before and after |

## Recorded, not judged

- 4,663 of the 5,120 elements in the declared columns changed (the rest rounded to their old bits). The largest relative change was 128.8x, on an element whose old value was tiny.
- **No changed element was one that an f32 delta could not write** (searched as the engine's own golden does). So this run shows the pipeline on real weight layout, and does *not* exercise the representability case that motivated the work. That case is covered only by the synthetic golden.
- The verifier took 18.8 s and **15.4 GB peak RSS** for this container (it holds every BF16 tensor of both containers as Python integer lists). That is fine at 270M and would not scale to a multi-billion-parameter model without a streaming or bytes-equal fast path for unedited tensors.

## What this does and does not show

It shows that the native compiler transformed a real Gemma 3 270M BF16 tensor, and that a separate implementation, using exact rationals and not the ledger, reconstructed every resulting bit. The direction is arbitrary (seeded Gaussian, unit norm); nothing here says it means anything, and no generation or behaviour was evaluated.

It does not show: that the interpreter and the verifier don't share a mistake in the profile definition (they implement the same specification); anything about derived/quantised representations (this container has none); the operand's provenance; or any behavioural effect.
