# MEASURE-PLAN-1 W9: a real run on gpt-oss-20b

This is witness W9 of `docs/measure-plan-1.md`. It is recorded here and not judged: the procedure has no gate.

- **Reference R:** `metal-lowered-f16` on the source container `gpt-oss-20b.vindex3`.
- **Candidate C:** `metal-lowered` on the conservative pack `gpt-oss-20b.nvfp4.vindex3`, stored.
- **Bank:** Q-BANK-1 as a token bank (`teacher-forced-token-bank/v1`), id `78139521…e6df`, 69 of 69 sequences.

## Acceptance: met

- The receipt is **admissible**.
- **Facts complete:**
  - null arm on 4 samples;
  - 2 arms attributed;
  - 4 protected and 6 sealed representations;
  - tokenizer checked on both arms;
  - 69 bank samples read;
  - 1664 positions.
- **Changed variable:** `target.decoder_stack@NVFP4`, and nothing else.
- **Sample size:** 69 ≥ 64, so the p99 reading is inside UNCERTAINTY-2's ±25% regime (`adequate_for_p99: true`).

## Reading (nats, full vocabulary)

| slice | n | KL mean | KL p99 | top-1 |
|---|---:|---:|---:|---:|
| all | 1664 | 2.914e-2 | 2.217e-1 | 92.67% |
| margin 0.0–0.1 | 461 | 4.358e-2 | 2.351e-1 | 76.36% |
| margin 0.1–0.5 | 577 | 3.957e-2 | 2.597e-1 | 97.75% |
| margin 0.5–1.0 | 626 | 8.884e-3 | 7.815e-2 | 100.00% |

Per-category rows are in `run.txt`. Longform carries the most error: mean KL 5.07e-2, 88.73% top-1.

## Two runs, one reading

- **The first run** was made on 2026-09-23 19:53 from the unmerged MEASURE-PLAN-1 PR 3 branch. It recorded no
  binary identity, and #524 (residual scale on the lowered Metal path) merged after it.
- **This run** used main `a7f69949`, with binary sha256 `d614b346…72b5` (`meta.txt`).

Across all 1664 positions, every field the two runs share is identical: KL, top-1, top-5, ΔNLL, margin and
entropy. Main adds `max_abs_delta` and `mean_abs_delta` per position, and a `provenance` block.

## Files

- `run.sh`: the exact invocation.
- `meta.txt`: binary identity, load, and `LARQL_*` env.
- `run.txt`: the procedure's summary.
- `out/report.json`, `out/receipt.json`, `out/positions.jsonl`: the procedure's outputs. The home directory is
  rewritten to `~`.

This reading is the yardstick configuration (R → C) for `docs/head-fid-2.md`.
