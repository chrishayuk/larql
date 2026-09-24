# HEAD-FID-2: gpt-oss-20b's NVFP4 output head, decided through MEASURE-PLAN-1

Frozen before any HEAD-FID-2 logit is computed. The arms, bank, rule, guard and order below are FROZEN.
Nothing here claims a result.

Programme: REPRESENT. This is the separate pre-registration that MEASURE-PLAN-1 (`docs/measure-plan-1.md`,
"Deciding the NVFP4 head") defers to, run after that rung's W9.

## Why

Compiling gpt-oss-20b's output head to NVFP4 took lowered Metal decode from 10.37 to 8.51 ms/token (C vs H in
`bench/v3-metal-eco-1/README.md`). That is the difference between LARQL's conservative tier (~96 tok/s) and the
4-bit-head tier (116 tok/s, within 5% of MLX-Q4). V3-METAL-ECO-1 cannot quote H until its fidelity is decided.

HEAD-FID-1 tried to decide it by hand over one prompt (331 positions) and was UNINFORMATIVE by its own guard:
C was 89.12% top-1 against R, below the guard's 90%. HEAD-FID-2 asks the same question through the procedure,
over a bank.

## What is known before this freeze (disclosed, because it informed the thresholds)

- **HEAD-FID-1, descriptive only:** mean KL(R ‖ H) / mean KL(R ‖ C) = 1.093 over 331 positions of one prompt;
  mean KL(C ‖ H) 0.00937 nats.
- **W9 (MEASURE-PLAN-1), recorded in `bench/measure-plan-1/w9/`:** R against C over Q-BANK-1's 69 sequences,
  admissible. Mean KL 2.914e-2, p99 2.217e-1, top-1 92.67%, and 100.00% in the ≥ 0.5 margin band. The
  guard below is therefore expected to pass, and this rule's thresholds imply that R → H may reach at most
  mean 3.64e-2 and p99 3.33e-1 if the yardstick reproduces. This R → C reading is not a HEAD-FID-2 outcome.
- No H logit has been computed on the bank by any procedure.

## Arms

All three are Metal, lowered, and read stored bytes (`--*-source stored` wherever a pack is named).

| arm | backend | container |
|---|---|---|
| **R** reference | `metal-lowered-f16` | `gpt-oss-20b.vindex3` (source) |
| **C** conservative | `metal-lowered` | `gpt-oss-20b.nvfp4.vindex3` (decoder stack NVFP4, head at source) |
| **H** head | `metal-lowered` | `gpt-oss-20b.nvfp4-head.vindex3` (as C, plus `target.output_head@NVFP4`) |

## Measurements

These are three `larql vindex3 measure` runs from one binary, in one session, exactly as
`bench/head-fid-2/run.sh` invokes them, over Q-BANK-1's 69 sequences
(`gpt-oss-20b.quality-bank-1.tokens`, bank id `78139521…e6df`):

1. **R → C:** the yardstick, meaning the cost of the pack H would join. This is W9's configuration.
2. **R → H:** the decision run.
3. **C → H:** the head in isolation. The procedure must report `changed: target.output_head@NVFP4` and nothing
   else. This run is descriptive, not a rule input.

## Rule

H is **ACCEPTABLE** only if all three runs are admissible with complete facts, and all of these hold:

1. **Mean:** mean KL(R ‖ H) ≤ 1.25 × mean KL(R ‖ C). The head may add at most a quarter of the pack's own
   error.
2. **Tail:** p99 KL(R ‖ H) ≤ 1.5 × p99 KL(R ‖ C).
3. **Confident band:** top-1 agreement(R, H) ≥ 99.5% on the positions where R's margin is ≥ 0.5.

If any of these fails, H is **NOT ACCEPTABLE**.

**Guard:** if R → C's top-1 agreement is below 90%, the result is **UNINFORMATIVE**, as in HEAD-FID-1. If any
run is inadmissible, the result is **UNINFORMATIVE**, not a pass.

A pass says that the head adds error small relative to the pack it joins. It does not say the pack is lossless,
and it does not say anything about MLX's or llama.cpp's 4-bit heads, which have no arm here.

## Order

1. Commit W9 evidence (MEASURE-PLAN-1 PR 4b) and this freeze.
2. Use the W9 re-run's binary. Its source is main plus only documentation commits. Record its sha and sha256.
3. Run the three measurements, interleaved in one session. Record `uptime` and `LARQL_*` env per run.
4. Adjudicate against the rule above, and commit the evidence and verdict under `bench/head-fid-2/`.

## Out of scope

- Speed. It was measured in V3-METAL-ECO-1 and is not re-measured here.
- External runtimes' fidelity. That needs an external arm, which is a separate freeze.
- Other models' heads.
