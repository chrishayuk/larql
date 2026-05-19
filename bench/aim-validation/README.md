# Aim-Validation Harness

V0 scaffolding for `ROADMAP.md` V1-V4. The goal is not to implement the
experiments here; it is to keep their inputs, output schema, and comparisons
consistent so the results are falsifiable instead of anecdotal.

## Files

- `matrix.json` - canonical model/prompt/metric matrix for V1-V4.
- `scripts/aim_validation.py` - helper for printing planned runs, creating run
  directories, and recording JSON results.

## Workflow

```bash
python3 scripts/aim_validation.py plan
python3 scripts/aim_validation.py init-run V1

# Run the experiment-specific command, then record its JSON artifact:
python3 scripts/aim_validation.py record V1 \
  --artifact /tmp/v1-hash-routing.json \
  --notes "top-k sweep on Gemma 3 4B and Llama 2 7B"
```

The helper writes under `experiments/V1-V4_aim_validation/` by default. That
directory is intentionally outside this repo's normal source tree unless you
choose to commit a summary.

## Result Contract

Every recorded artifact should preserve:

- `test_id`: one of `V1`, `V2`, `V3`, `V4`.
- `model`: matrix model id or explicit model path.
- `prompt_set`: the matrix prompt set used.
- `metrics`: benchmark or experiment-specific metrics.
- `git_rev`: source revision at the time of the run.
- `notes`: short human context for thermal state, machine, flags, or caveats.

V1/V2 are allowed to use experiment-specific metric keys, but the final summary
must include divergence/quality and tok/s or bandwidth impact. V3 must include
page-fault and p50/p99 disk-read behavior. V4 must compare measured stacked
speedup against the product predicted by V1-V3.

