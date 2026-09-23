# Gemma 3 4B GW-0 calibration input

This directory is the frozen, unexecuted input for the first VINDEX3 semantic
WALK census. It contains 142 semantic edges and 426 raw-completion executions
across capital, currency, language and hypernym relations.

`manifest.json` binds `input.jsonl`, the local VINDEX3 index/system graph/tokenizer
and every source corpus by SHA-256. `input.jsonl` assigns semantic triples, prompt
families, exact BOS-bearing prompt token IDs and capture positions, splits, target
token IDs and matched-control IDs before capture.

Validate it without loading weights:

```bash
python3 scripts/gw_input_manifest.py validate \
  --manifest bench/gw0/gemma3-4b-it-phase1/manifest.json
```

The manifest must remain `frozen_input_unexecuted` until a separate GW-0 runner
consumes it. Capture results belong in a new census/artifact bundle and must not
be written back into these files.
