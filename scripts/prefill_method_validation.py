"""Validate the batched fresh-prefill method against the per-prefix control.

usage: python3 -I scripts/prefill_method_validation.py --repo <repo> --revision <sha> \
    --container <dir>.vindex3 --larql <binary> --out <file>.json [--lengths 1,7,64,511,512,513]

Both methods run a one-shot causal prefill with no decode state; `batched`
runs ONE over the sequence and applies the head at each requested position,
`per-prefix` runs one per prefix and keeps its last row. They are compared
row by row over the FULL vocabulary as max |Δ log_softmax| in float64,
against the strict-f32 tolerance (equivalence_metrics.REFERENCE_MAX_ABS_LOGPROB),
for every arm. Fail closed: exits non-zero on any disagreement above it.
The sequence is the harness's own passage ids, so the positions are the
ones its window cases record.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import equivalence_metrics as m  # noqa: E402
import hf_equivalence as h  # noqa: E402

DEFAULT_LENGTHS = "1,7,64,511,512,513"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--container", required=True, type=Path)
    ap.add_argument("--larql", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lengths", default=DEFAULT_LENGTHS)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    tokenizer = h.AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    ids = h.passage_ids(tokenizer, max(lengths))
    record = {
        "schema": "larql.measurement-method-validation/1",
        "measurement_method": "batched_prefill",
        "scope": "measurement-method validation only: shows the batched fresh-prefill arm reproduces the "
                 "per-prefix arm it replaces. It is NOT model evidence and feeds no equivalence stage; it "
                 "says nothing about LARQL matching transformers.",
        "harness_version": m.HARNESS_VERSION,
        "identity": {"repo": args.repo, "revision": args.revision},
        "container": h.container_identity(args.container),
        "larql": h.larql_commit(args.larql),
        "hardware": h.hardware(),
        "prefix_lengths": lengths,
        "positions_zero_based": [n - 1 for n in lengths],
        "input_ids": ids,
        "tolerance": m.REFERENCE_MAX_ABS_LOGPROB,
        "metric": "max over full vocab of |log_softmax64(batched) - log_softmax64(per-prefix)|",
        "arms": {},
    }
    with tempfile.TemporaryDirectory() as work:
        larql = h.Larql(args.larql, args.container, Path(work))
        for arm in h.ARMS:
            batched = larql.prefill(arm, ids, lengths, "validation", "batched")
            control = larql.prefill(arm, ids, lengths, "validation", "per-prefix")
            delta = m.max_abs_logprob(batched, control)
            record["arms"][arm] = {
                "max_abs_logprob_per_length": dict(zip(map(str, lengths), delta.tolist())),
                "bit_exact": bool(np.array_equal(batched.view(np.uint32), control.view(np.uint32))),
                "pass": bool(delta.max() <= m.REFERENCE_MAX_ABS_LOGPROB),
            }
            print(arm, record["arms"][arm], flush=True)
    passed = all(a["pass"] for a in record["arms"].values())
    record["validated"] = "pass" if passed else "fail"
    record["max_delta"] = max(max(a["max_abs_logprob_per_length"].values()) for a in record["arms"].values())
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.out}; validated: {record['validated']}")
    if not passed:
        raise SystemExit("batched and per-prefix disagree beyond tolerance — stop")


if __name__ == "__main__":
    main()
