"""Append a stopped equivalence run to its record as `incomplete_runtime`.

usage: python3 -I scripts/record_incomplete_run.py --record <name>.equivalence.json \
    --repo <repo> --revision <sha> --profile <profile> --harness-version <v> \
    --larql-commit <sha> --started <iso> --reason <text> --log <harness stdout> \
    [--dump-dir <dir>]

A run stopped before it finished is neither a pass nor a failure: it is
appended (never written over anything) with what it completed — the case
lines the harness printed, parsed into per-arm cached / prefill full-vocab
maxima and gates — the reason it stopped, and the partial dumps it left
(names and sizes, so their existence is on the record).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import equivalence_metrics as m  # noqa: E402

CASE_LINE = re.compile(r"^\s*(?P<case>\w+) n=(?P<n>\d+)\s+(?P<arms>.*)$")
ARM_CELL = re.compile(
    r"(?P<arm>[\w-]+): cached (?P<cached>[\d.e+-]+) prefill (?P<prefill>[\d.e+-]+) \[(?P<gate>\w+)\]")
DECODE_LINE = re.compile(
    r"^\s*decode (?P<arm>[\w-]+): cached (?P<cached>[\d.e+-]+) prefill (?P<prefill>[\d.e+-]+) \[(?P<gate>\w+)\]")


def parse_completed(lines: list[str]) -> dict:
    """The case and decode results a harness log printed."""
    cases, decode = [], {}
    for line in lines:
        if (c := CASE_LINE.match(line)):
            arms = {a["arm"]: {"cached_max_abs_logprob": float(a["cached"]),
                               "prefill_max_abs_logprob": float(a["prefill"]), "gate": a["gate"]}
                    for a in ARM_CELL.finditer(c["arms"])}
            cases.append({"name": c["case"], "length": int(c["n"]), "arms": arms})
        elif (d := DECODE_LINE.match(line)):
            decode[d["arm"]] = {"cached_max_abs_logprob": float(d["cached"]),
                                "prefill_max_abs_logprob": float(d["prefill"]), "gate": d["gate"]}
    return {"cases": cases, "decode": decode}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("--record", "--repo", "--revision", "--profile", "--harness-version",
                 "--larql-commit", "--started", "--reason", "--log"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--dump-dir", default=None)
    args = ap.parse_args()
    identity = {"repo": args.repo, "revision": args.revision}
    completed = parse_completed(Path(args.log).read_text().splitlines())
    dumps = []
    if args.dump_dir and Path(args.dump_dir).is_dir():
        dumps = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(Path(args.dump_dir).iterdir())]
    pending = m.stage("PENDING", {"incomplete_runtime": True})
    run = {
        "status": "incomplete_runtime",
        "harness_version": args.harness_version,
        "profile": args.profile,
        "started": args.started,
        "identity": identity,
        "larql": {"commit": args.larql_commit},
        "stopped": {"reason": args.reason},
        "completed": completed,
        "partial_dumps": {"dir": args.dump_dir, "files": dumps},
        "note": "neither a pass nor a fail: the run was stopped before it finished",
        "ladder": {"equivalent": pending},
    }
    path = Path(args.record)
    record = json.loads(path.read_text()) if path.exists() else None
    record = m.append_run(record, run, identity)
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"appended incomplete_runtime run to {path}: {len(completed['cases'])} case(s), "
          f"{len(completed['decode'])} decode arm(s), {len(dumps)} partial dump(s)")


if __name__ == "__main__":
    main()
