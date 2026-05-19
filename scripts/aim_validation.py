#!/usr/bin/env python3
"""V0 helper for ROADMAP.md V1-V4 aim-validation runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "bench" / "aim-validation" / "matrix.json"
DEFAULT_OUT = REPO_ROOT.parent / "chris-experiments" / "V1-V4_aim_validation"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_test_id(matrix: dict[str, Any], test_id: str) -> None:
    if test_id not in matrix.get("tests", {}):
        valid = ", ".join(sorted(matrix.get("tests", {}).keys()))
        raise SystemExit(f"unknown test id '{test_id}' (valid: {valid})")


def cmd_plan(args: argparse.Namespace) -> int:
    matrix = load_json(args.matrix)
    prompt_set = matrix["default_prompt_set"]
    prompts = matrix["prompt_sets"][prompt_set]
    print(f"matrix: {args.matrix}")
    print(f"default prompt set: {prompt_set} ({len(prompts)} prompts)")
    print()
    for test_id, spec in matrix["tests"].items():
        metrics = ", ".join(spec["required_metrics"])
        print(f"{test_id}: {spec['name']}")
        print(f"  required metrics: {metrics}")
        for model in matrix["models"]:
            print(f"  - {model['id']:<24} {model['family']:<8} {model['role']}")
        print()
    return 0


def cmd_init_run(args: argparse.Namespace) -> int:
    matrix = load_json(args.matrix)
    ensure_test_id(matrix, args.test_id)
    out_dir = args.output_dir / args.test_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "test_id": args.test_id,
        "test": matrix["tests"][args.test_id],
        "created_at": utc_now(),
        "git_rev": git_rev(),
        "matrix": matrix,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    matrix = load_json(args.matrix)
    ensure_test_id(matrix, args.test_id)
    artifact = load_json(args.artifact)
    out_dir = args.output_dir / args.test_id
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "test_id": args.test_id,
        "recorded_at": utc_now(),
        "git_rev": git_rev(),
        "artifact_path": str(args.artifact),
        "notes": args.notes or "",
        "artifact": artifact,
    }
    records_path = out_dir / "records.jsonl"
    with records_path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(records_path)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    matrix = load_json(args.matrix)
    ensure_test_id(matrix, args.test_id)
    records_path = args.output_dir / args.test_id / "records.jsonl"
    if not records_path.exists():
        raise SystemExit(f"no records found: {records_path}")
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    required = matrix["tests"][args.test_id]["required_metrics"]
    summary = {
        "test_id": args.test_id,
        "records": len(records),
        "required_metrics": required,
        "records_path": str(records_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="print the V1-V4 matrix")

    init_run = sub.add_parser("init-run", help="create a run manifest")
    init_run.add_argument("test_id")

    record = sub.add_parser("record", help="append an experiment JSON artifact")
    record.add_argument("test_id")
    record.add_argument("--artifact", type=Path, required=True)
    record.add_argument("--notes", default="")

    summarize = sub.add_parser("summarize", help="summarize recorded artifacts")
    summarize.add_argument("test_id")

    args = parser.parse_args(argv)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "init-run":
        return cmd_init_run(args)
    if args.command == "record":
        return cmd_record(args)
    if args.command == "summarize":
        return cmd_summarize(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

