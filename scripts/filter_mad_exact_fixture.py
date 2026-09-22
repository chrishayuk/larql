#!/usr/bin/env python3
"""Materialise the sealed exact-success subset of a MAD query fixture."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = {}
    for line_number, line in enumerate(args.queries.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("id")
        if not sample_id or sample_id in rows:
            raise ValueError(f"invalid or duplicate id at line {line_number}")
        rows[sample_id] = row
    manifest = json.loads((args.capture / "manifest.json").read_text())
    selected = []
    for sample in manifest["samples"]:
        row = rows.get(sample["id"])
        if row is None:
            raise ValueError(f"fixture lacks captured sample {sample['id']}")
        for field in ("split", "group_id", "operation_id", "wording_id", "alias_id"):
            if row.get(field) != sample.get(field):
                raise ValueError(f"sample {sample['id']} differs at {field}")
        if sample.get("answer_exact") is True:
            selected.append(row)
    if len(selected) != sum(sample.get("answer_exact") is True for sample in manifest["samples"]):
        raise AssertionError("exact-success selection count drifted")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in selected))
    history = sum(row["split"] == "history" for row in selected)
    query = sum(row["split"] == "query" for row in selected)
    print(f"wrote {args.output}: {len(selected)} rows ({history} history, {query} query)")


if __name__ == "__main__":
    main()
