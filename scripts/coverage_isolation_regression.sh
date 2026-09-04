#!/usr/bin/env bash
# NEGATIVE CONTROL — not a reproducer. Read this before citing it.
#
# This runs models -> vindex -> models on ONE unchanged tree with no manual
# clean, and asserts run 1 and run 3 are identical. It was written expecting
# to reproduce the 2026-09-04 contamination, and it did not: the sequence
# passes on the PRE-FIX harness too, at the same commit, with a witnessed
# identical source tree. `cargo llvm-cov` cleans old build artifacts by
# default, which is consistent with that.
#
# What it therefore establishes, which is worth keeping:
#
#     ordinary repeated gates on an unchanged tree do NOT contaminate
#
# That kills the broader claim that cross-crate gate ordering is by itself
# sufficient. The contamination observed on 2026-09-04 remains unreproduced;
# a second differential experiment (models at tree A -> models at tree B,
# pre-fix vs cleaned) also produced no contamination in either arm.
#
# Do NOT tune this until it fails. If you extend it, state the new
# hypothesis first and keep this arm as the control.
#
# SLOW (~45 min): every gate now rebuilds. Manual, never CI.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
OUT=$(mktemp -d)
trap 'echo "artifacts in $OUT"' EXIT

percentages() {  # summary.json -> "path percent" per file, sorted
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
for f in sorted(d["data"][0]["files"], key=lambda f: f["filename"]):
    s = f["summary"]["lines"]
    print(f"{f['"'"'filename'"'"']} {s['"'"'percent'"'"']:.4f}")
' "$1"
}

echo "### 1/3  models"
make larql-models-coverage-summary > "$OUT/models1.log" 2>&1
cp coverage/larql-models/summary.json "$OUT/models1.json"

echo "### 2/3  vindex (the contaminant)"
make larql-vindex-coverage-summary > "$OUT/vindex.log" 2>&1

echo "### 3/3  models again, no manual clean"
make larql-models-coverage-summary > "$OUT/models2.log" 2>&1
cp coverage/larql-models/summary.json "$OUT/models2.json"

percentages "$OUT/models1.json" > "$OUT/models1.txt"
percentages "$OUT/models2.json" > "$OUT/models2.txt"

if diff -u "$OUT/models1.txt" "$OUT/models2.txt" > "$OUT/diff.txt"; then
  echo "PASS: identical per-file coverage across the A -> B -> A sequence"
else
  echo "FAIL: coverage moved with no source change - contamination is back" >&2
  head -40 "$OUT/diff.txt" >&2
  exit 1
fi

echo "### 4/4  both gates in ONE make invocation"
make larql-models-coverage-summary larql-vindex-coverage-summary \
  > "$OUT/multi.log" 2>&1
echo "PASS: multi-target invocation, both gates green"
