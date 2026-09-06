#!/usr/bin/env bash
# Evict a mapped expert bank from the page cache, so the next arm faults
# its pages from the device.
#
# There is no supported way to drop macOS's unified page cache for one
# file, so the bank is displaced by reading OTHER large files through the
# cache until nothing of it is left: the HF checkpoint and the .s6
# container beside it, ~92 GB each, on a 128 GiB machine.
#
# This is only the setup. The WITNESS that an arm ran cold is the
# instrument's own binding line — `mapped … 0.000 GB resident`, read with
# mincore over the mapping — and a sample whose binding line reports more
# than 0.1 GB resident does not count as cold.
#
#   scripts/evict-bank.sh                 # evict, then report the time
#   MODELS=/some/where scripts/evict-bank.sh
set -euo pipefail

MODELS="${MODELS:-$HOME/chris-models}"
DISPLACERS=(
  "$MODELS/Kimi-Linear-48B-A3B-Instruct"
  "$MODELS/Kimi-Linear-48B-A3B-Instruct.s6.vindex3"
)
# Small files do not move a 128 GiB cache and cost a syscall each.
MIN_SIZE="${MIN_SIZE:-+100M}"

started=$SECONDS
for src in "${DISPLACERS[@]}"; do
  if [[ ! -d "$src" ]]; then
    echo "evict-bank: no such displacer directory: $src" >&2
    exit 1
  fi
  echo "evict-bank: streaming $src"
  find "$src" -type f -size "$MIN_SIZE" -print0 | xargs -0 -n1 cat >/dev/null
done
echo "evict-bank: displaced in $((SECONDS - started))s — check the arm's binding line for the cold witness"
