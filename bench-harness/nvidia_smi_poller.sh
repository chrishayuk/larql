#!/bin/bash
# Poll nvidia-smi at 100ms intervals, write timestamped VRAM usage to $1.
# Stop when $2 (a sentinel file) appears.
#
# Usage: ./nvidia_smi_poller.sh results/cfg-vram.csv /tmp/stop-sentinel &
set -u
out="$1"
sentinel="$2"
echo "timestamp_ms,vram_used_mib" > "$out"
while [ ! -f "$sentinel" ]; do
  ts=$(date +%s%3N)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "${ts},${used}" >> "$out"
  sleep 0.1
done
