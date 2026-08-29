#!/bin/bash
set -e

# The public box holds no state worth keeping: the demo container is
# regenerated at every boot (kilobytes, deterministic), so a wiped
# machine rebuilds identically and there is no volume to manage.
DEMO_DIR="${DEMO_DIR:-/data/vindex3-demo}"
vindex3-demo "$DEMO_DIR"

# Working directory = the published catalogue: SHOW MODELS lists the
# process cwd, so the public listing is exactly what lives here.
cd "$(dirname "$DEMO_DIR")"

exec larql-server "$DEMO_DIR" \
  --public-explorer \
  --cors \
  --no-docs \
  --rate-limit "${RATE_LIMIT:-120/min}" \
  --max-concurrent "${MAX_CONCURRENT:-4}" \
  --trust-forwarded-for \
  --port "${PORT:-8080}" \
  --host 0.0.0.0
