#!/usr/bin/env bash
# Reconstruct the GLM-5.3-Flash oracle environment from nothing.
#
#   tools/oracles/glm5/bootstrap.sh [venv-path]
#
# Default venv path is $LARQL_GLM_ORACLE, else ./.glm-oracle-venv.
# Idempotent: an existing venv is reused, and the source check runs
# either way.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
venv="${1:-${LARQL_GLM_ORACLE:-$repo/.glm-oracle-venv}}"

if [ ! -x "$venv/bin/python" ]; then
  echo "creating $venv"
  # 3.12 because that is what the pinned wheels were resolved against.
  python3.12 -m venv "$venv" 2>/dev/null || python3 -m venv "$venv"
  "$venv/bin/pip" install --quiet --upgrade pip
  "$venv/bin/pip" install --quiet -r "$here/requirements.txt"
else
  echo "reusing $venv"
fi

"$here/verify_sources.sh" "$venv"
echo
echo "oracle ready:  export LARQL_GLM_ORACLE=$venv"
