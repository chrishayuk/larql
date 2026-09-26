#!/usr/bin/env bash
# The oracle is only an authority if its SOURCE is the one that was
# judged. This checks the installed reference against the recorded
# hashes and fails loudly on drift — a silent `transformers` upgrade
# would otherwise change every GLM parity number without notice.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
venv="${1:-${LARQL_GLM_ORACLE:-$repo/.glm-oracle-venv}}"

sp="$("$venv/bin/python" -c 'import transformers,os;print(os.path.dirname(transformers.__file__))')/models"
ver="$("$venv/bin/python" -c 'import transformers;print(transformers.__version__)')"
echo "transformers $ver at $sp"

# Only the glm5 rows; the file also pins Inkling, which this oracle does
# not use and must not be made to depend on.
grep '^[0-9a-f]\{64\}  glm5_next/' "$repo/scripts/glm_reference_sources.sha256" \
  | (cd "$sp" && shasum -a 256 -c -)
echo "reference sources match scripts/glm_reference_sources.sha256"
