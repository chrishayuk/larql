#!/bin/sh
# Reproduce the two published results from a LARQL checkout: download the checkpoint, build LARQL at each commit in a separate
# worktree, encode a container with it, measure the container, and compare with expected/qwen3-0.6b.json.
#
#   verifier/reproduce.sh [work directory, default verifier/work]
#
# Needs git, curl, a Rust toolchain and whatever LARQL needs to build, about 10 GB of disk, and this package installed in
# verifier/.venv (see README.md). Set VERIFIER to use another installation. Your checkout's files are not touched.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
WORK=$(mkdir -p "${1:-$HERE/work}" && cd "${1:-$HERE/work}" && pwd)
VERIFIER=${VERIFIER:-$HERE/.venv/bin/larql-verifier}
PY=$(dirname "$VERIFIER")/python
EXPECTED=$HERE/expected/qwen3-0.6b.json
expect() { "$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$EXPECTED" "$1"; }

REVISION=$(expect "d['source']['revision']")
SOURCE=$WORK/source
mkdir -p "$SOURCE"
for f in model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt LICENSE README.md; do
  [ -f "$SOURCE/$f" ] || curl -fsSL -o "$SOURCE/$f" "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/$REVISION/$f"
done

failed=0
for which in original corrected; do
  commit=$(expect "d['containers']['$which']['larql_commit']")
  echo "== $which: LARQL $commit"
  git -C "$REPO" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$REPO" fetch -q origin "$commit"
  tree=$WORK/larql-$which
  git -C "$REPO" worktree remove --force "$tree" 2>/dev/null || true
  git -C "$REPO" worktree add -q --detach "$tree" "$commit"
  (cd "$tree" && CARGO_TARGET_DIR=$WORK/target cargo build -q --release --locked -p larql-cli)
  git -C "$REPO" worktree remove --force "$tree"
  out=$WORK/$which
  rm -rf "$out"
  mkdir -p "$out"
  env -i PATH=/usr/bin:/bin HOME="$HOME" "$WORK/target/release/larql" vindex3 encode "$SOURCE" --output "$out/c.vindex3" >/dev/null 2>&1
  env -i PATH=/usr/bin:/bin HOME="$HOME" "$WORK/target/release/larql" vindex3 represent "$out/c.vindex3" --output "$out/q8_0.vindex3" --encoding Q8_0 >/dev/null 2>&1
  "$VERIFIER" contract --contract Q8_0_T2 --container "$out/q8_0.vindex3" --source "$SOURCE" > "$out/result.json"
  "$PY" -I - "$EXPECTED" "$which" "$out" <<'PY' || failed=1
import hashlib, json, sys
expected, which, out = json.load(open(sys.argv[1]))["containers"][sys.argv[2]], sys.argv[2], sys.argv[3]
r = json.load(open(f"{out}/result.json"))
h = hashlib.sha256()
with open(f"{out}/q8_0.vindex3/segments/target.decoder_stack@Q8_0.bin", "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
got = {"status": r["status"], "violations": r["violations"], "report_sha256": r.get("report_sha256"),
       "q8_0_segment_sha256": h.hexdigest()}
bad = {k: (v, expected[k]) for k, v in got.items() if v != expected[k]}
print(f"{which}: {r['status']}, {r['violations']} violations: " + ("matches expected" if not bad else f"DIFFERS {bad}"), r["reasons"] or "")
sys.exit(1 if bad else 0)
PY
done
[ "$failed" = 0 ] && echo "Reproduced." || { echo "NOT reproduced."; exit 1; }
