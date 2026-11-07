#!/bin/sh
# Reproduce the published results from a LARQL checkout: download the checkpoint, build LARQL at each commit in a separate
# worktree, encode a container with it, measure the container, and compare with expected/qwen3-0.6b.json. Then capture one
# run's kernel calls, judge them through VERIFY OPERATOR, and check that two deliberately faulty kernels are caught. The faults
# exist only in a temporary worktree.
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
# One real kernel call: capture a witness from the corrected container with the pinned LARQL, judge it through LQL and the
# verifier, then do the same with deliberately faulty kernels, which must be caught.
GEMV=$WORK/gemv
rm -rf "$GEMV"
mkdir -p "$GEMV"
commit=$(expect "d['operators']['larql_commit']")
tokens=$(expect "','.join(map(str, d['operators']['tokens']))")
tree=$WORK/larql-gemv
git -C "$REPO" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$REPO" fetch -q origin "$commit"
git -C "$REPO" worktree remove --force "$tree" 2>/dev/null || true
git -C "$REPO" worktree add -q --detach "$tree" "$commit"
capture() {  # capture <name>: build the tree as it stands, and write $GEMV/<name>/witness.json
  (cd "$tree" && LARQL_BUILD_COMMIT="$commit+$1" CARGO_TARGET_DIR=$WORK/target cargo build -q --release --locked -p larql-cli)
  mkdir -p "$GEMV/$1"
  env -i PATH=/usr/bin:/bin HOME="$HOME" "$WORK/target/release/larql" vindex3 exec "$WORK/corrected/q8_0.vindex3" \
    --backend production-q8 --representation-source stored --tokens "$tokens" --logit-dump "$GEMV/$1/logits.bin" \
    --gemv-witness "$GEMV/$1/witness" --gemv-witness-max-operands 400 >/dev/null 2>&1
}
ask() {  # ask <name> <operator> <contract>: run the statement, printing its exit status
  code=0
  env -i PATH=/usr/bin:/bin HOME="$HOME" LARQL_VERIFIER="$VERIFIER" "$WORK/target/release/larql" lql \
    "VERIFY OPERATOR \"$2\" USING CONTRACT \"$3\" IN WITNESS \"$GEMV/$1/witness/witness.json\" AGAINST \"$WORK/corrected/q8_0.vindex3\";" \
    >"$GEMV/$1/answer.txt" 2>&1 || code=$?
  echo "$code"
}
KERNEL=crates/larql-compute/src/cpu/kquant_gemv.rs
inject() {  # inject <old> <new>: change the kernel, and refuse to go on if that changed nothing
  git -C "$tree" checkout -q -- "$KERNEL"
  "$PY" -I - "$tree/$KERNEL" "$1" "$2" <<'PY'
import sys
path, old, new = sys.argv[1:4]
text = open(path).read()
if text.count(old) != 1:
    sys.exit(f"the kernel no longer contains exactly one {old!r}")
open(path, "w").write(text.replace(old, new))
PY
}
gemv_failed=0
want_exit() { [ "$1" = satisfied ] && echo 0 || echo 2; }

echo "== operators: LARQL $commit"
capture clean
"$PY" -I - "$EXPECTED" "$GEMV/clean" <<'PY' || gemv_failed=1
import hashlib, json, sys
e, d = json.load(open(sys.argv[1]))["operators"], sys.argv[2]
m = json.load(open(f"{d}/witness/witness.json"))
records = {r["tensor"]: r for r in m["records"]}
logits = hashlib.sha256(open(f"{d}/logits.bin", "rb").read()).hexdigest()
bad = []
if (len(records), m["calls_observed"]) != (e["records"], e["calls_observed"]):
    bad.append(f"{len(records)} operands of {m['calls_observed']} calls")
if logits != e["logit_dump_sha256"] or m["logit_dump"]["sha256"] != logits:
    bad.append(f"logits {logits}")
for op, want in e["checked"].items():
    if records[op]["x"]["sha256"] != want["x_sha256"] or records[op]["y"]["sha256"] != want["y_sha256"]:
        bad.append(f"{op}: x or y differs")
print(f"capture: {len(records)} operands of {m['calls_observed']} calls: " + ("matches expected" if not bad else f"DIFFERS {bad}"))
sys.exit(1 if bad else 0)
PY
for op in $(expect "' '.join(d['operators']['checked'])"); do
  for contract in Q8_0_GEMV_SEQ Q8_0_GEMV_B1; do
    code=$(ask clean "$op" "$contract")
    [ "$code" = 0 ] && verdict=matches || { verdict=DIFFERS; gemv_failed=1; }
    echo "clean: $op $contract: exit $code: $verdict"
  done
done
checked=$(expect "list(d['operators']['checked'])[0]")
for fault in reverse_blocks unsigned_codes; do
  case $fault in
    reverse_blocks) inject 'for b in 0..per_row {' 'for b in (0..per_row).rev() {' ;;
    unsigned_codes) inject '(c as i8 as f32)' '(c as f32)' ;;
  esac
  capture "$fault"
  for contract in Q8_0_GEMV_SEQ Q8_0_GEMV_B1; do
    want=$(want_exit "$(expect "d['operators']['faults']['$fault']['$contract']")")
    code=$(ask "$fault" "$checked" "$contract")
    [ "$code" = "$want" ] && verdict="matches expected" || { verdict="DIFFERS, expected exit $want"; gemv_failed=1; }
    echo "faulty kernel $fault: $contract: exit $code: $verdict"
  done
done
git -C "$REPO" worktree remove --force "$tree"
[ "$gemv_failed" = 0 ] || failed=1

[ "$failed" = 0 ] && echo "Reproduced." || { echo "NOT reproduced."; exit 1; }
