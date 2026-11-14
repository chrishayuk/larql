#!/bin/sh
# Reproduce the published results from a LARQL checkout: download the checkpoint, build LARQL at each commit in a separate
# worktree, encode a container with it, measure the container, and compare with expected/qwen3-0.6b.json. Then capture one
# run's kernel calls, judge them through VERIFY OPERATOR, and check that two deliberately faulty kernels are caught. The faults
# exist only in a temporary worktree.
#
#   verifier/reproduce.sh [work directory, default verifier/work]
#
# Needs git, curl, a Rust toolchain and whatever LARQL needs to build, about 12 GB of disk, and this package installed in
# verifier/.venv (see README.md). Set VERIFIER to use another installation. Your checkout's files are not touched.
#
# Every engine step writes its output to a log under the work directory. Whatever stops the run, it ends by printing
# "Reproduced." or "NOT reproduced.", and the temporary worktree is removed.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
WORK=$(mkdir -p "${1:-$HERE/work}" && cd "${1:-$HERE/work}" && pwd)
VERIFIER=${VERIFIER:-$HERE/.venv/bin/larql-verifier}
PY=$(dirname "$VERIFIER")/python
EXPECTED=$HERE/expected/qwen3-0.6b.json
expect() { "$PY" -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$EXPECTED" "$1"; }

tree=
reported=
remove_tree() {
  if [ -n "$tree" ]; then
    git -C "$REPO" worktree remove --force "$tree" 2>/dev/null || true
    tree=
  fi
}
on_exit() {
  status=$?
  remove_tree
  if [ "$status" != 0 ] && [ -z "$reported" ]; then
    echo "NOT reproduced: stopped early (exit $status). The logs are under $WORK."
  fi
}
trap on_exit EXIT
trap 'exit 130' INT TERM

REVISION=$(expect "d['source']['revision']")
SOURCE=$WORK/source
mkdir -p "$SOURCE"
for f in model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt LICENSE README.md; do
  [ -f "$SOURCE/$f" ] && continue
  # Download beside the target and rename, so an interrupted download is never mistaken for a complete file.
  curl -fsSL -o "$SOURCE/$f.part" "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/$REVISION/$f"
  mv "$SOURCE/$f.part" "$SOURCE/$f"
done

failed=0
for which in original corrected; do
  commit=$(expect "d['containers']['$which']['larql_commit']")
  echo "== $which: LARQL $commit"
  out=$WORK/$which
  rm -rf "$out"
  mkdir -p "$out"
  git -C "$REPO" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$REPO" fetch -q origin "$commit"
  tree=$WORK/larql-$which
  git -C "$REPO" worktree remove --force "$tree" 2>/dev/null || true
  git -C "$REPO" worktree add -q --detach "$tree" "$commit"
  (cd "$tree" && CARGO_TARGET_DIR=$WORK/target cargo build -q --release --locked -p larql-cli) >"$out/build.log" 2>&1 \
    || { echo "$which: build failed, see $out/build.log"; exit 1; }
  remove_tree
  env -i PATH=/usr/bin:/bin HOME="$HOME" "$WORK/target/release/larql" vindex3 encode "$SOURCE" --output "$out/c.vindex3" \
    >"$out/encode.log" 2>&1 || { echo "$which: encode failed, see $out/encode.log"; exit 1; }
  env -i PATH=/usr/bin:/bin HOME="$HOME" "$WORK/target/release/larql" vindex3 represent "$out/c.vindex3" --output "$out/q8_0.vindex3" \
    --encoding Q8_0 >"$out/represent.log" 2>&1 || { echo "$which: represent failed, see $out/represent.log"; exit 1; }
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
# One real kernel call: capture a witness from the expected container with the pinned LARQL, judge it through LQL and the
# verifier, then do the same with deliberately faulty kernels, which must be caught.
GEMV=$WORK/gemv
rm -rf "$GEMV"
mkdir -p "$GEMV"
commit=$(expect "d['operators']['larql_commit']")
tokens=$(expect "','.join(map(str, d['operators']['tokens']))")
CONTAINER=$WORK/$(expect "d['operators']['container']")/q8_0.vindex3
git -C "$REPO" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$REPO" fetch -q origin "$commit"
tree=$WORK/larql-gemv
git -C "$REPO" worktree remove --force "$tree" 2>/dev/null || true
git -C "$REPO" worktree add -q --detach "$tree" "$commit"
# These functions run as conditions, where `set -e` does not apply, so each step returns its own failure.
capture() {  # capture <name>: build the tree as it stands, and write $GEMV/<name>/witness.json; logs beside it
  mkdir -p "$GEMV/$1"
  (cd "$tree" && LARQL_BUILD_COMMIT="$commit+$1" CARGO_TARGET_DIR=$WORK/target cargo build -q --release --locked -p larql-cli) \
    >"$GEMV/$1/build.log" 2>&1 || { echo "$1: build failed, see $GEMV/$1/build.log"; return 1; }
  env -i PATH=/usr/bin:/bin HOME="$HOME" "$WORK/target/release/larql" vindex3 exec "$CONTAINER" \
    --backend production-q8 --representation-source stored --tokens "$tokens" --logit-dump "$GEMV/$1/logits.bin" \
    --gemv-witness "$GEMV/$1/witness" --gemv-witness-max-operands 400 >"$GEMV/$1/exec.log" 2>&1 \
    || { echo "$1: capture failed (exit $?), see $GEMV/$1/exec.log"; return 1; }
}
ask() {  # ask <name> <operator> <contract>: run the statement into $GEMV/<name>/<operator>.<contract>.txt, printing its exit status
  code=0
  env -i PATH=/usr/bin:/bin HOME="$HOME" LARQL_VERIFIER="$VERIFIER" "$WORK/target/release/larql" lql \
    "VERIFY OPERATOR \"$2\" USING CONTRACT \"$3\" IN WITNESS \"$GEMV/$1/witness/witness.json\" AGAINST \"$CONTAINER\";" \
    >"$GEMV/$1/$2.$3.txt" 2>&1 || code=$?
  echo "$code"
}
KERNEL=crates/larql-compute/src/cpu/kquant_gemv.rs
inject() {  # inject <old> <new>: change the kernel, and fail if that changed nothing
  git -C "$tree" checkout -q -- "$KERNEL" || return 1
  "$PY" -I - "$tree/$KERNEL" "$1" "$2" <<'PY' || return 1
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
if capture clean; then
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
    if op not in records or records[op]["x"]["sha256"] != want["x_sha256"] or records[op]["y"]["sha256"] != want["y_sha256"]:
        bad.append(f"{op}: missing, or x or y differs")
print(f"capture: {len(records)} operands of {m['calls_observed']} calls: " + ("matches expected" if not bad else f"DIFFERS {bad}"))
sys.exit(1 if bad else 0)
PY
  for op in $(expect "' '.join(d['operators']['checked'])"); do
    for contract in Q8_0_GEMV_SEQ Q8_0_GEMV_B1; do
      code=$(ask clean "$op" "$contract")
      [ "$code" = 0 ] && verdict=matches || { verdict=DIFFERS; gemv_failed=1; }
      echo "clean: $op $contract: exit $code: $verdict"
    done
    # The verifier reports the worst error as a fraction of the B1 bound; it is deterministic for a fixed witness.
    "$PY" -I - "$EXPECTED" "$op" "$GEMV/clean/$op.Q8_0_GEMV_B1.txt" <<'PY' || gemv_failed=1
import json, sys
want = json.load(open(sys.argv[1]))["operators"]["checked"][sys.argv[2]]["worst_b1_ratio"]
label = "worst error / bound:"
got = [line.strip()[len(label):].strip() for line in open(sys.argv[3]) if line.strip().startswith(label)]
ok = len(got) == 1 and abs(float(got[0]) - float(want)) <= 1e-9 * abs(float(want))
print(f"clean: {sys.argv[2]} B1 worst error / bound {got[0] if len(got) == 1 else 'not reported'}: "
      + ("matches expected" if ok else f"DIFFERS, expected {want}"))
sys.exit(0 if ok else 1)
PY
  done
  checked=$(expect "list(d['operators']['checked'])[0]")
  for fault in reverse_blocks unsigned_codes; do
    case $fault in
      reverse_blocks) old='for b in 0..per_row {'; new='for b in (0..per_row).rev() {' ;;
      unsigned_codes) old='(c as i8 as f32)'; new='(c as f32)' ;;
    esac
    if ! inject "$old" "$new"; then
      echo "faulty kernel $fault: could not inject the fault: NOT checked"
      gemv_failed=1
      continue
    fi
    if ! capture "$fault"; then
      echo "faulty kernel $fault: NOT checked"
      gemv_failed=1
      continue
    fi
    for contract in Q8_0_GEMV_SEQ Q8_0_GEMV_B1; do
      want=$(want_exit "$(expect "d['operators']['faults']['$fault']['$contract']")")
      code=$(ask "$fault" "$checked" "$contract")
      [ "$code" = "$want" ] && verdict="matches expected" || { verdict="DIFFERS, expected exit $want"; gemv_failed=1; }
      echo "faulty kernel $fault: $contract: exit $code: $verdict"
    done
  done
else
  echo "operators: NOT checked"
  gemv_failed=1
fi
remove_tree
[ "$gemv_failed" = 0 ] || failed=1

reported=1
[ "$failed" = 0 ] && echo "Reproduced." || { echo "NOT reproduced."; exit 1; }
