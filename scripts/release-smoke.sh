#!/usr/bin/env bash
# Post-publish smoke for a LARQL GitHub Release.
#
# Proves the README / docs/getting-started.md install + first-run path works
# against the archives a release actually published — not against a fresh
# build. Called by .github/workflows/release-smoke.yml (every OS) and
# .github/workflows/release-model-smoke.yml (Qwen, Linux); runs the same way
# on a laptop. Windows runs it under the runner's Git Bash.
#
#   release-smoke.sh install <tag> <work>  download <tag>'s archive for this
#                                          host the README way, extract, check
#                                          `larql --version` == tag and that
#                                          larql-server runs
#   release-smoke.sh archive <tag> <dist> <work>
#                                          pre-publish: the same version check
#                                          on this host's archive in <dist>
#                                          (release.yml, before it publishes)
#   release-smoke.sh latest  <tag> <work>  the README's /releases/latest/ path
#                                          serves a working binary of the
#                                          release GitHub marks latest (== tag
#                                          when <tag> is latest; retried, so a
#                                          just-published release can settle;
#                                          LARQL_SMOKE_EXPECT_LATEST=0 drops
#                                          the == tag requirement)
#   release-smoke.sh fixture <tag> <work>  offline: write a tiny checkpoint,
#                                          `larql vindex3 encode` it, `larql
#                                          run` it, `larql serve` it and answer
#                                          /v1/health, /v1/models and
#                                          /v1/chat/completions
#   release-smoke.sh model   <tag> <work>  the README quick start on
#                                          Qwen/Qwen3-0.6B from an empty cache:
#                                          run, chat, serve; each must say Paris
#   release-smoke.sh all     <tag> <work>  install, latest, fixture
#
# <work> is a scratch directory the script owns. `install` must run first;
# the other modes use the binaries it extracted.

set -euo pipefail

REPO="${LARQL_REPO:-${GITHUB_REPOSITORY:-chrishayuk/larql}}"
RELEASES="https://github.com/${REPO}/releases"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${LARQL_SMOKE_PORT:-18080}"
SERVER_READY_SECS=120          # model load: Qwen3-0.6B on a 4-core runner
REQUEST_TIMEOUT_SECS=600       # one chat completion on CPU
LATEST_ATTEMPTS=10             # /latest/ settling after make_latest
LATEST_BACKOFF_SECS=30
CURL_RETRIES=5

# Fixture smoke: tokens generated, and the turn the fixture's WordLevel
# tokenizer encodes (token `[i]` is id `i`; scripts/release_smoke_fixture.py).
FIXTURE_TOKENS=4
FIXTURE_PROMPT='[3] [4]'

# Model smoke: docs/getting-started.md's quick start, verbatim.
MODEL_HF="hf://Qwen/Qwen3-0.6B"
MODEL_NAME="Qwen3-0.6B"
MODEL_PROMPT="What is the capital of France? Answer in one word. /no_think"
MODEL_RAW_PROMPT="The capital of France is"
MODEL_RAW_TOKENS=12
MODEL_CHAT_TOKENS=32
MODEL_ANSWER="Paris"

log() { printf '==> %s\n' "$*" >&2; }
die() {
  # `::error::` is an annotation on GitHub runners and plain text elsewhere.
  printf '::error::release-smoke: %s\n' "$*" >&2
  exit 1
}

# ── Host ────────────────────────────────────────────────────────────────

# The release target this host installs, by the same names release.yml
# publishes and the README tells a user to pick.
host_target() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os/$arch" in
    Darwin/arm64) echo aarch64-apple-darwin ;;
    Linux/x86_64) echo x86_64-unknown-linux-gnu ;;
    MINGW*/x86_64 | MSYS*/x86_64 | CYGWIN*/x86_64) echo x86_64-pc-windows-msvc ;;
    *) die "no published LARQL binary for $os/$arch" ;;
  esac
}

is_windows() { [[ "$(host_target)" == *-windows-* ]]; }
exe() { if is_windows; then echo "$1.exe"; else echo "$1"; fi; }

asset_name() {
  local target
  target="$(host_target)"
  if is_windows; then echo "larql-$target.zip"; else echo "larql-$target.tar.gz"; fi
}

# A Python that runs (the Windows `python3` can be a Store stub that does not).
python_bin() {
  local p
  for p in "${PYTHON:-}" python3 python; do
    [ -n "$p" ] && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 8))' >/dev/null 2>&1 && {
      echo "$p"
      return
    }
  done
  die "no Python 3.8+ on PATH (needed for the fixture writer and JSON checks)"
}

# `$1` is v-prefixed; `larql --version` prints the bare semver.
tag_version() {
  [[ "$1" == v* ]] || die "tag '$1' does not start with v"
  echo "${1#v}"
}

# ── Download + install ──────────────────────────────────────────────────

# Download `$1` (a release URL) and extract it into `$2`, the way the README
# does: `curl -L … | tar xz` on Unix, the .zip on Windows.
fetch_extract() {
  local url="$1" dest="$2"
  rm -rf "$dest"
  mkdir -p "$dest"
  log "download $url"
  if is_windows; then
    curl -fsSL --retry "$CURL_RETRIES" --retry-all-errors -o "$dest/archive.zip" "$url"
    if command -v unzip >/dev/null 2>&1; then
      unzip -q "$dest/archive.zip" -d "$dest"
    else
      powershell.exe -NoProfile -Command \
        "Expand-Archive -LiteralPath '$(cygpath -w "$dest/archive.zip")' -DestinationPath '$(cygpath -w "$dest")'"
    fi
    rm -f "$dest/archive.zip"
  else
    curl -fsSL --retry "$CURL_RETRIES" --retry-all-errors "$url" | tar xz -C "$dest"
  fi
}

# The directory an archive extracted to: `larql-<target>/` holding both
# executables (release.yml's packaging).
extracted_bin_dir() {
  local dir
  dir="$1/larql-$(host_target)"
  [ -d "$dir" ] || die "archive did not contain larql-$(host_target)/ (got: $(ls "$1"))"
  echo "$dir"
}

# Assert both executables are present and report version `$2`.
check_binaries() {
  local bin="$1" want="$2" got name
  for name in larql larql-server; do
    [ -x "$bin/$(exe "$name")" ] || die "$(exe "$name") missing from the archive"
    got="$("$bin/$(exe "$name")" --version | tr -d '\r')"
    [ "$got" = "$name $want" ] || die "$name --version printed '$got', want '$name $want'"
    log "$got"
  done
  "$bin/$(exe larql-server)" --help >/dev/null || die "larql-server --help failed"
  "$bin/$(exe larql)" --help >/dev/null || die "larql --help failed"
}

cmd_install() {
  local tag="$1" work="$2" want bin
  want="$(tag_version "$tag")"
  fetch_extract "$RELEASES/download/$tag/$(asset_name)" "$work/tag"
  bin="$(extracted_bin_dir "$work/tag")"
  check_binaries "$bin" "$want"
}

# Pre-publish: the host's archive from a local directory (release.yml's
# downloaded build artefacts), so a tag whose workspace version was not
# bumped never reaches a release.
cmd_archive() {
  local tag="$1" dist="$2" work="$3" want archive
  want="$(tag_version "$tag")"
  archive="$dist/$(asset_name)"
  [ -f "$archive" ] || die "no $(asset_name) in $dist"
  rm -rf "$work/archive"
  mkdir -p "$work/archive"
  if is_windows; then
    unzip -q "$archive" -d "$work/archive"
  else
    tar xzf "$archive" -C "$work/archive"
  fi
  check_binaries "$(extracted_bin_dir "$work/archive")" "$want"
}

# The tag GitHub currently marks latest — read from the same redirect the
# README's /releases/latest/download/ URL follows, so no API token or rate
# limit is involved.
latest_tag() {
  curl -fsSI --retry "$CURL_RETRIES" --retry-all-errors "$RELEASES/latest" |
    tr -d '\r' | awk 'tolower($1) == "location:" { n = split($2, p, "/"); print p[n] }' | tail -n1
}

cmd_latest() {
  local tag="$1" work="$2" latest attempt bin got want
  for ((attempt = 1; attempt <= LATEST_ATTEMPTS; attempt++)); do
    latest="$(latest_tag)"
    [ -n "$latest" ] || die "could not resolve $RELEASES/latest"
    # A release smoked right after publishing should already be latest;
    # GitHub's redirect can trail make_latest briefly. A dispatch for an
    # older tag (or a newer release since) legitimately differs: then the
    # check is that /latest/ serves a working binary of whatever IS latest.
    if [ "$latest" = "$tag" ] || [ "${LARQL_SMOKE_EXPECT_LATEST:-1}" = 0 ]; then
      break
    fi
    if ((attempt == LATEST_ATTEMPTS)); then
      die "/releases/latest still points at $latest, not $tag, after $LATEST_ATTEMPTS attempts — was make_latest lost, or is a newer release out? (set LARQL_SMOKE_EXPECT_LATEST=0 to smoke an older tag)"
    fi
    log "/releases/latest -> $latest, waiting for $tag ($attempt/$LATEST_ATTEMPTS)"
    sleep "$LATEST_BACKOFF_SECS"
  done
  want="$(tag_version "$latest")"
  fetch_extract "$RELEASES/latest/download/$(asset_name)" "$work/latest"
  bin="$(extracted_bin_dir "$work/latest")"
  got="$("$bin/$(exe larql)" --version | tr -d '\r')"
  # The redirect and the asset download are two requests; if latest moved
  # between them the version names the culprit.
  [ "$got" = "larql $want" ] || die "/releases/latest/download/ served '$got', but /releases/latest is $latest"
  check_binaries "$bin" "$want"
}

# ── Running ─────────────────────────────────────────────────────────────

# Put the extracted binaries on PATH, as the README's install does —
# `larql serve` finds `larql-server` beside itself or on PATH.
use_installed() {
  local work="$1" bin
  bin="$(extracted_bin_dir "$work/tag")"
  export PATH="$bin:$PATH"
  log "using $(command -v "$(exe larql)")"
}

SERVER_PID=""

# Start `larql serve <args>` in the background and wait for /v1/health.
start_server() {
  local log_file="$1"
  shift
  larql serve "$@" --port "$PORT" >"$log_file" 2>&1 &
  SERVER_PID=$!
  local deadline=$((SECONDS + SERVER_READY_SECS))
  until curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      cat "$log_file" >&2
      die "larql serve exited before answering /v1/health"
    fi
    if ((SECONDS > deadline)); then
      cat "$log_file" >&2
      die "larql serve did not answer /v1/health within ${SERVER_READY_SECS}s"
    fi
    sleep 1
  done
  log "server up on :$PORT (pid $SERVER_PID)"
}

# Stop the server tree. `larql serve` spawns `larql-server` as a child and
# waits on it, so signalling `larql` alone would orphan the listener.
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  local pid="$SERVER_PID" children
  SERVER_PID=""
  if [ -r "/proc/$pid/winpid" ]; then
    # MSYS: the native tree is only reachable through its Windows pid.
    taskkill //F //T //PID "$(cat "/proc/$pid/winpid")" >/dev/null 2>&1 || true
  else
    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    # shellcheck disable=SC2086 # word-split the child pid list
    kill $children "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  local deadline=$((SECONDS + 10))
  while curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; do
    ((SECONDS < deadline)) || die "server on :$PORT still answering after shutdown"
    sleep 1
  done
  log "server stopped"
}
trap stop_server EXIT

# POST a chat completion; prints the body, fails on a non-200.
chat_completion() {
  local body="$1" out status
  out="$(mktemp)"
  status="$(curl -sS --max-time "$REQUEST_TIMEOUT_SECS" -o "$out" -w '%{http_code}' \
    -H 'Content-Type: application/json' -d "$body" \
    "http://127.0.0.1:$PORT/v1/chat/completions")"
  cat "$out"
  rm -f "$out"
  [ "$status" = 200 ] || die "/v1/chat/completions returned HTTP $status"
}

# Validate an OpenAI chat.completion body on stdin; print the reply text.
# `$1` = expected completion tokens, or "" for no count check.
check_chat_shape() {
  "$(python_bin)" -I -c '
import json, sys
want = sys.argv[1]
body = json.load(sys.stdin)
assert body.get("object") == "chat.completion", body
choice = body["choices"][0]
msg = choice["message"]
assert msg["role"] == "assistant", msg
assert isinstance(msg["content"], str) and msg["content"].strip(), msg
assert choice["finish_reason"] in ("stop", "length"), choice
usage = body["usage"]
assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0, usage
assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], usage
if want:
    assert usage["completion_tokens"] == int(want), usage
print(msg["content"])
' "$1"
}

check_health_and_models() {
  local health models
  health="$(curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/health")"
  models="$(curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/models")"
  "$(python_bin)" -I -c '
import json, sys
health, models = json.loads(sys.argv[1]), json.loads(sys.argv[2])
assert health.get("status") == "ok", health
data = models["data"]
assert len(data) == 1 and data[0]["id"] and data[0].get("loaded", True), models
print(data[0]["id"])
' "$health" "$models"
}

# A JSON string literal for `$1`.
json_str() { "$(python_bin)" -I -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"; }

cmd_fixture() {
  local work="$1/fixture" run_out reply model_id
  use_installed "$1"
  rm -rf "$work"
  mkdir -p "$work"
  # Nothing from the host's caches may stand in for the shipped binary.
  export HOME="$work/home"
  mkdir -p "$HOME"

  "$(python_bin)" -I "$SCRIPTS/release_smoke_fixture.py" "$work/checkpoint"
  larql vindex3 encode "$work/checkpoint" --output "$work/container"
  [ -f "$work/container/tokenizer.json" ] || die "encode did not carry tokenizer.json into the container"

  log "larql run (offline)"
  run_out="$(larql run "$work/container" "$FIXTURE_PROMPT" -n "$FIXTURE_TOKENS" | tr -d '\r')"
  printf '%s\n' "$run_out"
  "$(python_bin)" -I -c '
import re, sys
toks = re.findall(r"\[\d+\]", sys.argv[1])
assert len(toks) == int(sys.argv[2]), f"want {sys.argv[2]} fixture tokens, got {sys.argv[1]!r}"
' "$run_out" "$FIXTURE_TOKENS"

  log "larql serve"
  start_server "$work/serve.log" "$work/container"
  model_id="$(check_health_and_models)"
  log "serving model id '$model_id'"
  reply="$(chat_completion "{\"model\":$(json_str "$model_id"),\"max_tokens\":$FIXTURE_TOKENS,\"messages\":[{\"role\":\"user\",\"content\":$(json_str "$FIXTURE_PROMPT")}]}" |
    check_chat_shape "$FIXTURE_TOKENS")"
  log "chat reply: $reply"
  # `larql run` and `larql serve` share one prompt derivation and greedy
  # decode; a disagreement is worth a look but host numerics (Metal vs CPU)
  # can legitimately split near-ties on random weights, so it only warns.
  if [ "$(echo "$reply" | xargs)" != "$(echo "$run_out" | xargs)" ]; then
    printf '::warning::release-smoke: run (%s) and serve (%s) generated different fixture tokens\n' "$run_out" "$reply" >&2
  fi
  stop_server
  log "fixture smoke passed"
}

assert_answer() {
  local what="$1" text="$2"
  printf '%s\n' "$text"
  grep -q "$MODEL_ANSWER" <<<"$text" || die "$what did not answer '$MODEL_ANSWER'"
}

cmd_model() {
  local work="$1/model" out reply
  use_installed "$1"
  rm -rf "$work"
  mkdir -p "$work"
  # Cache-free on purpose: a fresh HOME means an empty ~/.cache/larql and
  # ~/.cache/huggingface, so this is the first-run encode a new user gets.
  export HOME="$work/home"
  mkdir -p "$HOME"
  unset HF_HOME HF_HUB_CACHE

  log "larql run $MODEL_HF (first run: download + encode)"
  out="$(larql run "$MODEL_HF" "$MODEL_PROMPT")"
  assert_answer "larql run $MODEL_HF" "$out"

  log "larql run $MODEL_NAME --raw (cached shorthand, no encode)"
  out="$(larql run "$MODEL_NAME" "$MODEL_RAW_PROMPT" -n "$MODEL_RAW_TOKENS" --raw)"
  assert_answer "larql run --raw" "$out"

  log "larql chat $MODEL_NAME"
  out="$(printf '%s\n' "$MODEL_PROMPT" | larql chat "$MODEL_NAME")"
  assert_answer "larql chat" "$out"

  log "larql serve $MODEL_NAME"
  start_server "$work/serve.log" "$MODEL_NAME"
  check_health_and_models >/dev/null
  reply="$(chat_completion "{\"model\":$(json_str "$MODEL_NAME"),\"temperature\":0,\"max_tokens\":$MODEL_CHAT_TOKENS,\"messages\":[{\"role\":\"user\",\"content\":$(json_str "$MODEL_PROMPT")}]}" |
    check_chat_shape "")"
  assert_answer "/v1/chat/completions" "$reply"
  stop_server
  log "model smoke passed"
}

main() {
  if [ "${1:-}" = archive ]; then
    [ $# -eq 4 ] || die "usage: $0 archive <tag> <dist-dir> <work-dir>"
    mkdir -p "$4"
    cmd_archive "$2" "$3" "$4"
    return
  fi
  [ $# -eq 3 ] || die "usage: $0 {install|latest|fixture|model|all} <tag> <work-dir>"
  local mode="$1" tag="$2" work="$3"
  mkdir -p "$work"
  work="$(cd "$work" && pwd)"
  tag_version "$tag" >/dev/null
  case "$mode" in
    install) cmd_install "$tag" "$work" ;;
    latest) cmd_latest "$tag" "$work" ;;
    fixture) cmd_fixture "$work" ;;
    model) cmd_model "$work" ;;
    all)
      cmd_install "$tag" "$work"
      cmd_latest "$tag" "$work"
      cmd_fixture "$work"
      ;;
    *) die "unknown mode '$mode'" ;;
  esac
}

main "$@"; exit $?  # one line: bash has parsed the whole file before running
