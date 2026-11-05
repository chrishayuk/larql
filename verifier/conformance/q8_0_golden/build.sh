#!/bin/sh
# Builds ggml's own Q8_0 reference encoder (quantize_row_q8_0_ref), unmodified, from the pinned commit. Each source file used is
# checked against ggml-SHA256SUMS, and the build refuses to go on if one differs.
#
# The omitted headers gguf.h, ggml-cpu.h and ggml-cpu/ggml-cpu-impl.h aren't used by the reference encoder, so empty stubs stand in
# on the include path. stubs.c supplies the other ggml_* symbols the file references. They abort if called.
set -eu
COMMIT=facdc44a67da17df4433adcc468b424f48e9ded2
HERE=$(cd "$(dirname "$0")" && pwd)
B=$HERE/build
G=$B/ggml
mkdir -p "$G" "$B/stub/ggml-cpu"
if [ ! -d "$G/.git" ]; then
  git -C "$G" init -q
  git -C "$G" fetch -q --depth 1 https://github.com/ggml-org/ggml.git "$COMMIT"
  git -C "$G" -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
[ "$(git -C "$G" rev-parse HEAD)" = "$COMMIT" ]
(cd "$G" && shasum -a 256 -c "$HERE/ggml-SHA256SUMS" >/dev/null)
: > "$B/stub/gguf.h"
: > "$B/stub/ggml-cpu.h"
: > "$B/stub/ggml-cpu/ggml-cpu-impl.h"
cc -O2 -fno-fast-math -std=c11 -D_DARWIN_C_SOURCE -I"$G/include" -I"$G/src" -I"$B/stub" \
   -dynamiclib -o "$B/libggmlq8ref.dylib" "$G/src/ggml-quants.c" "$HERE/stubs.c"
