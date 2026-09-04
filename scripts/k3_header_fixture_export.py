#!/usr/bin/env python3
"""Export the REAL safetensors headers of two Kimi-K3 layers, headers only.

`moonshotai/Kimi-K3` is 1.56 TB across 96 shards. This reads the safetensors
header of two of them over HTTP **range requests** — the 8-byte length
prefix, then exactly that many bytes — and downloads no payload at all. The
result is the fixture behind the K3 REPRESENTABLE probe, so that probe reads
K3's actual tensor names, dtypes, shapes and byte counts rather than a stub.

Two shards, chosen as a two-sided witness:

    model-00001  tensor layer 0  KDA  (config layer 1, in `kda_layers`)
    model-00004  tensor layer 3  MLA  (config layer 4, in `full_attn_layers`)

Layer 0 must be discovered as KDA; layer 3 must NOT be. They share
`self_attn.g_proj` and `self_attn.o_proj` at identical shapes, so a
classifier keying on those alone fails the negative side — which is the
point of carrying both.

Usage (from anywhere; the output path is explicit):

    python3 scripts/k3_header_fixture_export.py \
        crates/larql-vindex/src/format/vindex3/plan/tests/fixtures/k3_two_layer_headers.json
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = "moonshotai/Kimi-K3"
BASE = f"https://huggingface.co/{REPO}/resolve/main"

# The two shards, and the layer each one carries.
SHARDS = {
    "model-00001-of-000096.safetensors": 0,
    "model-00004-of-000096.safetensors": 3,
}

# safetensors: 8-byte little-endian header length, then that many bytes of
# JSON, then the payload we never touch.
LENGTH_PREFIX_BYTES = 8


def _range(url: str, start: int, end: int) -> bytes:
    """Inclusive byte range, as safetensors' own layout is described."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def header_of(shard: str) -> dict:
    """The shard's safetensors header, and nothing else from the file."""
    url = f"{BASE}/{shard}"
    length = int.from_bytes(_range(url, 0, LENGTH_PREFIX_BYTES - 1), "little")
    raw = _range(url, LENGTH_PREFIX_BYTES, LENGTH_PREFIX_BYTES + length - 1)
    return json.loads(raw)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(sys.argv[1]).resolve()

    shards = {}
    for shard, layer in SHARDS.items():
        header = header_of(shard)
        names = [n for n in header if n != "__metadata__"]
        # Refuse silently-wrong input: every tensor in a shard we claim
        # carries layer N must actually name layer N.
        stray = [n for n in names if f".layers.{layer}." not in n]
        if stray:
            print(f"{shard}: {len(stray)} tensors outside layer {layer}, "
                  f"e.g. {stray[0]}", file=sys.stderr)
            return 1
        shards[shard] = header
        print(f"{shard}: layer {layer}, {len(names)} tensors", file=sys.stderr)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repo": REPO, "shards": shards}, sort_keys=True))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
