#!/usr/bin/env python3
"""Export the REAL safetensors headers of the hyper-connection operands.

Three checkpoints declare a hyper-connection residual topology, and they
spell it three different ways. This reads the safetensors header of the
shards that carry those operands over HTTP **range requests** — the 8-byte
length prefix, then exactly that many bytes — and downloads no payload:

    deepseek-ai/DeepSeek-V4-Flash    159.61 GB
    zai-org/GLM-5.3-Flash            328.33 GB
    tencent/Hy4-preview            1,559.98 GB

The result is the fixture behind wave 18's classifier baseline, so that
witness reads each checkpoint's actual tensor names, dtypes and shapes
rather than names retyped from a forecast.

Layer 0 is carried in full for the two families whose layer fits in one
shard, which is what supplies the CONTROL operands: a witness reporting
"every hyper-connection spelling is unclassified" proves nothing unless
ordinary operands from the same checkpoint, through the same call, are
seen to classify.

Hy4 is the exception and the reason its entry looks thin: it scatters ONE
layer's four hyper-connection operands across four different shards
(00084, 00089, 00093, 00094), so its layer cannot be carried whole
without fetching most of a 131-shard checkpoint. Its controls come from
the other two.

Usage (from anywhere; the output path is explicit):

    python3 scripts/hc_header_fixture_export.py \
        crates/larql-vindex/src/format/vindex3/opplan/tests/fixtures/hc_operand_headers.json
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

LENGTH_PREFIX_BYTES = 8

# How many real members of one indexed spelling to carry. See the elision
# note in `main`.
REPEAT_CAP = 2
ELIDE_INDEX = re.compile(r"\.\d+\.")

# Per family: the shards to read, and which names to keep from them.
#   layer_prefix  keep every tensor under this prefix (controls + HC)
#   extra         keep every tensor whose name starts with one of these
FAMILIES = {
    "deepseek-ai/DeepSeek-V4-Flash": {
        "shards": [
            "model-00002-of-00046.safetensors",
            "model-00045-of-00046.safetensors",
            "model-00046-of-00046.safetensors",
        ],
        "layer_prefix": "layers.0.",
        "extra": ["hc_head_", "mtp."],
    },
    "zai-org/GLM-5.3-Flash": {
        "shards": ["model-00002-of-00062.safetensors"],
        "layer_prefix": "model.language_model.layers.0.",
        "extra": [],
    },
    "tencent/Hy4-preview": {
        "shards": [
            "model-00084-of-00131.safetensors",
            "model-00089-of-00131.safetensors",
            "model-00093-of-00131.safetensors",
            "model-00094-of-00131.safetensors",
        ],
        "layer_prefix": "model.layers.0.hc_",
        "extra": [],
    },
}


def fetch(url: str, start: int, length: int) -> bytes:
    end = start + length - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def shard_header(repo: str, shard: str) -> dict:
    url = f"https://huggingface.co/{repo}/resolve/main/{shard}"
    size = int.from_bytes(fetch(url, 0, LENGTH_PREFIX_BYTES), "little")
    return json.loads(fetch(url, LENGTH_PREFIX_BYTES, size))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])

    fixture: dict = {}
    for repo, spec in FAMILIES.items():
        kept: dict = {}
        seen: dict = {}
        for shard in spec["shards"]:
            header = shard_header(repo, shard)
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                if not (
                    name.startswith(spec["layer_prefix"])
                    or any(name.startswith(p) for p in spec["extra"])
                ):
                    continue
                # An indexed family (896 experts, 62 MTP layers) exercises
                # ONE classifier path however many times it repeats, so
                # carrying every member would inflate the fixture without
                # testing anything further. Keep REPEAT_CAP real members of
                # each elided spelling — more than one, so an off-by-one in
                # index handling still has something to fail against.
                elided = ELIDE_INDEX.sub(".N.", name)
                seen[elided] = seen.get(elided, 0) + 1
                if seen[elided] > REPEAT_CAP:
                    continue
                offsets = entry["data_offsets"]
                kept[name] = {
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                    "bytes": offsets[1] - offsets[0],
                    "shard": shard,
                }
            print(f"{repo} {shard}: {len(kept)} kept so far", file=sys.stderr)
        fixture[repo] = dict(sorted(kept.items()))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    total = sum(len(v) for v in fixture.values())
    print(f"wrote {out} — {total} tensors across {len(fixture)} checkpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
