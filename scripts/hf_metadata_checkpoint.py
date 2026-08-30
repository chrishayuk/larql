#!/usr/bin/env python3
"""Build a *header-only* HF checkpoint from a repo's metadata alone.

`larql inspect-hf` and `larql vindex3 plan` read a checkpoint's
`config.json` plus each safetensors shard's **JSON header** — the 8-byte
length prefix and the header bytes it announces. No tensor data is ever
touched (`larql-models/src/inventory/tensors.rs`). So the admission
instruments do not need the weights: they need the headers, and a
safetensors header is a few hundred kilobytes even when the payload is
hundreds of gigabytes.

This fetches those headers over HTTP range requests and writes a stub
checkpoint directory containing `<u64 header_len><header json>` per shard
and nothing else. `inspect-hf` and `vindex3 plan` then run against it and
produce the **same verdict they would produce against the real weights** —
identity, per-layer attention policy, every unconsumed config key, the full
tensor inventory with exact shapes, dtypes and byte counts.

    python scripts/hf_metadata_checkpoint.py zai-org/GLM-5.3-Flash --out stub
    larql inspect-hf stub --no-tensor-list --output inventory.json
    larql vindex3 plan stub --output plan.json

GLM-5.3-Flash: 18 MB of stub stands in for 328 GB of checkpoint, and the
inventory's `total_bytes` matches the index's declared `total_size` exactly
— which is the self-check that the stub is faithful.

**What a stub cannot do:** anything that reads a tensor value. Encoding,
verification, execution and every parity gate need the real bytes. The stub
answers "would this checkpoint be admitted, and what would it cost", which
is the question worth answering before spending the download.
"""

import argparse
import concurrent.futures
import json
import os
import struct
import subprocess
import sys
import time

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

#: Files copied verbatim when present. The index names the shards; the rest
#: is what the inventory's identity and interface readers consult.
METADATA_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
)

#: Matches `MAX_HEADER_BYTES` in larql-models/src/inventory/tensors.rs — a
#: length beyond this means the range read did not land on a safetensors file.
MAX_HEADER_BYTES = 256 * 1024 * 1024

#: Unsharded repos carry this single file instead of an index.
SINGLE_SHARD = "model.safetensors"

CURL_TIMEOUT_SECS = "120"

#: A short or error-page range response is retried rather than accepted:
#: hosts rate-limit concurrent readers and `curl` still reports success.
RETRY_ATTEMPTS = 4
RETRY_INITIAL_DELAY_SECS = 2


def _curl(url, extra=()):
    proc = subprocess.run(
        ["curl", "-sL", "-m", CURL_TIMEOUT_SECS, *extra, url],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {proc.stderr.decode()[:200]}")
    return proc.stdout


def _fetch_range_exact(url, start, end, what):
    """Bytes `[start, end]`, retried until the length is exactly right.

    A range request can come back short or with an error page in the body
    — the host rate-limits concurrent readers, and `curl` reports success
    because HTTP succeeded. Accepting a short read here writes a truncated
    header and the stub then misreports the checkpoint, which is worse
    than failing: the whole point of this tool is that its output is
    trusted as if the weights were present.
    """
    want = end - start + 1
    delay = RETRY_INITIAL_DELAY_SECS
    for attempt in range(RETRY_ATTEMPTS):
        body = _fetch_range(url, start, end)
        if len(body) == want:
            return body
        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(
        f"{what}: read {len(body)} of {want} bytes after {RETRY_ATTEMPTS} attempts"
    )


def _fetch_range(url, start, end):
    """Bytes `[start, end]` inclusive, the way HTTP Range states them."""
    return _curl(url, ("-H", f"Range: bytes={start}-{end}"))


def fetch_header(base_url, out_dir, shard):
    """Write one shard's stub: the length prefix and the header it announces."""
    dest = os.path.join(out_dir, shard)
    if os.path.exists(dest) and os.path.getsize(dest) > 8:
        return shard, "cached"

    url = f"{base_url}/{shard}"
    try:
        prefix = _fetch_range_exact(url, 0, 7, "length prefix")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len > MAX_HEADER_BYTES:
            return shard, f"ERROR: header claims {header_len} bytes (limit {MAX_HEADER_BYTES})"
        header = _fetch_range_exact(url, 8, 8 + header_len - 1, "header")
    except RuntimeError as err:
        return shard, f"ERROR: {err}"
    # Parse before writing: a stub that is not valid JSON would fail later,
    # inside the tool under test, and look like a tool defect.
    json.loads(header)

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "wb") as fh:
        fh.write(prefix)
        fh.write(header)
    return shard, f"ok ({header_len} B)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="HF repo id, e.g. zai-org/GLM-5.3-Flash")
    ap.add_argument("--out", required=True, help="stub checkpoint directory to write")
    ap.add_argument("--revision", default="main", help="branch, tag or commit sha")
    ap.add_argument("--jobs", type=int, default=8, help="concurrent shard header fetches")
    args = ap.parse_args()

    base = f"{HF_ENDPOINT}/{args.repo}/resolve/{args.revision}"
    os.makedirs(args.out, exist_ok=True)

    for name in METADATA_FILES:
        body = _curl(f"{base}/{name}")
        # A repo without an optional metadata file answers with an HTML error
        # page, not a 404 body curl can distinguish after -L; JSON-parse to tell.
        try:
            json.loads(body)
        except (ValueError, UnicodeDecodeError):
            continue
        with open(os.path.join(args.out, name), "wb") as fh:
            fh.write(body)
        print(f"  {name}: {len(body)} B")

    index_path = os.path.join(args.out, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as fh:
            index = json.load(fh)
        shards = sorted(set(index["weight_map"].values()))
        declared = index.get("metadata", {}).get("total_size")
    else:
        shards = [SINGLE_SHARD]
        declared = None
    print(f"\n{len(shards)} shard(s); fetching headers with {args.jobs} jobs")

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for shard, status in pool.map(lambda s: fetch_header(base, args.out, s), shards):
            if status.startswith("ERROR"):
                failures.append((shard, status))
                print(f"  {shard}: {status}", file=sys.stderr)

    stub_bytes = sum(
        os.path.getsize(os.path.join(args.out, s))
        for s in shards
        if os.path.exists(os.path.join(args.out, s))
    )
    print(f"\nstub written to {args.out}: {stub_bytes / 1e6:.2f} MB of headers")
    if declared:
        print(f"stands in for a checkpoint declaring {declared / 1e9:.1f} GB")
    if failures:
        print(f"{len(failures)} shard(s) failed — the stub is INCOMPLETE", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
