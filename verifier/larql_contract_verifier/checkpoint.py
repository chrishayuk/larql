"""The source checkpoint: its pinned identity, and its weights without torch.

A contract compares a container with a source the container did not choose, so the source must be identified by something
other than the container. It is identified by the size and SHA-256 of every file of one upstream revision.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct

import numpy as np

from refcheck.container import decode_bf16

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PINS: dict[str, tuple[int, str]] = {
    "model.safetensors": (1503300328, "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
    "config.json": (726, "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"),
    "generation_config.json": (239, "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
    "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (9732, "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "LICENSE": (11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
    "README.md": (13965, "1ab64a26fcb3b461423b89a433a8c858f1bf8d4086f979cbb3ff878d47cf20e9"),
}


class SourceError(Exception):
    """The source is not one this verifier can vouch for, or cannot be read."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source(root: str, pins: dict[str, tuple[int, str]] = PINS) -> int:
    """Check every pinned file by size and SHA-256, raising on any difference. Returns how many files matched."""
    for name, (size, digest) in pins.items():
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            raise SourceError(f"the source lacks {name}")
        if os.path.getsize(path) != size:
            raise SourceError(f"{name}: {os.path.getsize(path)} bytes, pinned {size}")
        got = sha256_file(path)
        if got != digest:
            raise SourceError(f"{name}: sha256 {got}, pinned {digest}")
    return len(pins)


def load_weights(root: str) -> dict[str, np.ndarray]:
    """Every tensor of `model.safetensors` as float32. BF16 widens exactly. Any other dtype is refused."""
    path = os.path.join(root, "model.safetensors")
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        out = {}
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if entry["dtype"] != "BF16":
                raise SourceError(f"{key}: dtype {entry['dtype']}, this verifier reads BF16 only")
            start, end = entry["data_offsets"]
            f.seek(base + start)
            data = f.read(end - start)
            if len(data) != end - start:
                raise SourceError(f"{key}: short read")
            out[key] = decode_bf16(data).reshape(entry["shape"])
    return out


def container_sha256(root: str) -> str:
    """A container is a directory, so this digests the lines `<relative path>\\0<size>\\0<file sha256>\\n` of every regular
    file, in sorted order. It covers every file, so it changes with anything recorded in the container, including the path
    `candidate.json` records for the encoder's input."""
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            lines.append(f"{rel}\0{os.path.getsize(p)}\0{sha256_file(p)}\n")
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()
