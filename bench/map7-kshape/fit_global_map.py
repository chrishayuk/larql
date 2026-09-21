#!/usr/bin/env python3
"""Fit a global rank-r boundary transition map from map7-kshape capture shards.

Implements the G-r arm family of docs/preregistration/map7-kshape-prereg.md
(§4.2/§4.3): delta- or absolute-mode ridge regression from boundary i_src to
i_tgt, rank-truncated by SVD. Fitting uses the named split/stratum shards;
the injection-mode rule (final position, single-row splice) applies to
EVALUATION (the Rust evaluator), not to fitting — fitting may use all
captured positions (--positions all) or last positions only.

Self-contained: numpy + stdlib (minimal safetensors reader/writer inline).

Usage:
  python3 bench/map7-kshape/fit_global_map.py \
      --capture-dir ~/chris-models/map7-kshape-capture-v1 \
      --pair 4,20 --mode delta --rank 30 --ridge 1e-2 \
      --splits train --strata general --positions all \
      --out ~/chris-models/map7-kshape-capture-v1/maps/t2-global-r30.safetensors

Ridge definition (documented, applied identically to every arm that uses
this fitter): W solves (C + ridge * mean(diag(C)) * I) W = X'Y/n with
C = X'X/n over centred rows — `ridge` is relative to the mean input
variance, so one grid spans layers/pairs of different scale.
"""

from __future__ import annotations

import argparse
import glob
import json
import struct
import sys
from pathlib import Path

import numpy as np

DTYPES = {"F32": (np.float32, 4), "U32": (np.uint32, 4)}
N_BOUNDARIES = 35


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        data = f.read()
    out = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype, _ = DTYPES[spec["dtype"]]
        b, e = spec["data_offsets"]
        out[name] = np.frombuffer(data[b:e], dtype=dtype).reshape(spec["shape"])
    return out


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        blob = arr.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    hbytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        for blob in blobs:
            f.write(blob)


def refuse(msg: str) -> None:
    print(f"REFUSE: {msg}", file=sys.stderr)
    sys.exit(1)


def load_rows(capture_dir: Path, splits, strata, positions, src, tgt):
    xs, ys, used_shards = [], [], []
    for split in splits:
        for stratum in strata:
            pattern = str(capture_dir / f"{split}-{stratum}-c*.safetensors")
            shards = sorted(
                p for p in glob.glob(pattern)
                if not p.endswith(".audit.safetensors")
            )
            if not shards:
                refuse(f"no shards for {split}-{stratum} in {capture_dir}")
            for shard in shards:
                meta_path = Path(shard).with_suffix("").with_suffix("")
                meta_path = Path(str(meta_path) + ".meta.jsonl")
                if not meta_path.exists():
                    refuse(f"incomplete shard (no meta): {shard}")
                tensors = read_safetensors(Path(shard))
                b = tensors["boundaries"]
                if b.shape[1] != N_BOUNDARIES:
                    refuse(f"{shard}: expected {N_BOUNDARIES} boundaries, got {b.shape[1]}")
                metas = [json.loads(l) for l in open(meta_path)]
                if len(metas) != b.shape[0]:
                    refuse(f"{shard}: meta rows {len(metas)} != tensor rows {b.shape[0]}")
                if positions == "last":
                    keep = [i for i, m in enumerate(metas)
                            if m["pos"] == m["seq_len"] - 1]
                else:
                    keep = list(range(len(metas)))
                xs.append(b[keep][:, src, :].astype(np.float64))
                ys.append(b[keep][:, tgt, :].astype(np.float64))
                used_shards.append(Path(shard).name)
    return np.concatenate(xs), np.concatenate(ys), used_shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--pair", required=True, help="src,tgt boundary indices, e.g. 4,20")
    ap.add_argument("--mode", choices=["delta", "absolute"], required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--ridge", type=float, required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--strata", nargs="+", default=["general"])
    ap.add_argument("--positions", choices=["all", "last"], default="all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir).expanduser()
    cm_path = capture_dir / "capture-manifest.json"
    if not cm_path.exists():
        refuse(f"no capture-manifest.json in {capture_dir}")
    identity = json.loads(cm_path.read_text())["identity"]

    src, tgt = (int(v) for v in args.pair.split(","))
    if not (0 <= src < tgt < N_BOUNDARIES):
        refuse(f"bad pair {src},{tgt}")

    x, y, used_shards = load_rows(
        capture_dir, args.splits, args.strata, args.positions, src, tgt
    )
    n, d = x.shape
    print(f"fit rows: {n} (d={d}) from {len(used_shards)} shards")
    if n < 10 * args.rank:
        refuse(f"only {n} rows for rank {args.rank}")

    target = y - x if args.mode == "delta" else y
    x_mean = x.mean(axis=0)
    y_mean = target.mean(axis=0)
    xc = x - x_mean
    yc = target - y_mean

    c = xc.T @ xc / n
    g = xc.T @ yc / n
    lam = args.ridge * float(np.mean(np.diag(c)))
    w = np.linalg.solve(c + lam * np.eye(d), g)

    u, s, vt = np.linalg.svd(w, full_matrices=False)
    r = args.rank
    u_r = u[:, :r]
    vt_r = (s[:r, None] * vt[:r, :])
    energy = float((s[:r] ** 2).sum() / (s**2).sum())

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(out, {
        "u": u_r, "vt": vt_r, "x_mean": x_mean, "y_mean": y_mean,
    })
    manifest = {
        "arm": f"global-rank{r}-{args.mode}-ridge{args.ridge:g}",
        "pair": [src, tgt],
        "mode": args.mode,
        "rank": r,
        "ridge_relative": args.ridge,
        "ridge_absolute": lam,
        "positions": args.positions,
        "splits": args.splits,
        "strata": args.strata,
        "n_rows": n,
        "svd_energy_captured": energy,
        "shards": used_shards,
        "identity": identity,
        "predict": "y_hat = y_mean + (x - x_mean) @ u @ vt"
                   + (" + x" if args.mode == "delta" else ""),
        "ridge_definition": "(C + ridge*mean(diag(C))*I) W = X'Y/n, centred",
    }
    Path(str(out) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"OK: wrote {out} (rank {r}, energy {energy:.4f}, n {n})")


if __name__ == "__main__":
    main()
