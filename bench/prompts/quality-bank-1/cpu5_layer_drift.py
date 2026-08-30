#!/usr/bin/env python3
"""CPU-5 layer localisation: where does the representation's error enter?

    python3 cpu5_layer_drift.py <ref-dump-dir> <cand-dump-dir> [--hidden N]

Both directories are `vindex3 exec --dump-layers` outputs over the SAME
token ids. Per layer it reports the last position's drift against the
reference, so a step in the curve localises the damage to a depth rather
than attributing it to the model as a whole.

Reported per layer:
    rel_rms   ||cand - ref|| / ||ref||   scale-free, comparable across depth
    cosine    direction agreement
    max|d|    the worst single channel
    growth    rel_rms(L) / rel_rms(L-1), so a STEP is visible as a spike

A representation whose error grows smoothly with depth is accumulating
rounding. One that steps at a particular depth has a site, and a site is
something an exception set can name.
"""
import numpy as np, sys, os, glob, json


def planes(d, hidden):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "layer_*.f32"))):
        a = np.fromfile(f, dtype=np.float32)
        h = hidden if (hidden and a.size % hidden == 0) else a.size
        out[int(os.path.basename(f)[6:9])] = a.reshape(-1, h)
    final = os.path.join(d, "final_norm.f32")
    if os.path.exists(final):
        a = np.fromfile(final, dtype=np.float32)
        h = hidden if (hidden and a.size % hidden == 0) else a.size
        out[-1] = a.reshape(-1, h)
    return out


def main():
    ref_dir, cand_dir = sys.argv[1], sys.argv[2]
    hidden = int(sys.argv[sys.argv.index("--hidden") + 1]) if "--hidden" in sys.argv else 0
    R, C = planes(ref_dir, hidden), planes(cand_dir, hidden)
    common = sorted(k for k in R if k in C and k >= 0)
    if not common:
        raise SystemExit("no layers in common — are these dumps of the same run?")

    print(f"{'layer':>6}{'rel_rms':>12}{'cosine':>12}{'max|d|':>11}{'growth':>9}")
    prev = None
    rows = []
    for k in common:
        r, c = R[k][-1].astype(np.float64), C[k][-1].astype(np.float64)
        if r.shape != c.shape:
            raise SystemExit(f"layer {k}: shape {r.shape} vs {c.shape}")
        d = c - r
        rel = float(np.sqrt((d ** 2).mean()) / np.sqrt((r ** 2).mean()))
        nr, nc = np.linalg.norm(r), np.linalg.norm(c)
        cos = float(r @ c / (nr * nc)) if nr and nc else float("nan")
        growth = rel / prev if prev else float("nan")
        prev = rel
        rows.append({"layer": k, "rel_rms": rel, "cosine": cos,
                     "max_abs": float(np.abs(d).max()), "growth": growth})
        print(f"{k:>6}{rel:>12.4e}{cos:>12.7f}{np.abs(d).max():>11.4f}{growth:>9.3f}")

    g = [r["growth"] for r in rows if np.isfinite(r["growth"])]
    if g:
        i = int(np.argmax(g)) + 1
        print(f"\nsteepest single-layer growth {max(g):.3f}x entering layer {rows[i]['layer']}")
        print("a STEP names a site an exception set can protect; a flat curve means the "
              "error is accumulating everywhere and no exception set will help.")
    if "--json" in sys.argv:
        json.dump(rows, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
