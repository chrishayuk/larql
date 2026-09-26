#!/usr/bin/env python3
"""Q8K-ACT-1 logit comparison (docs/q8k-act-1.md, C0-e and F2/F3).

Reads two teacher-forced `larql vindex3 exec --logit-dump` files of shape
[positions, vocab] f32 (row-major, native endian) and reports, per
position, KL(ref || cand) over the softmax distributions and whether the
two argmaxes agree.

    q8k_act_1_logits.py REF.f32 CAND.f32 --vocab 262208 [--json OUT]
"""

import argparse
import json

import numpy as np


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def compare(ref: np.ndarray, cand: np.ndarray) -> dict:
    if ref.shape != cand.shape:
        raise SystemExit(f"shape mismatch: {ref.shape} vs {cand.shape}")
    lp, lq = log_softmax(ref), log_softmax(cand)
    kl = (np.exp(lp) * (lp - lq)).sum(axis=-1)
    agree = ref.argmax(axis=-1) == cand.argmax(axis=-1)
    return {
        "positions": int(ref.shape[0]),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "kl_per_position": [float(v) for v in kl],
        "top1_agreement": float(agree.mean()),
        "top1_disagreements": int((~agree).sum()),
    }


def load(path: str, vocab: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % vocab:
        raise SystemExit(f"{path}: {raw.size} values is not a whole number of {vocab}-wide rows")
    return raw.reshape(-1, vocab)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("cand")
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--json")
    a = p.parse_args()
    r = compare(load(a.ref, a.vocab), load(a.cand, a.vocab))
    print(
        f"positions {r['positions']}  KL mean {r['kl_mean']:.4e}  max {r['kl_max']:.4e}  "
        f"top-1 agreement {r['top1_agreement']:.4f} ({r['top1_disagreements']} disagree)"
    )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
