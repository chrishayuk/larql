#!/usr/bin/env python3
"""Render a STATE-2 harness run: arm table + canonical text pre-screen."""

import glob
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

MODEL = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--ibm-granite--granite-4.2-3b/snapshots/*/"
    )
)[0]


def main(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    tok = AutoTokenizer.from_pretrained(MODEL)
    canonical = next(r for r in rows if r["arm"] == "canonical")
    text = tok.decode(canonical["greedy_tokens"])
    print(f"spec: {canonical['spec']}")
    print(f"canonical greedy ({canonical['seconds']:.0f}s): {text!r}\n")
    print(f"{'arm':<12} {'kept':>4} {'retired':>7} {'KL mean':>9} {'KL max':>9} "
          f"{'match':>6} {'1st div':>7} {'sec':>5}")
    for r in rows:
        if r["arm"] == "canonical":
            continue
        kl = r["kl_bits"]
        div = r["first_divergence"]
        print(
            f"{r['arm']:<12} {r['retained']:>4} {r['retired_rows']:>7} "
            f"{r['kl_mean_bits']:>9.4f} {max(kl):>9.4f} "
            f"{r['tokens_matched']:>3}/{r['steps']:<2} "
            f"{('-' if div is None else div):>7} {r['seconds']:>5.0f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
