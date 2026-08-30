#!/usr/bin/env python3
"""Rung 4 of the mamba2 witness ladder: LARQL vs the banked HF oracle.

Teacher-forces prompt + the oracle's own greedy continuation through
`larql vindex3 exec --logit-dump`, so both arms see identical context at
every position, then scores the boundaries that matter for an SSM:

  * every prefill position's full logit vector,
  * the first decode token and all 32 recurrent steps,
  * greedy argmax agreement along the whole trajectory,
  * all three prompt lengths (one crossing the SSD chunk boundary).

The acceptance bound is the oracle's OWN step-vs-scan disagreement
(~2.1e-4 max-abs): LARQL runs the recurrence for prefill where HF runs
the chunked scan, so agreement beyond that bound is not available even
to the reference against itself. Argmax must agree everywhere.

Usage:
  python3 scripts/mamba2_parity_check.py <container> <oracle-dir> [--larql path]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# The oracle's own internal step-vs-scan bound was ~2.1e-4; LARQL adds
# its own f32 reassociation on the same order. One order of headroom, and
# argmax must still agree at every scored position.
MAX_ABS = 5e-3
ORACLE_SELF_BOUND = 2.1e-4


def run_logit_dump(larql: str, container: Path, ids: list[int], out: Path) -> np.ndarray:
    subprocess.run(
        [
            larql, "vindex3", "exec", str(container),
            "--tokens", ",".join(map(str, ids)),
            "--backend", "reference",
            "--logit-dump", str(out),
        ],
        check=True,
        capture_output=True,
    )
    flat = np.fromfile(out, dtype=np.float32)
    return flat.reshape(len(ids), -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("container")
    ap.add_argument("oracle_dir")
    ap.add_argument("--larql", default="./target/debug/larql")
    args = ap.parse_args()

    oracle_dir = Path(args.oracle_dir).expanduser()
    manifest = json.loads((oracle_dir / "manifest.json").read_text())
    failed = False

    for name, meta in manifest["prompts"].items():
        z = np.load(oracle_dir / f"{name}.npz")
        prompt_ids = list(map(int, z["input_ids"]))
        gen_ids = list(map(int, z["generated_ids"]))
        n = len(prompt_ids)
        forced = prompt_ids + gen_ids

        with tempfile.NamedTemporaryFile(suffix=".f32") as tmp:
            ours = run_logit_dump(args.larql, Path(args.container), forced, Path(tmp.name))

        # Oracle logits at every forced position: prefill rows, then the
        # 32 stepwise decode rows (decode_logits[k] scores position n+k).
        oracle = np.concatenate([z["prefill_logits"], z["decode_logits"]])
        assert oracle.shape == ours.shape, (oracle.shape, ours.shape)

        diff = np.abs(ours - oracle).max(axis=1)
        argmax_ok = (ours.argmax(axis=1) == oracle.argmax(axis=1))
        # The trajectory gate: greedy over OUR logits reproduces the
        # oracle's continuation token for token.
        traj = [int(ours[n - 1 + k].argmax()) for k in range(len(gen_ids))]
        traj_ok = traj == gen_ids

        worst = float(diff.max())
        segs = {
            "prefill": float(diff[:n].max()),
            "first-token": float(diff[n - 1]),
            "decode-steps": float(diff[n:].max()) if len(diff) > n else 0.0,
        }
        ok = worst <= MAX_ABS and argmax_ok.all() and traj_ok
        failed |= not ok
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {name:6s} n={n:3d}+{len(gen_ids)}  "
            f"max|Δ| {worst:.3e}  (prefill {segs['prefill']:.3e} · "
            f"first {segs['first-token']:.3e} · steps {segs['decode-steps']:.3e})  "
            f"argmax {int(argmax_ok.sum())}/{len(argmax_ok)}  "
            f"trajectory {'exact' if traj_ok else 'DIVERGED at ' + str(next(i for i, (a, b) in enumerate(zip(traj, gen_ids)) if a != b))}"
        )

    print(
        f"\nbound: max|Δ| ≤ {MAX_ABS:.0e} "
        f"(oracle's own step-vs-scan bound {ORACLE_SELF_BOUND:.1e}); argmax + trajectory exact"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
