#!/usr/bin/env python3
"""The operand file of CONSUMPTION-BF16-1's qualification edit (docs/consumption-bf16-1.md §3): a non-uniform
rank-one `ADD LOW RANK` over every column of one `[rows, cols]` matrix.

    a[r] = s_a(r) × 2^-1        b[j] = s_b(j) × 2^-2        s(i) = +1 if SHA-256("consumption-bf16-1/<a|b>/<i>")[0] is odd, else −1

Every element of the update is ±2^-3. It is an instrument stimulus, chosen to move the layer's FFN write at every
position; it carries no behavioural intent.

    python3 -I scripts/consumption_edit_operands.py --rows 2560 --cols 10240 --out operands.json
"""

import argparse
import hashlib
import json
import struct

LABEL = "consumption-bf16-1"
A_MAGNITUDE = 2.0 ** -1
B_MAGNITUDE = 2.0 ** -2


def sign(which, i):
    return 1.0 if hashlib.sha256(f"{LABEL}/{which}/{i}".encode()).digest()[0] & 1 else -1.0


def f64_hex(x):
    return struct.pack(">d", x).hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    doc = {
        "rank": 1,
        "a": [f64_hex(sign("a", r) * A_MAGNITUDE) for r in range(args.rows)],
        "b": [f64_hex(sign("b", j) * B_MAGNITUDE) for j in range(args.cols)],
        "origin": "CONSUMPTION-BF16-1 qualification stimulus: hashed-sign rank-one, |a|=2^-1, |b|=2^-2; no behavioural claim",
    }
    with open(args.out, "w") as f:
        json.dump(doc, f)


if __name__ == "__main__":
    main()
