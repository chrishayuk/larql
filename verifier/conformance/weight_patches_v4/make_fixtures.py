"""Compile the weight-patches/v4 conformance containers with the LARQL engine (the system under test).

This is ENGINE-SIDE tooling: it drives `larql lql` to compile containers from the committed structured-patch base
(`../structured_patch/base`). It writes the operand files, the LQL each case ran, the compiled containers and a
provenance record. It computes no expected values: the verifier recomputes everything from the base and the ops the
compiled ledger carries, and must never read anything this script wrote as an oracle except the containers it judges.

    python3 -I make_fixtures.py --larql <path to a larql binary built from the engine revision under test>
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, "..", "structured_patch", "base"))
DOWN = "1.mlp.down_proj.weight"  # [64, 256]
O_PROJ = "1.self_attn.o_proj.weight"  # [64, 64]
OTHER = "0.mlp.down_proj.weight"
ROWS = 64
F64 = "f64-seq-rne1/v1"
F32 = "f32-seq-rne1/v1"


def f64_hex(x):
    return struct.pack(">d", x).hex()


def f32_hex(x):
    return struct.pack(">f", x).hex()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_operands(case_dir, name, rank, a, b, hexer, origin):
    os.makedirs(os.path.join(case_dir, "operands"), exist_ok=True)
    path = os.path.join(case_dir, "operands", name)
    with open(path, "w") as f:
        json.dump({"rank": rank, "a": [hexer(x) for x in a], "b": [hexer(x) for x in b], "origin": origin}, f, indent=1)
    return path


def write_direction(case_dir, name, u):
    os.makedirs(os.path.join(case_dir, "operands"), exist_ok=True)
    path = os.path.join(case_dir, "operands", name)
    with open(path, "w") as f:
        json.dump({"bits": [f64_hex(x) for x in u]}, f, indent=1)
    return path


def cols(xs):
    return ", ".join(str(x) for x in xs)


def add(operands, tensor, columns, profile):
    return f'ADD LOW RANK "{operands}" TO "{tensor}" COLUMNS ({cols(columns)}) NUMERICS "{profile}";'


def cases(case_root):
    """Each case: (name, list of statements builders). Operands are exact binary fractions."""
    out = {}

    # 1. Rank 2 over scattered columns, with zero rows, a -0 factor and a zero entry of b.
    d = os.path.join(case_root, "low_rank_k2")
    J = [3, 4, 5, 40, 200, 255]
    a = []
    for r in range(ROWS):
        for k in range(2):
            v = (((r * 7 + k * 3) % 11) - 5) / 64.0
            if r == 0 and k == 0:
                v = -0.0
            if r == 9:
                v = 0.0  # a whole row of a is zero: row 9 is untouched inside J
            a.append(v)
    b = []
    for k in range(2):
        for p in range(len(J)):
            v = (((k * 5 + p * 3) % 7) - 3) / 8.0
            if k == 1 and p == 2:
                v = 0.0
            b.append(v)
    ops = write_operands(d, "a_b.json", 2, a, b, f64_hex, "conformance low_rank_k2")
    out["low_rank_k2"] = (d, [add(ops, DOWN, J, F64)])

    # 2. Mixed: a low-rank op and a removal on one tensor (disjoint columns), and a delta on another tensor.
    d = os.path.join(case_root, "mixed")
    J = [8, 9]
    ops = write_operands(d, "a_b.json", 1, [1.0 / 32.0] * ROWS, [1.0, -0.5], f64_hex, "conformance mixed")
    u = [1.0 if r == 0 else 0.0 for r in range(ROWS)]
    direction = write_direction(d, "u.json", u)
    out["mixed"] = (
        d,
        [
            add(ops, DOWN, J, F64),
            f'REMOVE DIRECTION "{direction}" FROM "{DOWN}" COLUMNS (33, 34) SCALE 0.5 NUMERICS "{F64}";',
            f'UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = "{OTHER}" AND ROW = 5 AND COL IN (12, 13);',
        ],
    )

    # 3. A sham: zero-valued (a is -0 everywhere), compiled to the base's bytes.
    d = os.path.join(case_root, "sham")
    ops = write_operands(d, "a_b.json", 1, [-0.0] * ROWS, [1.0, 1.0, 1.0], f64_hex, "conformance sham")
    out["sham"] = (d, [add(ops, DOWN, [10, 11, 12], F64)])

    # 4. The f32 working profile, on another tensor.
    d = os.path.join(case_root, "f32_profile")
    J = [0, 1, 2, 63]
    a = [((r % 5) - 2) / 16.0 for r in range(ROWS)]
    ops = write_operands(d, "a_b.json", 1, a, [0.75, -1.0, 2.0, 0.125], f32_hex, "conformance f32_profile")
    out["f32_profile"] = (d, [add(ops, O_PROJ, J, F32)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--larql", required=True)
    args = ap.parse_args()
    larql = os.path.abspath(args.larql)
    provenance = {
        "generated_by": "verifier/conformance/weight_patches_v4/make_fixtures.py (engine-side; not an oracle)",
        "larql_binary_sha256": sha256_file(larql),
        "base": "../structured_patch/base",
        "base_index_sha256": sha256_file(os.path.join(BASE, "index.json")),
        "cases": {},
    }
    for name, (case_dir, statements) in cases(HERE).items():
        compiled = os.path.join(case_dir, "compiled")
        if os.path.exists(compiled):
            shutil.rmtree(compiled)
        # Compile into a temporary directory and move it in: the engine records its output path in candidate.json's
        # locator_hint, which must not name this machine's checkout.
        scratch = tempfile.mkdtemp()
        staged = os.path.join(scratch, "compiled.vindex3")
        script = [f'USE "{BASE}";'] + statements + [f'COMPILE CURRENT INTO VINDEX "{staged}";']
        text = " ".join(script)
        # Record the statements with paths relative to this directory, so the record does not name a machine.
        with open(os.path.join(case_dir, "statements.lql"), "w") as f:
            f.write(
                "\n".join(
                    s.replace(HERE + os.sep, "").replace(BASE, "../structured_patch/base").replace(staged, "compiled")
                    for s in script
                )
                + "\n"
            )
        run = subprocess.run([larql, "lql", text], capture_output=True, text=True)
        if run.returncode != 0 or not os.path.exists(os.path.join(staged, "weight_patches.json")):
            sys.exit(f"{name}: larql failed\n{run.stdout}\n{run.stderr}")
        shutil.move(staged, compiled)
        shutil.rmtree(scratch)
        provenance["cases"][name] = {
            "ledger_sha256": sha256_file(os.path.join(compiled, "weight_patches.json")),
            "index_sha256": sha256_file(os.path.join(compiled, "index.json")),
        }
        print(f"{name}: compiled")
    with open(os.path.join(HERE, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
