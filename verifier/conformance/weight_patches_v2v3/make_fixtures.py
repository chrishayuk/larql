"""Compile the weight-patches/v2 and v3 conformance containers with the LARQL engine (the system under test).

This is ENGINE-SIDE tooling: it drives `larql lql` to compile containers from the committed structured-patch base
(`../structured_patch/base`). It writes the operand files, the LQL each case ran, the compiled containers, the `.vlp`
each case's ops save as, and a provenance record. It computes no expected values: the verifier recomputes everything
from the base, the deltas the compiled ledger carries and the `remove_direction` ops of the saved patch, and must never
read anything this script wrote as an oracle except the containers and patches it judges.

    python3 -I make_fixtures.py --larql <path to a larql binary built from the engine revision under test>

The four cases are the three shapes `WEIGHT_PATCHES_V2` and `WEIGHT_PATCHES_V3` judge
(docs/verification/weight-patches-v2-v3.md):

    deltas       v2   literal deltas over two tensors
    removal      v3   one remove_direction on one tensor
    removal_two  v3   two remove_direction ops on one tensor, on disjoint columns
    mixed        v3   literal deltas on one tensor and a remove_direction on another
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
DOWN0 = "0.mlp.down_proj.weight"  # [64, 256]
DOWN1 = "1.mlp.down_proj.weight"  # [64, 256]
O_PROJ = "1.self_attn.o_proj.weight"  # [64, 64]
ROWS = 64
RNE1 = "f64-seq-rne1/v1"
RNE2 = "f64-seq-rne2/v1"
LEDGER_FILE = "weight_patches.json"
PATCH_FILE = "patch.vlp"


def f64_hex(x):
    return struct.pack(">d", x).hex()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_direction(case_dir, name, u, origin=None):
    os.makedirs(os.path.join(case_dir, "operands"), exist_ok=True)
    path = os.path.join(case_dir, "operands", name)
    doc = {"bits": [f64_hex(x) for x in u]}
    if origin is not None:
        doc["origin"] = origin
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return path


def cols(xs):
    return ", ".join(str(x) for x in xs)


def delta(value, tensor, row, columns):
    """`value` is the literal as LQL reads it: the lexer takes it as a double and narrows it to the f32 delta."""
    where = f"COL = {columns[0]}" if len(columns) == 1 else f"COL IN ({cols(columns)})"
    return f'UPDATE WEIGHTS SET DELTA = {value} WHERE TENSOR = "{tensor}" AND ROW = {row} AND {where};'


def remove(direction, tensor, columns, scale, profile):
    return (f'REMOVE DIRECTION "{direction}" FROM "{tensor}" COLUMNS ({cols(columns)}) SCALE {scale} '
            f'NUMERICS "{profile}";')


def cases(case_root):
    """Each case: (directory, statements). Operands are exact binary fractions."""
    out = {}

    # 1. Deltas only, over two tensors. `0.1` and `-0.3` are not binary32 values: the ledger prints the f32 each
    # narrows to in its shortest form, which a reader must take as a binary32. `-0.0625` is exact. `0.0000000001` is
    # far below half a BF16 step of the stored element, so that edit is recorded and changes no bits. (The LQL lexer
    # takes no exponent, so the small delta is written out.)
    d = os.path.join(case_root, "deltas")
    os.makedirs(d, exist_ok=True)
    out["deltas"] = (
        d,
        [
            delta("0.1", DOWN0, 5, [12, 13, 200]),
            delta("-0.0625", DOWN0, 63, [255]),
            delta("-0.3", O_PROJ, 0, [0, 63]),
            delta("0.0000000001", O_PROJ, 7, [7]),
        ],
    )

    # 2. One removal on one tensor, with an origin note (the op digest covers its text).
    d = os.path.join(case_root, "removal")
    u = [(((r * 5) % 7) - 3) / 8.0 for r in range(ROWS)]
    direction = write_direction(d, "u.json", u, origin="conformance removal")
    out["removal"] = (d, [remove(direction, DOWN1, [33, 34, 250], "0.5", RNE1)])

    # 3. Two removals on one tensor, on disjoint columns, in the two f64 profiles and with a negative scale: the
    # tensor is recorded by its result bits once, and both ops are needed to recompute it.
    d = os.path.join(case_root, "removal_two")
    u1 = [(((r * 3) % 5) - 2) / 4.0 for r in range(ROWS)]
    u2 = [0.5 if r % 9 == 0 else (-0.25 if r % 9 == 4 else 0.0) for r in range(ROWS)]
    first = write_direction(d, "u1.json", u1)
    second = write_direction(d, "u2.json", u2)
    out["removal_two"] = (
        d,
        [
            remove(first, DOWN0, [10, 11], "1", RNE1),
            remove(second, DOWN0, [200], "-0.25", RNE2),
        ],
    )

    # 4. Deltas on one tensor and a removal on another: a v3 ledger with delta targets and a result-bits target.
    d = os.path.join(case_root, "mixed")
    u = [1.0 if r == 2 else (0.5 if r in (0, 40) else 0.0) for r in range(ROWS)]
    direction = write_direction(d, "u.json", u)
    out["mixed"] = (
        d,
        [
            delta("0.1", DOWN0, 5, [12]),
            delta("-0.75", DOWN0, 6, [0, 255]),
            remove(direction, O_PROJ, [0, 63], "0.5", RNE1),
        ],
    )
    return out


def run_lql(larql, text, what):
    run = subprocess.run([larql, "lql", text], capture_output=True, text=True)
    if run.returncode != 0:
        sys.exit(f"{what}: larql failed\n{run.stdout}\n{run.stderr}")


def save_patch(larql, case_dir, statements):
    """Record the case's ops as a `.vlp`, in a second session: the same statements between BEGIN and SAVE PATCH. The
    op digests carry no time or path, so they are the ones the compile listed in its ledger's `origins`."""
    scratch = tempfile.mkdtemp()
    staged = os.path.join(scratch, PATCH_FILE)
    text = " ".join([f'USE "{BASE}";', f'BEGIN PATCH "{staged}";'] + statements + ["SAVE PATCH;"])
    run_lql(larql, text, f"{os.path.basename(case_dir)} (saving the patch)")
    if not os.path.exists(staged):
        sys.exit(f"{os.path.basename(case_dir)}: larql saved no patch")
    shutil.move(staged, os.path.join(case_dir, PATCH_FILE))
    shutil.rmtree(scratch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--larql", required=True)
    ap.add_argument("--engine-revision", default=None,
                    help="the engine commit the binary was built from, recorded as given (not checked)")
    args = ap.parse_args()
    larql = os.path.abspath(args.larql)
    provenance = {
        "generated_by": "verifier/conformance/weight_patches_v2v3/make_fixtures.py (engine-side; not an oracle)",
        "larql_binary_sha256": sha256_file(larql),
        "engine_revision_as_given": args.engine_revision,
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
        # Record the statements with paths relative to this directory, so the record does not name a machine.
        with open(os.path.join(case_dir, "statements.lql"), "w") as f:
            f.write(
                "\n".join(
                    s.replace(HERE + os.sep, "").replace(BASE, "../structured_patch/base").replace(staged, "compiled")
                    for s in script
                )
                + "\n"
            )
        run_lql(larql, " ".join(script), name)
        if not os.path.exists(os.path.join(staged, LEDGER_FILE)):
            sys.exit(f"{name}: larql wrote no ledger")
        shutil.move(staged, compiled)
        shutil.rmtree(scratch)
        save_patch(larql, case_dir, statements)
        with open(os.path.join(compiled, LEDGER_FILE)) as f:
            schema = json.load(f)["schema"]
        provenance["cases"][name] = {
            "ledger_schema": schema,
            "ledger_sha256": sha256_file(os.path.join(compiled, LEDGER_FILE)),
            "index_sha256": sha256_file(os.path.join(compiled, "index.json")),
            "patch_sha256": sha256_file(os.path.join(case_dir, PATCH_FILE)),
        }
        print(f"{name}: compiled ({schema})")
    with open(os.path.join(HERE, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
