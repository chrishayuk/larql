#!/usr/bin/env python3
"""STRUCTURED-REAL-1 harness. Runs PROTOCOL.md as written and records what happened; it judges nothing it was not told to.

    python3 bench/structured-real-1/run.py          (from the worktree root, after `cargo build --release -p larql-cli`)
"""
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "verifier"))
from larql_contract_verifier.contracts import structured_patch as S  # noqa: E402
from refcheck import container as C  # noqa: E402

BASE = os.path.expanduser("~/chris-models/larql-registry/gemma-3-270m-it.vindex3")
LARQL = os.path.join(REPO, "target", "release", "larql")
WORK = os.path.join(HERE, "work")
TENSOR, ROWS, COLS_TOTAL = "5.mlp.down_proj.weight", 640, 2048
COLUMNS = [0, 1, 2, 3, 1000, 1001, 2046, 2047]
SEED, SCALE, PROFILE = 20261112, "1.0", "f64-seq-rne1/v1"
CONTRACT = S.CONTRACT
BF16 = "BF16"
DELTA_NEIGHBOURS = 8
EXPECTED_ELEMENTS = 100_325_376 + 167_772_160 + 640
ORIGIN = f"structured-real-1:default_rng({SEED}).standard_normal({ROWS}):unit-norm-f64"
OTHER_TENSOR = "6.mlp.down_proj.weight"
UNTOUCHED = (0, 5)


def sh(cmd, env=None, timeout=None):
    """Run a command under /usr/bin/time -l: (returncode, stdout, stderr, wall seconds, peak RSS bytes)."""
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True, text=True, env=env, timeout=timeout)
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    return p.returncode, p.stdout, p.stderr, round(time.time() - t0, 1), int(m.group(1)) if m else None


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digests(root):
    out = {}
    for d, _, names in os.walk(root):
        for n in sorted(names):
            p = os.path.join(d, n)
            out[os.path.relpath(p, root)] = sha_file(p)
    return out


def q8(path):
    return str(path).replace("\\", "\\\\")


def bits_hex(x):
    return f"{struct.unpack('<Q', struct.pack('<d', float(x)))[0]:016x}"


def lql(statement, env=None):
    return sh([LARQL, "lql", statement], env=env)


def tensor_array(root, tensor, rep_prefix="target.decoder"):
    c = C.open_container(root)
    ref = next(t for t in c.tensors(BF16) if t.name == tensor and t.rep_id.startswith(rep_prefix))
    return np.frombuffer(c.read_bytes(ref), dtype="<u2").reshape(ref.shape)


def reseal(root, seg):
    with open(seg.path, "rb") as f:
        data = f.read()
    payload = data[seg.payload_start:seg.payload_start + seg.payload_bytes]
    p = os.path.join(root, "index.json")
    index = json.load(open(p))
    e = index["representations"][seg.rep_id]
    e["segment_sha256"], e["payload_sha256"] = hashlib.sha256(data).hexdigest(), hashlib.sha256(payload).hexdigest()
    json.dump(index, open(p, "w"))


def flip(root, tensor, row, col):
    c = C.open_container(root)  # opened first: it checks the digests that are about to go stale
    ref = next(t for t in c.tensors(BF16) if t.name == tensor)
    seg = c.segments[ref.rep_id]
    at = ref.abs_start + (row * ref.shape[-1] + col) * 2
    with open(ref.segment_path, "r+b") as f:
        f.seek(at)
        (b,) = struct.unpack("<H", f.read(2))
        f.seek(at)
        f.write(struct.pack("<H", b ^ 1))
    reseal(root, seg)


def f32_bits_to_value(b):
    return Fraction(struct.unpack("<f", struct.pack("<I", b))[0])


def delta_reachable(old, new):
    """Could `UPDATE WEIGHTS` have written this? Search f32 deltas near new-old as the engine's own golden does;
    `patched_bits` is RNE_bf16(f64(old)+f64(delta)), reproduced exactly on rationals."""
    f = lambda b: struct.unpack("<f", struct.pack("<I", b << 16))[0]  # noqa: E731
    centre = struct.unpack("<I", struct.pack("<f", float(np.float32(f(new)) - np.float32(f(old)))))[0]
    for s in range(-DELTA_NEIGHBOURS, DELTA_NEIGHBOURS + 1):
        d = struct.unpack("<f", struct.pack("<I", (centre + s) & 0xFFFFFFFF))[0]
        if d != d or d in (float("inf"), float("-inf")):
            continue
        exact = S._from_bf16_bits(old)  # noqa: SLF001
        total = S.add(exact, S._from_f64_bits(struct.unpack("<Q", struct.pack("<d", d))[0]))  # noqa: SLF001
        if S.to_bf16_bits(total, via_f32=False) == new:
            return True
    return False


def main():
    os.makedirs(WORK, exist_ok=True)
    R = {"protocol": "STRUCTURED-REAL-1", "criteria": {}, "recorded": {}}
    wrapper = os.path.join(WORK, "verifier.sh")
    open(wrapper, "w").write(f'#!/bin/sh\nexec python3 -c "import sys; sys.path.insert(0, \'{os.path.join(REPO, "verifier")}\'); '
                             f'from larql_contract_verifier.cli import main; main()" "$@"\n')
    os.chmod(wrapper, 0o755)
    env = dict(os.environ, LARQL_VERIFIER=wrapper)

    before = tree_digests(BASE)
    R["recorded"]["registry_digests_before"] = {"index.json": before["index.json"]}

    # direction and operand
    u = np.random.default_rng(SEED).standard_normal(ROWS)
    u = u / np.linalg.norm(u)
    operand = os.path.join(WORK, "u.json")
    json.dump({"bits": [bits_hex(x) for x in u], "origin": ORIGIN}, open(operand, "w"))
    R["recorded"]["operand_sha256"] = sha_file(operand)

    vlp, out1, out2 = (os.path.join(WORK, n) for n in ("patch.vlp", "out1.vindex3", "out2.vindex3"))
    for p in (vlp, out1, out2):
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))
    cols = ", ".join(map(str, COLUMNS))
    rc, so, se, wall, rss = lql(f'USE "{q8(BASE)}"; BEGIN PATCH "{q8(vlp)}"; REMOVE DIRECTION "{q8(operand)}" FROM "{TENSOR}" '
                                f'COLUMNS ({cols}) SCALE {SCALE} NUMERICS "{PROFILE}"; SAVE PATCH;', env)
    R["recorded"]["author"] = {"rc": rc, "stdout": so.strip().splitlines()[-6:], "wall_s": wall, "peak_rss": rss}
    if rc:
        R["recorded"]["author"]["stderr"] = se[-800:]
        return finish(R, 1)

    # 1 source binding
    op = json.load(open(vlp))["operations"][0]
    index = json.load(open(os.path.join(BASE, "index.json")))
    rep = next(r for r in C.open_container(BASE).tensors(BF16) if r.name == TENSOR).rep_id
    R["criteria"]["1_source_binding"] = (op["source"]["index_sha256"] == before["index.json"]
                                         and op["source"]["payload_sha256"] == index["representations"][rep]["payload_sha256"])

    # 2 native compile
    rc, so, se, wall, rss = lql(f'USE "{q8(BASE)}"; APPLY PATCH "{q8(vlp)}"; COMPILE CURRENT INTO VINDEX "{q8(out1)}";', env)
    ledger = json.load(open(os.path.join(out1, "weight_patches.json"))) if os.path.isdir(out1) else {}
    R["recorded"]["compile"] = {"rc": rc, "stdout": so.strip().splitlines()[-8:], "wall_s": wall, "peak_rss": rss}
    R["criteria"]["2_native_compile"] = rc == 0 and ledger.get("schema") == "weight-patches/v3"
    if rc:
        R["recorded"]["compile"]["stderr"] = se[-800:]
        return finish(R, 1)

    # 3 independent verification, through the LQL statement and directly
    rc, so, se, wall, rss = lql(f'VERIFY PATCH "{q8(vlp)}" USING CONTRACT "{CONTRACT}" FROM "{q8(BASE)}" AGAINST "{q8(out1)}";', env)
    R["recorded"]["verify_lql"] = {"rc": rc, "stdout": so.strip().splitlines()[-8:], "wall_s": wall, "peak_rss": rss}
    rc, so, se, wall, rss = sh([wrapper, "patch", "--contract", CONTRACT, "--patch", vlp, "--base", BASE, "--container", out1])
    answer = json.loads(so) if so.strip() else {}
    m = {x["name"]: x["value"] for x in answer.get("measurements", [])}
    R["recorded"]["verify_direct"] = {"rc": rc, "wall_s": wall, "peak_rss": rss, "answer": {k: answer.get(k) for k in (
        "status", "violations", "numerical_satisfaction", "ledger_integrity", "provenance", "provenance_detail", "reasons")},
        "measurements": m}
    compared, expected = int(m.get("elements compared", 0)), int(m.get("elements expected to change", 0))
    R["criteria"]["3_independent_verification"] = (answer.get("status") == "satisfied" and answer.get("violations") == 0
                                                   and m.get("elements differing from the recomputed bits") == "0"
                                                   and compared == EXPECTED_ELEMENTS and expected >= 1)

    # 4 preservation, by bytes and not by the verifier
    base_c, new_c = C.open_container(BASE), C.open_container(out1)
    base_t = {(t.rep_id, t.name): t for t in base_c.tensors(BF16)}
    new_t = {(t.rep_id, t.name): t for t in new_c.tensors(BF16)}
    other_changed, same = [], True
    for key, t in base_t.items():
        a, b = base_c.read_bytes(t), new_c.read_bytes(new_t[key])
        if key[1] != TENSOR and a != b:
            other_changed.append(key)
        if key[1] == TENSOR:
            old, new = np.frombuffer(a, "<u2").reshape(t.shape), np.frombuffer(b, "<u2").reshape(t.shape)
    outside = np.ones(old.shape[1], bool)
    outside[COLUMNS] = False
    changed_idx = np.argwhere(old != new)
    seg_same = {n: sha_file(os.path.join(BASE, "segments", n)) == sha_file(os.path.join(out1, "segments", n))
                for n in ("target.embedding.bin", "target.final_norm.bin")}
    R["criteria"]["4_preservation"] = (not other_changed and bool((old[:, outside] == new[:, outside]).all())
                                       and all(seg_same.values()) and len(changed_idx) >= 1)
    R["recorded"]["preservation"] = {"other_tensors_changed": len(other_changed), "embedding_final_norm_segments_identical": seg_same,
                                     "edited_columns_outside_declared_changed": int((old[:, outside] != new[:, outside]).sum())}

    # recorded: what changed
    f = lambda b: struct.unpack("<f", struct.pack("<I", int(b) << 16))[0]  # noqa: E731
    relchg = [abs(f(new[r, c]) - f(old[r, c])) / abs(f(old[r, c])) for r, c in changed_idx if f(old[r, c]) != 0]
    unreachable = [(int(r), int(c)) for r, c in changed_idx if not delta_reachable(int(old[r, c]), int(new[r, c]))]
    R["recorded"]["changed"] = {"elements": int(len(changed_idx)), "of_declared_columns": len(COLUMNS) * ROWS,
                                "max_relative_change": max(relchg) if relchg else None,
                                "unreachable_by_any_f32_delta_searched": len(unreachable), "first_unreachable": unreachable[:5]}

    # 5 mutation sensitivity (copies, digests re-sealed)
    r0, c0 = (int(changed_idx[0][0]), int(changed_idx[0][1]))
    mutations = {"changed_element": (TENSOR, r0, c0), "untouched_element": (TENSOR, *UNTOUCHED), "other_tensor": (OTHER_TENSOR, 0, 0)}
    results = {}
    for name, (tensor, row, col) in mutations.items():
        mroot = os.path.join(WORK, f"mut_{name}.vindex3")
        shutil.rmtree(mroot, ignore_errors=True)
        shutil.copytree(out1, mroot)
        flip(mroot, tensor, row, col)
        rc, so, se, wall, rss = sh([wrapper, "patch", "--contract", CONTRACT, "--patch", vlp, "--base", BASE, "--container", mroot])
        a = json.loads(so) if so.strip() else {}
        results[name] = {"status": a.get("status"), "violations": a.get("violations"), "wall_s": wall,
                         "first": (a.get("offending") or [{}])[0].get("detail")}
        shutil.rmtree(mroot, ignore_errors=True)
    R["recorded"]["mutations"] = results
    R["criteria"]["5_mutation_sensitivity"] = all(v["status"] == "violated" and v["violations"] == 1 for v in results.values())

    # 6 reproducibility
    rc, so, se, wall, rss = lql(f'USE "{q8(BASE)}"; APPLY PATCH "{q8(vlp)}"; COMPILE CURRENT INTO VINDEX "{q8(out2)}";', env)
    d1, d2 = tree_digests(out1), tree_digests(out2)
    d1.pop("candidate.json", None), d2.pop("candidate.json", None)  # carries the output's temp-path locator hint
    R["criteria"]["6_reproducible"] = rc == 0 and d1 == d2
    R["recorded"]["second_compile"] = {"rc": rc, "wall_s": wall, "differing_files": sorted(k for k in set(d1) | set(d2) if d1.get(k) != d2.get(k))}

    # 7 registry untouched
    R["criteria"]["7_registry_unchanged"] = tree_digests(BASE) == before
    return finish(R, 0)


def finish(R, rc):
    R["all_criteria_hold"] = bool(R["criteria"]) and all(R["criteria"].values()) and len(R["criteria"]) == 7
    json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=2, default=str)
    print(json.dumps({"criteria": R["criteria"], "all_criteria_hold": R["all_criteria_hold"]}, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
