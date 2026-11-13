#!/usr/bin/env python3
"""PATCH-TRAIN-1 L1 report: emulator exactness evidence and performance on the intended footprint. Writes emulator_report.json.
    python3 -I report_emulator.py
"""
import collections
import json
import os
import statistics
import sys
import time
import tracemalloc

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import crosscheck as X  # noqa: E402
import emulate as E  # noqa: E402
import test_emulator as T  # noqa: E402

sys.path.insert(0, os.path.join(X.REPO, "verifier"))
from refcheck import container as C  # noqa: E402

CONTAINER = os.path.expanduser("~/chris-models/larql-registry/gemma-3-270m-it.vindex3")
FOOTPRINT_TENSOR, FOOTPRINT_COLUMNS, FOOTPRINT_SEED = "9.mlp.down_proj.weight", 256, 20261113
RANKS, REPEATS = (1, 2, 4), 30


def exactness():
    out = {}
    cases = T.load_vectors()
    out["frozen_vectors"] = {"cases": len(cases), "mismatches": sum(
        E.run_case(c) != (T.expected_outcome(c)[0], list(T.expected_outcome(c)[1]) if T.expected_outcome(c)[0] == "bf16"
                          else T.expected_outcome(c)[1]) for c in cases)}
    rand = X.random_cases(T.RANDOM_CASES, T.RANDOM_SEED)
    ind = X.run_independent(rand)
    emu = {c["id"]: E.run_case(c) for c in rand}
    kinds = collections.Counter()
    for c in rand:
        kind, val = emu[c["id"]]
        kinds[(c["op"]["numerics"], "bf16" if kind == "bf16" else val)] += 1
    out["random_vs_independent_evaluator"] = {
        "cases": len(rand), "seed": T.RANDOM_SEED, "mismatches": len(X.differing(emu, ind)),
        "numerical_results_by_profile": {p: kinds[(p, "bf16")] for p in X.PROFILES},
        "refusals_by_code": dict(collections.Counter(v[1] for v in emu.values() if v[0] == "refuse"))}
    mutants = {}
    for name, m in {"start_from_zero": dict(start_from_zero=True), "no_selection": dict(no_selection=True),
                    "reverse_k": dict(reverse_k=True), "rne1_via_f32": dict(via_f32=True), "ties_away": dict(ties_away=True)}.items():
        frozen, random_wrong = T.SemanticMutants().caught(**m)
        mutants[name] = {"frozen_vectors_that_catch_it": frozen, "random_cases_that_catch_it_of_3000": random_wrong}
    out["semantic_mutants_caught"] = mutants
    return out


def footprint():
    c = C.open_container(CONTAINER)
    ref = next(t for t in c.tensors("BF16") if t.name == FOOTPRINT_TENSOR)
    base = np.frombuffer(c.read_bytes(ref), dtype="<u2").reshape(ref.shape).copy()
    rng = np.random.default_rng(FOOTPRINT_SEED)
    return base, np.sort(rng.choice(base.shape[1], FOOTPRINT_COLUMNS, replace=False)), rng


def performance():
    base, cols, rng = footprint()
    rows = base.shape[0]
    out = {"tensor": FOOTPRINT_TENSOR, "shape": list(base.shape), "columns_in_footprint": len(cols), "results": []}
    for label, columns in (("256 columns", cols), ("all 2048 columns", np.arange(base.shape[1]))):
        for profile in E.PROFILES:
            for k in RANKS:
                work = np.float32 if profile == E.PROFILE_F32 else np.float64
                a = (rng.standard_normal((rows, k)) * 1e-3).astype(work)
                b = (rng.standard_normal((k, len(columns))) * 1e-1).astype(work)
                times = []
                for _ in range(REPEATS):
                    t = time.perf_counter()
                    E.apply_columns(base, columns, a, b, profile)
                    times.append((time.perf_counter() - t) * 1000)
                tracemalloc.start()
                E.apply_columns(base, columns, a, b, profile)
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                out["results"].append({"footprint": label, "profile": profile, "rank": k,
                                       "median_ms": round(statistics.median(times), 2), "p95_ms": round(sorted(times)[int(0.95 * REPEATS) - 1], 2),
                                       "peak_python_alloc_MB": round(peak / 1e6, 2)})
    return out


def main():
    report = {"protocol": "PATCH-TRAIN-1 L1 emulator report", "exactness": exactness(), "performance": performance(),
              "numpy": np.__version__, "python": sys.version.split()[0]}
    with open(os.path.join(HERE, "emulator_report.json"), "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps({"exactness": report["exactness"]}, indent=1))
    for r in report["performance"]["results"]:
        if r["rank"] in (1, 4):
            print(r)


if __name__ == "__main__":
    main()
