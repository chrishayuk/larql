"""ANE-4 — the down_proj k-split. Placement, latency and accuracy at once.

`down_proj` is `17408 -> 5120`. ANE-2A found k > 16384 is not merely
unpreferred but **unsupported**, and ANE-4 step 1 confirmed that this one
op fractures the FFN graph, dragging SiLU, the gate*up multiply and the
residual add onto the CPU with it.

The obvious lowering — and VINDEX3 is the right layer to own it:

```
    y = W x                    W : [5120, 17408]   x : [17408]

    W = [W0 W1]                W0, W1 : [5120, 8704]
    x = [x0 x1]

    y = W0 x0 + W1 x1          both reductions ANE-admissible
```

Mathematically identical, same original weights, same input.

## Four arms, because three questions are being asked at once

```
CPU-full      17408 -> 5120, CPU_ONLY            the baseline the FFN uses today
ANE-split-2   8704 x 2 + sum, CPU_AND_NE         the candidate
CPU-split-2   8704 x 2 + sum, CPU_ONLY           ARITHMETIC CONTROL
ANE-split-4   4352 x 4 + sum, CPU_AND_NE         does finer splitting help or place?
CPU-split-4   4352 x 4 + sum, CPU_ONLY
```

**CPU-split-2 is the arm that makes the accuracy result readable.**
Without it, a change in error between CPU-full and ANE-split-2 conflates
"splitting changed the floating-point evaluation" with "the ANE changed
the precision". With it, the two are separable.

Note ANE-split-4's pieces are k = 4352, which ANE-2A put **below** the
f16 ANE preference edge of ~4992. It is expected to place on CPU, and is
run anyway: if the split has a lower bound as well as an upper one, that
is part of the capability contract.

## Pre-registration

- **Placement:** ANE-split-2's two projections should place on ANE
  (8704 is inside [~4992, 16384]). ANE-split-4's should not.
- **Latency:** unknown, and the bar is high. Step 1 measured the current
  tail at ~1.40 ms on CPU, and full artificial residency at 16384 bought
  only ~7% of block throughput. Two ANE projections plus a sum have to
  beat a competent CPU `down_proj` by enough to matter.
- **Accuracy:** no recovery is the mechanism-based prior, since ANE error
  was flat in reduction depth. It is NOT a prediction — splitting changes
  the numerical structure, and the CPU-split arm is here to tell the two
  causes apart.

Usage:
    python ane4_ksplit.py <session> [out.json]
"""

import json
import os
import shutil
import sys
import time

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan
from coremltools.models.utils import compile_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ane1_placement as ane1  # noqa: E402
import ane4_ffn_block as ffn  # noqa: E402

HIDDEN_SIZE = ane1.HIDDEN_SIZE
DOWN_K = ane1.INTERMEDIATE_SIZE  # 17408
WARMUP_ITERS = 16
MEASURED_ITERS = 64


def build_split(k_total, n, w, splits, work_dir, tag):
    """`splits` = 1 builds the unsplit projection; >1 builds the sum of
    `splits` partial reductions over the same weights."""
    piece = k_total // splits
    parts = [np.ascontiguousarray(w[:, i * piece:(i + 1) * piece]) for i in range(splits)]

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, k_total), dtype=types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        if splits == 1:
            return mb.linear(x=x, weight=parts[0], name="proj")
        chunks = mb.split(x=x, num_splits=splits, axis=-1, name="k_split")
        partials = [
            mb.linear(x=chunks[i], weight=parts[i], name=f"proj_{i}")
            for i in range(splits)
        ]
        acc = partials[0]
        for i in range(1, splits):
            acc = mb.add(x=acc, y=partials[i], name=f"partial_sum_{i}")
        return acc

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    pkg = os.path.join(work_dir, f"{tag}.mlpackage")
    mlc = os.path.join(work_dir, f"{tag}.mlmodelc")
    for stale in (pkg, mlc):
        if os.path.exists(stale):
            shutil.rmtree(stale)
    model.save(pkg)
    compile_model(pkg, destination_path=mlc)
    return mlc


def arm(mlc, units, x, ref, label, piece_k):
    plan = MLComputePlan.load_from_path(path=mlc, compute_units=units)
    main = plan.model_structure.program.functions["main"]
    linears, adds = [], []
    for op in main.block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is None:
            continue
        dev = ane1.device_short(type(u.preferred_compute_device).__name__)
        if op.operator_name.endswith("linear"):
            linears.append(dev)
        elif op.operator_name.endswith("add"):
            adds.append(dev)

    model = ct.models.CompiledMLModel(mlc, compute_units=units)
    for _ in range(WARMUP_ITERS):
        model.predict({"x": x})
    samples = []
    out = None
    for _ in range(MEASURED_ITERS):
        t = time.perf_counter()
        out = model.predict({"x": x})
        samples.append((time.perf_counter() - t) * 1e3)
    samples.sort()
    got = np.asarray(list(out.values())[0], dtype=np.float64).reshape(-1)
    err = float(np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref**2)))
    row = {
        "arm": label,
        "piece_k": piece_k,
        "linear_devices": linears,
        "sum_devices": adds,
        "min_ms": samples[0],
        "p50_ms": samples[len(samples) // 2],
        "rel_rms": err,
    }
    print(
        f"{label:<14}{piece_k:>8}{'+'.join(linears) or '-':>14}"
        f"{'+'.join(adds) or '-':>10}{row['min_ms']:>10.3f}"
        f"{row['p50_ms']:>10.3f}{err:>12.3e}"
    )
    return row


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "unlabelled"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    work_dir = os.environ.get("ANE4_WORK_DIR", "/tmp/ane4-work")
    os.makedirs(work_dir, exist_ok=True)

    w = ffn.prng((HIDDEN_SIZE, DOWN_K), seed=21)
    x = ffn.prng((1, DOWN_K), seed=22)
    # One common high-precision reference for every arm.
    ref = (w.astype(np.float64) @ x.astype(np.float64).reshape(-1))

    print(f"ANE-4 k-split — session '{session}', down_proj {DOWN_K} -> {HIDDEN_SIZE}")
    print(f"weights {w.nbytes / 1e6:.1f} MB, common f64 reference\n")
    print(f"{'arm':<14}{'piece k':>8}{'linear dev':>14}{'sum dev':>10}"
          f"{'min ms':>10}{'p50 ms':>10}{'rel_rms':>12}")

    rows = []
    for splits, label_ane, label_cpu in ((1, None, "CPU-full"),
                                         (2, "ANE-split-2", "CPU-split-2"),
                                         (4, "ANE-split-4", "CPU-split-4")):
        mlc = build_split(DOWN_K, HIDDEN_SIZE, w, splits, work_dir, f"ks_{splits}")
        piece = DOWN_K // splits
        if label_ane:
            rows.append(arm(mlc, ct.ComputeUnit.CPU_AND_NE, x, ref, label_ane, piece))
        else:
            # The unsplit projection under CPU_AND_NE is the status quo:
            # requested ANE, refused, runs CPU. Recorded once for the file.
            rows.append(arm(mlc, ct.ComputeUnit.CPU_AND_NE, x, ref, "full-req-ANE", piece))
        rows.append(arm(mlc, ct.ComputeUnit.CPU_ONLY, x, ref, label_cpu, piece))
        shutil.rmtree(mlc, ignore_errors=True)
        shutil.rmtree(mlc.replace(".mlmodelc", ".mlpackage"), ignore_errors=True)

    by = {r["arm"]: r for r in rows}
    base = by["CPU-full"]
    print("\n--- reading the three questions independently ---")

    print("\nplacement:")
    for name in ("full-req-ANE", "ANE-split-2", "ANE-split-4"):
        r = by[name]
        on_ane = sum(1 for d in r["linear_devices"] if d == "ANE")
        print(f"  {name:<14} {on_ane}/{len(r['linear_devices'])} projections on ANE"
              f"   sum on {'+'.join(r['sum_devices']) or 'n/a'}")

    print("\nlatency vs the CPU-full baseline the FFN uses today:")
    for name in ("ANE-split-2", "CPU-split-2", "ANE-split-4", "CPU-split-4"):
        r = by[name]
        print(f"  {name:<14} {r['min_ms']:.3f} ms   {r['min_ms'] / base['min_ms']:.2f}x baseline")

    print("\naccuracy, with the arithmetic control:")
    print(f"  CPU-full     {base['rel_rms']:.3e}   unsplit arithmetic, CPU precision")
    print(f"  CPU-split-2  {by['CPU-split-2']['rel_rms']:.3e}   "
          f"split arithmetic, CPU precision   -> isolates SPLITTING")
    print(f"  ANE-split-2  {by['ANE-split-2']['rel_rms']:.3e}   "
          f"split arithmetic, ANE precision   -> adds DEVICE")
    split_effect = by["CPU-split-2"]["rel_rms"] / base["rel_rms"]
    device_effect = by["ANE-split-2"]["rel_rms"] / by["CPU-split-2"]["rel_rms"]
    print(f"  splitting alone: {split_effect:.2f}x     device on top: {device_effect:.2f}x")

    if out_path:
        with open(out_path, "w") as fh:
            json.dump({"experiment": "ANE-4 k-split", "session": session,
                       "down_k": DOWN_K, "n": HIDDEN_SIZE, "rows": rows,
                       "split_effect": split_effect,
                       "device_effect": device_effect}, fh, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
