"""ANE-4 accuracy causality — is the ANE's accumulation actually the cause?

ANE-4 step 1 observed `rel_rms` 1.27e-03 for the FFN block with
`down_proj` on CPU (intermediate 17408) against 3.30e-02 with it on ANE
(intermediate 16384), and attributed the gap to the execution device.
**That attribution was not controlled**: the two variants differ in
device AND in every weight matrix, activation width and reduction
length. `rel_rms` can move nonlinearly with cancellation and reference
magnitude, so "6% width cannot explain 26x" is plausible, not causal.

This harness supplies the missing control, in the only form that isolates
the variable:

    ONE compiled artifact, loaded TWICE, differing only in `computeUnits`.

Same graph, same weights, same input, same reference. If the error moves,
the device moved it.

Two modes:

  control   the FFN block at intermediate 16384, CPU_AND_NE vs CPU_ONLY
  ladder    an isolated `linear`, k = 5120 / 8192 / 12288 / 16384, each
            run on both devices — because "the ANE is less accurate" is
            far less useful to a planner than an error-vs-reduction-depth
            curve.

The ladder starts at 5120, not 4096: ANE-2A showed f16 k=4096 is
CPU-preferred, so that pair would silently be CPU-vs-CPU. Every arm's
actual placement is verified and reported, and a pair whose "ANE" arm was
not placed on the ANE is marked unusable rather than compared.

Usage:
    python ane4_accuracy.py control|ladder <session> [out.json]
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
LADDER_KS = [5120, 8192, 12288, 16384]
WARMUP_ITERS = 16
MEASURED_ITERS = 64


def placement_for(mlmodelc, units):
    """Per-op placement AS REQUESTED — the CPU_ONLY arm must be verified
    to have actually run on CPU, not assumed."""
    plan = MLComputePlan.load_from_path(path=mlmodelc, compute_units=units)
    main = plan.model_structure.program.functions["main"]
    out = []
    for op in main.block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if u is None:
            continue
        out.append(
            {
                "op": op.operator_name,
                "preferred": ane1.device_short(type(u.preferred_compute_device).__name__),
            }
        )
    return out


def run_arm(mlmodelc, units, x):
    model = ct.models.CompiledMLModel(mlmodelc, compute_units=units)
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
    return got, samples[0], samples[len(samples) // 2]


def rel_rms(got, ref):
    return float(np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref**2)))


def build_linear(n, k, w, work_dir, tag):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, k), dtype=types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        return mb.linear(x=x, weight=w, name="proj")

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


def compare(tag, mlmodelc, x, ref, extra=None):
    """The control itself: one artifact, two devices."""
    rows = {}
    for label, units in (("ANE", ct.ComputeUnit.CPU_AND_NE), ("CPU", ct.ComputeUnit.CPU_ONLY)):
        ops = placement_for(mlmodelc, units)
        compute_ops = [o for o in ops if o["op"].endswith("linear")]
        devices = sorted({o["preferred"] for o in compute_ops}) or ["?"]
        got, mn, p50 = run_arm(mlmodelc, units, x)
        rows[label] = {
            "requested": label,
            "linear_devices": devices,
            "rel_rms": rel_rms(got, ref),
            "min_ms": mn,
            "p50_ms": p50,
        }
    usable = "ANE" in rows["ANE"]["linear_devices"] and rows["CPU"]["linear_devices"] == ["CPU"]
    ratio = rows["ANE"]["rel_rms"] / rows["CPU"]["rel_rms"] if rows["CPU"]["rel_rms"] else float("nan")
    print(
        f"{tag:>16}"
        f"{'+'.join(rows['ANE']['linear_devices']):>10}{rows['ANE']['rel_rms']:>12.3e}"
        f"{'+'.join(rows['CPU']['linear_devices']):>10}{rows['CPU']['rel_rms']:>12.3e}"
        f"{ratio:>10.1f}x{'' if usable else '   UNUSABLE'}"
    )
    row = {"tag": tag, "arms": rows, "error_ratio_ane_over_cpu": ratio, "usable": usable}
    if extra:
        row.update(extra)
    return row


def header():
    print(
        f"{'case':>16}{'ANE dev':>10}{'ANE rel_rms':>12}"
        f"{'CPU dev':>10}{'CPU rel_rms':>12}{'ratio':>11}"
    )


def run_control(work_dir):
    """The FFN block at 16384, one artifact, two devices."""
    intermediate = ffn.ELIGIBLE_INTERMEDIATE
    x = ffn.prng((1, HIDDEN_SIZE), seed=7)
    weights = (
        ffn.prng((intermediate, HIDDEN_SIZE), seed=1),
        ffn.prng((intermediate, HIDDEN_SIZE), seed=2),
        ffn.prng((HIDDEN_SIZE, intermediate), seed=3),
        ffn.prng((1, HIDDEN_SIZE), seed=4),
    )
    mlc = ffn.build_block(intermediate, work_dir, "acc_block", weights)
    ref = ffn.reference(x, weights, intermediate).astype(np.float64)
    print(f"\n-- FFN block, intermediate {intermediate}: ONE artifact, two devices --")
    header()
    row = compare(f"ffn_{intermediate}", mlc, x, ref, {"intermediate": intermediate})
    shutil.rmtree(mlc, ignore_errors=True)
    shutil.rmtree(mlc.replace(".mlmodelc", ".mlpackage"), ignore_errors=True)
    return [row]


def run_ladder(work_dir):
    """Error vs reduction depth, both devices, same tensors per pair."""
    rows = []
    print(f"\n-- isolated linear, n = {HIDDEN_SIZE}: error vs reduction depth --")
    header()
    for k in LADDER_KS:
        w = ffn.prng((HIDDEN_SIZE, k), seed=11)
        x = ffn.prng((1, k), seed=12)
        mlc = build_linear(HIDDEN_SIZE, k, w, work_dir, f"acc_k{k}")
        ref = (w.astype(np.float64) @ x.astype(np.float64).reshape(-1))
        rows.append(compare(f"k={k}", mlc, x, ref, {"k": k, "n": HIDDEN_SIZE}))
        shutil.rmtree(mlc, ignore_errors=True)
        shutil.rmtree(mlc.replace(".mlmodelc", ".mlpackage"), ignore_errors=True)
        del w

    usable = [r for r in rows if r["usable"]]
    if len(usable) >= 2:
        print("\nANE error vs reduction depth:")
        for r in usable:
            print(f"  k={r['k']:>6}  ANE {r['arms']['ANE']['rel_rms']:.3e}"
                  f"   CPU {r['arms']['CPU']['rel_rms']:.3e}"
                  f"   ratio {r['error_ratio_ane_over_cpu']:.1f}x")
        first, last = usable[0], usable[-1]
        growth = last["arms"]["ANE"]["rel_rms"] / first["arms"]["ANE"]["rel_rms"]
        depth = last["k"] / first["k"]
        print(f"  ANE error grew {growth:.2f}x across a {depth:.2f}x increase in k")
        print("  (sqrt(k) growth would be "
              f"{depth ** 0.5:.2f}x, linear-in-k would be {depth:.2f}x)")
    return rows


def main():
    mode = sys.argv[1]
    session = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    work_dir = os.environ.get("ANE4_WORK_DIR", "/tmp/ane4-work")
    os.makedirs(work_dir, exist_ok=True)

    print(f"ANE-4 accuracy causality [{mode}] — session '{session}', "
          f"coremltools {ct.__version__}")
    rows = run_control(work_dir) if mode == "control" else run_ladder(work_dir)

    if out_path:
        with open(out_path, "w") as fh:
            json.dump({"experiment": f"ANE-4 accuracy {mode}", "session": session,
                       "rows": rows}, fh, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
