"""ANE-1 — does a real LARQL decode-shaped op actually land on the ANE?

Scope is frozen: ONE f16 projection, `5120 -> 17408`, Qwen3.8-27B's FFN
gate/up. Nothing else. No int8 yet — f16 placement first, so that a later
int8 rung answers a clean second question (does compression change
placement, footprint or behaviour?) instead of debugging quantisation and
placement at the same time.

The rung answers, in this order:

  1. can Core ML compile it?
  2. does MLComputePlan prove the projection is assigned to the ANE?
  3. is the placement reader itself trustworthy (negative controls)?
  4. is the output numerically plausible?
  5. what is the warm steady-state latency distribution?
  6. what equivalent traffic rate would that latency imply?

**No placement proof, no benchmark result.** If CPU_AND_NE is requested
and the plan reports CPU, that is an ANE-1 placement failure, not "the
ANE was slow", and no timing is reported for it. Instead the harness
walks progressively smaller model-adjacent shapes to locate the
placement boundary — which is the first point on ANE-2's envelope curve
and tells us roughly how large an ANE-4 drafter may be.

"Equivalent rate" is deliberate wording. Core ML may specialise,
compress, cache or transform constants, so `weight_bytes / latency` is
what the latency would imply IF the weights were fetched each
prediction. It is not measured physical traffic. ANE-2 is where that
gets evidence.

Usage:
    python ane1_placement.py <session-label> [out.json]
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

# Qwen3.8-27B text config, snapshot 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408

# Mirrors ANE-0b's dispatch_floor: same idea, same width, so the two
# harnesses' overhead lines are comparable.
FLOOR_ROWS = 8

BYTES_PER_F16 = 2

WARMUP_ITERS = 32
MEASURED_ITERS = 256

# ANE-0b banked this exact row for the same weights and activation, on
# the Metal GPU. A cross-instrument agreement here is much stronger than
# any single-engine self-check.
ANE0B_PROBE_ROW = INTERMEDIATE_SIZE // 2
ANE0B_PROBE_VALUE = 0.077413

# Above the machine's ~400 GB/s fabric, an equivalent rate means the
# weights are not being moved as assumed — interesting, but not a
# bandwidth result.
IMPLAUSIBLE_GBS = 600.0

# f16 inputs through an f16 accumulate chain: agreement to ~1e-3 relative
# is plausible; orders of magnitude worse means it is not our projection.
MAX_REL_RMS = 5e-2

# The placement boundary sweep, largest first. Every entry is a real
# projection from the model except the last two, which are drafter-sized
# and exist to find where the ANE starts being preferred at all.
SWEEP = [
    ("ffn_gate_up", INTERMEDIATE_SIZE, HIDDEN_SIZE),
    ("attn_q", 12288, HIDDEN_SIZE),
    ("attn_o", HIDDEN_SIZE, 6144),
    ("linear_v", 6144, HIDDEN_SIZE),
    ("linear_qk", 2048, HIDDEN_SIZE),
    ("attn_kv", 1024, HIDDEN_SIZE),
    ("drafter_2048", 2048, 2048),
    ("drafter_1024", 1024, 1024),
    ("drafter_512", 512, 512),
]


def weights(n, k):
    """Identical generator to ANE-0b, so outputs are comparable."""
    return (np.arange(n * k) % 977 / 977.0 - 0.5).astype(np.float16).reshape(n, k)


def activation(k):
    return ((np.arange(k) % 13) * 0.01 - 0.06).astype(np.float16).reshape(1, k)


def build(n, k, w, work_dir, tag):
    """Convert, save and compile one linear projection. Returns the .mlmodelc."""

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
    t0 = time.perf_counter()
    model.save(pkg)
    compile_model(pkg, destination_path=mlc)
    compile_ms = (time.perf_counter() - t0) * 1e3
    return mlc, compile_ms


def placement(mlmodelc, compute_units):
    """Per-op preferred and supported devices, from MLComputePlan.

    `preferred` is the placement. `supported` says only that the device
    *could* host the op, which is not the same claim and must not be
    reported as one.
    """
    plan = MLComputePlan.load_from_path(path=mlmodelc, compute_units=compute_units)
    main = plan.model_structure.program.functions["main"]
    ops = []
    for op in main.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        ops.append(
            {
                "op": op.operator_name,
                "preferred": type(usage.preferred_compute_device).__name__,
                "supported": [
                    type(d).__name__ for d in usage.supported_compute_devices
                ],
            }
        )
    return ops


def device_short(name):
    return {
        "MLNeuralEngineComputeDevice": "ANE",
        "MLCPUComputeDevice": "CPU",
        "MLGPUComputeDevice": "GPU",
    }.get(name, name)


def compute_op(ops):
    """The projection itself — `const` ops carry no placement."""
    for o in ops:
        if "linear" in o["op"]:
            return o
    return ops[-1] if ops else None


def stats(samples):
    s = sorted(samples)
    n = len(s)
    mean = sum(s) / n
    var = sum((v - mean) ** 2 for v in s) / n
    return {
        "min": s[0],
        "p50": s[n // 2],
        "p90": s[(n * 9) // 10],
        "max": s[-1],
        "mean": mean,
        "stdev": var**0.5,
    }


def time_model(mlmodelc, x, compute_units):
    model = ct.models.CompiledMLModel(mlmodelc, compute_units=compute_units)
    for _ in range(WARMUP_ITERS):
        model.predict({"x": x})
    samples = []
    out = None
    for _ in range(MEASURED_ITERS):
        t = time.perf_counter()
        out = model.predict({"x": x})
        samples.append((time.perf_counter() - t) * 1e3)
    return stats(samples), np.asarray(list(out.values())[0]).reshape(-1)


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "unlabelled"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    work_dir = os.environ.get("ANE1_WORK_DIR", "/tmp/ane1-work")
    os.makedirs(work_dir, exist_ok=True)

    n, k = INTERMEDIATE_SIZE, HIDDEN_SIZE
    weight_bytes = n * k * BYTES_PER_F16
    report = {
        "experiment": "ANE-1",
        "session": session,
        "coremltools": ct.__version__,
        "shape": {"name": "ffn_gate_up", "n": n, "k": k, "weight_bytes": weight_bytes},
        "iters": {"warmup": WARMUP_ITERS, "measured": MEASURED_ITERS},
    }

    print(f"ANE-1 — session '{session}', coremltools {ct.__version__}")
    print(f"subject: ffn_gate_up {k} -> {n}, f16, {weight_bytes / 1e6:.2f} MB\n")

    # --- 1. compile -------------------------------------------------
    w = weights(n, k)
    x = activation(k)
    mlmodelc, compile_ms = build(n, k, w, work_dir, "primary")
    report["compile_ms"] = compile_ms
    print(f"1. compiled in {compile_ms:.0f} ms -> {mlmodelc}")

    # --- 2. placement -----------------------------------------------
    ops = placement(mlmodelc, ct.ComputeUnit.CPU_AND_NE)
    report["placement_cpu_and_ne"] = ops
    op = compute_op(ops)
    pref = device_short(op["preferred"])
    sup = [device_short(d) for d in op["supported"]]
    print(f"\n2. {k}->{n}   requested: CPU+ANE   actual: {pref}   (supported: {', '.join(sup)})")

    # --- 3. negative controls ---------------------------------------
    # (a) the reader must track the request, not report ANE by habit.
    cpu_ops = placement(mlmodelc, ct.ComputeUnit.CPU_ONLY)
    cpu_pref = device_short(compute_op(cpu_ops)["preferred"])
    control_a = cpu_pref == "CPU"
    report["control_request_tracking"] = {"requested": "CPU_ONLY", "actual": cpu_pref, "pass": control_a}
    print(f"3a. negative control — CPU_ONLY requested, reader says {cpu_pref}: "
          f"{'PASS' if control_a else 'FAIL'}")

    # (b) an op the ANE cannot host must be absent from `supported`.
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 64), dtype=types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def unsupported_prog(x):
        return mb.cumsum(x=x, axis=1, name="unsupported")

    um = ct.convert(
        unsupported_prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    upkg = os.path.join(work_dir, "unsupported.mlpackage")
    umlc = os.path.join(work_dir, "unsupported.mlmodelc")
    for stale in (upkg, umlc):
        if os.path.exists(stale):
            shutil.rmtree(stale)
    um.save(upkg)
    compile_model(upkg, destination_path=umlc)
    uops = placement(umlc, ct.ComputeUnit.CPU_AND_NE)
    ucompute = compute_op(uops) or {"op": "?", "preferred": "?", "supported": []}
    usup = [device_short(d) for d in ucompute["supported"]]
    control_b = "ANE" not in usup
    report["control_unsupported_op"] = {
        "op": ucompute["op"],
        "supported": usup,
        "pass": control_b,
    }
    print(f"3b. negative control — {ucompute['op']} supported on {usup or ['(none reported)']}: "
          f"{'PASS' if control_b else 'INCONCLUSIVE'}")

    # --- placement gate ---------------------------------------------
    if pref != "ANE":
        print(f"\nANE-1 PLACEMENT FAILURE: requested CPU+ANE, got {pref}.")
        print("No benchmark result is reported. Locating the placement boundary instead.\n")
        report["verdict"] = "placement-failure"
        report["sweep"] = sweep_boundary(work_dir)
        finish(report, out_path)
        return

    # --- 4. numerical control ---------------------------------------
    ts, out = time_model(mlmodelc, x, ct.ComputeUnit.CPU_AND_NE)
    ref = (w.astype(np.float64) @ x.astype(np.float64).reshape(-1))
    err = np.abs(out.astype(np.float64) - ref)
    rel_rms = float(np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(ref**2)))
    probe = float(out[ANE0B_PROBE_ROW])
    cross = abs(probe - ANE0B_PROBE_VALUE) / abs(ANE0B_PROBE_VALUE)
    report["numerical"] = {
        "rel_rms_vs_f64_reference": rel_rms,
        "max_abs_err": float(err.max()),
        "probe_row": ANE0B_PROBE_ROW,
        "ane_value": probe,
        "ane0b_gpu_value": ANE0B_PROBE_VALUE,
        "cross_instrument_rel": cross,
        "pass": rel_rms <= MAX_REL_RMS,
    }
    print(f"4. numerical: rel_rms {rel_rms:.3e} vs f64 reference "
          f"({'PASS' if rel_rms <= MAX_REL_RMS else 'FAIL'})")
    print(f"   cross-instrument: ANE row {ANE0B_PROBE_ROW} = {probe:.6f} vs "
          f"ANE-0b GPU {ANE0B_PROBE_VALUE:.6f}  (rel {cross:.3e})")

    # --- 5. latency + 6. equivalent rate -----------------------------
    floor_w = weights(FLOOR_ROWS, k)
    floor_mlc, _ = build(FLOOR_ROWS, k, floor_w, work_dir, "floor")
    floor_ts, _ = time_model(floor_mlc, x, ct.ComputeUnit.CPU_AND_NE)
    report["latency_ms"] = ts
    report["predict_floor_ms"] = floor_ts

    eq = {s: weight_bytes / (ts[s] / 1e3) / 1e9 for s in ("min", "p50")}
    report["equivalent_gbs"] = eq
    print(f"\n5. warm latency ms: min {ts['min']:.3f}  p50 {ts['p50']:.3f}  "
          f"p90 {ts['p90']:.3f}  max {ts['max']:.3f}  sd {ts['stdev']:.3f}")
    print(f"   predict floor ms: min {floor_ts['min']:.3f}  p50 {floor_ts['p50']:.3f}")
    print(f"\n6. equivalent rate if {weight_bytes / 1e6:.2f} MB moved per prediction:")
    print(f"   min {eq['min']:.1f} GB/s     p50 {eq['p50']:.1f} GB/s")
    print("   (equivalent, NOT measured traffic — Core ML may specialise,")
    print("    compress or cache constants. ANE-2 is where that gets evidence.)")
    if eq["min"] > IMPLAUSIBLE_GBS:
        print(f"   WARNING: above the ~{IMPLAUSIBLE_GBS:.0f} GB/s plausibility ceiling —")
        print("   this is a signal that the weights are NOT being moved as assumed.")

    report["verdict"] = "placed-on-ane"
    finish(report, out_path)


def sweep_boundary(work_dir):
    """Placement-only walk down the shape ladder. No timing: without a
    placement proof there is nothing to time."""
    print(f"{'shape':<16}{'n':>8}{'k':>8}{'MB':>9}   requested CPU+ANE -> actual")
    rows = []
    for name, n, k in SWEEP:
        w = weights(n, k)
        mlc, compile_ms = build(n, k, w, work_dir, f"sweep_{name}")
        ops = placement(mlc, ct.ComputeUnit.CPU_AND_NE)
        op = compute_op(ops)
        pref = device_short(op["preferred"])
        sup = [device_short(d) for d in op["supported"]]
        mb_ = n * k * BYTES_PER_F16 / 1e6
        print(f"{name:<16}{n:>8}{k:>8}{mb_:>9.2f}   {pref}   (supported: {', '.join(sup)})")
        rows.append(
            {
                "name": name,
                "n": n,
                "k": k,
                "weight_mb": mb_,
                "preferred": pref,
                "supported": sup,
                "compile_ms": compile_ms,
            }
        )
        shutil.rmtree(mlc, ignore_errors=True)
        shutil.rmtree(mlc.replace(".mlmodelc", ".mlpackage"), ignore_errors=True)
    return rows


def finish(report, out_path):
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {out_path}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
