"""ANE-4 step 1 — does a realistic Qwen FFN block stay ANE-resident?

Every ANE result so far came from an isolated `linear`. This is the first
production-shaped question:

> Does a useful Qwen block remain meaningfully ANE-resident when Core ML
> sees the WHOLE graph rather than one matmul?

The block, exactly as Qwen3.8-27B runs it:

    RMSNorm -> gate_proj 5120->I, up_proj 5120->I
            -> SiLU(gate) * up
            -> down_proj I->5120
            -> residual add

ANE-2A established that placement is governed by the reduction depth k:
CPU below ~4992, ANE from there to 16384, and **not supported at all**
above. `down_proj` has k = 17408, so it is the one op in the block that
cannot go to the ANE. The open question is what Core ML does *around* it.

## The A/B control

```
A   intermediate = 17408   down_proj k = 17408   INELIGIBLE (real Qwen)
B   intermediate = 16384   down_proj k = 16384   ELIGIBLE   (2^14, the edge)
```

Only `down_proj`'s eligibility differs; gate/up keep k = 5120 in both. If
B stays substantially more ANE-resident than A, that is causal evidence
that **the k > 16384 rule is what fractures the graph**, rather than some
unrelated limitation on norms or elementwise ops.

`down_proj` is deliberately NOT split here. Let the natural graph fail or
partition first — otherwise a later k-split cannot be told apart from
having simply satisfied the isolated capability rule.

Usage:
    python ane4_ffn_block.py <session> [out.json]
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
import ane1_placement as ane1  # noqa: E402  (frozen ANE-1 instrument)

HIDDEN_SIZE = ane1.HIDDEN_SIZE
QWEN_INTERMEDIATE = ane1.INTERMEDIATE_SIZE  # 17408, down_proj k INELIGIBLE
ELIGIBLE_INTERMEDIATE = 16384               # 2^14, the last ANE-eligible k

RMS_EPS = 1e-6
WARMUP_ITERS = 32
MEASURED_ITERS = 128

VARIANTS = [
    ("A_qwen_17408", QWEN_INTERMEDIATE, "down_proj k INELIGIBLE (real Qwen)"),
    ("B_eligible_16384", ELIGIBLE_INTERMEDIATE, "down_proj k eligible (control)"),
]


def prng(shape, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape, dtype=np.float32) * 0.02).astype(np.float16)


def build_block(intermediate, work_dir, tag, weights):
    wg, wu, wd, wnorm = weights

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, HIDDEN_SIZE), dtype=types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        sq = mb.mul(x=x, y=x, name="rms_square")
        ms = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True, name="rms_mean")
        rs = mb.rsqrt(x=mb.add(x=ms, y=np.float16(RMS_EPS)), name="rms_rsqrt")
        xn = mb.mul(x=x, y=rs, name="rms_scale")
        xn = mb.mul(x=xn, y=wnorm, name="rms_weight")
        gate = mb.linear(x=xn, weight=wg, name="gate_proj")
        up = mb.linear(x=xn, weight=wu, name="up_proj")
        act = mb.silu(x=gate, name="silu")
        h = mb.mul(x=act, y=up, name="gate_up_mul")
        down = mb.linear(x=h, weight=wd, name="down_proj")
        return mb.add(x=down, y=x, name="residual_add")

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


def reference(x, weights, intermediate):
    """The same block in numpy f32, from the same f16 weights."""
    wg, wu, wd, wnorm = (w.astype(np.float32) for w in weights)
    xf = x.astype(np.float32).reshape(-1)
    rms = np.sqrt(np.mean(xf * xf) + RMS_EPS)
    xn = (xf / rms) * wnorm.reshape(-1)
    gate = wg @ xn
    up = wu @ xn
    act = gate / (1.0 + np.exp(-gate))
    return wd @ (act * up) + xf


def per_op_placement(mlmodelc):
    plan = MLComputePlan.load_from_path(path=mlmodelc, compute_units=ct.ComputeUnit.CPU_AND_NE)
    main = plan.model_structure.program.functions["main"]
    rows = []
    for op in main.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        rows.append(
            {
                "op": op.operator_name,
                "name": getattr(op, "output_name", None),
                "preferred": ane1.device_short(type(usage.preferred_compute_device).__name__),
                "supported": [
                    ane1.device_short(type(d).__name__)
                    for d in usage.supported_compute_devices
                ],
            }
        )
    return rows


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "unlabelled"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    work_dir = os.environ.get("ANE4_WORK_DIR", "/tmp/ane4-work")
    os.makedirs(work_dir, exist_ok=True)

    x = prng((1, HIDDEN_SIZE), seed=7)
    report = {"experiment": "ANE-4 step 1 (FFN block placement)", "session": session,
              "coremltools": ct.__version__, "variants": []}

    for tag, intermediate, note in VARIANTS:
        print(f"\n=== {tag} — intermediate {intermediate}, {note} ===")
        weights = (
            prng((intermediate, HIDDEN_SIZE), seed=1),
            prng((intermediate, HIDDEN_SIZE), seed=2),
            prng((HIDDEN_SIZE, intermediate), seed=3),
            prng((1, HIDDEN_SIZE), seed=4),
        )
        mlc = build_block(intermediate, work_dir, tag, weights)

        ops = per_op_placement(mlc)
        print(f"{'op':>26}{'device':>8}{'supported':>14}")
        for r in ops:
            print(f"{r['op']:>26}{r['preferred']:>8}{'+'.join(r['supported']):>14}")
        on_ane = sum(1 for r in ops if r["preferred"] == "ANE")
        print(f"  -> {on_ane} of {len(ops)} placeable ops on ANE")

        model = ct.models.CompiledMLModel(mlc, compute_units=ct.ComputeUnit.CPU_AND_NE)
        for _ in range(WARMUP_ITERS):
            model.predict({"x": x})
        samples = []
        out = None
        for _ in range(MEASURED_ITERS):
            t = time.perf_counter()
            out = model.predict({"x": x})
            samples.append((time.perf_counter() - t) * 1e3)
        samples.sort()
        block_min, block_p50 = samples[0], samples[len(samples) // 2]

        got = np.asarray(list(out.values())[0], dtype=np.float64).reshape(-1)
        ref = reference(x, weights, intermediate).astype(np.float64)
        rel_rms = float(
            np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref**2))
        )

        # The isolated-op expectation, for contrast: three projections at
        # the ANE-1 equivalent rate would be roughly this many bytes.
        proj_bytes = (2 * intermediate * HIDDEN_SIZE + HIDDEN_SIZE * intermediate) * 2
        print(f"  block warm ms: min {block_min:.3f}  p50 {block_p50:.3f}")
        print(f"  projection bytes in block: {proj_bytes / 1e6:.1f} MB"
              f"  -> equivalent {proj_bytes / (block_min / 1e3) / 1e9:.1f} GB/s")
        print(f"  numerical rel_rms vs f32 reference: {rel_rms:.3e}")

        report["variants"].append(
            {
                "tag": tag,
                "intermediate": intermediate,
                "note": note,
                "ops": ops,
                "ops_on_ane": on_ane,
                "ops_total": len(ops),
                "ms": {"min": block_min, "p50": block_p50},
                "projection_bytes": proj_bytes,
                "equivalent_gbs_min": proj_bytes / (block_min / 1e3) / 1e9,
                "rel_rms": rel_rms,
            }
        )
        shutil.rmtree(mlc, ignore_errors=True)
        shutil.rmtree(mlc.replace(".mlmodelc", ".mlpackage"), ignore_errors=True)
        del weights

    a, b = report["variants"]
    print("\n=== A/B control ===")
    print(f"  A (k=17408, ineligible): {a['ops_on_ane']}/{a['ops_total']} ops on ANE")
    print(f"  B (k=16384, eligible):   {b['ops_on_ane']}/{b['ops_total']} ops on ANE")
    if b["ops_on_ane"] > a["ops_on_ane"]:
        print("  -> B is more ANE-resident: causal evidence that k > 16384 fractures the graph")
    elif a["ops_on_ane"] == b["ops_on_ane"]:
        print("  -> identical residency: the fracture (if any) is NOT down_proj's k")
    else:
        print("  -> A more resident than B: unexpected; read the per-op tables before interpreting")

    if out_path:
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
