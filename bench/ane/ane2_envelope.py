"""ANE-2 — the ANE operating envelope.

Three experiments, in this order, because each one's design depends on
the previous answer:

  2A  placement boundary   where does Core ML stop preferring CPU and
                           start preferring ANE? ANE-1 found 512x512
                           preferring CPU and 5120x17408 preferring ANE,
                           so placement is a policy surface, not a
                           capability boolean.
  2B  batch-N              T(N)/T(1) for N = 1, 2, 4, 8. Judged against
                           CPU-7B's concrete numbers (N=2 1.02x,
                           N=4 1.27x, N=8 2.41x), not against vague
                           accelerator expectations.
  2C  int8                 does compression change placement, footprint,
                           or latency? Three separate questions that need
                           not move together.

**The ANE-1 harness is frozen and is imported, not copied.** Every cell
here is built, placed and timed by exactly the code that produced the
banked ANE-1 result.

Usage:
    python ane2_envelope.py 2a <session> [out.json]
    python ane2_envelope.py 2b <session> [out.json]
    python ane2_envelope.py 2c <session> [out.json]
"""

import json
import os
import shutil
import sys

import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ane1_placement as ane1  # noqa: E402  (frozen ANE-1 instrument)

HIDDEN_SIZE = ane1.HIDDEN_SIZE
INTERMEDIATE_SIZE = ane1.INTERMEDIATE_SIZE
BYTES_PER_F16 = ane1.BYTES_PER_F16

# 2A: one dimension fixed at the model's hidden size, the other scaled.
WIDTHS = [512, 1024, 2048, 4096, 8192, 12288, INTERMEDIATE_SIZE]

# 2B: the shape ANE-1 proved, at increasing batch.
BATCHES = [1, 2, 4, 8]

# CPU-7B's banked ratios, for the same-axis comparison. Different engine,
# different representation (Q8[64] x asym-Q8[16], 5120x5120) — carried
# here as a reference point, not as a like-for-like control.
CPU7B_RATIOS = {1: 1.00, 2: 1.02, 4: 1.27, 8: 2.41}


def cell(n, k, w, x, work_dir, tag, keep=False):
    """Build, place and time one shape. Returns a report row."""
    mlc, compile_ms = ane1.build(n, k, w, work_dir, tag)
    ops = ane1.placement(mlc, ct.ComputeUnit.CPU_AND_NE)
    op = ane1.compute_op(ops)
    pref = ane1.device_short(op["preferred"])
    sup = [ane1.device_short(d) for d in op["supported"]]
    ts, out = ane1.time_model(mlc, x, ct.ComputeUnit.CPU_AND_NE)
    weight_bytes = n * k * BYTES_PER_F16
    row = {
        "n": n,
        "k": k,
        "weight_bytes": weight_bytes,
        "weight_mb": weight_bytes / 1e6,
        "preferred": pref,
        "supported": sup,
        "compile_ms": compile_ms,
        "ms": ts,
        "equivalent_gbs_min": weight_bytes / (ts["min"] / 1e3) / 1e9,
        "equivalent_gbs_p50": weight_bytes / (ts["p50"] / 1e3) / 1e9,
    }
    if not keep:
        shutil.rmtree(mlc, ignore_errors=True)
        shutil.rmtree(mlc.replace(".mlmodelc", ".mlpackage"), ignore_errors=True)
    return row, out


def header():
    print(
        f"{'shape':>16}{'MB':>9}{'device':>8}{'supported':>12}"
        f"{'min ms':>10}{'p50 ms':>10}{'equiv GB/s':>12}"
    )


def show(row):
    shape = f"{row['k']}->{row['n']}"
    print(
        f"{shape:>16}{row['weight_mb']:>9.2f}{row['preferred']:>8}"
        f"{'+'.join(row['supported']):>12}"
        f"{row['ms']['min']:>10.3f}{row['ms']['p50']:>10.3f}"
        f"{row['equivalent_gbs_min']:>12.1f}"
    )


def run_2a(work_dir):
    """Placement boundary, both orientations."""
    rows = []
    print("\n-- widening output (k = 5120 fixed) --")
    header()
    for n in WIDTHS:
        w = ane1.weights(n, HIDDEN_SIZE)
        x = ane1.activation(HIDDEN_SIZE)
        row, _ = cell(n, HIDDEN_SIZE, w, x, work_dir, f"a_out_{n}")
        row["orientation"] = "widen_output"
        show(row)
        rows.append(row)
        del w

    print("\n-- widening input (n = 5120 fixed) --")
    header()
    for k in WIDTHS:
        w = ane1.weights(HIDDEN_SIZE, k)
        x = ane1.activation(k)
        row, _ = cell(HIDDEN_SIZE, k, w, x, work_dir, f"a_in_{k}")
        row["orientation"] = "widen_input"
        show(row)
        rows.append(row)
        del w

    boundary(rows)
    return rows


def boundary(rows):
    """State the crossover explicitly rather than leaving it to the eye."""
    print("\nplacement crossover:")
    for orient in ("widen_output", "widen_input"):
        sub = [r for r in rows if r["orientation"] == orient]
        cpu = [r for r in sub if r["preferred"] == "CPU"]
        ane = [r for r in sub if r["preferred"] == "ANE"]
        if not ane:
            print(f"  {orient}: no cell preferred ANE")
        elif not cpu:
            print(f"  {orient}: every cell preferred ANE (boundary below {sub[0]['weight_mb']:.2f} MB)")
        else:
            largest_cpu = max(cpu, key=lambda r: r["weight_bytes"])
            smallest_ane = min(ane, key=lambda r: r["weight_bytes"])
            print(
                f"  {orient}: CPU up to {largest_cpu['k']}->{largest_cpu['n']} "
                f"({largest_cpu['weight_mb']:.2f} MB), "
                f"ANE from {smallest_ane['k']}->{smallest_ane['n']} "
                f"({smallest_ane['weight_mb']:.2f} MB)"
            )


def build_batched(n, k, batch, w, work_dir, tag):
    """Like `ane1.build`, but the input spec carries the batch dimension.

    2B cannot reuse the frozen ANE-1 builder because that builder pins
    `shape=(1, k)` — the batch size is part of the model description, not
    a call-time argument. Placement and timing still come from the frozen
    instrument; only the input spec differs, and N=1 through this builder
    reproduces ANE-1's banked cell as a self-consistency check.
    """
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(batch, k), dtype=types.fp16)],
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
    from coremltools.models.utils import compile_model

    compile_model(pkg, destination_path=mlc)
    return mlc


def run_2b(work_dir):
    """Batch-N on the shape ANE-1 proved."""
    n, k = INTERMEDIATE_SIZE, HIDDEN_SIZE
    w = ane1.weights(n, k)
    rows = []
    print(f"\n-- batch-N on {k}->{n} ({n * k * BYTES_PER_F16 / 1e6:.2f} MB) --")
    print(
        f"{'N':>4}{'device':>8}{'min ms':>10}{'p50 ms':>10}"
        f"{'T(N)/T(1)':>12}{'per-vector':>12}{'CPU-7B':>9}"
    )
    base = None
    for bsz in BATCHES:
        x = np.repeat(ane1.activation(k), bsz, axis=0)
        mlc = build_batched(n, k, bsz, w, work_dir, f"b_{bsz}")
        ops = ane1.placement(mlc, ct.ComputeUnit.CPU_AND_NE)
        op = ane1.compute_op(ops)
        ts, _ = ane1.time_model(mlc, x, ct.ComputeUnit.CPU_AND_NE)
        weight_bytes = n * k * BYTES_PER_F16
        row = {
            "n": n,
            "k": k,
            "weight_bytes": weight_bytes,
            "weight_mb": weight_bytes / 1e6,
            "preferred": ane1.device_short(op["preferred"]),
            "supported": [ane1.device_short(d) for d in op["supported"]],
            "ms": ts,
            "equivalent_gbs_min": weight_bytes / (ts["min"] / 1e3) / 1e9,
        }
        shutil.rmtree(mlc, ignore_errors=True)
        shutil.rmtree(mlc.replace(".mlmodelc", ".mlpackage"), ignore_errors=True)
        row["batch"] = bsz
        if base is None:
            base = row["ms"]["min"]
        ratio = row["ms"]["min"] / base
        row["ratio_vs_n1"] = ratio
        row["per_vector"] = ratio / bsz
        row["cpu7b_reference"] = CPU7B_RATIOS[bsz]
        print(
            f"{bsz:>4}{row['preferred']:>8}{row['ms']['min']:>10.3f}"
            f"{row['ms']['p50']:>10.3f}{ratio:>12.2f}{ratio / bsz:>12.2f}"
            f"{CPU7B_RATIOS[bsz]:>9.2f}"
        )
        rows.append(row)
    print("\nCPU-7B column is a REFERENCE POINT, not a control: different engine,")
    print("different representation (Q8[64] x asym-Q8[16], 5120x5120).")
    return rows


def run_2a_bisect(work_dir):
    """2A found `17408->5120` reporting `supported: [CPU]` — ANE absent
    entirely, not merely unpreferred. That is a hard capability limit on
    the reduction axis, and `down_proj` (17408 -> 5120) is a real
    projection in the model, so where the limit sits decides whether the
    FFN can be placed on ANE at all.

    Two questions: where on k does ANE support disappear, and does the
    output width n move it?
    """
    rows = []
    print("\n-- reduction axis: n = 5120 fixed, k bisected --")
    header()
    for k in [12288, 13312, 14336, 15360, 16384, 16512, INTERMEDIATE_SIZE]:
        w = ane1.weights(HIDDEN_SIZE, k)
        x = ane1.activation(k)
        row, _ = cell(HIDDEN_SIZE, k, w, x, work_dir, f"bis_k{k}")
        row["probe"] = "reduction_axis"
        show(row)
        rows.append(row)
        del w

    print("\n-- does output width move the limit? k = 17408 fixed --")
    header()
    for n in [512, 2048, HIDDEN_SIZE]:
        w = ane1.weights(n, INTERMEDIATE_SIZE)
        x = ane1.activation(INTERMEDIATE_SIZE)
        row, _ = cell(n, INTERMEDIATE_SIZE, w, x, work_dir, f"bis_n{n}")
        row["probe"] = "output_width_at_k17408"
        show(row)
        rows.append(row)
        del w

    supported = [r for r in rows if "ANE" in r["supported"]]
    unsupported = [r for r in rows if "ANE" not in r["supported"]]
    print("\nANE capability limit on the reduction axis:")
    if supported:
        print(f"  largest k with ANE support:  {max(r['k'] for r in supported)}")
    if unsupported:
        print(f"  smallest k without support:  {min(r['k'] for r in unsupported)}")
    return rows


def run_2a_small(work_dir):
    """The smallest useful unit of ANE work — the regime an ANE-4 drafter
    would live in.

    API bring-up saw `512x512` prefer CPU while `5120->512` (same output
    width, 10x the bytes) preferred ANE, so the small-end boundary is not
    a simple byte threshold. Square shapes isolate it.
    """
    rows = []
    print("\n-- square shapes: the drafter regime --")
    header()
    for d in [256, 512, 768, 1024, 1536, 2048, 3072]:
        w = ane1.weights(d, d)
        x = ane1.activation(d)
        row, _ = cell(d, d, w, x, work_dir, f"sq_{d}")
        row["probe"] = "square"
        show(row)
        rows.append(row)

    ane_cells = [r for r in rows if r["preferred"] == "ANE"]
    cpu_cells = [r for r in rows if r["preferred"] == "CPU"]
    print("\nsmallest useful unit of ANE work (square):")
    if ane_cells:
        smallest = min(ane_cells, key=lambda r: r["weight_bytes"])
        print(f"  smallest square preferring ANE: {smallest['k']}x{smallest['n']} "
              f"({smallest['weight_mb']:.2f} MB, {smallest['ms']['min']:.3f} ms)")
    if cpu_cells:
        largest = max(cpu_cells, key=lambda r: r["weight_bytes"])
        print(f"  largest square preferring CPU:  {largest['k']}x{largest['n']} "
              f"({largest['weight_mb']:.2f} MB, {largest['ms']['min']:.3f} ms)")
    return rows


def run_2a_lower(work_dir):
    """Test the hypothesis that emerged from 2A + the square sweep.

    Observed so far: k=3072 (square) CPU, k=4096 (n=5120) CPU, k=5120 CPU
    at n=512? no — k=5120 preferred ANE at BOTH n=512 and n=17408, while
    k=4096 at n=5120 preferred CPU. Bytes do not order these: 5120->512
    is 5.24 MB on ANE, 4096->5120 is 41.94 MB on CPU.

    Hypothesis: **the reduction depth k selects the device**, with ANE
    preferred for roughly 4096 < k <= 16384 and unsupported above.
    Falsifier: hold k on either side of the edge and vary n widely; if k
    is the discriminator, n must not flip the device.
    """
    rows = []
    print("\n-- lower edge on the reduction axis (n = 5120 fixed) --")
    header()
    for k in [4096, 4352, 4608, 4864, 5120]:
        w = ane1.weights(HIDDEN_SIZE, k)
        x = ane1.activation(k)
        row, _ = cell(HIDDEN_SIZE, k, w, x, work_dir, f"lo_k{k}")
        row["probe"] = "lower_edge"
        show(row)
        rows.append(row)
        del w

    print("\n-- falsifier: does n flip the device at fixed k? --")
    header()
    for k in [4096, 5120]:
        for n in [512, 2048, 8192]:
            w = ane1.weights(n, k)
            x = ane1.activation(k)
            row, _ = cell(n, k, w, x, work_dir, f"fx_{k}_{n}")
            row["probe"] = "n_independence"
            show(row)
            rows.append(row)
            del w

    fx = [r for r in rows if r.get("probe") == "n_independence"]
    for k in [4096, 5120]:
        devs = {r["preferred"] for r in fx if r["k"] == k}
        verdict = "n-INDEPENDENT" if len(devs) == 1 else "n-DEPENDENT"
        print(f"  k={k}: devices across n = {sorted(devs)} -> {verdict}")
    return rows


def prng_weights(n, k, seed=20260825):
    """Pseudorandom, incompressible weight material.

    2A and 2B used the periodic `i % 977` generator for comparability with
    ANE-0b. That generator must NOT be used here: it compresses, and a
    compression experiment run on it would flatter itself.
    """
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, k), dtype=np.float32) * 0.02).astype(np.float16)


def weight_bin_bytes(mlmodelc):
    p = os.path.join(mlmodelc, "weights", "weight.bin")
    return os.path.getsize(p) if os.path.exists(p) else -1


def build_pair(n, k, w, work_dir, tag):
    """The same pseudorandom weights compiled twice: f16 and int8.

    Returns (f16_mlmodelc, int8_mlmodelc). Building both from one MLModel
    is what makes a footprint difference attributable to the compression
    rather than to different weight material.
    """
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from coremltools.models.utils import compile_model
    import coremltools.optimize.coreml as cto

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, k), dtype=types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def prog(x):
        return mb.linear(x=x, weight=w, name="proj")

    base = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    quant = cto.linear_quantize_weights(
        base,
        config=cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8", granularity="per_channel"
            )
        ),
    )

    out = []
    for label, model in (("f16", base), ("int8", quant)):
        pkg = os.path.join(work_dir, f"{tag}_{label}.mlpackage")
        mlc = os.path.join(work_dir, f"{tag}_{label}.mlmodelc")
        for stale in (pkg, mlc):
            if os.path.exists(stale):
                shutil.rmtree(stale)
        model.save(pkg)
        compile_model(pkg, destination_path=mlc)
        out.append(mlc)
    return out


def run_2c(work_dir):
    """Does int8 move the placement boundaries?

    Primary question, ahead of any speed result. Three facts are reported
    independently and none is inferred from another: compiled artifact
    size, preferred/supported device, warm latency.
    """
    ladders = [
        ("lower boundary", [4096, 4864, 4992, HIDDEN_SIZE, 5248]),
        ("upper boundary", [16384, 16512, INTERMEDIATE_SIZE]),
    ]
    rows = []
    for title, ks in ladders:
        print(f"\n-- {title} (n = {HIDDEN_SIZE}, pseudorandom weights) --")
        print(
            f"{'k':>7}{'dtype':>7}{'weight.bin MB':>15}{'device':>8}"
            f"{'supported':>12}{'min ms':>10}{'p50 ms':>10}{'equiv GB/s':>12}"
        )
        for k in ks:
            w = prng_weights(HIDDEN_SIZE, k)
            x = ane1.activation(k)
            f16_mlc, int8_mlc = build_pair(HIDDEN_SIZE, k, w, work_dir, f"c_{k}")
            for label, mlc in (("f16", f16_mlc), ("int8", int8_mlc)):
                ops = ane1.placement(mlc, ct.ComputeUnit.CPU_AND_NE)
                op = ane1.compute_op(ops)
                ts, _ = ane1.time_model(mlc, x, ct.ComputeUnit.CPU_AND_NE)
                stored = weight_bin_bytes(mlc)
                row = {
                    "k": k,
                    "n": HIDDEN_SIZE,
                    "dtype": label,
                    "stored_bytes": stored,
                    "preferred": ane1.device_short(op["preferred"]),
                    "supported": [ane1.device_short(d) for d in op["supported"]],
                    "ms": ts,
                    # Equivalent rate over the bytes ACTUALLY stored, so the
                    # int8 arm is not credited with traffic it does not have.
                    "equivalent_gbs_min": stored / (ts["min"] / 1e3) / 1e9,
                    "ladder": title,
                }
                print(
                    f"{k:>7}{label:>7}{stored / 1e6:>15.2f}{row['preferred']:>8}"
                    f"{'+'.join(row['supported']):>12}"
                    f"{ts['min']:>10.3f}{ts['p50']:>10.3f}"
                    f"{row['equivalent_gbs_min']:>12.1f}"
                )
                rows.append(row)
                shutil.rmtree(mlc, ignore_errors=True)
                shutil.rmtree(mlc.replace(".mlmodelc", ".mlpackage"), ignore_errors=True)
            del w

    print("\nboundary comparison (f16 vs int8 placement):")
    moved = False
    for k in sorted({r["k"] for r in rows}):
        f16 = next(r for r in rows if r["k"] == k and r["dtype"] == "f16")
        i8 = next(r for r in rows if r["k"] == k and r["dtype"] == "int8")
        same = f16["preferred"] == i8["preferred"] and f16["supported"] == i8["supported"]
        if not same:
            moved = True
        print(
            f"  k={k:>6}  f16 {f16['preferred']}/{'+'.join(f16['supported'])}"
            f"   int8 {i8['preferred']}/{'+'.join(i8['supported'])}"
            f"   {'SAME' if same else 'MOVED'}"
        )
    print(
        "\nverdict: int8 "
        + ("MOVES the placement boundary" if moved else "does NOT move the placement boundary")
    )
    return rows


def run_2c_narrow(work_dir):
    """2C found int8 admitting k=4096 where f16 preferred CPU. How far
    down does it go?

    This directly sizes a narrow ANE drafter: under f16 the drafter had to
    carry the target's 5120 width to be placed at all, and if int8 admits
    k=1024 or 2048 that constraint largely dissolves.
    """
    rows = []
    print(f"\n-- how far down does int8 admit? (n = {HIDDEN_SIZE}) --")
    print(
        f"{'k':>7}{'dtype':>7}{'weight.bin MB':>15}{'device':>8}"
        f"{'supported':>12}{'min ms':>10}{'p50 ms':>10}"
    )
    for k in [512, 1024, 2048, 3072, 4096]:
        w = prng_weights(HIDDEN_SIZE, k)
        x = ane1.activation(k)
        f16_mlc, int8_mlc = build_pair(HIDDEN_SIZE, k, w, work_dir, f"cn_{k}")
        for label, mlc in (("f16", f16_mlc), ("int8", int8_mlc)):
            ops = ane1.placement(mlc, ct.ComputeUnit.CPU_AND_NE)
            op = ane1.compute_op(ops)
            ts, _ = ane1.time_model(mlc, x, ct.ComputeUnit.CPU_AND_NE)
            stored = weight_bin_bytes(mlc)
            row = {
                "k": k,
                "n": HIDDEN_SIZE,
                "dtype": label,
                "stored_bytes": stored,
                "preferred": ane1.device_short(op["preferred"]),
                "supported": [ane1.device_short(d) for d in op["supported"]],
                "ms": ts,
                "ladder": "narrow",
            }
            print(
                f"{k:>7}{label:>7}{stored / 1e6:>15.2f}{row['preferred']:>8}"
                f"{'+'.join(row['supported']):>12}{ts['min']:>10.3f}{ts['p50']:>10.3f}"
            )
            rows.append(row)
            shutil.rmtree(mlc, ignore_errors=True)
            shutil.rmtree(mlc.replace(".mlmodelc", ".mlpackage"), ignore_errors=True)
        del w

    i8_ane = [r for r in rows if r["dtype"] == "int8" and r["preferred"] == "ANE"]
    if i8_ane:
        print(f"\nsmallest k placed on ANE under int8: {min(r['k'] for r in i8_ane)}")
    else:
        print("\nno narrow int8 cell was placed on ANE")
    return rows


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    mode, session = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    work_dir = os.environ.get("ANE2_WORK_DIR", "/tmp/ane2-work")
    os.makedirs(work_dir, exist_ok=True)

    print(f"ANE-2{mode[-1].upper()} — session '{session}', coremltools {ct.__version__}")
    report = {
        "experiment": f"ANE-2{mode[-1].upper()}",
        "session": session,
        "coremltools": ct.__version__,
        "iters": {"warmup": ane1.WARMUP_ITERS, "measured": ane1.MEASURED_ITERS},
        "weights": "periodic i%977 (same generator as ANE-0b/ANE-1)",
    }

    if mode == "2a":
        report["rows"] = run_2a(work_dir)
    elif mode == "2c-narrow":
        report["rows"] = run_2c_narrow(work_dir)
        report["weights"] = "pseudorandom normal, seed 20260825 (incompressible)"
    elif mode == "2c":
        report["rows"] = run_2c(work_dir)
        report["weights"] = "pseudorandom normal, seed 20260825 (incompressible)"
    elif mode == "2a-lower":
        report["rows"] = run_2a_lower(work_dir)
    elif mode == "2a-small":
        report["rows"] = run_2a_small(work_dir)
    elif mode == "2a-bisect":
        report["rows"] = run_2a_bisect(work_dir)
    elif mode == "2b":
        report["rows"] = run_2b(work_dir)
    else:
        print(f"unknown mode {mode!r}")
        sys.exit(2)

    if out_path:
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
