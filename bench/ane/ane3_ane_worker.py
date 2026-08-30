"""ANE-3 ANE worker — the Core ML side of the concurrency rung.

Mirrors `ane3_gpu_worker.rs` exactly: same `5120 -> 17408` f16 projection,
same barrier protocol, same epoch-stamped samples. Model build, compile and
warm-up all happen before the readiness signal, so Core ML's setup and
queue behaviour is outside every measured sample.

f16 deliberately, not int8: ANE-0b's frozen denominator is f16, and 2C
showed int8 halves stored bytes while buying only ~1.15x latency — so its
physical traffic is not halved and mixing precisions would make the
contention result harder to interpret, not easier.

Usage:
    python ane3_ane_worker.py <run_dir> <duration_ms>
"""

import json
import os
import sys
import time

import coremltools as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ane1_placement as ane1  # noqa: E402  (frozen ANE-1 instrument)

WARMUP_ITERS = 64
POLL_S = 0.001


def main():
    run_dir, duration_ms = sys.argv[1], float(sys.argv[2])
    role = sys.argv[3] if len(sys.argv) > 3 else "measure"
    work_dir = os.environ.get("ANE3_WORK_DIR", "/tmp/ane3-work")
    os.makedirs(work_dir, exist_ok=True)

    n, k = ane1.INTERMEDIATE_SIZE, ane1.HIDDEN_SIZE
    weight_bytes = n * k * ane1.BYTES_PER_F16
    w = ane1.weights(n, k)
    x = ane1.activation(k)

    # Build/compile and record placement BEFORE the barrier: none of this
    # may land inside a measured sample.
    mlmodelc, _ = ane1.build(n, k, w, work_dir, "ane3_primary")
    ops = ane1.placement(mlmodelc, ct.ComputeUnit.CPU_AND_NE)
    op = ane1.compute_op(ops)
    preferred = ane1.device_short(op["preferred"])
    if preferred != "ANE":
        print(f"ane3_ane_worker: REFUSING — placement is {preferred}, not ANE", file=sys.stderr)
        sys.exit(1)

    model = ct.models.CompiledMLModel(mlmodelc, compute_units=ct.ComputeUnit.CPU_AND_NE)
    for _ in range(WARMUP_ITERS):
        model.predict({"x": x})

    with open(os.path.join(run_dir, "ane.ready"), "w") as fh:
        fh.write("1")
    go = os.path.join(run_dir, "go")

    # ANE-3b: ramp rather than idle while waiting for the barrier, so every
    # condition is entered from the same SoC power state. See the matching
    # comment in ane3_gpu_worker.rs — ANE-3's GPU-alone arm drifted ~10%
    # depending on whether it ran cold or after sustained load.
    ramp_iters = 0
    while not os.path.exists(go):
        model.predict({"x": x})
        ramp_iters += 1

    if role == "ramp":
        with open(os.path.join(run_dir, "ane.ramp.json"), "w") as fh:
            json.dump({"engine": "ane", "role": "ramp", "ramp_iters": ramp_iters}, fh)
        print(f"ane worker: ramp only, {ramp_iters} iters", file=sys.stderr)
        return

    window_start = time.time()
    clock = time.perf_counter()
    starts, ms = [], []
    while (time.perf_counter() - clock) * 1e3 < duration_ms:
        t_epoch = time.time()
        t = time.perf_counter()
        model.predict({"x": x})
        ms.append((time.perf_counter() - t) * 1e3)
        starts.append(t_epoch)
    window_end = time.time()

    doc = {
        "engine": "ane",
        "placement": preferred,
        "n": n,
        "k": k,
        "weight_bytes": weight_bytes,
        "window_start": window_start,
        "window_end": window_end,
        "iters": len(ms),
        "sample_start_epoch": starts,
        "sample_ms": ms,
    }
    with open(os.path.join(run_dir, "ane.json"), "w") as fh:
        json.dump(doc, fh)
    print(f"ane worker: {len(ms)} iters over {window_end - window_start:.3f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
