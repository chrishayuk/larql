"""Q8_0_T2: every element of every Q8_0 tensor is within the T2 reconstruction bound of its source value.

The measurement is refcheck.e1. T2 is derived from ggml's reference encoder; see docs/verification/contract-protocol.md.
"""

from __future__ import annotations

import hashlib

from refcheck import container as C
from refcheck import e1 as E1

from .. import checkpoint

STATEMENT = "every element of every Q8_0 tensor is within the T2 reconstruction bound of its source value"


def could_not(out: dict, *reasons: str) -> dict:
    out.update(status="could_not_evaluate", reasons=list(reasons))
    return out


def evaluate(out: dict, container: str, source: str, pins=checkpoint.PINS) -> dict:
    try:
        n = checkpoint.verify_source(source, pins)
    except checkpoint.SourceError as e:
        return could_not(out, f"the source is not the checkpoint this verifier pins: {e}")
    model_digest = pins["model.safetensors"][1] if "model.safetensors" in pins else "not pinned"
    out["source"]["identity"] = (f"{n} files match the pinned SHA-256 hashes of {checkpoint.MODEL} at {checkpoint.REVISION}; "
                                 f"model.safetensors {model_digest}")
    try:
        c = C.open_container(container)
    except C.ContainerError as e:
        return could_not(out, f"the validating reader refused the container: {e}")
    try:
        weights = checkpoint.load_weights(source)
    except checkpoint.SourceError as e:
        return could_not(out, str(e))
    rep = E1.e1_report(c, lambda k: weights[k], {k: tuple(v.shape) for k, v in weights.items()})
    tot = rep["total"]
    problems = []
    if not rep["coverage"]["ok"]:
        problems.append(f"the container's tensors do not match the source: {rep['coverage']}")
    if not tot["n_q8_tensors"]:
        problems.append("the container has no Q8_0 tensors, so the contract would hold vacuously")
    outside = sum(v["blocks"] for v in tot["outside_t2_domain"].values())
    if outside:
        problems.append(f"{outside} blocks have a stored scale outside T2's domain, about which T2 says nothing")
    if tot["n_upstream_nonfinite"]:
        problems.append(f"{tot['n_upstream_nonfinite']} source elements are not finite")
    if problems:
        return could_not(out, *problems)

    q8 = {k: t for k, t in rep["tensors"].items() if t["dtype"] == "Q8_0"}
    offending = sorted(((k, t) for k, t in q8.items() if t["n_exceed_t2"]), key=lambda kv: (-kv[1]["n_exceed_t2"], kv[0]))
    out.update(
        status="violated" if tot["n_exceed_t2"] else "satisfied",
        violations=tot["n_exceed_t2"],
        measurements=[
            {"name": "Q8_0 tensors", "value": str(tot["n_q8_tensors"])},
            {"name": "elements over T2", "value": str(tot["n_exceed_t2"])},
            {"name": "blocks over T2", "value": str(tot["n_blocks_exceed_t2"])},
            {"name": "zero-scale blocks", "value": str(tot["zero_scale_blocks"])},
            {"name": "worst error / T2 bound", "value": f"{tot['worst_ratio']:.6g}"},
            {"name": "max |source - decoded|", "value": f"{tot['max_abs_error']:.6g}"},
            {"name": "decode matches gguf-py", "value": str(tot["gguf_all_bit_identical"]).lower()},
        ],
        offending=[{"name": k, "violations": t["n_exceed_t2"],
                    "detail": f"{t['n_blocks_exceed_t2']} blocks, worst error / bound {t['worst_ratio']:.6g}"}
                   for k, t in offending],
        report_sha256=hashlib.sha256(E1.canonical_json(rep).encode()).hexdigest(),
    )
    return out
