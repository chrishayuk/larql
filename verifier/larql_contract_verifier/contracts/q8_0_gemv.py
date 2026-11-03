"""Q8_0_GEMV_SEQ and Q8_0_GEMV_B1: one recorded call of the CPU Q8_0 matrix-vector kernel, judged from a witness.

The measurement is refcheck.gemv. This module only establishes what the measurement is about: it checks the witness's
digests, that it names this operator once, and that its bytes are the container's. If anything doesn't line up, the answer
is could_not_evaluate, because there's nothing to judge.

A witness is the engine's own report. A result is about the recorded output, given the recorded input and the container's
bytes: whether it matches a declared program, or lies within a bound. It doesn't say the production kernel produced that
output. And a violation doesn't say why: a fault in the recording looks the same as a fault in the kernel.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat

import numpy as np

from refcheck import container as C
from refcheck import gemv

WITNESS_SCHEMA = "larql-gemv-witness/1"
KERNEL = "cpu.FusedKQuant.q8_0_gemv"
ENCODING = "Q8_0"
# Far more than any real manifest, and a stream's size comes from its shape: nothing here reads without a limit.
MANIFEST_LIMIT = 1 << 26
# What malformed input can raise from inside the validating reader or from the operating system. Everything that doesn't
# line up is could_not_evaluate. The copied reader can't be changed here, so what it lets through is caught here.
BAD_INPUT = (C.ContainerError, OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError)

STATEMENTS = {
    "Q8_0_GEMV_SEQ": "the recorded output of this call equals the declared sequential binary32 program, bit for bit. "
    "This is conformance to a declared program, not accuracy",
    "Q8_0_GEMV_B1": "the recorded output of this call is within |y - y*| <= gamma_k*S + k*2^-150*(1 + gamma_k) of exact "
    "arithmetic. This is an experimental, measured bound check, not a machine-checked proof",
}
CHECKS = {"Q8_0_GEMV_SEQ": gemv.check_seq, "Q8_0_GEMV_B1": gemv.check_b1}


def could_not(out: dict, *reasons: str) -> dict:
    out.update(status="could_not_evaluate", reasons=list(reasons))
    return out


class WitnessError(Exception):
    pass


def _read_regular(path: str, limit: int, what: str) -> bytes:
    """A regular file's bytes, up to a limit. A symlink, FIFO or device is refused without reading it."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, ValueError) as e:
        raise WitnessError(f"{what} can't be read: {e}") from e
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise WitnessError(f"{what} isn't a regular file")
        return f.read(limit + 1)


def _stream(directory: str, entry, size: int, what: str) -> bytes:
    """A stream's bytes, after checking its name, length and digest."""
    if not isinstance(entry, dict):
        raise WitnessError(f"{what}: not an object")
    name = entry.get("file")
    if not isinstance(name, str) or not name or name != os.path.basename(name) or name in (".", ".."):
        raise WitnessError(f"{what}: {name!r} is not a file name in the witness's directory")
    if type(entry.get("bytes")) is not int or entry["bytes"] != size:
        raise WitnessError(f"{what}: the witness declares {entry.get('bytes')!r} bytes, and this shape needs {size}")
    data = _read_regular(os.path.join(directory, name), size, what)
    if len(data) != size:
        raise WitnessError(f"{what}: {len(data)} bytes on disk or more, and this shape needs {size}")
    if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
        raise WitnessError(f"{what}: the digest doesn't match")
    return data


def load_record(path: str, operator: str):
    """The manifest's digest and fields, the one record for this operator, and its streams."""
    raw = _read_regular(path, MANIFEST_LIMIT, "the witness")
    if len(raw) > MANIFEST_LIMIT:
        raise WitnessError("the witness is larger than any real manifest")
    try:
        # Strict: a duplicate key would let a reader that keeps the first see a different witness under the same digest.
        m = C._strict_json(raw, "the witness")
    except (C.ContainerError, ValueError, RecursionError) as e:
        raise WitnessError(f"the witness isn't valid JSON: {e}") from e
    if not isinstance(m, dict):
        raise WitnessError("the witness isn't a JSON object")
    required = {"schema": WITNESS_SCHEMA, "kernel": KERNEL, "activation": "f32", "execution_arm": "Direct",
                "representation": ENCODING}
    for key, want in required.items():
        if m.get(key) != want:
            raise WitnessError(f"the witness's {key} is {m.get(key)!r}, and this contract needs {want!r}")
    calls = m.get("calls_observed")
    if type(calls) is not int or calls <= 0:
        raise WitnessError(f"the witness's calls_observed is {calls!r}")
    records = m.get("records")
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise WitnessError("the witness has no list of records")
    found = [r for r in records if r.get("tensor") == operator]
    if len(found) != 1:
        raise WitnessError(f"the witness holds {len(found)} records for {operator!r}, and this needs exactly 1")
    r = found[0]
    n, k = r.get("rows"), r.get("columns")
    if r.get("encoding") != ENCODING:
        raise WitnessError(f"the record's encoding is {r.get('encoding')!r}")
    if type(r.get("call")) is not int or r["call"] < 0:
        raise WitnessError(f"the record's call is {r.get('call')!r}")
    if type(n) is not int or type(k) is not int or n <= 0 or k <= 0 or k % 32:
        raise WitnessError(f"the record's shape is rows={n!r} columns={k!r}")
    if "object" in r and not isinstance(r["object"], str):
        raise WitnessError(f"the record's object is {r['object']!r}")
    directory = os.path.dirname(os.path.abspath(path))
    blocks = _stream(directory, r.get("blocks"), n * (k // 32) * 34, "blocks")
    x = _stream(directory, r.get("x"), k * 4, "x")
    y = _stream(directory, r.get("y"), n * 4, "y")
    return hashlib.sha256(raw).hexdigest(), m, r, n, k, blocks, x, y


def container_tensor(container: str, operator: str, n: int, k: int, obj=None):
    """The container's object and bytes for this operator, refusing any mismatch in identity, encoding or shape.

    A tensor name is only unique within its object. A witness that names the object is held to it, and one that doesn't
    still needs the name to be unique in the container. A container the reader warns about contradicts itself, so it's refused.
    """
    c = C.open_container(container)
    if c.warnings:
        raise WitnessError(f"the validating reader warns about this container: {'; '.join(c.warnings)}")
    refs = [t for t in c.tensors() if t.name == operator and (obj is None or t.object_id == obj)]
    if len(refs) != 1:
        raise WitnessError(f"the container holds {len(refs)} tensors named {operator!r}, and this needs exactly 1")
    t = refs[0]
    if t.dtype != ENCODING:
        raise WitnessError(f"the container's {operator!r} is {t.dtype}, not {ENCODING}")
    if tuple(t.shape) != (n, k):
        raise WitnessError(f"the container's {operator!r} has shape {tuple(t.shape)}, and the witness says ({n}, {k})")
    return t.object_id, c.read_bytes(t)


def evaluate(out: dict, contract: str, operator: str, witness: str, container: str) -> dict:
    try:
        digest, m, r, n, k, blocks, x, y = load_record(witness, operator)
    except WitnessError as e:
        return could_not(out, str(e))
    except BAD_INPUT as e:
        return could_not(out, f"the witness can't be read: {type(e).__name__}: {e}")
    out["witness"].update(sha256=digest, reported={key: m.get(key) for key in (
        "kernel", "execution_arm", "component", "container", "build_commit", "binary_sha256", "calls_observed")})
    recorded_index = m.get("container_index_sha256")
    if recorded_index is not None:
        try:
            index_digest = hashlib.sha256(_read_regular(os.path.join(container, "index.json"), MANIFEST_LIMIT, "index.json")).hexdigest()
        except (WitnessError, *BAD_INPUT) as e:
            return could_not(out, f"the witness can't be tied to this container: {e}")
        if index_digest != recorded_index:
            return could_not(out, "the witness's run recorded a different index.json for its container")
    try:
        obj, stored = container_tensor(container, operator, n, k, r.get("object"))
    except WitnessError as e:
        return could_not(out, f"the witness can't be tied to this container: {e}")
    except BAD_INPUT as e:
        return could_not(out, f"the witness can't be tied to this container: {type(e).__name__}: {e}")
    if stored != blocks:
        return could_not(out, f"the witness's bytes for {operator!r} aren't the container's")
    result = CHECKS[contract](blocks, np.frombuffer(x, dtype="<u4"), np.frombuffer(y, dtype="<u4"), n, k)
    if result.status == gemv.COULD_NOT_EVALUATE:
        return could_not(out, *result.reasons)
    out.update(status=result.status, violations=result.rows_violating)
    out["measurements"] = [{"name": "object", "value": obj},
                           {"name": "rows checked", "value": str(result.rows_checked)},
                           {"name": "rows violating", "value": str(result.rows_violating)}]
    if result.worst_ratio is not None:
        out["measurements"].append({"name": "worst error / bound", "value": result.worst_ratio})
        out["measurements"].append({"name": "worst row", "value": str(result.worst_ratio_row)})
    if result.reasons:
        out["measurements"].append({"name": "found", "value": "; ".join(result.reasons)})
    if result.rows_violating:
        out["offending"] = [{"name": operator, "violations": result.rows_violating,
                             "detail": f"first row {result.first_violating_row}"}]
    report = {"contract": contract, "operator": operator, "object": obj, "witness_sha256": digest,
              "container_sha256": out["container"]["sha256"], "call": r["call"],
              "blocks": hashlib.sha256(blocks).hexdigest(), "x": hashlib.sha256(x).hexdigest(),
              "y": hashlib.sha256(y).hexdigest(), "status": result.status, "rows_checked": result.rows_checked,
              "rows_violating": result.rows_violating, "first_violating_row": result.first_violating_row,
              "worst_ratio": result.worst_ratio, "worst_ratio_row": result.worst_ratio_row,
              "reasons": list(result.reasons)}
    out["report_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    return out
