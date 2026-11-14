"""`EXECUTION_RESIDENCY_V1`: is what a run made resident for each matrix operand the declared derivation of the
container's stored weights, and was it the representation the run's policy and host allow?

Written from `docs/verification/execution-residency-v1.md`. The engine reports a residency record
(`larql-residency-record/1`): per operand, the digest of the stored bytes it read, the representation it left
resident, the rule that takes one to the other and the digest of the resident bytes. This contract re-derives every
resident representation from the CONTAINER'S bytes by its own implementation of the named rule, and re-decides every
choice from the recorded policy and cache size. Nothing the engine wrote is used as an expected value.

Two verdicts, kept apart:

* **derivation**: every operand's stored digest is the container's tensor, and its resident digest is the named rule
  applied to those bytes. Checked for the three rules of §3: the loader's Q8, exact widening to binary32, and stored
  bytes kept.
* **selection**: every operand's representation is the one §4 gives for its size, the recorded cache size, the
  representation source and the caps.

`status` is `satisfied` only when both are, for every operand, and the record is complete (§5). An operand under a
rule this contract does not implement, a policy it does not specify, or a record that omits an operand is never
folded into a pass.

**Not established, and said so in the answer:** that the record is an honest account of what the process held
(it is the engine's own report), that any kernel consumed the resident bytes or computed correctly with them, and
that the recorded cache size is the host's.
"""

from __future__ import annotations

import hashlib

import numpy as np

from refcheck import container as C

from . import vlp_numerics as V
from .weight_patches_v4 import CannotEvaluate, Store, container_digest, file_digests, strict_json

CONTRACT = "EXECUTION_RESIDENCY_V1"
STATEMENT = (
    "every matrix operand the record names was made resident as the declared derivation of the container's stored "
    "bytes (derivation), in the representation its size, the recorded cache size, the representation source and the "
    "caps select (selection), and the record omits no matrix of an object it covers"
)
#: §2: a stored dtype the record may honestly give that this contract does not derive from.
STORED_PACK = C.Q8_0
RECORD_SCHEMA = "larql-residency-record/1"
DERIVATION = "derivation"
SELECTION = "selection"

RULE_Q8_LOAD = "q8-load/1"
RULE_WIDEN_F32 = "widen-f32/1"
RULE_STORED_BYTES = "stored-bytes"
RULE_STORED_BYTES_MAPPED = "stored-bytes-mapped"
#: §3: the digest layout each rule's resident digest is taken in.
DIGEST_LAYOUT = {
    RULE_Q8_LOAD: "codes-i8-row-major||scales-f32le-row-then-block",
    RULE_WIDEN_F32: "f32le-row-major",
    RULE_STORED_BYTES: "stored-bytes",
}
#: §3.1: the loader's Q8.
Q8_BLOCK = 64
Q8_CODE_LIMIT = 127
#: §4: bytes per element of the two images the size policy compares with the cache.
F32_BYTES = 4
BF16_BYTES = 2
#: §4: what each rule's resident format is called in a record.
FORMAT_OF = {RULE_Q8_LOAD: "Q8", RULE_WIDEN_F32: "F32", RULE_STORED_BYTES: "Bf16", RULE_STORED_BYTES_MAPPED: "Bf16"}
SOURCES = ("stored", "auto", "transient")
EXACT_CAP = "bf16"
OPERAND_FIELDS = {"order", "object", "tensor", "shape", "stored_dtype", "stored_sha256", "representation_source",
                  "requested_format", "resident_format", "manufactured", "rule", "resident_sha256",
                  "resident_digest_layout"}
RECORD_FIELDS = {"schema", "container", "container_index_sha256", "selection_inputs", "rules", "operands"}
INPUT_FIELDS = {"l2_bytes", "max_format", "arithmetic", "kquant_exec"}
INDEX_FILE = "index.json"
OFFENDING_LIMIT = 10
SHA_HEX = 64
BAD_INPUT = (OSError, ValueError, KeyError, TypeError, IndexError, C.ContainerError, CannotEvaluate)
NOT_ESTABLISHED = (
    "that the record is an honest account of what the process held (it is the engine's own report)",
    "that any kernel consumed the resident bytes, or computed correctly with them",
    "that the recorded cache size is the host's",
    "that the record names every object the run loaded: an object with no operand in the record is listed, not judged",
)


class Half:
    """One verdict: violations, reasons it could not be evaluated, and whether anything was judged."""

    def __init__(self, name: str):
        self.name = name
        self.offending: list[dict] = []
        self.reasons: list[str] = []
        self.judged = 0

    def violate(self, where: str, detail: str) -> None:
        self.offending.append({"name": where, "violations": 1, "detail": detail, "half": self.name})

    def cannot(self, reason: str) -> None:
        self.reasons.append(reason)

    def status(self) -> str:
        if self.offending:
            return "violated"
        if self.reasons or not self.judged:
            return "could_not_evaluate"
        return "satisfied"

    def report(self) -> dict:
        return {"status": self.status(), "violations": len(self.offending), "offending": self.offending,
                "reasons": list(self.reasons), "judged": self.judged}


# ── the derivation rules (§3) ─────────────────────────────────────────────────


def widen_bf16(raw: bytes) -> np.ndarray:
    """Stored BF16 as binary32: each little-endian 16-bit pattern is the top half of the binary32. Exact."""
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def sha256(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def q8_load(values: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """§3.1: `(codes [rows, cols] int8, scales [rows, blocks] float32)` of a row-major binary32 matrix. Blocks of
    64 along a row, the last one short; per block, `scale = peak / 127` in binary32 (1 for an all-zero block) and
    `code = round_half_away(w / scale)`, one binary32 division per element, clamped to ±127."""
    w = np.ascontiguousarray(values, dtype=np.float32).reshape(rows, cols)
    blocks = -(-cols // Q8_BLOCK)
    codes = np.empty((rows, cols), dtype=np.int8)
    scales = np.empty((rows, blocks), dtype=np.float32)
    for b in range(blocks):
        block = w[:, b * Q8_BLOCK:(b + 1) * Q8_BLOCK]
        peak = np.max(np.abs(block), axis=1)
        scale = np.where(peak > 0, peak / np.float32(Q8_CODE_LIMIT), np.float32(1)).astype(np.float32)
        quotient = (block / scale[:, None]).astype(np.float32)
        whole = np.trunc(quotient)
        # Halves away from zero, on the exact fraction: `quotient - trunc(quotient)` is exact in binary32.
        rounded = whole + np.where(np.abs(quotient - whole) >= np.float32(0.5), np.copysign(np.float32(1), quotient),
                                   np.float32(0))
        codes[:, b * Q8_BLOCK:(b + 1) * Q8_BLOCK] = np.clip(rounded, -Q8_CODE_LIMIT, Q8_CODE_LIMIT).astype(np.int8)
        scales[:, b] = scale
    return codes, scales


def resident_digest(rule: str, raw: bytes, shape: tuple[int, int]) -> str:
    """The digest of the resident bytes `rule` gives for a stored BF16 matrix, in the rule's layout (§3)."""
    if rule == RULE_STORED_BYTES:
        return sha256(raw)
    values = widen_bf16(raw)
    if rule == RULE_WIDEN_F32:
        return sha256(values.astype("<f4").tobytes())
    if rule == RULE_Q8_LOAD:
        codes, scales = q8_load(values, *shape)
        return sha256(codes.tobytes(), scales.astype("<f4").tobytes())
    raise KeyError(rule)


# ── the selection rule (§4) ──────────────────────────────────────────────────


def selected_rules(elements: int, source: str, l2_bytes: int, max_format) -> frozenset[str]:
    """§4: the rules a stored BF16 matrix of `elements` may be resident under. Defined for the default arithmetic
    arm and the `stored` and `auto` sources."""
    # The cap is read as the engine reads its setting: surrounding white space is not part of it, case is.
    cap = max_format.strip() if isinstance(max_format, str) else None
    if elements * F32_BYTES < l2_bytes:
        return frozenset({RULE_WIDEN_F32})
    if elements * BF16_BYTES >= l2_bytes and source == "auto" and cap != EXACT_CAP:
        return frozenset({RULE_Q8_LOAD})
    return frozenset({RULE_STORED_BYTES, RULE_STORED_BYTES_MAPPED})


# ── the judgement ────────────────────────────────────────────────────────────


def is_sha(v) -> bool:
    return isinstance(v, str) and len(v) == SHA_HEX and all(c in "0123456789abcdef" for c in v)


def is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


class Judge:
    def __init__(self, record_path: str, container: str):
        self.record_path, self.root = record_path, container
        self.D, self.S = Half(DERIVATION), Half(SELECTION)
        self.by_rule: dict[str, int] = {}
        self.objects: dict[str, set[str]] = {}
        self.uncovered: list[str] = []

    def both_cannot(self, reason: str) -> None:
        self.D.cannot(reason)
        self.S.cannot(reason)

    def run(self) -> None:
        V.check_platform()
        self.files = file_digests(self.root)
        with open(self.record_path, "rb") as f:
            data = f.read()
        self.record_sha256 = hashlib.sha256(data).hexdigest()
        try:
            record = strict_json(data, "the record")
        except ValueError as e:
            return self.both_cannot(f"the record is not valid JSON: {e}")
        if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
            return self.both_cannot(f"the record's schema is not {RECORD_SCHEMA!r}")
        extra = set(record) - RECORD_FIELDS
        if extra or RECORD_FIELDS - set(record):
            return self.both_cannot(f"the record's fields are not exactly {sorted(RECORD_FIELDS)}")
        if record["container_index_sha256"] != self.files[INDEX_FILE][1]:
            # A record about another container says nothing about this one: neither half is judged.
            return self.both_cannot("container_index_sha256 is not this container's index.json: the record was made "
                                    "from another container")
        inputs = record["selection_inputs"]
        if not isinstance(inputs, dict) or set(inputs) != INPUT_FIELDS or not is_int(inputs["l2_bytes"]) \
                or inputs["l2_bytes"] <= 0 or any(inputs[k] is not None and not isinstance(inputs[k], str)
                                                  for k in INPUT_FIELDS - {"l2_bytes"}):
            return self.both_cannot("selection_inputs is malformed: l2_bytes must be a positive integer and the "
                                    "three settings strings or null")
        self.inputs = inputs
        if inputs["arithmetic"] not in (None, "") or inputs["kquant_exec"] not in (None, ""):
            self.S.cannot("an arithmetic arm or a K-quant execution mode is set: §4 specifies the default arm only")
        operands = record["operands"]
        if not isinstance(operands, list) or not operands:
            return self.both_cannot("the record names no operand, so the contract would hold vacuously")
        self.store = Store(self.root)
        seen: set[tuple[str, str]] = set()
        for i, o in enumerate(operands):
            self.judge_operand(i, o, seen)
        self.check_coverage()

    def judge_operand(self, i: int, o, seen: set) -> None:
        D, S = self.D, self.S
        if not isinstance(o, dict) or set(o) != OPERAND_FIELDS or not isinstance(o.get("object"), str) \
                or not isinstance(o.get("tensor"), str) or not is_int(o.get("order")) \
                or not isinstance(o.get("shape"), list) or not all(is_int(n) for n in o["shape"]):
            D.violate(f"operands[{i}]", "malformed operand: its fields are not exactly the record's, or its order "
                                        "or shape is not of integers")
            S.cannot(f"operands[{i}]: malformed operand")
            return
        where = f"{o['object']}/{o['tensor']}"
        key = (o["object"], o["tensor"])
        if key in seen:
            D.violate(where, "the operand is recorded more than once")
            S.violate(where, "the operand is recorded more than once")
            return
        seen.add(key)
        self.objects.setdefault(o["object"], set()).add(o["tensor"])
        rule = o["rule"]
        if not isinstance(rule, str):
            D.violate(where, "the operand names no rule")
            S.cannot(f"{where}: no rule")
            return
        self.by_rule[rule] = self.by_rule.get(rule, 0) + 1
        rid = f"{o['object']}@{C.BF16}"
        if rid not in self.store.reps or "compiled_from" in self.store.reps[rid]:
            D.cannot(f"{where}: the container has no canonical BF16 representation of {o['object']!r}")
            S.cannot(f"{where}: no canonical BF16 representation")
            return
        seg = self.store.seg(rid)
        if o["tensor"] not in seg.by_name:
            D.violate(where, "the container's canonical representation holds no such tensor")
            S.cannot(f"{where}: no such tensor")
            return
        entry = seg.by_name[o["tensor"]]
        shape = tuple(entry.shape)
        if entry.dtype != C.BF16 or len(shape) != 2:
            D.cannot(f"{where}: stored as {entry.dtype} {list(shape)}; the rules are defined on BF16 matrices")
            S.cannot(f"{where}: not a BF16 matrix")
            return
        if o["stored_dtype"] is None or o["stored_sha256"] is None:
            D.cannot(f"{where}: the record gives no stored dtype or digest (an operand read through an overlay); "
                     "the rules are defined on a container's stored bytes")
            S.cannot(f"{where}: no stored dtype or digest")
            return
        if o["stored_dtype"] == STORED_PACK:
            D.cannot(f"{where}: read from a stored {STORED_PACK} representation, which this contract does not "
                     "derive from")
            S.cannot(f"{where}: read from a stored {STORED_PACK} representation")
            return
        if o["shape"] != list(shape) or o["stored_dtype"] != C.BF16:
            D.violate(where, f"recorded as {o['stored_dtype']} {o['shape']}, stored as BF16 {list(shape)}")
            S.cannot(f"{where}: the record does not describe the stored tensor")
            return
        raw = self.store.read(rid, o["tensor"])

        # derivation
        if rule not in DIGEST_LAYOUT:
            D.cannot(f"{where}: rule {rule!r} is not one this contract derives ({sorted(DIGEST_LAYOUT)})")
        elif rule == RULE_Q8_LOAD and not np.all(np.isfinite(widen_bf16(raw))):
            D.cannot(f"{where}: the stored tensor holds a non-finite element, outside {RULE_Q8_LOAD}'s domain (§3.1)")
        else:
            D.judged += 1
            if o["stored_sha256"] != sha256(raw):
                D.violate(where, "stored_sha256 is not the digest of the container's tensor: the loader read other "
                                 "bytes, or the record is stale")
            if o["resident_digest_layout"] != DIGEST_LAYOUT[rule]:
                D.violate(where, f"resident_digest_layout {o['resident_digest_layout']!r} is not {rule}'s "
                                 f"{DIGEST_LAYOUT[rule]!r}")
            elif not is_sha(o["resident_sha256"]):
                D.violate(where, "resident_sha256 is not a SHA-256")
            elif o["resident_sha256"] != resident_digest(rule, raw, shape):
                D.violate(where, f"resident_sha256 is not {rule} applied to the container's tensor")
            if o["resident_format"] != FORMAT_OF[rule]:
                D.violate(where, f"resident_format {o['resident_format']!r} is not {rule}'s {FORMAT_OF[rule]!r}")
            if o["manufactured"] is not (rule == RULE_Q8_LOAD):
                D.violate(where, f"manufactured is {o['manufactured']!r} under {rule}")

        # selection
        source = o["representation_source"]
        if S.reasons and "default arm only" in S.reasons[0]:
            return
        if source not in SOURCES:
            S.violate(where, f"representation_source {source!r} is not one of {list(SOURCES)}")
        elif source == "transient":
            S.cannot(f"{where}: §4 does not specify the transient source")
        elif rule not in FORMAT_OF:
            S.cannot(f"{where}: rule {rule!r} is not one §4 places")
        else:
            S.judged += 1
            allowed = selected_rules(shape[0] * shape[1], source, self.inputs["l2_bytes"], self.inputs["max_format"])
            if rule not in allowed:
                S.violate(where, f"resident under {rule}; a {shape[0]}x{shape[1]} matrix under source {source!r}, "
                                 f"cache {self.inputs['l2_bytes']} bytes and cap {self.inputs['max_format']!r} is "
                                 f"placed under {sorted(allowed)}")

    def check_coverage(self) -> None:
        """§5: for every object the record covers, every stored matrix of its canonical representation is in the
        record. A record that leaves one out has not accounted for it, whichever rule it would have had."""
        for obj, named in sorted(self.objects.items()):
            rid = f"{obj}@{C.BF16}"
            if rid not in self.store.reps or "compiled_from" in self.store.reps[rid]:
                continue
            matrices = {t.name for t in self.store.seg(rid).tensors if len(t.shape) == 2}
            for missing in sorted(matrices - named):
                self.D.violate(f"{obj}/{missing}", "a stored matrix of an object the record covers is not in the record")
                self.S.violate(f"{obj}/{missing}", "a stored matrix of an object the record covers is not in the record")
        # Objects the record says nothing about. Whether the run loaded them is not something the container shows,
        # so they are reported and not judged.
        for rid, e in sorted(self.store.reps.items()):
            if e["encoding"] == C.BF16 and "compiled_from" not in e and e["object"] not in self.objects \
                    and any(len(t.shape) == 2 for t in self.store.seg(rid).tensors):
                self.uncovered.append(e["object"])


def could_not(out: dict, *reasons: str) -> dict:
    out.update(status="could_not_evaluate", violations=0, offending=[], reasons=list(reasons))
    return out


def evaluate(out: dict, record: str, container: str) -> dict:
    """Fill `out` with the two verdicts and the combined status. Never raises for a bad input."""
    out["derived_from"] = ("the container's own stored tensor bytes, by this contract's implementation of the rule "
                           "each operand names; the choice re-decided from the recorded policy and cache size")
    judge = Judge(record, container)
    try:
        judge.run()
    except RecursionError:
        judge.both_cannot("an input can't be read: it is nested more deeply than this reader follows")
    except RuntimeError as e:
        judge.both_cannot(f"the platform cannot run the rules: {e}")
    except BAD_INPUT as e:
        judge.both_cannot(f"an input can't be read: {type(e).__name__}: {e}")
    files = getattr(judge, "files", None)
    if files is not None:
        try:
            out["container"]["sha256"] = container_digest(files)
        except ValueError:  # a file name that is not text (non-UTF-8 bytes, on a file system that allows them)
            judge.both_cannot("the container holds a file whose name is not valid UTF-8, so it has no container "
                              "digest")
    D, S = judge.D, judge.S
    halves = {DERIVATION: D.report(), SELECTION: S.report()}
    statuses = [h["status"] for h in halves.values()]
    status = "satisfied" if all(s == "satisfied" for s in statuses) else \
        "violated" if "violated" in statuses else "could_not_evaluate"
    offending = D.offending + S.offending
    if len(offending) > OFFENDING_LIMIT:
        tail = offending[OFFENDING_LIMIT - 1:]
        offending = offending[:OFFENDING_LIMIT - 1] + [{
            "name": f"{len(tail)} further findings", "violations": len(tail),
            "detail": "listed in full under `verdicts`", "half": ",".join(sorted({o["half"] for o in tail}))}]
    out["record"]["sha256"] = getattr(judge, "record_sha256", None)
    out.update(
        status=status,
        violations=len(D.offending) + len(S.offending) if status == "violated" else 0,
        offending=offending if status == "violated" else [],
        reasons=[f"{DERIVATION}: {r}" for r in D.reasons] + [f"{SELECTION}: {r}" for r in S.reasons],
        verdicts=halves,
        derivation=halves[DERIVATION]["status"],
        selection=halves[SELECTION]["status"],
        failed_halves=[k for k, h in halves.items() if h["status"] != "satisfied"],
        selection_inputs=getattr(judge, "inputs", None),
        uncovered_objects=list(judge.uncovered),
        measurements=[
            {"name": "operands whose derivation was recomputed", "value": str(D.judged)},
            {"name": "operands whose selection was re-decided", "value": str(S.judged)},
            {"name": "operands by rule", "value": ", ".join(f"{k}: {v}" for k, v in sorted(judge.by_rule.items()))},
            {"name": "objects covered", "value": ", ".join(sorted(judge.objects))},
            {"name": "objects with stored matrices that the record does not name (not judged)",
             "value": ", ".join(judge.uncovered) or "none"},
            {"name": "not established", "value": "; ".join(NOT_ESTABLISHED)},
        ],
    )
    if status == "could_not_evaluate" and not out["reasons"]:
        out["reasons"] = ["nothing was judged"]
    return out
