"""`WEIGHT_PATCHES_V2` and `WEIGHT_PATCHES_V3`: does a compiled container hold what its `weight-patches/v2` or
`weight-patches/v3` ledger declares of a base container?

Written from `docs/verification/weight-patches-v2-v3.md` ("the spec" below; §7 is the judgement). A v2 ledger records
literal delta edits only. A v3 ledger records the results of `remove_direction` ops by their bits, alone or beside
delta edits on other tensors. Both are made of element targets, which `weight_patches_v4.py` already judges inside a
v4 ledger, so this is that judge held to another schema, to that schema's form, and to a stricter reading of the
patch. Nothing here is a second implementation of the arithmetic:

* a delta is recomputed by `weight_patches_v4.exact_sum_to_bf16` (the exact rational sum, rounded once);
* a `remove_direction` op is recomputed by `vlp_numerics.RemoveDirection` (the frozen v1 numerics), from the ops of
  the patch given, each held to the ledger's `origins`.

What is added (spec §7 steps 2, 3, 8 and 9):

* the ledger's schema must be the contract's, and its contents must be of the form that schema is written for;
* a patch, when one is given, is held to the ledger in full: every entry of it is read, its `weight_delta` ops must
  be exactly the ledger's delta edits, every weight op must be one of the ledger's `origins` and every origin an op
  of the patch, and it may hold no `add_low_rank_columns` op;
* the two containers must hold the same directories, the ledger's path must not be a file of the base, and a
  re-derived segment must keep the base's header bytes.

`WEIGHT_PATCHES_V4` does not use this module. The corrections made to the shared v4 code are listed in spec §13.

**Not established, and said so in the answer:** the content of a re-derived (quantised) representation, which is
`Q8_0_PACK_DERIVATION`'s claim; `candidate.json`; and the ledger's `origins` when no patch is given.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import PurePosixPath

from . import vlp_numerics as V
from . import weight_patches_v4 as W

CONTRACT_V2 = "WEIGHT_PATCHES_V2"
CONTRACT_V3 = "WEIGHT_PATCHES_V3"
LEDGER_SCHEMA_V2 = "weight-patches/v2"
LEDGER_SCHEMA_V3 = "weight-patches/v3"
STATEMENTS = {
    CONTRACT_V2: (
        "the compiled container is the base with exactly the delta edits its weight-patches/v2 ledger declares, each "
        "recomputed independently as the BF16 nearest the exact sum of the base's element and the delta (numerical "
        "satisfaction), and the ledger is bound to the base, of the form its schema is written for and consistent "
        "with both containers and with the patch if one is given (ledger integrity and provenance)"
    ),
    CONTRACT_V3: (
        "the compiled container is the base with exactly the edits its weight-patches/v3 ledger declares, its "
        "remove_direction results recomputed independently from the base weights and the ops of the patch given and "
        "its delta edits as the BF16 nearest the exact sum (numerical satisfaction), and the ledger is bound to the "
        "base, of the form its schema is written for and consistent with both containers and with that patch "
        "(ledger integrity and provenance)"
    ),
}
#: Which contract judges which schema, for a refusal that can name the right one.
CONTRACT_OF = {LEDGER_SCHEMA_V2: CONTRACT_V2, LEDGER_SCHEMA_V3: CONTRACT_V3, W.LEDGER_SCHEMA_V4: W.CONTRACT}

# spec §3.1: the `.vlp` op that carries literal deltas
OP_DELTA = "weight_delta"
DELTA_OP_FIELDS = {"op", "rounding", "source", "tensor", "edits", "digest"}
DELTA_OP_EDIT_FIELDS = {"row", "col", "delta", "delta_bits"}
DELTA_DIGEST_HEADER = "weight_delta/v1"
F32_BITS = 1 << 32
KIND_SHOWN = 40
DIGEST_SHOWN = 12


def delta_op_digest(op: dict) -> str:
    """Spec §3.1: SHA-256 of the op's canonical text, every line ending in a newline, the edits sorted by
    `(row, col)`. Raises ValueError if a field cannot be rendered (a malformed op)."""
    source = op.get("source")
    if not isinstance(source, dict):
        raise ValueError("the op has no source")
    if not isinstance(op.get("rounding"), str) or not isinstance(op.get("tensor"), str):
        raise ValueError("rounding or tensor is not a string")
    lines = [DELTA_DIGEST_HEADER, f"rounding={op['rounding']}"]
    for f in W.SOURCE_FIELDS:
        if not isinstance(source.get(f), str):
            raise ValueError(f"source.{f} is not a string")
        lines.append(f"{f}={source[f]}")
    lines.append(f"tensor={op['tensor']}")
    edits = op.get("edits")
    if not isinstance(edits, list) or not all(
            isinstance(e, dict) and all(W.is_int(e.get(f)) for f in ("row", "col", "delta_bits")) for e in edits):
        raise ValueError("edits are not a list of row, col and delta_bits integers")
    for e in sorted(edits, key=lambda e: (e["row"], e["col"])):
        lines.append(f"{e['row']},{e['col']},{e['delta_bits']}")
    return hashlib.sha256("".join(line + "\n" for line in lines).encode("utf-8")).hexdigest()


def directories(root: str) -> set[str]:
    """Every directory under `root`, by relative path. (A symlinked one was refused when the files were read.)"""
    out = set()
    for dirpath, dirnames, _ in os.walk(root):
        for name in dirnames:
            out.add(os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/"))
    return out


class ElementLedgerJudge(W.Judge):
    """The judge of `weight_patches_v4.py`, for a ledger made of element targets only."""

    #: The contract this judge answers, for messages.
    CONTRACT = ""
    #: Whether the schema is the one written for result-bits edits (v3) or the one written for none (v2).
    RESULT_BITS_EDITS = False
    RECOMPUTED_FROM = ("the base container's weights and the deltas (delta_bits) the compiled container's ledger "
                       "records; the ledger's old and new bits and its digests are checked, never used as expected "
                       "values")
    RECOMPUTED_FROM_PATCH = ("; the patch given is held to the ledger in full, and for a tensor the ledger records by "
                             "result bits its remove_direction ops are what the tensor is recomputed from, each held "
                             "to the ledger's origins")

    def __init__(self, base: str, container: str):
        super().__init__(base, container)
        #: `weight_delta` ops of the patch given, as `(position in the patch, op)`.
        self.patch_deltas: list[tuple[int, dict]] = []
        #: Positions of the patch's `add_low_rank_columns` ops.
        self.patch_low_rank: list[int] = []
        self.patch_deltas_held = 0
        self.not_established = [
            "the content of re-derived (quantised) representations: their digests are chained to the compiled "
            "canonical payload and the tensors are not re-quantised (Q8_0_PACK_DERIVATION judges them)",
            "candidate.json (regenerated by a compile, not an authority, not judged)",
            "the ledger's origins when no patch is given (with a patch, each must be one of its weight ops)",
            "where the deltas or an operand came from (an origin note is covered by its op's digest and read by "
            "nothing)",
        ]

    # -- spec §7 step 1: the inputs ----------------------------------------------------------------------------
    def run(self) -> None:
        """A base whose `index.json` names a ledger is a compiled edit. The engine refuses to stage an edit on one,
        so no compile has it as its base, and there is nothing to judge."""
        base_index = W.strict_json(W.read_file(self.base_root, W.INDEX_FILE), W.INDEX_FILE)
        if isinstance(base_index, dict) and W.LEDGER_POINTER in base_index:
            raise W.CannotEvaluate("the base's index.json names a weight-patch ledger: it is a compiled edit, which "
                                   "the engine never edits, so it is the base of no compile (§7 step 1)")
        super().run()

    # -- spec §7 step 2: the schema ---------------------------------------------------------------------------
    def schema_refusal(self, schema) -> tuple[str, str]:
        other = CONTRACT_OF.get(schema) if isinstance(schema, str) else None
        whose = f"; {other} judges that schema" if other else ""
        return (f"ledger schema {schema!r} is not the one {self.CONTRACT} judges, which is {self.SCHEMA!r} only"
                f"{whose}",
                f"the ledger is not a {self.SCHEMA} ledger, so its edits are not reconstructed")

    # -- spec §7 step 3: the form ------------------------------------------------------------------------------
    def check_form(self, targets) -> None:
        """Each schema is written for one form of ledger and only for it (spec §2.1). This reads the targets as
        written, so a target refused for another reason is still counted."""
        L = self.L
        operations = sum(1 for t in targets if isinstance(t, dict) and "operations" in t)
        results = sum(1 for t in targets if isinstance(t, dict) and isinstance(t.get("edits"), list)
                      for e in t["edits"] if isinstance(e, dict) and e.get("encoding") == W.RESULT_BITS)
        if operations:
            L.violate("targets", f"{operations} operation target(s): an operation target needs "
                                 f"{W.LEDGER_SCHEMA_V4!r}, and this ledger is {self.SCHEMA!r}", operations)
        if results and not self.RESULT_BITS_EDITS:
            L.violate("targets", f"{results} result-bits edit(s): a result-bits edit needs {LEDGER_SCHEMA_V3!r}, and "
                                 f"this ledger is {self.SCHEMA!r}", results)
        if self.RESULT_BITS_EDITS and not results:
            L.violate("targets", f"a {self.SCHEMA!r} ledger is written only when some edit is a result-bits edit, "
                                 "and this one has none")
        rounding = self.ledger.get("rounding")
        if rounding != W.DELTA_ROUNDING:
            L.cannot(f"rounding rule {W.written(rounding)} is not {W.DELTA_ROUNDING!r}, the one this contract "
                     "implements")
        if "origins" in self.ledger and self.origins() is None:
            L.violate("origins", "origins is not a list of SHA-256 hex strings")

    def origins(self) -> list[str] | None:
        """The ledger's `origins`: `[]` when absent, `None` when it is not a list of SHA-256 hex strings."""
        origins = self.ledger.get("origins", [])
        if not isinstance(origins, list) or not all(W.is_sha(o) for o in origins):
            return None
        return origins

    def plan_target(self, i: int, t: dict):
        """Spec §7 step 3, for one target. A target of a form this contract does not judge is not reconstructed:
        the ledger half is violated for it (by `check_form`, or here) and its numerical half is not evaluated."""
        plan = super().plan_target(i, t)
        if plan is None:
            return None
        L, label = self.L, plan["label"]
        written = t.get("edits")
        nulls = sum(1 for e in written if isinstance(e, dict) and "encoding" in e and e["encoding"] is None) \
            if isinstance(written, list) else 0
        if nulls:
            L.violate(label, f"{nulls} edit(s) with a null encoding: the field is absent, \"delta\" or "
                             f"\"result_bits\"", nulls)
        edits = plan.get("edits", [])
        results = [e for e in edits if e.get("encoding") == W.RESULT_BITS]
        for e in results:
            if e["delta_bits"] != 0 or not W.delta_reads_as(e["delta"], 0):
                L.violate(f"{label} ({e['row']}, {e['col']})",
                          f"a result-bits edit carries a delta ({W.written(e['delta'])}, delta_bits "
                          f"{e['delta_bits']:08x}): it has none, and both are zero")
        mixed = bool(results) and len(results) != len(edits)
        if mixed:
            L.violate(label, f"{len(results)} result-bits and {len(edits) - len(results)} delta edits in one target: "
                             "a tensor is edited by deltas or by remove_direction ops, never by both")
        why = None
        if plan["kind"] == "operations":
            why = f"an operation target is not a form {self.CONTRACT} judges"
        elif results and not self.RESULT_BITS_EDITS:
            why = f"a result-bits edit is not a form {self.CONTRACT} judges"
        elif mixed:
            why = "a target that mixes delta and result-bits edits is not a form this contract judges"
        if why is not None:
            plan["ops"] = []
            if not plan["refused"]:
                plan["refused"] = True
                self.N.cannot(f"{label}: {why}, so the tensor is not reconstructed")
        return plan

    # -- spec §7 step 8: the patch, in full ----------------------------------------------------------------------
    def read_patch(self, data: bytes) -> dict:
        """Every entry of the patch (spec §3.3). Returns its `remove_direction` ops by tensor, as the judge this
        extends expects, and keeps its `weight_delta` ops and the positions of its low-rank ops. An entry of any
        other kind makes the patch unusable: it is held to the ledger in full or not at all."""
        patch = W.strict_json(data, "the patch", parse_float=W.DecimalFloat, parse_int=W.ledger_int)
        operations = patch.get("operations") if isinstance(patch, dict) else None
        if not isinstance(operations, list):
            raise W.CannotEvaluate("the patch has no operations list")
        removals: dict = {}
        deltas, low_rank = [], []
        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                raise W.CannotEvaluate(f"operation {i} of the patch is not an object")
            kind = op.get("op")
            if kind == W.OP_REMOVE:
                tensor = op.get("tensor")
                removals.setdefault(tensor if isinstance(tensor, str) else W.UNPLACED, []).append(op)
            elif kind == OP_DELTA:
                deltas.append((i, op))
            elif kind == W.OP_LOW_RANK:
                low_rank.append(i)
            else:
                raise W.CannotEvaluate(f"operation {i} of the patch is of kind {repr(kind)[:KIND_SHOWN]}, which is "
                                       "not a stored-weight op this contract reads; a patch is held to the ledger "
                                       "in full or not at all")
        self.patch_deltas, self.patch_low_rank = deltas, low_rank
        return removals

    def check_patch_removals_placed(self) -> None:
        """The patch given must be exactly what the container was compiled from, as far as the ledger can show. Its
        `remove_direction` ops are placed as the v4 route places them; then its other ops and the ledger's
        `origins` are held too."""
        super().check_patch_removals_placed()
        if self.patch_ops is None:  # no patch, or one that could not be read
            return
        if self.patch_low_rank:
            n = len(self.patch_low_rank)
            self.L.violate("patch", f"the patch holds {n} add_low_rank_columns op(s), first at operation "
                                    f"{self.patch_low_rank[0]}; a compile with one writes "
                                    f"{W.LEDGER_SCHEMA_V4!r}, so the patch is not the one this container was "
                                    "compiled from", n)
        self.check_patch_deltas()
        self.check_patch_origins()
        self.L.measure("weight_delta ops of the patch held to the ledger's delta edits",
                       f"{self.patch_deltas_held}/{len(self.patch_deltas)}")

    def base_source(self) -> dict:
        """The source binding every op of a patch compiled into this container carries."""
        src = self.ledger.get("source")
        return {"model": src.get("model") if isinstance(src, dict) else None,
                "index_sha256": self.base_index_sha, "representation": self.rep,
                "payload_sha256": self.bc.reps[self.rep]["payload_sha256"]}

    def ledger_delta_edits(self) -> dict[str, dict[tuple[int, int], int]]:
        """`{tensor: {(row, col): delta_bits}}` of the ledger's delta edits, read as written."""
        out: dict = {}
        targets = self.ledger.get("targets")
        for t in targets if isinstance(targets, list) else []:
            if not isinstance(t, dict) or not isinstance(t.get("tensor"), str) or not isinstance(t.get("edits"), list):
                continue
            for e in t["edits"]:
                if isinstance(e, dict) and e.get("encoding") in (None, W.DELTA_ENCODING) \
                        and all(W.is_int(e.get(f)) for f in ("row", "col", "delta_bits")):
                    out.setdefault(t["tensor"], {})[(e["row"], e["col"])] = e["delta_bits"]
        return out

    def admit_delta_op(self, where: str, op: dict, source: dict, origins: set[str]):
        """One `weight_delta` op of the patch, before it is compared with the ledger (spec §7 step 8): its fields,
        rounding rule, source binding, digest and origin. Returns its edits as `{(row, col): delta_bits}`, or `None`
        when the op is refused."""
        L = self.L
        extra = set(op) - DELTA_OP_FIELDS
        if extra:
            L.cannot(f"{where}: unsupported op fields {sorted(extra)}")
            return None
        if not isinstance(op.get("tensor"), str):
            L.violate(where, "the weight_delta op names no tensor")
            return None
        if op.get("rounding") != W.DELTA_ROUNDING:
            L.cannot(f"{where}: rounding rule {W.written(op.get('rounding'))} is not {W.DELTA_ROUNDING!r}")
            return None
        if op.get("source") != source:
            got = op.get("source") if isinstance(op.get("source"), dict) else {}
            fields = [f for f in W.SOURCE_FIELDS if got.get(f) != source[f]] or ["fields"]
            L.violate(where, f"stale or foreign source binding ({', '.join(fields)}): the op is not bound to this base")
            return None
        edits = op.get("edits")
        if not isinstance(edits, list) or not edits:
            L.violate(where, "the weight_delta op has no edits")
            return None
        held: dict = {}
        for e in edits:
            if not isinstance(e, dict) or set(e) - DELTA_OP_EDIT_FIELDS \
                    or not all(W.is_int(e.get(f)) for f in ("row", "col", "delta_bits")) \
                    or not isinstance(e.get("delta"), (int, float)) or isinstance(e.get("delta"), bool) \
                    or e["row"] < 0 or e["col"] < 0 or not 0 <= e["delta_bits"] < F32_BITS:
                L.violate(where, "malformed edit")
                return None
            if (e["row"], e["col"]) in held:
                L.violate(where, f"element ({e['row']}, {e['col']}) is edited twice")
                return None
            held[(e["row"], e["col"])] = e["delta_bits"]
        try:
            intact = delta_op_digest(op) == op.get("digest")
        except ValueError as e:
            L.violate(where, f"malformed op: {e}")
            return None
        if not intact:
            L.violate(where, "digest is not intact: the op's fields do not hash to its recorded digest (§3.1)")
            return None
        self.digests_checked += 1
        if op["digest"] not in origins:
            L.violate(where, "the op's digest is not among the ledger's origins: the patch is not the one this "
                             "container was compiled from")
            return None
        unread = [e for e in edits if (e["delta_bits"] & V.F32_EXP_MASK) == V.F32_EXP_MASK
                  or not W.delta_reads_as(e["delta"], e["delta_bits"])]
        if unread:
            e = unread[0]
            L.violate(where, f"{len(unread)} edit(s) whose delta does not read as a finite binary32 of its "
                             f"delta_bits, first at ({e['row']}, {e['col']}): {W.written(e['delta'])} for "
                             f"{e['delta_bits']:08x}", len(unread))
            return None
        return held

    def check_patch_deltas(self) -> None:
        """Spec §7 step 8: on each tensor, the elements of the patch's `weight_delta` ops are exactly the ledger's
        delta edits, with the same `delta_bits`."""
        L = self.L
        recorded = self.ledger_delta_edits()
        source, origins = self.base_source(), set(self.origins() or [])
        in_patch: dict[str, dict] = {}
        refused = False
        for i, op in self.patch_deltas:
            where = f"patch op {i}"
            held = self.admit_delta_op(where, op, source, origins)
            if held is None:
                refused = True
                continue
            tensor = op["tensor"]
            where = f"{where} on {tensor}"
            if tensor not in recorded:
                L.violate(where, "the ledger records no delta edit of this tensor: the patch is not the one this "
                                 "container was compiled from")
                continue
            merged = in_patch.setdefault(tensor, {})
            twice = sorted(set(held) & set(merged))
            if twice:
                L.violate(where, f"{len(twice)} element(s) are also edited by an earlier weight_delta op of the "
                                 f"patch, first {twice[0]}", len(twice))
                continue
            absent = sorted(at for at in held if at not in recorded[tensor])
            other = sorted(at for at, bits in held.items() if at in recorded[tensor] and recorded[tensor][at] != bits)
            if absent:
                L.violate(where, f"{len(absent)} delta(s) of the patch are at elements the ledger does not record, "
                                 f"first {absent[0]}", len(absent))
            if other:
                at = other[0]
                L.violate(where, f"{len(other)} delta(s) of the patch differ from the ledger's at the same element, "
                                 f"first {at}: delta_bits {held[at]:08x} in the patch, {recorded[tensor][at]:08x} in "
                                 "the ledger", len(other))
            merged.update(held)
            self.patch_deltas_held += int(not absent and not other)
        if refused:
            return  # a refused op is already reported; what it would have covered is not also reported as missing
        for tensor in sorted(recorded):
            missing = sorted(at for at in recorded[tensor] if at not in in_patch.get(tensor, {}))
            if missing:
                L.violate(f"patch, {tensor}", f"{len(missing)} delta edit(s) of the ledger are not in the patch, "
                                              f"first {missing[0]}: the patch is not the whole of what this "
                                              "container was compiled from", len(missing))

    def check_patch_origins(self) -> None:
        """Spec §7 step 8: every entry of `origins` is the digest of a weight op of the patch. (That each op of the
        patch is among the origins is held where the op is admitted.)"""
        origins = self.origins()
        if origins is None:
            return  # reported by check_form
        ops = [op for _, op in self.patch_deltas] + [op for held in self.patch_ops.values() for op in held]
        held = {op["digest"] for op in ops if isinstance(op.get("digest"), str)}
        missing = [o for o in origins if o not in held]
        if missing:
            self.L.violate("origins", f"{len(missing)} of the ledger's {len(origins)} origins are not the digest of "
                                      f"a weight op of the patch given, first {missing[0][:DIGEST_SHOWN]}: the patch "
                                      "is not the whole of what this container was compiled from", len(missing))

    # -- spec §7 step 9 and §5: nothing else differs ---------------------------------------------------------------
    def check_files(self) -> None:
        super().check_files()
        L = self.L
        ledger_path = "/".join(PurePosixPath(self.ledger_rel).parts)
        if ledger_path in self.base_files:
            L.violate(self.ledger_rel, "the ledger's path is a file of the base: a compile writes its ledger as a new "
                                       "file, and this one stands where the base holds another (§5 D6)")
        base_dirs, new_dirs = directories(self.base_root), directories(self.new_root)
        for rel in sorted(base_dirs ^ new_dirs):
            L.violate(f"{rel}/", "a directory present in only one of the containers (§5)")

    def check_derived(self) -> None:
        """Spec §5 D3: a re-derived segment keeps the base's header bytes, as well as its tensor table. Its payload
        is chained by digest and not recomputed."""
        super().check_derived()
        for rid in self.derived:
            if rid in self.nc.reps and W.segment_header(self.bc.seg(rid)) != W.segment_header(self.nc.seg(rid)):
                self.L.violate(f"{rid} segment header", "the re-derived segment's header differs from the base's "
                                                        "(§5 D3)")


class V2Judge(ElementLedgerJudge):
    CONTRACT = CONTRACT_V2
    SCHEMA = LEDGER_SCHEMA_V2
    RESULT_BITS_EDITS = False


class V3Judge(ElementLedgerJudge):
    CONTRACT = CONTRACT_V3
    SCHEMA = LEDGER_SCHEMA_V3
    RESULT_BITS_EDITS = True
    RECOMPUTED_FROM = ("the base container's weights and the deltas (delta_bits) the compiled container's ledger "
                       "records; a tensor the ledger records by result bits is recomputed only from the "
                       "remove_direction ops of a patch; the ledger's old and new bits and its digests are checked, "
                       "never used as expected values")


JUDGES = {CONTRACT_V2: V2Judge, CONTRACT_V3: V3Judge}


def evaluate(out: dict, base: str, container: str, patch: bytes | None = None,
             patch_problem: str | None = None) -> dict:
    """Fill `out` with the two verdicts and the combined status, for the contract `out` names. Never raises for a
    bad input. The arguments are those of `weight_patches_v4.evaluate`, which does the work with this module's
    judge."""
    return W.evaluate(out, base, container, patch, patch_problem, judge_class=JUDGES[out["contract"]])
