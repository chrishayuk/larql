"""`WEIGHT_PATCHES_V2` and `WEIGHT_PATCHES_V3` against the four engine-compiled fixtures in
`conformance/weight_patches_v2v3/`, golden vectors for the delta rule, and fault injection.

The contract is `docs/verification/weight-patches-v2-v3.md` ("the spec"). Three shapes are judged: a v2 ledger of
deltas (`deltas`), a v3 ledger of `remove_direction` results (`removal`, `removal_two`) and a v3 ledger with both
(`mixed`).

Every fault is injected into a copy of a fixture made inside the test. A careful tamperer re-seals what a checksum
would notice (the segment and payload digests in `index.json`, the ledger's result digests, an op's digest), so
that the container stays structurally valid and only the judgement can object. Each test says which half must fail.

The golden vectors of `ExactSumGolden` are derived here, from the bit fields and exact rationals, by code that
shares nothing with the contract's. The fixtures are engine output: inputs to judge, never expected values.
"""

import hashlib
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from larql_contract_verifier import cli  # noqa: E402
from larql_contract_verifier.contracts import LEDGER_CONTRACTS  # noqa: E402
from larql_contract_verifier.contracts import vlp_numerics as V  # noqa: E402
from larql_contract_verifier.contracts import weight_patches_v2v3 as E  # noqa: E402
from larql_contract_verifier.contracts import weight_patches_v4 as W  # noqa: E402
from refcheck import container as C  # noqa: E402
from refcheck import numconv  # noqa: E402

FIXTURES = os.path.join(ROOT, "conformance", "weight_patches_v2v3")
V4_FIXTURES = os.path.join(ROOT, "conformance", "weight_patches_v4")
BASE = os.path.join(ROOT, "conformance", "structured_patch", "base")
VECTORS_V1 = os.path.join(REPO, "conformance", "vlp-numerics", "v1", "vectors.json")
V2, V3, V4 = E.CONTRACT_V2, E.CONTRACT_V3, W.CONTRACT
#: Each fixture and the contract that judges it.
CASES = {"deltas": V2, "removal": V3, "removal_two": V3, "mixed": V3}
V4_CASES = ("low_rank_k2", "mixed", "sham", "f32_profile", "split")
SCHEMA_OF = {V2: E.LEDGER_SCHEMA_V2, V3: E.LEDGER_SCHEMA_V3, V4: W.LEDGER_SCHEMA_V4}
#: Every edited tensor of each fixture: `(target index, how its edits are recorded)`.
TARGETS = {"deltas": ((0, "delta"), (1, "delta")), "removal": ((0, "result"),), "removal_two": ((0, "result"),),
           "mixed": ((0, "delta"), (1, "result"))}
REP = "target.decoder_stack@BF16"
DERIVED = "target.decoder_stack@Q8_0"
NORM = "target.final_norm@BF16"
LEDGER_FILE = "weight_patches.json"
INDEX_FILE = "index.json"
CANDIDATE = "candidate.json"
PATCH_FILE = "patch.vlp"
DOWN0 = "0.mlp.down_proj.weight"  # [64, 256]
DOWN1 = "1.mlp.down_proj.weight"  # [64, 256]
O_PROJ = "1.self_attn.o_proj.weight"  # [64, 64]
#: Two tensors of the edited representation that no fixture edits: the same shape, and adjacent in the table.
GATE0, UP0 = "0.mlp.gate_proj.weight", "0.mlp.up_proj.weight"  # [256, 64]
SAT, VIOL, CNE = "satisfied", "violated", "could_not_evaluate"
#: 0.1 and -0.3 rounded to binary32, which the engine prints as `0.1` and `-0.3`.
TENTH_BINARY32_BITS = 0x3DCCCCCD
MINUS_THREE_TENTHS_BINARY32_BITS = 0xBE99999A
BF16_SIGN = 0x8000
F32_SIGN = 0x80000000
ZERO_SHA = "0" * 64
DEEP = 100000


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def numbers_alike(a, b) -> bool:
    """Both JSON numbers (a bool is not one): `0` and `0.0` are the same value written two ways."""
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b))


def write_json(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def tensor_location(root, rep, name):
    c = C.open_container(root, verify_digests=False)
    seg = c.segments[rep]
    e = seg.by_name[name]
    return seg, seg.payload_start + e.offset, e


def read_element(root, name, row, col, rep=REP):
    seg, start, e = tensor_location(root, rep, name)
    with open(seg.path, "rb") as f:
        f.seek(start + (row * e.shape[-1] + col) * 2)
        return struct.unpack("<H", f.read(2))[0]


def write_element(root, name, row, col, bits, rep=REP):
    seg, start, e = tensor_location(root, rep, name)
    with open(seg.path, "r+b") as f:
        f.seek(start + (row * e.shape[-1] + col) * 2)
        f.write(struct.pack("<H", bits))


def reseal_index(root, rep):
    """Make index.json agree with a representation's segment bytes again. Returns its payload digest."""
    c = C.open_container(root, verify_digests=False)
    seg = c.segments[rep]
    data = read_bytes(seg.path)
    index = read_json(os.path.join(root, INDEX_FILE))
    entry = index["representations"][rep]
    entry["segment_sha256"] = sha(data)
    entry["payload_sha256"] = sha(data[seg.payload_start:seg.payload_start + seg.payload_bytes])
    write_json(os.path.join(root, INDEX_FILE), index)
    return entry["payload_sha256"]


def reseal_compiled(root):
    """A careful tamperer: the index digests of the edited representation, the re-derived representation's source
    digest (in the index and the ledger) and the ledger's result digests, all made to agree with the bytes now in
    the container."""
    payload = reseal_index(root, REP)
    index = read_json(os.path.join(root, INDEX_FILE))
    index["representations"][DERIVED]["source_representation_digest"] = payload
    write_json(os.path.join(root, INDEX_FILE), index)
    ledger = read_json(os.path.join(root, LEDGER_FILE))
    entry = index["representations"][REP]
    ledger["result"]["canonical"] = {"payload_sha256": entry["payload_sha256"], "segment_sha256": entry["segment_sha256"]}
    for d in ledger["result"]["derived"]:
        d["source_representation_digest"] = payload
    write_json(os.path.join(root, LEDGER_FILE), ledger)


def reseal_derived(root):
    """The same, after the re-derived segment's bytes were changed: its index digests and the ledger's copy of them."""
    reseal_index(root, DERIVED)
    entry = read_json(os.path.join(root, INDEX_FILE))["representations"][DERIVED]
    ledger = read_json(os.path.join(root, LEDGER_FILE))
    for d in ledger["result"]["derived"]:
        d.update(payload_sha256=entry["payload_sha256"], segment_sha256=entry["segment_sha256"])
    write_json(os.path.join(root, LEDGER_FILE), ledger)


def restructure_segment(root, rep, change):
    """Rebuild a segment from its tensors, as `change` leaves the list of `[table entry, bytes]` pairs, with the
    offsets made contiguous again, and re-seal its index entry (count, length, both digests). The result passes the
    validating reader."""
    index_path = os.path.join(root, INDEX_FILE)
    index = read_json(index_path)
    entry = index["representations"][rep]
    path = os.path.join(root, entry["segment"])
    data = read_bytes(path)
    (hlen,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + hlen])
    payload = data[8 + hlen:]
    tensors = [[dict(t), payload[t["offset"]:t["offset"] + t["len"]]] for t in header["tensors"]]
    change(tensors)
    offset, body = 0, b""
    for t, raw in tensors:
        t.update(offset=offset, len=len(raw))
        body += raw
        offset += len(raw)
    header["tensors"] = [t for t, _ in tensors]
    text = json.dumps(header).encode()
    text += b" " * ((-(8 + len(text))) % 4)
    out = struct.pack("<Q", len(text)) + text + body
    with open(path, "wb") as f:
        f.write(out)
    entry.update(tensor_count=len(tensors), payload_bytes=len(body), segment_sha256=sha(out), payload_sha256=sha(body))
    write_json(index_path, index)


def position_of(tensors, name):
    return next(i for i, (t, _) in enumerate(tensors) if t["name"] == name)


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work)
        self.made = 0

    def fresh(self, stem):
        self.made += 1
        return os.path.join(self.work, f"{stem}-{self.made}")

    def copy(self, case, fixtures=FIXTURES):
        root = self.fresh(case)
        shutil.copytree(os.path.join(fixtures, case, "compiled"), root)
        return root

    def copy_base(self):
        root = self.fresh("base")
        shutil.copytree(BASE, root)
        return root

    def committed(self, case, fixtures=FIXTURES):
        return os.path.join(fixtures, case, "compiled")

    def patch_of(self, case, fixtures=FIXTURES):
        return os.path.join(fixtures, case, PATCH_FILE)

    def judge(self, case, container, patch=True, base=BASE, contract=None):
        """Ask the fixture's own contract. `patch`: `True` for the patch the fixture was compiled from, `None` or
        `False` for none, or a path."""
        if patch is True:
            patch = self.patch_of(case)
        return cli.evaluate_ledger(contract or CASES[case], base, container, patch or None)

    def ledger(self, root):
        return read_json(os.path.join(root, LEDGER_FILE))

    def edit_ledger(self, root, change):
        doc = self.ledger(root)
        change(doc)
        write_json(os.path.join(root, LEDGER_FILE), doc)

    def edit_index(self, root, change):
        doc = read_json(os.path.join(root, INDEX_FILE))
        change(doc)
        write_json(os.path.join(root, INDEX_FILE), doc)

    def patch_copy(self, case, change, fixtures=FIXTURES):
        doc = read_json(self.patch_of(case, fixtures))
        change(doc)
        path = self.fresh("patch") + ".vlp"
        write_json(path, doc)
        return path

    def assertHalves(self, out, numerical, ledger):
        self.assertEqual((out["numerical_satisfaction"], out["ledger_integrity"]), (numerical, ledger),
                         json.dumps({k: out["verdicts"][k] for k in out["verdicts"]}, indent=1)[:4000])
        self.assertEqual(out["verdicts"][W.NUMERICAL]["status"], numerical)
        self.assertEqual(out["verdicts"][W.LEDGER]["status"], ledger)
        failed = [h for h, s in ((W.NUMERICAL, numerical), (W.LEDGER, ledger)) if s != SAT]
        self.assertEqual(out["failed_halves"], failed)
        if numerical == SAT and ledger == SAT:
            self.assertEqual((out["status"], out["violations"], out["offending"]), (SAT, 0, []))
        elif VIOL in (numerical, ledger):
            self.assertEqual(out["status"], VIOL)
            self.assertGreater(out["violations"], 0)
            self.assertEqual(sum(o["violations"] for o in out["offending"]), out["violations"])
        else:
            self.assertEqual(out["status"], CNE)
            self.assertTrue(out["reasons"])

    def assertNotSatisfied(self, out):
        """The answer is well formed and is not a pass."""
        self.assertIn(out["status"], (VIOL, CNE), out)
        self.assertNotEqual((out["numerical_satisfaction"], out["ledger_integrity"]), (SAT, SAT))
        if out["status"] == VIOL:
            self.assertGreater(out["violations"], 0)
            self.assertEqual(sum(o["violations"] for o in out["offending"]), out["violations"])
        else:
            self.assertTrue(out["reasons"])
            self.assertEqual((out["violations"], out["offending"]), (0, []))

    def findings(self, out, half):
        return " | ".join(f"{o['name']}: {o['detail']}" for o in out["verdicts"][half]["offending"]) + " | " + \
            " | ".join(out["verdicts"][half]["reasons"])

    def measured(self, out):
        return {m["name"]: m["value"] for m in out["measurements"]}

    def changing_edit(self, root, target):
        """`(tensor, index, edit)` of the first edit of a target that changes its element."""
        t = self.ledger(root)["targets"][target]
        k = next(i for i, e in enumerate(t["edits"]) if e["old_bits"] != e["new_bits"])
        return t["tensor"], k, t["edits"][k]


class Untampered(Fixture):
    def test_each_fixture_with_its_patch_verifies_on_both_halves(self):
        for case, contract in CASES.items():
            with self.subTest(case=case):
                out = self.judge(case, self.committed(case))
                self.assertHalves(out, SAT, SAT)
                self.assertEqual((out["schema"], out["contract"]), (cli.LEDGER_SCHEMA, contract))
                self.assertEqual(out["contract_statement"], E.STATEMENTS[contract])
                self.assertEqual(out["engine_reported"]["schema"], SCHEMA_OF[contract])
                self.assertEqual(out["patch"], {"path": self.patch_of(case), "sha256": sha(read_bytes(self.patch_of(case)))})
                self.assertEqual(out["provenance_detail"], {"base_binding": SAT, "op_digest": SAT,
                                                            "operand_origin": "not_checked"})
                ledger = self.ledger(self.committed(case))
                measured = self.measured(out)
                # every element of every named tensor is compared, not only those listed
                elements = sum(t["shape"][0] * t["shape"][1] for t in ledger["targets"])
                self.assertEqual(measured["elements compared"], str(elements))
                results = sum(1 for t in ledger["targets"] if t["edits"][0].get("encoding") == W.RESULT_BITS)
                self.assertEqual(measured["result-bits targets recomputed from the patch given"], str(results))
                ops = read_json(self.patch_of(case))["operations"]
                deltas = sum(op["op"] == E.OP_DELTA for op in ops)
                self.assertEqual(measured["weight_delta ops of the patch held to the ledger's delta edits"],
                                 f"{deltas}/{deltas}")
                self.assertEqual(measured["op digests recomputed and intact (§3a)"], str(len(ops)))

    def test_a_delta_only_ledger_needs_no_patch(self):
        out = self.judge("deltas", self.committed("deltas"), patch=None)
        self.assertHalves(out, SAT, SAT)
        self.assertNotIn("patch", out)
        # no op was in hand, so no op digest was recomputed: the origins are not established without the patch
        self.assertEqual(out["provenance_detail"]["op_digest"], CNE)
        self.assertIn("the ledger's origins when no patch is given", self.measured(out)["not established"])
        self.assertIn("the deltas (delta_bits)", out["recomputed_from"])

    def test_a_v3_ledger_without_its_patch_is_never_satisfied(self):
        for case in ("removal", "removal_two", "mixed"):
            with self.subTest(case=case):
                out = self.judge(case, self.committed(case), patch=None)
                self.assertHalves(out, CNE, SAT)
                self.assertEqual(out["status"], CNE)
                self.assertIn("give the patch to recompute them", self.findings(out, W.NUMERICAL))

    def test_the_delta_targets_of_a_mixed_ledger_are_compared_even_without_the_patch(self):
        out = self.judge("mixed", self.committed("mixed"), patch=None)
        self.assertEqual(self.measured(out)["elements compared"], str(64 * 256))

    def test_the_fixtures_are_the_shapes_the_spec_names(self):
        deltas = self.ledger(self.committed("deltas"))
        self.assertEqual(deltas["schema"], E.LEDGER_SCHEMA_V2)
        self.assertEqual([t["tensor"] for t in deltas["targets"]], [DOWN0, O_PROJ])
        edits = [e for t in deltas["targets"] for e in t["edits"]]
        self.assertTrue(all("encoding" not in e for e in edits))
        with open(os.path.join(self.committed("deltas"), LEDGER_FILE)) as f:
            text = f.read()
        # a shortest binary32 rendering, a negative one, and a delta that changes no bits
        self.assertIn('"delta": 0.1,', text)
        self.assertIn('"delta": -0.3,', text)
        self.assertIn(TENTH_BINARY32_BITS, [e["delta_bits"] for e in edits])
        self.assertIn(MINUS_THREE_TENTHS_BINARY32_BITS, [e["delta_bits"] for e in edits])
        self.assertTrue(any(e["old_bits"] == e["new_bits"] for e in edits))
        for case, removals, tensors in (("removal", 1, 1), ("removal_two", 2, 1), ("mixed", 1, 2)):
            ledger = self.ledger(self.committed(case))
            self.assertEqual(ledger["schema"], E.LEDGER_SCHEMA_V3)
            self.assertEqual(len(ledger["targets"]), tensors)
            ops = read_json(self.patch_of(case))["operations"]
            self.assertEqual(sum(op["op"] == W.OP_REMOVE for op in ops), removals)
            self.assertEqual(ledger["origins"], [op["digest"] for op in ops])
        two = [op for op in read_json(self.patch_of("removal_two"))["operations"]]
        self.assertEqual(len({op["tensor"] for op in two}), 1)
        self.assertFalse(set(two[0]["columns"]) & set(two[1]["columns"]))
        mixed = self.ledger(self.committed("mixed"))
        self.assertEqual([sorted({e.get("encoding", "delta") for e in t["edits"]}) for t in mixed["targets"]],
                         [["delta"], [W.RESULT_BITS]])

    def test_the_provenance_record_names_the_engine_binary_and_the_committed_files(self):
        record = read_json(os.path.join(FIXTURES, "provenance.json"))
        self.assertRegex(record["larql_binary_sha256"], "^[0-9a-f]{64}$")
        self.assertEqual(record["base_index_sha256"], sha(read_bytes(os.path.join(BASE, INDEX_FILE))))
        self.assertEqual(sorted(record["cases"]), sorted(CASES))
        for case, entry in record["cases"].items():
            root = self.committed(case)
            self.assertEqual(entry["ledger_schema"], SCHEMA_OF[CASES[case]])
            self.assertEqual(entry["ledger_sha256"], sha(read_bytes(os.path.join(root, LEDGER_FILE))))
            self.assertEqual(entry["index_sha256"], sha(read_bytes(os.path.join(root, INDEX_FILE))))
            self.assertEqual(entry["patch_sha256"], sha(read_bytes(self.patch_of(case))))

    def test_every_patch_op_digest_is_the_specified_text(self):
        for case in CASES:
            for op in read_json(self.patch_of(case))["operations"]:
                digest = E.delta_op_digest(op) if op["op"] == E.OP_DELTA else W.op_digest(op)
                self.assertEqual(digest, op["digest"], (case, op["op"]))

    def test_the_cli_writes_one_json_object(self):
        env = dict(os.environ, PYTHONPATH=ROOT)
        for case, contract in (("deltas", V2), ("mixed", V3)):
            proc = subprocess.run([sys.executable, "-m", "larql_contract_verifier.cli", "ledger", "--contract", contract,
                                   "--base", BASE, "--container", self.committed(case), "--patch", self.patch_of(case)],
                                  capture_output=True, text=True, env=env, cwd=ROOT, check=True)
            answer = json.loads(proc.stdout)
            self.assertEqual((answer["contract"], answer["status"]), (contract, SAT))

    def test_the_ledger_route_knows_exactly_three_contracts(self):
        self.assertEqual(sorted(LEDGER_CONTRACTS), sorted((V2, V3, V4)))
        out = cli.evaluate_ledger("WEIGHT_PATCHES_V1", BASE, self.committed("deltas"))
        self.assertEqual((out["status"], out["contract_statement"]), (CNE, ""))
        self.assertIn(str(sorted(LEDGER_CONTRACTS)), out["reasons"][0])

    def test_the_removal_fixtures_agree_with_the_exact_rational_patch_route(self):
        """`STRUCTURED_REMOVE_DIRECTION` recomputes a removal on exact rationals, by other code than the ledger
        route's. It reads a patch of removals only, so it can speak for `removal` and `removal_two` and not for a
        patch that also holds a delta."""
        for case in ("removal", "removal_two"):
            out = cli.evaluate_patch("STRUCTURED_REMOVE_DIRECTION", self.patch_of(case), BASE, self.committed(case))
            self.assertEqual(out["status"], SAT, out["reasons"])
        out = cli.evaluate_patch("STRUCTURED_REMOVE_DIRECTION", self.patch_of("mixed"), BASE, self.committed("mixed"))
        self.assertEqual(out["status"], CNE)

    def test_each_fixtures_pack_is_the_derivation_of_its_edited_weights(self):
        for case in CASES:
            self.assertEqual(cli.evaluate_pack("Q8_0_PACK_DERIVATION", self.committed(case))["status"], SAT, case)


# ── golden vectors for `bf16-rne-of-exact-sum`, derived here ─────────────────────────────────────────────────


def bf16_exact(bits: int) -> Fraction:
    """A finite BF16 bit pattern as an exact rational, from its fields: 1 sign, 8 exponent, 7 fraction bits."""
    sign = -1 if bits >> 15 else 1
    field, fraction = (bits >> 7) & 0xFF, bits & 0x7F
    assert field != 0xFF, "not finite"
    if field == 0:
        return sign * Fraction(fraction, 2 ** 133)  # subnormal: fraction × 2^-133
    return sign * Fraction(128 + fraction, 128) * Fraction(2) ** (field - 127)


def f32_exact(bits: int) -> Fraction:
    """A finite binary32 bit pattern as an exact rational: 1 sign, 8 exponent, 23 fraction bits."""
    sign = -1 if bits >> 31 else 1
    field, fraction = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    assert field != 0xFF, "not finite"
    if field == 0:
        return sign * Fraction(fraction, 2 ** 149)
    return sign * Fraction(2 ** 23 + fraction, 2 ** 23) * Fraction(2) ** (field - 127)


def reference_sum(old_bits: int, delta_bits: int):
    """`RNE_bf16(exact(old) + exact(delta))` by integer arithmetic on rationals: the BF16 bits, or `None` when the
    rounded magnitude is 2^128 or more. Written from the spec (§2.4), not from the contract's code."""
    old, delta = bf16_exact(old_bits), f32_exact(delta_bits)
    total = old + delta
    if total == 0:
        both_negative_zeros = old == 0 and delta == 0 and old_bits >> 15 and delta_bits >> 31
        return BF16_SIGN if both_negative_zeros else 0
    sign = BF16_SIGN if total < 0 else 0
    magnitude = abs(total)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1  # now 2^exponent <= magnitude < 2^(exponent + 1)
    exponent = max(exponent, -126)  # below the smallest normal, the quantum stays 2^-133
    quanta = magnitude / Fraction(2) ** (exponent - 7)
    whole = quanta.numerator // quanta.denominator
    rest = quanta - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    if whole == 256:  # rounded up into the next binade
        whole, exponent = 128, exponent + 1
    if exponent > 127:
        return None
    if whole < 128:  # subnormal, or zero with the sum's own sign
        return sign | whole
    return sign | ((exponent + 127) << 7) | (whole - 128)


class ExactSumGolden(unittest.TestCase):
    """(old BF16 bits, delta binary32 bits) → new BF16 bits. Each expectation is worked by hand in the comment and
    recomputed by `reference_sum`; the contract's own computation is then held to it. `None`: the sum overflows."""

    TABLE = (
        # ties at the BF16 boundary, to even
        (0x3F80, 0x3B800000, 0x3F80),  # 1 + 2^-8: halfway between 1 and 1 + 2^-7; the even one is 1
        (0x3F81, 0x3B800000, 0x3F82),  # (1 + 2^-7) + 2^-8: halfway, the even neighbour is above
        (0x3F80, 0x3B800001, 0x3F81),  # one binary32 step above the tie: up
        (0x3F80, 0x3B7FFFFF, 0x3F80),  # one step below the tie: down
        (0x3F81, 0x3B7FFFFF, 0x3F81),  # below the tie from an odd neighbour: down, where ties-to-even would go up
        (0xBF80, 0xBB800000, 0xBF80),  # the first tie, negated
        (0x3FFF, 0x3B800000, 0x4000),  # (2 - 2^-7) + 2^-8: the even neighbour is 2, in the next binade
        # signed zeros
        (0x0000, 0x00000000, 0x0000),  # +0 + +0
        (0x8000, 0x80000000, 0x8000),  # -0 + -0 is -0
        (0x8000, 0x00000000, 0x0000),  # -0 + +0 is +0: a zero delta changes a stored -0
        (0x0000, 0x80000000, 0x0000),  # +0 + -0 is +0
        (0x3F80, 0xBF800000, 0x0000),  # 1 + -1: an exact zero is +0
        (0xBF80, 0x3F800000, 0x0000),  # -1 + 1
        (0x0000, 0x80000001, 0x8000),  # +0 + -2^-149: a nonzero sum that rounds to zero keeps its sign
        (0x8000, 0x00000001, 0x0000),  # -0 + 2^-149
        # subnormals: the quantum is 2^-133 below 2^-126
        (0x0001, 0x00010000, 0x0002),  # 2^-133 + 2^-133
        (0x0001, 0x00008000, 0x0002),  # 1.5 quanta: tie, to the even 2
        (0x0002, 0x00008000, 0x0002),  # 2.5 quanta: tie, to the even 2
        (0x007F, 0x00010000, 0x0080),  # the largest subnormal plus one quantum is the smallest normal
        (0x0080, 0x80010000, 0x007F),  # and back
        (0x0000, 0x00008000, 0x0000),  # half a quantum: tie, to the even 0
        (0x0000, 0x00008001, 0x0001),  # just over half a quantum
        (0x8000, 0x80008000, 0x8000),  # -0 + -half a quantum: tie to zero, sign kept
        # a sum that overflows BF16
        (0x7F7F, 0x7B000000, None),  # the largest BF16 plus half its step: the tie rounds to 2^128
        (0x7F7F, 0x7AFFFFFF, 0x7F7F),  # one binary32 step less: stays the largest
        (0x7F7F, 0x7F7FFFFF, None),  # plus the largest binary32
        (0xFF7F, 0xFB000000, None),  # the negative tie at the top
        (0x7F00, 0x7F000000, None),  # 2^127 + 2^127
        # a delta that cancels the stored value to a tiny nonzero
        (0x3F80, 0xBF7FFFFF, 0x3380),  # 1 - (1 - 2^-24) = 2^-24
        (0x3F80, 0xBF800001, 0xB400),  # 1 - (1 + 2^-23) = -2^-23
        (0xBF80, 0x3F7FFFFF, 0xB380),  # -1 + (1 - 2^-24) = -2^-24
        (0x0080, 0x807FFFFF, 0x0000),  # 2^-126 - (2^-126 - 2^-149) = 2^-149: below half a quantum, +0
        # a sum a binary64 cannot hold: the small term decides which side of the tie it is
        (0x3F80, 0x5D808000, 0x5D81),  # 1 + 2^60(1 + 2^-8): just above the tie
        (0xBF80, 0x5D808000, 0x5D80),  # -1 + the same delta: just below
        (0x2500, 0x3F808000, 0x3F81),  # 2^-53 + (1 + 2^-8)
        # and the plain ones
        (0x3F80, 0x00000000, 0x3F80),
        (0x3F80, 0xC0000000, 0xBF80),  # 1 - 2
    )

    def contract_sum(self, old, delta):
        try:
            return W.exact_sum_to_bf16(old, delta)
        except numconv.Overflow:
            return None

    def test_the_reference_gives_the_hand_worked_values(self):
        for old, delta, want in self.TABLE:
            with self.subTest(old=f"{old:04x}", delta=f"{delta:08x}"):
                self.assertEqual(reference_sum(old, delta), want)

    def test_the_contracts_computation_gives_them_too(self):
        for old, delta, want in self.TABLE:
            with self.subTest(old=f"{old:04x}", delta=f"{delta:08x}"):
                self.assertEqual(self.contract_sum(old, delta), want)

    def test_the_contract_agrees_with_the_reference_on_seeded_pairs(self):
        """Uniform bit patterns, and pairs made to cancel, to sit in the subnormals and to reach the top."""
        rng = random.Random(20261114)

        def finite16():
            while True:
                bits = rng.getrandbits(16)
                if (bits >> 7) & 0xFF != 0xFF:
                    return bits

        def finite32():
            while True:
                bits = rng.getrandbits(32)
                if (bits >> 23) & 0xFF != 0xFF:
                    return bits

        pairs = [(finite16(), finite32()) for _ in range(1500)]
        for _ in range(1500):  # a delta within a few binary32 steps of minus the stored value
            old = finite16()
            delta = (((old ^ BF16_SIGN) << 16) + rng.randint(-4, 4)) % (1 << 32)
            if (delta >> 23) & 0xFF != 0xFF:
                pairs.append((old, delta))
        for _ in range(1000):  # both near the bottom of the range
            pairs.append((rng.getrandbits(1) << 15 | rng.randint(0, 0x0180),
                          rng.getrandbits(1) << 31 | rng.randint(0, 0x01800000)))
        for _ in range(1000):  # both near the top
            pairs.append((rng.getrandbits(1) << 15 | rng.randint(0x7E80, 0x7F7F),
                          rng.getrandbits(1) << 31 | rng.randint(0x7A000000, 0x7F7FFFFF)))
        overflows = 0
        for old, delta in pairs:
            want = reference_sum(old, delta)
            overflows += want is None
            self.assertEqual(self.contract_sum(old, delta), want, f"{old:04x} + {delta:08x}")
        self.assertGreaterEqual(overflows, 50)  # 50 at this seed: enough results past the top of BF16 to matter
        self.assertGreater(len(pairs), 4900)


class GoldenThroughTheContract(Fixture):
    """The same vectors as a base, a ledger and a compiled container: the table's stored values are written into a
    copy of the base, the deltas into a v2 ledger, and the hand-worked results into the compiled tensor."""

    TENSOR, COL = DOWN0, 3

    def build(self, rows):
        """A base whose element `(k, COL)` of `TENSOR` is row k's stored value, and a container compiled from it by
        a ledger of row k's delta, holding row k's expected bits."""
        base, root = self.copy_base(), self.copy("deltas")
        for k, (old, _, _) in enumerate(rows):
            write_element(base, self.TENSOR, k, self.COL, old)
        payload = reseal_index(base, REP)
        self.edit_index(base, lambda i: i["representations"][DERIVED].update(source_representation_digest=payload))
        # the compiled container: that base, with each row's new bits
        segment = read_json(os.path.join(base, INDEX_FILE))["representations"][REP]["segment"]
        shutil.copyfile(os.path.join(base, segment), os.path.join(root, segment))
        edits = []
        for k, (old, delta, new) in enumerate(rows):
            write_element(root, self.TENSOR, k, self.COL, new)
            edits.append({"row": k, "col": self.COL, "delta": f32_from_bits(delta), "delta_bits": delta,
                          "old_bits": old, "new_bits": new})
        entry = read_json(os.path.join(base, INDEX_FILE))["representations"][REP]

        def rebind(doc):
            doc["source"]["index_sha256"] = sha(read_bytes(os.path.join(base, INDEX_FILE)))
            doc["source"]["canonical"] = {"payload_sha256": entry["payload_sha256"],
                                          "segment_sha256": entry["segment_sha256"]}
            doc["targets"] = [dict(doc["targets"][0], tensor=self.TENSOR, edits=edits)]

        self.edit_ledger(root, rebind)
        reseal_compiled(root)
        return base, root

    def test_every_vector_that_does_not_overflow_is_satisfied_as_a_compiled_edit(self):
        rows = [row for row in ExactSumGolden.TABLE if row[2] is not None]
        self.assertGreater(len(rows), 30)
        base, root = self.build(rows)
        out = cli.evaluate_ledger(V2, base, root)
        self.assertHalves(out, SAT, SAT)
        changed = sum(old != new for old, _, new in rows)
        self.assertEqual(self.measured(out)["elements expected to change"], str(changed))

    def test_each_vector_one_step_off_is_caught(self):
        """The ledger and the container agree on a neighbour of the right result. Only the recomputation objects."""
        rows = [row for row in ExactSumGolden.TABLE if row[2] is not None]
        for k in range(len(rows)):
            old, delta, new = rows[k]
            with self.subTest(old=f"{old:04x}", delta=f"{delta:08x}"):
                wrong = new ^ 1
                base, root = self.build(rows[:k] + [(old, delta, wrong)] + rows[k + 1:])
                out = cli.evaluate_ledger(V2, base, root)
                self.assertHalves(out, VIOL, VIOL)
                self.assertIn(f"new_bits {wrong:04x} is not the recomputed {new:04x}", self.findings(out, W.LEDGER))
                self.assertIn(f"expected {new:04x}, found {wrong:04x}", self.findings(out, W.NUMERICAL))
                shutil.rmtree(base)
                shutil.rmtree(root)

    def test_a_vector_that_overflows_is_a_ledger_violation_whatever_the_container_holds(self):
        for old, delta, new in ExactSumGolden.TABLE:
            if new is not None:
                continue
            with self.subTest(old=f"{old:04x}", delta=f"{delta:08x}"):
                base, root = self.build([(old, delta, old)])
                out = cli.evaluate_ledger(V2, base, root)
                self.assertHalves(out, CNE, VIOL)
                self.assertIn("the sum overflows BF16", self.findings(out, W.LEDGER))


class FrozenRemovalVectors(unittest.TestCase):
    """The frozen `conformance/vlp-numerics/v1` vectors for `remove_direction`, read only, through the form the
    ledger route gives the op: the `.vlp` op, whose direction is under `operand`."""

    def test_every_frozen_remove_direction_case_in_its_vlp_form(self):
        with open(VECTORS_V1) as f:
            cases = [c for c in json.load(f)["cases"] if c["op"]["kind"] == "remove_direction"]
        results = refusals = 0
        for case in cases:
            op = case["op"]
            if op.get("fixups"):
                continue  # a `.vlp` op carries none
            vlp = {"op": W.OP_REMOVE, "numerics": op.get("numerics"), "operand": op.get("u"), "alpha": op.get("alpha"),
                   "columns": op.get("columns")}
            base = np.array([int(h, 16) for h in case["base"]["bf16"]], dtype=np.uint16)
            with self.subTest(case=case["id"]):
                try:
                    parsed = V.RemoveDirection(vlp, op.get("shape"), u_key="operand")
                    out = parsed.apply(base, tensor_shape=tuple(case["base"]["shape"]),
                                       tensor_role=case.get("tensor_role"))
                    got = {"bf16": [f"{int(b):04x}" for b in out.reshape(-1)]}
                except V.Refusal as e:
                    got = {"refuse": e.code}
                want = {k: v for k, v in case["expect"].items() if k in ("bf16", "refuse")}
                self.assertEqual(got, want)
                results += "bf16" in want
                refusals += "refuse" in want
        self.assertGreaterEqual(results, 10)
        self.assertGreaterEqual(refusals, 10)


# ── numerical faults ─────────────────────────────────────────────────────────────────────────────────────────


class EveryShape(Fixture):
    """The faults every shape must reject, each with the digests re-sealed. `kind` is how the faulted tensor is
    recorded: a delta target is recomputable from the ledger alone, a result-bits target only with the patch."""

    def each_target(self):
        for case in CASES:
            for target, kind in TARGETS[case]:
                yield case, target, kind

    def test_a_recorded_result_changed_together_with_the_container_element(self):
        """The ledger and the container agree on a wrong value: caught only by recomputation."""
        for case, target, kind in self.each_target():
            with self.subTest(case=case, kind=kind):
                root = self.copy(case)
                name, k, edit = self.changing_edit(root, target)
                wrong = edit["new_bits"] ^ 1
                write_element(root, name, edit["row"], edit["col"], wrong)
                self.edit_ledger(root, lambda d: d["targets"][target]["edits"][k].update(new_bits=wrong))
                reseal_compiled(root)
                out = self.judge(case, root)
                self.assertHalves(out, VIOL, VIOL)
                self.assertIn(f"new_bits {wrong:04x} is not the recomputed {edit['new_bits']:04x}",
                              self.findings(out, W.LEDGER))
                self.assertIn("differ from the recomputed tensor", self.findings(out, W.NUMERICAL))
                without = self.judge(case, root, patch=None)
                if kind == "delta":  # the ledger carries the delta, so no patch is needed to see it
                    self.assertEqual(without["ledger_integrity"], VIOL)
                    self.assertEqual(without["numerical_satisfaction"], VIOL)
                else:  # without the patch there is nothing to recompute it from: not evaluated, and not a pass
                    self.assertHalves(without, CNE, SAT)

    def test_a_container_element_changed_alone(self):
        for case, target, kind in self.each_target():
            with self.subTest(case=case, kind=kind):
                root = self.copy(case)
                name, _, edit = self.changing_edit(root, target)
                write_element(root, name, edit["row"], edit["col"], edit["new_bits"] ^ 1)
                reseal_compiled(root)
                out = self.judge(case, root)
                self.assertHalves(out, VIOL, SAT)
                self.assertIn(f"({edit['row']}, {edit['col']})", self.findings(out, W.NUMERICAL))
                self.assertIn("1 that the ops change" if kind == "result" else "1 elements differ",
                              self.findings(out, W.NUMERICAL))

    def test_a_declared_edit_that_was_not_applied(self):
        """The container holds the base's value where the ledger records a new one."""
        for case, target, kind in self.each_target():
            with self.subTest(case=case, kind=kind):
                root = self.copy(case)
                name, _, edit = self.changing_edit(root, target)
                self.assertEqual(read_element(BASE, name, edit["row"], edit["col"]), edit["old_bits"])
                write_element(root, name, edit["row"], edit["col"], edit["old_bits"])
                reseal_compiled(root)
                out = self.judge(case, root)
                self.assertHalves(out, VIOL, SAT)
                self.assertIn(f"expected {edit['new_bits']:04x}, found {edit['old_bits']:04x}",
                              self.findings(out, W.NUMERICAL))
                without = self.judge(case, root, patch=None)
                if kind == "delta":
                    self.assertEqual(without["numerical_satisfaction"], VIOL)
                else:  # held to the recorded bits under the ledger half, and still not a pass
                    self.assertHalves(without, CNE, VIOL)
                    self.assertIn("differ from the base with the ledger's recorded bits", self.findings(without, W.LEDGER))

    def test_an_element_of_an_edited_tensor_outside_the_declared_edits(self):
        for case, target, kind in self.each_target():
            with self.subTest(case=case, kind=kind):
                root = self.copy(case)
                t = self.ledger(root)["targets"][target]
                listed = {(e["row"], e["col"]) for e in t["edits"]}
                row, col = 31, 20  # no fixture edits this element, and no removal writes this column
                self.assertNotIn(col, {c for _, c in listed})
                write_element(root, t["tensor"], row, col, read_element(root, t["tensor"], row, col) ^ 1)
                reseal_compiled(root)
                out = self.judge(case, root)
                self.assertHalves(out, VIOL, SAT)
                self.assertIn(f"({row}, {col})", self.findings(out, W.NUMERICAL))
                if kind == "result":
                    self.assertIn("1 outside it", self.findings(out, W.NUMERICAL))

    def test_an_element_a_removal_leaves_alone_inside_its_columns(self):
        # `removal`: u is zero in row 2, so (2, 33) keeps the base's bits though column 33 is in the footprint
        root = self.copy("removal")
        listed = {(e["row"], e["col"]) for e in self.ledger(root)["targets"][0]["edits"]}
        self.assertNotIn((2, 33), listed)
        self.assertIn((0, 33), listed)
        write_element(root, DOWN1, 2, 33, read_element(root, DOWN1, 2, 33) ^ 1)
        reseal_compiled(root)
        out = self.judge("removal", root)
        self.assertHalves(out, VIOL, SAT)
        self.assertIn("1 inside the footprint whose expected bits are the base's", self.findings(out, W.NUMERICAL))

    def test_an_element_of_a_tensor_the_ledger_does_not_name(self):
        for case in CASES:
            with self.subTest(case=case):
                root = self.copy(case)
                write_element(root, UP0, 1, 1, read_element(root, UP0, 1, 1) ^ 1)
                reseal_compiled(root)
                out = self.judge(case, root)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn(f"{REP}/{UP0}: a tensor the ledger does not edit differs from the base in 1 elements",
                              self.findings(out, W.LEDGER))
                self.assertEqual(self.judge(case, root, patch=None)["ledger_integrity"], VIOL)

    def test_a_negative_zero_made_positive_outside_the_edits(self):
        """Bits are compared, not values."""
        for case in CASES:
            with self.subTest(case=case):
                base, root = self.copy_base(), self.copy(case)
                name = self.ledger(root)["targets"][0]["tensor"]
                for r in (base, root):
                    write_element(r, name, 31, 20, BF16_SIGN)
                payload = reseal_index(base, REP)
                self.edit_index(base, lambda i: i["representations"][DERIVED].update(source_representation_digest=payload))
                entry = read_json(os.path.join(base, INDEX_FILE))["representations"][REP]
                index_sha = sha(read_bytes(os.path.join(base, INDEX_FILE)))
                digests = {}

                def rebind(doc):
                    doc["source"]["index_sha256"] = index_sha
                    doc["source"]["canonical"] = {"payload_sha256": entry["payload_sha256"],
                                                  "segment_sha256": entry["segment_sha256"]}
                    doc["origins"] = [digests.get(o, o) for o in doc["origins"]]

                def rebind_patch(doc):
                    for op in doc["operations"]:
                        old = op["digest"]
                        op["source"].update(index_sha256=index_sha, payload_sha256=entry["payload_sha256"])
                        op["digest"] = E.delta_op_digest(op) if op["op"] == E.OP_DELTA else W.op_digest(op)
                        digests[old] = op["digest"]

                patch = self.patch_copy(case, rebind_patch)
                self.edit_ledger(root, rebind)
                reseal_compiled(root)
                self.assertHalves(self.judge(case, root, patch=patch, base=base), SAT, SAT)
                write_element(root, name, 31, 20, 0)
                reseal_compiled(root)
                out = self.judge(case, root, patch=patch, base=base)
                self.assertHalves(out, VIOL, SAT)
                self.assertIn("expected 8000, found 0000", self.findings(out, W.NUMERICAL))


# ── the ledger's own record ───────────────────────────────────────────────────────────────────────────────────


class LedgerRecord(Fixture):
    def test_an_edit_dropped_from_the_ledger(self):
        # a delta: the container still holds the changed element, which no edit now explains
        root = self.copy("deltas")
        _, k, edit = self.changing_edit(root, 0)
        self.edit_ledger(root, lambda d: d["targets"][0]["edits"].pop(k))
        out = self.judge("deltas", root, patch=None)
        self.assertHalves(out, VIOL, SAT)
        self.assertIn(f"({edit['row']}, {edit['col']})", self.findings(out, W.NUMERICAL))
        with_patch = self.judge("deltas", root)
        self.assertHalves(with_patch, VIOL, VIOL)
        self.assertIn("at elements the ledger does not record", self.findings(with_patch, W.LEDGER))
        # a result: the patch's op changes an element the ledger no longer records
        for case, target in (("removal", 0), ("removal_two", 0), ("mixed", 1)):
            with self.subTest(case=case):
                root = self.copy(case)
                self.edit_ledger(root, lambda d: d["targets"][target]["edits"].pop())
                out = self.judge(case, root)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn("the ledger does not record it", self.findings(out, W.LEDGER))

    def test_a_dropped_edit_that_changed_nothing_is_seen_only_with_the_patch(self):
        """The ledger without it is still a true description of the container. The patch is what shows it is not
        the ledger of this compile."""
        root = self.copy("deltas")
        edits = self.ledger(root)["targets"][1]["edits"]
        k = next(i for i, e in enumerate(edits) if e["old_bits"] == e["new_bits"])
        self.edit_ledger(root, lambda d: d["targets"][1]["edits"].pop(k))
        self.assertHalves(self.judge("deltas", root, patch=None), SAT, SAT)
        out = self.judge("deltas", root)
        self.assertHalves(out, SAT, VIOL)
        self.assertIn("at elements the ledger does not record", self.findings(out, W.LEDGER))

    def test_an_edit_added_to_the_ledger(self):
        # a delta the container was never given
        root = self.copy("deltas")
        old = read_element(root, DOWN0, 40, 40)
        delta_bits = 0x3F000000  # 0.5
        new = W.exact_sum_to_bf16(old, delta_bits)
        self.assertNotEqual(new, old)
        extra = {"row": 40, "col": 40, "delta": 0.5, "delta_bits": delta_bits, "old_bits": old, "new_bits": new}
        self.edit_ledger(root, lambda d: d["targets"][0]["edits"].append(extra))
        out = self.judge("deltas", root, patch=None)
        self.assertHalves(out, VIOL, SAT)
        self.assertIn("(40, 40)", self.findings(out, W.NUMERICAL))
        with_patch = self.judge("deltas", root)
        self.assertHalves(with_patch, VIOL, VIOL)
        self.assertIn("delta edit(s) of the ledger are not in the patch, first (40, 40)", self.findings(with_patch, W.LEDGER))
        # a result the patch's ops do not make
        for case, target in (("removal", 0), ("removal_two", 0), ("mixed", 1)):
            with self.subTest(case=case):
                root = self.copy(case)
                t = self.ledger(root)["targets"][target]
                self.assertNotIn((31, 20), {(e["row"], e["col"]) for e in t["edits"]})
                old = read_element(root, t["tensor"], 31, 20)
                extra = {"row": 31, "col": 20, "delta": 0.0, "delta_bits": 0, "old_bits": old, "new_bits": old,
                         "encoding": W.RESULT_BITS}
                self.edit_ledger(root, lambda d: d["targets"][target]["edits"].append(extra))
                out = self.judge(case, root)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn("the patch's ops do not make", self.findings(out, W.LEDGER))

    def test_an_added_delta_that_changes_nothing_is_seen_only_with_the_patch(self):
        root = self.copy("deltas")
        old = read_element(root, DOWN0, 40, 40)
        extra = {"row": 40, "col": 40, "delta": 0.0, "delta_bits": 0, "old_bits": old, "new_bits": old}
        self.edit_ledger(root, lambda d: d["targets"][0]["edits"].append(extra))
        self.assertHalves(self.judge("deltas", root, patch=None), SAT, SAT)
        self.assertHalves(self.judge("deltas", root), SAT, VIOL)

    def test_a_delta_whose_text_and_bits_disagree(self):
        for case in ("deltas", "mixed"):
            for delta, bits in ((0.1, TENTH_BINARY32_BITS + 1), (0.25, TENTH_BINARY32_BITS), (-0.1, TENTH_BINARY32_BITS),
                                (1e300, TENTH_BINARY32_BITS)):
                with self.subTest(case=case, delta=delta, bits=f"{bits:08x}"):
                    root = self.copy(case)
                    edit = self.ledger(root)["targets"][0]["edits"][0]
                    self.assertEqual(edit["delta_bits"], TENTH_BINARY32_BITS)
                    new = W.exact_sum_to_bf16(edit["old_bits"], bits)
                    write_element(root, DOWN0, edit["row"], edit["col"], new)
                    self.edit_ledger(root, lambda d: d["targets"][0]["edits"][0].update(
                        delta=delta, delta_bits=bits, new_bits=new))
                    reseal_compiled(root)
                    # delta_bits is the delta: the weights are right for it, and the text beside it is wrong
                    out = self.judge(case, root, patch=None)
                    self.assertEqual(out["ledger_integrity"], VIOL)
                    self.assertIn("does not read as the binary32 of delta_bits", self.findings(out, W.LEDGER))
                    if case == "deltas":
                        self.assertHalves(out, SAT, VIOL)

    def test_the_shortest_binary32_rendering_is_read_as_a_binary32(self):
        # 0.1 is not 0x3dcccccd as a binary64; an honest compile must not be reported for it
        edit = self.ledger(self.committed("deltas"))["targets"][0]["edits"][0]
        self.assertEqual((edit["delta"], edit["delta_bits"]), (0.1, TENTH_BINARY32_BITS))
        self.assertNotEqual(Fraction(0.1), W.f32_value(TENTH_BINARY32_BITS))
        self.assertTrue(W.delta_reads_as(W.DecimalFloat("0.1"), TENTH_BINARY32_BITS))
        self.assertTrue(W.delta_reads_as(W.DecimalFloat("-0.3"), MINUS_THREE_TENTHS_BINARY32_BITS))
        self.assertFalse(W.delta_reads_as(W.DecimalFloat("0.3"), MINUS_THREE_TENTHS_BINARY32_BITS))

    def test_a_wrong_old_bits(self):
        for case in CASES:
            for target, kind in TARGETS[case]:
                with self.subTest(case=case, kind=kind):
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: d["targets"][target]["edits"][0].update(
                        old_bits=d["targets"][target]["edits"][0]["old_bits"] ^ 1))
                    out = self.judge(case, root)
                    self.assertHalves(out, SAT, VIOL)
                    self.assertIn("is not the base's", self.findings(out, W.LEDGER))
                    self.assertEqual(self.judge(case, root, patch=None)["ledger_integrity"], VIOL)

    def test_a_wrong_new_bits_in_the_ledger_alone(self):
        for case in CASES:
            for target, kind in TARGETS[case]:
                with self.subTest(case=case, kind=kind):
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: d["targets"][target]["edits"][0].update(
                        new_bits=d["targets"][target]["edits"][0]["new_bits"] ^ 1))
                    out = self.judge(case, root)
                    self.assertHalves(out, SAT, VIOL)
                    self.assertIn("is not the recomputed", self.findings(out, W.LEDGER))

    def test_non_finite_delta_bits_and_a_sum_that_overflows(self):
        for bits, said in ((0x7F800000, "is not finite"), (0x7FC00000, "is not finite"),
                           (0x7F7FFFFF, "the sum overflows BF16")):
            with self.subTest(bits=f"{bits:08x}"):
                root = self.copy("deltas")
                self.edit_ledger(root, lambda d: d["targets"][0]["edits"][0].update(delta=0.0, delta_bits=bits))
                out = self.judge("deltas", root, patch=None)
                self.assertHalves(out, CNE, VIOL)
                self.assertIn(said, self.findings(out, W.LEDGER))

    def test_a_malformed_edit(self):
        changes = {
            "a missing field": lambda e: e.pop("old_bits"),
            "an extra field": lambda e: e.update(note="x"),
            "a row outside the tensor": lambda e: e.update(row=64),
            "a negative column": lambda e: e.update(col=-1),
            "bits out of range": lambda e: e.update(new_bits=1 << 16),
            "a boolean": lambda e: e.update(delta_bits=True),
            "a string delta": lambda e: e.update(delta="0.1"),
            "a fractional row": lambda e: e.update(row=1.5),
        }
        for case in CASES:
            for name, change in changes.items():
                with self.subTest(case=case, change=name):
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: change(d["targets"][0]["edits"][0]))
                    out = self.judge(case, root)
                    self.assertEqual(out["ledger_integrity"], VIOL)
                    self.assertNotEqual(out["numerical_satisfaction"], SAT)

    def test_an_element_listed_twice(self):
        for case in CASES:
            with self.subTest(case=case):
                root = self.copy(case)
                self.edit_ledger(root, lambda d: d["targets"][0]["edits"].append(dict(d["targets"][0]["edits"][0])))
                out = self.judge(case, root)
                self.assertEqual(out["ledger_integrity"], VIOL)
                self.assertIn("is edited twice", self.findings(out, W.LEDGER))
                self.assertNotEqual(out["numerical_satisfaction"], SAT)

    def test_a_tensor_listed_twice(self):
        root = self.copy("deltas")
        self.edit_ledger(root, lambda d: d["targets"].append(json.loads(json.dumps(d["targets"][0]))))
        out = self.judge("deltas", root, patch=None)
        self.assertEqual(out["ledger_integrity"], VIOL)
        self.assertIn("the subject of more than one target", self.findings(out, W.LEDGER))

    def test_a_target_that_does_not_describe_its_tensor(self):
        changes = {
            "a tensor the base does not hold": lambda t: t.update(tensor="9.mlp.down_proj.weight"),
            "another shape": lambda t: t.update(shape=[t["shape"][1], t["shape"][0]]),
            "another dtype": lambda t: t.update(dtype="Q8_0"),
            "another object": lambda t: t.update(object="target.embedding"),
            "no edits": lambda t: t.update(edits=[]),
            "no tensor": lambda t: t.pop("tensor"),
            "a field of an operation target": lambda t: t.update(changed_elements=3),
        }
        for case in CASES:
            for name, change in changes.items():
                with self.subTest(case=case, change=name):
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: change(d["targets"][0]))
                    out = self.judge(case, root)
                    self.assertEqual(out["ledger_integrity"], VIOL, self.findings(out, W.LEDGER))
                    self.assertNotEqual(out["numerical_satisfaction"], SAT)

    def test_targets_in_two_representations_or_none(self):
        root = self.copy("deltas")
        self.edit_ledger(root, lambda d: d["targets"][1].update(representation="target.embedding@BF16"))
        self.assertHalves(self.judge("deltas", root, patch=None), CNE, CNE)
        root = self.copy("deltas")
        self.edit_ledger(root, lambda d: d.update(targets=[]))
        self.assertHalves(self.judge("deltas", root, patch=None), CNE, VIOL)
        root = self.copy("deltas")
        self.edit_ledger(root, lambda d: [t.update(representation=DERIVED) for t in d["targets"]])
        self.assertHalves(self.judge("deltas", root, patch=None), CNE, VIOL)

    def test_max_patch_elements_is_enforced_and_inclusive(self):
        with mock.patch.object(W, "MAX_PATCH_ELEMENTS", 7):  # `deltas` lists seven elements
            self.assertHalves(self.judge("deltas", self.committed("deltas")), SAT, SAT)
        with mock.patch.object(W, "MAX_PATCH_ELEMENTS", 6):
            out = self.judge("deltas", self.committed("deltas"))
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("MAX_PATCH_ELEMENTS", self.findings(out, W.LEDGER))

    def test_tampered_result_digests(self):
        changes = {
            "result.canonical.payload_sha256": lambda d: d["result"]["canonical"].update(payload_sha256=ZERO_SHA),
            "result.canonical.segment_sha256": lambda d: d["result"]["canonical"].update(segment_sha256=ZERO_SHA),
            "result.derived": lambda d: d["result"]["derived"][0].update(source_representation_digest=ZERO_SHA),
            "result.derived[target.decoder_stack@Q8_0].payload_sha256":
                lambda d: d["result"]["derived"][0].update(payload_sha256=ZERO_SHA),
            "result.derived[target.decoder_stack@Q8_0].encoder":
                lambda d: d["result"]["derived"][0].update(encoder={"algorithm": "other", "revision": 9}),
            "result.derived: is not a list": lambda d: d["result"].update(derived=None),
            "result: the ledger has no result object": lambda d: d.pop("result"),
            "result.derived: names": lambda d: d["result"].update(derived=[]),
        }
        for case in CASES:
            for said, change in changes.items():
                with self.subTest(case=case, field=said):
                    root = self.copy(case)
                    self.edit_ledger(root, change)
                    out = self.judge(case, root)
                    self.assertHalves(out, SAT, VIOL)
                    self.assertIn(said, self.findings(out, W.LEDGER))


# ── the source binding ────────────────────────────────────────────────────────────────────────────────────────


class SourceBinding(Fixture):
    def test_a_ledger_bound_to_another_base(self):
        for case in CASES:
            with self.subTest(case=case):
                other = self.copy_base()
                with open(os.path.join(other, INDEX_FILE), "a") as f:
                    f.write("\n")  # the same weights, another index.json
                out = self.judge(case, self.committed(case), base=other)
                self.assertEqual(out["ledger_integrity"], VIOL)
                self.assertNotEqual(out["numerical_satisfaction"], SAT)
                self.assertEqual(out["provenance_detail"]["base_binding"], VIOL)
                self.assertIn("source.index_sha256", self.findings(out, W.LEDGER))
                self.assertHalves(self.judge(case, self.committed(case), patch=None, base=other), CNE, VIOL)

    def test_each_field_of_the_source(self):
        changes = {
            "source.index_sha256": lambda s: s.update(index_sha256=ZERO_SHA),
            "source.model": lambda s: s.update(model="another"),
            "source.canonical.payload_sha256": lambda s: s["canonical"].update(payload_sha256=ZERO_SHA),
            "source.canonical.segment_sha256": lambda s: s["canonical"].update(segment_sha256=ZERO_SHA),
            "source.derived[target.decoder_stack@Q8_0].payload_sha256":
                lambda s: s["derived"][0].update(payload_sha256=ZERO_SHA),
            "source.derived[target.decoder_stack@Q8_0].encoder": lambda s: s["derived"][0].pop("encoder"),
            "source.derived: names": lambda s: s.update(derived=[]),
            "source.derived: is not a list": lambda s: s.update(derived={}),
        }
        for case in CASES:
            for said, change in changes.items():
                with self.subTest(case=case, field=said):
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: change(d["source"]))
                    out = self.judge(case, root)
                    self.assertEqual(out["ledger_integrity"], VIOL)
                    self.assertIn(said, self.findings(out, W.LEDGER))
                    # a stale binding is refused: nothing is recomputed against another base
                    self.assertEqual(out["numerical_satisfaction"], CNE)
                    self.assertEqual(out["provenance_detail"]["base_binding"], VIOL)

    def test_a_ledger_with_no_source(self):
        root = self.copy("deltas")
        self.edit_ledger(root, lambda d: d.pop("source"))
        self.assertHalves(self.judge("deltas", root, patch=None), CNE, VIOL)

    def test_a_base_whose_weights_differ_from_the_one_the_ledger_was_written_against(self):
        """The digests are re-sealed, so the binding passes: the wrong `old_bits` and the recomputation object."""
        for case in CASES:
            with self.subTest(case=case):
                other = self.copy_base()
                name, _, edit = self.changing_edit(self.committed(case), 0)
                write_element(other, name, edit["row"], edit["col"], edit["old_bits"] ^ 0x10)
                payload = reseal_index(other, REP)
                self.edit_index(other, lambda i: i["representations"][DERIVED].update(source_representation_digest=payload))
                entry = read_json(os.path.join(other, INDEX_FILE))["representations"][REP]
                root = self.copy(case)

                def rebind(doc):
                    doc["source"]["index_sha256"] = sha(read_bytes(os.path.join(other, INDEX_FILE)))
                    doc["source"]["canonical"] = {"payload_sha256": entry["payload_sha256"],
                                                  "segment_sha256": entry["segment_sha256"]}

                self.edit_ledger(root, rebind)
                out = self.judge(case, root, patch=None, base=other)
                self.assertEqual(out["ledger_integrity"], VIOL)
                self.assertIn("is not the base's", self.findings(out, W.LEDGER))
                self.assertNotEqual(out["numerical_satisfaction"], SAT)


# ── the schema and the form ───────────────────────────────────────────────────────────────────────────────────


class SchemaAndForm(Fixture):
    def all_cases(self):
        """Every engine-compiled ledger fixture: `(name, compiled container, schema, its patch or None)`."""
        for case, contract in CASES.items():
            yield case, self.committed(case), SCHEMA_OF[contract], self.patch_of(case)
        for case in V4_CASES:
            patch = self.patch_of(case, V4_FIXTURES) if case == "split" else None
            yield f"v4/{case}", self.committed(case, V4_FIXTURES), W.LEDGER_SCHEMA_V4, patch

    def test_every_contract_against_every_ledger(self):
        """Nine fixtures under three contracts: only a ledger of the contract's own schema is judged, and no
        pairing raises."""
        for name, container, schema, patch in self.all_cases():
            for contract in (V2, V3, V4):
                for given in (None, patch):
                    with self.subTest(fixture=name, contract=contract, patch=bool(given)):
                        out = cli.evaluate_ledger(contract, BASE, container, given)
                        if SCHEMA_OF[contract] != schema:
                            self.assertHalves(out, CNE, CNE)
                            self.assertIn(repr(schema), self.findings(out, W.LEDGER))
                        elif schema == E.LEDGER_SCHEMA_V3 and not given:
                            self.assertHalves(out, CNE, SAT)
                        elif name == "v4/split" and not given:
                            self.assertHalves(out, CNE, SAT)
                        else:
                            self.assertHalves(out, SAT, SAT)

    def test_a_refusal_names_the_contract_that_judges_the_schema(self):
        out = cli.evaluate_ledger(V3, BASE, self.committed("deltas"))
        self.assertIn(f"ledger schema {E.LEDGER_SCHEMA_V2!r} is not the one {V3} judges, which is "
                      f"{E.LEDGER_SCHEMA_V3!r} only; {V2} judges that schema", self.findings(out, W.LEDGER))
        out = cli.evaluate_ledger(V2, BASE, self.committed("low_rank_k2", V4_FIXTURES))
        self.assertIn(f"; {V4} judges that schema", self.findings(out, W.LEDGER))

    def test_a_schema_no_contract_judges(self):
        for schema in ("weight-patches/v1", "weight-patches/v5", "weight-patches/v2 ", "", None, 2, ["weight-patches/v2"]):
            for case in ("deltas", "removal"):
                with self.subTest(schema=schema, case=case):
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: d.update(schema=schema))
                    self.assertHalves(self.judge(case, root), CNE, CNE)

    def test_a_ledger_relabelled_for_the_contract_asked(self):
        """The schema string is changed to the one the contract judges. The contents still belong under the schema
        the engine wrote, so the form rule objects, with or without the ledger's own patch."""
        expectations = {
            # (fixture, relabelled as): what the ledger half says, and whether the tensor of that form is evaluated
            ("removal", V2): ("result-bits edit(s): a result-bits edit needs 'weight-patches/v3'", CNE),
            ("removal_two", V2): ("result-bits edit(s): a result-bits edit needs 'weight-patches/v3'", CNE),
            ("mixed", V2): ("result-bits edit(s): a result-bits edit needs 'weight-patches/v3'", CNE),
            ("deltas", V3): ("is written only when some edit is a result-bits edit", SAT),
            ("v4/low_rank_k2", V2): ("operation target(s): an operation target needs 'weight-patches/v4'", CNE),
            ("v4/low_rank_k2", V3): ("operation target(s): an operation target needs 'weight-patches/v4'", CNE),
            ("v4/mixed", V2): ("operation target(s): an operation target needs 'weight-patches/v4'", CNE),
            ("v4/mixed", V3): ("operation target(s): an operation target needs 'weight-patches/v4'", CNE),
            ("v4/sham", V2): ("operation target(s): an operation target needs 'weight-patches/v4'", CNE),
            ("v4/f32_profile", V3): ("operation target(s): an operation target needs 'weight-patches/v4'", CNE),
            ("v4/split", V2): ("operation target(s): an operation target needs 'weight-patches/v4'", CNE),
            ("v4/split", V3): ("operation target(s): an operation target needs 'weight-patches/v4'", CNE),
        }
        fixtures = {name: (container, patch) for name, container, _, patch in self.all_cases()}
        for (name, contract), (said, numerical) in expectations.items():
            container, patch = fixtures[name]
            for given in (None, patch):
                with self.subTest(fixture=name, relabelled=contract, patch=bool(given)):
                    root = self.fresh("relabelled")
                    shutil.copytree(container, root)
                    self.edit_ledger(root, lambda d: d.update(schema=SCHEMA_OF[contract]))
                    out = cli.evaluate_ledger(contract, BASE, root, given)
                    self.assertEqual(out["ledger_integrity"], VIOL, self.findings(out, W.LEDGER))
                    self.assertIn(said, self.findings(out, W.LEDGER))
                    self.assertEqual(out["numerical_satisfaction"], numerical, self.findings(out, W.NUMERICAL))
                    self.assertEqual(out["status"], VIOL)

    def test_an_operation_target_is_not_reconstructed_under_these_contracts(self):
        """`low_rank_k2` relabelled v2: the v4 judge would recompute its low-rank op and find the tensor right. These
        contracts do not judge that form, and say so on the numerical half."""
        root = self.copy("low_rank_k2", V4_FIXTURES)
        self.assertHalves(cli.evaluate_ledger(V4, BASE, root), SAT, SAT)
        self.edit_ledger(root, lambda d: d.update(schema=E.LEDGER_SCHEMA_V2))
        out = cli.evaluate_ledger(V2, BASE, root)
        self.assertHalves(out, CNE, VIOL)
        self.assertIn(f"an operation target is not a form {V2} judges, so the tensor is not reconstructed",
                      self.findings(out, W.NUMERICAL))
        self.assertEqual(self.measured(out)["elements compared"], "0")

    def test_a_result_bits_tensor_is_not_reconstructed_under_v2_even_with_its_patch(self):
        root = self.copy("removal")
        self.edit_ledger(root, lambda d: d.update(schema=E.LEDGER_SCHEMA_V2))
        out = cli.evaluate_ledger(V2, BASE, root, self.patch_of("removal"))
        self.assertEqual((out["numerical_satisfaction"], out["ledger_integrity"]), (CNE, VIOL))
        self.assertIn(f"a result-bits edit is not a form {V2} judges", self.findings(out, W.NUMERICAL))
        self.assertEqual(self.measured(out)["result-bits targets recomputed from the patch given"], "0")

    def test_an_operation_target_grafted_into_an_element_ledger(self):
        graft = self.ledger(self.committed("low_rank_k2", V4_FIXTURES))["targets"][0]
        self.assertIn("operations", graft)
        for case in CASES:
            with self.subTest(case=case):
                root = self.copy(case)
                self.assertNotIn(graft["tensor"], [t["tensor"] for t in self.ledger(root)["targets"]]
                                 if case != "removal" else [])

                def change(d):
                    d["targets"] = [t for t in d["targets"] if t["tensor"] != graft["tensor"]] + [graft]

                self.edit_ledger(root, change)
                out = self.judge(case, root)
                self.assertEqual(out["ledger_integrity"], VIOL)
                self.assertIn("an operation target needs", self.findings(out, W.LEDGER))
                self.assertNotEqual(out["numerical_satisfaction"], SAT)

    def test_a_target_with_both_operations_and_edits(self):
        root = self.copy("deltas")
        self.edit_ledger(root, lambda d: d["targets"][0].update(operations=[]))
        out = self.judge("deltas", root)
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("an operation target needs", self.findings(out, W.LEDGER))

    def test_a_target_that_mixes_the_two_edit_forms(self):
        for case, target in (("removal", 0), ("removal_two", 0), ("mixed", 1)):
            for given in (True, None):
                with self.subTest(case=case, patch=bool(given)):
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: d["targets"][target]["edits"][0].pop("encoding"))
                    out = self.judge(case, root, patch=given)
                    self.assertHalves(out, CNE, VIOL)
                    self.assertIn("a tensor is edited by deltas or by remove_direction ops, never by both",
                                  self.findings(out, W.LEDGER))
        # and the other way: a result-bits edit among the deltas of a mixed ledger's delta target
        root = self.copy("mixed")
        self.edit_ledger(root, lambda d: d["targets"][0]["edits"][0].update(encoding=W.RESULT_BITS, delta=0.0, delta_bits=0))
        out = self.judge("mixed", root)
        self.assertEqual((out["numerical_satisfaction"], out["ledger_integrity"]), (CNE, VIOL))

    def test_a_result_bits_edit_that_carries_a_delta(self):
        for case, target in (("removal", 0), ("removal_two", 0), ("mixed", 1)):
            for name, change in (("delta_bits", {"delta_bits": 1}), ("delta", {"delta": 0.5}),
                                 ("a negative zero", {"delta": -0.0}), ("both", {"delta": 0.1, "delta_bits": TENTH_BINARY32_BITS})):
                with self.subTest(case=case, change=name):
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: d["targets"][target]["edits"][0].update(change))
                    out = self.judge(case, root)
                    self.assertHalves(out, SAT, VIOL)
                    self.assertIn("a result-bits edit carries a delta", self.findings(out, W.LEDGER))

    def test_an_edit_encoding_that_is_null_or_unknown(self):
        for case in CASES:
            root = self.copy(case)
            self.edit_ledger(root, lambda d: d["targets"][0]["edits"][0].update(encoding=None))
            out = self.judge(case, root)
            self.assertEqual(out["ledger_integrity"], VIOL, case)
            self.assertIn("with a null encoding", self.findings(out, W.LEDGER))
            for encoding in ("zz", 1, ["delta"], {}):
                with self.subTest(case=case, encoding=encoding):
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: d["targets"][0]["edits"][0].update(encoding=encoding))
                    out = self.judge(case, root)
                    self.assertNotSatisfied(out)
                    self.assertNotEqual(out["numerical_satisfaction"], SAT)
                    self.assertIn("unknown edit encoding", self.findings(out, W.LEDGER))

    def test_an_explicit_delta_encoding_is_the_absent_one(self):
        root = self.copy("deltas")
        self.edit_ledger(root, lambda d: [e.update(encoding=W.DELTA_ENCODING) for t in d["targets"] for e in t["edits"]])
        self.assertHalves(self.judge("deltas", root), SAT, SAT)

    def test_a_rounding_rule_this_contract_does_not_implement(self):
        for rounding in ("bf16-rne-of-f32-sum", "", None, 7):
            # a delta ledger: the rule is what its numerics are
            root = self.copy("deltas")
            self.edit_ledger(root, lambda d: d.update(rounding=rounding))
            self.assertHalves(self.judge("deltas", root), CNE, CNE)
            # a ledger of removals alone: no delta is recomputed, and the ledger is still not one this contract reads
            root = self.copy("removal")
            self.edit_ledger(root, lambda d: d.update(rounding=rounding))
            out = self.judge("removal", root)
            self.assertHalves(out, SAT, CNE)
            self.assertEqual(out["status"], CNE)
            self.assertIn("rounding rule", self.findings(out, W.LEDGER))
        root = self.copy("removal")
        self.edit_ledger(root, lambda d: d.pop("rounding"))
        self.assertHalves(self.judge("removal", root), SAT, CNE)

    def test_origins_that_are_not_a_list_of_digests(self):
        for origins in ("abc", None, 5, {}, [5], ["zz"], [ZERO_SHA.upper().replace("0", "A")], [[ZERO_SHA]], [None]):
            for case in ("deltas", "removal"):
                with self.subTest(origins=origins, case=case):
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: d.update(origins=origins))
                    out = self.judge(case, root, patch=None)
                    self.assertEqual(out["ledger_integrity"], VIOL)
                    self.assertIn("origins is not a list of SHA-256 hex strings", self.findings(out, W.LEDGER))
                    self.assertNotSatisfied(self.judge(case, root))


# ── origins and the patch ─────────────────────────────────────────────────────────────────────────────────────


class OriginsAndPatch(Fixture):
    """Spec §7 steps 7 and 8: a patch given is held to the ledger in full."""

    def op_of(self, case, kind, fixtures=FIXTURES):
        return next(op for op in read_json(self.patch_of(case, fixtures))["operations"] if op["op"] == kind)

    def reseal(self, op):
        op["digest"] = E.delta_op_digest(op) if op["op"] == E.OP_DELTA else W.op_digest(op)
        return op["digest"]

    def test_emptied_origins(self):
        for case in CASES:
            for change in (lambda d: d.update(origins=[]), lambda d: d.pop("origins")):
                with self.subTest(case=case):
                    root = self.copy(case)
                    self.edit_ledger(root, change)
                    out = self.judge(case, root)
                    self.assertEqual(out["ledger_integrity"], VIOL)
                    self.assertIn("not among the ledger's origins", self.findings(out, W.LEDGER))
                    if case != "deltas":  # a removal that is not an origin is not what the tensor is recomputed from
                        self.assertEqual(out["numerical_satisfaction"], CNE)
                        self.assertEqual(self.measured(out)["result-bits targets recomputed from the patch given"], "0")

    def test_emptied_origins_of_a_delta_ledger_are_not_seen_without_the_patch(self):
        """The stated limit (spec §6): a `weight_delta` digest cannot be recomputed from the ledger."""
        root = self.copy("deltas")
        self.edit_ledger(root, lambda d: d.pop("origins"))
        out = self.judge("deltas", root, patch=None)
        self.assertHalves(out, SAT, SAT)
        self.assertIn("the ledger's origins when no patch is given", self.measured(out)["not established"])

    def test_an_origin_the_patch_does_not_hold(self):
        for case in CASES:
            with self.subTest(case=case):
                root = self.copy(case)
                self.edit_ledger(root, lambda d: d["origins"].append("ab" * 32))
                out = self.judge(case, root)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn("origins are not the digest of a weight op of the patch given", self.findings(out, W.LEDGER))

    def test_the_patch_of_another_fixture(self):
        for case in CASES:
            for other in CASES:
                if other == case:
                    continue
                with self.subTest(container=case, patch=other):
                    out = self.judge(case, self.committed(case), patch=self.patch_of(other))
                    self.assertEqual(out["ledger_integrity"], VIOL, self.findings(out, W.LEDGER))
                    self.assertIn("not the", self.findings(out, W.LEDGER))
                    self.assertEqual(out["status"], VIOL)
                    if CASES[case] == V3:  # no removal of this container's was given: nothing recomputed from it
                        self.assertEqual(out["numerical_satisfaction"], CNE)
            with self.subTest(container=case, patch="v4/split"):
                out = self.judge(case, self.committed(case), patch=self.patch_of("split", V4_FIXTURES))
                self.assertEqual(out["ledger_integrity"], VIOL)
                self.assertIn("add_low_rank_columns op(s)", self.findings(out, W.LEDGER))

    def test_a_delta_ledger_given_a_patch_with_a_removal(self):
        """A v2 compile has no removal: the patch is not the one this container was compiled from."""
        removal = self.op_of("removal", W.OP_REMOVE)
        for name, patch in (
                ("its own deltas and a removal", self.patch_copy("deltas", lambda d: d["operations"].append(removal))),
                ("a removal alone", self.patch_of("removal")),
                ("a removal on a tensor it edits", self.patch_copy("deltas", lambda d: d["operations"].append(
                    self.op_of("removal_two", W.OP_REMOVE))))):
            with self.subTest(patch=name):
                out = self.judge("deltas", self.committed("deltas"), patch=patch)
                self.assertEqual(out["ledger_integrity"], VIOL)
                self.assertEqual(out["status"], VIOL)
                self.assertIn("the ledger records no result-bits target for this tensor", self.findings(out, W.LEDGER))
        # even when the ledger is made to list the removal among its origins
        root = self.copy("deltas")
        self.edit_ledger(root, lambda d: d["origins"].append(removal["digest"]))
        out = self.judge("deltas", root, patch=self.patch_copy("deltas", lambda d: d["operations"].append(removal)))
        self.assertHalves(out, SAT, VIOL)

    def test_a_patch_with_a_low_rank_op(self):
        low_rank = self.op_of("split", W.OP_LOW_RANK, V4_FIXTURES)
        for case in CASES:
            with self.subTest(case=case):
                patch = self.patch_copy(case, lambda d: d["operations"].append(low_rank))
                out = self.judge(case, self.committed(case), patch=patch)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn("the patch holds 1 add_low_rank_columns op(s)", self.findings(out, W.LEDGER))
                # even when the ledger lists it among its origins
                root = self.copy(case)
                self.edit_ledger(root, lambda d: d["origins"].append(low_rank["digest"]))
                self.assertHalves(self.judge(case, root, patch=patch), SAT, VIOL)

    def test_a_patch_op_on_a_tensor_the_ledger_does_not_name(self):
        for case in CASES:
            root = self.copy(case)
            named = [t["tensor"] for t in self.ledger(root)["targets"]]
            self.assertNotIn(UP0, named)
            delta = dict(self.op_of("deltas", E.OP_DELTA), tensor=UP0)
            removal = dict(self.op_of("removal", W.OP_REMOVE), tensor=UP0, columns=[1, 2],
                           operand=self.op_of("removal", W.OP_REMOVE)["operand"] * 4)
            for op in (delta, removal):
                with self.subTest(case=case, op=op["op"]):
                    digest = self.reseal(op)
                    patch = self.patch_copy(case, lambda d: d["operations"].append(op))
                    self.edit_ledger(root, lambda d: d["origins"].append(digest))  # the tamperer lists it too
                    out = self.judge(case, root, patch=patch)
                    self.assertHalves(out, SAT, VIOL)
                    self.assertIn(UP0, self.findings(out, W.LEDGER))
                    self.assertIn("the ledger records no", self.findings(out, W.LEDGER))
                    self.edit_ledger(root, lambda d: d["origins"].pop())

    def test_a_patch_with_an_extra_removal(self):
        for case, target in (("removal", 0), ("removal_two", 0), ("mixed", 1)):
            tensor = self.ledger(self.committed(case))["targets"][target]["tensor"]
            own = next(op for op in read_json(self.patch_of(case))["operations"] if op["op"] == W.OP_REMOVE)
            cols = 256 if tensor != O_PROJ else 64
            extra = dict(own, columns=[cols - 2, cols - 3][::-1])
            digest = self.reseal(extra)
            patch = self.patch_copy(case, lambda d: d["operations"].append(extra))
            with self.subTest(case=case, origins="as compiled"):
                out = self.judge(case, self.committed(case), patch=patch)
                self.assertHalves(out, CNE, VIOL)
                self.assertIn("not among the ledger's origins", self.findings(out, W.LEDGER))
            with self.subTest(case=case, origins="made to list it"):
                # the op is admitted, and recomputing with it gives a tensor the container does not hold
                root = self.copy(case)
                self.edit_ledger(root, lambda d: d["origins"].append(digest))
                out = self.judge(case, root, patch=patch)
                self.assertHalves(out, VIOL, VIOL)
                self.assertIn("the ledger does not record it", self.findings(out, W.LEDGER))
                self.assertIn("differ from the recomputed tensor", self.findings(out, W.NUMERICAL))

    def test_a_patch_with_a_missing_removal(self):
        # one of the two a tensor needs: what is recomputed is not what the container holds
        for keep in (0, 1):
            with self.subTest(kept=keep):
                patch = self.patch_copy("removal_two", lambda d: d.update(operations=[d["operations"][keep]]))
                out = self.judge("removal_two", self.committed("removal_two"), patch=patch)
                self.assertHalves(out, VIOL, VIOL)
                self.assertIn("the ledger records an edit the patch's ops do not make", self.findings(out, W.LEDGER))
                self.assertIn("origins are not the digest of a weight op of the patch given", self.findings(out, W.LEDGER))
                self.assertIn("differ from the recomputed tensor", self.findings(out, W.NUMERICAL))
        # the only one: nothing to recompute the tensor from
        for case in ("removal", "removal_two", "mixed"):
            with self.subTest(case=case):
                patch = self.patch_copy(case, lambda d: d.update(
                    operations=[op for op in d["operations"] if op["op"] != W.OP_REMOVE]))
                out = self.judge(case, self.committed(case), patch=patch)
                self.assertHalves(out, CNE, VIOL)
                self.assertIn("holds no remove_direction op on this tensor", self.findings(out, W.NUMERICAL))
                self.assertIn("origins are not the digest of a weight op of the patch given", self.findings(out, W.LEDGER))

    def test_a_removal_from_the_patch_that_is_damaged_or_rewritten(self):
        for case in ("removal", "removal_two", "mixed"):
            def removal(doc):
                return next(op for op in doc["operations"] if op["op"] == W.OP_REMOVE)

            damaged = self.patch_copy(case, lambda d: removal(d).update(alpha="3fd0000000000000"))
            out = self.judge(case, self.committed(case), patch=damaged)
            self.assertHalves(out, CNE, VIOL)
            self.assertIn("digest is not intact", self.findings(out, W.LEDGER))
            self.assertEqual(out["provenance_detail"]["op_digest"], VIOL)

            def rescale(doc):
                removal(doc).update(alpha="3fd0000000000000")
                self.reseal(removal(doc))

            out = self.judge(case, self.committed(case), patch=self.patch_copy(case, rescale))
            self.assertHalves(out, CNE, VIOL)
            self.assertIn("not among the ledger's origins", self.findings(out, W.LEDGER))

            def rebound(doc):
                removal(doc)["source"].update(index_sha256=ZERO_SHA)
                self.reseal(removal(doc))

            out = self.judge(case, self.committed(case), patch=self.patch_copy(case, rebound))
            self.assertHalves(out, CNE, VIOL)
            self.assertIn("stale or foreign source binding (index_sha256)", self.findings(out, W.LEDGER))

    def test_a_rewritten_removal_the_ledger_is_made_to_list(self):
        """The whole chain re-sealed around another scale: the op is intact, bound and an origin. The recomputation
        is what is left."""
        for case, target in (("removal", 0), ("mixed", 1)):
            with self.subTest(case=case):
                digests = {}

                def rescale(doc):
                    op = next(op for op in doc["operations"] if op["op"] == W.OP_REMOVE)
                    old = op["digest"]
                    op.update(alpha="3fd0000000000000")  # 0.25 for 0.5
                    digests[old] = self.reseal(op)

                patch = self.patch_copy(case, rescale)
                root = self.copy(case)
                self.edit_ledger(root, lambda d: d.update(origins=[digests.get(o, o) for o in d["origins"]]))
                out = self.judge(case, root, patch=patch)
                self.assertHalves(out, VIOL, VIOL)
                self.assertIn("is not the recomputed", self.findings(out, W.LEDGER))
                self.assertIn("differ from the recomputed tensor", self.findings(out, W.NUMERICAL))

    def test_overlapping_removals_in_the_patch(self):
        def overlap(doc):
            doc["operations"][1].update(columns=[11, 200])
            self.reseal(doc["operations"][1])

        digests = [op["digest"] for op in read_json(self.patch_copy("removal_two", overlap))["operations"]]
        root = self.copy("removal_two")
        self.edit_ledger(root, lambda d: d.update(origins=digests))
        out = self.judge("removal_two", root, patch=self.patch_copy("removal_two", overlap))
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("pairwise disjoint columns", self.findings(out, W.LEDGER))

    # -- the patch's delta ops, held to the ledger's delta edits --------------------------------------------------
    def delta_cases(self):
        return ("deltas", "mixed")

    def test_a_patch_delta_that_is_damaged_stale_or_not_an_origin(self):
        for case in self.delta_cases():
            changes = {
                "digest is not intact": lambda op: op["edits"][0].update(delta_bits=op["edits"][0]["delta_bits"] + 1),
                # A different delta at the same element, resealed: an intact op that no compile of this ledger ran.
                "not among the ledger's origins": lambda op: (op["edits"][0].update(delta_bits=op["edits"][0]["delta_bits"] + 1),
                                                              self.reseal(op)),
                "stale or foreign source binding (payload_sha256)": lambda op: (op["source"].update(payload_sha256=ZERO_SHA),
                                                                                self.reseal(op)),
                "the weight_delta op names no tensor": lambda op: op.update(tensor=7),
                "the weight_delta op has no edits": lambda op: op.update(edits=[]),
                "malformed edit": lambda op: op["edits"][0].update(row="5"),
                "malformed edit ": lambda op: op["edits"][0].update(old_bits=0),
                "is edited twice": lambda op: op["edits"].append(dict(op["edits"][0])),
                "does not read as a finite binary32 of its delta_bits": lambda op: op["edits"][0].update(delta=0.25),
            }
            for said, change in changes.items():
                with self.subTest(case=case, defect=said):
                    patch = self.patch_copy(case, lambda d: change(d["operations"][0]))
                    out = self.judge(case, self.committed(case), patch=patch)
                    self.assertHalves(out, SAT, VIOL)
                    self.assertIn(said.strip(), self.findings(out, W.LEDGER))

    def test_a_patch_delta_this_contract_cannot_read(self):
        for case in self.delta_cases():
            for said, change in (("unsupported op fields ['note']", lambda op: op.update(note="x")),
                                 ("rounding rule", lambda op: op.update(rounding="other"))):
                with self.subTest(case=case, defect=said):
                    patch = self.patch_copy(case, lambda d: change(d["operations"][0]))
                    out = self.judge(case, self.committed(case), patch=patch)
                    self.assertHalves(out, SAT, CNE)
                    self.assertEqual(out["status"], CNE)
                    self.assertIn(said, self.findings(out, W.LEDGER))

    def test_a_patch_whose_deltas_are_not_the_ledgers(self):
        """Each time the tamperer re-seals the op and lists its new digest among the ledger's origins, so that the
        comparison of the deltas themselves is what objects."""
        for case in self.delta_cases():
            changes = {
                "differ from the ledger's at the same element": lambda op: op["edits"][0].update(
                    delta=0.5, delta_bits=0x3F000000),
                "at elements the ledger does not record": lambda op: op["edits"].append(
                    {"row": 40, "col": 40, "delta": 0.5, "delta_bits": 0x3F000000}),
                "delta edit(s) of the ledger are not in the patch": lambda op: op.update(edits=op["edits"][1:] or [
                    {"row": 5, "col": 12, "delta": 0.1, "delta_bits": TENTH_BINARY32_BITS}][:0] or op["edits"][:0]),
            }
            for said, change in changes.items():
                with self.subTest(case=case, defect=said):
                    digests = {}

                    def rewrite(doc):
                        ops = doc["operations"]
                        # the op with the most edits, so that dropping one leaves an op with edits
                        op = max((o for o in ops if o["op"] == E.OP_DELTA), key=lambda o: len(o["edits"]))
                        old = op["digest"]
                        change(op)
                        if not op["edits"]:
                            ops.remove(op)
                            digests[old] = None
                        else:
                            digests[old] = self.reseal(op)

                    patch = self.patch_copy(case, rewrite)
                    root = self.copy(case)
                    self.edit_ledger(root, lambda d: d.update(
                        origins=[digests.get(o, o) for o in d["origins"] if digests.get(o, o) is not None]))
                    out = self.judge(case, root, patch=patch)
                    self.assertHalves(out, SAT, VIOL)
                    self.assertIn(said, self.findings(out, W.LEDGER))

    def test_a_patch_missing_a_delta_op(self):
        for case in self.delta_cases():
            with self.subTest(case=case):
                patch = self.patch_copy(case, lambda d: d["operations"].pop(0))
                out = self.judge(case, self.committed(case), patch=patch)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn("delta edit(s) of the ledger are not in the patch", self.findings(out, W.LEDGER))
                self.assertIn("origins are not the digest of a weight op of the patch given", self.findings(out, W.LEDGER))

    def test_a_patch_with_a_delta_op_twice(self):
        for case in self.delta_cases():
            with self.subTest(case=case):
                patch = self.patch_copy(case, lambda d: d["operations"].append(dict(d["operations"][0])))
                out = self.judge(case, self.committed(case), patch=patch)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn("are also edited by an earlier weight_delta op of the patch", self.findings(out, W.LEDGER))

    def test_a_patch_delta_on_a_tensor_the_ledger_records_by_result_bits(self):
        ledger = self.ledger(self.committed("mixed"))
        result_tensor = ledger["targets"][1]["tensor"]
        op = dict(self.op_of("mixed", E.OP_DELTA), tensor=result_tensor)
        digest = self.reseal(op)
        root = self.copy("mixed")
        self.edit_ledger(root, lambda d: d["origins"].append(digest))
        out = self.judge("mixed", root, patch=self.patch_copy("mixed", lambda d: d["operations"].append(op)))
        self.assertHalves(out, SAT, VIOL)
        self.assertIn("the ledger records no delta edit of this tensor", self.findings(out, W.LEDGER))

    def test_an_empty_patch(self):
        for case in CASES:
            with self.subTest(case=case):
                out = self.judge(case, self.committed(case), patch=self.patch_copy(case, lambda d: d.update(operations=[])))
                self.assertEqual(out["ledger_integrity"], VIOL)
                self.assertIn("origins are not the digest of a weight op of the patch given", self.findings(out, W.LEDGER))
                self.assertEqual(out["status"], VIOL)

    # -- a patch that cannot be used -------------------------------------------------------------------------------
    def unusable(self, case):
        def raw(text):
            path = self.fresh("raw") + ".vlp"
            with open(path, "wb") as f:
                f.write(text)
            return path

        link = self.fresh("link") + ".vlp"
        os.symlink(self.patch_of(case), link)
        return {
            "not JSON": (raw(b'{"operations": ['), "the patch given cannot be used"),
            "a duplicate key": (raw(b'{"operations": [], "operations": []}'), "duplicate JSON key"),
            "nested too deeply": (raw(b"[" * DEEP + b"]" * DEEP), "nested more deeply"),
            "not an object": (raw(b"[]"), "no operations list"),
            "no operations": (self.patch_copy(case, lambda d: d.pop("operations")), "no operations list"),
            "an entry that is not an object": (self.patch_copy(case, lambda d: d["operations"].append("x")),
                                               "is not an object"),
            "a knowledge op": (self.patch_copy(case, lambda d: d["operations"].append({"op": "insert", "layer": 1})),
                               "which is not a stored-weight op this contract reads"),
            "an op of no kind": (self.patch_copy(case, lambda d: d["operations"].append({})),
                                 "which is not a stored-weight op this contract reads"),
            "no such file": (os.path.join(self.work, "absent.vlp"), "the patch must be a file"),
            "a directory": (self.work, "the patch must be a file"),
            "a symlink": (link, "symlink"),
        }

    def test_a_patch_that_cannot_be_used_is_never_a_pass(self):
        for case in CASES:
            for name, (patch, said) in self.unusable(case).items():
                with self.subTest(case=case, patch=name):
                    out = self.judge(case, self.committed(case), patch=patch)
                    # the ledger is judged without it; evidence was asked for and could not be used
                    self.assertHalves(out, CNE, SAT)
                    self.assertIn(said, " ".join(out["reasons"]))

    def test_a_patch_never_weakens_what_the_ledger_alone_establishes(self):
        for case in CASES:
            root = self.copy(case)
            write_element(root, UP0, 1, 1, read_element(root, UP0, 1, 1) ^ 1)
            reseal_compiled(root)
            self.assertEqual(self.judge(case, root, patch=None)["ledger_integrity"], VIOL)
            for name, (patch, _) in self.unusable(case).items():
                with self.subTest(case=case, patch=name):
                    out = self.judge(case, root, patch=patch)
                    self.assertEqual(out["status"], VIOL)
                    self.assertIn("a tensor the ledger does not edit differs", self.findings(out, W.LEDGER))

    def test_the_v4_route_reads_only_the_removals_of_a_patch(self):
        """What the spec (§3.3) says of `WEIGHT_PATCHES_V4`, which this change leaves as it was: a `weight_delta`
        op in the patch is not compared with the ledger. The v4 `mixed` fixture has a delta target."""
        compiled = self.committed("mixed", V4_FIXTURES)
        foreign = dict(self.op_of("deltas", E.OP_DELTA), tensor=UP0)
        self.reseal(foreign)
        patch = self.fresh("v4") + ".vlp"
        write_json(patch, {"operations": [foreign]})
        self.assertHalves(cli.evaluate_ledger(V4, BASE, compiled, patch), SAT, SAT)


# ── nothing else differs: spec §5 ─────────────────────────────────────────────────────────────────────────────


class AllowedDifferences(Fixture):
    """Spec §5: the exhaustive list of what may differ between the base and a compiled container. A wrong value in
    an allowed place is caught, or the place is deliberately not judged and the test says by whom it is."""

    def each(self):
        for case in CASES:
            yield case, self.copy(case)

    def assertLedgerViolation(self, case, root, said, numerical=SAT):
        out = self.judge(case, root)
        self.assertHalves(out, numerical, VIOL)
        self.assertIn(said, self.findings(out, W.LEDGER))
        return out

    # D1: the edited segment ------------------------------------------------------------------------------------
    def test_d1_the_edited_segments_tensor_table(self):
        def reorder(tensors):
            i, j = position_of(tensors, GATE0), position_of(tensors, UP0)
            tensors[i], tensors[j] = tensors[j], tensors[i]

        def rename(tensors):
            tensors[position_of(tensors, UP0)][0]["name"] = UP0 + ".renamed"

        def reshape(tensors):
            tensors[position_of(tensors, UP0)][0]["shape"] = [64, 256]

        def drop(tensors):
            del tensors[position_of(tensors, UP0)]

        def add(tensors):
            tensors.append([{"name": "zz.extra.weight", "dtype": "BF16", "shape": [2, 2]}, b"\0" * 8])

        changes = {"reordered": (reorder, f"{REP} segment header"), "renamed": (rename, "absent from the compiled container"),
                   "reshaped": (reshape, f"{REP} segment header"), "dropped": (drop, "absent from the compiled container"),
                   "added": (add, f"{REP} segment header")}
        for name, (change, said) in changes.items():
            for case, root in self.each():
                with self.subTest(case=case, tensor=name):
                    restructure_segment(root, REP, change)
                    reseal_compiled(root)
                    C.open_container(root)  # the container is structurally valid: only the judgement objects
                    out = self.assertLedgerViolation(case, root, said)
                    self.assertIn("the edited segment's header differs from the base's", self.findings(out, W.LEDGER))
                    if name in ("dropped", "added"):
                        self.assertIn(f"index.json representations.{REP}.tensor_count", self.findings(out, W.LEDGER))

    def test_d1_a_reordering_leaves_every_tensor_its_own_bytes(self):
        """So nothing but the comparison of the table itself can see it."""
        root = self.copy("deltas")

        def reorder(tensors):
            i, j = position_of(tensors, GATE0), position_of(tensors, UP0)
            tensors[i], tensors[j] = tensors[j], tensors[i]

        before = {t.name: sha(read_bytes(os.path.join(root, "segments", "target.decoder_stack.bin"))[s:s + t.len])
                  for t, s in ((t, tensor_location(root, REP, t.name)[1])
                               for t in C.open_container(root).segments[REP].tensors)}
        restructure_segment(root, REP, reorder)
        reseal_compiled(root)
        after = {t.name: sha(read_bytes(os.path.join(root, "segments", "target.decoder_stack.bin"))[s:s + t.len])
                 for t, s in ((t, tensor_location(root, REP, t.name)[1])
                              for t in C.open_container(root).segments[REP].tensors)}
        self.assertEqual(before, after)
        out = self.judge("deltas", root)
        self.assertHalves(out, SAT, VIOL)
        self.assertEqual([o["name"] for o in out["verdicts"][W.LEDGER]["offending"]], [f"{REP} segment header"])

    def test_d1_an_edited_tensor_renamed_or_reshaped(self):
        for case in CASES:
            name = self.ledger(self.committed(case))["targets"][0]["tensor"]

            def rename(tensors):
                tensors[position_of(tensors, name)][0]["name"] = name + ".renamed"

            def reshape(tensors):
                entry = tensors[position_of(tensors, name)][0]
                entry["shape"] = entry["shape"][::-1]

            for label, change, numerical, said in (("renamed", rename, VIOL, "the compiled container has no such tensor"),
                                                   ("reshaped", reshape, CNE, "or the compiled tensor's shape")):
                with self.subTest(case=case, tensor=label):
                    root = self.copy(case)
                    restructure_segment(root, REP, change)
                    reseal_compiled(root)
                    out = self.judge(case, root)
                    self.assertEqual((out["numerical_satisfaction"], out["ledger_integrity"]), (numerical, VIOL))
                    self.assertIn(said, self.findings(out, W.LEDGER))

    def test_d1_the_same_header_in_other_bytes(self):
        """The header of the edited segment is the base's bytes, not only the base's table."""
        for case, root in self.each():
            with self.subTest(case=case):
                restructure_segment(root, REP, lambda tensors: None)  # the same table, written by another JSON writer
                reseal_compiled(root)
                self.assertLedgerViolation(case, root, "the edited segment's header differs from the base's")

    # D2 and D4: digests in index.json ---------------------------------------------------------------------------
    def test_d2_d4_a_wrong_digest_in_the_index_is_a_container_that_does_not_agree_with_itself(self):
        for rep in (REP, DERIVED):
            for field in ("payload_sha256", "segment_sha256"):
                for case, root in self.each():
                    with self.subTest(case=case, rep=rep, field=field):
                        self.edit_index(root, lambda i: i["representations"][rep].update({field: ZERO_SHA}))
                        out = self.judge(case, root)
                        self.assertHalves(out, CNE, CNE)
                        self.assertIn("fails the validating reader", " ".join(out["reasons"]))

    def test_d2_d4_the_ledgers_copy_of_a_digest(self):
        for case, root in self.each():
            with self.subTest(case=case):
                self.edit_ledger(root, lambda d: d["result"]["canonical"].update(payload_sha256=ZERO_SHA))
                self.assertLedgerViolation(case, root, "result.canonical.payload_sha256")
        for case, root in self.each():
            with self.subTest(case=case):
                self.edit_ledger(root, lambda d: d["result"]["derived"][0].update(segment_sha256=ZERO_SHA))
                self.assertLedgerViolation(case, root, f"result.derived[{DERIVED}].segment_sha256")

    # D3: the re-derived segment -----------------------------------------------------------------------------------
    def corrupt_pack(self, root):
        """Change one stored byte of the re-derived pack and re-seal every digest that names it."""
        seg, start, entry = tensor_location(root, DERIVED, GATE0)
        with open(seg.path, "r+b") as f:
            f.seek(start + 5)
            byte = f.read(1)
            f.seek(start + 5)
            f.write(bytes([byte[0] ^ 0x55]))
        reseal_derived(root)

    def test_d3_the_packs_bytes_are_not_judged_here_and_are_judged_by_the_pack_contract(self):
        for case, root in self.each():
            with self.subTest(case=case):
                self.corrupt_pack(root)
                out = self.judge(case, root)
                self.assertHalves(out, SAT, SAT)  # chained by digest only
                self.assertIn("Q8_0_PACK_DERIVATION judges them", self.measured(out)["not established"])
                pack = cli.evaluate_pack("Q8_0_PACK_DERIVATION", root)
                self.assertEqual(pack["status"], VIOL)

    def test_d3_the_packs_bytes_without_their_digests(self):
        root = self.copy("deltas")
        seg, start, _ = tensor_location(root, DERIVED, GATE0)
        with open(seg.path, "r+b") as f:
            f.seek(start + 5)
            f.write(b"\x7f")
        self.assertHalves(self.judge("deltas", root), CNE, CNE)

    def test_d3_the_packs_tensor_table_and_header(self):
        def rename(tensors):
            tensors[position_of(tensors, UP0)][0]["name"] = UP0 + ".renamed"

        def reorder(tensors):
            i, j = position_of(tensors, GATE0), position_of(tensors, UP0)
            tensors[i], tensors[j] = tensors[j], tensors[i]

        for label, change, said in (("renamed", rename, "the re-derived representation's tensor table differs"),
                                    ("reordered", reorder, "the re-derived representation's tensor table differs"),
                                    ("the same table in other bytes", lambda tensors: None,
                                     "the re-derived segment's header differs from the base's")):
            for case, root in self.each():
                with self.subTest(case=case, pack=label):
                    restructure_segment(root, DERIVED, change)
                    reseal_derived(root)
                    self.assertLedgerViolation(case, root, said)

    # D5: the source digest of the re-derived representation ------------------------------------------------------
    def test_d5_the_source_representation_digest(self):
        for case, root in self.each():
            with self.subTest(case=case, where="index.json"):
                self.edit_index(root, lambda i: i["representations"][DERIVED].update(source_representation_digest=ZERO_SHA))
                self.assertLedgerViolation(case, root, "source_representation_digest is not the compiled canonical payload")
        for case, root in self.each():
            with self.subTest(case=case, where="the base's value, kept"):
                stale = read_json(os.path.join(BASE, INDEX_FILE))["representations"][DERIVED]["source_representation_digest"]
                self.edit_index(root, lambda i: i["representations"][DERIVED].update(source_representation_digest=stale))
                self.assertLedgerViolation(case, root, "source_representation_digest is not the compiled canonical payload")
        for case, root in self.each():
            with self.subTest(case=case, where="the ledger"):
                self.edit_ledger(root, lambda d: d["result"]["derived"][0].update(source_representation_digest=ZERO_SHA))
                self.assertLedgerViolation(case, root, "does not derive from the compiled canonical payload")

    # D6 and D7: the ledger pointer and the ledger file -----------------------------------------------------------
    def test_d6_a_pointer_that_names_no_usable_ledger(self):
        for pointer in (None, 5, ["weight_patches.json"], "absent.json", "../weight_patches.json", "/etc/hosts", ""):
            for case, root in self.each():
                with self.subTest(case=case, pointer=pointer):
                    self.edit_index(root, lambda i: i.update(weight_patches=pointer))
                    self.assertHalves(self.judge(case, root), CNE, CNE)
        for case, root in self.each():
            with self.subTest(case=case, pointer="removed"):
                self.edit_index(root, lambda i: i.pop("weight_patches"))
                self.assertHalves(self.judge(case, root), CNE, CNE)

    def test_d6_a_ledger_under_another_name_is_still_one_new_file(self):
        for case, root in self.each():
            with self.subTest(case=case):
                os.rename(os.path.join(root, LEDGER_FILE), os.path.join(root, "ledger.json"))
                self.edit_index(root, lambda i: i.update(weight_patches="ledger.json"))
                self.assertHalves(self.judge(case, root), SAT, SAT)
                # and the file left behind under the old name is a file the base does not have
                shutil.copyfile(os.path.join(root, "ledger.json"), os.path.join(root, LEDGER_FILE))
                self.assertLedgerViolation(case, root, f"{LEDGER_FILE}: present in only one of the containers")

    def test_d6_a_ledger_that_stands_where_the_base_holds_a_file(self):
        """The pointer may not turn a file of the base into "the ledger": that file would be replaced unseen."""
        for case in CASES:
            with self.subTest(case=case):
                base, root = self.copy_base(), self.copy(case)
                for r in (base, root):
                    with open(os.path.join(r, "tokenizer.json"), "w") as f:
                        f.write('{"model": "wsp"}')
                self.assertHalves(self.judge(case, root, base=base), SAT, SAT)  # carried unchanged: fine
                os.replace(os.path.join(root, LEDGER_FILE), os.path.join(root, "tokenizer.json"))
                self.edit_index(root, lambda i: i.update(weight_patches="tokenizer.json"))
                out = self.judge(case, root, base=base)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn("tokenizer.json: the ledger's path is a file of the base", self.findings(out, W.LEDGER))

    def test_d7_a_ledger_that_is_not_a_json_object(self):
        texts = {"not JSON": b'{"schema": ', "a duplicate key": None, "an array": b"[]", "a number": b"7",
                 "a non-finite constant": b'{"schema": NaN}', "not UTF-8": b"\xff\xfe"}
        for case in CASES:
            for name, text in texts.items():
                with self.subTest(case=case, ledger=name):
                    root = self.copy(case)
                    path = os.path.join(root, LEDGER_FILE)
                    if text is None:
                        text = read_bytes(path).replace(b'"schema"', b'"schema": "weight-patches/v9", "schema"', 1)
                    with open(path, "wb") as f:
                        f.write(text)
                    out = self.judge(case, root)
                    self.assertNotSatisfied(out)
                    self.assertEqual(out["numerical_satisfaction"], CNE)
                    if name != "not UTF-8":
                        self.assertEqual(out["ledger_integrity"], VIOL)

    def test_d7_a_ledger_nested_too_deeply_to_read(self):
        for case in CASES:
            root = self.copy(case)
            with open(os.path.join(root, LEDGER_FILE), "wb") as f:
                f.write(b"[" * DEEP + b"]" * DEEP)
            out = self.judge(case, root)
            self.assertHalves(out, CNE, CNE)
            self.assertIn("nested more deeply", " ".join(out["reasons"]))

    # D8: candidate.json ---------------------------------------------------------------------------------------------
    def test_d8_candidate_json_is_not_judged(self):
        """Deliberately: no verification contract answers for it, and the answer says it is not judged."""
        for case in CASES:
            for name in ("changed", "removed", "not JSON"):
                with self.subTest(case=case, candidate=name):
                    root = self.copy(case)
                    path = os.path.join(root, CANDIDATE)
                    if name == "removed":
                        os.remove(path)
                    else:
                        with open(path, "wb") as f:
                            f.write(b"{" if name == "not JSON" else read_bytes(path).replace(b"1", b"2"))
                    out = self.judge(case, root)
                    self.assertHalves(out, SAT, SAT)
                    self.assertIn("candidate.json", self.measured(out)["not established"])
            with self.subTest(case=case, candidate="absent from the base"):
                base = self.copy_base()
                os.remove(os.path.join(base, CANDIDATE))
                self.assertHalves(self.judge(case, self.committed(case), base=base), SAT, SAT)

    def test_d8_only_the_top_level_candidate_json(self):
        root = self.copy("deltas")
        shutil.copyfile(os.path.join(root, CANDIDATE), os.path.join(root, "segments", CANDIDATE))
        out = self.judge("deltas", root)
        self.assertHalves(out, SAT, VIOL)
        self.assertIn(f"segments/{CANDIDATE}: present in only one", self.findings(out, W.LEDGER))

    # D9: index.json's bytes ---------------------------------------------------------------------------------------
    def test_d9_the_index_in_other_bytes_with_the_same_values(self):
        for case, root in self.each():
            with self.subTest(case=case):
                path = os.path.join(root, INDEX_FILE)
                # Read before opening for writing: "w" truncates the file it would read.
                index = read_json(path)
                with open(path, "w") as f:
                    json.dump(index, f, sort_keys=True, separators=(",", ":"))
                self.assertHalves(self.judge(case, root), SAT, SAT)

    # everything that is not on the list --------------------------------------------------------------------------
    def test_unlisted_index_values(self):
        changes = {
            "index.json extra": lambda i: i.update(extra={"note": 1}),
            "index.json hidden_size": lambda i: i.update(hidden_size=i["hidden_size"] + 1),
            "index.json model": lambda i: i.update(model="other"),
            "index.json version": lambda i: i.pop("family"),
            "index.json precision_map.encoding": lambda i: i["precision_map"].update(encoding="Q4_K"),
            "index.json segments": lambda i: i["segments"].update({"segments/extra": 1}),
            f"index.json representations.{REP}.extra": lambda i: i["representations"][REP].update(extra=1),
            f"index.json representations.{NORM}.extra": lambda i: i["representations"][NORM].update(extra=1),
            f"index.json representations.{NORM}.source_representation_digest":
                lambda i: i["representations"][NORM].update(source_representation_digest=ZERO_SHA),
            f"index.json representations.{DERIVED}.encoder.revision":
                lambda i: i["representations"][DERIVED]["encoder"].update(revision=3),
            f"index.json representations.{DERIVED}.codec.revision":
                lambda i: i["representations"][DERIVED]["codec"].update(note="x"),
            f"index.json representations.{DERIVED}.extra": lambda i: i["representations"][DERIVED].update(extra=1),
            "index.json hidden_size ": lambda i: i.update(hidden_size=float(i["hidden_size"])),  # the type
        }
        for said, change in changes.items():
            for case, root in self.each():
                with self.subTest(case=case, change=said):
                    self.edit_index(root, change)
                    said_as = "index.json family" if "version" in said else said.strip()
                    if "codec" in said:  # the reader refuses a codec that is not exactly the Q8_0 identity
                        self.assertHalves(self.judge(case, root), CNE, CNE)
                    else:
                        self.assertLedgerViolation(case, root, said_as.rsplit(".extra", 1)[0] if False else said_as)

    def test_a_changed_field_of_an_unedited_representations_entry(self):
        for case, root in self.each():
            with self.subTest(case=case):
                self.edit_index(root, lambda i: i["representations"][NORM].update(tensor_count=2))
                # the entry no longer describes the segment it names: whole-file digest still agrees, the field does not
                self.assertLedgerViolation(case, root, f"index.json representations.{NORM}.tensor_count")

    def test_an_unedited_representation_altered(self):
        for case, root in self.each():
            with self.subTest(case=case):
                write_element(root, "weight", 0, 3, read_element(root, "weight", 0, 3, NORM) ^ 1, NORM)
                self.assertHalves(self.judge(case, root), CNE, CNE)  # un-sealed: the file is not its index entry's
                reseal_index(root, NORM)
                out = self.assertLedgerViolation(case, root, "segments/target.final_norm.bin: a file outside the edited "
                                                             "and re-derived segments differs from the base")
                self.assertIn(f"index.json representations.{NORM}.payload_sha256", self.findings(out, W.LEDGER))

    def test_a_carried_file_changed_removed_or_added(self):
        """A tokenizer or a template is no part of an edit: it is carried byte for byte."""
        for name in ("tokenizer.json", "chat_template.jinja", "knn_store.bin"):
            for case in CASES:
                base, root = self.copy_base(), self.copy(case)
                for r in (base, root):
                    with open(os.path.join(r, name), "wb") as f:
                        f.write(b'{"bos": "<s>"}')
                self.assertHalves(self.judge(case, root, base=base), SAT, SAT)
                with self.subTest(case=case, file=name, change="changed"):
                    with open(os.path.join(root, name), "wb") as f:
                        f.write(b'{"bos": "<S>"}')
                    out = self.judge(case, root, base=base)
                    self.assertHalves(out, SAT, VIOL)
                    self.assertIn(f"{name}: a file outside the edited and re-derived segments differs",
                                  self.findings(out, W.LEDGER))
                with self.subTest(case=case, file=name, change="removed"):
                    os.remove(os.path.join(root, name))
                    out = self.judge(case, root, base=base)
                    self.assertHalves(out, SAT, VIOL)
                    self.assertIn(f"{name}: present in only one of the containers", self.findings(out, W.LEDGER))
                with self.subTest(case=case, file=name, change="added"):
                    out = self.judge(case, self.copy(case), base=base)  # the base has it, this container never did
                    self.assertHalves(out, SAT, VIOL)
                    with open(os.path.join(root, name), "wb") as f:
                        f.write(b"new")
                    out = self.judge(case, root)  # this container has it, the base never did
                    self.assertHalves(out, SAT, VIOL)
                    self.assertIn(f"{name}: present in only one of the containers", self.findings(out, W.LEDGER))

    def test_the_system_graph_changed(self):
        for case, root in self.each():
            with self.subTest(case=case):
                path = os.path.join(root, "system_graph.json")
                graph = read_json(path)
                graph["note"] = "changed"
                write_json(path, graph)
                self.assertLedgerViolation(case, root, "system_graph.json: a file outside the edited and re-derived "
                                                       "segments differs")

    def test_a_segment_file_added_or_removed(self):
        for case, root in self.each():
            with self.subTest(case=case, segment="added, in no index entry"):
                shutil.copyfile(os.path.join(root, "segments", "target.final_norm.bin"),
                                os.path.join(root, "segments", "target.extra.bin"))
                self.assertLedgerViolation(case, root, "segments/target.extra.bin: present in only one of the containers")
        for case, root in self.each():
            with self.subTest(case=case, segment="added, with an index entry"):
                shutil.copyfile(os.path.join(root, "segments", "target.final_norm.bin"),
                                os.path.join(root, "segments", "target.extra.bin"))

                def add(i):
                    i["representations"]["target.output_head@Q8_0"] = dict(
                        i["representations"]["target.output_head@BF16"], encoding="Q8_0")

                self.edit_index(root, add)
                self.assertNotSatisfied(self.judge(case, root))
        for case, root in self.each():
            with self.subTest(case=case, segment="removed, its index entry kept"):
                os.remove(os.path.join(root, "segments", "target.final_norm.bin"))
                out = self.judge(case, root)
                self.assertHalves(out, CNE, CNE)  # the container is not intact: nothing is judged, and it is not a pass
        for case, root in self.each():
            with self.subTest(case=case, segment="removed, with its index entry"):
                os.remove(os.path.join(root, "segments", "target.final_norm.bin"))
                self.edit_index(root, lambda i: i["representations"].pop(NORM))
                out = self.assertLedgerViolation(case, root, "segments/target.final_norm.bin: present in only one")
                self.assertIn(f"index.json representations.{NORM}", self.findings(out, W.LEDGER))

    def test_the_set_of_re_derived_representations(self):
        for case, root in self.each():
            with self.subTest(case=case):
                self.edit_index(root, lambda i: i["representations"][DERIVED].pop("compiled_from"))
                out = self.judge(case, root)
                self.assertEqual(out["ledger_integrity"], VIOL)
                self.assertIn("derived representations: compiled container re-derives []", self.findings(out, W.LEDGER))

    def test_a_directory_in_only_one_container(self):
        for case in CASES:
            with self.subTest(case=case, where="the compiled container"):
                root = self.copy(case)
                os.mkdir(os.path.join(root, "extra"))
                self.assertLedgerViolation(case, root, "extra/: a directory present in only one of the containers")
            with self.subTest(case=case, where="the base"):
                base = self.copy_base()
                os.makedirs(os.path.join(base, "segments", "nested"))
                out = self.judge(case, self.committed(case), base=base)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn("segments/nested/: a directory present in only one", self.findings(out, W.LEDGER))

    def test_a_symlink_in_either_container(self):
        for case in CASES:
            for where in ("compiled", "base"):
                for kind in ("file", "directory", "a segment replaced by a link to itself"):
                    with self.subTest(case=case, where=where, link=kind):
                        base, root = self.copy_base(), self.copy(case)
                        target = root if where == "compiled" else base
                        if kind == "file":
                            os.symlink(os.path.join(BASE, INDEX_FILE), os.path.join(target, "linked.json"))
                        elif kind == "directory":
                            os.symlink(os.path.join(BASE, "segments"), os.path.join(target, "linked"))
                        else:
                            segment = os.path.join(target, "segments", "target.final_norm.bin")
                            kept = self.fresh("kept") + ".bin"
                            shutil.move(segment, kept)
                            os.symlink(kept, segment)
                        out = self.judge(case, root, base=base)
                        self.assertHalves(out, CNE, CNE)
                        self.assertIn("symlink", " ".join(out["reasons"]))


# ── fail closed, and never raise ──────────────────────────────────────────────────────────────────────────────


class NeverRaisesNeverPasses(Fixture):
    JUNK = (None, True, 0, -1, 1.5, 2 ** 70, "", "x", [], [[]], {}, {"x": 1}, [None], "weight-patches/v4")

    def places(self, doc):
        """Every place of a ledger that is judged: `(name, container, key)`."""
        yield from (("ledger." + k, doc, k) for k in ("schema", "rounding", "source", "targets", "result"))
        yield from (("source." + k, doc["source"], k) for k in doc["source"])
        yield from (("source.canonical." + k, doc["source"]["canonical"], k) for k in doc["source"]["canonical"])
        yield from (("result." + k, doc["result"], k) for k in doc["result"])
        yield from (("result.canonical." + k, doc["result"]["canonical"], k) for k in doc["result"]["canonical"])
        yield from (("result.derived[0]." + k, doc["result"]["derived"][0], k) for k in doc["result"]["derived"][0])
        yield from (("source.derived[0]." + k, doc["source"]["derived"][0], k) for k in doc["source"]["derived"][0])
        for n, target in enumerate(doc["targets"]):
            yield from ((f"targets[{n}]." + k, target, k) for k in target)
            yield from ((f"targets[{n}].edits[0]." + k, target["edits"][0], k) for k in target["edits"][0])
        yield ("targets[0]", doc["targets"], 0)
        yield ("targets[0].edits[0]", doc["targets"][0]["edits"], 0)

    def test_junk_in_any_judged_place_of_a_ledger(self):
        """Each judged value replaced by each of fourteen values of the wrong kind, and removed: no answer is a
        pass and none raises. (`writer` is reported, not judged.)"""
        tried = 0
        for case in CASES:
            pristine = self.ledger(self.committed(case))
            names = [name for name, _, _ in self.places(json.loads(json.dumps(pristine)))]
            root = self.copy(case)
            for name in names:
                for junk in self.JUNK + ("<removed>",):
                    doc = json.loads(json.dumps(pristine))
                    holder, key = next((h, k) for n, h, k in self.places(doc) if n == name)
                    if junk == "<removed>":
                        if isinstance(holder, list):
                            continue
                        del holder[key]
                    elif holder[key] == junk and (type(holder[key]) is type(junk) or numbers_alike(holder[key], junk)):
                        continue  # not a change: JSON has one number type, and 0 is the 0.0 a writer wrote
                    else:
                        holder[key] = junk
                    write_json(os.path.join(root, LEDGER_FILE), doc)
                    tried += 1
                    for given in (True, None):
                        with self.subTest(case=case, place=name, junk=junk, patch=bool(given)):
                            self.assertNotSatisfied(self.judge(case, root, patch=given))
        self.assertGreater(tried, 2000)

    def test_junk_in_any_place_of_a_patch_op(self):
        tried = 0
        for case in CASES:
            pristine = read_json(self.patch_of(case))
            for n, op in enumerate(pristine["operations"]):
                places = [(k, None) for k in op] + [("source", k) for k in op["source"]]
                if op["op"] == E.OP_DELTA:
                    places += [("edits", k) for k in op["edits"][0]]
                for outer, inner in places:
                    for junk in self.JUNK + ("<removed>",):
                        doc = json.loads(json.dumps(pristine))
                        target = doc["operations"][n]
                        holder, key = (target, outer) if inner is None else \
                            (target[outer][0] if outer == "edits" else target[outer], inner)
                        if junk == "<removed>":
                            del holder[key]
                        elif holder[key] == junk and type(holder[key]) is type(junk):
                            continue
                        else:
                            holder[key] = junk
                        path = os.path.join(self.work, "junk.vlp")
                        write_json(path, doc)
                        tried += 1
                        with self.subTest(case=case, op=n, place=(outer, inner), junk=junk):
                            self.assertNotSatisfied(self.judge(case, self.committed(case), patch=path))
        self.assertGreater(tried, 1500)

    def test_inputs_that_are_not_containers(self):
        for case, contract in CASES.items():
            for base, container in ((BASE, os.path.join(self.work, "absent")), (os.path.join(self.work, "absent"), BASE),
                                    (BASE, self.patch_of(case)), (BASE, BASE), (self.committed(case), self.committed(case)),
                                    (self.committed(case), BASE)):
                with self.subTest(case=case, base=os.path.basename(base), container=os.path.basename(container)):
                    out = cli.evaluate_ledger(contract, base, container, self.patch_of(case))
                    self.assertEqual(out["status"], CNE)
                    self.assertTrue(out["reasons"])

    def test_an_index_that_cannot_be_read(self):
        for text in (b"{", b"[" * DEEP + b"]" * DEEP, b'{"version": 4, "version": 4}', b"null"):
            for where in ("compiled", "base"):
                with self.subTest(index=text[:12], where=where):
                    base, root = self.copy_base(), self.copy("deltas")
                    with open(os.path.join(root if where == "compiled" else base, INDEX_FILE), "wb") as f:
                        f.write(text)
                    out = self.judge("deltas", root, base=base)
                    self.assertHalves(out, CNE, CNE)

    def test_the_inputs_must_not_change_during_the_judgement(self):
        root = self.copy("deltas")
        evaluate = E.evaluate

        def touch(*args, **kwargs):
            out = evaluate(*args, **kwargs)
            with open(os.path.join(root, CANDIDATE), "ab") as f:
                f.write(b" ")
            return out

        with mock.patch.object(E, "evaluate", touch):
            out = self.judge("deltas", root)
        self.assertEqual(out["status"], CNE)
        self.assertTrue(any(r.startswith("an input changed during the evaluation") for r in out["reasons"]), out["reasons"])
        self.assertEqual((out["numerical_satisfaction"], out["ledger_integrity"]), (CNE, CNE))


if __name__ == "__main__":
    unittest.main()
