"""`WEIGHT_PATCHES_V4` against the four engine-compiled fixtures in `conformance/weight_patches_v4/`, and fault injection.

Every fault is injected into a copy of a fixture made inside the test. A "careful tamperer" re-seals what a plain
checksum would notice (segment and payload digests in index.json, the ledger's result digests, result_tensor_sha256,
operands_sha256), so the only thing left to object is the verifier's own recomputation; each test asserts which half
of the verdict fails.

The committed compiled containers differ from the base in `candidate.json`, which §7 step 4 allows and does not judge.
"""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from larql_contract_verifier import cli  # noqa: E402
from larql_contract_verifier.contracts import vlp_numerics as V  # noqa: E402
from larql_contract_verifier.contracts import weight_patches_v4 as W  # noqa: E402
from refcheck import container as C  # noqa: E402

FIXTURES = os.path.join(ROOT, "conformance", "weight_patches_v4")
BASE = os.path.join(ROOT, "conformance", "structured_patch", "base")
CASES = ("low_rank_k2", "mixed", "sham", "f32_profile")
REP = "target.decoder_stack@BF16"
DERIVED = "target.decoder_stack@Q8_0"
LEDGER_FILE = "weight_patches.json"
CANDIDATE = "candidate.json"
DOWN1 = "1.mlp.down_proj.weight"
NEG_ZERO, POS_ZERO = 0x8000, 0x0000
SAT, VIOL, CNE = "satisfied", "violated", "could_not_evaluate"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    with open(path) as f:
        return json.load(f)


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


def tensor_bytes(root, name, rep=REP):
    seg, start, e = tensor_location(root, rep, name)
    with open(seg.path, "rb") as f:
        f.seek(start)
        return f.read(e.len)


def reseal_index(root, rep):
    """Make index.json agree with a representation's segment bytes again."""
    c = C.open_container(root, verify_digests=False)
    seg = c.segments[rep]
    with open(seg.path, "rb") as f:
        data = f.read()
    index = read_json(os.path.join(root, "index.json"))
    entry = index["representations"][rep]
    entry["segment_sha256"] = sha(data)
    entry["payload_sha256"] = sha(data[seg.payload_start:seg.payload_start + seg.payload_bytes])
    write_json(os.path.join(root, "index.json"), index)
    return entry["payload_sha256"]


def reseal_compiled(root):
    """A careful tamperer: index digests of the edited representation, the derived representation's source digest (in
    the index and the ledger), the ledger's result digests and every result_tensor_sha256, all made to agree with the
    bytes now in the container."""
    payload = reseal_index(root, REP)
    index = read_json(os.path.join(root, "index.json"))
    index["representations"][DERIVED]["source_representation_digest"] = payload
    write_json(os.path.join(root, "index.json"), index)
    ledger_path = os.path.join(root, LEDGER_FILE)
    ledger = read_json(ledger_path)
    entry = index["representations"][REP]
    ledger["result"]["canonical"] = {"payload_sha256": entry["payload_sha256"], "segment_sha256": entry["segment_sha256"]}
    for d in ledger["result"]["derived"]:
        d["source_representation_digest"] = payload
    for t in ledger["targets"]:
        if "operations" in t:
            t["result_tensor_sha256"] = sha(tensor_bytes(root, t["tensor"]))
    write_json(ledger_path, ledger)


def reseal_compiled_index_only(root):
    """Make the index and the ledger's result digests agree with the container again after the LEDGER FILE'S TEXT
    was rewritten by hand. Nothing here parses the ledger, so a number no JSON writer would emit survives."""
    reseal_index(root, REP)


def operand_sha(op, key):
    digits = 8 if op["numerics"] == V.PROFILE_F32 else 16
    field = "operand" if key == "u" else key
    values = [op[field]] if key == "alpha" else op[field]
    bits = V.parse_hex(values, digits, key)
    return sha(V.operand_digest_bytes(bits))


def rehash_operands(op):
    """A careful tamperer re-seals the operand digests and the op digest (§3a is not a signature)."""
    keys = ("u", "alpha") if op["op"] == W.OP_REMOVE else ("a", "b")
    op["operands_sha256"] = {k: operand_sha(op, k) for k in keys}
    op["digest"] = W.op_digest(op)


#: A verdict on one small fixture takes well under a second; this is the bound that separates it from a stall.
ANSWER_AT_ONCE_SECONDS = 5.0
#: 0.1 rounded to binary32, which an engine prints as `0.1`.
TENTH_BINARY32_BITS = 0x3DCCCCCD
BINARY32_INFINITY_BITS = 0x7F800000
BINARY32_NAN_BITS = 0x7FC00000


class Fixture(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work)

    def copy(self, case, normalise=False):
        self.copies = getattr(self, "copies", 0) + 1
        root = os.path.join(self.work, f"{case}-{self.copies}")
        shutil.copytree(os.path.join(FIXTURES, case, "compiled"), root)
        if normalise:
            shutil.copyfile(os.path.join(BASE, CANDIDATE), os.path.join(root, CANDIDATE))
        return root

    def judge(self, container, base=BASE):
        return cli.evaluate_ledger(W.CONTRACT, base, container)

    def ledger(self, root):
        return read_json(os.path.join(root, LEDGER_FILE))

    def edit_ledger(self, root, change):
        doc = self.ledger(root)
        change(doc)
        write_json(os.path.join(root, LEDGER_FILE), doc)

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

    def findings(self, out, half):
        return " | ".join(f"{o['name']}: {o['detail']}" for o in out["verdicts"][half]["offending"]) + " | " + \
            " | ".join(out["verdicts"][half]["reasons"])


class Untampered(Fixture):
    def test_each_committed_fixture_verifies_on_both_halves(self):
        for case in CASES:
            with self.subTest(case=case):
                out = self.judge(os.path.join(FIXTURES, case, "compiled"))
                self.assertHalves(out, SAT, SAT)
                measured = {m["name"]: m["value"] for m in out["measurements"]}
                self.assertEqual(measured["sham targets (every op zero-valued)"], "1" if case == "sham" else "0")
                ops = sum(len(t.get("operations", [])) for t in self.ledger(os.path.join(FIXTURES, case, "compiled"))
                          ["targets"])
                self.assertEqual(measured["op digests recomputed and intact (§3a)"], str(ops))
                self.assertEqual(out["provenance_detail"], {"base_binding": SAT, "op_digest": SAT,
                                                            "operand_origin": "not_checked"})
                self.assertNotIn("op digest", next(m["value"] for m in out["measurements"]
                                                   if m["name"] == "not established"))

    def test_every_fixture_op_digest_is_the_section_3a_text(self):
        for case in CASES:
            for t in self.ledger(os.path.join(FIXTURES, case, "compiled"))["targets"]:
                for op in t.get("operations", []):
                    self.assertEqual(W.op_digest(op), op["digest"], (case, op["op"]))

    def test_a_byte_changed_in_candidate_json_alone_is_not_a_violation(self):
        root = self.copy("low_rank_k2")
        path = os.path.join(root, CANDIDATE)
        with open(path, "r+b") as f:
            f.seek(10)
            b = f.read(1)
            f.seek(10)
            f.write(bytes([b[0] ^ 1]))
        self.assertHalves(self.judge(root), SAT, SAT)

    def test_the_answer_is_bound_to_the_verifier_and_keeps_engine_reports_apart(self):
        out = self.judge(self.copy("mixed"))
        self.assertEqual(out["schema"], cli.LEDGER_SCHEMA)
        self.assertEqual(out["verifier"]["code_sha256"], cli.code_sha256())
        self.assertNotIn("writer", out["verifier"])
        self.assertEqual(out["engine_reported"]["writer"], self.ledger(os.path.join(FIXTURES, "mixed", "compiled"))["writer"])
        self.assertEqual(out["engine_reported"]["schema"], W.LEDGER_SCHEMA_V4)

    def test_the_sham_compiles_to_the_bases_exact_bytes(self):
        root = self.copy("sham")
        self.assertEqual(tensor_bytes(root, DOWN1), tensor_bytes(BASE, DOWN1))
        out = self.judge(root)
        self.assertHalves(out, SAT, SAT)

    def test_a_graph_object_with_several_source_bindings_is_read(self):
        """Gemma 3's vision objects bind several tensor prefixes; refcheck's Qwen3 profile refuses that, this contract's
        reader does not (it is not a framing rule)."""
        base = os.path.join(self.work, "base")
        shutil.copytree(BASE, base)
        root = self.copy("low_rank_k2")
        for r in (base, root):
            path = os.path.join(r, "system_graph.json")
            graph = read_json(path)
            head = next(o for o in graph["objects"] if o["id"] == "target.output_head")
            head["source_bindings"].append(dict(head["source_bindings"][0], tensor_prefix="extra.prefix"))
            write_json(path, graph)
        with self.assertRaises(C.ContainerError):
            C.open_container(root)
        self.assertHalves(self.judge(root, base), SAT, SAT)

    def test_the_uncompiled_base_has_no_ledger_and_is_not_judged(self):
        self.assertHalves(self.judge(BASE), CNE, CNE)

    def test_the_cli_writes_one_json_object(self):
        root = self.copy("low_rank_k2")
        env = dict(os.environ, PYTHONPATH=ROOT)
        proc = subprocess.run([sys.executable, "-m", "larql_contract_verifier.cli", "ledger", "--contract", W.CONTRACT,
                               "--base", BASE, "--container", root], capture_output=True, text=True, env=env, cwd=ROOT,
                              check=True)
        answer = json.loads(proc.stdout)
        self.assertEqual(answer["status"], SAT)

    def test_an_unknown_contract_is_not_judged(self):
        out = cli.evaluate_ledger("OTHER", BASE, self.copy("sham"))
        self.assertEqual(out["status"], CNE)


class NumericalFaults(Fixture):
    """Faults in the weights, with the ledger re-sealed around them: only the numerical half can object."""

    def flip(self, case, name, row, col, mask=1):
        root = self.copy(case)
        write_element(root, name, row, col, read_element(root, name, row, col) ^ mask)
        reseal_compiled(root)
        return root

    def test_a_flipped_bit_in_a_written_element_inside_the_footprint(self):
        compiled = os.path.join(FIXTURES, "low_rank_k2", "compiled")
        before = np.frombuffer(tensor_bytes(BASE, DOWN1), "<u2").reshape(64, 256)
        after = np.frombuffer(tensor_bytes(compiled, DOWN1), "<u2").reshape(64, 256)
        row, col = (int(x) for x in np.argwhere(before != after)[0])  # an element the op changed
        out = self.judge(self.flip("low_rank_k2", DOWN1, row, col))
        self.assertHalves(out, VIOL, SAT)
        self.assertIn("1 elements differ", self.findings(out, W.NUMERICAL))
        self.assertIn("1 that the ops change", self.findings(out, W.NUMERICAL))
        self.assertIn(f"({row}, {col})", self.findings(out, W.NUMERICAL))

    def test_a_changed_element_outside_the_footprint(self):
        out = self.judge(self.flip("low_rank_k2", DOWN1, 7, 6))
        self.assertHalves(out, VIOL, SAT)
        self.assertIn("1 outside it", self.findings(out, W.NUMERICAL))

    def test_an_element_the_v2_selection_leaves_untouched_inside_the_footprint(self):
        # row 9 of `a` is all zero, so every element of row 9 in J keeps its base bits
        out = self.judge(self.flip("low_rank_k2", DOWN1, 9, 40))
        self.assertHalves(out, VIOL, SAT)
        self.assertIn("1 inside the footprint whose expected bits are the base's", self.findings(out, W.NUMERICAL))

    def test_the_f32_profile_and_the_mixed_target_are_checked_element_by_element(self):
        for case, name, row, col in (("f32_profile", "1.self_attn.o_proj.weight", 3, 63),
                                     ("mixed", DOWN1, 0, 33), ("mixed", DOWN1, 5, 8)):
            with self.subTest(case=case, at=(row, col)):
                self.assertHalves(self.judge(self.flip(case, name, row, col)), VIOL, SAT)

    def test_a_delta_target_element_changed_in_the_container(self):
        root = self.flip("mixed", "0.mlp.down_proj.weight", 5, 12)
        self.assertHalves(self.judge(root), VIOL, SAT)

    def test_a_sham_whose_tensor_changed(self):
        # re-sealing result_tensor_sha256 to the changed bytes also breaks the sham's result = base rule
        out = self.judge(self.flip("sham", DOWN1, 4, 11))
        self.assertHalves(out, VIOL, VIOL)
        self.assertIn("(4, 11)", self.findings(out, W.NUMERICAL))
        self.assertIn("a sham (every op zero-valued) must record", self.findings(out, W.LEDGER))

    def test_a_reordered_a_rehashed(self):
        root = self.copy("low_rank_k2")

        def change(doc):
            op = doc["targets"][0]["operations"][0]
            k = op["rank"]
            rows = [op["a"][r * k:(r + 1) * k] for r in range(len(op["a"]) // k)]
            op["a"] = [h for row in reversed(rows) for h in row]
            rehash_operands(op)

        self.edit_ledger(root, change)
        out = self.judge(root)
        self.assertEqual(out["numerical_satisfaction"], VIOL, out["verdicts"])

    def test_b_with_its_columns_reordered_rehashed(self):
        root = self.copy("low_rank_k2")

        def change(doc):
            op = doc["targets"][0]["operations"][0]
            n = len(op["columns"])
            op["b"] = [op["b"][k * n + (n - 1 - p)] for k in range(op["rank"]) for p in range(n)]
            rehash_operands(op)

        self.edit_ledger(root, change)
        self.assertEqual(self.judge(root)["numerical_satisfaction"], VIOL)


class NegativeZero(Fixture):
    """The base has no -0 in the edited tensor, so one is put into a copy of the base (and the compiled copy) at an
    element outside J and at an untouched element inside J, and the ledger is re-bound to that base. The control is
    satisfied; a -0 to +0 flip at either element is then a numerical violation."""

    OUTSIDE, INSIDE = (2, 100), (9, 3)

    def setUp(self):
        super().setUp()
        self.base = os.path.join(self.work, "base")
        shutil.copytree(BASE, self.base)
        self.root = self.copy("low_rank_k2")
        for root in (self.base, self.root):
            for row, col in (self.OUTSIDE, self.INSIDE):
                write_element(root, DOWN1, row, col, NEG_ZERO)
        payload = reseal_index(self.base, REP)
        base_index_sha = sha(open(os.path.join(self.base, "index.json"), "rb").read())
        segment = read_json(os.path.join(self.base, "index.json"))["representations"][REP]["segment_sha256"]
        reseal_compiled(self.root)
        base_tensor = sha(tensor_bytes(self.base, DOWN1))

        def rebind(doc):
            doc["source"]["index_sha256"] = base_index_sha
            doc["source"]["canonical"] = {"payload_sha256": payload, "segment_sha256": segment}
            for t in doc["targets"]:
                t["base_tensor_sha256"] = base_tensor
                for op in t["operations"]:
                    op["source"].update(index_sha256=base_index_sha, payload_sha256=payload)
                    op["digest"] = W.op_digest(op)  # re-binding an op is a new op: its digest is re-sealed

        self.edit_ledger(self.root, rebind)

    def test_the_control_with_negative_zeros_is_satisfied(self):
        self.assertHalves(self.judge(self.root, self.base), SAT, SAT)

    def test_a_negative_zero_flipped_to_positive_zero_outside_the_footprint(self):
        write_element(self.root, DOWN1, *self.OUTSIDE, POS_ZERO)
        reseal_compiled(self.root)
        out = self.judge(self.root, self.base)
        self.assertHalves(out, VIOL, SAT)
        self.assertIn("expected 8000, found 0000", self.findings(out, W.NUMERICAL))
        self.assertIn("1 outside it", self.findings(out, W.NUMERICAL))

    def test_a_negative_zero_flipped_to_positive_zero_untouched_inside_the_footprint(self):
        write_element(self.root, DOWN1, *self.INSIDE, POS_ZERO)
        reseal_compiled(self.root)
        out = self.judge(self.root, self.base)
        self.assertHalves(out, VIOL, SAT)
        self.assertIn("1 inside the footprint", self.findings(out, W.NUMERICAL))


class LedgerFaults(Fixture):
    """Faults in the ledger or outside the edited tensors: the ledger-integrity half objects."""

    def test_a_changed_element_in_another_tensor(self):
        root = self.copy("low_rank_k2")
        write_element(root, "0.mlp.up_proj.weight", 1, 1, read_element(root, "0.mlp.up_proj.weight", 1, 1) ^ 1)
        reseal_compiled(root)
        out = self.judge(root)
        self.assertHalves(out, SAT, VIOL)
        self.assertIn("0.mlp.up_proj.weight", self.findings(out, W.LEDGER))

    def test_a_changed_byte_in_a_segment_the_compile_does_not_touch(self):
        root = self.copy("low_rank_k2")
        write_element(root, "weight", 0, 3, read_element(root, "weight", 0, 3, "target.final_norm@BF16") ^ 1,
                      "target.final_norm@BF16")
        unsealed = self.judge(root)
        self.assertEqual(unsealed["status"], CNE)  # the reader refuses a segment that disagrees with its digests
        reseal_index(root, "target.final_norm@BF16")
        out = self.judge(root)
        self.assertHalves(out, SAT, VIOL)
        found = self.findings(out, W.LEDGER)
        self.assertIn("segments/target.final_norm.bin", found)
        self.assertIn("index.json representations.target.final_norm@BF16", found)

    def test_a_stale_source_binding_on_an_op(self):
        root = self.copy("low_rank_k2")
        self.edit_ledger(root, lambda d: d["targets"][0]["operations"][0]["source"].update(index_sha256="0" * 64))
        out = self.judge(root)
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("stale or foreign source binding (index_sha256)", self.findings(out, W.LEDGER))

    def test_a_ledger_bound_to_another_base(self):
        root = self.copy("low_rank_k2")
        other = os.path.join(self.work, "otherbase")
        shutil.copytree(BASE, other)
        with open(os.path.join(other, "index.json"), "a") as f:
            f.write("\n")
        out = self.judge(root, other)
        self.assertHalves(out, CNE, VIOL)
        self.assertEqual(out["provenance_detail"]["base_binding"], VIOL)
        self.assertIn("source.index_sha256", self.findings(out, W.LEDGER))

    def test_swapped_a_and_b_rehashed(self):
        root = self.copy("low_rank_k2")

        def change(doc):
            op = doc["targets"][0]["operations"][0]
            op["a"], op["b"] = op["b"], op["a"]
            rehash_operands(op)

        self.edit_ledger(root, change)
        out = self.judge(root)
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("shape_invalid", self.findings(out, W.LEDGER))

    def test_b_laid_out_rank_by_cols_instead_of_rank_by_j(self):
        root = self.copy("low_rank_k2")

        def change(doc):
            op = doc["targets"][0]["operations"][0]
            n, cols = len(op["columns"]), doc["targets"][0]["shape"][1]
            full = []
            for k in range(op["rank"]):
                row = ["0000000000000000"] * cols
                for p, j in enumerate(op["columns"]):
                    row[j] = op["b"][k * n + p]
                full += row
            op["b"] = full
            rehash_operands(op)

        self.edit_ledger(root, change)
        out = self.judge(root)
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("shape_invalid", self.findings(out, W.LEDGER))

    def test_a_tampered_operands_sha256(self):
        for key in ("a", "b"):
            with self.subTest(key=key):
                root = self.copy("low_rank_k2")
                self.edit_ledger(root, lambda d: d["targets"][0]["operations"][0]["operands_sha256"].update(
                    {key: "0" * 64}))
                out = self.judge(root)
                self.assertHalves(out, CNE, VIOL)  # §7a: nothing trustworthy to recompute
                self.assertIn(f"operands_sha256.{key}", self.findings(out, W.LEDGER))
                shutil.rmtree(root)

    def test_a_tampered_remove_direction_operands_sha256(self):
        root = self.copy("mixed")
        self.edit_ledger(root, lambda d: d["targets"][0]["operations"][1]["operands_sha256"].update(alpha="1" * 64))
        out = self.judge(root)
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("operands_sha256.alpha", self.findings(out, W.LEDGER))

    def test_a_tampered_result_tensor_sha256(self):
        root = self.copy("low_rank_k2")
        self.edit_ledger(root, lambda d: d["targets"][0].update(result_tensor_sha256="0" * 64))
        out = self.judge(root)
        self.assertHalves(out, SAT, VIOL)
        self.assertIn("result_tensor_sha256", self.findings(out, W.LEDGER))

    def test_a_tampered_base_tensor_sha256(self):
        root = self.copy("low_rank_k2")
        self.edit_ledger(root, lambda d: d["targets"][0].update(base_tensor_sha256="0" * 64))
        self.assertHalves(self.judge(root), SAT, VIOL)

    def test_tampered_changed_elements(self):
        def op_count(d):
            d["targets"][0]["operations"][0]["changed_elements"] += 1

        def target_count(d):
            d["targets"][0]["changed_elements"] += 1

        def both(d):
            op_count(d)
            target_count(d)

        for change in (op_count, target_count, both):
            with self.subTest(change=change.__name__):
                root = self.copy("low_rank_k2")
                self.edit_ledger(root, change)
                out = self.judge(root)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn("changed_elements", self.findings(out, W.LEDGER))
                shutil.rmtree(root)

    def test_a_sham_that_claims_a_change(self):
        root = self.copy("sham")

        def change(d):
            d["targets"][0]["operations"][0]["changed_elements"] = 1
            d["targets"][0]["changed_elements"] = 1

        self.edit_ledger(root, change)
        out = self.judge(root)
        self.assertHalves(out, SAT, VIOL)
        self.assertIn("zero-valued op (§4a) never changes an element", self.findings(out, W.LEDGER))

    def test_a_sham_whose_result_digest_differs_from_its_base_digest(self):
        root = self.copy("sham")
        self.edit_ledger(root, lambda d: d["targets"][0].update(result_tensor_sha256="0" * 64))
        out = self.judge(root)
        self.assertHalves(out, SAT, VIOL)

    def test_overlapping_columns_between_two_ops_of_one_target(self):
        root = self.copy("mixed")
        def overlap(d):
            op = d["targets"][0]["operations"][1]
            op["columns"] = [9, 34]
            op["digest"] = W.op_digest(op)

        self.edit_ledger(root, overlap)
        out = self.judge(root)
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("column 9 is also written by op 0", self.findings(out, W.LEDGER))

    def test_rank_over_the_bound(self):
        root = self.copy("low_rank_k2")
        self.edit_ledger(root, lambda d: d["targets"][0]["operations"][0].update(rank=W.MAX_STRUCTURED_RANK + 1))
        out = self.judge(root)
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("MAX_STRUCTURED_RANK", self.findings(out, W.LEDGER))

    def test_each_other_bound_is_enforced_and_inclusive(self):
        # low_rank_k2: 64x256 tensor, rank 2, |J| = 6: 128 + 12 operand elements, 768 multiply-adds
        for constant, at, over in (("MAX_STRUCTURED_OPERAND_ELEMENTS", 140, 139),
                                   ("MAX_STRUCTURED_TENSOR_ELEMENTS", 64 * 256, 64 * 256 - 1),
                                   ("MAX_STRUCTURED_WORK", 768, 767)):
            with self.subTest(constant=constant):
                with mock.patch.object(W, constant, at):
                    self.assertHalves(self.judge(self.copy_once("low_rank_k2")), SAT, SAT)
                with mock.patch.object(W, constant, over):
                    out = self.judge(self.copy_once("low_rank_k2"))
                self.assertHalves(out, CNE, VIOL)
                self.assertIn(constant, self.findings(out, W.LEDGER))

    def copy_once(self, case):
        if not hasattr(self, "shared"):
            self.shared = self.copy(case)
        return self.shared

    def test_an_unknown_schema(self):
        root = self.copy("low_rank_k2")
        self.edit_ledger(root, lambda d: d.update(schema="weight-patches/v5"))
        out = self.judge(root)
        self.assertHalves(out, CNE, CNE)
        self.assertIn("unknown ledger schema 'weight-patches/v5'", self.findings(out, W.LEDGER))

    def test_an_unknown_op_and_an_unknown_op_field(self):
        for change in (lambda op: op.update(op="add_outer"), lambda op: op.update(scale=2)):
            root = self.copy("low_rank_k2")
            self.edit_ledger(root, lambda d: change(d["targets"][0]["operations"][0]))
            out = self.judge(root)
            self.assertHalves(out, CNE, CNE)
            shutil.rmtree(root)

    def test_an_unsupported_profile(self):
        root = self.copy("low_rank_k2")
        self.edit_ledger(root, lambda d: d["targets"][0]["operations"][0].update(numerics="f16-seq/v1"))
        self.assertHalves(self.judge(root), CNE, CNE)

    def test_an_op_on_another_tensor(self):
        root = self.copy("low_rank_k2")
        self.edit_ledger(root, lambda d: d["targets"][0]["operations"][0].update(tensor="0.mlp.down_proj.weight"))
        self.assertHalves(self.judge(root), CNE, VIOL)

    def test_a_delta_edit_with_the_wrong_recorded_bits(self):
        root = self.copy("mixed")
        self.edit_ledger(root, lambda d: d["targets"][1]["edits"][0].update(new_bits=d["targets"][1]["edits"][0]
                                                                            ["new_bits"] ^ 1))
        out = self.judge(root)
        self.assertHalves(out, SAT, VIOL)
        self.assertIn("new_bits", self.findings(out, W.LEDGER))

    def retarget_delta(self, root, delta, delta_bits, new_bits=None):
        """Rewrite the first delta edit of the `mixed` fixture's element target as `delta`/`delta_bits`, with the
        container holding `new_bits` (default: what the exact sum of the base element and `delta_bits` rounds to)."""
        edit = self.ledger(root)["targets"][1]["edits"][0]
        name = self.ledger(root)["targets"][1]["tensor"]
        if new_bits is None:
            new_bits = W.exact_sum_to_bf16(edit["old_bits"], delta_bits)
        write_element(root, name, edit["row"], edit["col"], new_bits)
        reseal_compiled(root)
        self.edit_ledger(root, lambda d: d["targets"][1]["edits"][0].update(
            delta=delta, delta_bits=delta_bits, new_bits=new_bits))
        reseal_compiled(root)

    def test_a_delta_written_in_shortest_binary32_form_is_the_binary32_it_reads_as(self):
        # An engine prints 0x3dcccccd as `0.1`. Read as a binary64 that decimal is another number; read as a
        # binary32, it is exactly the delta. An honest compile must not be reported as violated for it.
        root = self.copy("mixed")
        self.retarget_delta(root, 0.1, TENTH_BINARY32_BITS)
        with open(os.path.join(root, LEDGER_FILE)) as f:
            self.assertIn('"delta": 0.1,', f.read())
        self.assertHalves(self.judge(root), SAT, SAT)

    def test_a_delta_that_reads_as_another_binary32_than_its_bits(self):
        root = self.copy("mixed")
        self.retarget_delta(root, 0.1, TENTH_BINARY32_BITS + 1)
        out = self.judge(root)
        self.assertHalves(out, SAT, VIOL)
        self.assertIn("does not read as the binary32", self.findings(out, W.LEDGER))

    def test_a_delta_beyond_binary32_is_a_violation_not_a_crash(self):
        root = self.copy("mixed")
        self.retarget_delta(root, 1e300, TENTH_BINARY32_BITS)
        out = self.judge(root)
        self.assertHalves(out, SAT, VIOL)
        self.assertIn("does not read as the binary32", self.findings(out, W.LEDGER))

    def set_delta_text(self, root, text, delta_bits=TENTH_BINARY32_BITS, field="delta"):
        """Put the literal JSON number `text` in the ledger: as the first delta edit's `delta`, or as a top-level
        field of that name. Written as text, because these are numbers a JSON writer would not emit."""
        marker = 123454321.5
        def place(d):
            if field == "delta":
                d["targets"][1]["edits"][0].update(delta=marker, delta_bits=delta_bits)
            else:
                d[field] = marker
        self.edit_ledger(root, place)
        path = os.path.join(root, LEDGER_FILE)
        with open(path) as f:
            ledger = f.read()
        self.assertEqual(ledger.count(repr(marker)), 1)
        with open(path, "w") as f:
            f.write(ledger.replace(repr(marker), text))
        reseal_compiled_index_only(root)

    def test_an_enormous_delta_is_answered_at_once_and_never_raises(self):
        # Nine characters must not become a million-digit integer (§5), and no number may raise out of the verifier.
        for text in ("1e400", "1e2000000", "1e999999999", "1e99999999999999999999", "-1e99999999999999999999",
                     "1e" + "9" * 5000, "123456789e31"):
            with self.subTest(delta=text[:24]):
                root = self.copy("mixed")
                self.set_delta_text(root, text)
                started = time.monotonic()
                out = self.judge(root)
                self.assertLess(time.monotonic() - started, ANSWER_AT_ONCE_SECONDS)
                self.assertEqual(out["ledger_integrity"], VIOL, out["reasons"])
                self.assertIn("does not read as the binary32", self.findings(out, W.LEDGER))

    def test_a_vanishing_delta_reads_as_a_zero_of_its_sign(self):
        for text, bits, reads in (("1e-99999999999999999999", 0x00000000, True),
                                  ("-1e-2000000", 0x80000000, True),
                                  ("1e-2000000", 0x80000000, False),
                                  ("-1e-" + "9" * 5000, 0x80000000, True),
                                  ("7e-46", 0x00000000, True),  # under half the smallest subnormal
                                  ("7.1e-46", 0x00000001, True)):  # over it
            with self.subTest(delta=text[:24], bits=f"{bits:08x}"):
                self.assertEqual(W.delta_reads_as(W.DecimalFloat(text), bits), reads)

    def test_the_sign_of_a_zero_delta_is_part_of_what_it_reads_as(self):
        # −0 + −0 is −0 (§3b), so a zero's sign is not decoration. A binary32 reader reads `-0.0` as 0x80000000.
        for text, bits, reads in (("0.0", 0x00000000, True), ("-0.0", 0x80000000, True),
                                  ("-0.0", 0x00000000, False), ("0.0", 0x80000000, False),
                                  ("0e5", 0x00000000, True), ("-0.000e-3", 0x80000000, True)):
            with self.subTest(delta=text, bits=f"{bits:08x}"):
                self.assertEqual(W.delta_reads_as(W.DecimalFloat(text), bits), reads)
        self.assertTrue(W.delta_reads_as(0, 0x00000000))  # written as an integer
        self.assertFalse(W.delta_reads_as(0, 0x80000000))

    def test_the_reading_is_one_rounding_of_the_exact_decimal_ties_to_even(self):
        tenth = TENTH_BINARY32_BITS
        # The exact midpoint of 0x3dcccccd and 0x3dccccce, as a decimal: it goes to the even pattern, …ce.
        midpoint = "0.10000000521540641784667968750"
        for text, bits, reads in (("0.1", tenth, True), ("0.10000000149011612", tenth, True),
                                  ("0.10000001", tenth, False), (midpoint, tenth + 1, True), (midpoint, tenth, False),
                                  ("1e-1", tenth, True), ("100000000e-9", tenth, True), ("0.000001E5", tenth, True),
                                  ("3.4028235e38", 0x7F7FFFFF, True), ("3.4028236e38", 0x7F7FFFFF, False),
                                  ("1e-45", 0x00000001, True)):
            with self.subTest(delta=text, bits=f"{bits:08x}"):
                self.assertEqual(W.delta_reads_as(W.DecimalFloat(text), bits), reads)
        self.assertTrue(W.delta_reads_as(1, 0x3F800000))
        self.assertFalse(W.delta_reads_as(2 ** 200, 0x7F7FFFFF))

    def test_an_exponent_is_read_by_its_value_not_its_length(self):
        # Five thousand leading zeros in an exponent are still an exponent of one, or of none.
        zeros = "0" * 5000
        for text, bits, reads in ((f"1e-{zeros}", 0x3F800000, True), (f"1e-{zeros}", 0x00000000, False),
                                  (f"1e{zeros}", 0x3F800000, True), (f"1e+{zeros}", 0x3F800000, True),
                                  (f"1e-{zeros}1", TENTH_BINARY32_BITS, True), (f"1e-{zeros}1", 0x00000000, False),
                                  (f"-1e-{zeros[1:]}38", 0x80000000, False),
                                  ("1e-" + "0" * 30 + "1", TENTH_BINARY32_BITS, True),
                                  ("1e" + "9" * 19, 0x7F7FFFFF, False), ("-1e-" + "9" * 19, 0x80000000, True)):
            with self.subTest(delta=text[:8] + "…" + text[-4:], bits=f"{bits:08x}"):
                self.assertEqual(W.delta_reads_as(W.DecimalFloat(text), bits), reads)
        root = self.copy("mixed")
        self.set_delta_text(root, f"1e-{zeros}", delta_bits=0x00000000)  # 1.0, recorded as zero
        out = self.judge(root)
        self.assertEqual(out["ledger_integrity"], VIOL, out["reasons"])

    def test_a_delta_of_any_length_is_judged(self):
        # More digits than an integer conversion takes: still a number, with one right reading.
        tenth, zeros = TENTH_BINARY32_BITS, "0" * 5000
        midpoint = "0.10000000521540641784667968750"  # of …cd and …ce: a tie, to the even …ce
        for text, bits, reads in ((f"0.1{zeros}", tenth, True), (f"0.2{zeros}", tenth, False),
                                  (f"0.{zeros}1e5000", tenth, True), (f"0.{zeros}1e5001", 0x3F800000, True),
                                  (f"1{zeros}e-5001", tenth, True), (f"{zeros}0.1", tenth, True),
                                  (midpoint + zeros, tenth + 1, True), (midpoint + zeros + "1", tenth + 2, False),
                                  (midpoint + zeros + "1", tenth + 1, True),  # a hair over the tie: up, not to even
                                  (midpoint[:-2] + "49" + "9" * 5000, tenth, True),  # a hair under it: down
                                  ("0.1" + "3" * 5000, 0x3DCCCCCD, False)):
            with self.subTest(delta=text[:12] + "…" + text[-6:], bits=f"{bits:08x}"):
                self.assertEqual(W.delta_reads_as(W.DecimalFloat(text), bits), reads)
        # The next midpoint, of …ce and …cf, ties DOWN to the even …ce. A nonzero digit five thousand places on
        # puts the number over it, so it reads as …cf: digits past any cut-off still decide a tie.
        upper = (W.f32_value(tenth + 1) + W.f32_value(tenth + 2)) / 2
        scale = 10 ** 60
        self.assertEqual(upper * scale % 1, 0)  # the decimal below is exact
        upper = "0." + str(int(upper * scale)).zfill(60)
        for text, bits in ((upper, tenth + 1), (upper + zeros, tenth + 1), (upper + zeros + "1", tenth + 2)):
            with self.subTest(over_a_tie_that_goes_down=len(text)):
                self.assertTrue(W.delta_reads_as(W.DecimalFloat(text), bits))
                self.assertFalse(W.delta_reads_as(W.DecimalFloat(text), bits ^ 3))
        for text, half in ((f"0.1{zeros}", SAT), (f"0.2{zeros}", VIOL)):
            with self.subTest(judged=text[:6]):
                root = self.copy("mixed")
                self.retarget_delta(root, 0.1, tenth)
                self.set_delta_text(root, text)
                started = time.monotonic()
                out = self.judge(root)
                self.assertLess(time.monotonic() - started, ANSWER_AT_ONCE_SECONDS)
                self.assertEqual(out["ledger_integrity"], half, out["reasons"])

    def test_the_first_significant_digit_is_found_through_any_zeros(self):
        hundred = "0" * 100
        for text, bits, reads in ((f"0.{hundred}1e60", 0x00001BE0, None),  # 1e-41, a subnormal
                                  (f"1{hundred}e-62", 0x7E967699, True),  # 1e38
                                  ("0.0e400000000", 0x00000000, True), ("-0.0e400000000", 0x80000000, True),
                                  ("0.0e400000000", 0x80000000, False),
                                  (f"0.{hundred}1e140", 0x7F7FFFFF, False),  # 1e39: beyond binary32
                                  (f"1{hundred}e-148", 0x00000000, True)):  # 1e-48: a zero
            if reads is None:
                bits = struct.unpack("<I", struct.pack("<f", 1e-41))[0]
                reads = True
            with self.subTest(delta=text[:10] + "…" + text[-6:], bits=f"{bits:08x}"):
                self.assertEqual(W.delta_reads_as(W.DecimalFloat(text), bits), reads)

    def test_the_integer_spelling_of_negative_zero_keeps_its_sign(self):
        for bits, half in ((0x80000000, SAT), (0x00000000, VIOL)):
            with self.subTest(bits=f"{bits:08x}"):
                root = self.copy("mixed")
                self.retarget_delta(root, 0.0, bits)
                self.set_delta_text(root, "-0", delta_bits=bits)
                out = self.judge(root)
                self.assertEqual(out["ledger_integrity"], half, self.findings(out, W.LEDGER))

    def test_a_violation_names_the_delta_as_it_was_written(self):
        root = self.copy("mixed")
        self.set_delta_text(root, "1e-" + "0" * 5000 + "1", delta_bits=0x00000000)
        out = self.judge(root)
        self.assertIn("delta 1e-0000", self.findings(out, W.LEDGER))
        self.assertIn("characters)", self.findings(out, W.LEDGER))

    def test_an_enormous_number_in_another_field_costs_nothing(self):
        # Every float of the ledger is read through the same hook: one that is not a delta must not raise or stall.
        root = self.copy("mixed")
        self.set_delta_text(root, "1e99999999999999999999", field="writer_note")
        out = self.judge(root)
        self.assertNotIn("Traceback", json.dumps(out))
        self.assertEqual(out["numerical_satisfaction"], SAT, out["reasons"])

    def test_non_finite_delta_bits_are_a_ledger_violation_not_a_crash(self):
        for bits in (BINARY32_INFINITY_BITS, BINARY32_NAN_BITS):
            with self.subTest(bits=f"{bits:08x}"):
                root = self.copy("mixed")
                edit = self.ledger(root)["targets"][1]["edits"][0]
                self.retarget_delta(root, 0.0, bits, new_bits=edit["new_bits"])
                out = self.judge(root)
                self.assertEqual(out["ledger_integrity"], VIOL, out)
                self.assertIn("is not finite", self.findings(out, W.LEDGER))

    def test_a_delta_edit_with_the_wrong_old_bits(self):
        root = self.copy("mixed")
        self.edit_ledger(root, lambda d: d["targets"][1]["edits"][1].update(old_bits=0))
        self.assertHalves(self.judge(root), SAT, VIOL)

    def test_a_target_with_both_operations_and_edits(self):
        root = self.copy("mixed")
        self.edit_ledger(root, lambda d: d["targets"][0].update(edits=[]))
        self.assertHalves(self.judge(root), CNE, VIOL)

    def test_a_tampered_result_derived_digest(self):
        root = self.copy("low_rank_k2")
        self.edit_ledger(root, lambda d: d["result"]["derived"][0].update(source_representation_digest="0" * 64))
        out = self.judge(root)
        self.assertHalves(out, SAT, VIOL)
        self.assertIn("result.derived", self.findings(out, W.LEDGER))

    def test_an_op_field_altered_without_resealing_its_digest(self):
        for field, change in (("origin", lambda op: op.update(origin="someone else")),
                              ("a", lambda op: op["a"].__setitem__(1, op["a"][3])),
                              ("columns", lambda op: op.update(columns=[3, 4, 5, 40, 200, 254])),
                              ("numerics", lambda op: op.update(numerics=V.PROFILE_RNE2)),
                              ("origin removed", lambda op: op.pop("origin"))):
            with self.subTest(field=field):
                root = self.copy("low_rank_k2")

                def edit(d):
                    op = d["targets"][0]["operations"][0]
                    change(op)
                    if field == "a":  # keep the operand digest right, so only §3a can object
                        op["operands_sha256"]["a"] = operand_sha(op, "a")

                self.edit_ledger(root, edit)
                out = self.judge(root)
                self.assertHalves(out, CNE, VIOL)
                self.assertIn("digest is not intact", self.findings(out, W.LEDGER))
                self.assertEqual(out["provenance_detail"]["op_digest"], VIOL)
                shutil.rmtree(root)

    def test_a_remove_direction_op_altered_without_resealing(self):
        root = self.copy("mixed")
        self.edit_ledger(root, lambda d: d["targets"][0]["operations"][1].update(alpha="3fd0000000000000"))
        out = self.judge(root)
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("digest is not intact", self.findings(out, W.LEDGER))

    def test_an_index_field_a_compile_may_not_change(self):
        for change in (lambda i: i.update(hidden_size=i["hidden_size"] + 1),
                       lambda i: i["representations"]["target.final_norm@BF16"].update(tensor_count=1, extra="x"),
                       lambda i: i["representations"][DERIVED].update(compiled_from=REP, encoder={"algorithm": "x"})):
            root = self.copy("low_rank_k2")
            path = os.path.join(root, "index.json")
            index = read_json(path)
            change(index)
            write_json(path, index)
            out = self.judge(root)
            self.assertHalves(out, SAT, VIOL)
            self.assertIn("index.json", self.findings(out, W.LEDGER))
            shutil.rmtree(root)

    def test_max_patch_elements_is_enforced_and_inclusive(self):
        with mock.patch.object(W, "MAX_PATCH_ELEMENTS", 2):  # mixed: one element target of two edits
            self.assertHalves(self.judge(self.copy("mixed")), SAT, SAT)
        with mock.patch.object(W, "MAX_PATCH_ELEMENTS", 1):
            out = self.judge(self.copy("mixed"))
        self.assertHalves(out, CNE, VIOL)
        self.assertIn("MAX_PATCH_ELEMENTS", self.findings(out, W.LEDGER))
        self.assertEqual(W.MAX_PATCH_ELEMENTS, 2 ** 18)

    def test_an_op_with_fixups_is_unsupported(self):
        root = self.copy("low_rank_k2")
        self.edit_ledger(root, lambda d: d["targets"][0]["operations"][0].update(fixups=[]))
        self.assertHalves(self.judge(root), CNE, CNE)

    def test_a_v4_ledger_with_no_operation_target(self):
        root = self.copy("mixed")
        self.edit_ledger(root, lambda d: d.update(targets=d["targets"][1:]))
        out = self.judge(root)
        self.assertEqual(out["ledger_integrity"], VIOL)
        self.assertIn("(§2)", self.findings(out, W.LEDGER))

    def test_a_ledger_pointer_that_is_missing(self):
        root = self.copy("low_rank_k2")
        index = read_json(os.path.join(root, "index.json"))
        del index["weight_patches"]
        write_json(os.path.join(root, "index.json"), index)
        self.assertHalves(self.judge(root), CNE, CNE)


def rewrite_segment_header(root, rep, change):
    """Rebuild a segment file with a changed header (same payload) and re-seal its index entry."""
    index_path = os.path.join(root, "index.json")
    index = read_json(index_path)
    entry = index["representations"][rep]
    path = os.path.join(root, entry["segment"])
    with open(path, "rb") as f:
        data = f.read()
    (hlen,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8:8 + hlen])
    payload = data[8 + hlen:]
    change(header)
    text = json.dumps(header).encode()
    pad = (-(8 + len(text))) % 4
    text += b" " * pad
    out = struct.pack("<Q", len(text)) + text + payload
    with open(path, "wb") as f:
        f.write(out)
    entry["tensor_count"] = len(header["tensors"])
    entry["segment_sha256"] = sha(out)
    entry["payload_sha256"] = sha(payload)
    write_json(index_path, index)


def split_final_norm_into_duplicate_names(header):
    (t,) = header["tensors"]
    half = t["shape"][0] // 2
    header["tensors"] = [dict(t, shape=[half], offset=0, len=t["len"] // 2),
                         dict(t, shape=[half], offset=t["len"] // 2, len=t["len"] // 2)]


class WholeFileSegments(Fixture):
    """§7 step 4: a segment that is neither edited nor re-derived is judged as a whole file and never parsed, so an
    ambiguous tensor table there (the Gemma 3 vision adapter's duplicate name) does not stop the judgement."""

    NORM = "target.final_norm@BF16"

    def setUp(self):
        super().setUp()
        self.base = os.path.join(self.work, "base")
        shutil.copytree(BASE, self.base)
        self.root = self.copy("low_rank_k2")
        for r in (self.base, self.root):
            rewrite_segment_header(r, self.NORM, split_final_norm_into_duplicate_names)
        base_index_sha = sha(open(os.path.join(self.base, "index.json"), "rb").read())

        def rebind(doc):
            doc["source"]["index_sha256"] = base_index_sha
            for t in doc["targets"]:
                for op in t["operations"]:
                    op["source"]["index_sha256"] = base_index_sha
                    op["digest"] = W.op_digest(op)

        self.edit_ledger(self.root, rebind)

    def test_an_untouched_segment_with_a_duplicate_name_identical_in_both_is_satisfied(self):
        with self.assertRaisesRegex(C.ContainerError, "duplicate tensor name"):
            C.open_container(self.base)  # a strict reader cannot parse this base at all
        seg = read_json(os.path.join(self.base, "index.json"))["representations"][self.NORM]["segment"]
        self.assertEqual(open(os.path.join(self.base, seg), "rb").read(), open(os.path.join(self.root, seg), "rb").read())
        out = self.judge(self.root, self.base)
        self.assertHalves(out, SAT, SAT)
        measured = {m["name"]: m["value"] for m in out["measurements"]}
        self.assertEqual(measured["untouched segments judged as whole files (byte-identical; internal structure not validated)"], "3")

    def test_the_same_segment_differing_by_one_byte_is_a_ledger_violation(self):
        index = read_json(os.path.join(self.root, "index.json"))
        path = os.path.join(self.root, index["representations"][self.NORM]["segment"])
        with open(path, "r+b") as f:
            f.seek(-3, os.SEEK_END)
            b = f.read(1)
            f.seek(-3, os.SEEK_END)
            f.write(bytes([b[0] ^ 1]))
        self.assertEqual(self.judge(self.root, self.base)["status"], CNE)  # unsealed: the file no longer matches its index
        rewrite_segment_header(self.root, self.NORM, lambda h: None)  # re-seal the index entry to the changed bytes
        out = self.judge(self.root, self.base)
        self.assertHalves(out, SAT, VIOL)
        self.assertIn("segments/target.final_norm.bin", self.findings(out, W.LEDGER))

    def test_a_duplicate_name_inside_the_edited_segment_is_still_refused(self):
        def duplicate(header):
            names = [t["name"] for t in header["tensors"]]
            header["tensors"][names.index("0.mlp.up_proj.weight")]["name"] = "0.mlp.gate_proj.weight"

        rewrite_segment_header(self.root, REP, duplicate)
        out = self.judge(self.root, self.base)
        self.assertHalves(out, CNE, CNE)
        self.assertTrue(any("duplicate tensor name" in r for r in out["reasons"]), out["reasons"])


class ExactSumRounding(unittest.TestCase):
    """`bf16-rne-of-exact-sum` where a binary64 sum is not the exact sum: the delta sits on a BF16 tie and the stored
    value is too small for a binary64 to keep. The exact sum is off the tie, so the small term decides the direction;
    a reader that adds in binary64 lands on the tie and goes to the even neighbour instead. Hand-derived expectations,
    so an engine and this verifier are each held to them independently."""

    #: (old BF16 bits, delta binary32 bits, expected BF16 bits)
    CASES = (
        (0x3F80, 0x5D808000, 0x5D81),  # 1.0 + 2^60·(1 + 2^-8): above the tie, up
        (0xBF80, 0x5D808000, 0x5D80),  # −1.0 + the same delta: below the tie, down
        (0x2500, 0x3F808000, 0x3F81),  # 2^-53 + (1 + 2^-8): the stored value is the small term
        (0x3F80, 0x5D818000, 0x5D82),  # an odd lower neighbour, above the tie: up
        (0xBF80, 0x5D818000, 0x5D81),  # the same tie from below: down, where ties-to-even alone would go up
        (0x3F80, 0x3B800000, 0x3F80),  # 1.0 + 2^-8 exactly: a true tie, to even
    )

    def test_the_small_term_breaks_the_tie(self):
        for old, delta, want in self.CASES:
            with self.subTest(old=f"{old:04x}", delta=f"{delta:08x}"):
                self.assertEqual(f"{W.exact_sum_to_bf16(old, delta):04x}", f"{want:04x}")


class NoArithmeticWithoutSelection(unittest.TestCase):
    """v2 §3: a zero-valued op's elements are never run through the arithmetic, so -0 in the base survives."""

    def test_a_zero_valued_op_over_negative_zeros_is_an_identity(self):
        base = np.full(12, NEG_ZERO, dtype=np.uint16)
        zero = "0000000000000000"
        op = {"numerics": V.PROFILE_RNE1, "rank": 1, "a": [zero] * 3, "b": ["3ff0000000000000"] * 2, "columns": [1, 2]}
        parsed = V.LowRank(op, (3, 4))
        self.assertTrue(parsed.zero_valued())
        self.assertEqual(parsed.apply(base).reshape(-1).tolist(), base.tolist())


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.environ.get("LCV_BENCH"), "set LCV_BENCH=1 to time a Gemma-3-4B-sized rank-one target")
class Benchmark(unittest.TestCase):
    """A [2560, 10240] rank-one `add_low_rank_columns` over every column (26.2M elements written), as one real
    `down_proj` edit would be: parse, recompute and compare every element. Prints the wall time."""

    ROWS, COLS = 2560, 10240

    def test_rank_one_over_a_whole_down_proj(self):
        import time
        rng = np.random.default_rng(1)
        base = (rng.standard_normal(self.ROWS * self.COLS).astype(np.float32) * 0.02).view(np.uint32)
        base = (base >> np.uint32(16)).astype(np.uint16)
        u = rng.standard_normal(self.ROWS) * 1e-3
        v = rng.standard_normal(self.COLS)
        op = {"numerics": V.PROFILE_RNE1, "rank": 1, "columns": list(range(self.COLS)),
              "a": [f"{b:016x}" for b in u.view(np.uint64).tolist()],
              "b": [f"{b:016x}" for b in v.view(np.uint64).tolist()]}
        t0 = time.perf_counter()
        parsed = V.LowRank(op, (self.ROWS, self.COLS))
        t1 = time.perf_counter()
        result = parsed.apply(base)
        t2 = time.perf_counter()
        compiled = result.copy()  # stands in for the compiled container's tensor
        compiled[17, 4242] ^= 1
        t3 = time.perf_counter()
        changed = int(np.count_nonzero(result.reshape(-1) != base))
        mismatches = int(np.count_nonzero(compiled != result))
        t4 = time.perf_counter()
        sys.stderr.write(f"\n[bench] {self.ROWS}x{self.COLS} rank 1: parse {t1 - t0:.2f}s, recompute {t2 - t1:.2f}s, "
                         f"compare every element {t4 - t3:.2f}s, total {t2 - t0 + t4 - t3:.2f}s; {changed} elements changed\n")
        self.assertEqual(mismatches, 1)
        self.assertGreater(changed, 0)
