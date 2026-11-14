"""`EXECUTION_RESIDENCY_V1`: the three derivation rules against hand-derived blocks, an exact-rational second route
and digests the engine's loader produced; the selection rule at its boundaries; and every way a record can be
wrong, incomplete or outside what the contract specifies."""

import copy
import hashlib
import json
import os
import struct
import sys
import tempfile
import shutil
import unittest
from unittest import mock
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from larql_contract_verifier import cli  # noqa: E402
from larql_contract_verifier.contracts import execution_residency as R  # noqa: E402
from refcheck import container as C  # noqa: E402
from refcheck import numconv  # noqa: E402

BASE = os.path.join(ROOT, "conformance", "structured_patch", "base")
ENGINE_DIGESTS = os.path.join(ROOT, "conformance", "execution_residency", "engine_digests.json")
OBJECT = "target.decoder_stack"
REP = f"{OBJECT}@BF16"
SAT, VIOL, CNE = "satisfied", "violated", "could_not_evaluate"
#: A cache size that puts the fixture's three matrix sizes in the three populations of §4: 16×64 and 64×64
#: (4,096 and 16,384 bytes as f32) are widened or kept, and 64×256 (32,768 bytes as BF16) is quantised under auto.
L2 = 10_000
LARGE = "1.mlp.down_proj.weight"  # [64, 256]: Q8 under auto
MIDDLE = "1.self_attn.o_proj.weight"  # [64, 64]: stored bytes
SMALL = "1.self_attn.k_proj.weight"  # [16, 64]: widened


def sha(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def matrices():
    seg = C.open_container(BASE).segments[REP]
    out = {}
    with open(seg.path, "rb") as f:
        for t in seg.tensors:
            if len(t.shape) == 2:
                f.seek(seg.payload_start + t.offset)
                out[t.name] = (tuple(t.shape), f.read(t.len))
    return out


MATRICES = matrices()
INDEX_SHA = sha(open(os.path.join(BASE, "index.json"), "rb").read())


def f32(x: Fraction) -> Fraction:
    return numconv.round_nearest_even(x, numconv.BINARY32)


def q8_exact(values: list[Fraction]) -> tuple[Fraction, list[int]]:
    """§3.1 for one block on exact rationals, every rounding explicit: `(scale, codes)`. Shares no arithmetic with
    the contract's vectorised rule."""
    peak = max(abs(v) for v in values)
    scale = f32(peak / R.Q8_CODE_LIMIT) if peak > 0 else Fraction(1)
    codes = []
    for v in values:
        q = f32(v / scale)
        whole = int(q)
        if abs(q - whole) >= Fraction(1, 2):
            whole += 1 if q > 0 else -1
        codes.append(max(-R.Q8_CODE_LIMIT, min(R.Q8_CODE_LIMIT, whole)))
    return scale, codes


def record(source="auto", l2=L2, max_format=None, only=None):
    """A complete, correct record of the fixture's decoder stack under `source` and cache size `l2`. Digests come
    from the contract's own rule here; that the rule is right is established by the other test classes, against
    hand-derived blocks, the exact route and the engine's loader."""
    operands = []
    for i, (name, (shape, raw)) in enumerate(sorted(MATRICES.items())):
        if only is not None and name not in only:
            continue
        rule = sorted(R.selected_rules(shape[0] * shape[1], source, l2, max_format))[0]
        operands.append({
            "order": i, "object": OBJECT, "tensor": name, "shape": list(shape), "stored_dtype": "BF16",
            "stored_sha256": sha(raw), "representation_source": source, "requested_format": R.FORMAT_OF[rule],
            "resident_format": R.FORMAT_OF[rule], "manufactured": rule == R.RULE_Q8_LOAD, "rule": rule,
            "resident_sha256": R.resident_digest(rule, raw, shape),
            "resident_digest_layout": R.DIGEST_LAYOUT[rule]})
    return {"schema": R.RECORD_SCHEMA, "container": BASE, "container_index_sha256": INDEX_SHA,
            "selection_inputs": {"l2_bytes": l2, "max_format": max_format, "arithmetic": None, "kquant_exec": None},
            "rules": {}, "operands": operands}


class TheQ8Rule(unittest.TestCase):
    def codes_and_scales(self, values, rows, cols):
        codes, scales = R.q8_load(np.asarray(values, dtype=np.float32), rows, cols)
        return [int(c) for c in codes.reshape(-1)], [float(s) for s in scales.reshape(-1)]

    def test_a_block_whose_peak_is_127_has_scale_one_and_its_integers_as_codes(self):
        values = [127.0, -127.0, 1.0, 0.0, 0.5, -0.5, 1.5, -2.5] + [3.0] * 56
        codes, scales = self.codes_and_scales(values, 1, 64)
        self.assertEqual(scales, [1.0])
        self.assertEqual(codes[:8], [127, -127, 1, 0, 1, -1, 2, -3])  # halves away from zero

    def test_an_all_zero_block_has_scale_one_not_zero(self):
        codes, scales = self.codes_and_scales([0.0, -0.0] * 32, 1, 64)
        self.assertEqual((codes, scales), ([0] * 64, [1.0]))

    def test_blocks_are_cut_along_a_row_and_the_last_is_short(self):
        # Two rows of 96: each row is one block of 64 and one of 32, and no block spans the rows.
        row0 = [1.0] * 64 + [2.0] * 32
        row1 = [4.0] * 64 + [8.0] * 32
        codes, scales = R.q8_load(np.asarray(row0 + row1, dtype=np.float32), 2, 96)
        self.assertEqual(scales.shape, (2, 2))
        np.testing.assert_array_equal(scales, np.array([[1, 2], [4, 8]], dtype=np.float32) / np.float32(127))
        self.assertTrue(np.all(codes == 127))

    def test_the_scale_is_a_binary32_and_the_code_a_division_by_it(self):
        # 100/127 is not exact in binary32; the largest element still maps to 127 and half of it to 64 (63.5 away).
        codes, _ = self.codes_and_scales([100.0, 50.0, -25.0] + [0.0] * 61, 1, 64)
        self.assertEqual(codes[:3], [127, 64, -32])

    def test_against_exact_rationals_on_a_whole_committed_tensor(self):
        (rows, cols), raw = MATRICES[LARGE]
        values = R.widen_bf16(raw).reshape(rows, cols)
        codes, scales = R.q8_load(values, rows, cols)
        for r in range(0, rows, 5):
            for b in range(cols // R.Q8_BLOCK):
                block = [Fraction(float(v)) for v in values[r, b * 64:(b + 1) * 64]]
                scale, want = q8_exact(block)
                self.assertEqual(Fraction(float(scales[r, b])), scale, (r, b))
                self.assertEqual([int(c) for c in codes[r, b * 64:(b + 1) * 64]], want, (r, b))

    def test_against_exact_rationals_on_seeded_blocks_across_the_bf16_range(self):
        rng = np.random.default_rng(20261114)
        for exponent in (-130, -40, -8, 0, 20, 100, 126):
            patterns = rng.integers(0, 1 << 15, size=(4, 64), dtype=np.int64)
            field = np.clip(127 + exponent - (patterns >> 7) % 4, 0, 254)
            signs = rng.integers(0, 2, size=(4, 64), dtype=np.int64) << 15
            values = ((((patterns & 0x7F) | (field << 7) | signs) << 16).astype(np.uint32)).view(np.float32)
            codes, scales = R.q8_load(values, 4, 64)
            for r in range(4):
                scale, want = q8_exact([Fraction(float(v)) for v in values[r]])
                self.assertEqual(Fraction(float(scales[r, 0])), scale, exponent)
                self.assertEqual([int(c) for c in codes[r]], want, exponent)


class TheOtherRules(unittest.TestCase):
    def test_widening_puts_each_bf16_pattern_in_the_top_half_of_a_binary32(self):
        raw = struct.pack("<4H", 0x3F80, 0xBF80, 0x0000, 0x8000)  # 1.0, −1.0, +0, −0
        self.assertEqual(R.resident_digest(R.RULE_WIDEN_F32, raw, (1, 4)),
                         sha(struct.pack("<4I", 0x3F800000, 0xBF800000, 0x00000000, 0x80000000)))

    def test_stored_bytes_are_the_stored_bytes(self):
        raw = b"\x01\x02\x03\x04"
        self.assertEqual(R.resident_digest(R.RULE_STORED_BYTES, raw, (1, 2)), sha(raw))


class TheEnginesLoader(unittest.TestCase):
    """Digests the engine's own loader produced for every matrix of the committed base, under each rule. Engine
    output, committed; the contract's rules must reproduce every one."""

    def test_every_rule_reproduces_the_engines_resident_digest_for_every_matrix(self):
        with open(ENGINE_DIGESTS) as f:
            doc = json.load(f)
        self.assertEqual(doc["container_index_sha256"], INDEX_SHA)
        seen = 0
        for o in doc["operands"]:
            shape, raw = MATRICES[o["tensor"]]
            self.assertEqual(o["stored_sha256"], sha(raw), o["tensor"])
            for rule in (R.RULE_Q8_LOAD, R.RULE_WIDEN_F32, R.RULE_STORED_BYTES):
                self.assertEqual(R.resident_digest(rule, raw, shape), o["resident"][rule], (o["tensor"], rule))
                seen += 1
        self.assertEqual(seen, 3 * len(MATRICES), "every matrix, under every rule")


class Selection(unittest.TestCase):
    def test_the_three_populations_and_their_boundaries(self):
        l2 = 1 << 20
        self.assertEqual(R.selected_rules(l2 // 4 - 1, "auto", l2, None), {R.RULE_WIDEN_F32})
        # At exactly the cache size the f32 image no longer fits: the stored bytes are kept.
        self.assertEqual(R.selected_rules(l2 // 4, "auto", l2, None), {R.RULE_STORED_BYTES, R.RULE_STORED_BYTES_MAPPED})
        self.assertEqual(R.selected_rules(l2 // 2 - 1, "auto", l2, None),
                         {R.RULE_STORED_BYTES, R.RULE_STORED_BYTES_MAPPED})
        # And at exactly the cache size the BF16 image streams: quantised, if anything may be.
        self.assertEqual(R.selected_rules(l2 // 2, "auto", l2, None), {R.RULE_Q8_LOAD})

    def test_the_stored_source_and_the_exact_cap_each_forbid_the_manufacture(self):
        l2, big = 1 << 20, 1 << 24
        kept = {R.RULE_STORED_BYTES, R.RULE_STORED_BYTES_MAPPED}
        self.assertEqual(R.selected_rules(big, "stored", l2, None), kept)
        self.assertEqual(R.selected_rules(big, "auto", l2, "bf16"), kept)
        self.assertEqual(R.selected_rules(big, "auto", l2, "q8"), {R.RULE_Q8_LOAD})  # only the exact word caps


class Fixture(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.work)

    def judge(self, doc, container=BASE):
        path = os.path.join(self.work, f"record-{len(os.listdir(self.work))}.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        return cli.evaluate_residency(R.CONTRACT, path, container)

    def assertHalves(self, out, derivation, selection):
        self.assertEqual((out["derivation"], out["selection"]), (derivation, selection),
                         json.dumps(out["verdicts"], indent=1)[:3000])
        want = SAT if (derivation, selection) == (SAT, SAT) else VIOL if VIOL in (derivation, selection) else CNE
        self.assertEqual(out["status"], want)
        self.assertEqual(out["failed_halves"], [h for h, s in ((R.DERIVATION, derivation), (R.SELECTION, selection))
                                                if s != SAT])
        if want == VIOL:
            self.assertEqual(sum(o["violations"] for o in out["offending"]), out["violations"])
            self.assertGreater(out["violations"], 0)
        else:
            self.assertEqual((out["violations"], out["offending"]), (0, []))

    def findings(self, out, half):
        v = out["verdicts"][half]
        return " | ".join(f"{o['name']}: {o['detail']}" for o in v["offending"]) + " | " + " | ".join(v["reasons"])

    def operand(self, doc, tensor):
        return next(o for o in doc["operands"] if o["tensor"] == tensor)


class CorrectRecords(Fixture):
    def test_a_complete_record_is_satisfied_on_both_halves_under_each_source(self):
        for source, rules in (("auto", {R.RULE_Q8_LOAD, R.RULE_STORED_BYTES, R.RULE_WIDEN_F32}),
                              ("stored", {R.RULE_STORED_BYTES, R.RULE_WIDEN_F32})):
            with self.subTest(source=source):
                doc = record(source)
                self.assertEqual({o["rule"] for o in doc["operands"]}, rules, "the fixture exercises each population")
                out = self.judge(doc)
                self.assertHalves(out, SAT, SAT)
                m = {x["name"]: x["value"] for x in out["measurements"]}
                self.assertEqual(m["operands whose derivation was recomputed"], str(len(MATRICES)))
                self.assertEqual(m["operands whose selection was re-decided"], str(len(MATRICES)))
                for limit in ("engine's own report", "kernel consumed", "cache size is the host's"):
                    self.assertIn(limit, m["not established"])

    def test_the_exact_cap_is_a_correct_record_with_nothing_manufactured(self):
        doc = record("auto", max_format="bf16")
        self.assertNotIn(R.RULE_Q8_LOAD, {o["rule"] for o in doc["operands"]})
        self.assertHalves(self.judge(doc), SAT, SAT)


class DerivationFaults(Fixture):
    def tampered(self, name, **changes):
        doc = record()
        self.operand(doc, name).update(changes)
        return self.judge(doc)

    def test_a_resident_digest_that_is_not_the_rules_result(self):
        # What altered codes or altered scales look like from outside: the resident bytes hash to something else.
        for tensor in (LARGE, MIDDLE, SMALL):
            with self.subTest(tensor=tensor):
                out = self.tampered(tensor, resident_sha256="0" * 64)
                self.assertHalves(out, VIOL, SAT)
                self.assertIn("applied to the container's tensor", self.findings(out, R.DERIVATION))

    def test_a_q8_residency_built_by_another_rule(self):
        # The stored Q8_0 pack's geometry (32-element blocks) is a different quantisation of the same weights.
        (rows, cols), raw = MATRICES[LARGE]
        w = R.widen_bf16(raw).reshape(-1, 32)
        scale = np.max(np.abs(w), axis=1) / np.float32(127)
        codes = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
        out = self.tampered(LARGE, resident_sha256=sha(codes.tobytes(), scale.astype("<f4").tobytes()))
        self.assertHalves(out, VIOL, SAT)

    def test_a_stale_source_digest(self):
        out = self.tampered(LARGE, stored_sha256="1" * 64)
        self.assertHalves(out, VIOL, SAT)
        self.assertIn("stored_sha256 is not the digest of the container's tensor", self.findings(out, R.DERIVATION))

    def test_a_record_that_misdescribes_what_it_holds(self):
        for change, said in (({"resident_format": "F32"}, "resident_format"), ({"manufactured": False}, "manufactured"),
                             ({"resident_digest_layout": "f32le-row-major"}, "resident_digest_layout"),
                             ({"resident_sha256": None}, "not a SHA-256")):
            with self.subTest(change=change):
                out = self.tampered(LARGE, **change)
                self.assertEqual(out["derivation"], VIOL, self.findings(out, R.DERIVATION))
                self.assertIn(said, self.findings(out, R.DERIVATION))

    def test_a_shape_or_dtype_that_is_not_the_stored_tensors(self):
        out = self.tampered(LARGE, shape=[256, 64])
        self.assertHalves(out, VIOL, CNE)
        out = self.tampered(LARGE, stored_dtype="F32")
        self.assertHalves(out, VIOL, CNE)

    def test_a_tensor_the_container_does_not_hold(self):
        out = self.tampered(LARGE, tensor="9.mlp.down_proj.weight")
        self.assertEqual(out["derivation"], VIOL)
        self.assertIn("holds no such tensor", self.findings(out, R.DERIVATION))


class SelectionFaults(Fixture):
    def test_an_incorrect_representation_choice(self):
        # Each operand correctly derived under a rule its size and policy do not select.
        for tensor, wrong in ((LARGE, R.RULE_STORED_BYTES), (MIDDLE, R.RULE_Q8_LOAD), (SMALL, R.RULE_STORED_BYTES),
                              (MIDDLE, R.RULE_WIDEN_F32)):
            with self.subTest(tensor=tensor, wrong=wrong):
                doc = record()
                shape, raw = MATRICES[tensor]
                self.operand(doc, tensor).update(
                    rule=wrong, resident_format=R.FORMAT_OF[wrong], manufactured=wrong == R.RULE_Q8_LOAD,
                    resident_sha256=R.resident_digest(wrong, raw, shape), resident_digest_layout=R.DIGEST_LAYOUT[wrong])
                out = self.judge(doc)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn(f"resident under {wrong}", self.findings(out, R.SELECTION))

    def test_a_manufacture_under_the_stored_source_or_the_exact_cap(self):
        for source, cap in (("stored", None), ("auto", "bf16")):
            with self.subTest(source=source, cap=cap):
                doc = record("auto")  # holds a Q8 residency
                doc["selection_inputs"]["max_format"] = cap
                for o in doc["operands"]:
                    o["representation_source"] = source
                out = self.judge(doc)
                self.assertHalves(out, SAT, VIOL)
                self.assertIn(LARGE, self.findings(out, R.SELECTION))

    def test_cache_metadata_the_choices_do_not_follow_from(self):
        # The record's choices were made for one cache size and it reports another.
        doc = record(l2=L2)
        doc["selection_inputs"]["l2_bytes"] = 16 * 1024 * 1024
        out = self.judge(doc)
        self.assertHalves(out, SAT, VIOL)

    def test_a_source_the_record_does_not_know(self):
        doc = record()
        self.operand(doc, LARGE)["representation_source"] = "exact"
        self.assertHalves(self.judge(doc), SAT, VIOL)


class Scope(Fixture):
    """What the answer says about the parts it did not judge."""

    def test_objects_the_record_does_not_name_are_listed_not_judged(self):
        out = self.judge(record())
        self.assertHalves(out, SAT, SAT)
        self.assertEqual(out["uncovered_objects"], ["target.embedding", "target.output_head"])
        self.assertIn("names every object", " ".join(m["value"] for m in out["measurements"]))

    def test_every_missing_matrix_is_counted(self):
        out = self.judge(record(only=[LARGE]))
        self.assertHalves(out, VIOL, VIOL)
        missing = len(MATRICES) - 1
        self.assertEqual(out["violations"], 2 * missing)

    def test_a_cap_is_read_without_the_white_space_around_it(self):
        for cap in (" bf16", "bf16 ", "\tbf16\n"):
            with self.subTest(cap=cap):
                self.assertHalves(self.judge(record(max_format=cap)), SAT, SAT)
                # The same choices as an uncapped run made, recorded under the cap: the cap forbids the manufacture.
                doc = record()
                doc["selection_inputs"]["max_format"] = cap
                out = self.judge(doc)
                self.assertHalves(out, SAT, VIOL)
        self.assertHalves(self.judge(record(max_format="BF16")), SAT, SAT)  # not the cap: case is kept
        self.assertEqual(R.selected_rules(10 ** 6, "auto", 1, "BF16"), {R.RULE_Q8_LOAD})

    def test_an_operand_with_no_stored_bytes_of_its_own_is_not_judged(self):
        for change in ({"stored_dtype": None, "stored_sha256": None}, {"stored_dtype": "Q8_0", "rule": "unspecified"}):
            with self.subTest(change=change):
                doc = record()
                self.operand(doc, LARGE).update(change)
                self.assertHalves(self.judge(doc), CNE, CNE)

    def test_a_non_finite_tensor_is_outside_the_q8_rule(self):
        real = R.Store.read

        def with_a_nan(store, rid, name):
            raw = real(store, rid, name)
            return b"\xc0\x7f" + raw[2:] if name == LARGE else raw

        with mock.patch.object(R.Store, "read", with_a_nan):
            out = self.judge(record())
        self.assertEqual((out["derivation"], out["status"]), (CNE, CNE), out["reasons"])
        self.assertIn("non-finite", " ".join(out["reasons"]))

    def test_an_operand_whose_order_or_shape_is_not_of_integers(self):
        for change in ({"shape": [64.0, 256.0]}, {"order": "x"}, {"order": True}, {"shape": "64x256"}):
            with self.subTest(change=change):
                doc = record()
                self.operand(doc, LARGE).update(change)
                out = self.judge(doc)
                self.assertEqual(out["derivation"], VIOL)
                self.assertIn("malformed operand", self.findings(out, R.DERIVATION))

    def test_a_container_that_changed_withdraws_both_verdicts(self):
        with mock.patch.object(cli.checkpoint, "container_sha256", lambda root: "0" * 64):
            out = self.judge(record())
        self.assertEqual((out["status"], out["derivation"], out["selection"]), (CNE, CNE, CNE))
        self.assertEqual(out["failed_halves"], [R.DERIVATION, R.SELECTION])
        self.assertEqual({h["status"] for h in out["verdicts"].values()}, {CNE})

    def test_an_answer_reached_before_the_record_is_read_has_both_verdicts(self):
        out = cli.evaluate_residency(R.CONTRACT, os.path.join(self.work, "absent.json"), BASE)
        self.assertEqual((out["status"], out["derivation"], out["selection"]), (CNE, CNE, CNE))
        self.assertEqual(out["failed_halves"], [R.DERIVATION, R.SELECTION])

    def test_a_record_nested_too_deeply_is_an_input_fault_not_a_platform_one(self):
        path = os.path.join(self.work, "deep.json")
        with open(path, "w") as f:
            f.write("[" * 100000 + "]" * 100000)
        out = cli.evaluate_residency(R.CONTRACT, path, BASE)
        self.assertEqual(out["status"], CNE)
        self.assertNotIn("platform", " ".join(out["reasons"]))

    def test_a_file_name_that_is_not_text_is_not_an_exception(self):
        real = R.file_digests

        def with_a_bad_name(root):
            files = real(root)
            files["segments/bad\udcff"] = (0, hashlib.sha256(b"").hexdigest())
            return files

        with mock.patch.object(R, "file_digests", with_a_bad_name):
            out = self.judge(record())
        self.assertEqual((out["status"], out["violations"]), (CNE, 0), out["reasons"])


class RecordFaults(Fixture):
    def test_a_container_the_reader_refuses_is_not_an_exception(self):
        root = os.path.join(self.work, "container")
        shutil.copytree(BASE, root)
        os.symlink("index.json", os.path.join(root, "link.json"))
        out = self.judge(record(), root)
        self.assertHalves(out, CNE, CNE)
        self.assertIn("symlink", self.findings(out, R.DERIVATION))

    def test_a_missing_operand_is_a_violation_of_both_halves(self):
        doc = record()
        doc["operands"] = [o for o in doc["operands"] if o["tensor"] != MIDDLE]
        out = self.judge(doc)
        self.assertHalves(out, VIOL, VIOL)
        self.assertIn(f"{OBJECT}/{MIDDLE}", self.findings(out, R.DERIVATION))
        self.assertIn("not in the record", self.findings(out, R.SELECTION))

    def test_a_duplicate_operand_identifier(self):
        doc = record()
        doc["operands"].append(copy.deepcopy(self.operand(doc, LARGE)))
        out = self.judge(doc)
        self.assertHalves(out, VIOL, VIOL)
        self.assertIn("recorded more than once", self.findings(out, R.DERIVATION))

    def test_an_unknown_rule_is_never_satisfied(self):
        for rule in ("unspecified", "q8-load/2", "stored-bytes-mapped"):
            with self.subTest(rule=rule):
                doc = record()
                self.operand(doc, LARGE).update(rule=rule, resident_sha256=None, resident_digest_layout=None)
                out = self.judge(doc)
                self.assertEqual(out["derivation"], CNE, self.findings(out, R.DERIVATION))
                self.assertNotEqual(out["status"], SAT)
                self.assertIn("is not one this contract derives", self.findings(out, R.DERIVATION))

    def test_policies_the_selection_rule_does_not_specify(self):
        doc = record()
        doc["selection_inputs"]["arithmetic"] = "q8xq8"
        out = self.judge(doc)
        self.assertHalves(out, SAT, CNE)
        doc = record()
        for o in doc["operands"]:
            o["representation_source"] = "transient"
        self.assertHalves(self.judge(doc), SAT, CNE)

    def test_a_record_made_from_another_container(self):
        doc = record()
        doc["container_index_sha256"] = "2" * 64
        out = self.judge(doc)
        self.assertHalves(out, CNE, CNE)
        self.assertIn("another container", " ".join(out["reasons"]))

    def test_malformed_records_are_not_judged(self):
        cases = {
            "schema": lambda d: d.update(schema="larql-residency-record/2"),
            "fields": lambda d: d.update(extra=1),
            "selection_inputs": lambda d: d["selection_inputs"].update(l2_bytes=0),
            "l2 type": lambda d: d["selection_inputs"].update(l2_bytes="16777216"),
            "no operand": lambda d: d.update(operands=[]),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                doc = record()
                change(doc)
                out = self.judge(doc)
                self.assertEqual((out["status"], out["violations"]), (CNE, 0), out["reasons"])
                self.assertTrue(out["reasons"])

    def test_a_malformed_operand(self):
        doc = record()
        del self.operand(doc, LARGE)["rule"]
        out = self.judge(doc)
        self.assertEqual(out["derivation"], VIOL)
        self.assertNotEqual(out["status"], SAT)

    def test_inputs_that_cannot_be_read_and_an_unknown_contract(self):
        path = os.path.join(self.work, "not-json")
        with open(path, "w") as f:
            f.write("{")
        self.assertEqual(cli.evaluate_residency(R.CONTRACT, path, BASE)["status"], CNE)
        self.assertEqual(cli.evaluate_residency(R.CONTRACT, os.path.join(self.work, "absent"), BASE)["status"], CNE)
        out = cli.evaluate_residency("NOPE", path, BASE)
        self.assertIn("unknown contract", out["reasons"][0])


if __name__ == "__main__":
    unittest.main()
