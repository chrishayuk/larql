"""Tests for prec_diag/analysis.py and prec_diag/freeze.py on synthetic
logits and captures with known discrepancies. No model runs here.

Run: python3 -I -m unittest discover -s scripts -p 'test_prec_diag_*.py'
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import equivalence_metrics as em  # noqa: E402
from prec_diag import analysis as an  # noqa: E402
from prec_diag import evidence as ev  # noqa: E402
from prec_diag import freeze as fz  # noqa: E402

N, V = 1040, 32
INPUTS = ("I_rep", "I_nr1", "I_nr2")
rng = np.random.default_rng(0)
BASE = {i: (rng.standard_normal((N, V)) * 3).astype(np.float32) for i in INPUTS}


def arr(value, n=N, late_value=None):
    a = np.full(n, value, dtype=np.float64)
    if late_value is not None:
        a[an.LATE_FIRST:] = late_value
    return a


class Measurements(unittest.TestCase):
    def test_identical_logits_measure_zero(self):
        m = an.position_metrics(BASE["I_rep"][:20], BASE["I_rep"][:20], "reference")
        self.assertEqual(float(m["d"].max()), 0.0)
        self.assertEqual(float(m["tvd"].max()), 0.0)
        self.assertTrue(all(v == em.PASS for v in m["top5"]))
        self.assertTrue((m["top1_ref"] == m["top1_x"]).all())

    def test_d_matches_the_registered_metric_and_names_its_token(self):
        ref = BASE["I_rep"][:40]
        x = ref.copy()
        x[7, 3] += np.float32(0.01)
        m = an.position_metrics(ref, x, "reference", block=6)
        np.testing.assert_array_equal(m["d"], em.max_abs_logprob(x, ref))
        self.assertEqual(int(m["d_token"][7]), 3)
        self.assertEqual(float(m["d"][6]), 0.0)
        self.assertGreater(m["tvd"][7], 0)
        self.assertGreaterEqual(float(m["kl"].min()), -1e-15)
        self.assertAlmostEqual(float(m["max_dlogit"][7]), 0.01, places=6)

    def test_a_shift_of_every_logit_is_not_a_difference_in_logprob(self):
        ref = BASE["I_rep"][:5]
        m = an.position_metrics(ref, ref + np.float32(2.0), "reference")
        self.assertLess(float(m["d"].max()), 1e-5)
        self.assertLess(float(m["max_dcentred"].max()), 1e-5)
        self.assertAlmostEqual(float(m["max_dlogit"].min()), 2.0, places=5)

    def test_the_ratio_floor_is_applied_and_counted(self):
        r, flagged = an.ratio(np.array([2e-5, 1e-3]), np.array([0.0, 1e-3]))
        np.testing.assert_allclose(r, [2.0, 1.0])
        self.assertEqual(flagged, 1)


class R1(unittest.TestCase):
    def call(self, dx, da, valid=True):
        return an.r1({i: dx for i in INPUTS}, {i: da for i in INPUTS}, valid)

    def test_comparable(self):
        self.assertEqual(self.call(arr(1.5e-4), arr(1e-4))["verdict"], "comparable")

    def test_materially_less_accurate_by_median(self):
        self.assertEqual(self.call(arr(5e-4), arr(1e-4))["verdict"], "materially less accurate")

    def test_materially_less_accurate_by_material_positions(self):
        dx = arr(1e-4)
        dx[[10, 20, 30]] = 2e-3
        r = self.call(dx, arr(1e-4))
        self.assertEqual(r["verdict"], "materially less accurate")
        self.assertEqual(len(r["material_discrepancies"]), 9)  # 3 positions x 3 inputs

    def test_three_material_positions_in_all_is_the_boundary(self):
        def call(n):
            dx = {i: arr(1e-4) for i in INPUTS}
            dx["I_nr1"][list(range(n))] = 2e-3
            return an.r1(dx, {i: arr(1e-4) for i in INPUTS}, True)["verdict"]
        self.assertEqual(call(2), "comparable")
        self.assertEqual(call(3), "materially less accurate")

    def test_a_large_x_where_a32_is_also_large_is_not_material(self):
        dx, da = arr(1e-4), arr(1e-4)
        dx[5], da[5] = 2e-3, 2e-3
        self.assertEqual(self.call(dx, da)["material_discrepancies"], [])

    def test_inconclusive_between_the_bands(self):
        self.assertEqual(self.call(arr(3e-4), arr(1e-4))["verdict"], "inconclusive")

    def test_invalid_a64_drops_the_reading(self):
        self.assertEqual(self.call(arr(1e-4), arr(1e-4), valid=False)["verdict"], "invalid")


class R2(unittest.TestCase):
    def d(self, rep, nr):
        return {b: {"I_rep": rep, "I_nr1": nr, "I_nr2": nr} for b in ("C", "D")}

    def test_supported(self):
        self.assertEqual(an.r2(self.d(arr(1e-4, late_value=8e-4), arr(1e-4)))["verdict"], "supported")

    def test_not_supported_by_ratio(self):
        self.assertEqual(an.r2(self.d(arr(1e-4, late_value=8e-4), arr(7e-4)))["verdict"], "not supported")

    def test_not_supported_by_gate_count(self):
        nr = arr(1e-5)
        nr[[1, 2, 3]] = 2e-3
        self.assertEqual(an.r2(self.d(arr(1e-4, late_value=8e-4), nr))["verdict"], "not supported")

    def test_one_or_two_positions_over_the_gate_is_neither(self):
        nr = arr(1e-5)
        nr[[1, 2]] = 2e-3
        self.assertEqual(an.r2(self.d(arr(1e-4, late_value=8e-4), nr))["verdict"], "inconclusive")

    def test_inconclusive(self):
        self.assertEqual(an.r2(self.d(arr(1e-4, late_value=8e-4), arr(5e-4)))["verdict"], "inconclusive")


class R3(unittest.TestCase):
    def call(self, base, h, lside, inputs=INPUTS):
        d = {b: {i: arr(base) for i in inputs} for b in ("C", "D")}
        return an.r3(d, {b: {i: arr(h) for i in inputs} for b in d}, {b: {i: arr(lside) for i in inputs} for b in d})

    def test_contributes(self):
        self.assertEqual(self.call(1e-3, 4e-4, 4e-4)["verdict"], "contributes")

    def test_one_sided(self):
        self.assertEqual(self.call(1e-3, 4e-4, 1e-3)["verdict"], "one-sided")
        self.assertEqual(self.call(1e-3, 1e-3, 4e-4)["verdict"], "one-sided")

    def test_not_material(self):
        self.assertEqual(self.call(1e-3, 9e-4, 1.1e-3)["verdict"], "not material")

    def test_partial(self):
        self.assertEqual(self.call(1e-3, 6e-4, 7e-4)["verdict"], "partial")

    def test_a_tiny_base_is_floored(self):
        r = self.call(1e-7, 1e-7, 1e-7)
        self.assertTrue(r["rho"]["C"]["I_rep"]["floored"])
        self.assertAlmostEqual(r["rho"]["C"]["I_rep"]["rho_H"], 1e-2)


def planes(layers, positions, seed=1, hidden=6, qw=8, kw=4):
    g = np.random.default_rng(seed)
    out = {}
    for key in an.site_sequence(layers):
        w = qw if key[0].startswith("query") else kw if key[0].startswith("key") else hidden
        out[key] = g.standard_normal((len(positions), w)).astype(np.float32)
    return out


class R4(unittest.TestCase):
    POS = sorted(set(an.REV2_SET + an.BOUNDARY_SET + an.P43_SET + an.P16_SET))

    def test_identical_planes_have_no_divergence(self):
        p = planes(2, self.POS)
        r = an.r4(p, p, self.POS, 2, f32_pair=True)["per_set"]["all"]
        self.assertIsNone(r["first_difference"])
        self.assertEqual(r["first_material_divergence"], "none")

    def test_first_difference_material_divergence_and_amplification(self):
        ref = planes(3, self.POS)
        x = {k: v.copy() for k, v in ref.items()}
        x[("ffn_input", 1)] = (x[("ffn_input", 1)] * np.float32(1 + 1e-6)).astype(np.float32)   # tiny, first
        for k in an.site_sequence(3)[an.site_sequence(3).index(("ffn_output", 1)):]:
            x[k] = (ref[k] * np.float32(1 + 1e-2)).astype(np.float32)                          # amplified
        r = an.r4(x, ref, self.POS, 3, f32_pair=True)["per_set"]["all"]
        self.assertEqual(r["first_difference"], ["ffn_input", 1])
        self.assertEqual(r["first_material_divergence"], ["ffn_output", 1])
        self.assertEqual(r["largest_amplification"]["site"], ["ffn_output", 1])
        self.assertEqual(r["largest_amplification"]["input_site"], ["ffn_input", 1])
        self.assertEqual(r["largest_output_among_floored_inputs"]["site"], ["ffn_input", 1])
        floored = r["largest_output_among_floored_inputs"]
        self.assertIsNone(floored["ratio"])
        self.assertTrue(floored["ratio_status"].startswith("undefined"))
        self.assertEqual(floored["input_rel_l2"], 0.0)   # raw errors kept beside the undefined ratio
        self.assertGreater(floored["output_rel_l2"], 0.0)
        self.assertEqual(r["largest_amplification"]["ratio_status"], "defined")
        # The earliest material divergence and the largest amplification are separate markers.
        self.assertIn("first_material_divergence", r)

    def test_first_difference_is_undefined_against_a64_and_at_transformed_sites(self):
        ref = planes(1, self.POS)
        x = {k: v + np.float32(1e-3) for k, v in ref.items()}
        self.assertIn("undefined", an.r4(x, ref, self.POS, 1, f32_pair=False)["per_set"]["all"]["first_difference"])
        r = an.r4(x, ref, self.POS, 1, f32_pair=True, transformed_sites=frozenset({"embedding"}))
        self.assertEqual(r["per_set"]["all"]["first_difference"]["undefined_at"], ["embedding", None])

    def test_spike_localisation_finds_the_boundary_site(self):
        ref = planes(2, self.POS)
        x = {k: (v * np.float32(1 + 1e-5)).astype(np.float32) for k, v in ref.items()}
        rows = [i for i, p in enumerate(self.POS) if p in an.BOUNDARY_SET]
        x[("attention_output", 1)][rows] *= np.float32(1.01)
        r = an.r4(x, ref, self.POS, 2, f32_pair=True, boundary=an.BOUNDARY_SET, matched=an.MATCHED_SET)
        self.assertEqual(r["spike_localisation"]["site"], ["attention_output", 1])

    def test_no_separating_site_is_amplified_at_the_head(self):
        ref = planes(1, self.POS)
        x = {k: (v * np.float32(1 + 1e-5)).astype(np.float32) for k, v in ref.items()}
        r = an.r4(x, ref, self.POS, 1, f32_pair=True, boundary=an.BOUNDARY_SET, matched=an.MATCHED_SET)
        self.assertEqual(r["spike_localisation"]["site"], "amplified at the head")

    def test_predecessors_cross_layers_and_reach_the_head(self):
        self.assertEqual(an.predecessor_key(("attention_input", 0), 3), ("embedding", None))
        self.assertEqual(an.predecessor_key(("attention_input", 2), 3), ("residual", 1))
        self.assertEqual(an.predecessor_key(("head_input", None), 3), ("residual", 2))
        self.assertIsNone(an.predecessor_key(("embedding", None), 3))


class R5R6Outcome(unittest.TestCase):
    def test_behaviour_changing_and_tail_only(self):
        ref = np.zeros((3, V), np.float32)
        ref[:, 0] = 10.0
        x = ref.copy()
        x[0, 1] = 11.0                       # top-1 flips: behaviour changing
        x[1, 31] = -0.0 + np.float32(-40.0)  # a deep-tail token moves: tail only
        m = an.position_metrics(ref, x, "reference")
        r = an.r5(m)
        self.assertEqual(r["top1_mismatches"], 1)
        self.assertEqual(r["behaviour_changing_positions"], 1)
        self.assertEqual(r["tail_only_positions"], 1)

    def test_a_tail_move_with_an_order_change_is_not_tail_only(self):
        ref = np.full((1, V), -40.0, np.float32)
        ref[0, 0] = 10.0
        ref[0, 1:7] = [0.0, -2e-4, -4e-4, -6e-4, -8e-4, -1e-3]
        x = ref.copy()
        x[0, 2], x[0, 3] = ref[0, 3], ref[0, 2]   # a near-tie swap inside the top 5
        x[0, 31] = -39.0                          # and a deep-tail token moves by a whole unit
        m = an.position_metrics(ref, x, "reference")
        self.assertGreater(m["d"][0], an.GATE)
        self.assertLess(m["tvd"][0], an.R5_TVD_TAIL_ONLY)
        self.assertEqual(m["top5"][0], em.TIE_SWAP)
        self.assertEqual(an.r5(m)["tail_only_positions"], 0)

    def test_r6_reports_identity_or_the_differences(self):
        a = BASE["I_rep"][:10]
        self.assertEqual(an.r6(a, a.copy(), a, "reference"), {"E1_identical": True})
        b = a.copy()
        b[3, 3] += 1
        r = an.r6(a, b, a, "reference")
        self.assertFalse(r["E1_identical"])
        self.assertEqual(r["d_cached_batched"]["argmax"], 3)

    def test_outcome_table(self):
        ok = {"C": {"verdict": "comparable"}, "D": {"verdict": "comparable"}}
        self.assertEqual(an.outcome(ok, "supported", "not material")["row"], an.OUTCOMES[("supported", "not material")])
        self.assertIn("one-sided", an.outcome(ok, "supported", "one-sided")["row"])
        self.assertIn("no conclusion forced", an.outcome(ok, "inconclusive", "contributes")["row"])
        self.assertIn("no conclusion forced", an.outcome(ok, "supported", "partial")["row"])
        bad = {"C": {"verdict": "materially less accurate"}, "D": {"verdict": "comparable"}}
        o = an.outcome(bad, "supported", "not material")
        self.assertEqual(o["per_backend"]["C"]["conclusion"], "possible LARQL computation error")
        self.assertEqual(o["per_backend"]["D"], {"conclusion": an.OUTCOMES[("supported", "not material")],
                                                 "accuracy": "comparable"})
        self.assertTrue(o["accuracy_qualified"])

    def test_an_unknown_r1_leaves_the_outcome_agreement_only_never_comparable(self):
        for verdict in ("not_admitted", "invalid", None):
            o = an.outcome({"C": {"verdict": verdict}, "D": {"verdict": "comparable"}}, "supported", "not material")
            self.assertEqual(o["per_backend"]["C"]["scope"], "agreement only", verdict)
            self.assertTrue(o["per_backend"]["C"]["accuracy"].startswith("unqualified"), verdict)
            self.assertNotIn("scope", o["per_backend"]["D"])
            self.assertFalse(o["accuracy_qualified"])


class Analyse(unittest.TestCase):
    ARMS = ("A32", "A32s", "A32r", "A64", "C", "D", "C32a", "D32a")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.runs = {}
        noise = {"A32": 0, "A32s": 1e-6, "A32r": 2e-6, "A64": 3e-6, "C": 4e-6, "D": 5e-6, "C32a": 3e-6,
                 "D32a": 4e-6}
        for arm in self.ARMS:
            for i in INPUTS:
                g = np.random.default_rng(hash((arm, i)) % 2**32)
                logits = BASE[i].astype(np.float64) + noise[arm] * g.standard_normal((N, V))
                rr = ev.RunRecord.begin(self.root / "runs" / f"{arm}.{i}", arm, "test", {"arm": arm, "input": i})
                if arm == "A64":
                    logits.astype("<f8").tofile(rr.run_dir / "logits.f64")
                    rr.complete(["logits.f64"])
                else:
                    logits.astype("<f4").tofile(rr.run_dir / "logits.f32")
                    rr.complete(["logits.f32"])
                self.runs[(arm, i)] = rr.run_dir
        self.expected = {k: {"arm": k[0], "input": k[1]} for k in self.runs}

    def analyse(self):
        rec = an.analyse(runs=self.runs, expected=self.expected, vocab=V, out_dir=self.root / "analysis",
                         log_path=self.root / "results.jsonl")
        return rec, {e["reading"]: e for e in ev.ResultsLog(self.root / "results.jsonl").verify()}

    def test_every_reading_is_logged_and_the_analysis_is_admitted(self):
        rec, log = self.analyse()
        self.assertEqual(rec["status"], ev.COMPLETE, rec.get("detail"))
        for name in ("R1 accuracy C", "R1 accuracy D", "R2 repetition amplification", "R3 RoPE construction",
                     "Outcome"):
            self.assertEqual(log[name]["status"], ev.READING_COMPLETE, name)
        self.assertEqual(log["R1 accuracy C"]["values"]["verdict"] in
                         ("comparable", "inconclusive", "materially less accurate"), True)
        self.assertEqual(len(rec["identity"]["consumed_records"]), len(self.runs))
        self.assertTrue(ev.admit(self.root / "analysis").admitted)
        self.assertEqual(rec["identity"]["thresholds"], json.loads(json.dumps(an.thresholds())))

    def test_a_tampered_input_makes_its_readings_not_admitted(self):
        p = self.runs[("C32a", "I_rep")] / "logits.f32"
        data = bytearray(p.read_bytes())
        data[0] ^= 1
        p.write_bytes(bytes(data))
        _, log = self.analyse()
        self.assertEqual(log["R3 RoPE construction"]["status"], ev.NOT_ADMITTED)
        self.assertIsNone(log["R3 RoPE construction"]["values"])
        self.assertEqual(log["R2 repetition amplification"]["status"], ev.READING_COMPLETE)
        self.assertEqual(log["Outcome"]["status"], ev.NOT_ADMITTED)

    def test_a_missing_a64_drops_accuracy_but_not_agreement(self):
        for i in INPUTS:
            del self.runs[("A64", i)]
        _, log = self.analyse()
        self.assertEqual(log["R1 accuracy C"]["status"], ev.NOT_ADMITTED)
        self.assertEqual(log["R2 repetition amplification"]["status"], ev.READING_COMPLETE)
        out = log["Outcome"]
        self.assertEqual(out["status"], ev.READING_COMPLETE)
        self.assertFalse(out["values"]["accuracy_qualified"])
        for b in ("C", "D"):
            self.assertEqual(out["values"]["per_backend"][b]["scope"], "agreement only")
        # The outcome names the exact log lines it was derived from, with their statuses.
        cited = {s["reading"]: s for s in out["sources"]}
        self.assertEqual(set(cited), {"R1 accuracy C", "R1 accuracy D", "R2 repetition amplification",
                                      "R3 RoPE construction"})
        self.assertEqual(cited["R1 accuracy C"]["status"], ev.NOT_ADMITTED)
        lines = ev.ResultsLog(self.root / "results.jsonl").verify()
        persisted = [ln for ln in (self.root / "results.jsonl").read_bytes().split(b"\n") if ln]
        for src in out["sources"]:
            self.assertEqual(lines[src["seq"]]["reading"], src["reading"])
            # The digest is of the exact bytes persisted for that line.
            self.assertEqual(ev.sha256_bytes(persisted[src["seq"]]), src["line_sha256"])
            # ...which is also the chain digest the next line records.
            self.assertEqual(lines[src["seq"] + 1]["prev_line_sha256"], src["line_sha256"])

    def test_a_tampered_a32s_never_reaches_an_admitted_reading(self):
        p = self.runs[("A32s", "I_nr2")] / "logits.f32"
        data = bytearray(p.read_bytes())
        data[100] ^= 1
        p.write_bytes(bytes(data))
        _, log = self.analyse()
        self.assertEqual(log["R1 yardstick A32s"]["status"], ev.NOT_ADMITTED)
        self.assertIsNone(log["R1 yardstick A32s"]["values"])
        for b in ("C", "D"):
            r1 = log[f"R1 accuracy {b}"]
            self.assertEqual(r1["status"], ev.READING_COMPLETE)
            self.assertFalse(any("A32s" in c["run_dir"] for c in r1["consumed"]))
            self.assertNotIn("r_s", json.dumps(r1["values"]))
        for name, line in log.items():
            if line["status"] == ev.READING_COMPLETE:
                self.assertFalse(any("A32s.I_nr2" in c["run_dir"] for c in line["consumed"]), name)

    def test_an_identity_mismatch_is_not_admitted(self):
        self.expected[("A32", "I_nr1")] = {"arm": "A32", "input": "I_nr2"}
        _, log = self.analyse()
        self.assertEqual(log["R2 repetition amplification"]["status"], ev.NOT_ADMITTED)

    def test_capture_readings_pair_larql_with_hf(self):
        pos = sorted(set(an.REV2_SET + an.BOUNDARY_SET + an.P43_SET + an.P16_SET))
        ref = planes(2, pos)
        for name, p, dt in (("C-capture", ref, "<f4"), ("D-capture", ref, "<f4"), ("A32-capture", ref, "<f4"),
                            ("A64-capture", ref, "<f8")):
            for i in ("I_rep", "I_nr1"):
                rr = ev.RunRecord.begin(self.root / "runs" / f"{name}.{i}", name, "test", {"arm": name})
                sites = rr.run_dir / "sites"
                sites.mkdir()
                files = []
                for (site, layer), plane in p.items():
                    fname = f"{site}.f" if layer is None else f"{site}.L{layer:03}.f"
                    plane.astype(dt).tofile(sites / fname)
                    files.append({"site": site, "layer": layer, "rows": plane.shape[0], "width": plane.shape[1],
                                  "file": fname, "dtype": dt})
                (sites / "manifest.json").write_text(json.dumps({"positions": pos, "files": files,
                                                                  "layers": [{"layer": 0}, {"layer": 1}]}))
                rr.complete(["sites/manifest.json"] + [f"sites/{f['file']}" for f in files])
                self.runs[(name, i)] = rr.run_dir
        rec, log = self.analyse()
        self.assertEqual(rec["status"], ev.COMPLETE, rec.get("detail"))
        r = log["R4 divergence C vs A32 I_rep"]
        self.assertEqual(r["status"], ev.READING_COMPLETE)
        self.assertIsNone(r["values"]["per_set"]["all"]["first_difference"])
        self.assertIn("undefined", log["R4 divergence C vs A64 I_rep"]["values"]["per_set"]["all"]["first_difference"])


class Freeze(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        files = {}
        for label in fz.REQUIRED_FILES:
            p = self.root / "files" / label
            p.parent.mkdir(exist_ok=True)
            p.write_text(f"content of {label}\n")
            files[label] = str(p)
        c = self.root / "c.vindex3"
        (c / "segments").mkdir(parents=True)
        (c / "segments" / "a.bin").write_bytes(b"segment")
        (c / "system_graph.json").write_text("{}")
        (c / "index.json").write_text(json.dumps({"representations": {"a@BF16": {
            "segment": "segments/a.bin", "segment_sha256": ev.sha256_bytes(b"segment"), "payload_sha256": "p"}}}))
        ck = self.root / "ckpt"
        ck.mkdir()
        (ck / "model.safetensors").write_bytes(b"weights")
        (ck / "config.json").write_text("{}")
        self.spec = {"files": files, "container": str(c), "checkpoint": {"directory": str(ck), "revision": "abc"},
                     "values": {k: k for k in fz.REQUIRED_VALUES}}
        self.c, self.ck = c, ck

    def test_build_is_deterministic_and_written_once(self):
        a, b = fz.build(self.spec), fz.build(self.spec)
        self.assertEqual(ev.canonical_json(a), ev.canonical_json(b))
        digest = fz.write(self.root / "FREEZE.json", a)
        self.assertEqual(digest, ev.sha256_file(self.root / "FREEZE.json"))
        with self.assertRaises(fz.FreezeError):
            fz.write(self.root / "FREEZE.json", a)
        self.assertEqual(fz.preflight(self.root / "FREEZE.json", digest), [])
        self.assertEqual(a["thresholds"], an.thresholds())

    def test_an_incomplete_spec_is_refused(self):
        del self.spec["files"]["script.hf_oracle"]
        with self.assertRaisesRegex(fz.FreezeError, "script.hf_oracle"):
            fz.build(self.spec)
        self.spec["files"]["script.hf_oracle"] = str(self.root / "absent")
        with self.assertRaisesRegex(fz.FreezeError, "not a file"):
            fz.build(self.spec)

    def test_a_segment_that_differs_from_its_index_is_refused(self):
        (self.c / "segments" / "a.bin").write_bytes(b"other")
        with self.assertRaisesRegex(fz.FreezeError, "differs from its index"):
            fz.build(self.spec)

    def test_preflight_names_every_drift(self):
        digest = fz.write(self.root / "FREEZE.json", fz.build(self.spec))
        Path(self.spec["files"]["script.analysis"]).write_text("edited\n")
        (self.ck / "model.safetensors").write_bytes(b"other weights")
        (self.c / "segments" / "a.bin").write_bytes(b"other")
        with mock.patch.object(an, "thresholds", return_value={"epsilon": 1}):
            problems = fz.preflight(self.root / "FREEZE.json", digest)
        self.assertTrue(any("script.analysis" in p for p in problems))
        self.assertTrue(any("checkpoint" in p for p in problems))
        self.assertTrue(any("container" in p for p in problems))
        self.assertTrue(any("thresholds" in p for p in problems))
        self.assertTrue(fz.preflight(self.root / "FREEZE.json", "0" * 64)[0].startswith("FREEZE.json differs"))


if __name__ == "__main__":
    unittest.main()
