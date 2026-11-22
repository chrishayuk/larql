"""GW-ELEMENT-1 confirmation engineering: synthetic qualification of the builder, admission, sealing and analysis.

No model and no element data are used: captures are synthetic. The only real input is the tokenizer,
used in the sealing tests.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwelem_admit as A  # noqa: E402
import gwelem_analyze as AN  # noqa: E402
import gwelem_candidates as C  # noqa: E402
import gwelem_design as D  # noqa: E402
import gwelem_plan as P  # noqa: E402

FIT = (1, 2, 4)
TARGETS = (3, 8)
LAYERS, HID = 6, 8
WORDS = {"A2": {"number": [",", " the", " atomic", " number"], "symbol": [",", " the", " chemical", " symbol"]},
         "A3": {"number": ["'", "s", " atomic", " number"], "symbol": ["'", "s", " chemical", " symbol"]}}
VOCAB = {}


def tid(s):
    return VOCAB.setdefault(s, 1000 + len(VOCAB))


def decode(ids):
    inv = {v: k for k, v in VOCAB.items()}
    return "".join(inv.get(i, f"<{i}>") for i in ids)


def synth_runs(zs):
    rs = {}
    for z in zs:
        for rel in D.RELATIONS:
            for ph in D.PHRASINGS:
                toks = [2, tid(f"elem{z}")] + [tid(w) for w in WORDS[ph][rel]] + [tid(" is")]
                rs[D.run_id(rel, ph, z)] = {"country": D.key(z), "tokens": toks, "window": [2, 3, 4, 5],
                                            "window_ids": toks[2:6], "expected_step_logit_sha256": ["x"] * len(toks)}
    return rs


def write_capture(d: Path, rs: dict, seed: int):
    rng = np.random.default_rng(seed)
    (d / "vectors").mkdir(parents=True)
    man = {"runs": {}}
    for i, (rid, r) in enumerate(sorted(rs.items())):
        pos = r["window"] + [len(r["tokens"]) - 1]
        a = rng.standard_normal((LAYERS, len(pos), HID)).astype("<f4")
        f = f"vectors/run-{i:04d}.f32"
        a.tofile(d / f)
        man["runs"][rid] = {"file": f, "layers": LAYERS, "positions": pos, "hidden": HID, "window": r["window"],
                            "vector_sha256": {f"{l}|{p}": C.vector_sha256(a[l, j]) for l in range(LAYERS)
                                              for j, p in enumerate(pos)}}
    (d / "manifest.json").write_text(json.dumps(man))


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rs = synth_runs(FIT + TARGETS)
        fit_rs = {k: v for k, v in self.rs.items() if int(k.split("|")[2]) in FIT}
        num_rs = {k: v for k, v in self.rs.items() if int(k.split("|")[2]) in TARGETS and k.startswith("number")}
        sym_rs = {k: v for k, v in self.rs.items() if int(k.split("|")[2]) in TARGETS and k.startswith("symbol")}
        self.dirs = {"fit": self.tmp / "fit", "num": self.tmp / "num", "sym": self.tmp / "sym"}
        for (k, r), s in zip(((n, x) for n, x in (("fit", fit_rs), ("num", num_rs), ("sym", sym_rs))), (1, 2, 3)):
            write_capture(self.dirs[k], r, s)
        allowed_c = set(fit_rs) | set(num_rs)
        self.cview = C.CandidateView(C.load_windows([self.dirs["fit"], self.dirs["num"]], allowed_c), FIT, TARGETS)
        self.rview = C.ReferenceView(C.load_windows([self.dirs["num"], self.dirs["sym"]], set(num_rs) | set(sym_rs)))
        self.store = C.Store(HID)
        self.targets = C.build_targets(self.rs, TARGETS, self.cview, self.rview, self.store, D.ARMS)
        sf = self.tmp / "store.f32"
        ssha = self.store.write(sf)
        self.plan = P.band_plan({"container": "/x", "tokenizer_sha256": "sha256:0"}, self.rs, self.targets, sf, ssha,
                                self.store.index, [1], ["XX"], hidden=HID)
        self.plan["layers"] = LAYERS
        self.caps_all = A.Caps([self.dirs["fit"], self.dirs["num"], self.dirs["sym"]])
        self.caps_c2 = A.Caps([self.dirs["fit"], self.dirs["num"]])

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def admit(self, plan=None, index=None, **kw):
        cells = {rid: r["tokens"] for rid, r in self.rs.items()}
        return A.admit_plan(plan or self.plan, index or self.store.index, self.caps_all, decode, {}, cells, fit=FIT, **kw)


class Builder(Fixture):
    def test_inventory(self):
        n = sum(len(t["arms"]) for t in self.targets)
        self.assertEqual(n, (3 + 3 + 1 + 1) * len(TARGETS))

    def test_candidate_cannot_read_target_symbol(self):
        with self.assertRaises(C.InformationRuleError):
            self.cview.read(D.run_id("symbol", "A2", 3))

    def test_view_refuses_forbidden_runs(self):
        w = C.load_windows([self.dirs["sym"]], {D.run_id("symbol", "A2", 3)})
        with self.assertRaises(C.InformationRuleError):
            C.CandidateView(w, FIT, TARGETS)
        with self.assertRaises(C.InformationRuleError):
            C.CandidateView({}, FIT, (1,))

    def test_c2_reads_only_own_number_and_fit(self):
        self.cview.reads.clear()
        C.c2(self.cview, 3, "A2", D.LAYER, 0)
        own = {r for r in self.cview.reads if r.endswith("|3")}
        self.assertEqual(own, {D.run_id("number", "A2", 3)})
        self.assertTrue(all(int(r.split("|")[2]) in FIT for r in self.cview.reads if not r.endswith("|3")))


class Admission(Fixture):
    def test_clean_plan_admits_and_recomputes(self):
        self.assertEqual(self.admit(), [])
        caps = {"C2": self.caps_c2, "S": self.caps_all, "ID": self.caps_all}
        r = A.recompute_all(self.plan, self.store.index, caps)
        self.assertEqual(r["failures"], [])
        self.assertEqual(r["vectors"], len(self.store.index))
        audit = A.dependency_audit(self.store.index, self.caps_c2, fit=FIT)
        self.assertEqual(audit["failures"], [])

    def _first(self, kind):
        for vid, e in self.store.index.items():
            if e["recipe"]["kind"] == kind:
                return vid
        raise AssertionError(kind)

    def test_planted_own_symbol_input(self):
        idx = copy.deepcopy(self.store.index)
        vid = self._first("C2")
        r = idx[vid]["recipe"]
        own = next(i for i in r["inputs"] if i[0] == "own")
        own[1] = "symbol"
        own[6] = self.caps_all.sha(f"symbol|{r['phrasing']}|{r['target']}", r["layer"], r["slot"])
        idx[vid]["recipe_sha256"] = A.rsha(r)
        plan = copy.deepcopy(self.plan)
        plan["vector_store"]["index"][vid]["recipe_sha256"] = idx[vid]["recipe_sha256"]
        f = self.admit(plan, idx)
        self.assertTrue(any("own symbol state" in x for x in f), f[:5])

    def test_planted_membership_and_hash_and_address(self):
        for mutate, needle in (
                (lambda r: r["inputs"].pop(1), "inputs differ"),
                (lambda r: r.__setitem__("slot", (r["slot"] + 1) % 4), "address"),
                (lambda r: r["inputs"][1].__setitem__(6, "0" * 64), "input hash"),
                (lambda r: r.__setitem__("members", r["members"][:-1]), "members differ")):
            idx = copy.deepcopy(self.store.index)
            vid = self._first("C2")
            mutate(idx[vid]["recipe"])
            idx[vid]["recipe_sha256"] = A.rsha(idx[vid]["recipe"])
            plan = copy.deepcopy(self.plan)
            plan["vector_store"]["index"][vid]["recipe_sha256"] = idx[vid]["recipe_sha256"]
            f = self.admit(plan, idx)
            self.assertTrue(any(needle in x for x in f), (needle, f[:5]))
        idx = copy.deepcopy(self.store.index)
        idx[self._first("S")]["recipe"]["target"] = "8"
        self.assertTrue(any("recipe or index hash" in x for x in self.admit(index=idx)))

    def test_planted_arm_defects(self):
        cases = []
        p = copy.deepcopy(self.plan)
        p["targets"][0]["arms"].pop(1)
        cases.append((p, "arm inventory"))
        p = copy.deepcopy(self.plan)
        a = p["targets"][0]["arms"][1]
        a["target_span"] = a["target_span"][1:]
        cases.append((p, "span"))
        p = copy.deepcopy(self.plan)
        p["targets"][0]["arms"][0].pop("identity")
        cases.append((p, "identity flag"))
        p = copy.deepcopy(self.plan)
        p["targets"][0]["arms"][1]["to_layer"] = D.LAYER + 1
        cases.append((p, "single carrier"))
        p = copy.deepcopy(self.plan)
        p["targets"][0]["arms"][2] = copy.deepcopy(p["targets"][0]["arms"][1]) | {"name": "S-full-L4"}
        cases.append((p, "duplicate"))
        p = copy.deepcopy(self.plan)
        p["targets"][0]["arms"][1]["vectors"] = dict(p["targets"][0]["arms"][0]["vectors"])
        cases.append((p, "identical to the natural carrier"))
        for plan, needle in cases:
            f = self.admit(plan)
            self.assertTrue(any(needle in x for x in f), (needle, f[:5]))

    def test_leaky_vector_with_honest_recipe_is_caught(self):
        """A C2 whose recipe is honest but whose vector was computed from the target's symbol state."""
        vid = self._first("C2")
        e = self.store.index[vid]
        r = e["recipe"]
        leak = self.caps_all.vec(f"symbol|{r['phrasing']}|{r['target']}", r["layer"], r["slot"]).astype("<f4")
        data = np.fromfile(self.plan["vector_store"]["file"], dtype="<f4")
        data[e["offset"]:e["offset"] + HID] = leak
        data.tofile(self.plan["vector_store"]["file"])
        idx = copy.deepcopy(self.store.index)
        idx[vid]["sha256"] = A.vsha(leak)
        plan = copy.deepcopy(self.plan)
        plan["vector_store"]["index"][vid]["sha256"] = idx[vid]["sha256"]
        self.assertEqual(self.admit(plan, idx), [])  # labels alone cannot see it
        res = A.audit_c2(plan, idx, self.caps_c2, self.caps_all, fit=FIT)
        self.assertFalse(res["pass"])
        self.assertIn(vid, res["recompute"]["failures"])

    def test_audit_refuses_c2_caps_with_target_symbol(self):
        res = A.audit_c2(self.plan, self.store.index, self.caps_all, self.caps_all, fit=FIT)
        self.assertFalse(res["pass"])
        self.assertTrue(res["c2_caps_symbol_leaks"])
        self.assertTrue(A.audit_c2(self.plan, self.store.index, self.caps_c2, self.caps_all, fit=FIT)["pass"])

    def test_mislabelled_run_tokens(self):
        plan = copy.deepcopy(self.plan)
        rid = plan["targets"][0]["run"]
        plan["runs"][rid]["tokens"] = list(plan["runs"][rid]["tokens"])
        plan["runs"][rid]["tokens"][1] = tid("elem8")
        self.assertTrue(any("registered prompt" in x for x in self.admit(plan)))

    def test_dependency_audit_detects(self):
        idx = copy.deepcopy(self.store.index)
        vid = self._first("C2")
        idx[vid]["recipe"]["inputs"][1][6] = "f" * 64
        self.assertTrue(A.dependency_audit(idx, self.caps_c2, fit=FIT)["failures"])
        self.assertTrue(A.dependency_audit(self.store.index, self.caps_all, fit=FIT)["failures"])

    def test_recompute_detects_store_tamper(self):
        data = np.fromfile(self.plan["vector_store"]["file"], dtype="<f4")
        data[0] += 1e-3
        data.tofile(self.plan["vector_store"]["file"])
        caps = {"C2": self.caps_c2, "S": self.caps_all, "ID": self.caps_all}
        self.assertTrue(A.recompute_all(self.plan, self.store.index, caps)["failures"])


class Sealing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gwtemplate_baseline_plan as bp
        cls.tok = bp.tokenizer()
        cls.subj = A.sealed_subjects(cls.tok)

    def plan(self, rid, country, text):
        return {"runs": {rid: {"country": country, "tokens": self.tok.encode(text, add_special_tokens=True).ids}}}

    def test_key_and_content(self):
        f = A.seal_check(self.plan("number|A2|26", "26", "For iron, the atomic number is"), self.subj)
        self.assertTrue(any("sealed confirmation key" in x for x in f))
        self.assertTrue(any("sealed element 26" in x for x in f))
        f = A.seal_check(self.plan("number|A2|1", "1", "For iron, the atomic number is"), self.subj)
        self.assertEqual(len(f), 1)
        f = A.seal_check(self.plan("number|A3|1", "1", "Iron's atomic number is"), self.subj)
        self.assertTrue(f)
        self.assertEqual(A.seal_check(self.plan("number|A2|1", "1", "For hydrogen, the atomic number is"), self.subj), [])
        self.assertEqual(A.seal_check(self.plan("number|A2|26", "26", "For iron, the atomic number is"), self.subj,
                                      release=True), [])

    def test_plan_builder_refuses_sealed(self):
        for z in (26, "26", "026"):
            with self.assertRaises(P.SealedError):
                P.runs([z], decode)

    def test_capitalisation_variants(self):
        for text in ("For Iron, the atomic number is", "iron's atomic number is", "Iron's atomic number is"):
            self.assertTrue(A.seal_check(self.plan("number|A2|1", "1", text), self.subj), text)
        self.assertEqual(len(self.subj), 4 * len(D.CONFIRMATION_SPLIT))


def fake_ro(cands, logits):
    lse = float(np.log(np.exp(np.array(logits) - max(logits)).sum()) + max(logits))
    nxt = cands[int(np.argmax(logits))]
    return {"candidate_logits": list(logits), "log_normalizer": lse, "next": nxt}


class Analysis(unittest.TestCase):
    def setUp(self):
        import gwtemplate_baseline_plan as bp
        import gwelem_feasibility as fz
        tok = bp.tokenizer()
        self.els = fz.ELEMENTS
        T0 = AN.Tokens.__new__(AN.Tokens)
        self.space = tok.encode(" ", add_special_tokens=False).ids[0]
        syms = {z: tok.encode(" " + s, add_special_tokens=False).ids[0] for z, s, _ in fz.ELEMENTS}
        self.cands = sorted(set(syms.values()) | {self.space})
        del T0
        members = [z for z in D.FIT + D.DEVELOPMENT]
        self.T = AN.Tokens(tok, self.cands, fz.ELEMENTS, members)
        self.syms = syms

    def records(self, switch_for, other_for=(), bias=0.0):
        targets, rows = [], []
        Ys = D.CONFIRMATION
        for Y in Ys:
            for ph in D.PHRASINGS:
                base = [0.0] * len(self.cands)
                nat = list(base)
                nat[self.T.pos[self.space]] = 10.0
                sep = list(base)
                sep[self.T.pos[self.syms[Y]]] = 10.0
                arms_n, recs = [], {}
                for name in D.ARMS[ph]:
                    arms_n.append({"name": name, "identity": name.startswith("ID")})
                    lg = list(nat)
                    if name.startswith("C2-full") and Y in switch_for:
                        lg = list(sep)
                        lg = [x + (bias if i != self.T.pos[self.syms[Y]] else 0) for i, x in enumerate(lg)]
                    if name.startswith("C2-full") and Y in other_for:
                        lg = list(base)
                        lg[self.T.pos[self.syms[D.FIT[0]]]] = 10.0
                    if name.startswith("S-"):
                        lg = list(sep)
                    recs[name] = {"readout": fake_ro(self.cands, lg)}
                rid_n, rid_s = D.run_id("number", ph, Y), D.run_id("symbol", ph, Y)
                targets.append({"id": rid_n, "run": rid_n, "arms": arms_n})
                rows.append({"id": rid_n, "natural": {"readout": fake_ro(self.cands, nat)}, "arms": recs})
                targets.append({"id": rid_s, "run": rid_s, "arms": [{"name": "ID-full-L4", "identity": True}]})
                rows.append({"id": rid_s, "natural": {"readout": fake_ro(self.cands, sep)},
                             "arms": {"ID-full-L4": {"readout": fake_ro(self.cands, sep)}}})
        return targets, rows

    def test_all_switch_passes_and_is_deterministic(self):
        t, r = self.records(set(D.CONFIRMATION))
        rep = AN.analyse(t, r, self.T)
        g = rep["gates"]["primary"]
        self.assertEqual(g["A"]["share"], 1.0)
        self.assertEqual(g["verdict"], "pass", g)
        self.assertEqual(json.dumps(AN.analyse(t, r, self.T)["gates"]), json.dumps(rep["gates"]))
        self.assertEqual(rep["groups"]["S-full-L4|A2+A3"]["shares"]["r'(Y)"]["mean"], 1.0)
        self.assertNotIn("secondary", rep["gates"])

    def test_no_switch_and_other_element(self):
        t, r = self.records(set(D.CONFIRMATION[:10]), other_for=set(D.CONFIRMATION[10:13]))
        g = AN.analyse(t, r, self.T)["gates"]["primary"]
        self.assertFalse(g["A"]["pass"])
        self.assertEqual(g["verdict"], "no switch")
        self.assertAlmostEqual(g["B"]["other_top1"], 3 / 25)

    def test_bias_fails_absolute_and_nonworse(self):
        t, r = self.records(set(D.CONFIRMATION), bias=0.8)
        g = AN.analyse(t, r, self.T)["gates"]["primary"]
        self.assertTrue(g["A"]["pass"])
        self.assertFalse(g["B"]["b3_mean_pass"])
        self.assertFalse(g["B"]["nonworse_pass"])
        self.assertTrue(g["verdict"].startswith("axis B fail"))

    def test_nonworse_requires_same_cells(self):
        t, r = self.records(set(D.CONFIRMATION))
        arms = [a for a in AN.per_arm(t, r, self.T) if not (a["arm"] == "S-full-L4" and a["target"] == 5)]
        with self.assertRaises(ValueError):
            AN.summarise(arms)


if __name__ == "__main__":
    unittest.main()
