"""Test 3 derived-vector path: independent admission and recipe recomputation.

It shares no code with the candidate builder (`gwtemplate_test3_candidates.py`)
or the runner. It re-implements:
- the canonical recipe hash;
- `vector_sha256`;
- the fit/validation split;
- the candidate formulas, as documented in design d2 §4.

**admit(...)** checks every injected (layer, position) → vector against its
recipe:
- the recipe hash;
- exact layer, slot and target position;
- target country, relation and phrasing;
- the arm kind;
- per-kind roles and membership, including the information rule;
- every input vector hash, against the capture manifests;
- the vector hash recorded in the runner output.

So a correctly hashed vector attached to the wrong recipe or address is
rejected. The runner cannot see recipes and would not catch that.

**recompute(...)** rebuilds a recipe's vector from the capture files, in
float64 over sorted members and cast to f32 once, and compares hashes.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

RELATIONS = ("capital", "currency_code_iso4217")
FITVAL_SALT = "gw-template-factorial/test3-fitval/v1|"
N_VAL = 20


def vsha(v: np.ndarray) -> str:
    return hashlib.sha256(struct.pack("<Q", v.size) + np.asarray(v, dtype="<f4").tobytes()).hexdigest()


def rsha(recipe: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_set(dev: list[str]) -> set[str]:
    o = sorted(dev, key=lambda c: hashlib.sha256((FITVAL_SALT + c).encode()).hexdigest())
    return set(o[N_VAL:])


def other(r: str) -> str:
    return RELATIONS[1 - RELATIONS.index(r)]


class Captures:
    """One or more stage-1b-format capture directories (manifest + f32 files)."""

    def __init__(self, dirs: list[Path]):
        self.entries, self.dirs = {}, {}
        for d in dirs:
            m = json.loads((d / "manifest.json").read_text())
            for rid, e in m["runs"].items():
                if rid in self.entries:
                    raise ValueError(f"run {rid} in two captures")
                self.entries[rid], self.dirs[rid] = e, d
        self._cache = {}

    def window_sha(self, rid, layer, slot):
        e = self.entries[rid]
        return e["vector_sha256"][f"{layer}|{e['window'][slot]}"]

    def window(self, rid) -> np.ndarray:
        if rid not in self._cache:
            e = self.entries[rid]
            a = np.fromfile(self.dirs[rid] / e["file"], dtype="<f4").reshape(e["layers"], len(e["positions"]), e["hidden"])
            self._cache[rid] = a[:, :len(e["window"]), :]
        return self._cache[rid]


ROLES = {  # kind -> allowed (role, relation-is-original?) patterns
    "ID": {"copy": "r"}, "S": {"copy": "rp"}, "C0": {"copy": "rp"}, "E0": {"copy": "r"},
    "C1": {"member": "rp"}, "C1A": {"member": "rp"}, "C1B": {"member": "rp"}, "C1E": {"member": "r"},
    "C2": {"own": "r", "plus": "rp", "minus": "r"}, "QUALMEAN": {"member": "any"},
}


def check_recipe(rec: dict, l: int, p: int, run_id: str, window: list[int], kind: str, caps: Captures,
                 fit: set[str] | None, holdout: set[str]) -> list[str]:
    f = []
    rel, ph, Y = run_id.split("|")
    if rec.get("layer") != l or rec.get("target_position") != p or rec.get("slot") != window.index(p):
        f.append("address")
    if (rec.get("target"), rec.get("relation"), rec.get("phrasing")) != (Y, rel, ph):
        f.append("target cell")
    if rec.get("kind") != kind:
        f.append(f"kind {rec.get('kind')} != {kind}")
    roles = ROLES.get(kind, {})
    members = set()
    for role, irel, iph, c, il, js, sha in rec.get("inputs", []):
        want = roles.get(role)
        if want is None or (want == "r" and irel != rel) or (want == "rp" and irel != other(rel)):
            f.append(f"input role {role}/{irel}")
        if iph != ph or il != l:
            f.append("input phrasing or layer")
        if c in holdout:
            f.append(f"holdout input {c}")
        if js != rec.get("source_slot"):
            f.append("input slot")
        try:
            if caps.window_sha(f"{irel}|{iph}|{c}", il, js) != sha:
                f.append(f"input hash {irel}|{iph}|{c} {il}|{js}")
        except (KeyError, IndexError):
            f.append(f"input not captured {irel}|{iph}|{c}")
        if role in ("member", "plus", "minus"):
            members.add(c)
        if role == "own" and c != Y:
            f.append("own input is not the target")
    if kind in ("C1", "C1A", "C1B", "C1E", "C2"):
        if Y in members:
            f.append("information rule: target in its own aggregate")
        if fit is not None and not members <= fit:
            f.append("information rule: non-fit member")
        if set(rec.get("members", [])) != members:
            f.append("members differ from inputs")
        if kind == "C1" and fit is not None and members != fit - {Y}:
            f.append("C1 membership is not fit minus target")
    if kind in ("C0", "E0") and (rec.get("donor") == Y or [i[3] for i in rec.get("inputs", [])] != [rec.get("donor")]):
        f.append("donor reference malformed")
    if kind in ("ID", "S") and [i[3] for i in rec.get("inputs", [])] != [Y]:
        f.append("self reference malformed")
    return f


def admit(plan: dict, index: dict, rows: list[dict] | None, caps: Captures, dev: list[str] | None) -> list[str]:
    fails = []
    fit = fit_set(dev) if dev else None
    holdout = set(plan["holdout"])
    out = {r["id"]: r for r in rows} if rows is not None else None
    for t in plan["targets"]:
        run = plan["runs"][t["run"]]
        for a in t["arms"]:
            kind = a["name"].split("-")[0]
            span = a["target_span"]
            want = {f"{l}|{p}" for l in range(a["from_layer"] - 1, plan["layers"]) for p in span}
            if set(a["vectors"]) != want:
                fails.append(f"{t['id']} {a['name']}: coverage")
            for key, vid in a["vectors"].items():
                l, p = map(int, key.split("|"))
                e = index.get(vid)
                if e is None:
                    fails.append(f"{t['id']} {a['name']} {key}: unknown vector")
                    continue
                if rsha(e["recipe"]) != e["recipe_sha256"]:
                    fails.append(f"{t['id']} {a['name']} {key}: recipe hash")
                for x in check_recipe(e["recipe"], l, p, t["run"], run["window"], kind, caps, fit, holdout):
                    fails.append(f"{t['id']} {a['name']} {key}: {x}")
                if out is not None:
                    rec = out[t["id"]]["arms"][a["name"]]
                    if rec["vectors"].get(key) != e["sha256"]:
                        fails.append(f"{t['id']} {a['name']} {key}: output vector hash")
                    if e["recipe_sha256"] not in rec["recipes"]:
                        fails.append(f"{t['id']} {a['name']} {key}: recipe not recorded")
            if out is not None and a.get("identity") and not out[t["id"]]["arms"][a["name"]].get("identity_bit_exact"):
                fails.append(f"{t['id']} {a['name']}: identity not bit-exact")
    return fails


def recompute(rec: dict, caps: Captures) -> np.ndarray:
    """Documented formulas (design d2 §4): float64 over sorted members, cast to f32 once."""
    l, js, ph, kind = rec["layer"], rec["source_slot"], rec["phrasing"], rec["kind"]
    get = lambda rel, c: caps.window(f"{rel}|{ph}|{c}")[l, js].astype(np.float64)  # noqa: E731

    def mean(role):
        xs = sorted((c, irel) for r_, irel, _, c, *_ in rec["inputs"] if r_ == role)
        acc = None
        for c, irel in xs:
            acc = get(irel, c) if acc is None else acc + get(irel, c)
        return (acc / len(xs)).astype("<f4")

    if kind in ("ID", "S", "C0", "E0"):
        (_, irel, _, c, *_), = rec["inputs"]
        return caps.window(f"{irel}|{ph}|{c}")[l, js].copy()
    if kind in ("C1", "C1A", "C1B", "C1E", "QUALMEAN"):
        return mean("member")
    if kind == "C2":
        (_, irel, _, c, *_), = [i for i in rec["inputs"] if i[0] == "own"]
        own = get(irel, c)
        # The documented order is own + (mean_plus - mean_minus); float addition is not associative.
        return (own + (mean("plus").astype(np.float64) - mean("minus").astype(np.float64))).astype("<f4")
    raise ValueError(kind)
