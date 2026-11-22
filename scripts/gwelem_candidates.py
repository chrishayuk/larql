"""GW-ELEMENT-1 confirmation: the C2 builder and its information rule.

For a target element Y (atomic-number prompt, phrasing p, window slot j, carrier layer 4):

    C2(Y, p, j) = h(Y, number, p, j) + mean_{e in F} h(e, symbol, p, j) - mean_{e in F} h(e, number, p, j)

F is the frozen 45-element fit population.

**Information rule (owner, 2026-11-21).** A candidate may read only:
- the target's own **number**-relation window;
- the fit population's windows in both relations.

The target's own **symbol**-relation state never enters a candidate. The rule is enforced structurally:
- `CandidateView` is built from an explicit allow-list of run ids. Any read outside it raises
  `InformationRuleError`.
- The confirmation execution constructs it from the fit capture and the confirmation **number**
  capture only. The confirmation symbol capture is a separate directory and is never opened by it.
- Y in F is refused, so there is no leave-one-out case.

References are built only through `ReferenceView`, and are labelled as references:
- **S:** Y's own symbol window, the natural symbol-state reference;
- **ID:** Y's own window in the run's relation.

**Numerics (fixed):**
- each mean is a float64 sum over F in ascending atomic number, divided by |F| and cast to f32 once;
- C2 = f32(float64(own) + (float64(mean_plus) - float64(mean_minus))).
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

import gwelem_design as D


class InformationRuleError(Exception):
    pass


def vector_sha256(v: np.ndarray) -> str:
    return hashlib.sha256(struct.pack("<Q", v.size) + np.asarray(v, dtype="<f4").tobytes()).hexdigest()


def recipe_sha(recipe: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_windows(cap_dirs: list[Path], allowed: set[str]) -> dict:
    """{run id: f32 [layers, window slots, hidden]} for allowed run ids only; others are never read."""
    w = {}
    for d in cap_dirs:
        man = json.loads((Path(d) / "manifest.json").read_text())
        for rid, e in man["runs"].items():
            if rid not in allowed:
                continue
            if rid in w:
                raise ValueError(f"{rid} in two captures")
            a = np.fromfile(Path(d) / e["file"], dtype="<f4").reshape(e["layers"], len(e["positions"]), e["hidden"])
            w[rid] = a[:, :len(e["window"]), :]
    missing = allowed - set(w)
    if missing:
        raise ValueError(f"allowed runs not captured: {sorted(missing)[:5]}")
    return w


class CandidateView:
    """The only capture access a candidate has."""

    def __init__(self, windows: dict, fit, targets):
        self.fit = tuple(sorted(int(z) for z in fit))
        self.targets = set(int(z) for z in targets)
        if self.targets & set(self.fit):
            raise InformationRuleError("a target is in the fit population")
        allowed = {D.run_id(r, p, z) for z in self.fit for r in D.RELATIONS for p in D.PHRASINGS}
        allowed |= {D.run_id(D.SOURCE, p, z) for z in self.targets for p in D.PHRASINGS}
        extra = set(windows) - allowed
        if extra:
            raise InformationRuleError(f"candidate view holds forbidden runs: {sorted(extra)[:5]}")
        self._w = windows
        self.reads: list[str] = []

    def read(self, rid: str) -> np.ndarray:
        if rid not in self._w:
            raise InformationRuleError(f"candidate may not read {rid}")
        self.reads.append(rid)
        return self._w[rid]


class ReferenceView:
    def __init__(self, windows: dict):
        self._w = windows

    def read(self, rid: str) -> np.ndarray:
        return self._w[rid]


def _mean(view: CandidateView, rel: str, ph: str, layer: int, slot: int) -> np.ndarray:
    acc = None
    for z in view.fit:  # ascending atomic number; the sum starts from +0.0 (as the documented formula)
        v = view.read(D.run_id(rel, ph, z))[layer, slot].astype(np.float64)
        acc = (np.zeros_like(v) if acc is None else acc) + v
    return (acc / len(view.fit)).astype("<f4")


def c2(view: CandidateView, Y: int, ph: str, layer: int, slot: int):
    """(vector f32, recipe without address) for target Y's number prompt."""
    if Y not in view.targets:
        raise InformationRuleError(f"{Y} is not a declared target")
    own_id = D.run_id(D.SOURCE, ph, Y)
    own = view.read(own_id)[layer, slot]
    plus = _mean(view, D.INJECTED, ph, layer, slot)
    minus = _mean(view, D.SOURCE, ph, layer, slot)
    v = (own.astype(np.float64) + (plus.astype(np.float64) - minus.astype(np.float64))).astype("<f4")
    inputs = [["own", D.SOURCE, ph, D.key(Y), layer, slot, vector_sha256(own)]]
    inputs += [["plus", D.INJECTED, ph, D.key(z), layer, slot,
                vector_sha256(view.read(D.run_id(D.INJECTED, ph, z))[layer, slot])] for z in view.fit]
    inputs += [["minus", D.SOURCE, ph, D.key(z), layer, slot,
                vector_sha256(view.read(D.run_id(D.SOURCE, ph, z))[layer, slot])] for z in view.fit]
    return v, {"kind": "C2", "target": D.key(Y), "relation": D.SOURCE, "phrasing": ph,
               "members": [D.key(z) for z in view.fit], "formula": D.FORMULA, "inputs": inputs}


def reference(view: ReferenceView, kind: str, Y: int, rel: str, ph: str, layer: int, slot: int):
    """S: Y's own symbol window injected into Y's number prompt. ID: Y's own window in the run's relation."""
    src = {"S": D.INJECTED, "ID": rel}[kind]
    if kind == "S" and rel != D.SOURCE:
        raise ValueError("S is defined for number targets only")
    v = view.read(D.run_id(src, ph, Y))[layer, slot].copy()
    return v, {"kind": kind, "target": D.key(Y), "relation": rel, "phrasing": ph, "formula": D.FORMULA,
               "inputs": [["copy", src, ph, D.key(Y), layer, slot, vector_sha256(v)]]}


class Store:
    def __init__(self, hidden: int):
        self.hidden, self.chunks, self.index, self.n = hidden, [], {}, 0

    def add(self, vec: np.ndarray, recipe: dict, layer: int, slot: int, position: int) -> str:
        rec = dict(recipe, layer=layer, slot=slot, target_position=position)
        rs, vs = recipe_sha(rec), vector_sha256(vec)
        vid = f"{rs[7:23]}-{vs[:16]}"
        if vid not in self.index:
            self.index[vid] = {"offset": self.n * self.hidden, "sha256": vs, "recipe_sha256": rs, "recipe": rec}
            self.chunks.append(np.asarray(vec, dtype="<f4"))
            self.n += 1
        return vid

    def write(self, path: Path) -> str:
        data = np.concatenate(self.chunks) if self.chunks else np.zeros(0, "<f4")
        Path(path).write_bytes(data.astype("<f4").tobytes())
        return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_targets(rs: dict, Ys, cview: CandidateView | None, rview: ReferenceView, store: Store,
                  arms_by_ph: dict, reference_arms=D.REFERENCE_ARMS) -> list[dict]:
    """Band-plan targets for Ys: number targets with arms_by_ph[ph], symbol targets with reference_arms.

    cview may be None only when no C2 arm is requested.
    """
    import gwelem_plan as P
    L = D.LAYER
    targets = []
    for Y in sorted(Ys):
        for ph in D.PHRASINGS:
            ref = reference_arms[ph] if isinstance(reference_arms, dict) else reference_arms
            for rel, names in ((D.SOURCE, arms_by_ph[ph]), (D.INJECTED, ref)):
                rid = D.run_id(rel, ph, Y)
                run = rs[rid]
                arms = []
                for name in names:
                    kind, span, _ = name.split("-")
                    vec = {}
                    for j in D.SPANS[span]:
                        if kind == "C2":
                            if cview is None:
                                raise InformationRuleError("C2 requested without a candidate view")
                            v, r = c2(cview, Y, ph, L, j)
                        else:
                            v, r = reference(rview, kind, Y, rel, ph, L, j)
                        vec[j] = store.add(v, r, L, j, run["window"][j])
                    arms.append(P.arm(name, run, vec, identity=(kind == "ID")))
                targets.append({"id": f"{rid}|L{L}", "run": rid, "candidate_set": P.CANDIDATE_SET, "arms": arms})
    return targets
