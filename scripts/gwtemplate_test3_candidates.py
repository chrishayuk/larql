"""GW-TEMPLATE-FACTORIAL test 3: candidate vector builder (ENGINEERING; not run on real data).

It builds the vectors the derived-vector runner injects. Every vector gets a
recipe: canonical JSON naming the kind, target, relation, phrasing, layer,
slot, member countries and formula version. The recipe's sha256 travels with
the vector into the runner's output.

**Information rule (design d2 §4).** A candidate for target Y (relation r,
phrasing p) may read only:
- h(Y, r, p), Y's own original-relation window;
- h(e, ·, p) for e in F∖{Y}, the fit countries other than Y.

Reading any other capture while building a candidate raises
`InformationRuleError`. In particular this blocks h(Y, r', p) for any Y,
and every non-fit country other than Y. References (C0 donor, S) are built
through a separate accessor and are labelled as references, not candidates.

Holdout countries refuse everywhere.

**Kinds:**
- C0: the donor's captured window, a reference;
- S: Y's own r' window, a reference;
- C1: the mean of h(e, r', p) over F∖{Y};
- C1A / C1B: the same over the disjoint halves of F, each without Y;
- C2: h(Y, r, p) + mean over F∖{Y} of h(e, r', p) − mean over F∖{Y} of
  h(e, r, p);
- C1E: the mean of h(e, r, p) over F∖{Y}, injected into Y's r prompt.

There is no C2E: the r→r displacement is zero, so it would be the identity
arm. C3 is deferred by owner decision.

Means are accumulated in float64 in sorted country order and cast to f32
once, so the result is deterministic and can be recomputed independently.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FITVAL_SALT = "gw-template-factorial/test3-fitval/v1|"
HALF_SALT = "gw-template-factorial/test3-fit-halves/v1|"
N_VAL = 20
FORMULA_VERSION = "test3-candidates/v1"
RELATIONS = ("capital", "currency_code_iso4217")


class InformationRuleError(Exception):
    pass


def salted_order(countries, salt: str) -> list[str]:
    return sorted(countries, key=lambda c: hashlib.sha256((salt + c).encode()).hexdigest())


def split(dev: list[str]) -> tuple[list[str], list[str]]:
    """Returns (fit, validation). The first N_VAL in salted order are validation."""
    o = salted_order(dev, FITVAL_SALT)
    return sorted(o[N_VAL:]), sorted(o[:N_VAL])


def halves(fit: list[str]) -> tuple[list[str], list[str]]:
    o = salted_order(fit, HALF_SALT)
    n = len(o) // 2
    return sorted(o[:n]), sorted(o[n:])


def other(rel: str) -> str:
    return RELATIONS[1 - RELATIONS.index(rel)]


def recipe_sha(recipe: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Capture:
    """Read-only window access to a stage 1b capture (or a synthetic one in tests).

    windows[(rel, phrasing, country)] -> float32 array [layers, slots, hidden]
    """

    def __init__(self, windows: dict, holdout: set[str]):
        self._w = windows
        self.holdout = holdout

    @classmethod
    def load(cls, cap_dir: Path, plan: dict) -> "Capture":
        man = json.loads((cap_dir / "manifest.json").read_text())
        w = {}
        for rid, e in man["runs"].items():
            rel, ph, c = rid.split("|")
            a = np.fromfile(cap_dir / e["file"], dtype="<f4").reshape(e["layers"], len(e["positions"]), e["hidden"])
            nwin = len(e["window"])
            w[(rel, ph, c)] = a[:, :nwin, :]
        return cls(w, set(plan["holdout"]))

    def raw(self, rel, ph, c):
        if c in self.holdout:
            raise InformationRuleError(f"holdout country {c}")
        return self._w[(rel, ph, c)]


class Builder:
    def __init__(self, cap: Capture, fit: list[str]):
        self.cap, self.fit = cap, sorted(fit)
        self.reads: list[tuple] = []

    def _cand_read(self, Y, r, ph, rel, c):
        """The only capture access for candidates; it enforces the information rule."""
        if c == Y and rel == r:
            pass
        elif c != Y and c in self.fit:
            pass
        else:
            raise InformationRuleError(f"candidate for {Y} ({r},{ph}) may not read h({c},{rel},{ph})")
        self.reads.append((Y, r, ph, rel, c))
        return self.cap.raw(rel, ph, c)

    def _mean(self, Y, r, ph, rel, members):
        acc = None
        for c in sorted(members):
            a = self._cand_read(Y, r, ph, rel, c).astype(np.float64)
            acc = a if acc is None else acc + a
        return (acc / len(members)).astype("<f4")

    def candidate(self, kind: str, Y: str, r: str, ph: str):
        """Returns (vectors [layers, slots, hidden] f32, recipe dict) for Y's r prompt."""
        rp = other(r)
        loo = [c for c in self.fit if c != Y]
        if kind == "C1":
            v, members = self._mean(Y, r, ph, rp, loo), loo
        elif kind in ("C1A", "C1B"):
            a, b = halves(self.fit)
            members = [c for c in (a if kind == "C1A" else b) if c != Y]
            v = self._mean(Y, r, ph, rp, members)
        elif kind == "C2":
            own = self._cand_read(Y, r, ph, r, Y).astype(np.float64)
            d = self._mean(Y, r, ph, rp, loo).astype(np.float64) - self._mean(Y, r, ph, r, loo).astype(np.float64)
            v, members = (own + d).astype("<f4"), loo
        elif kind == "C1E":
            v, members = self._mean(Y, r, ph, r, loo), loo
        else:
            raise ValueError(f"unknown candidate kind {kind}")
        spec = {"C1": [("member", rp, c) for c in members], "C1A": [("member", rp, c) for c in members],
                "C1B": [("member", rp, c) for c in members], "C1E": [("member", r, c) for c in members],
                "C2": [("own", r, Y)] + [("plus", rp, c) for c in members] + [("minus", r, c) for c in members]}[kind]
        return v, {"kind": kind, "target": Y, "relation": r, "phrasing": ph, "members": members,
                    "formula": FORMULA_VERSION, "inputs": spec}

    def reference(self, kind: str, Y: str, r: str, ph: str, donor: str | None = None):
        """C0 (donor r' window), E0 (donor r window), S (Y's r' window), ID (Y's own r window): references, outside the candidate rule."""
        if kind == "C0":
            return self.cap.raw(other(r), ph, donor), {"kind": "C0", "target": Y, "donor": donor, "relation": r,
                                                       "phrasing": ph, "formula": FORMULA_VERSION,
                                                       "inputs": [("copy", other(r), donor)]}
        if kind == "E0":
            return self.cap.raw(r, ph, donor), {"kind": "E0", "target": Y, "donor": donor, "relation": r,
                                                "phrasing": ph, "formula": FORMULA_VERSION,
                                                "inputs": [("copy", r, donor)]}
        if kind == "S":
            return self.cap.raw(other(r), ph, Y), {"kind": "S", "target": Y, "relation": r, "phrasing": ph,
                                                   "formula": FORMULA_VERSION, "inputs": [("copy", other(r), Y)]}
        if kind == "ID":
            return self.cap.raw(r, ph, Y), {"kind": "ID", "target": Y, "relation": r, "phrasing": ph,
                                            "formula": FORMULA_VERSION, "inputs": [("copy", r, Y)]}
        raise ValueError(kind)


class Store:
    """Accumulates vectors into one f32 file plus an index; deduplicates identical vectors with identical recipes."""

    def __init__(self, hidden: int):
        self.hidden, self.chunks, self.index, self.n = hidden, [], {}, 0

    def add(self, vec: np.ndarray, recipe: dict, layer: int, slot: int, target_position: int,
            cap: "Capture", source_slot: int | None = None) -> str:
        """Adds one vector. The stored recipe names the exact target address and every input vector's hash.

        source_slot: the input slot, when it differs from the target slot (a single-slot word map).
        """
        from gwtemplate_test3_capture_run import vector_sha256
        js = slot if source_slot is None else source_slot
        inputs = [[role, rel, recipe["phrasing"], c, layer, js, vector_sha256(cap.raw(rel, recipe["phrasing"], c)[layer, js])]
                  for role, rel, c in recipe["inputs"]]
        rec = dict(recipe, layer=layer, slot=slot, source_slot=js, target_position=target_position, inputs=inputs)
        rsha = recipe_sha(rec)
        vsha = vector_sha256(vec)
        vid = f"{rsha[7:23]}-{vsha[:16]}"
        if vid not in self.index:
            self.index[vid] = {"offset": self.n * self.hidden, "sha256": vsha, "recipe_sha256": rsha, "recipe": rec}
            self.chunks.append(vec.astype("<f4"))
            self.n += 1
        return vid

    def write(self, path: Path) -> str:
        data = np.concatenate(self.chunks) if self.chunks else np.zeros(0, "<f4")
        path.write_bytes(data.astype("<f4").tobytes())
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
