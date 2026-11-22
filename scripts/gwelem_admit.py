"""GW-ELEMENT-1 confirmation: independent admission, recomputation and the C2 dependency audit.

This module shares no code with the C2 builder (`gwelem_candidates.py`), the plan builder or the
runner. It imports only the frozen design constants (`gwelem_design`). It re-implements:
- `vector_sha256` and the canonical recipe hash;
- the C2 formula;
- every structural rule.

**Pre-run (`admit_plan`)**
- Sealing: a run whose key is in the 35-element confirmation split, or whose tokens contain any
  sealed element's subject tokens (A2 or A3 form), is refused unless the caller declares
  `release=True` (confirmation execution only). The runner enforces the same seal independently.
- Window: the decoded window equals the frozen slot strings.
- Inventory: number targets carry exactly the frozen arm names for their phrasing; symbol targets
  carry exactly the reference arm; no arm duplicates another.
- Arm shape: a single carrier band at L4 (from 5, to 4); the span is exactly the named slots;
  the vectors cover exactly L4 × the span.
- Every vector's recipe:
  - recipe hash;
  - address (layer, slot, target position);
  - target cell (element, relation, phrasing);
  - kind equals the arm kind;
  - input hashes equal the capture manifests;
  - the plan index equals the builder index.
- **C2 information rule:**
  - exactly one `own` input, the target's own number window;
  - `plus` = the fit population's symbol windows and `minus` = the fit population's number
    windows, each exactly the 45-element fit set;
  - members = fit;
  - the target is not in fit;
  - **no input is the target's symbol window**, and no input comes from any non-fit element
    other than the target's own number window.
- S: one copy of the target's own symbol window, on a number target. ID: one copy of the run's own window.

**Recompute (`recompute_all`).** Every vector is rebuilt from the capture files, through a capture
set the caller restricts. For confirmation, C2 is recomputed from the fit capture and the number
capture only, so the target's symbol capture is not even loaded.

**Dependency audit (`dependency_audit`).** For each (phrasing, slot), every C2 vector's plus and
minus inputs are identical across targets: the same 45 elements, the same hashes. The recomputed
displacement is therefore one target-independent vector, and the only target-specific input is the
target's number window.

**Post-run.** The generic band record admission (`gwcompress_band_admit.admit_records`) and the
sub-window record admission (`gwcompress_subspan_admit.admit_subspan_records`) apply unchanged. Both
are domain-free, and both check records against the capture's carrier hashes.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

import gwelem_design as D


def vsha(v) -> str:
    v = np.asarray(v, dtype="<f4")
    return hashlib.sha256(struct.pack("<Q", v.size) + v.tobytes()).hexdigest()


def rsha(recipe: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Caps:
    """Capture manifests and files for an explicit list of capture directories."""

    def __init__(self, dirs):
        self.entries, self.dirs, self._c = {}, {}, {}
        for d in dirs:
            m = json.loads((Path(d) / "manifest.json").read_text())
            for rid, e in m["runs"].items():
                if rid in self.entries:
                    raise ValueError(f"{rid} in two captures")
                self.entries[rid], self.dirs[rid] = e, Path(d)

    def sha(self, rid, layer, slot):
        e = self.entries[rid]
        return e["vector_sha256"][f"{layer}|{e['window'][slot]}"]

    def vec(self, rid, layer, slot) -> np.ndarray:
        if rid not in self._c:
            e = self.entries[rid]
            a = np.fromfile(self.dirs[rid] / e["file"], dtype="<f4").reshape(e["layers"], len(e["positions"]), e["hidden"])
            self._c[rid] = a
        return self._c[rid][layer, slot]


# ── sealing ──────────────────────────────────────────────────────────────────

SEAL_FORMS = ("mid_lower", "mid_cap", "start_cap", "start_lower")


def sealed_subjects(tok) -> dict:
    """{(Z, form): subject token ids} for every element in the confirmation split, re-rendered here.

    Four surface forms per element: lower-case and capitalised, mid-sentence (after "For ") and
    sentence-initial (after BOS). The A2 prompts use mid_lower and the A3 prompts use start_cap; the
    other two forms close the capitalisation variants (reviewer finding C2).
    """
    import gwelem_feasibility as fz  # the registered element list (artifact 2748's script)
    names = {z: n for z, _, n in fz.ELEMENTS}
    out = {}
    for z in D.CONFIRMATION_SPLIT:
        n = names[z]
        N = n[0].upper() + n[1:]
        for form, text, subj in (("mid_lower", f"For {n}, the atomic number is", n),
                                 ("mid_cap", f"For {N}, the atomic number is", N),
                                 ("start_cap", f"{N}'s atomic number is", N),
                                 ("start_lower", f"{n}'s atomic number is", n)):
            enc = tok.encode(text, add_special_tokens=True)
            s0 = text.index(subj)
            out[(z, form)] = [i for i, (a, b) in zip(enc.ids, enc.offsets) if a < s0 + len(subj) and b > s0 and i != 2]
    return out


def runs_digest(runs: dict) -> str:
    """The digest an authorisation binds to: sha256 over sorted run ids of 'id|country|t1,t2,...\n'."""
    h = hashlib.sha256()
    for rid in sorted(runs):
        r = runs[rid]
        h.update(f"{rid}|{r['country']}|{','.join(str(int(t)) for t in r['tokens'])}\n".encode())
    return "sha256:" + h.hexdigest()


def _contains(t, s):
    n = len(s)
    return any(t[i:i + n] == s for i in range(len(t) - n + 1))


def seal_check(plan: dict, subjects: dict, release: bool = False) -> list[str]:
    if release:
        return []
    keys = {D.key(z) for z in D.CONFIRMATION_SPLIT}
    f = []
    for rid, r in plan["runs"].items():
        if r["country"] in keys or any(p in keys for p in rid.split("|")):
            f.append(f"{rid}: sealed confirmation key")
        for (z, ph), s in subjects.items():
            if _contains(list(r["tokens"]), list(s)):
                f.append(f"{rid}: contains the subject tokens of sealed element {z} ({ph})")
    return f


# ── pre-run admission ────────────────────────────────────────────────────────

def _expect_inputs(kind, Y, rel, ph, layer, slot, fit):
    if kind == "C2":
        return ([("own", D.SOURCE, ph, D.key(Y))] + [("plus", D.INJECTED, ph, D.key(z)) for z in fit]
                + [("minus", D.SOURCE, ph, D.key(z)) for z in fit])
    if kind == "S":
        return [("copy", D.INJECTED, ph, D.key(Y))]
    if kind == "ID":
        return [("copy", rel, ph, D.key(Y))]
    return None


def check_vector(rec: dict, kind: str, Y: int, rel: str, ph: str, layer: int, slot: int, pos: int,
                 caps: Caps, fit: tuple) -> list[str]:
    f = []
    if (rec.get("layer"), rec.get("slot"), rec.get("target_position")) != (layer, slot, pos):
        f.append("address")
    if (rec.get("target"), rec.get("relation"), rec.get("phrasing"), rec.get("kind")) != (D.key(Y), rel, ph, kind):
        f.append("target cell or kind")
    if rec.get("formula") != D.FORMULA:
        f.append("formula version")
    inputs = rec.get("inputs", [])
    want = _expect_inputs(kind, Y, rel, ph, layer, slot, fit)
    if want is None:
        return f + [f"unknown kind {kind}"]
    got = [tuple(i[:4]) for i in inputs]
    if got != want:
        f.append("inputs differ from the frozen recipe (roles, relations, phrasing or membership)")
    for i in inputs:
        role, irel, iph, c, il, js, h = i
        if il != layer or js != slot:
            f.append(f"input address {c}")
        try:
            if caps.sha(f"{irel}|{iph}|{c}", il, js) != h:
                f.append(f"input hash {irel}|{iph}|{c}")
        except KeyError:
            f.append(f"input not captured {irel}|{iph}|{c}")
    if kind == "C2":
        if Y in fit:
            f.append("information rule: target in the fit population")
        if any(i[1] == D.INJECTED and i[3] == D.key(Y) for i in inputs):
            f.append("information rule: the target's own symbol state is an input")
        if any(i[0] != "own" and int(i[3]) not in fit for i in inputs):
            f.append("information rule: a non-fit aggregate member")
        if rec.get("members") != [D.key(z) for z in fit]:
            f.append("members differ from the fit population")
        if rel != D.SOURCE:
            f.append("C2 on a non-number target")
    if kind == "S" and rel != D.SOURCE:
        f.append("S on a non-number target")
    return f


def admit_plan(plan: dict, index: dict, caps: Caps, decode, subjects: dict, cells: dict, fit=D.FIT,
               release: bool = False, arms=D.ARMS, reference_arms=D.REFERENCE_ARMS) -> list[str]:
    """cells: {run id: reference token ids} (the registered baseline prompts); every run must equal its cell."""
    fit = tuple(sorted(fit))
    fails = seal_check(plan, subjects, release)
    for rid, r in plan["runs"].items():
        if list(r["tokens"]) != list(cells.get(rid, [])) or r["country"] != rid.split("|")[-1]:
            fails.append(f"{rid}: run tokens or key differ from the registered prompt for this id")
    for t in plan["targets"]:
        rid = t["run"]
        rel, ph, ykey = rid.split("|")
        Y = int(ykey)
        run = plan["runs"][rid]
        w = run["window"]
        dec = [decode([run["tokens"][p]]) for p in w]
        if dec != D.DECODED_WINDOWS.get((rel, ph)):
            fails.append(f"{t['id']}: decoded window {dec}")
            continue
        if w[-1] + 1 != len(run["tokens"]) - 1:
            fails.append(f"{t['id']}: window does not end before the readout")
        names = [a["name"] for a in t["arms"]]
        ref = reference_arms[ph] if isinstance(reference_arms, dict) else reference_arms
        inv = list(arms[ph]) if rel == D.SOURCE else list(ref)
        if sorted(names) != sorted(inv) or len(set(names)) != len(names):
            fails.append(f"{t['id']}: arm inventory {names} != {inv}")
        seen = {}
        for a in t["arms"]:
            nm = f"{t['id']} {a['name']}"
            kind, span, lay = a["name"].split("-")
            slots = D.SPANS.get(span)
            if slots is None or lay != f"L{D.LAYER}":
                fails.append(f"{nm}: name")
                continue
            if (a["from_layer"], a["to_layer"]) != (D.LAYER + 1, D.LAYER):
                fails.append(f"{nm}: not a single carrier at L{D.LAYER}")
            span_pos = [w[j] for j in slots]
            if a["target_span"] != span_pos:
                fails.append(f"{nm}: span {a['target_span']} != {span_pos}")
            if set(a["vectors"]) != {f"{D.LAYER}|{p}" for p in span_pos}:
                fails.append(f"{nm}: coverage")
            if bool(a.get("identity")) != (kind == "ID"):
                fails.append(f"{nm}: identity flag")
            k = (tuple(a["target_span"]), tuple(sorted(a["vectors"].items())))
            if k in seen:
                fails.append(f"{nm}: duplicate of {seen[k]}")
            seen[k] = a["name"]
            if kind != "ID":
                # r2: an intended non-identity arm must differ bit-exactly from the matched natural carrier
                # at >= 1 designated position (identity arms are exempt).
                differs = []
                for j in D.EFFECTIVE_SLOTS:
                    if j in slots:
                        vid = a["vectors"].get(f"{D.LAYER}|{w[j]}")
                        nat = caps.sha(rid, D.LAYER, j) if rid in caps.entries else None
                        pe = plan["vector_store"]["index"].get(vid) or {}
                        differs.append(nat is not None and pe.get("sha256") not in (None, nat))
                if not any(differs):
                    fails.append(f"{nm}: identical to the natural carrier at every designated position")
            for vk, vid in a["vectors"].items():
                l, p = map(int, vk.split("|"))
                e = index.get(vid)
                pe = plan["vector_store"]["index"].get(vid)
                if e is None or pe is None:
                    fails.append(f"{nm} {vk}: unknown vector")
                    continue
                if rsha(e["recipe"]) != e["recipe_sha256"] or pe["recipe_sha256"] != e["recipe_sha256"] \
                        or pe["sha256"] != e["sha256"]:
                    fails.append(f"{nm} {vk}: recipe or index hash")
                if p not in w:
                    fails.append(f"{nm} {vk}: outside the window")
                    continue
                for x in check_vector(e["recipe"], kind, Y, rel, ph, l, w.index(p), p, caps, fit):
                    fails.append(f"{nm} {vk}: {x}")
    return fails


# ── recompute and dependency audit ───────────────────────────────────────────

def recompute(rec: dict, caps: Caps) -> np.ndarray:
    l, j, ph = rec["layer"], rec["slot"], rec["phrasing"]
    get = lambda rel, c: caps.vec(f"{rel}|{ph}|{c}", l, j)  # noqa: E731
    if rec["kind"] in ("S", "ID"):
        (_, irel, _, c, *_), = rec["inputs"]
        return get(irel, c).copy()
    if rec["kind"] != "C2":
        raise ValueError(rec["kind"])

    def mean(role):
        xs = sorted(((int(c), irel) for r, irel, _, c, *_ in rec["inputs"] if r == role))
        acc = np.zeros(get(xs[0][1], xs[0][0]).shape, dtype=np.float64)
        for c, irel in xs:
            acc = acc + get(irel, c).astype(np.float64)
        return (acc / len(xs)).astype("<f4")

    (_, irel, _, c, *_), = [i for i in rec["inputs"] if i[0] == "own"]
    own = get(irel, c).astype(np.float64)
    return (own + (mean("plus").astype(np.float64) - mean("minus").astype(np.float64))).astype("<f4")


def recompute_all(plan: dict, index: dict, caps_for_kind: dict) -> dict:
    """caps_for_kind: kind -> Caps; C2 must get a Caps without any target symbol capture."""
    data = np.fromfile(plan["vector_store"]["file"], dtype="<f4")
    h = plan["vector_store"]["hidden"]
    res = {"vectors": 0, "failures": []}
    for vid, pe in plan["vector_store"]["index"].items():
        e = index[vid]
        res["vectors"] += 1
        v = recompute(e["recipe"], caps_for_kind[e["recipe"]["kind"]])
        stored = data[pe["offset"]:pe["offset"] + h]
        if vsha(v) != e["sha256"] or vsha(stored) != e["sha256"]:
            res["failures"].append(vid)
    return res


def dependency_audit(index: dict, c2_caps: Caps, fit=D.FIT) -> dict:
    """Every C2's aggregate inputs are identical across targets; the target's symbol run is absent from c2_caps."""
    groups, fails, targets = {}, [], set()
    for vid, e in index.items():
        r = e["recipe"]
        if r["kind"] != "C2":
            continue
        targets.add(r["target"])
        agg = tuple(tuple(i) for i in r["inputs"] if i[0] != "own")
        groups.setdefault((r["phrasing"], r["slot"], r["layer"]), set()).add(agg)
        own = [i for i in r["inputs"] if i[0] == "own"]
        if len(own) != 1 or own[0][1] != D.SOURCE or own[0][3] != r["target"]:
            fails.append(f"{vid}: own input")
        if f"{D.INJECTED}|{r['phrasing']}|{r['target']}" in c2_caps.entries:
            fails.append(f"{vid}: the target's symbol capture is loadable by the C2 recompute")
    for k, s in groups.items():
        if len(s) != 1:
            fails.append(f"{k}: aggregate inputs differ across targets")
        else:
            (agg,), = [tuple(s)]
            if sorted({int(i[3]) for i in agg}) != sorted(fit):
                fails.append(f"{k}: aggregate membership is not the fit population")
    return {"c2_targets": len(targets), "groups": len(groups), "failures": fails}


def audit_c2(plan: dict, index: dict, c2_caps: Caps, ref_caps: Caps, fit=D.FIT) -> dict:
    """Recompute and dependency audit through ONE restricted Caps for C2 (reviewer finding I5).

    c2_caps must hold no symbol run of any C2 target. It is used both to recompute every C2 and to
    audit dependency. S and ID references use ref_caps.
    """
    targets = {e["recipe"]["target"] for e in index.values() if e["recipe"]["kind"] == "C2"}
    leaks = sorted(r for r in c2_caps.entries if r.startswith(D.INJECTED + "|") and r.split("|")[-1] in targets)
    rec = recompute_all(plan, index, {"C2": c2_caps, "S": ref_caps, "ID": ref_caps})
    dep = dependency_audit(index, c2_caps, fit=fit)
    return {"recompute": rec, "dependency": dep, "c2_caps_symbol_leaks": leaks,
            "pass": not rec["failures"] and not dep["failures"] and not leaks}
