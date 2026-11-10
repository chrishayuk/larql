"""PREC-DIAG-1 analysis: measurements, readings R1–R6 and the outcome table.

The rules are rev 3 "Measurements", "Readings" and "Outcome table".
Thresholds are module constants; `FREEZE.json` records them (`thresholds()`).

Rules this module follows:
- Every function computes from arrays and returns plain JSON-able values,
  so the readings are unit-tested on synthetic logits with known
  discrepancies.
- `analyse()` consumes only admitted runs (`evidence.admit`) and is itself a
  run under the same lifecycle. Its record lists the `run.result.json`
  sha256 of every input it consumed.
- A reading over an unadmitted input is `not_admitted` and carries no
  values.
- Logits are streamed in blocks of positions; nothing holds two full
  `[1040, 262144]` f64 copies at once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import equivalence_metrics as em  # noqa: E402
from prec_diag import evidence as ev  # noqa: E402

# ------------------------------------------------------------- thresholds

EPSILON = 1e-5                 # absolute floor for every ratio's denominator
LATE_FIRST, LATE_LAST = 514, 1039
GATE = 1e-3                    # d above this is "material" (= the strict-f32 gate's value)
R1_COMPARABLE_MEDIAN, R1_COMPARABLE_P99, R1_LESS_ACCURATE_MEDIAN, R1_MATERIAL_COUNT = 2.0, 4.0, 4.0, 3
R2_SUPPORTED_RATIO, R2_NOT_SUPPORTED_RATIO, R2_NOT_SUPPORTED_COUNT = 0.5, 0.8, 3
R3_CONTRIBUTES, R3_BAND = 0.5, (0.8, 1.25)
R4_MATERIAL_REL_L2, R4_AMPLIFICATION_FLOOR, R4_SPIKE_RATIO = 1e-3, 1e-7, 3.0
R5_TVD_CHANGING, R5_TVD_TAIL_ONLY = 1e-3, 1e-4
MASS_BEARING_LOGPROB = em.MASS_BEARING_LOGPROB
TIE_TOLERANCE = em.TIE_TOLERANCE      # registered top-5 rule, per family
FAMILY = {"C": "reference", "C32a": "reference", "D": "production", "D32a": "production",
          "A32": "reference", "A32s": "reference", "A32r": "reference", "A64": "reference"}
BLOCK = 16                    # positions per streamed block


def thresholds() -> dict:
    """Every threshold the readings use, for FREEZE.json."""
    return {"epsilon": EPSILON, "late": [LATE_FIRST, LATE_LAST], "material_d": GATE,
            "R1": {"comparable_median": R1_COMPARABLE_MEDIAN, "comparable_p99": R1_COMPARABLE_P99,
                   "less_accurate_median": R1_LESS_ACCURATE_MEDIAN, "material_count": R1_MATERIAL_COUNT},
            "R2": {"supported_ratio": R2_SUPPORTED_RATIO, "not_supported_ratio": R2_NOT_SUPPORTED_RATIO,
                   "not_supported_count": R2_NOT_SUPPORTED_COUNT},
            "R3": {"contributes": R3_CONTRIBUTES, "not_material_band": list(R3_BAND)},
            "R4": {"material_rel_l2": R4_MATERIAL_REL_L2, "amplification_floor": R4_AMPLIFICATION_FLOOR,
                   "spike_ratio": R4_SPIKE_RATIO},
            "R5": {"tvd_behaviour_changing": R5_TVD_CHANGING, "tvd_tail_only": R5_TVD_TAIL_ONLY,
                   "tie_tolerance": TIE_TOLERANCE},
            "mass_bearing_logprob": MASS_BEARING_LOGPROB}


# ----------------------------------------------------------- measurements

def position_metrics(ref: np.ndarray, x: np.ndarray, family: str, block: int = BLOCK) -> dict:
    """Per-position measurements of `x` against `ref` (`[positions, vocab]`).

    Returns arrays (one entry per position) keyed by measurement name."""
    if ref.shape != x.shape:
        raise ValueError(f"shapes differ: {ref.shape} vs {x.shape}")
    n = ref.shape[0]
    out = {k: np.empty(n) for k in ("d", "d_ref_logprob", "mass_d", "max_dlogit", "max_dcentred", "tvd",
                                    "kl", "top1_prob_ref", "top1_prob_x", "entropy_ref", "entropy_x")}
    out |= {k: np.empty(n, dtype=np.int64) for k in ("d_token", "mass_count", "top1_ref", "top1_x")}
    out["top5"] = np.empty(n, dtype=object)
    tol = TIE_TOLERANCE[family]
    for s in range(0, n, block):
        r = np.asarray(ref[s:s + block], dtype=np.float64)
        a = np.asarray(x[s:s + block], dtype=np.float64)
        lr, la = em.log_softmax64(r), em.log_softmax64(a)
        diff = np.abs(lr - la)
        tok = diff.argmax(axis=-1)
        rows = np.arange(r.shape[0])
        pr, pa = np.exp(lr), np.exp(la)
        mass = lr >= MASS_BEARING_LOGPROB
        sl = slice(s, s + r.shape[0])
        out["d"][sl] = diff[rows, tok]
        out["d_token"][sl] = tok
        out["d_ref_logprob"][sl] = lr[rows, tok]
        out["mass_d"][sl] = np.where(mass, diff, 0.0).max(axis=-1)
        out["mass_count"][sl] = mass.sum(axis=-1)
        out["max_dlogit"][sl] = np.abs(r - a).max(axis=-1)
        out["max_dcentred"][sl] = np.abs((r - r.mean(-1, keepdims=True)) - (a - a.mean(-1, keepdims=True))).max(-1)
        out["tvd"][sl] = 0.5 * np.abs(pr - pa).sum(axis=-1)
        out["kl"][sl] = np.where(pr > 0, pr * (lr - la), 0.0).sum(axis=-1)
        out["top1_ref"][sl] = lr.argmax(-1)
        out["top1_x"][sl] = la.argmax(-1)
        out["top1_prob_ref"][sl] = pr.max(-1)
        out["top1_prob_x"][sl] = pa[rows, lr.argmax(-1)]
        out["entropy_ref"][sl] = -(pr * lr).sum(-1)
        out["entropy_x"][sl] = -(pa * la).sum(-1)
        for i in range(r.shape[0]):
            out["top5"][s + i] = em.order_verdict(lr[i], la[i], tol)[0]
    return out


def late(values: np.ndarray, positions=None) -> np.ndarray:
    pos = np.arange(len(values)) if positions is None else np.asarray(positions)
    return np.asarray(values)[(pos >= LATE_FIRST) & (pos <= LATE_LAST)]


def ratio(num: np.ndarray, den: np.ndarray) -> tuple[np.ndarray, int]:
    """num / max(den, ε), and how many positions hit the floor."""
    den = np.asarray(den, dtype=np.float64)
    flagged = den < EPSILON
    return np.asarray(num, dtype=np.float64) / np.where(flagged, EPSILON, den), int(flagged.sum())


def summary(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
            "max": float(v.max()), "argmax": int(v.argmax())}


# ---------------------------------------------------------------- readings

def r1(d_x_a64: dict[str, np.ndarray], d_a32_a64: dict[str, np.ndarray], a64_valid: bool) -> dict:
    """Accuracy for one backend X. Inputs keyed by input id."""
    if not a64_valid:
        return {"verdict": "invalid", "reason": "A64 audit failed or A64 not admitted"}
    per_input, material, verdict_inputs = {}, [], []
    for name, dx in d_x_a64.items():
        da = d_a32_a64[name]
        r, flagged = ratio(dx, da)
        mat = np.flatnonzero((dx > GATE) & (da <= GATE))
        material += [{"input": name, "position": int(p), "d_x": float(dx[p]), "d_a32": float(da[p])} for p in mat]
        entry = {"ratio": summary(r), "flagged_floor": flagged, "d_x_a64": summary(dx), "d_a32_a64": summary(da)}
        per_input[name] = entry
        verdict_inputs.append(entry["ratio"])
    if any(s["median"] > R1_LESS_ACCURATE_MEDIAN for s in verdict_inputs) or len(material) >= R1_MATERIAL_COUNT:
        verdict = "materially less accurate"
    elif (all(s["median"] <= R1_COMPARABLE_MEDIAN and s["p99"] <= R1_COMPARABLE_P99 for s in verdict_inputs)
          and len(material) < R1_MATERIAL_COUNT):
        verdict = "comparable"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "inputs": per_input, "material_discrepancies": material}


def r2(d: dict[str, dict[str, np.ndarray]], rep: str = "I_rep", nonrep=("I_nr1", "I_nr2")) -> dict:
    """Repetition amplification. `d[backend][input]` = per-position d(X, ref)."""
    stats = {b: {i: {"late": summary(late(v)), "over_gate": int((v > GATE).sum())} for i, v in by.items()}
             for b, by in d.items()}
    supported, not_supported = True, False
    for b in d:
        rep_p99 = stats[b][rep]["late"]["p99"]
        for i in nonrep:
            s = stats[b][i]
            if s["over_gate"] > 0 or not s["late"]["p99"] <= R2_SUPPORTED_RATIO * rep_p99:
                supported = False
            if s["over_gate"] >= R2_NOT_SUPPORTED_COUNT or s["late"]["p99"] >= R2_NOT_SUPPORTED_RATIO * rep_p99:
                not_supported = True
    verdict = "not supported" if not_supported else "supported" if supported else "inconclusive"
    return {"verdict": verdict, "stats": stats}


def _median_late(v):
    lv = late(v)
    return float(np.median(lv)) if lv.size else float("nan")


def r3(d_x_a32, d_x_a32r, d_x32a_a32, rep: str = "I_rep") -> dict:
    """RoPE construction, reciprocal. Each argument: `[backend][input]` arrays.
    `d_x32a_a32` is keyed by the base backend (C for C32a, D for D32a)."""
    rho = {}
    for b in d_x_a32:
        rho[b] = {}
        for i in d_x_a32[b]:
            base = max(_median_late(d_x_a32[b][i]), EPSILON)
            rho[b][i] = {"rho_H": _median_late(d_x_a32r[b][i]) / base,
                         "rho_L": _median_late(d_x32a_a32[b][i]) / base,
                         "median_late_base": _median_late(d_x_a32[b][i]), "floored": base == EPSILON}
    on_rep = [rho[b][rep] for b in rho]
    lo, hi = R3_BAND
    if all(r["rho_H"] <= R3_CONTRIBUTES and r["rho_L"] <= R3_CONTRIBUTES for r in on_rep):
        verdict = "contributes"
    elif any((r["rho_H"] <= R3_CONTRIBUTES) != (r["rho_L"] <= R3_CONTRIBUTES) for r in on_rep):
        verdict = "one-sided"
    elif all(lo <= r[k] <= hi for b in rho for r in rho[b].values() for k in ("rho_H", "rho_L")):
        verdict = "not material"
    else:
        verdict = "partial"
    return {"verdict": verdict, "rho": rho}


#: Forward order of E0a's sites within a layer, and the site whose error
#: each one's amplification is measured against (its immediate captured
#: predecessor; `prev_residual` is the previous layer's residual, or the
#: embedding at layer 0). Recorded for review: rev 2 names the ratio, not
#: the pairing.
FORWARD_ORDER = ("attention_input", "query_unpositioned", "key_unpositioned", "query_positioned",
                 "key_positioned", "attention_output", "attention_delta", "ffn_input", "ffn_output",
                 "ffn_delta", "residual")
PREDECESSOR = {"attention_input": "prev_residual", "query_unpositioned": "attention_input",
               "key_unpositioned": "attention_input", "query_positioned": "query_unpositioned",
               "key_positioned": "key_unpositioned", "attention_output": "attention_input",
               "attention_delta": "attention_output", "ffn_input": "attention_delta",
               "ffn_output": "ffn_input", "ffn_delta": "ffn_output", "residual": "ffn_delta",
               "head_input": "prev_residual"}


def predecessor_key(key: tuple, num_layers: int):
    """The captured site a site's amplification is measured against."""
    site, layer = key
    pred = PREDECESSOR.get(site)
    if pred is None:
        return None
    if pred != "prev_residual":
        return (pred, layer)
    if site == "head_input":
        return ("residual", num_layers - 1)
    return ("embedding", None) if layer == 0 else ("residual", layer - 1)


def site_sequence(num_layers: int) -> list[tuple[str, int | None]]:
    seq = [("embedding", None)]
    for L in range(num_layers):
        seq += [(s, L) for s in FORWARD_ORDER]
    return seq + [("head_input", None)]


def row_rel_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-30)


def r4(x_planes: dict, ref_planes: dict, positions: list[int], num_layers: int, f32_pair: bool,
       transformed_sites: frozenset = frozenset(), position_sets: dict | None = None,
       boundary: list[int] | None = None, matched: list[int] | None = None) -> dict:
    """Where the disagreement starts and grows, for one arm pair.

    `*_planes[(site, layer)]` = `[len(positions), width]` arrays."""
    seq = site_sequence(num_layers)
    pos = np.asarray(positions)
    sets = {"all": list(range(len(positions)))}
    for name, members in (position_sets or {}).items():
        sets[name] = [i for i, p in enumerate(positions) if p in set(members)]
    per_site = {}
    for key in seq:
        x, ref = x_planes[key], ref_planes[key]
        per_site[key] = {"rel": row_rel_l2(x, ref), "max_abs": np.abs(np.asarray(x, np.float64) - ref).max(-1),
                         "identical": (np.asarray(x).tobytes() == np.asarray(ref).tobytes())
                         if f32_pair else None}
    out = {"per_set": {}, "transformed_sites": sorted(transformed_sites)}
    for set_name, idx in sets.items():
        if not idx:
            continue
        rels = {k: float(np.linalg.norm(np.asarray(x_planes[k], np.float64)[idx] - np.asarray(ref_planes[k],
                                                                                                np.float64)[idx])
                         / max(np.linalg.norm(np.asarray(ref_planes[k], np.float64)[idx]), 1e-30)) for k in seq}
        first_diff = None
        if f32_pair:
            for k in seq:
                if k[0] in transformed_sites:
                    first_diff = {"undefined_at": list(k), "reason": "recorded transformation"}
                    break
                if np.asarray(x_planes[k])[idx].tobytes() != np.asarray(ref_planes[k])[idx].tobytes():
                    first_diff = list(k)
                    break
        material = next((list(k) for k in seq if rels[k] > R4_MATERIAL_REL_L2), "none")
        resolvable, floored = [], []
        for k in seq:
            pk = predecessor_key(k, num_layers)
            if pk is None:
                continue
            r_in, r_out = rels[pk], rels[k]
            cand = {"site": list(k), "input_site": list(pk), "input_rel_l2": r_in, "output_rel_l2": r_out}
            if r_in < R4_AMPLIFICATION_FLOOR:
                # No finite value is invented: the ratio is undefined, and the
                # raw errors on both sides stay on the record.
                floored.append({**cand, "ratio": None,
                                "ratio_status": f"undefined: input relative L2 below {R4_AMPLIFICATION_FLOOR}"})
            else:
                resolvable.append({**cand, "ratio": r_out / r_in, "ratio_status": "defined"})
        largest = max(resolvable, key=lambda c: c["ratio"]) if resolvable else None
        largest_floored = max(floored, key=lambda c: c["output_rel_l2"]) if floored else None
        out["per_set"][set_name] = {"first_difference": first_diff if f32_pair else "undefined (not an f32 pair)",
                                    "first_material_divergence": material,
                                    "largest_amplification": largest,
                                    "largest_output_among_floored_inputs": largest_floored,
                                    "growth": [{"site": list(k), "rel_l2": rels[k]} for k in seq]}
    if boundary is not None and matched is not None:
        bi = [i for i, p in enumerate(pos) if p in set(boundary)]
        mi = [i for i, p in enumerate(pos) if p in set(matched)]
        spike = "amplified at the head"
        detail = []
        for k in seq:
            mb, mm = float(np.median(per_site[k]["rel"][bi])), float(np.median(per_site[k]["rel"][mi]))
            detail.append({"site": list(k), "median_boundary": mb, "median_matched": mm})
            if mb > R4_SPIKE_RATIO * mm and spike == "amplified at the head":
                spike = list(k)
        out["spike_localisation"] = {"site": spike, "detail": detail}
    return out


def r5(m: dict) -> dict:
    """Behavioural stability from one `position_metrics` result."""
    top1_mismatch = m["top1_ref"] != m["top1_x"]
    order_change = top1_mismatch | (m["top5"] != em.PASS)
    changing = top1_mismatch | (m["top5"] == em.FAIL) | (m["tvd"] >= R5_TVD_CHANGING)
    tail = (m["d"] > GATE) & (m["tvd"] < R5_TVD_TAIL_ONLY) & ~order_change
    dprob = np.abs(m["top1_prob_ref"] - m["top1_prob_x"])
    return {"top1_mismatches": int(top1_mismatch.sum()),
            "top5": {v: int((m["top5"] == v).sum()) for v in (em.PASS, em.TIE_SWAP, em.FAIL)},
            "max_tvd": float(m["tvd"].max()), "max_tvd_position": int(m["tvd"].argmax()),
            "max_dtop1_prob": float(dprob.max()), "behaviour_changing_positions": int(changing.sum()),
            "behaviour_changing": bool(changing.any()), "tail_only_positions": int(tail.sum())}


def r6(cached: np.ndarray, batched: np.ndarray, a32: np.ndarray, family: str) -> dict:
    """Cached vs batched prefill within LARQL (prospective; not judged)."""
    identical = np.asarray(cached).tobytes() == np.asarray(batched).tobytes()
    out = {"E1_identical": bool(identical)}
    if not identical:
        out["d_cached_batched"] = summary(position_metrics(cached, batched, family)["d"])
        out["d_batched_a32"] = summary(position_metrics(a32, batched, family)["d"])
        out["d_cached_a32"] = summary(position_metrics(a32, cached, family)["d"])
    return out


#: R1's verdicts that qualify an outcome row (anything else leaves it unqualified).
R1_VERDICTS = ("comparable", "inconclusive", "materially less accurate")

OUTCOMES = {("supported", "not material"): "Repetition-driven amplification of f32 implementation differences",
            ("not supported", "contributes"): "The RoPE construction is a contributing cause independent of input",
            ("supported", "contributes"): "Interaction: RoPE-construction differences amplified by repetition",
            ("not supported", "not material"): "Neither; R4's localised site is the next suspect"}


def outcome(r1_by_backend: dict, r2_verdict: str, r3_verdict: str) -> dict:
    """The fixed outcome table, qualified per backend by R1."""
    if r3_verdict == "one-sided":
        row = "RoPE evidence is one-sided and is reported as such; no causal claim"
    elif r2_verdict == "inconclusive" or r3_verdict == "partial":
        row = "Reported as such; no conclusion forced"
    else:
        row = OUTCOMES.get((r2_verdict, r3_verdict), "Reported as such; no conclusion forced")
    per_backend = {}
    for b, r in r1_by_backend.items():
        verdict = r.get("verdict")
        if verdict == "materially less accurate":
            per_backend[b] = {"conclusion": "possible LARQL computation error", "accuracy": verdict}
        elif verdict in R1_VERDICTS:
            per_backend[b] = {"conclusion": row, "accuracy": verdict}
        else:
            # No R1 verdict: the row stands as an agreement-only conclusion.
            # An unknown accuracy is never read as "comparable".
            per_backend[b] = {"conclusion": row, "accuracy": f"unqualified (R1 {verdict})",
                              "scope": "agreement only"}
    return {"row": row, "R2": r2_verdict, "R3": r3_verdict, "per_backend": per_backend,
            "R1": {b: r.get("verdict") for b, r in r1_by_backend.items()},
            "accuracy_qualified": all(v.get("scope") != "agreement only" for v in per_backend.values())}


# ------------------------------------------------------- admitted analysis

def script_digests() -> dict:
    here = Path(__file__).resolve().parent
    return {"analysis.py": ev.sha256_file(here / "analysis.py"), "evidence.py": ev.sha256_file(here / "evidence.py"),
            "equivalence_metrics.py": ev.sha256_file(here.parent / "equivalence_metrics.py")}


def line_ref(line: dict) -> dict:
    """A results-log line, named by its sequence number and digest, as a
    derived reading cites it."""
    return {"reading": line["reading"], "seq": line["seq"], "status": line["status"],
            "line_sha256": ev.sha256_bytes(ev.canonical_json(line))}


def load_logits(run_dir: Path, vocab: int) -> np.ndarray:
    for name, dt in (("logits.f32", "<f4"), ("logits.f64", "<f8")):
        p = Path(run_dir) / name
        if p.is_file():
            return np.memmap(p, dtype=dt, mode="r").reshape(-1, vocab)
    raise FileNotFoundError(f"{run_dir}: no logits")


def logit_readings(runs: dict[tuple[str, str], Path], expected: dict[tuple[str, str], dict], vocab: int,
                   out_dir: Path, log: ev.ResultsLog, inputs=("I_rep", "I_nr1", "I_nr2")) -> dict:
    """R1, R2, R3, R5 and the outcome, from admitted logit runs.

    `runs[(arm, input)]` is a run directory; `expected[(arm, input)]` the
    identity the freeze manifest holds for it. Writes per-position metric
    arrays into `out_dir` and one results-log line per reading."""
    adm = {k: ev.admit(d, expected.get(k), None) for k, d in runs.items()}
    # R1 consumes the A64 runs, and A64 reaches `complete` only through a
    # passed audit; an invalid A64 therefore makes every accuracy reading
    # `not_admitted` through admission itself.
    cache: dict = {}

    def d(x: str, ref: str, i: str):
        key = (x, ref, i)
        if key not in cache:
            m = position_metrics(load_logits(runs[(ref, i)], vocab), load_logits(runs[(x, i)], vocab), FAMILY[x])
            np.savez(out_dir / f"metrics.{x}_vs_{ref}.{i}.npz", **{k: v for k, v in m.items() if k != "top5"},
                     top5=m["top5"].astype(str))
            cache[key] = m
        return cache[key]

    def admitted(*keys):
        return [adm[k] for k in keys if k in adm] + [ev.Admission(Path(f"<missing {k}>"), False, ["not run"])
                                                    for k in keys if k not in adm]

    def emit(name, keys, compute, **fields):
        consumed = admitted(*keys)
        if all(a.admitted for a in consumed):
            entry = ev.reading(name, ev.READING_COMPLETE, consumed, compute(), **fields)
        else:
            entry = ev.reading(name, ev.NOT_ADMITTED, consumed, **fields)
        return log.append(entry)

    results = {}
    backends = ("C", "D")
    results["R1"] = {}
    for b in backends:
        keys = [(b, i) for i in inputs] + [("A32", i) for i in inputs] + [("A64", i) for i in inputs]
        results["R1"][b] = emit(f"R1 accuracy {b}", keys, lambda b=b: r1(
            {i: d(b, "A64", i)["d"] for i in inputs}, {i: d("A32", "A64", i)["d"] for i in inputs}, True),
            question="accuracy")
    # HF's own yardstick r_s (A32s), reported not judged: a reading of its
    # own, so an A32s problem never reaches R1 and never invalidates it.
    keys_s = [("A32s", i) for i in inputs] + [("A32", i) for i in inputs] + [("A64", i) for i in inputs]
    results["R1 yardstick"] = emit("R1 yardstick A32s", keys_s, lambda: {
        i: {"r_s": summary(ratio(d("A32s", "A64", i)["d"], d("A32", "A64", i)["d"])[0]),
            "flagged_floor": ratio(d("A32s", "A64", i)["d"], d("A32", "A64", i)["d"])[1]} for i in inputs},
        question="accuracy (reported, not judged)")
    keys = [(b, i) for b in backends for i in inputs] + [("A32", i) for i in inputs]
    results["R2"] = emit("R2 repetition amplification", keys,
                         lambda: r2({b: {i: d(b, "A32", i)["d"] for i in inputs} for b in backends}),
                         question="agreement")
    results["R2 accuracy"] = emit("R2 repetition amplification vs A64", keys + [("A64", i) for i in inputs],
                                  lambda: r2({b: {i: d(b, "A64", i)["d"] for i in inputs} for b in backends}),
                                  question="accuracy")
    keys3 = keys + [("A32r", i) for i in inputs] + [(f"{b}32a", i) for b in backends for i in inputs]
    results["R3"] = emit("R3 RoPE construction", keys3, lambda: r3(
        {b: {i: d(b, "A32", i)["d"] for i in inputs} for b in backends},
        {b: {i: d(b, "A32r", i)["d"] for i in inputs} for b in backends},
        {b: {i: d(f"{b}32a", "A32", i)["d"] for i in inputs} for b in backends}), question="agreement")
    results["R5"] = {}
    for x in ("C", "D", "C32a", "D32a", "A32s", "A32r"):
        for ref in ("A32", "A64"):
            for i in inputs:
                results["R5"][(x, ref, i)] = emit(f"R5 behaviour {x} vs {ref} {i}", [(x, i), (ref, i)],
                                                  lambda x=x, ref=ref, i=i: r5(d(x, ref, i)), question="behaviour")
    results["R6"] = {}
    for b in backends:
        for i in inputs:
            if (f"{b}-batched", i) in runs:
                results["R6"][(b, i)] = emit(
                    f"R6 cached vs batched {b} {i}", [(b, i), (f"{b}-batched", i), ("A32", i)],
                    lambda b=b, i=i: r6(load_logits(runs[(b, i)], vocab), load_logits(runs[(f"{b}-batched", i)], vocab),
                                        load_logits(runs[("A32", i)], vocab), FAMILY[b]),
                    question="agreement (prospective, not judged)")
    r1v = {b: (results["R1"][b]["values"] or {"verdict": "not_admitted"}) for b in backends}
    r2v = (results["R2"]["values"] or {}).get("verdict", "not_admitted")
    r3v = (results["R3"]["values"] or {}).get("verdict", "not_admitted")
    sources = [line_ref(results["R1"][b]) for b in backends] + [line_ref(results["R2"]), line_ref(results["R3"])]
    if "not_admitted" in (r2v, r3v):
        entry = ev.reading("Outcome", ev.NOT_ADMITTED, [], None, sources=sources)
    else:
        entry = ev.reading("Outcome", ev.READING_COMPLETE, [], outcome(r1v, r2v, r3v), sources=sources)
    results["Outcome"] = log.append(entry)
    return results


#: R4 position sets on the 1,040-token inputs (rev 3 "Paths and positions").
REV2_SET = [0, 1, 20, 255, 510, 511, 512, 513, 514, 767, 1022, 1023, 1024, 1025, 1026, 1039]
BOUNDARY_SET = [80 + 80 * k for k in range(12)]
P43_SET = [44 + 80 * k for k in range(13)]
P16_SET = [17 + 80 * k for k in range(13)]
MATCHED_SET = [44 + 80 * k for k in range(12)] + [17 + 80 * k for k in range(12)]


def load_planes(run_dir: Path) -> tuple[dict, list[int], int]:
    """A capture's planes (LARQL E0a or hf_oracle layout), positions and layers."""
    sites = Path(run_dir) / "sites"
    man = json.loads((sites / "manifest.json").read_text())
    planes = {}
    for f in man["files"]:
        p = ev.contained(sites, f["file"])
        if p is None:
            raise ValueError(f"plane {f['file']!r} outside the capture")
        planes[(f["site"], f["layer"])] = np.fromfile(p, dtype=f.get("dtype", "<f4")).reshape(f["rows"], f["width"])
    return planes, man["positions"], len(man["layers"])


def capture_readings(runs: dict, expected: dict, log: ev.ResultsLog, inputs=("I_rep", "I_nr1"),
                     transformed_sites: frozenset = frozenset()) -> dict:
    """R4 for C and D against A32 (f32 pair) and A64, per input."""
    out = {}
    for i in inputs:
        for x in ("C", "D"):
            for ref in ("A32", "A64"):
                keys = [(f"{x}-capture", i), (f"{ref}-capture", i)]
                consumed = [ev.admit(runs[k], expected.get(k)) if k in runs
                            else ev.Admission(Path(f"<missing {k}>"), False, ["not run"]) for k in keys]
                name = f"R4 divergence {x} vs {ref} {i}"
                if all(a.admitted for a in consumed):
                    xp, xpos, layers = load_planes(runs[keys[0]])
                    rp, rpos, rlayers = load_planes(runs[keys[1]])
                    if xpos != rpos or layers != rlayers:
                        entry = ev.reading(name, ev.INVALID, consumed, None,
                                           reason="captures disagree on positions or layers")
                    else:
                        entry = ev.reading(name, ev.READING_COMPLETE, consumed, r4(
                            xp, rp, xpos, layers, f32_pair=(ref == "A32"), transformed_sites=transformed_sites,
                            position_sets={"rev2": REV2_SET, "boundary": BOUNDARY_SET, "P43": P43_SET,
                                           "P16": P16_SET},
                            boundary=BOUNDARY_SET, matched=MATCHED_SET), question="agreement" if ref == "A32"
                            else "accuracy")
                else:
                    entry = ev.reading(name, ev.NOT_ADMITTED, consumed)
                out[(x, ref, i)] = log.append(entry)
    return out


def analyse(*, runs: dict, expected: dict, vocab: int, out_dir: Path, log_path: Path,
            inputs=("I_rep", "I_nr1", "I_nr2")) -> dict:
    """The analysis as a run of its own: its record lists every consumed
    run record's digest, and its outputs are the metric arrays."""
    consumed = {f"{a}/{i}": (ev.record_digest(d) if (Path(d) / ev.RESULT_FILE).is_file() else None)
                for (a, i), d in sorted(runs.items())}
    rr = ev.RunRecord.begin(out_dir, "analysis", "prec-diag-1-analysis",
                            {"tool": "analysis", "scripts": script_digests(), "thresholds": thresholds(),
                             "consumed_records": consumed})
    try:
        log = ev.ResultsLog(log_path)
        results = logit_readings(runs, expected, vocab, out_dir, log, inputs)
        if any(a.endswith("-capture") for a, _ in runs):
            results["R4"] = capture_readings(runs, expected, log)
        (out_dir / "readings.json").write_text(json.dumps(
            {k if isinstance(k, str) else "|".join(k): v for k, v in _flatten(results).items()},
            indent=1, sort_keys=True, default=str))
        outputs = sorted(str(p.relative_to(out_dir)) for p in out_dir.iterdir()
                         if p.is_file() and p.name != ev.RESULT_FILE)
        return rr.complete(outputs)
    except Exception as e:
        return rr.fail(ev.FAULT, f"{type(e).__name__}: {e}")


def _flatten(results: dict) -> dict:
    out = {}
    for k, v in results.items():
        if isinstance(v, dict) and v and all(isinstance(x, dict) and "reading" in x for x in v.values()):
            for kk, vv in v.items():
                out[f"{k}|{kk if isinstance(kk, str) else '|'.join(kk)}"] = vv
        else:
            out[k] = v
    return out
