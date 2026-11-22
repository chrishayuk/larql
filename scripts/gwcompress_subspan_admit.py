"""GW-COMPRESS-1: sub-window admission extension (CP3b qualification addendum).

It adds to the band admission (`gwcompress_band_admit`, unchanged). It shares
no code with the band runner or with the CP3b position builder: positions are
re-derived here from the **decoded token strings**, not from token ids.

**Plan** (`admit_subspan_plan`):
- every arm `{kind}-{SET}-L{l}` has the span the decoded window gives for SET;
- no arm in a target repeats another arm's (band, span, vectors), so an alias
  such as P2 LAST (= REL) cannot be executed twice;
- the arm-name inventory per target is exactly the declared one.

**Records** (`admit_subspan_records`), using the record's per-window-position
carrier hashes:
- window positions **outside** the span equal the captured natural carriers
  at every layer <= the band's last layer (nothing at or below the
  intervention layer may change off-span);
- window positions **before** the span equal natural at **every** layer
  (causally they cannot see the injection);
- window positions after the span, above the band, may differ (causal
  downstream effect) and are not tested.
"""
from __future__ import annotations

DECODED = {"P2": [",", " the", " capital"], "P3": [",", " the", " capital", " city"]}
EXPECTED = {"REL": lambda w: [w[2]], "LAST": lambda w: [w[-1]], "LAST2": lambda w: w[-2:],
            "FULL": lambda w: list(w), "CTRL": lambda w: [w[0]]}


def _parse(name: str):
    kind, tag, layer = name.split("-")
    return kind, tag, int(layer[1:])


def admit_subspan_plan(plan: dict, decode, inventory: dict[str, list[str]]) -> list[str]:
    """inventory: phrasing -> exact list of arm names each target must carry."""
    fails = []
    for t in plan["targets"]:
        run = plan["runs"][t["run"]]
        ph = t["run"].split("|")[1]
        w = run["window"]
        got = [decode([run["tokens"][p]]) for p in w]
        if got != DECODED[ph]:
            fails.append(f"{t['id']}: decoded window {got} != {DECODED[ph]}")
            continue
        names = [a["name"] for a in t["arms"]]
        if sorted(names) != sorted(inventory[ph]):
            fails.append(f"{t['id']}: arm inventory differs ({len(names)} vs {len(inventory[ph])})")
        seen = {}
        for a in t["arms"]:
            kind, tag, _ = _parse(a["name"])
            if tag not in EXPECTED:
                fails.append(f"{t['id']} {a['name']}: unknown position set")
                continue
            if a["target_span"] != EXPECTED[tag](w):
                fails.append(f"{t['id']} {a['name']}: span {a['target_span']} is not {tag} {EXPECTED[tag](w)}")
            key = (a["from_layer"], a["to_layer"], tuple(a["target_span"]), tuple(sorted(a["vectors"].items())))
            if key in seen:
                fails.append(f"{t['id']} {a['name']}: duplicate of {seen[key]}")
            seen[key] = a["name"]
    return fails


def admit_subspan_records(plan: dict, rows: list[dict], caps) -> list[str]:
    fails = []
    out = {r["id"]: r for r in rows}
    for t in plan["targets"]:
        rec = out.get(t["id"])
        if rec is None:
            fails.append(f"{t['id']}: missing record")
            continue
        cap = caps.entries[t["run"]]["vector_sha256"]
        win = plan["runs"][t["run"]]["window"]
        for a in t["arms"]:
            r = rec["arms"].get(a["name"])
            if r is None:
                fails.append(f"{t['id']} {a['name']}: missing record")
                continue
            span, hi = a["target_span"], a["to_layer"]
            sc = r["span_carrier_sha256"]
            for p in win:
                if p in span:
                    continue
                top = plan["layers"] - 1 if p < span[0] else hi
                for l in range(top + 1):
                    k = f"{l}|{p}"
                    if sc.get(k) is None or sc.get(k) != cap.get(k):
                        fails.append(f"{t['id']} {a['name']} {k}: non-target window carrier is not natural")
    return fails
