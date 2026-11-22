"""GW-TEMPLATE-FACTORIAL: model-free tokenisation and inventory feasibility.

DESIGN STAGE ONLY. No model is run. It reads the pinned GeoNames
countryInfo.txt (the hash GW-BREADCRUMB-2 pinned) and the Gemma 3 4B
tokenizer (the pinned hash), and reports, per candidate phrasing, the
semantic coordinates (subject span, relation span, readout position) and
whether they can be located; and, per relation, answer uniqueness,
tokenisation and donor diversity.

Usage: gwtemplate_feasibility.py OUT.json
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
COUNTRY_INFO = ROOT / "output/gw-breadcrumb-2-sources/geonames-20261115/countryInfo.txt"
COUNTRY_SHA = "79f8f59ea01cbb40438bf1856dd6c2ff201f03f8fc72d0530b2646aaf603ef04"
TOKENIZER = Path("/Users/christopherhay/chris-models/gemma3-4b-it.vindex3/tokenizer.json")
TOKENIZER_SHA = "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"

# Natural phrasings: {X} is the subject; the relation word is marked; the
# prompt ends where the answer begins.
PHRASINGS = {
    "capital": {"P1": ("The capital of {X} is", "capital"),
                "P2": ("For {X}, the capital is", "capital"),
                "P3": ("In {X}, the capital city is", "capital")},
    "currency": {"P1": ("The currency of {X} is the", "currency"),
                 "P2": ("For {X}, the currency is the", "currency"),
                 "P3": ("In {X}, the currency used is the", "currency")},
    # Fallbacks (r3, fixed order): ISO 4217 currency code, then ISO 3166-1 alpha-3.
    "currency_code_iso4217": {"P1": ("The ISO currency code of {X} is", "code"),
                              "P2": ("For {X}, the ISO currency code is", "code"),
                              "P3": ("In {X}, the ISO currency code used is", "code")},
    "country_code_iso3166_alpha3": {"P1": ("The ISO alpha-3 code of {X} is", "code"),
                                    "P2": ("For {X}, the ISO alpha-3 code is", "code"),
                                    "P3": ("In {X}, the ISO alpha-3 country code is", "code")},
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def countries():
    out = []
    for line in COUNTRY_INFO.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        out.append({"iso": f[0], "iso3": f[1], "name": f[4], "capital": f[5], "population": int(f[7] or 0),
                    "continent": f[8], "currency_code": f[10], "currency_name": f[11]})
    return out


def coords(tok, template, relation_word, name):
    """Semantic coordinates by character offsets: subject span, relation
    span, readout position (the last prompt token)."""
    text = template.format(X=name)
    enc = tok.encode(text, add_special_tokens=True)
    s0 = text.index(name)
    s1 = s0 + len(name)
    r0 = text.index(relation_word, s1 if template.index("{X}") < template.index(relation_word) else 0)
    r1 = r0 + len(relation_word)
    subj = [i for i, (a, b) in enumerate(enc.offsets) if b > a and a < s1 and b > s0]
    rel = [i for i, (a, b) in enumerate(enc.offsets) if b > a and a < r1 and b > r0]
    clean = (bool(subj) and all(s0 <= enc.offsets[i][0] - (1 if text[enc.offsets[i][0]] == " " else 0) or
                                enc.offsets[i][0] >= s0 - 1 for i in subj)
             and all(enc.offsets[i][1] <= s1 for i in subj))
    return {"ids": enc.ids, "subject": subj, "relation": rel, "readout": len(enc.ids) - 1,
            "subject_tokens": [enc.ids[i] for i in subj], "subject_aligned": clean}


def first_token(tok, text):
    ids = tok.encode(" " + text, add_special_tokens=False).ids
    return ids[0] if ids else None


def main(argv):
    if sha(COUNTRY_INFO) != COUNTRY_SHA or sha(TOKENIZER) != TOKENIZER_SHA:
        raise SystemExit("a pinned source differs")
    tok = Tokenizer.from_file(str(TOKENIZER))
    cs = [c for c in countries() if c["population"] > 0]
    report = {"status": "MODEL-FREE DESIGN CHECK; no inference", "countries_with_population": len(cs)}

    # Capital: answer uniqueness by first token.
    cap = [c for c in cs if c["capital"]]
    cap_first = collections.Counter(first_token(tok, c["capital"]) for c in cap)
    report["capital"] = {
        "countries_with_capital": len(cap),
        "distinct_first_tokens": len(cap_first),
        "countries_with_unique_first_token": sum(cap_first[first_token(tok, c["capital"])] == 1 for c in cap),
        "shared_first_tokens": {tok.id_to_token(t): n for t, n in cap_first.most_common(10) if n > 1},
        "single_token_capitals": sum(len(tok.encode(" " + c["capital"], add_special_tokens=False).ids) == 1 for c in cap),
    }
    # Currency: answer uniqueness, many-to-one sharing, tokenisation.
    cur = [c for c in cs if c["currency_name"]]
    by_name = collections.Counter(c["currency_name"] for c in cur)
    cur_first = collections.Counter(first_token(tok, c["currency_name"]) for c in cur)
    report["currency"] = {
        "countries_with_currency": len(cur),
        "distinct_currency_names": len(by_name),
        "most_shared_names": dict(by_name.most_common(12)),
        "countries_whose_currency_name_is_unique": sum(by_name[c["currency_name"]] == 1 for c in cur),
        "distinct_first_tokens": len(cur_first),
        "countries_with_unique_first_token": sum(cur_first[first_token(tok, c["currency_name"])] == 1 for c in cur),
        "largest_first_token_classes": {tok.id_to_token(t): n for t, n in cur_first.most_common(8)},
        "note": "GeoNames names lack the country adjective (USD 'Dollar', CAD 'Dollar'); the model may answer "
                "'US' or 'Canadian' first, so currency answer sets need a curated accepted-token list per country",
    }
    # Coordinates per phrasing.
    phr = {}
    both = [c for c in cs if c["capital"] and c["currency_name"]]
    for rel, ps in PHRASINGS.items():
        for pid, (template, word) in ps.items():
            ok, aligned, same_subject, rel_single = 0, 0, 0, 0
            for c in both:
                try:
                    k = coords(tok, template, word, c["name"])
                    ref = coords(tok, PHRASINGS["capital"]["P1"][0], "capital", c["name"])
                except ValueError:
                    continue
                ok += 1
                aligned += k["subject_aligned"]
                same_subject += k["subject_tokens"] == ref["subject_tokens"]
                rel_single += len(k["relation"]) == 1
            phr[f"{rel} {pid}"] = {"template": template, "located": ok, "subject_aligned": aligned,
                                   "subject_tokens_equal_to_capital_P1": same_subject,
                                   "relation_word_single_token": rel_single, "of": len(both)}
    report["phrasings"] = phr
    # r3: the same entity must have identical subject token IDs in all three
    # phrasings of a relation (absolute positions may differ).
    same = {}
    for rel, ps in PHRASINGS.items():
        n_same = 0
        for c in both:
            toks = [tuple(coords(tok, t, w, c["name"])["subject_tokens"]) for t, w in ps.values()]
            n_same += len(set(toks)) == 1
        same[rel] = {"countries_with_identical_subject_ids_across_P1_P3": n_same, "of": len(both),
                     "subject_positions_differ": True}
    report["subject_identity_across_phrasings"] = same
    # r3: currency alias and class inventory (model-free; no curation).
    def forms(name):
        return sorted({name, name.lower(), name + "s", name.lower() + "s"})
    cls = collections.defaultdict(list)
    for c in cur:
        cls[c["currency_name"]].append(c["iso"])
    alias = {}
    for name, isos in sorted(cls.items(), key=lambda kv: -len(kv[1])):
        firsts = sorted({first_token(tok, f) for f in forms(name)})
        alias[name] = {"countries": len(isos), "codes": sorted({c["currency_code"] for c in cur if c["currency_name"] == name}),
                       "forms": forms(name), "first_token_ids": firsts,
                       "first_tokens": [tok.id_to_token(t) for t in firsts]}
    fset = {n: set(a["first_token_ids"]) for n, a in alias.items()}
    overlapping = sorted({tuple(sorted((a, b))) for a in fset for b in fset if a != b and fset[a] & fset[b]})
    report["currency_alias_inventory"] = {
        "distinct_codes": len({c["currency_code"] for c in cur}), "distinct_names": len(cls),
        "classes_by_name": alias, "name_pairs_with_overlapping_first_tokens": [list(x) for x in overlapping],
        "classes_with_one_country": sum(len(v) == 1 for v in cls.values()),
        "countries_in_classes_of_size_le_3": sum(len(v) for v in cls.values() if len(v) <= 3),
        "note": "forms are the GeoNames noun only (capitalised, lowercase, plural); national adjectives "
                "('US', 'Swiss') are not in the source and are NOT curated here"}
    report["countries_with_capital_and_currency"] = len(both)
    # Upper bound of the frozen cohort rule per relation, as if every country
    # were eligible in all three phrasings (the baseline can only lower it).
    import gwtemplate_cohort as cohort
    def firsts(forms):
        return sorted({first_token(tok, f) for f in forms})
    name_cls = collections.defaultdict(list)
    code_cls = collections.defaultdict(list)
    for c in cur:
        name_cls[c["currency_name"]].append(c["iso"])
        code_cls[c["currency_code"]].append(c["iso"])
    a3_cls = {c["iso3"]: [c["iso"]] for c in cs if c["iso3"]}
    inputs = {
        "currency_name": (name_cls, {n: firsts(forms(n)) for n in name_cls}),
        "currency_code_iso4217": (code_cls, {k: firsts([k]) for k in code_cls}),
        "country_code_iso3166_alpha3": (a3_cls, {k: firsts([k]) for k in a3_cls}),
    }
    upper = {}
    everyone = {c["iso"] for c in cs}
    for name in cohort.FALLBACK_ORDER:
        cl, ft = inputs[name]
        r = cohort.select(cl, ft, everyone)
        upper[name] = {"classes_available": len(cl), "conflict_edges": r["conflict_edges"],
                       "selected_classes": len(r["selected_classes"]), "cohort_countries": r["n_countries"],
                       "cohort_classes": r["n_classes"], "feasible_if_all_eligible": r["feasible"],
                       "ledgered": len([e for e in r["ledger"] if e["country"]]),
                       "answer_first_token_is_whole_code": (sum(len(tok.encode(" " + k, add_special_tokens=False).ids) == 1 for k in cl)
                                                            if name != "currency_name" else None)}
    report["cohort_rule_upper_bound"] = upper
    out = Path(argv[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
