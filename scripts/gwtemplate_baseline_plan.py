"""GW-TEMPLATE-FACTORIAL: plans for the template-agnostic natural baseline.

Model-free. It builds the intended population plan, every (country,
relation, phrasing) cell under the r3 design. It runs nothing. Each cell gets:

- its text and token IDs (the pinned tokenizer);
- its semantic coordinates, located by character offsets;
- its relation's candidate set;
- its country's accepted first-token set, for later scoring. The executable
  ignores this field.

The r3 design fixes the phrasings, relations and fallback order. This file
reads them from `gwtemplate_feasibility.PHRASINGS` and
`gwtemplate_cohort.FALLBACK_ORDER` and does not restate them.

Usage: gwtemplate_baseline_plan.py OUT_PLAN.json [--relations capital,...]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwtemplate_cohort as cohort  # noqa: E402
import gwtemplate_feasibility as feas  # noqa: E402

SCHEMA = "larql.gwtemplate.baseline-plan.v1"
CONTAINER = str(feas.TOKENIZER.parent)
CONTINUATION = 8
TOP_K = 64
# Relation name -> (phrasing key in feas.PHRASINGS, class of a country, forms of a class).
RELATIONS = ("capital",) + cohort.FALLBACK_ORDER
_PHRASING_KEY = {"capital": "capital", "currency_name": "currency",
                 "currency_code_iso4217": "currency_code_iso4217",
                 "country_code_iso3166_alpha3": "country_code_iso3166_alpha3"}


def tokenizer():
    from tokenizers import Tokenizer
    if feas.sha(feas.TOKENIZER) != feas.TOKENIZER_SHA:
        raise SystemExit("the pinned tokenizer differs")
    return Tokenizer.from_file(str(feas.TOKENIZER))


def feas_forms(name):
    """The r3 inventory's forms of a currency name (gwtemplate_feasibility)."""
    return sorted({name, name.lower(), name + "s", name.lower() + "s"})


def accepted(tok, relation: str, c: dict) -> list[int]:
    """The country's accepted first-token set, without curation, as in the
    r3 inventory and feasibility-03."""
    if relation == "capital":
        texts = [c["capital"]]
    elif relation == "currency_name":
        texts = feas_forms(c["currency_name"])
    elif relation == "currency_code_iso4217":
        texts = [c["currency_code"]]
    else:
        texts = [c["iso3"]]
    return sorted({feas.first_token(tok, t) for t in texts})


def span(enc, text: str, s0: int, s1: int) -> list[int]:
    """The independent Python location of a character span: the tokens
    overlapping [s0, s1). It must agree with the executable's."""
    return [i for i, (a, b) in enumerate(enc.offsets) if b > a and a < s1 and b > s0]


def cell(tok, cid: str, template: str, word: str, subject: str, cset: str, extra: dict) -> dict:
    text = template.format(X=subject)
    enc = tok.encode(text, add_special_tokens=True)
    s0 = text.index(subject)
    s1 = s0 + len(subject)
    after = template.index("{X}") < template.index(word)
    r0 = text.index(word, s1 if after else 0)
    r1 = r0 + len(word)
    return {"id": cid, "text": text, "token_ids": enc.ids,
            "subject": {"text": subject, "char": [s0, s1], "tokens": span(enc, text, s0, s1)},
            "relation": {"text": word, "char": [r0, r1], "tokens": span(enc, text, r0, r1)},
            "readout": len(enc.ids) - 1, "candidate_set": cset, **extra}


def population(relations=RELATIONS) -> dict:
    tok = tokenizer()
    cs = [c for c in feas.countries() if c["population"] > 0 and c["capital"] and c["currency_name"]]
    cells, sets = [], collections.defaultdict(set)
    for rel in relations:
        if rel not in RELATIONS:
            raise SystemExit(f"unknown relation {rel}")
        for c in cs:
            acc = accepted(tok, rel, c)
            sets[rel] |= set(acc)
            for pid, (template, word) in feas.PHRASINGS[_PHRASING_KEY[rel]].items():
                cells.append(cell(tok, f"{rel}|{pid}|{c['iso']}", template, word, c["name"], rel,
                                  {"country": c["iso"], "relation_name": rel, "phrasing": pid,
                                   "accepted_first_tokens": acc}))
    return {"schema": SCHEMA, "container": CONTAINER, "tokenizer_sha256": "sha256:" + feas.TOKENIZER_SHA,
            "continuation_tokens": CONTINUATION, "top_k": TOP_K,
            "purpose": "GW-TEMPLATE-FACTORIAL natural baseline population (r3); NOT AUTHORISED TO RUN",
            "sources": {"countryInfo_sha256": "sha256:" + feas.COUNTRY_SHA},
            "candidate_sets": {k: sorted(v) for k, v in sets.items()},
            "cells": cells}


def main(argv):
    rels = RELATIONS
    if len(argv) == 3 and argv[1] == "--relations":
        rels = tuple(argv[2].split(","))
    elif len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    plan = population(rels)
    out = Path(argv[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1) + "\n")
    by = collections.Counter(c["relation_name"] for c in plan["cells"])
    print(json.dumps({"cells": len(plan["cells"]), "by_relation": by,
                      "candidate_set_sizes": {k: len(v) for k, v in plan["candidate_sets"].items()},
                      "prompt_tokens_total": sum(len(c["token_ids"]) for c in plan["cells"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
