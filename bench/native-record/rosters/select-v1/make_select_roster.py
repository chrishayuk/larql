#!/usr/bin/env python3
"""ADDRESS-SELECT-1 Stage 1 fresh roster generator (select-v1).

Frozen before any selector work. See the address-select-1 registry record.

The roster is the evaluation set and is touched EXACTLY ONCE, after every
layer / selector-family / threshold choice has been fixed on the already-burned
campaign development material. Regenerating with the same seed reproduces it
byte-for-byte; any change to entities, forms, relations or counts is a NEW
roster version, never an edit.

Two banks of twelve slots each, matching decomposition-v1 / oracle-v2 exactly
(12 slots, L26, 1x down-column strength) so the C+A ceiling stays comparable:

  bank A  4 development entities  -> supplies the STRUCTURE-SHIFT stratum
  bank B  4 unseen entities       -> supplies the ENTITY-SHIFT and
                                     DOUBLE-SHIFT strata

Three positive strata, 24 prompts each, 8 per relation:

  structure_shift   known entities  x unseen forms
  entity_shift      unseen entities x known forms
  double_shift      unseen entities x unseen forms

Four negative families, scored SEPARATELY and never pooled:
uninstalled entities, near-names, wrong relations, unrelated.

Disjointness against every burned entity and every burned query form is
asserted from the campaign fixtures at generation time, not assumed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent.parent
RESULTS = CAMPAIGN / "results"

SEED = 20260906
ROSTER_VERSION = "address-select-roster-v1"
RELATIONS = ("capital", "currency", "language")

# --- bank A: development entities, already installed, canonical activations
# already stored by oracle-v2 / decomposition-v1. Their VALUES are unchanged
# from the factorial bank so Stage 2's ceiling is directly comparable.
BANK_A = [
    ("Avenlorn", {"capital": "Oslo", "currency": "Euro", "language": "German"}),
    ("Braskovia", {"capital": "Paris", "currency": "Dollar", "language": "Spanish"}),
    ("Celdrune", {"capital": "Rome", "currency": "Pound", "language": "Welsh"}),
    ("Dornessa", {"capital": "Tokyo", "currency": "Yen", "language": "Japanese"}),
]

# --- bank B: unseen entities, installed fresh for this roster. Values reuse
# the same value pool so value identity is not confounded with entity novelty.
BANK_B = [
    ("Ithuvane", {"capital": "Oslo", "currency": "Euro", "language": "German"}),
    ("Jorrenmark", {"capital": "Paris", "currency": "Dollar", "language": "Spanish"}),
    ("Kravanthe", {"capital": "Rome", "currency": "Pound", "language": "Welsh"}),
    ("Lomberask", {"capital": "Tokyo", "currency": "Yen", "language": "Japanese"}),
]

# --- forms. KNOWN forms are drawn from the burned campaign banks; UNSEEN forms
# must not appear anywhere in them (asserted below).
KNOWN_FORMS = [
    "The recorded {r} for {e} is",
    "With regard to {e}, its {r} is",
]
UNSEEN_FORMS = [
    "Consulting the gazetteer, {e} lists its {r} as",
    "Whatever else is true of {e}, its {r} remains",
]

# Enrollment form: one per record, deliberately distinct from every query form
# in both the burned banks and this roster, following the campaign convention.
ENROLLMENT_FORM = "The {r} listed for {e} is"

# --- negatives
UNINSTALLED_ENTITIES = [
    "Uruguay", "Namibia", "Paraguay", "Botswana",
    "Ecuador", "Zambia", "Honduras", "Senegal",
]
NEAR_NAME_SUFFIX = "dar"  # -eth / -ix / -a / -un are all burned near-name patterns
UNINSTALLED_RELATIONS = ["national bird", "time zone", "highest mountain", "coastline"]
UNRELATED = [
    "A kettle is used to boil",
    "The instrument with 88 keys is a",
    "Bees produce sweet",
    "The season after summer is",
    "A triangle has three",
    "Sailors navigate using the north",
    "Bread is baked inside an",
    "The wheel was invented to help people",
]


def burned_inventory() -> tuple[set[str], set[str]]:
    """Every entity name and every templatised prompt across all seven artifacts."""
    entities: set[str] = set()
    prompts: set[str] = set()

    def walk(obj) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key in ("facts", "records") and isinstance(val, list):
                    for item in val:
                        if isinstance(item, list) and item and isinstance(item[0], str):
                            entities.add(item[0])
                walk(val)
        elif isinstance(obj, list):
            for val in obj:
                walk(val)
        elif isinstance(obj, str) and len(obj) < 200 and obj.endswith((" is", " finds", " answer is")):
            prompts.add(obj)

    for path in sorted(glob.glob(str(RESULTS / "*" / "fixture.json"))):
        with open(path) as handle:
            walk(json.load(handle))

    # Names that only ever appear inside prompt text (controller rosters) still count.
    for prompt in list(prompts):
        for word in prompt.replace(".", " ").replace(",", " ").replace("'", " ").split():
            if word[:1].isupper() and word.isalpha() and len(word) > 5:
                entities.add(word)

    templates: set[str] = set()
    for prompt in prompts:
        text = prompt
        for ent in sorted(entities, key=len, reverse=True):
            text = text.replace(ent, "{e}")
        for rel in RELATIONS:
            text = text.replace(rel, "{r}")
        templates.add(text)
    return entities, templates


def assert_disjoint(burned_entities: set[str], burned_templates: set[str]) -> None:
    for name, _ in BANK_B:
        assert name not in burned_entities, f"bank B entity {name} is already burned"
        for burned in burned_entities:
            assert not (name.startswith(burned[:5]) or burned.startswith(name[:5])), (
                f"bank B entity {name} shares a stem with burned entity {burned}"
            )
    for form in UNSEEN_FORMS:
        assert form not in burned_templates, f"unseen form is burned: {form}"
    for form in KNOWN_FORMS:
        assert form in burned_templates, f"known form is NOT in the burned banks: {form}"
    assert ENROLLMENT_FORM not in {f for f in burned_templates}, "enrollment form collides with a query form"
    for name, _ in BANK_A:
        assert name in burned_entities, f"bank A entity {name} should be a development entity"


def build() -> dict:
    rows: list[dict] = []

    def add(bank: str, group: str, stratum: str | None, prompt: str,
            entity: str | None, relation: str | None, expected: str | None) -> None:
        rows.append({
            "id": f"{group}-{len(rows):04d}",
            "bank": bank,
            "group": group,
            "stratum": stratum,
            "prompt": prompt,
            "entity": entity,
            "relation": relation,
            "expected": expected,
        })

    # enrollment: the selector's candidate set, one per installed record
    for bank, records in (("A", BANK_A), ("B", BANK_B)):
        for entity, values in records:
            for rel in RELATIONS:
                add(bank, "enrollment", None,
                    ENROLLMENT_FORM.format(e=entity, r=rel), entity, rel, values[rel])

    # positives
    for entity, values in BANK_A:
        for rel in RELATIONS:
            for form in UNSEEN_FORMS:
                add("A", "positive", "structure_shift",
                    form.format(e=entity, r=rel), entity, rel, values[rel])
    for entity, values in BANK_B:
        for rel in RELATIONS:
            for form in KNOWN_FORMS:
                add("B", "positive", "entity_shift",
                    form.format(e=entity, r=rel), entity, rel, values[rel])
            for form in UNSEEN_FORMS:
                add("B", "positive", "double_shift",
                    form.format(e=entity, r=rel), entity, rel, values[rel])

    # negatives, four families, scored separately
    for i, country in enumerate(UNINSTALLED_ENTITIES):
        rel = RELATIONS[i % len(RELATIONS)]
        add("B" if i % 2 else "A", "negative_uninstalled", None,
            KNOWN_FORMS[0].format(e=country, r=rel), country, rel, None)

    near = [(f"{e[:-2]}{NEAR_NAME_SUFFIX}", "A") for e, _ in BANK_A]
    near += [(f"{e[:-2]}{NEAR_NAME_SUFFIX}", "B") for e, _ in BANK_B]
    for i, (name, bank) in enumerate(near):
        rel = RELATIONS[i % len(RELATIONS)]
        add(bank, "negative_near_name", None,
            KNOWN_FORMS[0].format(e=name, r=rel), name, rel, None)

    installed = [(e, "A") for e, _ in BANK_A] + [(e, "B") for e, _ in BANK_B]
    for i, (entity, bank) in enumerate(installed):
        rel = UNINSTALLED_RELATIONS[i % len(UNINSTALLED_RELATIONS)]
        add(bank, "negative_wrong_relation", None,
            KNOWN_FORMS[0].format(e=entity, r=rel), entity, rel, None)

    for i, prompt in enumerate(UNRELATED):
        add("B" if i % 2 else "A", "negative_unrelated", None, prompt, None, None, None)

    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()
    out_dir = Path(args.out)

    burned_entities, burned_templates = burned_inventory()
    assert_disjoint(burned_entities, burned_templates)

    roster = build()
    rows = roster["rows"]

    body = "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
    (out_dir / "roster.jsonl").write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()

    counts: dict[str, int] = {}
    for row in rows:
        key = row["stratum"] or row["group"]
        counts[key] = counts.get(key, 0) + 1

    relation_balance: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["group"] == "positive":
            relation_balance.setdefault(row["stratum"], {})
            relation_balance[row["stratum"]][row["relation"]] = (
                relation_balance[row["stratum"]].get(row["relation"], 0) + 1
            )

    manifest = {
        "roster_version": ROSTER_VERSION,
        "seed": SEED,
        "generator": "make_select_roster.py",
        "roster_sha256": digest,
        "total_rows": len(rows),
        "counts": counts,
        "relation_balance": relation_balance,
        "banks": {
            "A": {"entities": [e for e, _ in BANK_A], "role": "development entities, already installed"},
            "B": {"entities": [e for e, _ in BANK_B], "role": "unseen entities, installed fresh"},
        },
        "known_forms": KNOWN_FORMS,
        "unseen_forms": UNSEEN_FORMS,
        "enrollment_form": ENROLLMENT_FORM,
        "burned_entities_checked": sorted(burned_entities),
        "burned_template_count": len(burned_templates),
        "negative_families": [
            "negative_uninstalled", "negative_near_name",
            "negative_wrong_relation", "negative_unrelated",
        ],
        "rules": [
            "Evaluated EXACTLY ONCE, after layer/selector-family/threshold choices are frozen on burned development material.",
            "Negative families are scored separately and never pooled.",
            "No rescue after seeing strata: if structure_shift passes and entity_shift fails, that IS the result.",
            "Any change to entities, forms, relations or counts is a NEW roster version, never an edit.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
