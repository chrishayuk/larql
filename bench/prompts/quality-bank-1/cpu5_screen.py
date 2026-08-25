#!/usr/bin/env python3
"""Build a SCREEN view over an existing bank reference.

    python3 cpu5_screen.py <bank-outdir> <screen-outdir> [--per-category 2] [--longform 1]

An exception-set search runs many arms, and running each over all 69
prompts costs about an hour apiece. The screen is a fixed subset of the
DISCOVERY bank used to ORDER operand families by sensitivity; the chosen
candidate is then re-scored on the full discovery bank, and accepted or
rejected on quality-bank-2, which no search ever touches.

It reuses the banked reference logits rather than re-running BF16: the
entries it selects point back at the original dumps, so a screen costs
exactly one candidate run and nothing else.

**Selection is a RULE, not a choice.** The first `--per-category` entries
of each category in bank order, plus `--longform` long entries. Picking
which prompts to screen on after seeing where an arm struggled would make
the screen a way of choosing a flattering subset.
"""
import json, os, sys


def arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    bank_dir, screen_dir = sys.argv[1], sys.argv[2]
    per_cat = arg("--per-category", 2)
    n_long = arg("--longform", 1)

    meta = json.load(open(os.path.join(bank_dir, "reference.json")))
    picked, seen = [], {}
    for e in meta["entries"]:
        cat = e["category"]
        limit = n_long if cat == "longform" else per_cat
        if seen.get(cat, 0) < limit:
            seen[cat] = seen.get(cat, 0) + 1
            # Re-point the dump at the original bank, so the reference is
            # shared rather than duplicated.
            src = os.path.join(bank_dir, e["dump"])
            picked.append({**e, "dump": os.path.relpath(src, screen_dir)})

    os.makedirs(screen_dir, exist_ok=True)
    out = {**meta, "entries": picked,
           "screen_of": os.path.abspath(bank_dir),
           "selection": {"rule": "first N per category in bank order",
                         "per_category": per_cat, "longform": n_long}}
    json.dump(out, open(os.path.join(screen_dir, "reference.json"), "w"), indent=1)
    positions = sum(len(e["ids"]) for e in picked)
    print(f"screen: {len(picked)} entries, {positions} positions "
          f"({positions * 100 // sum(len(e['ids']) for e in meta['entries'])}% of the bank)")
    for c, n in sorted(seen.items()):
        print(f"  {c:<12} {n}")


if __name__ == "__main__":
    main()
