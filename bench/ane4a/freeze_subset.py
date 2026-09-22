"""ANE-4A1 — freeze the depth-ladder's prompt subset, before any depth runs.

The subset is fixed here and never revisited. Choosing prompts after
seeing depth results — even informally, even by noticing that one
category looks bad — is how a ladder ends up tuned against an easy
slice. So the selection rule is written down, executed once, and its
output checksummed.

**The rule**, applied to quality-bank-1's 69 entries:

  1. Allocate the budget across the bank's 7 categories in proportion to
     each category's size, largest-remainder rounded. A drafter that is
     excellent on factual prompts and hopeless on arithmetic is a
     materially different result from one that is uniformly mediocre,
     and only a stratified subset can tell those apart.
  2. Within a category, sort by prompt length and take evenly spaced
     indices INCLUDING the shortest and the longest. Length is the other
     axis a truncated model might behave differently along — a 4-token
     prompt and a 116-token one exercise very different amounts of
     recurrent state.

Both steps are deterministic. No randomness, no seed to fiddle with, no
"first N".

Usage:
    python3 freeze_subset.py [budget, default 20]
"""

import hashlib
import json
import sys
from collections import defaultdict

BANK = "/Users/christopherhay/chris-models/qbanks/Qwen3.8-27B/quality-bank-1"
OUT = "bench/ane4a/subset-v1.json"


def spread(items, k):
    """`k` evenly spaced items, endpoints included. `k == 1` takes the
    median rather than an arbitrary end."""
    n = len(items)
    if k >= n:
        return list(items)
    if k == 1:
        return [items[n // 2]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rows = [json.loads(line) for line in open(f"{BANK}/ref/_entries.jsonl")]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["id"].rsplit("-", 1)[0]].append(r)

    # Largest-remainder apportionment, so the allocation is reproducible
    # rather than depending on dict order or rounding luck.
    exact = {c: len(rs) * budget / len(rows) for c, rs in by_cat.items()}
    alloc = {c: int(v) for c, v in exact.items()}
    remainder = budget - sum(alloc.values())
    for cat in sorted(exact, key=lambda c: (-(exact[c] - alloc[c]), c))[:remainder]:
        alloc[cat] += 1

    chosen = []
    for cat in sorted(by_cat):
        ordered = sorted(by_cat[cat], key=lambda r: (len(r["ids"]), r["id"]))
        for r in spread(ordered, alloc[cat]):
            chosen.append({"id": r["id"], "category": cat,
                           "positions": len(r["ids"]), "ids": r["ids"]})
    chosen.sort(key=lambda e: e["id"])

    doc = {
        "subset": "ane4a-subset-v1",
        "frozen_before_any_depth_result": True,
        "rule": "proportional by category (largest remainder), then evenly "
                "spaced by prompt length including both extremes",
        "budget": budget,
        "bank": BANK,
        "allocation": {c: alloc[c] for c in sorted(alloc)},
        "total_positions": sum(e["positions"] for e in chosen),
        "entries": chosen,
    }
    body = json.dumps(doc, indent=2)
    digest = hashlib.sha256(body.encode()).hexdigest()
    with open(OUT, "w") as fh:
        fh.write(body + "\n")

    print(f"{'category':<14}{'alloc':>6}{'of':>5}   ids")
    for cat in sorted(by_cat):
        picked = [e for e in chosen if e["category"] == cat]
        print(f"{cat:<14}{alloc[cat]:>6}{len(by_cat[cat]):>5}   "
              + ", ".join(f"{e['id'].rsplit('-', 1)[1]}({e['positions']})" for e in picked))
    print(f"\n{len(chosen)} prompts, {doc['total_positions']} positions "
          f"(bank has {len(rows)} / {sum(len(r['ids']) for r in rows)})")
    print(f"wrote {OUT}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
