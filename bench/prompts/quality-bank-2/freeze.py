#!/usr/bin/env python3
"""Freeze quality-bank-2 and PROVE it is disjoint from everything already used.

    python3 freeze.py freeze <container>     # stamp digests + token bank
    python3 freeze.py verify <container>     # re-derive and compare

Quality Bank 2 exists to answer one question honestly: **is the mixed
precision plan good, or is it merely tuned?**

If the same prompts are used to (1) locate where Q4 fails, (2) choose
which operands to protect, and (3) declare the result acceptable, then
the acceptance number is partly a training number and the exception set
has been fitted to its own test. That is not a small effect when the
search is over operator families and the metric is a tail statistic.

So: bank 1 is the DISCOVERY bank and may be looked at freely. This bank
is scored ONCE, on the final candidate. Disjointness is verified here
rather than asserted in a comment, against both quality-bank-1 and the
SENSITIVITY-1B' calibration set, which has already been spent.
"""
import json, os, re, sys, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
SELF = os.path.join(HERE, "prompts.json")
PRIOR = [
    ("quality-bank-1", os.path.join(HERE, "..", "quality-bank-1", "prompts.json")),
    ("sensitivity-1b-calibration",
     os.path.join(HERE, "..", "quality-bank-1", "calibration-disjoint.json")),
]
# Inherited from run_bank.py, so the frozen ids are the ids it will run.
LIMIT = 128
MIN_IDS = 3


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def norm(s):
    return re.sub(r"\s+", " ", s.strip()).lower()


def prompts_of(path):
    d = json.load(open(path))
    return d["prompts"] if isinstance(d, dict) else d


def verify_disjoint(mine):
    """Disjointness is CHECKED, never trusted to a `note` field."""
    report = {}
    for name, path in PRIOR:
        other = prompts_of(path)
        ids = {p["id"] for p in other} & {p["id"] for p in mine}
        texts = {norm(p["text"]) for p in other} & {norm(p["text"]) for p in mine}
        # A 40-char prefix scan catches near-duplicates that differ only
        # in a trailing clause — exact matching would call those disjoint.
        pref = {norm(p["text"])[:40] for p in other}
        near = sorted(p["id"] for p in mine if norm(p["text"])[:40] in pref)
        if ids or texts or near:
            raise SystemExit(
                f"REFUSED: quality-bank-2 is not disjoint from {name}.\n"
                f"  id overlap        : {sorted(ids)}\n"
                f"  text overlap      : {len(texts)}\n"
                f"  near-duplicates   : {near}"
            )
        report[name] = {"prompts": len(other), "id_overlap": 0,
                        "text_overlap": 0, "near_duplicate_pairs": 0}
    seen = {}
    for p in mine:
        k = norm(p["text"])
        if k in seen:
            raise SystemExit(f"REFUSED: internal duplicate {seen[k]} / {p['id']}")
        seen[k] = p["id"]
    report["internal_duplicates"] = 0
    return report


def tokenize(container, mine):
    import tokenizers
    path = os.path.join(container, "tokenizer.json")
    tk = tokenizers.Tokenizer.from_file(path)
    out = []
    for p in mine:
        ids = tk.encode(p["text"]).ids[:LIMIT]
        if len(ids) < MIN_IDS:
            raise SystemExit(
                f"REFUSED: `{p['id']}` tokenises to {len(ids)} ids and run_bank.py "
                f"would drop it silently, shrinking the bank without saying so")
        out.append({"id": p["id"], "ids": ids})
    return out, path


def canonical(records):
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


def build(container):
    d = json.load(open(SELF))
    mine = d["prompts"]
    disjoint = verify_disjoint(mine)
    records, tkpath = tokenize(container, mine)
    text_digest = sha(canonical(
        [{"id": p["id"], "text": norm(p["text"])} for p in mine]))
    token_digest = sha(canonical(records))
    return d, mine, records, {
        "disjointness": {
            "method": "exact id match, whitespace/case-normalised text match, "
                      "and a 40-character prefix near-duplicate scan",
            "verified_against": disjoint,
        },
        "text_digest": {"algorithm": "sha256", "value": text_digest},
        "token_digest": {
            "algorithm": "sha256", "value": token_digest,
            "limit": LIMIT,
            "positions": sum(len(r["ids"]) for r in records),
            "tokenizer": {"path": os.path.relpath(tkpath, HERE),
                          "sha256": sha(open(tkpath).read())},
        },
    }


def main():
    cmd, container = sys.argv[1], sys.argv[2]
    d, mine, records, meta = build(container)
    if cmd == "freeze":
        d.update(meta)
        d["count"] = len(mine)
        json.dump(d, open(SELF, "w"), indent=1, ensure_ascii=False)
        with open(os.path.join(HERE, "prompts.tokens.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"froze {len(mine)} prompts, {meta['token_digest']['positions']} positions")
        print(f"  text  {meta['text_digest']['value']}")
        print(f"  token {meta['token_digest']['value']}")
    elif cmd == "verify":
        cur = json.load(open(SELF))
        for k in ("text_digest", "token_digest"):
            was, now = cur.get(k, {}).get("value"), meta[k]["value"]
            if was != now:
                raise SystemExit(f"REFUSED: {k} moved\n  frozen {was}\n  now    {now}")
        print(f"verified: {len(mine)} prompts unchanged since freeze")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
