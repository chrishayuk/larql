#!/usr/bin/env python3
"""Freeze quality-bank-3a AND 3b together, or neither.

    python3 cpu6_freeze.py freeze <container>
    python3 cpu6_freeze.py verify <container>

CPU-6 requires both validation banks to be frozen SIMULTANEOUSLY, before
either runs, so that 3A's composition cannot be influenced by 3B's and
neither by results. This tool makes that an INVARIANT rather than a
sentence in a protocol document: it refuses to freeze either bank unless
both exist and pass mutual disjointness.

Checked, never asserted:

    3A and 3B disjoint from quality-bank-1        (spent, discovery)
    3A and 3B disjoint from quality-bank-2        (spent on CPU-5)
    3A and 3B disjoint from the 1B' calibration   (spent)
    3A disjoint from 3B
    each bank matches the FROZEN category targets exactly
"""
import json, os, re, sys, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
BANKS = {"3a": os.path.join(HERE, "quality-bank-3a"),
         "3b": os.path.join(HERE, "quality-bank-3b")}
PRIOR = [
    ("quality-bank-1", os.path.join(HERE, "quality-bank-1", "prompts.json")),
    ("quality-bank-2", os.path.join(HERE, "quality-bank-2", "prompts.json")),
    ("sensitivity-1b-calibration",
     os.path.join(HERE, "quality-bank-1", "calibration-disjoint.json")),
]
# Frozen in CPU6-VALIDATION.md before either bank was authored.
TARGETS = {"factual": 44, "prose": 29, "code": 29, "arithmetic": 29,
           "uncertain": 29, "structured": 23, "longform": 17}
LIMIT, MIN_IDS = 128, 3


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def norm(s):
    return re.sub(r"\s+", " ", s.strip()).lower()


def prompts_of(path):
    d = json.load(open(path))
    return d["prompts"] if isinstance(d, dict) else d


def refuse(why):
    raise SystemExit(f"REFUSED: {why}")


def disjoint(name_a, a, name_b, b):
    ids = {p["id"] for p in a} & {p["id"] for p in b}
    texts = {norm(p["text"]) for p in a} & {norm(p["text"]) for p in b}
    pref = {norm(p["text"])[:40] for p in b}
    near = sorted(p["id"] for p in a if norm(p["text"])[:40] in pref)
    if ids or texts or near:
        refuse(f"{name_a} is not disjoint from {name_b}\n"
               f"  id overlap      : {sorted(ids)[:8]}\n"
               f"  text overlap    : {len(texts)}\n"
               f"  near-duplicates : {near[:8]}")


def check_one(key, path, mine):
    from collections import Counter
    got = Counter(p["category"] for p in mine)
    if dict(got) != TARGETS:
        refuse(f"bank {key} composition {dict(got)} != frozen targets {TARGETS}")
    seen = {}
    for p in mine:
        k = norm(p["text"])
        if k in seen:
            refuse(f"bank {key} internal duplicate: {seen[k]} / {p['id']}")
        seen[k] = p["id"]
    for name, prior in PRIOR:
        disjoint(f"bank {key}", mine, name, prompts_of(prior))


def tokenize(container, mine, key):
    import tokenizers
    path = os.path.join(container, "tokenizer.json")
    tk = tokenizers.Tokenizer.from_file(path)
    out = []
    for p in mine:
        ids = tk.encode(p["text"]).ids[:LIMIT]
        if len(ids) < MIN_IDS:
            refuse(f"bank {key} `{p['id']}` tokenises to {len(ids)} ids; run_bank.py "
                   f"would drop it silently and shrink the bank without saying so")
        out.append({"id": p["id"], "ids": ids})
    return out, path


def canonical(records):
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


def build(container):
    loaded = {}
    for key, d in BANKS.items():
        f = os.path.join(d, "prompts.json")
        if not os.path.exists(f):
            refuse(f"bank {key} does not exist at {f} — CPU-6 freezes BOTH banks or "
                   f"neither, so that one cannot be authored knowing the other")
        loaded[key] = json.load(open(f))

    for key in BANKS:
        check_one(key, os.path.join(BANKS[key], "prompts.json"), loaded[key]["prompts"])
    # The mutual check, which is the whole reason this tool exists.
    disjoint("bank 3a", loaded["3a"]["prompts"], "bank 3b", loaded["3b"]["prompts"])

    out = {}
    for key, d in loaded.items():
        records, tkpath = tokenize(container, d["prompts"], key)
        out[key] = (d, records, {
            "text_digest": {"algorithm": "sha256", "value": sha(canonical(
                [{"id": p["id"], "text": norm(p["text"])} for p in d["prompts"]]))},
            "token_digest": {"algorithm": "sha256", "value": sha(canonical(records)),
                             "limit": LIMIT,
                             "positions": sum(len(r["ids"]) for r in records),
                             "tokenizer": {"sha256": sha(open(tkpath).read())}},
            "disjointness": {"verified_against": [n for n, _ in PRIOR] + ["the other CPU-6 bank"],
                             "method": "exact id, whitespace/case-normalised text, "
                                       "40-character prefix near-duplicate scan"},
            "category_targets": TARGETS,
        })
    return out


def main():
    cmd, container = sys.argv[1], sys.argv[2]
    built = build(container)
    for key, (d, records, meta) in built.items():
        path = os.path.join(BANKS[key], "prompts.json")
        if cmd == "freeze":
            d.update(meta)
            d["count"] = len(d["prompts"])
            json.dump(d, open(path, "w"), indent=1, ensure_ascii=False)
            with open(os.path.join(BANKS[key], "prompts.tokens.jsonl"), "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"froze {key}: {d['count']} prompts, "
                  f"{meta['token_digest']['positions']} positions, "
                  f"token {meta['token_digest']['value'][:16]}")
        elif cmd == "verify":
            cur = json.load(open(path))
            for k in ("text_digest", "token_digest"):
                if cur.get(k, {}).get("value") != meta[k]["value"]:
                    refuse(f"bank {key} {k} moved since freeze")
            print(f"verified {key}: {len(d['prompts'])} prompts unchanged")
        else:
            raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
