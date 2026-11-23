#!/usr/bin/env python3
"""Mechanical check that every target in bank.json can be seen by the balance controller's matcher
on both tokenizers, and that registered targets do not share a first token.

The matcher (target_token_matches) accepts a decoded token that contains the target, or starts
with the target's first three characters, after dropping leading whitespace. So the first token of
" <target>" must be at least three characters long once trimmed, and must start with the target's
first three characters (or contain the target). This reads tokenizers only; it observes no model
behaviour.

Usage: check_targets.py <qwen tokenizer.json> <gemma tokenizer.json>
"""
import json, pathlib, sys
from tokenizers import Tokenizer

bank = json.loads(pathlib.Path(__file__).with_name("bank.json").read_text())
tokenizers = {name: Tokenizer.from_file(path) for name, path in zip(("qwen", "gemma"), sys.argv[1:3])}

def first_token(tk, target):
    ids = tk.encode(" " + target, add_special_tokens=False).ids
    return ids[0], tk.decode(ids[:1]).strip()

failures = []
for group in ("facts", "reserve", "development"):
    for fact in bank[group]:
        t = fact["target"]
        for name, tk in tokenizers.items():
            _, first = first_token(tk, t)
            if not (len(first) >= 3 and (first.startswith(t[:3]) or t in first)):
                failures.append(f"{group}: {t!r} on {name}: first token {first!r}")
for name, tk in tokenizers.items():
    ids = [first_token(tk, f["target"])[0] for f in bank["facts"]]
    if len(set(ids)) != len(ids):
        failures.append(f"registered targets share a first token on {name}")
if failures:
    print("\n".join(failures)); sys.exit(1)
print("all targets visible to the matcher on both tokenizers; registered first tokens distinct")
