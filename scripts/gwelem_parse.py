"""GW-ELEMENT-1: the frozen natural-answer parser (bound before any element output is seen).

It works on token ids, decoding each token separately with the pinned tokenizer.
Every rule here is fixed before the baseline runs. It never widens
acceptance after outputs are seen.

**Symbol answers** (prompt "... the chemical symbol is" / "...'s chemical symbol is"):
- correct iff continuation[0] equals the first token of " " + symbol
  (every token-eligible symbol is a single token);
- **and** the next token's decoded text does not begin with a lower-case
  ASCII letter. Otherwise the model wrote a longer symbol, e.g. " C"+"l" = Cl
  for carbon, which is refused;
- if continuation[0] is the right token but it is the last continuation token,
  the answer is "truncated" and refused (no terminator was observed).

**Atomic-number answers** (prompt "... the atomic number is"):
1. continuation[0] must be the bare-space token " " (Gemma's number form:
   " " followed by single digits). Otherwise `format: first_token_not_space`.
   This includes markdown such as "**".
2. The digit run is the maximal run of following tokens whose decoded text is
   non-empty and all ASCII digits.
3. An empty run gives `format: no_digits`.
4. **Terminator required:** the token after the run must exist inside the
   continuation. If the run reaches the end of the continuation, the answer is
   `truncated`, refused and never accepted.
5. The terminator is a special token (end of turn / end of sequence, passed
   in as `special`), or its decoded text begins with whitespace or ASCII
   punctuation, other than a leading digit (already excluded by the run).
   A letter ("8th") or any other character gives `format: bad_terminator`.
6. correct iff the run's digits equal str(Z) exactly. A leading zero or any
   other value is `wrong_value`.

Whitespace inside tokens is not stripped except where stated: the first token
must be exactly " ", and the terminator is judged on its first character.
"""
from __future__ import annotations

import string

PUNCT = set(string.punctuation)


def classify_symbol(cont: list[int], want_first: int, decode, special: frozenset = frozenset()) -> dict:
    if not cont:
        return {"correct": False, "reason": "empty"}
    if cont[0] != want_first:
        return {"correct": False, "reason": "wrong_first_token", "first": decode(cont[0])}
    if len(cont) < 2:
        return {"correct": False, "reason": "truncated"}
    if cont[1] in special:
        return {"correct": True, "reason": "ok"}
    nxt = decode(cont[1])
    if nxt[:1] != "" and nxt[:1] in string.ascii_lowercase:
        return {"correct": False, "reason": "symbol_continues", "next": nxt}
    return {"correct": True, "reason": "ok"}


def classify_number(cont: list[int], z: int, space_id: int, decode, special: frozenset = frozenset()) -> dict:
    if not cont or cont[0] != space_id:
        return {"correct": False, "reason": "format: first_token_not_space",
                "first": decode(cont[0]) if cont else None}
    digits, i = "", 1
    while i < len(cont):
        t = decode(cont[i])
        if t and all(ch in string.digits for ch in t):
            digits += t
            i += 1
        else:
            break
    if not digits:
        return {"correct": False, "reason": "format: no_digits", "next": decode(cont[1]) if len(cont) > 1 else None}
    if i >= len(cont):
        return {"correct": False, "reason": "truncated", "digits": digits}
    term = decode(cont[i])
    c0 = term[:1]
    if cont[i] not in special and not (c0.isspace() or c0 in PUNCT):
        return {"correct": False, "reason": "format: bad_terminator", "digits": digits, "terminator": term}
    if digits != str(z):
        return {"correct": False, "reason": "wrong_value", "digits": digits}
    return {"correct": True, "reason": "ok", "digits": digits}
