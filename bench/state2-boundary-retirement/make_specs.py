#!/usr/bin/env python3
"""STATE-2 corpus: tokenized specs for the boundary-retirement harness.

Tokens are produced HERE, never in the engine (the CLI's own rule: only
one side may choose the tokenizer). Spans are found by tokenizing full
prefixes, so BPE merges across piece boundaries cannot shift them.

The queried fact lives in C1's INTERIOR: the script refuses to emit a
spec where the fact's tokens sit within `MARGIN` positions of either
edge of C1, so every retained-tail arm (k <= 16) genuinely excludes it.
The question text asks for the fact without containing it (no filler
leakage).
"""

import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

import glob

MODEL = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--ibm-granite--granite-4.2-3b/snapshots/*/"
    )
)[0]
OUT = os.path.dirname(os.path.abspath(__file__))
MARGIN = 24  # interior fact must be at least this far from both C1 edges
SERIAL = "RX-4471"

PREAMBLE = "Site maintenance log, northern sector.\n\n"

C1 = (
    "Entry 118. The inspection team reached the auxiliary coolant plant "
    "shortly after dawn and began the quarterly survey of the pumping "
    "hall. The intake screens on the eastern channel were partially "
    "fouled with sediment and were cleared by hand before any pressure "
    "readings were taken. Flow through the primary loop measured 412 "
    "litres per minute, which is within the tolerance band recorded "
    "last quarter. The auxiliary coolant pump, serial number "
    + SERIAL
    + ", showed elevated bearing temperatures during the sustained load "
    "test, peaking at 78 degrees before settling. The team fitted a "
    "replacement gasket on the outlet flange and re-torqued the mounting "
    "bolts to specification. In the control room, the operators reported "
    "intermittent faults on the second display console, which were "
    "traced to a corroded connector behind the panel and resealed. The "
    "backup generator was run for twenty minutes under load and held "
    "voltage steadily. Fuel reserves stand at roughly two thirds of "
    "capacity, and a resupply was scheduled for the end of the month. "
    "By late afternoon the survey of the pumping hall was complete, and "
    "the team logged the site as operational with two advisory notes "
    "pending review."
)

CONTINUATION = "\n\nThe following morning, the crew returned to the station and"

QUERYBACK = (
    "\n\nQuestion: What is the serial number of the auxiliary coolant "
    "pump?\nAnswer: The serial number is"
)


def spans(tok, pieces):
    """Token index after each cumulative prefix of `pieces`."""
    ends = []
    text = ""
    for piece in pieces:
        text += piece
        ends.append(len(tok(text)["input_ids"]))
    return ends, tok(text)["input_ids"]


def build(tok, label, tail):
    ends, tokens = spans(tok, [PREAMBLE, C1, tail])
    c1_start, c1_end = ends[0], ends[1]
    fact_ids = tok(SERIAL, add_special_tokens=False)["input_ids"]
    # Locate the fact inside the tokenized C1 region.
    hits = [
        i
        for i in range(c1_start, c1_end - len(fact_ids) + 1)
        if tokens[i : i + len(fact_ids)] == fact_ids
    ]
    # BPE may merge the serial with surrounding space; fall back to a
    # containment scan over decoded windows.
    if not hits:
        hits = [
            i
            for i in range(c1_start, c1_end - 1)
            if SERIAL in tok.decode(tokens[i : i + 6])
        ]
    if not hits:
        sys.exit(f"{label}: fact tokens not found inside C1")
    fact = hits[0]
    if fact - c1_start < MARGIN or c1_end - fact < MARGIN:
        sys.exit(
            f"{label}: fact at {fact} is within {MARGIN} of a C1 edge "
            f"({c1_start}..{c1_end}); rewrite the passage"
        )
    spec = {
        "label": label,
        "model": MODEL,
        "tokens": tokens,
        "c1_start": c1_start,
        "c1_end": c1_end,
        "fact_position": fact,
    }
    path = os.path.join(OUT, f"spec_{label}.json")
    with open(path, "w") as f:
        json.dump(spec, f)
    print(
        f"{label}: {len(tokens)} tokens, C1 = {c1_start}..{c1_end} "
        f"({c1_end - c1_start} positions), fact at {fact} "
        f"({c1_end - fact} before the boundary) -> {path}"
    )


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    build(tok, "continuation", CONTINUATION)
    build(tok, "queryback", QUERYBACK)
    # The harness reports token ids; keep a decode table for reading its
    # greedy output without loading the tokenizer again.
    print("bos:", tok("x")["input_ids"][:1], "vocab:", tok.vocab_size)


if __name__ == "__main__":
    main()
