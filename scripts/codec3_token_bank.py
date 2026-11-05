#!/usr/bin/env python3
"""CONTINUATION-CODEC-3's held-out token bank (bank 3), built by a rule fixed
and committed before the text was fetched (docs/continuation-codec-3-
reconnaissance.md: bank 1 is calibration only; bank 2 is CODEC-2's and
stays sealed; the verdict is taken on a text no arm has run on).

Selection rule (written without having seen the text's bytes)
  George Gissing, *New Grub Street* (public domain), Project Gutenberg eBook
  #1709, plain-text UTF-8 from SOURCE_URL. Same register as banks 1 and 2
  (19th-century English narrative prose); a different author and work from
  both (Austen, Trollope). The fetched file's sha256 is recorded in the
  bank; Gutenberg updates files in place, so the IDs, not the file, are the
  authority.

  1. Body = the lines strictly between the first line starting with
     START_MARKER and the first later line starting with END_MARKER. Either
     marker missing: refuse.
  2. A paragraph is a run of non-blank lines; each line is stripped and the
     lines are joined with single spaces.
  3. The text starts at the LAST paragraph that is a chapter-one heading
     (HEADING with numeral I or 1), so a table of contents that also lists
     chapter one cannot start it. None: refuse.
  4. From there, a paragraph is DROPPED if it is a chapter heading (HEADING,
     any numeral) or contains no lowercase letter (titles, volume markers,
     ornaments). Every other paragraph is kept, in order, verbatim under 2.
  5. Paragraphs are separated by one blank line; the hashed passage has no
     trailing newline. The passage is the shortest prefix of kept
     paragraphs whose tokenisation reaches BANK_LEN IDs; the bank is its
     first BANK_LEN IDs. Fewer kept paragraphs than that: refuse.

  A refusal is recorded and the rule amended in a new commit BEFORE any
  tokenisation; the start is never chosen by reading the prose.

Tokenisation, the cross-container identity check and the unknown-token
check are CODEC-2's, unchanged (imported). `--verify-bank1` is the builder's
control: bank 1's committed passage must reproduce bank 1's IDs and digest.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec2_token_bank import (  # noqa: E402
    BANK_LEN,
    encode,
    ids_digest,
    load_tokenizer,
    sha256,
    shortest_passage,
    unknown_count,
    verify_bank1,
)

SOURCE_URL = "https://www.gutenberg.org/cache/epub/1709/pg1709.txt"
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"
HEADING = re.compile(r"^CHAPTER\s+([IVXLC]+|\d+)\b", re.IGNORECASE)
CHAPTER_ONE = {"I", "1"}
FORECASTS = Path(__file__).resolve().parent.parent / "docs/represent/forecasts"
BANK3 = FORECASTS / "continuation-codec-3-token-bank.json"
BANK3_PASSAGE = FORECASTS / "continuation-codec-3-passage.txt"


def raw_paragraphs(lines: list[str]) -> list[str]:
    """Rule 2: runs of non-blank lines, stripped and joined with spaces."""
    out, current = [], []
    for line in lines + [""]:
        if line.strip():
            current.append(line.strip())
        elif current:
            out.append(" ".join(current))
            current = []
    return out


def paragraphs(text: str) -> list[str]:
    """The kept body paragraphs under rules 1-4."""
    lines = text.replace("\r\n", "\n").split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith(START_MARKER)), None)
    if start is None:
        raise SystemExit("REFUSED (rule 1): no Gutenberg START marker")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith(END_MARKER)), None
    )
    if end is None:
        raise SystemExit("REFUSED (rule 1): no Gutenberg END marker after START")
    paras = raw_paragraphs(lines[start + 1 : end])
    heading = [HEADING.match(p) for p in paras]
    ones = [i for i, m in enumerate(heading) if m and m.group(1).upper() in CHAPTER_ONE]
    if not ones:
        raise SystemExit("REFUSED (rule 3): no chapter-one heading")
    return [
        p
        for p, m in zip(paras[ones[-1] :], heading[ones[-1] :])
        if not m and any(c.islower() for c in p)
    ]


def build(source: Path, containers: list[Path], fetched: str) -> None:
    raw = source.read_bytes()
    paras = paragraphs(raw.decode("utf-8"))
    tokenizers = [load_tokenizer(c) for c in containers]
    passage, count = shortest_passage(paras, tokenizers[0])
    encodings = [encode(t, passage) for t in tokenizers]
    if any(e != encodings[0] for e in encodings[1:]):
        raise SystemExit("containers tokenise the passage differently")
    ids = encodings[0][:BANK_LEN]
    unknown = unknown_count(tokenizers[0], ids)
    if unknown:
        raise SystemExit(f"{unknown} unknown tokens in the bank")
    tok_files = {
        name: sha256((containers[0] / name).read_bytes())
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    for c in containers[1:]:
        for name, digest in tok_files.items():
            if sha256((c / name).read_bytes()) != digest:
                raise SystemExit(f"{c.name}/{name} differs from {containers[0].name}")
    BANK3_PASSAGE.write_text(passage, encoding="utf-8")
    bank = {
        "authority": "CONTINUATION-CODEC-3's held-out token sequence (bank 3). These IDs, not a runtime tokenisation of the passage, are what every arm consumes; the passage is the human-readable source.",
        "rule": "scripts/codec3_token_bank.py (selection rule in its docstring, committed before the text was fetched)",
        "source": f"George Gissing, New Grub Street (public domain), Project Gutenberg eBook #1709 ({SOURCE_URL}), fetched {fetched} (file sha256 {sha256(raw)}; Gutenberg updates files in place, so the file is not the authority)",
        "passage": f"continuation-codec-3-passage.txt: the first {count} kept paragraphs from the last chapter-one heading, headings and paragraphs without a lowercase letter dropped, each paragraph's lines joined with single spaces, paragraphs separated by one blank line, no trailing newline",
        "passage_sha256": sha256(passage.encode("utf-8")),
        "tokenizer": f"the container tokenizer, byte-identical in {', '.join(c.name for c in containers)} (tokenizer.json sha256 {tok_files['tokenizer.json']}, tokenizer_config.json sha256 {tok_files['tokenizer_config.json']}); encoded with add_special_tokens, so position 0 is <bos> (id {ids[0]}); every container produces identical IDs",
        "count": len(ids),
        "why_8193": "the 8,192-position rung feeds IDs 0..8191; ID 8192 is the next token scored at the rung's last decode position. Shorter rungs use prefixes of the same IDs",
        "ids_sha256_u32le": ids_digest(ids),
        "unknown_tokens": unknown,
        "disjoint_from_banks_1_and_2": "different authors and works: bank 1 is Pride and Prejudice (continuation-codec-1-token-bank.json), bank 2 The Warden (continuation-codec-2-token-bank.json)",
        "ids": ids,
    }
    BANK3.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    print(f"bank 3: {count} paragraphs, {len(ids)} IDs, digest {bank['ids_sha256_u32le']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--container", type=Path, action="append", required=True,
                    help="a VINDEX3 container whose tokenizer.json to use (give every arm's)")
    ap.add_argument("--verify-bank1", action="store_true")
    ap.add_argument("--source", type=Path, help="the fetched Gutenberg #1709 plain-text file")
    ap.add_argument("--fetched", help="fetch date, recorded in the bank")
    args = ap.parse_args()
    verify_bank1(args.container)
    if args.verify_bank1:
        return
    if not (args.source and args.fetched):
        sys.exit("--source and --fetched are required to build bank 3")
    build(args.source, args.container, args.fetched)


if __name__ == "__main__":
    main()
