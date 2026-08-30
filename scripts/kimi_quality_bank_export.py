#!/usr/bin/env python3
"""Q2 — the TEACHER-FORCED quality bank's sequence export.

Many SHORT, INDEPENDENT sequences rather than one long trajectory. Two
reasons, and both are about what the bank is able to attribute:

* **Independence.** Each sequence starts from clean recurrent state, so
  a routing change in one cannot contaminate the next. A single long
  trajectory would let one early flip dominate every later position.
* **Union size.** The routed-expert union grows with positions. Measured
  on the real 19-position fixture: 65 experts a layer, against 116 if
  routing were uniform — real routing is concentrated, and the marginal
  decays to 1-3 new experts a position. One 32-position sequence needs
  ~79 experts a layer, about 29 GB of BF16 expert weight. A single
  4000-position trajectory would need essentially all 256, i.e. the
  whole checkpoint.

This script writes only TOKENS and their EMBEDDING ROWS — no weights and
no reference logits. The baseline arm is LARQL's own BF16 stack, which
is already gated byte-identical against `modeling_kimi.py` over the
16-token trajectory, so there is nothing for a Python reference to add
here and a great deal of export cost to avoid.

Sequences come from real prose (`data/gutenberg`), not random token ids:
a bank built on off-manifold input measures a region of the model that
deployment never visits.

    python scripts/kimi_quality_bank_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
        --sequences 256 --positions 32 --out /tmp/kimi_quality_bank
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import kimi_moe_export

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO / "data" / "gutenberg"


def load_tokenizer(checkpoint: Path):
    """Kimi ships its own tokenizer class beside the weights."""
    sys.path.insert(0, str(checkpoint))
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)


def corpus_text(corpus: Path) -> str:
    files = sorted(corpus.glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt under {corpus}")
    return "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--sequences", type=int, default=256)
    ap.add_argument("--positions", type=int, default=32)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="token offset of sequence 0 — a HELD-OUT bank uses the same "
        "stride with a start that lands every window in the gaps between "
        "the selection bank's windows, so the two banks never share a "
        "position while sampling the same corpus distribution",
    )
    ap.add_argument("--stride", type=int, default=0,
                    help="tokens between sequence starts, so they come from genuinely "
                         "different passages rather than adjacent windows. 0 spreads "
                         "them evenly over the whole corpus, which is what you want "
                         "unless you are deliberately sampling one region.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(args.checkpoint)
    ids = tok.encode(corpus_text(args.corpus))
    stride = args.stride or (len(ids) - args.positions) // max(args.sequences - 1, 1)
    if stride < args.positions:
        raise SystemExit(
            f"corpus gives {len(ids)} tokens, so {args.sequences} sequences would sit "
            f"{stride} tokens apart while each is {args.positions} long — they would "
            f"OVERLAP, and overlapping sequences are not independent samples. Add text "
            f"or lower --sequences/--positions."
        )
    print(
        f"[bank] corpus tokenised: {len(ids)} tokens; stride {stride} "
        f"({stride / args.positions:.1f}x the sequence length)",
        flush=True,
    )

    embed = kimi_moe_export.read_tensors(
        args.checkpoint, {"model.embed_tokens.weight"}
    )["model.embed_tokens.weight"]
    hidden = embed.shape[1]
    print(f"[bank] embedding table {tuple(embed.shape)}", flush=True)

    sequences = []
    last_needed = args.start + (args.sequences - 1) * stride + args.positions
    if last_needed > len(ids):
        raise SystemExit(
            f"--start {args.start} pushes the last window to {last_needed} tokens "
            f"but the corpus holds {len(ids)}"
        )
    for s in range(args.sequences):
        start = args.start + s * stride
        seq = ids[start : start + args.positions]
        sequences.append(seq)
        rows = embed[torch.tensor(seq, dtype=torch.long)].to(torch.float32)
        # One file per sequence: the runner streams them, and a bank of
        # 256 x 32 x 2304 f32 is only ~75 MB in total.
        rows.numpy().astype("<f4").tofile(args.out / f"seq_{s}.f32")

    manifest = {
        "sequences": args.sequences,
        "positions": args.positions,
        "hidden": hidden,
        "vocab_size": int(embed.shape[0]),
        "stride": stride,
        # Only a held-out bank carries a start: the canonical bank's
        # manifest bytes predate this knob, and its content-hash identity
        # must not change under regeneration.
        **({"start": args.start} if args.start else {}),
        "corpus": str(args.corpus),
        "token_ids": sequences,
        # Teacher forcing is a property of the BANK, not of a runner
        # flag: both arms are fed these exact tokens at every position,
        # and neither ever consumes its own output. Recorded so a reader
        # of the fixture cannot mistake it for a free-running trace.
        "regime": "teacher-forced",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[bank] wrote {args.sequences} sequences x {args.positions} positions "
        f"= {args.sequences * args.positions} evaluated positions -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
