#!/usr/bin/env python3
"""MLX parallel to `larql shannon score`.

Mirrors the windowing in crates/larql-cli/src/commands/primary/shannon_cmd.rs
(score_token_range): score targets 1..N with a sliding window of `context`
tokens, `stride` newly-scored targets per window. Reports bits/token and
bits/char to compare against the LARQL Rust path.

MLX is the established independent reference in this repo (see
scripts/compare_inference.py). Model parameters are cast to f32 to match
larql's f32 forward.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Must match `RESULT_PREFIX` in
# crates/larql-cli/src/commands/primary/shannon_cmd.rs. `larql shannon
# verify` greps stdout for this prefix to consume the structured result.
RESULT_PREFIX = "RESULT "

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm import load as mlx_load


def cast_f32(model):
    def to_f32(p):
        if isinstance(p, mx.array) and p.dtype != mx.float32:
            return p.astype(mx.float32)
        return p
    model.update(tree_map(to_f32, model.parameters()))
    return model


def score(model_id: str, corpus_path: Path, context: int, stride: int,
          emit_json: bool = False):
    text = corpus_path.read_text(encoding="utf-8")
    n_bytes = len(text.encode("utf-8"))
    n_chars = len(text)

    print(f"loading {model_id}...", file=sys.stderr)
    model, tok = mlx_load(model_id)
    cast_f32(model)
    mx.eval(model.parameters())

    # Match larql's encode_prompt: tokenizer.encode(prompt, add_special_tokens=True)
    # then prepend BOS only if arch has one and it's missing. mlx_lm wraps the HF
    # tokenizer; default add_special_tokens=True replicates the first step. We
    # rely on the tokenizer's own default to match larql.
    ids = tok.encode(text)
    n_tokens = len(ids)
    print(f"corpus tokenized to {n_tokens} tokens, {n_chars} chars, {n_bytes} bytes",
          file=sys.stderr)
    if n_tokens < 2:
        raise SystemExit("corpus too short")
    if stride >= context:
        raise SystemExit("--stride must be smaller than --context")

    total_bits = 0.0
    token_bits = []

    target_start = 1
    while target_start < n_tokens:
        target_end = min(target_start + stride, n_tokens)
        prefix_start = min(max(target_end - context, 0), target_start - 1)
        chunk = ids[prefix_start:target_end]
        chunk_t = mx.array(chunk)[None]
        logits = model(chunk_t)[0].astype(mx.float32)  # (chunk_len, vocab)
        # log_softmax
        m_max = mx.max(logits, axis=-1, keepdims=True)
        log_probs = (logits - m_max) - mx.log(
            mx.sum(mx.exp(logits - m_max), axis=-1, keepdims=True)
        )

        row_start = target_start - prefix_start - 1
        for offset, pos in enumerate(range(target_start, target_end)):
            target = ids[pos]
            lp = log_probs[row_start + offset, target].item()
            bits = -lp / math.log(2)
            total_bits += bits
            token_bits.append(bits)
        target_start = target_end

    n_scored = len(token_bits)
    print("done.")
    print(f"tokens scored:  {n_scored:>10}")
    print(f"bits/token:     {total_bits / max(n_scored, 1):>14.7f}")
    print(f"bits/char:      {total_bits / max(n_chars, 1):>14.7f}")
    print(f"bits/byte:      {total_bits / max(n_bytes, 1):>14.7f}")
    print(f"total bits:     {total_bits:>14.7f}")
    if emit_json:
        print(RESULT_PREFIX + json.dumps({
            "engine": "mlx",
            "model": model_id,
            "tokens_scored": n_scored,
            "chars": n_chars,
            "bytes": n_bytes,
            "total_bits": total_bits,
            "bits_per_token": total_bits / max(n_scored, 1),
            "bits_per_char": total_bits / max(n_chars, 1),
        }))
    return token_bits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--json", action="store_true",
                   help="emit a final 'RESULT {...}' JSON line for tool consumers")
    args = p.parse_args()
    score(args.model, args.corpus, args.context, args.stride, args.json)


if __name__ == "__main__":
    main()
