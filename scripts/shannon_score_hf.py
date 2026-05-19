#!/usr/bin/env python3
"""HF/PyTorch parallel to `larql shannon score`.

Third independent reference alongside `shannon_score_mlx.py` and the LARQL
Rust `shannon score` CLI. Mirrors the windowing in
crates/larql-cli/src/commands/primary/shannon_cmd.rs (score_token_range):
score targets 1..N with a sliding window of `context` tokens, `stride`
newly-scored targets per window.

Notes on apples-to-apples comparison:
- Normalize CRLF→LF on the corpus before running (LARQL Rust keeps CRLFs,
  Python text I/O does not — they disagree on token counts otherwise).
- Match LARQL's `encode_prompt(..., add_special_tokens=True)`: do NOT strip
  BOS. The HF tokenizer's default add_special_tokens=True replicates the
  Rust path's BOS handling.
- Cast model to F32 to compare against larql's F32 forward. F16/BF16 would
  introduce its own quantization noise.
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

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def score(model_id: str, corpus_path: Path, context: int, stride: int, device: str,
          emit_json: bool = False):
    text = corpus_path.read_text(encoding="utf-8")
    n_bytes = len(text.encode("utf-8"))
    n_chars = len(text)

    print(f"loading {model_id} on {device}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    torch_device = torch.device(device)
    model.to(torch_device)

    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
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
    with torch.no_grad():
        while target_start < n_tokens:
            target_end = min(target_start + stride, n_tokens)
            prefix_start = min(max(target_end - context, 0), target_start - 1)
            chunk = ids[prefix_start:target_end]
            chunk_t = torch.tensor([chunk], device=torch_device)
            logits = model(chunk_t).logits[0].float()  # (chunk_len, vocab)
            log_probs = F.log_softmax(logits, dim=-1)

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
            "engine": "hf",
            "model": model_id,
            "tokens_scored": n_scored,
            "chars": n_chars,
            "bytes": n_bytes,
            "total_bits": total_bits,
            "bits_per_token": total_bits / max(n_scored, 1),
            "bits_per_char": total_bits / max(n_chars, 1),
        }))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--device", default="cpu",
                   help="cpu (default, deterministic) or mps (faster)")
    p.add_argument("--json", action="store_true",
                   help="emit a final 'RESULT {...}' JSON line for tool consumers")
    args = p.parse_args()
    score(args.model, args.corpus, args.context, args.stride, args.device, args.json)


if __name__ == "__main__":
    main()
