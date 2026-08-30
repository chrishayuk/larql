#!/usr/bin/env python3
"""HF reference oracle for the mamba2-780m witness (rung 1 of the ladder).

The witness claim is "generates correctly through the generic V3 execution
path", so the oracle captures more than a sentence: full logits at every
prefill position, per-layer hidden states, and full logits at every greedy
decode step. SSM correctness lives in the continuation state — a one-token
match can conceal a broken state update — so the corpus spans prompt lengths
(including one crossing the SSD chunk_size boundary) and the script proves
the reference is self-consistent before anything is paritied against it:

  * determinism  — a fresh second prefill must be bitwise identical;
  * step-vs-scan — feeding the prompt token-by-token through the recurrent
    cache must agree with the one-shot chunked-scan prefill. This is the
    same state update the decode loop uses, checked against the scan.

Precision is a decision, not a default: fp32 is the oracle substrate (the
parity gate scores against fp32, not the F16 the checkpoint ships in).

Usage:
  python3 scripts/mamba2_reference_oracle.py ~/chris-models/mamba2-780m-hf \
      --out ~/chris-models/oracles/mamba2-780m-hf [--steps 32]

Writes per-prompt .npz archives plus manifest.json (token ids, greedy
continuations, self-check results, library versions).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fixed corpus: short / medium / long-enough-to-cross-chunk_size (256).
PARAGRAPH = (
    "The river rises in the high mountains where snow melts slowly through "
    "the spring months, gathering small streams as it descends past granite "
    "cliffs and pine forests toward the wide valley floor below. "
)
PROMPTS = {
    "short": "The capital of France is",
    "medium": (
        "In 1969 the Apollo 11 mission carried three astronauts to the Moon, "
        "and the lunar module descended to the Sea of Tranquility, where"
    ),
    "long": PARAGRAPH * 8 + "The river finally reaches",
}


def greedy_step_ids(model, cache, logits, pos, steps):
    """Greedy decode `steps` tokens through the recurrent cache.

    Returns (ids, stepwise full logits). `logits` is the distribution the
    first generated token is drawn from.
    """
    out_ids, out_logits = [], []
    for _ in range(steps):
        next_id = int(logits.argmax())
        out_ids.append(next_id)
        step = model(
            torch.tensor([[next_id]]),
            cache_params=cache,
            use_cache=True,
            cache_position=torch.tensor([pos]),
        )
        cache = step.cache_params
        logits = step.logits[0, -1].float()
        out_logits.append(logits.numpy())
        pos += 1
    return out_ids, np.stack(out_logits)


def stepwise_prefill_logits(model, ids):
    """Feed the prompt one token at a time through the recurrent cache."""
    cache = None
    logits = None
    for pos in range(ids.shape[1]):
        out = model(
            ids[:, pos : pos + 1],
            cache_params=cache,
            use_cache=True,
            cache_position=torch.tensor([pos]),
        )
        cache = out.cache_params
        logits = out.logits[0, -1].float()
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=32)
    a = ap.parse_args()

    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(a.checkpoint).expanduser() / "config.json"
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, dtype=torch.float32, low_cpu_mem_usage=True, device_map="cpu"
    ).eval()
    torch.set_grad_enabled(False)

    manifest = {
        "checkpoint": a.checkpoint,
        "config_sha256": config_sha,
        "dtype": "float32",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "steps": a.steps,
        "prompts": {},
    }

    for name, text in PROMPTS.items():
        ids = tok(text, return_tensors="pt").input_ids
        n = ids.shape[1]
        print(f"[{name}] {n} tokens", file=sys.stderr)

        out = model(ids, use_cache=True, output_hidden_states=True)
        prefill_logits = out.logits[0].float().numpy()  # [n, vocab]
        hidden = np.stack([h[0].float().numpy() for h in out.hidden_states])

        # Self-check 1: a fresh prefill must be bitwise identical.
        again = model(ids, use_cache=True)
        deterministic = bool(torch.equal(out.logits, again.logits))

        # Self-check 2: token-by-token recurrence vs one-shot scan.
        step_logits = stepwise_prefill_logits(model, ids)
        scan_logits = out.logits[0, -1].float()
        step_vs_scan = float((step_logits - scan_logits).abs().max())
        step_vs_scan_argmax = bool(step_logits.argmax() == scan_logits.argmax())

        gen_ids, decode_logits = greedy_step_ids(
            model, out.cache_params, scan_logits, n, a.steps
        )
        gen_text = tok.decode(gen_ids)
        print(f"[{name}] -> {gen_text!r}", file=sys.stderr)
        print(
            f"[{name}] deterministic={deterministic} "
            f"step_vs_scan_max_abs={step_vs_scan:.3e} "
            f"argmax_agree={step_vs_scan_argmax}",
            file=sys.stderr,
        )

        np.savez_compressed(
            out_dir / f"{name}.npz",
            input_ids=ids[0].numpy(),
            prefill_logits=prefill_logits,
            hidden_states=hidden,
            generated_ids=np.array(gen_ids),
            decode_logits=decode_logits,
        )
        manifest["prompts"][name] = {
            "text": text,
            "input_ids": ids[0].tolist(),
            "n_tokens": n,
            "generated_ids": gen_ids,
            "generated_text": gen_text,
            "first_token_id": gen_ids[0],
            "first_token": tok.decode([gen_ids[0]]),
            "deterministic": deterministic,
            "step_vs_scan_max_abs": step_vs_scan,
            "step_vs_scan_argmax_agree": step_vs_scan_argmax,
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir}/manifest.json + {len(PROMPTS)} npz", file=sys.stderr)


main()
