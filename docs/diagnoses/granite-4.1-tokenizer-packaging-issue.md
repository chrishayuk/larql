<!--
Draft GitHub issue text for an ibm-granite tracker (or a Hugging Face
"Community" discussion on the model repo). Body assumes a markdown-aware
renderer. The full diagnosis with measurement context lives in
docs/diagnoses/granite-4.1-tokenizer-packaging-bug.md — this file is the
copy-pasteable short form.

Suggested target:
- https://huggingface.co/ibm-granite/granite-4.1-3b → "Community" tab → New discussion
  (and the same on -8b / -30b / each -base variant)
- or whichever public IBM Granite issue tracker is canonical
-->

# `tokenizer.json` ships a cl100k-style pre-tokenizer regex but `tokenizer_config.json` declares `GPT2Tokenizer`

## Summary

In `ibm-granite/granite-4.1-3b` (and the rest of the 4.1 family — see
checklist below), `tokenizer.json` and `tokenizer_config.json` declare
**incompatible** tokenizers. Consumers that load `tokenizer.json`
directly (the `tokenizers` crate, `PreTrainedTokenizerFast(tokenizer_file=...)`,
vLLM's fast tokenizer path, llama.cpp's fast ingest, etc.) produce a
different token stream than `transformers.AutoTokenizer.from_pretrained(...)`,
which goes through the slow `GPT2Tokenizer` declared in
`tokenizer_config.json`.

`AutoTokenizer` is the training-time tokenizer — that's what the model
weights expect. So consumers using `tokenizer.json` feed the model
out-of-distribution token sequences. The model still loads, still runs,
still emits plausible-looking probabilities — they're just on the wrong
tokenization grid, so any log-likelihood, perplexity, KV-cache
footprint, or generation done off `tokenizer.json` is silently wrong.

## Where the regexes disagree

`tokenizer_config.json`:
```json
{ "tokenizer_class": "GPT2Tokenizer", ... }
```
This routes through HF's classic `GPT2Tokenizer` with regex
`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.

`tokenizer.json` → `pre_tokenizer.pretokenizers[0].pattern.Regex`:
```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```
This is the **cl100k_base** (GPT-4 / Llama 3) pre-tokenizer, not the
GPT-2 one. Same vocab + merges, different splitting.

Two visible consequences:

- `"re-use"` →
  - GPT-2: `["re", "-", "use"]` (3 tokens)
  - cl100k: `["re", "-use"]` (2 tokens — the leading non-letter is eaten into the word)
- `"\n    \n"` →
  - GPT-2: `["\n", "    ", "\n"]` (3 tokens)
  - cl100k: `["Ċ    Ċ"]` (1 token — mixed whitespace merged)

## Reproduction

```python
from pathlib import Path
from transformers import AutoTokenizer
from tokenizers import Tokenizer

REPO = "ibm-granite/granite-4.1-3b"

# Any moderately long ASCII text exercises the divergence;
# a 1 KB snippet of public-domain Frankenstein works well.
text = """The Project Gutenberg eBook of Frankenstein; or, the modern prometheus
    
This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online
at www.gutenberg.org."""

hf   = AutoTokenizer.from_pretrained(REPO)            # honours tokenizer_config.json
fast = Tokenizer.from_pretrained(REPO)                # uses tokenizer.json

hf_ids   = hf(text, add_special_tokens=True)["input_ids"]
fast_ids = fast.encode(text, add_special_tokens=True).ids

print(f"AutoTokenizer (GPT2Tokenizer regex): {len(hf_ids)} tokens")
print(f"tokenizer.json (cl100k regex):       {len(fast_ids)} tokens")
print(f"agree on every id? {hf_ids == fast_ids}")
```

Expected output (3B snapshot, current as of writing):
```
AutoTokenizer (GPT2Tokenizer regex): 161 tokens
tokenizer.json (cl100k regex):       148 tokens
agree on every id? False
```

Even `transformers.PreTrainedTokenizerFast(tokenizer_file=...)` produces
the cl100k counts — the shape of `tokenizer.json` itself is the
divergent surface, not the consumer.

## Measured impact

In a three-engine bits/char comparison (HF/PyTorch on CPU, Apple MLX,
and a third-party Rust forward path loading `tokenizer.json` via the
`tokenizers` crate) on a 1 KB Frankenstein excerpt:

| Engine | Tokenization source | Tokens | bits/char |
|---|---|---:|---:|
| HF / PyTorch (F32) | `AutoTokenizer` (GPT-2 regex) | 263 | 1.0378 |
| MLX | `AutoTokenizer` (GPT-2 regex) | 263 | 1.0378 |
| Rust forward via `tokenizer.json` | cl100k regex | 243 | 0.5975 |

HF and MLX agree to five decimals. The Rust path is off by 42 % — and
its forward pass is bit-identical to HF/MLX **given the same input**,
which confirms the gap is on the tokenizer, not on inference. Patching
`tokenizer.json`'s first pre-tokenizer regex back to the GPT-2 form
closes the gap to **0.000 %**.

## Suggested fix

Replace `tokenizer.json`'s
`pre_tokenizer.pretokenizers[0].pattern.Regex` with the classic GPT-2
pattern; leave the ByteLevel sub-tokenizer, the BPE model, and the
added-tokens list untouched. After the swap,
`tokenizers.Tokenizer.from_pretrained(...)` and
`transformers.AutoTokenizer.from_pretrained(...)` produce identical
token streams.

```python
import json
from pathlib import Path

path = Path("tokenizer.json")
tj = json.loads(path.read_text())
tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
path.write_text(json.dumps(tj, ensure_ascii=False))
```

Alternative — if the cl100k pre-tokenizer is the *intended* training-
time tokenizer for some future Granite revision: keep `tokenizer.json`
as-is and update `tokenizer_config.json`'s `tokenizer_class` to
`"PreTrainedTokenizerFast"`, then publish a model snapshot trained
against that regex. The two files have to agree.

## Affected repos

Confirmed empirically (cl100k regex in `tokenizer.json` +
`GPT2Tokenizer` declared in `tokenizer_config.json`):
- [x] `ibm-granite/granite-4.1-3b`
- [x] `ibm-granite/granite-4.1-8b`
- [x] `ibm-granite/granite-4.1-30b`

Inferred (same `tokenizer_config.json` / `tokenizer.json` packaging
pattern shared with the chat-tuned variant — please verify with the
snippet above):
- [ ] `ibm-granite/granite-4.1-3b-base`
- [ ] `ibm-granite/granite-4.1-8b-base`
- [ ] `ibm-granite/granite-4.1-30b-base`

(`ibm-granite/granite-4.0-micro` and the 3.x family use the same
`GPT2Tokenizer` declaration but ship a `tokenizer.json` with the
matching GPT-2 regex — so 4.1 is the regression.)

## Why it's silent

`AutoTokenizer.from_pretrained` is the dominant entry point on the
Python side, and it follows `tokenizer_class` rather than preferring
the fast tokenizer when the two disagree. So Python users running
`transformers` see the correct tokenization and never notice the
mismatch. Non-`transformers` consumers — anything reading
`tokenizer.json` as the source of truth — see broken inputs to a model
that nonetheless keeps running, with no error and no easy way to
distinguish "model is bad" from "tokenizer is bad" without a reference
scorer.
