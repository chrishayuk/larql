# IBM Granite 4.1 — `tokenizer.json` packaging bug

Status: **diagnosed**, worked around in LARQL at
`crates/larql-inference/src/tokenizer.rs::maybe_patch_gpt2_pretok`.
Upstream fix should land in the model repos on Hugging Face. Last
updated 2026-05-17.

Affected repos:

- [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b)
- [ibm-granite/granite-4.1-3b-base](https://huggingface.co/ibm-granite/granite-4.1-3b-base)
- [ibm-granite/granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b)
- [ibm-granite/granite-4.1-8b-base](https://huggingface.co/ibm-granite/granite-4.1-8b-base)
- [ibm-granite/granite-4.1-30b](https://huggingface.co/ibm-granite/granite-4.1-30b)
- [ibm-granite/granite-4.1-30b-base](https://huggingface.co/ibm-granite/granite-4.1-30b-base)

(3B, 8B, 30B all **confirmed empirically** by inspecting
`tokenizer.json::pre_tokenizer.pretokenizers[0].pattern.Regex` and
`tokenizer_config.json::tokenizer_class` on each snapshot, 2026-05-17.
All three carry the cl100k regex in `tokenizer.json` and declare
`GPT2Tokenizer` in `tokenizer_config.json`. The `-base` variants
inherit the same tokenizer artefacts and are presumed affected;
please verify with the snippet below.)

## TL;DR

The shipped `tokenizer.json` and `tokenizer_config.json` declare
**incompatible** tokenizers:

| File | Source of truth | Pre-tokenization regex |
|---|---|---|
| `tokenizer_config.json` | `"tokenizer_class": "GPT2Tokenizer"` | Classic GPT-2: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` |
| `tokenizer.json` | `pre_tokenizer.pretokenizers[0].pattern.Regex` | **cl100k_base** (GPT-4 / Llama 3 style): `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+` |

Both files share the same vocab + merges (`vocab.json` + `merges.txt`,
also embedded inside `tokenizer.json`), but the two pre-tokenizer
regexes split text **differently**, so they produce different token
streams from the same input.

`transformers.AutoTokenizer.from_pretrained(...)` honours
`tokenizer_class`, falls back to the slow `GPT2Tokenizer`, and uses
the GPT-2 regex — **this matches the model's training-time
tokenizer**. Any consumer that loads `tokenizer.json` directly (the
`tokenizers` crate, `PreTrainedTokenizerFast(tokenizer_file=...)`,
vLLM's fast path, llama.cpp's fast tokenizer ingest, …) will use the
cl100k regex and feed the model out-of-distribution tokens.

## Repro

The Frankenstein header used by LARQL's shannon-verify gate
(`tests/fixtures/shannon_frankenstein_2k.txt`, truncated to 1 KB)
exposes the divergence cleanly:

```python
SNAP = "ibm-granite/granite-4.1-3b"  # or any local snapshot path
from transformers import AutoTokenizer
from tokenizers import Tokenizer

text = open("tests/fixtures/shannon_frankenstein_2k.txt").read()[:1024]
text = text.replace("\r\n", "\n").replace("\r", "\n")

hf   = AutoTokenizer.from_pretrained(SNAP)              # uses tokenizer_config.json
fast = Tokenizer.from_file(f"{SNAP}/tokenizer.json")    # uses tokenizer.json

hf_ids   = hf(text, add_special_tokens=True)["input_ids"]
fast_ids = fast.encode(text, add_special_tokens=True).ids

print(f"hf   (GPT2Tokenizer regex): {len(hf_ids)} tokens")  # 264
print(f"fast (cl100k regex):        {len(fast_ids)} tokens")  # 244
```

Even `PreTrainedTokenizerFast(tokenizer_file=...)` produces 244 — the
shape of `tokenizer.json` itself is what drifts, not the consumer.

## What actually diverges

Two patterns dominate. Both come from the cl100k regex being more
greedy about absorbing adjacent characters into letter / whitespace
pretokens than the GPT-2 regex:

1. **Letter words preceded by punctuation.** GPT-2 splits `"re-use"`
   into `["re", "-", "use"]`. The cl100k regex's
   `[^\r\n\p{L}\p{N}]?\p{L}+` arm eats one leading non-letter into
   the word, producing `["re", "-use"]`.

   ```
   text: "...away or re-use..."
   GPT-2 : (12, '-'),     (817, 'use')        # 2 tokens
   cl100k: (25700, '-use')                    # 1 token
   ```

2. **Mixed-whitespace runs.** GPT-2's `\s+(?!\S)|\s+` arms keep
   `\n`, runs of spaces, and `\n` again as separate pretokens. The
   cl100k regex's `\s*[\r\n]+` arm merges them into one.

   ```
   text: "...modern prometheus\n    \nThis eBook..."
   GPT-2 : (198, '\n'), (257, '    '), (198, '\n')   # 3 tokens
   cl100k: (7361, 'Ċ    Ċ')                          # 1 token
   ```

Across the 1 KB Frankenstein header the net effect is **264 vs 244
tokens** (~7.6 % fewer with the cl100k regex). The first 8 ids agree
(both yield `[791, 5907, 52686, 58610, 315, 9454, 62756, 26]` for
*"The Project Gutenberg eBook of Frankenstein;"*), so the divergence
is invisible on short ASCII-letter-only prompts and surfaces as the
input length grows.

## Impact on inference

LARQL's three-engine shannon-verify gate showed the cost cleanly. The
LARQL Rust forward path is correct *given the input* — but the input
differs from what HF/MLX feed the same model:

| Engine | Tokenization source | Tokens | bits/char | Δ vs HF |
|---|---|---:|---:|---:|
| LARQL Rust (pre-fix) | `tokenizer.json` (cl100k) | 243 | 0.5975 | −42.43 % |
| MLX | HF AutoTokenizer (GPT-2) | 263 | 1.0378 | −0.000 % |
| HF / PyTorch | HF AutoTokenizer (GPT-2) | 263 | 1.0378 | — |

bits/char is supposed to be tokenization-invariant (the total
information content of the text is fixed). It isn't here because the
two tokenizations carve up different units. The merged tokens
(`Ċ    Ċ`, `-use`) are vocabulary entries the model assigns a much
higher probability to than the GPT-2 split forms, so the cl100k
stream sits in a lower-entropy regime of the model than it was
trained for. From a downstream consumer's perspective: any
log-likelihood, perplexity, or KV-cache footprint computed off the
shipped `tokenizer.json` is wrong.

A consumer with no reference scorer would have no signal that
anything is off; **the model still loads, still runs, still emits
plausible-looking probabilities** — they're just on the wrong
tokenization grid. Generation quality degrades silently (out-of-
distribution token sequences feeding the model).

## Suggested upstream fix

Replace `tokenizer.json`'s
`pre_tokenizer.pretokenizers[0].pattern.Regex` with the classic GPT-2
pattern, and leave the ByteLevel sub-tokenizer and everything else
untouched. After the swap, `tokenizers::Tokenizer.from_file(...)` and
`AutoTokenizer.from_pretrained(...)` produce identical token streams:

```python
import json
tj = json.load(open(f"{SNAP}/tokenizer.json"))
tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
json.dump(tj, open(f"{SNAP}/tokenizer.json", "w"))
```

Alternative if IBM intentionally wants the cl100k pre-tokenizer
(e.g. a future retraining with `\p{N}{1,3}` digit chunking and the
greedy letter arm): keep `tokenizer.json` as-is and change
`tokenizer_config.json`'s `tokenizer_class` to
`"PreTrainedTokenizerFast"`, then publish a new model snapshot
trained against that regex. The two files have to agree.

## LARQL workaround

`crates/larql-inference/src/tokenizer.rs::load_tokenizer` reads
`tokenizer_config.json` next to `tokenizer.json`; when the config
declares `tokenizer_class: "GPT2Tokenizer"`, it rewrites
`pre_tokenizer.pretokenizers[0].pattern.Regex` to the GPT-2 form in
memory before constructing the `tokenizers::Tokenizer`. Only the
first pre-tokenizer leaf is touched — the ByteLevel sibling, the BPE
model, the added tokens, and everything else round-trip untouched.

Unit tests pin the rewrite shape:

- `maybe_patch_gpt2_pretok_replaces_first_regex` — JSON in / JSON
  out, regex slot is now the GPT-2 pattern, ByteLevel sibling
  untouched.
- `maybe_patch_gpt2_pretok_returns_none_for_unexpected_shape` — a
  Llama-style Metaspace pre-tokenizer is left alone (we'd otherwise
  silently corrupt any future model with the same
  `tokenizer_class: GPT2Tokenizer` declaration but a non-Split
  pre-tokenizer).

End-to-end verification: `larql shannon verify
ibm-granite/granite-4.1-3b` now passes at **0.000 %** delta against
both HF and MLX on the 1 KB Frankenstein corpus.

## Related: original GPT-2

The original `gpt2` model on HF predates `tokenizer.json` entirely
(its `tokenizer_class` is also `GPT2Tokenizer`, vocab/merges live in
`vocab.json`/`merges.txt`). Granite 4.1 reuses the GPT-2 BPE
vocabulary class but ships an extra `tokenizer.json` with a regex
that doesn't match — this is the kind of regression that can happen
when a `tokenizer.json` is regenerated from a different reference
codebase (cl100k templates are common starting points). The
`tokenizer_class` field is the authoritative declaration of the
training-time tokenizer.
