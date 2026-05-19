# Cross-engine Shannon scoring

Three parallel scorers measure next-token bits on the same corpus through
independent forward passes. Compare bits/token and bits/char as an
end-to-end correctness check on LARQL Rust against HF/PyTorch and MLX
references.

## One-command wrapper

`larql shannon verify` orchestrates all three scorers, normalizes CRLF,
prints a delta table, and exits non-zero if any pair-wise delta exceeds
`--threshold`:

```bash
larql shannon verify google/gemma-3-4b-it \
    --corpus data/gutenberg/frankenstein.txt --bytes 1024 \
    --context 512 --stride 256 --threshold 0.5
```

Use `--engines mlx,hf` (default), `--engines hf`, or `--engines mlx` to
control which references run alongside the LARQL Rust path. LARQL Rust
always runs. The Python interpreter and script paths can be overridden
with `--python`, `--mlx-script`, `--hf-script` if the layout changes.

## Individual scripts

Use these directly when you want a single engine's number without the
verify wrapper.

| Script | Engine | Notes |
|---|---|---|
| `larql shannon score MODEL --corpus FILE` | LARQL Rust F32 (raw safetensors) | Built-in CLI; see `crates/larql-cli/src/commands/primary/shannon_cmd.rs` |
| `python scripts/shannon_score_mlx.py MODEL --corpus FILE [--json]` | MLX F32 (cast from bf16/fp16) | Requires `mlx_lm` |
| `python scripts/shannon_score_hf.py MODEL --corpus FILE [--json]` | HF transformers F32 (PyTorch, CPU recommended) | Requires `torch` |

`--json` appends a `RESULT {...}` line that `larql shannon verify` parses
to consume the result; the human-readable output is unchanged.

All three implement the same sliding-window scoring used by
`score_token_range` in `shannon_cmd.rs`: `context`-sized chunks, `stride`
newly-scored targets per chunk, summing `-log2(p[target])`.

## Gotchas — these will silently corrupt cross-engine comparisons

**1. CRLF normalization.** Python's `Path.read_text()` converts `\r\n` → `\n`
silently; LARQL Rust does not. On a 1KB Gutenberg corpus, this is a ~27-byte
delta and an 8-token tokenization divergence. Always preprocess:

```bash
tr -d '\r' < raw_corpus.txt > corpus.txt
```

**2. BOS handling.** LARQL's `encode_prompt(..., add_special_tokens=true)`
plus its `maybe_prepend_bos` shim matches HF's default
`tokenizer(text, add_special_tokens=True)` and MLX's `tok.encode(text)`
default. Do **not** strip BOS in the Python scripts — the scorers already
do the right thing. (Llama-3.2 prepends `<|begin_of_text|>`; Gemma prepends
`<bos>`; SmolLM2 has no auto-BOS.)

**3. F32 cast.** Both Python scorers cast to `float32` on load. Bf16/fp16
would introduce its own quantization noise that doesn't belong in a
correctness check of the LARQL Rust F32 path.

## Recommended invocation order

```bash
# Prep corpus (strip CRLF).
tr -d '\r' < raw.txt > corpus.txt

# Same args across all three engines.
CORPUS=corpus.txt
MODEL=google/gemma-3-4b-it  # or any HF model id LARQL handles

cargo run --release -p larql-cli -- shannon score $MODEL --corpus $CORPUS \
    --context 512 --stride 256
python scripts/shannon_score_mlx.py $MODEL --corpus $CORPUS \
    --context 512 --stride 256
python scripts/shannon_score_hf.py  $MODEL --corpus $CORPUS \
    --context 512 --stride 256 --device cpu
```

Compare `total bits`. HF and MLX should agree to within ~0.1% — they are
mutually validating the F32 reference. LARQL Rust deltas above ~0.5% are
worth investigating.

## What this instrument has found

First serious application (2026-05-15) on a 997-char Frankenstein corpus:

| Model | Layer pattern | LARQL Rust vs HF F32 |
|---|---|---:|
| SmolLM2-135M | 30 std layers, no SWA | <0.01% |
| Llama-3.2-1B | 16 std layers, no SWA | +0.59% |
| Gemma 3 4B | 28 sliding + 6 global | +5.4% |
| Mistral 7B v0.1 | 32 all-sliding | +8.2% |

The architecture-scaling pattern is consistent with a sliding-window
attention implementation drift in the LARQL Rust forward path. Treat
LARQL-Rust-routed *absolute* probability claims on SWA architectures as
having a several-percent error bar until the gap is closed; rankings
(top-1, top-k) are likely preserved.
