# larql-kv/baselines

Multi-engine accuracy snapshots from `larql accuracy`. Each baseline is a
pair: `<model>-<date>.md` (commentary + headline table + regression
thresholds) and `<model>-<date>.json` (per-prompt raw scores).

Why per-crate (not under `bench/baselines/`): `bench/baselines/cpu/` and
`bench/baselines/cross-arch/` measure the workspace end-to-end (us vs
`llama.cpp`, cross-architecture sweeps). These baselines measure
**`larql-kv` engine correctness specifically** — they regress only when
something in this crate's engines (or the `accuracy_suite` driver)
changes. Colocating them with the engines + the suite makes "if you
modify `larql-kv`, re-run these" a single-directory check.

The accuracy suite splits results by [`KnowledgeSource`](../src/accuracy_suite/prompts.rs):

| Axis | Where the answer lives |
|---|---|
| **Parametric** | In the model's weights — any K/V strategy should score ≈ identically. |
| **In-context** | Planted in the prompt; engines lose it as their cache strategy compresses. |
| **Conflict** | In-context premise contradicts pretraining; scores `followed_context` vs `parametric_fallback`. |

Each cell records both **top-1 match rate** (argmax verdict) and
**Shannon bits-per-token** (`-log2 P(expected_first_token | prompt)`,
lower = more confident). Top-1 is binary; bits is continuous, and
needed because greedy decode can produce a correct prefix that fails
literal substring matching (see the Gemma 3 4B baseline's needle
discussion).

## Reproducing a baseline

```sh
./target/release/larql accuracy <model> \
  --engines standard,markov-rs,unlimited-context,turbo-quant \
  --output-file crates/larql-kv/baselines/<model>-<date>.json
```

Then write the `.md` companion documenting:
1. The reproducer command (model path / commit / machine).
2. The headline table (paste from stdout).
3. Per-engine wall-clock.
4. Per-prompt findings — especially any engine that diverged. Bits
   matters here; aggregates frequently agree while individual prompts
   do not.
5. Regression gates: which numbers may not move without an explanation
   in the engine's code.

## Existing baselines

- [`gemma3-4b-2026-05-17.md`](gemma3-4b-2026-05-17.md) — first multi-engine
  baseline: standard / markov-rs / unlimited-context / turbo-quant on
  Gemma 3 4B Q4K, CPU, 8-thread. All four bit-exact on parametric;
  unlimited-context loses the needle at 1024 tokens (window=512); 4-bit
  turbo-quant indistinguishable from FP32 K/V on this corpus.

## Regression workflow

Same shape as workspace-level `bench/baselines/cpu/`:

1. Run the suite — `larql accuracy <model> --output-file <new>.json`.
2. Diff `<new>.json` against the committed baseline of the same name:
   - Param/InCtx `match_rate` should match within ±1 prompt.
   - Mean bits should match within ±0.05.
   - Conflict verdicts on the divergence-set prompts should match
     exactly (the listed prompts are tokeniser-deterministic).
3. Any drift outside those bands needs a one-paragraph note in the
   commit message: which engine moved, on which axis, why.
