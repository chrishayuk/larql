# LARQL Rust forward divergence vs HF / MLX — Shannon cross-engine diagnostic

Status: **FIXED. Loader now parses `rms_norm_eps` and the structured
`rope_scaling` field from config.json; the three architectures that
exposed bugs (Llama 3.2, Mistral 7B, Gemma 3 4B) all match HF/MLX to
within 0.003%.** Last updated 2026-05-16.

`scripts/diagnose_models.py` now passes 4/4 with **zero env-var
overrides** — the production loader path is correct.

## Resolution summary (2026-05-16)

Empirical bisection identified **three independent bugs** in LARQL's
config-loading path. All three are now fixed in `larql-models` directly;
the env vars used to diagnose them stay in tree as instruments.

| # | Bug | Pre-fix gap | Permanent fix | Post-fix gap |
|--:|---|---:|---|---:|
| 1 | `rope_scaling` per-layer-type (Gemma 3 applies linear factor only on full-attention layers) was not honoured | Gemma 3 4B: +5.4% | `parser.rs` parses the structured `{full_attention, sliding_attention}` shape into `RopeScaling { gemma3_global_only: true }`; `Gemma3Arch::rope_position_divisor_for_layer` returns the factor on global layers, 1.0 on sliding | +0.000% |
| 2 | `rms_norm_eps` from config.json was ignored — `ModelArchitecture::norm_eps()` hardcoded to 1e-6 | Mistral 7B: +8.2%; Llama 3.2 1B: +0.59% (partial) | `parser.rs` parses `rms_norm_eps` / `layer_norm_eps` into `ModelConfig.norm_eps`; default `norm_eps()` reads the parsed value, falls back to 1e-6 only when absent. CPU forward callers in `forward/{layer,ops}.rs` use new `rms_norm_for_arch` helper | +0.001% Mistral; +0.003% Llama (combined w/ bug 3) |
| 3 | `rope_scaling = llama3` (wavelength-dependent per-channel factors) was not implemented | Llama 3.2 1B residual after bug 2 | New `Llama3RopeScaling` type in `larql-models/config.rs`; `LlamaArch::llama3_rope_scaling` returns it when `rope_type=llama3`. `attention/rope.rs` has `Llama3Scaling::apply` (mirrors HF's `_compute_llama3_parameters`). Forward path goes through `apply_rope_partial_at_full` | +0.003% (with bug 2 fix) |

All three are independent: each closes its specific gap in isolation,
and they compose without interaction.

**Verification:** `.venv/bin/python scripts/diagnose_models.py` —
4/4 PASS at <0.5% delta, **zero env-var overrides applied**:

```
SmolLM2-135M    llama (small, no SWA)              0.001%   PASS
Llama-3.2-1B    llama (llama3 rope_scaling)        0.003%   PASS
Mistral-7B-v0.1 mistral (all SWA)                  0.001%   PASS
Gemma-3-4B-it   gemma3 (mixed SWA + global)        0.000%   PASS
+ Q4K Metal:    Gemma 3 4B  F32=1.0712  Q4K=1.4680  37.0%   PASS
```

## Diagnostic env vars (kept in tree)

The five env vars used to localise the bugs remain available for future
bisection on new architectures or for verifying changes to the forward
path. All are gated via `OnceLock` and are no-ops when unset.

| Var | Purpose |
|---|---|
| `LARQL_FORCE_GLOBAL_LAYERS=all\|<csv>` | Force listed layers onto the global rope_base. Useful for ruling out sliding-vs-global routing as a cause. |
| `LARQL_ROPE_POS_DIVISOR=<f64>` | Apply linear rope-scaling factor to *every* layer's positions. |
| `LARQL_ROPE_POS_DIVISOR_GLOBAL=<f64>` | Same but only on `!is_sliding_window_layer(layer)` layers. |
| `LARQL_LLAMA3_ROPE_SCALING=factor,low,high,old_ctx` | Force HF llama3 scaling params (overrides the arch's parsed value). |
| `LARQL_NORM_EPS_OVERRIDE=<f64>` | Override `arch.norm_eps()` (overrides config-parsed value). |

## TL;DR

LARQL Rust's F32 forward pass produces systematically lower next-token
probability on Gemma 3 4B (+5.4% bits/char) and Mistral 7B v0.1 (+8.2%)
than two mutually-validated F32 references (HF/PyTorch and MLX, which
agree with each other to within 0.06% on every model tested).
Architecturally-simpler models are clean (SmolLM2-135M <0.01%, Llama-3.2-1B
+0.59%). The instrument that surfaced this is `larql shannon verify`.

The initial hypothesis was sliding-window attention. Empirical bisection
ruled that out. The remaining suspects are arch-wide and need code-level
inspection.

## Instrument

`larql shannon verify MODEL --corpus FILE`

Runs the LARQL Rust forward in-process and spawns the HF/PyTorch and MLX
reference scorers as Python subprocesses on the same corpus, with the
same windowing logic (`context`, `stride`). Prints a delta table and
exits non-zero if any pair-wise delta exceeds `--threshold`. ~17s on a 4B
model end-to-end on M3 Max.

Underlying scripts: `scripts/shannon_score_mlx.py`,
`scripts/shannon_score_hf.py`, and the existing `larql shannon score`.

CRLF and BOS gotchas across the three engines are documented in
`scripts/README_shannon_score.md`.

## Triangle results

Corpus: 997 chars (1KB Frankenstein header, CRLF-normalized). All three
engines at F32. Same context=512 / stride=256 windowing.

### Gemma 3 4B (247 scored tokens)

| Engine | total bits | bits/char | Δ vs HF |
|---|---:|---:|---:|
| HF F32 (torch CPU) | **1068.0** | 1.0712 | — |
| MLX F32 | 1068.6 | 1.0718 | +0.06% |
| LARQL Rust F32 | 1126.3 | 1.130 | **+5.4%** |

### Mistral 7B v0.1 (289 scored tokens)

| Engine | total bits | bits/char | Δ vs HF |
|---|---:|---:|---:|
| HF F32 (torch CPU) | **508.86** | 0.5104 | — |
| MLX F32 | 508.86 | 0.5104 | <0.01% |
| LARQL Rust F32 | 550.4 | 0.552 | **+8.2%** |

### Llama-3.2-1B (234 scored tokens)

| Engine | total bits | bits/char | Δ vs HF |
|---|---:|---:|---:|
| HF F32 (torch CPU) | **579.39** | 0.5811 | — |
| MLX F32 | 579.40 | 0.5811 | <0.01% |
| LARQL Rust F32 | 582.8 | 0.585 | **+0.59%** |

### SmolLM2-135M (262 scored tokens)

| Engine | total bits | bits/char | Δ vs HF |
|---|---:|---:|---:|
| HF F32 | 812.830 | 0.8153 | — |
| MLX F32 | 812.840 | 0.8153 | +0.001% |
| LARQL Rust F32 | 812.767 | 0.8152 | -0.008% |

### Cross-architecture summary

| Model | Layer pattern | Hidden | LARQL Rust vs HF |
|---|---|---:|---:|
| SmolLM2-135M | 30 std, no SWA | 576 | <0.01% |
| Llama-3.2-1B | 16 std, no SWA | 2048 | +0.59% |
| Gemma 3 4B | 28 sliding + 6 global | 2560 | +5.4% |
| Mistral 7B v0.1 | 32 all-sliding | 4096 | +8.2% |

The two reference engines mutually validate to <0.1%. LARQL Rust is the
unique outlier; the divergence is always in the same direction (lower
probability on the true next token); magnitude scales with something that
correlates with sliding-window architectures and/or model size.

## Initial hypothesis and how it was tested

A priori suspect: the sliding-window attention code path. Reasoning:
- Llama-class (no SWA): small gap.
- Gemma 3 (mixed SWA + global): larger gap.
- Mistral (all SWA): largest gap.

Diagnostic instruments added to the LARQL Rust forward path
(`crates/larql-inference/src/layer_graph/pipeline_layer.rs`,
`attention/block.rs`, `attention/decode.rs`, `attention/rope.rs`):

- `LARQL_FORCE_GLOBAL_LAYERS=all|<csv>` — for the listed layer indices,
  forces `rope_base = arch.config().rope_base` (the global rope_theta)
  regardless of `arch.is_sliding_window_layer(layer)`. Effectively
  collapses the sliding-vs-global routing.
- `LARQL_ROPE_POS_DIVISOR=<f64>` — divides the RoPE position before phase
  computation, emulating HF's `rope_scaling = {type: linear, factor: N}`
  semantics that LARQL does not currently honour.

Both are gated via `OnceLock`. No behaviour change when unset.

## Bisection results — Gemma 3 4B

| Configuration | Total bits | Δ vs HF (1068) |
|---|---:|---:|
| Baseline (no override) | 1126.3 | +5.4% |
| `LARQL_FORCE_GLOBAL_LAYERS=all` | 1387.4 | +29.9% (much worse) |
| `LARQL_ROPE_POS_DIVISOR=8` (uniform) | 1649.9 | +54.5% (much worse) |
| `LARQL_ROPE_POS_DIVISOR_GLOBAL=8` (global layers only) | 1068.0 | **+0.000% (fixed)** |

### Reading

1. **LARQL's per-layer rope_base routing is correct.** Forcing global
   rope_base on every layer makes the gap dramatically worse, which
   means the sliding-layer rope_base = 10,000 is what HF actually
   expects. LARQL was already routing correctly.

2. **HF's `rope_scaling` is per-layer-type.** Querying HF's effective
   config exposed the structured form
   `{sliding_attention: {rope_type: default, rope_theta: 10000}, full_attention: {rope_type: linear, factor: 8.0, rope_theta: 1000000}}`.
   The `factor: 8.0` applies **only on full-attention (global) layers**,
   not on sliding layers. The raw config.json shows the flat form
   `{factor: 8.0, rope_type: linear}` which HF's Gemma3 config class
   expands to the structured form on load. LARQL was reading the flat
   form and not honouring either the per-layer-type structure or the
   factor at all.

3. **Applying the factor only on global layers closes Gemma 3's gap
   exactly.** `LARQL_ROPE_POS_DIVISOR_GLOBAL=8` drops LARQL Rust to
   1068.014 bits — matching HF's 1068.011 to within +0.000% (well
   under MLX's own +0.056% noise from HF). This is the bug.

## Bisection results — Mistral 7B v0.1 and Llama-3.2-1B

After fixing bug 1, the remaining gap on Mistral (+8.2%) and Llama
(+0.59%) was traced via the same toggle-and-measure approach.

**Mistral 7B v0.1:**

| Configuration | Total bits | Δ vs HF (508.86) |
|---|---:|---:|
| Baseline (no override) | 550.4 | +8.2% |
| `LARQL_NORM_EPS_OVERRIDE=1e-5` | 508.9 | **+0.001% (fixed)** |

The HF effective config for Mistral has `rms_norm_eps: 1e-05`. LARQL's
`norm_eps()` returns the hardcoded default `1e-6` from `config.rs:786`
for every architecture; the `rms_norm_eps` field from config.json is
never read by the parser. The 10× difference matters when residual
magnitudes are small relative to eps — which happens enough during a
34-layer (or 32-layer) forward to compound to 8.2% on bits/char.

**Llama-3.2-1B (after applying bug 2 fix):**

| Configuration | Total bits | Δ vs HF (579.39) |
|---|---:|---:|
| Baseline | 582.8 | +0.59% |
| `LARQL_NORM_EPS_OVERRIDE=1e-5` | 582.09 | +0.47% (partial) |
| `LARQL_NORM_EPS_OVERRIDE=1e-5` + `LARQL_LLAMA3_ROPE_SCALING=32,1,4,8192` | 579.40 | **+0.003% (fixed)** |

Llama-3.2-1B has `rope_scaling = {rope_type: llama3, factor: 32, low_freq_factor: 1, high_freq_factor: 4, original_max_position_embeddings: 8192}`. The `llama3` type is wavelength-dependent: per-channel
`inv_freq[i]` is divided by `factor` for slow-rotating channels
(`wavelength > old_ctx / low_freq`), passed through for fast-rotating
channels (`wavelength < old_ctx / high_freq`), and smoothly interpolated
in between. LARQL was implementing standard RoPE for all channels with
no scaling. The fix is in `crates/larql-inference/src/attention/rope.rs`
(`Llama3Scaling` struct + `apply_rope_partial_at_full`).

## Permanent fix plan

The env-var diagnostics confirm the bugs exhaustively. The permanent
fix is in three small parts:

1. **`crates/larql-models/src/detect/parser.rs`**: read
   `text_config["rms_norm_eps"]` and store it on `ModelConfig`. Default
   to 1e-6 only when absent.
2. **`crates/larql-models/src/config.rs`**: override `norm_eps()` to
   return the parsed config value via the default trait impl.
3. **`crates/larql-models/src/architectures/gemma3.rs`**: parse the
   structured `rope_scaling` field, expose `rope_position_divisor_for_layer`,
   and have `apply_rope_partial_at_full` consult it. Similar plumbing
   for `architectures/llama.rs` to expose the llama3 scaling parameters.

The env vars stay in tree as the diagnostic instrument set — they're
useful for future regression bisection on new architectures, and for
verifying that the permanent fix matches what the env-var diagnostic
proved.

## Footprint of HF-routed vs LARQL-Rust-routed results in this codebase

What this divergence does and does not affect:

- **Unaffected:** any result produced by Python + HF transformers
  (atlas, Lazarus mechanistic experiments, hierarchical surgery
  experiment 78, the published mech-interp work). HF agreed with itself
  across the triangle.
- **Affected, in proportion to the gap:** any LARQL Rust forward pass
  result on Gemma 3 4B or Mistral 7B v0.1 where *absolute* probabilities
  or gate values are the load-bearing claim. Top-1 rankings and KNN
  retrieval are likely preserved (the gap is small and consistent in
  direction) but absolute claims need an asterisk until the bug is
  fixed.
- **Q4_K vindex path** stacks an additional ~30% on top of the F32 gap.
  That's typical Q4_K_M quantization loss, but worth knowing the total
  gap from F32 reference is roughly: 5.4% (forward) + 30% (quant) for
  Gemma 3 4B Q4_K.

## How to reproduce

The multi-architecture sweep is the easiest entry point — runs all four
test cases with the env-var fixes applied and prints a pass/fail table:

```bash
.venv/bin/python scripts/diagnose_models.py
```

Expected output: 4/4 PASS at <0.5% threshold, plus the Q4K Metal path
showing ~37% Q4K quantization gap for Gemma 3 4B (expected).

To reproduce the individual bisection steps:

```bash
# Prep corpus (strip BOM + CRLF).
tail -c +4 data/gutenberg/frankenstein.txt | head -c 1024 \
    | tr -d '\r' > /tmp/larql_shannon_corpus_lf.txt

# Triangle on Gemma 3 4B without fix. ~17s. Expected: FAIL +5.4%.
larql shannon verify google/gemma-3-4b-it \
    --corpus /tmp/larql_shannon_corpus_lf.txt \
    --threshold 0.5

# Same with fix. Expected: PASS +0.000%.
LARQL_ROPE_POS_DIVISOR_GLOBAL=8 \
    larql shannon verify google/gemma-3-4b-it \
    --corpus /tmp/larql_shannon_corpus_lf.txt \
    --threshold 0.5

# Mistral 7B v0.1 with fix. Expected: PASS +0.001%.
LARQL_NORM_EPS_OVERRIDE=1e-5 \
    larql shannon verify mistralai/Mistral-7B-v0.1 \
    --corpus /tmp/larql_shannon_corpus_lf.txt \
    --threshold 0.5

# Llama-3.2-1B with both fixes. Expected: PASS +0.003%.
LARQL_NORM_EPS_OVERRIDE=1e-5 \
LARQL_LLAMA3_ROPE_SCALING=32,1,4,8192 \
    larql shannon verify meta-llama/Llama-3.2-1B \
    --corpus /tmp/larql_shannon_corpus_lf.txt \
    --threshold 0.5
```

Requires `torch` and `mlx_lm` in `.venv` for the HF and MLX reference
legs.

## Memory cross-references

- `memory/project_swa_forward_divergence.md` — quick-recall summary,
  links to this document.
- `memory/reference_shannon_scorer_triangle.md` — the instrument itself.
