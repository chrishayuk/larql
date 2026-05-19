# `larql dev fidelity` — Specification

**Status:** Draft.
**Source experiments:** Shannon Exp 34 (internal bit contribution),
Exp 35 (FFN functional fidelity), Exp 36 (patch propagation),
Exp 37 (bit-budget additivity).
**Crates touched:** `larql-cli` (new subcommand), `larql-inference`
(read-only consumer of internal forward-pass intermediates).
**Historical note:** earlier drafts pointed at reusable bridges in
`kv-cache-benchmark/examples/{q4k_ffn_raw_bridge,patch_propagation_q4k,
bit_budget_additivity_q4k}.rs`. That crate was retired in 2026-05-16 and
its experiment-specific bridges were dropped — the work needs to be
redone against the production `larql_kv::vindex_compare` API or rebuilt
fresh inside `larql-cli`.

## 1. Purpose

Promote bit-fidelity from one-off experiment scripts to a first-class CLI
subcommand. Given a baseline vindex and a candidate (a patched vindex, an
alternative top-K, or an alternative quantisation), measure how much of the
real FFN's predictive contribution the candidate recovers, in bits.

The shipping criterion in the Shannon arc is:

```
functional_fidelity = candidate_bits_saved / real_ffn_bits_saved
```

Cosine and MSE are diagnostic; they are not the contract. Exp 35 demonstrated
both directions of the failure: a `top256_walk` candidate at cos = 0.853
delivered fidelity 0.475 (cos high, fidelity halved); a 30° controlled
rotation at cos = 0.866 delivered fidelity 0.853 (cos and fidelity track each
other only when the perturbation is structurally aligned).

This spec defines the CLI surface, the metric, and the corpus contract.

## 2. Metric

For each scored token `t` in the corpus and a target layer `L`:

```
bits_real[t]      = -log2 p_real_ffn(target_token | prefix)         at L
bits_baseline[t]  = -log2 p_mean_replacement(target_token | prefix) at L
bits_candidate[t] = -log2 p_candidate(target_token | prefix)        at L

real_saved[t]      = bits_baseline[t] - bits_real[t]
candidate_saved[t] = bits_baseline[t] - bits_candidate[t]
```

Aggregate metrics:

```
fidelity      = sum(candidate_saved) / sum(real_saved)
correlation_t = pearson(candidate_saved, real_saved)
cosine        = cos(real_ffn_out_vector, candidate_out_vector)        # diagnostic only
```

The mean-replacement baseline is the dense FFN evaluated with its output
replaced by the corpus mean of `mlp_out` at layer `L`. This mirrors Exp 34/35
exactly — do not invent a new baseline here.

### 2.1 Success cone

```
fidelity ≤ 0     :  failed
fidelity ~ 0.5   :  partial
fidelity ≥ 0.7   :  useful
fidelity ≥ 0.9   :  near-dense
```

`larql dev fidelity` prints the headline fidelity number plus the cone label.
The cone is reported, not enforced — the command always exits 0 unless
something failed structurally (corpus unreadable, candidate incompatible).

## 3. CLI

```
larql dev fidelity <baseline.vindex> <candidate>           \
    --layer 30                                              \
    --corpus prompts.jsonl                                  \
    --tokens 840                                            \
    [--top-k full|256|1024|2048]                            \
    [--patch path/to/candidate.vlp]                         \
    [--out results.json]                                    \
    [--report text|json]
```

Where:

- `<baseline.vindex>` is the unmodified reference vindex.
- `<candidate>` is one of:
  - a path to another vindex (alternate quantisation, alternate weights),
  - the same vindex with `--patch <file>.vlp` overlay,
  - the same vindex with `--top-k <K>` (sparse-walk substitution against
    its own dense path).
- `--layer L` is required. Multi-layer reports are out of scope; run the
  command once per layer.
- `--corpus prompts.jsonl` defaults to `experiments/shannon/35_*/data/seed_corpus.jsonl`
  if present, otherwise the command errors clearly.
- `--tokens N` caps the scored token count for run-time control. Exp 35
  used 840.

### 3.1 Output

Default text report (one screen):

```
fidelity         : 0.912    (near-dense)
per-token corr   : 0.998
cosine (diag)    : 0.998
real bits saved  : 135.36 over 840 tokens
candidate saved  : 123.52 over 840 tokens
baseline used    : mean_replacement (corpus mean mlp_out @ L30)
layer            : 30
```

JSON report (with `--report json`) is the same fields plus per-token vectors.
Schema lives next to this spec — explicit, versioned.

## 4. Corpus contract

A corpus is a JSONL file of `{prompt, target_text}` records. The harness:

1. Tokenises `prompt + target_text` once using the model's tokenizer.
2. Runs forward, recording `mlp_in` and `mlp_out` at the target layer for
   each scored position.
3. Scores positions covered by `target_text` only. Prompt positions are
   warm-up.

Determinism rules:

- The same forward path runs for `real`, `baseline`, and `candidate` — only
  the layer-`L` FFN output differs. Exp 31 surfaced the encoder/decoder
  divergence problem; this spec inherits the lesson: no batched-prefill on
  one branch and stepwise on another.
- The mean-replacement baseline is computed once, on a held-out slice of
  the corpus, before scoring. The slice is recorded in the report.

## 5. Reuse

The closest current building block is
`crates/larql-kv/src/vindex_compare.rs` + its `vindex_compare` example,
which already forward-passes two vindexes through the same model and emits
the per-prompt + aggregate fidelity report (cos / KL / top-K jaccard /
argmax agreement). Lifting it into a CLI subcommand is mostly:

1. Replace the example's CLI-args main with a `clap` subcommand.
2. Add the Shannon-style internal-bit-contribution / patch-propagation /
   bit-budget-additivity scoring on top of the existing per-prompt
   `PromptReport` shape — these used to live in
   `kv-cache-benchmark/examples/{q4k_ffn_raw_bridge,patch_propagation_q4k,
   bit_budget_additivity_q4k}.rs` (retired 2026-05-16) and need a fresh
   port against the production `WalkFfn`/`kquant_ffn_forward_layer` API.
3. Wrap the existing harness in `commands/dev/fidelity/`.

The harness must remain reusable in the bench crate; do not delete the
example files — they are still the canonical reproducer for the experiment
write-ups they back.

## 6. Defaults and warnings

Operational guardrails surfaced by the experiments:

- **Top-K monotonicity is not assumed.** Exp 35 measured K=256 → 0.475,
  K=1024 → 0.839, K=2048 → 0.700. The command never extrapolates; report
  the measured K and stop.
- **Late-stack only.** The default-and-recommended target layer is the
  last third of the stack (Gemma 3 4B: L26 onwards). Earlier layers need a
  tuned readout (Exp 35 §"productive band readout"); using `lm_head ∘
  final_norm` to score L < ~L20 produces uninformative numbers. The
  command prints a warning when `--layer` falls outside the
  per-architecture supported range.
- **Final RMSNorm is part of the decoder.** Exp 34: bypassing it makes
  L33→final restore `20637.11 bits/token`. The harness must apply
  `final_norm` before `lm_head`; this is non-negotiable and verified at
  startup against a one-token sanity sample.

## 7. Out of scope

- Auto-search over candidate K. The command scores a fixed candidate.
- Multi-layer composite fidelity. Score one layer at a time; if needed,
  Exp 37's bit-budget additivity result governs how to compose.
- Productive-band tuned readouts. The Exp 35 synthesis flagged this as
  required before measuring early-layer FFN replacements; until that
  exists, this command refuses to score early layers and says so.
- Replacing existing inference benches. `decode_bench`, `accuracy_suite`,
  etc. continue to do their jobs; this is a fidelity-specific tool.

## 8. Test plan

```
1. Self-consistency: candidate := real_ffn → fidelity = 1.000 ± 1e-3,
   correlation = 1.000 ± 1e-4.
2. Mean-replacement: candidate := mean replacement → fidelity = 0.000 ± noise.
3. Anti-direction: candidate := -real_ffn (Exp 35: -0.1× real) →
   fidelity ≈ -3.165, command labels "failed".
4. Controlled rotation: 30° rotation of real → fidelity ~ 0.85,
   cosine ~ 0.87 (regression test against Exp 35 numbers).
5. Q4_K full walk: fidelity = 0.91 ± 0.02 against the dense baseline on
   the seed corpus (regression against Exp 35 §"q4k_full_walk").
6. Sparse-K K=8: fidelity reports negative, command labels "failed".
7. Determinism: two runs of the same (baseline, candidate, corpus,
   --tokens, --top-k) produce byte-identical JSON reports.
```

## 9. References

- `~/chris-source/chris-experiments/shannon/35_ffn_functional_fidelity/L30_SUCCESS_CONE.md`
- `~/chris-source/chris-experiments/shannon/35_ffn_functional_fidelity/CANDIDATE_RESULTS.md`
- `~/chris-source/chris-experiments/shannon/34_internal_bit_contribution/SPEC.md`
- `~/chris-source/chris-experiments/shannon/36_patch_propagation/SPEC.md`
- `~/chris-source/chris-experiments/shannon/37_bit_budget_additivity/SPEC.md`
- `crates/larql-kv/src/vindex_compare.rs` + `examples/vindex_compare.rs`
  — production A/B comparator; closest live equivalent of the retired
  `kv-cache-benchmark` experiment bridges (q4k_ffn_raw_bridge,
  patch_propagation_q4k, bit_budget_additivity_q4k).
- Memory: `feedback_isolated_vs_batched_kernel_profile` — batched fidelity
  is what predicts end-to-end; isolated kernel cosine does not.
