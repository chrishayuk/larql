# PREC-DIAG oracle: the `qwen3_5` family, qualification (diagnostic)

Date 2026-11-21. Refs #897. **Oracle qualification only. No accuracy result.** Nothing was compared between
LARQL and transformers: no logit difference, d, r, cosine or top-1 agreement was computed. No 1,040-token
forward ran. No FREEZE or experiment directory was created. The logits of the real-checkpoint runs were
deleted unread. Their shape comes from the file size, their dtype from the oracle's float64 writer, and
their finiteness from the oracle's own refusal of non-finite logits.

## What the family is

Code: `scripts/prec_diag/hf_qwen3_5.py`, wired through `hf_oracle.py` and `families.py`. The entry pins
transformers 5.5.0's `modeling_qwen3_5.py` at
`aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048`.

**Loading** (`load_text_model`):
- The checkpoint is composite. `from_pretrained(dtype=float64)` on its config leaves everything bf16, so
  the text model is loaded from `text_config`.
- The dtype of every parameter and buffer is then asserted as loaded: A64 float64, every other arm
  float32. There is no `.to(dtype)` that could hide a wrong load.
- Under A64, the two rotary buffers are rebuilt from the float64 formula. transformers computes them in
  float32 at construction and does not widen them.

**Loading gate** (`loading_facts`). Every checkpoint tensor is either loaded (under
`model.language_model.`, mapped to `model.`, with the head tied) or excluded under a declared prefix:
- `model.visual`: the vision tower;
- `mtp`: the MTP head.

The gate refuses:
- any other tensor;
- a text tensor the model does not load;
- an excluded tensor inside LARQL's closure;
- a loaded tensor outside LARQL's closure;
- a model key with no checkpoint source.

transformers' own `unexpected` list cannot be relied on here, because the modeling file hides `mtp.*` and
`model.visual.*` from it.

**Parameter audit** (part 1). Every parameter must equal its checkpoint tensor widened exactly. Sources
may be BF16 or F32; `A_log` and the DeltaNet norm weight are F32. The tied head is checked. The rotary
buffers must equal the float64 formula.

**A64 patches** (part 3). Each patch has a numpy float64 witness. Each call count is derived from the
config's `layer_types`:

| patch | count |
|---|---|
| (1+w) RMSNorm | 2·layers + 2·full + 1 |
| gated RMSNorm | linear |
| rotary (f64 inv_freq, partial rotary, M-RoPE, text-only positions) | 1 |
| eager softmax | full |
| DeltaNet forward (the decay `g` in f64; witness: conv + SiLU, β, g) | linear |
| chunked gated delta rule, per layer instance | linear |

The decode rule is replaced on every instance by a refusal.

**The recurrence witness.** `step_witness` is a numpy float64 step-by-step gated delta recurrence. It is
written from the published update (Yang et al., Gated Delta Networks, ICLR 2025):

`S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`, `o_t = S_tᵀ q_t`

with L2-normalised q and k, and q scaled by d_k^−½.

The patched chunked rule is checked against it on every DeltaNet layer of every A64 forward:
- the outputs at every position;
- the state, per head, at every chunk end (64k−1) and after the last token.

**Bound.** 1e-12, per position and per head. This is `F64_WITNESS_REL`, the bound the other witnesses use.
- Both sides are float64 and differ only by summation order: the chunked form uses a forward substitution
  and `exp(g_i − g_j)` decays.
- Measured agreement is about 1e-15:

  | input | worst agreement |
  |---|---|
  | synthetic, T ≤ 1,040 | 1.2e-15 |
  | real checkpoint, 130 tokens | 4.3e-15 |

  That is about three orders of magnitude inside the bound.
- transformers' shipped float32 rules miss the same witness by 1e-7 to 3e-6, more than five orders of
  magnitude outside it. A test shows this for both the chunk and the recurrent rule, on outputs and states.
- On float32 inputs, the patched rule is bit-identical to transformers' rule. So is the patched DeltaNet
  forward on a float32 model. Both are pinned by tests. A64 is therefore transformers' algorithm, run in
  float64.

**Part 5** (`op_audit.py`, mandatory). For this family it needed four ops classified:
- `constant_pad_nd` and `eye`: exact.
- `softplus`: an elementwise probe, on both sides of its threshold.
- 1-d `convolution`: an accumulating probe made of two exact sub-probes.
  - Input 1 with weights `[h, e, …]`: the accumulator and the weight operand.
  - Input `1 + 2^-26` with weights `[1, 0, …]`: the input operand.
  - A test shows that rounding only the input, or only the weight, fails it.

l2norm and the conv were verified by part 5 on the real forward, not assumed.

**The part's schema is now `larql.prec-diag-1.a64-op-audit/2`.** A freeze names its schema
(`values.a64_op_audit`), and the oracle refuses any other. So a run recorded under `/2` is never
mistaken for one made under the narrower `/1` op set, and vice versa. No existing freeze names a schema:
none sets `a64_op_audit`, so none of their identities changes. The #899 backfill ran under `/1`, and its
record says so.

**Arms.**
- **A32 and A32s** run transformers' shipped float32 path unmodified, after the same text-config load.
- **A32r** takes a table for the `full_attention` layers only, at the partial rotary width
  (rotary_dim/2), on text-only positions.
  - The three M-RoPE axes must be identical, so the injected table is the 1-D RoPE the modeling file
    computes.
  - A `linear_attention` table is refused: DeltaNet layers take no positions.
  - `experiment.rope_table_labels` asks only for the layer types that apply rotary
    (`families.rope_layer_types`).
- **A32rec** (diagnostic only) uses transformers' own float32 `torch_recurrent_gated_delta_rule` for the
  prefill. It is kept away from the gate in three ways:
  - the runner never plans it;
  - the oracle refuses it under a freeze (`bind`) and for any family without a recurrence;
  - its identity names arm `A32rec`, so it is never admitted as A32.

  `hf-f64-accuracy/2` reads only C, D, A32 and A64. A test pins this from `evaluate` itself.

## Real-checkpoint qualification (diagnostic, throwaway)

**Subject**
- `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`, from the local HF cache.
- safetensors `04b1c301…e4696`; config `b90b86f3…b4204`.
- transformers 5.5.0, torch 2.14.0, numpy 2.4.2; BLAS accelerate; 8 threads, interop 1, `OMP_NUM_THREADS=8`.

**Loading boundary.** `larql vindex3 plan <snapshot> --capability text-generation`:
- in closure: `model.language_model.{embed_tokens,layers,norm}`;
- allowlisted: `model.visual.*`;
- the MTP head is bound to no component.

The binary was larql 0.3.0 at main `9337e2c8`, the one the feasibility study used.

**Inputs.** Throwaway English text, tokenised. The runs went through `hf_oracle.run` with the real
lifecycle and the record. The scripts in the run identity are this PR's files.

| run | tokens | status | part 1 | parts 2–4 | part 5 | ops traced | non-f64 ops | unclassified | probes | worst witness (chunk rule) | wall | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A64 | 16 | complete | pass (321/321; 285 BF16 + 36 F32) | pass | **pass** | 20,494 | **0** | 0 | 101, all pass | 1.8e-15 (1 chunk) | 11.7 s | 10.4 GB |
| A64 | 130 | complete | pass | pass | **pass** | 22,798 | **0** | 0 | 104, all pass | 4.3e-15 (chunk ends 63, 127, 129) | 17.0 s | 10.9 GB |

**Call counts** in both runs equal the counts derived from the config: rmsnorm 61, gated_rmsnorm 18,
rotary 1, softmax 6, gdn 18, chunk_rule 18.

**The loading gate passed:**
- 320 text tensors loaded;
- 153 `model.visual` tensors excluded (LARQL allowlist);
- 15 `mtp` tensors excluded (bound to no LARQL component);
- 488 checkpoint tensors in all;
- the head shares the embedding.

**Part 5:**
- The only dtype conversion in the forward is int64 → float64 of the position ids, in the patched rotary.
- 187 linear matmuls were traced for 187 `nn.Linear` modules, so the trace is not partial.
- The largest probe was 2.29 GB (the lm_head-shaped `mm`).

**Timing (130 tokens).** "wall" is `hf_oracle.run` itself.
- load 0.3 s;
- chunk-rule witnesses 2.7 s;
- part 1: 0.5 s;
- part 5 probes 2.1 s;
- forward with all parts 15.3 s.

The machine was shared with other work, including a live experiment. An earlier pair of runs, on the same
code except one unused attribute and under heavier load, gave identical audit results and took 27 s and
35 s.

**Re-run after review.** Both runs were repeated after the review fixes: op-audit schema `/2`, and the
chunk-state count check. The audit results were identical (same op counts, 0 non-f64 ops, same witness
errors), with `states_compared` 1 and 3. The run identities name this PR's current files. Wall time was
14.3 s and 20.4 s.
Logits were `[16, 248320]` and `[130, 248320]` float64 and finite.

## Estimate at 1,040 tokens (not run)

**Time: about 1–3 minutes per A64 input**, depending on machine load.
- The forward without witnesses or probes grew by about 0.02 s per token, from 16 to 130 tokens, on top
  of a fixed ~7.5 s that is mostly dispatch-mode overhead. At 1,040 tokens that gives ~30 s; ~75 s under
  the heavier load of the first pair of runs.
- The recurrence witnesses run at about 0.02 s per token over 18 layers: ~22 s (~50 s under load).
- The probes are dominated by one f64 lm_head GEMM, about 0.5 TFLOP: ~10 s.
- Writing the 2.07 GB logits file adds a few seconds.

**Peak memory: about 16–20 GB**, against the 128 GB on this machine.
- The base is ~10 GB: f64 parameters plus the checkpoint dict of the parameter audit.
- Two copies of the f64 logits add 2 × 2.07 GB.
- The lm_head-shaped probe adds about 2.03 + 2.07 GB while the logits are still held.

A freeze should set its RSS limit from the `memory` record of a first run.

## Not done here

- **No accuracy comparison of any kind**, by rule.
- **No E2/LARQL-side work.**
  - C32a and D32a refuse on DeltaNet layers (feasibility study). v2 does not read them.
  - D's DeltaNet recurrence is shared with C (feasibility Q2). That is an owner decision for the freeze.
- **No site captures for qwen3_5** (`captures: False`). E3 refuses at preflight.
- **The vision tower and MTP head are out of scope.** A vision position layout (distinct M-RoPE axes) is
  refused.
