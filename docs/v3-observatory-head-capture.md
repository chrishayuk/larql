# Observatory CPU softmax head capture

Status: this describes separate local capture-tool work used to produce the
checked-in Granite evidence. The executor changes and `observatory_record`
example are not shipped with the Observatory frontend/bridge. The supported
merged producer is `larql vindex3 observe`; the bridge reads its JSONL directly.

This is an opt-in addition to the canonical CPU decode observer, separate from
[frozen V3-OBS-1 Standard capture](v3-obs-1-carrier-observation.md). It does not
change that rung's scope or claim that Standard contains individual heads.

`StepObserver::wants_attention_heads()` defaults to false. When requested,
`DecodeSession` calls `PlanBackend::attention_step_observed`. The default backend
implementation refuses; `ProductionBackend` supplies the tap inside its existing
softmax aggregation loop. Normal and observed execution share that loop, gate,
output projection, and subsequent carrier write. There is no second attention
calculation or transformer traversal for visualization.

The borrowed `AttentionHeadRecord` contains query position, query-head index,
KV-head index, source-start offset, actual softmax weights, and the actual
weighted-V head vector. The values are **before output gating, W_O, post-attention
normalization and residual scaling**. Source positions cover the executed window;
sink mass is preserved, never redistributed. The callback's layer is supplied by
the decode traversal. A requested all-head capture on a non-softmax plan refuses.
This initial implementation does not claim a batched-prefill, Metal, MLA,
recurrent-state, or per-head intervention tap.

## Granite runner witness

`observatory_record --heads CONTAINER OUTPUT.json PROMPT [PROBE_TEXT ...]`
opts into this tap plus owned carrier capture and the existing offline vocabulary
lens. Inputs remain bounded to 32 prompt tokens and 256 layers, with at most
16,384 head observations, 64 MiB of head arrays, and one 128 MiB W_O analysis
matrix at a time. Capturing is timing-intrusive. Device readbacks are zero on
this CPU path; durations are not representative inference benchmarks.

The runner currently admits single-stream, ungated, bias-free softmax attention
without post-attention normalization and with floating stored W_O. This supports
the Granite test bed; it deliberately refuses to pretend Gemma's post-attention
normalization is an independently additive per-head operation.

After execution:

1. A fresh-KV unobserved control must match every finite vocabulary logit
   bit-for-bit at every prompt position.
2. The captured head vectors are multiplied by the corresponding **stored** W_O
   columns in f64 and the plan's residual scale. Their sum must reconstruct every
   actual attention carrier delta within `1e-4` relative L2 (absolute L2 when the
   reference is zero). This explicitly checks physical-realization disagreement.
3. Content bars are f64 dots into the predeclared stored output-token rows.
   The first declared token is the heatmap target. These are selected raw token
   directions, not vocabulary top-k, normalized logit changes, or causal effects.
   No final norm, output multiplier or softcap is applied to a head contribution.
4. The sidecar binds to the exact event-record SHA-256. Raw head values and source
   weights are retained in a separate lossless f32-round-trip JSON payload, whose
   SHA-256 is in the sidecar. Carrier vectors remain in the existing f32 payload.

The executor tests exercise bit parity, once-per-head coverage, GQA mapping,
a sliding window that genuinely truncates, and unsupported-backend refusal.
The real recording supplies the model-backed parity and reconstruction witness.
Interventions, causal claims, dimensionality studies and KV-content accounting
remain separate evidence requirements.


## Published Granite witness (2026-09-20)

- Model: `granite-4.2-3b.s6.vindex3`; prompt: `The capital of France is`.
- Run: `standard-1789916123150343000-39449`.
- 5 prompt positions × 40 layers × 40 query heads = 8,000 observations.
- 400 carrier writes; 200 attention writes reconstructed.
- Maximum relative L2 reconstruction error: `3.4234811296986355e-7`.
- Observed and unobserved all-position vocabulary-logit SHA-256:
  `5f77c8f8043a599577ffc9948b683eb9ddd6534b562f234e564958781bc80608`.
- Event record SHA-256:
  `8915cc58b32cb5c131398ff122af71f26277fbb44f10dfc7766c580964feda01`.
- Output: ` Paris`, final normalized probability `0.9061932565969187`.
- At final prompt position, L35/H7's measured source weight on position 3
  (`France`) is `0.8205832242965698`. Its Paris direction probe is positive;
  this is not an intervention-supported retrieval claim.

The published source, lens, parity, replay audit and raw head payload are under
`observatory/public/recordings/`. The prior Standard recording is preserved under
`archive/`; the new evidence is not grafted onto the old run identity.
