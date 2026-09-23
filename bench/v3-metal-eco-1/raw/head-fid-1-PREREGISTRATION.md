# gpt-oss-20b NVFP4 head fidelity — pre-registered 2026-09-23, before any logit was dumped

Question: does compiling gpt-oss-20b's output head to NVFP4 cost acceptable fidelity, over and above the
conservative NVFP4 pack it would join?

Arms (all Metal, lowered, teacher-forced with `vindex3 exec --logit-dump`):
- R reference: `metal-lowered-f16` on the source container `gpt-oss-20b.vindex3` (f16 everywhere the
  lowering holds f16; the expert banks are the checkpoint's own MXFP4).
- C conservative: `metal-lowered` on `gpt-oss-20b.nvfp4.vindex3` (attention NVFP4, head at source → f16).
- H head: `metal-lowered` on `gpt-oss-20b.nvfp4-head.vindex3`. Byte-identical to C's container except the
  added `target.output_head@NVFP4` segment and metadata (verified by sha256 before this file).

Corpus: the chat-templated prompt "Write a detailed history of the Roman Empire from its founding to its
fall." plus R's own greedy continuation of 256 tokens. Every position is scored.

Metrics per arm X ∈ {C, H}, over all positions: mean and p99 of KL(R || X) in nats (softmax of f32 logits),
and top-1 agreement with R. Directly: mean KL(C || H), top-1 agreement C vs H.

Frozen rule. H is ACCEPTABLE if both hold:
1. top-1 agreement(H, R) ≥ top-1 agreement(C, R) − 1.0 percentage point;
2. mean KL(R || H) ≤ 1.5 × mean KL(R || C).
Otherwise NOT ACCEPTABLE. A pass says that the head adds error comparable to the rest of the pack's, not
that the pack is lossless. Thresholds are provisional (uncalibrated on this model) and are recorded as such.
If C itself shows top-1 agreement below 90% with R, the comparison is reported as uninformative, not as a pass.
