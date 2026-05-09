## ADDED Requirements

### Requirement: Post-8.04-ms checkpoint follow-up consolidation document

The repository SHALL ship
`openspec/changes/cuda-decode-perf-results-followup/proposal.md`
as the navigation aid for any future contributor
inheriting the CUDA decode performance work that landed after the
`cuda-decode-perf-results` retro's 8.04 ms/token checkpoint. It
SHALL contain:

- The post-`8.04` checkpoint progression table (decode ms/tok,
  tok/s, prefill ms) showing the `cuda-mmvq-hw-f16-cvt` win to
  7.44 ms/tok and the cumulative gap closure with
  `llama-cpp-turboquant` (decode 1.71×, prefill 1.71×).
- The mechanism by which `cuda-mmvq-hw-f16-cvt` delivered its
  -7.5% win (PTX `cvt.f32.f16` replacing software emulation).
- The mechanism by which `cuda-marlin-imma-probe` settled
  Path A as not viable (INT8 IMMA loses 3-7× to dp4a at batch=1
  due to fragment-row waste).
- An explicit closure table marking Paths A/B/D/E from the
  original retro as closed, and Path C as open-but-deferred.
- A cross-reference to the
  [`cuda-speculative-decoding`](../cuda-speculative-decoding/proposal.md)
  architectural pivot that picks up where the batch=1
  micro-optimisation work ends.

#### Scenario: navigation aid lets a fresh contributor identify the next-step architectural path

- **WHEN** a contributor opens
  `openspec/changes/cuda-decode-perf-results-followup/proposal.md`
  cold
- **THEN** they SHALL identify within ~5 minutes that
  micro-optimisation paths are exhausted at batch=1 and that
  speculative decoding (per
  `openspec/changes/cuda-speculative-decoding/`) is the only
  remaining mechanism for closing the gap with llama.cpp
<!-- test: unbacked -->
