# GW-CAR-1B — capital pre-execution review

The capital-only confirmation population was accepted on 28 September 2026.
The executable protocol was sealed before capture, fitting, or candidate replay.

- Protocol: `sha256:c35202a20adc6d8e02be49764fc05973ebfdded7e4d680681ec8c7e16b08a9b0`.
- Release executable: `sha256:24e9f5a437015c7741ce248b983f2df7e3035a1ae78be401c8cfe70f07b07110`.
- Model metadata and every historical model segment hash verified at seal.
- 24 synthetic Python tests pass, including the 104-readout family, control
  invalidation, template/shuffle discrimination, split isolation, source-only
  construction, and rehashed-candidate tampering refusal.
- The Rust context/head-selection test passes in release mode.
- The release example builds and passes Clippy with warnings denied.
- Its rustfmt check passes.
- The actual unsealed executable was invoked against a draft and refused before
  creating output or loading the model.

The [reviewed executable draft](../bench/gw-car-1/gemma3-4b-it-phase1/gwcar1b-capital-protocol.reviewed-draft.json)
was used for the seal. The earlier executable draft is retained as pre-review
history; it predates the final support-overlap diagnostic and is not executable.
The source-design draft also remains distinct from this executable protocol.

The population is 44 historical training subjects (396 rows), plus 67 fresh
capital subjects (201 rows; 34 validation / 33 test). The head interface remains
oracle target H1, donor other heads, and target-natural KV/continuation. No
latency, avoided-prefix, or global-relation claim is registered.

The planned post-seal checks are historical train-carrier identity, natural
head-mixture identity, fresh context-255 natural-continuation identity, and
candidate 0/1 identity against independent contexts 128/0. This review records
readiness and does not assert that these model checks have passed.
