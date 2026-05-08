# server-vindex-loading Specification

## Purpose
TBD - created by archiving change cuda-and-rotorquant-kv. Update Purpose after archive.
## Requirements
### Requirement: --role flag gates which weight families load

`larql-server` SHALL accept a `--role` CLI flag with values:

- `all` (default — current behaviour) — load every weight family.
- `ffn` — load only FFN expert weights, gate vectors, and metadata
  needed to serve `/v1/expert/*` and `/v1/walk_ffn`.
- `attention` — load only attention weights (Q/K/V/O), final norms,
  and the embedding matrix; do not load FFN expert weights.

Loading code SHALL skip the irrelevant family rather than load-then-drop,
to keep peak memory low.

#### Scenario: --role attention does not allocate FFN weights
- **WHEN** `larql-server --role attention --vindex /data/vindex` boots
- **THEN** RSS at the end of bootstrap SHALL be at least 50% lower than `--role all` on the same vindex
<!-- test: unbacked -->

#### Scenario: --role ffn refuses attention RPCs
- **WHEN** an `/v1/attention/decode` is sent to a `--role ffn` server
- **THEN** the server SHALL respond with HTTP 503 and a body containing "no attention weights loaded"
<!-- test: unbacked -->

#### Scenario: --role all keeps the existing single-binary semantics
- **WHEN** `larql-server` runs without an explicit `--role`
- **THEN** the boot sequence SHALL be byte-identical to a pre-change boot on the same vindex
<!-- test: unbacked -->

