## ADDED Requirements

### Requirement: RotorQuant engines join the engine registry

The KV-engine registry in `larql_inference::engines` SHALL gain four
new engines:

- `IsoQuantEngine { bits: 3 | 4 }` — wraps a KvFormat::Iso3 or Iso4 cache.
- `PlanarQuantEngine { bits: 3 | 4 }` — wraps Planar3 or Planar4.

These engines SHALL be selectable via the same engine-selection path
as `MarkovResidualEngine`, `ApolloEngine`, and `TurboQuantEngine`,
with no behavioural change to the existing engines.

#### Scenario: IsoQuantEngine 3-bit creates an Iso3 KV cache
- **WHEN** an `IsoQuantEngine { bits: 3 }` is constructed and used for prefill
- **THEN** the underlying KV cache SHALL have `format == KvFormat::Iso3`
<!-- test: unbacked -->

#### Scenario: Engine registry remains additive — existing engines unchanged
- **WHEN** an existing benchmark using `MarkovResidualEngine` is re-run
- **THEN** outputs SHALL be byte-identical to a pre-change run
<!-- test: unbacked -->

### Requirement: Engine selection prefers RotorQuant on CUDA + RotorQuant capability

Engine auto-selection SHALL prefer `IsoQuantEngine { bits: 3 }` over
the older `TurboQuantEngine` for first-token factual benchmarks
whenever the active backend reports `Capability::KvCompressionRotorQuant`
and the user has not explicitly requested an engine.

#### Scenario: CUDA box auto-selects IsoQuant
- **WHEN** an inference is launched on a CUDA backend with no engine override
- **THEN** the chosen engine SHALL be `IsoQuantEngine { bits: 3 }`
<!-- test: unbacked -->

#### Scenario: Metal box auto-selects unchanged
- **WHEN** an inference is launched on a Metal backend
- **THEN** the chosen engine SHALL be the existing default (Standard FP16) — RotorQuant is not advertised by Metal
<!-- test: unbacked -->
