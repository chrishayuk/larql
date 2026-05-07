## ADDED Requirements

### Requirement: KvEngine trait MUST expose cache_mut

The `KvEngine` trait SHALL gain `fn cache_mut(&mut self) ->
Option<&mut KvCache>` with a default impl returning `None`.
Engines that hold a standard `KvCache` SHALL override to return
`Some(&mut self.cache)`. Existing implementors that don't
override remain compilable and observably unchanged.

#### Scenario: engines without a KvCache return None
- **WHEN** `cache_mut` is called on `TurboQuantEngine` or `ApolloEngine`
- **THEN** the call SHALL return `None`
<!-- test: unbacked -->

#### Scenario: engines with a KvCache return Some
- **WHEN** `cache_mut` is called on `MarkovResidualEngine` or `UnlimitedContextEngine`
- **THEN** the call SHALL return `Some(&mut KvCache)` referencing the engine's standard cache
<!-- test: unbacked -->

### Requirement: RotorQuantEngine MUST decorate any inner engine

A new engine type `RotorQuantEngine { inner: Box<dyn KvEngine>, format: KvFormat }` SHALL implement `KvEngine` by delegating every method to `inner` and additionally calling `inner.cache_mut()?.quantize_layer(layer)` for every layer after each `decode_step`.

#### Scenario: decode step compresses the inner cache
- **WHEN** a `RotorQuantEngine { format: Iso3, inner: UnlimitedContextEngine }` runs `decode_step` and the inner cache holds an FP32 layer
- **THEN** after the call returns, the inner cache's `is_layer_compressed(0)` SHALL be `true`
<!-- test: unbacked -->

#### Scenario: prefill leaves layers FP32 (deferred-K)
- **WHEN** a `RotorQuantEngine` runs `prefill`
- **THEN** the inner cache's layers SHALL remain FP32 (`is_layer_compressed` returns false for every layer)
<!-- test: unbacked -->

### Requirement: EngineKind::RotorQuant MUST parse from a spec string

`EngineKind::from_name` SHALL accept spec strings of the form
`iso3:inner=unlimited-context` or
`planar3:inner=markov-rs:window=1024`, producing an
`EngineKind::RotorQuant { inner: Box<EngineKind>, format: KvFormat }`
that builds via `EngineKind::build` to a `RotorQuantEngine`. The
default inner SHALL be `unlimited-context:window=512` when no
`inner=` parameter is supplied.

#### Scenario: bare iso3 parses to default inner
- **WHEN** `EngineKind::from_name("iso3")` is called
- **THEN** the result SHALL be `Some(EngineKind::RotorQuant { inner: EngineKind::UnlimitedContext { window_size: 512 }, format: KvFormat::Iso3 })`
<!-- test: unbacked -->

#### Scenario: nested inner engine parses
- **WHEN** `EngineKind::from_name("planar3:inner=markov-rs")` is called
- **THEN** the inner SHALL parse as `MarkovResidual { window_size: None }` and the outer format SHALL be `KvFormat::Planar3`
<!-- test: unbacked -->
