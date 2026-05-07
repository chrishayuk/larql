## 1. KvEngine trait extension

- [ ] 1.1 Add `fn cache_mut(&mut self) -> Option<&mut KvCache>`
      with default impl returning `None`.
- [ ] 1.2 Override on `MarkovResidualEngine` returning the
      window's cache.
- [ ] 1.3 Override on `UnlimitedContextEngine` returning
      `current_window_kv.as_mut()`.
- [ ] 1.4 (Defaults stand on `TurboQuantEngine` and
      `ApolloEngine`.)

## 2. RotorQuantEngine decorator

- [ ] 2.1 New module `engines/kv_engines/rotorquant.rs` with
      `pub struct RotorQuantEngine { inner: Box<dyn KvEngine>,
      format: KvFormat }`.
- [ ] 2.2 `KvEngine` impl that delegates every method to `inner`
      and adds a post-decode `quantize_layer` sweep over every
      layer.
- [ ] 2.3 `info()` decorates the inner engine's name with
      `+iso3` / `+planar3` etc.

## 3. EngineKind variant + parsing

- [ ] 3.1 Add `EngineKind::RotorQuant { inner: Box<EngineKind>, format: KvFormat }`.
- [ ] 3.2 Extend `EngineKind::from_name` to recognise `iso3` /
      `iso4` / `planar3` / `planar4` as outer formats with
      optional `inner=<spec>` parameter (default
      `unlimited-context:window=512`).
- [ ] 3.3 `EngineKind::build` constructs a `RotorQuantEngine`
      around the inner engine.

## 4. Tests

- [ ] 4.1 `cache_mut_returns_none_for_turboquant_and_apollo`.
- [ ] 4.2 `cache_mut_returns_some_for_markov_and_unlimited`.
- [ ] 4.3 `decode_step_compresses_inner_cache` — using a mock
      inner engine that has a `KvCache` field, verify
      `is_layer_compressed` flips to true after `decode_step`.
- [ ] 4.4 `prefill_leaves_layers_fp32`.
- [ ] 4.5 `engine_kind_iso3_parses_with_default_inner`.
- [ ] 4.6 `engine_kind_planar3_inner_markov_rs_parses`.

## 5. Validation

- [ ] 5.1 `openspec validate engine-rotorquant-auto-compress --strict` passes.
- [ ] 5.2 `cargo check --workspace` passes.
- [ ] 5.3 `cargo test -p larql-inference --lib engines` passes.
- [ ] 5.4 `make traceability-check` and `make openspec-validate` pass.
