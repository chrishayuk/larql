## 1. ServerEntry surface

- [ ] 1.1 Add `pub capabilities: Vec<String>` to `ServerEntry`.
- [ ] 1.2 `impl ServerEntry { default_capabilities, supports }`.

## 2. Routing

- [ ] 2.1 `GridState::route_for_capability(model_id, layer, capability)`.

## 3. Construction sites

- [ ] 3.1 announce handler in `grid.rs` defaults capabilities to
      `default_capabilities()`.
- [ ] 3.2 test helper `entry()` defaults the same way.

## 4. Inline tests

- [ ] 4.1 `default_capabilities_advertise_both_attention_and_expert`.
- [ ] 4.2 `route_for_capability_filters_by_capability`.
- [ ] 4.3 `route_for_capability_returns_none_when_no_match`.
- [ ] 4.4 `route_for_capability_falls_back_to_default_caps_shard`.

## 5. Validation

- [ ] 5.1 `openspec validate router-heterogeneous-shards --strict` passes.
- [ ] 5.2 `cargo test -p larql-router` passes (11 grid tests, +4 new).
- [ ] 5.3 `make traceability-check` and `make openspec-validate` pass.
- [ ] 5.4 Commit references the parent change.
