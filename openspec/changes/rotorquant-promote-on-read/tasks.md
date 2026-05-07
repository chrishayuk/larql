## 1. KvCache API extension

- [ ] 1.1 Change `get_layer` signature from `&self` to `&mut self`
      (audit existing call sites).
- [ ] 1.2 Add `get_layer_lazy(&self, layer) -> Option<&SharedKV>`
      preserving the original immutable semantics.
- [ ] 1.3 Add `promote_on_read_count: AtomicU64` field; increment
      on each successful promote.

## 2. Auto-promote logic

- [ ] 2.1 Inside `get_layer`, on FP32 miss, check
      `quantized_kv[layer]`; if `Some`, call `dequantize_layer`
      and populate the FP32 slot.
- [ ] 2.2 Increment `promote_on_read_count` after a successful
      promote (not on cache hits where the FP32 slot is already
      populated).

## 3. Symmetric clear

- [ ] 3.1 `clear_layer(layer)` clears both `layers[layer]` and
      `quantized_kv[layer]`.

## 4. Tests

- [ ] 4.1 `get_layer_returns_fp32_after_compress`.
- [ ] 4.2 `get_layer_caches_promoted_copy` (counter increments by 1, not 2).
- [ ] 4.3 `get_layer_lazy_never_promotes`.
- [ ] 4.4 `clear_layer_erases_both_storages`.

## 5. Migration audit

- [ ] 5.1 Find every `cache.get_layer(...)` call site in the
      workspace; change `&` borrow to `&mut` or switch to
      `get_layer_lazy` based on intent.
- [ ] 5.2 Snapshot serialisation in
      `attention-service-routes` uses `get_layer_lazy` (read
      compressed bytes directly without promote).

## 6. Validation

- [ ] 6.1 `openspec validate rotorquant-promote-on-read --strict` passes.
- [ ] 6.2 `cargo check --workspace` passes.
- [ ] 6.3 `cargo test -p larql-inference --lib attention::decode` passes.
- [ ] 6.4 `make traceability-check` and `make openspec-validate` pass.
