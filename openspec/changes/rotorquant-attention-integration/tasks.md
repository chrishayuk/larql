## 1. Dep + struct extensions

- [ ] 1.1 Add `larql-rotorquant` to `larql-inference/Cargo.toml`.
- [ ] 1.2 Add `kv_format`, `quantized_kv` fields to `KvCache`.
- [ ] 1.3 Default-init those fields in `with_layers` and
      `with_window`.

## 2. New methods

- [ ] 2.1 `set_kv_format(format)`.
- [ ] 2.2 `quantize_layer(layer)` with FP32-slot-taken + restore-on-failure.
- [ ] 2.3 `dequantize_layer(layer)` (non-destructive; uses
      `dequantize_v_with_inverse_rotation` for V).
- [ ] 2.4 `promote_layer_to_fp32(layer)`.
- [ ] 2.5 `is_layer_compressed(layer)`.

## 3. Tests

- [ ] 3.1 `quantize_layer_no_op_when_format_unset`.
- [ ] 3.2 `quantize_then_dequantize_roundtrip_preserves_direction`.
- [ ] 3.3 `promote_layer_to_fp32_restores_layers_slot`.

## 4. Validation

- [ ] 4.1 `openspec validate rotorquant-attention-integration --strict` passes.
- [ ] 4.2 `cargo check --workspace` passes.
- [ ] 4.3 `cargo test -p larql-inference --lib attention::decode` passes (15 + 3 new).
- [ ] 4.4 `make traceability-check` and `make openspec-validate` pass.
- [ ] 4.5 Commit references the parent change.
