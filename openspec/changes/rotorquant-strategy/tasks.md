## 1. New strategy module

- [ ] 1.1 `crates/kv-cache-benchmark/src/rotorquant.rs` defining
      `RotorQuantStrategy` + four constructors.
- [ ] 1.2 KvStrategy impl: encode flattens, calls
      `larql_rotorquant::quantize_k` + `quantize_v`, serialises both
      to the binary wire format. Decode unwinds via
      `dequantize_k` + `dequantize_v_with_inverse_rotation` and
      reshapes back to `Vec<Vec<f32>>`.
- [ ] 1.3 `memory_bytes` derives the analytical compressed
      footprint per format (codes + norms + rotation indices,
      double for K+V).

## 2. Cargo + lib wiring

- [ ] 2.1 `crates/kv-cache-benchmark/Cargo.toml` adds
      `larql-rotorquant` to `[dependencies]`.
- [ ] 2.2 `crates/kv-cache-benchmark/src/lib.rs` adds `pub mod rotorquant;`.

## 3. Inline tests

- [ ] 3.1 `iso3_strategy_runs_through_harness` — exercises
      `run_strategy_benchmark` end-to-end on a small synthetic
      config; asserts positive cosine.
- [ ] 3.2 `planar3_strategy_runs_through_harness` — same for Planar3.
- [ ] 3.3 `memory_bytes_iso3_is_smaller_than_fp16` — analytical
      footprint comparison.

## 4. Validation

- [ ] 4.1 `openspec validate rotorquant-strategy --strict` passes.
- [ ] 4.2 `cargo check -p kv-cache-benchmark` passes.
- [ ] 4.3 `cargo test -p kv-cache-benchmark --lib rotorquant` passes (3 tests).
- [ ] 4.4 `make traceability-check` and `make openspec-validate` pass.
- [ ] 4.5 Commit references the parent change.
