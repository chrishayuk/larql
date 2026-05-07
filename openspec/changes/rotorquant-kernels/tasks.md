## 1. New workspace member

- [ ] 1.1 `crates/larql-rotorquant/Cargo.toml` declaring `cudarc`
      optional under the `cuda` feature, default features off.
- [ ] 1.2 Workspace `Cargo.toml` adds `larql-rotorquant` to both
      `members` and `default-members`.

## 2. Public API + reference impl

- [ ] 2.1 `src/format.rs` — `KvFormat` + `QuantizedKv` types.
- [ ] 2.2 `src/error.rs` — `RotorQuantError` typed errors.
- [ ] 2.3 `src/cpu_ref.rs` — full reference implementation:
      absmax row scaling, planar/iso rotation tables, brute-force
      best-rotation search, bit-packed code emission, dequantize.
- [ ] 2.4 `src/lib.rs` — public API: `quantize_k`, `quantize_v`,
      `dequantize_k`, `dequantize_v_with_inverse_rotation`. All
      route through `cpu_ref`.
- [ ] 2.5 `src/cuda.rs` — feature-flagged stub (no body).

## 3. Tests

- [ ] 3.1 `tests/round_trip.rs` with 9 tests:
      - `planar3_round_trip_k` / `_v`
      - `planar4_round_trip_k`
      - `iso3_round_trip_k` / `_v`
      - `iso4_round_trip_k`
      - `iso3_gemma4b_head_round_trip` (head_dim=320)
      - `iso3_v_round_trip_recovers_original_not_rotated`
      - `head_dim_divisibility_is_enforced`
- [ ] 3.2 Doctest on the crate-level example in `lib.rs`.

## 4. Validation

- [ ] 4.1 `openspec validate rotorquant-kernels --strict` passes.
- [ ] 4.2 `cargo check -p larql-rotorquant` passes.
- [ ] 4.3 `cargo check --workspace` passes.
- [ ] 4.4 `cargo test -p larql-rotorquant` passes (9 tests + 1 doctest).
- [ ] 4.5 `make traceability-check` and `make openspec-validate` pass.
- [ ] 4.6 Commit references the parent change in subject.
