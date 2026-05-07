## ADDED Requirements

### Requirement: `larql-rotorquant` crate exposes the parent-change API

The new workspace member `larql-rotorquant` SHALL export the public
API surface declared in the parent change's `kv-cache-rotorquant`
capability: `KvFormat` enum, `QuantizedKv` struct, `quantize_k`,
`quantize_v`, `dequantize_k`, `dequantize_v_with_inverse_rotation`,
`RotorQuantError` typed errors. The crate MUST compile without the
`cuda` feature.

#### Scenario: crate compiles on default features
- **WHEN** `cargo check -p larql-rotorquant` runs on a host with no CUDA toolkit
- **THEN** the build SHALL succeed
<!-- test: unbacked -->

#### Scenario: public API matches the parent-change declaration
- **WHEN** a caller writes `use larql_rotorquant::{KvFormat, QuantizedKv, quantize_k, quantize_v, dequantize_k, dequantize_v_with_inverse_rotation, RotorQuantError};`
- **THEN** the import SHALL resolve without error
<!-- test: unbacked -->

### Requirement: Round-trip parity for all four formats on K and V

Every `(format, kind)` combo where `format ∈ {Planar3, Planar4, Iso3, Iso4}` and `kind ∈ {K, V}` SHALL round-trip a synthetic input with cosine similarity ≥ 0.95.

#### Scenario: Planar3 K round-trip
- **WHEN** synthetic 64×32 input is quantised then dequantised through `KvFormat::Planar3`
- **THEN** cosine similarity SHALL be ≥ 0.95
<!-- test: larql_rotorquant::round_trip::planar3_round_trip_k -->

#### Scenario: Planar4 K round-trip
- **WHEN** synthetic 64×32 input is quantised then dequantised through `KvFormat::Planar4`
- **THEN** cosine similarity SHALL be ≥ 0.95
<!-- test: larql_rotorquant::round_trip::planar4_round_trip_k -->

#### Scenario: Iso3 K round-trip
- **WHEN** synthetic 64×32 input is quantised then dequantised through `KvFormat::Iso3`
- **THEN** cosine similarity SHALL be ≥ 0.95
<!-- test: larql_rotorquant::round_trip::iso3_round_trip_k -->

#### Scenario: Iso4 K round-trip
- **WHEN** synthetic 64×32 input is quantised then dequantised through `KvFormat::Iso4`
- **THEN** cosine similarity SHALL be ≥ 0.95
<!-- test: larql_rotorquant::round_trip::iso4_round_trip_k -->

#### Scenario: Planar3 V round-trip with inverse rotation
- **WHEN** synthetic 64×32 input is quantised as V then dequantised via `dequantize_v_with_inverse_rotation`
- **THEN** cosine similarity SHALL be ≥ 0.95
<!-- test: larql_rotorquant::round_trip::planar3_round_trip_v -->

#### Scenario: Iso3 V round-trip with inverse rotation
- **WHEN** synthetic 64×32 input is quantised as V then dequantised via `dequantize_v_with_inverse_rotation`
- **THEN** cosine similarity SHALL be ≥ 0.95
<!-- test: larql_rotorquant::round_trip::iso3_round_trip_v -->

#### Scenario: Iso3 round-trip on Gemma 4B head shape
- **WHEN** a 32-row, head_dim=320 input is round-tripped through `KvFormat::Iso3`
- **THEN** cosine similarity SHALL be ≥ 0.95
<!-- test: larql_rotorquant::round_trip::iso3_gemma4b_head_round_trip -->

### Requirement: V dequantize MUST recover original-space values

`dequantize_v_with_inverse_rotation` SHALL produce values in the
unrotated space (cosine ≥ 0.95 with the original input). This
invariant guards against the upstream commit `6e5a4aa` bug where
forward rotation was mistakenly applied on V dequantize, leaving
the cache in rotated-space and producing PPL = 15K instead of 7.05.

#### Scenario: V dequant recovers original (not rotated) values
- **WHEN** a synthetic V is quantised then `dequantize_v_with_inverse_rotation` is called
- **THEN** the recovered tensor SHALL have cosine similarity ≥ 0.95 with the input
<!-- test: larql_rotorquant::round_trip::iso3_v_round_trip_recovers_original_not_rotated -->

### Requirement: head_dim divisibility is enforced

The crate SHALL reject inputs whose `head_dim` is not divisible by
the format's block size with a typed error
(`RotorQuantError::HeadDimNotDivisible`). No silent corruption.

#### Scenario: Iso3 rejects head_dim=33
- **WHEN** `quantize_k(Iso3, ..., n_rows=4, head_dim=33)` is called
- **THEN** the call SHALL return `Err(RotorQuantError::HeadDimNotDivisible { .. })`
<!-- test: larql_rotorquant::round_trip::head_dim_divisibility_is_enforced -->
