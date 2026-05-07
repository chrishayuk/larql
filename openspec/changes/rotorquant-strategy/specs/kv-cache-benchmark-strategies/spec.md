## ADDED Requirements

### Requirement: RotorQuantStrategy MUST implement the KvStrategy trait

`kv_cache_benchmark::rotorquant::RotorQuantStrategy` SHALL implement
`KvStrategy` for every supported `KvFormat` (Iso3, Planar3, Iso4,
Planar4). Each format MUST have a constructor (`iso3()`,
`planar3()`, `iso4()`, `planar4()`) returning a strategy instance
with a unique `name()`.

#### Scenario: Iso3 strategy runs through the harness
- **WHEN** `run_strategy_benchmark(&RotorQuantStrategy::iso3(), config, seq_len, rng)` is called with a synthetic Gemma-shaped config
- **THEN** the call SHALL return a `StrategyResult` with `metrics.cosine_sim > 0.0` (no panic, no NaN)
<!-- test: kv_cache_benchmark::rotorquant::tests::iso3_strategy_runs_through_harness -->

#### Scenario: Planar3 strategy runs through the harness
- **WHEN** `run_strategy_benchmark(&RotorQuantStrategy::planar3(), config, seq_len, rng)` is called
- **THEN** the call SHALL return a result with positive cosine similarity
<!-- test: kv_cache_benchmark::rotorquant::tests::planar3_strategy_runs_through_harness -->

### Requirement: memory_bytes MUST be smaller than Standard FP16

`RotorQuantStrategy::memory_bytes(config, seq_len)` SHALL be strictly
less than `StandardKv::memory_bytes(config, seq_len)` for every
combination of model config and sequence length supported by the
benchmark. The exact ratio depends on `head_dim` (small head_dims
have rotation-index overhead near the codes); production head_dims
≥ 128 approach the upstream paper's 10× ratio.

#### Scenario: iso3 memory below fp16 baseline
- **WHEN** `iso3.memory_bytes(config, 1024)` and `StandardKv.memory_bytes(config, 1024)` are compared at synthetic head_dim=32
- **THEN** the iso3 result SHALL be strictly less than the FP16 result
<!-- test: kv_cache_benchmark::rotorquant::tests::memory_bytes_iso3_is_smaller_than_fp16 -->
