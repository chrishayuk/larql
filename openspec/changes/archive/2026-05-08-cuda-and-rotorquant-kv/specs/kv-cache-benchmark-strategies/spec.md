## ADDED Requirements

### Requirement: RotorQuantStrategy joins the KV-cache strategy enum

The `kv_cache_benchmark::KvStrategy` trait family SHALL gain a new
implementation, `RotorQuantStrategy { variant: Iso3 | Planar3 | Iso4 |
Planar4 }`. It MUST conform to the same `encode` / `decode` /
`memory_bytes` contract as the existing strategies (Standard,
TurboQuant, MarkovResidual, UnlimitedContext, Apollo).

#### Scenario: RotorQuantStrategy round-trip is bit-stable across runs
- **WHEN** the same input is encoded twice with the same RotorQuantStrategy variant
- **THEN** both encoded buffers SHALL be byte-identical
<!-- test: kv_cache_benchmark::rotorquant::tests::iso3_strategy_runs_through_harness -->
<!-- test: kv_cache_benchmark::rotorquant::tests::planar3_strategy_runs_through_harness -->

#### Scenario: memory_bytes reflects the format's compression ratio
- **WHEN** `memory_bytes()` is read on Iso3 vs Standard
- **THEN** Iso3 SHALL be ≤ 12% of Standard plus per-format overhead ≤ 2%
<!-- test: kv_cache_benchmark::rotorquant::tests::memory_bytes_iso3_is_smaller_than_fp16 -->

### Requirement: Accuracy harness measures RotorQuant on Gemma 3 4B

The `kv_cache_benchmark::accuracy_suite` SHALL exercise the new
RotorQuant variants against the same Gemma 3 4B / wikitext-2 fixture
used for TurboQuant baselines, reporting top-1 match, KL divergence,
and needle-in-haystack accuracy alongside the existing rows.

#### Scenario: Iso3 produces published-paper PPL on wikitext-2
- **WHEN** the accuracy harness runs against Llama 3.1 8B with RotorQuantStrategy{Iso3}
- **THEN** the reported PPL SHALL fall within ±2% of the upstream paper's 6.91
<!-- test: kv_cache_benchmark::test_accuracy_suite::synthetic_report_accepts_upstream_ppl_measurement_with_tolerance -->
<!-- test: kv_cache_benchmark::test_accuracy_suite::synthetic_report_flags_ppl_outside_upstream_tolerance -->

#### Scenario: Comparative table includes RotorQuant rows
- **WHEN** the comparative-strategy table is generated
- **THEN** rows for `Iso3`, `Planar3`, `Iso4`, `Planar4` SHALL appear with their compression / decode / PPL columns populated
<!-- test: kv_cache_benchmark::test_accuracy_suite::synthetic_strategy_report_includes_rotorquant_rows -->
<!-- test: kv_cache_benchmark::test_accuracy_suite::synthetic_strategy_report_formats_decode_throughput_and_ppl_placeholder -->
