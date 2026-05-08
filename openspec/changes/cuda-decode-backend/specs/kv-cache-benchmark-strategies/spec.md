## ADDED Requirements

### Requirement: CLI benchmark supports CUDA

`larql bench --backends cuda` SHALL run against a real vindex and report the
same timing columns as existing backends.

#### Scenario: CUDA bench row is emitted
- **WHEN** `larql bench <vindex> --backends cuda` runs on a CUDA host
- **THEN** the output SHALL include a `larql-cuda` row with prefill and decode timing
<!-- test: larql_cli::bench_cuda_manual_rtx4090 -->
