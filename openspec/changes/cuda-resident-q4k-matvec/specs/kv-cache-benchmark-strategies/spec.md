## MODIFIED Requirements

### Requirement: CLI benchmark supports CUDA

`larql bench --backends cuda` SHALL run against a real vindex and report the
same timing columns as existing backends. When CUDA quantized matvec kernels are
enabled, the benchmark output MUST expose enough stage timing to identify
prefill, decode forward, LM-head, and tokens/sec. A CUDA benchmark pass after
this change MUST complete successfully on the RTX host and record whether the
direct Q4_K path improves over the 9.25 s/token baseline captured before the
change.

#### Scenario: CUDA bench row is emitted
- **WHEN** `larql bench <vindex> --backends cuda` runs on a CUDA host
- **THEN** the output SHALL include a `larql-cuda` row with prefill and decode timing
<!-- test: larql_cli::bench_cuda_manual_rtx4090 -->

#### Scenario: CUDA bench records direct Q4_K result
- **WHEN** `larql bench output/gemma-3-4b-it-vindex --backends cuda --tokens 20 --warmup 3 --verbose` completes on the RTX host
- **THEN** the recorded result SHALL include prefill time, decode ms/token, tokens/sec, GPU forward time, LM-head time, and a comparison against the previous 9.25 s/token baseline
<!-- test: larql_cli::bench_cuda_direct_q4k_manual_rtx4090 -->
