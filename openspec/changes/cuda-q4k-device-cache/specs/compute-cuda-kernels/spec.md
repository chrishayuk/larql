## ADDED Requirements

### Requirement: CUDA caches packed Q4_K weights on device

`CudaBackend` SHALL cache immutable packed Q4_K weight buffers in device memory
and reuse the cached device buffer on repeated `q4k_matvec` calls for the same
host weight slice. The direct Q4_K kernel SHALL compute more than one output
row per CUDA block for large row counts. The cache MUST be backend-local and
MUST preserve the existing host-dequant debug fallback.

#### Scenario: Repeated Q4_K matvec reuses cached device weights
- **WHEN** `CudaBackend::q4k_matvec` is called twice with the same packed Q4_K weight slice and compatible input vectors
- **THEN** the second call SHALL reuse the cached device weight buffer and return a CPU-reference-equivalent result
<!-- test: larql_compute::test_cuda_q4::q4k_matvec_reuses_device_cache -->

#### Scenario: Host-dequant fallback bypasses the device cache
- **WHEN** `LARQL_CUDA_Q4K_HOST_DEQUANT=1` is set
- **THEN** `CudaBackend::q4k_matvec` SHALL use the host-dequant fallback without requiring a cached device buffer
<!-- test: larql_compute::test_cuda_q4::q4k_matvec_host_dequant_fallback_parity -->

### Requirement: CUDA benchmark loads Q4_K LM-head weights

`larql bench --backends cuda` SHALL load `lm_head_q4.bin` when it is present in
a Q4K vindex so the LM-head stage uses the same backend Q4_K matvec route as
generation.

#### Scenario: Q4K bench initializes quantized LM-head storage
- **WHEN** `larql bench <q4k-vindex> --backends cuda` loads a vindex containing `lm_head_q4.bin`
- **THEN** the benchmark SHALL populate the vindex Q4_K LM-head storage before decode begins
<!-- test: larql_cli::bench_loads_lm_head_q4_for_cuda_manual_rtx4090 -->

### Requirement: CUDA caches dequantized Q6_K weights on device

`CudaBackend` SHALL cache dequantized Q6_K matrices as f32 device buffers after
the first Q6_K matvec for a given immutable host weight slice. Repeated Q6_K
matvec calls for the same weight slice MUST reuse the cached device buffer and
MUST avoid repeated CPU dequantization and f32 host-to-device upload.

#### Scenario: Repeated Q6_K matvec reuses cached device weights
- **WHEN** `CudaBackend::q6k_matvec` is called twice with the same packed Q6_K weight slice and compatible input vectors
- **THEN** the second call SHALL reuse the cached dequantized device weight buffer and return a CPU-reference-equivalent result
<!-- test: larql_compute::test_cuda_q4::q6k_matvec_reuses_device_cache -->
