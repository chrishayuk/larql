## ADDED Requirements

### Requirement: CUDA Q4_K matvec uses a direct packed-weight kernel

`CudaBackend::q4k_matvec` SHALL use a CUDA kernel that reads packed Q4_K weight
blocks directly and accumulates f32 outputs without first materializing the
entire weight matrix as host f32. The implementation MUST retain a debug
fallback to the existing host-dequant + cuBLAS path.

#### Scenario: Direct Q4_K matvec matches CPU reference at FFN dimensions
- **WHEN** a 10240x2560 Q4_K matrix is multiplied by a 2560-element f32 vector on the CUDA backend
- **THEN** the direct CUDA result SHALL match the CPU reference with cosine similarity >= 0.9999 and max absolute difference <= 1e-3
<!-- test: larql_compute::test_cuda_q4::q4k_matvec_ffn_gate_parity -->

#### Scenario: Direct Q4_K matvec matches CPU reference at LM-head dimensions
- **WHEN** a production-like Q4_K LM-head matrix is multiplied by an f32 residual vector on the CUDA backend
- **THEN** the direct CUDA result SHALL match the CPU reference with cosine similarity >= 0.9999 and max absolute difference <= 1e-3
<!-- test: larql_compute::test_cuda_q4::q4k_matvec_lm_head_parity -->

#### Scenario: Host-dequant fallback remains available
- **WHEN** the CUDA Q4_K host-dequant fallback is explicitly requested
- **THEN** `CudaBackend::q4k_matvec` SHALL use the previous host-dequant + cuBLAS path and return a CPU-reference-equivalent result
<!-- test: larql_compute::test_cuda_q4::q4k_matvec_host_dequant_fallback_parity -->

### Requirement: CUDA decode routes Q4_K projections through quant matvec

`CudaBackend` SHALL route compatible Q4_K decode projections through
`QuantMatVec::q4k_matvec`. This MUST avoid locally dequantizing the full
matrix on the host when the direct CUDA Q4_K path can handle the projection
shape.

#### Scenario: Decode helper dispatches Q4_K through direct matvec
- **WHEN** a synthetic Q4_K decode projection is evaluated by the CUDA decode backend
- **THEN** the backend SHALL produce the same vector as the CPU reference while recording that the Q4_K direct path was selected
<!-- test: larql_compute::test_cuda_decode::decode_q4k_projection_uses_quant_matvec -->
