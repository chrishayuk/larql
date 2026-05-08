## ADDED Requirements

### Requirement: Attention service can select CUDA decode

The attention service SHALL route prefill/decode through CUDA when the selected
backend is CUDA and `DecodeToken`/`PrefillQ4` are supported.

#### Scenario: GPU container selects CUDA decode
- **WHEN** the GPU container starts with `LARQL_BACKEND=cuda`
- **THEN** attention-service decode SHALL use the CUDA backend rather than CPU fallback
<!-- test: larql_server::attention_cuda_manual_container_smoke -->
