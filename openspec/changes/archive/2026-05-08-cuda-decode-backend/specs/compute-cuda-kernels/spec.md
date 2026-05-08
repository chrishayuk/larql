## ADDED Requirements

### Requirement: CUDA backend implements KV-cached decode

`CudaBackend` SHALL implement `DecodeBackend::decode_token` for Q4/Q6
pipeline layers. The first implementation MAY dequantize weights on the host,
but attention and KV append/read SHALL execute through CUDA helpers.

#### Scenario: decode_token returns a vector instead of None
- **WHEN** CUDA is available and `decode_token` is called with a synthetic one-layer Q4 pipeline
- **THEN** it SHALL return `Some(Vec<f32>)` with length equal to hidden size
<!-- test: larql_compute::test_cuda_decode::decode_token_one_layer_returns_hidden -->

### Requirement: CUDA backend implements Q4 prefill

`CudaBackend` SHALL implement `DecodeBackend::prefill_q4` by populating the
same KV cache later used by `decode_token`.

#### Scenario: prefill_q4 populates cache length
- **WHEN** CUDA is available and `prefill_q4` runs over a synthetic prompt
- **THEN** `kv_cache_len()` SHALL equal the prompt sequence length
<!-- test: larql_compute::test_cuda_decode::prefill_populates_kv_cache_len -->

### Requirement: CUDA decode capability bits are truthful

`CudaBackend::supports` SHALL report `DecodeToken` and `PrefillQ4` only after
the CUDA decode and prefill paths return real results.

#### Scenario: decode capability is advertised
- **WHEN** CUDA backend construction succeeds
- **THEN** `supports(DecodeToken)` and `supports(PrefillQ4)` SHALL be true
<!-- test: larql_compute::cuda::backend::tests::supports_decode_after_cuda_decode_backend -->
