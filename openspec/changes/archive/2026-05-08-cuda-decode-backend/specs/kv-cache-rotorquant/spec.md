## ADDED Requirements

### Requirement: CUDA backend supports RotorQuant KV compression

After FP16 CUDA KV decode is working, `CudaBackend` SHALL support RotorQuant
KV quantize/dequantize and advertise `KvCompressionRotorQuant`.

#### Scenario: CUDA RotorQuant round-trip preserves direction
- **WHEN** a CUDA RotorQuant KV row is quantized and dequantized
- **THEN** cosine similarity with the original row SHALL be at least
  0.98 for 3-bit packed formats and at least 0.99 for 4-bit packed
  formats
<!-- test: larql_rotorquant::cuda_round_trip::iso3_cuda_preserves_direction -->
