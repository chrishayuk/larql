## Why

The CUDA backend now has cuBLAS matmul/GEMV, correctness-first Q4/Q6
matvec, and fused decode-attention helpers, but it still does not implement
`DecodeBackend`. As a result `larql bench` and the attention service cannot
select CUDA for KV-cached inference even on the RTX 4090 host.

This change turns the existing cudarc path into the production path for CUDA
decode before considering cuda-oxide. The first milestone is correctness, not
peak throughput: host dequant and host-visible KV are acceptable until the
trait contract is live and benchmarkable.

## What Changes

- Implement CUDA `DecodeBackend` for `prefill_q4` and `decode_token`.
- Add `larql bench --backends cuda`.
- Route attention-service prefill/decode through CUDA when available.
- Add RotorQuant CUDA KV compression after the FP16 KV path works.

## Impact

- Affected crates: `larql-compute`, `larql-cli`, `larql-server`,
  `larql-rotorquant`.
- The existing `cuda-oxide-migration` pilot remains independent and is not a
  prerequisite for this path.
- Initial CUDA decode may be slower than the final target because Q4/Q6
  weights are still host-dequantized before cuBLAS calls.
