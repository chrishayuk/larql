## ADDED Requirements

### Requirement: Q6_K × Q8_1 mmvq kernel SHALL match the existing f32 GEMV

A new `q6k_mmvq::matvec_device` SHALL produce output that agrees with the existing f32-cached Q6_K GEMV path (`q6k_matvec_device` via `with_q6k_f32_device_buf`) to max-element absolute difference ≤ 1e-3 on Q8_1-dequantised input. The kernel body SHALL be ported close-to-verbatim from upstream `vec_dot_q6_K_q8_1_impl_mmvq` (MIT, ggml authors); provenance SHALL be recorded in the NVRTC source comment.

#### Scenario: mmvq matches the f32 path on Q8_1-dequantised input within 1e-3

- **WHEN** a random Q6_K packed weight `[rows=2560, hidden=10240]` (Gemma 3 4B FFN down shape) is multiplied by a random f32 input that has been quantised to Q8_1 and dequantised back to f32; the dequantised f32 is fed to the existing f32-cached `q6k_matvec_device`, and the same Q8_1 form is fed to `q6k_mmvq::matvec_device`
- **THEN** the two output `Vec<f32>`s SHALL agree to max-element absolute difference ≤ 1e-3
<!-- test: unbacked -->

### Requirement: Q6_K matvec dispatch SHALL be runtime-selectable

`LARQL_CUDA_Q6K_MMVQ` env var SHALL select the Q6_K matvec kernel: `0` forces the existing f32 GEMV (with the dequantised-f32 device cache), `1` (the default) routes through the new mmvq path. The new code SHALL be additive — the f32 path stays compiled and reachable.

#### Scenario: env var routes between mmvq and f32 paths

- **WHEN** `LARQL_CUDA_Q6K_MMVQ=0` is set in the environment and the bench is run
- **THEN** the existing f32 GEMV path SHALL be invoked, producing the same numerical results as before this change
<!-- test: unbacked -->

### Requirement: proj_down profile bucket SHALL drop materially after the port

`proj_down` profile bucket SHALL drop to ≤ 1.5 ms (down from the post-`cuda-attn-rope-hoist` 4.06 ms) on the dev-box bench after this change ships. **Actual**: 1.58 ms — within 5.3% of target, well inside the 25% miss tolerance.

#### Scenario: bench cleared at acceptance OR profile-documented on miss

- **WHEN** `LARQL_CUDA_AVAILABLE=1 LARQL_CUDA_DECODE_PROFILE=1 ./target/release/larql bench output/gemma-3-4b-it-vindex --backends cuda --tokens 20 --warmup 3 --verbose` is run on the dev box after this change lands
- **THEN** EITHER `proj_down` SHALL be ≤ 1.5 ms AND `decode ms/token` ≤ 10 (acceptance hit), OR the change's `proposal.md` SHALL contain a profile-and-document write-up identifying the residual bottleneck
<!-- test: unbacked -->
