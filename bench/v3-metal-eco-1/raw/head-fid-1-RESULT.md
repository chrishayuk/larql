# Result — against PREREGISTRATION.md (sha256 9db1e383c37483b2…), computed after all three dumps

331 positions (75-token templated prompt + R's 256-token greedy continuation), vocab 201088, all finite.

| arm | mean KL(R‖X) | p99 | max | top-1 vs R |
|---|---:|---:|---:|---:|
| C conservative | 0.10192 | 1.3103 | 3.8449 | 89.12% |
| H head NVFP4 | 0.11136 | 1.4636 | 3.9343 | 90.33% |

Direct: mean KL(C‖H) 0.00937 (p99 0.0473), top-1 C vs H 92.45%.

Rule 1: 90.33 ≥ 88.12, holds. Rule 2: 0.11136 ≤ 0.15288, holds (ratio 1.093).
Guard: C's top-1 agreement with R is 89.12%, below 90%.

**VERDICT: UNINFORMATIVE by the frozen rule, not a pass.** The guard existed so a head could not pass
against a baseline already this far from R. Descriptive only: the head adds ~9% to the pack's mean KL.
