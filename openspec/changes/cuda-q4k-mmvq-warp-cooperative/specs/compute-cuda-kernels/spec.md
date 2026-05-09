## ADDED Requirements

### Requirement: Q4_K mmvq SHALL provide a 4-warp cooperative kernel for long-row shapes

`mul_mat_vec_q4_K_q8_1_f32_coop` SHALL launch with grid
`(rows, 1, 1)` and block `(WARP_SIZE = 32, NWARPS = 4, 1)` —
all 128 threads in a block cooperate on ONE output row.
`blocks_per_iter = 8` (each loop iteration covers 8
super-blocks across the 4 warps' 8 groups of 16 lanes). The
final reduction SHALL be warp-internal `__shfl_xor_sync`
followed by cross-warp accumulation through
`extern __shared__ float warp_sums[NWARPS]`.

#### Scenario: per-shape microbench shows expected pattern

- **WHEN** `q4k_mmvq_legacy_vs_coop_sweep` runs on a sm_89 host
  for the six Gemma 3 4B Q4_K projection shapes (q, kv, wo,
  gate, up, down)
- **THEN** the printed speedup ratios SHALL satisfy:
  - `down`  (n_super_blocks = 40): coop ≥ 1.10× legacy
  - `kv`    (rows = 1024): coop ≥ 1.10× legacy
  - `gate` / `up` (rows = 10240, n_sb = 10): coop ≤ 1.0× legacy
    (the dispatcher MUST route them to the legacy kernel)
<!-- test: unbacked -->

### Requirement: q4k_mmvq dispatcher SHALL choose coop for long-row or low-row-count shapes

`q4k_mmvq_use_coop(rows, hidden)` SHALL return true exactly when
`n_super_blocks ≥ 16 || rows ≤ 1024`. The
`LARQL_CUDA_Q4K_COOP` env var SHALL override the dispatcher:
`=1` forces coop on every shape, `=0` forces legacy on every
shape. Unset (or any other value) MUST yield the shape-aware
default.

#### Scenario: dispatcher choices on Gemma 3 4B

- **WHEN** the dispatcher is asked about Gemma 3 4B's six
  projection shapes
- **THEN** it SHALL pick:
  - q (2048 × 2560)   → legacy
  - kv (1024 × 2560)  → coop  (rows ≤ 1024)
  - wo (2560 × 2048)  → legacy
  - gate (10240 × 2560) → legacy
  - up (10240 × 2560)  → legacy
  - down (2560 × 10240) → coop (n_super_blocks = 40 ≥ 16)
<!-- test: unbacked -->
