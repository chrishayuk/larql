# cuda-q4k-mmvq-warp-cooperative — tasks

## 1. Cooperative-warp kernel

- [x] 1.1 Add `mul_mat_vec_q4_K_q8_1_f32_coop` to `Q4K_MMVQ_SRC`
      with grid `(rows, 1, 1)`, block `(32, 4, 1)`, and a
      cross-warp reduction via `extern __shared__ float warp_sums[]`.
- [x] 1.2 Reuse the existing `vec_dot_q4_K_q8_1_impl_vmmq` device
      function — no change to the inner dot product.
- [x] 1.3 Loop semantics: `kbx_lane = tid / 16`,
      `iqs = 2 × (tid % 16)`, increment `kbx_base += 8` per iter
      (`blocks_per_iter = 8`).

## 2. Rust wrappers

- [x] 2.1 `Q4K_MMVQ_COOP_FUNC` `OnceLock` cell + `q4k_mmvq_coop_function`
      compile/load helper.
- [x] 2.2 `matvec_device_into_with_dev_coop` — launches the new
      kernel with the right grid / block / shmem.
- [x] 2.3 Both `matvec_device_into` (host bytes) and
      `matvec_device_into_with_dev` route through the dispatcher.

## 3. Shape-aware dispatcher

- [x] 3.1 `q4k_mmvq_use_coop(rows, hidden)` — returns true when
      `n_super_blocks >= 16 || rows <= 1024`. `LARQL_CUDA_Q4K_COOP`
      env var override (`1` = force coop, `0` = force legacy,
      unset / anything else = shape-aware).

## 4. Tests + microbench

- [x] 4.1 Existing parity suite passes with the dispatcher's
      default settings (139 lib + 56 integration).
- [x] 4.2 `q4k_mmvq_legacy_vs_coop_sweep` ignored microbench
      records per-shape speedup. Run with:
      `LARQL_CUDA_AVAILABLE=1 cargo test --release -p larql-compute
      --features cuda --lib q4k_mmvq_legacy_vs_coop_sweep --
      --ignored --nocapture`.

## 5. Bench gate

- [x] 5.1 `LARQL_CUDA_AVAILABLE=1 LARQL_CUDA_PREFILL_TENSOR_CORES=1
      ./target/release/larql bench output/gemma-3-4b-it-vindex
      --backends cuda --tokens 20 --warmup 3` — 5-run avg.
- [x] 5.2 Result: 8.50 ms/tok → 8.23 ms/tok (-3.2%, 117.6 →
      121.5 tok/s).

## 6. Documentation + archive

- [x] 6.1 Per-shape speedup table in `proposal.md`.
- [x] 6.2 `LARQL_CUDA_Q4K_COOP={0,1}` env var documented.
- [ ] 6.3 Archive when reviewed.
