# cuda-oxide migration — tasks

## Phase 1 — Pilot

### 1. Toolchain + build dependencies

- [ ] 1.1 Pick a cuda-oxide commit hash to pin against. Record
      in `crates/larql-rotorquant/UPSTREAM.md` once the pilot
      lands. The pinned commit MUST be on `main` and have
      passing CI on the upstream side.
- [ ] 1.2 Document `nightly-2026-04-03` as the companion
      toolchain in a comment in `rust-toolchain.toml`. Do NOT
      change the workspace default.
- [ ] 1.3 Add a `cuda-oxide-doctor` Makefile target that
      shells out to `cargo +nightly-2026-04-03 oxide doctor`
      and prints actionable error messages.

### 2. Cargo feature wiring

- [ ] 2.1 Add `cuda-oxide` feature to
      `crates/larql-rotorquant/Cargo.toml`:
      ```toml
      cuda-oxide = ["dep:cuda-core", "dep:cuda-host"]
      ```
      with `cuda-core` and `cuda-host` declared as optional
      git deps pinned to the chosen upstream commit.
- [ ] 2.2 Add a `compile_error!` macro at the crate root that
      fires when `cuda` and `cuda-oxide` are both enabled —
      they're mutually exclusive.
- [ ] 2.3 Workspace-level docs in
      `Cargo.toml`: declare `cuda-oxide` as an unstable
      feature alongside `cuda`.

### 3. Pilot kernel: Iso3 quantize

- [ ] 3.1 New module `crates/larql-rotorquant/src/cuda_oxide/mod.rs`
      gated behind `#[cfg(feature = "cuda-oxide")]`. Module
      tree:
      ```
      cuda_oxide/
        mod.rs            // public API: quantize_iso3
        kernels.rs        // #[kernel] iso3_quantize_block
        device_tables.rs  // codebook + rotation table consts
      ```
- [ ] 3.2 Write `#[kernel] fn iso3_quantize_block(...)`:
      - input: `&[f32; head_dim]` row, `head_dim` is multiple
        of 4 (Iso block size).
      - output: per-block `(rotation_idx: u16, codes: [u8; 4])`,
        plus row L2 norm.
      - logic: mirror `cpu_ref.rs::quantize` Iso3 branch
        verbatim. Use `thread::index_2d()` for `(row, block)`
        indexing.
- [ ] 3.3 Write the host-side launcher:
      `pub fn quantize_iso3_oxide(ctx: &CudaContext, k: &[f32],
       n_rows: usize, head_dim: usize) -> QuantizedKv`.
      Allocates three `DeviceBuffer`s (codes / norms /
      rotation_indices), launches the kernel via
      `cuda_launch!`, copies results back, packages into the
      same `QuantizedKv` struct the CPU reference produces.

### 4. Tests

- [ ] 4.1 `crates/larql-rotorquant/tests/cuda_oxide_round_trip.rs`
      — gated on `cfg(feature = "cuda-oxide")` and
      `LARQL_CUDA_AVAILABLE=1`. Compares
      `cuda_oxide::quantize_iso3` output → `cpu_ref::dequantize`
      against the original synthetic 64 × 320 input. Asserts
      cosine ≥ 0.99 per row.
- [ ] 4.2 Skip with a clear message when `LARQL_CUDA_AVAILABLE`
      is not set. Don't fail the build on CPU-only hosts.
- [ ] 4.3 Cross-implementation parity (when
      `rotorquant-cuda-kernels` ships its cudarc variant in
      parallel): same input through both backends, max-element
      diff ≤ 1e-3.

### 5. Container image (GPU only)

- [ ] 5.1 Update `deploy/docker/Dockerfile.gpu`:
      ```dockerfile
      ARG ENABLE_CUDA_OXIDE=0
      RUN if [ "$ENABLE_CUDA_OXIDE" = "1" ]; then \
          curl -sSf https://apt.llvm.org/llvm.sh | bash -s -- 21 && \
          apt-get install -y libclang-common-21-dev clang-21 && \
          rustup toolchain install nightly-2026-04-03 && \
          rustup component add rust-src rustc-dev --toolchain nightly-2026-04-03 && \
          cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git \
              --rev <PINNED_COMMIT> cargo-oxide; \
        fi
      ```
- [ ] 5.2 Update `deploy/docker/docker-compose.yml` to expose
      `ENABLE_CUDA_OXIDE` as a build arg.
- [ ] 5.3 Document in `deploy/docker/README.md`: when to use
      the flag, expected image-size impact (~400 MB).

### 6. Make targets

- [ ] 6.1 `make cuda-oxide-pilot` — builds the rotorquant
      crate with `--features cuda-oxide`, runs the round-trip
      test if `LARQL_CUDA_AVAILABLE=1`. Uses the nightly
      toolchain transparently.
- [ ] 6.2 `make cuda-oxide-doctor` — runs `cargo oxide doctor`
      and reports missing toolchain pieces.

## Phase 2 — Evaluation (decision-only; no code)

- [ ] 7.1 Build cost: clean `cargo build --features cuda-oxide`
      on the dev box. Pass: ≤ 90 s.
- [ ] 7.2 PTX size: report cuda-oxide PTX bytes vs hand-written
      reference (from `rotorquant-cuda-kernels` if it shipped
      in parallel, otherwise from a quick CUDA C benchmark).
      Pass: ≤ 1.5×.
- [ ] 7.3 Throughput: bench Iso3 quantize on Gemma 4B
      head shape, RTX 4090. Pass: ≥ 0.75× CPU reference (i.e.
      cuda-oxide GPU is ≥ 25% faster than CPU; this is the
      floor — speed-of-light is much higher).
- [ ] 7.4 Stability: 2-week burn-in. Zero hard failures in CI;
      no upstream regressions that block our pinned commit.
- [ ] 7.5 Author experience: write up the kernel-authoring
      experience in `docs/cuda-oxide-pilot-report.md`. Include
      a Rust-vs-CUDA-C side-by-side for one representative
      block of code.
- [ ] 7.6 Decision: ship Phase 3 (yes/no/abort). Document the
      decision in the same report. If "no" or "abort", revert
      Phase 1 and close the change.

## Phase 3 — Conditional rollout (only if Phase 2 passes)

### 8. Remaining RotorQuant formats

- [ ] 8.1 Iso4 quantize (4-bit, same 4D rotation as Iso3).
- [ ] 8.2 Planar3 quantize (3-bit, 2D Givens rotation).
- [ ] 8.3 Planar4 quantize (4-bit, 2D Givens rotation).
- [ ] 8.4 The dequantize side for each (4 more kernels).
- [ ] 8.5 Cross-format parity tests: every (format, kind) combo
      cuda-oxide ↔ CPU within 1e-3 max-element.

### 9. Fused softmax / decode-attention kernel

- [ ] 9.1 Port the NVRTC-compiled fused-softmax kernel from
      `larql-compute/src/cuda/attn.rs` to cuda-oxide.
- [ ] 9.2 Keep the cuBLAS GEMM calls on cudarc — they wrap the
      cuda-oxide softmax, not replace it.
- [ ] 9.3 Existing `test_cuda_attn` parity tests must pass
      against the cuda-oxide variant.

### 10. Capability bit + docs

- [ ] 10.1 Flip `Capability::CudaOxide` (new) on `CudaBackend`
      when the feature is enabled.
- [ ] 10.2 Refresh `docs/cuda-rotorquant-status.md` with the
      Phase 3 results: throughput numbers, cold-start latency,
      PTX size on disk.
- [ ] 10.3 Mark this OpenSpec change ready for archive once
      the throughput acceptance bar is hit on the GPU box.

## Risk-gated checkpoints

- After 1.x: verify the doctor target works locally before
  starting any kernel work.
- After 3.x: a single working kernel is the go/no-go signal
  for Phase 2 evaluation. If it can't compile, abort.
- After Phase 2: explicit yes/no decision, written up in the
  pilot report. **Do NOT start Phase 3 without that document.**
