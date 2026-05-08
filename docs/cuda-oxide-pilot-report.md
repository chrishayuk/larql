# cuda-oxide Pilot Report

Status: Phase 1 working; Phase 2 measurements in progress.

Date: 2026-05-08

Pinned upstream: `NVlabs/cuda-oxide` commit
`6de050946cd1013335a33cf2c5144888a32efab3`.

## Environment

- GPU container: `nvidia/cuda:13.1.0-devel-ubuntu24.04`
- Rust toolchain: `nightly-2026-04-03`
- LLVM/clang: 21 from `apt.llvm.org`
- Target arch: `sm_89`
- Host requirement: run the container with `--gpus all`; `cargo-oxide`
  links against `libcuda.so.1`.

The local dev box's CUDA 12.5 headers are not enough for this pin because
`cuda-core` expects driver symbols generated from CUDA 13 headers.

## Correctness

The pilot kernel is `larql_rotorquant::cuda_oxide::iso3_dequantize_block`.
It dequantizes CPU-produced `KvFormat::Iso3` buffers and is loaded from
`larql_rotorquant.ptx`.

Validation command:

```bash
CUDA_OXIDE_PTX_DIR=$PWD cargo +nightly-2026-04-03 oxide build \
  --features cuda-oxide --arch sm_89
LARQL_CUDA_AVAILABLE=1 cargo +nightly-2026-04-03 test \
  --features cuda-oxide --test cuda_oxide_round_trip -- --test-threads=1
```

Result: passed. The 64 x 320 round-trip matched CPU dequantize with
max-element absolute difference <= 1e-3 and per-row cosine >= 0.99.

## Measurements

Measured in the CUDA 13.1 container on 2026-05-08.

| Metric | Result | Gate |
| --- | ---: | --- |
| Fresh isolated `cargo oxide build` after backend setup | 38-40 s | pass, <= 90 s |
| Cached-backend `cargo oxide build` | 11 s | pass, <= 90 s |
| cuda-oxide PTX size | 18,120 bytes | fail vs 1.5x size gate |
| Quick CUDA C reference PTX size | 8,126 bytes | reference |
| PTX size ratio | 2.23x | fail, target <= 1.5x |
| Iso3 CPU dequantize, 16,384 x 320 | 10.8414 ms/iter | reference |
| Iso3 cuda-oxide dequantize, 16,384 x 320 | 5.3844 ms/iter | pass |
| CPU throughput | 1.8015 GiB/s | reference |
| cuda-oxide throughput | 3.6274 GiB/s | pass |
| Max GPU-vs-CPU dequantize diff in benchmark | 0.00000036 | pass |

Benchmark command:

```bash
LARQL_CUDA_AVAILABLE=1 LARQL_CUDA_OXIDE_BENCH_ITERS=20 \
  cargo +nightly-2026-04-03 run --release \
  --features cuda-oxide --example cuda_oxide_iso3_bench
```

The benchmark times the current host wrapper, including device buffer
allocation and host/device copies. That makes the throughput number a
conservative end-to-end measurement, not a pure kernel-only number.

## Author Experience

What worked well:

- The kernel lives in Rust and shares the same scalar helpers as ordinary
  Rust code structure.
- Compile-time failures surfaced during `cargo oxide build`, not at server
  startup through NVRTC.
- Host launch code is type-checked through `DeviceBuffer`, `DisjointSlice`,
  and the generated kernel marker type.

What bit us:

- Device-side const arrays did not translate cleanly in this pin, so the
  codebook and rotation table had to be expressed as `match` functions.
- Artifact naming uses the crate stem (`larql_rotorquant.ptx`), while the
  package/bin name contains a hyphen.
- `cuda-host::load_kernel_module` looks in `CARGO_MANIFEST_DIR`, but
  cuda-oxide defaults PTX output next to the host binary. The pilot build
  must set `CUDA_OXIDE_PTX_DIR=$PWD`.
- The pin is effectively CUDA 13.1-only; CUDA 12.5 headers do not generate
  the expected `cuEventElapsedTime_v2` binding.

Representative CUDA C:

```cuda
unsigned elem = blockIdx.x * blockDim.x + threadIdx.x;
if (elem >= n_rows * head_dim) return;
unsigned row = elem / head_dim;
unsigned col = elem - row * head_dim;
unsigned block = col >> 2;
unsigned lane = col & 3u;
float recovered = lane == 0u ? a * r0 + c * r1 + b * r2
                : lane == 1u ? b * r0 + a * r1 + c * r2
                : lane == 2u ? c * r0 + b * r1 + a * r2
                : r3;
out[elem] = recovered * norms[row];
```

Equivalent cuda-oxide Rust:

```rust
let idx = thread::index_1d();
let elem = idx.get();
let Some(slot) = out.get_mut(idx) else { return; };
let row = elem / head_dim;
let col = elem - row * head_dim;
let block = col / 4;
let lane = col - block * 4;
let recovered = match lane {
    0 => a * rotated0 + c * rotated1 + b * rotated2,
    1 => b * rotated0 + a * rotated1 + c * rotated2,
    2 => c * rotated0 + b * rotated1 + a * rotated2,
    _ => rotated3,
};
*slot = recovered * norms[row];
```

## Decision

No Phase 3 decision yet.

The kernel is correct and faster than the CPU reference for the measured
Gemma-shaped workload, but the PTX size gate missed badly at 2.23x the quick
CUDA C reference. Phase 3 should stay blocked until:

1. the 2-week burn-in has no hard failures, and
2. we either reduce the PTX size or explicitly accept the size miss with a
   written "go" decision.
