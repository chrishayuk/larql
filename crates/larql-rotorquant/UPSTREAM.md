# RotorQuant Upstream Notes

LARQL's RotorQuant implementation is inspired by `scrya-com/rotorquant`
and vendors CUDA source from the public llama.cpp TurboQuant fork.

## Imported Source

- Upstream repository: `https://github.com/johndpope/llama-cpp-turboquant.git`
- Upstream branch: `feature/planarquant-kv-cache`
- Imported commit: `08e025c06ab521e4fa9e5c08b80af57614543e53`
- Import date: `2026-05-08`
- License: upstream llama.cpp / ggml license as carried by the source files.

The parent OpenSpec task names four files, `planar3.cu`, `iso3.cu`,
`planar4.cu`, and `iso4.cu`. The upstream branch currently ships these
four conversions in one combined CUDA translation unit plus headers:

| LARQL variant | Upstream source |
|---|---|
| `Planar3` | `ggml/src/ggml-cuda/cpy-planar-iso.cu::kernel_cpy_f16_planar3` |
| `Planar4` | `ggml/src/ggml-cuda/cpy-planar-iso.cu::kernel_cpy_f16_planar4` |
| `Iso3` | `ggml/src/ggml-cuda/cpy-planar-iso.cu::kernel_cpy_f16_iso3` |
| `Iso4` | `ggml/src/ggml-cuda/cpy-planar-iso.cu::kernel_cpy_f16_iso4` |

Vendored files live under `crates/larql-rotorquant/cuda/upstream/`:

- `cpy-planar-iso.cu`
- `cpy-planar-iso.cuh`
- `planar-iso-constants.cuh`
- `set-rows-planar-iso.cuh`

## Local Patches

None to the files under `cuda/upstream/`; they are byte-for-byte
copies from the imported commit. Rust CPU reference code in
`src/cpu_ref.rs` is a separate from-scratch implementation and is not
a patched copy of the upstream CUDA code.

Local integration files live outside `cuda/upstream/`:

- `cuda/shim/common.cuh`
- `cuda/shim/ggml-common.h`
- `cuda/larql_rotorquant_kernels.cu`
- `cuda/ARCHS.txt`

## cuda-oxide Pilot Pin

- Upstream repository: `https://github.com/NVlabs/cuda-oxide.git`
- Upstream branch: `main`
- Pinned commit: `6de050946cd1013335a33cf2c5144888a32efab3`
- Pin date: `2026-05-08`
- Purpose: experimental Iso3 dequantize kernel under the
  `cuda-oxide` cargo feature.

These files let the vendored translation unit compile outside the
full llama.cpp source tree and expose C ABI wrapper symbols for Rust.

## Sync

Run:

```bash
make rotorquant-sync
```

The target fetches the recorded commit and diffs the vendored files
against upstream. A non-empty diff exits non-zero.
