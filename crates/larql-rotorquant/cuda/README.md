# RotorQuant CUDA Strategy

`upstream/` contains byte-for-byte copies of the RotorQuant CUDA kernels
imported from the recorded llama.cpp TurboQuant commit. `../UPSTREAM.md`
records the source URL, branch, commit, import date, and sync command.

`shim/` contains the small local compatibility headers needed to compile the
vendored translation unit outside of the full llama.cpp tree. The build
wrapper `larql_rotorquant_kernels.cu` includes the vendored source and exposes
stable C ABI launcher symbols for the Rust FFI layer.

The CUDA build is feature-gated. `cargo check -p larql-rotorquant --features
cuda` finds `NVCC`, `CUDA_HOME`, `CUDA_PATH`, `nvcc` on `PATH`, or the highest
`/usr/local/cuda-*` toolkit. Architectures default to `ARCHS.txt` and can be
overridden with `LARQL_CUDA_ARCH`, for example:

```bash
LARQL_CUDA_ARCH=89 cargo test -p larql-rotorquant --features cuda
```
