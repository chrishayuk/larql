# RotorQuant CUDA Strategy

This directory is reserved for CUDA implementation notes and, if needed,
vendored kernel sources.

The current implementation keeps the public RotorQuant API in safe Rust with
a CPU reference path used by tests. Feature-gated CUDA support lives in
`src/cuda.rs` and uses `cudarc` rather than vendoring upstream CUDA files.

If the project later vendors kernels from the upstream llama.cpp
`feature/planarquant-kv-cache` branch, keep the imported files in this
directory and update `../UPSTREAM.md` with the source commit and local patch
notes.
