# RotorQuant Upstream Notes

LARQL's RotorQuant implementation is inspired by:

- Repository: `https://github.com/scrya-com/rotorquant`
- Related llama.cpp branch: `feature/planarquant-kv-cache`

Current status:

- The Rust API and CPU reference path are implemented from scratch in this crate.
- CUDA support is feature-gated behind `cuda` and uses `cudarc`.
- No upstream `.cu` files are vendored in this crate at this time.

If vendored kernels are introduced later, record the exact source URL,
commit SHA, import date, license, and any local patches in this file.
