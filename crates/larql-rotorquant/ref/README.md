# RotorQuant Reference Shim

This directory contains the optional Python/Triton reference harness used by
the RotorQuant round-trip tests.

- `upstream/triton_planarquant.py`
- `upstream/triton_isoquant.py`

Those two files are copied from `https://github.com/scrya-com/rotorquant.git`
at commit `fcd76768650659cad2f40a45a330612a7af8f928` (imported 2026-05-08).
`triton_reference.py` is the local shim that feeds deterministic tensors into
the upstream kernels.

The normal Rust test suite does not require Python GPU packages. To run the
reference-backed tests, point `LARQL_ROTORQUANT_REF_PY` at a Python interpreter
with `torch`, `triton`, and CUDA available:

```bash
LARQL_ROTORQUANT_REF_PY=/path/to/python \
  cargo test -p larql-rotorquant upstream_triton_reference
```
