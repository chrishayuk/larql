# larql-compute

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

CPU numerical substrate and shared backend traits. CPU forward-pass math,
attention, normalization, quantized kernels and per-layer state dispatch live
here. Metal implements the shared traits in [larql-compute-metal](../larql-compute-metal/README.md).
Engine sessions, tokenizers, distributed routing and VINDEX3 recording belong
to higher layers.

## Source map

| Source | Responsibility |
|---|---|
| [backend](src/backend/) | `ComputeBackend`, `MatMul`, `QuantMatVec`, `DecodeBackend` and capability probes |
| [cpu](src/cpu/) | CPU kernels, BLAS integration and quantized arithmetic |
| [attention](src/attention/), [forward](src/forward/), [residual.rs](src/residual.rs) | Attention spine, embedding/forward primitives and norms |
| [ffn](src/ffn.rs) | `FfnBackend` contract and weight-backed arithmetic |
| [kv_dispatch](src/kv_dispatch/) | Synchronous per-layer execution intent and CPU implementation |
| [async_compute_backend](src/async_compute_backend/) | Deferred-dispatch contract and CPU implementation |
| [kv_index.rs](src/kv_index.rs) | `KvIndex`: read-only substrate access without a dependency on `VectorIndex` |
| [kquant_forward](src/kquant_forward/) | Quantized forward/decode helpers |
| [encoders](src/encoders/) and [connectors](src/connectors/) | CPU vision encoder and projector forward operations |

`larql-vindex` implements `KvIndex for VectorIndex`. Adding a direct vindex or
inference dependency here would undo the substrate boundary. VINDEX3's
`PlanBackend` interpreter seam lives in `larql-vindex`; it composes numerical
primitives from this crate but is a different interface from `ComputeBackend`.

## Platforms and checks

The manifest selects Accelerate on macOS, system OpenBLAS on Linux/FreeBSD,
and the ndarray path without a BLAS backend on Windows. CPU compute has no
Metal feature. The `gpu` flags exposed by consumers select the sibling crate.

```bash
cargo test -p larql-compute
cargo test -p larql-compute --features heavy_tests
```

`heavy_tests` enables the slower integration suites; `test-utils` exposes
synthetic `KvIndex` fixtures. Benchmarks stay in this crate; user-facing
examples are catalogued in [larql-demos](../larql-demos/README.md).

Use [compute-substrate.md](../../docs/compute-substrate.md) for ownership and
[the Metal README](../larql-compute-metal/README.md) for GPU controls and the
measurement protocol. Older shader/decode notes under this crate predate the
backend split; they are linked as historical material, not current GPU ownership.
