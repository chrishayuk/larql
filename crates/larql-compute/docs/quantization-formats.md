# Quantized compute and storage layouts

**Class: CURRENT.** [CPU substrate](../README.md) ·
[Metal backend](../../larql-compute-metal/README.md).

Canonical GGML type IDs and block sizes live in
[larql-models::quant::ggml](../../larql-models/src/quant/ggml/mod.rs).
The previous guide's 148-byte Q4_K layout is obsolete. Use the actual
`Q4_K_BLOCK_BYTES`/`Q4_K_BLOCK_ELEMS` constants; do not recover a type or stride
from a rounded bytes-per-value figure.

Compute kernels consume declared layouts, which may include external scales,
interleaved matrices, padded rows or quantized activation scratch. Those are
not necessarily identical to checkpoint wire blocks. Shape and alignment
requirements are part of the kernel contract and must be checked by its caller.

[QuantMatVec](../src/backend/quant_matvec.rs) defines the compute dispatch
surface, [quant_route.rs](../src/quant_route.rs) describes routing and
[cpu/ops](../src/cpu/ops/) implements numerical kernels. Metal owns its own
[shaders](../../larql-compute-metal/src/shaders/) and host dispatch.

For V3, the [representation codec contract](../../../docs/represent-codec-contract.md)
and [lowering inventory](../../../docs/lowering-plane-inventory.md) separate
byte decoding, compilation, realization and backend support. Adding a decode
helper here does not admit a new V3 execution path or prove its quality.

The [source-format guide](../../larql-models/docs/quantization-formats.md) links
encoding authorities; the [archived guide](../../../docs/archive/README.md)
preserves earlier layout/benchmark descriptions for provenance.
