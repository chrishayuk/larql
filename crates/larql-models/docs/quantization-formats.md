# Quantized data formats

**Class: CURRENT.** [Crate overview](../README.md) ·
[representation contracts](../../../docs/represent-codec-contract.md).

`larql-models::quant` owns source encoding/decoding and wire-layout definitions.
CPU quantized arithmetic lives in `larql-compute`; Metal kernels and dispatch
live in `larql-compute-metal`. VINDEX3 representation codecs and compilation
are owned by `larql-vindex`.

| Source | Scope |
|---|---|
| [half](../src/quant/half.rs) | f16/bf16 conversions |
| [ggml](../src/quant/ggml/) | GGUF type IDs, block constants and supported GGML quantization/dequantization |
| [mxfp4](../src/quant/mxfp4.rs) | Microscaled FP4 values and scale interpretation |
| [fp8](../src/quant/fp8.rs) and [fp8_finegrained](../src/quant/fp8_finegrained.rs) | FP8 forms and scale layouts |
| [nvfp4](../src/quant/nvfp4.rs) and [nvfp4_ggml](../src/quant/nvfp4_ggml/) | NVFP4 forms and GGML interoperability |
| [fp4](../src/quant/fp4.rs) and [fp4_block](../src/quant/fp4_block.rs) | FP4 codebooks/block encodings |

Use the constants and supported-type dispatch in
[ggml/mod.rs](../src/quant/ggml/mod.rs) for wire IDs, block bytes and elements.
Q4_K must not be inferred from a byte-per-value ratio shared by another format.
The old private 148-byte Q4_K description is obsolete; the current wire
constant is `Q4_K_BLOCK_BYTES` and readers/writers must agree on it.

A wire block, an internal activation pack and a kernel-specific interleave can
share a precision label while having different layouts. External scale planes,
padding and operand geometry are part of the representation contract. Never
substitute a kernel scratch layout for a checkpoint encoding without an explicit
conversion.

Decoder availability, encoder availability, kernel support and behavioral
fidelity are separate claims. Use the VINDEX3 codec registry and execution
admission to establish the latter capabilities; the presence of a type ID in
this module alone is insufficient.
