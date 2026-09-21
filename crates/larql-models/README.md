# larql-models

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Model description, source inventory and weight loading. This crate owns
configuration, architecture traits, source tensor mappings, quantized data
formats and multimodal weight descriptions. It does not execute the VINDEX3
operation program or select a runtime backend.

## Responsibilities

| Source | Responsibility |
|---|---|
| [config](src/config/) | `ModelConfig`, `ModelArchitecture`, topology and config-derived defaults |
| [architectures](src/architectures/) | Family adapters and tensor-key mappings |
| [inventory](src/inventory/) | Source declarations and resolved facts used by VINDEX3 admission |
| [loading](src/loading/) and [weights](src/weights.rs) | Safetensors/GGUF loading, filtering, `ModelWeights` and weight views |
| [quant](src/quant/) | Encoding/decoding and quantized layout definitions |
| [multimodal](src/multimodal.rs) | Encoder/connector contracts and embedding-plan descriptions |
| [encoders](src/encoders/) and [connectors](src/connectors/) | Vision tower/projector configuration and weights |

`ModelArchitecture` describes source-model behavior; it is not the authority
for executing an already encoded VINDEX3 artifact. The V3 interpreter reads
its container graph and operation plan. Inventory recognition, semantic
admission, encoding and backend execution support are separate gates.

When adding a model, compare the source tensor inventory with the written
manifest: unrequested tensors can otherwise disappear silently. Config-backed
answers belong in trait defaults when shared across families. Validate the
first drifting layer against a reference forward, with fixtures long enough
to distinguish windowing and other declared behavior.

## Development

```bash
cargo test -p larql-models
```

The `test-utils` feature exposes shared synthetic weight fixtures for downstream
crate tests. Supported-family lists should come from detection/config code and
capability reports, not a fixed README count.

See [architecture extension](docs/architecture-trait.md),
[weight loading](docs/weight-loading.md), [quantized formats](docs/quantization-formats.md),
and the [source-to-execution guide](../../docs/compute-substrate.md).
