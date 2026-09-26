# Source weight loading

**Class: CURRENT.** [Crate overview](../README.md).

[loading](../src/loading/) converts supported source checkpoints into
`ModelWeights` and weight views. Safetensors and GGUF loaders own their parsing,
prefix handling, dtype interpretation and filtering. Their re-exports in
[loading/mod.rs](../src/loading/mod.rs) are the entry-point inventory.

`load_model_dir`, its validated/filtered variants, and the walk-only variants
serve the model-weight execution path. Validation checks model configuration;
filtering avoids loading/dequantizing tensors the caller does not need.
Walk-only loading must not be mistaken for a complete set of operands for
arbitrary later operations. `resolve_model_path` resolves the supported source
locations/cache conventions; it is not VINDEX3 container opening.

The weight representation is defined in [weights.rs](../src/weights.rs).
Do not assume every loader is zero-copy, every dtype stays encoded, or every
caller requires the whole tensor set: those properties depend on the loader,
format and requested view. Preserve explicit storage ownership and scratch
lifetimes when adding a path.

## VINDEX3 source access

V3 planning starts from inventory and semantic admission; remote source access
can stage config and safetensors headers before fetching payload ranges.
Encoding is owned by `larql-vindex`, not a conversion of an eagerly loaded
`ModelWeights` object. See [remote sources](../../../docs/vindex3-remote-source.md)
and [the encoding guide](../../../docs/vindex3/execution.md).

At source extension time compare checkpoint tensor names against the emitted
manifest and graph. Attention biases, sinks, norms and side components are
semantics, not ignorable extras. Distinguish a deliberately filtered view from
a silently incomplete extraction. Layer-diff validation should name the first
forward disagreement rather than relying solely on final scalar similarity.
