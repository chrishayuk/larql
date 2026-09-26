# ModelArchitecture and source admission

**Class: CURRENT.** [Crate overview](../README.md) ·
[source-to-execution guide](../../../docs/compute-substrate.md).

`ModelArchitecture` describes source-model semantics: tensor keys, activation,
normalization, position policy, layer topology and mixture-of-experts geometry.
Its definition and config-derived defaults live in
[config/architecture.rs](../src/config/architecture.rs). `family` and `config`
are required; other methods provide defaults or extension points as defined
there. A default must not contradict an explicit configuration field.

Family implementations live in [architectures](../src/architectures/).
[Detection](../src/detect/) selects adapters for source loading, while
[inventory](../src/inventory/) records declared and resolved facts for V3
admission. The existence of an adapter is not a promise that every tensor,
modality or backend for that family is supported.

## Extending support

1. Read the source configuration and tensor inventory together. Establish
   required weights, optional branches, norms, biases, sinks and layer policies.
2. Put shared config-backed answers in trait defaults. Use a family override
   only for an actual architectural distinction.
3. Add key mappings and loading support, then compare the written inventory
   with the source. A tensor that no extractor asks for may be silently omitted.
4. For V3, verify semantic admission, graph carriage and operand closure.
   Consuming a config key is not proof that it reaches execution.
5. Compare layer-level outputs against a reference. Choose prompts and shapes
   that exercise the claimed behavior, including crossing any attention window.

Multimodal descriptions live in [multimodal.rs](../src/multimodal.rs), with
vision/projector weights in their owning modules. CPU forwards live in
`larql-compute`. A declared modality interface is separate from a complete
CLI/server embedding and execution path.

An encoded VINDEX3 artifact carries its own graph and component program.
The canonical executor consumes those declarations rather than calling a
family adapter to reconstruct model meaning. See the
[VINDEX3 architecture](../../../docs/vindex3/architecture.md) for that boundary.
