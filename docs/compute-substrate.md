# Source description and numerical execution

**Class: CURRENT.** This guide connects models, compute, Metal and the VINDEX3
interpreter. [Workspace facts](generated/workspace-facts.md) record their
manifest dependencies; [architecture](architecture-stack.md) maps ownership.

## Source to program

`larql-models` reads configuration and tensor inventories, resolves architecture
facts and loads weight views. Family traits describe source semantics, including
activation, normalization, position policy, attention/mixer geometry and tensor
keys. A config field should not disappear behind a hardcoded trait default.

VINDEX3 admission in `larql-vindex` compares declared and resolved semantics
with what the system graph can carry. Encoding writes the admitted structure
and payloads. A component operation plan then establishes operand closure for
execution. Recognition, carriage, byte integrity and executable support are
separate findings, not successive spellings of “supported.”

A V2 `ModelWeights` forward and a V3 component plan are separate callers of
numerical kernels. V3 execution does not reconstruct `ModelWeights` or infer
its semantics from model-family names at runtime.

## Traits and implementations

| Contract | Owner | Consumer boundary |
|---|---|---|
| `ModelArchitecture` / source inventory | `larql-models` | Source loading, admission and model-weight execution |
| `ComputeBackend` / `MatMul` / `QuantMatVec` / `DecodeBackend` | `larql-compute` | CPU/Metal numerical dispatch |
| `KvDispatch` / `AsyncComputeBackend` | `larql-compute` | Per-layer synchronous/deferred execution intent |
| `FfnBackend` | `larql-compute` | Weight FFN plus engine-side sparse/remote implementations |
| `KvIndex` | `larql-compute` | Abstract index reads; concrete vindex implementation lives above |
| `PlanBackend` | `larql-vindex` | Numerical realization of the canonical V3 interpreter |
| `LogitsSession` | `larql-inference` | Generation/serving over logits and advancing state |

A format decoder, representation compiler, numerical kernel and lowering
provider are different capabilities. Adding a quantized encoding to source
loading does not make all backends able to execute it. The
[codec contract](represent-codec-contract.md) and
[lowering inventory](lowering-plane-inventory.md) define the V3 distinctions.

## Platforms and measurements

The CPU manifest selects Accelerate on macOS, system OpenBLAS on Linux/FreeBSD,
and a non-BLAS ndarray path on Windows. Metal implementation modules are
macOS-gated; consumers expose their own opt-in features. The top-level CLI
has a default GPU feature, while other crates can have empty defaults. Read
the actual manifest rather than treating one crate's default as workspace-wide.

[The Metal README](../crates/larql-compute-metal/README.md) owns operator
controls, command-buffer failure handling and GPU measurement rules. A measured
result requires declared model/representation/backend, warmed control brackets
and uncontended execution. A profiler changes execution timing; use it for
stage attribution and an unprofiled run for throughput.

A correctness fixture must exercise the behavior in question: a short prompt
cannot distinguish full from sliding attention before the window boundary.
Use the layer-dump/layer-diff workflow to locate the first disagreement before
explaining an end-to-end score. Historical performance records keep their
hardware, topology and control conditions.
