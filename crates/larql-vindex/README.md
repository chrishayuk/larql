# larql-vindex

**Class: CURRENT.** VINDEX container lifecycle and the VINDEX3 execution
substrate. [Overview](../../docs/vindex3/what-is-vindex3.md) ·
[status](../../docs/vindex3/status.md) ·
[generated facts](../../docs/generated/current-facts.md).
[Full stack map](../../docs/architecture-stack.md) and
[workspace dependencies/features](../../docs/generated/workspace-facts.md).

This crate owns source admission, the model-system graph, container encoding
and inspection, executable component plans, physical representations and the
canonical interpreter. It also retains the VINDEX2 mmap index, querying and
patch-overlay APIs.

```text
source inventory → semantic admission → system graph → encoded container
  → component operation plan → representation/operand binding
  → canonical execution → observation records and scoped research hooks
```

A container distinguishes logical objects from their bytes. The interpreter
executes the declared operations; a `PlanBackend` supplies numerical kernels.
Alternative representations retain encoding and fidelity identity. Observation
subscribes to execution boundaries, with provenance and record composition in
`larql-inference`; the Observatory consumes those records.

<a id="crate-structure"></a>

## Source map

| Area | Responsibility |
|---|---|
| [format/vindex3/plan](src/format/vindex3/plan/) | Semantic admission, findings, planner identity and capabilities |
| [format/vindex3/graph](src/format/vindex3/graph/) | Logical components, objects, interfaces and operator-derived surfaces |
| [format/vindex3/encode](src/format/vindex3/encode/) | Container materialization, including local and HF payload sources |
| [format/vindex3/opplan](src/format/vindex3/opplan/) | Operation plans, operand closure, canonical execution and backend seam |
| [format/vindex3/represent](src/format/vindex3/represent/) | Compilation, selection, accounting, candidate authority and evidence/search state |
| [format/vindex3/knowledge](src/format/vindex3/knowledge/) | Query/knowledge and descriptive-support machinery |
| [patch](src/patch/) | V2 immutable-base overlays and patch persistence |
| [format/generation.rs](src/format/generation.rs) | Generation discrimination and extraction default |

## Boundaries

- Admission, byte integrity, executable closure, numerical fidelity and
  behavioral evidence answer different questions. A successful encode does
  not prove all of them.
- Operator surfaces follow the declared program. An SSM or recurrent mixer
  must not acquire fictional softmax-attention operands.
- CPU kernels live in `larql-compute`; Metal arithmetic/lowering lives in
  `larql-compute-metal`. Session and generation orchestration lives in
  `larql-inference`.
- V2 base files are immutable. `VectorIndex`, `PatchedVindex`, gate KNN and
  `.vlp` overlays remain supported APIs; they are not the V3 container model.
- Mmap storage does not imply a universal inference RAM bound. Residency
  depends on topology, representations, backend and execution policy.

## Contracts and use

Use [vindex-cli](../vindex-cli/README.md) for format-native operations and
[larql-cli](../larql-cli/README.md) for execution and recording. The
[candidate specification](docs/vindex3-format-spec.md) owns the versioned
format; [architecture](../../docs/vindex3/architecture.md),
[execution](../../docs/vindex3/execution.md),
[representation](../../docs/vindex3/representation.md), and
[observation](../../docs/vindex3/observation-and-intervention.md) explain the
current system. V2 details remain in [format-spec.md](docs/format-spec.md)
and [operations-spec.md](docs/operations-spec.md).

```bash
cargo test -p larql-vindex
```

Use the pinned toolchain and the workspace's platform/build instructions.
Model-backed witnesses and performance protocols are separate from unit tests.
