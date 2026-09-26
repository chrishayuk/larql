# What is VINDEX3?

**Class: CURRENT.** [Status and machine-derived facts](status.md).

VINDEX3 describes a model as an executable, queryable artifact: its logical
objects, their physical representations, the operations that consume them,
and the provenance needed to check those declarations. LARQL is the reference
implementation and the surrounding execution and research system.

A checkpoint supplies weights and configuration. VINDEX3 makes the resolved
model structure explicit in a system graph. An operation plan binds those
objects to executable operands; the runtime executes that declared program.
Alternative representations can change physical storage and arithmetic while
retaining their relationship to the model and their declared fidelity.

The useful verbs are **encode · run · represent · observe · intervene · query**.
They span different maturity levels and interfaces. Encoding and inspection
are available through `vindex`; execution and recording through LARQL;
intervention and graph-walk research need the specific implementation and
evidence described in [status](status.md).

Observation lets a computation leave a record: which carrier was written,
where, under which representation and execution identity. Lenses and the
Observatory inspect those records. An attribution or projection is descriptive;
a causal claim requires an actual intervention and a declared comparison.

VINDEX2 remains supported. Its mmap weight index, gate queries and patch
overlays are part of LARQL, and default extraction still follows the
[generation policy](../vindex-generation-policy.md). A filename extension
does not establish a container's generation; the index schema does.

Continue with [architecture](architecture.md), [execution](execution.md),
[representation](representation.md), and
[observation and intervention](observation-and-intervention.md).
The [candidate specification](../../crates/larql-vindex/docs/vindex3-format-spec.md)
owns the format contract; this overview explains the implementation.
