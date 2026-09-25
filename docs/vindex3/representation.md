# Represent a model

**Class: CURRENT.** [Status](status.md).

REPRESENT explores physical realizations under declared behavior and cost
constraints. A logical object can have canonical bytes and alternate encoded
representations. A profile selects among representations; encoding a smaller
pack does not by itself establish that its behavior is acceptable.

The standalone tool supports reference compilation and inspection:

```bash
vindex represent model.vindex3 model-nvfp4.vindex3 --encoding NVFP4
vindex representations model-nvfp4.vindex3
vindex precision model-nvfp4.vindex3 --matrix
```

Supported encodings, geometry and export targets are checked by the compiler.
Use command help and an admitted artifact; the existence of a codec is not
a promise that every backend supports its execution.

The research machinery extends beyond compilation. Candidate authority binds
actual payloads to a representation state. Deterministic evidence ingestion
feeds the search state; measurement keys bind candidate, protocol and
instrument identity. Actuation reconstructs and validates the selected
measurement before preparation and execution. These boundaries make a search
loop auditable; none independently grants a candidate behavioral approval.

The implementation lives in
[`represent/`](../../crates/larql-vindex/src/format/vindex3/represent/), including
`candidate_authority`, `ingest`, `state`, `search_evidence` and `actuate`.
The [REPRESENT contract index](../represent-v1-contract-index.md) and
[optimizer contract index](../optimizer-contract-index.md) name the tests that
protect their respective frozen contracts. They are stronger authorities than
a blanket claim that the optimizer is finished.

The [codec contract](../represent-codec-contract.md) separates representation
decoding, compilation and execution support. The
[optimizer MCP design](../represent-optimizer-mcp.md) is a design document,
not an inventory of shipped commands.

[REPRESENT-CAL-1](../represent-cal-1.md) is the accepted contract for calibrated
recipe integration:
capture and derivation provenance, completion of fixed-grid GPTQ, matched
unweighted/weighted scale fitting, and separate admission evidence. Its gates
remain open, not a claim that these recipe paths are shipped.
