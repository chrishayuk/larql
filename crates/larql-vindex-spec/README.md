# larql-vindex-spec

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Dependency-light contract for the vindex manifest, shared extraction/storage
enums and validation thresholds. This crate has no `larql-*` dependencies.
It is not the home of the VINDEX3 system graph, operation plan or representation
codec contracts; those live in [larql-vindex](../larql-vindex/README.md).

## Authorities

| Artifact | Scope |
|---|---|
| [src/lib.rs](src/lib.rs) | Manifest types, `ExtractLevel`, storage/quant enums and validation |
| [SPEC.md](SPEC.md) | Versioned manifest contract |
| [schema/vindex-v1.schema.json](schema/vindex-v1.schema.json) | JSON Schema mirror |
| [src/thresholds.rs](src/thresholds.rs) | Validation thresholds and layer sampling |

The manifest compatibility tag is distinct from `index.json`'s container
schema, the VINDEX3 graph schema and planner semantics. Use the
[generated VINDEX3 facts](../../docs/generated/current-facts.md) for those axes.
Do not update a frozen manifest contract merely because the V3 graph changes.

The manifest contract covers provenance, structural dimensions, extraction
level, dtype/quantization, checksums and sharding. Loader-specific fields can
round-trip in `extra` without becoming fields validated by this contract.
The Rust implementation is the reference when checking its schema/prose mirror;
contract changes still require the deliberate versioning process in SPEC.md.

```bash
cargo test -p larql-vindex-spec
```

The `test-utils` feature provides consumer fixtures. Package versions and
features are listed in the generated workspace inventory rather than duplicated
here as release claims.
