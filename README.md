# LARQL

**Execute, query and study a model as an artifact.**

LARQL is the reference implementation of **VINDEX3** and a research system for
model execution, representation and evidence. VINDEX3 describes a model's
logical objects, physical representations and executable semantics. LARQL can
run that program, record its computation, inspect its structure, and test
claims about its behavior.

**Encode · Run · Represent · Observe · Intervene · Query** describes the
project's scope. Each surface has its own maturity and evidence boundary;
[status](docs/vindex3/status.md) distinguishes supported interfaces from active
research. **Planned:** sharded/partial execution and multimodal embedding
handoff; both are explicitly refused until their execution contracts and parity
gates exist ([planned capabilities](docs/vindex3/status.md#planned-capabilities)).
The graph-database thesis remains: the model itself is the object
being queried, rather than a separate database of extracted facts.

**Class: CURRENT.** Start with [What is VINDEX3?](docs/vindex3/what-is-vindex3.md).
[Machine-derived facts](docs/generated/current-facts.md) give this checkout's
versions, schemas and command inventory. [vindex3.org](https://vindex3.org)
teaches the format; the [candidate specification](crates/larql-vindex/docs/vindex3-format-spec.md)
is its versioned contract.

## Build and try it

Use the repository's pinned Rust toolchain:

```bash
cargo build --release -p vindex-cli -p larql-cli
```

On Linux and Windows add `--no-default-features`; the default GPU feature uses
Metal on macOS. Invoke the binaries from Cargo's release target directory or
add it to PATH. With a local, supported Hugging Face checkpoint:

```bash
# Admit the source, encode its declared structure, inspect the result.
vindex plan /path/to/checkpoint --json
vindex encode /path/to/checkpoint --output model.vindex3
vindex inspect model.vindex3 --json
vindex verify model.vindex3

# Read its executable program and record a real decode.
larql vindex3 ops model.vindex3
larql vindex3 observe model.vindex3 --backend production \
  --prompt "The capital of France is" --record run.jsonl
```

`plan` and `encode` also accept `hf://org/repo@revision`. Admission and backend
support are explicit gates, not a promise that every checkpoint executes.
See [execution](docs/vindex3/execution.md) for generation and serving, and the
[standalone vindex README](crates/vindex-cli/README.md) for format operations.

## A model can leave an execution record

The canonical decode path exposes carrier writes with site identity and
provenance. Optional lenses read states through the model's normalization and
head. The [Observatory](observatory/README.md) imports records for coordinated
inspection and replay. Head/source attribution and intervention research build
on these records, with separate contracts for descriptive evidence and causal
claims. A projected contribution is not a counterfactual.

[Observation and intervention](docs/vindex3/observation-and-intervention.md)
explains those boundaries. [Current research](docs/vindex3/status.md#current-research)
links the evidence and frozen protocols.

## Representations with evidence

VINDEX3 separates logical model identity from physical encodings. REPRESENT
compiles alternate representations and provides accounting, candidate identity,
evidence ingestion and measurement/search contracts. Smaller bytes, executable
support and acceptable behavior are separately established claims.

```bash
# Compile a 4-bit deployment image, then measure it against the source.
larql vindex3 represent model.vindex3 --output deploy.vindex3 --encoding NVFP4 --deployment
larql vindex3 token-bank export model.vindex3 \
  --prompts bench/prompts/quality-bank-1/prompts.json --output bank/
larql vindex3 measure --reference model.vindex3 --reference-backend production \
  --candidate deploy.vindex3 --candidate-backend production-nvfp4 \
  --bank bank/ --sequences 69 --label nvfp4 --output out/nvfp4
```

Start with [representation](docs/vindex3/representation.md) and its contract
indexes. Encodings and execution providers this build does not ship load from
shared libraries with `--plugin`; see [plugins](docs/vindex3/plugins.md).

## Existing vindexes and LQL

VINDEX2 extraction, mmap gate queries, LQL, patch overlays and compilation
remain part of LARQL. Default extraction still follows the
[generation policy](docs/vindex-generation-policy.md); use an explicit V3
encoding path for the workflow above. Base vindexes are immutable: mutations
use overlays and compilation produces a new artifact.

For those workflows, use the [LQL guide](docs/lql-guide.md),
[language specification](crates/larql-lql/docs/spec.md),
[operations and patches](crates/larql-vindex/docs/operations-spec.md),
[Factory](docs/vindex-factory.md), and [Python bindings](docs/larql-python.md).

## Architecture and development

[Stack architecture](docs/architecture-stack.md) maps ownership across every crate;
[generated workspace facts](docs/generated/workspace-facts.md) track dependencies
and features. [VINDEX3 architecture](docs/vindex3/architecture.md) explains the
container execution path.
`larql-vindex` owns the container and canonical interpreter; CPU and Metal
crates provide numerical backends; `larql-inference` owns runtime/session
composition. CLI, LQL and server layers expose those capabilities.

```bash
cargo test -p vindex-cli
cargo test -p larql-vindex
make ci
python3 scripts/current_facts.py --check
```

Use `--no-default-features` for portable Cargo builds/tests. Model-backed and
performance experiments have additional controls; a fixture pass is not a
model-wide fidelity or speed claim. [AGENTS.md](AGENTS.md) documents workspace
invariants and build conventions.

The [documentation index](docs/README.md) leads to specifications, runtime
contracts, research records and ADRs. The [documentation policy](docs/documentation-policy.md)
keeps current explanations separate from versioned contracts and historical
evidence. Historical benchmarks retain their original conditions in research
records rather than serving as a universal performance promise.

## License

See [LICENSE](LICENSE).
