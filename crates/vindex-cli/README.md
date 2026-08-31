# vindex

The format-native VINDEX3 tool. Every command answers from the container alone —
`index.json`, the system graph, the segment headers — the same authorities an
independent implementation would read. No inference runtime, no model registry,
no network.

**VINDEX3 is specified and explained at [vindex3.org](https://vindex3.org)**;
[vindex3.org/get-started](https://vindex3.org/get-started) walks this tool from
install to `verify` with recorded outputs, and the normative documents are
[`crates/larql-vindex/docs/vindex3-format-spec.md`](../larql-vindex/docs/vindex3-format-spec.md)
(the 3.0 Candidate ABI) and [`docs/vindex3-format.md`](../../docs/vindex3-format.md)
(the living spec).

## Install

```bash
# Prebuilt (macOS arm64)
curl -L https://github.com/chrishayuk/larql/releases/download/\
vindex-v0.5.0/vindex-0.5.0-macos-arm64.tar.gz | tar xz

# Or from source
cargo install --git https://github.com/chrishayuk/larql vindex-cli
```

`vindex update` installs the latest release, and only ever when asked: no verb
checks for updates on its own, and nothing phones home. `vindex update --check`
reports without installing.

## The verbs

| Command | What it answers |
|---|---|
| `inspect <container>` | The container reconstructed from itself: identity, census, coherence |
| `describe <container> <object>` | One logical object in full — bindings, representations, tensor-table head; `--peek <tensor>` decodes the first values |
| `representations <container>` | The physical directory: what exists as bytes, with recorded fidelity |
| `layers <container>` | Every layer's token-mixer programme, from the operation plan |
| `diff <container> <A> <B> <object>` | One object under two representations, decoded and compared value by value — the error derived, never asserted |
| `represent <container> <out>` | Compile a representation beside the original through the reference compiler. Nothing is destroyed |
| `precision <container>` | Bits per weight, derived from stored bytes over tensor elements; `--matrix` shows bits per layer × semantic role |
| `verify <container>` | The container against its own recorded hashes, re-derived from the artifact alone |

Addresses are semantic: `layer.24.ffn.down`, `layer.3.attention.q`,
`layer.7.mixer`. A layer whose programme has no such operand refuses by naming
what it does have, rather than inventing a surface.

```
$ vindex inspect granite-3b.vindex3
family         granite
generation     3
geometry       40 layers · hidden 2560
authority      canonical

COMPONENT   ROLE          LAYERS   HIDDEN
target      primarytext       40     2560

graph          4 object(s) · coherent
```

Every command takes a global `--json`: one result, three projections — terminal
text, structured JSON, and the designed panels vindex3.org renders from the same
shape. Exit codes: `1` on `verified: false`, `2` on error.

## What is deliberately not here

Execution, inference, mutation, benchmarking and representation search are
engine concerns and live in `larql`. VINDEX3 is defined by its documents, not by
any tool, and an artifact should not require an engine to be understood — this
binary is the canonical *reader*.

One impurity, recorded honestly: `larql-vindex` is not yet a dependency-light
core, so this version carries its tree. Carving `vindex-core` out of it is the
named next step, not a surprise.
