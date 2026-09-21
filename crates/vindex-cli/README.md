# vindex

**Class: CURRENT.** The format-native VINDEX3 tool: plan and encode sources,
inspect containers, compile alternate representations, and export supported
representations. Execution and observation live in LARQL.

[Generated package version and complete command inventory](../../docs/generated/current-facts.md)
are checked against source and the Clap command tree. Package version does not
establish which binaries have been published. The
[candidate specification](../larql-vindex/docs/vindex3-format-spec.md) owns the
format contract; [the overview](../../docs/vindex3/what-is-vindex3.md) explains it.

## Install from this checkout

```bash
cargo install --path crates/vindex-cli --locked
vindex --help
```

For prebuilt binaries, use the repository's
[release list](https://github.com/chrishayuk/larql/releases) and select a `vindex`
release and platform. `vindex update --check` checks availability;
`vindex update` installs an update only when explicitly requested.

## Workflow

```bash
vindex plan /path/to/checkpoint --json
vindex encode /path/to/checkpoint --output model.vindex3
vindex inspect model.vindex3 --json
vindex layers model.vindex3
vindex representations model.vindex3
vindex precision model.vindex3 --matrix
vindex verify model.vindex3

# Create an alternate representation in a new container.
vindex represent model.vindex3 model-nvfp4.vindex3 --encoding NVFP4
vindex describe model-nvfp4.vindex3 layer.0.ffn.down
vindex export model-nvfp4.vindex3 model.gguf
```

These paths are placeholders. Admission, geometry, encoding and export support
are checked for the actual model. `describe` accepts logical object addresses
and semantic operand addresses; an absent operand is refused. `diff` compares
representations held by a container. Use `<command> --help` for arguments;
the complete verb list is generated rather than duplicated here.

`plan` and `encode` accept `hf://org/repo@revision`: planning reads config and
headers, encoding reads payload ranges. These operations can use the network;
`update` also accesses release infrastructure. Local container inspection does
not require the source checkpoint or a model inference service.

The global `--json` option exposes structured results. Hash verification checks
the artifact against its recorded hashes; it does not establish forward-pass
parity or behavioral fidelity. `represent` compiles physical bytes; search and
measurement contracts are explained in [representation](../../docs/vindex3/representation.md).

The dependency on `larql-vindex` currently brings its broader workspace tree;
this binary is not yet an independently extracted, dependency-light format library.

## Checks

```bash
cargo test -p vindex-cli
python3 scripts/current_facts.py --check
```
