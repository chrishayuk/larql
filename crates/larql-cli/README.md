# larql-cli

**Class: CURRENT.** The `larql` binary composes container operations,
execution, observation, LQL, serving and research tools.
[Current VINDEX3 command inventory](../../docs/generated/current-facts.md) is
generated and compared with the Clap command tree.
[Full stack map](../../docs/architecture-stack.md) and
[workspace dependencies/features](../../docs/generated/workspace-facts.md).

```bash
cargo build --release -p larql-cli
larql vindex3 --help
larql vindex3 observe model.vindex3 --backend production \
  --prompt "The capital of France is" --record run.jsonl
larql serve model.vindex3 --port 8080
```

Add Cargo's release target directory to PATH. Build `larql-server` for
`serve`, which dispatches to that binary. Linux/Windows builds need
`--no-default-features` because the default GPU feature uses Metal.

The [execution guide](../../docs/vindex3/execution.md) covers source admission,
encoding and generation. [Observation and intervention](../../docs/vindex3/observation-and-intervention.md)
distinguishes the supported recorder from scoped research tooling.
The standalone [vindex tool](../vindex-cli/README.md) provides format operations.

Commands are thin adapters under [src/commands](src/commands/): primary verbs
include `run`, `chat`, `bench`, `serve`, `vindex3` and `shannon`; extraction,
query, diagnostics and `dev` hold their respective families. The
[main command tree](src/main.rs) owns dispatch and help headings. Historical
research spellings are rewritten to `larql dev`; new tooling should use the
explicit family.

V2 extraction, LQL and patch workflows remain available. Use the
[LQL guide](../../docs/lql-guide.md), [generation policy](../../docs/vindex-generation-policy.md)
and individual command help. The broader [CLI guide](../../docs/cli.md)
is supplementary and does not enumerate the complete VINDEX3 surface.

```bash
cargo test -p larql-cli --bin larql
```
`larql run` supports explicit V3 `row`, `standard` and `no-cache` modes,
CPU Gemma 3 image prefixes, `--v3-shards` for ordered CPU layer workers, and
`--v3-ffn-shards` for CPU dense FFN workers with local attention and row KV.
See [the scoped runtime guide](../../docs/vindex3/runtime-followups.md).
