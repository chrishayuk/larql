# larql-continuation-fixture

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

A **test fixture**, not a shipped provider and not an SDK example. It builds
the CONTINUATION-PLUGIN-1 C5 external continuation provider
(`hostile-test-provider/v77`) as a `cdylib`, so that larql-cli's C6 gate can
load it through the real `--plugin` path and hold it to the C5 journey bit
for bit against `canonical/v1`.

| Source | Responsibility |
|---|---|
| [src/lib.rs](src/lib.rs) | Registers the provider through `larql_plugin!` and `PluginRegistrar::continuation` |

- The provider is not re-written here. It is
  `crates/larql-kv/tests/external_continuation_provider/provider.rs`, included
  by `#[path]`, so the dylib and the C5 proof are one source.
- Nothing depends on this crate. The gate
  (`crates/larql-cli/tests/support/continuation_fixture.rs`) builds it into its
  own target directory and `dlopen`s the result; `cargo metadata` and a source
  scan check that none of it is linked into larql-cli.
- Its `gpu` feature mirrors larql-cli's. The gate builds it with the features
  of the CLI under test, because the plugin ABI stamp names the compiler and
  commit but not features.
- `publish = false`.

For writing a real continuation plugin, see [plugins](../../docs/vindex3/plugins.md).
