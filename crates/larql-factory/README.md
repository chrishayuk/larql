# larql-factory

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Recipe-driven build orchestration for vindex artifacts: structural validation,
reproducible build identity, size estimates, capability manifests, cards and a
staged build driver. The driver calls the extractor and verification tools;
this crate does not implement the VINDEX3 interpreter.

## API and authority

[src/lib.rs](src/lib.rs) exports `Recipe`, `validate`, `build_id`,
`estimate_size`, `capabilities_manifest`, `render_card` and `run_build`.
[build](src/build/) owns stages and `BuildRecord`; `CommandRunner` makes
subprocess execution an explicit boundary that tests can replace.

```bash
larql recipe validate recipe.yaml
larql recipe build-id recipe.yaml
larql recipe estimate recipe.yaml
```

A build can fetch, extract and publish artifacts according to its recipe;
inspect that recipe before invoking `larql recipe build`. Source revision,
extractor identity and generation belong in the build contract. Capability
manifests describe implemented architecture support, not proof that a particular
artifact passed all numerical or behavioral gates.

The current VERIFY stage establishes checksum integrity. It must not be
presented as source-forward parity or a representation-quality assessment.
MIRROR and REGISTER remain external orchestration concerns. See the
[Factory programme](../../docs/vindex-factory.md), including its explicit
implemented/design boundaries, and [source-to-serving interfaces](../../docs/runtime-surfaces.md).

```bash
cargo test -p larql-factory
```

The manifest dependencies are models and the public manifest contract; the
build driver's use of external commands is not a hidden Rust dependency on
`larql-vindex` or `larql-cli`.
