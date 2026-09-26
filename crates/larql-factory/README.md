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

`larql recipe build` currently refuses at preflight: every recipe requires
reconstruction and logit-agreement checks, while the driver only implements
local checksum verification. No fetch, extraction or publication starts when
these requirements cannot be enforced; `from_hub: true` is also reported as an
unsupported verification requirement. Validation, identity, estimation and card
APIs remain available. Completing the numerical verifier is required before
build-and-publish execution can resume; checksum success is insufficient.

MIRROR and REGISTER remain external orchestration concerns. See the
[Factory programme](../../docs/vindex-factory.md), including its explicit
implemented/design boundaries, and [source-to-serving interfaces](../../docs/runtime-surfaces.md).

```bash
cargo test -p larql-factory
```

The manifest dependencies are models and the public manifest contract; the
build driver's use of external commands is not a hidden Rust dependency on
`larql-vindex` or `larql-cli`.
