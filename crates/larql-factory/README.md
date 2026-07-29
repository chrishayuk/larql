# larql-factory

The Vindex Factory driver: recipe schema, `build_id` canonicaliser,
structural validator, capability manifest, Hub card generator, and the
PREFLIGHT→RELEASE build-stage driver for
[docs/vindex-factory.md](../../docs/vindex-factory.md).

This is the single implementation both a `chuk-vindex-recipes` GitHub
Action and a rig worker are meant to call (as `larql recipe`,
`larql capabilities`, `larql card render`) — see §3.1 of the spec for
why the driver lives here rather than in a separate repo.

```
crates/larql-factory/
├── src/
│   ├── recipe/          Recipe schema (§4) — one file per YAML section
│   ├── validate/         Structural validator (§6.1's non-network checks)
│   ├── build_id.rs        build_id canonicaliser (§5)
│   ├── capabilities/       Capability manifest (§15.2)
│   ├── card/                 Hub model-card generator (§9)
│   ├── estimate/               Size/cost estimate (§6.1 step 4) — the
│   │                           first module with network I/O
│   ├── build/                    PREFLIGHT→RELEASE build driver (§7)
│   │   ├── runner.rs                CommandRunner trait + Subprocess/Mock impls
│   │   ├── record.rs                 BuildRecord/Stage/BuildStatus/OutputRecord
│   │   └── stages/                     One file per stage — invocation()
│   │                                   builder + run() through a CommandRunner
│   ├── constants.rs            Facts shared across modules
│   └── hex.rs                   Lowercase hex encoding
└── testdata/
    └── gemma-3-4b-it.yaml   Sample recipe used throughout the test suite
```

## What it does

- **Recipe schema** ([`recipe`]) — Rust types for the v0.1 YAML schema:
  `source`, `extractor`, `outputs`, `verify`, `publish`, `budget`.
  `Recipe::from_yaml` parses; `larql_vindex_spec::{ExtractLevel,
  StorageDtype, QuantFormat}` are reused directly rather than
  duplicated.
- **`build_id`** ([`build_id`]) — SHA-256 over exactly `apiVersion` +
  `source` + `extractor` + `outputs`. Changing `verify`/`publish`/
  `budget`/`metadata` doesn't change it — those don't change the
  produced bytes.
- **Structural validator** ([`validate`]) — everything §6.1's PR check
  needs that doesn't require network I/O: full-SHA revision, released-tag
  extractor version, known preset names, threshold ranges. Reports every
  problem in one pass, not just the first.
- **Capability manifest** ([`capabilities`]) — which architectures the
  running `larql` recognises and what each supports, built from
  `larql_models::detect::ARCHITECTURE_REGISTRY` (a real registry, not a
  hand-duplicated list — see that module's docs).
- **Card generator** ([`card`]) — a Hub model card: frontmatter, dims,
  slice table, `USE` snippet with a computed revision tag, verification
  summary, inlined recipe. `VerificationReport`/`SliceSummary` are still
  a provisional shape — the build driver's VERIFY stage is checksums
  only, not the numeric reconstruction/logit-match report §8.1 assumes.
- **Size/cost estimate** ([`estimate`]) — upstream download size, a
  coarse per-output byte estimate (dims via the same
  `larql_models::detect::detect_from_json` path the real extractor
  uses), an executor recommendation, and a cost band from
  `docs/dec-funnel-v0.2.md` §7's rate basis. Prices the recipe's own
  declared `budget.max_wall_minutes` rather than inventing a duration
  prediction — there's no real throughput data anywhere to ground one
  in.
- **Build driver** ([`build`]) — `run_build` orchestrates PREFLIGHT →
  FETCH → EXTRACT → SLICE → MANIFEST → VERIFY → PUBLISH → RELEASE as
  subprocess calls into this same `larql` binary (`model pull`,
  `extract`, `slice`, `verify`, `hf publish --private`,
  `hf visibility --public`), scoping `HF_HUB_CACHE` per build so a
  stale cached revision can never leak into EXTRACT. PUBLISH always
  goes private first; RELEASE only flips a repo public once every
  output has verified — §8's "nothing goes public unverified"
  contract. Every subprocess call goes through a [`CommandRunner`]
  trait (`SubprocessRunner` for real builds, a `MockRunner` in tests),
  so the whole pipeline's stage ordering and failure handling is
  tested without spawning a process. Always returns a `BuildRecord`
  (JSON-printable either way) rather than a bare `Result` — a stage
  failure is data (`BuildStatus::Failed { stage, message }`), not an
  early panic. MIRROR and REGISTER aren't implemented — see "Not built
  here yet" below.

## CLI usage

```bash
# Structural validation — prints every problem found, exits 1 if any
larql recipe validate my-recipe.yaml

# The content hash that determines whether a build is a no-op / verify-only / rebuild
larql recipe build-id my-recipe.yaml

# This release's capability manifest, as JSON
larql capabilities

# Render a Hub model card from a recipe + manifest + verification report
larql card render \
  --recipe my-recipe.yaml \
  --manifest index.json \
  --verification verification.json \
  --slices slices.json   # optional

# Upstream size, per-output size, executor recommendation, cost band —
# hits the network (HF file listing + config.json)
larql recipe estimate my-recipe.yaml

# PREFLIGHT → FETCH → EXTRACT → SLICE → MANIFEST → VERIFY → PUBLISH → RELEASE
# Prints a BuildRecord as JSON; exits non-zero (and stops at the failed
# stage) if any stage fails. Scratch dir defaults to a temp directory.
larql recipe build my-recipe.yaml [--scratch-dir DIR]
```

## Not built here yet

MIRROR (R2 upload) and REGISTER (chuk-experiments-server) from §7's
stage list aren't implemented — nothing in this codebase talks to
either today, and the spec's own text assumes both are owned by the
rig worker, not the `larql` binary. `run_build`'s `BuildRecord` is the
structured hand-off an external wrapper would read to do both, the way
`dec0-loopback.sh` already wraps `dec-bench`'s own JSON output.
Reconstruction-fidelity and logit-match numeric verification (§8.1)
also aren't implemented — VERIFY here is checksum integrity only
(`larql verify`), since the numeric checks need per-architecture
tensor-naming knowledge that isn't validatable without real model
weights. A real `chuk-vindex-recipes` repo, and end-to-end runs against
real HF credentials, are still open — see the spec's §14 build
inventory.

## Tests

```sh
cargo test -p larql-factory
```

Every source file is at or above the 90% floor (`coverage-policy.json`);
the large majority are at 100% — `estimate/mod.rs`, `estimate/http.rs`,
`build/mod.rs`, and `build/runner.rs` are the main exceptions, all
network- or subprocess-facing.

## CI

```sh
make larql-factory-ci
```

GitHub Actions: `.github/workflows/larql-factory.yml`
Platforms: **Linux · Windows · macOS** (all in CI)
