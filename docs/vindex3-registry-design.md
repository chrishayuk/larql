# VINDEX3 registry/resolver — design notes

Status: **rung 1 (registry contract) + rung 2A (`serve` convergence)
implemented.** §1–§7 below are the pre-implementation investigation
rung 1's instruction required — *"Before implementation, report the
existing resolution seams and any places where the current code would
force an awkward compatibility compromise. Ground the design in the
code rather than adding a parallel resolver."* §8 records the three
architecture decisions that investigation surfaced, confirmed by the
user on 2026-08-23. §9 records what rung 1 built. §10 records the
"resolver convergence" rung's scope and the claimed/unclaimed decision,
and what 2A (`larql serve`) built.

The scope is deliberately narrow (see §7): a versioned VINDEX3-only
registry manifest, a name/variant grammar, one shared resolver
abstraction, and a static local test registry. No website, no remote
registry service, no Mac app wiring, no runtime-lifecycle wiring.

## 1. What exists today — model-string resolution

**There are three independent "string → path" resolvers today, and
they already disagree with each other at the margins.** Any new
resolver must consolidate toward one of these becoming canonical, not
add a fourth:

1. **`crates/larql-cli/src/commands/primary/cache.rs::resolve_model`**
   — the "full" resolver: `hf://owner/name[@rev]` → download; an
   existing local directory → used as-is; a string containing `/` →
   checked against the merged cache, else prefixed `hf://` and
   downloaded; a bare shorthand name → `resolve_shorthand` requires a
   *unique* match across both caches. Used by `run`, `chat`, `show`,
   `slice`, and (pre-exec only) `serve`.
2. **`crates/larql-server/src/bootstrap/load.rs::load_artifact`** —
   inside the actual serving binary, much narrower: `is_hf_path`
   prefix check, else a literal filesystem path. **No knowledge of
   `~/.cache/larql/local/` shorthand names or bare `owner/name`
   HF-cache lookup at all.** It only "works" for those forms because
   `larql serve`'s CLI trampoline (`main.rs::run_serve`) pre-resolves
   the string before spawning the `larql-server` subprocess — and
   **silently falls back to the raw, unresolved string**
   (`.unwrap_or_else(|_| path.clone())`) if that pre-resolution
   errors, handing an unresolved/ambiguous string across the process
   boundary to the weaker resolver.
3. **`crates/larql-cli/src/commands/primary/pull_cmd.rs::looks_like_hf_repo`**
   — a *third*, independent `owner/name` heuristic ("exactly one `/`,
   neither side empty, no dot in the owner segment"), structurally
   similar to but not shared with either of the above. It rejects an
   owner containing `.`; `cache::resolve_model`'s slash-branch does
   not check that at all.

**`larql run`/`larql chat` have no VINDEX3 execution path at all** —
only `larql serve` does. `run_cmd::run` resolves via
`cache::resolve_model` and then unconditionally hands the result to
`load_vindex_config`/`walk_cmd`, both VINDEX2-only; a VINDEX3 result
fails several calls deep with a generic `VindexError::WrongContainerGeneration`,
not a purpose-built refusal at the top of `run` the way `slice`/`verify`
do it (`run_cmd.rs:996-998` says exactly this — "container completeness
is a separate rung"). **Consequence for this design**: a resolver that
returns a `ResolvedVindex3` pointing at a real VINDEX3 container is
correct and testable end-to-end against `serve`, but `larql run
qwen3.8` cannot actually execute yet regardless of what the resolver
returns — that gap is pre-existing and out of scope here (§7 says not
to wire runtime lifecycle in this rung).

> **Closed 2026-09-01.** `run_cmd::run` now detects a VINDEX3 container
> from its `index.json` generation before any VINDEX2-only reader opens
> it, and routes to `run_cmd_vindex3`: the container is prepared through
> `vindex3_cmd::prepare` — the one authority `larql vindex3 exec` also
> uses — and text streams through the tokenizer the container carries.
> The resolver consolidation above is still open; the execution gap is
> not.

**The local cache has no persisted manifest at all — the filesystem
*is* the registry.** `~/.cache/larql/local/` is a directory of
symlinks (`<name>.vindex` → target dir); `~/.cache/huggingface/hub/`
is the standard hf-hub layout. Both are scanned fresh on every command
(`scan_cached_vindexes`) into a `CachedVindex { repo, snapshot,
size_bytes, source: HuggingFace | Local }` — **which carries no
generation field**. `list`/`resolve_shorthand`/`resolve_cached`/`rm`
treat V2 and V3 entries identically; the only place generation is
ever discovered is a directory scan calling `detect_generation`
per-path, downstream of the cache layer entirely
(`larql-server`'s `--dir` discovery mixes V2 and V3 freely and sorts
them after the fact into two separate collections).

## 2. What exists today — VINDEX2/VINDEX3 generation detection

The sole discriminator, everywhere in the codebase, is a plain integer
field: `index.json`'s `version`. No marker file, no magic byte, no
directory-shape heuristic — deliberately (`format/generation.rs:1-33`
states this as policy).

```rust
// crates/larql-vindex/src/format/generation.rs
pub const V2_MIN_SCHEMA: u32 = 1;
pub const V2_CURRENT_SCHEMA: u32 = 2;
pub const V3_MIN_SCHEMA: u32 = 3;
pub const V3_CURRENT_SCHEMA: u32 = 4;

pub enum ContainerGeneration { V2, V3 }

pub fn detect_generation(dir: &Path) -> Result<ContainerGeneration, VindexError> {
    // reads only index.json's `version` field via a minimal probe struct —
    // deliberately does not fully deserialize, so a V3 index.json (whose
    // shape a V2 config struct can't model) reports "wrong generation"
    // instead of a confusing parse error.
}
```

This is a solid, well-tested pattern worth imitating for a new
ABI-version probe, not something to route around.

`crates/larql-server/src/bootstrap/load.rs::load_artifact` is already
the single choke point that decides V2 vs V3 for *serving* — detect
once, dispatch once, refuse-with-named-flags for any option a V3
binding can't honor (`unsupported_v3_options`). This is the right
shape to extend with a `Vindex3Abi` check inserted before
`load_v3_model` is called — not a resolver-side reimplementation.

## 3. The ABI gap — genuinely new, not a wrap-around

**No ABI/runtime-compatibility version concept exists for VINDEX3
today.** Grepped exhaustively: no `Abi` type, no `abi_version`, no
`min_larql_version`, no `runtime_version` field anywhere.
`format/vindex3/profile.rs:47`'s own comment: *"the VINDEX3 ABI is
explicitly not frozen yet."*

`Vindex3Index.version: u32` is a **schema** version (currently 4) —
"does this binary's `index.json` parser understand this shape", not
"was this container built for a runtime capability this binary lacks."
Two things are conflated by that one field today; a `Vindex3Abi` in
the registry's `ResolvedVindex3` is new, separate machinery.

The word "admissible" is used heavily in the codebase, but exclusively
for **pre-encode** semantic representability (`larql vindex3 plan`:
can this HF checkpoint become a valid VINDEX3 graph at all) and its
post-hoc drift re-check — never for "can this runtime load this
already-built container." `capability::scope`'s `DocumentCapabilities`/
`ProfileCapabilities` similarly answer "what can this resolved profile
serve given the bytes present", not ABI compatibility. **A load-time
admissibility/ABI gate for an already-built container would be new
machinery**, though the `plan::report::Finding`/"collect every
blocking fact, verdict = no blockers" pattern is a good template to
copy for it.

"CAP-0/CAP-1" (mentioned in prior project memory) does not exist
anywhere in the repo under that name.

## 4. The provenance gap — the reason `registry/*.json` cannot be started yet

This is the finding that matters most for sequencing, per the explicit
instruction not to start with JSON files until `publish` was inspected:

- `larql publish`/`larql slice` write **no manifest of their own** —
  they move/upload files and re-derive display titles from
  `index.json`. `PublishOptions` is upload plumbing only.
- **An authoritative, versioned provenance schema already exists**:
  `larql_vindex_spec::VindexManifest` / `Source`
  (`crates/larql-vindex-spec/src/lib.rs`) — HF repo, revision, base
  model SHA, per-shard checksums, extractor version/SHA, timestamp.
  JSON-Schema-mirrored; the crate's own README states "Rust types win"
  on conflict. **But it is wired only to VINDEX2's `index.json`** — its
  own module doc describes "a dense Gemma-shaped extraction."
- **`Vindex3Index`, VINDEX3's own root manifest, carries zero
  provenance fields** — no `source`, no `checksums`, no HF repo/revision,
  no extractor SHA, no timestamp. Only identity (`model`/`family`) plus
  structure (`segments`/`profiles`/`variants`).
- `larql-factory::Recipe` is the closest thing to a build-intent
  registry (pinned `hf_repo`+`revision`, extractor tool/version/level,
  output presets, publish target) — but it describes *what should be
  built*, not a queryable index of what already exists, and its own
  code comments call the build-driver side "aspirational."
- The one existing "official artifact" discovery mechanism —
  `library_name: larql` HF model-card frontmatter tag
  (`crates/larql-factory/src/card/frontmatter.rs`), intended to make
  vindexes filterable via `huggingface.co/models?library=larql` — is
  hard-gated to `vindex_spec_version: 1` (VINDEX2/v1-manifest only,
  the validator rejects any other value) and is itself flagged as an
  unresolved open question in `docs/vindex-factory.md` §13.1 (model
  repo vs. dataset repo tension).

**Verdict**: there is no existing VINDEX3-native provenance struct to
wrap. The registry design has an open choice — carry provenance
out-of-band in the registry entry (reusing `Source`'s *shape*, not its
VINDEX2 wiring), or extend `Vindex3Index` itself with a
`Source`-shaped field. The latter is a container-format change with a
much larger blast radius than this rung's stated scope. **§8 proposes
out-of-band as the default for this rung** and flags it as the one
decision most worth confirming before writing the schema.

## 5. HF pull/download — asymmetric with publish, VINDEX2-shaped

- `resolve_hf_vindex`/`resolve_hf_vindex_with_progress`/`download_hf_weights`
  (`crates/larql-vindex/src/format/huggingface/download/mod.rs`) fetch
  a **fixed, hardcoded VINDEX2 filename list**
  (`VINDEX_METADATA_FILES`/`VINDEX_BIN_FILES`/`VINDEX_WEIGHT_FILES`).
  A VINDEX3 repo's actual payload (`moe_manifest.json`,
  `routed/layer_NNN.lyrw`) is **not in that list** — pulling a VINDEX3
  repo via `larql pull` today would fetch `index.json` (schema 3/4)
  and silently miss the container. Confirmed wired exactly this way in
  `larql-lql`'s `USE "hf://..."` path: resolve, *then* detect
  generation locally — generation is discovered strictly after a
  possibly-incomplete download.
- `publish`'s upload side (`enumerate_publishable_files`) is already
  **generic** — it walks the source directory's actual shape (root
  files + one level of subdirectories), so it structurally handles a
  VINDEX3 directory fine. The asymmetry (generic upload, fixed-list
  download) is the sharpest HF-layer seam.
- **No org/namespace enforcement exists anywhere.** `is_hf_path` is a
  bare `"hf://"` prefix check. `chrishayuk/*-vindex` appears
  throughout docs/tests purely as the author's personal example, never
  checked or allow-listed in code. There is no LARQL_HF_ORG-style
  environment-variable config. An official short-name → HF-repo
  mapping is greenfield.
- Sibling/preset conventions (`{repo}-{preset}`, `client`/`attn`/
  `embed`/`server`/`browse`) exist only because `larql slice` can
  carve V2 vindexes — `slice` explicitly refuses VINDEX3 containers.
  The naming *template* is reusable; the preset vocabulary is
  VINDEX2-file-layout-shaped and doesn't map onto VINDEX3 segments.

## 6. The existing analogue to a "variant" string

VINDEX3's own format already has almost exactly the vocabulary a
registry `variants` map needs, just not surfaced anywhere in the
CLI/cache layer yet:

```rust
// crates/larql-vindex/src/format/vindex3/variants.rs
pub struct StoredVariant { pub storage: String, pub fidelity: Fidelity }
pub struct RegionSetVariants { pub baseline: String, pub variants: BTreeMap<String, StoredVariant> }
pub struct VariantCatalogue { sets: BTreeMap<String, RegionSetVariants> } // #[serde(transparent)]

// crates/larql-vindex/src/format/vindex3/profile.rs
pub struct Profile { pub name: String, pub selects: BTreeMap<String, String> }
```

`Vindex3Index.select_profile(name)` is the resolution entry point;
`larql show <v3-dir>` is the *only* place in `larql-cli` that
currently exercises it, purely for display. **There is no CLI flag on
`run`/`chat`/`serve` to select a profile by name** — `load_v3_model`
always opens the container with no profile argument. A registry
`variant` string (e.g. `27b-nvfp4`) maps onto `Profile.name` almost
exactly; wiring a `--profile` flag through to `Vindex3Runtime::open`
is a natural, small follow-up but is not required for this rung
(the static test registry can name a profile in its manifest without
the CLI needing to pass it anywhere yet).

## 7. Scope for this rung (unchanged from the instruction, restated for reference)

1. Versioned registry manifest/schema — **VINDEX3-only**; `format:
   "vindex3"` structurally required, not a switchable field.
2. Model-name/variant grammar, defined and tested.
3. One shared resolver abstraction — `crate::ResolvedVindex3`, not a
   generic `ResolvedModel`.
4. A tiny static local test registry — no website, no remote registry
   service, no network API.
5. Deterministic default-variant selection; unknown-model/variant
   refusal; ABI/runtime compatibility refusal; explicit `hf://`/local
   resolution — all proven by tests.
6. No Mac app wiring, no runtime-lifecycle wiring.

**Explicitly not required by this rung** (raised in §1/§5, left as
follow-ups): consolidating the three existing resolvers into calling
the new one; wiring `--profile` through to `Vindex3Runtime::open`;
fixing `larql pull`'s VINDEX2-fixed-file-list download so a VINDEX3
repo actually round-trips; giving `larql run` a VINDEX3 execution path
(done 2026-09-01, `run_cmd_vindex3`).

## 8. Confirmed architecture decisions (2026-08-23)

1. **Provenance placement (§4): out-of-band in the registry manifest.**
   `Vindex3Index` describes the container itself; the registry manifest
   answers a different question — "where did this published build come
   from" — and the two are deliberately not coupled. A registry variant
   carries `source: { repo, revision }`, a **V3-registry-native type**
   (not a reuse of `larql_vindex_spec::Source` — reusing a VINDEX2-wired
   type just because its fields happen to match would leak the exact
   coupling this decision removes), and `revision` must be an immutable
   pin, never `main`/`latest`/`HEAD`/unfrozen-branch for an official
   entry. If VINDEX3 itself ever needs embedded reproducibility
   provenance (e.g. detached/offline verification with no registry
   present), that is a deliberate, separate container-format decision —
   not a side effect of this rung.
2. **Consolidation scope: additive only, this rung.** The new resolver
   establishes the authoritative VINDEX3 reference semantics without
   becoming responsible for preserving every existing resolver's quirk;
   it reuses genuinely shared primitives (`detect_generation`,
   `is_hf_path`) but does not delegate its semantics back to
   `cache::resolve_model`, `load_artifact`, or
   `pull_cmd::looks_like_hf_repo` — otherwise the new abstraction would
   just inherit the inconsistencies §1 found. "Additive" must not mean
   "speculative dead code": this rung's tests pin the full contract
   (deterministic default variant, unknown model/variant, ABI refusal,
   explicit hf://\+local resolution, VINDEX2 refusal, malformed
   references) end-to-end. **Banked follow-up rule**: once this
   resolver's contract is proven, the three existing resolution paths
   are meant to converge onto it as a "resolver convergence" rung — they
   are not intended to remain permanently parallel.
3. **Module home: `larql-vindex`, as a dedicated `registry` module —
   not under `format::vindex3`.** `larql-vindex` already owns the
   adjacent concepts (generation detection, HF path handling) and is
   already a shared dependency of `larql-cli`/`larql-server`. The
   registry is not part of the on-disk VINDEX3 format — it's a
   distribution/identity layer *for* VINDEX3 — so it lives at
   `crates/larql-vindex/src/registry/`, a sibling of `format/`, not
   nested inside it. No new crate: per "don't extract crates
   speculatively", extraction becomes evidence-driven only if registry
   logic later grows substantial independent networking/caching/
   signing/publishing machinery.

## 9. Rung 1 — what was built

`crates/larql-vindex/src/registry/` (see its module doc for the
full picture): `reference.rs` (the four-form grammar — `ModelName`/
`VariantName` newtypes, `ModelReference`/`ExplicitReference`, disjoint
by construction because a `ModelName` structurally cannot contain `/`),
`manifest.rs` (`RegistryManifest`/`RegistryModel`/`RegistryVariant`/
`RegistryArtifactRef`/`Provenance`, schema-versioned,
`validate()`/`from_json()` reject a dangling default variant or an
unpinned revision before the manifest is usable), `abi.rs`
(`Vindex3Abi` — one supported value, exact match, deliberately no
compatibility range invented ahead of a second value existing),
`resolver.rs` (`resolve()` — the one entry point; `Vindex3Resolution::
{Registry(ResolvedVindex3), Explicit(ArtifactRef)}`, kept as two output
shapes rather than forcing name/variant/ABI/provenance placeholders
onto an explicit `hf://`/local reference that has no registry identity
to report), `error.rs` (`RegistryError`, wraps `VindexError` via
`#[from]` for the one place this resolver reuses a VINDEX3 primitive —
`detect_generation`, in the explicit-local-path arm, which refuses a
VINDEX2 directory even through the escape hatch), `fixtures.rs` (the
tiny static test registry — `qwen3.8` with two variants, public and
unconditional, following the `format::vindex3::fixtures` precedent so
`larql-cli`/`larql-server` tests can reuse it later without duplicating
data). Colocated `*_tests.rs` files per source file (the
`generation.rs`/`generation_tests.rs` precedent) plus an end-to-end
`crates/larql-vindex/tests/vindex3_registry.rs` against the public API.

Gates: 64 colocated unit tests + 7 integration tests, all green;
`cargo clippy --all-targets -- -D warnings` clean; `cargo fmt --check`
clean; 100% region/line/function coverage on all five new source files
(`abi.rs`, `fixtures.rs`, `manifest.rs`, `reference.rs`, `resolver.rs`)
via `cargo llvm-cov`, well above the 90% floor. Downstream crates
(`larql-cli`, `larql-server`, `larql-lql`) still `cargo check` clean
against the new crate-root re-exports. Not wired into `larql run`/
`serve`/`pull`/the three existing resolvers, and not wired into the Mac
app or runtime lifecycle — both explicitly out of scope for this rung
(§7).

## 10. Rung 2 — resolver convergence (2026-08-23)

Next-rung decision (user, 2026-08-23): with rung 1's contract proven,
converge the three existing resolvers (`cache::resolve_model`,
`larql-server`'s `load_artifact`, `pull_cmd`'s HF heuristic) onto it
one seam at a time — not `--profile` CLI wiring next. Internal
structure: **2A** `cache`/`serve` convergence, **2B** server artifact
convergence, **2C** VINDEX3 HF pull/download correctness (the
finding in §5 that `larql pull` on a real VINDEX3 repo would silently
miss the container — a distribution/download-semantics fix, not a
resolver swap, tackled last).

### 10.1 The claimed/unclaimed boundary (confirmed)

Two facts changed convergence's shape from the literal reading of
"short name → VINDEX3 registry resolver ONLY, no fallback":
`resolve_shorthand` today searches both caches generation-mixed and is
the *only* way users reference a locally-linked/pulled model (V2 or
V3) by short name — there is no separate "legacy alias" mechanism; and
rung 1 shipped no real registry data, only a test fixture. Applied
literally today, every bare-shorthand reference would refuse until
real entries exist.

**Confirmed boundary**: registry ownership is authoritative *only once
a name is claimed by the registry*.

```text
bare name
  |
  +-- registry contains name (registry.models.contains_key)
  |      -> VINDEX3 registry resolution, exclusively
  |      -> success OR hard failure — no legacy/cache fallback, ever
  |
  +-- registry does not contain name
         -> existing cache-shorthand lookup (V2/V3 mixed, unchanged)
```

Encoded as membership (`registry.models.contains_key(name)`), **not**
by pattern-matching `resolve()`'s `UnknownModel` error and falling back
on that specific variant — the two look identical today but only
membership stays correct once a claimed name can fail for other
reasons (bad variant, incompatible ABI) that must never fall through.
The banned pattern, explicitly:

```rust
// NEVER: silently downgrades a claimed name's real failure into a guess.
resolve_registry(name).or_else(|_| resolve_legacy(name))
```

Controls this boundary must satisfy (pinned as tests — see 10.2):
unclaimed name + cached V2/V3 → legacy succeeds unchanged; claimed name
+ valid entry → registry wins even over a same-named cache entry;
claimed name + bad/missing variant, or incompatible ABI → hard error,
legacy never touched; explicit `hf://`/local-path references bypass the
claim check entirely (§10.2 explains why); malformed grammar falls
through to legacy (it was never a registry lookup to begin with).

Migration story this buys: `qwen3.8` today resolves via legacy cache
shorthand; the day it's added to the registry, it silently becomes
registry-owned — no CLI syntax change, no flag day where all shorthand
breaks before the registry has anything in it.

### 10.2 2A scope: `serve`'s trampoline only (confirmed, built)

`cache::resolve_model` is shared by `run`/`chat`/`show`/`slice` (all
VINDEX2-only — no VINDEX3 execution path exists for them, §1) and
`larql serve`'s CLI trampoline (the only command that can actually run
VINDEX3 today). 2A touches **only** the `serve` trampoline
(`crates/larql-cli/src/commands/primary/serve_resolve.rs`, called from
`main.rs::run_serve`) — `run`/`chat`/`show`/`slice` keep calling the
untouched `cache::resolve_model`.

Within `serve`'s own resolution, only the **bare-name branch** gets the
claimed/unclaimed check. An explicit `hf://owner/repo` or
`/local/path` argument bypasses the claim check and keeps using
`cache::resolve_model` unchanged, deliberately: both forms already
dispatch correctly on whichever generation they find
(`load_artifact`'s own `detect_generation` call downstream), so routing
them through the new resolver's stricter explicit arms — which refuse
a VINDEX2 local directory outright — would *regress* existing VINDEX2
`serve` usage, not fix anything. Widening those forms into the new
resolver is a later decision, not a side effect of this rung.

**The other fix landed here**: `run_serve` used to do
`cache::resolve_model(path).unwrap_or_else(|_| path.clone())` —
silently substituting the raw, unresolved string on *any* resolution
failure (including an ambiguous shorthand with a good error message)
and handing it across the process boundary to a server binary with no
shorthand knowledge at all, which then failed three layers down with a
confusing IO error. This now propagates the real error. Nothing
legitimate depended on the fallback: `cache::resolve_model`'s own
"already a local directory" branch already accepts a raw valid path
before ever reaching the failure cases the fallback used to catch.

**Production registry is still empty** — `serve_resolve::
production_registry()` returns a manifest with zero models. No official
VINDEX3 model is published yet, and where real registry data will
actually come from (embedded? a file? fetched?) remains an open,
separate decision. Shipping this wiring now means zero behaviour
change for any caller today, and the claimed/unclaimed split activates
itself correctly the moment a real entry is added — no second
migration required.

Gates (at landing): 10 unit tests via dependency-injected
`legacy`/`fetch_hf` closures pinning every control in 10.1, plus one
hermetic test of the real (non-injected) `resolve_serve_target` against
a tempdir; full `larql-cli` suite (763 tests) green; clippy/fmt clean.
Superseded by 10.3's refactor — see there for the current shape.

### 10.3 2B: `load_artifact` convergence, and hoisting the dispatch into `larql-vindex`

Grounding 2B in `larql-server`'s actual code (rather than assuming
"server artifact convergence" meant an ABI gate — a raw V3 container has
no ABI field to check against; that would need a container-format
change, out of scope) found `load_artifact` called from **three**
places, none going through the CLI trampoline at all: the server
binary's own CLI arg (`bootstrap/mod.rs`), `--dir` bulk discovery (same
file), and the `/v1/runtime/model` HTTP lifecycle endpoint
(`routes/runtime_lifecycle.rs`, landed in PR #300). None of the three do
shorthand resolution — `load_artifact` itself only ever understood
`hf://` or a literal path. `larql-server qwen3.8` invoked directly, or a
`POST /v1/runtime/model {"path": "qwen3.8"}`, would fail today exactly
like `larql serve qwen3.8` did before 2A.

**The dispatch moved into `larql-vindex::registry` (new `production`
module)**, rather than growing a second, independently-maintained copy
inside `larql-server`: two copies of "is this name claimed" is exactly
the divergence the initiative exists to remove — `qwen3.8` must mean
the same VINDEX3 identity whether reached via `larql serve`, the server
binary invoked directly, or the HTTP endpoint. `registry::production`
now owns:

- `production_registry()` — moved verbatim from `serve_resolve.rs`
  (still empty; the real-data-source question is still separate and
  still open).
- `resolve_claimed(raw, registry) -> Result<Option<PathBuf>, RegistryError>`
  — `Ok(None)` = not a claimed name, caller does its own thing; `Ok(Some(path))`
  = claimed, resolved, and materialised (fetched via
  `resolve_hf_vindex` if not already cached); `Err` = claimed but
  failed — the hard-refusal contract from 10.1, now enforced in one
  place for every caller.
- `resolve_claimed_with(raw, registry, fetch_hf)` — the injectable core,
  `pub` so every caller's own tests can prove the claimed/unclaimed
  wiring hermetically, matching this codebase's own
  `resolve_shorthand`/`resolve_shorthand_from` convention.

`resolver.rs` gained `lookup_claimed_variant` (`pub(super)`): the
model/variant/ABI lookup `resolve_registry` already did, factored out so
`production::resolve_claimed_with` can read a claimed variant's
`{repo, revision}` directly — a `RegistryArtifactRef` plain struct, not
the `ArtifactRef` enum `resolve_registry` wraps it into for the public
API. Consuming the concrete struct instead of the wider enum removed a
whole `match ArtifactRef::HuggingFace {..} else { unreachable!() }`
arm this file no longer needs to defend against (that arm was real
coverage debt: `RegistryArtifactRef` has no local form by schema, so the
`Local` arm was permanently dead code the 90% floor still had to count
against). `resolve_claimed`'s default `fetch_hf` is the bare
`resolve_hf_vindex` function item, not a closure literal wrapping it —
a closure creates its own never-covered region (the fetch only runs on
a real, successful claim; a unit test can't exercise that without
touching HF for real), a function-item reference has no separate body
to measure at all. Both changes were necessary, not cosmetic: without
them `production.rs` measured 78.79% lines against a new-file 90% floor
under the *actual* `scripts/check_coverage_policy.py` gate (verified
directly, not assumed) — after, 93.10%.

**`load_artifact` itself**: `resolve_artifact_path` (a new private
helper) tries `resolve_claimed` first; `Ok(None)` falls through to
exactly the prior `is_hf_path`/literal-`PathBuf` logic, so an `hf://`
reference or an existing local directory is untouched (both are
structurally never a bare `ModelReference::Registry` form, so
`resolve_claimed` always answers `None` for them — proven, not assumed,
by a dedicated `hf://` test). **One correctness fix landed alongside**:
the VINDEX2 branch used to call `load_single_vindex(path_str, ...)`
with the *original*, unresolved string — for an `hf://` input this just
meant re-resolving it a second time (wasteful, not wrong); for a claimed
registry name it would have been a real bug, since `load_single_vindex`
has no idea how to resolve a bare registry name and would have tried
`PathBuf::from("qwen3.8")` as a literal (nonexistent) relative path.
`load_artifact` now passes the already-resolved directory string to
both branches.

**Known accepted edge case**: a bare relative directory name with no
path separator at all (e.g. `larql-server myvindexdir` from a cwd
containing `myvindexdir/`) is syntactically a `ModelName` too. Today
this is inert — the production registry is empty, so `resolve_claimed`
always answers `None` and the existing literal-path behaviour applies
unchanged. Once a real name is registered, a local directory happening
to share that exact name would be shadowed by the registry claim — the
same namespace-collision tradeoff any claimed-name system accepts (npm,
pip, …), not something this rung needs to solve.

Gates: full crate suites green after the change — `larql-vindex` 74 registry
unit tests + 7 integration tests (up from 64/7 pre-2B), `larql-cli` 761
tests (down from 763: two success-path network tests were retired as
redundant with `production_tests.rs`'s own hermetic coverage of the
same contract), `larql-server` 575+ tests including 5 new
`resolve_artifact_path_tests`; clippy `--all-targets -D warnings` and
`cargo fmt --check` clean on all three crates; `--no-default-features`
build of `larql-cli` (matches its CI invocation exactly) still clean.
**Coverage verified against the actual CI gate script**, not just a raw
percentage: `cargo llvm-cov --package <crate> --summary-only` (full
suite, matching each crate's CI workflow exactly — `larql-server` even
reruns with `--test-threads=1` to match) followed by
`cargo llvm-cov report --json` piped into
`scripts/check_coverage_policy.py` against each crate's own
`coverage-policy.json`. Result: `larql-vindex` passes except a
**pre-existing, unrelated** `quant/convert.rs` baseline miss (81.54% vs
81.90%, a file this rung never touched); `larql-server` passes cleanly,
`bootstrap/load.rs` ratcheted 66.0% baseline → 79.74% actual;
`larql-cli` has no per-file policy gate in CI at all (confirmed by
reading its workflow, not assumed) — its own coverage-policy.json scopes
only `bench/`/`dec_bench/`.

### 10.4 2C scope, as reframed by the user: "VINDEX3 pull semantics," not a filename-list swap

The frozen rule going in: **never replace the VINDEX2 hardcoded
download list with a new VINDEX3 hardcoded download list** — that
would recreate the exact failure mode one generation later. 2C does two
things together: (1) kill `pull_cmd`'s independent HF/name heuristic,
routing it through the same claimed/unclaimed dispatch 2A/2B share; (2)
turn the actual HF downloader generation-aware, deriving VINDEX3
completeness from the container's own structure rather than a second
guessed list.

### 10.5 2C — VINDEX3 pull semantics (2026-08-24)

**Why "download the complete repo snapshot," not "enumerate from
`index.json`'s fields."** `Vindex3Index.segments` + `representations` +
`moe_manifest`/`system_graph` looked at first like enough information to
enumerate every required file exactly — cheaper on bandwidth than a full
snapshot. Rejected: the M2 migration rung's own capability-snapshot
side-channel (`tokenizer.json` and siblings) is copied onto a container
without ever being named in `index.json` at all. Hand-enumerating "which
index.json fields count as a file" would just be the fixed-list bug
this rung exists to fix, one layer of indirection removed — correct
today, silently wrong the next time the format grows a file class this
code doesn't know about. A repo dedicated to one vindex has no
meaningful unrelated bulk to over-fetch, so the repo's own file listing
(`repo.info().siblings`, the same enumeration
`resolve_hf_model_with_progress` already uses for upstream checkpoints)
*is* the required payload for VINDEX3 — downloaded in full, every listed
file required (a failed fetch is now a hard error naming the file, not
a silently-skipped candidate the way VINDEX2's optional metadata files
are).

**`resolve_hf_vindex_with_progress` is now generation-aware** — the one
shared transport function every VINDEX3 pull path routes through:
fetch `index.json` (minimal control metadata) → `detect_generation` →
VINDEX2 unchanged (`vindex_core_files()`, optional-skip, exactly as
before — no behaviour change for the shipped generation) → VINDEX3:
`repo.info()` lists every sibling, each one downloaded through the same
cache-aware `fetch` closure V2 already used, any failure a hard error
naming the file. `resolve_hf_vindex_complete` (new, no-op-progress
wrapper, mirrors this crate's `SilentXCallbacks` convention) gives
non-interactive callers (the registry) the same completeness without a
progress bar.

**Identity resolution, transport, and validation are three separate
phases**, per the user's explicit instruction — not one function that
guesses all of it:
- **Identity**: `registry::resolve_claimed_hf_reference` (new) —
  claimed-name lookup only, no fetch, returns the pinned
  `hf://repo@revision` string. Split out of `resolve_claimed_with` so
  `pull`'s own progress-bar UX keeps driving the actual download
  (`resolve_claimed_with` still exists, now built on top of this plus
  `fetch_hf`, for `serve`/`load_artifact`'s silent-fetch use).
  `pull_cmd::resolve_pull_hf_path` calls it first; `Ok(None)` (not
  claimed) falls through to the *existing* `normalise_hf_path`/
  `looks_like_hf_repo` heuristic — now purely the *unclaimed* path, not
  `pull`'s only identity concept. A claimed name's failure (unknown
  variant, incompatible ABI) is a real refusal, exactly the 10.1
  contract, never rescued by the legacy heuristic.
- **Transport**: `resolve_hf_vindex_with_progress` (§ above) — a fetched
  reference's bytes, complete, generation-aware.
- **Validation**: `format::vindex3::validate_downloaded_container` (new)
  — "is this on-disk directory actually a complete VINDEX3 container."
  The container's own `index.json` decides which of the two existing
  structural loaders applies (never guessed): `moe_manifest` present →
  routed-MoE → `Vindex3Container::open`; absent → system-graph →
  `inspect_container(dir, true)` (payload-verifying). Discovering these
  are genuinely two different container shapes with two different
  "real" loaders (`Vindex3Container::open` refuses a system-graph
  container by name, directing to the other) was itself a finding —
  there is no single existing "open any VINDEX3 container" function to
  reuse, so this one is a thin, deliberately-obvious dispatcher over the
  two that already exist, adding no new structural checks of its own.
  Called from both `pull_cmd::pull_one` (after every V3 download,
  claimed or explicit) and `registry::resolve_claimed_with` (so
  `serve`/`load_artifact` get the identical completeness guarantee
  `pull` does — not a separate, weaker one).

**Registry entries can gate ABI before download; explicit `hf://`
references cannot** (the user's explicit caution) — respected by
construction, not by a special case: ABI is a *registry manifest* fact
(`Vindex3Abi` lives on `RegistryVariant`, checked inside
`lookup_claimed_variant` before any network call), never a container
fact. An explicit `hf://` reference has no registry entry to declare an
ABI at all, so no ABI check applies to it — completeness (validation, §
above) is the only guarantee it gets, which is exactly what's honestly
available before a container's own bytes are local.

**The acceptance test**, per the user's explicit framing ("not these
expected filenames were requested — the pulled result actually opens"):
a real, self-encoded VINDEX3 fixture (`encode_fixture_container` +
`miniature_glimmer`) served over a mocked HF endpoint (`mockito`,
`HF_ENDPOINT`, this crate's existing `HfTestEnv` pattern), every file it
holds discovered dynamically by walking the fixture on disk — never
hardcoded in the test — then `resolve_hf_vindex_with_progress`'s result
opened through `inspect_container` (the real loader for this fixture's
shape). A sibling test claims a file in `repo.info()` the mock never
actually serves and asserts a hard failure naming it. Both pass.
`validate_downloaded_container` gets its own gates too: a complete
routed-MoE fixture validates, a complete system-graph fixture validates,
a routed-MoE fixture missing one segment file fails naming that segment,
and a directory with no `index.json` fails.

**Controls pinned** (the user's list, verified): claimed registry V3 →
complete pull succeeds (`a_claimed_name_fetches_and_validates_its_pinned_artifact`,
`a_claimed_name_resolves_to_its_pinned_hf_reference`); claimed with
pinned revision → exact revision used (asserted in the same tests via
the exact `hf://repo@revision` string); explicit `hf://` V3 → complete
pull succeeds (the download-layer acceptance test, generation-agnostic
— it doesn't know or care whether its caller was a claimed or explicit
resolution); explicit local path → pull is not involved (`pull_cmd` has
no local-path branch at all, unchanged); V2 repo → existing behaviour
unchanged (the `ContainerGeneration::V2` arm is byte-for-byte the
pre-2C code, gated by the existing test suite); V3 repo missing required
payload → hard failure (both at the transport layer — the mocked-missing-file
test — and the validation layer — the missing-segment test); claimed-name
registry failure → never falls into `pull_cmd`'s heuristic
(`a_claimed_name_with_an_unknown_variant_never_falls_through_to_the_heuristic`,
`..._incompatible_abi_never_falls_through...`).

Gates: full suites green post-change — `larql-vindex` 2728 lib tests +
every integration binary (up from 2723), `larql-cli` 766 (up from 761),
`larql-server` 575 unchanged (not touched this rung, re-verified against
the updated `larql-vindex`); clippy `--all-targets -D warnings` and
`cargo fmt --check` clean on all three crates; `larql-cli
--no-default-features` (its exact CI build) clean. Coverage verified
against the real CI gate exactly as in 2B: `larql-vindex` passes except
the same pre-existing, unrelated `quant/convert.rs` baseline miss (not
touched this rung either); every new/changed file individually clears
its bar (`registry/production.rs` 97.5%, `format/vindex3/verify.rs`
93.7%, `format/huggingface/download/mod.rs` 78.5% against its existing
64.0% debt baseline — ratcheted up, not down).

### 10.6 Resolver convergence: done for the product path

`serve qwen3.8`, `larql-server` invoked directly or via
`/v1/runtime/model`, and `pull qwen3.8` now all share one definition of
"what model did the user ask for" — the same claimed/unclaimed boundary,
the same registry data, the same completeness guarantee. Per the user's
own framing, this is the point resolver convergence is "complete enough
for the product path"; the next architectural gap is not `--profile`
CLI wiring but **the first real production registry entries / a
publishing pipeline** — the static test registry is what's actually
stopping this from being useful outside tests. Also still open,
explicitly deferred rather than forgotten: consolidating `run`/`chat`/
`show`/`slice`'s own use of `cache::resolve_model` (2A deliberately left
these untouched — no VINDEX3 execution path exists for them yet, per
the rung-1 inventory); widening explicit `hf://`/local-path forms into
the new resolver's stricter arms; `--profile` CLI wiring once a real
registry entry exists to select a variant of.

## 11. Rung 3A — the production registry's data source (2026-08-24)

Deliberately narrow, per the user's own scoping: canonical
`registry/index.json` + `registry/models/*.json` at the repo root,
parsed by the *existing* production Rust schema, containing one real
entry (`granite-4.1-3b`) — no promotion command in this rung. The
point of keeping R3A alone is proving production registry data can
exist, be validated, and be consumed by the resolver **without fixture
machinery**, before any promotion tooling exists that could mask a
problem in the representation itself.

**The data-source question §7 left open** ("embedded? a file?
fetched?") is answered: **compile-time embed**, via `include_str!`
(`crates/larql-vindex/src/registry/embedded.rs`), matching
[`fixtures`]'s existing "static, in-process, no network" precedent. A
runtime file read relative to the process only works from a source
checkout — release binaries (ADR-0026) ship standalone with no repo
nearby; a remote fetch would add a second service dependency beyond HF
for every `pull`/`serve`, which this rung's own non-goals already
ruled out. Accepted consequence: a new or updated entry reaches users
on the next binary *release*, not instantly on merge — no regression,
since "official status conferred by PR merge" (§8 of the publishing
doc) was already going to need a release to actually ship the entry.

**File split, and why two files, not one**: `registry/index.json`
names which models exist under which manifest schema version;
`registry/models/<name>.json` holds one model's full body
(`RegistryModel`: default variant + variants). `embedded.rs` parses
the index, then for each named model looks up a **small, explicit**
`include_str!` list (`embedded_model_json`) rather than a glob —
`include_str!` needs a compile-time literal path, and R3A is
deliberately one entry; a generalised N-model embed mechanism ahead of
a second real entry existing would be exactly the premature machinery
this initiative has avoided elsewhere. A second entry adds one
`include_str!` line and one match arm, reviewed in the same PR as its
`registry/models/*.json` file.

**Testability**: `embedded.rs`'s core (`assemble_registry`) takes an
injected `lookup` closure — the same convention `production.rs`
already used for `resolve_claimed`/`resolve_claimed_with` — so every
failure branch (malformed index, an index name with no matching embed,
malformed per-model JSON, a manifest that parses but fails
`validate()`) is provable against synthetic data, independent of the
real embedded files. `production_registry()` stays the plain,
infallible `RegistryManifest` every existing caller already expects;
it panics only if the checked-in `registry/` files themselves are
malformed, which R3B's CI conformance gate exists to catch before
merge — the same "fixture serialises" contract
`fixtures::tiny_static_registry_json` already relies on.

**R3A's acceptance gate**:

```text
production_registry()
    -> reads canonical repo registry
    -> granite-4.1-3b exists
    -> resolves to pinned VINDEX3 artifact
```

See `docs/vindex3-registry-publishing-design.md` §8 for the
`granite-4.1-3b` publish itself, including a real `create_hf_repo`
namespace bug found and fixed along the way.

**A second real bug, found by actually running the acceptance gate, not
just wiring it**: `larql pull granite-4.1-3b` + `larql serve
granite-4.1-3b` initially failed — `open VINDEX3 container: IO error:
No such file or directory` — even though `pull` itself reported
success. Root cause in `cached_snapshot_file`
(`format/huggingface/download/mod.rs`): its "already cached" fast path
accepted a blob whose bytes existed locally (deduped — this session had
already pulled a byte-identical `target.embedding.bin` from an
unrelated repo) but for which the *pinned revision's own* snapshot
symlink didn't exist, falling back to returning the bare blob path. The
V3 completeness loop that calls it discards the returned path entirely
(`fetch(...).ok_or_else(...)?` — only checks `Option`-ness), so it
reported success while the pinned revision's `snapshots/<rev>/` never
actually got the symlink the container loader needs. Fixed by making
the fast path only ever accept the pinned revision's own snapshot
symlink — an unpinned `revision: None` now always misses immediately
and falls through to `download_with_progress`, which creates the
correct symlink without re-transferring already-local bytes. Regression
tests pin the exact failure shape (blob present, symlink absent for the
pinned revision → miss) and the now-narrower unpinned-revision case.

**R3A acceptance gate: PASSED for real**, both bugs above fixed first —
`larql pull granite-4.1-3b` resolved through the production registry to
`hf://larql/granite-4.1-3b@1048a8eb2fec5812a698e76d7e603527d0475c17`
and downloaded a complete container; `larql serve granite-4.1-3b`
loaded it and served a real `/v1/chat/completions` request end to end
("The capital of France is" → "Paris."). No fixture registry, no manual
local path, no floating HF revision, anywhere in that chain. R3A
shipped as its own PR (#305), separate from and frozen ahead of R3B.

## 12. Rung 3B — registry CI/conformance (2026-08-24)

Pure conformance, no new model, no promotion command — proving a
checked-in `registry/` cannot become invalid unnoticed, using the exact
same Rust code `production_registry()` already trusts, not a parallel
Python JSON checker that could drift from the runtime's own definition
of "valid."

**`load_registry_from_dir`** (new `registry/check.rs`) is the
filesystem-reading counterpart to [`embedded`]'s compile-time
`include_str!` path — same `assemble_registry` core, same
`RegistryManifest::validate()`, same error types. `embedded.rs`'s index
parsing was factored out into a shared `parse_index` so both entry
points read `index.json`'s shape identically rather than each parsing
it their own way. A model name the index lists with no matching
`models/<name>.json` file is reported naming the exact path expected —
the filename is derived from the index-listed name by construction,
never read from a separate field inside the model's own JSON, so there
is no independent "manifest's own name" that could disagree with the
index (the same single-source-of-truth choice the schema already makes
for everything else).

**`larql registry check [PATH]`** (new `registry_cmd.rs`, a subcommand
group deliberately — R3D's future `larql registry promote` joins it
later, not a bare top-level verb): no `PATH` validates the registry
*embedded in this binary* — `load_production_registry()`'s own data,
which for a CI build is exactly the checked-out PR's `registry/` files,
since `include_str!` embeds whatever was present at compile time. A
`PATH` reads and validates that directory from disk instead, for R3D's
promotion workflow (`write model JSON -> larql registry check ->
git diff -> PR`) to check a *candidate* directory before it's moved
into place.

**CI wiring**: `.github/workflows/larql-cli.yml`'s existing cross-platform
`test` job (ubuntu/windows/macos-14) gained a `registry check` step
right after its existing test step — no new job, since the binary is
already being built there. Runs on all three OSes deliberately, so a
platform-specific path bug (the Windows-absolute-path class
`registry/reference.rs` already hit once, rung 1) would surface here
too, not just in unit tests.

Gates: larql-vindex 2828 lib tests + all integration binaries (+8 from
R3A's 2820: `check.rs`'s own tests); larql-cli 775 (+3); clippy
`-D warnings` and `cargo fmt --check` clean workspace-wide. Coverage
verified against the real `check_coverage_policy.py` gate: unchanged
94.30% total, 43 debt baselines (none newly added), `check.rs` 100%
line coverage, `embedded.rs` 94.12% (both new/changed files clear the
90% floor).

## 13. Rung 3C — publish emits a candidate, never writes the registry (2026-08-24)

One thing, narrowly: **turn a successful `larql publish` into a
deterministic candidate registry record.** Not `registry/models/`
itself (R3D), not a promotion command, not the NVFP4 runtime gap
([[project_vindex3_nvfp4_attention_widening_gap]] in memory) — that's
a separate programme.

**The one invariant, enforced structurally**: candidate generation must
never invent information. `CandidateInputs` (new `registry/candidate.rs`)
names each field by where it came from — `artifact.repo`/
`artifact.revision` are mechanically known (the *real* publish result's
own `{repo, revision}`, never re-derived or guessed), `abi` is an
explicit policy default (`CURRENT_VINDEX3_ABI` — the one value this
binary actually implements, not invented), and `source.repo`/
`source.revision`/`attested_by` are explicit caller input only — never
derived from `--repo`, a filename, or an HF cache path. Missing any
required input is a refusal before any candidate is built, never a
guess.

`build_candidate` validates through the **real** schema: it wraps its
one model into a one-entry `RegistryManifest` and calls
`.validate()` — the exact code `production_registry()` trusts and
`registry check` (R3B) calls — before returning anything. A candidate
that would fail real registry validation (floating revision, empty
attestation) is refused at generation time, not discovered later by
whoever eventually runs `registry check` on it.

**`larql publish --emit-registry-candidate`** (new
`publish_cmd/registry_candidate.rs`) is opt-in and additive only —
plain `larql publish` takes no new code path at all
(`parse_candidate_request` returns `Ok(None)` immediately when the
flag is absent). When passed, it requires `--registry-name`,
`--registry-variant`, `--source-repo`, `--source-revision`, and
`--attested-by` — checked up front, before any network call, so a
missing companion flag is reported immediately rather than after
paying for a real publish. Incompatible with `--no-full` (a candidate
names the full artifact's own revision, which doesn't exist without
publishing it). `--registry-candidate-out PATH` writes the JSON there;
omitted, it prints to stdout. The candidate body is exactly what would
live at `registry/models/<name>.json` — no `name` field inside it (the
name lives in the index and the filename, the same single-source-of-
truth choice `check.rs` already makes) — a caller feeds it straight
into `larql registry check` once R3D exists to consume it.

**Acceptance gate, run for real, not asserted piecewise**: a new test
(`format::huggingface::publish::tests::publish_then_candidate_end_to_end_against_a_mocked_hf_endpoint`)
publishes against a mocked HF endpoint, lets it return a pinned commit,
builds a candidate from *that exact* publish result, and asserts the
candidate's `artifact.revision` is the mocked endpoint's own returned
value (never a hand-typed stand-in) plus exact `source.repo`/
`source.revision`/`attestation` — then separately proves an empty
attestation refuses rather than silently building. Plain `larql
publish` behaviour is unchanged (all 36 pre-existing `publish_cmd`
tests pass unmodified; the flag-omitted path returns `Ok(None)`
before any of the new logic runs).

Gates: larql-vindex 2908 lib tests + all integration binaries (+80 net
over R3B's 2828 — includes unrelated work merged to `main` between R3B
and R3C, plus `candidate.rs`'s own 7 tests and the new end-to-end
acceptance test); larql-cli 787 (+7 from R3B's 775 net of the same
merged-in delta, plus `registry_candidate.rs`'s own tests); clippy
`-D warnings` and `cargo fmt --check` clean workspace-wide. Coverage
verified against the real `check_coverage_policy.py` gate: 94.30%
total, 43 debt baselines unchanged; `registry/candidate.rs` 100% line
coverage; `format/huggingface/publish/mod.rs` (a pre-existing 74.0%
debt baseline) now measures 90.29%, ratcheted further above its floor.

**Per explicit instruction, stopped here**: does not write into
`registry/models/`, does not touch `registry/index.json`, not
committed, no PR opened. R3D (promotion — re-fetch the pinned artifact,
re-validate, run whatever gates exist, write the registry entry) is
next, deliberately not bundled in.
