# Vindex Factory — spec v0.1

**Status:** draft for review
**Owner:** Chris Hayuk
**Depends on:** `larql` (extract/slice/publish/pull, vindex-v1 manifest crate), `chuk-mcp-training` (rig), `chuk-experiments-server` (record)

---

## 1. Purpose

Turn vindex production from a laptop ritual into a **reviewable, reproducible, remote-executed pipeline**.

Today a vindex is made by running `larql extract` on the Mac, `larql slice`, then `larql publish` — 15 GB uploads over a domestic uplink, no record of what parameters were used, and no way for anyone else to reproduce or contribute. The extractor and publisher already exist and work. What's missing is the *factory* around them.

**The one-line shape:**

> A recipe file merged to `main` is a promise that a specific vindex exists on the Hub, built from a pinned upstream by a pinned extractor, verified before it went public.

### 1.1 Non-goals

- Not a general CI system. It builds vindexes; nothing else.
- Not a replacement for `larql extract`. The factory *invokes* the CLI; all extraction logic stays in larql.
- Not a hosted service with an uptime SLO. Between builds, nothing runs.
- Not a community submission portal (yet). G3 opens PRs from outside; G0–G2 are single-owner.

---

## 2. The three planes

The single most important thing to get right is *who owns what*. Three existing systems already have clear ownership; the factory adds no fourth.

| Plane | System | Owns |
|---|---|---|
| **Intent + approval** | GitHub repo (`chuk-vindex-recipes`) | The recipe. What *should* exist. Review, approval, audit trail. |
| **Execution** | The rig (`chuk-mcp-training`, control plane on Fly, workers leased/BYO) | Doing the work. Budget walls, cleanup, telemetry. |
| **Record** | `chuk-experiments-server` | What *does* exist. Build records, lineage, artifact pins, verification reports. |

Corollaries, stated so they don't get relitigated:

- **GitHub holds no state.** No build status badges as truth, no artifacts, no results committed back. A recipe on `main` is a *declaration*; whether it built is a question you ask the experiments server. This preserves the existing artifact-ownership ruling (experiments-server is system of record; the harness delegates via the mirror).
- **GitHub Actions does not build.** The Action validates the recipe and dispatches to the rig. It never touches model weights, never holds an HF write token, never needs more than 2 minutes or 2 GB.
- **The rig does not decide what to build.** It executes a job spec handed to it. No recipe parsing on the worker beyond materialising the CLI invocation.

### 2.1 On Fly.io as the executor

Your sketch said "kicks off the build on something like fly.io (or something)". I'd split that:

**Fly stays the control plane** — it already is (`chuk-mcp-training.fly.dev`, `chuk-train-mcp.fly.dev/mcp`, `chuk-experiments-server.fly.dev`). Dispatch, queue, dashboards, MCP surface: all correct where they are.

**Fly is the wrong executor for anything large.** A Gemma-26B-class extraction wants ~2 TB of NVMe scratch, a fat downlink to pull safetensors from HF, and a fat uplink to push shards back. Fly volumes cap out well below that, are per-machine, are billed continuously, and egress is metered. The rig's leased-worker model — Vast box, hard budget wall, destroy on completion — is exactly the right shape for a burst job with a big disk, and it already exists. DEC-4 already specs an Inkling extraction as precisely this.

So: **two executor classes**, chosen by the recipe's declared footprint.

| Class | Where | For | Bound |
|---|---|---|---|
| `inline` | Fly machine (ephemeral, on-demand, destroyed after) | tiny models — v11, TinyModel, anything under the threshold | ≤ 40 GiB total working set, ≤ 30 min |
| `leased` | Rig worker (Vast/Lambda), single-use join token | everything else | recipe-declared `budget` |

The threshold is a PIN (§13.4). The point is that your instinct isn't wrong — it's just that "Fly" is the *small* path, not the general one, and the general one is already built.

---

## 3. Repo layout

```
chuk-vindex-recipes/
├── recipes/
│   ├── google--gemma-3-4b-it.yaml
│   ├── google--gemma-4-26b.yaml
│   ├── chrishayuk--tinystories-v11.yaml
│   └── ...
├── verify/
│   ├── gemma-smoke-32.jsonl          # prompt sets, content-addressed
│   └── tinystories-smoke-32.jsonl
├── policy/
│   ├── licences.yaml                 # redistribution allowlist (§9)
│   └── naming.yaml                   # repo/tag templates
├── schema/
│   └── vindexbuild.v1.json           # JSON Schema for recipes
├── .github/
│   ├── CODEOWNERS                    # recipes/ → @chrishayuk
│   └── workflows/
│       ├── validate.yml              # on PR
│       └── dispatch.yml              # on push to main
└── README.md
```

**Why its own repo, not inside `larql`:** recipes churn on a different cadence than the extractor, need different CODEOWNERS semantics, and — at G3 — need to accept PRs from people who should not be able to open PRs against the extractor.

### 3.1 Where the *code* lives — RESOLVED

The split is **code in `larql`, data in `chuk-vindex-recipes`**. Not code-vs-spec, and not a third repo for both.

**`larql` gains** (as a `larql-factory` crate in the existing workspace, surfaced as CLI subcommands):

| Piece | Subcommand |
|---|---|
| Recipe schema + validator | `larql recipe validate` |
| Canonicaliser → `build_id` | `larql recipe build-id` |
| Size/cost estimator | `larql recipe estimate` |
| Capability manifest (§15.2) | `larql capabilities` |
| Build-stage driver (§7) | `larql recipe build`¹ |
| Card generator | `larql card render` |
| Verify-from-hub harness | `larql verify`² |

**`chuk-vindex-recipes` holds** recipes, `policy/`, `verify/` prompt sets, and the two workflows — which become thin, because they shell out to a released `larql` binary rather than reimplementing anything.

¹ `larql build` already exists (a Vindexfile-based declarative build, unrelated to this spec) — the driver lives under the existing `larql recipe` subcommand group instead, alongside `validate`/`build-id`/`estimate`.

² Implemented as checksum integrity only, reusing the existing `larql verify` command as-is — not a new reconstruction-fidelity/logit-match harness. Building §8.1's numeric checks needs per-architecture tensor-naming knowledge (mapping "layer N's down_proj" to the right safetensors key per model family) that isn't validatable without real model weights; neither `larql shannon verify` nor `larql dec-bench drift` do vindex-vs-upstream comparison despite an earlier assumption here that one of them did.

Three reasons this line and not another:

1. **One version pin covers everything that determines the bytes.** The recipe pins `extractor.version`; if the build driver lives elsewhere it needs a second pin, or you get silent skew where a new driver changes stage semantics against an old extractor. Two pins is a real reproducibility tax for no gain.
2. **The canonicalisation hazard dissolves by construction.** `build_id` must be identical in the Action and on the worker. If the Action *calls* `larql recipe build-id`, there is exactly one implementation and nothing to keep in sync. This was the only subtle failure mode in §14 and it disappears.
3. **The driver has exactly one consumer.** A new repo earns its keep when a thing has several consumers and belongs to none of them — that was the argument for the tokenizer bench library. The factory driver consumes larql and produces larql's own artifacts. It belongs in larql.

The Action holding no logic is worth stating as a property: it cannot skew from the worker, because it isn't a second implementation of anything.

**Tripwire for revisiting:** if the factory ever builds artifacts that aren't vindexes — dataset packs, tokenizer artifacts, checkpoint conversions — it has become a general artifact-build system with multiple consumers and wants its own home. Until then it doesn't.

**Spec location:** this document goes to `larql/docs/vindex-factory.md`, since it specifies code that lives there; the recipes repo README links to it. Repo copy is canonical over chat outputs, same rule as the DEC spec.

---

## 4. The recipe

One YAML file per published vindex family. Schema-validated, `apiVersion`-stamped.

```yaml
apiVersion: vindex.chukai.io/v1
kind: VindexBuild
metadata:
  name: gemma-3-4b-it
  description: FFN knowledge index for Gemma 3 4B IT

spec:
  source:
    hf_repo: google/gemma-3-4b-it
    revision: 9b5b1a2f7c3e0d1a4b8f2c6e9d0a3b5c7e1f4a82   # REQUIRED, full commit sha
    licence: gemma                                        # must resolve in policy/licences.yaml

  extractor:
    tool: larql
    version: "0.14.2"          # released tag only, never a branch — see PIN 13.3
    level: inference
    dtype: f16
    options:
      down_q4k: true
      drop_gate_vectors: false
      preserve_mtp_head: true

  outputs:
    - preset: full
    - preset: client
    - preset: browse
    - preset: expert-server
      shard_cap_gib: 20
    - preset: attn
    - preset: embed

  verify:
    reconstruction:
      layers_sampled: 8
      seed: 1729
      max_abs_diff: 0.0        # vindex reconstruction is bit-identical; anything else is a bug
      min_cosine: 1.0
    logit_match:
      prompt_set: verify/gemma-smoke-32.jsonl
      top1_agreement_min: 0.99
      bits_per_char_drift_max: 0.005
    from_hub: true             # re-pull the published bytes and verify those (§8.2)

  publish:
    hub:
      owner: chrishayuk
      repo_template: "{model_slug}-vindex{slice_suffix}"
      repo_type: dataset       # PIN 13.1
      collection: "larql/vindex-v1"
      tags: [vindex, larql, mechanistic-interpretability]
      visibility: private-until-verified
    mirror:
      r2_bucket: chuk-vindex

  budget:
    executor: auto             # auto | inline | leased
    max_wall_minutes: 240
    max_usd: 12
    requires:
      disk_gib: 512
      ram_gib: 64
      net_down_mbps: 1000
```

### 4.1 Required-field rationale

- **`source.revision` is mandatory and must be a full commit sha.** A vindex is a derived artifact; without a pinned upstream it is not reproducible and the provenance in the vindex-v1 manifest is a lie. This is the same `base_model_sha` pin already agreed for the Hub model card.
- **`extractor.version` is a released tag.** Building from a git sha of a branch makes the artifact unreproducible by anyone who isn't you.
- **`verify` is not optional.** A recipe with no verify block fails schema validation. See §8.

---

## 5. Content addressing and idempotency

```
build_id = sha256(canonical_json({
    apiVersion,
    spec.source,          # repo + revision + licence
    spec.extractor,       # tool, version, level, dtype, options
    spec.outputs          # presets + shard caps
}))
```

Deliberately excluded: `verify`, `publish`, `budget`, `metadata`. None of them change the produced bytes. Tightening a verification threshold or renaming a Hub repo should not force a 4-hour rebuild — it should re-run verification or re-point publication against the existing artifact.

**Rules:**

1. If `build_id` already has a `PASSED` build record in the experiments server, the dispatch is a **no-op** and the PR check says so.
2. If `build_id` exists but only as `FAILED`, dispatch proceeds (retry).
3. If only `verify` changed, dispatch a **verify-only** job against the existing R2 artifact — cheap, no extraction.
4. If only `publish` changed, dispatch a **publish-only** job from the existing R2 artifact.

This gives you re-publication and re-verification as first-class cheap operations, which matters a lot the first few times you get the Hub metadata wrong.

`build_id` is also the join key everywhere: R2 prefix, experiments-server experiment key, HF revision tag component, `vindex.json` manifest field.

---

## 6. Approval flow

### 6.1 On pull request — `validate.yml`

Cheap, no weights, no secrets beyond a read-only HF token.

1. **Schema validation** against `schema/vindexbuild.v1.json`.
2. **Upstream existence**: HF API call confirming `source.hf_repo@source.revision` resolves, and that the revision is a commit sha not a branch name that happens to look like one.
3. **Licence gate**: `source.licence` must be present in `policy/licences.yaml` with `derivative_redistribution: allowed`. Unknown licence → hard fail, requires a policy PR (reviewed separately) to unblock. This is the check that stops you accidentally publishing a derived artifact you have no right to redistribute; it is the one gate I'd argue hardest for.
4. **Cost and size estimate**: from upstream file sizes and the preset table, estimate output bytes per slice, worker class, and USD. Posted as a PR comment so that *approval is informed*.
5. **`build_id` computation**, posted in the same comment, with a lookup: "already built and PASSED on 2026-07-14, this is a no-op" / "verify-only change" / "full rebuild, est. 3h 20m, ~$9".
6. **Name collision check**: resolved Hub repo names don't collide with an existing repo owned by a *different* recipe.

### 6.2 Approval

Merge to `main` is approval. `CODEOWNERS` puts `recipes/` and `policy/` behind you. No separate approval workflow, no bot commands — the PR review *is* the gate, which is the whole reason for putting this on GitHub rather than in a form.

At G3, `policy/` gets a stricter CODEOWNERS than `recipes/` so external contributors can propose a build but not widen the licence allowlist.

### 6.3 On merge — `dispatch.yml`

1. Diff `main` to find changed recipe files.
2. For each, recompute `build_id` and classify the change (full / verify-only / publish-only / no-op).
3. **Upsert an experiment** in `chuk-experiments-server` keyed on `build_id`, carrying the full canonicalised recipe as the experiment spec.
4. **Dispatch** via the harness (`submit_run_from_experiment`), which is the already-proven path — registry entry → single-use join token → worker → pulse metrics → artifacts to R2 → results mirrored back.
5. Action's job is done. It holds only a rig dispatch token. It does not wait, does not poll, does not report status back into GitHub.

The last point is deliberate. If you want build status visible from GitHub, add a link in the PR comment to the experiments-server page for that `build_id`. Don't mirror state.

---

## 7. Build pipeline (on the worker)

Stages, each emitting a pulse metric and each independently resumable:

```
 1. PREFLIGHT   disk/ram/net check against spec.budget.requires; abort early if short
 2. FETCH       hf download {source.hf_repo}@{source.revision} → scratch
                verify per-file sha256 against the HF API's reported blobs
 3. EXTRACT     larql extract --level {level} --dtype {dtype} {options}
                → {build_id}.vindex/
 4. SLICE       larql slice --preset P  for each P in outputs
                shard cap enforced per vindex-v1 (20 GiB default)
 5. MANIFEST    write/complete vindex.json:
                  vindex_schema_version, build_id, base_model_repo, base_model_sha,
                  extractor {tool, version}, per-shard sha256, byte counts,
                  slice cross-references, licence, build timestamp
 6. MIRROR      push {build_id}/ to R2 (durable, zero-egress source of truth for bytes)
 7. VERIFY-A    local: reconstruction + logit-match against spec.verify
 8. PUBLISH     larql publish → HF repos, created PRIVATE
                full first; slices second, referencing full's revision
 9. VERIFY-B    larql pull from Hub into clean dir; re-run spec.verify (§8.2)
10. RELEASE     flip visibility public; tag revision; add to collection;
                render README/model card from the manifest
11. REGISTER    write verification report + artifact pins back to experiments-server
12. TEARDOWN    scratch wiped, worker released
```

**Budget wall behaviour is the rig's existing rule, unchanged:** if the worker is budgeted for N minutes, that's what it has. A build that blows the wall dies at whatever stage it reached, R2 keeps whatever was mirrored, and the build record is `FAILED(wall)`. A retry resumes from R2 if stages 1–6 completed.

### 7.1 Batching

Several recipes merged in one PR should share one lease where footprints allow — this is the existing "fill every rented GPU hour" rule applied to a disk-and-bandwidth-bound job. Extraction is largely CPU/IO-bound, so a fat Vast box can plausibly run two extractions concurrently or one after another inside a single lease. Scheduling detail deferred to PIN §13.7.

---

## 8. Verification — the hard rule

> **Nothing goes public unverified.** Publication is a two-phase commit: private upload, verify the *published* bytes, then flip visibility.

### 8.1 What gets checked

- **Reconstruction fidelity.** A vindex is a re-layout, not a lossy transform; sampled layers must reconstruct the upstream tensor bit-identically (cosine 1.0, max abs diff 0.0). Any drift here is a bug in the extractor, not a tolerance to widen. Where a lossy option is explicitly enabled (`down_q4k`), that path gets its own pre-registered tolerance rather than relaxing the exact one.
- **Logit match.** Fixed prompt set, teacher-forced, top-1 agreement plus bits/char drift against a reference forward pass. This is the same shape as the DEC C6 drift instrument and should share the implementation rather than growing a second one.
- **Manifest integrity.** Every shard digest in `vindex.json` matches the uploaded blob.

### 8.2 Why verify from the Hub, not from disk

Verifying the local build directory tells you the extractor worked. Verifying a fresh `larql pull` from the Hub tells you *the thing people will actually download* works. It catches truncated LFS uploads, missing files, wrong repo layout, a slice preset that produced something unloadable, and the classic "works because of a stale file still sitting in the local dir".

It costs a full download of the published artifact. For a 15 GB vindex on a fat-pipe worker that's minutes, and it's the difference between publishing a claim and publishing a proof. Make it default-on; PIN §13.5 covers whether it can be relaxed to first-publish-only for a given repo.

### 8.3 Failure

Verification failure leaves the Hub repo **private** with the failed revision in place, writes the report to the experiments server, and marks the build `FAILED(verify)`. Nothing is deleted — a failed artifact you can inspect is worth more than a clean slate.

---

## 9. Hub conventions

- **Repo naming** from `policy/naming.yaml`, default `{owner}/{model_slug}-vindex` for full and `{owner}/{model_slug}-vindex-{preset}` for slices.
- **Revision tag**: `v{vindex_schema_version}-{extractor}{extractor_version}-{build_id[:8]}` — e.g. `v1-larql0.14.2-3f9a2c71`. Immutable. `main` moves; tags don't.
- **Card frontmatter**, generated from the manifest, never hand-written:
  ```yaml
  library_name: larql
  base_model: google/gemma-3-4b-it
  base_model_relation: quantized      # or the closest accurate relation
  base_model_sha: 9b5b1a2f...
  tags: [vindex, larql, mechanistic-interpretability]
  licence: gemma
  ```
- **Collection** membership added at RELEASE, so the collection only ever contains verified artifacts.
- **Body of the card** is generated: what a vindex is, layer/feature counts, slice table with sizes, the `USE "hf://..."` snippet, the verification report summary, and the exact recipe that produced it (inlined, so the artifact carries its own reproduction instructions).
- **Deprecation, not deletion.** A superseded vindex gets `deprecated: true` in the card frontmatter, a pointer to its replacement, and removal from the collection. The bytes stay. Hub history is effectively append-only; pretending otherwise creates dangling references in other people's configs.

---

## 10. Secrets

| Secret | Held by | Never sees |
|---|---|---|
| HF **read** token | GitHub Actions | anything else |
| Rig dispatch token | GitHub Actions | HF write, R2 |
| HF **write** token | Rig control plane → worker as `EnvValue::Secret` | GitHub |
| R2 credentials | Rig control plane → worker as `EnvValue::Secret` | GitHub |

The property worth stating explicitly: **GitHub never holds a token that can publish.** A compromised Action can request a build; it cannot write to the Hub. The wire protocol already models `EnvValue::Plain | EnvValue::Secret`, so this needs no new mechanism.

Worker tokens are per-lease and expire with the lease. A leaked build log therefore leaks nothing durable.

---

## 11. Staleness

When `larql` ships a new extractor version, previously published vindexes are silently behind. Handle it as a **report, not an automation**:

A scheduled job (weekly) compares each recipe's `extractor.version` against the latest released larql tag and each `source.revision` against the upstream's current `main`, then opens a single issue: "3 recipes pin larql 0.14.2, current is 0.15.0; 1 upstream has moved". You decide whether a rebuild is warranted. Auto-bumping recipes would mean auto-republishing weights, which is exactly the thing that should require a human merge.

---

## 12. Adoption gates

| Gate | Deliverable | Pass condition |
|---|---|---|
| **G0** | Recipe schema + validate.yml + one recipe for a tiny model (v11 or TinyModel), `inline` executor on Fly | Merge a recipe → a real vindex appears on the Hub, verified, in under 15 minutes, with a build record in the experiments server |
| **G1** | `leased` executor path; Gemma 3 4B recipe end-to-end on a Vast box | Re-publication of the existing `chrishayuk/gemma-3-4b-it-vindex` from a recipe, byte-identical to a local build, all slices, verify-from-hub green |
| **G2** | Idempotency + verify-only/publish-only paths + staleness report | A no-op merge costs nothing; a threshold change re-verifies without re-extracting |
| **G3** | External PRs: stricter CODEOWNERS on `policy/`, validator posts to the PR, contributor docs | A recipe merged from someone who has never had Hub write access produces a verified artifact under their own namespace |

G0 is deliberately the small path so the loop is proven before a 4-hour job is on the line. It also settles PIN §13.4 empirically.

---

## 13. Open PINs

**13.1 — Dataset repo or model repo?**
These two prior decisions conflict and need resolving before G1. Earlier you settled on `--repo-type dataset` (a vindex isn't a model in the Hub sense; no config.json expectations). Later you settled on registering `library_name: larql` with the Hub so that `huggingface.co/models?library=larql` becomes a de facto registry. That filter is a **models** surface — dataset repos won't appear in it. You can have the clean typing or the free registry, not obviously both. My lean: **model repos**, because the registry effect is the adoption mechanism and it's the harder thing to replicate; the config.json expectation is cosmetic and a generated card handles it. But this decides the URL of every artifact you publish, so it wants a deliberate call, and it's cheap now and expensive in six months.

**13.2 — Repo split. RESOLVED (§3.1):** factory code in `larql` as a `larql-factory` crate with CLI subcommands; recipes, policy and verify sets in `chuk-vindex-recipes`; workflows shell out to a released `larql` binary.

**13.3 — Extractor pinning granularity.** Released tag only (reproducible by others, but blocks building against unreleased fixes) vs allowing a git sha with a warning. Lean: tag only, and cut a release when you need one.

**13.4 — `inline` executor threshold.** What working-set size sends a build to Fly rather than a leased worker? 40 GiB is a guess; G0 measures it.

**13.5 — Verify-from-hub always, or first-publish-only per repo?** Always is honest and costs bandwidth on every rebuild. First-publish-only is cheaper but assumes upload reliability is a property of the repo rather than the run.

**13.6 — Build record entity.** New `vindex_build` type in the experiments server, or reuse experiment + run with a `kind: vindex_build` tag? Lean: reuse — the server is deliberately generic and a new entity type is a schema migration for no gain.

**13.7 — Multi-recipe lease packing.** Concurrent extractions on one fat box vs strict serial within a lease. Needs a real measurement of whether extraction is disk-bound or CPU-bound at scale before committing.

**13.8 — Does the factory build the *reference* vindexes only, or also the DEC programme's shard sets?** DEC-4's Inkling extraction is the same operation with different downstream consumers (R2 shards for the serving tier, not Hub publication). Folding it in means one pipeline; keeping it separate avoids coupling a research funnel to a publication pipeline. Lean: same pipeline, `publish.hub` optional — an R2-only build is a legitimate recipe.

---

## 14. Build inventory

**Reuse as-is:** `larql extract` / `slice` / `publish` / `pull`; vindex-v1 manifest crate; rig control plane, worker, join tokens, budget walls, pulse metrics, R2 upload; experiments-server registry and mirror; `submit_run_from_experiment`; DEC C6 drift instrument (for logit-match).

**Genuinely new:**

1. `larql-factory` crate: recipe schema + validator, canonicaliser producing `build_id`, estimator, capability manifest. Shipped as CLI subcommands so both the Action and the worker call one implementation (§3.1).
2. `validate.yml` / `dispatch.yml` and the estimator that turns upstream file sizes into a cost figure for the PR comment.
3. The build-stage driver on the worker (stages 1–12), which is mostly orchestration around existing CLI calls.
4. Verify-from-hub harness.
5. Card generator (manifest → README frontmatter + body).
6. `policy/licences.yaml` and its checker.

Items 2–6 are all thin because item 1 is a CLI: the Action installs a released `larql` and calls it, so the workflows contain no logic that could drift from the worker's behaviour.

**Blocker inherited from DEC:** the rig currently has a mock Vast provider only. G1 is gated on the real one, which is already on the DEC critical path — the two programmes want the same unblock.

---

## 15. Scaling ladder, and what breaks on the way up

The factory is worth building at tiny scale and *necessary* at K3 scale, but the v0.1 recipe above only survives to about Gemma-26B. This section names the ladder and the four places the spec breaks.

### 15.1 The ladder

| Stage | Model | Executor | New thing it proves | Est. cost |
|---|---|---|---|---|
| **VF-0** | v11 / TinyModel | `inline` (Fly) | merge → artifact on Hub → record in registry, whole loop under 15 min | ~£0 |
| **VF-1** | Gemma 3 4B IT | `leased` | **byte-compare against the existing hand-built `chrishayuk/gemma-3-4b-it-vindex`** | ~$3 |
| **VF-2** | Gemma 4 26B | `leased`, campaign | real scale, known arch; resumable stages, multi-slice fan-out | ~$10 |
| **VF-3** | Inkling | `leased`, campaign | first novel architecture; multimodal drop; extractor capability gate | ~$5–10 |
| **VF-4** | Kimi K3 | campaign, multi-box | novel arch + KDA client + routing-stats-dependent carve + three destinations | ~$50–150 |

**VF-1 is the load-bearing one**, and it's cheap. You already have a known-good published artifact built by hand. If the factory reproduces it byte-for-byte from a recipe, the factory doesn't lie. Every later stage rests on that.

**VF-3 and VF-4 are gated on extractor work, not factory work.** The Inkling extractor is DEC-4's genuinely-novel code and the KDA client port is DEC-6b. The factory does not help write either. What it does is make the *second through tenth* attempts affordable — which is the entire game at K3 scale, where the first carve will be wrong.

### 15.2 Break 1 — the recipe assumes the extractor already understands the architecture

Under v0.1 you can merge a recipe for an architecture `larql` has never seen, dispatch a 12-hour job, and have it die at stage 3 on an unknown config key. At Gemma prices that's annoying; at K3 prices it's a bad afternoon.

**Fix:** `larql` publishes a capability manifest per release — `{model_type, config_keys, quant_formats, attention_kinds}`. The PR check resolves `source.hf_repo@revision`'s `config.json` `model_type` against the pinned `extractor.version`'s manifest. Unsupported → hard fail at PR time, with the message "larql 0.14.2 does not support `kda`; this needs extractor work first, not a recipe."

### 15.3 Break 2 — `outputs` needs a carve and a destination, not just a preset name

Today's K3 work makes this concrete. The K3 artifact set isn't slices of one thing at one place — it's three tiers with different carves, sizes and homes:

| Tier | Carve | Size | Home |
|---|---|---|---|
| demo-vindex | hot experts only, down-row payloads, up folded to per-feature scalars, client slice included | ~55–65 GB | **Hub** (published — the tier people actually download) |
| full walk-vindex | all experts, walk-serving carve | ~450 GB | **R2** (and pulled to the cartridge) |
| exact extents | everything, lossless | ~1.35 TiB | **R2 only** — the verification reference |

Two consequences for the schema:

1. **Each output declares its destination**, `hub` or `r2` or both. This answers PIN §13.8 outright: an R2-only build is first-class, and `publish.hub` is optional. It is also the honest answer to Hub storage limits — a 1.35 TiB public repo is not a thing you should try to have.
2. **The carve is semantic, not a preset label.** `down_only: true`, `fold_up_into_scalars: true`, `experts: {policy: hot, top_n_per_layer: 50}` are extraction parameters that change the bytes, so they live under `outputs[].carve` and they hash into `build_id`.

The cartridge is deliberately *not* a destination. The factory writes R2; `larql pull` fills the drive. That's already built and it keeps a physical object out of the build graph.

### 15.4 Break 3 — hot-expert selection is data-dependent, so it must be pinned

`top-50-per-layer` is derived from routing statistics, which come from a harvest run over some prompt distribution. Two builds of the same recipe against different traffic produce **different artifacts**, which silently destroys reproducibility and makes `build_id` a lie.

**Fix:** routing statistics become a pinned input alongside the weights.

```yaml
  source:
    hf_repo: moonshotai/kimi-k3
    revision: <sha>
    routing_stats:
      experiment_id: dec3-k3-routing-pass2
      artifact_sha: <sha256>          # pinned R2 artifact, not "latest"
```

`routing_stats` hashes into `build_id`. Re-harvesting traffic therefore produces a new `build_id` and a new revision tag rather than quietly mutating a published repo — which is exactly the behaviour you want when the coverage curve is a claim in the video.

### 15.5 Break 4 — budget walls and verification don't survive the jump

**Extraction becomes a campaign.** A hard wall on a 12-hour job under v0.1 loses the job. Layer-range extraction is embarrassingly parallel and independently checkpointable, so:

- stages 3–5 run per layer-range, each range mirrored to R2 on completion;
- a wall death costs one range, not the run;
- ranges can fan out across several leased boxes under one `CampaignId` — the wire protocol already carries the field;
- MANIFEST becomes a reduce step over completed ranges.

**Verification goes size-tiered.** Re-pulling 1.35 TiB to verify it is not a plan. Replace the flat `from_hub: true` with:

| Output size | Verification |
|---|---|
| ≤ 100 GB | full pull-and-verify from the destination (unchanged — the demo tier qualifies) |
| > 100 GB | all shard digests checked against the manifest; **sampled** shard pull with reconstruction + logit-match on the sampled ranges |
| exact-extents tier | is itself the reference — the lossy tiers are diffed against it rather than against a re-run of the upstream forward pass |

That last row is the nice property: once the exact tier exists in R2, it's a local oracle, and every later carve gets verified against a thing you already trust instead of against a 2.8T-parameter forward pass you'd have to stand up again.

**Cost approval gets a second signal.** Add `budget.confirm_usd`; the PR check fails if it's absent on any build the estimator prices above a threshold (say $50), or if it disagrees with the estimate by more than a band. Merging a $150 build should be a deliberate act, not the same gesture as merging a $3 one.

### 15.6 Sequencing recommendation

Do **VF-0 and VF-1 now** — days of work, near-zero cost, and VF-1 needs no new extractor because the artifact it reproduces already exists.

Let the **first K3 harvest be manual.** The weights drop and the video timeline are the constraint; the factory is not on the critical path for a first pass, and you'll learn the right carve by doing it wrong twice. Then bring the working invocation *back* as a recipe, so the factory's first K3 job is a reproduction of a known result rather than a discovery. That's the same trick VF-1 plays on Gemma, applied where it matters most.

The one thing worth pulling forward before the harvest is **§15.4** — pin the routing statistics from the very first harvest run, even if everything else is hand-run. Unpinned routing stats are the one mistake you cannot retrofit, because the traffic that produced them won't exist again.
