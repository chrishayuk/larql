# Vindex Spec v1

The public contract for the vindex on-disk format. This document is
the prose pair of the Rust types in `src/lib.rs` and the JSON Schema
in `schema/vindex-v1.schema.json` — when the three disagree, the Rust
types win and the others are bugs.

The contract is versioned through the `larql-vindex-spec` crate's
semantic version. Tooling pins via Cargo; the integer
`vindex_spec_version` in the manifest is the compatibility tag, not
the evolution channel.

## 1. Scope

A *vindex* is a directory of mmap-friendly binary files (`*.bin`) plus
a JSON manifest (`index.json`) that catalogues their contents and pins
the upstream model the artifact was derived from. The spec defines:

- The structural fields the validator needs to do its job.
- Provenance hardening on `source`.
- Closed enums for `extract_level`, `dtype`, `quant`.
- The sharding rule for `.bin` files larger than a hard cap.
- The tokenizer rehosting requirement.
- Per-(quant, dtype) validation thresholds.

The spec **does not** define:

- The byte layout of the `.bin` files. That's the file-format version
  (`version: 2` on disk), which evolves independently.
- Loader-domain fields (`model_config`, `fp4`, `ffn_layout`,
  `layer_bands`). These are real on-disk fields that survive
  round-trip via `serde(flatten)`, but the spec doesn't validate
  their internal shape — that's the loader's job.

## 2. Manifest (`index.json`)

Lives at the vindex root. Top-level fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `vindex_spec_version` | u32 | yes | Must equal `1`. The validator rejects other values. |
| `version` | u32 | yes | On-disk file-format version. Independent of `vindex_spec_version`. |
| `model` | string | yes | Upstream model id, e.g. `google/gemma-3-4b-it`. |
| `family` | string | yes | Architecture family for loader dispatch. |
| `source` | object | yes | Provenance — see §3. Was nullable pre-v1; v1 hardens it. |
| `checksums` | object | yes | `{filename → sha256-hex}` for every `.bin` referenced. Was nullable pre-v1. |
| `num_layers` | u32 | yes | Transformer layer count. |
| `hidden_size` | u32 | yes | Hidden dim. |
| `intermediate_size` | u32 | yes | FFN intermediate dim. |
| `vocab_size` | u32 | yes | Vocabulary size. |
| `embed_scale` | f32 | yes | Embedding scaling factor. |
| `extract_level` | enum | yes | `browse` < `attention` < `inference` < `all`. See §4. |
| `dtype` | enum | yes | `f32` or `f16`. See §5. |
| `quant` | enum | yes | `none`, `q4k`, or `kquant`. See §6. |
| `layers` | array | yes | Per-layer offset table — see §7. |
| `down_top_k` | u32 | yes | K used at runtime for top-K gate-feature lookup. |
| `has_model_weights` | bool | yes | Whether full weight tensors (not just gate vectors) are present. |

Any field on disk that isn't named above is passed through unchanged
(`serde(flatten)` into the Rust `extra` map). Known loader-domain
fields that use this channel today: `layer_bands`, `model_config`,
`fp4`, `ffn_layout`.

## 3. Provenance (`source`)

All fields are REQUIRED in v1. The pre-v1 manifest allowed `null` on
`huggingface_revision` and `safetensors_sha256` (and `source` itself
was nullable); v1 retires both.

| Field | Type | Notes |
|---|---|---|
| `huggingface_repo` | string | Canonical upstream repo. |
| `huggingface_revision` | string | Branch or tag pulled at extract time. |
| `base_model_sha` | string | Upstream git commit SHA — the validator pulls exactly these bytes. |
| `base_safetensors_sha256` | object | `{shard_filename → sha256-hex}` for every safetensors shard. |
| `extracted_at` | string | ISO 8601 timestamp. |
| `larql_version` | string | `larql` crate version that produced the vindex. |
| `extractor_sha` | string | Git SHA of the larql repo at extract time. |

The combination of `base_model_sha` + `extractor_sha` is enough for a
validator to reproduce the extraction bit-for-bit modulo float
reduction non-determinism — which is what the cosine threshold
tolerates.

## 4. Extract level (`extract_level`)

Strictly increasing tier. Each level is a superset of the previous.

| `extract_level` | Adds | Enables |
|---|---|---|
| `browse` | gate + embed + down_meta + tokenizer | WALK / DESCRIBE / SELECT |
| `attention` | + attention + norms | client-side of remote-FFN inference |
| `inference` | + FFN up/down | full local INFER |
| `all` | + lm_head + COMPILE extras | COMPILE |

## 5. Storage dtype (`dtype`)

Float precision for non-quantised tensors (gate vectors, embeddings,
norms, ...). Closed enum:

| Value | Notes |
|---|---|
| `f32` | IEEE 754 binary32. Default for full-precision vindexes. |
| `f16` | IEEE 754 binary16. Default for size-optimised vindexes. |

`dtype` is independent of `quant` — they cover different files. FFN
weights are governed by `quant`; everything else by `dtype`.

## 6. Quant format (`quant`)

Quant scheme for FFN weight files. v1 enum:

| Value | Notes |
|---|---|
| `none` | Float storage controlled by `dtype`. |
| `q4k` | Q4_K / Q6_K family — the v1 canonical tag. Writers emit this for v1 vindexes. |
| `kquant` | Same Q4_K / Q6_K family as `q4k`; the post-rename canonical tag. Readers MUST accept it as an alias of `q4k`. A future v2 schema bump flips writers to emit `kquant` by default. |

Filename convention (set by the Rust constants in
`larql_vindex::format::filenames`):

- New canonical: `interleaved_kquant.bin`,
  `attn_weights_kquant.bin`, `lm_head_kquant.bin`,
  `down_features_kquant.bin` (each paired with a
  `*_manifest.json` sidecar where applicable). Writers emit these.
- Legacy: `interleaved_q4k.bin`, `attn_weights_q4k.bin`,
  `lm_head_q4.bin`, `down_features_q4k.bin`. Readers MUST accept
  these as the read-only fallback; writers no longer emit them.

The dual-naming is the on-disk counterpart of the `q4k` / `kquant`
enum alias: both names describe the same k-quant family of block
formats (Q4_K plus Q6_K mixed for the FFN down projection by
default). A vindex extracted with a pre-rename binary keeps loading
under post-rename readers without re-extraction.

FP4 storage is governed by a separate `fp4` loader field (see §11);
the spec doesn't model its internal config, but FP4 vindexes still
round-trip cleanly via the `extra` pass-through.

## 7. Layers (`layers`)

One entry per transformer layer. Each entry declares either a
single-file slot or a sharded one — never both, never neither. MoE
vindexes carry optional `num_experts` + `num_features_per_expert`.

```jsonc
// Single-file (typical):
{ "layer": 0, "num_features": 10240,
  "file": "interleaved_kquant.bin", "offset": 0, "length": 52428800 }

// Sharded (only when the underlying file exceeds MAX_SHARD_BYTES):
{ "layer": 0, "num_features": 10240,
  "shards": [
    { "file": "interleaved_kquant-00001-of-00003.bin", "offset": 0, "length": 52428800 }
  ] }

// MoE (optional fields):
{ "layer": 0, "num_features": 10240,
  "file": "experts_packed.bin", "offset": 0, "length": ...,
  "num_experts": 8, "num_features_per_expert": 1280 }
```

The validator requires that every `file` (single-file slot) and every
`file` inside `shards` appears as a key in the top-level `checksums`
map.

## 8. Sharding

Any single `.bin` exceeding **20 GiB** (`MAX_SHARD_BYTES`) must split.

Naming: `<base>-NNNNN-of-NNNNN.bin`, zero-padded to five digits,
1-indexed. The convention mirrors safetensors so existing tooling can
recognise the pattern.

When a file is sharded, the layer entries that reference it use the
`shards` array; the validator concatenates the per-shard ranges in
order before checking. The 20 GiB cap is per-shard, not per-tensor —
a 60 GiB tensor becomes three 20-GiB shards.

## 9. Tokenizer rehosting

The vindex repo carries its own tokenizer files:

- `tokenizer.json` (always required)
- `tokenizer_config.json` (required when upstream defines one)
- `chat_template.jinja` (required when upstream defines one)
- `special_tokens_map.json` (required when upstream defines one)

Rationale: upstream model repos do occasionally get deleted or have
breaking tokenizer fixes pushed without a new revision. Rehosting
makes vindexes self-sufficient. The drift cost — a tokenizer fix
upstream not propagating to a vindex — is acceptable because the
tokenizer is pinned to `base_model_sha` anyway; if upstream fixes it,
the next vindex revision picks up the fix.

## 10. Validation thresholds

Live in `thresholds.rs`, not the manifest. The validator picks them
from `(quant, dtype)`:

| `quant` | `dtype` | `cosine_min` | `max_diff` |
|---|---|---|---|
| `q4k` / `kquant` | (any) | 0.995 | 0.05 |
| `none` | `f16` | 0.9999 | 0.01 |
| `none` | `f32` | 0.99999 | 0.001 |

The two k-quant tags share thresholds because they describe the same
on-disk format family. When `quant` is either tag the quant
dominates loss and the dtype is ignored.

Sampled layers: `[0, L/4, L/2, 3L/4, L-1]` — five reads (deduped for
shallow models), deterministic, cheap even on 31B-class models.

The "bit-identical" framing in early discussions is an aspiration,
not a contract — float reduction order varies across CPU/Metal
backends. The cosine + max_diff pair operationalises "faithful
reconstruction" in a way that survives reductions while still
catching real errors.

## 11. Loader-domain fields (pass-through)

These exist in real manifests today and round-trip via `extra` /
`serde(flatten)`. The spec doesn't validate them; the loader does.

| Field | Role |
|---|---|
| `layer_bands` | Syntax / knowledge / output band partitioning for DESCRIBE. |
| `model_config` | Architecture metadata (head dim, RoPE base, Granite scalars, ...). |
| `fp4` | FP4/FP8 block storage config (per-projection precision, compliance gate). |
| `ffn_layout` | When `per_layer`, FFN weights live in `layers/layer_NN.weights`. |

These evolve under the on-disk `version` field, not
`vindex_spec_version`. Adding a new loader field doesn't require a
spec bump.

## 12. HuggingFace model card

The companion to the manifest is the repo's `README.md` YAML front
matter, which the Hub uses for filtering and indexing:

```yaml
---
base_model: google/gemma-3-4b-it
base_model_sha: 1adbacd6b6dee75c
library_name: larql
tags:
  - vindex
  - vindex-v1
  - vindex-q4k                 # mirrors quant value (legacy alias `vindex-kquant` also accepted)
  - vindex-extract-inference   # mirrors extract_level
vindex_spec_version: 1
pipeline_tag: text-generation
---
```

`library_name: larql` is the discovery anchor — once registered with
HF's `hub-docs`, every vindex becomes filterable via
`huggingface.co/models?library=larql`, the same trick `gguf` and
`mlx` use. HF repo-name suffixes (`-attn`, `-embed`, `-client`,
`-server`, `-browse`, `-expert-server`) are publish-time slicing
conventions, not manifest fields.

## 13. Versioning policy

`larql-vindex-spec` follows semantic versioning:

- **Patch** bumps: clarifications, schema bug fixes, doc edits.
- **Minor** bumps: adding new optional fields, adding closed-enum
  variants, loosening thresholds. Manifests written under an older
  minor remain valid.
- **Major** bumps: removing fields, removing enum variants,
  tightening thresholds, breaking the manifest shape. The
  `vindex_spec_version` integer bumps with the major version.

Reference vindexes are re-validated on every minor or major bump of
this crate; the validator's CI gate is the canary.

## 14. Out of scope for v1

- Multi-modal vindexes (vision encoders, audio frontends).
- Cross-vindex composition (a vindex that references tensors in
  another vindex by URI).
- Streaming / partial-fetch protocols beyond what mmap + LFS sparse
  reads already give you.
- bf16 storage. v1 covers `f32` and `f16` only.
- Quant schemes beyond the Q4_K / Q6_K family the `q4k` / `kquant`
  tags both describe. Adding one (e.g. Q5_K, Q8_K, FP4) is a minor
  bump.

## 15. Open questions (deferred)

- **Spec-version-1 freeze date.** Until the first three reference
  vindexes pass validation, the spec is mutable. Picking a freeze
  date is a product call.
- **FP4 in the spec proper.** Today FP4 is a loader pass-through
  field. A future minor bump may promote it into the spec types so
  the validator can apply FP4-specific thresholds — the
  `fp4_compliance.json` sidecar already does some of this at
  extract time.
- **Slice / component publishing.** The HF repo-suffix conventions
  for `-attn` / `-embed` / `-server` etc. aren't modelled in v1.
  A future minor bump may add an optional `slice` field declaring
  which tensor classes a published artifact carries.
