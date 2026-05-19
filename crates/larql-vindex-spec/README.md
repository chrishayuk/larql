# larql-vindex-spec

The public contract for the vindex on-disk format. Zero `larql-*`
dependencies — depended on by `larql-vindex` (writer/loader) and any
external validator that wants to parse vindex manifests without
pulling in the rest of the workspace.

This crate is small on purpose:

```
crates/larql-vindex-spec/
├── Cargo.toml
├── SPEC.md                              prose contract (start here)
├── schema/
│   └── vindex-v1.schema.json            JSON Schema 2020-12 mirror
└── src/
    ├── lib.rs                           Rust types (authoritative)
    └── thresholds.rs                    per-quant validation thresholds
```

When the three artifacts disagree, the Rust types in `src/lib.rs` win
and the others are bugs.

## What v1 models

The spec validates what the **validator** cares about:

- `vindex_spec_version` compatibility tag.
- Provenance hardening: `source` is required and includes
  `base_model_sha` + `base_safetensors_sha256` (per-shard map) +
  `extractor_sha`.
- `checksums` covers every `.bin` file the manifest references.
- Structural fields: dims, `extract_level`, `dtype`, `quant`,
  `layers` (single-file or sharded slots), `down_top_k`.

Loader-domain fields (`model_config`, `fp4`, `ffn_layout`,
`layer_bands`) round-trip cleanly via `serde(flatten)` into
`VindexManifest::extra` but are not validated by the spec. They evolve
under the on-disk `version` field, not `vindex_spec_version`.

## Sharding

Any single `.bin` exceeding `MAX_SHARD_BYTES` (20 GiB) must split into
`<base>-NNNNN-of-NNNNN.bin`, zero-padded to five digits, 1-indexed —
the convention mirrors safetensors. The manifest expresses shards as a
`shards` array with per-shard `file`/`offset`/`length`; the validator
concatenates per-shard ranges in order before checking.

## Validation thresholds

Live in [`src/thresholds.rs`](src/thresholds.rs), not the manifest:

| `quant` | `dtype` | `cosine_min` | `max_diff` |
|---|---|---|---|
| `q4k` / `kquant` | (any) | 0.995 | 0.05 |
| `none` | `f16` | 0.9999 | 0.01 |
| `none` | `f32` | 0.99999 | 0.001 |

Sampled layers: `[0, L/4, L/2, 3L/4, L-1]` — deduped for shallow
models. Five reads per validation regardless of model depth.

## HuggingFace model card

Every published vindex carries the spec stamp in its `README.md` YAML
front matter:

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
[`huggingface.co/models?library=larql`](https://huggingface.co/models?library=larql).

## Versioning

`larql-vindex-spec` follows semantic versioning:

- **Patch**: clarifications, schema bug fixes, doc edits.
- **Minor**: new optional fields, new closed-enum variants, loosening
  thresholds. Manifests written under an older minor remain valid.
- **Major**: removing fields, removing enum variants, tightening
  thresholds, breaking the manifest shape. The `vindex_spec_version`
  integer bumps with the major version.

See [SPEC.md](SPEC.md) for the full contract.
