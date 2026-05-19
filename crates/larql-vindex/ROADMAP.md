# Roadmap — larql-vindex

## Current state (as of 2026-05-16)

- **975 lib tests** on `larql-vindex` (`cargo test -p larql-vindex
  --lib`). Crate-local checks are wired through `make larql-vindex-ci`:
  fmt, clippy `-D warnings`, tests, example compile checks, bench
  compile/tests, and coverage policy. All four self-contained examples
  (`mmap_demo`, `demo_memit_solve`, `q4k_demo`, `walker_demo`) run
  end-to-end; the rest compile clean via `cargo check --examples`.
- **Folder layout decomposed**:
  - `index/{storage,compute,mutate}/` — substores, KNN dispatch, mutation
  - `index/compute/gate_knn/{mod,dispatch,scores_batch,hnsw_lifecycle}.rs`
    (round-4 split)
  - `index/storage/ffn_store/{mod,down,up,interleaved,interleaved_q4,interleaved_q4k,gate_q4,fp4,q4k_cache}.rs`
    (round-4 split)
  - `index/storage/lm_head/{mod,loaders,knn}.rs` (round-4 split)
  - `extract/build/{mod,down_meta,index_json,resume}.rs` (round-4 / 2026-05-09 re-split)
  - `extract/streaming/{mod,tensor_io}.rs` (round-5 phase 1, 2026-05-09)
  - `format/huggingface/publish/{mod,remote,upload,lfs,protocol}.rs`
    (round-5, 2026-05-09)
  - `format/weights/load/{mod,f32,q4k}.rs` (round-5, 2026-05-09)
  - `index/core/{mod,gate_lookup,patch_overrides,native_ffn,quantized_ffn,fp4_ffn}.rs`
    (round-5, 2026-05-09 — one capability impl per sibling)
  - `index/types/{mod,gate_lookup,patch_overrides,native_ffn,quantized_ffn,fp4_ffn,ffn_row}.rs`
    (round-5, 2026-05-09 — one trait per sibling, data structs in mod.rs)
  - `extract/streaming/stages/{mod,gate_vectors,router_weights,embeddings,down_meta,tokenizer,index_json,model_weights}.rs`
    (round-5, 2026-05-09 — one stage per sibling)
  - `format/weights/write_q4k/{mod,attn,ffn,moe_layers,norms,ple,lm_head,feature_major_down}.rs`
    (round-5, 2026-05-09 — one emitted artefact per sibling)
  - `format/{weights,filenames,fp4_codec,…}/`
  - `engine/` (was `storage/`) — StorageEngine + epoch + MEMIT
  - `config/{index,quantization,model,compliance,dtype}.rs` — was the
    624-line `types.rs` monolith
  - Large-file debt **partially closed 2026-05-10**: every file
    touched by rounds 4–5 is under the soft 600-LOC threshold, but
    the 2026-05-10 health pass found 8 files ≥800 LOC that were not
    in the round work-set (notably the new
    `index/storage/vindex_storage/mmap_storage.rs` 1182 L from the
    `VindexStorage` migration, and pre-existing
    `walker/vector_extractor.rs` 1213 L,
    `format/huggingface/download.rs` 941 L,
    `format/huggingface/publish/lfs.rs` 905 L,
    `index/types/ffn_row.rs` 875 L,
    `extract/build_helpers.rs` 848 L,
    `index/storage/gate_accessors.rs` 845 L,
    `format/huggingface/discovery.rs` 800 L). Tracked under P1
    "Residual large-file debt" below.
- **Quant dispatch via `quant::registry`** — adding the next K-quant is
  one table entry plus codec functions; ~3-file edit. Block sizes flow
  through `larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS` (round-4 M4).
  `LEGACY_BLOCK_Q4_K_STRIDE` names the 148-byte historical bug shape
  (round-4 M5).
- **Filename literals centralised** in `format::filenames` (252+
  occurrences → one constant module). Round-2 added 8 missed
  constants (LM_HEAD_BIN + FP4 family + attn_q4/q8 manifests). Round-4
  M1 closed `UP_WEIGHTS_BIN` / `DOWN_WEIGHTS_BIN`; the 2026-05-08
  pass added `ROUTER_WEIGHTS_BIN` and moved layer-weight paths through
  `LAYERS_DIR` / `layer_weights_filename`.
- **`DEFAULT_C_SCORE`** lifted on `index::types` so the patch overlay
  fallback and the vindexfile builder share one default (round-4 M3).
- **`VectorIndex` god struct decomposed** into four typed substores
  (`GateStore`, `FfnStore`, `ProjectionStore`, `MetadataStore`). Adding
  a new field is one edit in the relevant store.
- **5 storage formats**: f32, f16, Q4_0, Q4_K/Q6_K (Ollama-compatible),
  Q8, FP4/FP8 (exp 26).
- Mmap zero-copy with adaptive residency.
- HNSW graph index wired into `gate_knn` (opt-in via `--hnsw`).
- Q4_K dequant cache LRU-bounded via `--max-q4k-cache-layers`.
- Patch system for editable knowledge (`PatchedVindex` overlay).
  **2026-05-16 `gate_knn` optimization pass** (driven by the Exp 53
  shard service's bench gap): empty-base short-circuit when the
  layer has no native features, O(overrides²) → O(overrides) merge
  via `HashMap<feat, hit_idx>`, `argmax` instead of sort at
  `top_k = 1`, sort-skip when only deletions modified hits, lazy
  per-layer contiguous gate-override matrix with per-layer
  invalidation. Wins: vindex↔cache gap closed from ~20% to ~7% on
  the steady-state shard-query bench; cross-layer mutation cost
  cut by ~17% (218.8 → 181.9 µs at n=256/d=1024) because mutating
  layer B no longer evicts layer A's cache. See
  `crates/larql-server/benches/shard_query.rs`.
- **Vindexfile `FROM hf://...`** — HF resolution wired through the
  same resolver `larql run` and `larql extract` use.
- **Streaming extract checkpoints + auto-resume** — phase-level
  progress recorded to `.extract_checkpoint.json`; gate + down_meta
  phases auto-skip on a compatible checkpoint.
- **Stage labels centralised** in `extract::stage_labels` (15 labels;
  typo at any site is now a compile error).
- `make larql-vindex-coverage-summary` + `make
  larql-vindex-coverage-html` (cargo-llvm-cov) enforce both the
  aggregate line floor and `coverage-policy.json`.
- **Coverage ratchet**: aggregate floor is 71% lines from the
  2026-05-08 baseline of 71.56%; current measured **90.86% lines**
  as of 2026-05-16, with **105 of 147 files at the 90% default**
  and 42 debt baselines. The `patch/overlay.rs` baseline (82%) is
  the one entry the 2026-05-16 gate_knn optimization touched —
  added LayerGateCache code paths + targeted unit tests; file now
  sits at 88.61%.
- **Cross-platform CI**: `.github/workflows/larql-vindex.yml` runs
  format, check, examples, clippy, tests, and bench compile/tests on
  Linux, Windows, and macOS. Coverage policy runs on Ubuntu.
- Bench rig daemon-aware (`make bench-vindex-scaling` refuses if
  `larql-server` / `larql-router` are running on the host).

---

## P0: Active

### Modularity + magic-literal debt

**Status**: Mostly closed. Only the large-file decomposition bullet
remains open as of 2026-05-09.

Closed during the 2026-05-09 review (verified against the tree, not
just the audit doc):

- [x] Architecture-specific extraction literals — dense clustering
  routes through `LayerBands::for_family(...).knowledge` via
  `extract::build::knowledge_layer_range`. No hard-coded layer ranges
  remain in the extraction pipeline.
- [x] Vindex file layout literals — production paths fully routed
  through `format::filenames`. The 2026-05-09 sweep added
  `ROUTER_WEIGHTS_BIN` (was the last stray production literal in
  `extract/streaming.rs`). Test fixtures keep literals deliberately to
  pin the wire contract.
- [x] Stringly typed `ffn_layout` — already a typed
  `Option<FfnLayout>` enum (`config/index.rs`).
- [x] Algorithm parameters lifted — extraction batch sizes,
  relation-cluster cap, k-means iterations live in
  `extract::constants`; HNSW build parameters in
  `config::hnsw::HnswBuildConfig::{LAYER, EXPERT}`.
- [x] `GateIndex` split — narrower capability traits (`GateLookup`,
  `PatchOverrides`, `NativeFfnAccess`, `QuantizedFfnAccess`,
  `Fp4FfnAccess`, `FfnRowAccess`) with `GateIndex` retained as the
  compatibility composition for existing trait-object consumers.

Still open:

- [x] Large-file decomposition — **closed 2026-05-09**. Last file
  (`format/weights/write_q4k/mod.rs` 734 L) split into one sibling per
  emitted artefact (attn / ffn / moe_layers / norms / ple / lm_head),
  orchestrator down to 318 L. Closed today: `publish.rs` (997),
  `streaming.rs` (832), `load.rs` (817), `core.rs` (755), `types.rs`
  (715), `streaming/stages.rs` (644), `write_q4k/mod.rs` (734). No
  non-test file in vindex now exceeds the soft 600-LOC threshold.

**Acceptance bar:** no remaining production filename/layout magic
strings for vindex-owned files (met), extraction remains model-family
agnostic (met — see P1), `GateIndex` split into narrower capability
traits (met), and no new module grows past the current large-file
debt without a split plan.

### `VindexStorage` trait abstraction
**Impact**: Lets Redis / S3 / GPU-residency backends plug in
**Effort**: Medium
**Status**: ✅ Closed 2026-05-10 — all 7 migration steps shipped;
walk kernels, Metal dispatch, KNN, and overlay all consume the
trait. Future Redis / S3 backends land as parallel `VindexStorage`
impls without touching this code. Detail kept here as the migration
record; the section will move to `CHANGELOG.md` next time a P0 item
displaces it.

The substore extraction (`GateStore`, `FfnStore`, `ProjectionStore`,
`MetadataStore`) got most of the way there. Formalise a sealed
`VindexStorage` trait (mmap-agnostic row accessor) so Q4_K row reads
can route through Redis-cached or S3-buffered backends without
walk-kernel changes.

**Why now**: the per-layer FFN format (Phase 2, closed today) made
"give me the bytes for layer L, component C, feature F" the canonical
addressing contract. That's exactly the right granularity for a
storage trait — every concrete backend can satisfy it. The current
mmap path becomes one impl among several.

**Acceptance bar**:
- A sealed `VindexStorage` trait whose surface is defined by the
  current walk-kernel and decode hot-paths (`q4k_ffn_row_*`,
  `gate_*`, `down_features_*`, `attn_*_layer_data`).
- One impl: `MmapStorage` reproducing today's behavior bit-for-bit
  (no measurable decode regression on the cool-machine 4B/26B
  baselines).
- Walk kernels and Metal dispatch consume only the trait, not
  concrete `Mmap` types.
- Coverage: every trait method has a test against `MmapStorage`;
  the trait-object boxing path has at least one round-trip test.

**Out of scope for this round**: Redis / S3 backends (those land on
top of the trait once it's stable). Same for GPU-resident storage —
the trait surface should leave room for it but we don't implement.

**Migration steps:**

- [x] Step 1 — sealed trait skeleton in
  `index/storage/vindex_storage/mod.rs`. 14 byte-yielding methods
  covering hot-path substore reaches; FP4 + DownMeta deliberately
  not behind the trait (richer per-feature decoders).
- [x] Step 2 — `MmapStorage` parity impl
  (`from_substores(&FfnStore, &GateStore, &ProjectionStore, hidden)`).
- [x] Step 3 — Criterion perf gate
  (`benches/vindex_storage_dispatch.rs`). Initial `Bytes`-returning
  shape paid 6 atomic ops per fetch (12× direct); redesigned
  per-layer accessors to return `BytesView<'a>` with
  `as_slice() -> &'a [u8]` zero-atomic / `to_owned_bytes()` opt-in.
  Final: `Arc<dyn>` within ~5% of direct on both layer-fetch and
  per-row.
- [x] Step 4 — `storage: Arc<dyn VindexStorage>` field on
  `VectorIndex`, populated by `refresh_storage()` at the end of
  every loader that mutates substore mmaps. Per-layer
  byte-yielding accessors forward through `self.storage`
  (`attn_q4k_layer_data` / `attn_q8_layer_data` /
  `attn_q4_layer_slices` / `interleaved_q4k_layer_data` /
  `down_features_q4k_layer_data` / `gate_q4_data`). Whole-buffer
  accessors (`attn_q4_data`, `interleaved_q4*_mmap_ref`) deferred
  to step 5 — they need an API-shape decision once substore mmap
  fields drop.
- [x] Step 5 (partial) — `storage` field is now
  `Arc<MmapStorage>` (concrete, mutable via `Arc::make_mut`).
  Loaders rewritten to use `set_*()` setters; `refresh_storage()`
  removed. 12 setters + 11 `has_*` + 6 `*_view()` helpers added.
  Read sites migrated for `lm_head/knn.rs`, `interleaved_q4.rs`,
  `gate_accessors.rs` `is_some()` checks, `is_mmap()`,
  `attn_q4_data`. `MmapStorage` field visibility tightened to
  `pub(crate)` for test ergonomics.
- [x] Step 6 — gate KNN compute paths (`gate_accessors.rs`,
  `compute/gate_knn/*`, `gate_store.rs`, `mutate/mod.rs`,
  `patch/overlay.rs`) migrated to `self.storage.gate_layer_view(layer)`;
  trait-covered `Arc<Mmap>` substore fields dropped from
  `FfnStore` / `GateStore`. `ProjectionStore` deleted entirely (its
  9 fields all moved to `MmapStorage`). `MmapStorage::release_pages()`
  added — tracks `Arc<Mmap>` handles per setter; replaces the
  per-substore-field madvise loop in `release_mmap_pages`. Heap
  fields (`gate_vectors`, decode caches, HNSW caches, FP4 storage)
  stay on substores.
- [x] Step 7 — f32 native FFN paths (`up_features`,
  `down_features`, `interleaved_f32`) migrated to `MmapStorage`.
  `FfnStore` shrunk to caches + FP4 storage only.
  `release_mmap_pages` collapsed to a single
  `self.storage.release_pages()` call.

See `CHANGELOG.md` 2026-05-10 entry for the full step 1-3 writeup
plus bench numbers.

## P1: Active

### Architecture-independent extraction and weight writing

**Status**: Closed for current architectures (2026-05-09). Reopen when a
non-standard attention contract (MLA, MQA-with-shared-rotary, etc.) is
landed in `larql-models` and needs writer support.

The extraction stack now preserves architecture facts from
`ModelArchitecture` end-to-end, and a single capability helper gates
unsupported attention layouts before any output is written.

Work items:

- [x] Audit f32/Q4K writer entry points and loader surfaces for implicit
  standard-attention assumptions. Both writers (`write_model_weights` and
  `write_model_weights_q4k`) call `ensure_standard_attention_supported`
  on entry; the check lives in one place at
  `format/weights/capabilities.rs`.
- [x] Replace `extract/build_from_vectors.rs` model-name heuristics —
  audit (2026-05-09) found no `contains("gemma")` / `contains("llama")`
  string heuristics remain. The path routes through `arch.family()` and
  `LayerBands::for_family`.
- [x] Add an architecture capability check **before any partial write**.
  Added `ensure_extract_level_supported` (2026-05-09) wired into both
  `build_vindex` and `build_vindex_streaming`. Browse-level extracts of
  MLA architectures still succeed (no attention written); Attention /
  Inference / All tiers fail with a targeted `UnsupportedArchitecture`
  error before the output directory is created.
- [x] Centralise remaining protocol-like tensor/manifest tags. Quant tags
  flow through `quant::registry`; file-kind strings through
  `format::filenames`; capability surfaces through
  `SURFACE_F32_WEIGHT_WRITER` / `SURFACE_Q4K_WEIGHT_WRITER` /
  `SURFACE_EXTRACT_PIPELINE` constants.
- [ ] Extend f32/Q4K weight writers beyond standard Q/K/V/O when a concrete
  non-standard architecture contract is added. Won't fix until a
  `larql-models` MLA implementation lands.
- [x] Tests proving unsupported attention layouts are rejected before any
  partial vindex write — `build_inference_rejects_mla_before_writing`
  asserts `read_dir(output_dir).is_empty()` after the failure;
  `extract_level_*_rejects_mla` cover Attention/Inference/All and
  `extract_level_browse_passes_for_mla` covers the no-attention path.
- [x] Fixture tests that prove unknown/custom families do not inherit
  Gemma/Llama defaults through string matching — extended
  `unknown_family_does_not_inherit_known_bands_by_string_prefix` in
  `config::compliance` to compare lookalikes (`gemma3-clone`,
  `llamafied`) against canonical bands and prove the layer-count
  fallback is structurally distinct.

Acceptance bar (met 2026-05-09): vector-only and model-backed extracts
agree on family, embedding scale, layer bands, and required tensor
coverage; unsupported attention layouts fail before any file is written;
no string-prefix inheritance of curated band tables.

### Residual large-file debt (review finding 2026-05-10)
**Impact**: Code navigability + future split cost
**Effort**: Small–Medium per file
**Status**: ✅ Closed 2026-05-10 — all 7 listed files split or
documented as won't-split.

Round-6 splits (2026-05-10):

- [x] `walker/vector_extractor.rs` (1213 L) → 7 siblings under
  `walker/vector_extractor/` (largest 373 L). All siblings ≥94.8%
  line coverage.
- [x] `extract/build_helpers.rs` (848 L) → 6 siblings + test_support
  under `extract/build_helpers/` (largest 264 L). All ≥98%.
- [x] `format/huggingface/download.rs` (941 L) → minimal 2-way
  split (`download/mod.rs` 676 L, `download/helpers.rs` 270 L).
  Pure helpers at 100%; the network-bound mod.rs inherits the 74%
  baseline as `download/mod.rs:64.0` (HF API hard to mock without
  significant test infrastructure). `mod.rs` still over 600 LOC by
  user direction.
- [x] `format/huggingface/publish/lfs.rs` (905 L) → 4 siblings +
  test_support under `publish/lfs/` (largest 281 L). All ≥91%; old
  `lfs.rs:97` debt baseline removed (every sibling at default 90%).
- [x] `format/huggingface/discovery.rs` (800 L) → 3 siblings +
  test_support under `discovery/` (largest 386 L). All ≥94%.
- [x] `index/types/ffn_row.rs` (875 L) → 3 siblings under
  `index/types/ffn_row/` (largest 444 L). Trait declaration in
  `mod.rs` (87.1%, debt baseline added), shared `Stub` fixture in
  `test_support.rs` (84.9%, debt baseline added), tests in
  `tests.rs`. Production coverage was always at 87% — split made it
  visible by removing dilution from co-located tests.
- [x] `index/storage/gate_accessors.rs` (845 L) → 3 siblings under
  `gate_accessors/` (mod.rs 347 L, accessor_tests.rs 445 L,
  release_pages_tests.rs 53 L). Production-only mod.rs at 92.3%
  (up from old 70.5% combined-file baseline).
- [x] `index/storage/vindex_storage/mmap_storage.rs` (1182 L) —
  **kept as a single file by design**: 12-method trait impl, dense
  per-substore mmap storage. Splitting by trait method fragments
  without modularity gain.

**Outcome**: workspace coverage 90.30% (was 90.17% at start of
round); 104 files at 90% default (was 86); 42 debt baselines (was
40 — net +2 from ffn_row split). 932 lib tests + 19 integration
suites pass; clippy clean. No file outside the won't-split list is
≥800 LOC.

### Production-path `unwrap`/`expect` triage (review finding 2026-05-10)
**Impact**: Crash-safety on I/O failures
**Effort**: Small
**Status**: ✅ Closed 2026-05-10 — 6 hotspot files audited (56
production sites total); only `extract/build_from_vectors.rs`
needed real fixes (9 sites converted, 3 new error-path tests).
The other 47 sites are defensible idioms — left as-is.

Triage outcome:

| File | Sites | Verdict |
|---|---|---|
| `extract/build_from_vectors.rs` | 9 | **Fixed** — JSONL field-access panics on missing/malformed records converted to `VindexError::Parse` via two helpers (`parse_u64_field`, `parse_f32_vector`). 3 new error-path regression tests. |
| `index/storage/gate_store.rs` | 12 | Defensible — 5× mutex/RwLock locks, 4× shape ops with static invariants, 1× `as_slice` on owned-from-vec Array2, 2× cache slot just-assigned. The review's "concrete target" line 391 was inside `#[cfg(test)] mod gate_cache_lru_tests` (starts line 373) — false positive. |
| `index/compute/gate_knn/hnsw_lifecycle.rs` | 12 | Defensible — 9× mutex locks, 3× shape ops with caller-side invariants. Optional cleanup: line 56 could mirror the sibling at line 82 which uses `.ok()?`. |
| `index/compute/gate_knn/scores_batch.rs` | 9 | Defensible — 3× lock guards, 5× shape ops bounds-checked or invariant, 1× cache slot just-assigned. |
| `patch/knn_store.rs` | 8 | Defensible — 7× mutex locks, 1× `partial_cmp().unwrap_or()` fallback. |
| `index/storage/ffn_store/q4k_cache.rs` | 6 | Defensible — 6× mutex locks. |

Lesson: the review's "~120 sites" framing over-stated the
fix surface. The signal-to-noise was ~10% — `Mutex::lock().unwrap()`
and `ArrayView2::from_shape().unwrap()` with static shapes are
idiomatic Rust, not crash hazards. Future audits should pre-filter
these patterns before counting.

**Acceptance bar (met)**: every production `.unwrap()` in the six
hotspot files is either a mutex/RwLock lock, a statically-provable
shape op, a just-assigned cache slot, or has been converted to `?`
with a typed error. 922 lib tests + all integration suites pass;
clippy clean.

### Coverage round-7 (review finding 2026-05-10)
**Impact**: Per-file ratchet — 40 files still below the 90% default
**Effort**: Small per file
**Status**: Active

Aggregate is at 90.04% (workspace floor met for the first time),
but the per-file ratchet has 40 debt entries. Highest-leverage
targets by current coverage:

- [ ] `format/weights/write_q4k/moe_layers.rs` — **34.0%** (the
  single biggest tractable win)
- [ ] `config/compliance.rs` — 57.9%
- [ ] `format/load.rs` — 62.9%
- [ ] `engine/core.rs` — 82.0%

**Acceptance bar**: each listed file moves to ≥90% line coverage
or carries a documented rationale (e.g. error-path branches that
require a real S3 outage to exercise).

### Cached layer decode for template-fixed layers (L0–12) — parked
**Impact**: 155+ tok/s decode (skip 13 of 21 layers)
**Effort**: Medium
**Status**: ⏸ Parked — depends on upstream work that isn't ready yet.
Don't start until the prerequisite lands. Keep `CachedLayerGraph` in
`larql-inference` as the integration point.

### Layer-level resume within an incomplete phase
**Impact**: A run interrupted at gate-layer-30-of-34 today re-runs
all 34 layers; layer-level resume would skip 30
**Effort**: Medium
**Status**: Forward-looking — phase-level resume now in place
(2026-04-25 round-3); the layer-level extension needs mid-phase file
truncation to the last clean layer boundary, which is more delicate
than the phase flag.

## P2: Forward-looking

### Expert-level sharding protocol
**Impact**: Unlocks > 256-expert MoE sharding-within-layer
**Effort**: Medium
**Status**: Forward-looking

Today `larql-router` shards by layer, not by expert ID within a
layer. For DeepSeek-V4-class models (1K+ experts) experts need to
shard across servers. Add an `ExpertRoute` message type to
`larql-router-protocol` and wire `GridState` dispatch.

### Q5_K / Q3_K / BF16 quant additions
**Effort**: Small per format (≈ 3 files thanks to the registry)
**Status**: Not yet needed — add when a target model demands it

Path: implement codec functions in `larql-models/src/quant/ggml/`,
add one entry to `QUANT_FORMATS` in `quant::registry`, add match arm
in `larql-compute::backend::quant_matvec`. Verified by the round-2
audit.

### Multi-model vindex
**Status**: Research

Store features from multiple models in one vindex. Compare
representations across architectures.

### Incremental extraction
**Status**: Research

Add new layers / features to an existing vindex without full rebuild.

---

## Won't fix

- **`detect.rs` (1391 L) split** in `larql-models` — cohesive single
  entry point dispatching to 12 architectures. Splitting fragments
  without modularity gain. Reconsider when a second detection system
  emerges (auto-discovery from model ID, multi-modal config).

---

## History

Completed entries previously kept here have been moved to
[`CHANGELOG.md`](CHANGELOG.md), reverse-chronological by date. Active
P0/P1/P2 items above; once a row lands it migrates to the changelog.
